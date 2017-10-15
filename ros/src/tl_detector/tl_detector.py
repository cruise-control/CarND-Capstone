#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from copy import deepcopy  # for waypoints
import math

from scipy.spatial import KDTree

#import sys
#sys.path.insert(0, "../waypoint_updater")  # for Utility
#import Utility

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self._waypoint_tree = None
        self.DEBUG_USE_TRUTH = True
        
        # Base waypoints should only be published once
        # For now it's best to assume this because we don't take any special
        # care to re-initialize properly if we were to get a different set
        # This won't happen in the simulator, but who knows about the real
        # world?
        msg = rospy.wait_for_message('/base_waypoints', Lane)
        self.waypoints = msg.waypoints
        rospy.loginfo('%d Waypoints loaded in tl_detector', len(self.waypoints))

        # A kd-tree will be about as fast as possible for nearest neighbor
        # searches, or at a minimum, the fastest solution with least effort :)
        self._map_x = []
        self._map_y = []
        for wp in self.waypoints:
            self._map_x.append(wp.pose.pose.position.x)
            self._map_y.append(wp.pose.pose.position.y)
        self._waypoint_tree = KDTree(zip(self._map_x,self._map_y))


        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.bbox_image_pub = rospy.Publisher('/detector_image', Image, queue_size=1)


        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        """
        # shouldn't change during drive
        self.lights_pos_x = []
        self.lights_pos_y = []
        """

        # simulator only (not fed in real car)
        self.states = []

        self.map_x = None
        self.map_y = None

        rospy.spin()


    def pose_cb(self, msg):
        # msg is a PoseStamped
        self.pose = msg


    def traffic_cb(self, msg):
        """ a TrafficLightArray of TrafficLight messages
            see tl_detector/light_publisher for examples
        """
        self.lights = msg.lights  # store for later
        
        '''
        if self.DEBUG_USE_TRUTH:
            # only need the first time through
            if(self.lights_pos_x == None):
                for l in self.lights:
                    ps = PoseStamped()
                    ps = l.pose  # PoseStamped
                    self.lights_pos_x.append(ps.pose.position.x)
                    self.lights_pos_y.append(ps.pose.position.y)
            
            # constantly updating light colors - TODO: use until we get camera classifier working
            self.states = []
            for l in self.lights:
                self.states.append(l.state)  # uint8 of RED(0), YELLOW(1), GREEN(2), UNKNOWN(4)
            #rospy.loginfo('@_3 traffic_cb %s %s', self.lights_pos_x, self.lights_pos_y)
        '''



    def image_cb_test(self, msg):
        '''DEPRECATED'''
        self.camera_image = msg
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        if self.light_classifier:
            detector_image = self.light_classifier.get_classification(cv_image)
        #self.bbox_image_pub.publish(detector_image)


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        # FIXME -- temporarily disabling the THRESHOLD LOGIC, Uncomment later
        # As long as we are using truth data, assume it is perfect and no need
        # to count.  Too late to think about getting everything right! :)
#        if self.state != state:
#            self.state_count = 0
#            self.state = state
#        elif (self.state_count >= STATE_COUNT_THRESHOLD) and (light_wp < 100000):  # sometimes there's no light to publish (inf)
#            self.last_state = self.state
#            #light_wp = light_wp if state == TrafficLight.RED else -1  # TODO: put this back when we Classify
#            self.last_wp = light_wp
#            self.upcoming_red_light_pub.publish(Int32(light_wp))
#        else:
#            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
#        self.state_count += 1
        if light_wp >= 0:
            if state == TrafficLight.RED:
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            elif state == TrafficLight.YELLOW:
                # Are we getting too clever for our own good?
                # Negative index is a yellow light
                self.upcoming_red_light_pub.publish(Int32(-light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(-len(self._map_x)))

    def get_closest_waypoint(self, point):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
            .. or not :)
        Args:
            position (geometry_msgs/Point): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        # Udacity has led us astray!!! This isn't really a closes pair of points
        # problem, just a plain old nearest neighbor
        _, closest_wp = self._waypoint_tree.query((point.x, point.y))
        return closest_wp

    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify #DEPRECATED

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #if self.DEBUG_USE_TRUTH:
        #    # Get ground truth from topic
        #   return light.state
        #else:
        #Get classification
        return self.light_classifier.get_classification(cv_image)
            


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_wp = len(self.waypoints)

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):  # sometimes we get called before telemetry data
            car_wp = self.get_closest_waypoint(self.pose.pose.position)
        else:
            rospy.logwarn("Ack!  don't know where ego vehicle is in tl_detector/process_traffic_lights")
            return -1, TrafficLight.UNKNOWN

        # find the closest stop line ahead
        for line in stop_line_positions:
            line_point = Point()  # need to pass get_closest_waypoint a geometry_msgs/Point type
            line_point.x = line[0]
            line_point.y = line[1]
            stop_wp = self.get_closest_waypoint(line_point)  # nearest waypoint to this stop line
            if(stop_wp > car_wp-5):  # stop is ahead of us, not behind (waypoints ordered)
                if stop_wp < light_wp:  # here's a closer one
                    light_wp = stop_wp  # this is the one!  ...so far

        if(light_wp == len(self.waypoints)):
            # Didn't find one before course wraps around again
            return -1, TrafficLight.UNKNOWN

        # find the closest visible traffic light (if one exists)
        closest_light = -1
        closest_dist = float('inf')
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(len(self.lights)):
            dist = dl(self.waypoints[light_wp].pose.pose.position, self.lights[i].pose.pose.position)
            if(dist < closest_dist):
                closest_light = i
                closest_dist = dist

        if closest_light>-1:  # found one
            light = self.lights[closest_light]

        # rospy.loginfo('car wp: %s  light_wp: %s  light_index: %s  state: %s', car_wp, light_wp, closest_light, self.get_light_state())

        if light:
            state = self.get_light_state()
            return light_wp, state
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
