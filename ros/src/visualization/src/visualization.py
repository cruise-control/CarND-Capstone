#!/usr/bin/env python

import rospy
import tf

from geometry_msgs.msg import PoseStamped, Point
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight
from visualization_msgs.msg import Marker


# See following tutorial for reference
# http://library.isr.ist.utl.pt/docs/roswiki/rviz(2f)Tutorials(2f)Markers(3a20)Basic(20)Shapes.html
class Visualization(object):

    def __init__(self):
        rospy.init_node('visualization')
        #rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/final_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.tlight_cb)


        self.vis_pose_pub = rospy.Publisher('/pose_vis', Marker, queue_size=1)
        self.vis_light_pub = rospy.Publisher('/light_vis', Marker, queue_size=1)

        rospy.spin()


    def waypoints_cb(self, msg):
        if self.vis_pose_pub.get_num_connections() > 0:
            
            # Marker is a visualization msg for rviz
            marker = Marker()
            
            # Base link refers to vehicle coordinate system
            marker.header.frame_id = '/world'
            marker.header.stamp = rospy.Time.now()
            marker.id = 0
            marker.type = marker.SPHERE_LIST
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.ns = 'road'

            marker.action = marker.ADD
            #marker.pose.orientation.x = 0.
            #marker.pose.orientation.y = 0.
            #marker.pose.orientation.z = 0.
            #marker.pose.orientation.w = 1.0
            #marker.pose.position.x = 0. 
            #marker.pose.position.y = 0.
            #marker.pose.position.z = 0.
            
            # Set RGB-A channels
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Zero lifetime means it won't be deleted
            marker.lifetime = rospy.Duration(1)
            
            # Read out waypoint positions from lane and publish for rviz
            for waypoint in msg.waypoints:
                p = Point()
                p = waypoint.pose.pose.position
                marker.points.append(p)
            self.vis_pose_pub.publish(marker)

    def tlight_cb(self, msg):
        if self.vis_light_pub.get_num_connections() > 0:
            
            # Marker is a visualization msg for rviz
            marker = Marker()
            
            # Base link refers to vehicle coordinate system
            marker.header.frame_id = '/world'
            marker.header.stamp = rospy.Time.now()
            marker.id = 1
            marker.type = marker.SPHERE_LIST
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.ns = "road"

            marker.action = marker.ADD
            #marker.pose.orientation.x = 0.
            #marker.pose.orientation.y = 0.
            #marker.pose.orientation.z = 0.
            #marker.pose.orientation.w = 1.0
            #marker.pose.position.x = 0. 
            #marker.pose.position.y = 0.
            #marker.pose.position.z = 0.
            
            # Set RGB-A channels
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Zero lifetime means it won't be deleted
            marker.lifetime = rospy.Duration(0)
            
            # Read out waypoint positions from lane and publish for rviz
            for light in msg.lights:
                p = Point()
                p = light.pose.pose.position
                marker.points.append(p)
            self.vis_light_pub.publish(marker)

if __name__ == '__main__':
    try:
        Visualization()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start visualization node.')