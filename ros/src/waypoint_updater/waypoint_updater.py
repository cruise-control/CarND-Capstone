#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray
from std_msgs.msg import Int32
from geometry_msgs.msg import TwistStamped
import Utility
import math
import time
import numpy as np
from scipy.spatial import KDTree
from threading import Lock
from copy import deepcopy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

'''
Notes:
The Waypoint position information is contained in the Pose position and orientation of the
Lane data structure. The rest of the structures appear to be blanked out except for twist - linear velocity
- x which sometimes has a value set

Steps:
1 Get the current vehicle location wrt to the provided 'map' or waypoints
2 Create a simple trajectory that will move the vehicle along the path to the next waypoint
3 Publish this next trajectory set. (after the vehicle pose comes in)
'''

PLAN_INTERVAL = 2 # Seconds per path plan step
SPEED_LIMIT= 15
MAX_DECEL = 1.5

# FIXME! Magic number alert!
# FIXME - These belong in the world model
# Timed these, very approximate
RED_LIGHT_TIME = 20.0
YELLOW_LIGHT_TIME = 1.95
GREEN_LIGHT_TIME = 4.0
STICKY_APPROACH_HOLD_TIMEOUT = 3.5 # Unstick APPROACH or HOLD timeout
RED = 2
YELLOW = 1
GREEN = 0

# FIXME! More magic numbers!
HOLD_TOL = 3 # Command hold this far before (meters)
PAST_TOL = 3 # Tolate going past by this much (meters), FIXME does this make sense?
LYEL_TOL = HOLD_TOL/1.3 # If red transitions and in this margin, will be a late yellow (tunable parameter)

class World(object):

    @staticmethod
    def distance(x0, y0, x1, y1):
        return math.sqrt((x0-x1)**2 + (y0-y1)**2)


    def __init__(self):
        rospy.loginfo("Initializing world model")

        # Red light locations
        self._red_light_lock = Lock()
        self._red_light_wp = None
        self._red_light_time = None

        # Map waypoints
        self._map_x = []
        self._map_y = []
        self._map_s = []
        self._map_kd_tree = None

        # Block until we get the road waypoints
        waypoints = rospy.wait_for_message('/base_waypoints', Lane)
        self._update_map_waypoints(waypoints.waypoints)
        self.waypoints = waypoints.waypoints

        # Subscribe to traffic waypoints
        rospy.Subscriber('/traffic_waypoint', Int32, self._traffic_cb)


    def _update_map_waypoints(self, waypoints):
        for wp in waypoints:
            self._map_x.append(wp.pose.pose.position.x)
            self._map_y.append(wp.pose.pose.position.y)

        # Generate the s map
        self._map_s = Utility.generateMapS(self._map_x, self._map_y)

        # KDTree for nearest neighbor searches
        self._map_kd_tree = KDTree(zip(self._map_x, self._map_y))


    def _traffic_cb(self, msg):
        self._red_light_lock.acquire()
        self._red_light_wp = int(msg.data)
        self._red_light_time = rospy.Time.now()
        self._red_light_lock.release()


    def _road_distance(self, waypoints, wp1, wp2):
        '''
        Get the distance between waypoints indexed by wp1 and wp2.
        This returns the sum of the euclidean distance between
        all intermediate waypoints
        '''
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


    def closest_waypoint(self, x, y):
        ''' Find closest waypoint to given x,y
        '''
        d, idx = self._map_kd_tree.query((x,y))
        return idx


    def next_waypoint(self, x, y, theta):
        '''
        Get the next waypoint in front of the vehicle
        '''
        idx = self.closest_waypoint(x,y)
        mx = self._map_x[idx]
        my = self._map_y[idx]
        heading = math.atan2((my-y),(mx-x))
        angle = abs(theta - heading)
        if angle > math.pi / 4:
            idx += 1
        idx %= len(self._map_x)
        return idx


    def cartesian_to_frenet(self, x, y, theta):
        '''
        Convert from x / y coordinates to Frenet coordinates
        '''

        # FIXME - this really doesn't need the vehicle heading if we pick
        # the segment a little more thoughtfully, GHP

        # Get next waypoint index and previous one. Handle wrap around
        next_idx = self.next_waypoint(x,y,theta)
        prev_idx = next_idx - 1
        if next_idx == 0:
            prev_idx = len(self._map_x) - 1

        # Find project of x onto n
        nx = self._map_x[next_idx] - self._map_x[prev_idx]
        ny = self._map_y[next_idx] - self._map_y[prev_idx]
        xx = x -  self._map_x[prev_idx]
        yy = y -  self._map_y[prev_idx]

        proj_norm = (xx*nx + yy*ny)/(nx*nx + ny*ny)
        proj_x = proj_norm * nx
        proj_y = proj_norm * ny

        frenet_d = self.distance(xx,yy,proj_x,proj_y)

        # See if d should be positive or negative
        c_x = 1000-self._map_x[prev_idx]
        c_y = 2000-self._map_y[prev_idx]
        c_pos = self.distance(c_x,c_y,xx,yy)
        c_ref = self.distance(c_x,c_y,proj_x,proj_y)

        if(c_pos <= c_ref):
            frenet_d *= -1

        frenet_s = 0
        for i in range(0,prev_idx):
            frenet_s += self.distance(self._map_x[i],self._map_y[i],self._map_x[i+1],self._map_y[i+1])

        frenet_s += self.distance(0,0,proj_x,proj_y)

        # This final distance is weird... return it and offset X
        return frenet_s,frenet_d


    def frenet_to_cartesian(self, s, d):
        prev_idx = -1
        while(s > self._map_s[prev_idx+1] and prev_idx < len(self._map_s)-2):
            prev_idx += 1;
        wp = (prev_idx+1) % len(self._map_s)

        heading = math.atan2(self._map_y[wp]-self._map_y[prev_idx], self._map_x[wp]-self._map_x[prev_idx])
        seg_s = s - self._map_s[prev_idx]

        seg_x = self._map_x[prev_idx] + seg_s * math.cos(heading)
        seg_y = self._map_y[prev_idx] + seg_s * math.sin(heading)

        p_heading = heading - math.pi/2

        x = seg_x + d*math.cos(p_heading)
        y = seg_y + d*math.sin(p_heading)

        return x, y


    def traffic_lights_in_range(self, x, y, yaw, rng):
        '''Return upcoming traffic lights, return its cartesian position

        Starting from position <x,y>, look ahead <rng> meters for traffic lights
        '''
        # Acquire current red light idx
        self._red_light_lock.acquire()
        wp_idx = self._red_light_wp
        wp_time = self._red_light_time
        self._red_light_lock.release()

        if wp_idx is None:
            return None

        # Reject invalid waypoints
        if wp_idx >= len(self._map_x) or wp_idx <= -len(self._map_x):
            return None

        # Using negative indices for yellow lights
        # For now, treat them the same
        if wp_idx < 0:
            wp_state = YELLOW
            wp_idx = -wp_idx
        else:
            wp_state = RED

        rospy.loginfo('red_light_wp_idx: %d', wp_idx)
        wp_x = self._map_x[wp_idx]
        wp_y = self._map_y[wp_idx]

        # Convert origin position to frenet, and look out rng meters
        s, d = self.cartesian_to_frenet(x, y, yaw)
        end_x, end_y = self.frenet_to_cartesian(s+rng, 0)
        end_idx = self.closest_waypoint(end_x, end_y)

        # If the traffic light is within range, return it
        if wp_idx <= end_idx:
            return wp_x, wp_y, wp_state, wp_time
        else:
            return None


class WaypointUpdater(object):

    # Vehicle states
    START          = 0
    CRUISE         = 1
    APPROACH       = 2
    HOLD           = 3
    EMERGENCY_STOP = 4

    # Control modes
    CONTROL_GO = 0
    CONTROL_STOP = 1

    def __init__(self, world):

        rospy.Subscriber('/current_pose', PoseStamped, self._pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self._current_velocity_cb)
        #rospy.Subscriber('/obstacle_waypoint', Int32, self._obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        self.control_mode_pub = rospy.Publisher('/control_mode', Int32, queue_size=1)

        # Road data
        self.world = world

        # Ego data
        self.POSE_HIST_SZ = 2 # Save last 2 poses received

        # Pose history requires a lock because it us updated by the subscriber
        # callback
        self._pose_lock = Lock()
        self._pose_hist = []

        # Current velocity requires a lock because it is updated by the
        # subscriber callback
        self._current_vel_lock = Lock()
        self._current_vel = None

        # These are only referenced synchronously and are thus threadsafe
        # (no lock)
        self.pose = None
        self.heading = None
        self.s = None
        self.d = None
        self.speed = None

        # Plan data
        self.generated_waypoints = []
        self.hold_pos = None
        self.hold_state = None
        self.hold_time = None

        # FSM
        self._states = {self.START: self.st_start,
                        self.CRUISE: self.st_cruise,
                        self.APPROACH: self.st_approach,
                        self.HOLD: self.st_hold}
        self._state = self.START

        # Initialize system control mode
        self.control_mode_pub.publish(Int32(self.CONTROL_STOP))


    def _pose_hist_append(self, pose):
        self._pose_lock.acquire()
        if len(self._pose_hist) >= self.POSE_HIST_SZ:
            self._pose_hist.pop(0)
        self._pose_hist.append(pose)
        self._pose_lock.release()


    def _pose_cb(self, msg):
        self._pose_hist_append(msg.pose)


    def _current_velocity_cb(self, msg):
        self._current_vel_lock.acquire()
        self._current_vel = msg  # is a TwistStamped
        self._current_vel_lock.release()


    def _obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass


    def vel(self):
        current_vel = 0  # presume stopped if no info
        self._current_vel_lock.acquire()
        current_vel = self._current_vel
        self._current_vel_lock.release()
        return current_vel


    def pose_hist(self, num=1):
        '''Getter for vehicle pose history

        Return the last <num> poses
        '''
        poses = []
        if num>0:
            self._pose_lock.acquire()
            num = min(num, len(self._pose_hist))
            if num>0:
                poses.extend(deepcopy(self._pose_hist[-num:]))
            self._pose_lock.release()

        return poses


    def getSplinePath(self, pose, heading, speed=0.0):

        # Get frenet coordinates
        s, d = self.world.cartesian_to_frenet(pose.position.x, pose.position.y, heading)

        # Spline control point parameters
        # We know that the waypoint_follower node attempts to track a point 2
        # seconds out and to account for the interval in our plan step, we
        # plan at least 2+PLAN_INTERVAL seconds out
        ctrl_spline_length = (2+PLAN_INTERVAL) * max(speed, SPEED_LIMIT)
        ctrl_pt_spacing = 5 # meter spacing between control points
        n_ctrl_pts = int(math.ceil(ctrl_spline_length / ctrl_pt_spacing))

        # Setup the Coordinates to get the spline between
        # Append n_ctrl_points at increments of ctrl_pt_spacing
        # We no longer use the current position as a control point because 
        # the waypoint follower always targets a point ahead in the trajectory.
        # By keeping the first point ahead, we are further attempting to ensure
        # that if we get off track, we won't overcompensate back when because
        # the follower is using a point at our s (frenet) to calculate 
        # curvature, though this isn't guaranteed to work all the time.
        # TODO: It would be better to abstract this away here and compensate
        #       in the waypoint follower node
        px = []
        py = []
        for i in range(n_ctrl_pts):
            plan_s = s + speed*0.5 + i*ctrl_pt_spacing
            plan_d = 0
            plan_x, plan_y = self.world.frenet_to_cartesian(plan_s, plan_d)
            px.append(plan_x)
            py.append(plan_y)

        # Fit the spline
        tkc = Utility.getSplineCoeffs(px, py)

        pts = []

        sp = np.arange(0, tkc.s[-1], 1)
        index = 0
        for i in sp:
            # Fit spline point
            ix, iy = Utility.fitX(i, tkc)

            # Generate waypoint
            wp = Waypoint()
            wp.pose.pose.orientation = Quaternion(*Utility.getQuaternion(0,0,tkc.calc_yaw(i)))
            wp.pose.pose.position.x = ix
            wp.pose.pose.position.y = iy
            wp.pose.header.seq = i
            wp.pose.header.stamp = rospy.Time(0)
            wp.pose.header.frame_id = '/world'
            wp.twist.header.seq = i
            wp.twist.header.stamp = rospy.Time(0)
            wp.twist.header.frame_id = '/world'
            pts.append(wp)
            self.set_waypoint_velocity(pts,index,0)
            index += 1

        # Save generated points
        self.generated_waypoints = pts


    def update_state(self):

        # Get last ego pose
        poses = self.pose_hist()
        self.pose = poses[0]
        quaternion = (
            self.pose.orientation.x,
            self.pose.orientation.y,
            self.pose.orientation.z,
            self.pose.orientation.w)
        self.heading = Utility.getHeading(quaternion)
        self.s, self.d = self.world.cartesian_to_frenet(self.pose.position.x, self.pose.position.y, self.heading)

        # Get vehicle velocity
        my_vel = self.vel()
        if my_vel is not None:  # no _current_velocity_cb yet
            self.speed = math.sqrt(my_vel.twist.linear.x**2 + my_vel.twist.linear.y**2)


    def loop(self):

        LOOP_RATE = 2 # Hz

        # We only run the spline planner ever PLAN_INTERVAL seconds,
        # which necessitates this logic
        loop_ctr = 0
        plan_cnt = int(LOOP_RATE*PLAN_INTERVAL)

        # Main node loop
        rate = rospy.Rate(LOOP_RATE) # N Hz
        while not rospy.is_shutdown():
            # Determine if we need to run the spline planner
            if loop_ctr == 0:
                plan = True
            else:
                plan = False

            # Run the node logic
            self.step(plan)

            # Increment loop counter
            loop_ctr += 1
            loop_ctr %= plan_cnt

            # Wait till next loop
            rate.sleep()

    
    def sense(self):
        '''Update world state
        '''
        # Look for upcoming traffic lights
        # In the real world this would be a more generalized world
        # model/obstacle map update
        # FIXME! Magic number alert! Get current velocity and look ahead
        #        far enough that we will have time to stop
        LOOKAHEAD_DIST = 100 # meters

        s_back = self.s-PAST_TOL
        orig_x, orig_y = self.world.frenet_to_cartesian(s_back, 0)
        light = self.world.traffic_lights_in_range(orig_x, orig_y, self.heading, LOOKAHEAD_DIST)
        if light is not None:
            # Find a point some margin behind the stop position to target
            hold_x = light[0]
            hold_y = light[1]
            stop_s, _ = self.world.cartesian_to_frenet(hold_x, hold_y, self.heading)
            self.hold_pos = self.world.frenet_to_cartesian(stop_s-HOLD_TOL/2, _)
            self.hold_state = light[2]
            # Save time of arrival of the message, but only when light changes
            if self.hold_time is None:
                self.hold_time = light[3]
        else:
            self.hold_pos = None
            # Note: The below logic is inherently unsafe for real-world driving as upon
            # having X seconds of invalid traffic light observations, it will result in
            # the vehicle transitioning into CRUISE mode. This is ok for the scope of this
            # project
            
            # If there is a prior hold time
            if self.hold_time:
                # Only clear the lights when there is a safe state to do so in
                # Meaning, if we were previously approaching a light or already stopped
                # don't transition to CRUISE if we get an UNKNOWN light classification.
                # Only do that on positive classification or after a timeout in the event the 
                # vehicle is not providing any classification information.
                
                # Allow state clearing if not getting any classifications for X seconds
                timeout = (rospy.Time.now() - self.hold_time).to_sec() > STICKY_APPROACH_HOLD_TIMEOUT 
                
                if self._state != self.APPROACH and self._state != self.HOLD or timeout:
                    rospy.loginfo('Sticky Approach / Hold timeout!!!')
                    self.hold_pos = None
                    self.hold_state = None
                    self.hold_time = None




    def plan(self):
        '''Plan the next vehicle actions
        '''
        # Set the path we must follow, i.e. spline the road
        # This currently doesn't take into account any obstacles, because we
        # are not taking any evasive action other than to slow and/or stop
        # Velocity profile is updated in the robot state functions, i.e. the
        # planner tells the vehicle where to go, the state function decides
        # how to do it.
        self.getSplinePath(self.pose, self.heading)


    def act(self):
        # Choose the correct state plan execute function
        if self.hold_pos is None:
            self._state = self.CRUISE
        else:
            hold_s, _ = self.world.cartesian_to_frenet(self.hold_pos[0], self.hold_pos[1], self.heading)
            dtg = hold_s - self.s;

            if self.speed > 0:
                time_to_intercept = dtg/self.speed
            else:  # speed of 0, divide by 0
                time_to_intercept = 10

            if self.hold_state == RED:
                time_to_red = -float('inf')
            else:
                yellow_time_elapsed = rospy.Time.now()-self.hold_time
                time_to_red = YELLOW_LIGHT_TIME - yellow_time_elapsed.to_sec()
                # Where will the vehicle be at the red light transition?
                pos_at_red_transition = self.s + self.speed * time_to_red
                # Will that result in a late yellow?
                if pos_at_red_transition >= hold_s - LYEL_TOL:
                    # Yes, then set time to red as the time to intercept
                    time_to_red = time_to_intercept
                    rospy.loginfo('Late Yellow!!!')

            if dtg < -PAST_TOL or time_to_intercept < time_to_red:
                self._state = self.CRUISE
            elif dtg > HOLD_TOL or time_to_intercept == time_to_red:
                self._state = self.APPROACH
            else:
                self._state = self.HOLD

        # Call current state function
        self._states[self._state]()


    def st_start(self):
        # This is just a stub so that the vehicle will do nothing if
        # it hasn't chosen a starting state yet
        pass


    def st_cruise(self):
        self.control_mode_pub.publish(Int32(self.CONTROL_GO))

        # Set velocity profile
        self.set_constant_velocity_profile(SPEED_LIMIT)

        # Create a new lane message type and publish it
        l = Lane()
        l.header.frame_id = '/world'
        l.header.stamp = rospy.Time(0)
        l.waypoints = self.generated_waypoints
        self.final_waypoints_pub.publish(l)


    def st_approach(self):
        self.control_mode_pub.publish(Int32(self.CONTROL_GO))

        # Set velocity profile
        self.set_decelerate_profile(self.hold_pos)

        # Create a new lane message type and publish it
        l = Lane()
        l.header.frame_id = '/world'
        l.header.stamp = rospy.Time(0)
        l.waypoints = self.generated_waypoints
        self.final_waypoints_pub.publish(l)


    def st_hold(self):
        self.control_mode_pub.publish(Int32(self.CONTROL_STOP))


    def _print_waypoints(self):
        for i in range(len(self.generated_waypoints)):
            wp = self.generated_waypoints[i]
            rospy.loginfo('%d: %f, %f, %f',
                i,
                wp.pose.pose.position.x,
                wp.pose.pose.position.y,
                wp.twist.twist.linear.x)

    def step(self, do_plan=True):
        '''Main vehicle function
        '''
        if len(self._pose_hist)>=self.POSE_HIST_SZ:

            self.update_state()
            self.sense()
            if do_plan:
                self.plan()
            self.act()


    def set_constant_velocity_profile(self, vel):
        for i in range(len(self.generated_waypoints)):
            self.set_waypoint_velocity(self.generated_waypoints, i, vel)


    def set_decelerate_profile(self, stop_pos):
        # Decelerate the generated_waypoints like waypoint_loader/decelerate()
        # first, find the generated_waypoint closest to the stop line
        stop_x = stop_pos[0]
        stop_y = stop_pos[1]

        closest_wp = -1
        closest_dist = float('inf')
        for i in range(len(self.generated_waypoints)):
            dist = World.distance(self.generated_waypoints[i].pose.pose.position.x,
                                  self.generated_waypoints[i].pose.pose.position.y,
                                  stop_x,
                                  stop_y)
            if(dist<closest_dist):
                closest_wp = i
                closest_dist = dist


        # What if the last generated point is closest but it is still far from
        # the light?
        if closest_wp < len(self.generated_waypoints):

            # then, come to a stop at the end waypoint/stop line
            for i in range(0, closest_wp):
                dist = World.distance(self.generated_waypoints[i].pose.pose.position.x,
                                      self.generated_waypoints[i].pose.pose.position.y,
                                      self.generated_waypoints[closest_wp].pose.pose.position.x,
                                      self.generated_waypoints[closest_wp].pose.pose.position.y)

                # Assume applying max decel to the stop point
                # Time from this point to the stop
                t = math.sqrt(dist/MAX_DECEL)

                # Velocity that that time
                vel = 2*MAX_DECEL*t

                # Cap velocity at the speed limit
                vel = min(vel, SPEED_LIMIT)

                self.set_waypoint_velocity(self.generated_waypoints, i, vel)
            # The for loop may not be entered, guard against it.
            if i is not None:
                # i is now closest waypoint, always set to 0.0 velocity
                self.set_waypoint_velocity(self.generated_waypoints, i, 0)


    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity


if __name__ == '__main__':
    try:
        rospy.init_node('waypoint_updater')

        # The world constructor blocks waiting for the road waypoints
        world = World()

        # We have constructed a model of the world, get ready to start driving!
        wpu = WaypointUpdater(world)

        # .. and go
        wpu.loop()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
