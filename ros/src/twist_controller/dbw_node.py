#!/usr/bin/env python

from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from threading import Lock
import rospy
import math

from twist_controller import Controller
from yaw_controller import YawController

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):

    # FIXME These are not configured specially, they are just hardcoded in two
    #       places
    CONTROL_GO = 0
    CONTROL_STOP = 1

    @staticmethod
    def mph2mps(mph):
        CONVERSION_FACTOR = 0.44704
        return mph * CONVERSION_FACTOR

    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self._controller = Controller(wheel_base,steer_ratio,0.0,max_lat_accel,max_steer_angle)

        self._brake_deadband = brake_deadband

        self._mode_lock = Lock()

        # Subscribe to all topics
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/control_mode', Int32, self.control_mode_cb)

        # Instance variables
        self.mode = self.CONTROL_STOP
        self.dbw_enabled = False
        self.target_twist = None
        self.current_velocity = None
        self.prev_throttle = 0
        self.prev_brake = 0

        # Main function
        self.loop()

    def current_velocity_cb(self, cv):
        '''
        cv is a TwistStamped command with values in meters per second
        for reference:
        100MPH = ~44 m/s
        50MPH = ~22 m/s
        '''

        self.current_velocity = cv.twist

        #rospy.loginfo('@_1 Curr velx %s yawdot %s', str(self.current_velocity.linear.x), str(self.current_velocity.angular.z))

    def dbw_enabled_cb(self, dbw):
        self.dbw_enabled = bool(dbw.data)

    def twist_cb(self, twistCommand):
        '''Pull in the TwistCommand.'''

        self.target_twist = twistCommand.twist
        #rospy.loginfo('@_1 Target velx %s yawdot %s', str(twistCommand.twist.linear.x), str(twistCommand.twist.angular.z))

    def control_mode_cb(self, msg):
        '''Get the mode message.'''

        self._mode_lock.acquire()
        self.mode = int(msg.data)
        self._mode_lock.release()

    def mode_go(self):
        '''Go control mode
        Use a PID to control brake/throttle/steering
        '''
        # Return if there is no target velocity to meet
        if self.target_twist and self.current_velocity:

            brake_error = 0
            velocity_error = (self.target_twist.linear.x - self.current_velocity.linear.x)

            # Normalise the brake error as a fraction of 25 mph
            # meaning that 25 mph overspeed is maximum this will assume the system will achieve
            if velocity_error < -self._brake_deadband:  # let it coast a little
                brake_error = -velocity_error   # work in positives
                self._controller.reset_speed()  # get rid of throttle I

            #rospy.loginfo('@_1 Curr velx %s target %s', str(self.current_velocity.linear.x), str(self.target_twist.linear.x))
            #rospy.loginfo('@_1 Computing PID vel_err %s & brk_err %s',str(velocity_error), str(brake_error))

            # Pass in the normalized velocity error, brake error and steering values to the controller
            t, b, s = self._controller.control(
                error_velocity = velocity_error,
                error_brake = brake_error,
                yaw_values=(self.target_twist.linear.x,
                            self.target_twist.angular.z,
                            self.current_velocity.linear.x)
                )

            # If the human driver is driving then reset the controller
            #if not self.dbw_enabled:
                #self._controller.reset_brake()
                #self._controller.reset_speed()

            # If there is no brake error, zero out any command
            #if brake_error == 0:
            #    b = 0
            # Scale b up by maximum torque as listed in the brake command message
            #b *= BrakeCmd.TORQUE_MAX

            rospy.loginfo('@_1 PID OUT: THR %s BRK %s YAW %s', str(t), str(b), str(s))

            # If the human driver is not driving then publish commands
            if self.dbw_enabled:
                self.publish(t,b,s)
            else:
                self._controller.reset_speed()
                self._controller.reset_brake()

    def mode_stop(self):
        '''Stop control mode
        Just hold the brake to the floor!
        '''
        self.publish(0, 1, 0)
        self._controller.reset_speed()

    def loop(self):
        rate = rospy.Rate(10) # NHz
        while not rospy.is_shutdown():
            if self.mode == self.CONTROL_GO:
                self.mode_go()
            else:
                self.mode_stop()
            rate.sleep()

    def publish(self, throttle, brake, steer):
        # Only publish with updated values. This is to
        # fix the lag issue
        if brake != self.prev_brake:
            self.prev_brake = brake
            bcmd = BrakeCmd()
            bcmd.enable = True
            #bcmd.pedal_cmd_type = BrakeCmd.CMD_PERCENT
            #bcmd.pedal_cmd = brake
            # scale by maximum torque with brake 0-1 to mimic CMD_PERCENT
            bcmd.pedal_cmd = brake * BrakeCmd.TORQUE_MAX
            bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE  # CMD_PERCENT seems to be ineffective
            self.brake_pub.publish(bcmd)

        # The simulator latches the commands, so we always need to check
        # whether or not to apply throttle
        if brake <= 0:
            #if throttle != self.prev_throttle:
            self.prev_throttle = throttle
            tcmd = ThrottleCmd()
            tcmd.enable = True
            tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
            tcmd.pedal_cmd = throttle
            self.throttle_pub.publish(tcmd)
            self.prev_throttle = throttle

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

if __name__ == '__main__':
    DBWNode()
