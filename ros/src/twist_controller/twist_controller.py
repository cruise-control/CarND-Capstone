from yaw_controller import YawController
from pid import PID

# https://en.wikipedia.org/wiki/PID_controller
# Ziegler-Nichols with Relay method, target velocity 15 m/s in sim
# observed in sim:
#    3.5 seconds 0-15 m/s at 100% throttle (1/4 period), then
#    40  seconds to coast to a stop with 0% throttle
#    2.1 seconds to stop from 15 m/s with 100% brake (1/4 period)

pi = 3.141592
# Speed
a = 15.0  # amplitude of process variable (effect) max speed in m/s
b =  1.0  # amplitude of control variable (cause)
Ku = (4.0*b)/(pi*a)  # ultimate gain, where oscillation begins
Tu = 4*3.5  # seconds, observed full period of oscillation
# ratios for Z-G PID
#speed_p = 0.6*Ku
#speed_i = 1.2*(Ku/Tu)
#speed_d = 3*Ku*Tu/40
# Manually Twiddled
speed_p = 0.2
speed_i = 0.05
speed_d = 0.01

# Brake
a = 3412.0  # amplitude of process variable (effect) max brake torque from BrakeCmd
b =  1.0  # amplitude of control variable (cause)
Ku = (4.0*b)/(pi*a)  # ultimate gain, where oscillation begins
Tu = 4*2.1  # seconds, observed full period of oscillation
# ratios for Z-G PD
#brake_p = 0.6*Ku
#brake_i = 1.2*(Ku/Tu)
#brake_d = 3*Ku*Tu/40
# Manually Twiddled
brake_p = 0.0001
brake_i = 0.0
brake_d = 0.001

pid_min = 0.0
pid_max = 1.0

class Controller(object):
    throttle = 0
    brake = 0
    steer = 0
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):

        self.pid_speed = PID(speed_p, speed_i, speed_d, pid_min, pid_max)
        self.pid_brake = PID(brake_p, brake_i, brake_d, pid_min, pid_max)

        self.yaw_controller = YawController(wheel_base,steer_ratio,0.0,max_lat_accel,max_steer_angle)


    def reset_speed(self):
        self.pid_speed.reset()

    def reset_brake(self):
        self.pid_brake.reset()


    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        for key in kwargs:
            if key is 'error_velocity':
                self.v_err = kwargs[key]
                # Pass in error and time step
                self.throttle = self.pid_speed.step(self.v_err,0.1)
            if key is 'error_brake':
                self.b_err = kwargs[key]
                # Pass in error and time step
                self.brake = self.pid_brake.step(self.b_err,0.1)
            if key is 'yaw_values':
                self.steer = self.yaw_controller.get_steering(
                    linear_velocity=kwargs[key][0],angular_velocity=kwargs[key][1],current_velocity=kwargs[key][2]
                    )

        return self.throttle, self.brake, self.steer
