from yaw_controller import YawController
from pid import PID

class Controller(object):
    throttle = 0
    brake = 0
    steer = 0
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        
        # TODO Choose proper values for here
        self.pid_speed = PID(4,0.1,0.5,0,1)
        self.pid_brake = PID(8,0.1,0.5,0.1)
        
        self.yaw_controller = YawController(wheel_base,steer_ratio,0.0,max_lat_accel,max_steer_angle)
        
        pass

    def reset(self):
        self.pid_speed.reset()
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
                if self.b_err == 0:
                    self.brake = 0
                else:
                    # Pass in error and time step
                    self.brake = self.pid_brake.step(self.b_err,0.1)                
            if key is 'yaw_values':
                self.steer = self.yaw_controller.get_steering(
                    linear_velocity=kwargs[key][0],angular_velocity=kwargs[key][1],current_velocity=kwargs[key][2]
                    )

        return self.throttle, self.brake, self.steer
