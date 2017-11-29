import rospy
import math

class SpeedController(object):
    def __init__(self, wheel_radius, vehicle_mass, accel_limit, decel_limit, brake_deadband):
        self.wheel_radius = wheel_radius
        self.vehicle_mass = vehicle_mass
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.brake_deadband = brake_deadband

        self.max_torque = vehicle_mass * accel_limit * wheel_radius

    def get_throttle_brake(self, proposed_linear_vel, current_linear_vel, time_duration):
        if time_duration > 0:
            accel = (proposed_linear_vel - current_linear_vel) / time_duration

            accel = min(accel, self.accel_limit)
            accel = max(accel, self.decel_limit)

            torque = self.vehicle_mass * accel * self.wheel_radius
            throttle = torque / self.max_torque

            brake = 0.
            # deceleration => brake cmd
            if throttle < -self.brake_deadband:
                brake = math.fabs(torque)

            throttle = max(throttle, 0.)
            #rospy.logwarn("DBWNode: target_vel=%f current_vel=%f desired_acc=%f", proposed_linear_vel, current_linear_vel, desired_acc)
            rospy.logwarn("DBWNode: throttle=%f brake=%f", throttle, brake)
            return throttle, brake
        else:
            return 0.0, 0.0
