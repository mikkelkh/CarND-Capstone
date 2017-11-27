#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
import math

from twist_controller import Controller

from tf.transformations import euler_from_quaternion

from styx_msgs.msg import Lane

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
    def __init__(self):
        rospy.init_node('dbw_node')

        rospy.logwarn("DBWNode started")

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

        # TODO: Create `TwistController` object
        # self.controller = TwistController(<Arguments you wish to provide>)

        # TODO: Subscribe to all the topics you need to
        self.current_velocity = None
        self.proposed_velocity = None
        self.dbw_enabled = False

        self.sub1 = rospy.Subscriber("/twist_cmd", TwistStamped, self.twist_cmd_callback, queue_size=1)
        self.sub2 = rospy.Subscriber("/current_velocity", TwistStamped, self.current_velocity_callback, queue_size=1)
        self.sub3 = rospy.Subscriber("/vehicle/dbw_enabled", Bool, self.dbw_enabled_callback, queue_size=1)

        self.sub4 = rospy.Subscriber("/current_pose", PoseStamped, self.current_pose_callback, queue_size=1)
        self.sub5 = rospy.Subscriber("/base_waypoints", Lane, self.base_waypoints_callback, queue_size=1)

        self.loop()

    def get_yaw(self, orientation_q):
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion (orientation_list)
        # roll and pitch are always 0 anyways ...
        return yaw

    def base_waypoints_callback(self, msg):
        num_waypoints = len(msg.waypoints)
        rospy.logwarn("num_waypoints=%d", num_waypoints) # 10902
        #for i in range(500,510):
        for i in range(0,20):
            wp = msg.waypoints[i]
            x = wp.pose.pose.position.x
            y = wp.pose.pose.position.y
            z = wp.pose.pose.position.z
            yaw = self.get_yaw(wp.pose.pose.orientation)
            theta_z = wp.twist.twist.angular.z

            # Note that the coordinates for linear velocity are vehicle-centered 
            # So only the x-direction linear velocity should be nonzero.
            vx = wp.twist.twist.linear.x
            # all zero
            #vy = wp.twist.twist.linear.y
            #vz = wp.twist.twist.linear.z
            #theta_x = wp.twist.twist.angular.x
            #theta_y = wp.twist.twist.angular.y
            #theta_z = wp.twist.twist.angular.z
            rospy.logwarn("wp[%d] x=%f y=%f z=%f yaw=%f theta_z=%f vx=%f", i, x, y, z, yaw, theta_z, vx)

    def current_pose_callback(self, msg):
        self.ego_x = msg.pose.position.x
        self.ego_y = msg.pose.position.y
        self.ego_z = msg.pose.position.z
        yaw = self.get_yaw(msg.pose.orientation)
        #rospy.logwarn("ego x=%f y=%f z=%f yaw=%f", self.ego_x, self.ego_y, self.ego_z, yaw)

    def twist_cmd_callback(self, msg):
        self.proposed_velocity = msg.twist # linear and angular

    def current_velocity_callback(self, msg):
        self.current_velocity = msg.twist # linear and angular

    def dbw_enabled_callback(self, msg):
        self.dbw_enabled = msg.data

    def loop(self):
        rospy.logwarn("DBWNode loop started")
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # rospy.logwarn("DBWNode looping\n")
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            # throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            # if <dbw is enabled>:
            #   self.publish(throttle, brake, steer)
            throttle, brake, steer = 1, 0, 0 # Just check if we can move the car ...
            throttle, brake, steer = 1, 0, -0.02 # Just check if we can move the car ...
            self.publish(throttle, brake, steer)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
