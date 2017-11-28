#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

from tf.transformations import euler_from_quaternion

import math

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.ego_x = None
        self.ego_y = None
        self.ego_z = None
        self.ego_yaw = None
        self.frame_id = None
        self.msg_seq = 1

        self.frame_id = None
        self.waypoints = None
        self.num_waypoints = None
        self.closest_wp_index = None
        self.closest_wp_dist = None

        #rospy.spin()
        self.loop()

    def pose_cb(self, msg):
        # TODO: Implement
        self.ego_x = msg.pose.position.x
        self.ego_y = msg.pose.position.y
        self.ego_z = msg.pose.position.z
        self.ego_yaw = self.get_yaw(msg.pose.orientation)
        self.frame_id = msg.header.frame_id
        #rospy.logwarn("WP_UPDATER: ego x=%f y=%f z=%f yaw=%f", self.ego_x, self.ego_y, self.ego_z, self.ego_yaw)

    def waypoints_cb(self, msg):
        # TODO: Implement
        self.waypoints = msg.waypoints
        self.num_waypoints = len(self.waypoints)
        rospy.logwarn("WP_UPDATER: num_waypoints=%d", self.num_waypoints) # 10902

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def loop(self):
        rospy.logwarn("WP_UPDATER loop started")
        rate = rospy.Rate(10) # 50Hz
        while not rospy.is_shutdown():
            if self.waypoints is not None and self.ego_x is not None:
                if self.closest_wp_index is None:
                    wp_min = 0
                    wp_max = self.num_waypoints - 1
                    self.closest_wp_index = self.get_closest_wp_index(wp_min, wp_max) 
                else:
                    wp_min = self.closest_wp_index - LOOKAHEAD_WPS
                    wp_max = self.closest_wp_index + LOOKAHEAD_WPS
                    self.closest_wp_index = self.get_closest_wp_index(wp_min, wp_max) 
                #self.publish(throttle, brake, steer)
                rospy.logwarn("WP_UPDATER closest_wp_index=%d closest_wp_dist=%f", self.closest_wp_index, self.closest_wp_dist)

                final_waypoints = []
                for i in range(LOOKAHEAD_WPS):
                    final_waypoints.append(self.waypoints[ (self.closest_wp_index + i) % self.num_waypoints ])

                lane_msg = Lane()
                lane_msg.header.seq = self.msg_seq
                lane_msg.header.frame_id = self.frame_id
                lane_msg.header.stamp = rospy.Time.now()
                lane_msg.waypoints = final_waypoints
                self.final_waypoints_pub.publish(lane_msg)
                self.msg_seq += 1
            rate.sleep()


    # --------------------------------------------------------------------------------
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    # --------------------------------------------------------------------------------

    def get_yaw(self, orientation_q):
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion (orientation_list)
        # roll and pitch are always 0 anyways ...
        return yaw

    def transform_into_ego_coord(self, x, y):
        # translation
        x = x - self.ego_x
        y = y - self.ego_x
        # rotation(-yaw)
        #  [ cos(-yaw)    -sin(-yaw) ] [ x ]
        #  [ sin(-yaw)     cos(-yaw) ] [ y ]
        x =   x * math.cos(self.ego_yaw) + y * math.sin(self.ego_yaw)
        y = - x * math.sin(self.ego_yaw) + y * math.cos(self.ego_yaw)
        return x, y

    def is_wp_behind_ego(self, wp_index):
        wp = self.waypoints[wp_index]
        wp_x = wp.pose.pose.position.x
        wp_y = wp.pose.pose.position.y
        ego_coord_wp_x, ego_coord_wp_y = self.transform_into_ego_coord(wp_x, wp_y)
        if ego_coord_wp_x < 0:
            return True
        else:
            return False

    def get_closest_wp_index(self, wp1, wp2): 
        closest_dist = 1e10
        closest_wp_index = -1
        for i in range(wp1, wp2+1):
            wp = self.waypoints[i % self.num_waypoints]
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            wp_z = wp.pose.pose.position.z
            dist = math.sqrt( (self.ego_x - wp_x)**2 + (self.ego_y - wp_y)**2 + (self.ego_z - wp_z)**2 )
            if dist < closest_dist:
                closest_dist = dist
                closest_wp_index = i
        if self.is_wp_behind_ego(closest_wp_index) is True:
            closest_wp_index += 1
            if closest_wp_index == self.num_waypoints:
                closest_wp_index = 0

        self.closest_wp_dist = closest_dist # just for logging / check purposes
        return closest_wp_index


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
