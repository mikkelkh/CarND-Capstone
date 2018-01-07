#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import yaml
import numpy as np
import os

import math

import math

STATE_COUNT_THRESHOLD = 3

SAVE_IMAGE = False
TRAINING = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        

        config_string = rospy.get_param("/traffic_light_config")
        
        # Folder of where images are stored. 
        self.dirData = rospy.core.rospkg.environment.get_test_results_dir()
        self.config = yaml.load(config_string)

        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_line_waypoints = []

        
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.idx = 0
        self.waypointlist = np.array([])
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.ego_x = None
        self.ego_y = None
        self.closest_wp_index = None

        self.num_waypoints = 0
        self.light_wp = -1
        self.state_red_count = -1

        
        self.tree = []
        self.image_count = 0
        self.waypoints_prepared = False
        self.idx_traffic_light_found = False
        self.last_pose = None
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.gpu_ready_pub = rospy.Publisher('/gpu_ready', Int32, queue_size=1)

        # Philippe
        sub6 = rospy.Subscriber('/image_color', Image, self.image_callback)
        self.light_classifier = TLClassifier(deep_learning=False)
        self.gpu_ready_pub.publish(Int32(1))
        self.loop()

        # Peter and Mikkel
        #sub6 = rospy.Subscriber('/image_color', Image, self.image_callback)
        #self.light_classifier = TLClassifier(deep_learning=True)
        #self.gpu_ready_pub.publish(Int32(1))
        #self.loop2()

    # -----------------------------------------------------------------------------------

    def loop2(self):
        # every 250 ms is OK on GTX 1080 TI: GPU decoding is arround 120 ms on GTX 1080 TI
        # every 333ms ms, so we have some margin on lower end GPUs: eg on GTX 980 TI decoding is around 250 ms
        # TESTED with GTX 1080 TI => Sould be OK with GTX TITAN X on Carla Self-Driving Car
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            if self.camera_image is not None:
                self.image_cb(self.camera_image)
            rate.sleep()

    def loop(self):
        # every 250 ms is OK on GTX 1080 TI: GPU decoding is arround 120 ms on GTX 1080 TI
        # every 333ms ms, so we have some margin on lower end GPUs: eg on GTX 980 TI decoding is around 250 ms
        # TESTED with GTX 1080 TI => Sould be OK with GTX TITAN X on Carla Self-Driving Car
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            if self.camera_image is not None:

                # search wp closest to our car
#                if self.closest_wp_index is None:
                wp_min = 0
                wp_max = self.num_waypoints - 1
#                else:
#                    wp_min = self.closest_wp_index - 200
#                    wp_max = self.closest_wp_index + 200

                self.closest_wp_index = self.get_closest_wp_index(self.ego_x, self.ego_y, wp_min, wp_max)

                closest_light_red_index = -1
                closest_light_red_dist = 1e10
                for i in range(len(self.lights)):
                    light = self.lights[i]
                    dist = self.stop_line_waypoints[i] - self.closest_wp_index
                    if dist >= 0 and dist < 150 and dist  < closest_light_red_dist:
                        closest_light_red_dist = dist
                        closest_light_red_index = i

                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                #Get classification
                state = self.light_classifier.get_classification(cv_image)

                if state == TrafficLight.RED and closest_light_red_index is not -1:
                    print('RED')
                    self.light_wp = self.stop_line_waypoints[closest_light_red_index]
                    self.state_red_count = STATE_COUNT_THRESHOLD
                else:
                    print('NOT RED')
                    self.state_red_count -= 1

                if self.state_red_count > 0:
                    print("traffic_waypoint=" + str(self.light_wp))
                    self.upcoming_red_light_pub.publish(Int32(self.light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(-1))

            rate.sleep()


    def image_callback(self, msg):
        if self.num_waypoints > 0 and self.ego_x is not None:
            self.camera_image = msg

    def get_closest_wp_index(self, x, y, wp1, wp2): 
        closest_dist = 1e10
        closest_wp_index = -1
        for i in range(wp1, wp2):
            wp = self.waypoints[i % self.num_waypoints]
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            dist = math.sqrt( (x - wp_x)**2 + (y - wp_y)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_wp_index = i
        return closest_wp_index

    # -----------------------------------------------------------------------------------

    def image_simu_cb(self, msg):
        if self.num_waypoints > 0 and self.ego_x is not None:

            # search wp closest to our car
#            if self.closest_wp_index is None:
            wp_min = 0
            wp_max = self.num_waypoints - 1
#            else:
#                wp_min = self.closest_wp_index - 200
#                wp_max = self.closest_wp_index + 200
            self.closest_wp_index = self.get_closest_wp_index(self.ego_x, self.ego_y, wp_min, wp_max)

            # simulate traffic light RED detection
            closest_light_red_index = -1
            closest_light_red_dist = 1e10
            for i in range(len(self.lights)):
                light = self.lights[i]
                if light.state == TrafficLight.RED:
                    dist = self.stop_line_waypoints[i] - self.closest_wp_index
                    # something realistic in our Field Of View
                    if dist >= 0 and dist < 150 and dist  < closest_light_red_dist:
                        closest_light_red_dist = dist
                        closest_light_red_index = i

            if (closest_light_red_index >= 0):
                # RED or YELLOW light detected
                self.light_wp = self.stop_line_waypoints[closest_light_red_index]
                self.state_red_count = STATE_COUNT_THRESHOLD
            else:
                self.state_red_count -= 1

            if self.state_red_count > 0:
                print("traffic_waypoint=" + str(self.light_wp))
                self.upcoming_red_light_pub.publish(Int32(self.light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(-1))

    def pose_cb(self, msg):
        self.pose = msg
        
        # Philippe
        self.ego_x = msg.pose.position.x
        self.ego_y = msg.pose.position.y

    # this callback is called just once at start
    def waypoints_cb(self, msg):
        if self.num_waypoints == 0:
            self.waypoints = msg.waypoints
            self.num_waypoints = len(self.waypoints)
            for i in range(len(self.stop_line_positions)):
                x = self.stop_line_positions[i][0]
                y = self.stop_line_positions[i][1]
                stop_line_waypoint = self.get_closest_wp_index(x, y, 0, self.num_waypoints)
                self.stop_line_waypoints.append(stop_line_waypoint)
                print(stop_line_waypoint)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    # -----------------------------------------------------------------------------------

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if self.num_waypoints > 0 and self.ego_x is not None:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.has_image = True
            self.image_count = self.image_count+1
            
            light_wp, state = self.process_traffic_lights(img)
    
    
            if state == TrafficLight.RED:
                print('RED')
                self.light_wp = self.stop_line_waypoints[light_wp]
                self.state_red_count = STATE_COUNT_THRESHOLD
            else:
                print('NOT RED')
                self.state_red_count -= 1

            if self.state_red_count > 0:
                print("traffic_waypoint=" + str(self.light_wp))
                self.upcoming_red_light_pub.publish(Int32(self.light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(-1))    
    
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        #Get classification
        if TRAINING is True:
            return TrafficLight.UNKNOWN
        else:
            return self.light_classifier.get_classification(light)

    def process_traffic_lights(self, img):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
            
        # For a valid vehicle pose. Get the closest waypoint of vehicle.         
        if (self.lights) and (self.pose):
            # search wp closest to our car
#            if self.closest_wp_index is None:
            wp_min = 0
            wp_max = self.num_waypoints - 1
#            else:
#                wp_min = self.closest_wp_index - 200
#                wp_max = self.closest_wp_index + 200
            self.closest_wp_index = self.get_closest_wp_index(self.ego_x, self.ego_y, wp_min, wp_max)

            # simulate traffic light RED detection
            closest_light_index = -1
            closest_light_dist = 1e10
            for i in range(len(self.lights)):
                dist = self.stop_line_waypoints[i] - self.closest_wp_index
                # something realistic in our Field Of View
                if dist >= 0 and dist < 200 and dist  < closest_light_dist:
                    closest_light_dist = dist
                    closest_light_index = i
         
            if closest_light_index is not -1:
                light = self.lights[closest_light_index]

            if SAVE_IMAGE and self.has_image:       
                strState = str(light.state)

                # Is the traffic light visible in the image?
                if (closest_light_index >= 0):
                    light_visible = True
                else:
                    light_visible = False
                    
                if (light_visible==False):
                    save_factor = 30
                    strState = str(4)
                elif (light.state==0): # RED LIGHT --> Slow save factor
                    save_factor = 10
                elif(light.state==1):  # Yellow LIGHT --> Faster save factor
                    save_factor = 1
                elif(light.state==2):  # Green LIGHT --> Faster save factor
                    save_factor = 1
                    
                # Is the car moving? (only save images when it is)
                if self.last_pose is not None:
                    diff_dist = math.sqrt((self.pose.pose.position.x-self.last_pose.pose.position.x)**2+(self.pose.pose.position.y-self.last_pose.pose.position.y)**2+(self.pose.pose.position.z-self.last_pose.pose.position.z)**2)
                else:
                    diff_dist = 0
                    
                if (diff_dist > 0.001) and (np.mod(self.image_count,save_factor)==0):
                    # Stores images to test results folder
                    tmpName = str(self.image_count).zfill(4)
                    dirOut = os.path.join(self.dirData,tmpName+"_"+strState+"_"+ str(int(closest_light_dist)) + '.jpg')
                    cv2.imwrite(dirOut,img)
                    rospy.loginfo("SaveImage")
                          
        self.last_pose = self.pose
                    
        if light:
            light_wp = closest_light_index
            state = self.get_light_state(img)
            return light_wp, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
