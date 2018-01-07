

import tensorflow as tf
import numpy as np
import cv2
import os
import rospy

from timeit import default_timer as timer


CLASS_TRAFFIC_LIGHT = 10

MODEL_DIR = 'light_classification/models/'
IMG_DIR = 'light_classification/img/'
DEBUG_DIR = 'light_classification/debug/'
CLASSIFIER = '1513151224'

import tensorflow as tf
import numpy as np
import cv2
import os
import rospy

from timeit import default_timer as timer


CLASS_TRAFFIC_LIGHT = 10

MODEL_DIR = 'light_classification/models/'
IMG_DIR = 'light_classification/img/'
DEBUG_DIR = 'light_classification/debug/'

class TLClassifier(object):
    def __init__(self, profiling=True, deep_learning=True):
        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)
        #rospy.logwarn("%s", os.getcwd())
        self.detector = MODEL_DIR + 'faster_rcnn_inception_v2.pb'
        rospy.logwarn("----------------------------------------------------------------------------------")
        rospy.logwarn("With SIMULATOR: use lowest screen resolution 640x480 with Graphics Quality FASTEST")
        rospy.logwarn("With SIMULATOR: so that the GPU is mainly dedicated to running object detection")
        rospy.logwarn("----------------------------------------------------------------------------------")
        rospy.logwarn("PLEASE WAIT for TENSORFLOW init ... There will be a READY TO GO message")
        rospy.logwarn("----------------------------------------------------------------------------------")
        self.sess, _ = self.load_graph(self.detector)
        detection_graph = self.sess.graph
        
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # The classification of the object (integer id).
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        self.profiling = profiling
        self.deep_learning = deep_learning
        
        # Load traffic light classifier
        if deep_learning:
            self.classifier = MODEL_DIR + CLASSIFIER
            graph_file = os.path.join(self.classifier,'saved_model.pb')
            tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.classifier)        
            self.predictor = tf.contrib.predictor.from_saved_model(self.classifier)
        
        # the very first decoding is slow: al inits are done
        # => do that in advance before real decoding
        for i in range(1):
            #test_image = cv2.imread('light_classification/img/left0144.jpg')
            test_image = cv2.imread(IMG_DIR + 'left0144.jpg')
            
#            pred_image, is_red = self.detect_tl(test_image)
            image_np, box_coords, classes, scores = self.detect_tl(test_image)
            
            # Traditional traffic light classifier
            pred_image, is_red = self.classify_red_tl(image_np, box_coords, classes, scores)
            if is_red:
                print("Traditional classifier: RED")
            else:
                print("Traditional classifier: NOT RED")
                
            # Deep learning traffic light classifier
            if deep_learning:
                pred_image, is_red = self.classify_tl_cnn(image_np, box_coords, classes, scores)
                if is_red:
                    print("Deep learning classifier: RED")
                else:
                    print("Deep learning classifier: NOT RED")                
            
            #cv2.imwrite("light_classification/img/pred_image.png", pred_image)
            cv2.imwrite(IMG_DIR + 'pred_image.png', pred_image)

        rospy.logwarn("----------------------------------------------------------------------------------")
        rospy.logwarn("TENSORFLOW init done ... READY TO GO")
        rospy.logwarn("----------------------------------------------------------------------------------")
        self.num_image = 1

    def _float32_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))    

    def load_graph(self, graph_file, use_xla=False):
        #config = tf.ConfigProto()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        if use_xla:
            jit_level = tf.OptimizerOptions.ON_1
            config.graph_options.optimizer_options.global_jit_level = jit_level
    
        with tf.Session(graph=tf.Graph(), config=config) as sess:
            gd = tf.GraphDef()
            with tf.gfile.Open(graph_file, 'rb') as f:
                data = f.read()
                gd.ParseFromString(data)
            tf.import_graph_def(gd, name='')
            ops = sess.graph.get_operations()
            n_ops = len(ops)
            print("number of operations = %d" % n_ops)
            return sess, ops

    def filter_boxes(self, min_score, boxes, scores, classes, keep_classes=None):
        """Return traffic light boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                if ((keep_classes is None) or (int(classes[i]) in keep_classes)):
                    idxs.append(i)
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes
    
    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        return box_coords
    
    def select_lighton_real(self, img): # HLS for real
        """Applies color selection for high L and S """
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        lower = np.array([ 50,   150, 150], dtype="uint8")
        upper = np.array([ 100, 255, 255], dtype="uint8")
        tl_mask = cv2.inRange(hls_img, lower, upper)
        return cv2.bitwise_and(img, img, mask = tl_mask)
    
    def select_red_simu(self, img): # BGR for simu
        lower = np.array([ 0,   0, 200], dtype="uint8")
        upper = np.array([ 50, 50, 255], dtype="uint8")
        red_mask = cv2.inRange(img, lower, upper)
        return cv2.bitwise_and(img, img, mask = red_mask)
    
    def classify_tl_cnn(self, image_np, boxes, classes, scores, thickness=4):
        """Draw bounding boxes on the image"""
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            score = scores[i]
#            if class_id == CLASS_TRAFFIC_LIGHT:
            h = top - bot
            w = right - left
            if h <= 1.5 * w:
                continue # Truncated Traffic Ligth box
#            cv2.rectangle(image_np,(left, top), (right, bot), (255, 0, 0), thickness) # BGR format for color
            tl_img = image_np[int(bot):int(top), int(left):int(right)]
            
            equal_size_img = cv2.resize(tl_img,dsize=(30,90))
            equal_size_img = np.asarray(equal_size_img, dtype=np.float32)    
            model_input= tf.train.Example(features=tf.train.Features(feature={'x': self._float32_feature(equal_size_img.reshape(-1))})) 
            model_input=model_input.SerializeToString()
            output_dict= self.predictor({"inputs":[model_input]})

            if output_dict['classes'][0]=='red':
                text = "RED LIGHT score=%.3f" % score
                cv2.putText(image_np, text, (int(left), int(bot)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                return image_np, True
              # if normalized center of mass of RED points 
#              # is in the upper part of detected Traffic Light Box
#              # THEN it is a RED traffic light
#              if npoints > 10 and mean_row < 0.33:
#                  text = "RED LIGHT score=%.3f" % score
#                  cv2.putText(image_np, text, (int(left), int(bot)), cv2.FONT_HERSHEY_SIMPLEX, 
#                                               1, (0,0,255), 2)
#                  return image_np, True
        return image_np, False    
    
    def classify_red_tl(self, image_np, boxes, classes, scores, thickness=4):
        """Draw bounding boxes on the image"""
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            score = scores[i]
#            if class_id == CLASS_TRAFFIC_LIGHT:
            h = top - bot
            w = right - left
            if h <= 1.5 * w:
                continue # Truncated Traffic Ligth box
            cv2.rectangle(image_np,(left, top), (right, bot), (255, 0, 0), thickness) # BGR format for color
            tl_img = image_np[int(bot):int(top), int(left):int(right)]

            tl_img_simu = self.select_red_simu(tl_img) # SELECT RED
            tl_img_real = self.select_lighton_real(tl_img) # SELECT TL
            tl_img = (tl_img_simu + tl_img_real) / 2

            gray_tl_img = cv2.cvtColor(tl_img, cv2.COLOR_RGB2GRAY)
            nrows, ncols = gray_tl_img.shape[0], gray_tl_img.shape[1]

            # compute center of mass of RED points
            mean_row = 0
            mean_col = 0
            npoints = 0
            for row in range(nrows):
                for col in range(ncols):
                    if (gray_tl_img[row, col] > 0): 
                        mean_row += row
                        mean_col += col
                        npoints += 1
            if npoints > 0:
              mean_row = float(mean_row / npoints) / nrows
              mean_col = float(mean_col / npoints) / ncols
              #print(mean_row, mean_col, npoints)

              # if normalized center of mass of RED points 
              # is in the upper part of detected Traffic Light Box
              # THEN it is a RED traffic light
              if npoints > 10 and mean_row < 0.33:
                  text = "RED LIGHT score=%.3f" % score
                  cv2.putText(image_np, text, (int(left), int(bot)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                  return image_np, True
        return image_np, False

    def detect_tl(self, image):
        trt_image = np.copy(image)
        image_np = np.expand_dims(np.asarray(trt_image, dtype=np.uint8), 0)
    
        
        # Actual detection.
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                            feed_dict={self.image_tensor: image_np})
    
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
    
        confidence_cutoff = 0.8
        
        # Filter traffic light boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes, keep_classes=[CLASS_TRAFFIC_LIGHT])
    
        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        image_np = np.squeeze(image_np)
        width = image_np.shape[1]
        height = image_np.shape[0]
        box_coords = self.to_image_coords(boxes, height, width)
        
        return image_np, box_coords, classes, scores
        
    def classify_tl(self,image_np, box_coords, classes, scores):
        # Each class with be represented by a differently colored box
        if self.deep_learning:
            image_np, is_red = self.classify_tl_cnn(image_np, box_coords, classes, scores)
        else:
            image_np, is_red = self.classify_red_tl(image_np, box_coords, classes, scores)
    
        return image_np, is_red

    def get_classification(self, image):
        from styx_msgs.msg import TrafficLight
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # light detection        
        start = timer()
        image_np, box_coords, classes, scores = self.detect_tl(image)
        end_inference = timer()
        
        # light color classification
        pred_image, is_red = self.classify_tl(image_np, box_coords, classes, scores)
        end = timer()
        
        if self.profiling:
            time_inference = end_inference - start
            time_img_processing = (end - start) - time_inference
            print("time: inference {:.6f} post-processing {:.6f}".format(time_inference, time_img_processing))
        

        fimage = DEBUG_DIR + 'image' + str(self.num_image) + '.png'
        cv2.imwrite(fimage, pred_image)
        self.num_image += 1

        if is_red:
            return TrafficLight.RED
        else:
            return TrafficLight.UNKNOWN
