from styx_msgs.msg import TrafficLight
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Lock
import tensorflow as tf
import image_geometry
import numpy as np
import rospy
import cv2
import os


class TLState(object):
    # Number of guess allowed in the case where no lights are classified
    MAX_GUESS = 2
    
    def __init__(self):
        self.now = TrafficLight.UNKNOWN
        self.last = TrafficLight.UNKNOWN
        self.guess_count = 0
        self.x = -1
        self.y = -1

    def processClassifications(self, boxes, predictions, confidences):
        '''Process new classifications.
        
        A simple voting mechanism. Take the label which has the highest sum
        of confidences.
        '''
        if len(predictions) > 0: # There is a prediction
            res = {TrafficLight.RED:0,TrafficLight.YELLOW:0,TrafficLight.GREEN:0,TrafficLight.UNKNOWN:0}
            for box, pred, conf in zip(boxes,predictions, confidences):
                
                # Nasty hack to dismiss any classifications at the very 
                # bottom of the screen (fixes a system bug)
                if box[2] < 550:
                    res[pred-1] += conf
                    
            self.now = max(res, key=res.get)
            
            # If there are no classification in the chosen category, then set to unknown
            if res[self.now] == 0:
                self.now = TrafficLight.UNKNOWN
                
            self.guess_count = 0  # Clear guess counter
        else: # There is no prediction
            # If the last light was a Red or Yellow, then set output to Red
            if self.last == TrafficLight.YELLOW or self.last == TrafficLight.RED:
                self.now = TrafficLight.RED
                # After N guesses, revert to unknown
                if self.guess_count > self.MAX_GUESS: 
                    self.now = TrafficLight.UNKNOWN
                self.guess_count +=1
            else:
                self.now = TrafficLight.UNKNOWN
                self.guess_count = 0
                
        self.last = self.now

class TLClassifier(object):
                
    def __init__(self):
        #print os.getcwd()
        graph_file = './light_classification/model/frozen_inference_graph.pb'
        self.graph = self._load_graph(graph_file)
        self.sess = tf.Session(graph=self.graph)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')

        self.COLOR_LIST = [(255,0,0),(255,255,0),(0,255,0)]
        self.bridge = CvBridge()
        self.predictor = TLState()
        self.counter = 0
        self.skip = 0
        self._classify_lock = Lock()
        
        self._classified_image_publisher = rospy.Publisher('/classified_image', Image, queue_size=1)
        # Add the pinhole camera mode - for tranformation of traffic 
        # light position to pixel position
        # self._cam_model = image_geometry.PinholeCameraModel()
        # FIXME: Needs the camera info to setup the model
                           

    def get_classification(self, image, light):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light
            light (TrafficLight): next traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        self.counter += 1

        # Hack to reduce processing to every second image
        if self.skip >= 3:
            
            # Wrap in lock - (assumption that threading causing stuck classification)
            self._classify_lock.acquire()
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                            feed_dict={self.image_tensor: image_np})
                                            
            self._classify_lock.release()

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.6

            boxes, scores, classes = self._filter_boxes(confidence_cutoff, boxes, scores, classes)
            height = image.shape[0]
            width = image.shape[1]
            box_coords = self._to_image_coords(boxes, height, width)
            
            DEBUG_CLASSIFIER = True
            if DEBUG_CLASSIFIER:
                '''
                Get this working or remove
                #print(light)
                print(box_coords)
                print(self._cam_model.projectionMatrix())
                p3d = [1,2,1]
                #p3d = ((light.pose.pose.position.x,light.pose.pose.position.y,light.pose.pose.position.z))
                print(self._cam_model.project3dToPixel(p3d))
                '''
                self._draw_boxes(image, box_coords, classes)
                # Publish the classified image
                image_message = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
                self._classified_image_publisher.publish(image_message)
            
            self.predictor.processClassifications(box_coords, classes, scores)
            label = {4:'UNKNOWN',2:'GREEN',1:'YELLOW',0:'RED'}
            rospy.loginfo('@_4 New Prediction: %s %s', self.predictor.now, label[self.predictor.now])
            self.skip = 0
        self.skip += 1
        
        return self.predictor.now
        
        #ros_image = self.bridge.cv2_to_imgmsg(image, "rgb8")
        #return ros_image

    def _load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    def _filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def _to_image_coords(self, boxes, height, width):
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

    def _draw_boxes(self, image, boxes, classes, thickness=4):
        """Draw bounding boxes on the image"""
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i]) - 1
            color = self.COLOR_LIST[class_id]
            cv2.rectangle(image, (left, top), (right, bot), color=color, thickness=thickness)
