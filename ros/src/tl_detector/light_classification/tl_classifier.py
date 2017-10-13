import tensorflow as tf
import numpy as np
import cv2
import os
from cv_bridge import CvBridge

from styx_msgs.msg import TrafficLight

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
        self.counter = 0


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        self.counter += 1

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                        feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.6

        boxes, scores, classes = self._filter_boxes(confidence_cutoff, boxes, scores, classes)

        height = image.shape[0]
        width = image.shape[1]
        box_coords = self._to_image_coords(boxes, height, width)

        self._draw_boxes(image, box_coords, classes)

        cv2.imwrite('./light_classification/detection_images/{}.png'.format(str(self.counter).zfill(3)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        #return TrafficLight.UNKNOWN



        ros_image = self.bridge.cv2_to_imgmsg(image, "rgb8")

        return ros_image

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