#! /usr/bin/env python3
# coding: utf-8

import cv2
from imageai.Detection import ObjectDetection


class YOLOV3Detector:
    """YOLOV3 detector class

    Note : This class is based on the usage of imageai python package

    Args:
        cls (list of str): list of coco classes to detect (i.e ['car','truck','motorcycle'])
        wieghts_path (str): path to the yolov3 coco path

    Attributes:
        cls (list of str): list of coco classes to detect
        wieghts_path (str): path to the yolov3 coco path
        detector (object): ObjectDetection class object
        custom_objects (dict):  dictionnary of coco customs objects to detect (i.e {'car':True,'Truck':True} )
    """

    def __init__(self, cls, weights_path):
        self.cls = cls
        self.weights_path = weights_path
        self.detector = ObjectDetection()
        self._load_model()
        self.custom_objects = self.detector.CustomObjects(**dict(zip(self.cls, [True] * len(self.cls))))

    def _load_model(self):
        """load the YOLOv3 model and set the weights
        """
        self.detector.setModelTypeAsYOLOv3()
        #self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(self.weights_path)
        self.detector.loadModel()

    def detect(self, img):
        """detect method performs class detection

        Args:
            img (numpy 2D array): input image

        Returns:
            result (numpy 2D array) : the input image with detected objects drawn
            detections (list of tuple): list of dictionnary of the detected objects
                                        (i.e [{'name': 'truck', 'percentage_probability': 51.10,
                                        'box_points': [829, 485, 904, 552]},
                                        {'name': 'truck', 'percentage_probability': 61.18,
                                        'box_points': [1171, 796, 1357, 935]}]


        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Object detection
        result_img, detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom_objects,
                                                                            input_image=img,
                                                                            input_type='array',
                                                                            output_type="array",
                                                                            minimum_percentage_probability=60)
        result_img = cv2.cvtColor(result_img,cv2.COLOR_RGB2BGR)

        detections = reformat_detection(detections)

        return result_img, detections

    def close(self):
        """close method, ends the tensorflow session
        """
        self.detector.sess.close()


def reformat_detection(detections):
    """
    Reformat the detections
    Args:
        detections : detections from ObjectDetection object class instance

    Returns:
        res (list of tuples): contains the detected object class confidence and bounding boxes coordinates
    """
    res = []
    for detection in detections:
        x1 = detection['box_points'][0]
        y1 = detection['box_points'][1]
        x2 = detection['box_points'][2]
        y2 = detection['box_points'][3]
        cls = detection['name']
        conf = detection['percentage_probability']
        res.append((x1, y1, x2, y2, conf, cls))
    return res
