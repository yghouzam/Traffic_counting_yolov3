#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from scipy.spatial import distance as dist

from centroid_tracker import CentroidTracker
from utils import *


class TrackerSpeedEstimator:
    """TrackerSpeedEstimator class manages the tracking and the speed estimation of elements in a video.

    Note: Works with CentroidTracker and Control_zone classes

    Args:
        czones (list of object class): list of Control_zone object class that have been initialized
        video_capture (opencv object): opencv video iterator

    Attributes:
        czones (list of object class): list of Control_zone object class that have been initialized
        fps (int): number of frames per second of the video
        cap (opencv object): opencv video iterator
        ct (object): CentroidTracker object
        mapped_centroid_classes (dict): dictionnary of tracked elements
                                            and their detected classes
                                            (ie {0:'car',1: 'car',2:'truck'} )
        tracked_objects_status (dict): dictionnary of tracked elements
                                            and their control zone status,
                                            idzone belongings and number of displayed time
                                            (ie  {0: (1, 1, 0),
                                                  1: (0, None, 0),
                                                  2: (2, 1, 19)} )
        frameid_control (dict): dictionnary of tracked elements and their frames ids in the control zone
                                            (ie {4: [24, 68, 69],
                                                 0: [54, 79, 80, 81, 82],
                                                 8: [84, 112]})
        estimated_speed (dict): dictionnary of the tracked elements and their estimated speed for each control zone
                                            (ie {1: {0: 104.4, 8: 104.4},
                                                 2: {2: 100.8, 6: 127.095}
                                                 })

    """

    def __init__(self, video_capture, czones):
        self.czones = czones
        self.fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        self.cap = video_capture
        self.ct = CentroidTracker(maxDisappeared=8)
        self.mapped_centroid_classes = {}
        self.tracked_objects_status = {}
        self.frameid_control = {}
        self.estimated_speed = {}

    def track(self, detections):
        """update the tracker with the centroid of the detected elements
        Args:
            detections (list of numpy array): list of bounding box coordinates of the detected elements
                                        (ie [np.array([252,266,175,112]]),np.array([112,186,375,121]])])
        """
        # object Tracking
        self.detections = detections
        bboxes = [np.array(i[:4]).astype(int) for i in self.detections]
        self.objects = self.ct.update(bboxes)


    def map_centroid_class(self):
        """Maps the tracked objects ids with the detected object classes using centroids and bounding boxes
        Note :
              the mapping is realized according the the minimal distance bewteen two centroids
        """
        centroid_bboxes = np.array([((x1 + x2) / 2, ((y1 + y2) / 2)) for x1, y1, x2, y2, conf, cls in self.detections])
        for (objectID, centroid) in self.objects.items():
            distances = dist.cdist(np.expand_dims(centroid, axis=0), centroid_bboxes)
            imaped_bbox = distances.argmin(axis=1)[0]
            self.mapped_centroid_classes[objectID] = self.detections[imaped_bbox][5]

    def _update_status(self, obj_id, centroid, cz):
        """update the tracked elements informations to know when an element is not yet in the control zone,
        is currently in the control zone or has already crossed the control zone.

        Frame ids are saved for each tracked elements when entering in the control zone and exiting the controle zone

        Note :
              status is defined as :
                                   0 - when a tracked element has not crossed any control zone yet,
                                   1 - when a tracked element has crossed the starting zone of a control zone
                                   2 - when a tracked element has crossed the ending zone of the same control zone

        Args:
            obj_id (int): object id of the tracked element
            centroid (tuple of int): x,y coordinates tracked centroid
            cz (Control_zone object): control zone
        """
        frameid = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        if not (obj_id in self.tracked_objects_status.keys()):
            self.tracked_objects_status[obj_id] = (0, None, 0)

        status, idczone, ndisplay = self.tracked_objects_status[obj_id]
        if cz.entering_zone(centroid):
            # if status != 1:
            self.tracked_objects_status[obj_id] = (1, cz.idczone, 0)
            self.frameid_control[obj_id] = [frameid]

        if idczone == cz.idczone:
            if cz.exiting_zone(centroid):
                self.tracked_objects_status[obj_id] = (2, idczone, 0)
                self.frameid_control[obj_id].append(frameid)

    def compute_speed(self):
        """Compute the speed for each tracked elements crossing each control zone

        Note :
              only elements with status = 2 are measured using information of the entering and exiting frame ids
              knowing the length of between the entering and the exiting zone ,
              the number of frames in the control zone and the frame rate, we can estimate the speed of the object.
        """
        for czone in self.czones:
            if not (czone.idczone in self.estimated_speed.keys()):
                self.estimated_speed[czone.idczone] = {}
            for (objectID, centroid) in self.objects.items():
                self._update_status(objectID, centroid, czone)

                status, idczone, ndisplay = self.tracked_objects_status[objectID]

                if status == 2 and (idczone == czone.idczone):
                    n_present_frames = self.frameid_control[objectID][-1] - self.frameid_control[objectID][0]
                    speed = ((czone.ckzn_d / (n_present_frames / self.fps)) * 3600) / 1000  # km/h
                    self.estimated_speed[idczone].update({objectID: speed})

    def display_speed(self, img, ndisplay_frames=20):
        """Displays the speed of each measured objects and the average speed for each control zone
        Note :
              only the speed of status = 2 elements is displayed during ndisplay_frames
              If an element if over the speed limit, the speed is displayed as red
        Args:
             img (numpy 2D array): input image
             ndisplay_frames (int): maximum number of displayed speed frames for each tracked objects
        """
        speedlimits = {}
        for czone in self.czones:
            speedlimits[czone.idczone] = czone.speedlimit

        shape = img.shape[:2]
        for (objectID, centroid) in self.objects.items():
            status, idczone, ndisplay = self.tracked_objects_status[objectID]
            if status == 2:
                if ndisplay <= ndisplay_frames:
                    speed = self.estimated_speed[idczone][objectID]
                    centroid = self.objects[objectID]
                    cv2.putText(img, "{0:.1f} : km/h".format(speed), (centroid[0] - 15, centroid[1] + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                over_speed_color((0, 255, 0), speed, speedlimits[idczone]),
                                1)
                    self.tracked_objects_status[objectID] = (status, idczone, ndisplay + 1)
                else:
                    self.tracked_objects_status[objectID] = (0, None, 0)

        i = 0
        for czone in self.czones:
            idczone = czone.idczone
            if len(self.estimated_speed[idczone]) > 0:
                mspeed = np.array(list(self.estimated_speed[idczone].values())).mean()
                offset_r, offset_c = offset_loc(czone.draw_loc)
                x, y = (
                    (shape[1] // 2) + (offset_c * (shape[1] // 4)),
                    -((shape[0] // 2) - 20) * offset_r + (shape[0] // 2))
                cv2.rectangle(img, (x - 10, y + 5), (x + 170, y - 15), czone.col, -1)
                cv2.putText(img, "Avg : {0:.1f} : km/h".format(mspeed), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            i += 1

    def display_tracking(self, img):
        """Displays the centroid and the IDs of the tracked objects
        Args:
             img (numpy 2D array): input image
        """
        for (objectID, centroid) in self.objects.items():
            col = (0, 255, 0)
            status, _, _ = self.tracked_objects_status[objectID]
            if status == 1:
                col = (255, 255, 255)
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.circle(img, (centroid[0], centroid[1]), 4, col, -1)
