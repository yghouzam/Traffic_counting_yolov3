#! /usr/bin/env python3
# coding: utf-8

import numpy as np
from utils import *


class Control_zone:
    """Control Zone class manages the speed checking zone.

    Note: Controle zone is defined with two lines and each lines is definied with two points (x,y).

    here two examples :

              Start line
    (x1,y1)+--------------+(x2,y2)                (x1,y1)+--------------+(x4,y4)
           |              |                              |              |
           |    Control   |                      Start   |    Control   |  End
           |     Zone     |                       Line   |     Zone     |  Line
           |              |                              |              |
    (x4,y4)+--------------+(x3,y3)                (x2,y2)+--------------+(x3,y3)
              End line

    Args:
        height (int): image height in pixels
        width (int): image widht in pixels
        idczone (int): unique id of the controle zone
        x1y1x2y2 (list of int): list of x,y positions that defines the start line of the control zone ([(x1,y1,x2,y2)])
        x3y3x4y4 (list of int): list of the point 3 and 4 that defines the end line of the control zone ([(x3,y3,x4,y4)])
        ckzn_d (int): distance in meters between the start and the end line of control zone
        col (tuple of int): RGB color of the control zone
        draw_loc (str): location of the average speed of the control zone ['top-left','bottom-left','bottom-right','top-right']

    Attributes:
        height (int): image height in pixels
        width (int): image widht in pixels
        idczone (int): unique id of the controle zone
        border1 (list of int): list of x,y positions that defines the start line of the control zone ([(x1,y1,x2,y2)])
        border2 (list of int): list of the point 3 and 4 that defines the end line of the control zone ([(x3,y3,x4,y4)])
        ckzn_d (int): distance in meters between the start and the end line of control zone
        col (tuple of int): RGB color of the control zone
        draw_loc (str): location of the average speed of the control zone ['top-left','bottom-left','bottom-right','top-right']

    """

    def __init__(self, idczone, height, width, x1y1x2y2, x3y3x4y4, ckzn_d, speedlimit, col, draw_loc):
        self.height = height
        self.width = width
        self.border1 = x1y1x2y2
        self.border2 = x3y3x4y4
        self.ckzn_d = ckzn_d
        self.speedlimit = speedlimit
        self.idczone = idczone
        self.col = col
        self.draw_loc = draw_loc
        self._construct_zone()

    def _construct_zone(self):
        """Construct the homography matrix of the control zone in order to accurately measure the speed
        """
        check_zone_pts1 = np.float32([self.border1[0], self.border1[1], self.border2[0], self.border2[1]])
        check_zone_pts2 = np.float32([[0, self.height], [self.width, self.height], [self.width, 0], [0, 0]])
        self.zone_mtx, mask = cv2.findHomography(check_zone_pts1, check_zone_pts2, cv2.RANSAC, 5.0)

    def _project_to_zone(self, x, y):
        """Projects the point (x,y) to the homography plane of the control zone
        Args:
            x (int): x coordinate in pixel of the point
            y (int): y coordinate in pixel of the point
        Returns:
            px (int): projected coordinate in pixel of the point
            py (int): projected coordinate in pixel of the point
        """
        p = np.array((x, y, 1)).reshape((3, 1))
        temp_p = self.zone_mtx.dot(p)
        suma = np.sum(temp_p, 1)
        px = int(round(suma[0] / suma[2]))
        py = int(round(suma[1] / suma[2]))
        return px, py

    def in_zone(self, xy):
        """Checks if the point (x,y) is in the control zone
        Args:
            xy (list of int): [x,y] coordinates of the point
        Returns:
            bool : True if the projected point is in the control zone
                   ,False otherwise
        """
        x_ref, y_ref = self._project_to_zone(xy[0], xy[1])
        if ((x_ref >= 0) & (x_ref <= self.width)) & ((y_ref >= 0) & (y_ref <= self.height)):
            return True
        return False

    def entering_zone(self, xy):
        """Checks if the point (x,y) is in the entering zone (start line)
        Note :
            a very small zone around the start line is constructed.
        Args:
            xy (list of int): [x,y] coordinates of the point
        Returns:
            bool : True if the projected point is in start line of the control zone
                   ,False otherwise
        """
        return is_crossing_line(xy[0], xy[1], self.border1, tresh=0.015)

    def exiting_zone(self, xy):
        """Checks if the point (x,y) is in the exiting zone (start line)
        Note :
            a very small zone around the end line is constructed.
        Args:
            xy (list of int): [x,y] coordinates of the point
        Returns:
            bool : True if the projected point is in end line of the control zone
                   ,False otherwise
        """
        return is_crossing_line(xy[0], xy[1], self.border2, tresh=0.015)

    def display_zone(self, img):
        """Displays the control zone on the input image
        Args:
            img (numpy 2D array): input image
        Returns:
            image_new (numpy 2D array): new image with the drawn control zone
        """
        alpha = 0.4
        overlay = img.copy()
        pts = np.array([self.border1[0], self.border1[1], self.border2[0], self.border2[1]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img=overlay, pts=[pts], color=self.col)
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return image_new
