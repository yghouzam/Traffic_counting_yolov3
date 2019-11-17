#! /usr/bin/env python3
# coding: utf-8

import cv2
import os


def point_inside_polygon(x, y, poly):
    """
    Check if a point (x,y) is in a polygone

    Args:
        x (int): x coordinate of the point
        y (int): y coordinate of the point
        poly (list of tuple int): polygone list of (x,y) coordinates
    Returns :
            bool : True is the point is in the polygone , False otherwise
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def is_crossing_line(x, y, line_pts, tresh=0.01):
    """
    Check if a point (x,y) is near to a line
    Note : we build a small polygone from the line

    Args:
        x (int): x coordinate of the point
        y (int): y coordinate of the point
        line_pts (tuple of int):
        tresh (float): treshold of the extented polygone

    Returns :
            bool : True is the point is in the line , False otherwise
    """
    tresh_up = 1 + tresh
    tresh_dnw = 1 - tresh
    x1, y1 = line_pts[0]
    x2, y2 = line_pts[1]
    x3, y3 = x1, y1 * tresh_dnw
    x4, y4 = x2, y2 * tresh_dnw
    x5, y5 = x2, y2 * tresh_up
    x6, y6 = x1, y1 * tresh_up

    poly = [(x3, y3), (x4, y4), (x5, y5), (x6, y6)]
    return point_inside_polygon(x, y, poly)


def load_icons(classes):
    """Load and resize icons classes
       Args:
            icon of the objects classes
       Returns :
                dictionnary containing the icons
    """
    dictio = {}
    for obj in classes:
        dirname = os.path.dirname(os.path.realpath(__file__))
        obj_icon = cv2.imread(os.path.join(dirname,"icons", obj + '.png'))
        obj_icon = cv2.resize(obj_icon, (20, 20))
        dictio[obj] = {"icon": obj_icon, "h": obj_icon.shape[0], "w": obj_icon.shape[1]}
    return dictio


def crossing_color(border_color, is_crossing):
    """returns the a white RGB color when is_crossing is True
       Args:
             border_color (tuple int): RGB input border color
             is_crossing (bool): is the border currently crossed by an element
        Returns :
                 white RGB color when is_crossing is True, border_color otherwise
    """
    if is_crossing:
        return 255, 255, 255
    return border_color


def over_speed_color(color, speed, speed_limit):
    """returns the a red RGB color when the speed is > than the speed limit
        Args:
             color (tuple int): RGB input color
             speed (float): speed
             speed_limit (float): speed limit
        Returns :
                 RGB color (tuple int): red RGB color or input RGB color
    """
    if speed > speed_limit:
        return 0, 0, 255
    return color


def offset_loc(loc):
    """Determine the offet factor for the rows and the columns index in function of the location of a text
        Args:
             loc (str): location in ['top-left','bottom-left','bottom-right','top-right']
        Returns :
                 offset_r (int): rows offset factor (-1 for bottom and 1 for top)
                 offset_c (int): columns offset factor (1 for left and -1 for right)
    """
    if loc == "top-left":
        offset_r = 1
        offset_c = 1
    elif loc == "bottom-left":
        offset_r = -1
        offset_c = 1
    elif loc == "bottom-right":
        offset_r = -1
        offset_c = -1
    elif loc == "top-right":
        offset_r = 1
        offset_c = -1
    else:
        raise ValueError("loc type {} not in ['top-left','bottom-left','bottom-right','top-right']".format(loc))

    return offset_r, offset_c
