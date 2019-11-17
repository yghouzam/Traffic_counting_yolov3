#! /usr/bin/env python3
# coding: utf-8

from utils import *


class Counter:
    """Counter class manages the counting of objects in the video

    Args:
        border (tuple of int): list of two (x,y) positions that defines the counting line ([(x1,y1,x2,y2)])
        cls (list of str): list of objects classes you want to be counted (yolov3 classes)
        color (tuple of int): RGB color of the counter
        draw_loc (str): location of the counter in the image

    Attributes:
        border (tuple of int): list of two (x,y) positions that defines the counting line ([(x1,y1,x2,y2)])
        cls (list of str): list of objects classes you want to be counted (yolov3 classes)
        color (tuple of int): RGB color of the counter
        draw_loc (str): location of the counter in the image
        objects_seen (list): list of id elements already counted
        is_crossing (bool): is any element currently crossing the border
        counts_classes (dict): dictionnary of classes and their counts

    """

    def __init__(self, border, cls, color, draw_loc):
        self.color = color
        self.cls = cls
        self.counts_classes = dict(zip(list(cls), [0] * len(cls)))
        self.objects_seen = []
        self.border = border
        self.draw_loc = draw_loc
        self.is_crossing = False

    def count_class(self, objects, mapped_centroid_classes):
        """Count crossing elements and increments the element counter.
        if any element is crossing the border self.is_crossing is set to be True

        Args:
            objects (dict): dictionnary of centroid coordinated
                            of tracked elements in the image (ie : {0: [250,470],1: [410,520]} )
            mapped_centroid_classes (dict): dictionnary of tracked elements
                                            and their detected classes
                                            (ie {0:'car',1: 'car',2:'truck'} )

        Returns:
            self.counts_classe (dict): dictionnary of the counted classes (ie {'car': 43, 'truck': 5, 'motorbike': 0})
        """
        self.is_crossing = False
        x1, y1, x2, y2 = self.border
        for (objectID, centroid) in objects.items():
            if is_crossing_line(centroid[0], centroid[1], ((x1, y1), (x2, y2))):
                if objectID not in self.objects_seen:
                    cls = mapped_centroid_classes[objectID]
                    if cls in self.counts_classes.keys():
                        self.counts_classes[cls] += 1
                    else:
                        self.counts_classes[cls] = 1
                    self.objects_seen.append(objectID)
                    self.is_crossing = True
        return self.counts_classes

    def count_display(self, img, icons, draw_line=True):
        """Displays out the counter on the input image

        Args:
            img (numpy 2D array): Input image
            icons (dict): dictionnary of icons
            draw_line (bool): draw the counter line (border) if True
        """
        # Counting Display
        shape = img.shape[:2]
        offset_r, offset_c = offset_loc(self.draw_loc)

        for num_o, obj in enumerate(self.cls):
            r_index_start = offset_r * (icons[obj]["h"] + 20)
            r_index_end = (r_index_start + icons[obj]["h"])
            c_index_start = offset_c * ((icons[obj]["w"] * num_o) + (40 * (num_o + 2)))
            c_index_end = (c_index_start + icons[obj]["w"])

            img[r_index_start:r_index_end, c_index_start:c_index_end] = icons[obj]["icon"]
            cv2.putText(img, "{}".format(self.counts_classes[obj]), (
                abs(min(0, offset_c) * shape[1]) + c_index_end + 5,
                abs((min(0, offset_r) * img.shape[0])) + r_index_end),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        if draw_line:
            cv2.line(img, self.border[:2], self.border[2:],
                     crossing_color(self.color, self.is_crossing), 3)
