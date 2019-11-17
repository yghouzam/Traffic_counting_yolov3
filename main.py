#! /usr/bin/env python3
# coding: utf-8

from controlzone import Control_zone
from counter import Counter
from detector import YOLOV3Detector
from config import Config
from trackerspeedestimator import TrackerSpeedEstimator
from utils import *
import argparse


def main():
    video_path, output_path, yolov3_weights_path = get_args()

    config = Config()

    video_capture = cv2.VideoCapture(video_path)

    HEIGHT = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    FPS = int(video_capture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

    classes = ['car', 'truck', 'motorcycle']

    R_col = (255, 0, 0)
    G_col = (0, 255, 0)
    B_col = (0, 0, 255)

    icons = load_icons(classes)

    # Set the counters parameters
    counters_ini = config.parse_counters()

    # MODIFY THIS DEPENDING ON THE NUMBER OF COUNTERS YOU HAVE SET IN THE config.ini FILE
    counters_params = [dict(cls=classes, color=R_col, draw_loc='bottom-left'),
                       dict(cls=classes, color=G_col, draw_loc='bottom-right')]

    assert len(counters_params) == len(counters_ini), 'Same number of counters must be in config.ini and main.py '
    for counter_param_main, counter_param_ini in zip(counters_params, counters_ini):
        counter_param_main.update(dict(border=counter_param_ini))

    # BUILD THE COUNTERS
    counters = []
    for counter_params in counters_params:
        counter = Counter(**counter_params)
        counters.append(counter)

    # Set the control zones parameters
    czones_ini = config.parse_czones()

    # MODIFY THIS DEPENDING ON THE NUMBER OF CONTROL ZONES YOU HAVE SET IN THE config.ini FILE
    czones_params = [dict(height=HEIGHT, width=WIDTH, col=R_col, draw_loc='top-right'),
                     dict(height=HEIGHT, width=WIDTH, col=G_col, draw_loc='top-left')]

    assert len(counters_params) == len(counters_ini), 'Same number of control zones must be in config.ini and main.py '
    for czone_param_main, czone_param_ini in zip(czones_params, czones_ini):
        czone_param_main.update(dict(idczone=czone_param_ini['id'],
                                     ckzn_d=czone_param_ini['cz_distance'],
                                     speedlimit=czone_param_ini['speed_limit'],
                                     x1y1x2y2=czone_param_ini['start'],
                                     x3y3x4y4=czone_param_ini['exit']))

    # BUILD THE CONTROL ZONES
    czones = []
    for czone_params in czones_params:
        czone = Control_zone(**czone_params)
        czones.append(czone)

    # Build the detector
    detector = YOLOV3Detector(cls=classes, weights_path=yolov3_weights_path)

    # Build the speed tracker
    trackerspeed = TrackerSpeedEstimator(video_capture=video_capture, czones=czones)

    while video_capture.isOpened():
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            break

        img = frame.copy()

        result, detections = detector.detect(img)
        if len(detections) == 0:
            continue

        trackerspeed.track(detections)

        # counting objects
        trackerspeed.map_centroid_class()

        # Couting Display
        for counter in counters:
            _ = counter.count_class(trackerspeed.objects, trackerspeed.mapped_centroid_classes)
            counter.count_display(img=result, icons=icons)

        cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
        for czone in czones:
            result = czone.display_zone(img=result)

        trackerspeed.compute_speed()
        trackerspeed.display_speed(img=result)

        trackerspeed.display_tracking(img=result)

        out.write(result)
        cv2.imshow("video", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    video_capture.release()
    cv2.destroyAllWindows()
    out.release()


def get_args():
    parser = argparse.ArgumentParser(description='Run the traffic counting demo script')
    parser.add_argument('--in', dest='input_path', default="videos/Road_traffic_cut.mp4",
                        help='Path the input video')
    parser.add_argument('--out', dest='output_path', default="output.avi",
                        help='Path the output video (MUST BE .avi)')
    parser.add_argument('-w', dest='weights', default="yolov3_weights/pretrained-yolov3.h5",
                        help='Path the yolov3 weights')
    args = parser.parse_args()
    return args.input_path, args.output_path, args.weights


if __name__ == "__main__":
    main()
