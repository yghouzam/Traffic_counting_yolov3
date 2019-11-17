# [Traffic counting using yolov3]

![Example of trafic counting](examples/example.gif.gif "Image Title")

## part 1. Quick start
1. Clone this file
```bashrc
$ git clone https://github.com/yghouzam/Traffic_counting_yolov3.git
```
2.  You are supposed  to install some dependencies before getting out hands with these codes.
```bashrc
$ cd Traffic_counting_yolov3
$ pip install -r requirements.txt
```
3. Download pretrained yolov3 COCO weights
```bashrc
$ cd yolov3_weights
$ wget https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5
```

4. Configure the config.ini file
(choose your checking zones and counters positions)
```bashrc
[COUNTERS]
COUNTER1 = {'x1' : 325, 'y1' : 400, 'x2' : 600, 'y2' : 400}
COUNTER2 = {'x1' : 665, 'y1' : 400, 'x2' : 912, 'y2' : 400}


[CONTROL_ZONE]
# define the control zones
# Speed_limit is in km/h
# cz_distance is in meters
CZ1 = {'id':0, 'speed_limit' : 130, 'cz_distance' : 52 ,
       'start': {'x1' : 630, 'y1' : 342, 'x2' : 451, 'y2' : 342},
       'exit' : {'x3' : 170, 'y3' : 475, 'x4' : 585, 'y4' : 475}}

CZ2 = {'id':1, 'speed_limit' : 130, 'cz_distance' : 52 ,
       'start': {'x1' : 675, 'y1' : 475, 'x2' : 1035, 'y2' : 475},
       'exit' : {'x3' : 820, 'y3' : 348, 'x4' : 665, 'y4' : 348}}
```

5. Run the demo script
```bashrc
$ python main.py --in videos/Road_traffic_2.mp4 --out output.avi
```

## part 2. References

[-**`Imageai`**](https://github.com/OlafenwaMoses/ImageAI)<br>

[-**`Understanding YOLO`**](https://hackernoon.com/understanding-yolo-f5a74bbc7967)

