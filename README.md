## Self Driving Car Final Project Report

### Team Name: Cruise Control


      _____            _             _____            _             _
     / ____|          (_)           / ____|          | |           | |
    | |     _ __ _   _ _ ___  ___  | |     ___  _ __ | |_ _ __ ___ | |
    | |    | '__| | | | / __|/ _ \ | |    / _ \| '_ \| __| '__/ _ \| |
    | |____| |  | |_| | \__ \  __/ | |___| (_) | | | | |_| | | (_) | |
     \_____|_|   \__,_|_|___/\___|  \_____\___/|_| |_|\__|_|  \___/|_|_

### Team Member Names:
Garrett Pitcher <garrett.pitcher@gmail.com>  
M. M. (need approval)  
Hanqiu Jiang <hanq.jiang@gmail.com>  
Shaun Cosgrove <shaun.cosgrove@bogglingtech.com>  
W. G. (need approval)


<img src="./doc/traffic_light_stop.gif">  



### System Architecture

<img src="./doc/software_architecture_requirements.png" width="700">  
<img src="./doc/Overview_Messages_Nodes.png" width="900">  
<img src="./doc/rosgraph.png" width="700">

### Perception

<img src="./doc/ssd_architecture.png" width="700">

A SSD MobileNet architecture is deployed for recorded our traffic light detection. We therefore train our own labeled images on a pretrained model from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) utilizing the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).  

<img src="./doc/tl_detection.gif">  

### Planning

<img src="./doc/rviz_waypoints.png" width="500">

### Control

<img src="./doc/controller_tuning.png" width="500">  
