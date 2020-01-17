# Multi-Object-Tracking
Computer Vision project done as a Synthesis Special Course at DTU Elektro 2019


Objective: develop computer vision multi-object tracking system for mobile robotics localization in an outdoor environment


The aim of the project is to develop a part of a system that will improve localization of a mobile robot based on vision recognition of natural landmarks in an outdoor environment. Current robots localization system consists of IMU and wheel odometry data, which at long distances accumulates drift and are unable to precisely localize robot. Therefore, there is a need to relocalize robot according to visible landmarks in the natural environment. The environment that the robot will operate is a walking path around Technical University of Denmark. The robot should be able to localize itself independently of season of a year, which is important, because the environment changes throughout a year. The only consistent element of the environment are trees in the environment. Therefore, the idea of this project is to track the trees in order to build a map that a robot could refer to in order to refine its global position. The project focuses on tracking already detected trees. The assumption is that in each image bounding boxes and binary masks are available. The point is to connect multiple objects with their corresponding detections in a sequence of frames as shown below.



![](Example_imgs/example_gif.gif)



In this project, a successful multi-object tracking system was developed. First, an analysis of the given dataset and problem was thoroughly considered. Then based on initial tries of implemented tracking systems from OpenCV library, the MedianFlow algorithm was chosen for the development of the final algorithm. The project extended capabilities of the MedianFlow algorithm to track not only single object, but multiple objects, as well as it introduced a correction step based on the information provided by a detection algorithm. From the evaluation metrics calculated on three sequences that varied in tracking difficulty, it is possible to conclude that the tracking system is working properly. It can handle tracking of multiple objects throughout a series of images and report when the tracking is lost.
