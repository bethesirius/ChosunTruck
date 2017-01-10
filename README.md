# ChosunTruck

## Introduction
Chosun Truck is a Euro Truck Simulator 2 autonomous driving solution.
In recent years, autonomous driving technology has become a big issue and we have studied the technology related to this.
It was developed in a simulator environment called Euro Truck Simulator 2 to study with real vehicles.
Because this simulator provides an environment similar to a real road, we determined it to be a good test environment.

## This Project...
* help to understand the principle of autonomous driving.
* Autonomous driving only works on the expressway.

## master
OS: window 7 64bits
Development Tool: Visual Studio 2015
OpenCV version: 3.0

Because mouse movements or keyboard input do not work in Windows to navigate the truck in Euro Truck Simulator 2, the master version is only available for lane detection.

## linux
OS: ubuntu
OpenCV version: 3.0
```
g++ -o auto_drive main2.cpp IPM.cpp lineinder.cpp uinput.c `pkg-config opencv --cflags --libs` -std=c++11 -lX11 -Wall -fopenmp -O3 -march=native
```

## Video URL
https://drive.google.com/open?id=0B0NMsYygQ6icMXUzcEJhS1RRT2M
