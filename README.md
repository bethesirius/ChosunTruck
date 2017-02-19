# <img src="https://github.com/bethesirius/ChosunTruck/blob/master/README/Logo.png", width="64">ChosunTruck

# ChosunTruck

## Introduction
ChosunTruck is an autonomous driving solution for [Euro Truck Simulator 2](https://eurotrucksimulator2.com/).
Recently, autonomous driving technology has become a big issue and as a result we have been studying technology that incorporates this.
It is being developed in a simulated environment called Euro Truck Simulator 2 to allow us to study it using vehicles.
We chose Euro Truck Simulator 2 because this simulator provides a good test environment that is similar to the real road.

## Features
* You can drive a vehicle without handling it yourself.
* You can understand the principles of autonomous driving.
* (Experimental) You can detect where other vehicles are.

## How To Run It
### Windows

#### Dependencies
- OS: Windows 7 (64bit)

- IDE: Visual Studio 2013

- OpenCV version: 3.2

- [Cuda Toolkit 7.5](https://developer.nvidia.com/cuda-75-downloads-archive) (Note: Do an ADVANCED INSTALLATION. ONLY install the Toolkit + Integration to Visual Studio. Do NOT install the drivers + other stuff it would normally give you. Once installed, your project properties should look like this: https://i.imgur.com/e7IRtjy.png)

#### Required to allow input to work in Windows:
##### Note: Input is not currently working correctly.
- **Go to C:\Users\YOURUSERNAME\Documents\Euro Truck Simulator 2\profiles and edit controls.sii from di8.keyboard/fusion.mouse to sys.keyboard/sys.mouse.** (thanks Komat!)

#### Then:
- Open the visual studio project and build it. 
- Run ETS2 in windowed mode and set resolution to 1024 * 768.(It will work properly with 1920 * 1080 screen resolution and 1024 * 768 window mode ETS2.)



### Linux
#### Dependencies
- OS: Ubuntu 16.04 LTS


- [OpenCV version: 3.1](http://embedonix.com/articles/image-processing/installing-opencv-3-1-0-on-ubuntu/)

- (Optional) Tensorflow version: 0.12.1

### Build the source code with the following command (inside the linux directory).
```
make
```
#### Then:
- Run ETS2 in windowed mode and set its resolution to 1024 * 768. (It will work properly with 1920 * 1080 screen resolution and 1024 * 768 windowed mode ETS2)
- It cannot find the ETS2 window automatically. Move the ETS2 window to the right-down corner to fix this.
- In ETS2 Options, set controls to 'Keyboard + Mouse Steering', 'left click' to acclerate, and 'right click' to brake.
- Go to a highway and set the truck's speed to 40~60km/h. (I recommend you turn on cruise mode to set the speed easily)
- Run this program!

#### To enable car detection mode, add -D or --Car_Detection.
```
./ChosunTruck [-D|--Car_Detection]
```
## Troubleshooting
### OpenCV CUDA Libraries

**Having trouble building OpenCV libraries with CMake? No worries, just use the prebuilt OpenCV libraries provided through nuget.**

- For Release Libaries: https://www.nuget.org/packages/opencvcuda-release/
- For Debug Libraries: https://www.nuget.org/packages/opencvcuda-debug/

### Linker Errors

**Getting Linker Errors (LNKxxxx)?**

Go to: 

```
 My Project --> properties --> linker --> input --> additional dependencies
```
and change the path of your Linker dependencies to

```
opencv_calib3d310.lib;opencv_core310.lib;opencv_features2d310.lib;opencv_flann310.lib;opencv_highgui310.lib;opencv_imgcodecs310.lib;opencv_imgproc310.lib;opencv_ml310.lib;opencv_objdetect310.lib;opencv_photo310.lib;opencv_shape310.lib;opencv_stitching310.lib;opencv_superres310.lib;opencv_ts310.lib;opencv_video310.lib;opencv_videoio310.lib;opencv_videostab310.lib;opencv_viz310.lib;%(AdditionalDependencies)
```

Then, go to 
```
C/C++ --> general --> additional #include dependencies
```

and change the path to where your opencv libraries are located

### Assertion failed

**Getting an error that looks like this when in Debug mode?**

```
OpenCV Error: Assertion failed (src.checkVector(2, CV_32F) == 4 && dst.checkVect or(2, CV_32F) == 4) in cv::getPerspectiveTransform, file ...\opencv_sources\modules\imgproc\src\imgwarp.cpp, line 6353
```
- Switch to Release build for now. Usually this is caused by a wrong Mat type being used (in this case, CV_32F).

**This error occurs ONLY at the first execution after the build, but you can ignore it.**

```
OpenCV Error: Assertion failed (ssize.area() > 0) in cv::remapBilinear, file ...\opencv\sources\modules\imgproc\src\imgwarp.cpp, line 3868 
```
- Program still runs well and no assertion error occurs after the second execution

If you have some problems running this project, reference the demo video below. Or, [open a issue to contact our team](https://github.com/bethesirius/ChosunTruck/issues).

## Demo Video
Lane Detection (Youtube link)

[![youtube link](http://img.youtube.com/vi/vF7J_uC045Q/0.jpg)](http://www.youtube.com/watch?v=vF7J_uC045Q)

Lane Detection + Vehicle Detection (Youtube link)

[![youtube link](http://img.youtube.com/vi/w6H2eGEvzvw/0.jpg)](http://www.youtube.com/watch?v=w6H2eGEvzvw)

## Founders
- Chiwan Song, chi3236@gmail.com

- JaeCheol Sim, simjaecheol@naver.com

- Seongjoon Chu, hs4393@gmail.com

## Contributors
- [zappybiby](https://github.com/zappybiby)

## How To Contribute
Anyone who is interested in this project is welcome! Just fork it and pull requests!

## License
ChosunTruck, Euro Truck Simulator 2 auto driving solution
Copyright (C) 2017 chi3236, bethesirius, uoyssim

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
