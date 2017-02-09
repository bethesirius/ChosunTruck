# ChosunTruck

## Introduction
ChosunTruck is an autonomous driving solution for [Euro Truck Simulator 2](https://eurotrucksimulator2.com/).
Recently, autonomous driving technology has become a big issue and we have studied the technology related to this.
It is being developed in a simulator environment called Euro Truck Simulator 2 to study it with vehicles.
Because this simulator provides a good test environment that is similar to the real road, we chose it.

## Features
* You can drive a vehicle without handling it yourself.
* You can understand the principle of autonomous driving.
* These features only work while driving on a highway.

## How To Run It
### Windows
OS: Windows 7 64bits

IDE: Visual Studio 2015

OpenCV version: 3.0

** **NOTICE: Because mouse and keyboard input DOES NOT work on Windows, this is only available for lane detection. For more informations, see also [#4](https://github.com/bethesirius/ChosunTruck/issues/4)**

- Open the visual studio project and build it. 
- Run ETS2 in windowed mode and set resolution to 1024 * 768.(It will work properly with 1920 * 1080 screen resolution and 1024 * 768 window mode ETS2.)

### Linux
OS: Ubuntu 16.04 LTS

OpenCV version: 3.0

- Clone the linux branch.
```
git clone -b linux https://github.com/bethesirius/ChosunTruck.git
```
- Build the source code with below command.
```
g++ -o auto_drive main2.cpp IPM.cpp lineinder.cpp uinput.c `pkg-config opencv --cflags --libs` -std=c++11 -lX11 -Wall -fopenmp -O3 -march=native
```
- Run ETS2 in windowed mode and set resolution to 1024 * 768. (It will work properly with 1920 * 1080 screen resolution and 1024 * 768 window mode ETS2)
- It cannot find the ETS2 window automatically, you should move the ETS2 window to right-down corner to fix this.
- In ETS2 Options, set controls to`Keyboard + Mouse Steering`.
- Go to a highway and set truck's speed to 40~60km/h. (I recommend you turn on cruise mode to set the speed easily)
- Run this program as root!
```
sudo ./auto_drive
```
----
If you have some problems to run this project, reference the demo video below. Or, open a issue to contact our team.

## Demo Video
https://goo.gl/photos/aHQ5ZMKdTQJyMuaM9

## Contributers
Chiwan Song, chi3236@gmail.com
JaeCheol Sim, ssimpcp@gmail.com
Seongjoon Chu, hs4393@gmail.com

## How To Contribute
Anyone who is interest in this procject is welcome! Just Fork it and Pull Requests!

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
