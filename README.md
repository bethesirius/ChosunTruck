# ChosunTruck

## Introduction
Chosun Truck is autonomous driving solution on Euro Truck Simulator 2(https://eurotrucksimulator2.com/).
Recently, autonomous driving technology has become a big issue and we have studied the technology related to this.
It is being develop in a simulator environment called Euro Truck Simulator 2 to study it with vehicles.
Because this simulator provides a good test environment that is similar to the real road, we choose it.

## Features
* You can drive a vehicle without handling.
* You can understand the principle of autonomous driving.
* This feature only works on the highway.

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

## Demo Video
https://goo.gl/photos/aHQ5ZMKdTQJyMuaM9

## Contributers
Chiwan Song, chi3236@gmail.com
JaeCheol Sim, ssimpcp@gmail.com
Seongjoon Chu, hs4393@gmail.com

## How to Contribute
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
