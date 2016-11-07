# ChosunTruck
Euro Truck Simulator 2 autonomous driving solution

g++ -o auto_drive main2.cpp IPM.cpp lineinder.cpp uinput.c `pkg-config opencv --cflags --libs` -std=c++11 -lX11 -Wall -fopenmp -O3 -march=native
