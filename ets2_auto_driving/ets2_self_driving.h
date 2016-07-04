#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Windows.h>
#include <iostream>
#include <string>
#include <chrono>


using namespace cv;
using namespace std;

Mat hwnd2mat(HWND hwnd);
void cudaf();