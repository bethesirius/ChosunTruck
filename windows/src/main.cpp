#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/imageproc/imageproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudafilters.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <Windows.h>
#include <iostream>
#include <string>
#include <chrono>
#include "ets2_self_driving.h"
#include "IPM.h"

#define PI 3.1415926

using namespace cv;
using namespace std;

void Thinning(Mat input, int row, int col);

void GetDesktopResolution(int& monitorHorizontal, int& monitorVertical)
{
	RECT desktop;
	// Get a handle to the desktop window
	const HWND hDesktop = GetDesktopWindow();
	// Get the size of screen to the variable desktop
	GetWindowRect(hDesktop, &desktop);
	// The top left corner will have coordinates (0,0)
	// and the bottom right corner will have coordinates
	// (horizontal, vertical)
	monitorHorizontal = desktop.right;
	monitorVertical = desktop.bottom;
}

int main() {

	//cudaf();


	int monitorHorizontal = 0;
	int monitorVertical = 0;
	long long int sum = 0;
	long long int i = 0;
	int diffOld = 0;

	while (true) {
		// Press '+' to pause
		if (GetAsyncKeyState(VK_OEM_PLUS) & 0x8000){
			while (true) {
				// Press '-' to start
				if (GetAsyncKeyState(VK_OEM_MINUS) & 0x8000){
					break;
				}
			}
		}
		auto begin = chrono::high_resolution_clock::now();
		// Grab the game window
		HWND hWnd = FindWindow("prism3d", NULL);
		// Grab the console window
		HWND consoleWindow = GetConsoleWindow();
		// Grab Monitor
		GetDesktopResolution(monitorHorizontal, monitorVertical);
		
		Mat image, outputImg;
		hwnd2mat(hWnd).copyTo(image);

		// Mat to GpuMat
		//cuda::GpuMat imageGPU;
		//imageGPU.upload(image);

		medianBlur(image, image, 3); 
		//cv::cuda::bilateralFilter(imageGPU, imageGPU, );

		int width = 0, height = 0;

		RECT windowsize;
		GetClientRect(hWnd, &windowsize);

		height = windowsize.bottom; // change this to whatever size you want to resize to
		width = windowsize.right;

		// The 4-points at the input image	
		vector<Point2f> origPoints;
		
		origPoints.push_back(Point2f(0-400, (height-50)));
		origPoints.push_back(Point2f(width+400, height-50));
		origPoints.push_back(Point2f(width/2+80, height/2+30));
		origPoints.push_back(Point2f(width/2-80, height/2+30));
		

		// The 4-points correspondences in the destination image
		vector<Point2f> dstPoints;
		dstPoints.push_back(Point2f(0, height));
		dstPoints.push_back(Point2f(width, height));
		dstPoints.push_back(Point2f(width, 0));
		dstPoints.push_back(Point2f(0, 0));

		// IPM object
		IPM ipm(Size(width, height), Size(width, height), origPoints, dstPoints);

		// Process
		//clock_t begin = clock();
		ipm.applyHomography(image, outputImg);
		//clock_t end = clock();
		//double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		//printf("%.2f (ms)\r", 1000 * elapsed_secs);
		//ipm.drawPoints(origPoints, image);

		//Mat row = outputImg.row[0];
		cv::Mat gray;
		cv::Mat blur;
		cv::Mat sobel;
		cv::Mat contours;
		cv::resize(outputImg, outputImg, cv::Size(320, 240));
		cv::cvtColor(outputImg, gray, COLOR_RGB2GRAY);
		cv::blur(gray, blur, cv::Size(10, 10));
		cv::Sobel(blur, sobel, blur.depth(), 1, 0, 3, 0.5, 127);
		cv::threshold(sobel, contours, 145, 255, CV_THRESH_BINARY);
		//Thinning(contours, contours.rows, contours.cols);
		//cv::Canny(gray, contours, 125, 350);
		
		LineFinder ld; // 인스턴스 생성

		// 확률적 허프변환 파라미터 설정하기
		
		ld.setLineLengthAndGap(20, 120);
		ld.setMinVote(55);

		std::vector<cv::Vec4i> li = ld.findLines(contours);
		ld.drawDetectedLines(contours);
		
		// cv::cvtColor(contours, contours, COLOR_GRAY2RGB);
		
		/*
		auto end = chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		ms++;
		sum += ms;
		cout << 1000 / ms << "fps       avr:" << 1000 / (sum / (++i)) << endl;
		*/
		
		SetActiveWindow(hWnd);
		
		// Creates pt.x and pt.y which are the coordinates of the mouse position.
		POINT pt;
		// Find current mouse position.
		GetCursorPos(&pt);
		
		cout << "current mouse pos: " << "x: " << pt.x << "y: " << pt.y << endl;

		int bottom_center = 160;
		int sum_centerline = 0;
		int count_centerline = 0;
		int first_centerline = 0;
		int last_centerline = 0;
		double avr_center_to_left = 0;
		double avr_center_to_right = 0;

		//#pragma omp parallel for
		for (int i = 240; i > 30; i--){
			double center_to_right = -1;
			double center_to_left = -1;

			for (int j = 0; j < 150; j++) {
				if (contours.at<uchar>(i, bottom_center + j) == 112 && center_to_right == -1) {
					center_to_right = j;
				}
				if (contours.at<uchar>(i, bottom_center - j) == 112 && center_to_left == -1) {
					center_to_left = j;
				}
			}
			if (center_to_left != -1 && center_to_right != -1){
				int centerline = (center_to_right - center_to_left + 2 * bottom_center) / 2;
				if (first_centerline == 0) {
					first_centerline = centerline;
				}
				cv::circle(outputImg, Point(centerline, i), 1, Scalar(30, 255, 30), 3);
				cv::circle(outputImg, Point(centerline + center_to_right + 20, i), 1, Scalar(255, 30, 30), 3);
				cv::circle(outputImg, Point(centerline - center_to_left + 10, i), 1, Scalar(255, 30, 30), 3);
				sum_centerline += centerline;
				avr_center_to_left = (avr_center_to_left * count_centerline + center_to_left) / count_centerline + 1;
				avr_center_to_right = (avr_center_to_right * count_centerline + center_to_right) / count_centerline + 1;
				last_centerline = centerline;
				count_centerline++;
			}
			else {}
		}
		
		int diff = 0;
		
		// Sets the x-coordinate of the mouse position to the center
		pt.x = width / 2;
		
		if (count_centerline != 0) {
			// In testing, we found that "bottom_center - 25" gave the best results.
			diff = sum_centerline / count_centerline - bottom_center - 25;


		imshow("Lines", contours);
		imshow("Road", outputImg);
		cv::moveWindow("Lines", monitorHorizontal / 1.6, monitorVertical / 10.8);
		cv::moveWindow("Road", monitorHorizontal / 1.2673, monitorVertical / 10.8);
		SetWindowPos(consoleWindow, 0, monitorHorizontal / 1.6, monitorVertical / 2.7, 600, 400, SWP_NOZORDER);
		SetWindowPos(hWnd, 0, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
		waitKey(1);
		// WORK IN PROGRESS FOR INPUT IMPLEMENTATION
		/*
		unsigned char row_center = gray.at<uchar>(10, 160);

		unsigned char row_left = 0;
		unsigned char row_right = 0;

		int left = 0;
		int right = 0;
		int i = 0;
		int row_number = 5;
		while (i < 150) {
			if (i == 149) {
				i = 0;
				row_left = 0;
				row_right = 0;
				left = 0;
				right = 0;
				row_number++;

			}
			if (row_left == 255 && row_right == 255) {
				row_number = 5;
				break;
			}
			if (row_left != 255) {
				// If matrix is of type CV_8U then use Mat.at<uchar>(y,x) (http://bit.ly/2kINZBI)
				row_left = gray.at<uchar>(row_number, 159 + left);  
				left--;

			// diff_max was determined by finding the maxmimum diff that can be used to go from center to the very edge of the lane.
			// It is an approximation. In testing, 65px was the farthest we could go from center in-game without losing lane.
			int diff_max = 70;

			// jerk_factor = how fast the wheel will turn
			// (1/70) = Limits steering to move 1px MAXMIMUM every time step (1 second).
			double jerk_factor = 1 / 70;

			// linearized_diff = diff on a scale of -1 to 1
			double linearized_diff = diff / diff_max;

			// our new wheel position is determined by adding a number, turn_amount, to the previous wheel position
			double turn_amount = linearized_diff * jerk_factor;

			if (turn_amount < .5){
				turn_amount = 0;
			}
			else {
				turn_amount = 1;
			}
			
			int moveMouse = (pt.x + diffOld + turn_amount);

			cout << "Steer: " << turn_amount << "px ";

			SetCursorPos(moveMouse, height / 2);
			
			// int degree = atan2(last_centerline - first_centerline, count_centerline) * 180 / PI;
			diffOld = diff;
		}
	}
	return 0;
}
