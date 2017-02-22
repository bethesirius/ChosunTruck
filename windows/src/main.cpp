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

int main() {

	//cudaf();


	long long int sum = 0;
	long long int i = 0;

	while (true) {
		auto begin = chrono::high_resolution_clock::now();
		// ETS2
		HWND hWnd = FindWindow("prism3d", NULL);
		HWND consoleWindow = GetConsoleWindow();
		
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
		
		//cv::cvtColor(contours, contours, COLOR_GRAY2RGB);
		
		/*
		auto end = chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		ms++;
		sum += ms;
		cout << 1000 / ms << "fps       avr:" << 1000 / (sum / (++i)) << endl;
		*/

		int bottom_center = 160;
		int sum_centerline = 0;
		int count_centerline = 0;
		int first_centerline = 0;
		int last_centerline = 0;
		double avr_center_to_left = 0;
		double avr_center_to_right = 0;

		//#pragma omp parallel for
		for (int i = 240; i>10; i--) {
			double center_to_right = -1;
			double center_to_left = -1;

			for (int j = 0; j<150; j++) {
				if (contours.at<uchar>(i, bottom_center + j) == 112 && center_to_right == -1) {
					center_to_right = j;
				}
				if (contours.at<uchar>(i, bottom_center - j) == 112 && center_to_left == -1) {
					center_to_left = j;
				}
			}
			if (center_to_left != -1 && center_to_right != -1) {
				int centerline = (center_to_right - center_to_left + 2 * bottom_center) / 2;
				if (first_centerline == 0) {
					first_centerline = centerline;
				}
				cv::circle(outputImg, Point(centerline, i), 1, Scalar(30, 255, 30), 3);
				cv::circle(outputImg, Point(centerline + center_to_right+20, i), 1, Scalar(255, 30, 30), 3);
				cv::circle(outputImg, Point(centerline - center_to_left+10, i), 1, Scalar(255, 30, 30), 3);
				sum_centerline += centerline;
				avr_center_to_left = (avr_center_to_left * count_centerline + center_to_left) / count_centerline + 1;
				avr_center_to_right = (avr_center_to_right * count_centerline + center_to_right) / count_centerline + 1;
				last_centerline = centerline;
				count_centerline++;
			}
			else {
			}
		}

		imshow("Lines", contours);
		imshow("Road", outputImg);
		cv::moveWindow("Lines", width / 1.6, height / (10.8));
		cv::moveWindow("Road", width / (128/101), height / (10.8));
		SetWindowPos(consoleWindow, 0, width / 1.6, height / 2.7, 600, 400, SWP_NOZORDER);
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

			}
			if (row_right != 255) {
				row_right = gray.at<uchar>(row_number, 159 + right); 
				right++;

			}
			i++;

		}
		SetActiveWindow(hWnd);

		int average = (left == -150 || right == 150) ? 0 : left + right;
		if (left + right < -50)
		{
			cout << "go left ";

			INPUT input[2];
			input[0].type = INPUT_KEYBOARD;
			// Translating 'A' to Scan Code, then pressing down
			input[0].ki.wScan = MapVirtualKey(0x41, MAPVK_VK_TO_VSC);
			input[0].ki.dwFlags = KEYEVENTF_SCANCODE;
			// Translating 'A' to Scan Code, then releasing key
			input[1].ki.wScan = MapVirtualKey(0x41, MAPVK_VK_TO_VSC);
			input[1].ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
			SendInput(2, input, sizeof(INPUT));
		}
		else if (left + right > -50 && left + right < 50){
			cout << "go straight ";
			for (int x = 0, y = 0; x < 700 && y < 700; x += 10, y += 10)
			{
				/*
				INPUT input[2]; // Using SendInput to send input commands
				input[0].type = INPUT_KEYBOARD;
				// Translating 'A' to Scan Code, then releasing key
				input[0].ki.wScan = MapVirtualKey(0x41, MAPVK_VK_TO_VSC);
				input[0].ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
				// Translating 'D' to Scan Code, then releasing key
				input[1].ki.wScan = MapVirtualKey(0x44, MAPVK_VK_TO_VSC);
				input[1].ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
				SendInput(2, input, sizeof(INPUT));
				
			}
		}
		/* else{
			cout << "go right ";
			{
				INPUT input[2];
				input[0].type = INPUT_KEYBOARD;
				// Translating 'D' to Scan Code, then pressing down
				input[0].ki.wScan = MapVirtualKey(0x44, MAPVK_VK_TO_VSC);
				input[0].ki.dwFlags = KEYEVENTF_SCANCODE;
				// Translating 'D' to Scan Code, then releasing key
				input[1].ki.wScan = MapVirtualKey(0x44, MAPVK_VK_TO_VSC);
				input[1].ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
				SendInput(2, input, sizeof(INPUT));
			}
		}
	cout << "left: " << left << ", right: " << right << ", average: " << average << endl;
	*/
	}
	return 0;
}
