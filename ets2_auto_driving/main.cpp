#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/imageproc/imageproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/cudaarithm.hpp"
#include <opencv2/highgui/highgui_c.h>
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
		// NOTEPAD
		//HWND hWnd = FindWindow("Photo_Light", NULL);
		Mat image, outputImg;
		hwnd2mat(hWnd).copyTo(image);

		// Mat to GpuMat
		//cuda::GpuMat imageGPU;
		//imageGPU.upload(image);

		medianBlur(image, image, 3); 
		//bilateralFilter(imageGPU, imageGPU, 15, 80, 80);

		int width = 0, height = 0;

		RECT windowsize;
		GetClientRect(hWnd, &windowsize);

		height = 768; // change this to whatever size you want to resize to
		width = 1024;

		// The 4-points at the input image	
		vector<Point2f> origPoints;
		
		origPoints.push_back(Point2f(0, (height-50)));
		origPoints.push_back(Point2f(width, height-50));
		origPoints.push_back(Point2f(width/2+125, height/2+30));
		origPoints.push_back(Point2f(width/2-125, height/2+30));
		

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

		//imageGPU.download(image);
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
		imshow("Test", contours);
		waitKey(1);
		/*
		auto end = chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		ms++;
		sum += ms;
		cout << 1000 / ms << "fps       avr:" << 1000 / (sum / (++i)) << endl;
		*/
		///////////////////////////////////////
		
		unsigned char row_center = gray.at<unsigned char>(10, 160);
		
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
				row_left = gray.at<unsigned char>(row_number, 159 + left);
				left--;
				
			}
			if (row_right != 255) {
				row_right = gray.at<unsigned char>(row_number, 159 + right);
				right++;
				
			}
			i++;
			
		}
		SetActiveWindow(hWnd);
		int average = (left == -150 || right == 150) ? 0 : left + right;
		if (left + right < -50){
			cout << "go left ";
								
								//SendMessage(hWnd, WM_KEYUP, 0x44, 0);
								//Sleep(100);
								//SendMessage(hWnd, WM_KEYDOWN, 0x74, 0);
								//Sleep(100);
								//SendMessage(hWnd, WM_KEYUP, 0x74, 0);
								
								HKL kbl = GetKeyboardLayout(0);
								
								// Create a generic keyboard event structure
								INPUT ip;
								Sleep(1000);
								ip.type = INPUT_KEYBOARD;
								ip.ki.wScan = 0;
								ip.ki.time = 0;
								ip.ki.dwExtraInfo = 0;
								//while (1)
								{
									ip.ki.wVk = 0x74;
									ip.ki.dwFlags = 0;
									SendInput(1, &ip, sizeof(ip));

									ip.ki.wVk = 0x74;
									ip.ki.dwFlags = KEYEVENTF_KEYUP; // Releases 0x74
									SendInput(1, &ip, sizeof(ip));

									ip.ki.dwFlags = 0;
									ip.ki.wVk = 0x44; // Presses 'D'
									SendInput(1, &ip, sizeof(ip));
								}
								
		}
		else if (left + right > -50 && left + right < 50){
			cout << "go straight ";
			for (int x = 0, y = 0; x < 700 && y<700; x += 10, y += 10)
				 {
				SendMessage(hWnd, WM_MOUSEMOVE, 0, MAKELPARAM(x, y));
				Sleep(10);
				}
									//SendMessage(hWnd, WM_KEYUP, 0x44, 0);
									//Sleep(10);
									//SendMessage(hWnd, WM_KEYUP, 0x41, 0);
									
									INPUT ip;
									Sleep(5000);
									ip.type = INPUT_KEYBOARD;
									ip.ki.wScan = 0;
									ip.ki.time = 0;
									ip.ki.dwExtraInfo = 0;
									while (1)
										ip.ki.wVk = 0x44;
										ip.ki.dwFlags = KEYEVENTF_KEYUP; // Releases 0x44
										SendInput(1, &ip, sizeof(ip));

										ip.ki.wVk = 0x41;
										ip.ki.dwFlags = KEYEVENTF_KEYUP; // Releases 0x41
										SendInput(1, &ip, sizeof(ip));
										Sleep(1000);
									
		}
		else{
			cout << "go right ";
						//SendMessage(hWnd, WM_KEYUP, 0x41, 0);
						//Sleep(100);
						//SendMessage(hWnd, WM_KEYDOWN, 0x74, 0);
						//Sleep(100);
						//SendMessage(hWnd, WM_KEYUP, 0x74, 0);

						INPUT ip;
						Sleep(1000);
						ip.type = INPUT_KEYBOARD;
						ip.ki.wScan = 0;
						ip.ki.time = 0;
						ip.ki.dwExtraInfo = 0;
						while (1)
						{
							ip.ki.wVk = 0x74;
							ip.ki.dwFlags = 0; // Presses 0x41
							SendInput(1, &ip, sizeof(ip));

							ip.ki.wVk = 0x74;
							ip.ki.dwFlags = KEYEVENTF_KEYUP; // Releases 0x41
							SendInput(1, &ip, sizeof(ip));

							ip.ki.dwFlags = 0;
							ip.ki.wVk = 0x41; // Presses 'A'
							SendInput(1, &ip, sizeof(ip));

							Sleep(1000);
						}
						// keybd_event(VK_RIGHT, 0, KEYEVENTF_KEYUP, 0);
		}
		cout << "left: " << left << ", right: " << right << ", average: " << average << endl;
				///////////////////////////////////////
			
			
			imshow("Test", gray);
	}
	return 0;
}