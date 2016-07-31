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

int main() {

	//cudaf();


	long long int sum = 0;
	long long int i = 0;

	while (true) {
		auto begin = chrono::high_resolution_clock::now();
		// ETS2
		HWND hWnd = FindWindow("prism3d", NULL);
		// NOTEPAD
		//HWND hWnd = FindWindow("Notepad", NULL);
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
		origPoints.push_back(Point2f(0, height));
		origPoints.push_back(Point2f(width, height));
		origPoints.push_back(Point2f(width / 2 + 30, 140));
		origPoints.push_back(Point2f(width / 2 - 50, 140));

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
		ipm.drawPoints(origPoints, image);

		cv::Mat contours;
		cv::Canny(outputImg, contours, 125, 350);
		LineFinder ld; // 인스턴스 생성

		// 확률적 허프변환 파라미터 설정하기
		ld.setLineLengthAndGap(100, 30);
		ld.setMinVote(50);

		std::vector<cv::Vec4i> li = ld.findLines(contours);
		ld.drawDetectedLines(outputImg);

		imshow("Test", outputImg);
		waitKey(1);
		auto end = chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		ms++;
		sum += ms;
		cout << 1000 / ms << "fps       avr:" << 1000 / (sum / (++i)) << endl;

		/*// View
		imshow("Input", image);
		imshow("Output", outputImg);
		waitKey(1);*/
	}

	/*cv::Mat contours;
	cv::Canny(image, contours, 125, 350);
	LineFinder ld; // 인스턴스 생성

	// 확률적 허프변환 파라미터 설정하기
	ld.setLineLengthAndGap(100, 30);
	ld.setMinVote(50);

	std::vector<cv::Vec4i> li = ld.findLines(contours);
	ld.drawDetectedLines(image);

	imshow("Test", image);
	waitKey(1);
	auto end = chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	ms++;
	sum += ms;
	cout << 1000 / ms << "fps       avr:" << 1000 / (sum / (++i)) << endl;
	*/

	return 0;
}