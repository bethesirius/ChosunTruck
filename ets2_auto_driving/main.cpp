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

#define PI 3.1415926

using namespace cv;
using namespace std;



class LineFinder {
private:
	cv::Mat image; // 원 영상
	std::vector<cv::Vec4i> lines; // 선을 감지하기 위한 마지막 점을 포함한 벡터
	double deltaRho;
	double deltaTheta; // 누산기 해상도 파라미터
	int minVote; // 선을 고려하기 전에 받아야 하는 최소 투표 개수
	double minLength; // 선에 대한 최소 길이
	double maxGap; // 선에 따른 최대 허용 간격

public:
	LineFinder() : deltaRho(1), deltaTheta(PI / 180), minVote(10), minLength(0.), maxGap(0.) {}
	// 기본 누적 해상도는 1각도 1화소 
	// 간격이 없고 최소 길이도 없음

	// 해당 세터 메소드들

	// 누적기에 해상도 설정
	void setAccResolution(double dRho, double dTheta) {
		deltaRho = dRho;
		deltaTheta = dTheta;
	}

	// 투표 최소 개수 설정
	void setMinVote(int minv) {
		minVote = minv;
	}

	// 선 길이와 간격 설정
	void setLineLengthAndGap(double length, double gap) {
		minLength = length;
		maxGap = gap;
	}

	// 허프 선 세그먼트 감지를 수행하는 메소드
	// 확률적 허프 변환 적용
	std::vector<cv::Vec4i> findLines(cv::Mat& binary) {
		lines.clear();
		cv::HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, 200, maxGap);
		return lines;
	} // cv::Vec4i 벡터를 반환하고, 감지된 각 세그먼트의 시작과 마지막 점 좌표를 포함.

	// 위 메소드에서 감지한 선을 다음 메소드를 사용해서 그림
	// 영상에서 감지된 선을 그리기
	void drawDetectedLines(cv::Mat &image, cv::Scalar color = cv::Scalar(0, 0, 255)) {

		// 선 그리기
		std::vector<cv::Vec4i>::const_iterator it2 = lines.begin();

		while (it2 != lines.end()) {
			cv::Point pt1((*it2)[0], (*it2)[1]);
			cv::Point pt2((*it2)[2], (*it2)[3]);
			cv::line(image, pt1, pt2, color, 3);
			++it2;
		}
	}
};


Mat hwnd2mat(HWND hwnd) {

	HDC hwindowDC, hwindowCompatibleDC;

	int height, width, srcheight, srcwidth;
	HBITMAP hbwindow;
	Mat src;
	BITMAPINFOHEADER  bi;

	hwindowDC = GetDC(hwnd);
	hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

	RECT windowsize;    // get the height and width of the screen
	GetClientRect(hwnd, &windowsize);

	srcheight = windowsize.bottom / 2 + 200;// change this to whatever size you want to resize to
	srcwidth = windowsize.right;
	height = windowsize.bottom / 2 + 200; // change this to whatever size you want to resize to
	width = windowsize.right;

	src.create(height, width, CV_8UC4);

	// create a bitmap
	hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
	bi.biWidth = width;
	bi.biHeight = -height;  //this is the line that makes it draw upside down or not
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	// use the previously created device context with the bitmap
	SelectObject(hwindowCompatibleDC, hbwindow);
	// copy from the window device context to the bitmap device context
	StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 340, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
	GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

	// avoid memory leak
	DeleteObject(hbwindow); DeleteDC(hwindowCompatibleDC); ReleaseDC(hwnd, hwindowDC);

	return src;
}

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
	Mat image = hwnd2mat(hWnd);
	
	// Mat to GpuMat
	//cuda::GpuMat imageGPU;
	//imageGPU.upload(image);

	medianBlur((Mat)image, (Mat)image, 3);
	//cuda::bilateralFilter(imageGPU, imageGPU, );

	cv::Mat contours;
	cv::Canny((Mat)image, contours, 125, 350);
	LineFinder ld; // 인스턴스 생성

	// 확률적 허프변환 파라미터 설정하기
	ld.setLineLengthAndGap(100, 30);
	ld.setMinVote(50);

	std::vector<cv::Vec4i> li = ld.findLines(contours);
	ld.drawDetectedLines((Mat)image);

	imshow("Test", (Mat)image);
	waitKey(1);
	auto end = chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	ms++;
	sum += ms;
	cout << 1000 / ms << "fps       avr:" << 1000 / (sum / (++i)) << endl;
	}	

	return 0;
}