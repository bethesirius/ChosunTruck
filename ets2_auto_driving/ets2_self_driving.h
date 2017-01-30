#ifndef __ets_self_driving_h__
#define __ets_self_driving_h__

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

#define PI 3.1415926


using namespace cv;
using namespace std;

class LineFinder{

private:
	cv::Mat image; // 원 영상
	std::vector<cv::Vec4i> lines; // 선을 감지하기 위한 마지막 점을 포함한 벡터
	double deltaRho;
	double deltaTheta; // 누산기 해상도 파라미터
	int minVote; // 선을 고려하기 전에 받아야 하는 최소 투표 개수
	double minLength; // 선에 대한 최소 길이
	double maxGap; // 선에 따른 최대 허용 간격

public:
	LineFinder::LineFinder() : deltaRho(1), deltaTheta(PI / 180), minVote(50), minLength(50), maxGap(10) {}
	// 기본 누적 해상도는 1각도 1화소 
	// 간격이 없고 최소 길이도 없음
	void setAccResolution(double dRho, double dTheta);
	void setMinVote(int minv);
	void setLineLengthAndGap(double length, double gap);
	std::vector<cv::Vec4i> findLines(cv::Mat& binary);
	void drawDetectedLines(cv::Mat &image, cv::Scalar color = cv::Scalar(112, 112, 0));
};
Mat hwnd2mat(HWND hWnd);
void cudaf();

#endif