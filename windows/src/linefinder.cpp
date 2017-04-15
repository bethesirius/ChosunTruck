#include "ets2_self_driving.h"
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
#include <iostream>
#include <string>

#define PI 3.1415926

cv::Point prev_point;

using namespace cv;
using namespace std;

// 해당 세터 메소드들
// 누적기에 해상도 설정
void LineFinder::setAccResolution(double dRho, double dTheta) {
	deltaRho = dRho;
	deltaTheta = dTheta;
}

// 투표 최소 개수 설정
void LineFinder::setMinVote(int minv) {
	minVote = minv;
}

// 선 길이와 간격 설정
void LineFinder::setLineLengthAndGap(double length, double gap) {
	minLength = length;
	maxGap = gap;
}

// 허프 선 세그먼트 감지를 수행하는 메소드
// 확률적 허프 변환 적용
std::vector<cv::Vec4i> LineFinder::findLines(cv::Mat& binary) {
	lines.clear();
	cv::HoughLinesP(binary, lines, deltaRho, deltaTheta, minVote, minLength, maxGap);
	return lines;
} // cv::Vec4i 벡터를 반환하고, 감지된 각 세그먼트의 시작과 마지막 점 좌표를 포함.

// 위 메소드에서 감지한 선을 다음 메소드를 사용해서 그림
// 영상에서 감지된 선을 그리기
void LineFinder::drawDetectedLines(cv::Mat &image, cv::Scalar color) {

	// 선 그리기
	cv::Point endPoint;

	for (auto it2&& : lines) {
		cv::Point startPoint(it2[0], it2[1]);
		endPoint = cv::Point(it2[2], it2[3]);
		cv::line(image, startPoint, endPoint, color, 3);
	}
}

