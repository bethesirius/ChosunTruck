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
//#include <Windows.h>
#include <iostream>
#include <string>
#include <chrono>
#include "ets2_self_driving.h"
#include "IPM.h"
#include "getScreen_linux.cpp"
#include "uinput.h"

#define PI 3.1415926

using namespace cv;
using namespace std;

int main() {
	int counter = 0;
	while(true) {
		auto begin = chrono::high_resolution_clock::now();
		cv::Mat img, outputImg;
		
		int height = 540;
                int width = 860;	
		
		int Width = 0;
		int Height = 0;
		int Bpp = 0;
		std::vector<std::uint8_t> Pixels;

		ImageFromDisplay(Pixels, Width, Height, Bpp);
		
   			
			img = Mat(Height, Width, Bpp > 24 ? CV_8UC4 : CV_8UC3, &Pixels[0]);
			cv::Rect myROI(820, 500, width, height);
			cv::Mat croppedRef(img, myROI);
			cv::Mat cropped;
			croppedRef.copyTo(cropped);
			
			vector<Point2f> origPoints;

                	origPoints.push_back(Point2f(0, (height-120)));
                	origPoints.push_back(Point2f(width, height-120));
               		origPoints.push_back(Point2f(width/2+125, height/2-70));
               		origPoints.push_back(Point2f(width/2-125, height/2-70));

			vector<Point2f> dstPoints;

                	dstPoints.push_back(Point2f(0, height));
               		dstPoints.push_back(Point2f(width, height));
        	        dstPoints.push_back(Point2f(width, 0));
	                dstPoints.push_back(Point2f(0, 0));
			
			IPM ipm(Size(width, height), Size(width, height), origPoints, dstPoints);

			ipm.applyHomography(cropped, outputImg);
			
			Mat gray;
        	        cv::resize(outputImg, outputImg, cv::Size(320,240));
	                cv::cvtColor(outputImg, gray, COLOR_RGB2GRAY);
	              	//cv::blur(gray, gray, cv::Size(10,10));
        	        //cv::Sobel(gray, gray, gray.depth(), 1, 0, 3, 0.5, 127);
	                //cv::threshold(gray, gray, 145, 255, CV_THRESH_BINARY);
		/*	
			int bottom_center = 160;
                
                	struct Detection{
                	        double left;
        	                double right;
	                        double center;
        	                int row;
	                };

			vector<Detection> data;
                	double center_to_left = -1; //init center_to_left
	                double center_to_right = -1;  //init center_to_right
			
			for(int i=240; i>30; i++){
                        	for (int j=0;j<150;j++) {
                                	if (gray.at<unsigned char>(i, bottom_center+j) == 255 && center_to_right == -1) {
                                        	center_to_right = j;
                                	}
                                	if (gray.at<unsigned char>(i, bottom_center-j) == 255 && center_to_left == -1) {
                                        	center_to_left = j;
                                	}
                        	}
                        	if(center_to_left!=-1 && center_to_right!=-1){
                                	Detection d = {center_to_left, center_to_right,  bottom_center, i};
					data.push_back(d);
                        	}else{
					Detection d = {center_to_left, center_to_right, bottom_center, i};
					data.push_back(d);
				}
                	}
			for(vector<int>::size_type i = 0; i < data.size(); i++) {
                        	Detection d = {center_to_left, center_to_right, bottom_center, i};
                        	if (data.size() != 0) {
                                	data.push_back(d);
                        	}

                        	if (center_to_right != -1 && center_to_left != -1) {
                                	if (data.size() == 0) {
                                        	Detection d = {center_to_left, center_to_right, bottom_center, i};
                                        	data.push_back(d);
                                	}
                        	        bottom_center += (center_to_right - center_to_left) / 2;
                	        } else {

        	                }

	                }
			double sum_right = 0;
        	        double sum_left = 0;
	                double sum_center = 0;
			for (vector<int>::size_type i = 0; i < data.size(); i++) {
                        	Detection& d = data[i];
                        	int tmp = d.center;
                        	if (d.right == -1 || (i>1 && abs(d.right - data[i-1].right) > 15)) {
                                	if (sum_right == 0) {
                                        	d.right = data[i-1].right;
                                	} else {
                                        	d.right = data[i-1].right + (abs(sum_right)/sum_right)*(abs(sum_right)/i);
                                	}
                        	}
                        	if (d.left == -1 || (i>1 && abs(d.left - data[i-1].left) > 15)) {
                                	if (sum_left == 0) { 
                                        	d.left = data[i-1].left;
                                	} else {
                                        	d.left = data[i-1].left + (abs(sum_left)/sum_left)*(abs(sum_left)/i);
                                	}
                        	}
                        	if (i>1) {
                                	double delta_right = d.right - data[i-1].right;
                                	double delta_left = d.left - data[i-1].left;
                                	sum_right += delta_right;
                                	//cout << d.right << " " << data[i-1].right << " " << sum_right << " ";
                                	sum_left += delta_left;
                        	}
                        	cv::circle(gray, Point(tmp + (int)d.right, d.row), 1, Scalar(200, 200, 200) , 1);
                        	cv::circle(gray, Point((int)d.center - (int)d.left, d.row), 1, Scalar(200, 200, 200) , 1);
                        	cv::circle(gray, Point(d.center, d.row), 1, Scalar(50, 50, 50) , 3);
                        	sum_center += d.center;
                	}
			///////////////////////////////////////
                	unsigned char row_center = gray.at<unsigned char>(10, 160);

                	unsigned char row_left=0;
                	unsigned char row_right=0;

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
			int average = (left == -150 || right == 150) ? 0: left+right;
                	int direction = sum_center/data.size() - 160;
                
                	if (direction < -15){
                        	cout << "go left ";
                        	int move_mouse_pixels = -3;
                        	if (direction < -30) {
                                	move_mouse_pixels += -1;
                        	}
                        	if(direction < -50) {
                                	move_mouse_pixels += -1;
                        	}
                        	if(direction < -70) {
                                	move_mouse_pixels += -1;
                        	}
                        	counter+=goDirection(move_mouse_pixels);
			}else if (direction > -15 && direction < 15){
                        	cout << "go straight ";
                        	goDirection(0 - counter);
                        	counter = 0;
                        }else{
                        	cout << "go right ";
                        	int move_mouse_pixels = 3;
                        	if(direction > 30) {
                                	move_mouse_pixels += 1;
                        	}
                        	if (direction > 50) {
                                	move_mouse_pixels += 1;
                        	}
                        	if(direction > 70) {
                                	move_mouse_pixels += 1;
                        	}
                        	counter+=goDirection(move_mouse_pixels);	
			}
                	cout << "left: " << left << ", right: " << right << ", average: "<< average<<endl;
*/	

			imshow("TEST", gray);
			waitKey(1);
		 
	}
	return 0;
}
