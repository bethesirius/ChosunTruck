#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/photo.hpp>
#include <string>
#include <chrono>
#include <iostream>

using namespace std;
using namespace cv;

int main(){

	string imageName("input.bmp");


	auto begin = chrono::high_resolution_clock::now();
	//CPU version
	Mat image = imread(imageName.c_str(), IMREAD_GRAYSCALE);
	Mat reslutCPU;
	fastNlMeansDenoising(image, reslutCPU, 2.5, 7, 31);
	imwrite("cpu.bmp", reslutCPU);

	auto end = chrono::high_resolution_clock::now();
	auto dur = end - begin;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	cout << ms  << endl;




	begin = chrono::high_resolution_clock::now();
	//CUDA version       
	cuda::GpuMat imageGPU;
	cuda::GpuMat reslutGPU;
	Mat buff;
	imageGPU.upload(image);
	cuda::fastNlMeansDenoising(imageGPU, reslutGPU, 2.5, 7, 31);
	reslutGPU.download(buff);
	imwrite("gpu.bmp", buff);

	end = chrono::high_resolution_clock::now();
	dur = end - begin;
	ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	cout << ms << endl;

	return 0;
}