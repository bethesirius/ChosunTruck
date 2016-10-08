#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;
#define M_PI 3.14159265358979323846

void createFilter(double gKernel[][2])
{
	// set standard deviation to 1.0
	double sigma = 1.0;
	double r, s = 2.0 * sigma * sigma;

	// sum is for normalization
	double sum = 0.0;

	// generate 2x2 kernel
	for (int x = -2; x <= 2; x++)
	{
		for (int y = -2; y <= 2; y++)
		{
			r = sqrt(x*x + y*y);
			gKernel[x + 2][y + 2] = (exp(-(r*r) / s)) / (M_PI * s);
			sum += gKernel[x + 2][y + 2];
		}
	}

	// normalize the Kernel
	for (int i = 0; i < 2; ++i)
	for (int j = 0; j < 2; ++j)
		gKernel[i][j] /= sum;

}

/*int main()
{
	double gKernel[2][2];
	createFilter(gKernel);
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
			cout << gKernel[i][j] << "\t";
		cout << endl;
	}
}*/