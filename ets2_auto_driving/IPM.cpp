#include "IPM.h"

using namespace cv;
using namespace std;

// Public
IPM::IPM( const cv::Size& _origSize, const cv::Size& _dstSize, const std::vector<cv::Point2f>& _origPoints, const std::vector<cv::Point2f>& _dstPoints )
	: m_origSize(_origSize), m_dstSize(_dstSize), m_origPoints(_origPoints), m_dstPoints(_dstPoints)
{
	assert( m_origPoints.size() == 4 && m_dstPoints.size() == 4 && "Orig. points and Dst. points must vectors of 4 points" );
	m_H = getPerspectiveTransform( m_origPoints, m_dstPoints );
	m_H_inv = m_H.inv();
		
	createMaps();	
}
void IPM::drawPoints( const std::vector<cv::Point2f>& _points, cv::Mat& _img ) const
{
	assert(_points.size() == 4);

	line(_img, Point(static_cast<int>(_points[0].x), static_cast<int>(_points[0].y)), Point(static_cast<int>(_points[3].x), static_cast<int>(_points[3].y)), CV_RGB( 205,205,0), 2);
	line(_img, Point(static_cast<int>(_points[2].x), static_cast<int>(_points[2].y)), Point(static_cast<int>(_points[3].x), static_cast<int>(_points[3].y)), CV_RGB( 205,205,0), 2);
	line(_img, Point(static_cast<int>(_points[0].x), static_cast<int>(_points[0].y)), Point(static_cast<int>(_points[1].x), static_cast<int>(_points[1].y)), CV_RGB( 205,205,0), 2);
	line(_img, Point(static_cast<int>(_points[2].x), static_cast<int>(_points[2].y)), Point(static_cast<int>(_points[1].x), static_cast<int>(_points[1].y)), CV_RGB( 205,205,0), 2);
	
	for(size_t i=0; i<_points.size(); i++)
	{
		circle(_img, Point(static_cast<int>(_points[i].x), static_cast<int>(_points[i].y)), 2, CV_RGB(238,238,0), -1);
		circle(_img, Point(static_cast<int>(_points[i].x), static_cast<int>(_points[i].y)), 5, CV_RGB(255,255,255), 2);
	}
}
void IPM::getPoints(vector<Point2f>& _origPts, vector<Point2f>& _ipmPts)
{
	_origPts = m_origPoints;
	_ipmPts = m_dstPoints;
}
void IPM::applyHomography(const Mat& _inputImg, Mat& _dstImg, int _borderMode)
{
	// Generate IPM image from src
	remap(_inputImg, _dstImg, m_mapX, m_mapY, INTER_LINEAR, _borderMode);//, BORDER_CONSTANT, Scalar(0,0,0,0));
}
void IPM::applyHomographyInv(const Mat& _inputImg, Mat& _dstImg, int _borderMode)
{
	// Generate IPM image from src
	remap(_inputImg, _dstImg, m_mapX, m_mapY, INTER_LINEAR, _borderMode);//, BORDER_CONSTANT, Scalar(0,0,0,0));
}
Point2d IPM::applyHomography( const Point2d& _point )
{
	return applyHomography( _point, m_H );
}
Point2d IPM::applyHomographyInv( const Point2d& _point )
{
	return applyHomography( _point, m_H_inv );
}
Point2d IPM::applyHomography( const Point2d& _point, const Mat& _H )
{
	Point2d ret = Point2d( -1, -1 );
	
	const double u = _H.at<double>(0,0) * _point.x + _H.at<double>(0,1) * _point.y + _H.at<double>(0,2);
	const double v = _H.at<double>(1,0) * _point.x + _H.at<double>(1,1) * _point.y + _H.at<double>(1,2);
	const double s = _H.at<double>(2,0) * _point.x + _H.at<double>(2,1) * _point.y + _H.at<double>(2,2);
	if ( s != 0 )
	{
		ret.x = ( u / s );
		ret.y = ( v / s );
	}
	return ret;	
}
Point3d IPM::applyHomography( const Point3d& _point )
{
	return applyHomography( _point, m_H );
}
Point3d IPM::applyHomographyInv( const Point3d& _point )
{
	return applyHomography( _point, m_H_inv );
}
Point3d IPM::applyHomography( const Point3d& _point, const cv::Mat& _H )
{
	Point3d ret = Point3d( -1, -1, 1 );
	
	const double u = _H.at<double>(0,0) * _point.x + _H.at<double>(0,1) * _point.y + _H.at<double>(0,2) * _point.z;
	const double v = _H.at<double>(1,0) * _point.x + _H.at<double>(1,1) * _point.y + _H.at<double>(1,2) * _point.z;
	const double s = _H.at<double>(2,0) * _point.x + _H.at<double>(2,1) * _point.y + _H.at<double>(2,2) * _point.z;
	if ( s != 0 )
	{
		ret.x = ( u / s );
		ret.y = ( v / s );
	}
	else
		ret.z = 0;
	return ret;	
}

// Private
void IPM::createMaps()
{  
	// Create remap images
	m_mapX.create(m_dstSize, CV_32F);
	m_mapY.create(m_dstSize, CV_32F);
	for( int j = 0; j < m_dstSize.height; ++j )
	{
		float* ptRowX = m_mapX.ptr<float>(j);
		float* ptRowY = m_mapY.ptr<float>(j);
		for( int i = 0; i < m_dstSize.width; ++i )
		{
			Point2f pt = applyHomography( Point2f( static_cast<float>(i), static_cast<float>(j) ), m_H_inv );
			ptRowX[i] = pt.x;
			ptRowY[i] = pt.y;
		}
	}

	m_invMapX.create(m_origSize, CV_32F);
	m_invMapY.create(m_origSize, CV_32F);

	for( int j = 0; j < m_origSize.height; ++j )
	{
		float* ptRowX = m_invMapX.ptr<float>(j);
		float* ptRowY = m_invMapY.ptr<float>(j);
		for( int i = 0; i < m_origSize.width; ++i )
		{
			Point2f pt = applyHomography( Point2f( static_cast<float>(i), static_cast<float>(j) ), m_H );
			ptRowX[i] = pt.x;
			ptRowY[i] = pt.y;			
		}
	}	
}