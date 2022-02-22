#pragma once
#include <opencv2/core/mat.hpp>
//#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>


using namespace cv;

Mat getMean(const std::vector<Mat3b>& images);
Mat create_bg();
void create_backgrounds();
