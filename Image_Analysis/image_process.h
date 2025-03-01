#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#include <opencv2/opencv.hpp>
using namespace cv;

Mat ImageBlur(const Mat &input, Point start = Point(0, 0), Size area = Size(), Size blur_size = Size(50, 50));
Mat ImageGray(const Mat &input, int option = 1);
Mat ImageBinary(const Mat &input, const std::string &mode_type = "normal", int low_threshold_force = 127);
Mat ImageComicMesh_Mix(const Mat &input, int block_size = 20, int gap_size = 2, const std::string &option = "normal", std::string color_option = "std");
Mat Pixelized(const Mat &input, int pixel_size = 10);
int image_binary_color_check(const Mat &input, int threshold = 80);
void image_preProcess(Mat &input);
void Compare_View(const Mat &input, const Mat &input2, const Mat &input3);

#endif // IMAGE_PROCESS_H
