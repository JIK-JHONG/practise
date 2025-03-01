#include "image_process.h"
#include <iostream>
#include <cmath>

Mat ImageBlur(const Mat &input, Point start, Size area, Size blur_size)
{
    Mat output = input.clone();
    if (area.width == 0 || area.height == 0)
    {
        blur(output, output, blur_size);
    }
    else
    {
        Rect roi(start, area);
        Mat blurred_roi;
        blur(output(roi), blurred_roi, blur_size);
        blurred_roi.copyTo(output(roi));
    }
    return output;
}

int image_binary_color_check(const Mat &input, int threshold)
{
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    int white = 0;
    int black = 0;
    Mat tmp;
    if (channel == 3)
    {
        cvtColor(input, tmp, COLOR_BGR2GRAY);
    }
    else if (channel == 4)
    {
        cvtColor(input, tmp, COLOR_BGRA2GRAY);
    }
    else
    {
        tmp = input.clone();
    }
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            if (tmp.at<Vec3b>(j, i)[0] > threshold)
            {
                white += 1;
            }
            else
            {
                black += 1;
            }
        }
    }
    printf("Auto_Check Ratio >> W = %.2f %% , B = %.2f %%\n", float(white)/(white+black)*100., float(black)/(white+black)*100.);
    if (white > black)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


Mat ImageGray(const Mat &input, int option)
{
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    Mat gray(height, width, CV_8UC3, Scalar(255, 255, 255));
    int max_color = 1;
    int min_color = 255;
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int tmp_balance = round(input.at<Vec3b>(j, i)[0] * 0.299 + input.at<Vec3b>(j, i)[1] * 0.587 + input.at<Vec3b>(j, i)[2] * 0.114);
            if (tmp_balance < min_color)
            {
                min_color = tmp_balance;
            }
            if (tmp_balance > max_color)
            {
                max_color = tmp_balance;
            }
        }
    }
    double factor_max = 255. / max_color;
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int tmp_balance = round(input.at<Vec3b>(j, i)[0] * 0.299 + input.at<Vec3b>(j, i)[1] * 0.587 + input.at<Vec3b>(j, i)[2] * 0.114);
            for (int k = 0; k < channel; k++)
            {
                int tmp = round(tmp_balance * factor_max);
                if (tmp > 255)
                {
                    tmp = 255;
                }
                gray.at<Vec3b>(j, i)[k] = tmp;
            }
        }
    }
    return gray;
}

Mat ImageBinary(const Mat &input, const std::string &mode_type, int low_threshold_force)
{
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    Mat output(height, width, CV_8UC3, Scalar(255, 255, 255));

    int low_threshold = low_threshold_force;
    int high_threshold = 255;

    if (mode_type == "normal")
    {
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                int tmp_balance = (input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2]) / 3;
                for (int k = 0; k < channel; k++)
                {
                    if (tmp_balance > low_threshold)
                    {
                        tmp_balance = 255;
                    }
                    else
                    {
                        tmp_balance = 0;
                    }
                    output.at<Vec3b>(j, i)[k] = tmp_balance;
                }
            }
        }
    }
    else if (mode_type == "inverted")
    {
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                int tmp_balance = (input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2]) / 3;
                for (int k = 0; k < channel; k++)
                {
                    int tmp = tmp_balance;
                    if (tmp_balance > low_threshold)
                    {
                        tmp = 0;
                    }
                    else
                    {
                        tmp = 255;
                    }
                    output.at<Vec3b>(j, i)[k] = tmp;
                }
            }
        }
    }
    else if (mode_type == "enhance")
    {
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                int tmp_balance = (input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2]) / 3;
                for (int k = 0; k < channel; k++)
                {
                    int tmp = tmp_balance;
                    if (tmp_balance >= 0 && tmp_balance < 85)
                    {
                        tmp = 0;
                    }
                    else if (tmp_balance >= 85 && tmp_balance < 170)
                    {
                        tmp = 170;
                    }
                    else
                    {
                        tmp = 255;
                    }
                    output.at<Vec3b>(j, i)[k] = tmp;
                }
            }
        }
    }

    return output;
}


Mat ImageComicMesh_Mix(const Mat &input, int block_size, int gap_size, const std::string &option, std::string color_option)
{
    Mat input2;
    input2 = input.clone();
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    printf("Image = %d * %d (%d)\n", width, height, channel);
    int threshold = 5;
    Mat output(height, width, CV_8UC3, Scalar(255, 255, 255));
    int basic_color = 0;
    Vec3b color_mark(0, 0, 255);
    int fac = -1;
    Point initial_spot(round((block_size + gap_size)), round((block_size + gap_size)));
    printf("Scan_range = %d * %d (block_size = %d , gap_size = %d), mesh_size = %d \n", height / ((block_size + gap_size) * 2), width / ((block_size + gap_size) * 2), block_size, gap_size, (block_size + gap_size) * 2);

    int block_mesh_size = (block_size + gap_size) * 2 * (block_size + gap_size) * 2;
    int total_mesh_unit = 255 / (height / ((block_size + gap_size) * 2) * width / ((block_size + gap_size) * 2) / block_mesh_size);
    if (total_mesh_unit == 0)
    {
        total_mesh_unit = 1;
    }
    printf("block_mesh_size = %d\n", block_mesh_size);
    for (int j = 0; j < height / ((block_size + gap_size) * 2); j++)
    {
        for (int i = 0; i < width / ((block_size + gap_size) * 2); i++)
        {
            int sum_of_area = 0;
            int sum_color_b = 0;
            int sum_color_g = 0;
            int sum_color_r = 0;

            for (int kj = 0; kj < (block_size + gap_size) * 2; kj++)
            {
                for (int ki = 0; ki < (block_size + gap_size) * 2; ki++)
                {
                    int target_x = initial_spot.x + i * (block_size + gap_size) * 2 + ki - (block_size + gap_size);
                    int target_y = initial_spot.y + j * (block_size + gap_size) * 2 + kj - (block_size + gap_size);
                    if (target_x < width && target_y < height)
                    {
                        sum_of_area += (input2.at<Vec3b>(target_y, target_x)[0] + input2.at<Vec3b>(target_y, target_x)[1] + input2.at<Vec3b>(target_y, target_x)[2]) / 3;
                        sum_color_b += (input2.at<Vec3b>(target_y, target_x)[0]);
                        sum_color_g += (input2.at<Vec3b>(target_y, target_x)[1]);
                        sum_color_r += (input2.at<Vec3b>(target_y, target_x)[2]);
                    }
                }
            }
            sum_of_area /= block_mesh_size;
            sum_color_b /= block_mesh_size;
            sum_color_g /= block_mesh_size;
            sum_color_r /= block_mesh_size;
            if (sum_of_area <= 200)
            {
                Point center(initial_spot.x + i * (block_size + gap_size) * 2,
                             initial_spot.y + j * (block_size + gap_size) * 2);
                if (option == "normal")
                {
                    if (color_option == "std")
                    {
                        circle(output, center, block_size, Scalar(sum_color_b, sum_color_g, sum_color_r), fac);
                    }
                    else
                    {
                        circle(output, center, block_size, Scalar(255, 0, 0 + (i + j) * (total_mesh_unit)), fac);
                    }
                }
                else if (option == "square")
                {
                    if (color_option == "std")
                    {
                        rectangle(output,
                                  Point(center.x - block_size, center.y - block_size),
                                  Point(center.x + block_size, center.y + block_size),
                                  Scalar(sum_color_b, sum_color_g, sum_color_r),
                                  FILLED); // `FILLED` 代表填滿
                    }
                    else
                    {
                        rectangle(output,
                                  Point(center.x - block_size, center.y - block_size),
                                  Point(center.x + block_size, center.y + block_size),
                                  Scalar(255, 0, 0 + (i + j) * total_mesh_unit),
                                  FILLED); // `FILLED` 代表填滿
                    }
                }
            }
        }
    }

    return output;
}
