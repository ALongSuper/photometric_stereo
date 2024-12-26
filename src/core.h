/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-07-16 15:53:27
 * @LastEditors: SJL
 * @LastEditTime: 2024-08-06 11:25:14
 */
#include <torch/torch.h>
#include <c10/util/Half.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



//光源标定
bool light_calib(cv::Mat img, std::vector<cv::Mat> calib_images,cv::Mat & lights_mat);

//相对深度图重建
void get_depth_map(torch::Tensor tensor_gx,torch::Tensor tensor_gy,cv::Mat & depth_map);

//旋度图与平均曲率图重建
void get_curvature_map(torch::Tensor tensor_gx,torch::Tensor tensor_gy,cv::Mat & curvature_map,cv::Mat & curl_map);