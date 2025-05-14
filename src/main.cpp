/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2024-07-16 15:27:22
 * @LastEditors: SJL
 * @LastEditTime: 2025-05-14 14:32:05
 */
#include <torch/torch.h>
#include <c10/util/Half.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "core.h"

// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/io/ply_io.h>
#define PI 3.1415926

int main() {
   try{
        //std::string obj = "calib_ball";
        //std::string obj = "flooring";
        std::string obj = "bearing";
        
        // 光源方向标定    
        cv::Mat img_ball;
        std::vector<cv::Mat> calib_images;
        cv::String img_calib_ball_src = "../../img/calib_ball/*bmp";
        std::vector<cv::String> img_files;
        cv::glob(img_calib_ball_src, img_files);	
        for (size_t idx = 0; idx < img_files.size(); ++idx)
        {		
            std::string img_file = img_files[idx];
            cv::Mat ori_image = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
            calib_images.push_back(ori_image);
        }
        img_ball = cv::imread("../../img/calib.bmp", cv::IMREAD_GRAYSCALE);
        cv::Mat lights_mat;
        
        if(light_calib(img_ball,calib_images,lights_mat))
            std::cout << lights_mat << std::endl;
        else
        {
            std::cout << "光源标定数据异常" << std::endl;
            return false;
        }

        //flooring采用的是halcon中的数据
        if (obj == "flooring")
        {
            std::vector<float> Slants {39.4, 40.5, 39.5, 38.4};
            std::vector<float> Tilts {-6.0, 83.7, 172.9, -98.2};
            for (size_t i = 0; i < 4; i++)
            {
                float theta = Tilts[i] * M_PI / 180.0;
                float phi = Slants[i] * M_PI / 180.0;
                // 计算方向向量的分量
                float x = sin(phi) * cos(theta);
                float y = sin(phi) * sin(theta);
                float z = cos(phi);

                lights_mat.at<cv::Vec3f>(i)= cv::Vec3f(x, y, z);
            } 
        }

        //光度立体重建
        //需对每个像素求解 N = L逆 * I 的方程组，可以将L(光源方向向量)作为权重，借助libtorch对输入图像I进行卷积，达到并行加速的作用
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        //将所有方向的图像merge为一张多通道的图像，并转为float类型，便于转换为tensor张量
        std::vector<cv::Mat> input_images;
        cv::String input_images_src = "../../img/"+obj+"/*bmp";
        cv::glob(input_images_src, img_files);	
        for (size_t idx = 0; idx < img_files.size(); ++idx)
        {		
            std::string img_file = img_files[idx];
            cv::Mat ori_image = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
            input_images.push_back(ori_image);
        }
        cv::Mat input_img;
        cv::merge(input_images, input_img);

        cv::Mat input_img_roi = input_img.clone();
        cv::Mat input_f_img;
        input_img_roi.convertTo(input_f_img, CV_32FC(input_img.channels()));
        torch::Tensor tensor_input = torch::from_blob(input_f_img.data, {1, input_f_img.rows,input_f_img.cols, input_f_img.channels()}, torch::kFloat32);
        tensor_input = tensor_input.permute({0, 3, 1, 2});
        tensor_input = tensor_input.to(device);

        cv::Mat lights_inv;
        cv::invert(lights_mat, lights_inv, cv::DECOMP_SVD);
        torch::Tensor tensor_weights = torch::from_blob(lights_inv.data, {1, lights_inv.rows,lights_inv.cols, 1}, torch::kFloat32);
        tensor_weights = tensor_weights.permute({1, 2, 0, 3});
        tensor_weights = tensor_weights.to(device);

        // 创建卷积层并搬运到GPU上
        auto conv_layer = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_f_img.channels(), 3, 1).bias(false));
        conv_layer->to(device);
        torch::NoGradGuard no_grad;
        //设置权重，并进行卷积，输出为3通道，即光度立体法计算得到的法向量
        conv_layer->weight.copy_(tensor_weights);
        torch::Tensor result = conv_layer->forward(tensor_input);
        int height = result.size(2), width = result.size(3),channel = result.size(0);
        //*******************1. 反照率图************************
        //法向量的模(反照率)
        torch::Tensor norm = result.norm(2, 1, true);
        //法向量的单位向量
        torch::Tensor normalized_tensor = result.div(norm);
        torch::Tensor nanMask = torch::isnan(normalized_tensor);
        normalized_tensor.masked_fill_(nanMask, 1);

        torch::Tensor tensor_albedo = torch::clamp(norm, norm.min().item<float>(), norm.max().item<float>());
        tensor_albedo = (tensor_albedo - tensor_albedo.min()) / (tensor_albedo.max() - tensor_albedo.min());
        tensor_albedo = tensor_albedo.mul(255.0);
        tensor_albedo = tensor_albedo.to(torch::kUInt8);
        tensor_albedo = tensor_albedo.to(torch::kCPU);
        cv::Mat albedo_f_map(cv::Size(width, height), CV_8UC1, tensor_albedo.data_ptr());

        //*********************2 . 深度图***************************
        // //根据法向量normalized_tensor进行高度重建(相对高度，对梯度在xy方向进行积分，频域进行,即求解泊松方程)
        std::vector<torch::Tensor> split_tensors = normalized_tensor.chunk(3,1);
        torch::Tensor tensor_gx = split_tensors[0].div(split_tensors[2]);
        torch::Tensor tensor_gy = split_tensors[1].div(split_tensors[2]);
        cv::Mat depth_f_map;
        get_depth_map(tensor_gx,tensor_gy,depth_f_map);

        
        // //转为点云  如需请将PCL相关的注释恢复
        // normalized_tensor = normalized_tensor.permute({0,2,3,1}).contiguous().to(torch::kCPU);
        // cv::Mat normal_f_map(cv::Size(width, height), CV_32FC3, normalized_tensor.data_ptr());
        // pcl::PointCloud<pcl::PointNormal>::Ptr clouddepth(new pcl::PointCloud<pcl::PointNormal>);
        // pcl::PointNormal point;
        // for(int i = 0 ;i < depth_f_map.rows; i++)
        // {
        //     for(int j = 0; j <depth_f_map.cols;j++)
        //     {
        //         point.x = i;
        //         point.y = j;
        //         point.z = depth_f_map.at<float>(i,j);
        //         point.normal_x = normal_f_map.at<cv::Vec3f>(i,j)[0]*3;
        //         point.normal_y = normal_f_map.at<cv::Vec3f>(i,j)[1]*3;
        //         point.normal_z = normal_f_map.at<cv::Vec3f>(i,j)[2]*3;
        //         clouddepth->push_back(point);
        //     }
        // }
        // std::cout << "save begin" << std::endl;
        // pcl::io::savePCDFileASCII("../../img/"+obj+"/pointcloud.pcd", *clouddepth);
        // std::cout << "save end" << std::endl;

        //*********************3 .法向角度图***************************
        torch::Tensor dotProduct = split_tensors[2];//normalized_tensor与相机方向向量[0,0,1]点乘，结果即normalized_tensor的第三维
        torch::Tensor cosTheta = dotProduct;//均为单位向量模为1
        torch::Tensor angleRadians = torch::acos(cosTheta);
        torch::Tensor angleDegrees = angleRadians * (180.0 / M_PI);
        
        torch::Tensor tensor_normalized_angle = torch::clamp(angleDegrees, angleDegrees.min().item<float>(), angleDegrees.max().item<float>());
        angleDegrees = angleDegrees.to(torch::kCPU);
        cv::Mat angle_f_map(cv::Size(width, height), CV_32FC1, angleDegrees.data_ptr());

        //*********************4 .法向量渲染图***************************    
        torch::Tensor tensor_r = ((split_tensors[0]+1)*0.5*255).to(torch::kUInt8);
        torch::Tensor tensor_g = ((split_tensors[1]+1)*0.5*255).to(torch::kUInt8);
        torch::Tensor tensor_b = ((split_tensors[2]+1)*0.5*255).to(torch::kUInt8);
        torch::Tensor normals_tensor2 = torch::cat({ tensor_b, tensor_g, tensor_r },1);
        normals_tensor2 = normals_tensor2.permute({ 0, 2, 3, 1 }).contiguous();
        normals_tensor2 = normals_tensor2.to(torch::kCPU);
        cv::Mat normals_map = cv::Mat(cv::Size(width, height), CV_8UC3,normals_tensor2.data_ptr());

        // *********5 .旋度图与曲率图**************
        cv::Mat curvature_f_map,curl_f_map;
        get_curvature_map(tensor_gx,tensor_gy,curvature_f_map,curl_f_map);

        tensor_gx = tensor_gx.to(torch::kCPU);
        tensor_gy = tensor_gy.to(torch::kCPU);
        cv::Mat gx_f_map(cv::Size(width, height), CV_32FC1, tensor_gx.data_ptr());
        cv::Mat gy_f_map(cv::Size(width, height), CV_32FC1, tensor_gy.data_ptr());

        cv::imwrite("../../img/"+obj+"/depth_f_map.tif",depth_f_map);
        cv::imwrite("../../img/"+obj+"/albedo_f_map.tif",albedo_f_map);
        cv::imwrite("../../img/"+obj+"/angle_f_map.tif",angle_f_map);
        //cv::imwrite("../../img/"+obj+"/gx_f_map.tif",gx_f_map);
        //cv::imwrite("../../img/"+obj+"/gy_f_map.tif",gy_f_map);
        cv::imwrite("../../img/"+obj+"/curvature.tif",curvature_f_map);
        cv::imwrite("../../img/"+obj+"/curf.tif",curl_f_map);
        cv::imwrite("../../img/"+obj+"/normals.png",normals_map);

    }catch(const c10::Error  &e)
    {
        std::cerr << "error: " << e.what() << std::endl;
    } catch (const std::exception& e) 
    {
        std::cerr << "error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "error: " << std::endl;
    }
    return 0;
}