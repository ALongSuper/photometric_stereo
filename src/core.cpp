#include "core.h"

bool light_calib(cv::Mat img, std::vector<cv::Mat> calib_images,cv::Mat & lights_mat)
{


    //计算各光源的中心位置
    std::vector<cv::Point2f> light_pos;
    for (size_t i = 0; i < calib_images.size(); i++)
    {
        cv::Mat image = calib_images[i].clone();
        double vmin,vmax;
        cv::minMaxIdx(image, &vmin, &vmax);
        cv::Mat binary = image > (vmax * 0.95);
        float sum_row(0), sum_col(0), sum_weight(0);
        for (size_t row = 0; row < binary.rows; row++)
        {
            uchar* rowDatabin = binary.ptr<uchar>(row);
            uchar* rowDataraw = image.ptr<uchar>(row);
            for (size_t col = 0; col < binary.cols; col++)
            {
                if (rowDatabin[col] > 0)
                {
                    sum_row += rowDataraw[col] * row;
                    sum_col += rowDataraw[col] * col;
                    sum_weight += rowDataraw[col];
                }
            }
        }
        light_pos.push_back(cv::Point2f(sum_col / sum_weight, sum_row / sum_weight));
    }
    //计算标定球的位置信息
    cv::Mat binary;
    binary = img > 40;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary,contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& cntr1, const std::vector<cv::Point>& cntr2)
        { return cv::contourArea(cntr1) > cv::contourArea(cntr2); });

    cv::Point2f circle_center(cv::Point2f(0,0));
    float radius = 0;
    cv::minEnclosingCircle(contours[0], circle_center, radius);

    lights_mat = cv::Mat(calib_images.size(), 3, CV_32F, cv::Scalar::all(0));
    //依据标定球信息与光源位置与半径，计算光源在三维空间的方向向量
    for (size_t i = 0; i < calib_images.size(); i++)
    {
        //光源中心是否位于标定球内部
        double dis = cv::norm(light_pos[i] - circle_center);
        if (dis > radius)
           return false;
        lights_mat.at<float>(i, 1) = (light_pos[i].x - circle_center.x) / radius;
        lights_mat.at<float>(i, 0) = (light_pos[i].y - circle_center.y) / radius;
        lights_mat.at<float>(i, 2) = sqrt(1 - pow(lights_mat.at<float>(i, 0), 2) - pow(lights_mat.at<float>(i, 1), 2));
    }

    return true;
    
}

    void get_depth_map(torch::Tensor tensor_gx,torch::Tensor tensor_gy,cv::Mat & depth_map)
    {
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        int height = tensor_gx.size(2), width = tensor_gx.size(3),channel = tensor_gx.size(0);
        //fft
        torch::Tensor dft_tensor_gx = torch::fft::fft2(tensor_gx);
        torch::Tensor dft_tensor_gy = torch::fft::fft2(tensor_gy);
        //分离实部与虚部
        torch::Tensor real_part_gx = torch::real(dft_tensor_gx);
        torch::Tensor imag_part_gx = torch::imag(dft_tensor_gx);
        torch::Tensor real_part_gy = torch::real(dft_tensor_gy);
        torch::Tensor imag_part_gy = torch::imag(dft_tensor_gy);
        //创建索引张量
        torch::Tensor row_indices = torch::arange(height, torch::kInt).unsqueeze(1).unsqueeze(0).expand({1, 1, height, width});
        torch::Tensor col_indices = torch::arange(width, torch::kInt).unsqueeze(0).unsqueeze(0).expand({1, 1, height, width});
        row_indices = row_indices.to(device);
        col_indices = col_indices.to(device);
        float lambda = 1.0f;
        float mu = 1.0f;
        torch::Tensor u = torch::sin(2 * M_PI*row_indices/height);
        torch::Tensor v = torch::sin(2 * M_PI*col_indices/width);
        torch::Tensor uv = torch::pow(u, 2) + torch::pow(v, 2);
        torch::Tensor d = (1.0f + lambda) * uv + mu * torch::pow(uv, 2);
        torch::Tensor z_real = (u * imag_part_gx + v* imag_part_gy).div(d);
        torch::Tensor z_imag = (-1* u *real_part_gx - v* real_part_gy).div(d);
        z_real[0][0][0][0] = 0;
        z_imag[0][0][0][0] = 0;
        //逆变换
        torch::Tensor dft_Z = torch::complex(z_real,z_imag);
        torch::Tensor tensor_Z = torch::fft::ifft2(dft_Z);
        tensor_Z = torch::real(tensor_Z).to(torch::kFloat32);
        torch::Tensor tensor_normalized_depth = torch::clamp(tensor_Z, tensor_Z.min().item<float>(), tensor_Z.max().item<float>());
        tensor_Z = tensor_Z.to(torch::kCPU);
        cv::Mat depth_f_map(cv::Size(width, height), CV_32FC1, tensor_Z.data_ptr());
        depth_map = depth_f_map.clone();
    }
    
    void get_curvature_map(torch::Tensor tensor_gx,torch::Tensor tensor_gy,cv::Mat & curvature_map,cv::Mat & curl_map)
    {
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        int height = tensor_gx.size(2), width = tensor_gx.size(3),channel = tensor_gx.size(0);
        torch::Tensor sobel_y = torch::tensor({
            {{1, 0, -1},
            {2, 0, -2},
            {1, 0, -1}}
        }, torch::kFloat32).to(device);
        torch::Tensor sobel_x = torch::tensor({
            {{1, 2, 1},
            {0, 0, 0},
            {-1, -2, -1}}
        },torch::kFloat32).to(device);
        auto sobel_layer = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 3).bias(false).padding(1));
        torch::NoGradGuard no_grad1;
        sobel_layer->to(device);
        sobel_layer->weight.copy_(sobel_x);
        torch::Tensor tensor_gxx = sobel_layer->forward(tensor_gx);
        torch::Tensor tensor_gyx = sobel_layer->forward(tensor_gy);
        sobel_layer->weight.copy_(sobel_y);
        torch::Tensor tensor_gyy = sobel_layer->forward(tensor_gy);
        torch::Tensor tensor_gxy = sobel_layer->forward(tensor_gx);

        torch::Tensor tensor_A = torch::pow(tensor_gx, 2).add(1).mul(tensor_gyy);
        torch::Tensor tensor_B = tensor_gy.mul(tensor_gy).mul(tensor_gxy+tensor_gyx);
        torch::Tensor tensor_C = torch::pow(tensor_gy, 2).add(1).mul(tensor_gxx);
        torch::Tensor tensor_D = torch::pow(tensor_gx, 2).add(1) + tensor_gx.mul(tensor_gy);
        
        tensor_D = tensor_D.pow(1.5);
        torch::Tensor zeroMask = tensor_D == 0;
        tensor_D.masked_fill_(zeroMask, 0.00001);
        torch::Tensor tensor_curvature = (tensor_A - tensor_B + tensor_C).div(tensor_D);
        torch::Tensor limitdownMask = tensor_curvature < 0;
        tensor_curvature.masked_fill_(limitdownMask, 0);
        torch::Tensor limitupMask = tensor_curvature > 1;
        tensor_curvature.masked_fill_(limitupMask, 1);
        // float invgamma = 1/0.9;
        // tensor_curvature = tensor_curvature.pow(invgamma);

        torch::Tensor tensor_curf = tensor_gxx - tensor_gyy;
        tensor_curvature = tensor_curvature.to(torch::kCPU);
        tensor_curf = tensor_curf.to(torch::kCPU);
        cv::Mat curvature_f_map(cv::Size(width, height), CV_32FC1, tensor_curvature.data_ptr());
        cv::Mat curl_f_map(cv::Size(width, height), CV_32FC1, tensor_curf.data_ptr());

        //cv::GaussianBlur(curvature_f_map,curvature_f_map,cv::Size(5,5),1.075);
        curvature_map = curvature_f_map.clone();
        curl_map = curl_f_map.clone();
    }
