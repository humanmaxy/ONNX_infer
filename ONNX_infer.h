#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<opencv2/dnn.hpp>
#include <core/utils/logger.hpp>

class ONNX_process{
public:
	ONNX_process(int width, int height, std::string model_path) {
		 {
			image_width = width;
			image_height = height;
			// 设置 OpenCV 日志级别为 WARNING
			cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);

			// 创建一个dnn模型
			//std::string model_path = ".\\test.onnx";
			mynet = cv::dnn::readNetFromONNX(model_path);
			//std::vector<cv::String> input_names = mynet.getUnconnectedOutLayersNames();
			//std::vector<cv::String> output_names = mynet.getUnconnectedOutLayersNames();
			mynet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
			mynet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

		}
	};
	~ONNX_process() = default;
	int image_width = 960; 
	int image_height = 512;
	cv::dnn::Net mynet;
	cv::Mat u16Normalization(cv::Mat& inputMat);

	cv::Mat infer(cv::Mat img0);
	
};


cv::Mat ONNX_process::u16Normalization(cv::Mat& inputMat) {
	// Ensure the input matrix has the correct type (16-bit unsigned integer)
	if (inputMat.type() != CV_16UC1) {
		std::cerr << "Input matrix should be 16-bit unsigned integer type (CV_16UC1)." << std::endl;
		return inputMat;
	}
	//std::cout << inputMat.type() << inputMat.cols << inputMat.rows << endl;

	// 将输入图像转换为浮点数类型
	cv::Mat floatMat;
	inputMat.convertTo(floatMat, CV_32FC1);

	// 归一化
	cv::Mat x1 = floatMat / 65535.0 * 255.0 / 255 * 2 - 1;


	cv::Mat x2, x3;
	inputMat.convertTo(x2, CV_32FC1);
	inputMat.convertTo(x3, CV_32FC1);


	// Perform the u8 normalization for each pixel
	for (int row = 0; row < inputMat.rows; ++row) {
		for (int col = 0; col < inputMat.cols; ++col) {
			uint pixel = inputMat.at<ushort>(row, col);

			// 获取整数部分的前八位和后八位
			uint high_eight_bits = pixel / 255;
			uint low_eight_bits = pixel % 255;

			float high = (static_cast<float>(high_eight_bits) / 255.0f) * 2.0f - 1.0f;
			float low = (static_cast<float>(low_eight_bits) / 255.0f) * 2.0f - 1.0f;
			x2.at<float>(row, col) = high;
			x3.at<float>(row, col) = low;
		}
	}
	cv::Mat result;
	cv::merge(std::vector<cv::Mat>{ x2, x3, x1}, result);

	return result;
}
cv::Mat ONNX_process::infer(cv::Mat img0) {
	//cv::Mat img0;

	//int img1=rand()%256;
	cv::Mat float_img = img0.clone();
	//float_img.convertTo(img0, CV_32FC1); // 将图像转换为浮点型
	//cv::Mat img1 = img0 / 65535 * 255.0 / 255 * 2 - 1;
	//cv::Mat img2, img3;
	//float_img.convertTo(img2, CV_32FC1, 1.0 / 255.0);
	//float_img.convertTo(img3, CV_32FC1, 1.0 / 65535.0);
	cv::Mat res = u16Normalization(float_img);
	/*cv::merge(std::vector<cv::Mat>{img2, img3, img1}, res);*/
	cv::Mat blob = cv::dnn::blobFromImage(res);
	//cv::dnn::Net mynet;
	//cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);
	//std::string model_path = ".\\test.onnx";
	//mynet = cv::dnn::readNetFromONNX(model_path);
	std::vector<cv::String> input_names = mynet.getUnconnectedOutLayersNames();
	std::vector<cv::String> output_names = mynet.getUnconnectedOutLayersNames();
	//mynet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
	//mynet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	//srand(2023816);
	mynet.setInput(blob);
	std::vector<cv::Mat> outputs;

	mynet.forward(outputs, output_names);
	cv::Mat result = outputs[0];
	cv::Mat segament_result(this->image_height, this->image_width, CV_8UC1);
	//std::cout << segament_result.type() << endl;

	cv::Mat result2(this->image_height, this->image_width, CV_32FC3);
	int height = this->image_height;
	int width = this->image_width;

	for (int y = 0; y < image_height; y++) {
		for (int x = 0; x < image_width; x++) {
			cv::Vec3f& pixel = result2.at<cv::Vec3f>(y, x);
			int max_class_idx = 0;
			float max_prob = result.ptr<float>(0, 0, y)[x];
			for (int c = 0; c < 3; c++) {
				float prob = result.ptr<float>(0, c, y)[x];
				if (prob > max_prob) {
					max_prob = prob;
					max_class_idx = c;
				}
				segament_result.at<uchar>(y, x) = max_class_idx;
				pixel[c] = prob;
			}


		}
	}
	return segament_result;
}