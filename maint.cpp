#include<iostream>
#include<vector>
#include<memory>
#include"ONNX_infer.h"
//opencv
#include<opencv2/opencv.hpp>
using namespace std;

template<typename T,typename Q>
T max_num(T a, Q b) {
	if (a > b) {
		return a;
	}
	return b;
}
class A {
public:
	A(int i) :data(2,vector<int>(i)){
		this->data[1][1] = 500;
		data.at(1).at(2) = 100;
	}
	vector<vector<int>> data; //定义一个二维数组
	//定义一个函数模板

	int& operator()(int row, int col)
	{
		return data[row][col];
	}
	~A() {
		
		cout << "Destructor called" << endl;
	}
};

class Point {
public:
	Point(int x, int y) :x(x), y(y) {
		vector<int>* p = new vector<int>();
		//unique_ptr<vector<int>> p1= make_unique<vector<int>>();
		//cout<<p1->size();
	}
	int x, y;
	vector<int>* p;
	~Point() {
		delete p;
		cout << "Destructor called" << endl;
	}
};
std::vector<std::vector<int>> generateRandomMatrix(int rows, int cols) {
	std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
	srand(time(0)); // 设置随机种子

	for (auto& row : matrix) {
		for (auto& num : row) {
			num = rand() % 100; // 生成0-99的随机数
		}
	}
	return matrix;
}
cv::Mat u16Normalization(cv::Mat& inputMat) {
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

	//// 转换为8位灰度图像
	//x1.convertTo(x1, CV_8UC1);
	//// 转换为8位灰度图像
	//x2.convertTo(x2, CV_8UC1);
	//// 转换为8位灰度图像
	//x3.convertTo(x3, CV_8UC1);

	// 创建一个新的Mat对象，将两个8位灰度图组装在一起
	cv::Mat result;
	cv::merge(std::vector<cv::Mat>{ x2, x3, x1}, result);

	return result;
}
void printMatrix(const std::vector<std::vector<int>>& matrix) {
	for (const auto& row : matrix) {
		for (int num : row) {
			std::cout << num << "\t";
		}
		std::cout << "\n";
	}
}
//static cv::Mat ONNX_process(cv::Mat img0) {
//	//cv::Mat img0;
//	
//	//int img1=rand()%256;
//	cv::Mat float_img = img0.clone();
//	//float_img.convertTo(img0, CV_32FC1); // 将图像转换为浮点型
//	//cv::Mat img1 = img0 / 65535 * 255.0 / 255 * 2 - 1;
//	//cv::Mat img2, img3;
//	//float_img.convertTo(img2, CV_32FC1, 1.0 / 255.0);
//	//float_img.convertTo(img3, CV_32FC1, 1.0 / 65535.0);
//	cv::Mat res = u16Normalization(float_img);
//	/*cv::merge(std::vector<cv::Mat>{img2, img3, img1}, res);*/
//	cv::Mat blob = cv::dnn::blobFromImage(res);
//	cv::dnn::Net mynet;
//	//cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);
//	string model_path = ".\\test.onnx";
//	mynet = cv::dnn::readNetFromONNX(model_path);
//	std::vector<cv::String> input_names = mynet.getUnconnectedOutLayersNames();
//	std::vector<cv::String> output_names = mynet.getUnconnectedOutLayersNames();
//	mynet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
//	mynet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//	srand(2023816);
//	mynet.setInput(blob);
//	std::vector<cv::Mat> outputs;
//
//	mynet.forward(outputs, output_names);
//	cv::Mat result = outputs[0];
//	cv::Mat segament_result(512, 960, CV_8UC1);
//	//std::cout << segament_result.type() << endl;
//
//	cv::Mat result2(512, 960, CV_32FC3);
//	int height = 512;
//	int width = 960;
//
//	for (int y = 0; y < 512; y++) {
//		for (int x = 0; x < 960; x++) {
//			cv::Vec3f& pixel = result2.at<cv::Vec3f>(y, x);
//			int max_class_idx = 0;
//			float max_prob = result.ptr<float>(0, 0, y)[x];
//     		for (int c = 0; c < 3; c++) {
//				float prob = result.ptr<float>(0, c, y)[x];
//				if (prob > max_prob) {
//					max_prob = prob;
//					max_class_idx = c;
//				}
//			segament_result.at<uchar>(y, x)=max_class_idx;
//			pixel[c] = prob;
//				}
//
//
//		}
//	}
//	return segament_result;
//}



int main() {
	cout << "Hello, World!" << endl;
	cv::Mat img;
	srand(unsigned int(10000));
	int rows = 3, cols = 4; // 默认3行4列
	auto randomMatrix = generateRandomMatrix(rows, cols);
	printMatrix(randomMatrix);
	cv::Mat randomMat(200, 200, CV_8UC1);
	//cv::randu(randomMat, 0, 256);
	cv::randn(randomMat, 100, 100);
	//cv::imshow("aaa", randomMat);
	cv::Mat img0 = cv::imread("D:\\datachange\\119Lz_29tier\\2025_04_25_30RZ_LFP_msa_ng2_34tier\\crop_org\\2025_04_25_30RZ_LFP_msa_ng_down_34tier_5.tif", cv::IMREAD_UNCHANGED);
	//cv::Mat result=ONNX_process(img0);
	string model_path = ".\\test.onnx";
	ONNX_process onnx_process(512,960,model_path);
	cv::Mat result = onnx_process.infer(img0);
	img = cv::abs(cv::Mat(512, 512, CV_8UC1, cv::Scalar(0)));
	//cv::imshow("Image", img);
	cv::waitKey(0);
	cout << "Max of 5 and 10 is: " << max_num(5, 10.1) << endl;
	A a(3);
	cout << a(1, 1) << endl;
	Point p(1, 2);
	vector<Point> points;
	points.push_back(p);
	points.push_back(Point(3, 4));
	cout << "Point1111 coordinates: (" << points[1].x << ", " << points[1].y << ")" << endl;
	cout << "Point coordinates: (" << p.x << ", " << p.y << ")" << endl;
	return 0;
}