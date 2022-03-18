#ifndef  MAT_TRANSFORM_H_
#define MAT_TRANSFORM_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>
#include <assert.h>
#include <time.h>

using namespace std;

class ComposeMatLambda
{
public:
	using FuncionType = std::function<cv::Mat(const cv::Mat&)>;

	ComposeMatLambda() = default;
	ComposeMatLambda(const vector<FuncionType>& lambda) :lambda_(lambda)
	{
		;
	}
	cv::Mat operator()(cv::Mat& img)
	{
		for (auto func : lambda_)
			img = func(img);
		return img;
	}
private:
	vector<FuncionType> lambda_;
};

class LetterResize
{
public:
	LetterResize(cv::Size& new_shape = cv::Size(640, 640),
		cv::Scalar& color = cv::Scalar(114, 114, 114),
		int stride = 32) :new_shape_(new_shape), color_(color), stride_(stride) {}

	cv::Mat operator()(const cv::Mat& img)
	{
		int img_h = img.rows;
		int img_w = img.cols;

		int shape_h = new_shape_.height;
		int shape_w = new_shape_.width;

		double r = std::min(double(shape_h) / double(img_h), double(shape_w) / double(img_w));
		cout << "r: " << r << endl;


		cv::Size new_unpad = cv::Size(int(round(r * img_w)), int(round(r * img_h)));

		int dw = new_shape_.width - new_unpad.width;
		int dh = new_shape_.height - new_unpad.height;
		dw = dw % stride_;
		dh = dh % stride_;

		dw /= 2;
		dh /= 2;

		cv::Mat resize_mat;
		if (img.rows != new_unpad.height || img.cols != new_unpad.width)
			cv::resize(img, resize_mat, new_unpad, 0, 0, cv::INTER_LINEAR);
		int top = int(round(dh - 0.1));
		int bottom = int(round(dh + 0.1));
		int left = int(round(dw - 0.1));
		int right = int(round(dw + 0.1));
		cv::Mat pad_mat;
		cv::copyMakeBorder(resize_mat, pad_mat, top, bottom, left, right, cv::BORDER_CONSTANT, color_);

		return std::move(pad_mat);
	}

private:
	cv::Size new_shape_;
	cv::Scalar color_;
	int stride_;
};

class MatDivConstant
{
public:
	MatDivConstant(float constant) :constant_(constant) {}
	cv::Mat operator()(const cv::Mat& img)
	{
		cv::Mat tmp;
		cv::cvtColor(img, tmp, cv::COLOR_BGR2RGB);
		tmp.convertTo(tmp, CV_32FC3, 1, 0);
		tmp = tmp / constant_;
		return move(tmp);
	}

private:
	float constant_;
};

#endif