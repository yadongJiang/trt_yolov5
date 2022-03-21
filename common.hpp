#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <vector>

using namespace std;

class Shape
{
public:
	Shape() :num_(0), channels_(0), height_(0), width_(0), no_(0) {}
	Shape(int num, int channels, int height, int width, int no=1) :
		num_(num), channels_(channels), height_(height), width_(width), no_(no) {}

	inline const int num() { return num_; }
	inline const int channels() { return channels_; }
	inline const int height() { return height_; }
	inline const int width() { return width_; }
	inline const int no() { return no_; }
	inline const int count() { return num_ * channels_ * height_ * width_ * no_; }

	inline void set_height(int height) { height_ = height; }
	inline void set_width(int width) { width_ = width; }
	inline void set_no(int no) { no_ = no; }

	void Reshape(int num, int channels, int height, int width)
	{
		num_ = num;
		channels_ = channels;
		height_ = height;
		width_ = width;
	}

private:
	int num_;
	int channels_;
	int height_;
	int width_;
	int no_; // 模型在每一个anchor的输出数量(4(xywh) + 1(obj) + num_classes)
};

class Tensor2VecMat
{
public:
	Tensor2VecMat() {}
	vector<cv::Mat> operator()(float* h_input_tensor, int channels, int height, int width)
	{
		vector<cv::Mat> input_channels;
		/*cout << *input_data << endl;*/
		for (int i = 0; i < channels; i++)
		{
			cv::Mat channel(height, width, CV_32FC1, h_input_tensor);
			input_channels.push_back(channel);
			h_input_tensor += height * width;
		}
		return std::move(input_channels);
	}
};

struct BoxInfo
{
public:
	int x1;
	int y1;
	int x2;
	int y2;
	float class_conf;
	float score;
	int class_idx;

	BoxInfo() : x1(0), y1(0), x2(0), y2(0), class_conf(0), score(0), class_idx(-1) {}
	BoxInfo(int lx, int ly, int rx, int ry, float conf, float s, int idx)
		: x1(lx), y1(ly), x2(rx), y2(ry), class_conf(conf), score(s), class_idx(idx) {}
};

#endif