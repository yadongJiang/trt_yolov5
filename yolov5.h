#ifndef YOLOV5_H_
#define YOLOV5_H_

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvOnnxParser.h"
#include "common.hpp"
#include <mutex>

using namespace std;

struct OnnxDynamicNetInitParam
{
	std::string onnx_model;
	int gpu_id = 0;
	std::string rt_stream_path = "./";
	std::string rt_model_name = "defaule.gie";
	bool use_fp16_ = true;
	int num_classes;
};

class YOLOV5
{
public:
	YOLOV5() = delete;
	YOLOV5(const OnnxDynamicNetInitParam& param);

	~YOLOV5();

	vector<BoxInfo> Extract(const cv::Mat& img);

private:
	class Logger : public nvinfer1::ILogger
	{
	public:
		void log(nvinfer1::ILogger::Severity severity, const char* msg)
		{
			switch (severity)
			{
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
				std::cerr << "kINTERNAL_ERROR: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kERROR:
				std::cerr << "kERROR: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kWARNING:
				std::cerr << "kWARNING: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kINFO:
				std::cerr << "kINFO: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kVERBOSE:
				std::cerr << "kVERBOSE: " << msg << std::endl;
				break;
			default:
				break;
			}
		}
	};

private:
	bool CheckFileExist(const std::string& path);
	// 直接加载onnx模型，并转换成trt模型
	void LoadOnnxModel(const std::string& onnx_file);
	void deserializeCudaEngine(const void* blob, std::size_t size);
	void SaveRtModel(const std::string& path);

	bool LoadGieStreamBuildContext(const std::string& gie_file);
	void mallocInputOutput();

private:
	void Forward();
	void PreprocessCPU(const cv::Mat& img);
	void PreprocessGPU(const cv::Mat& img);
	
	vector<BoxInfo> PostprocessCPU();
	vector<BoxInfo> PostprocessGPU();

	void DecodeBoxes(float *ptr, int channels, int height, int width, int stride, int layer_idx);
	void DecodeBoxesGPU(float* ptr, int channels, int height, int width);

	vector<BoxInfo> NMS();

	inline void sigmoid(float& val)
	{
		val = 1.0 / (exp(-val) + 1);
	}
	static bool compose(BoxInfo& box1, BoxInfo& box2)
	{
		return box1.score > box2.score;
	}

	inline void FindMaxConfAndIdx(const vector<float>& vec,
						float& class_conf, int& class_pred);

	// 调整预测框，使框的值处于合理范围
	inline void RefineBoxes();

	inline float IOU(BoxInfo& b1, BoxInfo& b2);

	void coord_scale(const cv::Mat &img, vector<BoxInfo>& pred_boxes);

private:
	OnnxDynamicNetInitParam params_;
	cudaStream_t stream_;

	Logger logger_;
	bool use_fp16_;

	nvinfer1::IRuntime* runtime_;
	nvinfer1::ICudaEngine* engine_;
	nvinfer1::IExecutionContext* context_;
	nvinfer1::IHostMemory* gie_model_stream_{ nullptr };

	int max_height_ = 640;
	int max_width_ = 640;

	cv::Size crop_size_{ 640, 640 };

	Shape input_shape_{1, 3, 640, 640};
	Shape out_shape8_{1, 3, 80, 80, 5+1}; // 默认num_classes=1,可在params中设置num_classes
	Shape out_shape16_{ 1, 3, 40, 40, 5+1 };
	Shape out_shape32_{ 1, 3, 20, 20, 5+1 };

	vector<vector<vector<float>>> anchors_{
		{ {10, 13}, {16, 30}, { 33, 23} },
		{ {30, 61}, {62, 45}, { 59, 119} },
		{ {116, 90},{156,198 },{373, 326} }
	};

	float conf_thres_ = 0.3;
	float iou_thres_ = 0.3;

	vector<BoxInfo> filted_pred_boxes_;
	
	float* h_input_tensor_;
	float* d_input_tensor_;
	
	float* h_output_tensor8_;
	float* d_output_tensor8_;
	float* h_output_tensor16_;
	float* d_output_tensor16_;
	float* h_output_tensor32_;
	float* d_output_tensor32_;

	float* dev_ptr_;

	vector<void *> buffers_;

	float rate_;
	std::mutex mtx_;
};

#endif