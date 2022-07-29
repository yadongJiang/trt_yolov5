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

class Yolov5
{
public:
    Yolov5(const OnnxDynamicNetInitParam& param);
    ~Yolov5();

	vector<BoxInfo> Extract(const cv::Mat& img);

private:
    Yolov5() {}

private:
	bool CheckFileExist(const std::string& path);
	//加载onnx模型，生成engine模型
	void LoadOnnxModel(const std::string& onnx_file);
	void deserializeCudaEngine(const void* blob, std::size_t size);
	void SaveRtModel(const std::string& path);

	bool LoadGieStreamBuildContext(const std::string& gie_file);
	void mallocInputOutput();
	
	static bool compose(BoxInfo& box1, BoxInfo& box2)
	{
		return box1.score > box2.score;
	}

private:
	void Forward();
	void preProcessCPU(const cv::Mat& img);
	vector<BoxInfo> postProcessCPU();
	std::vector<BoxInfo> Decode();
	void coord_scale(const cv::Mat &img, 
						vector<BoxInfo>& pred_boxes);
	vector<BoxInfo> NMS(std::vector<BoxInfo> &boxes);
	// 将目标框限制在合理范围内
	inline void RefineBoxes(std::vector<BoxInfo> &boxes);
	inline float IOU(BoxInfo& b1, BoxInfo& b2);

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
	Shape output_shape_{1, 25200, 5+1, 1};

	vector<vector<vector<float>>> anchors_{
		{ {10, 13}, {16, 30}, { 33, 23} },
		{ {30, 61}, {62, 45}, { 59, 119} },
		{ {116, 90},{156,198 },{373, 326} }
	};

	float conf_thres_ = 0.25;
	float iou_thres_ = 0.45;

	vector<BoxInfo> filted_pred_boxes_;
	
	float* h_input_tensor_;
	float* d_input_tensor_;
	
	float* d_output_tensor8_;
	float* d_output_tensor16_;
	float* d_output_tensor32_;
	
	float *h_output_tensor_;
	float *d_output_tensor_;

	vector<void *> buffers_;

	float rate_;
	std::mutex mtx_;
};

#endif