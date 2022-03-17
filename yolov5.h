#ifndef YOLOV5_H_
#define YOLOV5_H_

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvOnnxParser.h"

using namespace std;

struct OnnxDynamicNetInitParam
{
	std::string onnx_model;
	int gpu_id = 0;
	std::string rt_stream_path = "./";
	std::string rt_model_name = "defaule.gie";
	bool use_fp16_ = true;
};

class YOLOV5
{
public:
	YOLOV5() = delete;
	YOLOV5(const OnnxDynamicNetInitParam& param);

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

private:
	OnnxDynamicNetInitParam params_;
	cudaStream_t stream_;

	Logger logger_;
	bool use_fp16_;

	nvinfer1::IRuntime* runtime_;
	nvinfer1::ICudaEngine* engine_;
	nvinfer1::IExecutionContext* context_;
	nvinfer1::IHostMemory* gie_model_stream_{ nullptr };
};

#endif