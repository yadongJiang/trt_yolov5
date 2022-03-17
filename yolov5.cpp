#include "yolov5.h"
#include <fstream>
#include <sstream>
#include <assert.h>

YOLOV5::YOLOV5(const OnnxDynamicNetInitParam& params) : params_(params)
{
	cout << "start init ..." << endl;
	cudaSetDevice(params.gpu_id);
	cudaStreamCreate(&stream_);

	LoadOnnxModel(params.onnx_model);
	SaveRtModel(params.rt_stream_path + params.rt_model_name);
}

bool YOLOV5::CheckFileExist(const std::string& path)
{
	std::fstream check_file(path);
	bool found = check_file.is_open();
	return found;
}

void YOLOV5::LoadOnnxModel(const std::string& onnx_file)
{
	if (!CheckFileExist(onnx_file))
	{
		std::cerr << "onnx file is not found " << onnx_file << std::endl;
		exit(0);
	}

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger_);
	assert(builder != nullptr);

	const auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx解析器
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger_);
	assert(parser->parseFromFile(onnx_file.c_str(), 2));

	nvinfer1::IBuilderConfig* build_config = builder->createBuilderConfig();
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
	nvinfer1::ITensor* input = network->getInput(0);
	std::cout << "********************* : " << input->getName() << std::endl;
	nvinfer1::Dims dims = input->getDimensions();
	std::cout << "batchsize: " << dims.d[0] << " channels: " << dims.d[1] << " height: " << dims.d[2] << " width: " << dims.d[3] << std::endl;

	{
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, dims.d[1], 1, 1 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 1, dims.d[1], 640, 640 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ 1, dims.d[1], 640, 640 });
		build_config->addOptimizationProfile(profile);
	}

	build_config->setMaxWorkspaceSize(1 << 30);

	if (params_.use_fp16_)
		use_fp16_ = builder->platformHasFastFp16();
	if (use_fp16_)
	{
		builder->setHalf2Mode(true);
		std::cout << "useFP16		" << use_fp16_ << std::endl;
	}
	else
		std::cout << "Using GPU FP32 !" << std::endl;

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *build_config);
	assert(engine != nullptr);

	gie_model_stream_ = engine->serialize();

	parser->destroy();
	engine->destroy();
	builder->destroy();
	network->destroy();
}

void YOLOV5::deserializeCudaEngine(const void* blob, std::size_t size)
{
	// 创建运行时
	runtime_ = nvinfer1::createInferRuntime(logger_);
	assert(runtime_ != nullptr);
	// 由运行时根据读取的序列化的模型反序列化生成engine
	engine_ = runtime_->deserializeCudaEngine(blob, size, nullptr);
	assert(engine_ != nullptr);

	// 利用engine创建执行上下文
	context_ = engine_->createExecutionContext();
	assert(context_ != nullptr);
}

void YOLOV5::SaveRtModel(const std::string& path)
{
	std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
	outfile.write((const char*)gie_model_stream_->data(), gie_model_stream_->size());
	outfile.close();
}