#include "yolov5.h"
#include <fstream>
#include <sstream>
#include <assert.h>
#include "mat_transform.hpp"
// #include "gpu_func.cuh"
#include <time.h>

Yolov5::Yolov5(const OnnxDynamicNetInitParam& params) : params_(params)
{
	cout << "start init ..." << endl;
	cudaSetDevice(params.gpu_id);
	cudaStreamCreate(&stream_);
	
	output_shape_.Reshape(output_shape_.num(), output_shape_.channels(), 
							5+params.num_classes, 1);

	if (!LoadGieStreamBuildContext(params.rt_stream_path + params.rt_model_name))
	{
		LoadOnnxModel(params.onnx_model);
		SaveRtModel(params.rt_stream_path + params.rt_model_name);
	}
}

Yolov5::~Yolov5()
{
	cudaStreamSynchronize(stream_);
	cudaStreamDestroy(stream_);
	if (h_input_tensor_ != nullptr)
		cudaFreeHost(h_input_tensor_);
	if (h_output_tensor_ != nullptr)
		cudaFreeHost(h_output_tensor_);
	if (d_input_tensor_ != NULL)
		cudaFree(d_input_tensor_);
	if (d_output_tensor8_ != NULL)
		cudaFree(d_output_tensor8_);
	if (d_output_tensor16_ != NULL)
		cudaFree(d_output_tensor16_);
	if (d_output_tensor32_ != NULL)
		cudaFree(d_output_tensor32_);
	if (d_output_tensor_ != NULL)
		cudaFree(d_output_tensor_);
}

bool Yolov5::CheckFileExist(const std::string& path)
{
	std::fstream check_file(path);
	bool found = check_file.is_open();
	return found;
}

void Yolov5::LoadOnnxModel(const std::string& onnx_file)
{
	if (!CheckFileExist(onnx_file))
	{
		std::cerr << "onnx file is not found " << onnx_file << std::endl;
		exit(0);
	}

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger_);
	assert(builder != nullptr);

	const auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::
										NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx������
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger_);
	assert(parser->parseFromFile(onnx_file.c_str(), 2));

	nvinfer1::IBuilderConfig* build_config = builder->createBuilderConfig();
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
	nvinfer1::ITensor* input = network->getInput(0);
	std::cout << "********************* : " << input->getName() << std::endl;
	nvinfer1::Dims dims = input->getDimensions();
	std::cout << "batchsize: " << dims.d[0] << " channels: " << dims.d[1] << " height: " << dims.d[2] << " width: " << dims.d[3] << std::endl;

	{
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, 
													nvinfer1::Dims4{ 1, dims.d[1], 1, 1 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, 
													nvinfer1::Dims4{ 1, dims.d[1], 640, 640 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, 
													nvinfer1::Dims4{ 1, dims.d[1], 640, 640 });
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

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *build_config); // 
	assert(engine != nullptr);

	gie_model_stream_ = engine->serialize();

	parser->destroy();
	engine->destroy();
	builder->destroy();
	network->destroy();

	deserializeCudaEngine(gie_model_stream_->data(), gie_model_stream_->size());
}

void Yolov5::deserializeCudaEngine(const void* blob, std::size_t size)
{
	// ��������ʱ
	runtime_ = nvinfer1::createInferRuntime(logger_);
	assert(runtime_ != nullptr);
	// ������ʱ���ݶ�ȡ�����л���ģ�ͷ����л�����engine
	engine_ = runtime_->deserializeCudaEngine(blob, size, nullptr);
	assert(engine_ != nullptr);

	// ����engine����ִ��������
	context_ = engine_->createExecutionContext();
	assert(context_ != nullptr);

	mallocInputOutput();
}

void Yolov5::SaveRtModel(const std::string& path)
{
	std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
	outfile.write((const char*)gie_model_stream_->data(), gie_model_stream_->size());
	outfile.close();
}

bool Yolov5::LoadGieStreamBuildContext(const std::string& gie_file)
{
	std::ifstream fgie(gie_file, std::ios_base::in | std::ios_base::binary);
	if (!fgie)
		return false;

	std::stringstream buffer;
	buffer << fgie.rdbuf();

	std::string stream_model(buffer.str());

	deserializeCudaEngine(stream_model.data(), stream_model.size());

	return true;
}

void Yolov5::mallocInputOutput()
{
	buffers_.clear();

	int num = engine_->getNbBindings();
	nvinfer1::Dims input_dim = engine_->getBindingDimensions(0);
	nvinfer1::Dims output_dim = engine_->getBindingDimensions(4);

	cudaHostAlloc((void**)&h_input_tensor_, input_shape_.count() * sizeof(float), 
													cudaHostAllocDefault);  // 3 * 640 * 640
	cudaMalloc((void**)&d_input_tensor_, input_shape_.count() * sizeof(float)); // 3 * 640 * 640
	
	cudaHostAlloc((void **)&h_output_tensor_, output_shape_.count()*sizeof(float), 
													cudaHostAllocDefault);
	cudaMalloc((void **)&d_output_tensor_, output_shape_.count()*sizeof(float));

	cudaMalloc((void **)&d_output_tensor8_, 3*(5+params_.num_classes)*input_shape_.height()*
									input_shape_.width()/64 * sizeof(float));
	cudaMalloc((void **)&d_output_tensor16_, 3*(5+params_.num_classes)*input_shape_.height()*
									input_shape_.width()/256 * sizeof(float));
	cudaMalloc((void **)&d_output_tensor32_, 3*(5+params_.num_classes)*input_shape_.height()*
									input_shape_.width()/1024 * sizeof(float));

	buffers_.push_back(d_input_tensor_); 
	buffers_.push_back(d_output_tensor8_);
	buffers_.push_back(d_output_tensor16_);
	buffers_.push_back(d_output_tensor32_);
	buffers_.push_back(d_output_tensor_);
}

vector<BoxInfo> Yolov5::Extract(const cv::Mat& img)
{
	if (img.empty())
		return vector<BoxInfo> {};

	std::lock_guard<std::mutex> lock(mtx_);
	preProcessCPU(img);
	Forward();
	auto res_det = postProcessCPU();
	coord_scale(img, res_det);
	
	return std::move(res_det);
	// return std::vector<BoxInfo> {} ; 
}

void Yolov5::preProcessCPU(const cv::Mat &img)
{
	cv::Mat tmp = img;

	ComposeMatLambda compose({
		LetterResize(crop_size_),
		MatDivConstant(255.)
	});

	tmp = compose(tmp);
	input_shape_.Reshape(1, tmp.channels(), tmp.rows, tmp.cols);

	float rh = (float)tmp.rows / img.rows;
	float rw = (float)tmp.cols / img.cols;
	rate_ = rh < rw ? rh : rw;

	std::vector<cv::Mat> channels;
	cv::split(tmp, channels);

	int offset = 0;
	for(const auto &channel : channels)
	{
		cudaMemcpy(d_input_tensor_ + offset, channel.data, 
					channel.total() * sizeof(float), cudaMemcpyHostToDevice);
		offset += channel.total();
	}

	cudaMemcpy(h_input_tensor_, d_input_tensor_, tmp.channels() * 
					tmp.rows*tmp.cols*sizeof(float), cudaMemcpyDeviceToHost);
}

void Yolov5::Forward()
{
	nvinfer1::Dims4 input_dims{ 1, input_shape_.channels(), 
					input_shape_.height(), input_shape_.width() };
	context_->setBindingDimensions(0, input_dims);
	context_->enqueueV2(buffers_.data(), stream_, nullptr);

	cudaStreamSynchronize(stream_);
}

vector<BoxInfo> Yolov5::postProcessCPU()
{
	int in_height = input_shape_.height();
	int in_width = input_shape_.width();
	cudaMemcpy(h_output_tensor_, d_output_tensor_, 3*(in_height*in_width/64 + 
													  in_height*in_width/256 + 
													  in_height*in_width/1024) * 
													  (5+params_.num_classes) * sizeof(float), cudaMemcpyDeviceToHost);

	std::vector<BoxInfo> boxes = Decode();
	return std::move(boxes);
}

std::vector<BoxInfo> Yolov5::Decode()
{
	std::vector<BoxInfo> boxes;

	int in_height = input_shape_.height();
	int in_width = input_shape_.width();
	int num_anchors = 3*(in_height*in_width/64 + 
						in_height*in_width/256 + 
						in_height*in_width/1024);
	int out_channels = 5 + params_.num_classes;

	for(int n=0; n<num_anchors; n++)
	{
		int pos = n * out_channels;
		
		float obj_score = h_output_tensor_[pos + 4];
		if(obj_score < conf_thres_)
			continue;
		
		BoxInfo box;
		for(int i=0; i<4; i++)
		{
			float x = h_output_tensor_[pos + 0];
			float y = h_output_tensor_[pos + 1];
			float w = h_output_tensor_[pos + 2];
			float h = h_output_tensor_[pos + 3];
			box.x1 = int(x - w/2);
			box.y1 = int(y - h/2);
			box.x2 = int(x + w/2);
			box.y2 = int(y + h/2);
		}

		float max_score = FLT_MIN;
		float max_cls_score = FLT_MIN;
		int max_cls_id = -1;
		for(int i=5; i<out_channels; i++)
		{
			float score = h_output_tensor_[pos + i] * obj_score;
			if(score > max_score)
			{
				max_score = score;
				max_cls_id = i - 5;
				max_cls_score = h_output_tensor_[pos + i];
			}
		}
		if (max_score < conf_thres_)
			continue;
		
		box.score = obj_score;
		box.class_idx = max_cls_id;
		box.class_conf = max_cls_score;

		boxes.push_back(std::move(box));
	}

	auto res = NMS(boxes);
	
	return std::move(res);
}

void Yolov5::coord_scale(const cv::Mat &img, 
						vector<BoxInfo>& pred_boxes)
{
	int h = int(round(img.rows * rate_));
	int w = int(round(img.cols * rate_));

	int dw = (crop_size_.width - w) % 32;
	int dh = (crop_size_.height - h) % 32;
	float fdw = dw / 2.;
	float fdh = dh / 2.;

	int top = int(round(fdh - 0.1));
	int left = int(round(fdw - 0.1));

	for (auto& box : pred_boxes)
	{
		box.x1 = (box.x1 - left) / rate_;
		box.x2 = (box.x2 - left) / rate_;
		box.y1 = (box.y1 - top)/ rate_;
		box.y2 = (box.y2 - top) / rate_;
	}
}

vector<BoxInfo> Yolov5::NMS(std::vector<BoxInfo> &boxes)
{
	vector<BoxInfo> pred_boxes;
	if (boxes.empty())
		return pred_boxes;

	sort(boxes.begin(), boxes.end(), compose);

	RefineBoxes(boxes);
	char* removed = (char*)malloc(boxes.size() * sizeof(char));
	memset(removed, 0, boxes.size() * sizeof(char));
	for (int i = 0; i < boxes.size(); i++)
	{
		if (removed[i])
			continue;

		pred_boxes.push_back(boxes[i]);
		for (int j = i + 1; j < boxes.size(); j++)
		{
			if (boxes[i].class_idx != boxes[j].class_idx)
				continue;
			float iou = IOU(boxes[i], boxes[j]);
			if (iou >= iou_thres_)
				removed[j] = 1;
		}
	}
	return std::move(pred_boxes);
}

void Yolov5::RefineBoxes(std::vector<BoxInfo> &boxes)
{
	for (auto& box : boxes)
	{
		box.x1 = box.x1 < 0. ? 0. : box.x1;
		box.x1 = box.x1 > 640. ? 640. : box.x1;
		box.y1 = box.y1 < 0. ? 0. : box.y1;
		box.y1 = box.y1 > 640. ? 640. : box.y1;
		box.x2 = box.x2 < 0. ? 0. : box.x2;
		box.x2 = box.x2 > 640. ? 640. : box.x2;
		box.y2 = box.y2 < 0. ? 0. : box.y2;
		box.y2 = box.y2 > 640. ? 640. : box.y2;
	}
}

float Yolov5::IOU(BoxInfo& b1, BoxInfo& b2)
{
	float x1 = b1.x1 > b2.x1 ? b1.x1 : b2.x1;
	float y1 = b1.y1 > b2.y1 ? b1.y1 : b2.y1;
	float x2 = b1.x2 < b2.x2 ? b1.x2 : b2.x2;
	float y2 = b1.y2 < b2.y2 ? b1.y2 : b2.y2;

	float inter_area = ((x2 - x1) < 0 ? 0 : (x2 - x1)) * ((y2 - y1) < 0 ? 0 : (y2 - y1));
	float b1_area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
	float b2_area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);

	return inter_area / (b1_area + b2_area - inter_area + 1e-5);
}