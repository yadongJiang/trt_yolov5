#include "yolov5.h"
#include <fstream>
#include <sstream>
#include <assert.h>
#include "mat_transform.hpp"
#include "gpu_func.cuh"

YOLOV5::YOLOV5(const OnnxDynamicNetInitParam& params) : params_(params)
{
	cout << "start init ..." << endl;
	cudaSetDevice(params.gpu_id);
	cudaStreamCreate(&stream_);

	if (!LoadGieStreamBuildContext(params.rt_stream_path + params.rt_model_name))
	{
		LoadOnnxModel(params.onnx_model);
		SaveRtModel(params.rt_stream_path + params.rt_model_name);
	}
}

YOLOV5::~YOLOV5()
{
	cudaStreamSynchronize(stream_);
	cudaStreamDestroy(stream_);
	if (h_input_tensor_ != NULL)
		cudaFreeHost(h_input_tensor_);
	if (h_output_tensor8_ != NULL)
		cudaFreeHost(h_output_tensor8_);
	if (h_output_tensor16_ != NULL)
		cudaFreeHost(h_output_tensor16_);
	if (h_output_tensor32_ != NULL)
		cudaFreeHost(h_output_tensor32_);
	if (d_input_tensor_ != NULL)
		cudaFree(d_input_tensor_);
	if (d_output_tensor8_ != NULL)
		cudaFree(d_output_tensor8_);
	if (d_output_tensor16_ != NULL)
		cudaFree(d_output_tensor16_);
	if (d_output_tensor32_ != NULL)
		cudaFree(d_output_tensor32_);
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

	deserializeCudaEngine(gie_model_stream_->data(), gie_model_stream_->size());
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

	mallocInputOutput();
}

void YOLOV5::SaveRtModel(const std::string& path)
{
	std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
	outfile.write((const char*)gie_model_stream_->data(), gie_model_stream_->size());
	outfile.close();
}

bool YOLOV5::LoadGieStreamBuildContext(const std::string& gie_file)
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

void YOLOV5::mallocInputOutput()
{
	buffers_.clear();
	cudaHostAlloc((void**)&h_input_tensor_, input_shape_.count() * sizeof(float), cudaHostAllocDefault);  // 3 * 640 * 640
	cudaMalloc((void**)&d_input_tensor_, input_shape_.count() * sizeof(float)); // 3 * 640 * 640
	
	cudaHostAlloc((void**)&h_output_tensor8_, out_shape8_.count() * sizeof(float), cudaHostAllocDefault); // 3 * 80 * 80 * 6
	cudaMalloc((void**)&d_output_tensor8_, out_shape8_.count() * sizeof(float));  // 3 * 80 * 80 * 6

	cudaHostAlloc((void**)&h_output_tensor16_, out_shape16_.count() * sizeof(float), cudaHostAllocDefault);  // 3 * 40 * 40 * 6
	cudaMalloc((void**)&d_output_tensor16_, out_shape16_.count() * sizeof(float));  // 3 * 40 * 40 * 6

	cudaHostAlloc((void**)&h_output_tensor32_, out_shape32_.count() * sizeof(float), cudaHostAllocDefault);  // 3 * 20 * 20 * 6
	cudaMalloc((void**)&d_output_tensor32_, out_shape32_.count() * sizeof(float));  // 3 * 20 * 20 * 6

	buffers_.push_back(d_input_tensor_);
	buffers_.push_back(d_output_tensor8_);
	buffers_.push_back(d_output_tensor16_);
	buffers_.push_back(d_output_tensor32_);
}

// ---------------------------- FOR INFERENCE -----------------------------

void YOLOV5::Extract(const cv::Mat& img)
{
	if (img.empty())
		return;

	PreprocessCPU(img);
	Forward();
	//PostprocessCPU();
	PostprocessGPU();
}

void YOLOV5::Forward()
{
	cudaMemcpy(d_input_tensor_, h_input_tensor_, input_shape_.count() * sizeof(float), cudaMemcpyHostToDevice);
	nvinfer1::Dims4 input_dims{ 1, input_shape_.channels(), input_shape_.height(), input_shape_.width() };
	context_->setBindingDimensions(0, input_dims);
	context_->enqueueV2(buffers_.data(), stream_, nullptr);

	cudaStreamSynchronize(stream_);
}

void YOLOV5::PreprocessCPU(const cv::Mat& img)
{
	cv::Mat img_tmp = img;

	ComposeMatLambda compose({
		LetterResize(crop_size_),
		MatDivConstant(255.)
	});

	cv::Mat sample_float = compose(img_tmp);

	input_shape_.Reshape(1, sample_float.channels(), sample_float.rows, sample_float.cols);
	out_shape8_.Reshape(1, 3, sample_float.rows / 8, sample_float.cols / 8);
	out_shape16_.Reshape(1, 3, sample_float.rows / 16, sample_float.cols / 16);
	out_shape32_.Reshape(1, 3, sample_float.rows / 32, sample_float.cols / 32);

	Tensor2VecMat tensor2mat;
	std::vector<cv::Mat> channels = tensor2mat(h_input_tensor_, sample_float.channels(), 
											   sample_float.rows, sample_float.cols);
	cv::split(sample_float, channels);
}

void YOLOV5::PostprocessCPU()
{
	cudaMemcpy(h_output_tensor8_, d_output_tensor8_, out_shape8_.count()*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_tensor16_, d_output_tensor16_, out_shape16_.count() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_tensor32_, d_output_tensor32_, out_shape32_.count() * sizeof(float), cudaMemcpyDeviceToHost);

	filted_pred_boxes_.clear();
	DecodeBoxes(h_output_tensor8_, out_shape8_.channels(), out_shape8_.height(), out_shape8_.width(), 8, 0);
	DecodeBoxes(h_output_tensor16_, out_shape16_.channels(), out_shape16_.height(), out_shape16_.width(), 16, 1);
	DecodeBoxes(h_output_tensor32_, out_shape32_.channels(), out_shape32_.height(), out_shape32_.width(), 32, 2);
	cout << "filted_pred_boxes_ size: " << filted_pred_boxes_.size() << endl;

	vector<BoxInfo> pred_boxes = NMS();
	cout << "pred boxes size: " << pred_boxes.size() << endl;

	/*cv::Mat img = cv::imread("float_sample2.jpg");
	for (auto& box : pred_boxes)
	{
		cv::rectangle(img, cv::Rect(box.x1, box.y1,
			box.x2 - box.x1, box.y2 - box.y1), cv::Scalar(0, 0, 255), 2);
	}
	cv::imshow("img", img);
	cv::waitKey();*/
}

void YOLOV5::PostprocessGPU()
{
	int height = input_shape_.height();
	int width = input_shape_.width();
	int no = out_shape8_.no();

	float* dev_ptr;
	int count8 = 3 * (height / 8) * (width / 8) * no;
	int count16 = 3 * (height / 16) * (width / 16) * no;
	int count32 = 3 * (height / 32) * (width / 32) * no;
	int counts = count8 + count16 + count32;
	cudaMalloc((void**)&dev_ptr, counts * sizeof(float));
	cudaMemcpy(dev_ptr, d_output_tensor8_, count8 * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_ptr + count8, d_output_tensor16_, count16 * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_ptr + count8 + count16, d_output_tensor32_, count32 * sizeof(float), cudaMemcpyDeviceToDevice);

	postprocess(dev_ptr, height, width, no, counts);

	cudaMemcpy(h_output_tensor8_, dev_ptr, count8 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_tensor16_, dev_ptr + count8, count16 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_tensor32_, dev_ptr + count8 + count16, count32 * sizeof(float), cudaMemcpyDeviceToHost);

	filted_pred_boxes_.clear();
	DecodeBoxesGPU(h_output_tensor8_, out_shape8_.channels(), out_shape8_.height(), out_shape8_.width());
	DecodeBoxesGPU(h_output_tensor16_, out_shape16_.channels(), out_shape16_.height(), out_shape16_.width());
	DecodeBoxesGPU(h_output_tensor32_, out_shape32_.channels(), out_shape32_.height(), out_shape32_.width());
	cout << "filted_pred_boxes_ size: " << filted_pred_boxes_.size() << endl;

	vector<BoxInfo> pred_boxes = NMS();
	cout << "pred boxes size: " << pred_boxes.size() << endl;

	/*cv::Mat img = cv::imread("float_sample2.jpg");
	for (auto& box : pred_boxes)
	{
		cv::rectangle(img, cv::Rect(box.x1, box.y1,
			box.x2 - box.x1, box.y2 - box.y1), cv::Scalar(0, 0, 255), 2);
	}
	cv::imshow("img", img);
	cv::waitKey();*/

	cudaFree(dev_ptr);
}

void YOLOV5::DecodeBoxes(float* ptr, int channels, int height, int width, int stride, int layer_idx)
{
	int no = out_shape8_.no();
	vector<vector<float>>  anchors = anchors_[layer_idx];

	for (int i = 0; i < channels * height * width * no; i++)
		sigmoid(ptr[i]);

	for (int c = 0; c < channels; c++)
	{
		vector<float> anchor = anchors[c];
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				int offset = c * height * width * no + (row * width + col) * no;
				float obj_conf = ptr[offset + 4];
				if (obj_conf <= conf_thres_)
					continue;

				float class_conf;
				int class_pred;
				vector<float> vec(ptr + offset + 4, ptr + offset + no);
				FindMaxConfAndIdx(vec, class_conf, class_pred);

				float score = class_conf * obj_conf;
				if (score <= conf_thres_)
					continue;

				ptr[offset + 0] = (ptr[offset + 0] * 2.0 - 0.5 + col) * stride;
				ptr[offset + 1] = (ptr[offset + 1] * 2.0 - 0.5 + row) * stride;

				ptr[offset + 2] = pow(ptr[offset + 2] * 2.0, 2) * anchor[0];
				ptr[offset + 3] = pow(ptr[offset + 3] * 2.0, 2) * anchor[1];

				BoxInfo box(ptr[offset + 0] - ptr[offset + 2] / 2,
							ptr[offset + 1] - ptr[offset + 3] / 2,
							ptr[offset + 0] + ptr[offset + 2] / 2,
							ptr[offset + 1] + ptr[offset + 3] / 2,
						    class_conf, score, class_pred);

				filted_pred_boxes_.emplace_back(box);
			}
		}
	}
}

void YOLOV5::DecodeBoxesGPU(float* ptr, int channels, int height, int width)
{
	int no = out_shape8_.no();

	for (int c = 0; c < channels; c++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				int offset = c * height * width * no + (row * width + col) * no;
				float obj_conf = ptr[offset + 4];
				if (obj_conf <= conf_thres_)
					continue;

				float class_conf;
				int class_pred;
				vector<float> vec(ptr + offset + 4, ptr + offset + no);
				FindMaxConfAndIdx(vec, class_conf, class_pred);

				float score = class_conf * obj_conf;
				if (score <= conf_thres_)
					continue;

				BoxInfo box(ptr[offset + 0] - ptr[offset + 2] / 2,
					ptr[offset + 1] - ptr[offset + 3] / 2,
					ptr[offset + 0] + ptr[offset + 2] / 2,
					ptr[offset + 1] + ptr[offset + 3] / 2,
					class_conf, score, class_pred);

				filted_pred_boxes_.emplace_back(box);
			}
		}
	}
}

void YOLOV5::FindMaxConfAndIdx(const vector<float>& vec,
	float& class_conf, int& class_pred)
{
	float max_val = FLT_MIN;
	int max_idx = -1;
	for (int i = 0; i < vec.size(); i++)
	{
		if (max_val < vec[i])
		{
			max_val = vec[i];
			max_idx = i;
		}
	}
	class_conf = max_val;
	class_pred = max_idx;
}

vector<BoxInfo> YOLOV5::NMS()
{
	vector<BoxInfo> pred_boxes;
	if (filted_pred_boxes_.empty())
		return pred_boxes;

	sort(filted_pred_boxes_.begin(), filted_pred_boxes_.end(), compose);
	RefineBoxes();

	char* removed = (char*)malloc(filted_pred_boxes_.size() * sizeof(char));
	memset(removed, 0, filted_pred_boxes_.size() * sizeof(char));
	for (int i = 0; i < filted_pred_boxes_.size(); i++)
	{
		if (removed[i])
			continue;

		pred_boxes.push_back(filted_pred_boxes_[i]);
		for (int j = i + 1; j < filted_pred_boxes_.size(); j++)
		{
			if (filted_pred_boxes_[i].class_idx != filted_pred_boxes_[j].class_idx)
				continue;
			float iou = IOU(filted_pred_boxes_[i], filted_pred_boxes_[j]);
			if (iou >= iou_thres_)
				removed[j] = 1;
		}
	}
	return std::move(pred_boxes);
}

// 调整预测框，使框的值处于合理范围
void YOLOV5::RefineBoxes()
{
	for (auto& box : filted_pred_boxes_)
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

float YOLOV5::IOU(BoxInfo& b1, BoxInfo& b2)
{
	b1.x1 = min<float>(max<float>(b1.x1, 0.), 640.);
	b1.y1 = min<float>(max<float>(b1.y1, 0.), 640.);
	b1.x2 = min<float>(max<float>(b1.x2, 0.), 640.);
	b1.y2 = min<float>(max<float>(b1.y2, 0.), 640.);

	b2.x1 = min<float>(max<float>(b2.x1, 0.), 640.);
	b2.y1 = min<float>(max<float>(b2.y1, 0.), 640.);
	b2.x2 = min<float>(max<float>(b2.x2, 0.), 640.);
	b2.y2 = min<float>(max<float>(b2.y2, 0.), 640.);

	float x1 = max<float>(b1.x1, b2.x1);
	float y1 = max<float>(b1.y1, b2.y1);
	float x2 = min<float>(b1.x2, b2.x2);
	float y2 = min<float>(b1.y2, b2.y2);

	float inter_area = max<float>(x2 - x1, 0) * max<float>(y2 - y1, 0);
	float b1_area = max<float>(b1.x2 - b1.x1, 0) * max<float>(b1.y2 - b1.y1, 0);
	float b2_area = max<float>(b2.x2 - b2.x1, 0) * max<float>(b2.y2 - b2.y1, 0);
	return inter_area / (b1_area + b2_area - inter_area + 1e-5);
}