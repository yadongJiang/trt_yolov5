#include "yolov5.h"

int main()
{
    OnnxDynamicNetInitParam params;
	params.onnx_model = "../models/yolov5s.onnx"; // onnx模型路径
	params.rt_model_name = "../models/yolov5.engine"; // 生成的engine模型的保存路径，如果该模型已经存在，将直接调用，不会再通过onnx模型生成
	params.num_classes = 80; // ģ�������

	// 创建类的对象
	Yolov5 yolov5(params);

	cv::Mat img = cv::imread("../bus.jpg");
	// 输入图像，进行检测，返回检测输出
	std::vector<BoxInfo> preds = yolov5.Extract(img);

	std::cout<<"preds size: "<<preds.size()<<std::endl;

	// 检测效果可视化
	for(auto &box : preds)
		cv::rectangle(img, cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1), (0,0,255), 2);

	cv::imshow("kimg", img);
	cv::waitKey();
}