#include "yolov5.h"

int main()
{
	OnnxDynamicNetInitParam params;
	params.onnx_model = "./yolov5/weights/onnx/shelf_yolov5_s.onnx";
	params.rt_model_name = "yolov5.engine";

	YOLOV5 yolov5(params);

	 cv::Mat img = cv::imread("./assets/image--01c1276cb6e346e893cc3d3ce6c6b9df.jpg");
	 yolov5.Extract(img);
	 cv::imshow("img", img);
	 cv::waitKey();
	
}