#include "yolov5.h"

int main()
{
	OnnxDynamicNetInitParam params;
	params.onnx_model = "E:/BaiduNetdiskDownload/yolov5/weights/onnx/shelf_yolov5_s.onnx";
	params.rt_model_name = "yolov5.engine";

	YOLOV5 yolov5(params);

	// cv::Mat img = cv::imread("E:/BaiduNetdiskDownload/YOLOX-main/assets/image--0dcd0d1781c04c6c80acf5435dc051cc.jpg");
	
}