#include "yolov5.h"
#include <time.h>

int main()
{
	OnnxDynamicNetInitParam params;
	params.onnx_model = "./yolov5/weights/onnx/shelf_yolov5_s.onnx";
	params.rt_model_name = "yolov5.engine";
	params.num_classes = 1;

	YOLOV5 yolov5(params);

	cv::Mat img = cv::imread("./assets/image--01c1276cb6e346e893cc3d3ce6c6b9df.jpg");

	std::clock_t start, end;
	yolov5.Extract(img);
	yolov5.Extract(img);
	int total = 0;
	for (int i = 0; i < 100; i++)
	{
		start = clock();
		yolov5.Extract(img);
		end = clock();
		std::cout << "cost time: " << end - start << endl;
		total += (end - start);
	}
	cout << "ave cost time: " << total / 100. << endl;

	cv::imshow("img", img);
	cv::waitKey();
	
}