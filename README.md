# TensorRT实现YOLOV5部署
实现cpu预处理、gpu预处理、cpu后处理、gpu后处理，yolov5-s模型在nvidia 1060最快可达100帧/s(gpu预处理+gpu后处理)

## 环境依赖

1. Opencv3.1
2. TensorRT7.2
3. Cuda10.2

## 代码示例
具体可以参照main.cpp中的main函数，OnnxDynamicNetInitParam结构体中主要用来设置一些必要的参数，如onnx模型路径、生成的.engine模型的保存路径已经名称、模型的类别数(这个比较重要，必须设置)、gpu信息等。
```
    OnnxDynamicNetInitParam params;
    # onnx模型路径
    params.onnx_model = "./yolov5/weights/onnx/shelf_yolov5_s.onnx";
    # 生成的engine模型的保存名称，保存路径使用的是默认路径(当前文件夹)
    params.rt_model_name = "yolov5.engine";
    # 类别数
    params.num_classes = 1;

    # 构造YOLOV5类对象
    YOLOV5 yolov5(params);
    # 返回检测框
    vector<BoxInfo> pred_boxes = yolov5.Extract(img);
```

## 应用示例
yolov5+tensorrt对红外微小目标的检测，可以达到100帧以上(640x640的输入)，[视频](https://www.bilibili.com/video/BV1Fq4y1a7SS/)