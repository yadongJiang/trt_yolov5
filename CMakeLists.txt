cmake_minimum_required(VERSION 3.0)

project(yolov5)

# add_definitions(-std=c++11)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_53;code=sm_53)

find_package(OpenCV REQUIRED)

link_directories(
    "/usr/local/cuda-10.2/lib64"
    "/home/jyd/download/TensorRT/TensorRT-7.2.1.6/lib"
)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    "/home/jyd/download/TensorRT/TensorRT-7.2.1.6/include"
)

file(GLOB SOURCE 
    ./src/*.cpp
)

cuda_add_executable(${PROJECT_NAME} ${SOURCE} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} nvonnxparser)

target_link_libraries(${PROJECT_NAME} nvinfer_plugin)