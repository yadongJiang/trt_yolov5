#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void dev_sigmoid(float *ptr, int len);
__global__ void process_kernel(float *dev_ptr, int height, int width, int no, int total_pix);  // , float* anchors

void postprocess(float* dev_ptr, int height, int width, int no, int counts);

#endif