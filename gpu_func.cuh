#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned char uchar;

__device__ void dev_sigmoid(float *ptr, int len);
__global__ void process_kernel(float *dev_ptr, int height, int width, int no, int total_pix);  // , float* anchors

__global__ void kernel_resize(float *d_dst, int channel, 
			int src_h, int src_w, 
			int dst_h, int dst_w, 
			int top, int bottom, int left, int right, 
			uchar *d_src);

void postprocess(float* dev_ptr, int height, int width, int no, int counts);

void mysize(uchar *ptr, float *d_input_tensor, 
			int channel, 
			int src_h, int src_w, 
			int dst_h, int dst_w, 
			int top, int bottom, int left, int right);
#endif
