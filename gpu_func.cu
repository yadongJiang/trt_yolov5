#include "gpu_func.cuh"

__constant__  float con_anchors[18] = { 10, 13 , 16, 30, 33, 23, 
										30, 61, 62, 45, 59, 119, 
										116, 90, 156, 198, 373, 326 };

__device__ void dev_sigmoid(float* ptr, int len)
{
	for (int i = 0; i < len; i++)
		ptr[i] = 1.0 / (exp(-ptr[i]) + 1.0);
}

__global__ void process_kernel(float* dev_ptr, int height, int width, int no, int total_pix) // , float* anchor
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = offset * no;
	if (offset >= total_pix)
		return;

	dev_sigmoid(dev_ptr + pos, no);
	
	int stride, c, row, col, anchor_w, anchor_h;
	int element_size8 = height * width / 64;
	int element_size16 = height * width / 256;
	int element_size32 = height * width / 1024;
	if (offset < (3 * element_size8))
	{
		stride = 8;
		c = offset / element_size8;
		row = (offset - c * element_size8) / (width / 8);
		col = (offset - c * element_size8) % (width / 8);
		anchor_w = con_anchors[c * 2 + 0];
		anchor_h = con_anchors[c * 2 + 1];

		dev_ptr[pos + 0] = (dev_ptr[pos + 0] * 2.0 - 0.5 + col) * stride;
		dev_ptr[pos + 1] = (dev_ptr[pos + 1] * 2.0 - 0.5 + row) * stride;
		dev_ptr[pos + 2] = powf((dev_ptr[pos + 2] * 2.0), 2) * anchor_w;
		dev_ptr[pos + 3] = powf((dev_ptr[pos + 3] * 2.0), 2) * anchor_h;
	}
	else if (offset >= (3 * element_size8) && offset < (3 * element_size8 + 3 * element_size16))
	{
		stride = 16;
		c = (offset - (3 * element_size8)) / element_size16;
		row = (offset - 3 * element_size8 - c * element_size16) / (width / 16);
		col = (offset - 3 * element_size8 - c * element_size16) % (width / 16);
		anchor_w = con_anchors[6 + c * 2 + 0];
		anchor_h = con_anchors[6 + c * 2 + 1];

		dev_ptr[pos + 0] = (dev_ptr[pos + 0] * 2.0 - 0.5 + col) * stride;
		dev_ptr[pos + 1] = (dev_ptr[pos + 1] * 2.0 - 0.5 + row) * stride;
		dev_ptr[pos + 2] = powf((dev_ptr[pos + 2] * 2.0), 2) * anchor_w;
		dev_ptr[pos + 3] = powf((dev_ptr[pos + 3] * 2.0), 2) * anchor_h;
	}
	else
	{
		stride = 32;
		c = (offset - (3 * element_size8 + 3 * element_size16)) / (element_size32);
		row = (offset - (3 * element_size8 + 3 * element_size16) - c * element_size32) / (width / 32);
		col = (offset - (3 * element_size8 + 3 * element_size16) - c * element_size32) % (width / 32);
		anchor_w = con_anchors[12 + c * 2 + 0];
		anchor_h = con_anchors[12 + c * 2 + 1];

		dev_ptr[pos + 0] = (dev_ptr[pos + 0] * 2.0 - 0.5 + col) * stride;
		dev_ptr[pos + 1] = (dev_ptr[pos + 1] * 2.0 - 0.5 + row) * stride;
		dev_ptr[pos + 2] = powf((dev_ptr[pos + 2] * 2.0), 2) * anchor_w;
		dev_ptr[pos + 3] = powf((dev_ptr[pos + 3] * 2.0), 2) * anchor_h;
	}
}

void postprocess(float* dev_ptr, int height, int width, int no, int counts)
{
	// float anchors[18] = { 10, 13 , 16, 30, 33, 23, 
	// 					  30, 61, 62, 45, 59, 119, 
	// 					  116, 90, 156, 198, 373, 326 };
	// cudaMemcpyToSymbol(con_anchors, anchors, 18 * sizeof(float));

	dim3 grids(std::ceil(float(counts/no) / 32));
	dim3 blocks(32);
	process_kernel << <grids, blocks >> > (dev_ptr, height, width, no, counts / no); 
}