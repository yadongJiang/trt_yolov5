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

__global__ void kernel_resize(float *d_dst, int channel,
							  int src_h, int src_w,
							  int dst_h, int dst_w,
							  int top, int bottom, int left, int right,
							  uchar* d_src)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = y * dst_w + x;

	if (y < top || y >= dst_h - bottom || x < left || x >= dst_w - right)
	{
		d_dst[dst_h * dst_w * 0 + offset] = 114. / 255.;
		d_dst[dst_h * dst_w * 1 + offset] = 114. / 255.;
		d_dst[dst_h * dst_w * 2 + offset] = 114. / 255.;

		return;
	}

	float scale_x = float(src_w) / dst_w;
	float scale_y = float(src_h) / dst_h;

	float src_x = (x + 0.5) * scale_x - 0.5;
	float src_y = (y + 0.5) * scale_y - 0.5;

	int src_x_0 = int(floor(src_x));
	int src_y_0 = int(floor(src_y));
	int src_x_1 = src_x_0 + 1 <= src_w - 1 ? src_x_0 + 1 : src_w - 1; 
	int src_y_1 = src_y_0 + 1 <= src_h - 1 ? src_y_0 + 1 : src_h - 1; 

	for (int c = 0; c < channel; c++)
	{
		uchar v00 = d_src[(src_y_0 * src_w + src_x_0) * channel + c];
		uchar v01 = d_src[(src_y_0 * src_w + src_x_1) * channel + c];
		uchar v10 = d_src[(src_y_1 * src_w + src_x_0) * channel + c];
		uchar v11 = d_src[(src_y_1 * src_w + src_x_1) * channel + c];
		uchar value0 = (src_x_1 - src_x) * v00 + (src_x - src_x_0) * v01;
		uchar value1 = (src_x_1 - src_x) * v10 + (src_x - src_x_0) * v11;

		uchar value = uchar((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1);
		float v = float(value) / 255.;
		d_dst[(2 - c) * dst_h * dst_w + offset] = v;
	}
}

void mysize(uchar* ptr, float* d_input_tensor,
			int channel, 
			int src_h, int src_w,
			int dst_h, int dst_w,
			int top, int bottom, int left, int right)
{
	uchar* d_img;
	cudaMalloc((void**)&d_img, channel * src_h * src_w * sizeof(uchar));
	cudaMemcpy(d_img, ptr, channel * src_h * src_w * sizeof(uchar), cudaMemcpyHostToDevice);
	
	// dst_w与dst_h一定可以被32整除，不需要考虑不能整除的情况
	dim3 grids(dst_w / 32, dst_h / 32);
	dim3 blocks(32, 32);

	kernel_resize << <grids, blocks >> > (d_input_tensor, channel, 
										  src_h, src_w, dst_h, dst_w, 
										  top, bottom, left, right, d_img);
	cudaFree(d_img);
}