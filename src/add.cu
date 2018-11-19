#include "kernels.h"

__global__ void add_x(float* in1, float* in2, float* out)
{
	int idx = threadIdx.x;
	float f1 = in1[idx];
	float f2 = in2[idx];

	out[idx] = f1 + f2;
}