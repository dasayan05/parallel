// element-wise squaring kernel
__global__
void square_x(float* d_in, float* d_out)
{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f;
}