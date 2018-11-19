// left shift operation
__global__ void leftshift_x(float* f)
{
	int x = threadIdx.x;
	if (x < blockDim.x - 1)
	{
		float temp = f[x + 1];
		__syncthreads();
		f[x] = temp;
	}
}

// right shift operation
__global__ void rightshift_x(float* f)
{
	int x = threadIdx.x;
	if (x < blockDim.x - 1)
	{
		float temp = f[x];
		__syncthreads();
		f[x + 1] = temp;
	}
}