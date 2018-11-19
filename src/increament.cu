// in-place increament operation
__global__ void atomic_increament_x(float* d_in)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    i = i % 10;
    atomicAdd(&d_in[i], 1.0f);
}