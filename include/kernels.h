// cuda kernel headers

__global__ void add_x(float*, float*, float*);
__global__ void square_x(float* ,float*);
__global__ void leftshift_x(float*);
__global__ void rightshift_x(float*);
__global__ void atomic_increament_x(float*);