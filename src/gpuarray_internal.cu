#include "gpuarray.h"

// standard includes
#include <iostream>
#include <exception>
#include <cstdlib>

// cuda includes
#include <cuda.h>

// allocate array on GPU
FloatArray::type* FloatArray::allocate_gpu(index_t elem_count)
{
	type* ptr; // temporarily hold the pointer to allocated GPU memory

	// cudaMalloc call
	cudaError_t err = cudaMalloc((type**) &ptr, elem_count * sizeof(type));
	if (err != cudaSuccess)
	{
		std::cerr << "cudaMalloc did not return \'cudaSuccess\'" << std::endl;
		return nullptr;
	}
	else
		return ptr;
}

// allocate array on CPU
FloatArray::type* FloatArray::allocate_cpu(index_t elem_count)
{
	type* ptr; // temporarily hold the pointer to allocated CPU memory

	// normal malloc call
	cudaError_t err;
	if (!this->_pinned)
	{
		ptr = (type*) std::malloc(elem_count * sizeof(type));

		if (ptr == nullptr)
			std::cerr << "std::malloc returned nullptr" << std::endl;
	}
	else
	{
		err = cudaMallocHost((void**) &ptr, elem_count * sizeof(type));
		if (err != cudaSuccess)
		{
			std::cerr << "cudaMallocHost did not succeed" << std::endl;
			return nullptr;
		}
	}

	return ptr;
}

// transfer to GPU
int FloatArray::gpu()
{
	const index_t elems = _C * _H * _W;

	type* _d = allocate_gpu(elems);
	if (_d == nullptr)
		return -1;
	cudaError_t err = cudaMemcpy((void*) _d, (void*) this->_array, elems * sizeof(type), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		std::cerr << "cudaMemcpy did not return \'cudaSuccess\'" << std::endl;
		return -1;
	}
	else
	{
		// success
		if (!this->_pinned)
			std::free(this->_array);
		else
			err = cudaFreeHost(this->_array);
			// TODO: an error check needed here

		this->_array = _d;
		this->_device = Device::GPU;

		return 0;
	}
}

// transfer to CPU
int FloatArray::cpu()
{
	const index_t elems = _C * _H * _W;
	
	type* _h = allocate_cpu(elems);
	if (_h == nullptr)
		return -1;

	cudaError_t err = cudaMemcpy((void*) _h, (void*) this->_array, elems * sizeof(type), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cerr << "cudaMemcpy did not return \'cudaSuccess\'" << std::endl;
		return -1;
	}
	else
	{
		// success
		cudaFree(this->_array);
		if (err != cudaSuccess)
			std::cerr << "cudaFree did not return \'cudaSuccess\'" << std::endl;

		this->_array = _h;
		this->_device = Device::CPU;

		return 0;
	}
}