#include "gpuarray.h"

// standard includes
#include <iostream>
#include <iomanip>
#include <exception>
#include <cstdlib>

// public constructor
FloatArray::FloatArray(const shape_t shape, ArrayInit* initializer, Device device, bool pinned)
	: _C(std::get<0>(shape)),
	  _H(std::get<1>(shape)),
	  _W(std::get<2>(shape)),
	  _pinned(pinned)
{
	index_t elems = _C * _H * _W;

	this->_array = allocate_cpu(elems);
	this->_device = Device::CPU;

	for(index_t c = 0; c < _C; ++c)
	{
		for(index_t h = 0; h < _H; ++h)
		{
			for(index_t w = 0; w < _W; ++w)
			{
				index_t _i = c * _H * _W + h * _W + w;
                this->_array[_i] = initializer->operator()(c, h, w);
            }
		}
	}

    if (device == Device::GPU)
        this->gpu();
}

void FloatArray::to(Device device)
{
	int err = 0;

	if ((this->_device == Device::CPU) && (device == Device::GPU))
		err = this->gpu();

	if ((this->_device == Device::GPU) && (device == Device::CPU))
		err = this->cpu();

	if (err != 0)
		std::cerr << "Transfer unsuccessful" << std::endl;
}

bool FloatArray::is_gpu() const
{
	return (_device == Device::GPU) ? true : false;
}

bool FloatArray::is_cpu() const
{
	return (_device == Device::CPU) ? true : false;
}

FloatArray::shape_t FloatArray::shape() const
{
	return shape_t({_C, _H, _W});
}

// destructor for de-allocating memory resources
FloatArray::~FloatArray()
{
	// TODO: is the error checking really necessary ??
	if (this->_device == Device::GPU)
	{
		cudaError_t err = cudaFree(this->_array);

		if (err != cudaSuccess)
			std::cerr << "cudaFree did not return \'cudaSuccess\'" << std::endl;
	}
	else
	{
		cudaError_t err;

		if (!this->_pinned)
			std::free(this->_array);
		else
		{
			err = cudaFreeHost(this->_array);
			if (err != cudaSuccess)
				std::cerr << "cudaFreeHost did not succeed" << std::endl;
		}
	}
}

// display array
void FloatArray::display(int precision)
{
	for(index_t c = 0; c < _C; ++c)
	{
		std::cout << "[Channel:" << c << "]" << '\n';
		for(index_t h = 0; h < _H; ++h)
		{
			if (h == 0)
				std::cout << "[[";
			else
				std::cout << " [";

			for(index_t w = 0; w < _W; ++w)
			{
				index_t _i = c * _H * _W + h * _W + w;
				if (w == _W - 1)
					std::cout << std::setprecision(precision) << this->_array[_i] << "]";
				else
					std::cout << std::setprecision(precision) << this->_array[_i] << ",";
			}

			if (h == _H - 1)
				std::cout << "]\n";
			else
				std::cout << '\n';
		}
		std::cout << '\n';
	}

	std::cout << std::flush;
}