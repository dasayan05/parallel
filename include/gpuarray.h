#ifndef GPUARRAY_H
#define GPUARRAY_H

#include <cstdio>
#include <array>

#include "arrayinit.h"

// a device enum 
enum class Device
{
	CPU, // indicated the array is on CPU memory
	GPU  // indicated the array is on GPU memory
};

/*
 * Represents an array of floats which
 * could potentially reside on GPU memory
 */
class FloatArray
{
	// typedef of the datatype of the array
	using type = float;
	// typdef of the datatype of indexing
	using index_t = std::size_t;
	// handy typedef for dimension sizes
	using shape_t = std::array<index_t, 3>;

private:

	// raw pointer to the GPU array object
	type* _array;
	const index_t _C, _H, _W;
	Device _device;
	bool _pinned;

public:

	shape_t shape() const;

public:	// definitions in 'gpuarray.cu'

	// default c'tor deleted
	FloatArray() = delete;

	// non-default c'tor for creating generalized 3D array
    FloatArray(const shape_t /* shape */,
    		ArrayInit* /* initializer */,
    		Device /* device */ = Device::CPU,
			bool /* pinned */ = false);

	// the d'tor
	~FloatArray();

	// transfer to specific device (host/device)
	void to(Device /* device */);

	// check if array is on GPU/CPU memory
	bool is_gpu() const;
	bool is_cpu() const;

	// display
	void display(int /* precision */ = 2);

private: // definitions in 'gpuarray_internal.cu'

	// cpu to gpu conversion
	int gpu();	int cpu();

	// internal allocation functions
	type* allocate_cpu(index_t);
	type* allocate_gpu(index_t);

public: // kernels
	
	friend int kernel_add(FloatArray&, FloatArray&, FloatArray&);
	friend int kernel_square(FloatArray&, FloatArray&);
	friend int kernel_shift_left(FloatArray&);
	friend int kernel_shift_right(FloatArray&);
	friend int kernel_increament(FloatArray&);
};

#endif
