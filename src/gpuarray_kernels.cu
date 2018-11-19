#include "gpuarray.h"
#include "utils.h"
#include "kernels.h"

// standard headers
#include <iostream>

int kernel_add(FloatArray& f1, FloatArray& f2, FloatArray& f)
{
	if ( utils::check_shape_eq(f1, f2) && utils::check_shape_eq(f2, f) )
		std::cout << "shape compatible" << std::endl;

	const auto shape = f1.shape();

	// can access provates of 'FloatArray's

	add_x<<<dim3(1,1,1), dim3(shape[0]*shape[1]*shape[2],1,1)>>>(f1._array, f2._array, f._array);

	return 0;
}

int kernel_square(FloatArray& f1, FloatArray& f2)
{
	if ( utils::check_shape_eq(f1, f2) )
		std::cout << "shape compatible" << std::endl;

	const auto shape = f1.shape();

	square_x<<<dim3(1,1,1),dim3(shape[0]*shape[1]*shape[2],1,1)>>>(f1._array, f2._array);

	return 0;
}

int kernel_shift_left(FloatArray& f)
{
	const auto shape = f.shape();

	leftshift_x<<<1, dim3(shape[0]*shape[1]*shape[2], 1, 1)>>>(f._array);

	return 0;
}

int kernel_shift_right(FloatArray& f)
{
	const auto shape = f.shape();

	rightshift_x<<<1, dim3(shape[0]*shape[1]*shape[2], 1, 1)>>>(f._array);

	return 0;
}

int kernel_increament(FloatArray& f)
{
	const auto shape = f.shape();

	atomic_increament_x<<<10,10>>>(f._array);

	return 0;
}