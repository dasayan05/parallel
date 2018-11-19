#include "utils.h"

// standard headers
#include <algorithm>

bool utils::check_shape_eq(const FloatArray& f1, const FloatArray& f2)
{
	auto f1_shape = f1.shape();
	auto f2_shape = f2.shape();

	if ((f1_shape[0] == f2_shape[0]) && (f1_shape[1] == f2_shape[1]) && (f1_shape[2] == f2_shape[2]))
		return true;
	else
		return false;
}