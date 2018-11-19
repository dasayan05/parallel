#include "gpuarray.h"

// standard header
#include <ctime>

int main(int , char** )
{
    FloatArray arr({1,1,10}, new ZeroArrayInit(), Device::CPU);
    arr.display();

    arr.to(Device::GPU);

    kernel_increament(arr);

    arr.to(Device::CPU);
    arr.display();
    return 0;
}
