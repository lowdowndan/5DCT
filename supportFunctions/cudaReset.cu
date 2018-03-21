#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>


// Mandatory global methods for C++ support
extern "C" void __cxa_pure_virtual()
{
    // Do nothing or print an error message.
}
//void *__dso_handle = 0;
extern "C" int __cxa_atexit(void (*destructor) (void *), void *arg, void *dso)
{
    //arg;
    //dso;
    return 0;
}
extern "C" void __cxa_finalize(void *f)
{
    //f;
}

void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, mxArray const *prhs[]){

	cudaDeviceReset();
	return;
}
