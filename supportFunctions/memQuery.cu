#include "mex.h"
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
	
// Allocate return variables
plhs[0] = mxCreateNumericMatrix(1, 1, mxINT64_CLASS,mxREAL);
plhs[1] = mxCreateNumericMatrix(1, 1, mxINT64_CLASS,mxREAL);

// Get pointers
size_t* freeMem = (size_t*) mxGetData(plhs[0]);
size_t* totalMem = (size_t*) mxGetData(plhs[1]);

// Get amount of free device memory
cudaError_t cudaStat;    
cudaStat = cudaMemGetInfo(freeMem, totalMem);

if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory query failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        return;
}

return;
}
