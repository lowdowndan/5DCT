#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
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

// Macros for input/output
#define modelParams plhs[0]
#define modelErrors plhs[1]

#define X prhs[0]
#define dvf prhs[1]

// cuda and cuBLAS setup
cudaError_t cudaStat;    
cublasStatus_t stat;
cublasHandle_t handle;

stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        mexPrintf ("CUBLAS initialization failed.\n");
        return;
    }

// Get dimensions and number of system to solve
unsigned int M = mxGetM(X);
unsigned int numParams = 3;
unsigned int N = 3;
unsigned int numVoxels = mxGetN(X) / N;
unsigned int numDims = 3;

// Memory allocation sizes
size_t numBytesX = M * N * sizeof(float);
size_t numBytesDvf = M * numDims * sizeof(float);

// Host memory for info matrix`
int * info = (int *)malloc(numVoxels * sizeof(int));

// Get pointers to v,f matrices and dvf matrices from matlab
float * h_X = (float*) mxGetData(X);
float * h_dvf = (float*) mxGetData(dvf);

// Host pointer arrays to device data
float * hd_XPtr;
float * hd_X2Ptr;
float * hd_dvfPtr;

float ** hd_X = (float **)malloc(numVoxels * sizeof(float*));
float ** hd_X2 = (float **)malloc(numVoxels * sizeof(float*));
float ** hd_dvf = (float **)malloc(numVoxels * sizeof(float*));

// Device pointers to device data
float ** d_X;
float ** d_X2;
float ** d_dvf;

// Allocate GPU memory for v,f and DVF

cudaStat = cudaMalloc((void**)&hd_XPtr, numBytesX * numVoxels);
cudaStat = cudaMalloc((void**)&hd_X2Ptr, numBytesX * numVoxels);

// Check if allocation was successful
if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for v,f matrices failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        return;
}

cudaStat = cudaMalloc((void**)&hd_dvfPtr, numBytesDvf * numVoxels);

// Check if allocation was successful
if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for DVF matrices failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        return;
}

// Copy data to device
stat = cublasSetMatrix(M, (N * numVoxels), sizeof(float), h_X, M, hd_XPtr, M);

if (stat != CUBLAS_STATUS_SUCCESS) {
        mexPrintf("V,F data transfer to GPU failed.\n");
        cudaFree(hd_X[0]);
        cublasDestroy(handle);
        return;
}

stat = cublasSetMatrix(M, (N * numVoxels), sizeof(float), h_X, M, hd_X2Ptr, M);

if (stat != CUBLAS_STATUS_SUCCESS) {
        mexPrintf("V,F data transfer to GPU failed.\n");
        cudaFree(hd_X[0]);
        cublasDestroy(handle);
        return;
}
	
stat = cublasSetMatrix(M, (numDims * numVoxels), sizeof(float), h_dvf, M, hd_dvfPtr, M);

if (stat != CUBLAS_STATUS_SUCCESS) {
	mexPrintf("DVF Data transfer to GPU failed.\n");
        cudaFree(hd_dvf[0]);
        cublasDestroy(handle);
        return;
}


// Set arrays of pointers to submatrices
for (int i = 0; i < numVoxels; i++){
	hd_X[i] = hd_XPtr + (i * M * N);
	hd_X2[i] = hd_X2Ptr + (i * M * N);
	hd_dvf[i] = hd_dvfPtr + (i * M * numDims);
}

// Allocate memory for array of device pointers on device
cudaStat = cudaMalloc((void**)&d_X, numVoxels * sizeof(float*));
cudaStat = cudaMalloc((void**)&d_X2, numVoxels * sizeof(float*));
cudaStat = cudaMalloc((void**)&d_dvf, numVoxels * sizeof(float*));

if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for array of device pointers failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
}

// Copy array of pointers to device
cudaStat = cudaMemcpy(d_X,hd_X, numVoxels * sizeof(float*),cudaMemcpyHostToDevice);
cudaStat = cudaMemcpy(d_X2,hd_X2, numVoxels * sizeof(float*),cudaMemcpyHostToDevice);
cudaStat = cudaMemcpy(d_dvf,hd_dvf, numVoxels * sizeof(float*),cudaMemcpyHostToDevice);
	
if (cudaStat != cudaSuccess) {
       	mexPrintf("Copy of pointer arrays to device failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
}



// Fit model parameters
stat = cublasSgelsBatched(handle, CUBLAS_OP_N, M, N,numDims,d_X,M,d_dvf,M,info,NULL,numVoxels);
if (stat != CUBLAS_STATUS_SUCCESS) {
        mexPrintf("Batched solver failed.  Error code %d.\n", stat);
        cublasDestroy(handle);
        return;
}

// Allocate host memory for parameters

modelParams = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetM(modelParams, numParams);
mxSetN(modelParams, (numDims * numVoxels));
mxSetData(modelParams, mxMalloc(sizeof(float) * numParams * numDims * numVoxels));
float * h_params = (float*) mxGetData(modelParams);

// Copy parameters to host
stat = cublasGetMatrix(numParams, (numDims * numVoxels), sizeof(float), hd_dvfPtr, M, h_params, numParams);

if (stat != CUBLAS_STATUS_SUCCESS) {
mexPrintf("Transfer of model parameters to host failed.  Error code %d.\n", stat);
cublasDestroy(handle);
return;
}

// Multiply to get fits


// Set constants for matrix Sgemm
float * const alpha = (float *)malloc(sizeof(float));
*alpha = 1.0;

float * const beta = (float *)malloc(sizeof(float));
*beta = 0.0;
	
// Allocate device memory for model fit
float * hd_fitPtr;
int numBytesError = sizeof(float) * M * numDims;
cudaStat = cudaMalloc((void**) &hd_fitPtr, numBytesError * numVoxels);

// Check if allocation was successful
if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for model fit failed %d.");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        return;
}

// Array of pointers to fit submatrices

float ** hd_fit = (float **)malloc(numVoxels * sizeof(float*));
float **  d_fit;

for(int i = 0; i < numVoxels; i++){
	hd_fit[i] = hd_fitPtr + (i * M * numDims);
}

// Check if allocation was successful
if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for model fit failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
       	return;
}

// Copy array of fit pointers to device
cudaStat = cudaMalloc((void**)&d_fit, numVoxels * sizeof(float*));
cudaStat = cudaMemcpy(d_fit,hd_fit,numVoxels * sizeof(float*), cudaMemcpyHostToDevice);

// Calcluate model fit
stat = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,numDims,M,numDims,alpha,(const float **) d_dvf,M,(const float **) d_X2,M,beta,d_fit, numParams, numVoxels);

if (stat != CUBLAS_STATUS_SUCCESS) {
        mexPrintf("Model fit calculation failed.  Error code %d.\n", stat);
        cublasDestroy(handle);
	   return;
}

// Allocate host memory for fit

modelErrors = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetM(modelErrors, numDims);
mxSetN(modelErrors, (M * numVoxels));
mxSetData(modelErrors, mxMalloc(sizeof(float) * M * numDims * numVoxels));
float * h_fit = (float*) mxGetData(modelErrors);

// Copy fit to host
stat = cublasGetMatrix(numDims, (M * numVoxels), sizeof(float), hd_fitPtr, numDims, h_fit, numDims);

if (stat != CUBLAS_STATUS_SUCCESS) {
mexPrintf("Transfer of model fits to host failed.  Error code %d.\n", stat);
cublasDestroy(handle);
return;
}


// Okay, we're done here.  Free memory, close cublas and exit
cudaFree(hd_fitPtr);
cudaFree(hd_XPtr);
cudaFree(hd_X2Ptr);
cudaFree(hd_dvfPtr);
cudaFree(d_fit);
cudaFree(d_X);
cudaFree(d_X2);
cudaFree(d_dvf);

free(hd_X);
free(hd_X2);
free(hd_dvf);
free(hd_fit);
free(alpha);
free(beta);
free(info);

cublasDestroy(handle);

//Reset device for profiling
return;
}
