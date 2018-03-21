#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

// map DVFs to float4 rather than 3 floats
// map DVFs to float4 rather than 3 floats
// map DVFs to float4 rather than 3 floats

// Mandatory global methods for C++ support
//extern "C" void __cxa_pure_virtual()
//{
//    // Do nothing or print an error message.
//}
////void *__dso_handle = 0;
//extern "C" int __cxa_atexit(void (*destructor) (void *), void *arg, void *dso)
//{
//    //arg;
//    //dso;
//    return 0;
//}
//extern "C" void __cxa_finalize(void *f)
//{
//    //f;
//}

// cuda error checking
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      mexErrMsgIdAndTxt("MATLAB:cudaError","Error: %s \n In file %s at line %d.\n", cudaGetErrorString(code), file, line);
   }
}

// mex clean up (will crash periodically if not run)

void cleanup(){

	cudaThreadExit();
}

// getInverse
__global__ void getInverse(cudaTextureObject_t XTex, cudaTextureObject_t YTex, cudaTextureObject_t ZTex, float* hd_iX, float* hd_iY, float* hd_iZ, size_t dimX, size_t dimY, size_t dimZ, size_t numVoxels, size_t numIterations){


	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int z = (blockIdx.z * blockDim.z) + threadIdx.z;

	// Bounds check
	if ( x > (dimX - 1) || y > (dimY - 1) || z > (dimZ - 1)){
		return;
	}

	// Get linear index 
	//unsigned int ind = x + (y * dimY) + (z * dimX * dimY);
	
	unsigned int ind = x + (y * dimX) + (z * dimX * dimY);
	// Get x index of this thread
//	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//	unsigned int z = (blockIdx.z * blockDim.z) + threadIdx.z;
//
//
//	// Get linear index 
//
//	// Bounds check
//	if ( ind > (numVoxels - 1)){
//	return;
//	}

	float ix = 0;
	float iy = 0;
	float iz = 0;

	for(int i = 1; i < numIterations; i++){

	//ix = tex3D<float>(XTex, (y + iy + 0.5), (x + ix + 0.5), (z + iz + 0.5)) * -1;
	//iy = tex3D<float>(YTex, (y + iy + 0.5), (x + ix + 0.5), (z + iz + 0.5)) * -1;
	//iz = tex3D<float>(ZTex, (y + iy + 0.5), (x + ix + 0.5), (z + iz + 0.5)) * -1;

	//ix = tex3D<float>(XTex, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * -1;
	//iy = tex3D<float>(YTex, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * -1;
	//iz = tex3D<float>(ZTex, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f)) * -1;

	ix = tex3D<float>(XTex, (x + ix + 0.5f), (y + 0.5f), (z + 0.5f)) * -1;
	iy = tex3D<float>(YTex, (x + 0.5f), (y + iy + 0.5f), (z + 0.5f)) * -1;
	iz = tex3D<float>(ZTex, (x + 0.5f), (y + 0.5f), (z + iz + 0.5f)) * -1;
	}

	// Write output
	hd_iX[ind] = ix;
	hd_iY[ind] = iy;
	hd_iZ[ind] = iz;

}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]){

// Macros for input/output
#define dfX prhs[0]
#define dfY prhs[1]
#define dfZ prhs[2]
#define numIt prhs[3]

#define iX plhs[0]
#define iY plhs[1]
#define iZ plhs[2]

// Set thread block size
#define BLOCKWIDTH 8
#define BLOCKHEIGHT 8 
#define BLOCKDEPTH 4

// Initialize GPU
//mxInitGPU();

// Check number of arguments 
if (nrhs != 4 || nlhs > 3){
	mexErrMsgIdAndTxt("MATLAB:badInput","Usage: [iX, iY, iZ] = invertDeformationField(dfX,dfY,dfZ, numIterations)\n");
}

// Get dimensions of image volume
const mwSize *dfDims = mxGetDimensions(dfX);

mxClassID classCheck;
const mwSize *dimCheck;
mwSize sizeCheck;

// Check that arguments are correct class and dimensionality
for (int i=0; i < (nrhs - 1); i++){

	classCheck = mxGetClassID(prhs[i]);
	if (classCheck != mxSINGLE_CLASS){
	mexErrMsgIdAndTxt("MATLAB:badInput","dfX, dfY, and dfZ must be of data type single.\n");
	}

	sizeCheck = mxGetNumberOfDimensions(prhs[i]);
	if (sizeCheck != 3){
	mexErrMsgIdAndTxt("MATLAB:badInput","dfX, dfY, and dfZ must be 3D matrices.\n");
	}

	if (i > 1){
	dimCheck = mxGetDimensions(prhs[i]);

	for (int j = 0; j < 3; j++){

		if(dfDims[j] != dimCheck[j]){
		mexErrMsgIdAndTxt("MATLAB:badInput","dfX, dfY, and dfZ must be the same size.\n");
		}
		}
	}


}


// Set dimensions of deformation fields
size_t dimX = dfDims[0];
size_t dimY = dfDims[1];
size_t dimZ = dfDims[2];

size_t numVoxels = mxGetNumberOfElements(dfX);	

// Calculate memory allocation sizes
size_t numBytes = numVoxels * sizeof(float);

// Get pointer to image data and interpolation grids
float* hh_dfX = (float*) mxGetData(dfX);
float* hh_dfY = (float*) mxGetData(dfY);
float* hh_dfZ = (float*) mxGetData(dfZ);

// Get number of iterations
double* hh_numIterations = (double*) mxGetData(numIt);
size_t numIterations = *hh_numIterations;

// Allocate GPU memory for inverse deformation
float* hd_iX;
float* hd_iY;
float* hd_iZ;

cudaCheck(cudaMalloc((void**)&hd_iX, numBytes));
cudaCheck(cudaMalloc((void**)&hd_iY, numBytes));
cudaCheck(cudaMalloc((void**)&hd_iZ, numBytes));

// Allocate CUDA arrays in device memory for deformation components
cudaArray* hd_dfX;
cudaArray* hd_dfY;
cudaArray* hd_dfZ;

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
struct cudaExtent extent = make_cudaExtent(dimX, dimY, dimZ);

cudaCheck(cudaMalloc3DArray(&hd_dfX, &channelDesc, extent));
cudaCheck(cudaMalloc3DArray(&hd_dfY, &channelDesc, extent));
cudaCheck(cudaMalloc3DArray(&hd_dfZ, &channelDesc, extent));

// Get pitched pointers to deformation components in host memory
cudaPitchedPtr hh_dfXp = make_cudaPitchedPtr((void*) hh_dfX, dimX * sizeof(float), dimX, dimY);
cudaPitchedPtr hh_dfYp = make_cudaPitchedPtr((void*) hh_dfY, dimX * sizeof(float), dimX, dimY);
cudaPitchedPtr hh_dfZp = make_cudaPitchedPtr((void*) hh_dfZ, dimX * sizeof(float), dimX, dimY);


// Copy deformation components to device
cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = hh_dfXp;
copyParams.dstArray = hd_dfX;
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;
cudaCheck(cudaMemcpy3D(&copyParams));

copyParams.srcPtr = hh_dfYp;
copyParams.dstArray = hd_dfY;
cudaCheck(cudaMemcpy3D(&copyParams));

copyParams.srcPtr = hh_dfZp;
copyParams.dstArray = hd_dfZ;
cudaCheck(cudaMemcpy3D(&copyParams));

// Create texture objects

// X
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = hd_dfX;

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

cudaTextureObject_t XTex = 0;
cudaCreateTextureObject(&XTex, &resDesc, &texDesc, NULL);

// Y
resDesc.res.array.array = hd_dfY;
cudaTextureObject_t YTex = 0;
cudaCreateTextureObject(&YTex, &resDesc, &texDesc, NULL);

// Z
resDesc.res.array.array = hd_dfZ;
cudaTextureObject_t ZTex = 0;
cudaCreateTextureObject(&ZTex, &resDesc, &texDesc, NULL);


// Get inverse
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT,BLOCKDEPTH);
const dim3 gridSize(dimX/BLOCKWIDTH + 1, dimY/BLOCKWIDTH + 1, dimZ/BLOCKDEPTH + 1);

getInverse<<<gridSize,blockSize>>>(XTex, YTex, ZTex, hd_iX, hd_iY, hd_iZ, dimX, dimY, dimZ, numVoxels, numIterations);
//cudaCheck(cudaPeekAtLastError());
cudaCheck(cudaDeviceSynchronize());

// Allocate host memory for inverse deformation 
iX = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetDimensions(iX, dfDims, sizeCheck);
mxSetData(iX, mxMalloc(numBytes));
float * hh_iX = (float*) mxGetData(iX);

iY = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetDimensions(iY, dfDims, sizeCheck);
mxSetData(iY, mxMalloc(numBytes));
float * hh_iY = (float*) mxGetData(iY);

iZ = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetDimensions(iZ, dfDims, sizeCheck);
mxSetData(iZ, mxMalloc(numBytes));
float * hh_iZ = (float*) mxGetData(iZ);


// Copy inverse deformation to host
cudaCheck(cudaMemcpy(hh_iX, hd_iX, numBytes, cudaMemcpyDeviceToHost));
cudaCheck(cudaMemcpy(hh_iY, hd_iY, numBytes, cudaMemcpyDeviceToHost));
cudaCheck(cudaMemcpy(hh_iZ, hd_iZ, numBytes, cudaMemcpyDeviceToHost));

// Free allocated memory
cudaFree(hd_iX);
cudaFree(hd_iY);
cudaFree(hd_iZ);

cudaDestroyTextureObject(XTex);
cudaDestroyTextureObject(YTex);
cudaDestroyTextureObject(ZTex);

cudaFreeArray(hd_dfX);
cudaFreeArray(hd_dfY);
cudaFreeArray(hd_dfZ);

//Reset device for profiling
cudaDeviceReset();

mexAtExit(cleanup);
return;
}
