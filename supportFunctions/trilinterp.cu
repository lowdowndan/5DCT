//TODO:
// eliminate redundant data transfer
// - only send needed indices of X, Y and Z to gpu
// chunking for compatability with gpus that have <12gb ram
// check for memory leaks
// optimize gird/block size
// handle mutiple voxels with each thread?
// pinned memory?

// allocate big chunk of pinned host memory for interpolation grid
// use host memcpy to copy grouped coordinates (x1 y1 z1 x2 y2 z2) for faster read later



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



// Interpolation kernel

__global__ void interp3D(cudaTextureObject_t imgTex, float* hd_out, float* hd_X, float* hd_Y, float* hd_Z, size_t outDimX, size_t outDimY, size_t outDimZ, int numVoxelsOut){


	// Get x index of this thread
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int z = (blockIdx.z * blockDim.z) + threadIdx.z;


	// Get linear index into interpolation grid
	unsigned int ind = y + (x * outDimY) + (z * outDimX * outDimY);

	// Bounds check
	if ( x > (outDimX - 1) || y > (outDimY - 1) || z > (outDimZ - 1)){
		return;
	}

	// Interpolate each column
	

	// Get interpolation coordinates
	float xi = hd_X[ind];
	float yi = hd_Y[ind];
	float zi = hd_Z[ind];
	
	// Interpolate and write value to output
	hd_out[ind] = tex3D<float>(imgTex, xi-0.5f, yi-0.5f, zi-0.5f);
	

}



void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[]){


// Macros for input/output
#define img prhs[0]

// Y and X are swapped to correspond with the output from meshgrid
#define Y prhs[1]
#define X prhs[2]
#define Z prhs[3]

#define out plhs[0]

// Set thread block size
#define BLOCKWIDTH 8
#define BLOCKHEIGHT 8 
#define BLOCKDEPTH 2


// Check number of arguments 
if (nrhs != 4 || nlhs > 1){
	mexErrMsgIdAndTxt("MATLAB:badInput","Usage: imageOut = trilinterp(imageIn, Xi, Yi, Zi)\n");
}

// Get dimensions of image volume
const mwSize *imgDims = mxGetDimensions(img);
const mwSize *outDims = mxGetDimensions(X);

mxClassID classCheck;
const mwSize *dimCheck;
mwSize sizeCheck;

// Check that arguments are correct class and dimensionality
for (int i=0; i < nrhs; i++){

	classCheck = mxGetClassID(prhs[i]);
	if (classCheck != mxSINGLE_CLASS){
	mexErrMsgIdAndTxt("MATLAB:badInput","imageIn, Xi, Yi, and Zi must be of data type single.\n");
	}

	sizeCheck = mxGetNumberOfDimensions(prhs[i]);
	if (sizeCheck != 3){
	mexErrMsgIdAndTxt("MATLAB:badInput","imageIn, Xi, Yi, and Zi must be 3D matrices.\n");
	}

	if (i > 1){
	dimCheck = mxGetDimensions(prhs[i]);

	for (int j = 0; j < 3; j++){

		if(outDims[j] != dimCheck[j]){
		mexErrMsgIdAndTxt("MATLAB:badInput","Xi, Yi, and Zi must be the same size.\n");
		}
		}
	}


}

// cuda error checking setup

cudaError_t cudaStat;    

// Set dimensions of input and output images
size_t imgDimX = imgDims[0];
size_t imgDimY = imgDims[1];
size_t imgDimZ = imgDims[2];

size_t outDimX = outDims[0];
size_t outDimY = outDims[1];
size_t outDimZ = outDims[2];

size_t numVoxelsImg = mxGetNumberOfElements(img);	
size_t numVoxelsOut = mxGetNumberOfElements(X);	

// Calculate memory allocation sizes
size_t numBytesOut = numVoxelsOut * sizeof(float);

// Get pointer to image data and interpolation grids
float* hh_img = (float*) mxGetData(img);
float* hh_X = (float*) mxGetData(X);
float* hh_Y = (float*) mxGetData(Y);
float* hh_Z = (float*) mxGetData(Z);

// Allocate GPU memory for image volume, interpolation grids and output
float* hd_X;
float* hd_Y;
float* hd_Z;
float* hd_out;


cudaStat = cudaMalloc((void**)&hd_X, numBytesOut);
if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for interpolation grid failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}

cudaStat = cudaMalloc((void**)&hd_Y, numBytesOut);
if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for interpolation grid failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}
cudaStat = cudaMalloc((void**)&hd_Z, numBytesOut);
if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for interpolation grid failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}

cudaStat = cudaMalloc((void**)&hd_out, numBytesOut);
if (cudaStat != cudaSuccess) {
	mexPrintf("Device memory allocation for output image failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}


// Copy grids

cudaStream_t streamX;
cudaStream_t streamY;
cudaStream_t streamZ;
cudaStream_t streamImg;

cudaStreamCreate(&streamX);
cudaStreamCreate(&streamY);
cudaStreamCreate(&streamZ);
cudaStreamCreate(&streamImg);

cudaStat = cudaMemcpy(hd_X, hh_X, numBytesOut, cudaMemcpyHostToDevice);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy interpolation grid to device.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}

cudaStat = cudaMemcpy(hd_Y, hh_Y, numBytesOut, cudaMemcpyHostToDevice);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy interpolation grid to device.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}

cudaStat = cudaMemcpy(hd_Z, hh_Z, numBytesOut, cudaMemcpyHostToDevice);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy interpolation grid to device.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}


// Allocate CUDA array in device memory
cudaArray* hd_imgBuffer;

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
struct cudaExtent extent = make_cudaExtent(imgDimX, imgDimY, imgDimZ);

cudaStat = cudaMalloc3DArray(&hd_imgBuffer, &channelDesc, extent);
if (cudaStat != cudaSuccess) {
	mexPrintf("Texture memory allocation for filtered image failed.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}

// Get pitched pointer to image in host memory
cudaPitchedPtr hh_imgInPitched = make_cudaPitchedPtr((void*) hh_img, imgDimX * sizeof(float), imgDimX, imgDimY);

// Copy filtered image to texture memory
cudaMemcpy3DParms copyParams = {0};
copyParams.srcPtr = hh_imgInPitched;
copyParams.dstArray = hd_imgBuffer;
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;

cudaStat = cudaMemcpy3D(&copyParams);
if (cudaStat != cudaSuccess) {
	mexPrintf("Failed to copy filtered image to texture memory.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}



// Create texture object
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = hd_imgBuffer;


cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

cudaTextureObject_t imgTex = 0;
cudaCreateTextureObject(&imgTex, &resDesc, &texDesc, NULL);

// Launch interpolation kernels for each axial slice
const dim3 blockSize(BLOCKWIDTH,BLOCKHEIGHT,BLOCKDEPTH);
const dim3 gridSize(outDimX/BLOCKWIDTH + 1, outDimY/BLOCKWIDTH + 1, outDimZ/BLOCKDEPTH + 1);

interp3D<<<gridSize,blockSize>>>(imgTex, hd_out, hd_X, hd_Y, hd_Z, outDimX, outDimY, outDimZ, numVoxelsOut);



cudaStat = cudaPeekAtLastError();
if (cudaStat != cudaSuccess) {
	mexPrintf("Interpolation kernel launch failure.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}
cudaStat = cudaDeviceSynchronize();
if (cudaStat != cudaSuccess) {
	mexPrintf("Error in interpolation kernel.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}


// Allocate host memory for output image
out = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetDimensions(out, outDims, sizeCheck);
mxSetData(out, mxMalloc(numBytesOut));
float * hh_out = (float*) mxGetData(out);

// Copy output image to host
cudaStat = cudaMemcpy(hh_out, hd_out, numBytesOut, cudaMemcpyDeviceToHost);
if (cudaStat != cudaSuccess) {
	mexPrintf("Error copying output image to host.\n");
	mexPrintf("Error code %d: %s\n",cudaStat,cudaGetErrorString(cudaStat));
        mexErrMsgIdAndTxt("MATLAB:cudaFail","Interpolation failed.\n");
}

// Free allocated memory
cudaFree(hd_X);
cudaFree(hd_Y);
cudaFree(hd_Z);
cudaFree(hd_out);


cudaDestroyTextureObject(imgTex);
cudaFreeArray(hd_imgBuffer);
//Reset device for profiling
cudaDeviceReset();
return;
}
