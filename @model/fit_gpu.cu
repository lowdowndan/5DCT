#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <iostream>
#include "helper_cuda.h"
#include "helper_functions.h"


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
//{ //f;
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

// cuBLAS error checking
#define cublasCheck(ans) { gpuAssert_cublas((ans), __FILE__, __LINE__); }
inline void gpuAssert_cublas(cublasStatus_t code, const char *file, int line)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      mexErrMsgIdAndTxt("MATLAB:cublasError","cuBLAS error: %s \n In file %s at line %d.\n", _cudaGetErrorEnum(code), file, line);
   }
}


void cleanup(){

	cudaThreadExit();
}

// Use the z component of DVFs for linear interpolation of surrogate (track each voxel)
__global__ void deform_surrogate_kernel(cudaTextureObject_t vTex, cudaTextureObject_t fTex, float* hd_registration, float* hd_X, float* hd_X2, int nSlices, int nScans, int nVoxels, int nx, int ny, int nz, int startSlice){

	// Calculate indices
	
	// Linear index -- what voxel am i? 
	int ind = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Index into X matrix (1st column for this voxel)
	int xInd = ind * nScans * 3;

	// Index into z component registration matrix 
	int rInd = (ind * nScans * 3) + (nScans * 2);

	// Bounds check
	if (ind > (nVoxels - 1)){
		return;
	}
	
	// Get the slice number of this thread.  Integer division here is intentional
	int z = (startSlice - 1) + (ind / (nx * ny));
	float z2;

	// Interpolate to get voltage and flow from each scan by applying the z
	// component of the DVF for this scan

	float v;
	float f;
	float dz;

	for(int i = 0; i < nScans; i++){

	// Get deformed z coordinate
	dz = hd_registration[rInd + i];
	z2 = (float)z + dz; 

	// CUDA uses row major ordering, so indexing is transosed from matlab v matrix
	v = tex2D<float>(vTex, (z2 + 0.5f), ((float) i + 0.5f));
	f = tex2D<float>(fTex, (z2 + 0.5f), ((float) i + 0.5f));

	// Write output
	hd_X[xInd + i ] = 1.0f;
	hd_X[xInd + i + (1 * nScans)] = v;
	hd_X[xInd + i + (2 * nScans)] = f;

	hd_X2[xInd + i ] = 1.0f;
	hd_X2[xInd + i + (1 * nScans)] = v;
	hd_X2[xInd + i + (2 * nScans)] = f;

	}
}


// Set thread block size
#define BLOCKWIDTH 512
//#define BLOCKHEIGHT 16 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

// input
#define m_vSlices prhs[0]
#define m_fSlices prhs[1]
#define m_dim prhs[2]
#define m_startSlice prhs[3]
#define m_registration prhs[4]

// output
//#define m_debug plhs[0]
#define m_parameters plhs[0]
#define m_model plhs[1]
//#define m_residual plhs[2]

const static int N_PARAMETERS = 3;
const static int N_DIMS = 3;

// rows and columns

int nScans = (int) mxGetM(m_registration);
int nVoxels = (int) mxGetN(m_registration) / N_DIMS;
int nSlices = (int) mxGetM(m_vSlices);

// slice dimensions
double* hh_dim = (double*) mxGetData(m_dim);
int nx = (int) hh_dim[0];
int ny = (int) hh_dim[1];
int nz = (int) hh_dim[2];

// Get the starting slice of this chunk
double* hh_startSlice = (double*) mxGetData(m_startSlice);
int startSlice = (int) *hh_startSlice;

// cuBLAS initialization
cublasStatus_t cublasStat;

cublasHandle_t handle;
cublasStat = cublasCreate(&handle);
cublasCheck(cublasStat);


// Memory allocation sizes
size_t nBytesRegistration = nScans * nVoxels * N_DIMS * sizeof(float);
size_t nBytesX = nBytesRegistration;
size_t nBytesSurrogate = nScans * nSlices * sizeof(float);

// Host memory for info matrix
int * info = (int *)malloc(nVoxels * sizeof(int));

// Get pointers to host data
float* hh_registration = (float*) mxGetData(m_registration);
float* hh_vSlices = (float*) mxGetData(m_vSlices);
float* hh_fSlices = (float*) mxGetData(m_fSlices);


// Allocate and copy
cudaArray* hda_vSlices;
cudaArray* hda_fSlices;

float* hd_registration;
float* hd_X;
float* hd_X2;

cudaCheck(cudaMalloc((void**)&hd_registration, nBytesRegistration));
cudaCheck(cudaMalloc((void**)&hd_X, nBytesX));
cudaCheck(cudaMalloc((void**)&hd_X2, nBytesX));

cudaCheck(cudaMemcpy(hd_registration,hh_registration, nBytesRegistration, cudaMemcpyHostToDevice));

// Copy v,f slice measurements to textures

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

cudaCheck(cudaMallocArray(&hda_vSlices, &channelDesc,nSlices, nScans));
cudaCheck(cudaMallocArray(&hda_fSlices, &channelDesc,nSlices, nScans));

cudaCheck(cudaMemcpyToArray(hda_vSlices,0,0,hh_vSlices,nBytesSurrogate,cudaMemcpyHostToDevice)); 
cudaCheck(cudaMemcpyToArray(hda_fSlices,0,0,hh_fSlices,nBytesSurrogate,cudaMemcpyHostToDevice));

// Create texture objects

// v
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

cudaTextureObject_t vTex = 0;
resDesc.res.array.array = hda_vSlices;
cudaCreateTextureObject(&vTex, &resDesc, &texDesc, NULL);

// f
resDesc.res.array.array = hda_fSlices;
cudaTextureObject_t fTex = 0;
cudaCreateTextureObject(&fTex, &resDesc, &texDesc, NULL);


const dim3 blockSize(BLOCKWIDTH);
const dim3 gridSize((nVoxels/BLOCKWIDTH) + 1);

//float* hd_buffer;
//int nBytesBuffer = nx * ny * sizeof(float);
//cudaCheck(cudaMalloc((void**)&hd_buffer, nBytesBuffer));

// Deform bellows by z vector of dvf, calculate appropriate v and f for all slices
deform_surrogate_kernel<<<gridSize,blockSize>>>(vTex, fTex, hd_registration, hd_X, hd_X2, nSlices, nScans, nVoxels, nx, ny, nz, startSlice);
cudaCheck(cudaDeviceSynchronize());

//return;
//m_debug = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
//mxSetM(m_debug, nx);
//mxSetN(m_debug, ny);
//mxSetData(m_debug, mxMalloc(nBytesBuffer));
//float * hh_debug = (float*) mxGetData(m_debug);
//cudaCheck(cudaMemcpy(hh_debug,hd_buffer,(nBytesBuffer),cudaMemcpyDeviceToHost));
//return;

// Host pointer arrays to device data
float ** hda_X = (float **)malloc(nVoxels * sizeof(float*));
float ** hda_X2 = (float **)malloc(nVoxels * sizeof(float*));
float ** hda_registration = (float **)malloc(nVoxels * sizeof(float*));

// Set arrays of pointers to submatrices
for (int i = 0; i < nVoxels; i++){
	hda_X[i] = hd_X + (i * nScans * N_DIMS);
	hda_X2[i] = hd_X2 + (i * nScans * N_DIMS);
	hda_registration[i] = hd_registration + (i * nScans * N_DIMS);
}

// Device pointers to device data
float ** dd_X;
float ** dd_X2;
float ** dd_registration;

cudaCheck(cudaMalloc((void**)&dd_X, nVoxels * sizeof(float*)));
cudaCheck(cudaMalloc((void**)&dd_X2, nVoxels * sizeof(float*)));
cudaCheck(cudaMalloc((void**)&dd_registration, nVoxels * sizeof(float*)));

// Copy array of pointers to device
cudaCheck(cudaMemcpy(dd_X, hda_X, nVoxels * sizeof(float*), cudaMemcpyHostToDevice));
cudaCheck(cudaMemcpy(dd_X2, hda_X2, nVoxels * sizeof(float*), cudaMemcpyHostToDevice));
cudaCheck(cudaMemcpy(dd_registration, hda_registration, nVoxels * sizeof(float*), cudaMemcpyHostToDevice));

// Solve
cublasStat = cublasSgelsBatched(handle, CUBLAS_OP_N, (int) nScans, (int) N_PARAMETERS, (int) N_DIMS, dd_X, nScans, dd_registration,nScans,info,NULL,nVoxels);
cublasCheck(cublasStat);


// Allocate host memory for parameters
m_parameters = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetM(m_parameters, N_PARAMETERS);
mxSetN(m_parameters, (N_DIMS * nVoxels));
mxSetData(m_parameters, mxMalloc(sizeof(float) * N_PARAMETERS * N_DIMS * nVoxels));
float * h_parameters = (float*) mxGetData(m_parameters);

// Copy parameters to host
cublasStat = cublasGetMatrix(N_PARAMETERS, (N_DIMS * nVoxels), sizeof(float), hd_registration, nScans, h_parameters, N_PARAMETERS);
cublasCheck(cublasStat);


// Multiply to get fits

// Set constants for matrix Sgemm
float * const alpha = (float *)malloc(sizeof(float));
*alpha = 1.0;

float * const beta = (float *)malloc(sizeof(float));
*beta = 0.0;
	
// Allocate device memory for model fit
float * hd_model;
unsigned int nBytesModel = sizeof(float) * nScans * N_DIMS * nVoxels;
cudaCheck(cudaMalloc((void**) &hd_model, nBytesModel));


// Array of pointers to fit submatrices
float ** hda_model = (float **)malloc(nVoxels * sizeof(float*));
float **  dd_model;

for(int i = 0; i < nVoxels; i++){
	hda_model[i] = hd_model + (i * nScans * N_DIMS);
}


// Copy array of fit pointers to device
cudaCheck(cudaMalloc((void**)&dd_model, nVoxels * sizeof(float*)));
cudaCheck(cudaMemcpy(dd_model,hda_model,nVoxels * sizeof(float*), cudaMemcpyHostToDevice));

// Calcluate model fit
cublasStat = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_T,N_DIMS,nScans,N_DIMS,alpha,(const float **) dd_registration,nScans,(const float **) dd_X2,nScans,beta,dd_model, N_PARAMETERS, nVoxels);
cublasCheck(cublasStat);

// Allocate host memory for fit
m_model = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS,mxREAL);
mxSetM(m_model, N_DIMS);
mxSetN(m_model, (nScans * nVoxels));
mxSetData(m_model, mxMalloc(nBytesModel));
float * h_model = (float*) mxGetData(m_model);

// Copy fit to host
cudaCheck(cudaMemcpy(h_model,hd_model,nBytesModel,cudaMemcpyDeviceToHost));
cublasStat = cublasGetMatrix(N_DIMS, (nScans * nVoxels), sizeof(float), hd_model, N_DIMS, h_model, N_DIMS);
cublasCheck(cublasStat);


// Okay, we're done here.  Free memory, close cublas and exit
free(hda_X);
free(hda_X2);
free(hda_registration);
free(hda_model);
free(info);
free(alpha);
free(beta);

cudaFreeArray(hda_vSlices);
cudaFreeArray(hda_fSlices);

cudaFree(hd_X);
cudaFree(hd_X2);
cudaFree(hd_registration);
cudaFree(hd_model);
cudaFree(dd_X);
cudaFree(dd_X2);
cudaFree(dd_registration);
cudaFree(dd_model);

cudaDestroyTextureObject(vTex);
cudaDestroyTextureObject(fTex);

cublasDestroy(handle);
//free(resDesc);
//free(texDesc);
//free(channelDesc);

//Reset device for profiling
cudaDeviceReset();

mexAtExit(cleanup);
//return;
}

