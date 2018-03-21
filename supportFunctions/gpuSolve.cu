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
{ //f;
}

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

//kernel

// deform bellows
__global__ void deformBellows(cudaTextureObject_t vTex, cudaTextureObject_t fTex, float* hd_Registration, float* hd_X, float* hd_X2, unsigned int nSlices, unsigned int nScans, unsigned int nVoxels, unsigned int nColumns){

	// Calculate indices
	unsigned int ind = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int xInd = ind * nScans * 3;
	unsigned int rInd = (ind * nScans * 3) + (nScans * 2);

	// Bounds check
	if (ind > (nVoxels - 1)){
		return;
	}
	
	// Get the slice number of this thread
	unsigned int z = ind / nColumns;
//	unsigned int z = (unsigned int) floorf((float) ind / (float) nColumns);
	float z2;


	// Interpolate to get voltage and flow from each scan
	float v;
	float f;

	for(int i = 0; i < nScans; i++){

      // Get deformed z coordinate
	z2 = z + hd_Registration[rInd + i];

	v = tex2D<float>(vTex, (z2 - 0.5f), (i + 0.5f));
	f = tex2D<float>(fTex, (z2 - 0.5f), (i + 0.5f));

      // Write output
	hd_X[xInd + i ] = 1;
	hd_X[xInd + i + (1 * nScans)] = v;
	hd_X[xInd + i + (2 * nScans)] = f;

	hd_X2[xInd + i ] = 1;
	hd_X2[xInd + i + (1 * nScans)] = v;
	hd_X2[xInd + i + (2 * nScans)] = f;

	//hd_X2[xInd + i ] = (float) nScans;
	//hd_X2[xInd + i + (1 * nScans)] = (float) nScans;
	//hd_X2[xInd + i + (2 * nScans)] = (float) nScans;
	}

}


// Set thread block size
#define BLOCKWIDTH 32
#define BLOCKHEIGHT 32 

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

// input
#define m_vSlices prhs[0]
#define m_fSlices prhs[1]
#define m_registration prhs[2]

// output
//#define m_parameters plhs[0]
#define m_parameters plhs[0]
#define m_model plhs[1]
#define m_debug plhs[2]
//#define m_residual plhs[1]

//cudaDeviceReset();
const static int N_PARAMETERS = 3;
const static int N_DIMS = 3;

// rows and columns
unsigned int M = mxGetM(m_registration);
unsigned int N = mxGetN(m_registration);
unsigned int nVoxels = N / N_DIMS;
unsigned int nScans = mxGetN(m_vSlices);
unsigned int nSlices = mxGetM(m_vSlices);
unsigned int nColumns = nVoxels / nSlices;

// cuda and cuBLAS initialization
cublasStatus_t cublasStat;

cublasHandle_t handle;
cublasStat = cublasCreate(&handle);
cublasCheck(cublasStat);


// Memory allocation sizes
size_t nBytesRegistration = M * N * sizeof(float);
size_t nBytesX = M * N * sizeof(float);
size_t nBytesSlices = nScans * nSlices * sizeof(float);

// Host memory for info matrix`
int * info = (int *)malloc(nVoxels * sizeof(int));

// Get pointers to host data
float* h_registration = (float*) mxGetData(m_registration);
float* h_vSlices = (float*) mxGetData(m_vSlices);
float* h_fSlices = (float*) mxGetData(m_fSlices);


// Allocate and copy
cudaArray* hd_vSlices;
cudaArray* hd_fSlices;

float* hd_registration;
float* hd_X;
float* hd_X2;

cudaCheck(cudaMalloc((void**)&hd_registration, nBytesRegistration));
cudaCheck(cudaMalloc((void**)&hd_X, nBytesX));
cudaCheck(cudaMalloc((void**)&hd_X2, nBytesX));

cudaCheck(cudaMemcpy(hd_registration,h_registration, nBytesRegistration, cudaMemcpyHostToDevice));

// Copy v,f slice measurements to textures
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

cudaCheck(cudaMallocArray(&hd_vSlices, &channelDesc,nSlices, nScans));
cudaCheck(cudaMallocArray(&hd_fSlices, &channelDesc,nSlices, nScans));

cudaCheck(cudaMemcpyToArray(hd_vSlices,0,0,h_vSlices,nBytesSlices,cudaMemcpyHostToDevice)); 
cudaCheck(cudaMemcpyToArray(hd_fSlices,0,0,h_fSlices,nBytesSlices,cudaMemcpyHostToDevice));

// Create texture objects

// v
cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = hd_vSlices;

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

cudaTextureObject_t vTex = 0;
cudaCreateTextureObject(&vTex, &resDesc, &texDesc, NULL);

// f
resDesc.res.array.array = hd_fSlices;
cudaTextureObject_t fTex = 0;
cudaCreateTextureObject(&fTex, &resDesc, &texDesc, NULL);


const dim3 blockSize(BLOCKWIDTH);
const dim3 gridSize((nVoxels/BLOCKWIDTH) + 1);

// Deform bellows by z vector of dvf, calculate appropriate v and f for all slices
deformBellows<<<gridSize,blockSize>>>(vTex, fTex, hd_registration, hd_X, hd_X2, nSlices, nScans, nVoxels, nColumns);


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
//cublasStat = cublasGetMatrix(N_DIMS, (nScans * nVoxels), sizeof(float), hd_model, N_DIMS, h_model, N_DIMS);
//cublasCheck(cublasStat);


// Okay, we're done here.  Free memory, close cublas and exit
free(hda_X);
free(hda_X2);
free(hda_registration);
free(hda_model);
free(info);
free(alpha);
free(beta);

cudaFreeArray(hd_vSlices);
cudaFreeArray(hd_fSlices);

cudaFree(hd_X);
cudaFree(hd_X2);
cudaFree(hd_registration);
cudaFree(hd_model);
cudaFree(hd_vSlices);
cudaFree(hd_fSlices);
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
return;
}

