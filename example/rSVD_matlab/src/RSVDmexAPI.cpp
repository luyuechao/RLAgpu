#include "gpuErrorCheck.h"
#include "rsvd.h"
#include <mex.h>
#include <stdio.h>


// Matlab Host function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // input from matlab
    double *ptr0, *ptr1, *ptr2;
    
    if (nrhs != 3){
        mexErrMsgIdAndTxt("MyToolbox:poseletDistance:nrhs",
                          "3 inputs required.");
    }
    if (nlhs != 3){
        mexErrMsgIdAndTxt("MyToolbox:poseletDistance:nlhs",
                          "3 output required.");
    }
    
    // Create pointers to the input data
    ptr0 = (double*)mxGetData(prhs[0]); //the input matrix
    ptr1 = (double*)mxGetData(prhs[1]);
    ptr2 = (double*)mxGetData(prhs[2]);
    
    // Get dimensions of input data
    int m = (int)(mxGetDimensions(prhs[0])[0]);
    int n = (int)(mxGetDimensions(prhs[0])[1]);
    
    double *h_A;
    CHECK_CUDA( cudaMallocHost((void**)&h_A, m * n * sizeof(double)) );
    CHECK_CUDA( cudaMemcpyAsync(h_A, ptr0, m * n * sizeof(double), cudaMemcpyHostToHost) );
    
    const int l = *ptr1;
    const int q = *ptr2;
    //printf("m = %d, n = %d, l = %d, q = %d \n",m, n, l, q);
    
    // allocate return maxtirx
    plhs[0] = mxCreateNumericMatrix(m, l, mxDOUBLE_CLASS, mxREAL);
    double *h_U = (double *) mxGetData(plhs[0]); // U
    
    plhs[1] = mxCreateNumericMatrix(l, 1, mxDOUBLE_CLASS, mxREAL);
    double *h_S = (double *) mxGetData(plhs[1]); // S
    
    plhs[2] = mxCreateNumericMatrix(l, n, mxDOUBLE_CLASS, mxREAL);
    double *h_VT = (double *) mxGetData(plhs[2]); // VT
    
    double dataSize = m * n * sizeof(double);
    // get available data memory
    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    const uint64_t batch = dataSize / (freeMem / 4);
    
    // create cusolverDn/cublas handle
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH) );
    CHECK_CUBLAS( cublasCreate(&cublasH) );
    
    // main prcoess
    rsvdOoC(h_U, h_S, h_VT, h_A, m, n, l, q, batch, cusolverH, cublasH);
    
    
    // clean up
    CHECK_CUBLAS( cublasDestroy(cublasH) );
    CHECK_CUSOLVER( cusolverDnDestroy(cusolverH) );
    CHECK_CUDA( cudaFreeHost(h_A) );
    
}
