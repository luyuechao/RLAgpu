#include "gpuErrorCheck.h"
#include "rpca.h"
#include <mex.h>
#include <stdio.h>

__host__ __device__
static inline uint64_t roundup_to_32X(const uint64_t x){
    return ((x + 32 - 1) / 32) * 32;
}

// Matlab Host function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Interface with matlab
    double *h_M;
    double *ptr1, *ptr2, *ptr3, *ptr4;
    
    if (nrhs != 5)
    {
        mexErrMsgIdAndTxt("MyToolbox:poseletDistance:nrhs",
                          "4 inputs required.");
    }
    if (nlhs != 3)
    {
        mexErrMsgIdAndTxt("MyToolbox:poseletDistance:nlhs",
                          "3 output required.");
    }
    
    // Create pointers to the input data
    h_M =  (double*)mxGetData(prhs[0]); //the input matrix
    // Get dimensions of input data
    int m = (int)(mxGetDimensions(prhs[0])[0]);
    int n = (int)(mxGetDimensions(prhs[0])[1]);
    
    ptr1 = (double*)mxGetData(prhs[1]);
    ptr2 = (double*)mxGetData(prhs[2]);
    ptr3 = (double*)mxGetData(prhs[3]);
    ptr4 = (double*)mxGetData(prhs[4]);
    
    const double lambda = *ptr1;
    const double tol =    *ptr2;
    const unsigned int maxit = (unsigned int) *ptr3;
    const unsigned int k  = (unsigned int) *ptr4;
    //const double tol = 1.0e-7;
    printf("m = %d, n = %d, lambda = %0.2e, tol = %0.2e, iter = %d, rank = %d \n",m, n, lambda, tol, maxit, k);
    

    
    // allocate return maxtirx
    plhs[0] = mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS, mxREAL);
    double *h_LO = (double *) mxGetData(plhs[0]);// Low-rank
    
    plhs[1] = mxCreateNumericMatrix(m, n, mxDOUBLE_CLASS, mxREAL);
    double *h_SP = (double *) mxGetData(plhs[1]);// sparse
    
    // iteration number
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    // allocate device memory
    const unsigned long long lddM  = roundup_to_32X( m );  // multiple of 32 by default
    double *d_M, *d_LO, *d_SP;

    CHECK_CUDA( cudaMalloc((void **)&d_M,  lddM * n * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void **)&d_LO, lddM * n * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void **)&d_SP, lddM * n * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_LO,   0, lddM * n * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_SP,   0, lddM * n * sizeof(double)) );
    
    // copy M to device
    CHECK_CUBLAS( cublasSetMatrix( m, n, sizeof(double), h_M, m, d_M, lddM) );
    
    unsigned int iter = 0;
    
    // main process
    rpca(d_LO, d_SP, d_M, iter, m, n, k, tol, maxit);
    
    // copy process data to host
    CHECK_CUBLAS( cublasGetMatrix(m, n, sizeof(double), d_LO, lddM, h_LO, m) );
    CHECK_CUBLAS( cublasGetMatrix(m, n, sizeof(double), d_SP, lddM, h_SP, m) );
    
    // copy iteration to API
    double *ptr_temp = mxGetPr(plhs[2]);
    *ptr_temp = (double)iter;
    
    CHECK_CUDA( cudaFree( d_M )  );
    CHECK_CUDA( cudaFree( d_LO ) );
    CHECK_CUDA( cudaFree( d_SP ) );

}
