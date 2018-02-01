/*This file implements the out of core by cublasXt */
#include "gpuErrorCheck.h"
#include "rsvd.h"
#include <cublasXt.h>

using namespace std;

void cublasXtRsvdColSampling(double *U, double *S, double *VT, double *A,
                             const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                             cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH,
                             cublasXtHandle_t &cublasXtH){
    
    // double_one & double_zero for matrix multiplication
    const double double_one = 1.0, double_zero = 0.0;
    
    /*********** Step 1: Y = A * Omega, ************/
    double *Omega;
    CHECK_CUDA( cudaMallocHost((void**)&Omega, n * l * sizeof(double)) );
    genNormalRand(Omega, n, l);
    
    const uint64_t ldY = roundup_to_32X( m );
    
    
    int QR_workSpace = orth_CAQR_size(m, l);
    
    double *d_Y;
    
    //CHECK_CUDA(cudaMalloc((void**)&d_Y, ldY * l * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**)&d_Y, QR_workSpace * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_Y,  0, QR_workSpace * sizeof(double)) );
    
    
    
    // Y = A * Omega, Y[mxl] = A[mxn] * Omega[nxl] memory usage:  m(l+n) + nl
    CHECK_CUBLAS( cublasXtDgemm( cublasXtH,  CUBLAS_OP_N, CUBLAS_OP_N,
                                m, l, n,
                                &double_one,
                                A, m,
                                Omega, n,
                                &double_zero,
                                d_Y, ldY) );
    
    CHECK_CUDA( cudaFreeHost(Omega) );
    
    /********** Step 2: power iteration *********/
    const uint64_t ldP = roundup_to_32X( n );
    double *d_P;
    //= (double*)malloc(n * l sizeof(double));
    CHECK_CUDA( cudaMalloc((void**)&d_P, ldP * l * sizeof(double)) );
    
    for(uint64_t i = 0; i < q; i++){
        // P = A' * Y, P[nxl] = A'[nxm] * Y[mxl]
        CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, l, m,
                                    &double_one,
                                    A,   m,
                                    d_Y, ldY,
                                    &double_zero,
                                    d_P, ldP) );
        
        //CHECK_CUDA( cudaMemset(Y, 0, ldA * l * sizeof(double)) );
        
        //Y = A * P, Y[mxl] = A[mxn] * P[nxl]
        CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, l, n,
                                    &double_one,
                                    A,   m,
                                    d_P, ldP,
                                    &double_zero,
                                    d_Y, ldY) );
        
    }
    
    CHECK_CUDA( cudaFree(d_P) );
    orth_CAQR(d_Y, m, l);
    //orthogonalization(cusolverH, d_Y, m, l); orth by cusolver
    
    double *Q;
    CHECK_CUDA( cudaMallocHost((void**)&Q, m * l * sizeof(double)) );
    CHECK_CUBLAS( cublasGetMatrix(m, l, sizeof(d_Y), d_Y, ldY, Q, m) );
    
    /************ Step 3: B' = A' * Q, B'[nxl] = A'[nxm] * Q[mxl] **********/
    // allocate for SVD memory
    const uint64_t ldBT = roundup_to_32X( n );
    double *d_BT;
    CHECK_CUDA(cudaMalloc((void**)&d_BT, ldBT   * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_BT, 0, ldBT   * l * sizeof(double)) );
    
    // Note: becuase cusolver 8.0 can not solve m < n matrix,
    // B is transposed, which different from Halko's algorithm
    // BT = A' * Q, BT[nxl] = A'[nxm] * Q[mxl]
    CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_T, CUBLAS_OP_N,
                                n, l, m,
                                &double_one,
                                A,   m,
                                d_Y, ldY,
                                &double_zero,
                                d_BT, ldBT) );
    
    CHECK_CUDA( cudaFree(d_Y) );
    
    /********** Step 5: SVD on BT (nxl) *********/
    const uint64_t ldUhat = roundup_to_32X( l );
    const uint64_t ldV    = roundup_to_32X( n );
    
    double *d_UhatT, *d_S, *d_V;
    CHECK_CUDA( cudaMalloc((void**)&d_UhatT, ldUhat * l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&d_V,     ldV    * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_UhatT,0,   ldUhat * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_V,    0,   ldV    * l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&d_S,              l * sizeof(double)) );
    
    CHECK_CUDA( cudaThreadSynchronize() );
    
    //d_V[nxl] * d_S[lxl] * d_UhatT[lxl] = d_BT[nxl]
    svd(cusolverH, d_V, d_S, d_UhatT, d_BT, n, l);

    CHECK_CUDA( cudaMemcpyAsync(S, d_S, l * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaFree(d_BT) );
    
    /********** Step 6:  U = Q * Uhat, U[mxl] = Q[mxl] * Uhat[lxl] *********/
    CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_N, CUBLAS_OP_T,
                                m, l, l,
                                &double_one,
                                Q, m,
                                d_UhatT, ldUhat,
                                &double_zero,
                                U, m) );
    
    CHECK_CUDA( cudaFreeHost(Q) );
    CHECK_CUDA( cudaFree(d_UhatT) );
    
    /**********Step 7: transpose V ****/
    const uint64_t ldVT = roundup_to_32X( l );
    double *d_VT;
    CHECK_CUDA( cudaMalloc((void**)&d_VT, ldVT * n * sizeof(double)) );
    
    transposeGPU(cublasH, d_VT, d_V, n, l);

    CHECK_CUBLAS( cublasGetMatrix(l, n, sizeof(VT), d_VT, ldVT, VT, l) );
    
    CHECK_CUDA( cudaFree(d_V) );
    CHECK_CUDA( cudaFree(d_VT));
    CHECK_CUDA( cudaFree(d_S) );
    
}

void cublasXtRsvdRowSampling(double *U, double *S, double *VT, double *A,
                             const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                             cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH,
                             cublasXtHandle_t &cublasXtH){
    
    // double_one & double_zero for matrix multiplication
    const double double_one = 1.0, double_zero = 0.0;
    

    
    /*********** Step 1: Y =  A' * Omega, ************/
    double *Omega;
    CHECK_CUDA( cudaMallocHost((void**)&Omega, m * l * sizeof(double)) );
    genNormalRand(Omega, m, l);
    
    const uint64_t ldY = roundup_to_32X( n );
    
    double *d_Y;
    
    //CHECK_CUDA(cudaMalloc((void**)&d_Y, ldY * l * sizeof(double)) );
    // set Y to zero, very important
    int QR_workSpace = orth_CAQR_size(n, l);
    CHECK_CUDA( cudaMalloc((void**)&d_Y,    QR_workSpace * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_Y, 0,     QR_workSpace * sizeof(double)) );
    
    // Y = A' * Omega, Y[nxl] = A'[nxm] * Omega[mxl]
    CHECK_CUBLAS( cublasXtDgemm( cublasXtH,  CUBLAS_OP_T, CUBLAS_OP_N,
                                n, l, m,
                                &double_one,
                                A, m,
                                Omega, m,
                                &double_zero,
                                d_Y, ldY) );
    
    CHECK_CUDA( cudaFreeHost(Omega) );
    
    /********** Step 2: power iteration *********/
    const uint64_t ldP = roundup_to_32X( m );
    double *d_P;
    CHECK_CUDA( cudaMalloc((void**)&d_P, ldP * l * sizeof(double)) );
    
    for(uint64_t i = 0; i < q; i++){
        // P =  A * Y, P[mxl] =  A[mxn] * Y[nxl]
        CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, l, n,
                                    &double_one,
                                    A,   m,
                                    d_Y, ldY,
                                    &double_zero,
                                    d_P, ldP) );
        
        //Q = A' * P, Q[nxl] = A'[nxm] * P[mxl], Y is used to save Q
        CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, l, m,
                                    &double_one,
                                    A,   m,
                                    d_P, ldP,
                                    &double_zero,
                                    d_Y, ldY) );
    }
    
    CHECK_CUDA( cudaFree(d_P) );
    
    orth_CAQR(d_Y, n, l);
    //orthogonalization(cusolverH, d_Y, n, l); orth by cusolver
    
    double *Q;
    CHECK_CUDA( cudaMallocHost((void**)&Q, n * l * sizeof(double)) );
    CHECK_CUBLAS( cublasGetMatrix(n, l, sizeof(double), d_Y, ldY, Q, n) );
    
    /************ Step 3: B = A * Q **********/
    const uint64_t ldB = roundup_to_32X( m );
    double *d_B;
    CHECK_CUDA(cudaMalloc((void**)&d_B, ldB * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_B, 0, ldB * l * sizeof(double)) );
    
    //   B[mxl] = A[mxn] * Q[nxl]
    CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_N, CUBLAS_OP_N,
                                m, l, n,
                                &double_one,
                                A,   m,
                                d_Y, ldY,
                                &double_zero,
                                d_B, ldB) );
    
    CHECK_CUDA( cudaFree(d_Y) );
    
    /********** Step 5: SVD on BT (mxl) *********/
    const uint64_t ldU     = roundup_to_32X( m );
    const uint64_t ldVThat = roundup_to_32X( l );
    
    double *d_U, *d_S, *d_VThat;
    CHECK_CUDA( cudaMalloc((void**)&d_U,     ldU     * l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&d_S,               l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&d_VThat, ldVThat * l * sizeof(double)) );
    //CHECK_CUDA( cudaMemsetAsync(d_VThat, 0,  ldVThat * l * sizeof(double)) );
    
    //CHECK_CUDA( cudaThreadSynchronize() );
    
    //U[mxl] * S[lxl] * VThat[lxl] = BT[mxl]
    svd(cusolverH, d_U, d_S, d_VThat, d_B, m, l);
    CHECK_CUDA( cudaMemcpyAsync(S, d_S, l * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaFree(d_B) );
    CHECK_CUBLAS( cublasGetMatrix(m, l, sizeof(double), d_U, ldU, U, m) );
    CHECK_CUDA( cudaFree(d_U) );
    
    /********** Step 6:  VT =  VThat * Q', VT[lxn] = VThat[lxl] * Q'[lxn] *********/
    CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_N, CUBLAS_OP_T,
                                l, n, l,
                                &double_one,
                                d_VThat, ldVThat,
                                Q,  n,
                                &double_zero,
                                VT, l) );
    
    CHECK_CUDA( cudaFreeHost(Q) );
    CHECK_CUDA( cudaFree(d_S) );
    CHECK_CUDA( cudaFree(d_VThat) );
    
}

void cublasXtRsvd(double *U, double *S, double *VT, double *A,
                  const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                  cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    
    // initilize cublasXt
    cublasXtHandle_t cublasXtH = NULL;
    cublasXtCreate(&cublasXtH);
    
    // setup device
    int devices[1] = { 0 };
    CHECK_CUBLAS( cublasXtDeviceSelect(cublasXtH, 1, devices) );
    
    if(m >= n){
        cublasXtRsvdColSampling(U, S, VT, A, m, n, l, q, cusolverH, cublasH, cublasXtH);
    }else{
        cublasXtRsvdRowSampling(U, S, VT, A, m, n, l, q, cusolverH, cublasH, cublasXtH);
    }
    
    CHECK_CUBLAS( cublasXtDestroy(cublasXtH) );
}
