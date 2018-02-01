#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <curand.h>

#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "gpuErrorCheck.h"
#include "rsvd.h"
#include "nccl.h"

/*
 ToDo list
 
 */
void enableGpuAccessPeer(int nDev, bool testAccess = false){
    
    int access;
    
    for (int i = 0; i < nDev; i++){
        CHECK_CUDA( cudaSetDevice( i ) );
        for (int j = 0; j < nDev; j++){
            if(i != j){
                CHECK_CUDA( cudaDeviceCanAccessPeer(&access, i, j) );
                if (access){
                    //printf("GPU %d can access GPU %d\n", i, j);
                    CHECK_CUDA( cudaDeviceEnablePeerAccess(j, 0) );
                    //CHECK_CUDA( cudaSetDevice(j) );
                    //CHECK_CUDA( cudaDeviceEnablePeerAccess(i, 0) );
                    //CHECK_CUDA( cudaSetDevice(i) );
                }else{
                    //printf("GPU %d can NOT access GPU %d\n", i, j);
                }
            }
        }
    }
    
    if (!testAccess) return;
    
    // test peer copy
    double *d_send[nDev];
    double *d_recv[nDev];
    cudaStream_t cuStream[nDev];
    int test_size = 1024;
    
    for (int i = 0; i < nDev; ++i){
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaMalloc(&d_send[i],   test_size * sizeof(double)) );
        CHECK_CUDA( cudaMalloc(&d_recv[i],   test_size * sizeof(double)) );
        CHECK_CUDA( cudaMemset(d_send[i], 1, test_size * sizeof(double)) );
        CHECK_CUDA( cudaMemset(d_recv[i], 0, test_size * sizeof(double)) );
        CHECK_CUDA( cudaStreamCreate(&cuStream[i]) );
        
    }
    
    // copy data from device 0 to all other devices
    
    for (int i = 1; i < nDev; ++i){ // be careful of i=1
        CHECK_CUDA( cudaSetDevice(i) );
        for (int j = 0; j < nDev; j++){
            if(i != j){
                CHECK_CUDA( cudaMemcpyPeerAsync( d_recv[j], j,
                                                d_send[i], i,
                                                test_size * sizeof(double),
                                                cuStream[j] ) );
            }
        }
    }
    
    
    // copy send 0 to recv 1
    // CHECK_CUDA( cudaSetDevice(0) );
    //CHECK_CUDA( cudaMemcpyPeerAsync( d_recv[1], 1, d_send[0], 0, 1, cuStream[0] ) );
    
    // copy send 1 to recv 0
    // CHECK_CUDA( cudaSetDevice(1) );
    // CHECK_CUDA( cudaMemcpyPeerAsync( d_recv[0], 0, d_send[1], 1, 1, cuStream[1] ) );
    
    // cleanup
    for (int i = 0; i < nDev; ++i){
        
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaFree(d_send[i]) );
        CHECK_CUDA( cudaFree(d_recv[i]) );
        CHECK_CUDA( cudaStreamDestroy(cuStream[i]) );
        
    }
    printf("The system passed peer copy test.\n");
}

void powerIterationMulti_colSampling( double **Y, double **TEMP, double **Ac,
                                     const uint64_t m, const uint64_t nb, const uint64_t l,
                                     const uint64_t q, const uint64_t i,
                                     cublasHandle_t *cublasH){
    
    const uint64_t ldAc = roundup_to_32X( m );
    const uint64_t ldY = ldAc;
    
    const double double_one = 1.0, double_zero = 0.0;
    
    if(q == 0){// no iteration
        return;
    }
    
    const uint64_t ldTEMP = roundup_to_32X( nb );
    CHECK_CUDA( cudaMalloc((void**)&TEMP[i], ldTEMP * l * sizeof(double)) );
    
    for(uint64_t j = 0; j < q; j++){
        // TEMP = Ac' * Y, P[nsxl] = Ac'[nsxm] * Y[mxl]
        CHECK_CUBLAS( cublasDgemm( cublasH[i], CUBLAS_OP_T, CUBLAS_OP_N,
                                  nb, l, m,
                                  &double_one,
                                  Ac[i],   ldAc,
                                  Y[i],   ldY,
                                  &double_zero,
                                  TEMP[i], ldTEMP) );
        
        //Y = Ac * TEMP, Y[mxl] = Ac[mxns] * TEMP[nsxl]
        CHECK_CUBLAS( cublasDgemm( cublasH[i], CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, nb,
                                  &double_one,
                                  Ac[i],   ldAc,
                                  TEMP[i], ldTEMP,
                                  &double_zero,
                                  Y[i],   ldY) );
        
    }

    CHECK_CUDA( cudaFree(TEMP[i]) );
    
}

void brsvdMulti_colSampling(double *U, double *Sv, double *VT, double *d_A,
                            const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                            const uint64_t batch,
                            ncclComm_t *comms, cusolverDnHandle_t &cusolverH){
    
    uint64_t nb = 0, lastbatch = 0;
    
    if(batch == 0){
        nb = n;
        lastbatch = n;
    }else{
        nb = n / batch;
        lastbatch = n - nb * batch;
    }
    
    int nDev = batch;
    
    double    *Ac[nDev];
    double *Omega[nDev];
    double     *Y[nDev];
    double  *TEMP[nDev];
    
    //setup parameters
    const uint64_t ldAc    = roundup_to_32X( m ); // pad columns into multiple of 32
    const uint64_t ldOmega = roundup_to_32X( nb);
    const uint64_t ldY = ldAc;
    
    // double_one & double_zero for matrix multiplication
    const double double_one = 1.0, double_zero = 0.0;
    
    curandGenerator_t randGen[nDev];
    cublasHandle_t    cublasH[nDev];
    cudaStream_t     cuStream[nDev];
    
    // initialize curand and cuBLAS
    for (int i = 0; i < nDev; ++i){
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaMalloc(&Ac[i], ldAc * nb * sizeof(double)) );
        CHECK_CUDA( cudaMalloc(&Y[i], ldY * l  * sizeof(double)) );
        // create curand handle
        CHECK_CURAND( curandCreateGenerator(&randGen[i], CURAND_RNG_PSEUDO_DEFAULT) );
        // seeds for curand
        CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(randGen[i], time(NULL)) );
        // create cublas handle
        CHECK_CUBLAS( cublasCreate(&cublasH[i]) );
        // create and set stream
        CHECK_CUDA( cudaStreamCreate(&cuStream[i]) );
        CHECK_CURAND( curandSetStream(randGen[i], cuStream[i]) );
        CHECK_CUBLAS( cublasSetStream(cublasH[i], cuStream[i]) );
        
    }
    
    CHECK_CUDA( cudaDeviceSynchronize() );
    
    // print_device_matrix(d_A, m, n, ldAc, "A");
    // scatter A, generate random number, multiplication
    for (int i = 0; i < nDev; ++i){
        CHECK_CUDA( cudaSetDevice(i) );
        // scatter A
        CHECK_CUDA( cudaMemcpyPeerAsync(Ac[i], i,
                                        d_A + ldAc * nb * i, 0,
                                        ldAc * nb * sizeof(double), cuStream[i]) );
        CHECK_CUDA( cudaMalloc(&Omega[i], ldOmega * l * sizeof(double)) );

        // generate double normal distribution with mean = 0.0, stddev = 1.0
        CHECK_CURAND( curandGenerateNormalDouble(randGen[i], Omega[i], ldOmega * l, 0.0, 1.0) );
        
        /*********** Y[mxl] = Ac[mxnb] * Omega[nbxl] ************/
        CHECK_CUBLAS( cublasDgemm( cublasH[i],  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, nb,
                                  &double_one,
                                  Ac[i],    ldAc,
                                  Omega[i], ldOmega,
                                  &double_zero,
                                  Y[i],    ldY) );
        //printf("Ac[%d]", i);
        //print_device_matrix(Ac[i], m, nb, ldAc, " ");
        //printf("Y[%d]", i);
        //print_device_matrix(Y[i], m, l, ldY, " ");
        CHECK_CUDA( cudaFree(Omega[i]) );
        
        /********** Step 2: power iteration *********/
        powerIterationMulti_colSampling(Y, TEMP, Ac, m, nb, l, q, i, cublasH);
    
    }
    

    // sum Y to Q
    double *Q[nDev];

    CHECK_CUDA( cudaSetDevice(0) );
    int QR_workSpace = orth_CAQR_size(m, l);
    
    // Q[0] is larger than others for CAQR decomposition
    CHECK_CUDA( cudaMalloc(&Q[0],        QR_workSpace * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(Q[0], 0, QR_workSpace * sizeof(double)) );
    
    for (int i = 1; i < nDev; ++i){
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaMalloc(&Q[i],        ldY * l * sizeof(double)) );
        CHECK_CUDA( cudaMemsetAsync(Q[i], 0, ldY * l * sizeof(double)) );
    }
    
    CHECK_NCCL( ncclGroupStart() );
    const int nccl_root = 0; // set root to dev 0
    for (int i = 0; i < nDev; ++i){
        CHECK_NCCL( ncclReduce((const void*)Y[i],
                               (void*)Q[i], ldY * l, ncclDouble,
                               ncclSum, nccl_root, comms[i], cuStream[i]) );
    }
    
    CHECK_NCCL( ncclGroupEnd() );
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaStreamSynchronize(cuStream[i]) );
        CHECK_CUDA( cudaFree(Y[i]) );
    }
    
    // device 0 to process last batch
    if(lastbatch != 0){
        int i = 0;
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaMalloc(&Y[i], ldY * l  * sizeof(double)) );
        CHECK_CUDA( cudaMalloc((void**)&Omega[i], lastbatch * l * sizeof(double)) );
        CHECK_CURAND( curandGenerateNormalDouble(randGen[i], Omega[i], lastbatch * l, 0.0, 1.0));
        CHECK_CURAND( curandDestroyGenerator( randGen[i] ) );
        
        /***********  Y[mxl] = As[mxlastbatch] * Omega[lastbatchxl] ************/
        CHECK_CUBLAS( cublasDgemm( cublasH[i],  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, lastbatch,
                                  &double_one,
                                  d_A + ldAc * nb * batch, ldAc,
                                  Omega[i], lastbatch,
                                  &double_one, // 1.0 (add to Y[0])
                                  Y[i], ldY) );
        
        CHECK_CUDA( cudaFree(Omega[i]) );
        
        if(q == 0){// no iteration
            return;
        }
        const uint64_t ldTEMP = roundup_to_32X( nb );
        CHECK_CUDA( cudaMalloc((void**)&TEMP[i], ldTEMP * l * sizeof(double)) );
        
        for(uint64_t j = 0; j < q -1; j++){
            // TEMP = Ac' * Y, P[nsxl] = Ac'[nsxm] * Y[mxl]
            CHECK_CUBLAS( cublasDgemm( cublasH[i], CUBLAS_OP_T, CUBLAS_OP_N,
                                      nb, l, m,
                                      &double_one,
                                      Ac[i],   ldAc,
                                      Y[i],   ldY,
                                      &double_zero,
                                      TEMP[i], ldTEMP) );
            
            //Y = Ac * TEMP, Y[mxl] = Ac[mxns] * TEMP[nsxl]
            CHECK_CUBLAS( cublasDgemm( cublasH[i], CUBLAS_OP_N, CUBLAS_OP_N,
                                      m, l, nb,
                                      &double_one,
                                      Ac[i],   ldAc,
                                      TEMP[i], ldTEMP,
                                      &double_zero,
                                      Y[i],   ldY) );
            
        }
        // TEMP = Ac' * Y, P[nsxl] = Ac'[nsxm] * Y[mxl]
        CHECK_CUBLAS( cublasDgemm( cublasH[i], CUBLAS_OP_T, CUBLAS_OP_N,
                                  nb, l, m,
                                  &double_one,
                                  Ac[i],   ldAc,
                                  Y[i],   ldY,
                                  &double_zero,
                                  TEMP[i], ldTEMP) );
        
        //Q = Q + Ac * TEMP, Y[mxl] = Ac[mxns] * TEMP[nsxl]
        CHECK_CUBLAS( cublasDgemm( cublasH[i], CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, nb,
                                  &double_one,
                                  Ac[i],   ldAc,
                                  TEMP[i], ldTEMP,
                                  &double_one, // one
                                  Q[i],   ldY) );
        
        CHECK_CUDA( cudaFree(TEMP[i]) );
        
        CHECK_CUDA( cudaFree(Y[i]) );
    }
    

    //QR on device 0
    CHECK_CUDA( cudaSetDevice(0) );
    //print_device_matrix(Q[0], m, l, ldY, "Q");
    orth_CAQR(Q[0], m, l);
    
    //print_device_matrix(Q[0], m, l, ldY, "Q");
    const uint64_t ldB = roundup_to_32X( l );
    double *B[nDev];
    
    for (int i = 0; i < nDev; ++i){
        CHECK_CUDA( cudaSetDevice(i) );
        // scatter Q
        if(i > 0){
            CHECK_CUDA( cudaMemcpyPeerAsync(Q[i], i,
                                            Q[0], 0,
                                            ldY * l * sizeof(double), cuStream[i] ) );
            CHECK_CUDA( cudaMalloc(&B[i], ldB * nb * sizeof(double)) );
        }else{ // i = 0
            CHECK_CUDA( cudaMalloc(&B[i], ldB * n * sizeof(double)) );
        }
        
        // Bc = Q’ * Ac, Bc[lxnb] = Q'[lxm] * Ac[mxnb]
        CHECK_CUBLAS( cublasDgemm( cublasH[i], CUBLAS_OP_T, CUBLAS_OP_N,
                                  l, nb, m,
                                  &double_one,
                                  Q[i],  ldY,
                                  Ac[i], ldAc,
                                  &double_zero,
                                  B[i], ldB) );
        
        // printf("B[%d]", i);
        // print_device_matrix(B[i], l, nb, ldB, " ");
        
        CHECK_CUDA( cudaFree(Ac[i]) );
        
    }

    CHECK_CUDA( cudaSetDevice(0) );

    
    // gather Bc to B on device 0
    for (int i = 1; i < nDev; ++i){
        CHECK_CUDA( cudaMemcpyPeerAsync(B[0] + ldB * nb * i, 0,
                                        B[i], i,
                                        ldB * nb * sizeof(double), cuStream[i] ) );
    }
    
    if(lastbatch != 0){
        // Bc = Q’ * Ac, Bc[lxlastbatch] = Q'[lxm] * Ac[mxlastbatch]
        CHECK_CUBLAS( cublasDgemm( cublasH[0], CUBLAS_OP_T, CUBLAS_OP_N,
                                  l, lastbatch, m,
                                  &double_one,
                                  Q[0], ldY,
                                  d_A + ldAc * nb * batch, ldAc,
                                  &double_zero,
                                  B[0] + ldB * nb * batch, ldB) );
    }
    
    //CHECK_CUDA( cudaFree(d_A) );
    // print_device_matrix(B[0], l, n, ldB, "B");
    // transpose B
    const uint64_t ldBT = roundup_to_32X( n );
    double *BT;
    CHECK_CUDA( cudaMalloc((void**)&BT, ldBT * l * sizeof(double)) );
    
    // synchronize the gather of B (necessary)
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA( cudaStreamSynchronize(cuStream[i]) );
    }
    
    transposeGPU(cublasH[0], BT, B[0], l, n);
    
    for (int i = 0; i < nDev; ++i){
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaFree(B[i]) );
    }
    
    CHECK_CUDA( cudaSetDevice(0) );
    /********** Step 5: SVD on BT (nxl) *********/
    //  max memroy usage: B[lxn] + UhatT[lxl] + V[nxl] = l(2n+1)
    const uint64_t ldUhat = roundup_to_32X( l ), ldV = roundup_to_32X( n );
    
    /********** Step 5: SVD on BT (nxl) *********/
    double *UhatT, *V;
    CHECK_CUDA( cudaMalloc((void**)&UhatT, ldUhat * l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&V,     ldV    * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(UhatT,0,   ldUhat * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(V,    0,   ldV    * l * sizeof(double)) );
    
    //V[nxl] * Sv[lxl] * UhatT[lxl] = BT[nxl]
    svd(cusolverH, V, Sv, UhatT, BT, n, l);
    
    CHECK_CUDA( cudaFree(BT) );
    
    /********** Step 6:  U = Q * Uhat, U[mxl] = Q[mxl] * Uhat[lxl] *********/
    CHECK_CUBLAS( cublasDgemm( cublasH[0], CUBLAS_OP_N, CUBLAS_OP_T,
                              m, l, l,
                              &double_one,
                              Q[0], ldY,
                              UhatT, ldUhat,
                              &double_zero,
                              U, ldAc) );
    
    CHECK_CUDA( cudaFree(UhatT) );

    /**********Step 7: transpose V ****/
    transposeGPU(cublasH[0], VT, V, n, l);
    CHECK_CUDA( cudaFree(V) );
    
    // cleanup
    for (int i = 0; i < nDev; ++i){
        
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaFree(Q[i]) );
        CHECK_CUBLAS( cublasDestroy(cublasH[i]) );
        CHECK_CUDA( cudaStreamDestroy( cuStream[i] ) );
    }
    
    // go back device 0
    //CHECK_CUDA( cudaSetDevice(0) );
    //CHECK_CUDA( cudaMalloc((void**)&d_A,  ldAc * n * sizeof(double)) );
    //CHECK_CUDA( cudaMemsetAsync(d_A, 0,   ldAc * n * sizeof(double)) );
    
}

void rsvd_multi_gpu(double *dev_U, double *dev_S, double *dev_VT, double *dev_A,
                    const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                    cusolverDnHandle_t &cusolverH){
    
    int nDev;
    CHECK_CUDA( cudaGetDeviceCount(&nDev) );
    //printf("%d GPU in this system.\n", nDev);
    
    /*nDev = 1;
    enableGpuAccessPeer(nDev, true);
    int devs[nDev] = {0};
    */
    // initialize multi-GPU peer access
    enableGpuAccessPeer(nDev, false);
    int devs[nDev] = {0,1};
    
    // initializing NCCL
    ncclComm_t comms[nDev];
    CHECK_NCCL( ncclCommInitAll(comms, nDev, devs) );
    
    // main process
    brsvdMulti_colSampling(dev_U, dev_S, dev_VT, dev_A, m, n, l, q,
                           nDev, comms, cusolverH);
    
    CHECK_CUDA( cudaSetDevice(0) );
}

