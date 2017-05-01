/*This file implements the out of core */
#include "gpuErrorCheck.h"
#include "rsvd.h"
#include <fstream>
#define TimerON true

using namespace std;

void TimerStart(cudaEvent_t &start){ cudaEventRecord(start, 0); };
void TimerStop(cudaEvent_t &start, cudaEvent_t &stop, float &timer, fstream &fs){
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
    fs << timer << ",";
};

void powerIterationOoC_colSampling( double *Y, double *Yb, const double *Ac,
                                   const uint64_t m, const uint64_t ns, const uint64_t l,
                                   const uint64_t q,
                                   cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    
    const uint64_t ldAc = roundup_to_32X( m );
    const uint64_t ldYb = ldAc, ldY = ldAc;
    
    const double double_one = 1.0, double_zero = 0.0;
    
    if(q == 0){// no iteration
        // Y[mxl] += Yb[mxl] in-place addition
        CHECK_CUBLAS( cublasDgeam( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                   m, l,
                                   &double_one, // 1.0
                                   Y,  ldY,
                                   &double_one, // 1.0
                                   Yb, ldYb,
                                   Y,  ldY));
        return;
    }
    
    const uint64_t ldTEMP = roundup_to_32X( ns );
    double *TEMP;
    CHECK_CUDA( cudaMalloc((void**)&TEMP, ldTEMP * l * sizeof(double)) );
    
    
    for(uint64_t i = 0; i < q - 1; i++){
        // P = As' * TEMP, P[nsxl] = As'[nsxm] * TEMP[mxl]
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                  ns, l, m,
                                  &double_one,
                                  Ac,   ldAc,
                                  Yb,   ldYb,
                                  &double_zero,
                                  TEMP, ldTEMP) );
        
        //Yb = As * TEMP, Yb[mxl] = As[mxns] * TEMP[nsxl]
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, ns,
                                  &double_one,
                                  Ac,   ldAc,
                                  TEMP, ldTEMP,
                                  &double_zero,
                                  Yb,   ldYb) );
        
    }
    // for the last iteration
    // TEMP = As' * Yb, TEMP[nsxl] = As'[nsxm] * Yb[mxl]
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                              ns, l, m,
                              &double_one,
                              Ac,   ldAc,
                              Yb,   ldYb,
                              &double_zero,
                              TEMP, ldTEMP) );
    
    //Y = As * TEMP + Y, Y[mxl] += As[mxns] * TEMP[nsxl]
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, l, ns,
                              &double_one,
                              Ac, ldAc,
                              TEMP, ldTEMP,
                              &double_one, // 1.0
                              Y, ldY) );
    
    CHECK_CUDA( cudaFree(TEMP) );
}

void rsvdOoC_colSampling(double *host_U, double *host_S, double *host_VT, const double *host_A,
                         const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                         const uint64_t batch,
                         cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    
    fstream fs;

    cudaEvent_t	start0, stop0, start1, stop1, start2, stop2, start3, stop3, start4, stop4;
    float timer0 = 0.0F, timer1 = 0.0F, timer2 = 0.0F, timer3 = 0.0F, timer4 = 0.0F;

    if(TimerON){
        fs.open("OoC_colSampling.csv", fstream::out | fstream::app);
        fs << m << "," << n << "," << l << ",";
        cudaEventCreate(&start0);	cudaEventCreate(&start1);	cudaEventCreate(&start2);
        cudaEventCreate(&start3);   cudaEventCreate(&start4);
        cudaEventCreate(&stop0);	cudaEventCreate(&stop1);	cudaEventCreate(&stop2);
        cudaEventCreate(&stop3);    cudaEventCreate(&stop4);
    }
    
    // timer 0 for sampling and power iteration
    if (TimerON)	{ TimerStart(start0); }
    
    // cuda stream for asynchronize
    cudaStream_t cudaStream1;
    CHECK_CUDA( cudaStreamCreate(&cudaStream1) );

    uint64_t nb = 0, lastbatch = 0;
    
    if(batch == 0){
        nb = n;
        lastbatch = n;
    }else{
        nb = n / batch;
        lastbatch = n - nb * batch;
    }
    
    //setup parameters
    const uint64_t ldAc    = roundup_to_32X( m ); // pad columns into multiple of 32
    const uint64_t ldOmega = roundup_to_32X( nb);


    const uint64_t ldY = ldAc, ldYb = ldAc;
    
    // double_one & double_zero for matrix multiplication
    const double double_one = 1.0, double_zero = 0.0;
    
    // allocate device memory
    double *Ac, *Omega, *Y,  *Yb;
    CHECK_CUDA( cudaMalloc((void**)&Ac, ldAc * nb * sizeof(double)) );

    CHECK_CUDA( cudaMalloc((void**)&Yb, ldYb * l  * sizeof(double)) );
    
    // set Y to 0, VERY IMPORTANT.
    int QR_workSpace = orth_CAQR_size(m, l);
    
    //cout << "m * l        = " << ldAc * l << endl;
    //cout << "QR_workSpace = " << QR_workSpace << endl;
    
    //double allocated = 0.0;
    
    //allocated += ldAc * nb + ldYb * l + ldAc * l + QR_workSpace;
    //cout << "allocated = " << ( allocated * sizeof(double) ) / (1024.0 * 1024.0 * 1024.0);
    
    CHECK_CUDA( cudaMalloc((void**)&Y, QR_workSpace * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(Y,  0, QR_workSpace * sizeof(double)) );
    
    curandGenerator_t randGen;
    CHECK_CURAND( curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT) );
    // seeds for curand
    CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)) );
    
    // memory usage: As[mxns] + TEMP[mxl] + Y[mxl] + Omega[nsxl] + P[nsxl] = m(ns + 2l)< mn
    for(uint64_t i = 0; i < batch; i++){

        CHECK_CUBLAS( cublasSetMatrixAsync(m, nb, sizeof(double),
                                           host_A + m * nb * i, m, Ac, ldAc, cudaStream1) );
        
        //cout << i;print_device_matrix(Ac, m, nb, ldAc, "Ac");
        /*********** Yb[mxl] = Ac[mxnb] * Omega[nbxl] ************/
        CHECK_CUDA( cudaMalloc((void**)&Omega, ldOmega * l * sizeof(double)) );
        // generate double normal distribution with mean = 0.0, stddev = 1.0
        CHECK_CURAND( curandGenerateNormalDouble(randGen, Omega, ldOmega * l, 0.0, 1.0) );
        
        CHECK_CUBLAS( cublasDgemm( cublasH,  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, nb,
                                  &double_one,
                                  Ac,    ldAc,
                                  Omega, ldOmega,
                                  &double_zero,
                                  Yb,    ldYb) );
        
        CHECK_CUDA( cudaFree(Omega) );
        
        /********** Step 2: power iteration *********/
        powerIterationOoC_colSampling(Y, Yb, Ac, m, nb, l, q, cusolverH, cublasH);
        
    }
    
    // last batch
    if(lastbatch != 0){
        
        CHECK_CUBLAS( cublasSetMatrixAsync(m, lastbatch, sizeof(host_A),
                                           host_A + m * nb * batch, m, Ac, ldAc, cudaStream1) );
        /*********** TEMP[mxl] = As[mxlastbatch] * Omega[lastbatchxl] ************/
        CHECK_CUDA( cudaMalloc((void**)&Omega, lastbatch * l * sizeof(double)) );
        CHECK_CURAND(curandGenerateNormalDouble(randGen, Omega, lastbatch * l, 0.0, 1.0));
        CHECK_CUBLAS( cublasDgemm( cublasH,  CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, lastbatch,
                                  &double_one,
                                  Ac,    ldAc,
                                  Omega, lastbatch,
                                  &double_zero,
                                  Yb,    ldYb) );
        
        CHECK_CUDA( cudaFree(Omega) );
        powerIterationOoC_colSampling(Y, Yb, Ac, m, lastbatch, l, q, cusolverH, cublasH);
    
    }
    
    CHECK_CURAND( curandDestroyGenerator( randGen ) );
    CHECK_CUDA( cudaFree(Ac) );
    CHECK_CUDA( cudaFree(Yb) );

    if (TimerON)	{ TimerStop( start0, stop0, timer0, fs );}
    
    if (TimerON)	{ TimerStart(start1); }
    
    orth_CAQR(Y, m, l);
    //orthogonalization(cusolverH, Y, m, l);

    if (TimerON)	{ TimerStop( start1, stop1, timer1, fs );}
    
    if (TimerON)	{ TimerStart(start2); }
    double *host_Q;
    CHECK_CUDA( cudaHostAlloc( (void**)&host_Q,  ldY * l * sizeof(double), cudaHostAllocPortable ) );
    CHECK_CUDA( cudaMemcpyAsync(host_Q, Y,       ldY * l * sizeof(double), cudaMemcpyDeviceToHost) );
    
    const uint64_t ldBT = roundup_to_32X( n );
    double *BT;
    CHECK_CUDA( cudaMalloc((void**)&BT, ldBT * l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&Ac, ldAc* nb * sizeof(double)) );
    
    for(uint64_t i = 0; i < batch; i++){
         CHECK_CUBLAS( cublasSetMatrix(m, nb, sizeof(double), host_A + m * nb * i, m, Ac, ldAc) );
        // BT =  Ac' * Q, Bc[nbxl] = Ac'[nbxm] * Q[mxl]
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                  nb, l, m,
                                  &double_one,
                                  Ac, ldAc,
                                  Y,  ldY,
                                  &double_zero,
                                  BT + nb * i , ldBT) );
    }
    
    if(lastbatch != 0){
        CHECK_CUBLAS( cublasSetMatrix(m, lastbatch, sizeof(double), host_A + m * nb * batch, m, Ac, ldAc) );
        //BTlast[lastbatch x l] = Alast'[lastbatch x m] *  Q[mxl]
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                  lastbatch, l, m,
                                  &double_one,
                                  Ac, ldAc,
                                  Y,  ldY,
                                  &double_zero,
                                  BT + nb * batch, ldBT) );
    }
    
    CHECK_CUDA( cudaFree(Ac) );
    CHECK_CUDA( cudaFree(Y) );
    
    if (TimerON)	{ TimerStop( start2, stop2, timer2, fs );}
    
//    float gflops;
//    if(TimerON) {
//        gflops = 2 * m * n * l / (1e9 *(timer2 / 1e3));
//        printf("gflops = %0.2e\n",gflops);
//    }
    
    if (TimerON)	{ TimerStart(start3); }
    /********** Step 5: SVD on BT (nxl) *********/
    //  max memroy usage: B[lxn] + UhatT[lxl] + V[nxl] = l(2n+1)
    const uint64_t ldUhat = roundup_to_32X( l ), ldV = roundup_to_32X( n );
    
    double *UhatT, *Sv, *V;
    CHECK_CUDA( cudaMalloc((void**)&UhatT, ldUhat * l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&Sv,             l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&V,     ldV    * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(UhatT,0,   ldUhat * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(V,    0,   ldV    * l * sizeof(double)) );
    CHECK_CUDA( cudaThreadSynchronize() );
    
    //V[nxl] * Sv[lxl] * UhatT[lxl] = BT[nxl]
    svd(cusolverH, V, Sv, UhatT, BT, n, l);

    CHECK_CUDA( cudaFree(BT) );
    // copy Q to device
    double *Q;
    const uint64_t ldA = ldAc;
    CHECK_CUDA( cudaMalloc((void**)&Q, ldA * l * sizeof(double)) );
    CHECK_CUDA( cudaMemcpyAsync(Q, host_Q, ldA * l * sizeof(double), cudaMemcpyHostToDevice) );
    
    // copy Sv to host
    CHECK_CUDA( cudaMemcpyAsync(host_S, Sv, l * sizeof(double), cudaMemcpyDeviceToHost) );
    // copy VT to host
    double *VT;
    const uint64_t ldVT = roundup_to_32X( l );
    CHECK_CUDA( cudaMalloc((void **)&VT, ldVT * n * sizeof(double)) );
    transposeGPU(cublasH, VT, V, n, l);
    CHECK_CUDA( cudaFree(V) );
    CHECK_CUBLAS( cublasGetMatrixAsync(l, n, sizeof(double), VT, ldVT, host_VT, l, cudaStream1) );
    
    if (TimerON)	{ TimerStop( start3, stop3, timer3, fs );}
    
    if (TimerON)    { TimerStart(start4); }
 
    double *U;
    CHECK_CUDA( cudaMalloc((void**)&U, ldA * l * sizeof(double)) );


    
    /********** Step 6:  U = Q * Uhat, U[mxl] = Q[mxl] * Uhat[lxl] *********/
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                               m, l, l,
                               &double_one,
                               Q, ldA,
                               UhatT, ldUhat,
                               &double_zero,
                               U, ldA) );
    
    CHECK_CUBLAS( cublasGetMatrix(m, l, sizeof(double), U, ldA, host_U, m) );
    
    // clean up
    CHECK_CUDA( cudaFreeHost(host_Q) );
    CHECK_CUDA( cudaFree(Sv) );
    CHECK_CUDA( cudaFree(UhatT) );
    CHECK_CUDA( cudaFree(U) );
    CHECK_CUDA( cudaFree(Q) );
    

    CHECK_CUDA( cudaFree(VT));
    CHECK_CUDA( cudaStreamDestroy(cudaStream1) );
    
    if (TimerON)	{ TimerStop( start4, stop4, timer4, fs );}
    
    if(TimerON){
        fs << endl;
        fs.close();
        cudaEventDestroy(start0);	cudaEventDestroy(start1);	cudaEventDestroy(start2);
        cudaEventDestroy(start3);   cudaEventDestroy(start4);
        cudaEventDestroy(stop0);	cudaEventDestroy(stop1);	cudaEventDestroy(stop2);
        cudaEventDestroy(stop3);    cudaEventDestroy(stop4);
    }

}

// the following code is for row sampling
void powerIterationOoC_rowSampling( double *Y, double *Yb, const double *Ar,
                                   const uint64_t mb, const uint64_t n, const uint64_t l,
                                   const uint64_t q,
                                   cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    
    const uint64_t ldAr   = roundup_to_32X( mb);
    const uint64_t ldY    = roundup_to_32X( n );
    const uint64_t ldYb   = roundup_to_32X( l );
    const uint64_t ldTEMP = ldYb;
    
    const double double_one = 1.0, double_zero = 0.0;
    
    if(q == 0){// no iteration
        // Y[nxl] += Yb'[nxl] in-place addition
        CHECK_CUBLAS( cublasDgeam( cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                                  n, l,
                                  &double_one,
                                  Y,  ldY,
                                  &double_one, // 1.0
                                  Yb, ldYb,
                                  Y,  ldY));
        return;
    }
    
    double *TEMP;
    CHECK_CUDA( cudaMalloc((void**)&TEMP, ldTEMP * mb * sizeof(double)) );
    
    for(uint64_t i = 0; i < q - 1; i++){
        // TEMP =  Yb * Ar',  TEMP[lxmb] = Yb[lxn] * Ar'[nxmb]
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                                  l, mb, n,
                                  &double_one,
                                  Yb,   ldYb,
                                  Ar,   ldAr,
                                  &double_zero,
                                  TEMP, ldTEMP) );
        
        //Yb = TEMP * Ar, Yb[lxn] = TEMP[lxmb] * Ar[mbxn]
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  l, n, mb,
                                  &double_one,
                                  TEMP, ldTEMP,
                                  Ar, ldAr,
                                  &double_zero,
                                  Yb, ldYb) );
        
    }
    // for the last iteration
    // TEMP = Yb * Ar', TEMP[lxmb] = Yb[lxn] * Ar'[nxmb]
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                              l, mb, n,
                              &double_one,
                              Yb,   ldYb,
                              Ar,   ldAr,
                              &double_zero,
                              TEMP, ldTEMP) );
    
    //Y += Ar' * TEMP', Y[nxl] +=  Ar'[n x mb ] * TEMP'[mb x l]
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_T,
                              n, l, mb,
                              &double_one,
                              Ar,   ldAr,
                              TEMP, ldTEMP,
                              &double_one, // 1.0
                              Y, ldY) );
    
    CHECK_CUDA( cudaFree(TEMP) );
    
}

void rsvdOoC_rowSampling(double *host_U, double *host_S, double *host_VT, const double *host_A,
                         const uint64_t m, const uint64_t n, const uint64_t l,
                         const uint64_t q, const uint64_t batch,
                         cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    

    
    fstream fs;
    
    cudaEvent_t	start0, stop0, start1, stop1, start2, stop2, start3, stop3, start4, stop4;
    float timer0 = 0.0F, timer1 = 0.0F, timer2 = 0.0F, timer3 = 0.0F, timer4 = 0.0F;
    
    if(TimerON){
        
        fs.open("OoC_rowSampling.csv", fstream::out | fstream::app);
        fs << m << "," << n << "," << l << ",";
        cudaEventCreate(&start0);	cudaEventCreate(&start1);	cudaEventCreate(&start2);
        cudaEventCreate(&start3);   cudaEventCreate(&start4);
        cudaEventCreate(&stop0);	cudaEventCreate(&stop1);	cudaEventCreate(&stop2);
        cudaEventCreate(&stop3);    cudaEventCreate(&stop4);
        
    }
    if(TimerON) { TimerStart(start0); }
    
    cudaStream_t cudaStream1;
    CHECK_CUDA( cudaStreamCreate(&cudaStream1) );
    uint64_t mb = 0, lastbatch = 0;
    if(batch == 0){
        mb = m;
        lastbatch = m;
    }else{
        mb = m / batch;
        lastbatch = m - mb * batch;
    }
    
    //setup parameters
    const uint64_t ldAr    = roundup_to_32X( mb); // pad columns into multiple of 32
    const uint64_t ldOmega = roundup_to_32X( l );
    const uint64_t ldY     = roundup_to_32X( n );
    const uint64_t ldYb    = ldOmega;
    
    // double_one & double_zero for matrix multiplication
    const double double_one = 1.0, double_zero = 0.0;
    
    // allocate device memory
    curandGenerator_t randGen;
    CHECK_CURAND(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));
    // seeds for curand
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)));
    // generate double normal distribution with mean = 0.0, stddev = 1.0
    double *Omega;

    double *Ar, *Y, *Yb;
    CHECK_CUDA( cudaMalloc((void**)&Ar, ldAr * n * sizeof(double)) );// Ar means row partition of A

    CHECK_CUDA( cudaMalloc((void**)&Yb, ldYb * n * sizeof(double)) );
    
    // set Y to zero, very important
    int QR_workSpace = orth_CAQR_size(n, l);
    CHECK_CUDA( cudaMalloc((void**)&Y,    QR_workSpace * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(Y, 0,     QR_workSpace * sizeof(double)) );

    
    for(uint64_t i = 0; i < batch; i++){
        
        CHECK_CUDA( cudaMalloc((void**)&Omega, ldOmega * mb * sizeof(double)) );
        CHECK_CURAND(curandGenerateNormalDouble(randGen, Omega, ldOmega * mb, 0.0, 1.0));
        
        // copy A batch from host to device
        CHECK_CUBLAS( cublasSetMatrix(mb, n, sizeof(double), host_A + mb * i, m, Ar, ldAr) );

        
        /*********** Yb[lxn] = Omega[lxmb] * Ar[mbxn] ************/
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  l, n, mb,
                                  &double_one,
                                  Omega, ldOmega,
                                  Ar,    ldAr,
                                  &double_zero,
                                  Yb,    ldYb) );
        
        CHECK_CUDA( cudaFree(Omega) );
        powerIterationOoC_rowSampling( Y, Yb, Ar, mb, n, l, q, cusolverH, cublasH );

    }
    
    // last batch
    if(lastbatch != 0){
        const uint64_t ldlast = roundup_to_32X( lastbatch );
        CHECK_CUDA( cudaMalloc((void**)&Omega, ldOmega * lastbatch * sizeof(double)) );
        CHECK_CURAND(curandGenerateNormalDouble(randGen, Omega, ldOmega * lastbatch, 0.0, 1.0));
        // copy A batch from host to device
        CHECK_CUBLAS( cublasSetMatrix(lastbatch, n, sizeof(double), host_A + mb * batch, m, Ar, ldlast) );
        
        /*********** Yb[lxn] = Omega[lxlastbatch] * Alast[lastbatchxn]************/
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  l, n,  lastbatch,
                                  &double_one,
                                  Omega, ldOmega,
                                  Ar,    ldlast,
                                  &double_zero,
                                  Yb,    ldYb) );
        
        CHECK_CUDA( cudaFree(Omega) );
        powerIterationOoC_rowSampling( Y, Yb, Ar, lastbatch, n, l, q, cusolverH, cublasH );
    
    }
    
    CHECK_CURAND( curandDestroyGenerator( randGen ) );
    CHECK_CUDA( cudaFree(Ar) );
    CHECK_CUDA( cudaFree(Yb) );
    
    if (TimerON)	{ TimerStop(start0, stop0, timer0, fs );}
    if (TimerON)	{ TimerStart(start1); }
    
    // orthogonalization on GPU
    orth_CAQR(Y, n, l);
    //orthogonalization(cusolverH, Y, n, l);
    
    if (TimerON)	{ TimerStop(start1, stop1, timer1, fs );}

    if (TimerON) 	{ TimerStart(start2); }

    double *host_Y;
    CHECK_CUDA( cudaHostAlloc( (void**)&host_Y, ldY * l * sizeof(double), cudaHostAllocPortable ) );
    CHECK_CUDA( cudaMemcpyAsync(host_Y,      Y, ldY * l * sizeof(double), cudaMemcpyDeviceToHost) );
    
    const uint64_t ldB = roundup_to_32X( m );
    double *B;
    CHECK_CUDA( cudaMalloc( (void**)&B,   ldB * l * sizeof(double)) );
    CHECK_CUDA( cudaMemset(B,          0, ldB * l * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**)&Ar,   ldAr *n * sizeof(double)) );
    
    for(uint64_t i = 0; i < batch; i++){
        /************ Step 4: Br = Ar * Y, Br[mbxl] = Ar[mbxn] * Y[nxl] **********/
        CHECK_CUBLAS( cublasSetMatrix(mb, n, sizeof(double), host_A + mb * i, m, Ar, ldAr) );
        //print_device_matrix(Ar, mb, n, ldAr, "Ar");
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  mb, l, n,
                                  &double_one,
                                  Ar, ldAr,
                                  Y,  ldY,
                                  &double_zero,
                                  B + mb * i, ldB) );
    }
    
    if(lastbatch != 0){
        const uint64_t ldlast = roundup_to_32X( lastbatch );
         //Blast[lastbatchxl] = Alast[lastbatch x n] * Y[n x l]
        CHECK_CUBLAS( cublasSetMatrix(lastbatch, n, sizeof(double), host_A + mb * batch, m, Ar, ldlast) );
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  lastbatch, l, n,
                                  &double_one,
                                  Ar, ldlast,
                                  Y,  ldY,
                                  &double_zero,
                                  B + mb * batch, ldB) );
    }
    
    CHECK_CUDA( cudaFree(Ar) );
    CHECK_CUDA( cudaFree(Y) );
    if (TimerON)	{ TimerStop(start2, stop2, timer2, fs );}
    
    if (TimerON) 	{ TimerStart(start3); }

    /********** Step 5: SVD on B (mxl) *********/
    const uint64_t ldU    = roundup_to_32X( m );
    const uint64_t ldVhatT =roundup_to_32X( l );
    
    double *U, *Sv, *VhatT;
    CHECK_CUDA( cudaMalloc((void**)&U,         ldU * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(U,  0,         ldU * l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&Sv,              l * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&VhatT, ldVhatT * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(VhatT, 0,  ldVhatT * l * sizeof(double)) );
    CHECK_CUDA( cudaThreadSynchronize() );
    
    //U[mxl] * Sv[lxl] * VhatT[lxl] = B[mxl]
    svd(cusolverH, U, Sv, VhatT, B, m, l);

    // copy Sv to host
    CHECK_CUDA( cudaMemcpyAsync(host_S, Sv, l * sizeof(double), cudaMemcpyDeviceToHost) );
    // copy U to host
    CHECK_CUBLAS( cublasGetMatrixAsync(m, l, sizeof(double), U, ldU, host_U, m, cudaStream1) );
    
    CHECK_CUDA( cudaFree(B) );
    // copy Q to device
    CHECK_CUDA( cudaMalloc((void**)&Y,       ldY * l * sizeof(double)) );
    CHECK_CUDA( cudaMemcpyAsync(Y,  host_Y,  ldY * l * sizeof(double), cudaMemcpyHostToDevice) );
    
    CHECK_CUDA( cudaThreadSynchronize() );
    
    CHECK_CUDA( cudaFree(Sv) );
    CHECK_CUDA( cudaFree(U) );
    CHECK_CUDA( cudaFreeHost(host_Y) );
    
    if (TimerON)	{ TimerStop(start3, stop3, timer3, fs );}

    if (TimerON) 	{ TimerStart(start4); }
    
    const uint64_t ldVT = roundup_to_32X(l);
    double *VT;
    CHECK_CUDA( cudaMalloc((void**)&VT, ldVT* n * sizeof(double)) );
   
    /********** Step 6:  VT = VhatT * Y, VT[lxn] = VhatT[lxl] * Q'[lxn] **********/
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                              l, n, l,
                              &double_one,
                              VhatT, ldVhatT,
                              Y,     ldY,
                              &double_zero,
                              VT,    ldVT) );
    
    CHECK_CUBLAS( cublasGetMatrix(l, n, sizeof(double), VT, ldVT, host_VT, l) );
    
    // clean up

    CHECK_CUDA( cudaFree(VT) );
    CHECK_CUDA( cudaFree(Y) );
    CHECK_CUDA( cudaStreamDestroy(cudaStream1) );
    
    if (TimerON)	{ TimerStop(start4, stop4, timer4, fs );}
    if (TimerON){
        fs << endl;
        fs.close();
        cudaEventDestroy(start0);	cudaEventDestroy(start1);	cudaEventDestroy(start2);
        cudaEventDestroy(start3);   cudaEventDestroy(start4);
        cudaEventDestroy(stop0);	cudaEventDestroy(stop1);	cudaEventDestroy(stop2);
        cudaEventDestroy(stop3);    cudaEventDestroy(stop4);
    }


}

void rsvdOoC(double *host_U, double *host_S, double *host_VT, const double *host_A,
             const uint64_t m, const uint64_t n, const uint64_t l,
             const uint64_t q, const uint64_t s,
             cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    
    if(m > n){
        rsvdOoC_colSampling(host_U, host_S, host_VT, host_A, m, n, l, q, s, cusolverH, cublasH);
    }else{
        rsvdOoC_rowSampling(host_U, host_S, host_VT, host_A, m, n, l, q, s, cusolverH, cublasH);
    }
    
}
