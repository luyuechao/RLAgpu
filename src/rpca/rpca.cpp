#include "gpuErrorCheck.h"

#include "rsvd.h"
#include "rpca.h"
#include <fstream>
#define TimerON true

using namespace std;

/*******************The following code is for debug*****************************/
void TimerStart(cudaEvent_t &start){ cudaEventRecord(start, 0); };
void TimerStop(cudaEvent_t &start, cudaEvent_t &stop, float &timer, fstream &fs){
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);
    timer /= 1e3;
    fs << timer << ",";
};
/******************************************************************************/

void rpca(double *LO, double *SP, const double *M, unsigned int &iter,
          const int m, const int n, const int k,
          const double tol, const unsigned int maxit){
    
    // create cusolverDn/cublas handle
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH) );
    CHECK_CUBLAS( cublasCreate(&cublasH) );
    
    /*******************The following code is for debug*****************************/
    
    fstream fs;
    
    cudaEvent_t	start0, stop0, start1, stop1, start2, stop2, start3, stop3, start4, stop4;
    float timer0 = 0.0F, timer1 = 0.0F, timer2 = 0.0F, timer3 = 0.0F, timer4 = 0.0F;
    
    if(TimerON){
        fs.open("rpca.csv", fstream::out | fstream::app);

        cudaEventCreate(&start0);	cudaEventCreate(&start1);	cudaEventCreate(&start2);
        cudaEventCreate(&start3);   cudaEventCreate(&start4);
        cudaEventCreate(&stop0);	cudaEventCreate(&stop1);	cudaEventCreate(&stop2);
        cudaEventCreate(&stop3);    cudaEventCreate(&stop4);
    }
    /*******************************************************************************/
    /*******************************************************************************/
    // timer 0 for initial
    if (TimerON)	{ TimerStart(start0); }
    
    assert( m*n >= 1024 && "For small matrix (m * n < 1024), please use CPU insteal of GPU");
    
    const unsigned int p = k;    // oversampling number
    const unsigned int l = p + k;
    
    //setup parameter
    const unsigned int ldM = roundup_to_32X( m ); // pad columns into multiple of 32
    const unsigned int ldVT= roundup_to_32X( l );
    
    // allocate device memory
    double *Y, *A, *Z, *U, *SV, *VT;
    CHECK_CUDA(cudaMalloc((void**)&Y,     sizeof(double) * ldM * n) );
    CHECK_CUDA(cudaMalloc((void**)&A,     sizeof(double) * ldM * n) );
    CHECK_CUDA(cudaMalloc((void**)&Z,     sizeof(double) * ldM * n) );
    CHECK_CUDA(cudaMalloc((void**)&U,     sizeof(double) * ldM * l) );
    CHECK_CUDA(cudaMalloc((void**)&SV,    sizeof(double) * l      ) );
    CHECK_CUDA(cudaMalloc((void**)&VT,    sizeof(double) * ldVT *n) );
    
    // set memory to 0, It's VERY IMPORTANT.
    CHECK_CUDA( cudaMemsetAsync(LO,   0,  sizeof(double) * ldM * n) );
    CHECK_CUDA( cudaMemsetAsync(SP,   0,  sizeof(double) * ldM * n) );
    CHECK_CUDA( cudaMemsetAsync(Y,    0,  sizeof(double) * ldM * n) );
    CHECK_CUDA( cudaMemsetAsync(A,    0,  sizeof(double) * ldM * n) );
    CHECK_CUDA( cudaMemsetAsync(Z,    0,  sizeof(double) * ldM * n) );
    CHECK_CUDA( cudaMemsetAsync(U,    0,  sizeof(double) * ldM * l) );
    CHECK_CUDA( cudaMemsetAsync(SV,   0,  sizeof(double) *  l     ) );
    CHECK_CUDA( cudaMemsetAsync(VT,   0,  sizeof(double) * ldVT* n) );

    // allocate host memory as pinned memory
    double *h_SV;
    CHECK_CUDA( cudaHostAlloc( (void**)&h_SV, sizeof(double) * min(m, n), cudaHostAllocPortable ) );
    
    // Synchronize for the memsetAsync()
    CHECK_CUDA( cudaThreadSynchronize() );
    
    int svlength = min(m, n); // singular value threashold index
    
    const double lambda = 1.0 / sqrt(m);
    
    //cout << "m = " << m << ", n = " << n << ", l = " << l << endl;
    
    const double norm_inf = matrixAbsMax(cublasH, M, m, n, ldM) / lambda;
        cout << scientific << setprecision(2) << "lambda = " << lambda
             << ", |M|_inf = " << matrixAbsMax(cublasH, M, m, n, ldM)
             << ", norm_inf = " << norm_inf << endl;
    
    double mu = 1.25 / norm_inf;
    
    const double mu_limit = mu * 1.0e7;
    const double rho = 1.5;
    
    // Y = M / norm_inf;
    const double  alpha = 1.0 / norm_inf, f_zero = 0.0;
    CHECK_CUBLAS( cublasDgeam( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, // output dimension
                              &alpha,  M,   ldM,
                              &f_zero, NULL,ldM,
                              Y, ldM ) );
    
    
    const double fNormM = FrobeniusNorm(cublasH, M, m, n, ldM);
        cout << "frobenius norm of M = " << fNormM
             << ", inv_mu = " << 1 / mu << endl;
    
    shrinkage(SP, A, LO, Y, M, lambda, mu, m, n);

    
    int i = 0;

    const int q = 1; // power iteration number
    
    
    if (TimerON)	{ TimerStop( start0, stop0, timer0, fs );}
    
    float svd_time = 0.0, shrink_time = 0.0, inv_svd_time = 0.0, update_time = 0.0;
    
    if(TimerON){
    

        cout << "iter \t svl \t |Z| \t";
        cout << "svd[s] \t\t shrink[s] \t inv_svd[s] \t update[s]" << endl;
    
    }
    
    for (i = 0; i < maxit; i++){
        
  
        if (TimerON)	{ TimerStart(start1); }
        
        //svd(cusolverH, U, SV, VT, A, m, n); // use GPU svd
        
        rsvd_gpu(U, SV, VT, A, m, n, l, q, cusolverH, cublasH);
        
        if (TimerON)	{ TimerStop( start1, stop1, timer1, fs );}
        svd_time += timer1;
    
        if (TimerON)	{ TimerStart(start2); }
        
        // Shrinkage singular value on CPU
        shrink_singluar_value(SV, h_SV, l, mu, svlength);
        
        if (TimerON)	{ TimerStop( start2, stop2, timer2, fs );}
        
        shrink_time += timer2;
        
        if (TimerON)	{ TimerStart(start3); }
        // LO = U[:, svl] * Si[svl] * VT[svl, :]
        inverse_svd(cublasH, LO, U, SV, VT, m, n, svlength, ldVT);
        
        if (TimerON)	{ TimerStop( start3, stop3, timer3, fs );}
        
        inv_svd_time += timer3;
        
        if (TimerON)    { TimerStart(start4); }
        // update mu
        mu = fmin(mu * rho, mu_limit);
        
        // update Y, Z, SP, A, LO, M
        update(Y, Z, SP, A, LO, M, lambda, mu, m, n);
        
        double fNormZ  = FrobeniusNorm(cublasH, Z,  m, n, ldM);
        
        double errZ = fNormZ / fNormM;
        if (TimerON)	{ TimerStop( start4, stop4, timer4, fs );}
        update_time += timer4;
        
        if(TimerON){
            cout << i << scientific << setprecision(1)
            << "\t" << svlength <<  "\t" << fNormZ << "\t\t"
            << timer1 << "\t\t" << timer2 << "\t\t"
            << timer3 << "\t\t" << timer4 << endl;
        }
        
        if(errZ < tol) { break; }// if converage
        
    }
    
    iter =  i + 1;
    
    if(TimerON){
    cout << "initialization = " << timer0 <<  endl <<
          "SVD = "     << svd_time << endl <<
          "Shrink = "  << shrink_time << endl <<
         "Inverse SVD =" << inv_svd_time << endl <<
         "Update = "     << update_time << endl;
    }
    
    // clean up
    CHECK_CUDA( cudaFreeHost(h_SV) );
    CHECK_CUDA( cudaFree(Y)  );
    CHECK_CUDA( cudaFree(A)  );
    CHECK_CUDA( cudaFree(Z)  );
    CHECK_CUDA( cudaFree(U)  );
    CHECK_CUDA( cudaFree(SV) );
    CHECK_CUDA( cudaFree(VT) );
    
    CHECK_CUBLAS( cublasDestroy(cublasH) );
    CHECK_CUSOLVER( cusolverDnDestroy(cusolverH) );
    
    /******************* The following code is for debug *****************************/
    
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

void shrink_singluar_value(double *d_SV, double *h_SV,
                           const int svMax,
                           const double mu, int &svlength){
    
    CHECK_CUDA( cudaMemcpy(h_SV, d_SV, sizeof(double) * svMax, cudaMemcpyDeviceToHost) );
    
    CHECK_CUDA( cudaThreadSynchronize() );
    CHECK_CUDA( cudaMemset(d_SV, 0, sizeof(double) * svMax) );
    // CHECK_CUDA( cudaMemsetAsync(d_SV, 0, sizeof(double) * svMax) );
    
    
    int i = 0;
    double inv_mu = 1.0 / mu;
    for(; i < svMax; i++){
        
        if(h_SV[i] <= inv_mu) { break; }
        else                  { h_SV[i] -=  inv_mu; }
    
    }
    svlength = i;
    
    //    static int sv  = 10;
    //    if (svlength < sv){
    //        sv = min(svlength + 1, svMax);
    //    }else{
    //        sv = min(svlength + int(0.05 * svMax), svMax);
    //    }
    
//    cout << "sv = ";
//    for (int j = 0; j < 10; j++){
//        cout << scientific << setprecision(2) << h_SV[j] << " ";
//    }
//    cout << endl;
    
    // copy to device
    CHECK_CUDA( cudaMemcpy(d_SV, h_SV, sizeof(double) * svlength, cudaMemcpyHostToDevice));
    //CHECK_CUDA( cudaThreadSynchronize() );
    
}

