#include "gpuErrorCheck.h"
#include "rsvd.h"

#define GRID_SIZE 1024
#define MAX_THREADS 1024

// 1D kenrel
// Slow kernel which saves S difference between every iteration
__global__ void shrinage_kernel(double *SP, double *A,
                                const double *LO, const double *Y,
                                const double *M,
                                const double alpha, const double mu_inv,
                                const int maxSize){
    
    // initialize the memory acces index
    int dataIdx = threadIdx.x + blockIdx.x * blockDim.x;
    
    double M_mY, t, sp;
    
    const int stride = gridDim.x * blockDim.x;// loop stride
    
    for (int i = 0; i < maxSize / stride + 1; i++){
        
        if(dataIdx >= maxSize)  return;
        
        M_mY = M[dataIdx] + mu_inv * Y[dataIdx] ;
        t = M_mY - LO[dataIdx];
        sp = fmax(t - alpha, 0.0) + fmin(t + alpha, 0.0);
        
        SP[dataIdx] = sp;
        A[dataIdx] = M_mY - sp;
    
        dataIdx += stride;
        
    }
    
}

void shrinkage(double *SP, double *A, const double *LO,
              const double *Y,  const double *M,
              const double lambda, const double mu,
              const int m, const int n){
    
    const unsigned int ldA = roundup_to_32X(m);
    
    //unsigned int grid_number = (ldA * n - 1) / MAX_THREADS + 1;
    
    shrinage_kernel <<< GRID_SIZE, MAX_THREADS >>> (SP, A, LO, Y, M, lambda / mu, 1.0 / mu, ldA * n);
    
    CHECK_CUDA(cudaThreadSynchronize());
    CHECK_CUDA( cudaGetLastError());
    
}

__global__ void update_kernel(double *Y, double *Z, double *SP, double *A,
                       const double *LO, const double *M,
                       const double alpha, const double mu, const int maxSize){
    
    // initialize the memory acces index
    int dataIdx = threadIdx.x + blockIdx.x * blockDim.x;
    
    const int stride = gridDim.x * blockDim.x;// loop stride
    
    double lo, m, z, y, M_mY, t, sp;
    
    for (int i = 0; i < maxSize / stride + 1; i++){
        
        if(dataIdx >= maxSize) return;
        
        lo = LO[dataIdx];
        m = M[dataIdx];
        z = m - lo - SP[dataIdx];
        
        Z[dataIdx] = z;
        
        // Y +=  mu * Z;
        y = Y[dataIdx] + mu * Z[dataIdx];
        Y[dataIdx]  = y;
        
        M_mY = m + y / mu;
        t = M_mY - lo;
        sp = fmax(t - alpha, 0.0) + fmin(t + alpha, 0.0);
        
        // TODO use atomic function to exchange the value
        SP[dataIdx] = sp;
        A[dataIdx] = M_mY - sp;
        
        dataIdx += stride;
    
    }
    
}

void update(double *Y, double *Z, double *SP, double *A,
            const double *LO, const double *M,
            const double lambda, const double mu,
            const int m, const int n){
    
    const unsigned int lda = roundup_to_32X( m );

    update_kernel <<< GRID_SIZE, MAX_THREADS >>>
                  (Y, Z, SP, A, LO, M, lambda / mu, mu, lda * n);

    
    CHECK_CUDA( cudaThreadSynchronize() );
    CHECK_CUDA( cudaGetLastError());
}
