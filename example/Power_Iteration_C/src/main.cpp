#include "gpuErrorCheck.h"
#include "rsvd.h"
#include <fstream>
#include <chrono> // for timer

#define testError true

using namespace std;

// timer
uint64_t getCurrTime()
{
    return chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void rsvd_test(const uint64_t m, const uint64_t n, const uint64_t k, const double sparsity){
    
    // create cusolverDn/cublas handle
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH) );
    CHECK_CUBLAS( cublasCreate(&cublasH) );
    
    const uint64_t p = k; // oversampling number
    const uint64_t l = k + p;

    //cout << "l = " << l << ", m = " << m << ", n = " << n << endl;
    assert(l < min(m, n) && "k+p must be < min(m, n)" );
    const uint64_t ldA  = roundup_to_32X( m );  // multiple of 32 by default
    const uint64_t ldVT = roundup_to_32X( l );
    const uint64_t ldU = ldA;
    
    // allocate device memory
    
    
    // allocate host memory as pinned memory
    double *host_S1;

    CHECK_CUDA( cudaHostAlloc( (void**)&host_S1,     l * sizeof(double), cudaHostAllocPortable ) );
    
    double *host_A, *host_U, *host_S, *host_VT;
    CHECK_CUDA( cudaMallocHost((void**)&host_A, m * n * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&host_U, m * l * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&host_S,     l * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&host_VT,l * n * sizeof(double)) );
    
    /* generate random low rank matrix A ***/
    //genLowRankMatrixGPU(cublasH, dev_A, m, n, k, ldA);
    genLowRankMatrix(host_A, m, n, k);

    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    double dataSize = m * n * sizeof(double);

    double cublasXtTime, myOoCErr, tick, tock;
    /******************************* Out-of-Core by me *********************************/
    // print to screen
    double dataGB = m * n * 8 / pow(2, 30);
    
    cout << setprecision(2) << dataGB << "\t"<< m << "\t" << n << "\t" << k;
    
    //save to file
    fstream fs;
    fs.open("RSVD_power.csv", fstream::out | fstream::app);
    
    fs << dataGB << ","
    << m << "," << n << "," << k << ",";
    
    // power iteration factor
    for (uint64_t q = 0; q<4; q++){
        tick = getCurrTime();
        // TODO find a better size
        const uint64_t batch = dataSize / (freeMem / 4);
        rsvdOoC(host_U, host_S, host_VT, host_A, m, n, l, q, batch, cusolverH, cublasH);
        tock = getCurrTime();
        cublasXtTime = (tock - tick) / 1e6; // from ms to s

        myOoCErr = 0;
        if(testError == true){
            myOoCErr = svdFrobeniusDiff(host_A, host_U, host_S, host_VT, m, n, l);
        }
        
        cout <<"\t"<< cublasXtTime << "\t" << myOoCErr;
        fs << ","<< cublasXtTime << "," << myOoCErr;
        
    }
    
    cout << endl;
    fs.close();
    
    
    // clean up
    CHECK_CUDA( cudaFreeHost(host_S1));
    CHECK_CUDA( cudaFreeHost(host_A) );
    CHECK_CUDA( cudaFreeHost(host_U) );
    CHECK_CUDA( cudaFreeHost(host_S) );
    CHECK_CUDA( cudaFreeHost(host_VT));
    CHECK_CUBLAS( cublasDestroy(cublasH) );
    CHECK_CUSOLVER( cusolverDnDestroy(cusolverH) );
    
}

int main(int argc, char **argv) {
    
    assert((argc >= 3) && "Please supply M, N, K as the argument");
    
    uint64_t m = atoll(argv[1]);
    uint64_t n = atoll(argv[2]);
    uint64_t k = atoll(argv[3]);
    double    s = stof(argv[4]);
    
    //cudaDeviceProp deviceProp;
    
    //CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    
    //size_t gpuGlobalMem = deviceProp.totalGlobalMem;
    
    //fprintf(stderr, "GPU global memory = %zu Bytes\n", gpuGlobalMem);
    
    
    
    
    //double GB = double(1024 * 1024 * 1024);
    // fprintf(stderr, "Free = %0.1f GB, Total = %0.1f GB\n", double(freeMem) / GB, double(totalMem) / GB);
    
    
    rsvd_test(m, n, k, s);
    //rQR_test();
    
    CHECK_CUDA(cudaDeviceReset());
    
    return 0;
}
