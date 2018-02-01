#include <fstream>

#include "gpuErrorCheck.h"
#include "rsvd.h"
#include "rpca.h"

#include <stdio.h>
#include <string.h>


using namespace std;

void rpca_multiGPU(double *LO, double *SP, const double *M, unsigned int &iter,
                   const int m, const int n, const int k,
                   const double tol, const unsigned int maxit);



// timer
#include <chrono>
uint64_t getCurrTime()
{
    return chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void backgroundSubstract(char* fileName, const uint64_t m, const uint64_t n,
                         const uint64_t k, const uint64_t nframes){
    
    double *h_M;
    CHECK_CUDA( cudaMallocHost((void**)&h_M, m * n * nframes * sizeof(double)) );
    
    FILE *picFile;
    if ((picFile = fopen(fileName, "r")) == NULL){
        printf("%s is not exist\n", fileName);
        return;
    }
    
    fread(h_M, sizeof(double), m * n * nframes, picFile);
    fclose(picFile);
    
    const uint64_t p = k; // oversampling number
    assert(k + p < min(m, n) && "k+p must be smaller than min(m, n)" );
    
    const uint64_t lddM  = roundup_to_32X( m * n );  // multiple of 32 by default
    
    // allocate device memory
    double *d_M, *d_LO, *d_SP;
    
    CHECK_CUDA( cudaMalloc((void **)&d_M,  lddM * nframes * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void **)&d_LO, lddM * nframes * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void **)&d_SP, lddM * nframes * sizeof(double)) );
    
    // LO and SP must be set to zero!
    CHECK_CUDA( cudaMemsetAsync(d_LO, 0, lddM * nframes * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_SP, 0, lddM * nframes * sizeof(double)) );
    
    CHECK_CUBLAS( cublasSetMatrix(m * n, nframes, sizeof(double),
                                  h_M, m * n, d_M, lddM) );
    
    /*************************** Robust PCA on GPU *****************************************/
    // parameter for RPCA
    const unsigned int maxit = 50; // TODO: 50 is temporay number
    const double tol = 1.0e-5;
    unsigned int iter = 0;
    // main process
    uint64_t tick = getCurrTime();
    
    rpca_multiGPU(d_LO, d_SP, d_M, iter, m * n, nframes, k, tol, maxit);
    
    uint64_t tock = getCurrTime();
    double rpca_time = (tock - tick) / 1e6;
    
    
    // copy SP, LO to host
    double *h_LO, *h_SP;
    CHECK_CUDA( cudaMallocHost((void**)&h_LO, m * n * nframes * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&h_SP, m * n * nframes * sizeof(double)) );
    
    CHECK_CUBLAS( cublasGetMatrix(m * n, nframes, sizeof(double), d_LO, lddM, h_LO, m * n) );
    CHECK_CUBLAS( cublasGetMatrix(m * n, nframes, sizeof(double), d_SP, lddM, h_SP, m * n) );
    
    // write out
    FILE *fileLowRank, *fileSparse;
    
    if ((fileLowRank = fopen("lowRank.bin", "wb")) == NULL){
        printf("Low-rank file can not open\n");
        exit(1);
    }
    
    fwrite(h_LO, sizeof(double), m * n * nframes, fileLowRank);
    
    fclose( fileLowRank );
    
    if ((fileSparse = fopen("sparse.bin", "wb")) == NULL){
        printf("sparse file can not open\n");
        exit(1);
    }
    
    fwrite(h_SP, sizeof(double), m * n * nframes, fileSparse);
    
    fclose( fileSparse );
    
    // print to screen
    cout << "row \t col \t k \t time \t iter \n" << endl;
    cout << m << " \t " << n  << " \t " << k << " \t "
    << scientific << setprecision(3) << rpca_time << " \t " << iter << " \t "
    <<
    endl;
    
    // save test result on disk
    fstream fs;
    fs.open("RPCA_double.csv", fstream::out | fstream::app);
    fs << m <<","<< n << "," << k << "," << "," << iter << ","
    << rpca_time << endl;
    
    fs.close();
    
    // clean up
    CHECK_CUDA( cudaFreeHost(h_M) );
    CHECK_CUDA( cudaFreeHost(h_LO) );
    CHECK_CUDA( cudaFreeHost(h_SP) );
    CHECK_CUDA( cudaFree( d_M )  );
    CHECK_CUDA( cudaFree( d_LO ) );
    CHECK_CUDA( cudaFree( d_SP ) );
    
}

int main(int argc, char **argv){
    
    assert((argc >= 4) && "Please supply fileName, m, n, k and numFrame as the argument");
    
    uint64_t m       = atoi(argv[2]);
    uint64_t n       = atoi(argv[3]);
    uint64_t k       = atoi(argv[4]);
    uint64_t nframes = atoi(argv[5]);
    
    backgroundSubstract(argv[1], m, n, k, nframes);
    
    CHECK_CUDA(cudaDeviceReset());
    
    return 0;
    
}
