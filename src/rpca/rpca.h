#ifndef _RPCA_H
#define _RPCA_H

void rpca(double *LO, double *SP, const double *M, unsigned int &iter,
          const int m, const int n, const int k,
          const double tol, const unsigned int maxit);

void shrink_singluar_value(double *d_SV, double *h_SV,
                           const int svMax,
                           const double mu, int &svSize);

void shrinkage(double *SP, double *A, const double *LO,
               const double *Y, const double *M,
               const double lambda, const double mu,
               const int m, const int n);

void update(double *Y, double *Z, double *SP, double *A,
            const double *LO, const double *M,
            const double lambda, const double mu,
            const int m, const int n);


// for data type transfer
void castChar2DoubleGPU(double *out, const unsigned char *in,
                        const unsigned long long maxSize);

void castDouble2SingleGPU(float *out, const double *in,
                          const unsigned long long maxSize);


#endif // _RPCA_H
