#!/usr/bin/env python
import subprocess
import os
import math
subprocess.call(["rm", "RSVD_double.csv", "OoC_colSampling.csv", "OoC_rowSampling.csv"])

program = './build/RLAgpuTest'

T = pow(2,26)
I = pow(2,30)

K = 16      # low-rank
N = 32 * K
M = 32 * N

S = 0.0    # sparsity


print ("Randomized SVD VS cublasXt SVD in double-precison:");
print ("Size[GB] m \t n \t k \t In_Core[s] \t CublasXt[s] \t My_OoC[s] \t In_Core_Err \t CublasXt_Err \t My_OoC_Err \t SingularValueDiff");
print ("---------------------------------------------------------------------------------------------------------------------------------");

for i in range(1, 122):
    N = math.sqrt(T / 32)
    M = N * 32
    K = N / 32
    args=[program, str(M), str(N), str(K), str(S)]
    subprocess.call(args)
    T = T + I

print ("finished testing, check the result in RSVD_double.csv")
