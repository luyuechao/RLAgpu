#!/usr/bin/env python
import subprocess

subprocess.call(["rm", "RSVD_double.csv"])

program = './build/RLAgpuTest'
M = 256
#N = 32
#M = 8192
#N = 8192
T = pow(2,24)
N = T / M
K = 64      # low-rank
S = 0.00    # sparsity

print ("Randomized SVD VS SVD in double-precison:");
print ("Size[GB] m \t n \t k \t In_Core[s] \t CublasXt[s] \t My_OoC[s] \t In_Core_Err \t My_OoC_Err \t CublasXt_Err \t SingularValueDiff");
print ("---------------------------------------------------------------------------------------------------------------------------------");

for i in range(1, 11):
    args=[program, str(M), str(N), str(K), str(S)]
    subprocess.call(args)
    #M = M * 2 + 1
    M = M * 2
    N = T / M
print ("finished testing, check the result in RSVD_double.csv")
