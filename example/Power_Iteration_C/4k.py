#!/usr/bin/env python
import subprocess

subprocess.call(["rm", "RSVD_double.csv"])

program = './build/RLAgpuTest'
M = 4147200
#M = 8294400
N = 100
K = 10      # low-rank
S = 0.00   # sparsity

print ("Randomized SVD VS SVD in double-precison:");
print ("Size[GB] m \t n \t k \t In_Core[s] \t CublasXt[s] \t My_OoC[s] \t In_Core_Err \t My_OoC_Err \t CublasXt_Err \t SingularValueDiff");
print ("---------------------------------------------------------------------------------------------------------------------------------");

args=[program, str(M), str(N), str(K), str(S)]
subprocess.call(args)
print ("finished testing, check the result in RSVD_double.csv")
