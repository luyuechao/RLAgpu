#!/usr/bin/env python
import subprocess
import os
import math
subprocess.call(["rm", "RSVD_power.csv"])

program = './build/RLAgpuTest'

T = pow(2,27)
I = pow(2,30)

K = 16      # low-rank
N = 32 * K
M = 32 * N

S = 0.0    # sparsity


print (":");
print ("Size[GB] m \t n \t k \t p1[s] \t p1_err \t p2[s] \t p2_err \t p3[s] \t p3_err \t ");
print ("---------------------------------------------------------------------------------------------------------------------------------");

for i in range(1, 12):
    N = math.sqrt(T / 32)
    M = N * 32
    K = N / 32
    args=[program, str(M), str(N), str(K), str(S)]
    subprocess.call(args)
    T = T + I

print ("finished testing, check the result in RSVD_double.csv")
