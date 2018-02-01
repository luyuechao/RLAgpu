#!/usr/bin/env python
import subprocess

program = './build/RPCAgpuTest'
data = './data/rawDouble.bin'

m = 1920
n = 1080
k = 10
numFrames = 100

args=[program, data, str(m), str(n), str(k), str(numFrames)]
subprocess.call(args)

print ("finished testing dataset 1")
