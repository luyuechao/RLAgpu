# RLAgpu

CUDA implementation of randomized singular value decomposition (rSVD).

# Requirements
## Hardware
* Nvidia CUDA compatiable GPU (computation ability > 2.0)

## OS
* Ubuntu 14.04 & CentOS 7.2 & Mac OS X 10.11
* Windows has not been tested yet.

# Install

## Dependnecy
* [CMake] (https://cmake.org/download)
* [CUDA >= 7.5](https://developer.nvidia.com/cuda-downloads)

## Ubuntu 14.04 / CentOS 7.2 / Mac OS X 10.11
* Step 1:
Install CMake, CUDA.

* Step 2:
    1. git clone https://github.com/luyuechao/RLAgpu
    2. In the project directory, edit CMakeList.txt file.
    
     change "arch" and "code" compiler flag according to your GPU architecture.
       Refer to https://en.wikipedia.org/wiki/CUDA
     (e.g. GTX TITAN X Maxwell (Compute ability 5.2) Set "arch" and "code" to "arch=compute_52" and "code=sm_52")
    
```
> mkdir build && cd build
> cmake ..
> make -j4
> sudo make install
> cd ../example/RSVD_C && mkdir build
> cd build && cmake .. && make -j4
> cd ..&&./run_me.py

```

# Performance Comparsion
see [WiKi](https://github.com/luyuechao/RLAgpu/wiki)

