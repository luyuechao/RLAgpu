CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)

PROJECT(RPCAgpuTest)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

FIND_PACKAGE(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS}
                    ../../src
                    ../../src/rpca              
		    )

FILE(GLOB SOURCES "src/*.cpp")

ADD_EXECUTABLE(RPCAgpuTest ${SOURCES})

#for host compiler
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -msse2 -msse3 -Wall -Wno-unused-local-typedefs")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -gencode arch=compute_52,code=sm_52 -Xptxas -dlcm=cg)


add_library(RLAGPU STATIC IMPORTED) # or SHARED

set_target_properties(RLAGPU PROPERTIES
  IMPORTED_LOCATION "../../../build/libRLAgpu.a"
)

target_link_libraries(RPCAgpuTest 
                                  RLAGPU
                                  ${CUDA_LIBRARIES}
                                  ${CUDA_cublas_LIBRARY} 
                                  ${CUDA_cusolver_LIBRARY}
                                  ${CUDA_curand_LIBRARY}
                    )
