CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)

PROJECT(RLAgpu)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})

FIND_PACKAGE(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS}
                    src
                   )

FILE(GLOB SOURCES "src/*.cu" "src/*.cpp" "src/rpca/*.cu" "src/rpca/*.cpp")

CUDA_ADD_LIBRARY(RLAgpu ${SOURCES})

LIST(APPEND CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -Wno-unused-local-typedefs")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-ftz=true;--prec-div=false;--prec-sqrt=false")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -gencode arch=compute_52,code=sm_52 -Xptxas -dlcm=cg)

target_link_libraries(RLAgpu ${CUDA_LIBRARIES} 
                             ${CUDA_cublas_LIBRARY} 
                             ${CUDA_cusolver_LIBRARY}
			     ${CUDA_curand_LIBRARY}
                      )

install (TARGETS RLAgpu DESTINATION bin)
install (FILES src/rsvd.h DESTINATION include)
