cmake_minimum_required(VERSION 2.8)

INCLUDE(CMakeForceCompiler)

project(rSVDmex)

set(CMAKE_MACOSX_RPATH 1)

find_package(Matlab COMPONENTS MX_LIBRARY REQUIRED)
find_package(CUDA REQUIRED)

MESSAGE("Matlab_LIBRARIES:  " ${Matlab_LIBRARIES})
MESSAGE("Matlab_MX_LIBRARY  :" ${Matlab_MX_LIBRARY})

SET(CMAKE_BUILD_TYPE "Release")

FILE(GLOB SOURCES "src/*.cu" "src/*.cpp")

include_directories( ${Matlab_INCLUDE_DIRS}
                     ${CUDA_INCLUDE_DIRS}
                   )

add_library(RLAGPU STATIC IMPORTED) # or SHARED
set_target_properties(RLAGPU PROPERTIES
  IMPORTED_LOCATION "../../../build/libRLAgpu.a"
)

matlab_add_mex(NAME rSVDmex 
               SRC  ${SOURCES}
               LINK_TO       
                               ${Matlab_MEX_LIBRARY}
                               ${Matlab_MX_LIBRARY}                               
                               ${CUDA_LIBRARIES} 
                               ${CUDA_cublas_LIBRARY} 
                               ${CUDA_cusolver_LIBRARY}
			   	${CUDA_curand_LIBRARY}
				RLAGPU

)
