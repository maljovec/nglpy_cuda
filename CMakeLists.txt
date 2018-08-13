cmake_minimum_required(VERSION 3.0)
project(nglcu)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -I include/)

cuda_add_library(ngl_cuda SHARED src/ngl_cuda.cu)
cuda_add_executable(test_nglcu src/test.cpp)
target_link_libraries(test_nglcu nglcu)
