cmake_minimum_required(VERSION 3.0)
project(nglcu)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30)

cuda_add_library(ngl_cuda SHARED ngl_cuda.cu)