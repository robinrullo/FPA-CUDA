cmake_minimum_required(VERSION 3.30)
project(CudaFPA LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

# GTX1050 - Architecture 6.1 (sm_61)
set(CMAKE_CUDA_ARCHITECTURES "native")

add_executable(CudaFPA
        main.cpp
        kernel.cu
        kernel.cuh
        kernel.cpp
)
set_target_properties(CudaFPA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
