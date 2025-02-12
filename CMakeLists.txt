cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "61") # GTX1050 - Architecture 6.1

project(CudaFPA2024_2025_02 LANGUAGES CUDA CXX)

include_directories(.)

add_executable(CudaFPA2024_2025_02
        kernel.cpp
        kernel.cu
        kernel.h
        main.cpp)
