cmake_minimum_required(VERSION 3.18)
project(turboquant_kv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(DEBUG_KERNEL "Build CUDA kernels with -G -lineinfo" OFF)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -lineinfo")
if(DEBUG_KERNEL)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(tqkv_kernels STATIC
    src/attention_baseline.cu
    src/kv_compress.cu
    src/kv_reconstruct.cu
    src/attention_int8_fused.cu
    src/attention_int8_fused_online.cu
)
set_target_properties(tqkv_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_executable(tqkv_decode src/main.cpp)
target_link_libraries(tqkv_decode PRIVATE tqkv_kernels CUDA::cudart)

add_executable(tqkv_bench src/benchmark.cpp)
target_link_libraries(tqkv_bench PRIVATE tqkv_kernels CUDA::cudart)
