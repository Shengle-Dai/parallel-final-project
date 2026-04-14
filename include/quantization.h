#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Symmetric blockwise INT8 quantization along the head-dim of a single token.
//
// Layout for both K and V cache tensors:
//   X      : __half  [B, H, S, D]
//   Xq     : int8_t  [B, H, S, D]
//   scales : __half  [B, H, S, D / block_size]
//
// Constraints: D in {64, 128}; block_size in {32, 64, 128}; D % block_size == 0.

void launch_kv_compress(
    const __half* dX,
    int8_t*       dXq,
    __half*       dScales,
    int B, int H, int S, int D, int block_size,
    cudaStream_t stream);

void launch_kv_reconstruct(
    const int8_t* dXq,
    const __half* dScales,
    __half*       dY,
    int B, int H, int S, int D, int block_size,
    cudaStream_t stream);
