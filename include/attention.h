#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

void launch_decode_attn_baseline(
    const __half* dQ,
    const __half* dK,
    const __half* dV,
    __half* dO,
    int B, int H, int S, int D,
    cudaStream_t stream);

// Fused INT8 reconstruct + decode-attention (Kernel C).
// Reads compressed K/V + per-block fp16 scales, reconstructs in registers,
// never materializes the full fp16 KV in global memory.
void launch_decode_attn_int8_fused(
    const __half* dQ,
    const int8_t* dKq, const __half* dKs,
    const int8_t* dVq, const __half* dVs,
    __half* dO,
    int B, int H, int S, int D, int block_size,
    cudaStream_t stream);

// Same I/O as launch_decode_attn_int8_fused, but the kernel runs a flash-
// attention-style online softmax over tiles of TileN tokens and each thread
// owns LanesPerThread head-dim lanes (uchar4 / uchar2 loads).
void launch_decode_attn_int8_fused_online(
    const __half* dQ,
    const int8_t* dKq, const __half* dKs,
    const int8_t* dVq, const __half* dVs,
    __half* dO,
    int B, int H, int S, int D, int block_size,
    int lanes_per_thread, int tile_n,
    cudaStream_t stream);
