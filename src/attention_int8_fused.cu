#include "attention.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

namespace {

// Fused decode-attention with on-the-fly INT8 KV reconstruction.
// Mirrors decode_attn_baseline_kernel structurally — same grid (B,H), same
// blockDim D, same dynamic-shared-mem scores[S], same two-pass softmax.
// The only differences: passes 1 and 4 read int8 + per-block fp16 scale
// instead of fp16, and reconstruct in fp32 registers before the dot/weighted
// sum. No reconstructed KV is ever written to global memory.
template <int D, int BlockSize>
__global__ void decode_attn_int8_fused_kernel(
    const __half*  __restrict__ Q,         // [B,H,D]
    const int8_t*  __restrict__ Kq,        // [B,H,S,D]
    const __half*  __restrict__ K_scales,  // [B,H,S, D/BlockSize]
    const int8_t*  __restrict__ Vq,        // [B,H,S,D]
    const __half*  __restrict__ V_scales,  // [B,H,S, D/BlockSize]
    __half*        __restrict__ O,         // [B,H,D]
    int B, int H, int S,
    float scale)
{
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128,
                  "BlockSize must be 32, 64, or 128");
    static_assert(D % BlockSize == 0, "BlockSize must divide D");
    constexpr int N_WARPS = D / 32;
    constexpr int N_SUB   = D / BlockSize;

    extern __shared__ float smem[];
    float* scores = smem;                 // [S]
    __shared__ float redbuf[4];

    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int sub_idx = tid / BlockSize;
    const unsigned mask = 0xffffffffu;

    const __half*  qbh   = Q  + (static_cast<size_t>(b) * H + h) * D;
    const int8_t*  kqbh  = Kq + (static_cast<size_t>(b) * H + h) * S * D;
    const int8_t*  vqbh  = Vq + (static_cast<size_t>(b) * H + h) * S * D;
    const __half*  ksbh  = K_scales + (static_cast<size_t>(b) * H + h) * S * N_SUB;
    const __half*  vsbh  = V_scales + (static_cast<size_t>(b) * H + h) * S * N_SUB;
    __half*        obh   = O  + (static_cast<size_t>(b) * H + h) * D;

    const float q_val = __half2float(qbh[tid]);

    // ---- Pass 1: scores[t] = (Q . K[t]) * scale, with on-the-fly reconstruct.
    for (int t = 0; t < S; ++t) {
        const float k_scale = __half2float(ksbh[t * N_SUB + sub_idx]);
        const int8_t kq     = kqbh[t * D + tid];
        const float k_val   = static_cast<float>(kq) * k_scale;
        float prod = q_val * k_val;

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            prod += __shfl_xor_sync(mask, prod, off);
        }
        if (lane == 0) redbuf[warp_id] = prod;
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane < N_WARPS) ? redbuf[lane] : 0.0f;
            #pragma unroll
            for (int off = N_WARPS / 2; off > 0; off >>= 1) {
                v += __shfl_xor_sync(mask, v, off);
            }
            if (lane == 0) scores[t] = v * scale;
        }
        __syncthreads();
    }

    // ---- Pass 2: max over scores. Identical to baseline.
    float local_max = -INFINITY;
    for (int t = tid; t < S; t += D) {
        local_max = fmaxf(local_max, scores[t]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, off));
    }
    if (lane == 0) redbuf[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < N_WARPS) ? redbuf[lane] : -INFINITY;
        #pragma unroll
        for (int off = N_WARPS / 2; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(mask, v, off));
        }
        if (lane == 0) redbuf[0] = v;
    }
    __syncthreads();
    const float m = redbuf[0];

    // ---- Pass 3: exp + sum + normalize. Identical to baseline.
    float local_sum = 0.0f;
    for (int t = tid; t < S; t += D) {
        float e = __expf(scores[t] - m);
        scores[t] = e;
        local_sum += e;
    }
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_xor_sync(mask, local_sum, off);
    }
    if (lane == 0) redbuf[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < N_WARPS) ? redbuf[lane] : 0.0f;
        #pragma unroll
        for (int off = N_WARPS / 2; off > 0; off >>= 1) {
            v += __shfl_xor_sync(mask, v, off);
        }
        if (lane == 0) redbuf[0] = v;
    }
    __syncthreads();
    const float inv_Z = 1.0f / redbuf[0];

    for (int t = tid; t < S; t += D) {
        scores[t] *= inv_Z;
    }
    __syncthreads();

    // ---- Pass 4: weighted V sum, with on-the-fly reconstruct.
    float out = 0.0f;
    for (int t = 0; t < S; ++t) {
        const float v_scale = __half2float(vsbh[t * N_SUB + sub_idx]);
        const int8_t vq     = vqbh[t * D + tid];
        const float v_val   = static_cast<float>(vq) * v_scale;
        out += scores[t] * v_val;
    }
    obh[tid] = __float2half(out);
}

template <int D, int BlockSize>
static void launch_one(
    const __half* dQ,
    const int8_t* dKq, const __half* dKs,
    const int8_t* dVq, const __half* dVs,
    __half* dO,
    int B, int H, int S, float scale, cudaStream_t stream)
{
    dim3 grid(B, H);
    size_t smem_bytes = static_cast<size_t>(S) * sizeof(float);
    decode_attn_int8_fused_kernel<D, BlockSize><<<grid, D, smem_bytes, stream>>>(
        dQ, dKq, dKs, dVq, dVs, dO, B, H, S, scale);
}

} // namespace

void launch_decode_attn_int8_fused(
    const __half* dQ,
    const int8_t* dKq, const __half* dKs,
    const int8_t* dVq, const __half* dVs,
    __half* dO,
    int B, int H, int S, int D, int block_size,
    cudaStream_t stream)
{
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    if (D == 64) {
        if      (block_size == 32) launch_one<64, 32>(dQ, dKq, dKs, dVq, dVs, dO, B, H, S, scale, stream);
        else if (block_size == 64) launch_one<64, 64>(dQ, dKq, dKs, dVq, dVs, dO, B, H, S, scale, stream);
        else goto bad;
    } else if (D == 128) {
        if      (block_size == 32)  launch_one<128, 32 >(dQ, dKq, dKs, dVq, dVs, dO, B, H, S, scale, stream);
        else if (block_size == 64)  launch_one<128, 64 >(dQ, dKq, dKs, dVq, dVs, dO, B, H, S, scale, stream);
        else if (block_size == 128) launch_one<128, 128>(dQ, dKq, dKs, dVq, dVs, dO, B, H, S, scale, stream);
        else goto bad;
    } else {
        goto bad;
    }
    return;

bad:
    std::fprintf(stderr,
        "launch_decode_attn_int8_fused: unsupported (D=%d, block_size=%d). "
        "Supported: D in {64,128}, block_size in {32,64,128}, block_size <= D.\n",
        D, block_size);
    std::abort();
}
