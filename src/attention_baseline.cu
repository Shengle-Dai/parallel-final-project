#include "attention.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace {

// One block per (batch, head). blockDim.x == D. D in {64, 128}.
// Dynamic shared mem holds float scores[S].
template <int D>
__global__ void decode_attn_baseline_kernel(
    const __half* __restrict__ Q,   // [B,H,D]
    const __half* __restrict__ K,   // [B,H,S,D]
    const __half* __restrict__ V,   // [B,H,S,D]
    __half* __restrict__       O,   // [B,H,D]
    int B, int H, int S,
    float scale)
{
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    constexpr int N_WARPS = D / 32;

    extern __shared__ float smem[];
    float* scores = smem;                       // [S]
    __shared__ float redbuf[4];                 // up to N_WARPS entries (max 4)

    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    const __half* qbh = Q + (static_cast<size_t>(b) * H + h) * D;
    const __half* kbh = K + (static_cast<size_t>(b) * H + h) * S * D;
    const __half* vbh = V + (static_cast<size_t>(b) * H + h) * S * D;
    __half*       obh = O + (static_cast<size_t>(b) * H + h) * D;

    // Each thread holds one Q lane.
    const float q_val = __half2float(qbh[tid]);
    const unsigned mask = 0xffffffffu;

    // Pass 1: scores[t] = (Q . K[t]) * scale.
    for (int t = 0; t < S; ++t) {
        float k_val = __half2float(kbh[t * D + tid]);
        float prod  = q_val * k_val;

        // Warp reduction.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            prod += __shfl_xor_sync(mask, prod, off);
        }
        if (lane == 0) redbuf[warp_id] = prod;
        __syncthreads();

        // Final reduction across warps (in warp 0).
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

    // Pass 2: max over scores.
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

    // Pass 3: scores[t] = exp(scores[t] - m); accumulate Z.
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

    // Normalize scores in place: scores[t] becomes the softmax probability.
    for (int t = tid; t < S; t += D) {
        scores[t] *= inv_Z;
    }
    __syncthreads();

    // Pass 4: out[lane] = sum_t scores[t] * V[t, lane].
    float out = 0.0f;
    for (int t = 0; t < S; ++t) {
        float vv = __half2float(vbh[t * D + tid]);
        out += scores[t] * vv;
    }
    obh[tid] = __float2half(out);
}

} // namespace

void launch_decode_attn_baseline(
    const __half* dQ,
    const __half* dK,
    const __half* dV,
    __half* dO,
    int B, int H, int S, int D,
    cudaStream_t stream)
{
    dim3 grid(B, H);
    size_t smem_bytes = static_cast<size_t>(S) * sizeof(float);
    float scale = 1.0f / sqrtf(static_cast<float>(D));

    if (D == 64) {
        decode_attn_baseline_kernel<64><<<grid, 64, smem_bytes, stream>>>(
            dQ, dK, dV, dO, B, H, S, scale);
    } else if (D == 128) {
        decode_attn_baseline_kernel<128><<<grid, 128, smem_bytes, stream>>>(
            dQ, dK, dV, dO, B, H, S, scale);
    } else {
        std::fprintf(stderr, "launch_decode_attn_baseline: D must be 64 or 128 (got %d)\n", D);
        std::abort();
    }
}
