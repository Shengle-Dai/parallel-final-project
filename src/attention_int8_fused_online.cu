#include "attention.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

namespace {

// Fused INT8 decode-attention with online softmax (FlashAttention-style) and
// vectorised K/V loads. One block per (batch, head); each thread owns L
// consecutive head-dim lanes (uchar4 at L=4, uchar2 at L=2). Sync count per
// tile is 1 for single-warp blocks, 2 for two-warp blocks.
template <int D, int BlockSize, int LanesPerThread, int TileN>
__global__ void decode_attn_int8_fused_online_kernel(
    const __half*  __restrict__ Q,
    const int8_t*  __restrict__ Kq,
    const __half*  __restrict__ K_scales,
    const int8_t*  __restrict__ Vq,
    const __half*  __restrict__ V_scales,
    __half*        __restrict__ O,
    int B, int H, int S,
    float scale)
{
    static_assert(D == 64 || D == 128);
    static_assert(LanesPerThread == 2 || LanesPerThread == 4);
    static_assert(TileN == 32 || TileN == 64);
    static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128);
    static_assert(D % LanesPerThread == 0);
    static_assert(D % BlockSize == 0);
    static_assert(BlockSize >= LanesPerThread);

    constexpr int ThreadsPerBlock = D / LanesPerThread;
    constexpr int WarpsPerBlock   = ThreadsPerBlock / 32;
    constexpr int SubBlocksPerRow = D / BlockSize;
    static_assert(WarpsPerBlock == 1 || WarpsPerBlock == 2);

    __shared__ float s_tile[TileN];
    __shared__ float warp_sums[WarpsPerBlock][TileN];

    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_in_warp = tid & 31;
    const int warp_id      = tid >> 5;
    const int base_lane    = tid * LanesPerThread;
    const unsigned mask    = 0xffffffffu;

    const __half* qbh  = Q        + (static_cast<size_t>(b) * H + h) * D;
    const int8_t* kqbh = Kq       + (static_cast<size_t>(b) * H + h) * S * D;
    const int8_t* vqbh = Vq       + (static_cast<size_t>(b) * H + h) * S * D;
    const __half* ksbh = K_scales + (static_cast<size_t>(b) * H + h) * S * SubBlocksPerRow;
    const __half* vsbh = V_scales + (static_cast<size_t>(b) * H + h) * S * SubBlocksPerRow;
    __half*       obh  = O        + (static_cast<size_t>(b) * H + h) * D;

    float q_vals[LanesPerThread];
    #pragma unroll
    for (int j = 0; j < LanesPerThread; ++j) {
        q_vals[j] = __half2float(qbh[base_lane + j]);
    }
    const int sub_idx = base_lane / BlockSize;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[LanesPerThread];
    #pragma unroll
    for (int j = 0; j < LanesPerThread; ++j) acc[j] = 0.0f;

    const int n_tiles = (S + TileN - 1) / TileN;
    for (int tile = 0; tile < n_tiles; ++tile) {
        const int t0      = tile * TileN;
        const int t_count = (S - t0 < TileN) ? (S - t0) : TileN;

        #pragma unroll
        for (int i = 0; i < TileN; ++i) {
            float p = 0.0f;
            if (i < t_count) {
                const int t = t0 + i;
                const int8_t* kp = kqbh + t * D + base_lane;
                int8_t kbytes[LanesPerThread];
                if constexpr (LanesPerThread == 4) {
                    uchar4 v = *reinterpret_cast<const uchar4*>(kp);
                    kbytes[0] = static_cast<int8_t>(v.x);
                    kbytes[1] = static_cast<int8_t>(v.y);
                    kbytes[2] = static_cast<int8_t>(v.z);
                    kbytes[3] = static_cast<int8_t>(v.w);
                } else {
                    uchar2 v = *reinterpret_cast<const uchar2*>(kp);
                    kbytes[0] = static_cast<int8_t>(v.x);
                    kbytes[1] = static_cast<int8_t>(v.y);
                }
                const float ks = __half2float(ksbh[t * SubBlocksPerRow + sub_idx]);
                #pragma unroll
                for (int j = 0; j < LanesPerThread; ++j) {
                    p += q_vals[j] * static_cast<float>(kbytes[j]) * ks;
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                p += __shfl_xor_sync(mask, p, off);
            }
            if constexpr (WarpsPerBlock == 1) {
                if (lane_in_warp == 0) {
                    s_tile[i] = (i < t_count) ? p * scale : -INFINITY;
                }
            } else {
                if (lane_in_warp == 0) {
                    warp_sums[warp_id][i] = p;
                }
            }
        }

        if constexpr (WarpsPerBlock > 1) {
            __syncthreads();
            for (int i = tid; i < TileN; i += ThreadsPerBlock) {
                const float bsum = warp_sums[0][i] + warp_sums[1][i];
                s_tile[i] = (i < t_count) ? bsum * scale : -INFINITY;
            }
        }
        __syncthreads();

        float m_tile = -INFINITY;
        #pragma unroll
        for (int i = 0; i < TileN; ++i) {
            m_tile = fmaxf(m_tile, s_tile[i]);
        }
        const float m_new = fmaxf(m_i, m_tile);
        // tile 0: m_i = -INF -> alpha = 0; acc[] and l_i are still 0, so a no-op.
        const float alpha = __expf(m_i - m_new);

        float z_tile = 0.0f;
        #pragma unroll
        for (int i = 0; i < TileN; ++i) {
            z_tile += (i < t_count) ? __expf(s_tile[i] - m_new) : 0.0f;
        }

        #pragma unroll
        for (int j = 0; j < LanesPerThread; ++j) acc[j] *= alpha;
        l_i = l_i * alpha + z_tile;

        #pragma unroll
        for (int i = 0; i < TileN; ++i) {
            if (i >= t_count) continue;
            const int t = t0 + i;
            const int8_t* vp = vqbh + t * D + base_lane;
            int8_t vbytes[LanesPerThread];
            if constexpr (LanesPerThread == 4) {
                uchar4 v = *reinterpret_cast<const uchar4*>(vp);
                vbytes[0] = static_cast<int8_t>(v.x);
                vbytes[1] = static_cast<int8_t>(v.y);
                vbytes[2] = static_cast<int8_t>(v.z);
                vbytes[3] = static_cast<int8_t>(v.w);
            } else {
                uchar2 v = *reinterpret_cast<const uchar2*>(vp);
                vbytes[0] = static_cast<int8_t>(v.x);
                vbytes[1] = static_cast<int8_t>(v.y);
            }
            const float vs = __half2float(vsbh[t * SubBlocksPerRow + sub_idx]);
            const float p_i = __expf(s_tile[i] - m_new);
            #pragma unroll
            for (int j = 0; j < LanesPerThread; ++j) {
                acc[j] += p_i * static_cast<float>(vbytes[j]) * vs;
            }
        }

        m_i = m_new;
    }

    const float inv_Z = 1.0f / l_i;
    #pragma unroll
    for (int j = 0; j < LanesPerThread; ++j) {
        obh[base_lane + j] = __float2half(acc[j] * inv_Z);
    }
}

template <int D, int BlockSize, int LanesPerThread, int TileN>
static void launch_one(
    const __half* dQ,
    const int8_t* dKq, const __half* dKs,
    const int8_t* dVq, const __half* dVs,
    __half* dO,
    int B, int H, int S, float scale, cudaStream_t stream)
{
    dim3 grid(B, H);
    constexpr int ThreadsPerBlock = D / LanesPerThread;
    decode_attn_int8_fused_online_kernel<D, BlockSize, LanesPerThread, TileN>
        <<<grid, ThreadsPerBlock, 0, stream>>>(
            dQ, dKq, dKs, dVq, dVs, dO, B, H, S, scale);
}

} // namespace

void launch_decode_attn_int8_fused_online(
    const __half* dQ,
    const int8_t* dKq, const __half* dKs,
    const int8_t* dVq, const __half* dVs,
    __half* dO,
    int B, int H, int S, int D, int block_size,
    int lanes_per_thread, int tile_n,
    cudaStream_t stream)
{
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    #define LAUNCH(D_, BS_, L_, T_) \
        launch_one<D_, BS_, L_, T_>(dQ, dKq, dKs, dVq, dVs, dO, B, H, S, scale, stream)

    if (D == 128 && lanes_per_thread == 4 && tile_n == 32) {
        if      (block_size == 32)  { LAUNCH(128, 32,  4, 32); return; }
        else if (block_size == 64)  { LAUNCH(128, 64,  4, 32); return; }
        else if (block_size == 128) { LAUNCH(128, 128, 4, 32); return; }
    } else if (D == 128 && lanes_per_thread == 4 && tile_n == 64) {
        if      (block_size == 32)  { LAUNCH(128, 32,  4, 64); return; }
        else if (block_size == 64)  { LAUNCH(128, 64,  4, 64); return; }
        else if (block_size == 128) { LAUNCH(128, 128, 4, 64); return; }
    } else if (D == 128 && lanes_per_thread == 2 && tile_n == 32) {
        if      (block_size == 32)  { LAUNCH(128, 32,  2, 32); return; }
        else if (block_size == 64)  { LAUNCH(128, 64,  2, 32); return; }
        else if (block_size == 128) { LAUNCH(128, 128, 2, 32); return; }
    } else if (D == 128 && lanes_per_thread == 2 && tile_n == 64) {
        if      (block_size == 32)  { LAUNCH(128, 32,  2, 64); return; }
        else if (block_size == 64)  { LAUNCH(128, 64,  2, 64); return; }
        else if (block_size == 128) { LAUNCH(128, 128, 2, 64); return; }
    } else if (D == 64 && lanes_per_thread == 2 && tile_n == 32) {
        if      (block_size == 32)  { LAUNCH(64, 32, 2, 32); return; }
        else if (block_size == 64)  { LAUNCH(64, 64, 2, 32); return; }
    } else if (D == 64 && lanes_per_thread == 2 && tile_n == 64) {
        if      (block_size == 32)  { LAUNCH(64, 32, 2, 64); return; }
        else if (block_size == 64)  { LAUNCH(64, 64, 2, 64); return; }
    }
    #undef LAUNCH

    std::fprintf(stderr,
        "launch_decode_attn_int8_fused_online: unsupported "
        "(D=%d, block_size=%d, L=%d, T=%d)\n",
        D, block_size, lanes_per_thread, tile_n);
    std::abort();
}
