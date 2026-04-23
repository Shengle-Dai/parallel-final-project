#include "quantization.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

// One block per (b, h, t) row. blockDim.x == D. Threads with the same
// sub-block index (tid / BlockSize) cooperatively reduce max(|x|).
template <int D, int BlockSize>
__global__ void kv_compress_kernel(
    const __half* __restrict__ X,       // [B,H,S,D]
    int8_t*       __restrict__ Xq,      // [B,H,S,D]
    __half*       __restrict__ scales,  // [B,H,S, D/BlockSize]
    int B, int H, int S)
{
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    static_assert(BlockSize == 32 || BlockSize == 64 || BlockSize == 128,
                  "BlockSize must be 32, 64, or 128");
    static_assert(D % BlockSize == 0, "BlockSize must divide D");
    constexpr int N_SUB         = D / BlockSize;            // sub-blocks per row
    constexpr int WARPS_PER_SUB = (BlockSize + 31) / 32;    // 1, 2, or 4

    const long long row = blockIdx.x;        // 0 .. B*H*S - 1
    const long long total_rows = static_cast<long long>(B) * H * S;
    if (row >= total_rows) return;

    const int tid = threadIdx.x;
    const int sub = tid / BlockSize;         // which sub-block this thread is in
    const int sub_lane = tid % BlockSize;    // position within the sub-block
    const int warp_in_sub = sub_lane >> 5;   // 0..WARPS_PER_SUB-1
    const int lane_in_warp = sub_lane & 31;
    const unsigned mask = 0xffffffffu;

    const __half* x_row = X  + row * D;
    int8_t*       q_row = Xq + row * D;
    __half*       s_row = scales + row * N_SUB;

    // Load value, take fp32 abs.
    float xv = __half2float(x_row[tid]);
    float a  = fabsf(xv);

    // Stage 1: reduce within each warp.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(mask, a, off));
    }
    // Now lane 0 of each warp holds the warp-max of |x|.

    // Stage 2: reduce across warps within the same sub-block (only if BlockSize > 32).
    float sub_max;
    if constexpr (WARPS_PER_SUB == 1) {
        sub_max = __shfl_sync(mask, a, 0);   // broadcast warp-0 max to all lanes
    } else {
        // One shared-mem slot per warp. Using D/32 slots (max 4 for D=128).
        __shared__ float warp_max[D / 32];
        if (lane_in_warp == 0) {
            warp_max[(sub * WARPS_PER_SUB) + warp_in_sub] = a;
        }
        __syncthreads();
        // First WARPS_PER_SUB lanes of the sub-block read; reduce via shuffle.
        float v = (sub_lane < WARPS_PER_SUB)
                  ? warp_max[(sub * WARPS_PER_SUB) + sub_lane]
                  : 0.0f;
        #pragma unroll
        for (int off = WARPS_PER_SUB / 2; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(mask, v, off));
        }
        // Broadcast lane 0's value within the sub-block. The whole warp shares
        // the same `v` after the shuffle reduction, so a sub-warp broadcast is
        // a single shfl. For BlockSize==64 sub spans 2 warps; we pull through
        // shared mem to be safe.
        __shared__ float sub_max_buf[N_SUB];
        if (sub_lane == 0) sub_max_buf[sub] = v;
        __syncthreads();
        sub_max = sub_max_buf[sub];
    }

    // Compute scale (fp32). All-zero block ⇒ s = 1 so quantized values are 0.
    float s = sub_max / 127.0f;
    if (s == 0.0f) s = 1.0f;

    // Thread 0 of each sub-block writes the scale.
    if (sub_lane == 0) {
        s_row[sub] = __float2half(s);
    }

    // Quantize.
    float scaled = xv / s;
    int   qi     = __float2int_rn(scaled);
    if (qi >  127) qi =  127;
    if (qi < -127) qi = -127;
    q_row[tid] = static_cast<int8_t>(qi);
}

template <int D, int BlockSize>
static void launch_one(
    const __half* dX, int8_t* dXq, __half* dScales,
    int B, int H, int S, cudaStream_t stream)
{
    const long long total_rows = static_cast<long long>(B) * H * S;
    const int grid = static_cast<int>(total_rows);
    kv_compress_kernel<D, BlockSize><<<grid, D, 0, stream>>>(
        dX, dXq, dScales, B, H, S);
}

} // namespace

void launch_kv_compress(
    const __half* dX, int8_t* dXq, __half* dScales,
    int B, int H, int S, int D, int block_size,
    cudaStream_t stream)
{
    if (D == 64) {
        if      (block_size == 32) launch_one<64, 32>(dX, dXq, dScales, B, H, S, stream);
        else if (block_size == 64) launch_one<64, 64>(dX, dXq, dScales, B, H, S, stream);
        else goto bad;
    } else if (D == 128) {
        if      (block_size == 32)  launch_one<128, 32 >(dX, dXq, dScales, B, H, S, stream);
        else if (block_size == 64)  launch_one<128, 64 >(dX, dXq, dScales, B, H, S, stream);
        else if (block_size == 128) launch_one<128, 128>(dX, dXq, dScales, B, H, S, stream);
        else goto bad;
    } else {
        goto bad;
    }
    return;

bad:
    std::fprintf(stderr,
        "launch_kv_compress: unsupported (D=%d, block_size=%d). "
        "Supported: D in {64,128}, block_size in {32,64,128}, block_size <= D.\n",
        D, block_size);
    std::abort();
}
