#include "quantization.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

template <int D>
__global__ void kv_reconstruct_kernel(
    const int8_t* __restrict__ Xq,     // [B,H,S,D]
    const __half* __restrict__ scales, // [B,H,S, D/block_size]
    __half*       __restrict__ Y,      // [B,H,S,D]
    int B, int H, int S, int block_size)
{
    static_assert(D == 64 || D == 128, "D must be 64 or 128");

    const long long row = blockIdx.x;
    const long long total_rows = static_cast<long long>(B) * H * S;
    if (row >= total_rows) return;

    const int tid = threadIdx.x;
    const int sub = tid / block_size;
    const int n_sub = D / block_size;

    const int8_t* q_row = Xq + row * D;
    const __half* s_row = scales + row * n_sub;
    __half*       y_row = Y + row * D;

    float qv = static_cast<float>(q_row[tid]);
    float sv = __half2float(s_row[sub]);
    y_row[tid] = __float2half(qv * sv);
}

} // namespace

void launch_kv_reconstruct(
    const int8_t* dXq, const __half* dScales, __half* dY,
    int B, int H, int S, int D, int block_size,
    cudaStream_t stream)
{
    const bool valid_d  = (D == 64 || D == 128);
    const bool valid_bs = (block_size == 32 || block_size == 64 || block_size == 128);
    if (!valid_d || !valid_bs || (D % block_size != 0)) {
        std::fprintf(stderr,
            "launch_kv_reconstruct: unsupported (D=%d, block_size=%d).\n",
            D, block_size);
        std::abort();
    }

    const long long total_rows = static_cast<long long>(B) * H * S;
    const int grid = static_cast<int>(total_rows);

    if (D == 64) {
        kv_reconstruct_kernel<64><<<grid, 64, 0, stream>>>(
            dXq, dScales, dY, B, H, S, block_size);
    } else {
        kv_reconstruct_kernel<128><<<grid, 128, 0, stream>>>(
            dXq, dScales, dY, B, H, S, block_size);
    }
}
