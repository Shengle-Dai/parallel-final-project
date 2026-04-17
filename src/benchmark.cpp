#include "attention.h"
#include "io_util.h"
#include "kv_cache.h"
#include "quantization.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do {                                                     \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                     __FILE__, __LINE__, cudaGetErrorString(_e));              \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

struct Stats {
    float mean_us;
    float std_us;
    float min_us;
};

template <typename Launch>
static Stats time_event_loop(int warmup, int iters, cudaEvent_t e0, cudaEvent_t e1, Launch&& launch) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> ts(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(e0));
        launch();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        ts[i] = ms * 1000.0f;  // us
    }
    double sum = 0.0, sumsq = 0.0;
    float  mn  = 1e30f;
    for (float t : ts) { sum += t; sumsq += static_cast<double>(t) * t; mn = std::min(mn, t); }
    double mean = sum / iters;
    double var  = sumsq / iters - mean * mean;
    Stats out;
    out.mean_us = static_cast<float>(mean);
    out.std_us  = static_cast<float>(std::sqrt(std::max(0.0, var)));
    out.min_us  = mn;
    return out;
}

int main(int argc, char** argv) {
    std::string csv_path = "results.csv";
    int warmup = 5;
    int iters  = 50;
    std::string load_kv_dir;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next_arg = [&](){
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--csv")           csv_path = next_arg();
        else if (a == "--warmup")        warmup   = std::atoi(next_arg());
        else if (a == "--iters")         iters    = std::atoi(next_arg());
        else if (a == "--load-kv-from")  load_kv_dir = next_arg();
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--csv FILE] [--warmup N] [--iters N]\n"
                        "          [--load-kv-from DIR]   (single-shape run on captured K/V)\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    std::vector<int> Bs = {1, 4};
    std::vector<int> Hs = {16, 32};
    std::vector<int> Ds = {64, 128};
    std::vector<int> Ss = {512, 1024, 2048, 4096};
    const std::vector<int> BSs = {32, 64, 128};

    // If --load-kv-from is set, override the sweep grid to a single shape from
    // the captured meta.json. The mode loops below still iterate over fp16 /
    // int8_nonfused / int8_fused / int8_fused_online — only the (B,H,D,S) axis
    // collapses to the captured shape.
    if (!load_kv_dir.empty()) {
        const std::string meta = tqkv_slurp(load_kv_dir + "/meta.json");
        Bs = { tqkv_read_meta_int(meta, "B") };
        Hs = { tqkv_read_meta_int(meta, "H") };
        Ds = { tqkv_read_meta_int(meta, "D") };
        Ss = { tqkv_read_meta_int(meta, "S") };
        std::printf("loading K/V from %s: B=%d H=%d D=%d S=%d\n",
                    load_kv_dir.c_str(), Bs[0], Hs[0], Ds[0], Ss[0]);
    }

    std::ofstream csv(csv_path);
    csv << "mode,B,H,D,S,block_size,iters,attn_us,reconstruct_us,compress_us,total_us,std_us,min_us,bytes_kv,tile_n,lanes_per_thread\n";

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int B : Bs)
    for (int H : Hs)
    for (int D : Ds)
    for (int S : Ss) {
        AttnShape sh{B, H, S, D};

        __half *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
        CUDA_CHECK(cudaMalloc(&dQ, sh.q_elems() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dK, sh.k_elems() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dV, sh.v_elems() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dO, sh.o_elems() * sizeof(__half)));

        if (load_kv_dir.empty()) {
            size_t max_elems = std::max({sh.q_elems(), sh.k_elems(), sh.v_elems()});
            std::vector<__half> staging(max_elems);
            for (auto& x : staging) x = __float2half(dist(rng) * 0.1f);
            CUDA_CHECK(cudaMemcpy(dQ, staging.data(), sh.q_elems() * sizeof(__half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dK, staging.data(), sh.k_elems() * sizeof(__half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dV, staging.data(), sh.v_elems() * sizeof(__half), cudaMemcpyHostToDevice));
        } else {
            std::vector<__half> hQ(sh.q_elems()), hK(sh.k_elems()), hV(sh.v_elems());
            tqkv_read_bin(load_kv_dir + "/Q.bin", hQ.data(), hQ.size() * sizeof(__half));
            tqkv_read_bin(load_kv_dir + "/K.bin", hK.data(), hK.size() * sizeof(__half));
            tqkv_read_bin(load_kv_dir + "/V.bin", hV.data(), hV.size() * sizeof(__half));
            CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), sh.q_elems() * sizeof(__half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dK, hK.data(), sh.k_elems() * sizeof(__half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dV, hV.data(), sh.v_elems() * sizeof(__half), cudaMemcpyHostToDevice));
        }

        // ---- fp16 baseline ----
        {
            Stats st = time_event_loop(warmup, iters, e0, e1, [&](){
                launch_decode_attn_baseline(dQ, dK, dV, dO, B, H, S, D, /*stream=*/0);
            });
            size_t bytes_kv = (sh.k_elems() + sh.v_elems()) * sizeof(__half);
            csv << "fp16," << B << "," << H << "," << D << "," << S << ","
                << 0 << "," << iters << ","
                << st.mean_us << ",0,0," << st.mean_us << ","
                << st.std_us << "," << st.min_us << "," << bytes_kv << ",0,0\n";
            csv.flush();
            std::printf("fp16              B=%d H=%d D=%d S=%-5d  attn=%8.2f us  min=%8.2f us\n",
                        B, H, D, S, st.mean_us, st.min_us);
        }

        // ---- int8_nonfused: one row per valid block_size ----
        for (int block_size : BSs) {
            if (block_size > D || (D % block_size != 0)) continue;

            const size_t kv_elems       = sh.k_elems();
            const size_t scales_per_row = static_cast<size_t>(D) / block_size;
            const size_t scale_elems    = static_cast<size_t>(B) * H * S * scales_per_row;

            int8_t *dKq = nullptr, *dVq = nullptr;
            __half *dKs = nullptr, *dVs = nullptr;
            __half *dKr = nullptr, *dVr = nullptr;
            CUDA_CHECK(cudaMalloc(&dKq, kv_elems    * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&dVq, kv_elems    * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&dKs, scale_elems * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&dVs, scale_elems * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&dKr, kv_elems    * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&dVr, kv_elems    * sizeof(__half)));

            // Compression cost: prefill-only, measured once outside the hot loop.
            Stats stc = time_event_loop(/*warmup=*/2, /*iters=*/5, e0, e1, [&](){
                launch_kv_compress(dK, dKq, dKs, B, H, S, D, block_size, 0);
                launch_kv_compress(dV, dVq, dVs, B, H, S, D, block_size, 0);
            });
            // Make sure dKq/dVq/dKs/dVs reflect a real compression before the
            // reconstruct + attn loop runs (the warmup runs already did this,
            // but be explicit).
            launch_kv_compress(dK, dKq, dKs, B, H, S, D, block_size, 0);
            launch_kv_compress(dV, dVq, dVs, B, H, S, D, block_size, 0);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Reconstruct timing.
            Stats str = time_event_loop(warmup, iters, e0, e1, [&](){
                launch_kv_reconstruct(dKq, dKs, dKr, B, H, S, D, block_size, 0);
                launch_kv_reconstruct(dVq, dVs, dVr, B, H, S, D, block_size, 0);
            });

            // Attn-on-reconstructed timing.
            Stats sta = time_event_loop(warmup, iters, e0, e1, [&](){
                launch_decode_attn_baseline(dQ, dKr, dVr, dO, B, H, S, D, 0);
            });

            float total_mean = str.mean_us + sta.mean_us;
            float total_min  = str.min_us  + sta.min_us;
            // For std, sum the variances (independent measurements).
            float total_std  = std::sqrt(str.std_us * str.std_us + sta.std_us * sta.std_us);

            size_t bytes_kv = 2 * kv_elems /*K+V int8*/ + 2 * scale_elems * sizeof(__half) /*K+V scales*/;
            csv << "int8_nonfused," << B << "," << H << "," << D << "," << S << ","
                << block_size << "," << iters << ","
                << sta.mean_us << "," << str.mean_us << "," << stc.mean_us << ","
                << total_mean << "," << total_std << "," << total_min << "," << bytes_kv << ",0,0\n";
            csv.flush();
            std::printf("int8_nf bs=%-3d   B=%d H=%d D=%d S=%-5d  attn=%8.2f us  reco=%8.2f us  total=%8.2f us\n",
                        block_size, B, H, D, S, sta.mean_us, str.mean_us, total_mean);

            cudaFree(dKq); cudaFree(dVq);
            cudaFree(dKs); cudaFree(dVs);
            cudaFree(dKr); cudaFree(dVr);
        }

        // ---- int8_fused: same compression artifacts, but the attention
        // kernel reads int8 + scales directly (no reconstruct kernel).
        for (int block_size : BSs) {
            if (block_size > D || (D % block_size != 0)) continue;

            const size_t kv_elems       = sh.k_elems();
            const size_t scales_per_row = static_cast<size_t>(D) / block_size;
            const size_t scale_elems    = static_cast<size_t>(B) * H * S * scales_per_row;

            int8_t *dKq = nullptr, *dVq = nullptr;
            __half *dKs = nullptr, *dVs = nullptr;
            CUDA_CHECK(cudaMalloc(&dKq, kv_elems    * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&dVq, kv_elems    * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&dKs, scale_elems * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&dVs, scale_elems * sizeof(__half)));

            Stats stc = time_event_loop(/*warmup=*/2, /*iters=*/5, e0, e1, [&](){
                launch_kv_compress(dK, dKq, dKs, B, H, S, D, block_size, 0);
                launch_kv_compress(dV, dVq, dVs, B, H, S, D, block_size, 0);
            });
            launch_kv_compress(dK, dKq, dKs, B, H, S, D, block_size, 0);
            launch_kv_compress(dV, dVq, dVs, B, H, S, D, block_size, 0);
            CUDA_CHECK(cudaDeviceSynchronize());

            Stats sta = time_event_loop(warmup, iters, e0, e1, [&](){
                launch_decode_attn_int8_fused(
                    dQ, dKq, dKs, dVq, dVs, dO, B, H, S, D, block_size, 0);
            });

            size_t bytes_kv = 2 * kv_elems + 2 * scale_elems * sizeof(__half);
            csv << "int8_fused," << B << "," << H << "," << D << "," << S << ","
                << block_size << "," << iters << ","
                << sta.mean_us << ",0," << stc.mean_us << ","
                << sta.mean_us << "," << sta.std_us << "," << sta.min_us << "," << bytes_kv << ",0,0\n";
            csv.flush();
            std::printf("int8_fu bs=%-3d   B=%d H=%d D=%d S=%-5d  attn=%8.2f us  min=%8.2f us\n",
                        block_size, B, H, D, S, sta.mean_us, sta.min_us);

            // ---- int8_fused_online: same compressed inputs, online-softmax kernel. ----
            // Headline config: D=128 ⇒ lanes=4 (single-warp block, uchar4 loads),
            // D=64 ⇒ lanes=2 (single-warp block, uchar2 loads). tile_n=32 keeps
            // register pressure low and gives ~50% theoretical occupancy.
            const int online_lanes  = (D == 128) ? 4 : 2;
            const int online_tile_n = 32;
            if (block_size >= online_lanes) {
                Stats sto = time_event_loop(warmup, iters, e0, e1, [&](){
                    launch_decode_attn_int8_fused_online(
                        dQ, dKq, dKs, dVq, dVs, dO, B, H, S, D, block_size,
                        online_lanes, online_tile_n, 0);
                });

                csv << "int8_fused_online," << B << "," << H << "," << D << "," << S << ","
                    << block_size << "," << iters << ","
                    << sto.mean_us << ",0," << stc.mean_us << ","
                    << sto.mean_us << "," << sto.std_us << "," << sto.min_us << "," << bytes_kv
                    << "," << online_tile_n << "," << online_lanes << "\n";
                csv.flush();
                std::printf("int8_fo bs=%-3d   B=%d H=%d D=%d S=%-5d  attn=%8.2f us  min=%8.2f us  (T=%d L=%d)\n",
                            block_size, B, H, D, S, sto.mean_us, sto.min_us,
                            online_tile_n, online_lanes);
            }

            cudaFree(dKq); cudaFree(dVq);
            cudaFree(dKs); cudaFree(dVs);
        }

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 0;
}
