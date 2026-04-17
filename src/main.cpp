#include "attention.h"
#include "io_util.h"
#include "kv_cache.h"
#include "quantization.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

#define CUDA_CHECK(x) do {                                                     \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                     __FILE__, __LINE__, cudaGetErrorString(_e));              \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

static const char* arg_or_die(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value after %s\n", argv[i]);
        std::exit(2);
    }
    return argv[++i];
}

int main(int argc, char** argv) {
    AttnShape sh{1, 4, 128, 64};
    uint32_t seed = 0;
    std::string out_dir = "tq_out";
    std::string mode = "fp16";        // "fp16" | "int8_nonfused" | "int8_fused" | "int8_fused_online"
    int block_size = 64;
    int tile_n = 32;
    int lanes_per_thread = 4;
    std::string load_kv_dir;          // if non-empty: read Q/K/V from this dir instead of RNG

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--B")             sh.B = std::atoi(arg_or_die(i, argc, argv));
        else if (a == "--H")             sh.H = std::atoi(arg_or_die(i, argc, argv));
        else if (a == "--D")             sh.D = std::atoi(arg_or_die(i, argc, argv));
        else if (a == "--S")             sh.S = std::atoi(arg_or_die(i, argc, argv));
        else if (a == "--seed")          seed = static_cast<uint32_t>(std::atoll(arg_or_die(i, argc, argv)));
        else if (a == "--out")           out_dir = arg_or_die(i, argc, argv);
        else if (a == "--mode")          mode = arg_or_die(i, argc, argv);
        else if (a == "--block-size")    block_size = std::atoi(arg_or_die(i, argc, argv));
        else if (a == "--tile-n")        tile_n = std::atoi(arg_or_die(i, argc, argv));
        else if (a == "--lanes")         lanes_per_thread = std::atoi(arg_or_die(i, argc, argv));
        else if (a == "--load-kv-from")  load_kv_dir = arg_or_die(i, argc, argv);
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--B N] [--H N] [--D N] [--S N] [--seed N] [--out DIR]\n"
                        "          [--mode {fp16,int8_nonfused,int8_fused,int8_fused_online}]\n"
                        "          [--block-size N] [--tile-n {32,64}] [--lanes {2,4}]\n"
                        "          [--load-kv-from DIR]   (read Q/K/V + shape from DIR instead of RNG)\n",
                        argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    if (!load_kv_dir.empty()) {
        const std::string meta = tqkv_slurp(load_kv_dir + "/meta.json");
        sh.B = tqkv_read_meta_int(meta, "B");
        sh.H = tqkv_read_meta_int(meta, "H");
        sh.D = tqkv_read_meta_int(meta, "D");
        sh.S = tqkv_read_meta_int(meta, "S");
    }

    if (sh.D != 64 && sh.D != 128) {
        std::fprintf(stderr, "D must be 64 or 128 (got %d)\n", sh.D);
        return 2;
    }
    if (mode != "fp16" && mode != "int8_nonfused" && mode != "int8_fused"
        && mode != "int8_fused_online") {
        std::fprintf(stderr,
            "--mode must be fp16, int8_nonfused, int8_fused, or int8_fused_online (got %s)\n",
            mode.c_str());
        return 2;
    }
    const bool int8_mode = (mode == "int8_nonfused" || mode == "int8_fused"
                            || mode == "int8_fused_online");
    if (int8_mode) {
        if (block_size != 32 && block_size != 64 && block_size != 128) {
            std::fprintf(stderr, "--block-size must be 32, 64, or 128 (got %d)\n", block_size);
            return 2;
        }
        if (sh.D % block_size != 0 || block_size > sh.D) {
            std::fprintf(stderr, "block_size (%d) must divide D (%d)\n", block_size, sh.D);
            return 2;
        }
    }
    if (mode == "int8_fused_online") {
        if (tile_n != 32 && tile_n != 64) {
            std::fprintf(stderr, "--tile-n must be 32 or 64 (got %d)\n", tile_n);
            return 2;
        }
        if (lanes_per_thread != 2 && lanes_per_thread != 4) {
            std::fprintf(stderr, "--lanes must be 2 or 4 (got %d)\n", lanes_per_thread);
            return 2;
        }
        if (sh.D == 64 && lanes_per_thread == 4) {
            std::fprintf(stderr,
                "D=64 with --lanes=4 yields a 16-thread block (half-warp); rejected\n");
            return 2;
        }
        if (block_size < lanes_per_thread) {
            std::fprintf(stderr,
                "block_size (%d) must be >= lanes_per_thread (%d)\n", block_size, lanes_per_thread);
            return 2;
        }
    }

    std::vector<__half> hQ(sh.q_elems());
    std::vector<__half> hK(sh.k_elems());
    std::vector<__half> hV(sh.v_elems());
    std::vector<__half> hO(sh.o_elems());

    if (load_kv_dir.empty()) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        auto fill = [&](std::vector<__half>& v) {
            for (auto& x : v) x = __float2half(dist(rng) * 0.1f);
        };
        fill(hQ);
        fill(hK);
        fill(hV);
    } else {
        tqkv_read_bin(load_kv_dir + "/Q.bin", hQ.data(), hQ.size() * sizeof(__half));
        tqkv_read_bin(load_kv_dir + "/K.bin", hK.data(), hK.size() * sizeof(__half));
        tqkv_read_bin(load_kv_dir + "/V.bin", hV.data(), hV.size() * sizeof(__half));
    }

    __half *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, sh.q_elems() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dK, sh.k_elems() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dV, sh.v_elems() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dO, sh.o_elems() * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), sh.q_elems() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), sh.k_elems() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), sh.v_elems() * sizeof(__half), cudaMemcpyHostToDevice));

    // Buffers used only by int8_nonfused.
    int8_t *dKq = nullptr, *dVq = nullptr;
    __half *dKs = nullptr, *dVs = nullptr;
    __half *dKr = nullptr, *dVr = nullptr;
    std::vector<int8_t> hKq, hVq;
    std::vector<__half> hKs, hVs;

    if (mode == "fp16") {
        launch_decode_attn_baseline(dQ, dK, dV, dO, sh.B, sh.H, sh.S, sh.D, /*stream=*/0);
        CUDA_CHECK(cudaGetLastError());
    } else {
        const size_t kv_elems       = sh.k_elems();
        const size_t scales_per_row = static_cast<size_t>(sh.D) / block_size;
        const size_t scale_elems    = static_cast<size_t>(sh.B) * sh.H * sh.S * scales_per_row;

        CUDA_CHECK(cudaMalloc(&dKq, kv_elems * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&dVq, kv_elems * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&dKs, scale_elems * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dVs, scale_elems * sizeof(__half)));

        launch_kv_compress(dK, dKq, dKs, sh.B, sh.H, sh.S, sh.D, block_size, 0);
        launch_kv_compress(dV, dVq, dVs, sh.B, sh.H, sh.S, sh.D, block_size, 0);
        CUDA_CHECK(cudaGetLastError());

        if (mode == "int8_nonfused") {
            CUDA_CHECK(cudaMalloc(&dKr, kv_elems * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&dVr, kv_elems * sizeof(__half)));

            launch_kv_reconstruct(dKq, dKs, dKr, sh.B, sh.H, sh.S, sh.D, block_size, 0);
            launch_kv_reconstruct(dVq, dVs, dVr, sh.B, sh.H, sh.S, sh.D, block_size, 0);
            CUDA_CHECK(cudaGetLastError());

            launch_decode_attn_baseline(dQ, dKr, dVr, dO, sh.B, sh.H, sh.S, sh.D, 0);
            CUDA_CHECK(cudaGetLastError());
        } else if (mode == "int8_fused") {
            launch_decode_attn_int8_fused(
                dQ, dKq, dKs, dVq, dVs, dO,
                sh.B, sh.H, sh.S, sh.D, block_size, 0);
            CUDA_CHECK(cudaGetLastError());
        } else {  // int8_fused_online
            launch_decode_attn_int8_fused_online(
                dQ, dKq, dKs, dVq, dVs, dO,
                sh.B, sh.H, sh.S, sh.D, block_size,
                lanes_per_thread, tile_n, 0);
            CUDA_CHECK(cudaGetLastError());
        }

        // Pull the int8 + scale tensors back for the Python side.
        hKq.resize(kv_elems);
        hVq.resize(kv_elems);
        hKs.resize(scale_elems);
        hVs.resize(scale_elems);
        CUDA_CHECK(cudaMemcpy(hKq.data(), dKq, kv_elems    * sizeof(int8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hVq.data(), dVq, kv_elems    * sizeof(int8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hKs.data(), dKs, scale_elems * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hVs.data(), dVs, scale_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, sh.o_elems() * sizeof(__half), cudaMemcpyDeviceToHost));

    std::filesystem::create_directories(out_dir);
    auto write_bin = [&](const std::string& name, const void* p, size_t bytes) {
        std::ofstream f(out_dir + "/" + name, std::ios::binary);
        f.write(reinterpret_cast<const char*>(p), bytes);
    };
    write_bin("Q.bin",      hQ.data(), hQ.size() * sizeof(__half));
    write_bin("K.bin",      hK.data(), hK.size() * sizeof(__half));
    write_bin("V.bin",      hV.data(), hV.size() * sizeof(__half));
    write_bin("O_cuda.bin", hO.data(), hO.size() * sizeof(__half));
    if (int8_mode) {
        write_bin("Kq.bin",       hKq.data(), hKq.size() * sizeof(int8_t));
        write_bin("Vq.bin",       hVq.data(), hVq.size() * sizeof(int8_t));
        write_bin("K_scales.bin", hKs.data(), hKs.size() * sizeof(__half));
        write_bin("V_scales.bin", hVs.data(), hVs.size() * sizeof(__half));
    }

    {
        std::ofstream f(out_dir + "/meta.json");
        f << "{\n"
          << "  \"B\": "          << sh.B << ",\n"
          << "  \"H\": "          << sh.H << ",\n"
          << "  \"D\": "          << sh.D << ",\n"
          << "  \"S\": "          << sh.S << ",\n"
          << "  \"seed\": "       << seed << ",\n"
          << "  \"dtype\": \"float16\",\n"
          << "  \"mode\": \""     << mode << "\",\n"
          << "  \"block_size\": " << (int8_mode ? block_size : 0);
        if (mode == "int8_fused_online") {
            f << ",\n  \"tile_n\": "          << tile_n
              << ",\n  \"lanes_per_thread\": " << lanes_per_thread;
        }
        f << "\n}\n";
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    if (dKq) cudaFree(dKq);
    if (dVq) cudaFree(dVq);
    if (dKs) cudaFree(dKs);
    if (dVs) cudaFree(dVs);
    if (dKr) cudaFree(dKr);
    if (dVr) cudaFree(dVr);

    std::printf("wrote tensors to %s (B=%d H=%d D=%d S=%d mode=%s",
                out_dir.c_str(), sh.B, sh.H, sh.D, sh.S, mode.c_str());
    if (int8_mode) std::printf(" block_size=%d", block_size);
    if (mode == "int8_fused_online") std::printf(" tile_n=%d lanes=%d", tile_n, lanes_per_thread);
    std::printf(")\n");
    return 0;
}
