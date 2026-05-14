# FP16 baseline + INT8 (non-fused, fused, online-fused)

Standalone CUDA microbenchmark for studying GPU-parallel KV-cache compression.
Four decode paths are implemented:

1. **fp16** — uncompressed baseline (Phase 1).
2. **int8_nonfused** — blockwise INT8 compression with a separate reconstruct
   kernel that materializes fp16 K/V back to global memory before attention
   (Phase 2). Negative-result reference.
3. **int8_fused** — int8 + scales loaded directly inside the attention kernel,
   reconstructed in registers, never written back to global memory (Phase 4).
4. **int8_fused_online** — single-pass online-softmax fused INT8 kernel with
   on-the-fly scales (Phase 6). **Main systems contribution.**

## Requirements

- Linux + NVIDIA GPU (target sm_80 = Ampere, e.g. A100 / RTX 30xx)
- CUDA Toolkit 11.8+ (12.x recommended)
- CMake 3.18+
- Python 3.9+ with `torch` and `numpy`

The local Mac dev machine cannot build this — run on a cloud GPU box (Colab,
Lambda, RunPod, etc.).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Override the architecture if needed:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   # Ada (RTX 40xx)
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90   # Hopper (H100)
```

For kernel debugging:

```bash
cmake -S . -B build -DDEBUG_KERNEL=ON
```

## Correctness check

### FP16 baseline

```bash
./build/tqkv_decode --B 1 --H 4 --D 64 --S 128 --seed 0 --out /tmp/tq
python python/check_correctness.py /tmp/tq
```

Expected output ends with `PASS (max_abs < 1e-2)`. Repeat at `--D 128 --S 2048`
for a fuller check.

### INT8 non-fused

```bash
./build/tqkv_decode --B 2 --H 8 --D 128 --S 2048 \
    --mode int8_nonfused --block-size 64 --seed 1 --out /tmp/tq_q
python python/check_correctness.py /tmp/tq_q
```

Runs two checks — quantization round-trip residual and INT8-attention against
a torch reference computed on the dequantized KV. Both must pass.

### INT8 fused

```bash
./build/tqkv_decode --B 2 --H 8 --D 128 --S 2048 \
    --mode int8_fused --block-size 64 --seed 1 --out /tmp/tq_f
python python/check_correctness.py /tmp/tq_f
```

Same artifacts and checks as `int8_nonfused`; only the attention kernel
changes (reconstructs in registers).

`--block-size` must divide `D`. Valid: `D=64 → {32,64}`, `D=128 → {32,64,128}`.

## Benchmark sweep

The bench binary (built from `src/benchmark.cpp`) sweeps a grid of
`B × H × D × S` with warmup + timed launches per cell for all four modes
(FP16, INT8 non-fused, INT8 fused, INT8 fused-online) at block sizes
`32 / 64 / 128`.

```bash
./build/tqkv_bench --help
```

For a one-shot A100 pipeline (build + sweep + NCU capture):

```bash
bash scripts/run_a100.sh
```

## Real-KV end-to-end

Replay attention against real GPT-2 KV tensors captured from HuggingFace:

```bash
python python/capture_kv.py --out tq_real_kv/
python python/run_real_kv_bench.py --kv tq_real_kv/ --bin ./build/tqkv_bench
```

## NCU profiling

```bash
bash scripts/ncu_capture.sh ./build/tqkv_bench
python python/parse_ncu.py results/ncu/...
python scripts/ncu_extract_long.py results/ncu/...
```

## Plotting

```bash
python python/plot_results.py        # Figures 1-6
python python/plot_extras.py         # Figures 7-12
```

## Layout

| Path | Role |
|---|---|
| `include/attention.h`                | Decode kernel launcher signatures |
| `include/quantization.h`             | Compress + reconstruct launcher signatures |
| `include/kv_cache.h`                 | `AttnShape` helper |
| `include/io_util.h`                  | Binary I/O helpers |
| `src/attention_baseline.cu`          | FP16 single-token decode kernel |
| `src/kv_compress.cu`                 | Kernel A — FP16 → INT8 + per-block scales |
| `src/kv_reconstruct.cu`              | Kernel B — INT8 + scales → FP16 scratch |
| `src/attention_int8_fused.cu`        | Kernel C — fused INT8-reconstruct + attention |
| `src/attention_int8_fused_online.cu` | Kernel D — online-softmax fused INT8 (main contribution) |
| `src/main.cpp`                       | `tqkv_decode` correctness driver |
| `src/benchmark.cpp`                  | `tqkv_bench` sweep harness with CUDA-event timing |
| `python/check_correctness.py`        | PyTorch reference + error report |
| `python/capture_kv.py`               | GPT-2 KV-tensor capture |
| `python/run_real_kv_bench.py`        | End-to-end bench against captured KV |
| `python/parse_ncu.py`                | NCU profile parser (roofline + memory metrics) |
| `python/plot_results.py`             | Figures 1-6 |
| `python/plot_extras.py`              | Figures 7-12 |
| `scripts/run_a100.sh`                | One-shot A100 build + sweep pipeline |
| `scripts/ncu_capture.sh`             | Capture NCU profiles for each kernel |
| `scripts/ncu_extract_long.py`        | Long-form NCU metric extraction |
