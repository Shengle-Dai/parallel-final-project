#!/usr/bin/env bash
# Build, run the full sweep + real-KV + ncu profile, regenerate plots, bundle
# the artifacts into /tmp/tqkv_a100_<timestamp>.tar.gz.
#
# Usage:
#   scripts/run_a100.sh         # warmup=5 iters=50, ~3 min on A100
#   scripts/run_a100.sh quick   # warmup=2 iters=10, ~30 s smoke test
#   PY=/path/to/python scripts/run_a100.sh
#
# Requires: nvcc (CUDA 11.8+), cmake >= 3.18, python with torch, transformers,
# numpy, scipy, pandas, matplotlib. ncu (Nsight Compute) is optional.

set -euo pipefail

MODE="${1:-full}"
TS=$(date +%Y%m%d_%H%M%S)
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$MODE" = "quick" ]; then
    WARMUP=2; ITERS=10; REAL_KV_ITERS=10
else
    WARMUP=5; ITERS=50; REAL_KV_ITERS=30
fi

if ! command -v nvcc >/dev/null 2>&1; then
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        if [ -x "$d/nvcc" ]; then export PATH="$d:$PATH"; break; fi
    done
fi
command -v nvcc  >/dev/null 2>&1 || { echo "nvcc not found"  >&2; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "cmake not found" >&2; exit 1; }

PY="${PY:-python3}"
$PY -c "import torch, transformers, numpy, scipy, pandas, matplotlib" 2>/dev/null \
    || { echo "missing python deps; install with: $PY -m pip install -r requirements.txt pandas matplotlib" >&2
         exit 1; }

# Auto-detect compute capability from the live GPU.
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d '. ')
SM="${SM:-80}"

echo "[build] sm_${SM}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$SM" >/dev/null
cmake --build build -j

echo "[correctness]"
mkdir -p /tmp/tqkv_smoke
for cfg in "64 2" "128 2" "128 4"; do
    read D L <<< "$cfg"
    ./build/tqkv_decode --B 2 --H 8 --D "$D" --S 2048 \
        --mode int8_fused_online --tile-n 32 --lanes "$L" \
        --block-size 64 --seed 1 --out "/tmp/tqkv_smoke/${D}_${L}" >/dev/null
    $PY python/check_correctness.py "/tmp/tqkv_smoke/${D}_${L}" | tail -1
done

mkdir -p results
echo "[phase 6 sweep] warmup=$WARMUP iters=$ITERS"
./build/tqkv_bench --csv results/phase6_int8_fused_online.csv \
    --warmup "$WARMUP" --iters "$ITERS" | tail -8

echo "[phase 8 real-KV (GPT-2 S=1024)]"
$PY python/run_real_kv_bench.py --seq 1024 --layer -1 \
    --csv results/phase8_real_kv.csv \
    --warmup "$WARMUP" --iters "$REAL_KV_ITERS" | tail -10

echo "[ncu]"
if ! command -v ncu >/dev/null 2>&1; then
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/nvidia/nsight-compute/*/host/target-linux-x64; do
        if [ -x "$d/ncu" ]; then export PATH="$d:$PATH"; break; fi
    done
fi
if command -v ncu >/dev/null 2>&1; then
    NCU_RUN=""
    if ! ncu --target-processes all --section Occupancy --csv \
        ./build/tqkv_decode --B 1 --H 4 --D 64 --S 256 --mode fp16 \
        --out /tmp/tqkv_ncu_probe >/dev/null 2>&1; then
        if sudo -n true 2>/dev/null; then
            NCU_RUN="sudo -n env PATH=$PATH"
        else
            echo "  ncu needs admin and passwordless sudo isn't set up; skipping"
            NCU_RUN="SKIP"
        fi
    fi
    if [ "$NCU_RUN" != "SKIP" ]; then
        $NCU_RUN bash scripts/ncu_capture.sh ./build/tqkv_decode results/ncu
        if [ -n "$NCU_RUN" ]; then sudo -n chown -R "$USER:$USER" results/ncu; fi
        $PY scripts/ncu_extract_long.py results/ncu
        $PY python/parse_ncu.py results/ncu/*.csv > results/ncu/summary.md
    fi
else
    echo "  ncu not found; skipping"
fi

echo "[plots]"
{ head -1 results/phase6_int8_fused_online.csv;
  tail -n +2 results/phase6_int8_fused_online.csv;
  tail -n +2 results/phase8_real_kv.csv; } > results/all_modes_a100.csv
$PY python/plot_results.py --csv results/all_modes_a100.csv --out results/plots_a100
$PY python/plot_extras.py \
    --csv results/phase6_int8_fused_online.csv \
    --real-kv-csv results/phase8_real_kv.csv \
    --ncu-dir results/ncu \
    --out results/plots_a100

ARTIFACT="/tmp/tqkv_a100_${TS}.tar.gz"
TAR_FILES=(
    results/phase6_int8_fused_online.csv
    results/phase8_real_kv.csv
    results/all_modes_a100.csv
    results/plots_a100
)
[ -d results/ncu ] && TAR_FILES+=(results/ncu)
tar czf "$ARTIFACT" "${TAR_FILES[@]}"
SIZE=$(du -h "$ARTIFACT" | cut -f1)

HOST=$(hostname -f 2>/dev/null || hostname)
WHO=$(whoami)
echo
echo "Artifact: $ARTIFACT ($SIZE)"
echo "Download:  scp ${WHO}@${HOST}:${ARTIFACT} ."
