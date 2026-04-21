#!/usr/bin/env bash
# Capture Nsight Compute metrics for the three decode-attention kernels at
# two representative shapes. Drives `tqkv_decode` (single-shape, single-launch
# binary) so ncu sees exactly one kernel launch per invocation.
#
# Usage:
#   scripts/ncu_capture.sh [BIN] [OUTDIR]
#
# Defaults: BIN=./build/tqkv_decode, OUTDIR=results/ncu
#
# After running, parse with:
#   python python/parse_ncu.py "$OUTDIR"/*.csv > "$OUTDIR"/summary.md

set -euo pipefail

BIN="${1:-./build/tqkv_decode}"
OUT="${2:-results/ncu}"

mkdir -p "$OUT"

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu (Nsight Compute) not found on PATH" >&2
    echo "On A100 boxes ncu typically lives in /usr/local/cuda/bin or similar." >&2
    exit 1
fi

# Capture a curated section list. --set full is overkill and slow; these four
# cover the metrics that drive the §7.3 sync model.
SECTIONS=(
    --section SchedulerStats
    --section WarpStateStats
    --section MemoryWorkloadAnalysis
    --section Occupancy
)

capture () {
    local tag="$1"; shift
    local rep_path="${OUT}/${tag}"
    local kernel_out_dir="${OUT}/tmp_${tag}"
    mkdir -p "$kernel_out_dir"
    echo "  capturing ${tag} ..."
    # --csv prints metrics CSV to stdout; redirect it to the .csv file.
    # The default section CSV is long-form (one row per metric); parse_ncu.py
    # expects exactly that. For the raw per-stall-reason metrics not surfaced
    # in the default sections, re-export with `ncu --import {rep}.ncu-rep
    # --csv --page raw` and reshape (see scripts/ncu_extract_long.py).
    ncu --target-processes all \
        "${SECTIONS[@]}" \
        --csv \
        -o "${rep_path}" --force-overwrite \
        "$BIN" "$@" --out "$kernel_out_dir" > "${rep_path}.csv"
    echo "  wrote ${rep_path}.ncu-rep and ${rep_path}.csv"
}

# Win cell: long S, large B*H — where int8_fused is expected to win most.
echo "===== win cell: B=4 H=32 D=128 S=4096 ====="
capture win_fp16 \
    --B 4 --H 32 --D 128 --S 4096 --mode fp16
capture win_int8_fused \
    --B 4 --H 32 --D 128 --S 4096 --mode int8_fused --block-size 64
capture win_int8_fused_online \
    --B 4 --H 32 --D 128 --S 4096 --mode int8_fused_online \
    --block-size 64 --lanes 4 --tile-n 32

# Regression cell: short S, small B*H — where int8_fused does not help.
echo "===== regression cell: B=1 H=16 D=64 S=512 ====="
capture reg_fp16 \
    --B 1 --H 16 --D 64 --S 512 --mode fp16
capture reg_int8_fused \
    --B 1 --H 16 --D 64 --S 512 --mode int8_fused --block-size 64
capture reg_int8_fused_online \
    --B 1 --H 16 --D 64 --S 512 --mode int8_fused_online \
    --block-size 64 --lanes 2 --tile-n 32

echo ""
echo "all reports saved under ${OUT}/"
echo "next: python python/parse_ncu.py ${OUT}/*.csv > ${OUT}/summary.md"
