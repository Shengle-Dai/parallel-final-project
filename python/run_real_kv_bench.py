#!/usr/bin/env python3
"""End-to-end wrapper: capture KV activations from GPT-2, then run the C++
benchmark binary on the captured data.

This is a convenience layer around two pieces that work independently:

    python python/capture_kv.py --seq 1024 --out tq_real_kv
    ./build/tqkv_bench --load-kv-from tq_real_kv --csv results/phase8_real_kv.csv

After this script runs, ``results/phase8_real_kv.csv`` has the same column
schema as the synthetic-data sweep (with ``tile_n, lanes_per_thread``
trailing). Compare its rows to the synthetic equivalents at the same shape
to see whether the per-channel outlier structure of real LLM activations
shifts INT8 fused's win/loss profile.

Usage:
    python python/run_real_kv_bench.py [--seq 1024] [--layer -1] \\
        [--bin ./build/tqkv_bench] [--csv results/phase8_real_kv.csv]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is Paris. ")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--bin", default="./build/tqkv_bench",
                    help="path to tqkv_bench binary")
    ap.add_argument("--capture-dir", default=None,
                    help="where to keep the captured Q/K/V (default: a temp dir)")
    ap.add_argument("--csv", default="results/phase8_real_kv.csv")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--keep", action="store_true",
                    help="retain the capture dir after the run (default: delete)")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(f"benchmark binary not found at {args.bin} — build first with `cmake --build build -j`",
              file=sys.stderr)
        return 2

    cleanup = False
    if args.capture_dir is None:
        capture_dir = tempfile.mkdtemp(prefix="tq_real_kv_")
        cleanup = not args.keep
    else:
        capture_dir = args.capture_dir
        os.makedirs(capture_dir, exist_ok=True)

    try:
        # ---- Phase 1: capture ----
        cap_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "capture_kv.py"),
            "--prompt", args.prompt,
            "--seq", str(args.seq),
            "--layer", str(args.layer),
            "--out", capture_dir,
        ]
        print("running:", " ".join(cap_cmd))
        subprocess.run(cap_cmd, check=True)

        # ---- Phase 2: benchmark ----
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        bench_cmd = [
            args.bin,
            "--load-kv-from", capture_dir,
            "--csv", args.csv,
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
        ]
        print("running:", " ".join(bench_cmd))
        subprocess.run(bench_cmd, check=True)

        print(f"\ndone — CSV at {args.csv}")
    finally:
        if cleanup:
            shutil.rmtree(capture_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
