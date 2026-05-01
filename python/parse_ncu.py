#!/usr/bin/env python3
"""Parse Nsight Compute --csv output and emit a markdown summary.

Each capture (one (cell, kernel) pair) is one CSV file. ncu writes its CSV
in long format: one row per (kernel-launch, section, metric). We aggregate
into a wide table keyed by the file's basename.

Usage:
    python python/parse_ncu.py results/ncu/*.csv > results/ncu/summary.md

The metrics we care about for the §7.3 sync model:
  - barrier-induced warp stall rate    → t_sync coefficient
  - DRAM throughput / utilisation      → BW_eff coefficient
  - issue-eligible-warp percentage     → occupancy reality check

ncu metric names vary across CUDA versions, so the parser does substring
matching rather than exact equality.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


# Each entry is (label, [substring patterns to match against "Metric Name"]).
# First match wins. Substring matching tolerates the version-to-version
# suffix drift like ".pct" vs ".pct_of_peak_sustained_active".
METRICS = [
    ("dur_us",            ["gpu__time_duration.sum"]),
    ("achieved_occ_pct",  ["sm__warps_active.avg.pct_of_peak_sustained_active"]),
    ("dram_GBps",         ["dram__bytes.sum.per_second"]),
    ("dram_util_pct",     ["dram__throughput.avg.pct_of_peak_sustained_elapsed"]),
    ("barrier_stall_pct", ["smsp__average_warps_issue_stalled_barrier"]),
    ("mem_throt_pct",     ["smsp__average_warps_issue_stalled_mem_throttle",
                           "smsp__average_warps_issue_stalled_drain"]),
    ("long_sb_pct",       ["smsp__average_warps_issue_stalled_long_scoreboard"]),
    ("issue_elig_pct",    ["smsp__warps_eligible"]),
]


def parse_one(path):
    """Read one ncu CSV file. Return (kernel_name, {label: 'value unit'})."""
    out = {}
    kernel_name = ""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not kernel_name:
                    kernel_name = row.get("Kernel Name", "").strip()
                mname = row.get("Metric Name", "")
                mval = row.get("Metric Value", "").strip()
                munit = row.get("Metric Unit", "").strip()
                for label, patterns in METRICS:
                    if label in out:
                        continue
                    for p in patterns:
                        if p in mname:
                            out[label] = f"{mval} {munit}".strip()
                            break
    except FileNotFoundError:
        print(f"warning: file not found: {path}", file=sys.stderr)
    except Exception as e:
        print(f"warning: failed to parse {path}: {e}", file=sys.stderr)
    return kernel_name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+",
                    help="ncu --csv output files (one per kernel x cell capture)")
    args = ap.parse_args()

    rows = []
    for path in args.csvs:
        tag = os.path.splitext(os.path.basename(path))[0]
        kernel_name, metrics = parse_one(path)
        rows.append((tag, kernel_name, metrics))

    labels = [l for l, _ in METRICS]
    cols = ["capture", "kernel"] + labels

    print("# Nsight Compute summary\n")
    print("One row per (cell × kernel) capture. Empty cells mean the metric was not")
    print("present in the captured ncu sections — adjust `SECTIONS` in")
    print("`scripts/ncu_capture.sh` and re-run if you need them.\n")
    print("Column hints:\n")
    print("- **barrier_stall_pct** — `__syncthreads()` time per warp-active cycle.")
    print("  Drives the §7.3 `t_sync` term; should be high on `int8_fused` and")
    print("  significantly lower on `int8_fused_online`.")
    print("- **dram_GBps / dram_util_pct** — achieved DRAM throughput. The §8.7")
    print("  roofline gap (kernel below HBM peak) shows up here.")
    print("- **achieved_occ_pct** — fraction of theoretical occupancy actually")
    print("  achieved. Below ~25% suggests register-pressure or block-count caps.")
    print("- **long_sb_pct** — long-scoreboard stalls (typically waiting on global")
    print("  memory). Vectorised loads should reduce this.\n")

    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for tag, kname, m in rows:
        # Trim namespace clutter from kernel name for readability.
        short_kname = kname.split("(")[0].split("::")[-1]
        cells = [tag, short_kname] + [m.get(lab, "") for lab in labels]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    sys.exit(main())
