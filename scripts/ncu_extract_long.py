#!/usr/bin/env python3
# Reshape ncu's wide 'raw' page export into the long-form CSV that
# parse_ncu.py expects (one row per metric, with the original "Metric Name"
# column populated). Filters to the decode-attention kernel only.
#
# usage: ncu_extract_long.py <ncu_results_dir>

from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys


ID_COLS = {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
           "Context", "Stream", "Block Size", "Grid Size", "Device", "CC"}

ATTN_HINTS = ("decode_attn_baseline", "decode_attn_int8_fused")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: ncu_extract_long.py <ncu_results_dir>", file=sys.stderr)
        return 2
    out_dir = sys.argv[1]
    rep_paths = sorted(glob.glob(os.path.join(out_dir, "*.ncu-rep")))
    if not rep_paths:
        print(f"no .ncu-rep files in {out_dir}", file=sys.stderr)
        return 1

    id_order = ["ID", "Process ID", "Process Name", "Host Name",
                "Kernel Name", "Context", "Stream", "Block Size",
                "Grid Size", "Device", "CC"]

    for rep in rep_paths:
        tag = os.path.splitext(os.path.basename(rep))[0]
        wide = subprocess.run(
            ["ncu", "--import", rep, "--csv", "--page", "raw"],
            capture_output=True, text=True, check=True,
        ).stdout

        reader = csv.reader(wide.splitlines())
        try:
            header = next(reader)
        except StopIteration:
            print(f"  {tag}: empty CSV", file=sys.stderr)
            continue
        rows = list(reader)

        id_idx = {c: i for i, c in enumerate(header) if c in ID_COLS}
        metric_cols = [(i, h) for i, h in enumerate(header) if h not in ID_COLS]
        kn_idx = id_idx.get("Kernel Name", -1)
        if kn_idx < 0:
            print(f"  {tag}: no 'Kernel Name' column", file=sys.stderr)
            continue

        attn_rows = [r for r in rows if any(h in r[kn_idx] for h in ATTN_HINTS)]
        if not attn_rows:
            print(f"  {tag}: no attention kernel; skipping")
            continue

        out_csv = os.path.join(out_dir, f"{tag}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(id_order + ["Section Name", "Metric Name",
                                   "Metric Unit", "Metric Value"])
            for row in attn_rows:
                base = [row[id_idx[c]] for c in id_order]
                for mi, mname in metric_cols:
                    w.writerow(base + ["", mname, "", row[mi]])
        kn = attn_rows[0][kn_idx].split("(")[0].split("::")[-1]
        print(f"{tag}: {len(attn_rows)} row(s) for {kn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
