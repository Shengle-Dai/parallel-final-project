#!/usr/bin/env python3
# Six focused figures complementing plot_results.py:
#   07 peak-cell bars, 08 regression-cell flip, 09 ncu stalls,
#   10 speedup heatmap, 11 real-vs-synth correctness, 12 real-KV bars.

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODE_STYLE = {
    "fp16":              dict(color="#1f77b4", label="FP16 baseline"),
    "int8_nonfused":     dict(color="#d62728", label="INT8 non-fused"),
    "int8_fused":        dict(color="#2ca02c", label="INT8 fused"),
    "int8_fused_online": dict(color="#ff7f0e", label="INT8 fused + online + uchar4"),
}
MODE_ORDER = ["fp16", "int8_nonfused", "int8_fused", "int8_fused_online"]


def _pick_block_size(sub, mode, prefer=None):
    # Best (lowest total_us) row across block sizes for this mode at this shape.
    m = sub[sub["mode"] == mode]
    if m.empty:
        return None
    if mode == "fp16":
        return m.iloc[0]
    return m.loc[m["total_us"].idxmin()]


# -------------------- 7. Peak win cell --------------------

def plot_win_cell(df, out_path, B=4, H=32, D=64, S=1024):
    sub = df[(df["B"]==B)&(df["H"]==H)&(df["D"]==D)&(df["S"]==S)].copy()
    rows = []
    for mode in MODE_ORDER:
        r = _pick_block_size(sub, mode, prefer=64)
        if r is not None:
            rows.append((mode, float(r["total_us"]),
                         int(r["block_size"]) if r["block_size"] else 0))
    fp16_us = next(t for m, t, _ in rows if m == "fp16")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(rows))
    bars = ax.bar(
        xs,
        [t for _, t, _ in rows],
        color=[MODE_STYLE[m]["color"] for m, _, _ in rows],
        edgecolor="white", linewidth=1.5,
    )
    for x, (mode, t, bs) in zip(xs, rows):
        speed = fp16_us / t
        label = f"{t:.0f} µs\n{speed:.2f}× vs FP16"
        ax.text(x, t + max(t * 0.02, 8), label, ha="center", va="bottom",
                fontsize=10, fontweight="bold" if mode == "int8_fused_online" else "normal")

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [MODE_STYLE[m]["label"] for m, _, _ in rows], rotation=12, ha="right", fontsize=10
    )
    ax.set_ylabel("latency (µs)  — lower is better", fontsize=11)
    ax.set_title(f"Peak-speedup cell — B={B} H={H} D={D} S={S}", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ymax = max(t for _, t, _ in rows) * 1.18
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# -------------------- 8. Regression cell flip --------------------

def plot_regression_flip(df, out_path, B=1, H=16, D=64, S=512):
    sub = df[(df["B"]==B)&(df["H"]==H)&(df["D"]==D)&(df["S"]==S)].copy()
    rows = []
    for mode in MODE_ORDER:
        r = _pick_block_size(sub, mode, prefer=64)
        if r is not None:
            rows.append((mode, float(r["total_us"])))
    fp16_us = next(t for m, t in rows if m == "fp16")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(rows))
    bars = ax.bar(
        xs,
        [t for _, t in rows],
        color=[MODE_STYLE[m]["color"] for m, _ in rows],
        edgecolor="white", linewidth=1.5,
    )
    for x, (mode, t) in zip(xs, rows):
        speed = fp16_us / t
        sign = "↓" if speed < 1.0 else "↑"
        label = f"{t:.0f} µs\n{speed:.2f}× {sign}"
        ax.text(x, t + max(t * 0.02, 5), label, ha="center", va="bottom",
                fontsize=10, fontweight="bold" if mode == "int8_fused_online" else "normal")
    ax.axhline(fp16_us, color="black", linestyle=":", alpha=0.6, label="FP16 reference")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [MODE_STYLE[m]["label"] for m, _ in rows], rotation=12, ha="right", fontsize=10
    )
    ax.set_ylabel("latency (µs)  — lower is better", fontsize=11)
    ax.set_title(
        f"Sync-bound regression cell — B={B} H={H} D={D} S={S}\n"
        f"Where v2 INT8 fused loses to FP16; online softmax converts it to a win",
        fontsize=12,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ymax = max(t for _, t in rows) * 1.20
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# -------------------- 9. NCU stall comparison --------------------

NCU_METRICS = {
    "barrier_stall_pct": "smsp__average_warps_issue_stalled_barrier",
    "long_sb_pct":       "smsp__average_warps_issue_stalled_long_scoreboard",
    "achieved_occ_pct":  "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram_GBps":         "dram__bytes.sum.per_second",
}


def _read_ncu_metric(path, substr):
    """Read first matching metric from a long-form ncu CSV."""
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if substr in row.get("Metric Name", ""):
                    try:
                        return float(row.get("Metric Value", "nan"))
                    except ValueError:
                        return float("nan")
    except FileNotFoundError:
        pass
    return float("nan")


def _ncu_table(ncu_dir):
    """Return {(cell, mode): {metric_label: value}} for the 6 standard captures."""
    captures = ["reg_fp16", "reg_int8_fused", "reg_int8_fused_online",
                "win_fp16", "win_int8_fused", "win_int8_fused_online"]
    out = {}
    for tag in captures:
        path = os.path.join(ncu_dir, f"{tag}.csv")
        out[tag] = {lab: _read_ncu_metric(path, sub) for lab, sub in NCU_METRICS.items()}
    return out


def plot_ncu_stalls(ncu_dir, out_path):
    data = _ncu_table(ncu_dir)
    cells = ["reg", "win"]
    cell_label = {"reg": "Regression cell (B=1 H=16 D=64 S=512)",
                  "win": "Win cell (B=4 H=32 D=128 S=4096)"}
    modes = [("fp16", "FP16"), ("int8_fused", "INT8 fused"),
             ("int8_fused_online", "INT8 fused + online + uchar4")]
    metrics = [("barrier_stall_pct", "Barrier stall %"),
               ("long_sb_pct",       "Long-scoreboard stall %")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    bar_w = 0.25
    x = np.arange(len(modes))

    for ax_idx, (metric, metric_title) in enumerate(metrics):
        ax = axes[ax_idx]
        for ci, cell in enumerate(cells):
            ys = [data[f"{cell}_{mkey}"][metric] for mkey, _ in modes]
            offset = (ci - 0.5) * bar_w
            ax.bar(
                x + offset, ys,
                bar_w,
                color=["#999" if ci == 0 else "#444"][0],
                alpha=0.85 if ci == 1 else 0.55,
                edgecolor="white", linewidth=1.0,
                label=cell_label[cell],
            )
            for xi, v in zip(x, ys):
                if not np.isnan(v):
                    ax.text(xi + offset, v + max(v, 0.02) * 0.05,
                            f"{v:.2g}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in modes], fontsize=10)
        ax.set_ylabel(metric_title, fontsize=11)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_title(metric_title, fontsize=12)
        # Give the legend headroom: extend the y-axis upper bound by ~3x so the
        # legend lives above all bar labels.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 3.5)
        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.suptitle("Nsight Compute — warp-issue stalls (log scale)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# -------------------- 10. Speedup heatmap --------------------

def plot_speedup_heatmap(df, out_path):
    """int8_fused_online speedup over FP16 at each (B*H, S) cell."""
    fo = df[df["mode"] == "int8_fused_online"].copy()
    fp = df[df["mode"] == "fp16"].copy()
    # Per shape, pick the best block_size for online.
    fo_best = fo.loc[fo.groupby(["B","H","D","S"])["total_us"].idxmin()]
    fp16_us = fp.set_index(["B","H","D","S"])["total_us"]
    fo_best = fo_best.copy()
    fo_best["fp16_us"] = fo_best.apply(
        lambda r: fp16_us.get((r["B"], r["H"], r["D"], r["S"]), float("nan")), axis=1)
    fo_best["speedup"] = fo_best["fp16_us"] / fo_best["total_us"]

    Ss = sorted(df["S"].unique())
    Ds = sorted(df["D"].unique())
    fig, axes = plt.subplots(1, len(Ds), figsize=(4 * len(Ds) + 1, 5), sharey=True)
    if len(Ds) == 1:
        axes = [axes]
    for ax, D in zip(axes, Ds):
        sub = fo_best[fo_best["D"] == D].copy()
        sub["BH"] = sub["B"] * sub["H"]
        BHs = sorted(sub["BH"].unique())
        mat = np.full((len(BHs), len(Ss)), np.nan)
        for i, bh in enumerate(BHs):
            for j, S in enumerate(Ss):
                r = sub[(sub["BH"] == bh) & (sub["S"] == S)]
                if not r.empty:
                    mat[i, j] = r.iloc[0]["speedup"]
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                       vmin=0.8, vmax=2.8, origin="lower")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}",
                            ha="center", va="center", fontsize=9,
                            color="black" if 1.1 < mat[i,j] < 2.2 else "white")
        ax.set_xticks(range(len(Ss)))
        ax.set_xticklabels(Ss)
        ax.set_yticks(range(len(BHs)))
        ax.set_yticklabels(BHs)
        ax.set_xlabel("sequence length S", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("B × H (effective batch · heads)", fontsize=11)
        ax.set_title(f"D = {D}", fontsize=12)
    fig.suptitle("INT8 fused + online softmax speedup vs FP16 — A100",
                 fontsize=13, y=1.02)
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("speedup", fontsize=10)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# -------------------- 11. Real-KV correctness --------------------

def plot_real_correctness(out_path):
    """Hand-coded from the §4.6 numbers — small enough to inline."""
    metrics = ["K round-trip\nmax-abs", "V round-trip\nmax-abs",
               "End-to-end\noutput max-abs"]
    synth = [2.01e-3, 2.29e-3, 3.81e-6]
    real  = [3.52e-2, 5.37e-2, 9.77e-4]
    threshold = 1e-2

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    w = 0.36
    ax.bar(x - w/2, synth, w, color="#1f77b4", label="Synthetic Gaussian",
           edgecolor="white", linewidth=1.0)
    ax.bar(x + w/2, real,  w, color="#ff7f0e", label="GPT-2 layer 11",
           edgecolor="white", linewidth=1.0)
    ax.axhline(threshold, color="red", linestyle="--", alpha=0.6,
               label="PASS threshold (1e-2)")
    for xi, (s, r) in enumerate(zip(synth, real)):
        ax.text(xi - w/2, s * 1.4, f"{s:.1e}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w/2, r * 1.4, f"{r:.1e}", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("error (log scale)", fontsize=11)
    ax.set_title("Quantization error — synthetic Gaussian vs real GPT-2 activations\n"
                 "(int8_fused_online, B=1 H=12 D=64 S=1024)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# -------------------- 12. Real-KV speedup at GPT-2 shape --------------------

def plot_real_speedup(real_csv, out_path):
    df = pd.read_csv(real_csv)
    fp16_us = df[df["mode"] == "fp16"]["total_us"].iloc[0]
    rows = []
    for mode in MODE_ORDER:
        m = df[df["mode"] == mode]
        if m.empty:
            continue
        r = m.loc[m["total_us"].idxmin()]
        rows.append((mode, float(r["total_us"]),
                     int(r["block_size"]) if r["block_size"] else 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(rows))
    ax.bar(xs, [t for _, t, _ in rows],
           color=[MODE_STYLE[m]["color"] for m, _, _ in rows],
           edgecolor="white", linewidth=1.5)
    for x, (mode, t, bs) in zip(xs, rows):
        speed = fp16_us / t
        ax.text(x, t + max(t * 0.02, 4),
                f"{t:.0f} µs\n{speed:.2f}× vs FP16",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold" if mode == "int8_fused_online" else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [MODE_STYLE[m]["label"] for m, _, _ in rows], rotation=12, ha="right", fontsize=10
    )
    ax.set_ylabel("latency (µs) — lower is better", fontsize=11)
    ax.set_title("Real GPT-2 KV — same kernels, B=1 H=12 D=64 S=1024",
                 fontsize=12, pad=20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ymax_data = max(t for _, t, _ in rows)
    ax.set_ylim(0, ymax_data * 1.20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/phase6_int8_fused_online.csv")
    ap.add_argument("--real-kv-csv", default="results/phase8_real_kv.csv")
    ap.add_argument("--ncu-dir", default="results/ncu")
    ap.add_argument("--out", default="results/plots_a100")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    df["block_size"] = df["block_size"].astype(int)

    plot_win_cell(df,           out / "07_win_cell_bars.png")
    plot_regression_flip(df,    out / "08_regression_flip.png")
    plot_ncu_stalls(args.ncu_dir, out / "09_ncu_stalls.png")
    plot_speedup_heatmap(df,    out / "10_speedup_heatmap.png")
    plot_real_correctness(      out / "11_real_vs_synth_correctness.png")
    if os.path.exists(args.real_kv_csv):
        plot_real_speedup(args.real_kv_csv, out / "12_real_kv_speedup.png")


if __name__ == "__main__":
    main()
