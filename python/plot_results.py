#!/usr/bin/env python3
"""Generate report plots from a benchmark CSV.

Usage:
    python plot_results.py [--csv results/all_modes.csv] \\
                           [--out results/plots]

Produces:
    01_latency_vs_seqlen.png      - latency vs S, faceted by (B,H,D)
    02_speedup_vs_seqlen.png      - speedup vs fp16 baseline
    03_bandwidth_vs_seqlen.png    - effective KV BW with HBM ceiling
    04_cost_decomposition.png     - stacked attn + reconstruct (int8_nonfused)
    05_blocksize_ablation.png     - latency vs block_size at fixed shape
    06_roofline.png               - arithmetic intensity vs achieved TFLOPS

Standalone — runs anywhere matplotlib + pandas are installed (Mac is fine).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# A100-SXM4-40GB peak numbers used as roofline ceilings.
A100_HBM_GBPS   = 1555.0          # GB/s (HBM2e theoretical)
A100_FP16_TFLOPS = 312.0          # FP16 tensor-core peak
A100_FP32_TFLOPS = 19.5           # FP32 CUDA-core peak (closer to what the kernel runs at)

# Mode plotting style.
MODE_STYLE = {
    "fp16":                dict(color="#1f77b4", marker="o", label="FP16 baseline"),
    "int8_nonfused":       dict(color="#d62728", marker="s", label="INT8 non-fused"),
    "int8_fused":          dict(color="#2ca02c", marker="^", label="INT8 fused"),
    "int8_fused_online":   dict(color="#ff7f0e", marker="X", label="INT8 fused + online softmax + uchar4"),
    "turboquant_fused":    dict(color="#9467bd", marker="D", label="TurboQuant harness (Alg-1, multi-launch)"),
    "turboquant_ref_fused":dict(color="#8c564b", marker="P", label="TurboQuant fused decode (Alg-2, single Triton kernel)"),
}


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["block_size"] = df["block_size"].astype(int)
    df["S"] = df["S"].astype(int)
    return df


def per_shape_panels(df: pd.DataFrame, plot_fn, suptitle: str, out_path: str,
                     figsize=(14, 8)):
    """Draw a 2x4 grid of subplots, one per (B,H,D) cell."""
    shapes = sorted(df.groupby(["B", "H", "D"]).groups.keys())
    n = len(shapes)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for i, (B, H, D) in enumerate(shapes):
        ax = axes[i // cols][i % cols]
        sub = df[(df["B"] == B) & (df["H"] == H) & (df["D"] == D)]
        plot_fn(ax, sub, B, H, D)
        ax.set_title(f"B={B} H={H} D={D}", fontsize=10)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(suptitle, fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------- 1. Latency vs sequence length -----------------

def plot_latency(ax, sub, B, H, D):
    # Pin block_size=64 for INT8 modes; TurboQuant uses value_group_size (32) — pass through.
    bs_pin = 64
    rep = sub[
        (sub["mode"] == "fp16")
        | ((sub["mode"].isin(("int8_nonfused", "int8_fused", "int8_fused_online"))) & (sub["block_size"] == bs_pin))
        | (sub["mode"].isin(("turboquant_fused", "turboquant_ref_fused")))
    ].copy()

    for mode, style in MODE_STYLE.items():
        s = rep[rep["mode"] == mode].sort_values("S")
        if s.empty:
            continue
        ax.plot(s["S"], s["total_us"], **style)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("seq_len S")
    ax.set_ylabel("latency (us)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper left")


# ----------------- 2. Speedup vs fp16 -----------------

def plot_speedup(ax, sub, B, H, D):
    # INT8 modes are pinned at block_size=64; TurboQuant has its own (value_group=32).
    rep = sub[
        (sub["mode"] == "fp16")
        | ((sub["mode"].isin(("int8_nonfused", "int8_fused", "int8_fused_online"))) & (sub["block_size"] == 64))
        | (sub["mode"].isin(("turboquant_fused", "turboquant_ref_fused")))
    ].copy()
    fp = rep[rep["mode"] == "fp16"].set_index("S")["total_us"]
    for mode in ("int8_nonfused", "int8_fused", "int8_fused_online",
                 "turboquant_fused", "turboquant_ref_fused"):
        s = rep[rep["mode"] == mode].sort_values("S")
        if s.empty:
            continue
        speedup = fp.reindex(s["S"]).values / s["total_us"].values
        ax.plot(s["S"], speedup, **MODE_STYLE[mode])
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("seq_len S")
    ax.set_ylabel("speedup vs FP16")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7)


# ----------------- 3. Bandwidth vs S -----------------

def plot_bandwidth(ax, sub, B, H, D):
    bs_pin = 64
    rep = sub[
        (sub["mode"] == "fp16")
        | ((sub["mode"].isin(("int8_nonfused", "int8_fused", "int8_fused_online"))) & (sub["block_size"] == bs_pin))
        | (sub["mode"].isin(("turboquant_fused", "turboquant_ref_fused")))
    ].copy()
    peak_seen = 0.0
    for mode, style in MODE_STYLE.items():
        s = rep[rep["mode"] == mode].sort_values("S")
        if s.empty:
            continue
        gbps = (s["bytes_kv"] / (s["total_us"] * 1e-6) / 1e9).values
        peak_seen = max(peak_seen, float(gbps.max()))
        ax.plot(s["S"], gbps, **style)
    # Linear y-axis: HBM peak (1555 GB/s) is ~20× our top measurement, so
    # don't draw it; annotate %-of-peak instead.
    pct = 100.0 * peak_seen / A100_HBM_GBPS
    ax.text(0.97, 0.04, f"max {peak_seen:.0f} GB/s = {pct:.1f}% of HBM peak",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("seq_len S")
    ax.set_ylabel("effective KV BW (GB/s)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper left")


# ----------------- 4. Cost decomposition (int8_nonfused) -----------------

def plot_cost_decomp(df: pd.DataFrame, out_path: str):
    nf = df[(df["mode"] == "int8_nonfused") & (df["block_size"] == 64)].copy()
    nf = nf.sort_values(["B", "H", "D", "S"])
    fig, ax = plt.subplots(figsize=(14, 5))
    nf["label"] = nf.apply(
        lambda r: f"B={r['B']} H={r['H']} D={r['D']} S={r['S']}", axis=1)
    x = np.arange(len(nf))
    ax.bar(x, nf["attn_us"],        label="attn",         color="#9467bd")
    ax.bar(x, nf["reconstruct_us"], bottom=nf["attn_us"], label="reconstruct", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(nf["label"], rotation=90, fontsize=7)
    ax.set_ylabel("latency (us)")
    ax.set_title("INT8 non-fused: attn + reconstruct decomposition (block_size=64)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------- 5. Block-size ablation -----------------

def plot_blocksize_ablation(df: pd.DataFrame, out_path: str):
    # Look at D=128 only (only D supporting all three block sizes).
    sub = df[(df["D"] == 128) &
             (df["mode"].isin(("int8_nonfused", "int8_fused"))) &
             (df["S"] == 4096)].copy()
    if sub.empty:
        print(f"skipping blocksize ablation (no rows match)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ("int8_nonfused", "int8_fused")):
        m = sub[sub["mode"] == mode]
        for (B, H), g in m.groupby(["B", "H"]):
            g = g.sort_values("block_size")
            ax.plot(g["block_size"], g["total_us"], marker="o",
                    label=f"B={B} H={H}")
        ax.set_title(f"{mode}, D=128, S=4096")
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel("block_size")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("total latency (us)")
    fig.suptitle("Block-size ablation at D=128, S=4096")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------- 6. Roofline -----------------

def estimate_flops_per_call(B, H, D, S):
    """Decode-attention FLOPs per call.
    Q.K^T   : B*H*S*D*2  (multiply + add per element)
    softmax : ~B*H*S*5   (exp + sum + div)
    P.V     : B*H*S*D*2
    """
    return 4.0 * B * H * S * D + 5.0 * B * H * S


def estimate_bytes_per_call(row):
    """Bytes moved per attention call for the decode hot path.
    Excludes Q/output (negligible) and the prefill compress cost.
    """
    B, H, D, S = row["B"], row["H"], row["D"], row["S"]
    n_kv_elems = B * H * S * D
    if row["mode"] == "fp16":
        return 2 * n_kv_elems * 2          # read K + read V, fp16
    bs = max(int(row["block_size"]), 1)
    n_scale_elems = B * H * S * (D // bs)
    if row["mode"] == "int8_nonfused":
        # int8 read + scale read + fp16 write + fp16 read, for both K and V
        return 2 * (n_kv_elems * 1 + n_scale_elems * 2 + n_kv_elems * 2 + n_kv_elems * 2)
    if row["mode"] in ("int8_fused", "int8_fused_online"):
        # int8 read + scale read, both K and V — same byte budget as int8_fused.
        return 2 * (n_kv_elems * 1 + n_scale_elems * 2)
    if row["mode"] == "turboquant_fused":
        # K: 4-bit packed indices (D/2 bytes/row) + fp16 norm (2 B/row).
        # V: 4-bit packed values (D/2 bytes/row) + fp16 scales/zeros per group of 32.
        n_rows = B * H * S
        value_group = max(int(row["block_size"]), 1)
        n_v_groups = D // value_group
        return (n_rows * (D // 2) + n_rows * 2                    # K
                + n_rows * (D // 2)                                # V data
                + 2 * n_rows * n_v_groups * 2)                     # V scales + zeros
    if row["mode"] == "turboquant_ref_fused":
        # Algorithm 2: MSE indices + QJL signs + 2 fp16 norms per row.
        # MSE is at (bits-1) bits; for the default bits=3 this is 2 bits → 2/8 = D/4 bytes.
        # QJL is 1 bit per coord = D/8 bytes.
        n_rows = B * H * S
        value_group = max(int(row["block_size"]), 1)
        n_v_groups = D // value_group
        return (n_rows * (D // 4) + n_rows * (D // 8) + 2 * n_rows * 2  # K
                + n_rows * (D // 2)                                      # V data
                + 2 * n_rows * n_v_groups * 2)                           # V scales + zeros
    return np.nan


def plot_roofline(df: pd.DataFrame, out_path: str):
    work = df.copy()
    work["flops"]     = work.apply(lambda r: estimate_flops_per_call(r["B"], r["H"], r["D"], r["S"]), axis=1)
    work["bytes"]     = work.apply(estimate_bytes_per_call, axis=1)
    work["intensity"] = work["flops"] / work["bytes"]
    work["tflops"]    = work["flops"] / (work["total_us"] * 1e-6) / 1e12

    # Pick block_size=64 as the representative for int8 modes; fp16 has no block_size;
    # turboquant_fused uses value_group=32 — pass through unfiltered.
    work = work[
        (work["mode"] == "fp16")
        | ((work["mode"].isin(("int8_nonfused", "int8_fused", "int8_fused_online"))) & (work["block_size"] == 64))
        | (work["mode"].isin(("turboquant_fused", "turboquant_ref_fused")))
    ]

    # Square canvas + larger type for poster / slide readability.
    fig, ax = plt.subplots(figsize=(9, 9))
    title_fs = 16
    label_fs = 14
    tick_fs = 12
    legend_fs = 11
    note_fs = 11
    # Callout labels (e.g. green INT8-fused arrows) — keep larger than body legend.
    ann_fs = 14

    # Roofline ceilings.
    intensity_grid = np.logspace(-2, 3, 200)  # FLOPs/byte
    bw_ceiling     = (intensity_grid * A100_HBM_GBPS / 1e3)  # TFLOPS = GB/s * intensity / 1e3
    ax.plot(intensity_grid, np.minimum(bw_ceiling, A100_FP32_TFLOPS),
            color="black", linestyle="-", linewidth=1.2,
            label=f"Roofline (HBM={A100_HBM_GBPS:.0f} GB/s, FP32 peak={A100_FP32_TFLOPS:.0f} TFLOPS)")

    for mode, style in MODE_STYLE.items():
        s = work[work["mode"] == mode]
        if s.empty:
            continue
        ax.scatter(s["intensity"], s["tflops"],
                   color=style["color"], marker=style["marker"], s=45,
                   alpha=0.7, label=style["label"], edgecolor="white", linewidth=0.5)

    # Annotate two contrasting cells so the reader can locate the regimes.
    def annotate_cell(B, H, D, S, mode, label, dx=10, dy=10):
        sel = work[(work["B"] == B) & (work["H"] == H) & (work["D"] == D) &
                   (work["S"] == S) & (work["mode"] == mode)]
        if sel.empty:
            return
        r = sel.iloc[0]
        ax.annotate(label, xy=(r["intensity"], r["tflops"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=ann_fs, color=MODE_STYLE[mode]["color"],
                    arrowprops=dict(arrowstyle="-", color=MODE_STYLE[mode]["color"], lw=0.9))

    annotate_cell(4, 32, 128, 4096, "int8_fused",
                  "fused win cell\nB=4 H=32 D=128 S=4096", dx=12, dy=18)
    annotate_cell(1, 16,  64,  512, "int8_fused",
                  "sync-bound regression\nB=1 H=16 D=64 S=512", dx=-150, dy=-30)

    # Sync-bound textbox.
    ax.text(0.02, 0.97,
            "All kernels sit far below both ceilings:\n"
            "the bottleneck is per-token __syncthreads(),\n"
            "not memory bandwidth or arithmetic peak.",
            transform=ax.transAxes, va="top", ha="left", fontsize=note_fs,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff8d6", ec="gray", alpha=0.9))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)", fontsize=label_fs)
    ax.set_ylabel("achieved TFLOPS", fontsize=label_fs)
    ax.set_title("Roofline — A100-SXM4-40GB (kernel-as-FP32 peak)", fontsize=title_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.tick_params(axis="both", which="minor", labelsize=tick_fs - 1)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=legend_fs, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/all_modes.csv")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()

    df = load(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_shape_panels(df, plot_latency,
                     "Latency vs sequence length (block_size=64 for INT8)",
                     out_dir / "01_latency_vs_seqlen.png")
    per_shape_panels(df, plot_speedup,
                     "Speedup vs FP16 baseline (block_size=64 for INT8)",
                     out_dir / "02_speedup_vs_seqlen.png")
    per_shape_panels(df, plot_bandwidth,
                     "Effective KV bandwidth vs sequence length",
                     out_dir / "03_bandwidth_vs_seqlen.png")
    plot_cost_decomp(df,        out_dir / "04_cost_decomposition.png")
    plot_blocksize_ablation(df, out_dir / "05_blocksize_ablation.png")
    plot_roofline(df,           out_dir / "06_roofline.png")


if __name__ == "__main__":
    main()
