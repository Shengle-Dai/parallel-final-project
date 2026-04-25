#!/usr/bin/env python3
"""Compare CUDA decode-attention output against a PyTorch reference.

Usage: python check_correctness.py <out_dir>

Branches on meta["mode"]:

  fp16          - load Q/K/V/O_cuda; run torch attention in fp32; check max abs.
  int8_nonfused - load also Kq/Vq/K_scales/V_scales; run two checks:
                    1. quantization round-trip residual
                    2. INT8-attn output vs torch attention on dequantized K,V

Exit 0 = pass, 1 = fail.
"""

import json
import math
import os
import sys

import numpy as np
import torch


PASS_THRESHOLD_FP16 = 1e-2
PASS_THRESHOLD_INT8_ATTN = 1e-2
ROUNDTRIP_SLACK = 1.5  # allow 1.5x the analytic per-element bound


def load_tensor(path: str, shape, dtype):
    a = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if a.size != expected:
        raise RuntimeError(f"{path}: have {a.size} elems, expected {expected} for shape {shape}")
    return a.reshape(shape)


def torch_attention_fp32(Q, K, V, scale):
    Qt = torch.from_numpy(Q).to(torch.float32)
    Kt = torch.from_numpy(K).to(torch.float32)
    Vt = torch.from_numpy(V).to(torch.float32)
    scores = torch.einsum("bhd,bhsd->bhs", Qt, Kt) * scale
    probs  = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bhsd->bhd", probs, Vt).to(torch.float16).numpy()


def dequantize(Xq, scales, block_size):
    # Xq:     int8   [B,H,S,D]
    # scales: float16 [B,H,S, D/block_size]
    # Reconstruct in fp32 to match the kernel's intermediate precision, then
    # cast to fp16 to match the kernel's output dtype.
    B, H, S, D = Xq.shape
    n_sub = D // block_size
    s_full = scales.reshape(B, H, S, n_sub).astype(np.float32)
    s_full = np.repeat(s_full, block_size, axis=-1)  # [B,H,S,D]
    return (Xq.astype(np.float32) * s_full).astype(np.float16)


def check_fp16(out_dir, meta):
    B, H, D, S = meta["B"], meta["H"], meta["D"], meta["S"]
    Q      = load_tensor(os.path.join(out_dir, "Q.bin"),      (B, H, D),    np.float16)
    K      = load_tensor(os.path.join(out_dir, "K.bin"),      (B, H, S, D), np.float16)
    V      = load_tensor(os.path.join(out_dir, "V.bin"),      (B, H, S, D), np.float16)
    O_cuda = load_tensor(os.path.join(out_dir, "O_cuda.bin"), (B, H, D),    np.float16)

    O_ref = torch_attention_fp32(Q, K, V, 1.0 / math.sqrt(D))

    diff = O_cuda.astype(np.float32) - O_ref.astype(np.float32)
    max_abs  = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    denom    = np.maximum(np.abs(O_ref.astype(np.float32)), 1e-3)
    max_rel  = float(np.max(np.abs(diff) / denom))

    print(f"shape    B={B} H={H} D={D} S={S}  mode=fp16")
    print(f"max_abs  = {max_abs:.4e}")
    print(f"mean_abs = {mean_abs:.4e}")
    print(f"max_rel  = {max_rel:.4e}")
    if max_abs < PASS_THRESHOLD_FP16:
        print(f"PASS (max_abs < {PASS_THRESHOLD_FP16})")
        return 0
    print(f"FAIL (max_abs >= {PASS_THRESHOLD_FP16})")
    return 1


def check_int8(out_dir, meta):
    mode = meta["mode"]
    B, H, D, S = meta["B"], meta["H"], meta["D"], meta["S"]
    block_size = meta["block_size"]
    n_sub = D // block_size

    Q      = load_tensor(os.path.join(out_dir, "Q.bin"),       (B, H, D),    np.float16)
    K      = load_tensor(os.path.join(out_dir, "K.bin"),       (B, H, S, D), np.float16)
    V      = load_tensor(os.path.join(out_dir, "V.bin"),       (B, H, S, D), np.float16)
    Kq     = load_tensor(os.path.join(out_dir, "Kq.bin"),      (B, H, S, D), np.int8)
    Vq     = load_tensor(os.path.join(out_dir, "Vq.bin"),      (B, H, S, D), np.int8)
    Ks     = load_tensor(os.path.join(out_dir, "K_scales.bin"),(B, H, S, n_sub), np.float16)
    Vs     = load_tensor(os.path.join(out_dir, "V_scales.bin"),(B, H, S, n_sub), np.float16)
    O_cuda = load_tensor(os.path.join(out_dir, "O_cuda.bin"),  (B, H, D),    np.float16)

    print(f"shape    B={B} H={H} D={D} S={S}  mode={mode}  block_size={block_size}")

    # ---- Check 1: quantization round-trip residual ----
    K_recon = dequantize(Kq, Ks, block_size)
    V_recon = dequantize(Vq, Vs, block_size)

    res_K = np.max(np.abs(K.astype(np.float32) - K_recon.astype(np.float32)))
    res_V = np.max(np.abs(V.astype(np.float32) - V_recon.astype(np.float32)))

    # Analytic per-element bound: max |x| in any block / 127 / 2 from rounding,
    # but worst case is approximately scale = max/127, so |residual| <= scale/2.
    # We bound by max scale * 0.5 plus fp16 cast slop.
    s_max = max(float(np.max(np.abs(Ks.astype(np.float32)))),
                float(np.max(np.abs(Vs.astype(np.float32)))))
    bound = s_max * 0.5 + 1e-3
    print(f"  roundtrip K max_abs = {res_K:.4e}")
    print(f"  roundtrip V max_abs = {res_V:.4e}")
    print(f"  analytic bound      = {bound:.4e}  (slack {ROUNDTRIP_SLACK}x = {bound*ROUNDTRIP_SLACK:.4e})")
    rt_ok = (res_K <= bound * ROUNDTRIP_SLACK) and (res_V <= bound * ROUNDTRIP_SLACK)
    print(f"  roundtrip {'PASS' if rt_ok else 'FAIL'}")

    # ---- Check 2: INT8-attention reference (attn on the dequantized tensors) ----
    O_int8_ref = torch_attention_fp32(Q, K_recon, V_recon, 1.0 / math.sqrt(D))
    diff = O_cuda.astype(np.float32) - O_int8_ref.astype(np.float32)
    int8_max_abs  = float(np.max(np.abs(diff)))
    int8_mean_abs = float(np.mean(np.abs(diff)))
    print(f"  int8-attn  max_abs  = {int8_max_abs:.4e}")
    print(f"  int8-attn  mean_abs = {int8_mean_abs:.4e}")
    int8_ok = int8_max_abs < PASS_THRESHOLD_INT8_ATTN
    print(f"  int8-attn  {'PASS' if int8_ok else 'FAIL'} (threshold {PASS_THRESHOLD_INT8_ATTN})")

    # ---- Informational: drift vs FP16 ground truth ----
    O_fp16 = torch_attention_fp32(Q, K, V, 1.0 / math.sqrt(D))
    drift = float(np.max(np.abs(O_cuda.astype(np.float32) - O_fp16.astype(np.float32))))
    print(f"  drift vs fp16-truth = {drift:.4e}  (informational)")

    if rt_ok and int8_ok:
        print("PASS")
        return 0
    print("FAIL")
    return 1


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_correctness.py <out_dir>", file=sys.stderr)
        return 2
    out_dir = sys.argv[1]
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    mode = meta.get("mode", "fp16")
    if mode == "fp16":
        return check_fp16(out_dir, meta)
    if mode in ("int8_nonfused", "int8_fused", "int8_fused_online"):
        return check_int8(out_dir, meta)
    print(f"unknown mode: {mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
