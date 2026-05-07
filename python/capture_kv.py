#!/usr/bin/env python3
# Capture Q/K/V activations from GPT-2 into the binary layout that
# tqkv_decode / tqkv_bench consume via --load-kv-from.

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "gpt2"


def _gpt2_hook(model, layer_idx, captured):
    cfg = model.config
    D_total = cfg.hidden_size
    H = cfg.n_head
    D = D_total // H

    def hook(_module, _input, output):
        # GPT-2 fuses Q, K, V into one (B, S, 3*D_total) projection.
        B, S, _ = output.shape
        Q = output[..., :D_total].reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()
        K = output[..., D_total:2 * D_total].reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()
        V = output[..., 2 * D_total:].reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()
        captured["Q"] = Q
        captured["K"] = K
        captured["V"] = V

    return model.transformer.h[layer_idx].attn.c_attn.register_forward_hook(hook)


def _resolve_layer(model, layer_idx: int) -> int:
    n = len(model.transformer.h)
    if layer_idx < 0:
        layer_idx = n + layer_idx
    if layer_idx < 0 or layer_idx >= n:
        raise ValueError(f"layer {layer_idx} out of range [0, {n})")
    return layer_idx


def capture(prompt, seq_len, layer_idx, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    layer_idx = _resolve_layer(model, layer_idx)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids
    while input_ids.shape[1] < seq_len:
        input_ids = torch.cat([input_ids, enc.input_ids], dim=1)
    input_ids = input_ids[:, :seq_len].to(device)

    captured = {}
    handle = _gpt2_hook(model, layer_idx, captured)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()
    return captured["Q"], captured["K"], captured["V"], layer_idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is Paris. ")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--out", default="tq_real_kv")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available.", file=sys.stderr)
        return 2
    device = torch.device(device_name)

    # CPU + fp16 is supported but slow on most boxes; run model in fp32 and
    # cast at write time. GPU paths use whatever --dtype was requested.
    if device_name == "cpu" and args.dtype == "float16":
        dtype = torch.float32
    else:
        dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]

    Q, K, V, layer_idx = capture(args.prompt, args.seq, args.layer, device, dtype)
    B, H, S, D = K.shape
    Q_decode = Q[:, :, -1, :].contiguous()

    if D not in (64, 128):
        print(f"warning: head_dim={D} not in {{64, 128}}; tqkv_bench will reject it",
              file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    Q_decode.to(torch.float16).cpu().numpy().tofile(os.path.join(args.out, "Q.bin"))
    K.to(torch.float16).cpu().numpy().tofile(os.path.join(args.out, "K.bin"))
    V.to(torch.float16).cpu().numpy().tofile(os.path.join(args.out, "V.bin"))

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "B": int(B), "H": int(H), "D": int(D), "S": int(S),
            "seed": 0,
            "dtype": "float16",
            "mode": "real_kv",
            "block_size": 0,
            "source_model": MODEL_NAME,
            "source_model_type": "gpt2",
            "source_layer": int(layer_idx),
        }, f, indent=2)
        f.write("\n")

    print(f"wrote {args.out} (B={B} H={H} D={D} S={S} layer={layer_idx})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
