#!/usr/bin/env python3
"""
Prefill-specific check for layer-0 post-attention chain:
  1) post_attention_norm output vs local RMSNorm(out_proj) for all prompt tokens
  2) residual_add output vs (post_attention_norm_output + residual_saved)

TODO(model-contracts):
  - Run as a standard prefill contract for every supported template/family.
  - Pull expected norm behavior from per-model contract metadata to avoid
    hard-coded assumptions when onboarding future architectures.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_DIR = Path("/tmp/gemma_ctx256_ck_build")


def load_json(p: Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_op(ops: list[dict[str, Any]], layer: int, op_name: str, occurrence: int = 0) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    idxs = [i for i, op in enumerate(ops) if op.get("layer") == layer and op.get("op") == op_name]
    if len(idxs) <= occurrence:
        return None, None
    i = idxs[occurrence]
    return i, ops[i]


def resolve_abs_offset(lowered: dict[str, Any], rel_off: int) -> int:
    arena = lowered.get("memory", {}).get("arena", {})
    mode = str(arena.get("mode", ""))
    act_base = int(arena.get("activations_base", 0))
    if mode == "region":
        return act_base + int(rel_off)
    return int(rel_off)


def read_f32(base_ptr: int, abs_offset: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + abs_offset, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def zero_activations(base_ptr: int, lowered: dict[str, Any]) -> None:
    memory = lowered.get("memory", {})
    arena = memory.get("arena", {})
    act = memory.get("activations", {})
    act_size = int(act.get("size", 0))
    if act_size <= 0:
        return
    act_base = int(arena.get("activations_base", 0))
    ctypes.memset(base_ptr + act_base, 0, act_size)


def rmsnorm_ref_rows(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    # x: [T, D], gamma: [D]
    out = np.empty_like(x, dtype=np.float32)
    for t in range(x.shape[0]):
        row = x[t]
        mean_sq = np.mean(row * row, dtype=np.float32)
        rstd = 1.0 / np.sqrt(mean_sq + np.float32(eps))
        out[t] = row * rstd * gamma
    return out


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    d = np.abs(a - b)
    return float(np.max(d)), float(np.mean(d)), int(np.argmax(d))


def parse_tokens_csv(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check prefill post-attention norm/residual chain")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", default="2,9259", help="comma-separated token ids (count>1 to force prefill)")
    args = ap.parse_args()

    tokens = parse_tokens_csv(args.tokens)
    if len(tokens) <= 1:
        raise RuntimeError("Need at least 2 tokens to force prefill path")

    model_dir = args.model_dir.expanduser().resolve()
    lowered = load_json(model_dir / "lowered_prefill.json")
    lowered_call = load_json(model_dir / "lowered_prefill_call.json")
    manifest = load_json(model_dir / "weights_manifest.json")
    ops = lowered.get("operations", [])
    call_ops = lowered_call.get("operations", [])
    cfg = lowered.get("config", {})
    embed_dim = int(cfg.get("embed_dim", 640))
    eps = float(cfg.get("rms_eps", 1e-6))
    tcount = len(tokens)

    res0_idx, res0_op = find_op(ops, args.layer, "residual_save", 0)
    out_idx, out_op = find_op(ops, args.layer, "out_proj", 0)
    pan_idx, pan_op = find_op(ops, args.layer, "post_attention_norm", 0)
    add_idx, add_op = find_op(ops, args.layer, "residual_add", 0)
    if None in (res0_op, out_op, pan_op, add_op):
        raise RuntimeError("missing one of required prefill ops")
    res0_stop_idx, _ = find_op(call_ops, args.layer, "residual_save", 0)
    out_stop_idx, _ = find_op(call_ops, args.layer, "out_proj", 0)
    pan_stop_idx, _ = find_op(call_ops, args.layer, "post_attention_norm", 0)
    add_stop_idx, _ = find_op(call_ops, args.layer, "residual_add", 0)
    if None in (res0_stop_idx, out_stop_idx, pan_stop_idx, add_stop_idx):
        raise RuntimeError("missing one of required prefill stop ops in lowered_prefill_call")

    res0_rel = int(res0_op.get("outputs", {}).get("dst", {}).get("activation_offset", 0))
    out_outputs = out_op.get("outputs", {})
    out_binding = out_outputs.get("y") or out_outputs.get("C") or out_outputs.get("out") or {}
    out_rel = int(out_binding.get("activation_offset", 0))
    pan_rel = int(pan_op.get("outputs", {}).get("output", {}).get("activation_offset", 0))
    add_rel = int(add_op.get("outputs", {}).get("out", {}).get("activation_offset", 0))
    res0_abs = resolve_abs_offset(lowered, res0_rel)
    out_abs = resolve_abs_offset(lowered, out_rel)
    pan_abs = resolve_abs_offset(lowered, pan_rel)
    add_abs = resolve_abs_offset(lowered, add_rel)

    pan_w = pan_op.get("weights", {}).get("post_attention_norm")
    if not isinstance(pan_w, dict):
        raise RuntimeError("post_attention_norm binding missing")
    pan_name = str(pan_w.get("name"))
    entry_by_name = {e.get("name"): e for e in manifest.get("entries", []) if isinstance(e, dict)}
    ent = entry_by_name.get(pan_name)
    if not ent:
        raise RuntimeError(f"manifest entry missing: {pan_name}")
    pan_file_off = int(ent["file_offset"])
    pan_size = int(ent["size"])

    lib = ctypes.CDLL(str(model_dir / "libmodel.so"))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    if hasattr(lib, "ck_model_kv_cache_reset"):
        lib.ck_model_kv_cache_reset.argtypes = []
        lib.ck_model_kv_cache_reset.restype = None
    if hasattr(lib, "ck_model_free"):
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

    if lib.ck_model_init(str(model_dir / "weights.bump").encode()) != 0:
        raise RuntimeError("ck_model_init failed")
    base_ptr = int(lib.ck_model_get_base_ptr())
    if not base_ptr:
        raise RuntimeError("base ptr null")

    tok_arr = (ctypes.c_int32 * tcount)(*tokens)

    try:
        os.environ["CK_STOP_OP"] = str(out_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_embed_tokens(tok_arr, tcount) != 0:
            raise RuntimeError("embed_tokens failed at out_proj stop")
        y_out = read_f32(base_ptr, out_abs, tcount * embed_dim).reshape(tcount, embed_dim)

        os.environ["CK_STOP_OP"] = str(pan_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_embed_tokens(tok_arr, tcount) != 0:
            raise RuntimeError("embed_tokens failed at post_attention_norm stop")
        y_pan = read_f32(base_ptr, pan_abs, tcount * embed_dim).reshape(tcount, embed_dim)
        y_res0 = read_f32(base_ptr, res0_abs, tcount * embed_dim).reshape(tcount, embed_dim)

        os.environ["CK_STOP_OP"] = str(add_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_embed_tokens(tok_arr, tcount) != 0:
            raise RuntimeError("embed_tokens failed at residual_add stop")
        y_add = read_f32(base_ptr, add_abs, tcount * embed_dim).reshape(tcount, embed_dim)
    finally:
        os.environ.pop("CK_STOP_OP", None)
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()

    with (model_dir / "weights.bump").open("rb") as f:
        f.seek(pan_file_off)
        gamma = np.frombuffer(f.read(pan_size), dtype=np.float32).copy()
    if gamma.size != embed_dim:
        raise RuntimeError(f"gamma size mismatch: {gamma.size} vs {embed_dim}")

    pan_ref = rmsnorm_ref_rows(y_out.astype(np.float32), gamma, eps)
    add_ref = y_pan + y_res0

    pan_max, pan_mean, pan_worst = diff_stats(y_pan, pan_ref)
    add_max, add_mean, add_worst = diff_stats(y_add, add_ref)
    pan_t, pan_d = divmod(pan_worst, embed_dim)
    add_t, add_d = divmod(add_worst, embed_dim)

    print("=" * 88)
    print("POST-ATTN PREFILL CHAIN CHECK")
    print("=" * 88)
    print(f"model_dir       : {model_dir}")
    print(f"layer/tokens    : {args.layer}/{tokens}")
    print(
        f"indices         : stop(res0/out/pan/add)="
        f"{res0_stop_idx}/{out_stop_idx}/{pan_stop_idx}/{add_stop_idx} "
        f"(lowered={res0_idx}/{out_idx}/{pan_idx}/{add_idx})"
    )
    print(f"embed_dim/eps   : {embed_dim}/{eps}")
    print(f"post_attn_gamma : {pan_name} (file_offset={pan_file_off}, size={pan_size})")
    print("")
    print("post_attention_norm vs local RMSNorm(out_proj)")
    print(f"  max_diff      : {pan_max:.6e}")
    print(f"  mean_diff     : {pan_mean:.6e}")
    print(f"  worst [t,d]   : [{pan_t},{pan_d}]")
    print(f"  runtime[t,d]  : {float(y_pan[pan_t, pan_d]):.8f}")
    print(f"  ref[t,d]      : {float(pan_ref[pan_t, pan_d]):.8f}")
    print("")
    print("residual_add vs (post_attn_norm + residual_saved)")
    print(f"  max_diff      : {add_max:.6e}")
    print(f"  mean_diff     : {add_mean:.6e}")
    print(f"  worst [t,d]   : [{add_t},{add_d}]")
    print(f"  runtime[t,d]  : {float(y_add[add_t, add_d]):.8f}")
    print(f"  ref[t,d]      : {float(add_ref[add_t, add_d]):.8f}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
