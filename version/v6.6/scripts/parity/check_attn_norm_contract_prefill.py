#!/usr/bin/env python3
"""
Prefill-specific check for layer-0 (or chosen layer) attn_norm contract:
  1) Capture attn_norm input right before op execution
  2) Capture attn_norm output after op execution
  3) Compute local RMSNorm reference with ln1_gamma + eps for all prompt tokens
  4) Report runtime vs reference mismatch stats

TODO(model-contracts):
  - Run this per model family/template as a standard onboarding gate.
  - Read norm-contract metadata (norm variant, epsilon key, scaling behavior)
    from template/config once available, instead of assuming plain RMSNorm.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_tokens_csv(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def find_op(
    ops: list[dict[str, Any]],
    layer: int,
    op_name: str,
    occurrence: int = 0,
) -> tuple[int | None, dict[str, Any] | None]:
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
    eps32 = np.float32(eps)
    for t in range(x.shape[0]):
        row = x[t]
        mean_sq = np.mean(row * row, dtype=np.float32)
        rstd = np.float32(1.0) / np.sqrt(mean_sq + eps32)
        out[t] = row * rstd * gamma
    return out


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    d = np.abs(a - b)
    return float(np.max(d)), float(np.mean(d)), int(np.argmax(d))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check prefill attn_norm runtime vs local RMSNorm reference")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="ck_build model directory")
    ap.add_argument("--layer", type=int, default=0, help="layer index")
    ap.add_argument("--occurrence", type=int, default=0, help="which attn_norm occurrence in layer")
    ap.add_argument("--tokens", default="2,9259", help="comma-separated token ids (need >1 for prefill)")
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
    eps = float(cfg.get("rms_eps", cfg.get("rms_norm_eps", 1e-6)))
    tcount = len(tokens)

    attn_idx, attn_op = find_op(ops, args.layer, "attn_norm", args.occurrence)
    if attn_op is None or attn_idx is None:
        raise RuntimeError(f"attn_norm not found for layer={args.layer}, occurrence={args.occurrence}")
    attn_stop_idx, _ = find_op(call_ops, args.layer, "attn_norm", args.occurrence)
    if attn_stop_idx is None:
        raise RuntimeError(f"attn_norm stop index not found in lowered_prefill_call for layer={args.layer}, occurrence={args.occurrence}")
    if attn_stop_idx <= 0:
        raise RuntimeError(f"attn_norm stop index invalid for pre-op capture: idx={attn_stop_idx}")
    prev_stop_idx = attn_stop_idx - 1

    in_rel = int(attn_op.get("activations", {}).get("input", {}).get("activation_offset", 0))
    out_rel = int(attn_op.get("outputs", {}).get("output", {}).get("activation_offset", 0))
    in_abs = resolve_abs_offset(lowered, in_rel)
    out_abs = resolve_abs_offset(lowered, out_rel)

    ln1 = attn_op.get("weights", {}).get("ln1_gamma")
    if not isinstance(ln1, dict):
        raise RuntimeError("attn_norm ln1_gamma binding missing")
    ln1_name = str(ln1.get("name"))
    entry_by_name = {e.get("name"): e for e in manifest.get("entries", []) if isinstance(e, dict)}
    ent = entry_by_name.get(ln1_name)
    if not ent:
        raise RuntimeError(f"manifest entry missing for {ln1_name}")
    gamma_file_off = int(ent["file_offset"])
    gamma_size = int(ent["size"])

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
        os.environ["CK_STOP_OP"] = str(prev_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_embed_tokens(tok_arr, tcount) != 0:
            raise RuntimeError(f"embed_tokens failed at CK_STOP_OP={prev_stop_idx}")
        x_pre = read_f32(base_ptr, in_abs, tcount * embed_dim).reshape(tcount, embed_dim)

        os.environ["CK_STOP_OP"] = str(attn_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_embed_tokens(tok_arr, tcount) != 0:
            raise RuntimeError(f"embed_tokens failed at CK_STOP_OP={attn_stop_idx}")
        y_runtime = read_f32(base_ptr, out_abs, tcount * embed_dim).reshape(tcount, embed_dim)
    finally:
        os.environ.pop("CK_STOP_OP", None)
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()

    with (model_dir / "weights.bump").open("rb") as f:
        f.seek(gamma_file_off)
        gamma = np.frombuffer(f.read(gamma_size), dtype=np.float32).copy()
    if gamma.size != embed_dim:
        raise RuntimeError(f"ln1_gamma size mismatch: {gamma.size} vs embed_dim={embed_dim}")

    y_ref = rmsnorm_ref_rows(x_pre.astype(np.float32), gamma, eps)
    max_diff, mean_diff, worst = diff_stats(y_runtime, y_ref)
    worst_t, worst_d = divmod(worst, embed_dim)

    print("=" * 88)
    print("ATTN_NORM PREFILL CONTRACT CHECK")
    print("=" * 88)
    print(f"model_dir       : {model_dir}")
    print(f"layer/occ       : {args.layer}/{args.occurrence}")
    print(f"tokens          : {tokens}")
    print(f"indices         : stop_prev={prev_stop_idx}, stop_attn_norm={attn_stop_idx} (lowered_idx={attn_idx})")
    print(f"embed_dim/eps   : {embed_dim}/{eps}")
    print(f"ln1_gamma       : {ln1_name} (file_offset={gamma_file_off}, size={gamma_size})")
    print(f"in/out offsets  : {in_rel}/{out_rel} (abs {in_abs}/{out_abs})")
    print("")
    print("attn_norm runtime vs local RMSNorm(input)")
    print(f"  max_diff      : {max_diff:.6e}")
    print(f"  mean_diff     : {mean_diff:.6e}")
    print(f"  worst [t,d]   : [{worst_t},{worst_d}]")
    print(f"  runtime[t,d]  : {float(y_runtime[worst_t, worst_d]):.8f}")
    print(f"  ref[t,d]      : {float(y_ref[worst_t, worst_d]):.8f}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
