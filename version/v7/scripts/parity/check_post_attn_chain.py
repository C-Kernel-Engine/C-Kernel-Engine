#!/usr/bin/env python3
"""
Check layer-N post-attention chain contracts:
  1) If present, post_attention_norm output vs local RMSNorm reference of
     out_proj output.
  2) residual_add output vs expected residual fusion input.

Supports both:
  - older Gemma-style chain with explicit post_attention_norm op
  - current chain where out_proj feeds residual_add directly
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


def rmsnorm_ref(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    mean_sq = np.mean(x * x, dtype=np.float32)
    rstd = 1.0 / np.sqrt(mean_sq + np.float32(eps))
    return x * rstd * gamma


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    d = np.abs(a - b)
    return float(np.max(d)), float(np.mean(d)), int(np.argmax(d))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check post-attention norm/residual chain")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token", type=int, default=5)
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    lowered = load_json(model_dir / "lowered_decode.json")
    lowered_call = load_json(model_dir / "lowered_decode_call.json")
    manifest = load_json(model_dir / "weights_manifest.json")
    ops = lowered.get("operations", [])
    call_ops = lowered_call.get("operations", [])
    cfg = lowered.get("config", {})
    embed_dim = int(cfg.get("embed_dim", 640))
    eps = float(cfg.get("rms_eps", 1e-6))

    # Required ops in layer chain.
    res0_idx, res0_op = find_op(ops, args.layer, "residual_save", 0)   # after embedding
    out_idx, out_op = find_op(ops, args.layer, "out_proj", 0)
    pan_idx, pan_op = find_op(ops, args.layer, "post_attention_norm", 0)
    add_idx, add_op = find_op(ops, args.layer, "residual_add", 0)
    if None in (res0_op, out_op, add_op):
        raise RuntimeError("missing one of required ops: residual_save(0), out_proj, residual_add(0)")
    res0_stop_idx, _ = find_op(call_ops, args.layer, "residual_save", 0)
    out_stop_idx, _ = find_op(call_ops, args.layer, "out_proj", 0)
    pan_stop_idx, _ = find_op(call_ops, args.layer, "post_attention_norm", 0)
    add_stop_idx, _ = find_op(call_ops, args.layer, "residual_add", 0)
    if None in (res0_stop_idx, out_stop_idx, add_stop_idx):
        raise RuntimeError("missing one of required stop ops in lowered_decode_call")

    # Offsets.
    res0_rel = int(res0_op.get("outputs", {}).get("dst", {}).get("activation_offset", 0))
    out_outputs = out_op.get("outputs", {})
    out_binding = out_outputs.get("y") or out_outputs.get("C") or out_outputs.get("out") or {}
    out_rel = int(out_binding.get("activation_offset", 0))
    pan_rel = int((pan_op or {}).get("outputs", {}).get("output", {}).get("activation_offset", 0))
    add_rel = int(add_op.get("outputs", {}).get("out", {}).get("activation_offset", 0))
    res0_abs = resolve_abs_offset(lowered, res0_rel)
    out_abs = resolve_abs_offset(lowered, out_rel)
    pan_abs = resolve_abs_offset(lowered, pan_rel) if pan_op is not None else None
    add_abs = resolve_abs_offset(lowered, add_rel)

    pan_name = None
    pan_file_off = None
    pan_size = None
    if pan_op is not None:
        pan_w = pan_op.get("weights", {}).get("post_attention_norm")
        if not isinstance(pan_w, dict):
            raise RuntimeError("post_attention_norm weight binding missing")
        pan_name = str(pan_w.get("name"))
        entry_by_name = {e.get("name"): e for e in manifest.get("entries", []) if isinstance(e, dict)}
        ent = entry_by_name.get(pan_name)
        if not ent:
            raise RuntimeError(f"manifest entry missing for {pan_name}")
        pan_file_off = int(ent["file_offset"])
        pan_size = int(ent["size"])

    lib = ctypes.CDLL(str(model_dir / "libmodel.so"))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
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

    try:
        # Stop at out_proj.
        os.environ["CK_STOP_OP"] = str(out_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_decode(ctypes.c_int32(args.token), None) != 0:
            raise RuntimeError("decode failed at out_proj stop")
        y_out = read_f32(base_ptr, out_abs, embed_dim)

        # Stop at residual_add.
        os.environ["CK_STOP_OP"] = str(add_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_decode(ctypes.c_int32(args.token), None) != 0:
            raise RuntimeError("decode failed at residual_add stop")
        y_add = read_f32(base_ptr, add_abs, embed_dim)

        # Capture residual_save from same stop.
        y_res0 = read_f32(base_ptr, res0_abs, embed_dim)

        if pan_op is not None:
            # Stop at post_attention_norm when that op exists.
            os.environ["CK_STOP_OP"] = str(pan_stop_idx)
            if hasattr(lib, "ck_model_kv_cache_reset"):
                lib.ck_model_kv_cache_reset()
            zero_activations(base_ptr, lowered)
            if lib.ck_model_decode(ctypes.c_int32(args.token), None) != 0:
                raise RuntimeError("decode failed at post_attention_norm stop")
            y_pan = read_f32(base_ptr, int(pan_abs), embed_dim)
        else:
            y_pan = None
    finally:
        os.environ.pop("CK_STOP_OP", None)
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()

    if pan_op is not None:
        with (model_dir / "weights.bump").open("rb") as f:
            f.seek(int(pan_file_off))
            gamma = np.frombuffer(f.read(int(pan_size)), dtype=np.float32).copy()
        if gamma.size != embed_dim:
            raise RuntimeError(f"post_attention_norm gamma size mismatch: {gamma.size} vs {embed_dim}")
        pan_ref = rmsnorm_ref(y_out.astype(np.float32), gamma, eps)
        add_ref = y_pan + y_res0
        pan_max, pan_mean, pan_worst = diff_stats(y_pan, pan_ref)
    else:
        gamma = None
        pan_ref = None
        pan_max = pan_mean = 0.0
        pan_worst = 0
        add_ref = y_out + y_res0
    add_max, add_mean, add_worst = diff_stats(y_add, add_ref)

    print("=" * 88)
    print("POST-ATTN CHAIN CHECK")
    print("=" * 88)
    print(f"model_dir       : {model_dir}")
    print(f"layer/token     : {args.layer}/{args.token}")
    if pan_op is not None:
        print(
            f"indices         : stop(res0/out/pan/add)="
            f"{res0_stop_idx}/{out_stop_idx}/{pan_stop_idx}/{add_stop_idx} "
            f"(lowered={res0_idx}/{out_idx}/{pan_idx}/{add_idx})"
        )
    else:
        print(
            f"indices         : stop(res0/out/add)={res0_stop_idx}/{out_stop_idx}/{add_stop_idx} "
            f"(lowered={res0_idx}/{out_idx}/{add_idx})"
        )
    print(f"embed_dim/eps   : {embed_dim}/{eps}")
    if pan_op is not None:
        print(f"post_attn_gamma : {pan_name} (file_offset={pan_file_off}, size={pan_size})")
    else:
        print("post_attn_gamma : (none, direct out_proj -> residual_add chain)")
    print("")
    if pan_op is not None:
        print("post_attention_norm vs local RMSNorm(out_proj)")
        print(f"  max_diff      : {pan_max:.6e}")
        print(f"  mean_diff     : {pan_mean:.6e}")
        print(f"  worst idx     : {pan_worst}")
        print(f"  runtime[idx]  : {float(y_pan[pan_worst]):.8f}")
        print(f"  ref[idx]      : {float(pan_ref[pan_worst]):.8f}")
        print("")
        print("residual_add vs (post_attn_norm + residual_saved)")
    else:
        print("residual_add vs (out_proj + residual_saved)")
    print(f"  max_diff      : {add_max:.6e}")
    print(f"  mean_diff     : {add_mean:.6e}")
    print(f"  worst idx     : {add_worst}")
    print(f"  runtime[idx]  : {float(y_add[add_worst]):.8f}")
    print(f"  ref[idx]      : {float(add_ref[add_worst]):.8f}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
