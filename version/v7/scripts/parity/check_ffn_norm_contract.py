#!/usr/bin/env python3
"""
Check layer-N MLP pre-norm contract:
  - capture input right before MLP pre-norm (after first residual_add)
  - capture pre-norm output
  - compute local RMSNorm reference with bound ln2_gamma + eps
  - compare runtime vs reference

Supports both older IR naming (ffn_norm) and current naming (rmsnorm
occurrence #1 in the decoder block).
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


def _resolve_mlp_prenorm_op(
    ops: list[dict[str, Any]],
    call_ops: list[dict[str, Any]],
    layer: int,
) -> tuple[int, dict[str, Any], int, str]:
    """Find MLP pre-norm op across IR naming variants."""
    # Older naming.
    op_idx, op = find_op(ops, layer, "ffn_norm", 0)
    stop_idx, _ = find_op(call_ops, layer, "ffn_norm", 0)
    if op is not None and stop_idx is not None:
        return op_idx, op, stop_idx, "ffn_norm"

    # Current naming: second rmsnorm in the block.
    op_idx, op = find_op(ops, layer, "rmsnorm", 1)
    stop_idx, _ = find_op(call_ops, layer, "rmsnorm", 1)
    if op is not None and stop_idx is not None:
        return op_idx, op, stop_idx, "rmsnorm(occ=1)"

    raise RuntimeError("MLP pre-norm op not found (expected ffn_norm or rmsnorm occurrence 1)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check ffn_norm runtime vs local reference")
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

    ffn_idx, ffn_op, ffn_stop_idx, ffn_label = _resolve_mlp_prenorm_op(ops, call_ops, args.layer)
    res_idx, res_op = find_op(ops, args.layer, "residual_add", 0)
    if res_op is None:
        raise RuntimeError("residual_add op#0 not found")
    res_stop_idx, _ = find_op(call_ops, args.layer, "residual_add", 0)
    if res_stop_idx is None:
        raise RuntimeError("residual_add stop op not found in lowered_decode_call")

    out_binding = (
        ffn_op.get("outputs", {}).get("output")
        or ffn_op.get("outputs", {}).get("out")
        or ffn_op.get("outputs", {}).get("y")
        or {}
    )
    ffn_out_rel = int(out_binding.get("activation_offset", 0))
    ffn_out_abs = resolve_abs_offset(lowered, ffn_out_rel)
    res_out_rel = int(res_op.get("outputs", {}).get("out", {}).get("activation_offset", 0))
    res_out_abs = resolve_abs_offset(lowered, res_out_rel)

    ln2 = (
        ffn_op.get("weights", {}).get("ln2_gamma")
        or ffn_op.get("weights", {}).get("gamma")
        or ffn_op.get("weights", {}).get("weight")
    )
    if not isinstance(ln2, dict):
        raise RuntimeError("MLP pre-norm gamma binding missing")
    ln2_name = str(ln2.get("name"))
    entry_by_name = {e.get("name"): e for e in manifest.get("entries", []) if isinstance(e, dict)}
    ent = entry_by_name.get(ln2_name)
    if not ent:
        raise RuntimeError(f"manifest entry not found: {ln2_name}")
    ln2_file_off = int(ent["file_offset"])
    ln2_size = int(ent["size"])

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
        raise RuntimeError("base_ptr null")

    try:
        os.environ["CK_STOP_OP"] = str(res_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_decode(ctypes.c_int32(args.token), None) != 0:
            raise RuntimeError("decode failed at residual_add stop")
        x_pre = read_f32(base_ptr, res_out_abs, embed_dim)

        os.environ["CK_STOP_OP"] = str(ffn_stop_idx)
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset()
        zero_activations(base_ptr, lowered)
        if lib.ck_model_decode(ctypes.c_int32(args.token), None) != 0:
            raise RuntimeError("decode failed at ffn_norm stop")
        y_runtime = read_f32(base_ptr, ffn_out_abs, embed_dim)
    finally:
        os.environ.pop("CK_STOP_OP", None)
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()

    with (model_dir / "weights.bump").open("rb") as f:
        f.seek(ln2_file_off)
        gamma = np.frombuffer(f.read(ln2_size), dtype=np.float32).copy()

    if gamma.size != embed_dim:
        raise RuntimeError(f"gamma size mismatch: {gamma.size} vs embed_dim={embed_dim}")

    y_ref = rmsnorm_ref(x_pre.astype(np.float32), gamma, eps)
    diff = np.abs(y_runtime - y_ref)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    worst = int(np.argmax(diff))

    print("=" * 88)
    print("MLP PRE-NORM CONTRACT CHECK")
    print("=" * 88)
    print(f"model_dir      : {model_dir}")
    print(f"layer/token    : {args.layer}/{args.token}")
    print(f"norm op        : {ffn_label}")
    print(f"residual_add idx: stop={res_stop_idx}, lowered={res_idx}")
    print(f"prenorm idx    : stop={ffn_stop_idx}, lowered={ffn_idx}")
    print(f"embed_dim/eps  : {embed_dim}/{eps}")
    print(f"ln2_gamma      : {ln2_name} (file_offset={ln2_file_off}, size={ln2_size})")
    print("")
    print(f"max_diff       : {max_diff:.6e}")
    print(f"mean_diff      : {mean_diff:.6e}")
    print(f"worst idx      : {worst}")
    print(f"runtime[idx]   : {float(y_runtime[worst]):.8f}")
    print(f"ref[idx]       : {float(y_ref[worst]):.8f}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
