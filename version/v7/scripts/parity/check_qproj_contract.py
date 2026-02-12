#!/usr/bin/env python3
"""
Focused projection contract checker (q_proj/k_proj/v_proj).

What it does in one run:
1) Prints projection call dims (M, K) and binding details from lowered IR.
2) Captures runtime attn_norm output (x) and projection output (y) via CK_STOP_OP.
3) Builds local reference y_ref = dequant(W) @ x + b using exact bound offsets.
4) Reports mismatch stats and simple bias diagnostics.

TODO(model-contracts):
  - Keep this per-family projection contract in sync with template flags
    (qk-norm on/off, bias behavior, head layout).
  - Add a companion rope contract check (pre/post RoPE Q/K) for each family,
    since projection parity can pass while RoPE layout still diverges.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from probe_defaults import getenv_decode_token_id


DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v7/models/unsloth--gemma-3-270m-it-GGUF/ck_build"
SUPPORTED_OPS = ("q_proj", "k_proj", "v_proj")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_layer_op(ops: list[dict[str, Any]], layer: int, op_name: str) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    for i, op in enumerate(ops):
        if op.get("layer") == layer and op.get("op") == op_name:
            return i, op
    return None, None


def _find_layer_op_any(ops: list[dict[str, Any]], layer: int, op_names: tuple[str, ...]) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    for i, op in enumerate(ops):
        if op.get("layer") == layer and op.get("op") in op_names:
            return i, op
    return None, None


def _read_f32(base_ptr: int, offset: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def _zero_activations(base_ptr: int, lowered: dict[str, Any]) -> None:
    memory = lowered.get("memory", {})
    arena = memory.get("arena", {})
    act = memory.get("activations", {})
    act_size = int(act.get("size", 0))
    if act_size <= 0:
        return

    act_base = int(arena.get("activations_base", 0))
    rope_buf = None
    for buf in act.get("buffers", []):
        if buf.get("name") == "rope_cache":
            rope_buf = buf
            break

    def abs_offset(buf: dict[str, Any]) -> int:
        if "abs_offset" in buf:
            return int(buf["abs_offset"])
        return act_base + int(buf.get("offset", 0))

    if rope_buf is None:
        ctypes.memset(base_ptr + act_base, 0, act_size)
        return

    rope_off = abs_offset(rope_buf)
    rope_size = int(rope_buf.get("size", 0))
    if rope_size <= 0:
        ctypes.memset(base_ptr + act_base, 0, act_size)
        return

    pre = max(0, rope_off - act_base)
    if pre:
        ctypes.memset(base_ptr + act_base, 0, pre)
    end = act_base + act_size
    post_start = rope_off + rope_size
    if post_start < end:
        ctypes.memset(base_ptr + post_start, 0, end - post_start)


def _resolve_abs_offset(lowered: dict[str, Any], activation_offset: int) -> int:
    memory = lowered.get("memory", {})
    arena = memory.get("arena", {})
    mode = str(arena.get("mode", ""))
    act_base = int(arena.get("activations_base", 0))
    if mode == "region":
        return act_base + int(activation_offset)
    return int(activation_offset)


def _dequant_q5_1(weights_bytes: bytes, rows: int, cols: int) -> np.ndarray:
    if cols % 32 != 0:
        raise ValueError(f"q5_1 expected cols multiple of 32, got {cols}")

    n_vals = rows * cols
    n_blocks = n_vals // 32
    block_size = 24
    expected_bytes = n_blocks * block_size
    if len(weights_bytes) < expected_bytes:
        raise ValueError(f"q5_1 bytes too small: got={len(weights_bytes)} expected={expected_bytes}")

    out = np.empty(n_vals, dtype=np.float32)
    p = 0
    o = 0
    for _ in range(n_blocks):
        d = float(np.frombuffer(weights_bytes[p : p + 2], dtype=np.float16)[0])
        m = float(np.frombuffer(weights_bytes[p + 2 : p + 4], dtype=np.float16)[0])
        qh = int.from_bytes(weights_bytes[p + 4 : p + 8], byteorder="little", signed=False)
        qs = weights_bytes[p + 8 : p + 24]

        for j in range(16):
            lo = qs[j] & 0x0F
            hi = ((qh >> j) & 1) << 4
            out[o + j] = d * float(lo | hi) + m

        for j in range(16):
            lo = qs[j] >> 4
            hi = ((qh >> (j + 16)) & 1) << 4
            out[o + 16 + j] = d * float(lo | hi) + m

        p += block_size
        o += 32

    return out.reshape(rows, cols)


def _dequant_q8_0(weights_bytes: bytes, rows: int, cols: int) -> np.ndarray:
    if cols % 32 != 0:
        raise ValueError(f"q8_0 expected cols multiple of 32, got {cols}")

    n_vals = rows * cols
    n_blocks = n_vals // 32
    block_size = 34
    expected_bytes = n_blocks * block_size
    if len(weights_bytes) < expected_bytes:
        raise ValueError(f"q8_0 bytes too small: got={len(weights_bytes)} expected={expected_bytes}")

    out = np.empty(n_vals, dtype=np.float32)
    p = 0
    o = 0
    for _ in range(n_blocks):
        d = float(np.frombuffer(weights_bytes[p : p + 2], dtype=np.float16)[0])
        qs = np.frombuffer(weights_bytes[p + 2 : p + 34], dtype=np.int8).astype(np.float32)
        out[o : o + 32] = d * qs
        p += block_size
        o += 32
    return out.reshape(rows, cols)


def _run_until_op(
    lib: ctypes.CDLL,
    base_ptr: int,
    lowered: dict[str, Any],
    token_id: int,
    stop_idx: int,
    out_offset: int,
    out_count: int,
) -> np.ndarray:
    os.environ["CK_STOP_OP"] = str(stop_idx)
    if hasattr(lib, "ck_model_kv_cache_reset"):
        lib.ck_model_kv_cache_reset()
    _zero_activations(base_ptr, lowered)
    ret = lib.ck_model_decode(ctypes.c_int32(token_id), None)
    if ret != 0:
        raise RuntimeError(f"ck_model_decode failed at CK_STOP_OP={stop_idx}, ret={ret}")
    return _read_f32(base_ptr, out_offset, out_count)


def _finite_stats(name: str, arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    n_finite = int(np.count_nonzero(finite))
    n_total = int(arr.size)
    if n_finite == 0:
        return f"{name}: finite=0/{n_total}"
    vals = arr[finite]
    return (
        f"{name}: finite={n_finite}/{n_total}, "
        f"min={float(np.min(vals)):.6e}, max={float(np.max(vals)):.6e}, mean={float(np.mean(vals)):.6e}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Check projection runtime contract against local reference")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="ck_build model directory")
    ap.add_argument("--op", choices=SUPPORTED_OPS, default="q_proj", help="projection op to validate")
    ap.add_argument("--layer", type=int, default=0, help="layer index")
    ap.add_argument("--token", type=int, default=getenv_decode_token_id(), help="single decode token id")
    args = ap.parse_args()

    proj_op = args.op
    weight_key = {"q_proj": "wq", "k_proj": "wk", "v_proj": "wv"}[proj_op]
    bias_key = {"q_proj": "bq", "k_proj": "bk", "v_proj": "bv"}[proj_op]

    model_dir = args.model_dir.expanduser().resolve()
    lowered_path = model_dir / "lowered_decode.json"
    lowered_call_path = model_dir / "lowered_decode_call.json"
    manifest_path = model_dir / "weights_manifest.json"
    lib_path = model_dir / "libmodel.so"
    weights_path = model_dir / "weights.bump"
    for p in (lowered_path, lowered_call_path, manifest_path, lib_path, weights_path):
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")

    lowered = _load_json(lowered_path)
    lowered_call = _load_json(lowered_call_path)
    manifest = _load_json(manifest_path)
    ops = lowered.get("operations", [])
    call_ops = lowered_call.get("operations", [])
    entry_by_name = {
        e.get("name"): e
        for e in (manifest.get("entries", []) or [])
        if isinstance(e, dict) and e.get("name")
    }

    proj_idx, proj_ir = _find_layer_op(ops, args.layer, proj_op)
    if proj_ir is None:
        raise RuntimeError(f"{proj_op} not found for layer {args.layer}")
    proj_call_idx, proj_call_ir = _find_layer_op(call_ops, args.layer, proj_op)
    if proj_call_ir is None:
        raise RuntimeError(f"{proj_op} call op not found for layer {args.layer}")
    n_idx, n_op = _find_layer_op_any(ops, args.layer, ("attn_norm", "rmsnorm"))
    if n_op is None:
        raise RuntimeError(f"attn norm op not found for layer {args.layer}")

    proj_out = proj_ir.get("outputs", {}).get("y", {})
    proj_w = proj_ir.get("weights", {})
    w_ref_ir = proj_w.get(weight_key)
    b_ref_ir = proj_w.get(bias_key)
    n_out = n_op.get("outputs", {}).get("output", {})

    if w_ref_ir is None:
        raise RuntimeError(f"{proj_op} missing {weight_key} binding in lowered_decode.json")
    if "activation_offset" not in proj_out:
        raise RuntimeError(f"{proj_op} missing output activation_offset")
    if "activation_offset" not in n_out:
        raise RuntimeError("attn_norm missing output activation_offset")

    call_args = {a.get("name"): a.get("expr") for a in proj_call_ir.get("args", [])}
    try:
        m_rows = int(call_args.get("M", "0"))
        k_cols = int(call_args.get("K", "0"))
    except ValueError as e:
        raise RuntimeError(f"failed parsing call dims from lowered_decode_call.json: {e}") from e
    if m_rows <= 0 or k_cols <= 0:
        raise RuntimeError(f"invalid call dims from call IR: M={m_rows}, K={k_cols}")

    w_off = int(w_ref_ir["bump_offset"])
    w_size = int(w_ref_ir["size"])
    w_dtype = str(w_ref_ir.get("dtype"))
    if w_dtype not in ("q5_1", "q8_0"):
        raise RuntimeError(f"unsupported projection dtype for this checker: {w_dtype}")

    if b_ref_ir is not None:
        b_off = int(b_ref_ir["bump_offset"])
        b_size = int(b_ref_ir["size"])
    else:
        b_off = -1
        b_size = 0

    n_off_rel = int(n_out["activation_offset"])
    proj_off_rel = int(proj_out["activation_offset"])
    n_off = _resolve_abs_offset(lowered, n_off_rel)
    proj_off = _resolve_abs_offset(lowered, proj_off_rel)

    arena = lowered.get("memory", {}).get("arena", {})
    weights_base = int(arena.get("weights_base", 0))

    print("=" * 88)
    print(f"{proj_op.upper()} CONTRACT CHECK")
    print("=" * 88)
    print(f"model_dir        : {model_dir}")
    print(f"layer            : {args.layer}")
    print(f"token id         : {args.token}")
    print(f"attn_norm op idx : {n_idx}")
    print(f"{proj_op} op idx    : {proj_idx}")
    print(f"{proj_op} call idx  : {proj_call_idx}")
    print("")
    print(f"{proj_op} call")
    print(f"  function       : {proj_call_ir.get('function')}")
    print(f"  M (rows)       : {m_rows}")
    print(f"  K (cols)       : {k_cols}")
    print(f"  x source       : {call_args.get('x')}")
    print(f"  y source       : {call_args.get('y')}")
    print(f"  W source       : {call_args.get('W')}")
    print("")
    print(f"{proj_op} bindings")
    print(f"  {weight_key} name/dtype  : {w_ref_ir.get('name')} / {w_dtype}")
    print(f"  {weight_key} offset/size : {w_off} / {w_size}")
    if b_ref_ir is not None:
        print(f"  {bias_key} name/dtype    : {b_ref_ir.get('name')} / {b_ref_ir.get('dtype')}")
        print(f"  {bias_key} offset/size   : {b_off} / {b_size}")
    else:
        print(f"  {bias_key}               : <none>")
    print("")
    print("memory layout")
    print(f"  mode             : {arena.get('mode')}")
    print(f"  weights_base     : {weights_base}")
    print(f"  activations_base : {arena.get('activations_base')}")
    print(f"  attn_norm out    : rel={n_off_rel}, abs={n_off}")
    print(f"  {proj_op} out      : rel={proj_off_rel}, abs={proj_off}")
    print("")

    def _file_off(weight_name: str, bump_off: int) -> int:
        ent = entry_by_name.get(weight_name)
        if ent and ent.get("file_offset") is not None:
            return int(ent["file_offset"])
        return int(bump_off + weights_base)

    w_file_off = _file_off(str(w_ref_ir.get("name")), w_off)
    b_file_off = _file_off(str(b_ref_ir.get("name")), b_off) if b_ref_ir is not None else -1

    print("file offsets")
    print(f"  {weight_key} file_offset : {w_file_off}")
    if b_ref_ir is not None:
        print(f"  {bias_key} file_offset : {b_file_off}")
    print("")

    lib = ctypes.CDLL(str(lib_path))
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

    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        raise RuntimeError(f"ck_model_init failed with code {ret}")
    base_ptr_void = lib.ck_model_get_base_ptr()
    if not base_ptr_void:
        raise RuntimeError("ck_model_get_base_ptr returned NULL")
    base_ptr = int(base_ptr_void)

    try:
        x_runtime = _run_until_op(lib, base_ptr, lowered, args.token, int(n_idx), n_off, k_cols)
        y_runtime = _run_until_op(lib, base_ptr, lowered, args.token, int(proj_idx), proj_off, m_rows)
    finally:
        os.environ.pop("CK_STOP_OP", None)
        if hasattr(lib, "ck_model_free"):
            lib.ck_model_free()

    with weights_path.open("rb") as f:
        f.seek(w_file_off)
        w_bytes = f.read(w_size)
        if b_ref_ir is not None:
            f.seek(b_file_off)
            b_vec = np.frombuffer(f.read(b_size), dtype=np.float32).copy()
        else:
            b_vec = np.zeros((m_rows,), dtype=np.float32)

    if b_vec.size != m_rows:
        raise RuntimeError(f"bias size mismatch: {bias_key}.size={b_vec.size} vs M={m_rows}")

    if w_dtype == "q5_1":
        w_ref = _dequant_q5_1(w_bytes, rows=m_rows, cols=k_cols)
    elif w_dtype == "q8_0":
        w_ref = _dequant_q8_0(w_bytes, rows=m_rows, cols=k_cols)
    else:
        raise RuntimeError(f"unsupported dtype: {w_dtype}")

    # Avoid BLAS/OpenMP path for deterministic local reference.
    y_no_bias = np.sum(w_ref * x_runtime[np.newaxis, :], axis=1, dtype=np.float64).astype(np.float32)
    y_ref = y_no_bias + b_vec
    y_double = y_no_bias + (2.0 * b_vec)

    def err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        d = np.abs(a - b)
        return float(np.max(d)), float(np.mean(d))

    max_ref, mean_ref = err(y_runtime, y_ref)
    max_no_bias, mean_no_bias = err(y_runtime, y_no_bias)
    max_double, mean_double = err(y_runtime, y_double)
    d = np.abs(y_runtime - y_ref)
    first_idx = int(np.argmax(d))

    print("runtime vs local reference")
    print(f"  y_runtime shape : {tuple(y_runtime.shape)}")
    print(f"  y_ref shape     : {tuple(y_ref.shape)}")
    print(f"  max_diff        : {max_ref:.6e}")
    print(f"  mean_diff       : {mean_ref:.6e}")
    print(f"  worst idx       : {first_idx}")
    print(f"  runtime[idx]    : {float(y_runtime[first_idx]):.8f}")
    print(f"  ref[idx]        : {float(y_ref[first_idx]):.8f}")
    print("")
    print(_finite_stats("x_runtime", x_runtime))
    print(_finite_stats("y_runtime", y_runtime))
    print(_finite_stats("y_ref", y_ref))
    print("")
    print("bias diagnostics (smaller is better)")
    print(f"  vs (W@x + b)    : max={max_ref:.6e}, mean={mean_ref:.6e}")
    print(f"  vs (W@x)        : max={max_no_bias:.6e}, mean={mean_no_bias:.6e}")
    print(f"  vs (W@x + 2*b)  : max={max_double:.6e}, mean={mean_double:.6e}")
    print("")
    best = min(
        [("W@x+b", mean_ref), ("W@x", mean_no_bias), ("W@x+2b", mean_double)],
        key=lambda x: x[1],
    )[0]
    print(f"best-matching bias model: {best}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
