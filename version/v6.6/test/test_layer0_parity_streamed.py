#!/usr/bin/env python3
"""
Layer 0 parity test (streamed/packed layout).

Compares CK intermediate tensors against llama.cpp dumps using offsets
from lowered_decode.json (IR Lower 2).
"""

import argparse
import os
import ctypes
import json
from pathlib import Path

import numpy as np


def load_llama_tensor(name: str, dump_dir: Path) -> np.ndarray | None:
    path = dump_dir / f"{name}.bin"
    if not path.exists():
        return None
    return np.fromfile(str(path), dtype=np.float32)


def read_f32(base_ptr: int, offset: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def find_op(
    ops,
    op_name: str,
    layer: int,
    occurrence: int = 0,
    output_buffer: str | None = None,
    weight_key: str | None = None,
):
    matches = [op for op in ops if op.get("op") == op_name and op.get("layer") == layer]
    if weight_key:
        matches = [op for op in matches if weight_key in (op.get("weights") or {})]
    if output_buffer:
        filtered = []
        for op in matches:
            outputs = op.get("outputs", {})
            if any(out.get("buffer") == output_buffer for out in outputs.values()):
                filtered.append(op)
        matches = filtered
    if len(matches) <= occurrence:
        return None
    return matches[occurrence]


def get_output_offset(
    ops,
    op_name: str,
    layer: int,
    output_key: str,
    occurrence: int = 0,
    output_buffer: str | None = None,
    weight_key: str | None = None,
) -> int | None:
    op = find_op(ops, op_name, layer, occurrence, output_buffer=output_buffer, weight_key=weight_key)
    if not op:
        return None
    out = op.get("outputs", {}).get(output_key)
    if not out:
        return None
    return int(out.get("activation_offset", 0))


def get_op_sequence_index(ops, op_name: str, layer: int, occurrence: int = 0, output_buffer: str | None = None, weight_key: str | None = None) -> int | None:
    matches = []
    for idx, op in enumerate(ops):
        if op.get("op") != op_name or op.get("layer") != layer:
            continue
        if weight_key and weight_key not in (op.get("weights") or {}):
            continue
        if output_buffer:
            outputs = op.get("outputs", {})
            if not any(out.get("buffer") == output_buffer for out in outputs.values()):
                continue
        matches.append(idx)
    if len(matches) <= occurrence:
        return None
    return matches[occurrence]


def zero_activations(base_ptr: int, arena: dict, memory: dict) -> None:
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

    def abs_offset(buf: dict) -> int:
        if "abs_offset" in buf:
            return int(buf["abs_offset"])
        return act_base + int(buf.get("offset", 0))

    if not rope_buf:
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


def compare_tensors(name: str, llama: np.ndarray | None, ck: np.ndarray | None, tol: float = 0.05) -> dict:
    if llama is None:
        return {"name": name, "status": "llama_missing"}
    if ck is None:
        return {"name": name, "status": "ck_missing"}

    llama_flat = llama.flatten()
    ck_flat = ck.flatten()
    n = min(len(llama_flat), len(ck_flat))
    llama_flat = llama_flat[:n]
    ck_flat = ck_flat[:n]

    diff = np.abs(llama_flat - ck_flat)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    max_abs = max(float(np.max(np.abs(llama_flat))), 1e-9)
    rel_err = max_diff / max_abs

    return {
        "name": name,
        "status": "pass" if rel_err < tol else "FAIL",
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_err": rel_err,
        "size": n,
    }


def rmsnorm_reference(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    gamma = gamma.astype(np.float32, copy=False)
    mean_sq = np.mean(x * x, dtype=np.float32)
    rstd = 1.0 / np.sqrt(mean_sq + np.float32(eps))
    return x * rstd * gamma


def get_rmsnorm_op(ops, layer: int, output_buffer: str | None) -> dict | None:
    matches = [op for op in ops if op.get("op") == "rmsnorm" and op.get("layer") == layer]
    if output_buffer:
        filtered = []
        for op in matches:
            outputs = op.get("outputs", {})
            if any(out.get("buffer") == output_buffer for out in outputs.values()):
                filtered.append(op)
        matches = filtered
    return matches[0] if matches else None


def pick_rmsnorm_gamma(op: dict, kind: str) -> dict | None:
    weights = op.get("weights", {})
    if kind == "attn_norm":
        return weights.get("ln1_gamma") or next(iter(weights.values()), None)
    if kind == "ffn_norm":
        return weights.get("ln2_gamma") or next(iter(weights.values()), None)
    return next(iter(weights.values()), None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer 0 parity test using streamed layout")
    ap.add_argument("--model-dir", required=True, help="CK model directory")
    ap.add_argument("--dump-dir", default="llama_dump", help="llama.cpp dump directory")
    ap.add_argument("--lowered", default=None, help="Path to lowered_decode.json (IR Lower 2)")
    ap.add_argument("--token", type=int, default=9707, help="Token ID")
    ap.add_argument("--layer", type=int, default=0, help="Layer index")
    ap.add_argument("--debug-rmsnorm", action="store_true", help="Dump RMSNorm reference comparisons")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    dump_dir = Path(args.dump_dir)
    lowered_path = Path(args.lowered) if args.lowered else model_dir / "lowered_decode.json"

    if not lowered_path.exists():
        print(f"ERROR: lowered IR not found: {lowered_path}")
        return 1

    with open(lowered_path, "r") as f:
        lowered = json.load(f)
    ops = lowered.get("operations", [])
    cfg = lowered.get("config", {})
    memory = lowered.get("memory", {})
    arena = memory.get("arena", {})
    if not arena:
        print("ERROR: lowered IR missing memory arena; packed layout required for this test")
        return 1
    if arena.get("mode") != "packed":
        print(f"ERROR: layout mode is '{arena.get('mode')}', but this test requires packed layout")
        return 1

    embed_dim = int(cfg.get("embed_dim", 896))
    num_heads = int(cfg.get("num_heads", 14))
    num_kv_heads = int(cfg.get("num_kv_heads", 2))
    head_dim = int(cfg.get("head_dim", 64))
    intermediate = int(cfg.get("intermediate_size", cfg.get("intermediate_dim", 4864)))
    rms_eps = float(cfg.get("rms_eps", 1e-6))

    # Mapping: llama dump tensor -> (op_name, output_key, occurrence, size)
    mapping = [
        ("inp_embd", "dense_embedding_lookup", "output", 0, embed_dim, None, None, -1),
        ("attn_norm-0", "rmsnorm", "output", 0, embed_dim, "layer_input", None, None),
        ("Qcur-0", "bias_add", "y", 0, num_heads * head_dim, None, "bq", None),
        ("Kcur-0", "bias_add", "y", 0, num_kv_heads * head_dim, None, "bk", None),
        ("Vcur-0", "bias_add", "y", 0, num_kv_heads * head_dim, None, "bv", None),
        ("__fattn__-0", "attn", "out", 0, num_heads * head_dim, None, None, None),
        ("kqv_out-0", "bias_add", "y", 0, embed_dim, None, "bo", None),
        ("ffn_inp-0", "residual_add", "out", 0, embed_dim, None, None, None),
        ("ffn_norm-0", "rmsnorm", "output", 0, embed_dim, "embedded_input", None, None),
        ("ffn_swiglu-0", "silu_mul", "out", 0, intermediate, None, None, None),
        ("ffn_out-0", "bias_add", "y", 0, embed_dim, None, "b2", None),
        ("l_out-0", "residual_add", "out", 1, embed_dim, None, None, None),
    ]

    lib_path = model_dir / "libmodel.so"
    if not lib_path.exists():
        print(f"ERROR: libmodel.so not found in {model_dir}")
        return 1

    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    lib.ck_model_kv_cache_reset.argtypes = []
    lib.ck_model_kv_cache_reset.restype = None

    weights_path = str(model_dir / "weights.bump")
    if lib.ck_model_init(weights_path.encode()) != 0:
        print("ERROR: ck_model_init failed")
        return 1

    base_ptr = lib.ck_model_get_base_ptr()
    if not base_ptr:
        print("ERROR: ck_model_get_base_ptr returned NULL")
        return 1

    results = []
    rms_debug = None
    for dump_name, op_name, out_key, occ, size, out_buf, weight_key, layer_override in mapping:
        llama = load_llama_tensor(dump_name, dump_dir)
        layer = args.layer if layer_override is None else layer_override
        offset = get_output_offset(ops, op_name, layer, out_key, occ, output_buffer=out_buf, weight_key=weight_key)
        seq_idx = get_op_sequence_index(ops, op_name, layer, occ, output_buffer=out_buf, weight_key=weight_key)
        ck = None
        if offset is not None and seq_idx is not None:
            os.environ["CK_STOP_OP"] = str(seq_idx)
            lib.ck_model_kv_cache_reset()
            zero_activations(base_ptr, arena, memory)
            if lib.ck_model_decode(ctypes.c_int32(args.token), None) != 0:
                print("ERROR: ck_model_decode failed")
                return 1
            ck = read_f32(base_ptr, offset, size)
            if args.debug_rmsnorm and dump_name == "attn_norm-0":
                op = get_rmsnorm_op(ops, layer, output_buffer="layer_input")
                if op:
                    inp = op.get("activations", {}).get("input", {})
                    out = op.get("outputs", {}).get("output", {})
                    gamma_info = pick_rmsnorm_gamma(op, "attn_norm")
                    if inp and out and gamma_info:
                        inp_off = int(inp.get("activation_offset", 0))
                        out_off = int(out.get("activation_offset", 0))
                        gamma_off = int(gamma_info.get("bump_offset", 0))
                        x = read_f32(base_ptr, inp_off, embed_dim)
                        gamma = read_f32(base_ptr, gamma_off, embed_dim)
                        y_ref = rmsnorm_reference(x, gamma, rms_eps)
                        y_ck = read_f32(base_ptr, out_off, embed_dim)
                        llama_attn = llama
                        rms_debug = {
                            "eps": rms_eps,
                            "ck_vs_ref": compare_tensors("ck_vs_ref", y_ref, y_ck),
                            "ref_vs_llama": compare_tensors("ref_vs_llama", llama_attn, y_ref),
                        }
        results.append(compare_tensors(dump_name, llama, ck))

    print("\nLayer 0 parity (streamed layout)")
    print(f"Model dir: {model_dir}")
    print(f"Dump dir:  {dump_dir}")
    print(f"Layer:     {args.layer}")
    print(f"{'Tensor':<18} {'Status':<6} {'RelErr':<10} {'MaxDiff':<12} {'Size':<8}")
    for r in results:
        status = r["status"]
        rel = r.get("rel_err", 0.0)
        mx = r.get("max_diff", 0.0)
        sz = r.get("size", 0)
        print(f"{r['name']:<18} {status:<6} {rel:>9.2%} {mx:>12.6f} {sz:>8}")

    if rms_debug:
        print("\nRMSNorm debug (attn_norm-0)")
        print(f"eps: {rms_debug['eps']}")
        ck_ref = rms_debug["ck_vs_ref"]
        ref_llama = rms_debug["ref_vs_llama"]
        print(
            f"ck_vs_ref   status={ck_ref['status']} rel_err={ck_ref['rel_err']:.2%} max_diff={ck_ref['max_diff']:.6f}"
        )
        print(
            f"ref_vs_llama status={ref_llama['status']} rel_err={ref_llama['rel_err']:.2%} max_diff={ref_llama['max_diff']:.6f}"
        )

    # Fail if any FAIL
    failed = [r for r in results if r["status"] == "FAIL"]
    if failed:
        print(f"\nFAIL: {len(failed)} tensors exceeded tolerance")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
