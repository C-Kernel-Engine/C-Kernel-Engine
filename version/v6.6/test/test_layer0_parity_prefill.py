#!/usr/bin/env python3
"""
Layer 0 prefill parity test.

Compares llama.cpp prompt-eval dumps (multi-token) against CK decode
run sequentially over the same token list. Uses CK_STOP_OP to capture
intermediate outputs for a given op.

Expected dumps: llama_dump/*.bin + prompt_tokens.json
"""

import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np


def load_llama_tensor(name: str, dump_dir: Path) -> np.ndarray | None:
    path = dump_dir / f"{name}.bin"
    if not path.exists():
        return None
    return np.fromfile(str(path), dtype=np.float32)


def read_f32(base_ptr: int, offset_bytes: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + offset_bytes, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,)).copy()


def parse_tokens(tokens_arg: str | None, dump_dir: Path) -> list[int] | None:
    if tokens_arg:
        p = Path(tokens_arg)
        if p.exists():
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("tokens", data.get("prompt_tokens", data))
            return [int(x) for x in data]
        # Allow JSON list literal or comma-separated
        try:
            data = json.loads(tokens_arg)
            if isinstance(data, list):
                return [int(x) for x in data]
        except json.JSONDecodeError:
            pass
        return [int(x.strip()) for x in tokens_arg.split(",") if x.strip()]

    token_path = dump_dir / "prompt_tokens.json"
    if token_path.exists():
        with open(token_path, "r") as f:
            data = json.load(f)
        return [int(x) for x in data]

    return None


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


def find_op(
    ops,
    op_name: str,
    layer: int,
    occurrence: int = 0,
    output_buffer: str | None = None,
    weight_key: str | None = None,
    section: str | None = None,
):
    matches = []
    for op in ops:
        if op.get("op") != op_name or op.get("layer") != layer:
            continue
        if section and op.get("section") != section:
            continue
        if weight_key and weight_key not in (op.get("weights") or {}):
            continue
        if output_buffer:
            outputs = op.get("outputs", {})
            if not any(out.get("buffer") == output_buffer for out in outputs.values()):
                continue
        matches.append(op)
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
    section: str | None = None,
):
    op = find_op(ops, op_name, layer, occurrence, output_buffer, weight_key, section)
    if not op:
        return None
    out = op.get("outputs", {}).get(output_key)
    if not out:
        return None
    return int(out.get("activation_offset", 0))


def get_op_sequence_index(
    ops,
    op_name: str,
    layer: int,
    occurrence: int = 0,
    output_buffer: str | None = None,
    weight_key: str | None = None,
    section: str | None = None,
):
    matches = []
    for idx, op in enumerate(ops):
        if op.get("op") != op_name or op.get("layer") != layer:
            continue
        if section and op.get("section") != section:
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


def compare_batches(name: str, llama: np.ndarray | None, ck: np.ndarray | None, tol: float = 0.05) -> dict:
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


def collect_ck_tokens(
    lib,
    base_ptr: int,
    arena: dict,
    memory: dict,
    tokens: list[int],
    seq_idx: int,
    offset_bytes: int,
    size: int,
) -> np.ndarray:
    out = np.zeros((len(tokens), size), dtype=np.float32)
    for i, tok in enumerate(tokens):
        lib.ck_model_kv_cache_reset()
        os.environ["CK_STOP_OP"] = "-1"
        for t in tokens[:i]:
            zero_activations(base_ptr, arena, memory)
            lib.ck_model_decode(ctypes.c_int32(t), None)
        os.environ["CK_STOP_OP"] = str(seq_idx)
        zero_activations(base_ptr, arena, memory)
        lib.ck_model_decode(ctypes.c_int32(tok), None)
        out[i, :] = read_f32(base_ptr, offset_bytes, size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer 0 prefill parity test")
    ap.add_argument("--model-dir", required=True, help="CK model directory")
    ap.add_argument("--dump-dir", default="llama_dump", help="llama.cpp dump directory")
    ap.add_argument("--lowered", default=None, help="Path to lowered_decode.json (IR Lower 2)")
    ap.add_argument("--tokens", default=None, help="Prompt tokens (JSON file or comma list)")
    ap.add_argument("--layer", type=int, default=0, help="Layer index")
    ap.add_argument("--max-tokens", type=int, default=0, help="Limit number of tokens (0 = all)")
    ap.add_argument("--allow-size-mismatch", action="store_true", help="Allow token count mismatch")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    dump_dir = Path(args.dump_dir)
    lowered_path = Path(args.lowered) if args.lowered else model_dir / "lowered_decode.json"

    if not lowered_path.exists():
        print(f"ERROR: lowered IR not found: {lowered_path}")
        return 1
    if not dump_dir.exists():
        print(f"ERROR: dump dir not found: {dump_dir}")
        return 1

    tokens = parse_tokens(args.tokens, dump_dir)
    if not tokens:
        print("ERROR: prompt tokens not found. Provide --tokens or dump_dir/prompt_tokens.json")
        return 1

    if args.max_tokens and len(tokens) > args.max_tokens:
        tokens = tokens[: args.max_tokens]

    with open(lowered_path, "r") as f:
        lowered = json.load(f)
    ops = lowered.get("operations", [])
    cfg = lowered.get("config", {})
    memory = lowered.get("memory", {})
    arena = memory.get("arena", {})
    if not arena:
        print("ERROR: lowered IR missing memory arena; packed layout required")
        return 1
    if arena.get("mode") != "packed":
        print(f"ERROR: layout mode is '{arena.get('mode')}', but this test requires packed layout")
        return 1

    embed_dim = int(cfg.get("embed_dim", 896))
    num_heads = int(cfg.get("num_heads", 14))
    num_kv_heads = int(cfg.get("num_kv_heads", 2))
    head_dim = int(cfg.get("head_dim", 64))
    intermediate = int(cfg.get("intermediate_size", cfg.get("intermediate_dim", 4864)))

    mapping = [
        ("inp_embd", "dense_embedding_lookup", "output", 0, embed_dim, None, None, None, 0),
        ("attn_norm-0", "rmsnorm", "output", 0, embed_dim, "layer_input", None, None, 0),
        ("Qcur-0", "bias_add", "y", 0, num_heads * head_dim, None, "bq", None, 0),
        ("Kcur-0", "bias_add", "y", 0, num_kv_heads * head_dim, None, "bk", None, 0),
        ("Vcur-0", "bias_add", "y", 0, num_kv_heads * head_dim, None, "bv", None, 0),
        ("__fattn__-0", "attn", "out", 0, num_heads * head_dim, None, None, None, 0),
        ("kqv_out-0", "bias_add", "y", 0, embed_dim, None, "bo", None, 0),
        ("ffn_inp-0", "residual_add", "out", 0, embed_dim, None, None, None, 0),
        ("ffn_norm-0", "rmsnorm", "output", 0, embed_dim, "embedded_input", None, None, 0),
        # ffn_gate/up are slices of mlp_gate_up output (2 * intermediate)
        ("ffn_gate-0", "mlp_gate_up", "y", 0, intermediate, None, None, None, 0),
        ("ffn_up-0", "mlp_gate_up", "y", 0, intermediate, None, None, None, intermediate),
        ("ffn_swiglu-0", "silu_mul", "out", 0, intermediate, None, None, None, 0),
        ("ffn_out-0", "bias_add", "y", 0, embed_dim, None, "b2", None, 0),
        ("l_out-0", "residual_add", "out", 1, embed_dim, None, None, None, 0),
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
    for dump_name, op_name, out_key, occ, size, out_buf, weight_key, section, slice_start in mapping:
        llama = load_llama_tensor(dump_name, dump_dir)
        if llama is None:
            results.append({"name": dump_name, "status": "llama_missing"})
            continue

        elems = llama.size
        if elems % size != 0:
            print(f"WARNING: {dump_name} size {elems} not divisible by {size}")
            if not args.allow_size_mismatch:
                results.append({"name": dump_name, "status": "llama_size_mismatch"})
                continue
        n_tokens_llama = elems // size
        if n_tokens_llama != len(tokens):
            msg = f"WARNING: {dump_name} token count {n_tokens_llama} != prompt tokens {len(tokens)}"
            print(msg)
            if not args.allow_size_mismatch:
                results.append({"name": dump_name, "status": "token_count_mismatch"})
                continue
        n_tokens = min(n_tokens_llama, len(tokens))

        llama_batch = llama[: n_tokens * size].reshape(n_tokens, size)

        offset = get_output_offset(ops, op_name, args.layer, out_key, occ, output_buffer=out_buf, weight_key=weight_key, section=section)
        seq_idx = get_op_sequence_index(ops, op_name, args.layer, occ, output_buffer=out_buf, weight_key=weight_key, section=section)
        if offset is None or seq_idx is None:
            results.append({"name": dump_name, "status": "ck_missing"})
            continue

        offset_bytes = offset + (slice_start * 4)
        ck_batch = collect_ck_tokens(lib, base_ptr, arena, memory, tokens[:n_tokens], seq_idx, offset_bytes, size)

        results.append(compare_batches(dump_name, llama_batch, ck_batch))

    print("\nLayer 0 prefill parity")
    print(f"Model dir: {model_dir}")
    print(f"Dump dir:  {dump_dir}")
    print(f"Tokens:    {len(tokens)}")
    print(f"Layer:     {args.layer}")
    print(f"{'Tensor':<18} {'Status':<8} {'RelErr':<10} {'MaxDiff':<12} {'Size':<8}")
    for r in results:
        status = r.get("status", "")
        rel = r.get("rel_err", 0.0)
        mx = r.get("max_diff", 0.0)
        sz = r.get("size", 0)
        print(f"{r['name']:<18} {status:<8} {rel:>9.2%} {mx:>12.6f} {sz:>8}")

    failed = [r for r in results if r.get("status") == "FAIL"]
    if failed:
        print(f"\nFAIL: {len(failed)} tensors exceeded tolerance")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
