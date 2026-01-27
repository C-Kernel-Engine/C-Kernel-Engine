#!/usr/bin/env python3
"""
test_prefill_sanity.py - Runtime prefill path validation via stop-and-inspect.

PURPOSE:
    Tests the prefill (multi-token processing) path by running the model with
    CK_STOP_OP to halt after each operation and inspect output buffers. Similar
    to decode sanity test but for the batched prefill kernel path.

WHAT IT CHECKS:
    1. NaN/Inf detection - Floating point errors in prefill kernel outputs
    2. All-zeros detection - Catches operations producing no output when they should
    3. Extreme value detection - Identifies numerical blow-ups (>1e6)
    4. Layout size sanity - Fails if activation arena exceeds threshold (8GB default)

KEY FEATURES:
    - Per-op inspection: Stops after each prefill operation to check outputs
    - Multi-token support: Processes batches of tokens (default: 9707,13,42,128)
    - Crash isolation: --stop-on-crash runs each op in subprocess to catch segfaults
    - Layer scanning: --scan-all-layers tests across all layers
    - Memory guard: Fails early on excessively large activation layouts

USAGE:
    python test_prefill_sanity.py                                     # Default run
    python test_prefill_sanity.py --tokens "1,2,3,4"                  # Custom tokens
    python test_prefill_sanity.py --layer 5                           # Test layer 5
    python test_prefill_sanity.py --scan-all-layers                   # Test all layers
    python test_prefill_sanity.py --stop-on-crash                     # Subprocess crash test
    python test_prefill_sanity.py --stop-on-failure                   # Stop at first error
    python test_prefill_sanity.py --max-ops 30                        # Limit ops checked
    python test_prefill_sanity.py --allow-large                       # Allow large layouts

EXIT CODES:
    0 = PASS (all checks passed)
    1 = FAIL (issues found or initialization failed)

FILES REQUIRED (in model-dir):
    - libmodel.so        (compiled model library)
    - weights.bump       (model weights in BUMP format)
    - layout_prefill.json or layout_decode.json (activation buffer layout)
    - lowered_prefill_call.json (IR operations list for prefill)

INTERNAL HELPERS:
    _check_tensor() - Validates tensor for NaN/Inf/zero/extreme values
    _run_single_op() - Runs model up to specific op and inspects outputs
"""

import argparse
import ctypes
import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
DEFAULT_TOKENS = "9707,13,42,128"

NONZERO_REQUIRED = {
    "dense_embedding_lookup",
    "rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv_proj",
    "rope_qk",
    "attn",
    "out_proj",
    "mlp_gate_up",
    "mlp_down",
}


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _build_buffer_maps(layout: dict) -> tuple[dict, dict]:
    """Return macro->(offset,size) and offset->(name,size) for activation buffers."""
    buffers = layout.get("memory", {}).get("activations", {}).get("buffers", [])
    macro_map = {}
    offset_map = {}
    for buf in buffers:
        name = buf.get("name")
        macro = buf.get("define", f"A_{name}".upper())
        off = int(buf.get("abs_offset", buf.get("offset", 0)))
        size = int(buf.get("size", 0))
        macro_map[macro] = (off, size, name)
        offset_map[off] = (name, size)
    return macro_map, offset_map


def _resolve_output_buffers(op: dict, macro_map: dict, offset_map: dict) -> list[tuple[str, int, int]]:
    """Return list of (name, offset, size) for output args."""
    outputs = []
    for arg in op.get("args", []):
        src = arg.get("source", "")
        name = arg.get("name", "")
        if not (src.startswith("output") or src.startswith("scratch") or name in {"y", "out", "output", "dst", "q", "k", "v"}):
            continue
        expr = arg.get("expr", "")
        # Macro match
        macro = None
        for key in macro_map:
            if key in expr:
                macro = key
                break
        if macro:
            off, size, buf_name = macro_map[macro]
            outputs.append((buf_name, off, size))
            continue
        # Numeric offset match
        digits = [int(tok) for tok in expr.replace("(", " ").replace(")", " ").split() if tok.isdigit()]
        if digits:
            off = digits[0]
            if off in offset_map:
                buf_name, size = offset_map[off]
                outputs.append((buf_name, off, size))
            else:
                # Try to find a containing buffer
                for base, (buf_name, size) in offset_map.items():
                    if base <= off < base + size:
                        outputs.append((buf_name, base, size))
                        break
    return outputs


def _read_f32(base_ptr: int, offset: int, count: int) -> np.ndarray:
    ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
    arr = np.ctypeslib.as_array(ptr, shape=(count,))
    return arr.copy()


def _check_tensor(arr: np.ndarray, require_nonzero: bool) -> tuple[bool, str]:
    if arr.size == 0:
        return False, "empty"
    if not np.all(np.isfinite(arr)):
        return False, "nan_or_inf"
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 1e6:
        return False, f"blowup(max_abs={max_abs:.3e})"
    if require_nonzero and max_abs < 1e-8:
        return False, "all_zero"
    return True, f"ok(max_abs={max_abs:.3e})"


def _get_layout_and_size(model_dir: Path) -> tuple[dict, int]:
    # Prefer prefill layout, but fallback to decode if shared layout
    layout_path = model_dir / "layout_prefill.json"
    if not layout_path.exists():
        layout_path = model_dir / "layout_decode.json"
    layout = _load_json(layout_path)
    arena = layout.get("memory", {}).get("arena", {})
    total = int(arena.get("total_size", 0))
    return layout, total


def _filter_ops(ops: list, layer: int, scan_all_layers: bool, max_ops: int) -> list:
    if scan_all_layers:
        filtered = ops
    else:
        filtered = []
        for op in ops:
            if op.get("section") == "header" or op.get("layer") == layer:
                filtered.append(op)
    if max_ops and max_ops > 0:
        return filtered[:max_ops]
    return filtered


def _run_single_op(seq_idx: int, op: dict, model_dir: Path, weights_path: Path,
                   token_arr: ctypes.Array, macro_map: dict, offset_map: dict,
                   max_elems: int) -> int:
    op_idx = op.get("idx", seq_idx)
    op_type = op.get("op", "unknown")
    outputs = _resolve_output_buffers(op, macro_map, offset_map)
    if not outputs:
        return 0

    lib_path = model_dir / "libmodel.so"
    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    lib.ck_model_free()
    os.environ["CK_STOP_OP"] = str(seq_idx)
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"[seq {seq_idx}] init failed")
        return 1

    ret = lib.ck_model_embed_tokens(token_arr, len(token_arr))
    if ret != 0:
        print(f"[seq {seq_idx}] prefill failed")
        lib.ck_model_free()
        return 1

    base_ptr = lib.ck_model_get_base_ptr()
    if base_ptr == 0:
        print(f"[seq {seq_idx}] base ptr NULL")
        lib.ck_model_free()
        return 1

    op_failed = False
    for name, off, size in outputs:
        count = min(size // 4, max_elems)
        data = _read_f32(base_ptr, off, count)
        require_nonzero = op_type in NONZERO_REQUIRED
        ok, msg = _check_tensor(data, require_nonzero)
        if not ok:
            op_failed = True
            print(f"[seq {seq_idx:4d} op {op_idx:4d} {op_type:12s}] {name}: FAIL ({msg})")
        else:
            print(f"[seq {seq_idx:4d} op {op_idx:4d} {op_type:12s}] {name}: OK ({msg})")

    lib.ck_model_free()
    return 1 if op_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefill sanity test")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokens", type=str, default=DEFAULT_TOKENS,
                        help="Comma-separated token IDs for prefill")
    parser.add_argument("--max-ops", type=int, default=60, help="Limit to first N ops in prefill")
    parser.add_argument("--layer", type=int, default=0, help="Layer to focus on")
    parser.add_argument("--max-elems", type=int, default=4096, help="Max elements to sample per buffer")
    parser.add_argument("--max-bytes", type=int, default=8 * 1024**3,
                        help="Fail if layout total_size exceeds this (default 8GB)")
    parser.add_argument("--allow-large", action="store_true",
                        help="Allow large layouts (may OOM)")
    parser.add_argument("--scan-all-layers", action="store_true",
                        help="Scan all layers/ops instead of a single layer")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Stop on first op failure")
    parser.add_argument("--stop-on-crash", action="store_true",
                        help="Run each op in a subprocess and stop on first crash")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seq", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    model_dir = args.model_dir
    lib_path = model_dir / "libmodel.so"
    weights_path = model_dir / "weights.bump"
    lowered_path = model_dir / "lowered_prefill_call.json"

    for p in [lib_path, weights_path, lowered_path]:
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    layout, total_size = _get_layout_and_size(model_dir)
    if total_size and (total_size > args.max_bytes) and not args.allow_large:
        print(f"ERROR: layout total_size={total_size} exceeds max-bytes={args.max_bytes}")
        print("Hint: rebuild with smaller --context-len or pass --allow-large")
        return 1

    ir = _load_json(lowered_path)
    ops = ir.get("operations", [])

    macro_map, offset_map = _build_buffer_maps(layout)

    ops = _filter_ops(ops, args.layer, args.scan_all_layers, args.max_ops)

    # Parse tokens
    token_ids = [int(t.strip()) for t in args.tokens.split(",") if t.strip()]
    if len(token_ids) < 2:
        print("ERROR: need at least 2 tokens for prefill")
        return 1
    token_arr = (ctypes.c_int32 * len(token_ids))(*token_ids)

    if args.child:
        if args.seq < 0 or args.seq >= len(ops):
            return 0
        op = ops[args.seq]
        return _run_single_op(
            args.seq, op, model_dir, weights_path, token_arr,
            macro_map, offset_map, args.max_elems
        )

    print("=" * 72)
    print("PREFILL SANITY")
    print("=" * 72)
    print(f"Model dir: {model_dir}")
    print(f"Tokens: {token_ids}")
    print(f"Ops checked: {len(ops)}")

    failures = 0
    if args.stop_on_crash:
        for seq_idx, op in enumerate(ops):
            cmd = [
                sys.executable, __file__,
                "--model-dir", str(model_dir),
                "--tokens", args.tokens,
                "--max-ops", str(args.max_ops),
                "--layer", str(args.layer),
                "--max-elems", str(args.max_elems),
                "--max-bytes", str(args.max_bytes),
                "--child", "--seq", str(seq_idx),
            ]
            if args.scan_all_layers:
                cmd.append("--scan-all-layers")
            if args.allow_large:
                cmd.append("--allow-large")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                op_type = op.get("op", "unknown")
                op_fn = op.get("function", op.get("kernel", "unknown"))
                print(f"[CRASH/FAIL] seq {seq_idx} op={op_type} fn={op_fn} return={result.returncode}")
                failures += 1
                if args.stop_on_failure:
                    break
        print("-" * 72)
        return 1 if failures else 0

    for seq_idx, op in enumerate(ops):
        rc = _run_single_op(
            seq_idx, op, model_dir, weights_path, token_arr,
            macro_map, offset_map, args.max_elems
        )
        if rc != 0:
            failures += 1
            if args.stop_on_failure:
                break

    print("-" * 72)
    if failures:
        print(f"FAIL: {failures} ops reported issues")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
