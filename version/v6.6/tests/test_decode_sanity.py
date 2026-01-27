#!/usr/bin/env python3
"""
test_decode_sanity.py - Runtime decode path validation via stop-and-inspect.

PURPOSE:
    Tests the decode (single-token generation) path by running the model with
    CK_STOP_OP environment variable to halt after each operation, then inspecting
    the output buffers for numerical health. Catches runtime issues that static
    analysis might miss.

WHAT IT CHECKS:
    1. NaN/Inf detection - Floating point errors in kernel outputs
    2. All-zeros detection - Catches operations producing no output when they should
    3. Extreme value detection - Identifies numerical blow-ups (>1e6)
    4. KV cache consistency - Verifies KV cache slot 0 is stable after multiple
       decodes and slot 1 is properly populated (optional --kv-consistency)
    5. Logits sanity - Checks final logits are valid (optional --check-logits)

KEY FEATURES:
    - Per-op inspection: Stops after each operation to check outputs
    - Layer-focused testing: Use --layer to test specific layers
    - Full layer scan: Use --scan-all-layers to test all layers
    - KV cache validation: Checks cache write/read consistency

USAGE:
    python test_decode_sanity.py                                      # Default run
    python test_decode_sanity.py --token 9707                         # Custom token
    python test_decode_sanity.py --layer 5                            # Test layer 5
    python test_decode_sanity.py --scan-all-layers                    # Test all layers
    python test_decode_sanity.py --kv-consistency                     # Check KV cache
    python test_decode_sanity.py --check-logits                       # Check final logits
    python test_decode_sanity.py --stop-on-failure                    # Stop at first error

EXIT CODES:
    0 = PASS (all checks passed)
    1 = FAIL (issues found or initialization failed)

FILES REQUIRED (in model-dir):
    - libmodel.so       (compiled model library)
    - weights.bump      (model weights in BUMP format)
    - layout_decode.json (activation buffer layout for decode)
    - lowered_decode_call.json (IR operations list)

INTERNAL HELPERS:
    _check_tensor() - Validates tensor for NaN/Inf/zero/extreme values
    _kv_consistency_check() - Validates KV cache write behavior
"""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

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


def _find_buffer(layout: dict, name: str) -> tuple[int, int] | None:
    buffers = layout.get("memory", {}).get("activations", {}).get("buffers", [])
    for buf in buffers:
        if buf.get("name") == name:
            off = int(buf.get("abs_offset", buf.get("offset", 0)))
            size = int(buf.get("size", 0))
            return off, size
    return None


def _kv_consistency_check(lib, weights_path: Path, base_ptr_fn, layout: dict, config: dict,
                          token_a: int, token_b: int, layer: int, head: int,
                          eps: float = 1e-6) -> tuple[bool, str]:
    kv_buf = _find_buffer(layout, "kv_cache")
    if not kv_buf:
        return False, "kv_cache buffer not found"
    kv_base, kv_size = kv_buf

    num_kv_heads = config.get("num_kv_heads", 0)
    head_dim = config.get("head_dim", 0)
    max_seq_len = config.get("context_len", config.get("context_length", config.get("max_seq_len", 0)))
    num_layers = config.get("num_layers", 0)
    if not all([num_kv_heads, head_dim, max_seq_len, num_layers]):
        return False, "missing kv dims in config"

    stride = num_kv_heads * max_seq_len * head_dim
    if (layer * 2 + 1) * stride * 4 > kv_size:
        return False, "kv_cache layout too small for requested layer"

    def _slice_offset(kind: int, pos: int) -> int:
        # kind: 0=K, 1=V
        return kv_base + ((layer * 2 + kind) * stride + head * max_seq_len * head_dim + pos * head_dim) * 4

    # Reset + init
    lib.ck_model_free()
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        return False, "init failed"

    # Optional reset if available
    try:
        lib.ck_model_kv_cache_reset()
    except Exception:
        pass

    # Decode token A
    ret = lib.ck_model_decode(ctypes.c_int32(token_a), None)
    if ret != 0:
        return False, "decode token A failed"
    base_ptr = base_ptr_fn()
    if base_ptr == 0:
        return False, "base ptr NULL after token A"

    k0_off = _slice_offset(0, 0)
    v0_off = _slice_offset(1, 0)
    k0 = _read_f32(base_ptr, k0_off, head_dim)
    v0 = _read_f32(base_ptr, v0_off, head_dim)
    if np.max(np.abs(k0)) < eps or np.max(np.abs(v0)) < eps:
        return False, "kv slot 0 still zero after token A"

    # Decode token B
    ret = lib.ck_model_decode(ctypes.c_int32(token_b), None)
    if ret != 0:
        return False, "decode token B failed"
    base_ptr = base_ptr_fn()
    if base_ptr == 0:
        return False, "base ptr NULL after token B"

    k0_b = _read_f32(base_ptr, k0_off, head_dim)
    v0_b = _read_f32(base_ptr, v0_off, head_dim)
    if np.max(np.abs(k0_b - k0)) > 1e-3 or np.max(np.abs(v0_b - v0)) > 1e-3:
        return False, "kv slot 0 changed after token B (should be stable)"

    k1_off = _slice_offset(0, 1)
    v1_off = _slice_offset(1, 1)
    k1 = _read_f32(base_ptr, k1_off, head_dim)
    v1 = _read_f32(base_ptr, v1_off, head_dim)
    if np.max(np.abs(k1)) < eps or np.max(np.abs(v1)) < eps:
        return False, "kv slot 1 still zero after token B"

    return True, "kv consistency OK"


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode sanity test")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--token", type=int, default=9707)
    parser.add_argument("--max-ops", type=int, default=40, help="Limit to first N ops in decode")
    parser.add_argument("--layer", type=int, default=0, help="Layer to focus on")
    parser.add_argument("--scan-all-layers", action="store_true",
                        help="Scan all layers and stop at first failure")
    parser.add_argument("--stop-on-failure", action="store_true", default=True,
                        help="Stop at first failure (default)")
    parser.add_argument("--kv-consistency", action="store_true",
                        help="Check KV cache consistency across two decode steps")
    parser.add_argument("--kv-layer", type=int, default=0)
    parser.add_argument("--kv-head", type=int, default=0)
    parser.add_argument("--kv-token-a", type=int, default=9707)
    parser.add_argument("--kv-token-b", type=int, default=13)
    parser.add_argument("--check-logits", action="store_true",
                        help="Check logits buffer after decode")
    parser.add_argument("--max-elems", type=int, default=4096, help="Max elements to sample per buffer")
    args = parser.parse_args()

    model_dir = args.model_dir
    lib_path = model_dir / "libmodel.so"
    weights_path = model_dir / "weights.bump"
    layout_path = model_dir / "layout_decode.json"
    lowered_path = model_dir / "lowered_decode_call.json"

    for p in [lib_path, weights_path, layout_path, lowered_path]:
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    layout = _load_json(layout_path)
    ir = _load_json(lowered_path)
    ops_full = ir.get("operations", [])

    macro_map, offset_map = _build_buffer_maps(layout)

    # Load library
    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    print("=" * 72)
    print("DECODE SANITY")
    print("=" * 72)
    print(f"Model dir: {model_dir}")
    print(f"Token: {args.token}")
    config = ir.get("config", {})
    num_layers = int(config.get("num_layers", 0) or 0)

    layers = [args.layer]
    if args.scan_all_layers and num_layers:
        layers = list(range(num_layers))

    print(f"Ops checked (per layer): {args.max_ops}")

    # Optional KV consistency check
    if args.kv_consistency:
        ok, msg = _kv_consistency_check(
            lib, weights_path, lib.ck_model_get_base_ptr,
            layout, config, args.kv_token_a, args.kv_token_b,
            args.kv_layer, args.kv_head
        )
        print(f"KV check: {'PASS' if ok else 'FAIL'} ({msg})")
        if not ok and args.stop_on_failure:
            return 1

    # Optional logits check
    if args.check_logits:
        lib.ck_model_free()
        ret = lib.ck_model_init(str(weights_path).encode())
        if ret == 0:
            ret = lib.ck_model_decode(ctypes.c_int32(args.token), None)
        if ret != 0:
            print("Logits check: FAIL (decode failed)")
            if args.stop_on_failure:
                return 1
        else:
            logits_buf = _find_buffer(layout, "logits")
            if not logits_buf:
                print("Logits check: FAIL (logits buffer not found)")
                if args.stop_on_failure:
                    return 1
            else:
                base_ptr = lib.ck_model_get_base_ptr()
                off, size = logits_buf
                count = min(size // 4, args.max_elems)
                data = _read_f32(base_ptr, off, count)
                ok, msg = _check_tensor(data, True)
                print(f"Logits check: {'PASS' if ok else 'FAIL'} ({msg})")
                if not ok and args.stop_on_failure:
                    return 1
        lib.ck_model_free()

    failures = 0
    for layer in layers:
        # Filter ops for this layer but keep seq index from full list
        ops = []
        for seq_idx, op in enumerate(ops_full):
            if op.get("section") == "header" or op.get("layer") == layer:
                ops.append((seq_idx, op))
            if len(ops) >= args.max_ops:
                break

        print(f"\n-- Layer {layer} (ops: {len(ops)}) --")

        for seq_idx, op in ops:
            op_idx = op.get("idx", seq_idx)
            op_type = op.get("op", "unknown")
            outputs = _resolve_output_buffers(op, macro_map, offset_map)
            if not outputs:
                continue

            lib.ck_model_free()
            os.environ["CK_STOP_OP"] = str(seq_idx)
            ret = lib.ck_model_init(str(weights_path).encode())
            if ret != 0:
                print(f"[seq {seq_idx} op {op_idx}] init failed")
                failures += 1
                if args.stop_on_failure:
                    return 1
                continue

            ret = lib.ck_model_decode(ctypes.c_int32(args.token), None)
            if ret != 0:
                print(f"[seq {seq_idx} op {op_idx}] decode failed")
                failures += 1
                lib.ck_model_free()
                if args.stop_on_failure:
                    return 1
                continue

            base_ptr = lib.ck_model_get_base_ptr()
            if base_ptr == 0:
                print(f"[seq {seq_idx} op {op_idx}] base ptr NULL")
                failures += 1
                lib.ck_model_free()
                if args.stop_on_failure:
                    return 1
                continue

            op_failed = False
            for name, off, size in outputs:
                count = min(size // 4, args.max_elems)
                data = _read_f32(base_ptr, off, count)
                require_nonzero = op_type in NONZERO_REQUIRED
                ok, msg = _check_tensor(data, require_nonzero)
                if not ok:
                    op_failed = True
                    print(f"[seq {seq_idx:4d} op {op_idx:4d} {op_type:12s}] {name}: FAIL ({msg})")
                else:
                    print(f"[seq {seq_idx:4d} op {op_idx:4d} {op_type:12s}] {name}: OK ({msg})")

            if op_failed:
                failures += 1
                lib.ck_model_free()
                if args.stop_on_failure:
                    return 1
            else:
                lib.ck_model_free()

    print("-" * 72)
    if failures:
        print(f"FAIL: {failures} ops reported issues")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
