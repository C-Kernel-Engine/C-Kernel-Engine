#!/usr/bin/env python3
"""
test_layer0_parity.py - Compare CK layer 0 outputs against llama.cpp

Tests numerical parity for layer 0 operations:
1. Embedding lookup
2. RMSNorm (attention)
3. Q/K/V projections
4. Attention output
5. Output projection + residual
6. RMSNorm (MLP)
7. MLP (gate, up, swiglu, down)
8. Final residual

Usage:
    python test_layer0_parity.py
    python test_layer0_parity.py --model-dir=/path/to/model --dump-dir=/path/to/llama_dump
"""

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

# Defaults
DEFAULT_MODEL_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
DEFAULT_DUMP_DIR = Path.home() / "Workspace/C-Kernel-Engine/llama_dump"
V66_ROOT = Path(__file__).parent.parent


def load_llama_tensor(name: str, dump_dir: Path) -> np.ndarray:
    """Load a tensor dumped from llama.cpp."""
    path = dump_dir / f"{name}.bin"
    if not path.exists():
        return None
    return np.fromfile(str(path), dtype=np.float32)


def read_f32_from_ptr(base_ptr: int, offset: int, count: int) -> np.ndarray:
    """Read float32 array from memory at base_ptr + offset."""
    ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
    arr = np.ctypeslib.as_array(ptr, shape=(count,))
    return arr.copy()


def compare(name: str, expected: np.ndarray, actual: np.ndarray, rtol: float = 0.05, atol: float = 1e-5) -> dict:
    """Compare two tensors and return results."""
    if expected is None:
        return {"name": name, "status": "SKIP", "reason": "no llama dump"}
    if actual is None:
        return {"name": name, "status": "SKIP", "reason": "no CK output"}

    expected = expected.flatten()
    actual = actual.flatten()

    n = min(len(expected), len(actual))
    expected = expected[:n]
    actual = actual[:n]

    diff = np.abs(expected - actual)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    max_val = max(float(np.max(np.abs(expected))), 1e-9)
    rel_err = max_diff / max_val

    passed = rel_err < rtol and max_diff < (atol + rtol * max_val)

    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "rel_err": rel_err,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "size": n,
        "max_val": max_val,
    }


def get_activation_offset(lowered: dict, op_name: str, layer: int, output_key: str, section: str = "body") -> int:
    """Get activation offset from lowered IR."""
    ops = lowered.get("operations", lowered.get("ops", []))

    for op in ops:
        if (op.get("op") == op_name and
            op.get("layer") == layer and
            op.get("section", "body") == section):
            outputs = op.get("outputs", {})
            if output_key in outputs:
                return int(outputs[output_key].get("activation_offset", 0))
    return None


def main():
    parser = argparse.ArgumentParser(description="Layer 0 numerical parity test")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--token", type=int, default=9707, help="Token ID to test")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir
    dump_dir = args.dump_dir

    print("=" * 70)
    print("LAYER 0 NUMERICAL PARITY TEST")
    print("=" * 70)
    print(f"Model dir: {model_dir}")
    print(f"Dump dir:  {dump_dir}")
    print(f"Token ID:  {args.token}")
    print()

    # Check files exist
    lib_path = model_dir / "libmodel.so"
    weights_path = model_dir / "weights.bump"
    lowered_path = model_dir / "lowered_decode.json"
    layout_path = model_dir / "layout_decode.json"

    if not lib_path.exists():
        print(f"ERROR: {lib_path} not found")
        return 1
    if not weights_path.exists():
        print(f"ERROR: {weights_path} not found")
        return 1
    if not lowered_path.exists():
        print(f"ERROR: {lowered_path} not found")
        return 1
    if not layout_path.exists():
        print(f"ERROR: {layout_path} not found")
        return 1
    if not dump_dir.exists():
        print(f"ERROR: {dump_dir} not found")
        return 1

    # Load lowered IR for offsets
    with open(lowered_path) as f:
        lowered = json.load(f)

    # Load layout to get activations base offset
    with open(layout_path) as f:
        layout = json.load(f)
    memory = layout.get("memory", {})
    acts = memory.get("activations", {})
    bufs = acts.get("buffers", [])
    act_base = bufs[0].get("abs_offset", 0) if bufs else 0
    print(f"Activations base offset: {act_base}")

    config = lowered.get("config", {})
    embed_dim = config.get("embed_dim", 896)
    num_heads = config.get("num_heads", 14)
    num_kv_heads = config.get("num_kv_heads", 2)
    head_dim = config.get("head_dim", 64)
    intermediate_size = config.get("intermediate_size", 4864)

    # Load the model
    print("Loading model...")
    lib = ctypes.CDLL(str(lib_path))

    # Define function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    # Initialize model
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"ERROR: ck_model_init failed with code {ret}")
        return 1

    # Run single token decode
    print(f"Running decode with token {args.token}...")
    logits = (ctypes.c_float * config.get("vocab_size", 151936))()
    ret = lib.ck_model_decode(ctypes.c_int32(args.token), logits)
    if ret != 0:
        print(f"ERROR: ck_model_decode failed with code {ret}")
        lib.ck_model_free()
        return 1

    # Get base pointer
    base_ptr = lib.ck_model_get_base_ptr()
    if base_ptr == 0:
        print("ERROR: ck_model_get_base_ptr returned NULL")
        lib.ck_model_free()
        return 1

    print(f"Base pointer: 0x{base_ptr:x}")
    print()

    # Define test cases: (llama_name, ck_op_name, output_key, size, section, layer)
    # Map llama dump names to CK op names and output keys
    # Note: layer=-1 for header ops (embedding), layer=0 for first transformer layer
    test_cases = [
        ("inp_embd", "dense_embedding_lookup", "output", embed_dim, "header", -1),
        ("attn_norm-0", "rmsnorm", "output", embed_dim, "body", 0),
        ("Qcur-0", "q_proj", "y", num_heads * head_dim, "body", 0),
        ("Kcur-0", "k_proj", "y", num_kv_heads * head_dim, "body", 0),
        ("Vcur-0", "v_proj", "y", num_kv_heads * head_dim, "body", 0),
        ("kqv_out-0", "out_proj", "y", embed_dim, "body", 0),
        ("ffn_inp-0", "residual_add", "out", embed_dim, "body", 0),  # First residual
        ("ffn_norm-0", "rmsnorm", "output", embed_dim, "body", 0),  # Second rmsnorm
        ("ffn_out-0", "mlp_down", "y", embed_dim, "body", 0),
        ("l_out-0", "residual_add", "out", embed_dim, "body", 0),  # Second residual
    ]

    results = []

    # Track which rmsnorm/residual we're looking at
    rmsnorm_count = 0
    residual_count = 0

    for llama_name, ck_op, out_key, size, section, layer in test_cases:
        # Load llama reference
        llama_tensor = load_llama_tensor(llama_name, dump_dir)

        # For ops that appear multiple times, find the right one
        ops = lowered.get("operations", lowered.get("ops", []))
        offset = None

        if ck_op == "rmsnorm":
            # Find the nth rmsnorm in the specified layer
            count = 0
            for op in ops:
                if op.get("op") == ck_op and op.get("layer") == layer and op.get("section") == section:
                    if count == rmsnorm_count:
                        outputs = op.get("outputs", {})
                        if out_key in outputs:
                            offset = int(outputs[out_key].get("activation_offset", 0))
                        break
                    count += 1
            if "ffn_norm" in llama_name:
                rmsnorm_count += 1

        elif ck_op == "residual_add":
            # Find the nth residual_add in the specified layer
            count = 0
            for op in ops:
                if op.get("op") == ck_op and op.get("layer") == layer:
                    if count == residual_count:
                        outputs = op.get("outputs", {})
                        if out_key in outputs:
                            offset = int(outputs[out_key].get("activation_offset", 0))
                        break
                    count += 1
            if "l_out" in llama_name:
                residual_count += 1
        else:
            offset = get_activation_offset(lowered, ck_op, layer, out_key, section)

        # Read CK output
        # Note: activation_offset from IR is relative to activations section
        # Need to add act_base to get absolute offset from bump base
        ck_tensor = None
        if offset is not None:
            abs_offset = offset + act_base
            try:
                ck_tensor = read_f32_from_ptr(base_ptr, abs_offset, size)
            except Exception as e:
                if args.verbose:
                    print(f"  Error reading {llama_name}: {e}")

        # Compare
        result = compare(llama_name, llama_tensor, ck_tensor)
        results.append(result)

        if args.verbose:
            print(f"  {llama_name}: rel_offset={offset}, abs_offset={offset + act_base if offset else None}, size={size}")

    # Print results
    print(f"{'Tensor':<18} {'Status':<6} {'RelErr':>10} {'MaxDiff':>12} {'Size':>8}")
    print("-" * 60)

    for r in results:
        status = r["status"]
        if status == "SKIP":
            print(f"{r['name']:<18} {status:<6} {'N/A':>10} {'N/A':>12} {'N/A':>8}  ({r.get('reason', '')})")
        else:
            rel = r.get("rel_err", 0)
            mx = r.get("max_diff", 0)
            sz = r.get("size", 0)
            print(f"{r['name']:<18} {status:<6} {rel:>9.2%} {mx:>12.6f} {sz:>8}")

    # Summary
    print()
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    skipped = [r for r in results if r["status"] == "SKIP"]

    print("=" * 70)
    print(f"PASSED: {len(passed)}, FAILED: {len(failed)}, SKIPPED: {len(skipped)}")

    if failed:
        print("\nFailed tensors:")
        for r in failed:
            print(f"  - {r['name']}: rel_err={r['rel_err']:.2%}, max_diff={r['max_diff']:.6f}")

    # Cleanup
    lib.ck_model_free()

    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
