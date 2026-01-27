#!/usr/bin/env python3
"""
debug_stop_seq_v65_v66.py - Use stop_seq to find divergence between v6.5 and v6.6

This script uses the CK_STOP_OP environment variable to stop execution at each
operation and compare the activation state between v6.5 and v6.6.

Usage:
    python debug_stop_seq_v65_v66.py --token 100
    python debug_stop_seq_v65_v66.py --token 100 --stop-at 5 --verbose
    python debug_stop_seq_v65_v66.py --token 100 --layer0-only
"""

import argparse
import ctypes
import json
import numpy as np
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

V65_DIR = Path.home() / ".cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V66_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

# Model dimensions
EMBED_DIM = 896
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
VOCAB_SIZE = 151936

# V6.6 activation offsets (from layout_decode.json)
V66_OFFSETS = {
    'A_EMBEDDED_INPUT': 396942152,
    'A_LAYER_INPUT': 400612168,
    'A_RESIDUAL': 404282184,
    'A_Q_SCRATCH': 433380168,
    'A_K_SCRATCH': 437050184,
    'A_V_SCRATCH': 437574472,
    'A_ATTN_SCRATCH': 438098760,
    'A_LAYER_OUTPUT': 481614664,
    'A_LOGITS': 485284680,
}

# Operation sequence for layer 0 (from generated code)
# Each entry: (stop_seq, op_name, activation_buffer, size)
LAYER0_OPS = [
    (0, 'embedding', 'A_EMBEDDED_INPUT', EMBED_DIM),
    (1, 'rmsnorm', 'A_LAYER_INPUT', EMBED_DIM),
    (2, 'residual_save', 'A_RESIDUAL', EMBED_DIM),
    (3, 'q_proj', 'A_Q_SCRATCH', NUM_HEADS * HEAD_DIM),
    (4, 'q_bias', 'A_Q_SCRATCH', NUM_HEADS * HEAD_DIM),
    (5, 'k_proj', 'A_K_SCRATCH', NUM_KV_HEADS * HEAD_DIM),
    (6, 'k_bias', 'A_K_SCRATCH', NUM_KV_HEADS * HEAD_DIM),
    (7, 'v_proj', 'A_V_SCRATCH', NUM_KV_HEADS * HEAD_DIM),
    (8, 'v_bias', 'A_V_SCRATCH', NUM_KV_HEADS * HEAD_DIM),
    (9, 'rope_q', 'A_Q_SCRATCH', NUM_HEADS * HEAD_DIM),
    (10, 'rope_k', 'A_K_SCRATCH', NUM_KV_HEADS * HEAD_DIM),
    (11, 'kv_store', None, 0),  # KV cache store, no simple activation
    (12, 'attention', 'A_ATTN_SCRATCH', NUM_HEADS * HEAD_DIM),
    (13, 'attn_proj', 'A_ATTN_SCRATCH', EMBED_DIM),
    (14, 'attn_bias', 'A_ATTN_SCRATCH', EMBED_DIM),
    (15, 'residual_add', 'A_LAYER_OUTPUT', EMBED_DIM),
]


# ============================================================================
# MODEL LOADER
# ============================================================================

@dataclass
class ModelHandle:
    """Handle to a loaded model."""
    lib: ctypes.CDLL
    base_ptr: int
    version: str
    model_dir: Path


def load_model(model_dir: Path, version: str) -> Optional[ModelHandle]:
    """Load model and return handle."""
    # Find library
    lib_path = None
    for name in ["ck-kernel-inference.so", "libmodel.so"]:
        p = model_dir / name
        if p.exists():
            lib_path = p
            break

    if not lib_path:
        print(f"ERROR: No .so found in {model_dir}")
        return None

    # Load engine library first if exists
    engine_path = model_dir / "libckernel_engine.so"
    if engine_path.exists():
        ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

    lib = ctypes.CDLL(str(lib_path))

    # Bind common functions
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    # Bind decode
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int

    # Try to get base pointer function
    try:
        lib.ck_model_get_base_ptr.argtypes = []
        lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
        has_base_ptr = True
    except AttributeError:
        has_base_ptr = False

    # Initialize
    weights_path = model_dir / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"ERROR: ck_model_init failed for {version}: {ret}")
        return None

    # Get base pointer
    if has_base_ptr:
        base_ptr = lib.ck_model_get_base_ptr()
    else:
        base_ptr = 0

    return ModelHandle(lib=lib, base_ptr=base_ptr, version=version, model_dir=model_dir)


def read_activation(handle: ModelHandle, offset: int, size: int) -> np.ndarray:
    """Read activation from memory."""
    if handle.base_ptr == 0:
        return np.zeros(size, dtype=np.float32)

    ptr = ctypes.cast(handle.base_ptr + offset, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(size,)).copy()


def run_decode_with_stop(handle: ModelHandle, token: int, stop_op: int) -> Tuple[int, np.ndarray]:
    """Run decode and stop at specific operation."""
    os.environ["CK_STOP_OP"] = str(stop_op)

    vocab_size = handle.lib.ck_model_get_vocab_size()
    output = (ctypes.c_float * vocab_size)()

    ret = handle.lib.ck_model_decode(ctypes.c_int32(token), output)

    del os.environ["CK_STOP_OP"]

    return ret, np.array(output[:], dtype=np.float32)


# ============================================================================
# COMPARISON
# ============================================================================

def compare_arrays(name: str, v65: np.ndarray, v66: np.ndarray, tol: float = 1e-4) -> Tuple[bool, str]:
    """Compare two arrays and return (passed, message)."""
    if v65.shape != v66.shape:
        return False, f"Shape mismatch: v65={v65.shape}, v66={v66.shape}"

    abs_diff = np.abs(v65 - v66)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    if max_diff < tol:
        return True, f"OK (max_diff={max_diff:.2e})"

    # Find where max diff is
    max_idx = np.argmax(abs_diff)

    return False, f"DIFF at [{max_idx}]: v65={v65[max_idx]:.6f}, v66={v66[max_idx]:.6f}, max_diff={max_diff:.4f}"


def print_array_stats(name: str, arr: np.ndarray, prefix: str = ""):
    """Print array statistics."""
    if arr is None or len(arr) == 0:
        print(f"{prefix}{name}: <empty>")
        return

    print(f"{prefix}{name}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}")
    print(f"{prefix}  first 5: {arr[:5]}")


# ============================================================================
# V6.5 ACTIVATION READER
# ============================================================================

def get_v65_activation_offset(handle: ModelHandle, buffer_name: str) -> Optional[int]:
    """Get activation offset for v6.5.

    V6.5 has different layout - need to read from its manifest.
    """
    # Try to load v6.5 manifest
    manifest_path = handle.model_dir / "weights_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        # V6.5 uses runtime_offset in entries
        entries = manifest.get('entries', [])
        for entry in entries:
            if entry.get('define') == buffer_name:
                return entry.get('runtime_offset')
    except Exception:
        pass

    return None


# ============================================================================
# MAIN COMPARISON LOOP
# ============================================================================

def run_comparison(token: int, stop_at: Optional[int] = None,
                   layer0_only: bool = False, verbose: bool = False):
    """Run comparison between v6.5 and v6.6."""

    print("=" * 70)
    print(f"STOP_SEQ COMPARISON: v6.5 vs v6.6")
    print(f"Token: {token}")
    print("=" * 70)

    # Load both models
    print("\nLoading v6.5...")
    h65 = load_model(V65_DIR, "v6.5")
    if not h65:
        print("Failed to load v6.5")
        return 1

    print("Loading v6.6...")
    h66 = load_model(V66_DIR, "v6.6")
    if not h66:
        print("Failed to load v6.6")
        h65.lib.ck_model_free()
        return 1

    print(f"\nv6.5 base_ptr: 0x{h65.base_ptr:x}")
    print(f"v6.6 base_ptr: 0x{h66.base_ptr:x}")

    # Determine which ops to test
    if layer0_only:
        ops = LAYER0_OPS
        max_stop = 15
    else:
        max_stop = stop_at if stop_at else 15  # First layer for now
        ops = [(i, f"op_{i}", None, 0) for i in range(max_stop + 1)]

    if stop_at is not None:
        ops = [(s, n, b, sz) for s, n, b, sz in ops if s <= stop_at]

    # Run comparison at each stop point
    results = []
    first_failure = None

    print("\n" + "=" * 70)
    print("OPERATION-BY-OPERATION COMPARISON")
    print("=" * 70)

    for stop_seq, op_name, buffer_name, size in ops:
        print(f"\n--- Stop {stop_seq}: {op_name} ---")

        # Run v6.5
        ret65, logits65 = run_decode_with_stop(h65, token, stop_seq)

        # Run v6.6
        ret66, logits66 = run_decode_with_stop(h66, token, stop_seq)

        if ret65 != 0 or ret66 != 0:
            print(f"  Decode returned: v6.5={ret65}, v6.6={ret66}")

        # Compare logits at this stop point
        passed, msg = compare_arrays(f"logits@{stop_seq}", logits65, logits66, tol=0.01)

        if passed:
            print(f"  [PASS] Logits: {msg}")
        else:
            print(f"  [FAIL] Logits: {msg}")
            if first_failure is None:
                first_failure = (stop_seq, op_name, "logits")

        # If we have a buffer to check, read and compare activations
        if buffer_name and h66.base_ptr and size > 0:
            offset = V66_OFFSETS.get(buffer_name)
            if offset:
                v66_act = read_activation(h66, offset, size)

                if verbose:
                    print_array_stats(f"  v6.6 {buffer_name}", v66_act, "  ")

                # Try to read v6.5 activation at same logical offset
                # Note: v6.5 has different layout, this may not work directly
                if h65.base_ptr:
                    v65_offset = get_v65_activation_offset(h65, buffer_name)
                    if v65_offset:
                        v65_act = read_activation(h65, v65_offset, size)

                        if verbose:
                            print_array_stats(f"  v6.5 {buffer_name}", v65_act, "  ")

                        act_passed, act_msg = compare_arrays(buffer_name, v65_act, v66_act, tol=0.01)
                        if act_passed:
                            print(f"  [PASS] {buffer_name}: {act_msg}")
                        else:
                            print(f"  [FAIL] {buffer_name}: {act_msg}")
                            if first_failure is None:
                                first_failure = (stop_seq, op_name, buffer_name)

        results.append((stop_seq, op_name, passed))

        # Stop early if we found a failure and not in verbose mode
        if first_failure and not verbose:
            print(f"\n  Stopping at first failure.")
            break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, _, p in results if p)
    failed_count = len(results) - passed_count

    print(f"Passed: {passed_count}/{len(results)}")
    print(f"Failed: {failed_count}/{len(results)}")

    if first_failure:
        stop_seq, op_name, buffer = first_failure
        print(f"\nFIRST DIVERGENCE at stop_seq={stop_seq} ({op_name})")
        print(f"Buffer: {buffer}")
        print(f"\nThis means the bug is in the operation that runs BEFORE stop_seq={stop_seq}")

        if stop_seq > 0:
            prev_op = ops[stop_seq - 1] if stop_seq <= len(ops) else None
            if prev_op:
                print(f"Check operation: stop_seq={prev_op[0]} ({prev_op[1]})")
    else:
        print("\nAll operations matched!")

    # Cleanup
    h65.lib.ck_model_free()
    h66.lib.ck_model_free()

    return 0 if first_failure is None else 1


# ============================================================================
# ALTERNATIVE: Compare full runs only (no activation reading)
# ============================================================================

def run_simple_comparison(token: int, max_stop: int = 20, verbose: bool = False):
    """Simple comparison: just compare logits at each stop point."""

    print("=" * 70)
    print(f"SIMPLE STOP_SEQ COMPARISON: v6.5 vs v6.6")
    print(f"Token: {token}, Max stop: {max_stop}")
    print("=" * 70)

    # Load models
    print("\nLoading v6.5...")
    h65 = load_model(V65_DIR, "v6.5")
    if not h65:
        return 1

    print("Loading v6.6...")
    h66 = load_model(V66_DIR, "v6.6")
    if not h66:
        h65.lib.ck_model_free()
        return 1

    # Run at each stop point
    print("\n" + "-" * 70)
    print(f"{'Stop':>5} | {'v6.5 argmax':>12} | {'v6.6 argmax':>12} | {'Max Diff':>12} | Status")
    print("-" * 70)

    first_divergence = None

    for stop_seq in range(max_stop + 1):
        # Run v6.5
        os.environ["CK_STOP_OP"] = str(stop_seq)
        vocab_size = h65.lib.ck_model_get_vocab_size()
        out65 = (ctypes.c_float * vocab_size)()
        h65.lib.ck_model_decode(ctypes.c_int32(token), out65)
        logits65 = np.array(out65[:], dtype=np.float32)

        # Run v6.6
        out66 = (ctypes.c_float * vocab_size)()
        h66.lib.ck_model_decode(ctypes.c_int32(token), out66)
        logits66 = np.array(out66[:], dtype=np.float32)

        del os.environ["CK_STOP_OP"]

        # Compare
        max_diff = np.max(np.abs(logits65 - logits66))
        argmax65 = np.argmax(logits65)
        argmax66 = np.argmax(logits66)

        if max_diff < 0.01:
            status = "PASS"
        else:
            status = "FAIL"
            if first_divergence is None:
                first_divergence = stop_seq

        print(f"{stop_seq:>5} | {argmax65:>12} | {argmax66:>12} | {max_diff:>12.4f} | {status}")

        if verbose:
            print(f"       v6.5: min={logits65.min():.2f}, max={logits65.max():.2f}")
            print(f"       v6.6: min={logits66.min():.2f}, max={logits66.max():.2f}")

        # Stop at first failure if not verbose
        if first_divergence and not verbose:
            break

    print("-" * 70)

    if first_divergence is not None:
        print(f"\nFIRST DIVERGENCE at stop_seq={first_divergence}")
        print(f"The bug is in the operation at or just before stop_seq={first_divergence}")
    else:
        print(f"\nAll {max_stop + 1} stop points matched!")

    # Cleanup
    h65.lib.ck_model_free()
    h66.lib.ck_model_free()

    return 0 if first_divergence is None else 1


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Debug v6.5 vs v6.6 using stop_seq")
    parser.add_argument("--token", type=int, default=100, help="Token ID to test")
    parser.add_argument("--stop-at", type=int, help="Stop at specific operation")
    parser.add_argument("--max-stop", type=int, default=20, help="Maximum stop sequence")
    parser.add_argument("--layer0-only", action="store_true", help="Only test layer 0")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--simple", action="store_true", help="Simple comparison (logits only)")
    args = parser.parse_args()

    # Set library path
    os.environ["LD_LIBRARY_PATH"] = f"{V66_DIR}:{V65_DIR}:" + os.environ.get("LD_LIBRARY_PATH", "")

    if args.simple:
        return run_simple_comparison(args.token, args.max_stop, args.verbose)
    else:
        return run_comparison(args.token, args.stop_at, args.layer0_only, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
