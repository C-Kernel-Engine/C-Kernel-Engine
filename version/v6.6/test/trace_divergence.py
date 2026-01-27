#!/usr/bin/env python3
"""
Trace divergence between v6.5 and v6.6 by running each version independently
and comparing activations at each stage.
"""

import ctypes
import numpy as np
import os
import subprocess
import sys
from pathlib import Path

V66_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V65_DIR = Path.home() / ".cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

EMBED_DIM = 896
VOCAB_SIZE = 151936

# V6.6 activation offsets (from layout_decode.json)
V66_A_EMBEDDED_INPUT = 396942152
V66_A_LAYER_INPUT = 400612168
V66_A_RESIDUAL = 404282184
V66_A_Q_SCRATCH = 433380168
V66_A_K_SCRATCH = 437050184
V66_A_V_SCRATCH = 437574472
V66_A_ATTN_SCRATCH = 438098760


def run_v66_and_dump(token: int, stop_seq: int = -1):
    """Run v6.6 and return activations."""
    # Load in fresh process context
    engine_path = V66_DIR / "libckernel_engine.so"
    if engine_path.exists():
        ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

    lib_path = V66_DIR / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = V66_DIR / "libmodel.so"

    lib = ctypes.CDLL(str(lib_path))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64

    weights_path = V66_DIR / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        raise RuntimeError(f"v6.6 init failed: {ret}")

    base_ptr = lib.ck_model_get_base_ptr()
    vocab_size = lib.ck_model_get_vocab_size()

    # Set stop if requested
    if stop_seq >= 0:
        os.environ["CK_STOP_OP"] = str(stop_seq)

    output = (ctypes.c_float * vocab_size)()
    ret = lib.ck_model_decode(token, output)

    if stop_seq >= 0:
        del os.environ["CK_STOP_OP"]

    # Read activations
    def read_act(offset, size):
        ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(ptr, shape=(size,)).copy()

    result = {
        'logits': np.array(output[:], dtype=np.float32),
        'embedding': read_act(V66_A_EMBEDDED_INPUT, EMBED_DIM),
        'layer_input': read_act(V66_A_LAYER_INPUT, EMBED_DIM),
        'residual': read_act(V66_A_RESIDUAL, EMBED_DIM),
        'q_scratch': read_act(V66_A_Q_SCRATCH, EMBED_DIM),
        'k_scratch': read_act(V66_A_K_SCRATCH, 128),
        'v_scratch': read_act(V66_A_V_SCRATCH, 128),
        'attn_scratch': read_act(V66_A_ATTN_SCRATCH, EMBED_DIM),
    }

    lib.ck_model_free()
    return result


def run_v65_and_dump(token: int, stop_seq: int = -1):
    """Run v6.5 and return activations."""
    lib_path = V65_DIR / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = V65_DIR / "libmodel.so"

    lib = ctypes.CDLL(str(lib_path))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    # V6.5 may not have base_ptr
    try:
        lib.ck_model_get_base_ptr.argtypes = []
        lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
        has_base_ptr = True
    except:
        has_base_ptr = False

    weights_path = V65_DIR / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        raise RuntimeError(f"v6.5 init failed: {ret}")

    vocab_size = lib.ck_model_get_vocab_size()

    if stop_seq >= 0:
        os.environ["CK_STOP_OP"] = str(stop_seq)

    output = (ctypes.c_float * vocab_size)()
    ret = lib.ck_model_decode(token, output)

    if stop_seq >= 0:
        del os.environ["CK_STOP_OP"]

    result = {
        'logits': np.array(output[:], dtype=np.float32),
    }

    # Try to read activations if we have base_ptr
    if has_base_ptr:
        base_ptr = lib.ck_model_get_base_ptr()
        # V6.5 has different offsets - would need to load from its manifest
        # For now, just return logits
        pass

    lib.ck_model_free()
    return result


def compare_stats(name: str, v65_arr, v66_arr):
    """Compare array statistics."""
    if v65_arr is None or v66_arr is None:
        return

    if len(v65_arr) != len(v66_arr):
        print(f"{name}: SHAPE MISMATCH v65={len(v65_arr)}, v66={len(v66_arr)}")
        return

    diff = np.abs(v65_arr - v66_arr)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    max_idx = np.argmax(diff)

    same = max_diff < 0.001

    print(f"{name}:")
    print(f"  v6.5: min={v65_arr.min():10.4f}, max={v65_arr.max():10.4f}, mean={v65_arr.mean():10.4f}")
    print(f"  v6.6: min={v66_arr.min():10.4f}, max={v66_arr.max():10.4f}, mean={v66_arr.mean():10.4f}")
    print(f"  diff: max={max_diff:10.4f}, mean={mean_diff:10.4f}, max_at={max_idx} {'OK' if same else 'DIFF!'}")


def main():
    token = 100

    print("=" * 70)
    print(f"TRACING DIVERGENCE: v6.5 vs v6.6 for token {token}")
    print("=" * 70)

    # Run full decode for both
    print("\n--- FULL DECODE ---")

    print("\nRunning v6.5...")
    v65 = run_v65_and_dump(token)
    print(f"  Logits: min={v65['logits'].min():.4f}, max={v65['logits'].max():.4f}")
    print(f"  Argmax: {np.argmax(v65['logits'])}")

    print("\nRunning v6.6...")
    v66 = run_v66_and_dump(token)
    print(f"  Logits: min={v66['logits'].min():.4f}, max={v66['logits'].max():.4f}")
    print(f"  Argmax: {np.argmax(v66['logits'])}")

    # Compare logits
    print("\n--- LOGITS COMPARISON ---")
    compare_stats("logits", v65['logits'], v66['logits'])

    # Print first 20 logit differences
    print("\nFirst 20 logit values:")
    print("Index |     v6.5 |     v6.6 |     Diff")
    print("-" * 45)
    for i in range(20):
        diff = abs(v65['logits'][i] - v66['logits'][i])
        marker = "***" if diff > 0.1 else ""
        print(f"{i:5d} | {v65['logits'][i]:8.4f} | {v66['logits'][i]:8.4f} | {diff:8.4f} {marker}")

    # Show top-5 tokens
    print("\nTop-5 predictions:")
    top5_65 = np.argsort(v65['logits'])[-5:][::-1]
    top5_66 = np.argsort(v66['logits'])[-5:][::-1]
    vals65 = [f'{v65["logits"][i]:.2f}' for i in top5_65]
    vals66 = [f'{v66["logits"][i]:.2f}' for i in top5_66]
    print(f"  v6.5: {list(top5_65)} values={vals65}")
    print(f"  v6.6: {list(top5_66)} values={vals66}")

    # Show v6.6 activations
    print("\n--- V6.6 ACTIVATION VALUES ---")
    for name in ['embedding', 'layer_input', 'residual', 'q_scratch', 'k_scratch', 'v_scratch', 'attn_scratch']:
        arr = v66.get(name)
        if arr is not None:
            print(f"{name:15s}: min={arr.min():10.4f}, max={arr.max():10.4f}, mean={arr.mean():10.4f}")
            print(f"                 first 5: {arr[:5]}")

    # Now let's run step-by-step with stop_seq to see when they diverge
    print("\n\n" + "=" * 70)
    print("STEP-BY-STEP STOP_SEQ TEST")
    print("=" * 70)

    for stop in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
        print(f"\n--- Stop at op {stop} ---")

        # Need to reload models for clean state
        # We'll use subprocess to isolate
        v66_stop = run_v66_and_dump(token, stop)

        print(f"  v6.6 embedding: min={v66_stop['embedding'].min():.4f}, max={v66_stop['embedding'].max():.4f}")
        print(f"  v6.6 layer_in:  min={v66_stop['layer_input'].min():.4f}, max={v66_stop['layer_input'].max():.4f}")
        print(f"  v6.6 q_scratch: min={v66_stop['q_scratch'].min():.4f}, max={v66_stop['q_scratch'].max():.4f}")


if __name__ == "__main__":
    main()
