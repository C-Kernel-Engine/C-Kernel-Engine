#!/usr/bin/env python3
"""
Minimal test to run v6.6 inference and compare with v6.5.
"""

import ctypes
import sys
import numpy as np
from pathlib import Path

V66_DIR = Path("/home/antshiv/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
V65_DIR = Path("/home/antshiv/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")

def load_model(model_dir):
    """Load model library."""
    lib_path = model_dir / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = model_dir / "libmodel.so"
    if not lib_path.exists():
        print(f"Error: Model library not found in {model_dir}")
        return None

    lib = ctypes.CDLL(str(lib_path))

    # Setup function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    lib.ck_model_get_context_window.argtypes = []
    lib.ck_model_get_context_window.restype = ctypes.c_int

    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    # Try decode API
    try:
        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int
        has_decode = True
    except AttributeError:
        has_decode = False

    # Initialize
    weights_path = model_dir / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"Error: Failed to init model, code {ret}")
        return None

    return lib, has_decode

def run_inference(lib, has_decode, token_id):
    """Run single token inference and return logits."""
    vocab_size = lib.ck_model_get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    if has_decode:
        out_logits = (ctypes.c_float * vocab_size)()
        ret = lib.ck_model_decode(token_id, out_logits)
        if ret != 0:
            print(f"Decode failed with code {ret}")
            return None
        return np.array(out_logits)
    else:
        print("No decode API, using forward")
        return None

def main():
    print("="*60)
    print("V6.6 vs V6.5 INFERENCE COMPARISON")
    print("="*60)

    # Test token - use a common token
    test_token = 100  # Usually a common character

    print(f"\nTest token: {test_token}")

    # Test v6.5
    print("\n=== Loading v6.5 ===")
    result65 = load_model(V65_DIR)
    if result65:
        lib65, has_decode65 = result65
        print(f"v6.5 loaded, has_decode={has_decode65}")

        logits65 = run_inference(lib65, has_decode65, test_token)
        if logits65 is not None:
            print(f"v6.5 logits: min={logits65.min():.4f}, max={logits65.max():.4f}, mean={logits65.mean():.4f}")
            print(f"v6.5 top 5 logits: {np.argsort(logits65)[-5:]}")
            print(f"v6.5 argmax: {np.argmax(logits65)}")
        lib65.ck_model_free()
    else:
        logits65 = None

    # Test v6.6
    print("\n=== Loading v6.6 ===")
    result66 = load_model(V66_DIR)
    if result66:
        lib66, has_decode66 = result66
        print(f"v6.6 loaded, has_decode={has_decode66}")

        logits66 = run_inference(lib66, has_decode66, test_token)
        if logits66 is not None:
            print(f"v6.6 logits: min={logits66.min():.4f}, max={logits66.max():.4f}, mean={logits66.mean():.4f}")
            print(f"v6.6 top 5 logits: {np.argsort(logits66)[-5:]}")
            print(f"v6.6 argmax: {np.argmax(logits66)}")
        lib66.ck_model_free()
    else:
        logits66 = None

    # Compare
    print("\n=== COMPARISON ===")
    if logits65 is not None and logits66 is not None:
        diff = np.abs(logits65 - logits66)
        print(f"Logits max diff: {diff.max():.4f}")
        print(f"Logits mean diff: {diff.mean():.4f}")

        # Check if same argmax
        if np.argmax(logits65) == np.argmax(logits66):
            print("✓ Same predicted token")
        else:
            print(f"✗ Different predictions: v6.5={np.argmax(logits65)}, v6.6={np.argmax(logits66)}")

        # Check for NaN or Inf
        if np.isnan(logits66).any():
            print("✗ v6.6 has NaN values!")
        if np.isinf(logits66).any():
            print("✗ v6.6 has Inf values!")

        # Sample some logits
        print(f"\nSample logits (indices 0-9):")
        print(f"  v6.5: {logits65[:10]}")
        print(f"  v6.6: {logits66[:10]}")

if __name__ == "__main__":
    main()
