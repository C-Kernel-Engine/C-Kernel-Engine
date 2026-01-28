#!/usr/bin/env python3
"""Test decode-only (single token, no prefill) to verify the mlp_down fix."""

import ctypes
import numpy as np
from pathlib import Path

# Find the model
cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
model_dirs = list(cache_dir.glob("*Qwen*")) + list(cache_dir.glob("*qwen*"))

if not model_dirs:
    print("Error: No Qwen model found in cache")
    exit(1)

model_dir = model_dirs[0]
lib_path = model_dir / "libmodel.so"
weights_path = model_dir / "weights.bump"

print(f"Model dir: {model_dir}")
print(f"Lib path: {lib_path}")
print(f"Weights path: {weights_path}")

if not lib_path.exists():
    print(f"Error: {lib_path} not found")
    exit(1)

# Load library
lib = ctypes.CDLL(str(lib_path))
lib.ck_model_init.argtypes = [ctypes.c_char_p]
lib.ck_model_init.restype = ctypes.c_int
lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
lib.ck_model_decode.restype = ctypes.c_int
lib.ck_model_get_vocab_size.restype = ctypes.c_int

# Initialize
print("\n--- Initializing model ---")
ret = lib.ck_model_init(str(weights_path).encode())
print(f"Init result: {ret}")

if ret != 0:
    print("Error: Model init failed")
    exit(1)

vocab = lib.ck_model_get_vocab_size()
print(f"Vocab size: {vocab}")

# Run decode
print("\n--- Running decode (token 9707 = 'Hello') ---")
logits = (ctypes.c_float * vocab)()
lib.ck_model_decode(9707, logits)

arr = np.ctypeslib.as_array(logits)
nan_count = np.isnan(arr).sum()
inf_count = np.isinf(arr).sum()

print(f"NaN count: {nan_count}/{len(arr)}")
print(f"Inf count: {inf_count}/{len(arr)}")

if nan_count == 0 and inf_count == 0:
    print(f"Logits range: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"Logits mean: {arr.mean():.4f}")
    print(f"Logits std: {arr.std():.4f}")
    top_5 = np.argsort(arr)[-5:][::-1]
    print(f"Top 5 token IDs: {top_5}")
    print(f"Top 5 logits: {arr[top_5]}")
    print("\n✓ DECODE TEST PASSED - No NaN/Inf")
else:
    print("\n✗ DECODE TEST FAILED - Contains NaN/Inf")
    # Show some non-NaN values if any
    valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
    if valid_mask.sum() > 0:
        print(f"Valid values: {valid_mask.sum()}")
        print(f"Valid range: [{arr[valid_mask].min():.4f}, {arr[valid_mask].max():.4f}]")
