#!/usr/bin/env python3
"""
Trace where NaN first appears by stopping at different ops.
Uses CK_STOP_OP environment variable to stop execution early.
"""

import ctypes
import numpy as np
import os
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

# Key ops to check (from generated code comments)
# Op 0: embedding
# Op 1: rmsnorm layer 0
# Op 2-5: qkv projections
# etc.

def test_at_stop(stop_op):
    """Run with CK_STOP_OP and check for NaN in activations."""
    os.environ['CK_STOP_OP'] = str(stop_op)

    # Need to reload library for fresh state
    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    lib.ck_model_free.restype = None

    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"  Init failed: {ret}")
        return None

    vocab = lib.ck_model_get_vocab_size()
    base_ptr = lib.ck_model_get_base_ptr()

    # Run decode
    logits = (ctypes.c_float * vocab)()
    lib.ck_model_decode(9707, logits)

    # Check logits
    arr = np.ctypeslib.as_array(logits)
    nan_count = np.isnan(arr).sum()

    # Also read some activation buffers
    # A_EMBEDDED_INPUT offset from layout (approximation - check actual value)
    # We'll just check logits for now

    lib.ck_model_free()
    del os.environ['CK_STOP_OP']

    return nan_count

# Binary search to find first NaN
print("\n=== Searching for first op that produces NaN ===")

# First, test full run
print("\nTesting full decode (no stop)...")
full_nan = test_at_stop(9999)
print(f"Full run NaN count: {full_nan}")

if full_nan == 0:
    print("No NaN in full run - model is working!")
    exit(0)

# Test at various checkpoints
checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600]

print("\nTesting at checkpoints...")
last_good = 0
first_bad = None

for cp in checkpoints:
    nan_count = test_at_stop(cp)
    if nan_count is None:
        continue
    status = "NaN!" if nan_count > 0 else "OK"
    print(f"  Stop at op {cp}: {status} ({nan_count} NaN)")

    if nan_count == 0:
        last_good = cp
    elif first_bad is None:
        first_bad = cp

if first_bad:
    print(f"\n=== Result ===")
    print(f"Last good: op {last_good}")
    print(f"First bad: op {first_bad}")
    print(f"NaN appears between op {last_good} and op {first_bad}")

    # Fine-grained search
    print(f"\nFine-grained search between {last_good} and {first_bad}...")
    for op in range(last_good, first_bad + 1):
        nan_count = test_at_stop(op)
        if nan_count is None:
            continue
        status = "NaN!" if nan_count > 0 else "OK"
        print(f"  Stop at op {op}: {status}")
        if nan_count > 0 and last_good < op:
            print(f"\n*** NaN first appears at op {op} ***")
            print("Check the generated code to see what op this is:")
            print(f"  grep 'Op {op}:' ~/.cache/ck-engine-v6.6/models/*/model_v6_6.c")
            break
else:
    print("\nCould not find where NaN first appears")
