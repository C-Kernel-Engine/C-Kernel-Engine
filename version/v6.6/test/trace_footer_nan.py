#!/usr/bin/env python3
"""
Trace NaN in footer ops (600-630) to find exact op that produces NaN.
"""

import ctypes
import numpy as np
import os
from pathlib import Path

cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
model_dirs = list(cache_dir.glob("*Qwen*")) + list(cache_dir.glob("*qwen*"))

if not model_dirs:
    print("Error: No Qwen model found in cache")
    exit(1)

model_dir = model_dirs[0]
lib_path = model_dir / "libmodel.so"
weights_path = model_dir / "weights.bump"

print(f"Model dir: {model_dir}")

def test_at_stop(stop_op):
    """Run with CK_STOP_OP and check for NaN in logits."""
    os.environ['CK_STOP_OP'] = str(stop_op)

    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_free.restype = None

    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        del os.environ['CK_STOP_OP']
        return None

    vocab = lib.ck_model_get_vocab_size()
    logits = (ctypes.c_float * vocab)()
    lib.ck_model_decode(9707, logits)

    arr = np.ctypeslib.as_array(logits)
    nan_count = np.isnan(arr).sum()

    lib.ck_model_free()
    del os.environ['CK_STOP_OP']

    return nan_count

print("\n=== Tracing footer ops (600-630) ===")
print("Looking for first op that produces NaN...\n")

last_good = 600
first_bad = None

for op in range(600, 640):
    nan_count = test_at_stop(op)
    if nan_count is None:
        print(f"  Op {op}: Init failed")
        continue

    status = "NaN!" if nan_count > 0 else "OK"
    print(f"  Op {op}: {status} ({nan_count} NaN)")

    if nan_count == 0:
        last_good = op
    elif first_bad is None:
        first_bad = op
        print(f"\n*** NaN first appears at stop_op={op} ***")
        print(f"This means op {op-1} or op {op} introduces NaN")
        print(f"\nCheck the generated code:")
        print(f"  grep -B2 -A10 'Op {op-1}:' ~/.cache/ck-engine-v6.6/models/*/model_v6_6.c")
        print(f"  grep -B2 -A10 'Op {op}:' ~/.cache/ck-engine-v6.6/models/*/model_v6_6.c")
        break

if first_bad is None:
    print(f"\nNo NaN found up to op 639. Checking higher ops...")
    for op in [650, 700, 800, 1000]:
        nan_count = test_at_stop(op)
        if nan_count is None:
            continue
        status = "NaN!" if nan_count > 0 else "OK"
        print(f"  Op {op}: {status} ({nan_count} NaN)")
        if nan_count > 0:
            print(f"\nNaN appears between op {last_good} and op {op}")
            break
        last_good = op
