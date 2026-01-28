#!/usr/bin/env python3
"""
Binary search to find the first op that produces NaN in A_EMBEDDED_INPUT.
"""

import ctypes
import numpy as np
import os
from pathlib import Path

cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
model_dirs = list(cache_dir.glob("*Qwen*")) + list(cache_dir.glob("*qwen*"))

model_dir = model_dirs[0]
lib_path = model_dir / "libmodel.so"
weights_path = model_dir / "weights.bump"

import json
layout_path = model_dir / "layout_decode.json"
with open(layout_path) as f:
    layout = json.load(f)

embed_dim = layout.get("config", {}).get("embed_dim", 896)

act_buffers = {}
for buf in layout.get("memory", {}).get("activations", {}).get("buffers", []):
    act_buffers[buf["name"]] = buf["abs_offset"]

def check_nan_at_stop(stop_op):
    """Check if A_EMBEDDED_INPUT has NaN at given stop point."""
    os.environ['CK_STOP_OP'] = str(stop_op)

    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_free.restype = None

    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        del os.environ['CK_STOP_OP']
        return None

    base_ptr = lib.ck_model_get_base_ptr()
    vocab = lib.ck_model_get_vocab_size()
    logits = (ctypes.c_float * vocab)()
    lib.ck_model_decode(9707, logits)

    del os.environ['CK_STOP_OP']

    # Check A_EMBEDDED_INPUT
    emb_offset = act_buffers["embedded_input"]
    ptr = ctypes.cast(base_ptr + emb_offset, ctypes.POINTER(ctypes.c_float))
    arr = np.ctypeslib.as_array(ptr, shape=(embed_dim,)).copy()

    # Also check A_LAYER_INPUT
    li_offset = act_buffers["layer_input"]
    ptr2 = ctypes.cast(base_ptr + li_offset, ctypes.POINTER(ctypes.c_float))
    arr2 = np.ctypeslib.as_array(ptr2, shape=(embed_dim,)).copy()

    lib.ck_model_free()

    return {
        'embedded_nan': np.isnan(arr).sum(),
        'layer_nan': np.isnan(arr2).sum(),
        'embedded_arr': arr,
        'layer_arr': arr2
    }

print("Tracing where NaN first appears...")
print("Checking A_EMBEDDED_INPUT and A_LAYER_INPUT at various ops\n")

# First, check key checkpoints to narrow down
checkpoints = [1, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 624]

last_good = 0
first_bad = None

for cp in checkpoints:
    result = check_nan_at_stop(cp)
    if result is None:
        continue

    emb_nan = result['embedded_nan']
    li_nan = result['layer_nan']
    status = "OK" if emb_nan == 0 else f"NaN({emb_nan})"

    print(f"Op {cp:3d}: A_EMBEDDED_INPUT={status:12s}  A_LAYER_INPUT={'OK' if li_nan == 0 else f'NaN({li_nan})'}")

    if emb_nan == 0:
        last_good = cp
    elif first_bad is None:
        first_bad = cp

if first_bad:
    print(f"\n--- Fine-grained search between op {last_good} and op {first_bad} ---\n")

    for op in range(last_good, first_bad + 1):
        result = check_nan_at_stop(op)
        if result is None:
            continue

        emb_nan = result['embedded_nan']
        li_nan = result['layer_nan']
        status = "OK" if emb_nan == 0 else f"NaN({emb_nan})"

        print(f"Op {op:3d}: A_EMBEDDED_INPUT={status:12s}  A_LAYER_INPUT={'OK' if li_nan == 0 else f'NaN({li_nan})'}")

        if emb_nan > 0 and last_good < op:
            print(f"\n*** NaN first appears in A_EMBEDDED_INPUT at op {op} ***")
            print(f"Check generated code:")
            print(f"  grep -B2 -A10 'Op {op-1}:' ~/.cache/ck-engine-v6.6/models/*/model_v6_6.c")
            print(f"  grep -B2 -A10 'Op {op}:' ~/.cache/ck-engine-v6.6/models/*/model_v6_6.c")

            # Show some details
            arr = result['embedded_arr']
            valid = ~np.isnan(arr)
            if valid.sum() > 0:
                print(f"\nPartial data (first 20 values): {arr[:20]}")
            break
        if emb_nan == 0:
            last_good = op
else:
    print("\nNo NaN found in checkpoints - model might be working!")
