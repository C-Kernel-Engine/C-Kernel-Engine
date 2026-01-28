#!/usr/bin/env python3
"""
Trace where NaN first appears in prefill.
"""

import ctypes
import numpy as np
import os
from pathlib import Path

cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
model_dirs = list(cache_dir.glob("*Qwen*")) + list(cache_dir.glob("*qwen*"))

if not model_dirs:
    print("No Qwen model found")
    exit(1)

model_dir = model_dirs[0]
lib_path = model_dir / "libmodel.so"
weights_path = model_dir / "weights.bump"

import json
layout_path = model_dir / "layout_prefill.json"
if not layout_path.exists():
    layout_path = model_dir / "layout_decode.json"

with open(layout_path) as f:
    layout = json.load(f)

embed_dim = layout.get("config", {}).get("embed_dim", 896)
vocab_size = layout.get("config", {}).get("vocab_size", 151936)

act_buffers = {}
for buf in layout.get("memory", {}).get("activations", {}).get("buffers", []):
    act_buffers[buf["name"]] = buf["abs_offset"]

def check_prefill_at_stop(stop_op, num_tokens=3):
    """Run prefill stopping at given op and check buffers."""
    os.environ['CK_STOP_OP'] = str(stop_op)

    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    # Prefill signature: (tokens, count, logits)
    lib.ck_model_prefill.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.ck_model_prefill.restype = ctypes.c_int
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

    # Use simple tokens: [1, 2, 3]
    tokens = (ctypes.c_int32 * num_tokens)(1, 2, 3)
    lib.ck_model_prefill(tokens, num_tokens, logits)

    del os.environ['CK_STOP_OP']

    # Check A_EMBEDDED_INPUT (num_tokens * embed_dim floats)
    emb_offset = act_buffers.get("embedded_input", 0)
    ptr = ctypes.cast(base_ptr + emb_offset, ctypes.POINTER(ctypes.c_float))
    emb_arr = np.ctypeslib.as_array(ptr, shape=(num_tokens * embed_dim,)).copy()

    # Check A_LAYER_INPUT
    li_offset = act_buffers.get("layer_input", 0)
    ptr2 = ctypes.cast(base_ptr + li_offset, ctypes.POINTER(ctypes.c_float))
    li_arr = np.ctypeslib.as_array(ptr2, shape=(num_tokens * embed_dim,)).copy()

    lib.ck_model_free()

    return {
        'embedded_nan': np.isnan(emb_arr).sum(),
        'layer_nan': np.isnan(li_arr).sum(),
        'embedded_inf': np.isinf(emb_arr).sum(),
        'layer_inf': np.isinf(li_arr).sum(),
        'embedded_arr': emb_arr,
        'layer_arr': li_arr,
        'logits_nan': np.isnan(np.ctypeslib.as_array(logits)).sum() if stop_op > 600 else 0
    }

print("Tracing NaN in prefill...")
print(f"Using {embed_dim}-dim embeddings, checking both buffers\n")

# Check key ops
# Op 0: embedding
# Op 1: rmsnorm
# Op 2: residual_save
# Op 3: quantize_input_0
# Op 4-6: q/k/v_proj
checkpoints = [0, 1, 2, 3, 4, 5, 6, 10, 20, 50, 100]

for cp in checkpoints:
    result = check_prefill_at_stop(cp)
    if result is None:
        print(f"Op {cp:3d}: FAILED TO RUN")
        continue

    emb_status = "OK" if result['embedded_nan'] == 0 else f"NaN({result['embedded_nan']})"
    li_status = "OK" if result['layer_nan'] == 0 else f"NaN({result['layer_nan']})"

    if result['embedded_inf'] > 0:
        emb_status += f"+Inf({result['embedded_inf']})"
    if result['layer_inf'] > 0:
        li_status += f"+Inf({result['layer_inf']})"

    print(f"Op {cp:3d}: A_EMBEDDED_INPUT={emb_status:20s}  A_LAYER_INPUT={li_status}")

    # If NaN found, show details
    if result['embedded_nan'] > 0 or result['layer_nan'] > 0:
        print(f"\n  First 10 A_EMBEDDED_INPUT values: {result['embedded_arr'][:10]}")
        print(f"  First 10 A_LAYER_INPUT values: {result['layer_arr'][:10]}")

        # Find first NaN index
        emb_nan_idx = np.where(np.isnan(result['embedded_arr']))[0]
        if len(emb_nan_idx) > 0:
            print(f"  First NaN in embedded at index: {emb_nan_idx[0]}")
        li_nan_idx = np.where(np.isnan(result['layer_arr']))[0]
        if len(li_nan_idx) > 0:
            print(f"  First NaN in layer at index: {li_nan_idx[0]}")
        break

print("\nDone.")
