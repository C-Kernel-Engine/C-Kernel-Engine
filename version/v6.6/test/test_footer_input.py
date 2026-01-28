#!/usr/bin/env python3
"""
Check what's in A_EMBEDDED_INPUT before the footer rmsnorm runs.
Stop at op 625 (before rmsnorm) vs op 626 (after quantize).
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

config = layout.get("config", {})
embed_dim = config.get("embed_dim", 896)

act_buffers = {}
for buf in layout.get("memory", {}).get("activations", {}).get("buffers", []):
    act_buffers[buf["name"]] = buf["abs_offset"]

def check_buffers_at_stop(stop_op, description):
    """Run decode stopping at given op and check activation buffers."""
    print(f"\n{'='*60}")
    print(f"Stop at op {stop_op}: {description}")
    print('='*60)

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
        print(f"Init failed: {ret}")
        del os.environ['CK_STOP_OP']
        return

    base_ptr = lib.ck_model_get_base_ptr()
    vocab = lib.ck_model_get_vocab_size()
    logits = (ctypes.c_float * vocab)()
    lib.ck_model_decode(9707, logits)

    del os.environ['CK_STOP_OP']

    # Check A_EMBEDDED_INPUT
    if "embedded_input" in act_buffers:
        emb_offset = act_buffers["embedded_input"]
        ptr = ctypes.cast(base_ptr + emb_offset, ctypes.POINTER(ctypes.c_float))
        arr = np.ctypeslib.as_array(ptr, shape=(embed_dim,))

        print(f"\nA_EMBEDDED_INPUT (offset {emb_offset}):")
        print(f"  NaN count: {np.isnan(arr).sum()}/{embed_dim}")
        print(f"  Inf count: {np.isinf(arr).sum()}/{embed_dim}")
        if np.isnan(arr).sum() == 0 and np.isinf(arr).sum() == 0:
            print(f"  Range: [{arr.min():.6f}, {arr.max():.6f}]")
            print(f"  Mean: {arr.mean():.6f}")
            print(f"  First 10: {arr[:10]}")
        else:
            # Show which values are valid
            valid = ~np.isnan(arr) & ~np.isinf(arr)
            print(f"  Valid values: {valid.sum()}")
            if valid.sum() > 0:
                print(f"  Valid range: [{arr[valid].min():.6f}, {arr[valid].max():.6f}]")

    # Check A_LAYER_INPUT
    if "layer_input" in act_buffers:
        li_offset = act_buffers["layer_input"]
        ptr = ctypes.cast(base_ptr + li_offset, ctypes.POINTER(ctypes.c_float))
        arr = np.ctypeslib.as_array(ptr, shape=(embed_dim,))

        print(f"\nA_LAYER_INPUT (offset {li_offset}):")
        print(f"  NaN count: {np.isnan(arr).sum()}/{embed_dim}")
        print(f"  Inf count: {np.isinf(arr).sum()}/{embed_dim}")
        if np.isnan(arr).sum() == 0 and np.isinf(arr).sum() == 0:
            print(f"  Range: [{arr.min():.6f}, {arr.max():.6f}]")
            print(f"  Mean: {arr.mean():.6f}")
            print(f"  First 10: {arr[:10]}")

    lib.ck_model_free()

# Check at different points
check_buffers_at_stop(624, "Before footer (after last layer residual_add)")
check_buffers_at_stop(625, "After footer rmsnorm")
check_buffers_at_stop(626, "After quantize_final_output")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print("""
Footer ops sequence:
  Op 624: residual_add (last layer) -> writes to A_EMBEDDED_INPUT
  Op 625: rmsnorm (footer) -> reads A_EMBEDDED_INPUT, writes A_LAYER_INPUT
  Op 626: quantize_final_output -> reads A_LAYER_INPUT, writes A_EMBEDDED_INPUT (Q8_0)
  Op 627: gemv_q8_0_q8_0 -> reads A_EMBEDDED_INPUT (Q8_0), writes A_LOGITS

If A_EMBEDDED_INPUT is NaN at stop=624, the issue is in the last layer.
If A_EMBEDDED_INPUT is OK at stop=624 but A_LAYER_INPUT is NaN at stop=625,
the issue is in the footer rmsnorm.
""")
