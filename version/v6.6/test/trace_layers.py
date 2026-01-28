#!/usr/bin/env python3
"""Trace layer by layer to find where NaN first appears."""
import ctypes
import numpy as np
import os
from pathlib import Path

def load_model():
    model_dir = Path.home() / '.cache/ck-engine-v6.6/models/qwen2-0_5b-instruct-q4_k_m'
    kernel_lib = ctypes.CDLL(str(model_dir / 'libckernel_engine.so'), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(model_dir / 'libmodel.so'))
    
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None
    
    weights_path = model_dir / 'weights.bump'
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        raise RuntimeError(f"Failed to init: {ret}")
    return lib

# Correct buffer offsets from generated code
A_EMBEDDED_INPUT = 396921692
A_LAYER_INPUT = 396925276
A_RESIDUAL = 396928860

lib = load_model()
bump = lib.ck_model_get_base_ptr()
vocab_size = 151936
out = (ctypes.c_float * vocab_size)()
token_id = 9707

# Key ops to check (looking at decode section):
# Op 0: embedding (writes to A_EMBEDDED_INPUT)
# Op 1-2: layer 0 starts (quantize, q_proj, etc)
# ...each layer ~26 ops

# Stop at op 0 (embedding)
os.environ['CK_STOP_OP'] = '0'
lib.ck_model_decode(token_id, out)
embedded_ptr = ctypes.cast(bump + A_EMBEDDED_INPUT, ctypes.POINTER(ctypes.c_float))
embedded = np.ctypeslib.as_array(embedded_ptr, shape=(896,))
print(f"After Op 0 (embedding):")
print(f"  A_EMBEDDED_INPUT[0:5]: {embedded[:5]}")
print(f"  NaN: {np.isnan(embedded).sum()}, Range: [{embedded.min():.4f}, {embedded.max():.4f}]")

# Stop after layer 0 attention (roughly op 17)
os.environ['CK_STOP_OP'] = '17'
lib.ck_model_decode(token_id, out)
embedded = np.ctypeslib.as_array(embedded_ptr, shape=(896,))
print(f"\nAfter Op 17 (layer 0 attn):")
print(f"  A_EMBEDDED_INPUT[0:5]: {embedded[:5]}")
print(f"  NaN: {np.isnan(embedded).sum()}")

# Stop after layer 0 complete (roughly op 26)
os.environ['CK_STOP_OP'] = '26'
lib.ck_model_decode(token_id, out)
embedded = np.ctypeslib.as_array(embedded_ptr, shape=(896,))
print(f"\nAfter Op 26 (layer 0 complete):")
print(f"  A_EMBEDDED_INPUT[0:5]: {embedded[:5]}")
print(f"  NaN: {np.isnan(embedded).sum()}")

# Check each op from 0 to 5 to narrow down
print("\n--- Detailed trace ops 0-10 ---")
for op in range(11):
    os.environ['CK_STOP_OP'] = str(op)
    lib.ck_model_decode(token_id, out)
    embedded = np.ctypeslib.as_array(embedded_ptr, shape=(896,))
    nan_count = np.isnan(embedded).sum()
    print(f"Op {op:2d}: NaN={nan_count:3d}, A_EMBEDDED_INPUT[0:3]={embedded[:3]}")

lib.ck_model_free()
