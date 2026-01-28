#!/usr/bin/env python3
"""Trace where NaN first appears by stopping at each op."""
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
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
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
A_LOGITS = 1210997596

lib = load_model()
bump = lib.ck_model_get_base_ptr()
vocab_size = 151936
out = (ctypes.c_float * vocab_size)()
token_id = 9707

print("Testing decode...")
print(f"Bump base pointer: {hex(bump)}")

# Clear stop env
if 'CK_STOP_OP' in os.environ:
    del os.environ['CK_STOP_OP']

# Do a full run
lib.ck_model_decode(token_id, out)

# Check logits via API
logits_ptr = lib.ck_model_get_logits()
logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
print(f"\nLogits from API:")
print(f"  Logits[0:5]: {logits[:5]}")
print(f"  NaN count: {np.isnan(logits).sum()}")

# Check logits from A_LOGITS buffer directly
logits_direct = ctypes.cast(bump + A_LOGITS, ctypes.POINTER(ctypes.c_float))
logits_arr = np.ctypeslib.as_array(logits_direct, shape=(vocab_size,))
print(f"\nLogits from A_LOGITS buffer:")
print(f"  Logits[0:5]: {logits_arr[:5]}")
print(f"  NaN count: {np.isnan(logits_arr).sum()}")

# Check embedded input (post-rmsnorm)
embedded_ptr = ctypes.cast(bump + A_EMBEDDED_INPUT, ctypes.POINTER(ctypes.c_float))
embedded = np.ctypeslib.as_array(embedded_ptr, shape=(896,))
print(f"\nA_EMBEDDED_INPUT (post-rmsnorm):")
print(f"  [0:5]: {embedded[:5]}")
print(f"  NaN count: {np.isnan(embedded).sum()}")

lib.ck_model_free()
