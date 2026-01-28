#!/usr/bin/env python3
"""Narrow down where NaN first appears in layer 0."""
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

A_EMBEDDED_INPUT = 396921692

lib = load_model()
bump = lib.ck_model_get_base_ptr()
vocab_size = 151936
out = (ctypes.c_float * vocab_size)()
token_id = 9707

embedded_ptr = ctypes.cast(bump + A_EMBEDDED_INPUT, ctypes.POINTER(ctypes.c_float))

print("--- Detailed trace ops 17-30 ---")
for op in range(17, 31):
    os.environ['CK_STOP_OP'] = str(op)
    lib.ck_model_decode(token_id, out)
    embedded = np.ctypeslib.as_array(embedded_ptr, shape=(896,))
    nan_count = np.isnan(embedded).sum()
    inf_count = np.isinf(embedded).sum()
    max_val = np.abs(np.nan_to_num(embedded, nan=0, posinf=0, neginf=0)).max()
    print(f"Op {op:2d}: NaN={nan_count:3d}, Inf={inf_count:3d}, maxabs={max_val:.4e}, [0:3]={embedded[:3]}")

lib.ck_model_free()
