#!/usr/bin/env python3
"""Detailed trace ops 17-27 to verify A_RESIDUAL is not corrupted."""
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

# NEW correct offset
A_RESIDUAL = 396931420

lib = load_model()
bump = lib.ck_model_get_base_ptr()
out = (ctypes.c_float * 151936)()
token_id = 9707

residual_ptr = ctypes.cast(bump + A_RESIDUAL, ctypes.POINTER(ctypes.c_float))

print("Tracing A_RESIDUAL ops 17-27 (with proper Q8_K gap)...")
for op in range(17, 28):
    os.environ['CK_STOP_OP'] = str(op)
    lib.ck_model_decode(token_id, out)
    residual = np.ctypeslib.as_array(residual_ptr, shape=(896,))
    nan_count = np.isnan(residual).sum()
    max_abs = np.abs(np.nan_to_num(residual, nan=0, posinf=0, neginf=0)).max()
    print(f"Op {op}: NaN={nan_count:3d}, maxabs={max_abs:.4e}, [0:3]={residual[:3]}")

lib.ck_model_free()
