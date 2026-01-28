#!/usr/bin/env python3
"""Check A_RESIDUAL buffer contents at each step."""
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
A_LAYER_INPUT = 396925276
A_RESIDUAL = 396928860

lib = load_model()
bump = lib.ck_model_get_base_ptr()
vocab_size = 151936
out = (ctypes.c_float * vocab_size)()
token_id = 9707

embedded_ptr = ctypes.cast(bump + A_EMBEDDED_INPUT, ctypes.POINTER(ctypes.c_float))
layer_input_ptr = ctypes.cast(bump + A_LAYER_INPUT, ctypes.POINTER(ctypes.c_float))
residual_ptr = ctypes.cast(bump + A_RESIDUAL, ctypes.POINTER(ctypes.c_float))

print("Checking buffer contents at key ops...")
print()

# Find residual_save (should be before op 1 - let's check)
for op in [0, 1, 2, 15, 16, 17, 24, 25]:
    os.environ['CK_STOP_OP'] = str(op)
    lib.ck_model_decode(token_id, out)
    embedded = np.ctypeslib.as_array(embedded_ptr, shape=(896,))
    residual = np.ctypeslib.as_array(residual_ptr, shape=(896,))
    layer_input = np.ctypeslib.as_array(layer_input_ptr, shape=(896,))
    
    print(f"After Op {op}:")
    print(f"  A_EMBEDDED_INPUT[0:3]: {embedded[:3]}, nan={np.isnan(embedded).sum()}")
    print(f"  A_RESIDUAL[0:3]:       {residual[:3]}, nan={np.isnan(residual).sum()}")
    # Interpret layer_input as uint8 (Q8_0 quantized)
    layer_input_q8 = np.ctypeslib.as_array(ctypes.cast(bump + A_LAYER_INPUT, ctypes.POINTER(ctypes.c_uint8)), shape=(100,))
    print(f"  A_LAYER_INPUT bytes:   {layer_input_q8[:10]}")
    print()

lib.ck_model_free()
