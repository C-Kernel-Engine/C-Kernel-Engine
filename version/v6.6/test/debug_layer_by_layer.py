#!/usr/bin/env python3
"""
Debug layer-by-layer by dumping activations at each op.
Uses CK_STOP_OP to stop at specific ops and examine intermediate state.
"""

import ctypes
import numpy as np
import os
from pathlib import Path

def setup_lib(model_dir):
    """Load the CK-Engine library."""
    # Load kernel engine library first (has kernel symbols)
    kernel_lib_path = model_dir / 'libckernel_engine.so'
    if kernel_lib_path.exists():
        ctypes.CDLL(str(kernel_lib_path), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(model_dir / 'libmodel.so'))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    # Try to get activation buffer pointer
    try:
        lib.ck_model_get_base_ptr.argtypes = []
        lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
    except:
        pass

    return lib

def read_buffer(lib, offset, count, dtype='float32'):
    """Read values from bump allocation at given offset."""
    try:
        base = lib.ck_model_get_base_ptr()
        if base == 0:
            return None
        # Create pointer to the offset
        ptr_type = ctypes.POINTER(ctypes.c_float) if dtype == 'float32' else ctypes.POINTER(ctypes.c_uint8)
        ptr = ctypes.cast(base + offset, ptr_type)
        return np.ctypeslib.as_array(ptr, shape=(count,)).copy()
    except:
        return None

def main():
    model_dir = Path.home() / '.cache/ck-engine-v6.6/models/qwen2-0_5b-instruct-q4_k_m'

    # Buffer offsets from generated code
    # These are specific to the model - check model_v6_6.c for exact values
    A_EMBEDDED_INPUT = 118095872  # From generated #defines
    A_LAYER_INPUT = 119930880
    A_RESIDUAL = 121765888
    A_Q_SCRATCH = 123600896
    A_K_SCRATCH = 124496896
    A_V_SCRATCH = 124759040
    A_ATTN_SCRATCH = 125021184
    A_MLP_SCRATCH = 125856192

    EMBED_DIM = 896

    print("=" * 60)
    print("Layer-by-layer activation dump")
    print("=" * 60)

    lib = setup_lib(model_dir)
    weights_path = model_dir / 'weights.bump'
    lib.ck_model_init(str(weights_path).encode())

    vocab_size = lib.ck_model_get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    token_id = 9707  # "Hello"

    # Run full forward and check logits
    out = (ctypes.c_float * vocab_size)()
    ret = lib.ck_model_decode(token_id, out)
    logits = np.array(out)

    print(f"\nFull forward pass (token={token_id}):")
    print(f"  Return: {ret}")
    print(f"  Logits: min={logits.min():.4f}, max={logits.max():.4f}")
    print(f"  NaN: {np.isnan(logits).sum()}, Inf: {np.isinf(logits).sum()}")
    print(f"  Top 5: {np.argsort(logits)[-5:][::-1]}")

    # Check intermediate buffers if possible
    embedded = read_buffer(lib, A_EMBEDDED_INPUT, EMBED_DIM)
    if embedded is not None:
        print(f"\n  A_EMBEDDED_INPUT (after full forward):")
        print(f"    min={embedded.min():.4f}, max={embedded.max():.4f}")
        print(f"    first 5: {embedded[:5]}")

    lib.ck_model_free()
    print("\nDone.")

if __name__ == '__main__':
    main()
