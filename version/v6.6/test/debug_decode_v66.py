#!/usr/bin/env python3
"""
Debug v6.6 decode by dumping activations at each step.
"""

import ctypes
import numpy as np
import os
import sys

V66_DIR = os.path.expanduser("~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")

def load_model():
    """Load v6.6 model."""
    so_path = os.path.join(V66_DIR, "ck-kernel-inference.so")
    if not os.path.exists(so_path):
        so_path = os.path.join(V66_DIR, "libmodel.so")
    if not os.path.exists(so_path):
        print(f"Error: No .so file found in {V66_DIR}")
        sys.exit(1)

    lib = ctypes.CDLL(so_path)

    # Initialize
    weights_path = os.path.join(V66_DIR, "weights.bump")
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    ret = lib.ck_model_init(weights_path.encode())
    if ret != 0:
        print(f"Model init failed: {ret}")
        sys.exit(1)

    return lib

def get_activation_ptr(lib, offset, count):
    """Get activation buffer pointer."""
    lib.ck_model_get_activations.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_get_bump.restype = ctypes.c_void_p

    bump = lib.ck_model_get_bump()
    ptr = ctypes.cast(bump + offset, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(count,))

def test_decode():
    """Test decode with step-by-step output."""
    lib = load_model()

    # Set tokens to embed - "Hello!" chat template tokens
    tokens = [151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]  # Approximate

    # Prefill
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    tokens_arr = (ctypes.c_int32 * len(tokens))(*tokens)
    ret = lib.ck_model_embed_tokens(tokens_arr, len(tokens))
    print(f"Embed tokens returned: {ret}")

    # Forward
    lib.ck_model_forward.argtypes = []
    lib.ck_model_forward.restype = ctypes.c_int

    ret = lib.ck_model_forward()
    print(f"Forward returned: {ret}")

    # Get logits
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    logits_ptr = lib.ck_model_get_logits()
    logits = np.ctypeslib.as_array(logits_ptr, shape=(151936,))

    top_5 = np.argsort(logits)[-5:][::-1]
    print(f"Prefill top-5 tokens: {top_5}")
    print(f"Prefill top-5 logits: {logits[top_5]}")

    # Decode step
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.c_void_p]
    lib.ck_model_decode.restype = ctypes.c_int

    next_token = int(top_5[0])
    print(f"\nDecode with token: {next_token}")

    ret = lib.ck_model_decode(next_token, None)
    print(f"Decode returned: {ret}")

    # Get decode logits
    logits = np.ctypeslib.as_array(logits_ptr, shape=(151936,))
    top_5 = np.argsort(logits)[-5:][::-1]
    print(f"Decode top-5 tokens: {top_5}")
    print(f"Decode top-5 logits: {logits[top_5]}")

    # Check for NaN/Inf
    if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
        print("WARNING: NaN or Inf in logits!")
        print(f"NaN count: {np.sum(np.isnan(logits))}")
        print(f"Inf count: {np.sum(np.isinf(logits))}")

if __name__ == "__main__":
    test_decode()
