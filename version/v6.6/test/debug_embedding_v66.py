#!/usr/bin/env python3
"""
Debug v6.6 embedding and early operations.
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path

V66_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V65_DIR = Path.home() / ".cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

EMBED_DIM = 896
VOCAB_SIZE = 151936

# V6.6 activation offsets
A_EMBEDDED_INPUT = 396942152
A_LAYER_INPUT = 400612168
A_Q_SCRATCH = 433380168


def test_v66():
    print("=" * 70)
    print("V6.6 EMBEDDING DEBUG")
    print("=" * 70)

    # Load engine
    engine_path = V66_DIR / "libckernel_engine.so"
    if engine_path.exists():
        ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

    lib_path = V66_DIR / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = V66_DIR / "libmodel.so"

    lib = ctypes.CDLL(str(lib_path))

    # Bind functions
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    # Try to get base pointer
    try:
        lib.ck_model_get_base_ptr.argtypes = []
        lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
        has_base_ptr = True
    except:
        has_base_ptr = False
        print("WARNING: ck_model_get_base_ptr not available")

    # Init
    weights_path = V66_DIR / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"Init failed: {ret}")
        return 1

    print(f"Model initialized successfully")

    vocab_size = lib.ck_model_get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    if has_base_ptr:
        base_ptr = lib.ck_model_get_base_ptr()
        print(f"Base pointer: 0x{base_ptr:x}")
    else:
        base_ptr = 0

    # Test 1: Run full decode (no stop)
    print("\n--- Test 1: Full decode ---")
    output = (ctypes.c_float * vocab_size)()
    ret = lib.ck_model_decode(100, output)
    print(f"Decode returned: {ret}")

    logits = np.array(output[:], dtype=np.float32)
    print(f"Logits: min={logits.min():.4f}, max={logits.max():.4f}")
    print(f"Logits argmax: {np.argmax(logits)}")
    print(f"First 10 logits: {logits[:10]}")
    print(f"Non-zero count: {np.count_nonzero(logits)}")

    # Test 2: Run with stop_seq=0 (after embedding)
    print("\n--- Test 2: Stop after embedding (stop_seq=0) ---")
    os.environ["CK_STOP_OP"] = "0"
    output = (ctypes.c_float * vocab_size)()
    ret = lib.ck_model_decode(100, output)
    del os.environ["CK_STOP_OP"]

    logits = np.array(output[:], dtype=np.float32)
    print(f"Decode returned: {ret}")
    print(f"Logits: min={logits.min():.4f}, max={logits.max():.4f}")
    print(f"Non-zero count: {np.count_nonzero(logits)}")

    # Test 3: Read embedding output from memory
    if base_ptr:
        print("\n--- Test 3: Read embedding activation ---")
        ptr = ctypes.cast(base_ptr + A_EMBEDDED_INPUT, ctypes.POINTER(ctypes.c_float))
        embed = np.ctypeslib.as_array(ptr, shape=(EMBED_DIM,)).copy()
        print(f"Embedding: min={embed.min():.4f}, max={embed.max():.4f}")
        print(f"First 10: {embed[:10]}")
        print(f"Non-zero count: {np.count_nonzero(embed)}")

        # Read layer input (after rmsnorm)
        print("\n--- Test 4: Read layer_input activation ---")
        ptr = ctypes.cast(base_ptr + A_LAYER_INPUT, ctypes.POINTER(ctypes.c_float))
        layer_in = np.ctypeslib.as_array(ptr, shape=(EMBED_DIM,)).copy()
        print(f"Layer input: min={layer_in.min():.4f}, max={layer_in.max():.4f}")
        print(f"First 10: {layer_in[:10]}")

        # Read Q scratch
        print("\n--- Test 5: Read Q scratch ---")
        ptr = ctypes.cast(base_ptr + A_Q_SCRATCH, ctypes.POINTER(ctypes.c_float))
        q = np.ctypeslib.as_array(ptr, shape=(EMBED_DIM,)).copy()
        print(f"Q scratch: min={q.min():.4f}, max={q.max():.4f}")
        print(f"First 10: {q[:10]}")

    lib.ck_model_free()
    return 0


def test_v65():
    print("\n" + "=" * 70)
    print("V6.5 REFERENCE (for comparison)")
    print("=" * 70)

    lib_path = V65_DIR / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = V65_DIR / "libmodel.so"

    lib = ctypes.CDLL(str(lib_path))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    weights_path = V65_DIR / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"Init failed: {ret}")
        return 1

    vocab_size = lib.ck_model_get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # Full decode
    print("\n--- Full decode ---")
    output = (ctypes.c_float * vocab_size)()
    ret = lib.ck_model_decode(100, output)
    print(f"Decode returned: {ret}")

    logits = np.array(output[:], dtype=np.float32)
    print(f"Logits: min={logits.min():.4f}, max={logits.max():.4f}")
    print(f"Logits argmax: {np.argmax(logits)}")
    print(f"First 10 logits: {logits[:10]}")

    lib.ck_model_free()
    return 0


if __name__ == "__main__":
    test_v66()
    test_v65()
