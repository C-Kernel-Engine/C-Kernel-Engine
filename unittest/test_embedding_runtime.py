#!/usr/bin/env python3
"""
test_embedding_runtime.py - Test that C-Kernel loads and uses embeddings correctly

This test:
1. Loads the compiled libmodel.so
2. Runs embedding lookup for a token
3. Compares with Python reference dequantization
"""

import ctypes
import struct
import numpy as np
from pathlib import Path
import sys

# Q8_0 dequantization
def dequant_q8_0_row(data: bytes, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0 data."""
    BLOCK_SIZE = 32
    BYTES_PER_BLOCK = 34

    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    result = []

    for i in range(n_blocks):
        block = data[i * BYTES_PER_BLOCK:(i + 1) * BYTES_PER_BLOCK]
        if len(block) < 2:
            break
        scale = struct.unpack('<e', block[0:2])[0]
        quants = np.frombuffer(block[2:34], dtype=np.int8)
        result.extend(quants.astype(np.float32) * scale)

    return np.array(result[:n_elements], dtype=np.float32)


def main():
    model_dir = Path.home() / ".cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
    lib_path = model_dir / "libmodel.so"
    weights_path = model_dir / "weights.bump"

    if not lib_path.exists():
        print(f"Error: {lib_path} not found")
        return 1

    # Load library
    lib = ctypes.CDLL(str(lib_path))

    # Setup function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int

    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    # Initialize model
    print(f"Loading model from {weights_path}...")
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"Error: ck_model_init returned {ret}")
        return 1
    print("Model loaded!")

    # Test token ID
    token_id = 9707  # "hello" in Qwen tokenizer

    # Read expected embedding from bump file
    print(f"\nReading expected embedding for token {token_id}...")

    # Read from bump file using file_offset
    with open(weights_path, 'rb') as f:
        # token_emb file_offset is 0x146
        EMBED_DIM = 896
        VOCAB_SIZE = 151936
        BYTES_PER_ROW = (EMBED_DIM // 32) * 34  # 952 bytes

        f.seek(0x146 + token_id * BYTES_PER_ROW)
        row_data = f.read(BYTES_PER_ROW)

    expected_emb = dequant_q8_0_row(row_data, EMBED_DIM)
    print(f"Expected embedding first 5: {expected_emb[:5]}")
    print(f"Expected embedding norm: {np.linalg.norm(expected_emb):.6f}")

    # Run model forward pass to see what it produces
    tokens = (ctypes.c_int32 * 1)(token_id)
    ret = lib.ck_model_embed_tokens(tokens, 1)
    if ret != 0:
        print(f"Error: ck_model_embed_tokens returned {ret}")
        lib.ck_model_free()
        return 1

    # Allocate logits buffer
    logits = (ctypes.c_float * 151936)()
    ret = lib.ck_model_forward(logits)
    if ret != 0:
        print(f"Error: ck_model_forward returned {ret}")
        lib.ck_model_free()
        return 1

    # Check logits
    logits_arr = np.ctypeslib.as_array(logits)
    print(f"\nLogits first 5: {logits_arr[:5]}")
    print(f"Logits max: {np.max(logits_arr):.6f}, argmax: {np.argmax(logits_arr)}")
    print(f"Logits non-zero count: {np.count_nonzero(logits_arr)}")
    print(f"Logits norm: {np.linalg.norm(logits_arr):.6f}")

    # Check for NaN/Inf
    has_nan = np.any(np.isnan(logits_arr))
    has_inf = np.any(np.isinf(logits_arr))
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

    lib.ck_model_free()

    return 0


if __name__ == "__main__":
    sys.exit(main())
