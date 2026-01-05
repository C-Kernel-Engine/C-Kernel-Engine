#!/usr/bin/env python3
"""Test if C-Kernel Q5_0 dequantization matches llama.cpp"""
import ctypes
import numpy as np
import struct
import os
import json

# Load C-Kernel library
LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "libckernel_engine.so")

def fp16_to_fp32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            return (-1)**sign * 0.0
        return (-1)**sign * (mant / 1024) * (2**-14)
    elif exp == 31:
        return float('inf') if sign == 0 else float('-inf')
    return (-1)**sign * (1 + mant/1024) * (2**(exp-15))

def dequant_q5_0_llama_cpp(data):
    """Dequantize Q5_0 block using llama.cpp's exact algorithm"""
    assert len(data) == 22

    d = struct.unpack('<H', data[0:2])[0]
    scale = fp16_to_fp32(d)
    qh = struct.unpack('<I', data[2:6])[0]
    qs = data[6:22]

    result = np.zeros(32, dtype=np.float32)

    for j in range(16):
        lo = qs[j] & 0x0F
        hi = qs[j] >> 4

        # llama.cpp order: indices 0-15 use low nibbles, 16-31 use high nibbles
        xh_0 = ((qh >> j) & 1) << 4
        xh_1 = ((qh >> (j + 16)) & 1) << 4

        q0 = (lo | xh_0) - 16
        q1 = (hi | xh_1) - 16

        result[j] = scale * q0
        result[j + 16] = scale * q1

    return result

def test_q5_0_dequant():
    """Test Q5_0 dequantization matches llama.cpp reference"""
    lib = ctypes.CDLL(LIB_PATH)

    lib.gemm_nt_q5_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    BUMP_PATH = os.path.expanduser("~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump")
    MANIFEST_PATH = os.path.expanduser("~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights_manifest.json")

    if not os.path.exists(BUMP_PATH) or not os.path.exists(MANIFEST_PATH):
        print("SKIP: Model files not found")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    wq_offset = None
    for entry in manifest['entries']:
        if entry['name'] == 'layer.0.wq':
            wq_offset = entry['file_offset']
            break

    if wq_offset is None:
        print("SKIP: layer.0.wq not found in manifest")
        return

    with open(BUMP_PATH, 'rb') as f:
        f.seek(wq_offset)
        block_data = f.read(22)

    print("=== Q5_0 Dequantization Test ===")
    print(f"Raw bytes: {block_data.hex()}")

    py_result = dequant_q5_0_llama_cpp(block_data)
    print(f"Python first 8: {py_result[:8]}")

    K = 32
    N = 1
    M = 1

    B = np.frombuffer(block_data, dtype=np.uint8)

    print("\n=== Individual Weight Extraction ===")
    errors = 0
    for test_idx in range(32):
        A_test = np.zeros(K, dtype=np.float32)
        A_test[test_idx] = 1.0
        C_test = np.zeros(N, dtype=np.float32)

        lib.gemm_nt_q5_0(
            A_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.c_void_p),
            None,
            C_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, N, K
        )

        match = np.isclose(C_test[0], py_result[test_idx], rtol=1e-5)
        if not match:
            print(f"FAIL Weight[{test_idx:2d}]: C={C_test[0]:10.6f}, Py={py_result[test_idx]:10.6f}")
            errors += 1

    if errors == 0:
        print("PASS: All 32 weights match llama.cpp reference!")
    else:
        print(f"FAIL: {errors}/32 weights do not match")

    # Sum test
    A = np.ones(K, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)

    lib.gemm_nt_q5_0(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.c_void_p),
        None,
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, N, K
    )

    print(f"\nSum test: C={C[0]:.6f}, Py={py_result.sum():.6f}, Match={np.isclose(C[0], py_result.sum())}")

if __name__ == "__main__":
    test_q5_0_dequant()
