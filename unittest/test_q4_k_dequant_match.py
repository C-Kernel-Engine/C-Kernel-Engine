#!/usr/bin/env python3
"""Test if C-Kernel Q4_K dequantization matches llama.cpp"""
import ctypes
import numpy as np
import struct
import os
import json

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

def unpack_q4_k_scales(scales_bytes):
    """Unpack Q4_K scale bytes - 12 bytes for 8 scales + 8 mins

    This matches llama.cpp's get_scale_min_k4() function exactly.
    Layout:
      - bytes 0-3: 6-bit scales[0-3] (high 2 bits used for scales[4-7])
      - bytes 4-7: 6-bit mins[0-3] (high 2 bits used for mins[4-7])
      - bytes 8-11: low 4 bits for scales[4-7], high 4 bits for mins[4-7]
    """
    sc = [0] * 8
    m = [0] * 8

    # Direct 6-bit values for indices 0-3
    sc[0] = scales_bytes[0] & 0x3F
    sc[1] = scales_bytes[1] & 0x3F
    sc[2] = scales_bytes[2] & 0x3F
    sc[3] = scales_bytes[3] & 0x3F

    m[0] = scales_bytes[4] & 0x3F
    m[1] = scales_bytes[5] & 0x3F
    m[2] = scales_bytes[6] & 0x3F
    m[3] = scales_bytes[7] & 0x3F

    # 6-bit values for indices 4-7: low 4 bits from bytes 8-11,
    # high 2 bits from upper bits of bytes 0-3 (scales) and 4-7 (mins)
    sc[4] = (scales_bytes[8]  & 0x0F) | ((scales_bytes[0] >> 6) << 4)
    sc[5] = (scales_bytes[9]  & 0x0F) | ((scales_bytes[1] >> 6) << 4)
    sc[6] = (scales_bytes[10] & 0x0F) | ((scales_bytes[2] >> 6) << 4)
    sc[7] = (scales_bytes[11] & 0x0F) | ((scales_bytes[3] >> 6) << 4)

    m[4] = (scales_bytes[8]  >> 4) | ((scales_bytes[4] >> 6) << 4)
    m[5] = (scales_bytes[9]  >> 4) | ((scales_bytes[5] >> 6) << 4)
    m[6] = (scales_bytes[10] >> 4) | ((scales_bytes[6] >> 6) << 4)
    m[7] = (scales_bytes[11] >> 4) | ((scales_bytes[7] >> 6) << 4)

    return sc, m

def dequant_q4_k_llama_cpp(data):
    """Dequantize Q4_K block using llama.cpp's exact algorithm"""
    assert len(data) == 144  # 2+2+12+128 bytes

    d = struct.unpack('<H', data[0:2])[0]
    dmin = struct.unpack('<H', data[2:4])[0]
    d_f = fp16_to_fp32(d)
    dmin_f = fp16_to_fp32(dmin)

    scales_raw = data[4:16]
    sc, m = unpack_q4_k_scales(scales_raw)

    qs = data[16:144]  # 128 bytes for 256 nibbles

    result = np.zeros(256, dtype=np.float32)

    # llama.cpp processes in groups of 64 weights
    q_ptr = 0
    is_idx = 0
    y_ptr = 0

    for j in range(0, 256, 64):  # 4 iterations
        d1 = d_f * sc[is_idx]
        m1 = dmin_f * m[is_idx]
        d2 = d_f * sc[is_idx + 1]
        m2 = dmin_f * m[is_idx + 1]

        # First 32 weights: low nibbles (formula: d * q - m)
        for l in range(32):
            q = (qs[q_ptr + l] & 0x0F)
            result[y_ptr + l] = d1 * q - m1

        # Next 32 weights: high nibbles
        for l in range(32):
            q = (qs[q_ptr + l] >> 4)
            result[y_ptr + 32 + l] = d2 * q - m2

        q_ptr += 32
        is_idx += 2
        y_ptr += 64

    return result

def test_q4_k_dequant():
    """Test Q4_K dequantization matches llama.cpp reference"""
    lib = ctypes.CDLL(LIB_PATH)

    lib.gemm_nt_q4_k.argtypes = [
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

    # Find a Q4_K layer (w2 layers use q4_k or q6_k)
    w2_offset = None
    for entry in manifest['entries']:
        if entry['name'] == 'layer.2.w2' and entry['dtype'] == 'q4_k':
            w2_offset = entry['file_offset']
            break

    if w2_offset is None:
        print("SKIP: No Q4_K layer found")
        return

    with open(BUMP_PATH, 'rb') as f:
        f.seek(w2_offset)
        block_data = f.read(144)  # One Q4_K block

    print("=== Q4_K Dequantization Test ===")
    print(f"Block size: {len(block_data)} bytes")

    py_result = dequant_q4_k_llama_cpp(block_data)
    print(f"Python first 8: {py_result[:8]}")
    print(f"Python 32-40:   {py_result[32:40]}")

    K = 256  # One Q4_K block has 256 weights
    N = 1
    M = 1

    B = np.frombuffer(block_data, dtype=np.uint8)

    print("\n=== Individual Weight Extraction (sampled) ===")
    errors = 0
    test_indices = [0, 1, 15, 31, 32, 33, 63, 64, 127, 128, 191, 192, 255]

    for test_idx in test_indices:
        A_test = np.zeros(K, dtype=np.float32)
        A_test[test_idx] = 1.0
        C_test = np.zeros(N, dtype=np.float32)

        lib.gemm_nt_q4_k(
            A_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.c_void_p),
            None,
            C_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, N, K
        )

        match = np.isclose(C_test[0], py_result[test_idx], rtol=1e-4)
        status = "✓" if match else "✗"
        print(f"Weight[{test_idx:3d}]: C={C_test[0]:10.6f}, Py={py_result[test_idx]:10.6f} {status}")
        if not match:
            errors += 1

    # Full verification
    print("\n=== Full Weight Verification ===")
    full_errors = 0
    for test_idx in range(256):
        A_test = np.zeros(K, dtype=np.float32)
        A_test[test_idx] = 1.0
        C_test = np.zeros(N, dtype=np.float32)

        lib.gemm_nt_q4_k(
            A_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.c_void_p),
            None,
            C_test.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, N, K
        )

        if not np.isclose(C_test[0], py_result[test_idx], rtol=1e-4):
            full_errors += 1

    if full_errors == 0:
        print("PASS: All 256 weights match llama.cpp reference!")
    else:
        print(f"FAIL: {full_errors}/256 weights do not match")

if __name__ == "__main__":
    test_q4_k_dequant()
