#!/usr/bin/env python3
"""
Test direct Q5_0 x Q8_0 and Q8_0 x Q8_0 vec_dot parity between CK and llama.cpp.

These tests bypass the FP32-to-Q8_0 quantization step, using pre-quantized inputs
to isolate kernel correctness from quantization differences.
"""

import ctypes
import struct
import numpy as np
from pathlib import Path

# Library paths
BASE_DIR = Path(__file__).resolve().parents[2]
LLAMA_LIB = BASE_DIR / "llama.cpp" / "build" / "bin" / "libggml_kernel_test.so"
CK_LIB = BASE_DIR / "build" / "libck_parity.so"

# Block sizes
QK5_0 = 32
QK8_0 = 32
BLOCK_Q5_0_SIZE = 22  # 2 (d) + 4 (qh) + 16 (qs)
BLOCK_Q8_0_SIZE = 34  # 2 (d) + 32 (qs)

def fp16_to_bytes(val: float) -> bytes:
    return struct.pack('<e', val)

def create_q5_0_block(d: float, values: np.ndarray) -> bytes:
    """Create a Q5_0 block from scale and 32 int8 values."""
    block = bytearray()
    block.extend(fp16_to_bytes(d))

    # Compute qh (high bits) and qs (low nibbles)
    qh = 0
    for i in range(32):
        if (values[i] & 0x10):  # 5th bit
            qh |= (1 << i)
    block.extend(struct.pack('<I', qh))

    # qs: pack pairs of 4-bit values
    for i in range(16):
        lo = values[i * 2] & 0x0F
        hi = values[i * 2 + 1] & 0x0F
        block.append(lo | (hi << 4))

    return bytes(block)

def create_q8_0_block(d: float, qs: np.ndarray) -> bytes:
    """Create a Q8_0 block from scale and 32 int8 values."""
    block = bytearray()
    block.extend(fp16_to_bytes(d))
    block.extend(qs.astype(np.int8).tobytes())
    return bytes(block)

def test_vec_dot_q5_0_q8_0():
    """Test direct Q5_0 x Q8_0 dot product."""
    print("=" * 60)
    print("Testing direct Q5_0 x Q8_0 vec_dot")
    print("=" * 60)

    try:
        libck = ctypes.CDLL(str(CK_LIB))
        libllama = ctypes.CDLL(str(LLAMA_LIB))
        libllama.test_init()
    except OSError as e:
        print(f"ERROR: Could not load libraries: {e}")
        return False

    # Setup function signatures
    libck.ck_test_vec_dot_q5_0_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    libck.ck_test_vec_dot_q5_0_q8_0.restype = None

    libllama.test_vec_dot_q5_0_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    libllama.test_vec_dot_q5_0_q8_0.restype = None

    test_sizes = [32, 64, 128, 256, 512, 896]
    all_pass = True

    for K in test_sizes:
        n_blocks = K // QK5_0
        np.random.seed(42 + K)

        # Create Q5_0 weights
        q5_weights = bytearray()
        expected_q5_dequant = []
        for _ in range(n_blocks):
            d = np.random.uniform(0.01, 0.1)
            # Q5_0 values are 5-bit: 0-31, centered around 16
            raw_vals = np.random.randint(0, 32, size=32, dtype=np.uint8)
            q5_weights.extend(create_q5_0_block(d, raw_vals))
            # Dequantize: d * (qs - 16)
            for v in raw_vals:
                expected_q5_dequant.append(d * (v - 16))

        # Create Q8_0 activations
        q8_activations = bytearray()
        expected_q8_dequant = []
        for _ in range(n_blocks):
            d = np.random.uniform(0.01, 0.1)
            qs = np.random.randint(-127, 128, size=32, dtype=np.int8)
            q8_activations.extend(create_q8_0_block(d, qs))
            for v in qs:
                expected_q8_dequant.append(d * v)

        # Compute expected dot product using dequantized values
        expected = sum(a * b for a, b in zip(expected_q5_dequant, expected_q8_dequant))

        # Create ctypes buffers
        w_ptr = (ctypes.c_uint8 * len(q5_weights)).from_buffer_copy(bytes(q5_weights))
        a_ptr = (ctypes.c_uint8 * len(q8_activations)).from_buffer_copy(bytes(q8_activations))

        ck_out = ctypes.c_float(0.0)
        llama_out = ctypes.c_float(0.0)

        # Call both implementations
        libck.ck_test_vec_dot_q5_0_q8_0(w_ptr, a_ptr, ctypes.byref(ck_out), K)
        libllama.test_vec_dot_q5_0_q8_0(w_ptr, a_ptr, ctypes.byref(llama_out), K)

        # Compare
        diff = abs(ck_out.value - llama_out.value)
        max_val = max(abs(ck_out.value), abs(llama_out.value), 1e-6)
        rel_diff = diff / max_val

        status = "PASS" if rel_diff < 0.001 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"K={K:4d}: CK={ck_out.value:12.6f} llama={llama_out.value:12.6f} "
              f"diff={diff:10.6f} rel={rel_diff:.6f} [{status}]")

    return all_pass

def test_vec_dot_q8_0_q8_0():
    """Test direct Q8_0 x Q8_0 dot product."""
    print("\n" + "=" * 60)
    print("Testing direct Q8_0 x Q8_0 vec_dot")
    print("=" * 60)

    try:
        libck = ctypes.CDLL(str(CK_LIB))
        libllama = ctypes.CDLL(str(LLAMA_LIB))
        libllama.test_init()
    except OSError as e:
        print(f"ERROR: Could not load libraries: {e}")
        return False

    # Setup function signatures
    libck.ck_test_vec_dot_q8_0_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    libck.ck_test_vec_dot_q8_0_q8_0.restype = None

    libllama.test_vec_dot_q8_0_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    libllama.test_vec_dot_q8_0_q8_0.restype = None

    test_sizes = [32, 64, 128, 256, 512, 896]
    all_pass = True

    for K in test_sizes:
        n_blocks = K // QK8_0
        np.random.seed(42 + K)

        # Create Q8_0 weights
        q8_weights = bytearray()
        for _ in range(n_blocks):
            d = np.random.uniform(0.01, 0.1)
            qs = np.random.randint(-127, 128, size=32, dtype=np.int8)
            q8_weights.extend(create_q8_0_block(d, qs))

        # Create Q8_0 activations
        q8_activations = bytearray()
        for _ in range(n_blocks):
            d = np.random.uniform(0.01, 0.1)
            qs = np.random.randint(-127, 128, size=32, dtype=np.int8)
            q8_activations.extend(create_q8_0_block(d, qs))

        # Create ctypes buffers
        w_ptr = (ctypes.c_uint8 * len(q8_weights)).from_buffer_copy(bytes(q8_weights))
        a_ptr = (ctypes.c_uint8 * len(q8_activations)).from_buffer_copy(bytes(q8_activations))

        ck_out = ctypes.c_float(0.0)
        llama_out = ctypes.c_float(0.0)

        # Call both implementations
        libck.ck_test_vec_dot_q8_0_q8_0(w_ptr, a_ptr, ctypes.byref(ck_out), K)
        libllama.test_vec_dot_q8_0_q8_0(w_ptr, a_ptr, ctypes.byref(llama_out), K)

        # Compare
        diff = abs(ck_out.value - llama_out.value)
        max_val = max(abs(ck_out.value), abs(llama_out.value), 1e-6)
        rel_diff = diff / max_val

        status = "PASS" if rel_diff < 0.001 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"K={K:4d}: CK={ck_out.value:12.6f} llama={llama_out.value:12.6f} "
              f"diff={diff:10.6f} rel={rel_diff:.6f} [{status}]")

    return all_pass

if __name__ == "__main__":
    print("Direct vec_dot parity tests (bypass FP32-to-Q8 quantization)")
    print("=" * 60)

    q5_pass = test_vec_dot_q5_0_q8_0()
    q8_pass = test_vec_dot_q8_0_q8_0()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Q5_0 x Q8_0: {'PASS' if q5_pass else 'FAIL'}")
    print(f"Q8_0 x Q8_0: {'PASS' if q8_pass else 'FAIL'}")

    if q5_pass and q8_pass:
        print("\nAll direct vec_dot tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        exit(1)
