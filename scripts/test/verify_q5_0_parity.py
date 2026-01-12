#!/usr/bin/env python3
"""
Direct parity test: CK vec_dot_q5_0_q8_0 vs llama.cpp ggml_vec_dot_q5_0_q8_0

This test creates identical Q5_0 weights and Q8_0 inputs, then calls both
implementations to verify they produce identical results.
"""

import ctypes
import struct
import numpy as np
from pathlib import Path

# Library paths
BASE_DIR = Path(__file__).resolve().parents[2]
LLAMA_LIB = BASE_DIR / "llama.cpp" / "libggml_kernel_test.so"
CK_LIB = BASE_DIR / "build" / "libck_parity.so"

# Block sizes
QK5_0 = 32
QK8_0 = 32
BLOCK_Q5_0_SIZE = 22  # 2 (d) + 4 (qh) + 16 (qs)
BLOCK_Q8_0_SIZE = 34  # 2 (d) + 32 (qs)

def fp16_to_bytes(val: float) -> bytes:
    """Convert FP32 to FP16 bytes."""
    return struct.pack('<e', val)

def create_simple_q5_0_block(scale: float, values: list) -> bytes:
    """Create a Q5_0 block with specific values.

    Q5_0 format:
    - d: FP16 scale
    - qh: 4 bytes (32 high bits)
    - qs: 16 bytes (32 x 4-bit nibbles, packed as pairs)

    Weight j = d * ((qs[j%16] nibble | qh_bit[j] << 4) - 16)
    """
    assert len(values) == 32

    data = bytearray()

    # d: FP16 scale
    data.extend(fp16_to_bytes(scale))

    # Convert values to Q5_0 encoding
    qh = 0  # High bits
    qs = [0] * 16  # Low nibbles

    for j in range(32):
        # Each value is in range [-16, 15] (signed 5-bit)
        v = max(-16, min(15, int(values[j])))
        q5 = v + 16  # Convert to unsigned [0, 31]

        lo_nibble = q5 & 0x0F
        hi_bit = (q5 >> 4) & 1

        if j < 16:
            # Low nibble of qs[j]
            qs[j] |= lo_nibble
            # High bit at position j
            if hi_bit:
                qh |= (1 << j)
        else:
            # High nibble of qs[j-16]
            qs[j - 16] |= (lo_nibble << 4)
            # High bit at position j (not j+12 - that's for the access pattern)
            # Actually, looking at the ref implementation: qh bit (j+12) for second weight
            # Weight j+16 uses qh bit (j+12) where j is 0-15
            # So for our value at index j (16-31), the qh bit is at position (j - 16 + 12) = j - 4
            # Wait no - let me re-read the ref code more carefully
            pass

    # Let me re-examine the Q5_0 layout from the ref implementation:
    # for j in 0..15:
    #   x0 uses qs[j] & 0x0F, qh bit j       -> weight index j
    #   x1 uses qs[j] >> 4,  qh bit (j+12)   -> weight index j+16
    #
    # So to encode:
    # - Weight j (0-15): low nibble in qs[j], high bit at qh[j]
    # - Weight j (16-31): high nibble in qs[j-16], high bit at qh[j-16+12] = qh[j-4]

    qh = 0
    qs = [0] * 16

    for j in range(16):
        # Weight j: value stored in low nibble
        v0 = max(-16, min(15, int(values[j])))
        q5_0 = v0 + 16  # [0, 31]
        lo_0 = q5_0 & 0x0F
        hi_0 = (q5_0 >> 4) & 1
        qs[j] |= lo_0
        if hi_0:
            qh |= (1 << j)

        # Weight j+16: value stored in high nibble
        v1 = max(-16, min(15, int(values[j + 16])))
        q5_1 = v1 + 16  # [0, 31]
        lo_1 = q5_1 & 0x0F
        hi_1 = (q5_1 >> 4) & 1
        qs[j] |= (lo_1 << 4)
        if hi_1:
            qh |= (1 << (j + 12))

    # Pack qh as 4 bytes (little-endian)
    data.extend(struct.pack('<I', qh))

    # Pack qs
    data.extend(bytes(qs))

    return bytes(data)

def create_simple_q8_0_block(scale: float, values: list) -> bytes:
    """Create a Q8_0 block with specific values.

    Q8_0 format:
    - d: FP16 scale
    - qs: 32 x int8 values
    """
    assert len(values) == 32

    data = bytearray()

    # d: FP16 scale
    data.extend(fp16_to_bytes(scale))

    # qs: 32 x int8
    for v in values:
        iv = max(-128, min(127, int(v)))
        data.extend(struct.pack('b', iv))  # signed byte

    return bytes(data)

def test_single_block_parity():
    """Test a single block with known values."""
    print("=" * 70)
    print("TEST: Single block Q5_0 x Q8_0 parity")
    print("=" * 70)

    # Load libraries
    try:
        libck = ctypes.CDLL(str(CK_LIB))
        libllama = ctypes.CDLL(str(LLAMA_LIB))
        libllama.test_init()
    except OSError as e:
        print(f"ERROR: Could not load libraries: {e}")
        return False

    # Simple test: scale=1.0, all weights=1, all inputs=1
    # Expected dot product: 32 * 1 * 1 * 1 * 1 = 32

    # Create Q5_0 block: all values = 1 (which is encoded as 1+16=17, lo=1, hi=1)
    # Wait, Q5_0 encoding is: value = d * (q5 - 16), so to get value=1:
    # 1 = d * (q5 - 16) => q5 = 17 (assuming d=1)
    # q5=17 = 0b10001, lo_nibble=1, hi_bit=1

    w_scale = 1.0
    x_scale = 1.0

    # Create weights: all 1s
    w_values = [1.0] * 32

    # Create input: all 1s (as float, then quantized to Q8_0 with scale=1/127)
    # Q8_0: value = d * qs, so if qs=127 and d=1/127, value=1
    # But for simplicity, let's use actual quantization

    # Actually, let's use the test wrapper that quantizes FP32 input
    # That's what the comprehensive test does

    print("\nTest 1: Using test wrappers (FP32 input -> Q8_0 quantization)")

    # Setup CK function
    libck.ck_test_gemv_q5_0_q8_0.argtypes = [
        ctypes.c_void_p,                    # weight_q5_0
        ctypes.POINTER(ctypes.c_float),     # input_f32
        ctypes.POINTER(ctypes.c_float),     # output
        ctypes.c_int,                       # rows
        ctypes.c_int                        # cols
    ]
    libck.ck_test_gemv_q5_0_q8_0.restype = None

    # Setup llama function
    libllama.test_gemv_q5_0.argtypes = [
        ctypes.c_void_p,                    # weight_q5_0
        ctypes.POINTER(ctypes.c_float),     # input_f32
        ctypes.POINTER(ctypes.c_float),     # output
        ctypes.c_int                        # cols
    ]
    libllama.test_gemv_q5_0.restype = None

    K = 32  # One block

    # Create simple Q5_0 weights
    weights = create_simple_q5_0_block(1.0, [1.0] * 32)
    w_ptr = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)

    # Create FP32 input
    input_f32 = np.ones(K, dtype=np.float32)
    input_ptr = input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Output buffers
    ck_out = np.zeros(1, dtype=np.float32)
    llama_out = np.zeros(1, dtype=np.float32)
    ck_out_ptr = ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    llama_out_ptr = llama_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call both
    libck.ck_test_gemv_q5_0_q8_0(w_ptr, input_ptr, ck_out_ptr, 1, K)
    libllama.test_gemv_q5_0(w_ptr, input_ptr, llama_out_ptr, K)

    print(f"  Weights: scale=1.0, values=[1]*32")
    print(f"  Input:   FP32 [1.0]*32")
    print(f"  CK result:    {ck_out[0]:.6f}")
    print(f"  llama result: {llama_out[0]:.6f}")
    print(f"  Difference:   {abs(ck_out[0] - llama_out[0]):.6e}")

    test1_pass = abs(ck_out[0] - llama_out[0]) < 1e-3
    print(f"  PASS: {'YES' if test1_pass else 'NO'}")

    # Test 2: Random values
    print("\nTest 2: Random values")
    np.random.seed(42)

    # Generate random Q5_0 weights
    w_scale = np.float32(np.random.uniform(0.01, 0.1))
    w_values = np.random.randint(-16, 16, size=32).astype(np.float32)
    weights = create_simple_q5_0_block(w_scale, w_values.tolist())
    w_ptr = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)

    # Random FP32 input
    input_f32 = np.random.randn(K).astype(np.float32)
    input_ptr = input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call both
    ck_out[0] = 0.0
    llama_out[0] = 0.0
    libck.ck_test_gemv_q5_0_q8_0(w_ptr, input_ptr, ck_out_ptr, 1, K)
    libllama.test_gemv_q5_0(w_ptr, input_ptr, llama_out_ptr, K)

    print(f"  Weights: scale={w_scale:.4f}, values=random[-16,15]")
    print(f"  Input:   FP32 random")
    print(f"  CK result:    {ck_out[0]:.6f}")
    print(f"  llama result: {llama_out[0]:.6f}")
    print(f"  Difference:   {abs(ck_out[0] - llama_out[0]):.6e}")

    test2_pass = abs(ck_out[0] - llama_out[0]) < 0.02  # Looser tolerance for random
    print(f"  PASS: {'YES' if test2_pass else 'NO'}")

    # Test 3: Larger dimension (896 elements = 28 blocks)
    print("\nTest 3: Larger dimension (896 elements)")
    K = 896
    n_blocks = K // QK5_0

    # Generate multi-block weights using same random pattern as comprehensive test
    np.random.seed(42)
    weights = bytearray()
    for _ in range(n_blocks):
        d = np.float32(np.random.uniform(0.01, 0.1))
        weights.extend(fp16_to_bytes(d))
        qh = np.random.randint(0, 256, size=4, dtype=np.uint8)
        weights.extend(qh.tobytes())
        qs = np.random.randint(0, 256, size=16, dtype=np.uint8)
        weights.extend(qs.tobytes())

    weights = bytes(weights)
    w_ptr = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)

    # Random input
    input_f32 = np.random.randn(K).astype(np.float32)
    input_ptr = input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    ck_out = np.zeros(1, dtype=np.float32)
    llama_out = np.zeros(1, dtype=np.float32)
    ck_out_ptr = ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    llama_out_ptr = llama_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    libck.ck_test_gemv_q5_0_q8_0(w_ptr, input_ptr, ck_out_ptr, 1, K)
    libllama.test_gemv_q5_0(w_ptr, input_ptr, llama_out_ptr, K)

    print(f"  Dimensions: M=1, K={K} ({n_blocks} blocks)")
    print(f"  CK result:    {ck_out[0]:.6f}")
    print(f"  llama result: {llama_out[0]:.6f}")
    print(f"  Difference:   {abs(ck_out[0] - llama_out[0]):.6e}")

    test3_pass = abs(ck_out[0] - llama_out[0]) < 0.1
    print(f"  PASS: {'YES' if test3_pass else 'NO'}")

    # Test 4: Many random tests
    print("\nTest 4: 100 random tests")
    K = 256  # 8 blocks
    n_blocks = K // QK5_0
    max_diff = 0.0
    failures = 0

    for trial in range(100):
        np.random.seed(trial + 1000)

        # Generate weights
        weights = bytearray()
        for _ in range(n_blocks):
            d = np.float32(np.random.uniform(0.01, 0.1))
            weights.extend(fp16_to_bytes(d))
            qh = np.random.randint(0, 256, size=4, dtype=np.uint8)
            weights.extend(qh.tobytes())
            qs = np.random.randint(0, 256, size=16, dtype=np.uint8)
            weights.extend(qs.tobytes())

        w_ptr = (ctypes.c_uint8 * len(weights)).from_buffer_copy(bytes(weights))

        # Generate input
        input_f32 = np.random.randn(K).astype(np.float32)
        input_ptr = input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ck_out = np.zeros(1, dtype=np.float32)
        llama_out = np.zeros(1, dtype=np.float32)
        ck_out_ptr = ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        llama_out_ptr = llama_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        libck.ck_test_gemv_q5_0_q8_0(w_ptr, input_ptr, ck_out_ptr, 1, K)
        libllama.test_gemv_q5_0(w_ptr, input_ptr, llama_out_ptr, K)

        diff = abs(ck_out[0] - llama_out[0])
        if diff > max_diff:
            max_diff = diff

        if diff > 0.1:
            failures += 1
            if failures <= 3:
                print(f"  Trial {trial}: CK={ck_out[0]:.6f}, llama={llama_out[0]:.6f}, diff={diff:.6e}")

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Failures (diff > 0.1): {failures}/100")
    test4_pass = failures == 0
    print(f"  PASS: {'YES' if test4_pass else 'NO'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print(f"  Test 1 (simple values):    {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (random values):    {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Test 3 (larger dimension): {'PASS' if test3_pass else 'FAIL'}")
    print(f"  Test 4 (100 random):       {'PASS' if test4_pass else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    return all_pass

if __name__ == "__main__":
    test_single_block_parity()
