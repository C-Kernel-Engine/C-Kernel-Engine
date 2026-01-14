#!/usr/bin/env python3
"""
Unit Tests for Q6_K x Q8_K Quantized Kernels

Tests:
1. vec_dot_q6_k_q8_k - Single dot product parity with llama.cpp
2. gemv_q6_k_q8_k - GEMV correctness vs FP32 reference
3. gemm_nt_q6_k_q8_k - Full GEMM correctness

Usage:
    python unittest/test_q6k_q8k_parity.py
"""

import ctypes
import numpy as np
import os
import struct
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LIB_PATH = PROJECT_ROOT / "build/libq6k_q8k_test.so"

# Color output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# Block sizes
QK_K = 256
BLOCK_Q6_K_SIZE = 210  # 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
BLOCK_Q8_K_SIZE = 292  # 4 (d) + 256 (qs) + 32 (bsums)


def fp16_to_fp32(h):
    """Convert FP16 (uint16) to FP32"""
    return np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0].astype(np.float32)


def fp32_to_fp16_bytes(f):
    """Convert FP32 to FP16 bytes"""
    h = np.array([f], dtype=np.float16).tobytes()
    return h[0], h[1]


def setup_test_lib():
    """Compile and load the test library"""
    src_files = [
        "src/kernels/gemm_kernels_q6k_q8k.c",
        "src/kernels/gemm_kernels_q6k.c",
        "src/kernels/gemm_kernels_q4k_q8k.c",
        "src/kernels/gemm_kernels_q4k_q8k_avx2.c",
        "src/kernels/gemm_kernels_q4k_sse.c",
        "src/kernels/quantize_row_q8_k_sse.c",
        "src/cpu_features.c",
    ]
    include_dir = PROJECT_ROOT / "include"
    (PROJECT_ROOT / "build").mkdir(exist_ok=True)

    cmd = f"gcc -O3 -march=native -fPIC -shared -I{include_dir} {' '.join(str(PROJECT_ROOT / f) for f in src_files)} -o {LIB_PATH} -lm"
    print(f"Compiling: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"{RED}Compilation failed!{RESET}")
        return None

    return ctypes.CDLL(str(LIB_PATH))


def create_q6k_block(d_scale, scales, values):
    """
    Create a Q6_K block from parameters.

    Q6_K layout:
      - ql[128]: low 4 bits of each of 256 weights
      - qh[64]: high 2 bits of each of 256 weights
      - scales[16]: int8 sub-block scales
      - d[2]: FP16 super-block scale
    """
    assert len(scales) == 16
    assert len(values) == 256

    block = bytearray(BLOCK_Q6_K_SIZE)

    # Pack values into ql and qh
    # Values should be in range 0-63 (6-bit unsigned)
    values = np.clip(values, 0, 63).astype(np.uint8)

    # ql: low 4 bits, packed as pairs
    for i in range(128):
        lo = values[i * 2] & 0x0F
        hi = values[i * 2 + 1] & 0x0F
        block[i] = lo | (hi << 4)

    # qh: high 2 bits, packed as quads
    for i in range(64):
        qh_val = 0
        for j in range(4):
            qh_val |= ((values[i * 4 + j] >> 4) & 0x03) << (j * 2)
        block[128 + i] = qh_val

    # scales
    for i, s in enumerate(scales):
        block[192 + i] = np.uint8(np.int8(s).view(np.uint8))

    # d (FP16)
    d_lo, d_hi = fp32_to_fp16_bytes(d_scale)
    block[208] = d_lo
    block[209] = d_hi

    return bytes(block)


def create_q8k_block(d_scale, values):
    """
    Create a Q8_K block from parameters.

    Q8_K layout:
      - d[4]: FP32 scale
      - qs[256]: int8 values
      - bsums[32]: int16 block sums (16 sums)
    """
    assert len(values) == 256

    block = bytearray(BLOCK_Q8_K_SIZE)

    # d (FP32)
    d_bytes = struct.pack('<f', d_scale)
    block[0:4] = d_bytes

    # qs (int8)
    values = np.clip(values, -128, 127).astype(np.int8)
    block[4:260] = values.tobytes()

    # bsums (int16) - sum of each 16-element sub-block
    bsums = []
    for i in range(16):
        bsum = int(np.sum(values[i*16:(i+1)*16]))
        bsums.append(np.int16(bsum))
    block[260:292] = np.array(bsums, dtype=np.int16).tobytes()

    return bytes(block)


def dequant_q6k_ref(data):
    """Reference Q6_K dequantization"""
    assert len(data) == BLOCK_Q6_K_SIZE

    ql = data[0:128]
    qh = data[128:192]
    scales_raw = data[192:208]
    d_raw = struct.unpack('<H', data[208:210])[0]
    d = fp16_to_fp32(d_raw)

    # Convert scales to signed int8
    scales = np.frombuffer(scales_raw, dtype=np.int8)

    result = np.zeros(256, dtype=np.float32)

    # Unpack and dequantize
    ql_idx = 0
    qh_idx = 0
    sc_idx = 0

    for n in range(QK_K // 128):  # 2 iterations of 128
        for l in range(32):
            is_val = l // 16

            # Extract 6-bit values
            q1 = (ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)
            q2 = (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)
            q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)
            q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)

            # Dequantize (q6 - 32 converts to signed)
            idx_base = n * 128
            result[idx_base + l] = d * scales[sc_idx + is_val] * (q1 - 32)
            result[idx_base + l + 32] = d * scales[sc_idx + is_val + 2] * (q2 - 32)
            result[idx_base + l + 64] = d * scales[sc_idx + is_val + 4] * (q3 - 32)
            result[idx_base + l + 96] = d * scales[sc_idx + is_val + 6] * (q4 - 32)

        ql_idx += 64
        qh_idx += 32
        sc_idx += 8
        ql = data[ql_idx:ql_idx+64]
        qh = data[128 + qh_idx:128 + qh_idx + 32]

    return result


def dot_q6k_q8k_ref(w_block, x_block):
    """Reference dot product Q6_K x Q8_K"""
    # Extract Q6_K values and dequantize
    ql = np.frombuffer(w_block[0:128], dtype=np.uint8)
    qh = np.frombuffer(w_block[128:192], dtype=np.uint8)
    scales = np.frombuffer(w_block[192:208], dtype=np.int8)
    d_w = fp16_to_fp32(struct.unpack('<H', w_block[208:210])[0])

    # Extract Q8_K values
    d_x = struct.unpack('<f', x_block[0:4])[0]
    q8 = np.frombuffer(x_block[4:260], dtype=np.int8)

    d = d_w * d_x
    sumf = 0.0

    q8_idx = 0
    ql_idx = 0
    qh_idx = 0
    sc_idx = 0

    for n in range(QK_K // 128):  # 2 iterations
        ql_slice = ql[ql_idx:ql_idx+64]
        qh_slice = qh[qh_idx:qh_idx+32]
        sc_slice = scales[sc_idx:sc_idx+8]
        q8_slice = q8[q8_idx:q8_idx+128]

        for l in range(32):
            is_val = l // 16

            q1 = int((ql_slice[l] & 0x0F) | (((qh_slice[l] >> 0) & 3) << 4)) - 32
            q2 = int((ql_slice[l + 32] & 0x0F) | (((qh_slice[l] >> 2) & 3) << 4)) - 32
            q3 = int((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) - 32
            q4 = int((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) - 32

            sumf += d * float(sc_slice[is_val + 0]) * float(q1) * float(q8_slice[l + 0])
            sumf += d * float(sc_slice[is_val + 2]) * float(q2) * float(q8_slice[l + 32])
            sumf += d * float(sc_slice[is_val + 4]) * float(q3) * float(q8_slice[l + 64])
            sumf += d * float(sc_slice[is_val + 6]) * float(q4) * float(q8_slice[l + 96])

        q8_idx += 128
        ql_idx += 64
        qh_idx += 32
        sc_idx += 8

    return sumf


def test_vec_dot_q6k_q8k(lib):
    """Test vec_dot_q6_k_q8_k against reference"""
    print(f"\n{YELLOW}Testing vec_dot_q6_k_q8_k...{RESET}")

    lib.vec_dot_q6_k_q8_k.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_void_p
    ]

    np.random.seed(42)

    # Create test data
    K = QK_K  # Single block
    d_w = 0.1
    scales = np.random.randint(-8, 8, size=16).astype(np.int8)
    values_6bit = np.random.randint(0, 64, size=256).astype(np.uint8)

    w_block = create_q6k_block(d_w, scales, values_6bit)

    d_x = 0.05
    values_8bit = np.random.randint(-50, 50, size=256).astype(np.int8)
    x_block = create_q8k_block(d_x, values_8bit)

    # Reference
    ref_result = dot_q6k_q8k_ref(w_block, x_block)

    # CK-Engine
    result = ctypes.c_float(0.0)
    lib.vec_dot_q6_k_q8_k(
        K,
        ctypes.byref(result),
        (ctypes.c_ubyte * len(w_block)).from_buffer_copy(w_block),
        (ctypes.c_ubyte * len(x_block)).from_buffer_copy(x_block)
    )

    diff = abs(result.value - ref_result)
    rel_err = diff / (abs(ref_result) + 1e-10)

    print(f"  Reference: {ref_result:.6f}")
    print(f"  CK-Engine: {result.value:.6f}")
    print(f"  Abs Diff:  {diff:.6e}")
    print(f"  Rel Error: {rel_err:.6e}")

    if rel_err < 1e-5:
        print(f"  {GREEN}PASS{RESET}")
        return True
    else:
        print(f"  {RED}FAIL{RESET}")
        return False


def test_gemv_q6k_q8k(lib):
    """Test gemv_q6_k_q8_k against FP32 reference"""
    print(f"\n{YELLOW}Testing gemv_q6_k_q8_k...{RESET}")

    lib.gemv_q6_k_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]

    np.random.seed(123)

    M = 32  # Output dimension
    K = QK_K  # Input dimension (1 block)

    # Create weight matrix (M rows)
    W_blocks = []
    for _ in range(M):
        d_w = np.random.rand() * 0.1
        scales = np.random.randint(-8, 8, size=16).astype(np.int8)
        values = np.random.randint(0, 64, size=256).astype(np.uint8)
        W_blocks.append(create_q6k_block(d_w, scales, values))

    W_buffer = b''.join(W_blocks)

    # Create input vector (1 block)
    d_x = 0.1
    x_vals = np.random.randint(-50, 50, size=256).astype(np.int8)
    x_block = create_q8k_block(d_x, x_vals)

    # Reference
    ref_y = np.zeros(M, dtype=np.float32)
    for row in range(M):
        w_block = W_blocks[row]
        ref_y[row] = dot_q6k_q8k_ref(w_block, x_block)

    # CK-Engine
    y = np.zeros(M, dtype=np.float32)
    lib.gemv_q6_k_q8_k(
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        (ctypes.c_ubyte * len(W_buffer)).from_buffer_copy(W_buffer),
        (ctypes.c_ubyte * len(x_block)).from_buffer_copy(x_block),
        M, K
    )

    mse = np.mean((y - ref_y) ** 2)
    max_diff = np.max(np.abs(y - ref_y))

    print(f"  MSE:      {mse:.6e}")
    print(f"  Max Diff: {max_diff:.6e}")

    if max_diff < 1e-4:
        print(f"  {GREEN}PASS{RESET}")
        return True
    else:
        print(f"  {RED}FAIL{RESET}")
        return False


def test_gemm_nt_q6k_q8k(lib):
    """Test gemm_nt_q6_k_q8_k for full batched inference"""
    print(f"\n{YELLOW}Testing gemm_nt_q6_k_q8_k...{RESET}")

    lib.gemm_nt_q6_k_q8_k.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]

    np.random.seed(456)

    M = 4   # Batch size (tokens)
    N = 16  # Output dimension
    K = QK_K  # Input dimension (1 block)

    # Create weight matrix (N rows)
    W_blocks = []
    for _ in range(N):
        d_w = np.random.rand() * 0.1
        scales = np.random.randint(-8, 8, size=16).astype(np.int8)
        values = np.random.randint(0, 64, size=256).astype(np.uint8)
        W_blocks.append(create_q6k_block(d_w, scales, values))

    W_buffer = b''.join(W_blocks)

    # Create input matrix (M rows, each is 1 Q8_K block)
    X_blocks = []
    for _ in range(M):
        d_x = np.random.rand() * 0.1
        x_vals = np.random.randint(-50, 50, size=256).astype(np.int8)
        X_blocks.append(create_q8k_block(d_x, x_vals))

    X_buffer = b''.join(X_blocks)

    # Reference
    ref_C = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            ref_C[m, n] = dot_q6k_q8k_ref(W_blocks[n], X_blocks[m])

    # CK-Engine
    C = np.zeros((M, N), dtype=np.float32)
    lib.gemm_nt_q6_k_q8_k(
        (ctypes.c_ubyte * len(X_buffer)).from_buffer_copy(X_buffer),
        (ctypes.c_ubyte * len(W_buffer)).from_buffer_copy(W_buffer),
        None,  # No bias
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, N, K
    )

    mse = np.mean((C - ref_C) ** 2)
    max_diff = np.max(np.abs(C - ref_C))

    print(f"  Shape:    ({M}, {N})")
    print(f"  MSE:      {mse:.6e}")
    print(f"  Max Diff: {max_diff:.6e}")

    if max_diff < 1e-4:
        print(f"  {GREEN}PASS{RESET}")
        return True
    else:
        print(f"  {RED}FAIL{RESET}")
        return False


def test_vs_fp32_gemv(lib):
    """Compare Q6_K x Q8_K GEMV vs FP32 GEMV for numerical accuracy"""
    print(f"\n{YELLOW}Testing Q6_K x Q8_K vs FP32 GEMV accuracy...{RESET}")

    lib.gemv_q6_k_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    lib.gemv_q6_k.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
    ]
    lib.quantize_row_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int
    ]

    np.random.seed(789)

    M = 64
    K = QK_K

    # Create random FP32 input
    x_fp32 = np.random.randn(K).astype(np.float32) * 0.5

    # Quantize to Q8_K
    x_q8k = (ctypes.c_ubyte * BLOCK_Q8_K_SIZE)()
    lib.quantize_row_q8_k(
        x_fp32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_q8k,
        K
    )

    # Create Q6_K weights
    W_blocks = []
    for _ in range(M):
        d_w = np.random.rand() * 0.1
        scales = np.random.randint(-8, 8, size=16).astype(np.int8)
        values = np.random.randint(0, 64, size=256).astype(np.uint8)
        W_blocks.append(create_q6k_block(d_w, scales, values))

    W_buffer = b''.join(W_blocks)

    # Run FP32 GEMV
    y_fp32 = np.zeros(M, dtype=np.float32)
    lib.gemv_q6_k(
        y_fp32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        (ctypes.c_ubyte * len(W_buffer)).from_buffer_copy(W_buffer),
        x_fp32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K
    )

    # Run INT8 GEMV
    y_int8 = np.zeros(M, dtype=np.float32)
    lib.gemv_q6_k_q8_k(
        y_int8.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        (ctypes.c_ubyte * len(W_buffer)).from_buffer_copy(W_buffer),
        x_q8k,
        M, K
    )

    # Compare
    mse = np.mean((y_fp32 - y_int8) ** 2)
    max_diff = np.max(np.abs(y_fp32 - y_int8))
    corr = np.corrcoef(y_fp32, y_int8)[0, 1]

    print(f"  MSE:         {mse:.6e}")
    print(f"  Max Diff:    {max_diff:.6e}")
    print(f"  Correlation: {corr:.6f}")

    # INT8 should correlate well with FP32, but exact match not expected
    if corr > 0.99:
        print(f"  {GREEN}PASS - High correlation with FP32{RESET}")
        return True
    else:
        print(f"  {RED}FAIL - Low correlation with FP32{RESET}")
        return False


def main():
    print(f"\n{'='*60}")
    print(f"Q6_K x Q8_K Kernel Unit Tests")
    print(f"{'='*60}")

    lib = setup_test_lib()
    if lib is None:
        return 1

    results = []
    results.append(("vec_dot_q6_k_q8_k", test_vec_dot_q6k_q8k(lib)))
    results.append(("gemv_q6_k_q8_k", test_gemv_q6k_q8k(lib)))
    results.append(("gemm_nt_q6_k_q8_k", test_gemm_nt_q6k_q8k(lib)))
    results.append(("vs_fp32_accuracy", test_vs_fp32_gemv(lib)))

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
