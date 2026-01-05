"""
Q4_K Kernel Unit Tests

Tests dequantization and GEMM accuracy for Q4_K (GGML k-quant format).
Compares C kernel output against reference Python implementation.
"""
import ctypes
import sys
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "unittest"
for path in (ROOT, UNITS):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)

# Load the library
try:
    lib = load_lib("libckernel_quant.so", "libckernel_engine.so")
except Exception as e:
    print(f"Warning: Could not load quantization library: {e}")
    print("Run 'make libckernel_quant.so' first")
    sys.exit(0)

# Q4_K constants
QK_K = 256  # Weights per super-block
BLOCK_Q4_K_SIZE = 144  # Bytes per block


# ============================================================================
# Reference Python Implementation
# ============================================================================

def fp16_to_fp32(h: int) -> float:
    """Convert FP16 (uint16) to FP32."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0:
        if mant == 0:
            return (-1.0 if sign else 1.0) * 0.0
        # Denormalized
        return (-1.0 if sign else 1.0) * (mant / 1024.0) * (2.0 ** -14)
    elif exp == 31:
        if mant == 0:
            return float('-inf') if sign else float('inf')
        return float('nan')
    else:
        return (-1.0 if sign else 1.0) * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


def unpack_q4_k_scales(scales: bytes) -> tuple:
    """Unpack 6-bit scales and mins from 12-byte packed array.

    This matches llama.cpp's get_scale_min_k4() function exactly.
    Layout:
      - bytes 0-3: 6-bit scales[0-3] (high 2 bits used for scales[4-7])
      - bytes 4-7: 6-bit mins[0-3] (high 2 bits used for mins[4-7])
      - bytes 8-11: low 4 bits for scales[4-7], high 4 bits for mins[4-7]
    """
    sc = [0] * 8
    m = [0] * 8

    # Direct 6-bit values for indices 0-3
    sc[0] = scales[0] & 0x3F
    sc[1] = scales[1] & 0x3F
    sc[2] = scales[2] & 0x3F
    sc[3] = scales[3] & 0x3F

    m[0] = scales[4] & 0x3F
    m[1] = scales[5] & 0x3F
    m[2] = scales[6] & 0x3F
    m[3] = scales[7] & 0x3F

    # 6-bit values for indices 4-7: low 4 bits from bytes 8-11,
    # high 2 bits from upper bits of bytes 0-3 (scales) and 4-7 (mins)
    sc[4] = (scales[8]  & 0x0F) | ((scales[0] >> 6) << 4)
    sc[5] = (scales[9]  & 0x0F) | ((scales[1] >> 6) << 4)
    sc[6] = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4)
    sc[7] = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4)

    m[4] = (scales[8]  >> 4) | ((scales[4] >> 6) << 4)
    m[5] = (scales[9]  >> 4) | ((scales[5] >> 6) << 4)
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4)
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4)

    return sc, m


def dequant_q4_k_block_ref(block_data: bytes) -> np.ndarray:
    """Reference dequantization of a Q4_K block (256 floats).

    This matches llama.cpp's dequantize_row_q4_K exactly:
    - Formula: weight = d * scale * q - dmin * m
    - Layout: 4 iterations of 64 weights each
      - First 32: low nibbles with scale[2*iter], min[2*iter]
      - Next 32: high nibbles with scale[2*iter+1], min[2*iter+1]
    """
    # Parse block header
    d_bits = struct.unpack('<H', block_data[0:2])[0]
    dmin_bits = struct.unpack('<H', block_data[2:4])[0]
    scales = block_data[4:16]
    qs = block_data[16:144]

    d = fp16_to_fp32(d_bits)
    dmin = fp16_to_fp32(dmin_bits)

    sc, m = unpack_q4_k_scales(scales)

    output = np.zeros(256, dtype=np.float32)

    # llama.cpp layout: 4 iterations of 64 weights each
    for iter in range(4):
        d1 = d * sc[2 * iter]
        m1 = dmin * m[2 * iter]
        d2 = d * sc[2 * iter + 1]
        m2 = dmin * m[2 * iter + 1]

        q_ptr = iter * 32
        y_ptr = iter * 64

        # First 32 weights: low nibbles
        for l in range(32):
            q = qs[q_ptr + l] & 0x0F
            output[y_ptr + l] = d1 * q - m1

        # Next 32 weights: high nibbles
        for l in range(32):
            q = qs[q_ptr + l] >> 4
            output[y_ptr + 32 + l] = d2 * q - m2

    return output


def create_random_q4_k_block() -> bytes:
    """Create a random Q4_K block for testing.

    Packing matches llama.cpp's get_scale_min_k4() layout:
      - bytes 0-3: (sc[0..3] & 0x3F) | ((sc[4..7] high 2 bits) << 6)
      - bytes 4-7: (m[0..3] & 0x3F) | ((m[4..7] high 2 bits) << 6)
      - bytes 8-11: (sc[4..7] low 4 bits) | ((m[4..7] low 4 bits) << 4)
    """
    # Random scale and min (FP16 format)
    d = np.random.uniform(0.01, 0.5)
    dmin = np.random.uniform(0.0, 0.1)

    # Convert to FP16 bits (simplified)
    d_bits = np.float16(d).view(np.uint16)
    dmin_bits = np.float16(dmin).view(np.uint16)

    # Random sub-block scales (6-bit, 0-63)
    sc = np.random.randint(0, 64, 8, dtype=np.uint8)
    m = np.random.randint(0, 64, 8, dtype=np.uint8)

    # Pack according to llama.cpp layout
    scales = bytes([
        (sc[0] & 0x3F) | ((sc[4] >> 4) << 6),  # byte 0
        (sc[1] & 0x3F) | ((sc[5] >> 4) << 6),  # byte 1
        (sc[2] & 0x3F) | ((sc[6] >> 4) << 6),  # byte 2
        (sc[3] & 0x3F) | ((sc[7] >> 4) << 6),  # byte 3
        (m[0] & 0x3F) | ((m[4] >> 4) << 6),    # byte 4
        (m[1] & 0x3F) | ((m[5] >> 4) << 6),    # byte 5
        (m[2] & 0x3F) | ((m[6] >> 4) << 6),    # byte 6
        (m[3] & 0x3F) | ((m[7] >> 4) << 6),    # byte 7
        (sc[4] & 0x0F) | ((m[4] & 0x0F) << 4), # byte 8
        (sc[5] & 0x0F) | ((m[5] & 0x0F) << 4), # byte 9
        (sc[6] & 0x0F) | ((m[6] & 0x0F) << 4), # byte 10
        (sc[7] & 0x0F) | ((m[7] & 0x0F) << 4), # byte 11
    ])

    # Random 4-bit weights
    qs = np.random.randint(0, 256, 128, dtype=np.uint8).tobytes()

    # Pack block
    block = struct.pack('<H', d_bits) + struct.pack('<H', dmin_bits) + scales + qs
    return block


# ============================================================================
# Test Functions
# ============================================================================

def test_dequant_q4_k():
    """Test Q4_K dequantization accuracy."""
    np.random.seed(42)

    # Create test block
    block_data = create_random_q4_k_block()

    # Reference implementation
    ref_output = dequant_q4_k_block_ref(block_data)

    # C kernel
    try:
        lib.dequant_q4_k_row.argtypes = [
            ctypes.c_void_p,   # src
            ctypes.POINTER(ctypes.c_float),  # dst
            ctypes.c_size_t    # n_elements
        ]
        lib.dequant_q4_k_row.restype = None

        c_output = np.zeros(256, dtype=np.float32)
        block_arr = np.frombuffer(block_data, dtype=np.uint8)

        lib.dequant_q4_k_row(
            block_arr.ctypes.data_as(ctypes.c_void_p),
            c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(256)
        )

        diff = np.max(np.abs(c_output - ref_output))
        return diff <= 1e-5, diff
    except Exception as e:
        return False, str(e)


def test_gemv_q4_k():
    """Test Q4_K GEMV accuracy."""
    np.random.seed(42)

    M = 64   # Output size
    K = 256  # Input size (one Q4_K block per row)

    # Create random weights (M x K in Q4_K format)
    blocks = b''.join([create_random_q4_k_block() for _ in range(M)])

    # Random input vector
    x = np.random.randn(K).astype(np.float32)

    # Reference: dequantize all weights, then matmul
    W_fp32 = np.zeros((M, K), dtype=np.float32)
    for row in range(M):
        block_data = blocks[row * BLOCK_Q4_K_SIZE : (row + 1) * BLOCK_Q4_K_SIZE]
        W_fp32[row, :] = dequant_q4_k_block_ref(block_data)

    ref_y = W_fp32 @ x

    # C kernel
    try:
        lib.gemv_q4_k.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # y
            ctypes.c_void_p,                 # W
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.c_int,                    # M
            ctypes.c_int                     # K
        ]
        lib.gemv_q4_k.restype = None

        c_y = np.zeros(M, dtype=np.float32)
        blocks_arr = np.frombuffer(blocks, dtype=np.uint8)

        lib.gemv_q4_k(
            c_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            blocks_arr.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M),
            ctypes.c_int(K)
        )

        diff = np.max(np.abs(c_y - ref_y))
        # For Q4_K (4-bit quantized weights), tolerance of 1e-3 is reasonable
        # due to FP16 scale precision and accumulated rounding
        return diff <= 1e-3, diff
    except Exception as e:
        return False, str(e)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print_system_info()

    print("\n" + "=" * 70)
    print("  Q4_K Kernel Unit Tests")
    print("=" * 70)

    # Test dequantization
    print("\nTest 1: Q4_K Dequantization")
    passed, result = test_dequant_q4_k()
    if passed:
        print(f"  [PASS] max_diff = {result:.2e}")
    else:
        print(f"  [FAIL] {result}")

    # Test GEMV
    print("\nTest 2: Q4_K GEMV")
    passed, result = test_gemv_q4_k()
    if passed:
        print(f"  [PASS] max_diff = {result:.2e}")
    else:
        print(f"  [FAIL] {result}")

    print("\n" + "=" * 70)
