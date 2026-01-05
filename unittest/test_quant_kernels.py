"""
Quantized Kernel Unit Tests

Tests all quantization formats:
- Q4_0: Simple 4-bit (32 weights/block)
- Q4_K: K-quant 4-bit with nested scales (256 weights/block)
- Q8_0: Simple 8-bit (32 weights/block)
- F16: IEEE half-precision

Tests both forward (GEMV/GEMM) and backward passes.
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
from test_utils import print_system_info

# Load the library
try:
    lib = load_lib("libckernel_quant.so", "libckernel_engine.so")
except Exception as e:
    print(f"Warning: Could not load quantization library: {e}")
    print("Run 'make libckernel_quant.so' first")
    sys.exit(0)

# Constants
QK4_0 = 32          # Q4_0 block size
QK5_0 = 32          # Q5_0 block size
QK5_1 = 32          # Q5_1 block size
QK8_0 = 32          # Q8_0 block size
QK_K = 256          # Q4_K block size
BLOCK_Q4_0_SIZE = 18
BLOCK_Q5_0_SIZE = 22  # 2 (scale) + 4 (high bits) + 16 (low 4-bits)
BLOCK_Q5_1_SIZE = 24  # 2 (scale) + 2 (min) + 4 (high bits) + 16 (low 4-bits)
BLOCK_Q8_0_SIZE = 34
BLOCK_Q4_K_SIZE = 144


# ============================================================================
# FP16 Utilities
# ============================================================================

def fp16_to_fp32(h: int) -> float:
    """Convert FP16 (uint16) to FP32."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0:
        if mant == 0:
            return (-1.0 if sign else 1.0) * 0.0
        return (-1.0 if sign else 1.0) * (mant / 1024.0) * (2.0 ** -14)
    elif exp == 31:
        if mant == 0:
            return float('-inf') if sign else float('inf')
        return float('nan')
    else:
        return (-1.0 if sign else 1.0) * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


# ============================================================================
# Q4_0 Reference Implementation
# ============================================================================

def create_random_q4_0_block() -> bytes:
    """Create a random Q4_0 block for testing."""
    d = np.random.uniform(0.01, 0.5)
    d_bits = np.float16(d).view(np.uint16)
    qs = np.random.randint(0, 256, QK4_0 // 2, dtype=np.uint8).tobytes()
    return struct.pack('<H', d_bits) + qs


def dequant_q4_0_block_ref(block_data: bytes) -> np.ndarray:
    """Reference dequantization of Q4_0 block."""
    d_bits = struct.unpack('<H', block_data[0:2])[0]
    d = fp16_to_fp32(d_bits)
    qs = block_data[2:18]

    output = np.zeros(QK4_0, dtype=np.float32)
    for i in range(QK4_0 // 2):
        packed = qs[i]
        q0 = (packed & 0x0F) - 8
        q1 = (packed >> 4) - 8
        output[2*i + 0] = d * q0
        output[2*i + 1] = d * q1
    return output


def gemv_q4_0_ref(W_blocks: bytes, x: np.ndarray, M: int, K: int) -> np.ndarray:
    """Reference Q4_0 GEMV."""
    blocks_per_row = K // QK4_0
    y = np.zeros(M, dtype=np.float32)

    for row in range(M):
        for b in range(blocks_per_row):
            offset = (row * blocks_per_row + b) * BLOCK_Q4_0_SIZE
            block_data = W_blocks[offset:offset + BLOCK_Q4_0_SIZE]
            w_fp32 = dequant_q4_0_block_ref(block_data)
            y[row] += np.dot(w_fp32, x[b * QK4_0:(b + 1) * QK4_0])
    return y


# ============================================================================
# Q5_0 Reference Implementation
# ============================================================================

def create_random_q5_0_block() -> bytes:
    """Create a random Q5_0 block for testing."""
    d = np.random.uniform(0.01, 0.5)
    d_bits = np.float16(d).view(np.uint16)
    # High bits: 4 bytes = 32 bits (one per weight)
    qh = np.random.randint(0, 2**32, dtype=np.uint32)
    qh_bytes = qh.tobytes()
    # Low 4-bits: 16 bytes (2 weights per byte)
    qs = np.random.randint(0, 256, QK5_0 // 2, dtype=np.uint8).tobytes()
    return struct.pack('<H', d_bits) + qh_bytes + qs


def dequant_q5_0_block_ref(block_data: bytes) -> np.ndarray:
    """Reference dequantization of Q5_0 block."""
    d_bits = struct.unpack('<H', block_data[0:2])[0]
    d = fp16_to_fp32(d_bits)
    qh = struct.unpack('<I', block_data[2:6])[0]
    qs = block_data[6:22]

    output = np.zeros(QK5_0, dtype=np.float32)
    for i in range(QK5_0 // 2):
        packed = qs[i]
        # Extract low 4 bits
        lo0 = packed & 0x0F
        lo1 = packed >> 4
        # Extract high bits
        hi0 = ((qh >> (2 * i + 0)) & 1) << 4
        hi1 = ((qh >> (2 * i + 1)) & 1) << 4
        # Combine: 5-bit signed value (-16 to +15)
        q0 = (lo0 | hi0) - 16
        q1 = (lo1 | hi1) - 16
        output[2*i + 0] = d * q0
        output[2*i + 1] = d * q1
    return output


def gemv_q5_0_ref(W_blocks: bytes, x: np.ndarray, M: int, K: int) -> np.ndarray:
    """Reference Q5_0 GEMV."""
    blocks_per_row = K // QK5_0
    y = np.zeros(M, dtype=np.float32)

    for row in range(M):
        for b in range(blocks_per_row):
            offset = (row * blocks_per_row + b) * BLOCK_Q5_0_SIZE
            block_data = W_blocks[offset:offset + BLOCK_Q5_0_SIZE]
            w_fp32 = dequant_q5_0_block_ref(block_data)
            y[row] += np.dot(w_fp32, x[b * QK5_0:(b + 1) * QK5_0])
    return y


# ============================================================================
# Q5_1 Reference Implementation
# ============================================================================

def create_random_q5_1_block() -> bytes:
    """Create a random Q5_1 block for testing."""
    d = np.random.uniform(0.01, 0.5)
    m = np.random.uniform(-0.5, 0.0)
    d_bits = np.float16(d).view(np.uint16)
    m_bits = np.float16(m).view(np.uint16)
    # High bits: 4 bytes = 32 bits (one per weight)
    qh = np.random.randint(0, 2**32, dtype=np.uint32)
    qh_bytes = qh.tobytes()
    # Low 4-bits: 16 bytes (2 weights per byte)
    qs = np.random.randint(0, 256, QK5_1 // 2, dtype=np.uint8).tobytes()
    return struct.pack('<H', d_bits) + struct.pack('<H', m_bits) + qh_bytes + qs


def dequant_q5_1_block_ref(block_data: bytes) -> np.ndarray:
    """Reference dequantization of Q5_1 block."""
    d_bits = struct.unpack('<H', block_data[0:2])[0]
    m_bits = struct.unpack('<H', block_data[2:4])[0]
    d = fp16_to_fp32(d_bits)
    m = fp16_to_fp32(m_bits)
    qh = struct.unpack('<I', block_data[4:8])[0]
    qs = block_data[8:24]

    output = np.zeros(QK5_1, dtype=np.float32)
    for i in range(QK5_1 // 2):
        packed = qs[i]
        # Extract low 4 bits
        lo0 = packed & 0x0F
        lo1 = packed >> 4
        # Extract high bits
        hi0 = ((qh >> (2 * i + 0)) & 1) << 4
        hi1 = ((qh >> (2 * i + 1)) & 1) << 4
        # Combine: 5-bit unsigned value (0 to 31)
        q0 = lo0 | hi0
        q1 = lo1 | hi1
        # Dequantize: w = d * q + m
        output[2*i + 0] = d * q0 + m
        output[2*i + 1] = d * q1 + m
    return output


def gemv_q5_1_ref(W_blocks: bytes, x: np.ndarray, M: int, K: int) -> np.ndarray:
    """Reference Q5_1 GEMV."""
    blocks_per_row = K // QK5_1
    y = np.zeros(M, dtype=np.float32)

    for row in range(M):
        for b in range(blocks_per_row):
            offset = (row * blocks_per_row + b) * BLOCK_Q5_1_SIZE
            block_data = W_blocks[offset:offset + BLOCK_Q5_1_SIZE]
            w_fp32 = dequant_q5_1_block_ref(block_data)
            y[row] += np.dot(w_fp32, x[b * QK5_1:(b + 1) * QK5_1])
    return y


# ============================================================================
# Q8_0 Reference Implementation
# ============================================================================

def create_random_q8_0_block() -> bytes:
    """Create a random Q8_0 block for testing."""
    d = np.random.uniform(0.01, 0.5)
    d_bits = np.float16(d).view(np.uint16)
    qs = np.random.randint(-128, 128, QK8_0, dtype=np.int8).tobytes()
    return struct.pack('<H', d_bits) + qs


def dequant_q8_0_block_ref(block_data: bytes) -> np.ndarray:
    """Reference dequantization of Q8_0 block."""
    d_bits = struct.unpack('<H', block_data[0:2])[0]
    d = fp16_to_fp32(d_bits)
    qs = np.frombuffer(block_data[2:34], dtype=np.int8)
    return d * qs.astype(np.float32)


def gemv_q8_0_ref(W_blocks: bytes, x: np.ndarray, M: int, K: int) -> np.ndarray:
    """Reference Q8_0 GEMV."""
    blocks_per_row = K // QK8_0
    y = np.zeros(M, dtype=np.float32)

    for row in range(M):
        for b in range(blocks_per_row):
            offset = (row * blocks_per_row + b) * BLOCK_Q8_0_SIZE
            block_data = W_blocks[offset:offset + BLOCK_Q8_0_SIZE]
            w_fp32 = dequant_q8_0_block_ref(block_data)
            y[row] += np.dot(w_fp32, x[b * QK8_0:(b + 1) * QK8_0])
    return y


# ============================================================================
# Q4_K Reference Implementation
# ============================================================================

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


def create_random_q4_k_block() -> bytes:
    """Create a random Q4_K block for testing.

    Packing matches llama.cpp's get_scale_min_k4() layout:
      - bytes 0-3: (sc[0..3] & 0x3F) | ((sc[4..7] high 2 bits) << 6)
      - bytes 4-7: (m[0..3] & 0x3F) | ((m[4..7] high 2 bits) << 6)
      - bytes 8-11: (sc[4..7] low 4 bits) | ((m[4..7] low 4 bits) << 4)
    """
    d = np.random.uniform(0.01, 0.5)
    dmin = np.random.uniform(0.0, 0.1)
    d_bits = np.float16(d).view(np.uint16)
    dmin_bits = np.float16(dmin).view(np.uint16)

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

    qs = np.random.randint(0, 256, 128, dtype=np.uint8).tobytes()
    return struct.pack('<H', d_bits) + struct.pack('<H', dmin_bits) + scales + qs


def dequant_q4_k_block_ref(block_data: bytes) -> np.ndarray:
    """Reference dequantization of Q4_K block.

    This matches llama.cpp's dequantize_row_q4_K exactly:
    - Formula: weight = d * scale * q - dmin * m
    - Layout: 4 iterations of 64 weights each
      - First 32: low nibbles with scale[2*iter], min[2*iter]
      - Next 32: high nibbles with scale[2*iter+1], min[2*iter+1]
    """
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


def gemv_q4_k_ref(W_blocks: bytes, x: np.ndarray, M: int, K: int) -> np.ndarray:
    """Reference Q4_K GEMV."""
    blocks_per_row = K // QK_K
    y = np.zeros(M, dtype=np.float32)

    for row in range(M):
        for b in range(blocks_per_row):
            offset = (row * blocks_per_row + b) * BLOCK_Q4_K_SIZE
            block_data = W_blocks[offset:offset + BLOCK_Q4_K_SIZE]
            w_fp32 = dequant_q4_k_block_ref(block_data)
            y[row] += np.dot(w_fp32, x[b * QK_K:(b + 1) * QK_K])
    return y


# ============================================================================
# F16 Reference Implementation
# ============================================================================

def gemv_f16_ref(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference F16 GEMV (W is uint16 F16 format)."""
    M, K = W.shape
    y = np.zeros(M, dtype=np.float32)
    for row in range(M):
        for k in range(K):
            w = fp16_to_fp32(W[row, k])
            y[row] += w * x[k]
    return y


# ============================================================================
# Test Functions
# ============================================================================

def test_dequant_q4_0():
    """Test Q4_0 dequantization accuracy."""
    np.random.seed(42)
    block_data = create_random_q4_0_block()
    ref_output = dequant_q4_0_block_ref(block_data)

    try:
        lib.dequant_q4_0_row.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
        ]
        lib.dequant_q4_0_row.restype = None

        c_output = np.zeros(QK4_0, dtype=np.float32)
        block_arr = np.frombuffer(block_data, dtype=np.uint8)
        lib.dequant_q4_0_row(
            block_arr.ctypes.data_as(ctypes.c_void_p),
            c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(QK4_0)
        )
        diff = np.max(np.abs(c_output - ref_output))
        return diff <= 1e-5, diff
    except Exception as e:
        return False, str(e)


def test_gemv_q4_0():
    """Test Q4_0 GEMV accuracy."""
    np.random.seed(42)
    M, K = 32, 64
    blocks = b''.join([create_random_q4_0_block() for _ in range(M * K // QK4_0)])
    x = np.random.randn(K).astype(np.float32)
    ref_y = gemv_q4_0_ref(blocks, x, M, K)

    try:
        lib.gemv_q4_0.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
        ]
        lib.gemv_q4_0.restype = None

        c_y = np.zeros(M, dtype=np.float32)
        blocks_arr = np.frombuffer(blocks, dtype=np.uint8)
        lib.gemv_q4_0(
            c_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            blocks_arr.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M), ctypes.c_int(K)
        )
        diff = np.max(np.abs(c_y - ref_y))
        return diff <= 1e-3, diff
    except Exception as e:
        return False, str(e)


def test_dequant_q5_0():
    """Test Q5_0 dequantization accuracy."""
    np.random.seed(42)
    block_data = create_random_q5_0_block()
    ref_output = dequant_q5_0_block_ref(block_data)

    try:
        lib.dequant_q5_0_row.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
        ]
        lib.dequant_q5_0_row.restype = None

        c_output = np.zeros(QK5_0, dtype=np.float32)
        block_arr = np.frombuffer(block_data, dtype=np.uint8)
        lib.dequant_q5_0_row(
            block_arr.ctypes.data_as(ctypes.c_void_p),
            c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(QK5_0)
        )
        diff = np.max(np.abs(c_output - ref_output))
        return diff <= 1e-5, diff
    except Exception as e:
        return False, str(e)


def test_gemv_q5_0():
    """Test Q5_0 GEMV accuracy."""
    np.random.seed(42)
    M, K = 32, 64
    blocks = b''.join([create_random_q5_0_block() for _ in range(M * K // QK5_0)])
    x = np.random.randn(K).astype(np.float32)
    ref_y = gemv_q5_0_ref(blocks, x, M, K)

    try:
        lib.gemv_q5_0.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
        ]
        lib.gemv_q5_0.restype = None

        c_y = np.zeros(M, dtype=np.float32)
        blocks_arr = np.frombuffer(blocks, dtype=np.uint8)
        lib.gemv_q5_0(
            c_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            blocks_arr.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M), ctypes.c_int(K)
        )
        diff = np.max(np.abs(c_y - ref_y))
        return diff <= 1e-3, diff
    except Exception as e:
        return False, str(e)


def test_dequant_q5_1():
    """Test Q5_1 dequantization accuracy."""
    np.random.seed(42)
    block_data = create_random_q5_1_block()
    ref_output = dequant_q5_1_block_ref(block_data)

    try:
        lib.dequant_q5_1_row.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
        ]
        lib.dequant_q5_1_row.restype = None

        c_output = np.zeros(QK5_1, dtype=np.float32)
        block_arr = np.frombuffer(block_data, dtype=np.uint8)
        lib.dequant_q5_1_row(
            block_arr.ctypes.data_as(ctypes.c_void_p),
            c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(QK5_1)
        )
        diff = np.max(np.abs(c_output - ref_output))
        return diff <= 1e-5, diff
    except Exception as e:
        return False, str(e)


def test_dequant_q8_0():
    """Test Q8_0 dequantization accuracy."""
    np.random.seed(42)
    block_data = create_random_q8_0_block()
    ref_output = dequant_q8_0_block_ref(block_data)

    try:
        lib.dequant_q8_0_row.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
        ]
        lib.dequant_q8_0_row.restype = None

        c_output = np.zeros(QK8_0, dtype=np.float32)
        block_arr = np.frombuffer(block_data, dtype=np.uint8)
        lib.dequant_q8_0_row(
            block_arr.ctypes.data_as(ctypes.c_void_p),
            c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(QK8_0)
        )
        diff = np.max(np.abs(c_output - ref_output))
        return diff <= 1e-5, diff
    except Exception as e:
        return False, str(e)


def test_gemv_q8_0():
    """Test Q8_0 GEMV accuracy."""
    np.random.seed(42)
    M, K = 32, 64
    blocks = b''.join([create_random_q8_0_block() for _ in range(M * K // QK8_0)])
    x = np.random.randn(K).astype(np.float32)
    ref_y = gemv_q8_0_ref(blocks, x, M, K)

    try:
        lib.gemv_q8_0.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
        ]
        lib.gemv_q8_0.restype = None

        c_y = np.zeros(M, dtype=np.float32)
        blocks_arr = np.frombuffer(blocks, dtype=np.uint8)
        lib.gemv_q8_0(
            c_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            blocks_arr.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M), ctypes.c_int(K)
        )
        diff = np.max(np.abs(c_y - ref_y))
        return diff <= 1e-3, diff
    except Exception as e:
        return False, str(e)


def test_dequant_q4_k():
    """Test Q4_K dequantization accuracy."""
    np.random.seed(42)
    block_data = create_random_q4_k_block()
    ref_output = dequant_q4_k_block_ref(block_data)

    try:
        lib.dequant_q4_k_row.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
        ]
        lib.dequant_q4_k_row.restype = None

        c_output = np.zeros(QK_K, dtype=np.float32)
        block_arr = np.frombuffer(block_data, dtype=np.uint8)
        lib.dequant_q4_k_row(
            block_arr.ctypes.data_as(ctypes.c_void_p),
            c_output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(QK_K)
        )
        diff = np.max(np.abs(c_output - ref_output))
        return diff <= 1e-5, diff
    except Exception as e:
        return False, str(e)


def test_gemv_q4_k():
    """Test Q4_K GEMV accuracy."""
    np.random.seed(42)
    M, K = 64, 256
    blocks = b''.join([create_random_q4_k_block() for _ in range(M)])
    x = np.random.randn(K).astype(np.float32)
    ref_y = gemv_q4_k_ref(blocks, x, M, K)

    try:
        lib.gemv_q4_k.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
        ]
        lib.gemv_q4_k.restype = None

        c_y = np.zeros(M, dtype=np.float32)
        blocks_arr = np.frombuffer(blocks, dtype=np.uint8)
        lib.gemv_q4_k(
            c_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            blocks_arr.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M), ctypes.c_int(K)
        )
        diff = np.max(np.abs(c_y - ref_y))
        return diff <= 1e-3, diff
    except Exception as e:
        return False, str(e)


def test_gemv_f16():
    """Test F16 GEMV accuracy."""
    np.random.seed(42)
    M, K = 32, 64

    # Create F16 weights
    W_fp32 = np.random.randn(M, K).astype(np.float32)
    W_f16 = W_fp32.astype(np.float16).view(np.uint16)
    x = np.random.randn(K).astype(np.float32)

    # Reference using numpy's float16
    ref_y = (W_fp32.astype(np.float16).astype(np.float32) @ x)

    try:
        lib.gemv_f16.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
        ]
        lib.gemv_f16.restype = None

        c_y = np.zeros(M, dtype=np.float32)
        lib.gemv_f16(
            c_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            W_f16.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M), ctypes.c_int(K)
        )
        diff = np.max(np.abs(c_y - ref_y))
        return diff <= 1e-3, diff
    except Exception as e:
        return False, str(e)


def test_backward_q4_k():
    """Test Q4_K backward pass accuracy."""
    np.random.seed(42)
    M, K = 64, 256
    blocks = b''.join([create_random_q4_k_block() for _ in range(M)])
    dY = np.random.randn(M).astype(np.float32)

    # Reference: dX = W^T @ dY (dequantize W, then transpose matmul)
    W_fp32 = np.zeros((M, K), dtype=np.float32)
    for row in range(M):
        offset = row * BLOCK_Q4_K_SIZE
        block_data = blocks[offset:offset + BLOCK_Q4_K_SIZE]
        W_fp32[row, :] = dequant_q4_k_block_ref(block_data)
    ref_dX = W_fp32.T @ dY

    try:
        lib.gemv_q4_k_backward.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
        ]
        lib.gemv_q4_k_backward.restype = None

        c_dX = np.zeros(K, dtype=np.float32)
        blocks_arr = np.frombuffer(blocks, dtype=np.uint8)
        lib.gemv_q4_k_backward(
            c_dX.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            blocks_arr.ctypes.data_as(ctypes.c_void_p),
            dY.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(M), ctypes.c_int(K)
        )
        diff = np.max(np.abs(c_dX - ref_dX))
        return diff <= 1e-2, diff  # Relaxed tolerance for backward
    except Exception as e:
        return False, str(e)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print_system_info()

    print("\n" + "=" * 70)
    print("  Quantized Kernel Unit Tests")
    print("=" * 70)

    tests = [
        ("Q4_0 Dequantization", test_dequant_q4_0),
        ("Q4_0 GEMV Forward", test_gemv_q4_0),
        ("Q5_0 Dequantization", test_dequant_q5_0),
        ("Q5_0 GEMV Forward", test_gemv_q5_0),
        ("Q5_1 Dequantization", test_dequant_q5_1),
        ("Q8_0 Dequantization", test_dequant_q8_0),
        ("Q8_0 GEMV Forward", test_gemv_q8_0),
        ("Q4_K Dequantization", test_dequant_q4_k),
        ("Q4_K GEMV Forward", test_gemv_q4_k),
        ("Q4_K GEMV Backward", test_backward_q4_k),
        ("F16 GEMV Forward", test_gemv_f16),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        success, result = test_fn()
        if success:
            print(f"  [PASS] max_diff = {result:.2e}")
            passed += 1
        else:
            print(f"  [FAIL] {result}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
