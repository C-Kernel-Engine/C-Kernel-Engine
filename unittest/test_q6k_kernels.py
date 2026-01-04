"""
Q6_K Kernel Unit Tests

Tests dequantization and GEMV/GEMM accuracy for Q6_K (GGML k-quant format).
Compares C kernel output against a Python reference implementation.
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
from test_utils import TestReport, TestResult, get_cpu_info, print_system_info

# Load the library
try:
    lib = load_lib("libckernel_quant.so", "libckernel_engine.so")
except Exception as e:
    print(f"Warning: Could not load quantization library: {e}")
    print("Run 'make libckernel_quant.so' first")
    sys.exit(0)

QK_K = 256
BLOCK_Q6_K_SIZE = 210


def fp16_to_fp32(h: int) -> float:
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0:
        if mant == 0:
            return (-1.0 if sign else 1.0) * 0.0
        return (-1.0 if sign else 1.0) * (mant / 1024.0) * (2.0 ** -14)
    if exp == 31:
        if mant == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")
    return (-1.0 if sign else 1.0) * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


def dequant_q6_k_block_ref(block_data: bytes) -> np.ndarray:
    ql = np.frombuffer(block_data, dtype=np.uint8, count=QK_K // 2, offset=0)
    qh = np.frombuffer(block_data, dtype=np.uint8, count=QK_K // 4, offset=QK_K // 2)
    scales = np.frombuffer(block_data, dtype=np.int8, count=QK_K // 16, offset=QK_K // 2 + QK_K // 4)
    d_bits = struct.unpack_from("<H", block_data, QK_K // 2 + QK_K // 4 + QK_K // 16)[0]
    d = fp16_to_fp32(d_bits)

    out = np.zeros(QK_K, dtype=np.float32)
    y = out
    ql_ptr = ql
    qh_ptr = qh
    sc_ptr = scales

    for _ in range(0, QK_K, 128):
        for l in range(32):
            iscale = l // 16
            q1 = ((int(ql_ptr[l + 0]) & 0xF) | (((int(qh_ptr[l]) >> 0) & 3) << 4)) - 32
            q2 = ((int(ql_ptr[l + 32]) & 0xF) | (((int(qh_ptr[l]) >> 2) & 3) << 4)) - 32
            q3 = ((int(ql_ptr[l + 0]) >> 4) | (((int(qh_ptr[l]) >> 4) & 3) << 4)) - 32
            q4 = ((int(ql_ptr[l + 32]) >> 4) | (((int(qh_ptr[l]) >> 6) & 3) << 4)) - 32
            y[l + 0] = d * float(sc_ptr[iscale + 0]) * q1
            y[l + 32] = d * float(sc_ptr[iscale + 2]) * q2
            y[l + 64] = d * float(sc_ptr[iscale + 4]) * q3
            y[l + 96] = d * float(sc_ptr[iscale + 6]) * q4
        y = y[128:]
        ql_ptr = ql_ptr[64:]
        qh_ptr = qh_ptr[32:]
        sc_ptr = sc_ptr[8:]

    return out


def create_random_q6_k_block() -> bytes:
    d = np.random.uniform(0.01, 0.5)
    d_bits = np.float16(d).view(np.uint16)
    ql = np.random.randint(0, 256, size=(QK_K // 2,), dtype=np.uint8)
    qh = np.random.randint(0, 256, size=(QK_K // 4,), dtype=np.uint8)
    scales = np.random.randint(-32, 32, size=(QK_K // 16,), dtype=np.int8)
    return ql.tobytes() + qh.tobytes() + scales.tobytes() + struct.pack("<H", int(d_bits))


def test_dequant_q6_k() -> float:
    np.random.seed(42)
    block_data = create_random_q6_k_block()
    ref = dequant_q6_k_block_ref(block_data)

    lib.dequant_q6_k_row.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.dequant_q6_k_row.restype = None

    c_out = np.zeros(QK_K, dtype=np.float32)
    buf = ctypes.create_string_buffer(block_data, len(block_data))
    lib.dequant_q6_k_row(
        ctypes.cast(buf, ctypes.c_void_p),
        c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(QK_K),
    )

    return float(np.max(np.abs(c_out - ref)))


def dequant_q6_k_row_ref(row_bytes: bytes, k: int) -> np.ndarray:
    nb = k // QK_K
    out = np.zeros(k, dtype=np.float32)
    for b in range(nb):
        block = row_bytes[b * BLOCK_Q6_K_SIZE:(b + 1) * BLOCK_Q6_K_SIZE]
        out[b * QK_K:(b + 1) * QK_K] = dequant_q6_k_block_ref(block)
    return out


def test_gemv_q6_k(m_out: int, k: int) -> float:
    np.random.seed(123)
    x = (np.random.randn(k).astype(np.float32) * 0.1).astype(np.float32)
    blocks_per_row = k // QK_K

    rows = []
    for _ in range(m_out):
        blocks = b"".join(create_random_q6_k_block() for _ in range(blocks_per_row))
        rows.append(blocks)
    W = b"".join(rows)

    w_deq = np.vstack([dequant_q6_k_row_ref(row, k) for row in rows])
    y_ref = w_deq @ x

    lib.gemv_q6_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemv_q6_k.restype = None

    y_out = np.zeros(m_out, dtype=np.float32)
    w_buf = ctypes.create_string_buffer(W, len(W))
    lib.gemv_q6_k(
        y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(w_buf, ctypes.c_void_p),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(m_out),
        ctypes.c_int(k),
    )

    return float(np.max(np.abs(y_out - y_ref)))


def test_gemm_nt_q6_k(m_out: int, k: int) -> float:
    np.random.seed(321)
    x = (np.random.randn(k).astype(np.float32) * 0.1).astype(np.float32)
    blocks_per_row = k // QK_K

    rows = []
    for _ in range(m_out):
        blocks = b"".join(create_random_q6_k_block() for _ in range(blocks_per_row))
        rows.append(blocks)
    W = b"".join(rows)

    w_deq = np.vstack([dequant_q6_k_row_ref(row, k) for row in rows])
    y_ref = w_deq @ x

    lib.gemm_nt_q6_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gemm_nt_q6_k.restype = None

    y_out = np.zeros(m_out, dtype=np.float32)
    w_buf = ctypes.create_string_buffer(W, len(W))
    lib.gemm_nt_q6_k(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(w_buf, ctypes.c_void_p),
        None,
        y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(1),
        ctypes.c_int(m_out),
        ctypes.c_int(k),
    )

    return float(np.max(np.abs(y_out - y_ref)))


if __name__ == "__main__":
    print_system_info()

    report = TestReport(
        test_name="Q6_K Kernels",
        dtype="q6_k",
        shape="block=256",
        cpu_info=get_cpu_info(),
    )

    diff = test_dequant_q6_k()
    report.add_result(TestResult(
        name="dequant_q6_k_row",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
    ))

    diff = test_gemv_q6_k(m_out=32, k=256)
    report.add_result(TestResult(
        name="gemv_q6_k (M=32,K=256)",
        passed=diff <= 5e-4,
        max_diff=diff,
        tolerance=5e-4,
    ))

    diff = test_gemm_nt_q6_k(m_out=32, k=256)
    # gemm_nt accumulates in a different order than the numpy reference, so allow a small drift.
    report.add_result(TestResult(
        name="gemm_nt_q6_k (M=1,N=32,K=256)",
        passed=diff <= 5e-4,
        max_diff=diff,
        tolerance=5e-4,
    ))

    report.print_report()
    if not report.all_passed():
        sys.exit(1)
