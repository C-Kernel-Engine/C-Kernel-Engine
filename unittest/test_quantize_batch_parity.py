#!/usr/bin/env python3
"""
Test quantize_batch_q8_0 and quantize_batch_q8_k parity with row-by-row quantization.

This validates that:
1. quantize_batch_q8_0(x, y, num_rows, k) == loop { quantize_row_q8_0(x[i], y[i], k) }
2. quantize_batch_q8_k(x, y, num_rows, k) == loop { quantize_row_q8_k(x[i], y[i], k) }

Usage:
    python unittest/test_quantize_batch_parity.py
"""

import ctypes
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# Q8_0 block: 2 (FP16 d) + 32 (int8 qs) = 34 bytes
QK8_0 = 32
BLOCK_Q8_0_SIZE = 34

# Q8_K block: 292 bytes per 256 elements
QK8_K = 256
BLOCK_Q8_K_SIZE = 292


def load_library():
    """Load CK library."""
    lib_path = BUILD / "libckernel_engine.so"
    if not lib_path.exists():
        print(f"{RED}Error: {lib_path} not found{RESET}")
        sys.exit(1)
    return ctypes.CDLL(str(lib_path))


def test_quantize_batch_q8_0(lib, num_rows=8, k=896, seed=42):
    """Test quantize_batch_q8_0 vs row-by-row quantize_row_q8_0."""
    print(f"\n{'='*60}")
    print(f"Testing quantize_batch_q8_0: {num_rows} rows x {k} elements")
    print(f"{'='*60}")

    np.random.seed(seed)

    # Create random FP32 input
    x = np.random.randn(num_rows, k).astype(np.float32)

    # Calculate output size
    nb = k // QK8_0
    out_bytes_per_row = nb * BLOCK_Q8_0_SIZE
    total_out_bytes = num_rows * out_bytes_per_row

    # Allocate outputs
    y_batch = np.zeros(total_out_bytes, dtype=np.uint8)
    y_rowwise = np.zeros(total_out_bytes, dtype=np.uint8)

    # Setup function signatures
    lib.quantize_batch_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.quantize_row_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int
    ]

    # Run batch quantize
    lib.quantize_batch_q8_0(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_batch.ctypes.data_as(ctypes.c_void_p),
        num_rows,
        k
    )

    # Run row-by-row quantize
    for row in range(num_rows):
        in_ptr = (x[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        out_ptr = ctypes.cast(
            y_rowwise.ctypes.data + row * out_bytes_per_row,
            ctypes.c_void_p
        )
        lib.quantize_row_q8_0(in_ptr, out_ptr, k)

    # Compare outputs
    if np.array_equal(y_batch, y_rowwise):
        print(f"{GREEN}PASS{RESET}: Batch output matches row-by-row output exactly")
        return True
    else:
        diff_count = np.sum(y_batch != y_rowwise)
        print(f"{RED}FAIL{RESET}: {diff_count}/{len(y_batch)} bytes differ")
        # Show first difference
        for i in range(len(y_batch)):
            if y_batch[i] != y_rowwise[i]:
                print(f"  First diff at byte {i}: batch={y_batch[i]}, rowwise={y_rowwise[i]}")
                break
        return False


def test_quantize_batch_q8_k(lib, num_rows=8, k=896, seed=42):
    """Test quantize_batch_q8_k vs row-by-row quantize_row_q8_k."""
    print(f"\n{'='*60}")
    print(f"Testing quantize_batch_q8_k: {num_rows} rows x {k} elements")
    print(f"{'='*60}")

    # K must be multiple of 256 for Q8_K
    if k % QK8_K != 0:
        k_adjusted = (k // QK8_K + 1) * QK8_K
        print(f"  Adjusting K from {k} to {k_adjusted} (must be multiple of 256)")
        k = k_adjusted

    np.random.seed(seed)

    # Create random FP32 input
    x = np.random.randn(num_rows, k).astype(np.float32)

    # Calculate output size
    nb = k // QK8_K
    out_bytes_per_row = nb * BLOCK_Q8_K_SIZE
    total_out_bytes = num_rows * out_bytes_per_row

    # Allocate outputs
    y_batch = np.zeros(total_out_bytes, dtype=np.uint8)
    y_rowwise = np.zeros(total_out_bytes, dtype=np.uint8)

    # Setup function signatures
    lib.quantize_batch_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.quantize_row_q8_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int
    ]

    # Run batch quantize
    lib.quantize_batch_q8_k(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_batch.ctypes.data_as(ctypes.c_void_p),
        num_rows,
        k
    )

    # Run row-by-row quantize
    for row in range(num_rows):
        in_ptr = (x[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        out_ptr = ctypes.cast(
            y_rowwise.ctypes.data + row * out_bytes_per_row,
            ctypes.c_void_p
        )
        lib.quantize_row_q8_k(in_ptr, out_ptr, k)

    # Compare outputs
    if np.array_equal(y_batch, y_rowwise):
        print(f"{GREEN}PASS{RESET}: Batch output matches row-by-row output exactly")
        return True
    else:
        diff_count = np.sum(y_batch != y_rowwise)
        print(f"{RED}FAIL{RESET}: {diff_count}/{len(y_batch)} bytes differ")
        return False


def main():
    print("=" * 60)
    print("Batch Quantize Parity Tests")
    print("=" * 60)

    lib = load_library()

    results = []

    # Test Q8_0 batch
    results.append(("quantize_batch_q8_0 (8x896)", test_quantize_batch_q8_0(lib, 8, 896)))
    results.append(("quantize_batch_q8_0 (1x896)", test_quantize_batch_q8_0(lib, 1, 896)))
    results.append(("quantize_batch_q8_0 (32x4096)", test_quantize_batch_q8_0(lib, 32, 4096)))

    # Test Q8_K batch
    results.append(("quantize_batch_q8_k (8x1024)", test_quantize_batch_q8_k(lib, 8, 1024)))
    results.append(("quantize_batch_q8_k (1x256)", test_quantize_batch_q8_k(lib, 1, 256)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
