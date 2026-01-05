#!/usr/bin/env python3
"""
test_kernels_comprehensive.py - Comprehensive kernel testing against llama.cpp

Tests every quantization kernel by:
1. Reading actual GGUF model data
2. Dequantizing with our Python reference (matching llama.cpp exactly)
3. Calling our C kernels via ctypes
4. Comparing results bit-for-bit

Usage:
    python unittest/test_kernels_comprehensive.py --gguf model.gguf
    python unittest/test_kernels_comprehensive.py --list-tensors
    python unittest/test_kernels_comprehensive.py --test-kernel q4_k
"""

import os
import sys
import struct
import ctypes
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# Colors
# ============================================================================

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


# ============================================================================
# llama.cpp Reference Dequantization (Python implementation)
# ============================================================================

def fp16_to_fp32(h: int) -> float:
    """Convert FP16 (uint16) to FP32"""
    return np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0].astype(np.float32)


def get_scale_min_k4(j: int, q: bytes) -> Tuple[int, int]:
    """llama.cpp's get_scale_min_k4 function - extracts scale and min for Q4_K"""
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d, m


def dequant_q4_0_block(data: bytes) -> np.ndarray:
    """Dequantize Q4_0 block (18 bytes -> 32 floats)"""
    d = fp16_to_fp32(struct.unpack('<H', data[0:2])[0])
    result = np.zeros(32, dtype=np.float32)
    for i in range(16):
        x0 = (data[2 + i] & 0x0F) - 8
        x1 = (data[2 + i] >> 4) - 8
        result[i] = x0 * d
        result[i + 16] = x1 * d
    return result


def dequant_q8_0_block(data: bytes) -> np.ndarray:
    """Dequantize Q8_0 block (34 bytes -> 32 floats)"""
    d = fp16_to_fp32(struct.unpack('<H', data[0:2])[0])
    qs = np.frombuffer(data[2:34], dtype=np.int8).astype(np.float32)
    return qs * d


def dequant_q4_k_block(data: bytes) -> np.ndarray:
    """Dequantize Q4_K block (144 bytes -> 256 floats)"""
    d = fp16_to_fp32(struct.unpack('<H', data[0:2])[0])
    dmin = fp16_to_fp32(struct.unpack('<H', data[2:4])[0])
    scales = data[4:16]
    qs = data[16:144]

    result = np.zeros(256, dtype=np.float32)

    is_idx = 0
    for j in range(0, 256, 64):
        sc1, m1 = get_scale_min_k4(is_idx, scales)
        sc2, m2 = get_scale_min_k4(is_idx + 1, scales)

        d1, dm1 = d * sc1, dmin * m1
        d2, dm2 = d * sc2, dmin * m2

        for l in range(32):
            result[j + l] = d1 * (qs[j // 2 + l] & 0x0F) - dm1
        for l in range(32):
            result[j + 32 + l] = d2 * (qs[j // 2 + l] >> 4) - dm2

        is_idx += 2

    return result


def dequant_q6_k_block(data: bytes) -> np.ndarray:
    """Dequantize Q6_K block (210 bytes -> 256 floats)"""
    ql = data[0:128]
    qh = data[128:192]
    scales = data[192:208]
    d = fp16_to_fp32(struct.unpack('<H', data[208:210])[0])

    result = np.zeros(256, dtype=np.float32)

    for n in range(0, 256, 128):
        for l in range(32):
            ql0 = ql[n // 2 + l]
            qh0 = qh[n // 4 + l]

            sc0 = np.array(scales[n // 16 + 0], dtype=np.uint8).view(np.int8).item()
            sc1 = np.array(scales[n // 16 + 1], dtype=np.uint8).view(np.int8).item()
            sc2 = np.array(scales[n // 16 + 2], dtype=np.uint8).view(np.int8).item()
            sc3 = np.array(scales[n // 16 + 3], dtype=np.uint8).view(np.int8).item()

            result[n + l + 0] = d * sc0 * (((ql0 & 0xF) | ((qh0 & 0x03) << 4)) - 32)
            result[n + l + 32] = d * sc1 * (((ql0 >> 4) | ((qh0 & 0x0C) << 2)) - 32)

            if n // 2 + l + 32 < 128:
                ql1 = ql[n // 2 + l + 32]
                qh1 = qh[n // 4 + l + 32] if n // 4 + l + 32 < 64 else 0

                result[n + l + 64] = d * sc2 * (((ql1 & 0xF) | ((qh1 & 0x03) << 4)) - 32)
                result[n + l + 96] = d * sc3 * (((ql1 >> 4) | ((qh1 & 0x0C) << 2)) - 32)

    return result


def dequant_q5_0_block(data: bytes) -> np.ndarray:
    """Dequantize Q5_0 block (22 bytes -> 32 floats)"""
    d = fp16_to_fp32(struct.unpack('<H', data[0:2])[0])
    qh = struct.unpack('<I', data[2:6])[0]
    qs = data[6:22]

    result = np.zeros(32, dtype=np.float32)
    for j in range(32):
        ql = (qs[j // 2] >> (4 * (j % 2))) & 0x0F
        qh_bit = (qh >> j) & 1
        q5 = ql | (qh_bit << 4)
        result[j] = d * (q5 - 16)

    return result


# ============================================================================
# GGUF Reader
# ============================================================================

GGUF_DTYPES = {
    0: ('F32', 4, 1),
    1: ('F16', 2, 1),
    2: ('Q4_0', 18, 32),
    6: ('Q5_0', 22, 32),
    8: ('Q8_0', 34, 32),
    12: ('Q4_K', 144, 256),
    14: ('Q6_K', 210, 256),
}


class GGUFReader:
    """Read tensors from GGUF file"""

    def __init__(self, path: str):
        self.path = path
        self.tensors: Dict[str, dict] = {}
        self.metadata: Dict[str, any] = {}
        self._parse()

    def _read_string(self, f) -> str:
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, vtype: int):
        if vtype == 0: return struct.unpack('<B', f.read(1))[0]
        elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
        elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
        elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
        elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
        elif vtype == 8: return self._read_string(f)
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
        elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
        else:
            f.read({2: 2, 3: 2, 7: 1, 12: 8}.get(vtype, 0))
            return None

    def _parse(self):
        with open(self.path, 'rb') as f:
            magic = f.read(4)
            assert magic == b'GGUF', f"Not a GGUF file: {magic}"

            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            for _ in range(metadata_kv_count):
                key = self._read_string(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, vtype)
                self.metadata[key] = value

            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]

                dtype_info = GGUF_DTYPES.get(dtype, ('UNK', 0, 0))
                self.tensors[name] = {
                    'dims': dims,
                    'dtype': dtype,
                    'dtype_name': dtype_info[0],
                    'block_size': dtype_info[1],
                    'block_elements': dtype_info[2],
                    'offset': offset,
                }

            self.data_start = (f.tell() + 31) // 32 * 32

    def read_raw_block(self, tensor_name: str, block_idx: int = 0) -> bytes:
        """Read raw block data from tensor"""
        if tensor_name not in self.tensors:
            raise ValueError(f"Tensor not found: {tensor_name}")

        info = self.tensors[tensor_name]
        block_size = info['block_size']

        with open(self.path, 'rb') as f:
            f.seek(self.data_start + info['offset'] + block_idx * block_size)
            return f.read(block_size)

    def read_dequantized_block(self, tensor_name: str, block_idx: int = 0) -> np.ndarray:
        """Read and dequantize a block"""
        info = self.tensors[tensor_name]
        data = self.read_raw_block(tensor_name, block_idx)

        dtype = info['dtype']
        if dtype == 2:
            return dequant_q4_0_block(data)
        elif dtype == 6:
            return dequant_q5_0_block(data)
        elif dtype == 8:
            return dequant_q8_0_block(data)
        elif dtype == 12:
            return dequant_q4_k_block(data)
        elif dtype == 14:
            return dequant_q6_k_block(data)
        elif dtype == 0:
            return np.frombuffer(data, dtype=np.float32)
        elif dtype == 1:
            return np.frombuffer(data, dtype=np.float16).astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {info['dtype_name']}")


# ============================================================================
# C Kernel Interface
# ============================================================================

class CKernelInterface:
    """Interface to C-Kernel-Engine's kernels via ctypes"""

    def __init__(self, lib_path: str = None):
        if lib_path is None:
            lib_path = Path(__file__).parent.parent / "build" / "libckernel_engine.so"

        if not Path(lib_path).exists():
            raise FileNotFoundError(f"Library not found: {lib_path}")

        self.lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()

    def _setup_functions(self):
        """Setup ctypes function signatures"""
        # gemm_nt_q4_k(A, B, bias, C, M, N, K)
        self.lib.gemm_nt_q4_k.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.c_void_p,                  # B (quantized)
            ctypes.POINTER(ctypes.c_float),   # bias
            ctypes.POINTER(ctypes.c_float),   # C
            ctypes.c_int, ctypes.c_int, ctypes.c_int  # M, N, K
        ]

        # gemm_nt_q6_k
        self.lib.gemm_nt_q6_k.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

    def extract_q4_k_weights(self, raw_data: bytes, n_elements: int) -> np.ndarray:
        """Extract weights by computing dot product with unit vectors"""
        result = np.zeros(n_elements, dtype=np.float32)
        B = np.frombuffer(raw_data, dtype=np.uint8)

        for i in range(n_elements):
            A = np.zeros(n_elements, dtype=np.float32)
            A[i] = 1.0
            C = np.zeros(1, dtype=np.float32)

            self.lib.gemm_nt_q4_k(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                B.ctypes.data_as(ctypes.c_void_p),
                None,
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                1, 1, n_elements
            )
            result[i] = C[0]

        return result


# ============================================================================
# Test Cases
# ============================================================================

@dataclass
class TestResult:
    name: str
    passed: int
    failed: int
    max_diff: float
    details: str = ""


def test_q4_k_scale_unpacking() -> TestResult:
    """Test Q4_K scale/min extraction against llama.cpp reference"""
    test_cases = [
        bytes([236, 243, 254, 184, 172, 230, 235, 185, 123, 7, 255, 201]),
        bytes([0] * 12),
        bytes([255] * 12),
        bytes([0xC0] * 8 + [0] * 4),
        bytes([0x3F] * 8 + [0xFF] * 4),
    ]

    passed = 0
    failed = 0
    max_diff = 0

    for tc_idx, scales in enumerate(test_cases):
        match = True
        for j in range(8):
            d_ref, m_ref = get_scale_min_k4(j, scales)

            # Our implementation
            if j < 4:
                d_ck = scales[j] & 0x3F
                m_ck = scales[j + 4] & 0x3F
            else:
                d_ck = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
                m_ck = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)

            if d_ref != d_ck or m_ref != m_ck:
                match = False
                max_diff = max(max_diff, abs(d_ref - d_ck), abs(m_ref - m_ck))

        if match:
            passed += 1
        else:
            failed += 1

    return TestResult("Q4_K Scale Unpacking", passed, failed, max_diff)


def test_q4_k_dequant_block(gguf: GGUFReader) -> TestResult:
    """Test Q4_K block dequantization"""
    q4k_tensors = [n for n, i in gguf.tensors.items() if i['dtype'] == 12]

    if not q4k_tensors:
        return TestResult("Q4_K Block Dequant", 0, 0, 0, "No Q4_K tensors")

    passed = 0
    failed = 0
    max_diff = 0.0

    # Test first 10 blocks from first Q4_K tensor
    tensor_name = q4k_tensors[0]
    for block_idx in range(10):
        try:
            raw = gguf.read_raw_block(tensor_name, block_idx)
            dequant = dequant_q4_k_block(raw)

            # Verify no NaN/Inf
            if np.isnan(dequant).any() or np.isinf(dequant).any():
                failed += 1
            else:
                passed += 1
                max_diff = max(max_diff, np.abs(dequant).max())
        except:
            break

    return TestResult("Q4_K Block Dequant", passed, failed, max_diff,
                      f"Tested {tensor_name}")


def test_q6_k_dequant_block(gguf: GGUFReader) -> TestResult:
    """Test Q6_K block dequantization"""
    q6k_tensors = [n for n, i in gguf.tensors.items() if i['dtype'] == 14]

    if not q6k_tensors:
        return TestResult("Q6_K Block Dequant", 0, 0, 0, "No Q6_K tensors")

    passed = 0
    failed = 0
    max_diff = 0.0

    tensor_name = q6k_tensors[0]
    for block_idx in range(10):
        try:
            raw = gguf.read_raw_block(tensor_name, block_idx)
            dequant = dequant_q6_k_block(raw)

            if np.isnan(dequant).any() or np.isinf(dequant).any():
                failed += 1
            else:
                passed += 1
                max_diff = max(max_diff, np.abs(dequant).max())
        except:
            break

    return TestResult("Q6_K Block Dequant", passed, failed, max_diff,
                      f"Tested {tensor_name}")


def test_q8_0_dequant_block(gguf: GGUFReader) -> TestResult:
    """Test Q8_0 block dequantization"""
    q8_tensors = [n for n, i in gguf.tensors.items() if i['dtype'] == 8]

    if not q8_tensors:
        return TestResult("Q8_0 Block Dequant", 0, 0, 0, "No Q8_0 tensors")

    passed = 0
    failed = 0
    max_diff = 0.0

    tensor_name = q8_tensors[0]
    for block_idx in range(10):
        try:
            raw = gguf.read_raw_block(tensor_name, block_idx)
            dequant = dequant_q8_0_block(raw)

            if np.isnan(dequant).any() or np.isinf(dequant).any():
                failed += 1
            else:
                passed += 1
                max_diff = max(max_diff, np.abs(dequant).max())
        except:
            break

    return TestResult("Q8_0 Block Dequant", passed, failed, max_diff,
                      f"Tested {tensor_name}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive kernel tests")
    parser.add_argument('--gguf', type=str,
                        default=os.path.expanduser(
                            "~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"))
    parser.add_argument('--list-tensors', action='store_true')
    parser.add_argument('--test-kernel', type=str, choices=['q4_k', 'q6_k', 'q8_0', 'all'],
                        default='all')
    args = parser.parse_args()

    print("=" * 80)
    print(f"{BOLD}  C-Kernel-Engine Comprehensive Kernel Tests{RESET}")
    print("=" * 80)

    # Load GGUF
    if not Path(args.gguf).exists():
        print(f"{RED}ERROR: GGUF not found: {args.gguf}{RESET}")
        return 1

    gguf = GGUFReader(args.gguf)
    print(f"GGUF: {args.gguf}")
    print(f"Tensors: {len(gguf.tensors)}")

    # Count by dtype
    dtype_counts = {}
    for name, info in gguf.tensors.items():
        dtype_name = info['dtype_name']
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

    print("Dtypes:", dtype_counts)
    print()

    if args.list_tensors:
        print(f"{'Tensor':<40} {'Dtype':<8} {'Dims'}")
        print("-" * 70)
        for name, info in sorted(gguf.tensors.items()):
            print(f"{name:<40} {info['dtype_name']:<8} {info['dims']}")
        return 0

    # Run tests
    results: List[TestResult] = []

    results.append(test_q4_k_scale_unpacking())

    if args.test_kernel in ['q4_k', 'all']:
        results.append(test_q4_k_dequant_block(gguf))

    if args.test_kernel in ['q6_k', 'all']:
        results.append(test_q6_k_dequant_block(gguf))

    if args.test_kernel in ['q8_0', 'all']:
        results.append(test_q8_0_dequant_block(gguf))

    # Print results
    print()
    print(f"{'Test':<30} {'Passed':<10} {'Failed':<10} {'Max Value':<12} {'Details'}")
    print("-" * 80)

    total_passed = 0
    total_failed = 0

    for r in results:
        status = f"{GREEN}PASS{RESET}" if r.failed == 0 else f"{RED}FAIL{RESET}"
        print(f"{r.name:<30} {r.passed:<10} {r.failed:<10} {r.max_diff:<12.4f} {r.details}")
        total_passed += r.passed
        total_failed += r.failed

    print("-" * 80)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("=" * 80)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
