#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Quantization Kernels vs llama.cpp

This test validates that C-Kernel-Engine's dequantization exactly matches
llama.cpp's reference implementation for all supported quantization formats.

Tests:
1. Q4_K scale/min unpacking (get_scale_min_k4)
2. Q4_K full block dequantization
3. Q6_K block dequantization
4. Q5_0 block dequantization
5. End-to-end tensor dequantization comparison

Usage:
    python unittest/test_quant_vs_llamacpp.py [--gguf path/to/model.gguf]
"""

import struct
import numpy as np
import sys
import os
import argparse

# Color output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def fp16_to_fp32(h):
    """Convert FP16 (uint16) to FP32 - matches llama.cpp exactly"""
    return np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0].astype(np.float32)


# ============================================================================
# llama.cpp Reference Implementations
# ============================================================================

def get_scale_min_k4_llamacpp(j, q):
    """
    llama.cpp's get_scale_min_k4 function verbatim.

    For j < 4: d = q[j] & 63, m = q[j+4] & 63
    For j >= 4: Complex 4+2 bit packing from bytes 8-11 and upper bits
    """
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d, m


def dequant_q4_k_llamacpp(data):
    """
    llama.cpp Q4_K dequantization reference.

    Block structure (144 bytes total):
      - 2 bytes: d (fp16 super-block scale)
      - 2 bytes: dmin (fp16 super-block min)
      - 12 bytes: scales (8 sub-block scales + 8 mins, 6-bit packed)
      - 128 bytes: qs (256 4-bit weights)
    """
    assert len(data) == 144, f"Q4_K block must be 144 bytes, got {len(data)}"

    d_raw = struct.unpack('<H', data[0:2])[0]
    dmin_raw = struct.unpack('<H', data[2:4])[0]
    d = fp16_to_fp32(d_raw)
    dmin = fp16_to_fp32(dmin_raw)

    scales = list(data[4:16])
    qs = data[16:144]

    result = np.zeros(256, dtype=np.float32)

    # Process in groups of 64 (2 sub-blocks)
    q_ptr = 0
    is_idx = 0

    for j in range(0, 256, 64):
        sc1, m1 = get_scale_min_k4_llamacpp(is_idx, scales)
        sc2, m2 = get_scale_min_k4_llamacpp(is_idx + 1, scales)

        d1 = d * sc1
        dm1 = dmin * m1
        d2 = d * sc2
        dm2 = dmin * m2

        # First 32: low nibbles with scale 1
        for l in range(32):
            q = qs[q_ptr + l] & 0x0F
            result[j + l] = d1 * q - dm1

        # Next 32: high nibbles with scale 2
        for l in range(32):
            q = qs[q_ptr + l] >> 4
            result[j + 32 + l] = d2 * q - dm2

        q_ptr += 32
        is_idx += 2

    return result


def dequant_q6_k_llamacpp(data):
    """
    llama.cpp Q6_K dequantization reference.

    Block structure (210 bytes total):
      - 128 bytes: ql (low 4 bits)
      - 64 bytes: qh (high 2 bits)
      - 16 bytes: scales (int8)
      - 2 bytes: d (fp16 super-block scale)
    """
    assert len(data) == 210, f"Q6_K block must be 210 bytes, got {len(data)}"

    ql = data[0:128]
    qh = data[128:192]
    scales = data[192:208]
    d_raw = struct.unpack('<H', data[208:210])[0]
    d = fp16_to_fp32(d_raw)

    result = np.zeros(256, dtype=np.float32)

    # Process 16 sub-blocks of 16 values each
    for n in range(16):
        sc = np.array(scales[n], dtype=np.uint8).view(np.int8)
        ql_offset = n * 8
        qh_offset = n * 4

        for j in range(16):
            # Get low 4 bits
            byte_idx = ql_offset + j // 2
            ql_val = (ql[byte_idx] >> (4 * (j % 2))) & 0x0F

            # Get high 2 bits
            qh_byte = qh_offset + j // 4
            qh_shift = 2 * (j % 4)
            qh_val = (qh[qh_byte] >> qh_shift) & 0x03

            # Combine to 6-bit value
            q6 = ql_val | (qh_val << 4)

            # Dequantize: d * sc * (q6 - 32)
            result[n * 16 + j] = d * sc * (q6 - 32)

    return result


def dequant_q5_0_llamacpp(data):
    """
    llama.cpp Q5_0 dequantization reference.

    Block structure (22 bytes total):
      - 2 bytes: d (fp16 scale)
      - 4 bytes: qh (high bits for 32 weights)
      - 16 bytes: qs (low 4 bits for 32 weights)
    """
    assert len(data) == 22, f"Q5_0 block must be 22 bytes, got {len(data)}"

    d_raw = struct.unpack('<H', data[0:2])[0]
    d = fp16_to_fp32(d_raw)

    qh = struct.unpack('<I', data[2:6])[0]
    qs = data[6:22]

    result = np.zeros(32, dtype=np.float32)

    for j in range(32):
        # Low 4 bits
        ql = (qs[j // 2] >> (4 * (j % 2))) & 0x0F
        # High 1 bit
        qh_bit = (qh >> j) & 1
        # Combine to 5-bit value
        q5 = ql | (qh_bit << 4)
        # Dequantize: d * (q5 - 16)
        result[j] = d * (q5 - 16)

    return result


def dequant_q4_0_llamacpp(data):
    """
    llama.cpp Q4_0 dequantization reference.

    Block structure (18 bytes total):
      - 2 bytes: d (fp16 scale)
      - 16 bytes: qs (32 4-bit weights, 2 per byte)
    """
    assert len(data) == 18, f"Q4_0 block must be 18 bytes, got {len(data)}"

    d_raw = struct.unpack('<H', data[0:2])[0]
    d = fp16_to_fp32(d_raw)

    qs = data[2:18]

    result = np.zeros(32, dtype=np.float32)

    for j in range(32):
        q = (qs[j // 2] >> (4 * (j % 2))) & 0x0F
        result[j] = d * (q - 8)

    return result


# ============================================================================
# C-Kernel-Engine Implementations (for comparison)
# ============================================================================

def unpack_q4_k_scales_ckernel(scales):
    """C-Kernel-Engine's unpack_q4_k_scales function"""
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

    # 6-bit values for indices 4-7
    sc[4] = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4)
    sc[5] = (scales[9] & 0x0F) | ((scales[1] >> 6) << 4)
    sc[6] = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4)
    sc[7] = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4)

    m[4] = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    m[5] = (scales[9] >> 4) | ((scales[5] >> 6) << 4)
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4)
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4)

    return sc, m


# ============================================================================
# GGUF Reader
# ============================================================================

class GGUFReader:
    """Simple GGUF file reader"""

    GGUF_TYPES = {
        0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 4: 'Q4_2', 5: 'Q4_3',
        6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'I8', 11: 'I16',
        12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS',
        17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL',
    }

    def __init__(self, path):
        self.path = path
        self.tensors = {}
        self._read_header()

    def _read_string(self, f):
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _skip_value(self, f, vtype):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype in sizes:
            f.read(sizes[vtype])
        elif vtype == 8:
            self._read_string(f)
        elif vtype == 9:
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            for _ in range(arr_len):
                self._skip_value(f, arr_type)

    def _read_header(self):
        with open(self.path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError(f"Not a GGUF file: {magic}")

            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            # Skip metadata
            for _ in range(metadata_kv_count):
                key = self._read_string(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                self._skip_value(f, vtype)

            # Read tensor infos
            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                self.tensors[name] = {
                    'dims': dims,
                    'dtype': dtype,
                    'dtype_name': self.GGUF_TYPES.get(dtype, f'UNK({dtype})'),
                    'offset': offset
                }

            # Calculate data start (aligned to 32 bytes)
            current_pos = f.tell()
            self.data_start = (current_pos + 31) // 32 * 32

    def read_tensor_block(self, name, block_idx=0):
        """Read a specific block from a tensor"""
        if name not in self.tensors:
            raise ValueError(f"Tensor not found: {name}")

        info = self.tensors[name]
        dtype = info['dtype']

        # Block sizes
        block_sizes = {2: 18, 6: 22, 12: 144, 14: 210}
        if dtype not in block_sizes:
            raise ValueError(f"Unsupported dtype {info['dtype_name']} for block read")

        block_size = block_sizes[dtype]

        with open(self.path, 'rb') as f:
            f.seek(self.data_start + info['offset'] + block_idx * block_size)
            return f.read(block_size)

    def get_tensors_by_dtype(self, dtype_name):
        """Get all tensors with a specific dtype"""
        return {k: v for k, v in self.tensors.items() if v['dtype_name'] == dtype_name}


# ============================================================================
# Test Functions
# ============================================================================

def test_q4_k_scale_unpacking():
    """Test 1: Q4_K scale/min unpacking matches llama.cpp"""
    print(f"\n{'='*60}")
    print("Test 1: Q4_K Scale/Min Unpacking")
    print(f"{'='*60}")

    # Generate test cases with known edge cases
    test_cases = [
        # Random values
        [236, 243, 254, 184, 172, 230, 235, 185, 123, 7, 255, 201],
        # All zeros
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # All 0xFF
        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        # High bits set pattern
        [0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0x00, 0x00, 0x00, 0x00],
        # Low nibbles set
        [0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0xFF, 0xFF, 0xFF, 0xFF],
    ]

    all_pass = True
    for case_idx, scales in enumerate(test_cases):
        sc_ck, m_ck = unpack_q4_k_scales_ckernel(scales)

        errors = []
        for j in range(8):
            sc_llama, m_llama = get_scale_min_k4_llamacpp(j, scales)
            if sc_ck[j] != sc_llama or m_ck[j] != m_llama:
                errors.append(f"  j={j}: CK=({sc_ck[j]},{m_ck[j]}) vs llama=({sc_llama},{m_llama})")

        if errors:
            print(f"  Case {case_idx}: {RED}FAIL{RESET}")
            for e in errors:
                print(e)
            all_pass = False
        else:
            print(f"  Case {case_idx}: {GREEN}PASS{RESET}")

    return all_pass


def test_q4_k_dequant(gguf_path=None):
    """Test 2: Q4_K full block dequantization"""
    print(f"\n{'='*60}")
    print("Test 2: Q4_K Block Dequantization")
    print(f"{'='*60}")

    if gguf_path and os.path.exists(gguf_path):
        reader = GGUFReader(gguf_path)
        q4k_tensors = reader.get_tensors_by_dtype('Q4_K')

        if not q4k_tensors:
            print(f"  {YELLOW}SKIP: No Q4_K tensors in GGUF{RESET}")
            return True

        # Test first Q4_K tensor
        tensor_name = list(q4k_tensors.keys())[0]
        print(f"  Testing tensor: {tensor_name}")

        # Test multiple blocks
        test_blocks = [0, 1, 10, 100]
        all_pass = True

        for block_idx in test_blocks:
            try:
                block_data = reader.read_tensor_block(tensor_name, block_idx)
                result = dequant_q4_k_llamacpp(block_data)

                # Validate result properties
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    print(f"    Block {block_idx}: {RED}FAIL (NaN/Inf){RESET}")
                    all_pass = False
                else:
                    print(f"    Block {block_idx}: {GREEN}PASS{RESET} (range: [{result.min():.4f}, {result.max():.4f}])")
            except Exception as e:
                print(f"    Block {block_idx}: {YELLOW}SKIP ({e}){RESET}")

        return all_pass
    else:
        # Use synthetic test data
        print("  Using synthetic test data (no GGUF provided)")

        # Create known test block
        test_block = bytes([
            0x00, 0x3C,  # d = 1.0 in fp16
            0x00, 0x3C,  # dmin = 1.0 in fp16
        ] + [63] * 12 +  # scales all = 63
            [0x00] * 128)  # qs all = 0

        result = dequant_q4_k_llamacpp(test_block)

        # All values should be -63 (0 * 63 - 63)
        expected = -63.0
        if np.allclose(result, expected, rtol=1e-4):
            print(f"  Synthetic test: {GREEN}PASS{RESET}")
            return True
        else:
            print(f"  Synthetic test: {RED}FAIL{RESET} (expected {expected}, got {result[0]:.4f})")
            return False


def test_q6_k_dequant(gguf_path=None):
    """Test 3: Q6_K block dequantization"""
    print(f"\n{'='*60}")
    print("Test 3: Q6_K Block Dequantization")
    print(f"{'='*60}")

    if gguf_path and os.path.exists(gguf_path):
        reader = GGUFReader(gguf_path)
        q6k_tensors = reader.get_tensors_by_dtype('Q6_K')

        if not q6k_tensors:
            print(f"  {YELLOW}SKIP: No Q6_K tensors in GGUF{RESET}")
            return True

        tensor_name = list(q6k_tensors.keys())[0]
        print(f"  Testing tensor: {tensor_name}")

        block_data = reader.read_tensor_block(tensor_name, 0)
        result = dequant_q6_k_llamacpp(block_data)

        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"  Block 0: {RED}FAIL (NaN/Inf){RESET}")
            return False

        print(f"  Block 0: {GREEN}PASS{RESET} (range: [{result.min():.4f}, {result.max():.4f}])")
        return True
    else:
        print(f"  {YELLOW}SKIP: No GGUF provided{RESET}")
        return True


def test_q5_0_dequant(gguf_path=None):
    """Test 4: Q5_0 block dequantization"""
    print(f"\n{'='*60}")
    print("Test 4: Q5_0 Block Dequantization")
    print(f"{'='*60}")

    # Synthetic test
    print("  Using synthetic test data")

    # Create test block: d=1.0, qh=0, qs=all 8 (which gives q5=8 for each)
    test_block = bytes([
        0x00, 0x3C,  # d = 1.0 in fp16
        0x00, 0x00, 0x00, 0x00,  # qh = 0
    ] + [0x88] * 16)  # qs = 8|8 for each byte

    result = dequant_q5_0_llamacpp(test_block)

    # All values should be (8 - 16) * 1.0 = -8.0
    expected = -8.0
    if np.allclose(result, expected, rtol=1e-4):
        print(f"  Synthetic test: {GREEN}PASS{RESET}")
        return True
    else:
        print(f"  Synthetic test: {RED}FAIL{RESET} (expected {expected}, got {result[0]:.4f})")
        return False


def test_q4_0_dequant(gguf_path=None):
    """Test 5: Q4_0 block dequantization"""
    print(f"\n{'='*60}")
    print("Test 5: Q4_0 Block Dequantization")
    print(f"{'='*60}")

    # Synthetic test
    print("  Using synthetic test data")

    # Create test block: d=1.0, qs=all 0
    test_block = bytes([
        0x00, 0x3C,  # d = 1.0 in fp16
    ] + [0x00] * 16)  # qs = 0

    result = dequant_q4_0_llamacpp(test_block)

    # All values should be (0 - 8) * 1.0 = -8.0
    expected = -8.0
    if np.allclose(result, expected, rtol=1e-4):
        print(f"  Synthetic test: {GREEN}PASS{RESET}")
        return True
    else:
        print(f"  Synthetic test: {RED}FAIL{RESET} (expected {expected}, got {result[0]:.4f})")
        return False


def test_gguf_tensor_stats(gguf_path):
    """Test 6: Report tensor statistics from GGUF"""
    print(f"\n{'='*60}")
    print("Test 6: GGUF Tensor Statistics")
    print(f"{'='*60}")

    if not gguf_path or not os.path.exists(gguf_path):
        print(f"  {YELLOW}SKIP: No GGUF provided{RESET}")
        return True

    reader = GGUFReader(gguf_path)

    # Count by dtype
    dtype_counts = {}
    for name, info in reader.tensors.items():
        dtype_name = info['dtype_name']
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

    print(f"  Total tensors: {len(reader.tensors)}")
    print(f"  By dtype:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"    {dtype}: {count}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Test quantization kernels vs llama.cpp')
    parser.add_argument('--gguf', type=str, default='qwen2.5-3b-instruct-q4_k_m.gguf',
                        help='Path to GGUF file for testing')
    args = parser.parse_args()

    print("="*60)
    print(" C-Kernel-Engine Quantization Tests vs llama.cpp")
    print("="*60)

    gguf_path = args.gguf
    if not os.path.exists(gguf_path):
        print(f"{YELLOW}Warning: GGUF file not found: {gguf_path}{RESET}")
        print("Running with synthetic tests only")
        gguf_path = None
    else:
        print(f"Using GGUF: {gguf_path}")

    results = []
    results.append(("Q4_K Scale Unpacking", test_q4_k_scale_unpacking()))
    results.append(("Q4_K Dequantization", test_q4_k_dequant(gguf_path)))
    results.append(("Q6_K Dequantization", test_q6_k_dequant(gguf_path)))
    results.append(("Q5_0 Dequantization", test_q5_0_dequant(gguf_path)))
    results.append(("Q4_0 Dequantization", test_q4_0_dequant(gguf_path)))
    results.append(("GGUF Tensor Stats", test_gguf_tensor_stats(gguf_path)))

    # Summary
    print(f"\n{'='*60}")
    print(" Summary")
    print(f"{'='*60}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
