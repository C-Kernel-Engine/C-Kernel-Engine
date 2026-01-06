#!/usr/bin/env python3
"""
test_q6k_kernels.py - Q6_K kernel parity tests

Tests Q6_K quantization and GEMV kernels against llama.cpp/GGML.

Q6_K format:
- 256 elements per block
- 210 bytes per block:
  - ql[128]: low 4 bits
  - qh[64]: high 2 bits
  - scales[16]: int8 scales
  - d: fp16 super-block scale
- Uses FP32 activations (unlike Q4_K which uses Q8_K)
"""

import ctypes
import numpy as np
import sys
import struct
import json
import mmap
from pathlib import Path

# Configuration
BASE_DIR = Path("/home/antshiv/Workspace/C-Kernel-Engine")
QK_K = 256  # Elements per super-block
BLOCK_Q6_K_SIZE = 210  # bytes

# Load libraries
print("Loading libraries...")
try:
    libggml = ctypes.CDLL(str(BASE_DIR / "llama.cpp/libggml_kernel_test.so"))
    libck = ctypes.CDLL(str(BASE_DIR / "build/libck_parity.so"))
    print("  GGML kernel test library loaded")
    print("  CK parity library loaded")
except OSError as e:
    print(f"Error loading libraries: {e}")
    print("\nMake sure to build the libraries:")
    print("  cd llama.cpp && make libggml_kernel_test.so")
    print("  make libck_parity.so")
    sys.exit(1)

# Set up function signatures
# GGML functions
libggml.test_dequant_q6_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

# CK functions
libck.ck_test_dequant_q6_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]


def test_dequant_q6k_real_weights():
    """Test Q6_K dequantization using real weights from bump file.

    Compares CK dequant_q6_k_row against GGML dequantize_row_q6_K.
    """
    print(f"\n{'='*60}")
    print("TEST: dequant_q6_k (real weights)")
    print(f"{'='*60}")

    bump_path = BASE_DIR / "build/smollm/weights.bump"
    manifest_path = BASE_DIR / "build/smollm/weights_manifest.json"

    if not bump_path.exists():
        print(f"Bump file not found: {bump_path}")
        print("Run: python scripts/ck_run_v5.py <gguf> to generate")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find all Q6_K tensors
    q6k_entries = [e for e in manifest.get('entries', [])
                   if e.get('dtype', '').lower() == 'q6_k']

    if not q6k_entries:
        print("No Q6_K tensors found in manifest")
        return False

    print(f"Found {len(q6k_entries)} Q6_K tensors")

    # Test multiple tensors
    all_pass = True
    test_count = 0

    with open(bump_path, 'rb') as f:
        bump_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Test token_emb and a few layer weights
        test_entries = []
        for entry in q6k_entries:
            name = entry['name']
            # Include embedding and a few layer weights
            if 'token_emb' in name or 'layer.0' in name or 'layer.12' in name or 'layer.23' in name:
                test_entries.append(entry)

        if len(test_entries) > 6:
            test_entries = test_entries[:6]

        for entry in test_entries:
            name = entry['name']
            offset = entry.get('file_offset', 0)
            size = entry['size']
            n_blocks = size // BLOCK_Q6_K_SIZE
            n_elements = n_blocks * QK_K

            # Test first few blocks
            test_blocks = min(8, n_blocks)
            test_elements = test_blocks * QK_K
            test_bytes = test_blocks * BLOCK_Q6_K_SIZE

            print(f"\nTesting: {name}")
            print(f"  Total: {n_elements} elements ({n_blocks} blocks)")
            print(f"  Testing first {test_blocks} blocks")

            # Read weight data
            bump_mmap.seek(offset)
            q6k_data = bump_mmap.read(test_bytes)

            # Dequantize with GGML
            ggml_out = np.zeros(test_elements, dtype=np.float32)
            libggml.test_dequant_q6_k(
                q6k_data,
                ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                test_elements
            )

            # Dequantize with CK
            ck_out = np.zeros(test_elements, dtype=np.float32)
            libck.ck_test_dequant_q6_k(
                q6k_data,
                ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                test_elements
            )

            # Compare
            diff = np.abs(ggml_out - ck_out)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            rel_err = 100 * max_diff / (np.max(np.abs(ggml_out)) + 1e-9)

            passed = max_diff < 1e-5
            status = "PASS" if passed else "FAIL"

            print(f"  [{status}] max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_err={rel_err:.4f}%")

            if not passed:
                all_pass = False
                # Show first mismatch
                idx = np.argmax(diff)
                print(f"       First mismatch at idx {idx}: GGML={ggml_out[idx]:.6f}, CK={ck_out[idx]:.6f}")

                # Show block-level stats
                for b in range(min(3, test_blocks)):
                    block_diff = diff[b*QK_K:(b+1)*QK_K]
                    print(f"       Block {b} max_diff: {np.max(block_diff):.2e}")

            test_count += 1

        bump_mmap.close()

    print(f"\n{'='*60}")
    if all_pass:
        print(f"[PASS] dequant_q6_k: All {test_count} tests passed")
    else:
        print(f"[FAIL] dequant_q6_k: Some tests failed")

    return all_pass


def test_gemv_q6k_real_weights():
    """Test Q6_K GEMV using real weights.

    Flow:
    1. Load Q6_K weights from bump file
    2. Dequantize to FP32 for reference
    3. Create random FP32 activations
    4. Compute reference: W_f32 . activations
    5. Run CK gemv_q6_k
    6. Compare results
    """
    print(f"\n{'='*60}")
    print("TEST: gemv_q6_k (real weights)")
    print(f"{'='*60}")

    bump_path = BASE_DIR / "build/smollm/weights.bump"
    manifest_path = BASE_DIR / "build/smollm/weights_manifest.json"

    if not bump_path.exists():
        print(f"Bump file not found: {bump_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find Q6_K tensors
    q6k_entries = [e for e in manifest.get('entries', [])
                   if e.get('dtype', '').lower() == 'q6_k']

    if not q6k_entries:
        print("No Q6_K tensors found")
        return False

    # Use layer.0.wv for testing (typically smaller)
    test_entry = None
    for entry in q6k_entries:
        if 'layer.0.wv' in entry['name']:
            test_entry = entry
            break

    if not test_entry:
        test_entry = q6k_entries[0]

    print(f"Using tensor: {test_entry['name']}")
    print(f"  Size: {test_entry['size']} bytes")

    # Load weight data
    with open(bump_path, 'rb') as f:
        bump_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offset = test_entry.get('file_offset', 0)
        size = test_entry['size']
        bump_mmap.seek(offset)
        q6k_data = bump_mmap.read(size)
        bump_mmap.close()

    n_blocks = size // BLOCK_Q6_K_SIZE
    n_elements = n_blocks * QK_K
    print(f"  Elements: {n_elements} ({n_blocks} blocks)")

    # Test with different row counts
    np.random.seed(42)
    all_pass = True

    for test_rows in [1, 4, 8]:
        row_blocks = min(test_rows, n_blocks)
        row_elements = row_blocks * QK_K
        row_bytes = row_blocks * BLOCK_Q6_K_SIZE
        q6k_row = q6k_data[:row_bytes]

        print(f"\nTest: {row_blocks} rows ({row_elements} elements)")

        # Dequantize weights for reference
        weight_f32 = np.zeros(row_elements, dtype=np.float32)
        libggml.test_dequant_q6_k(
            q6k_row,
            weight_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            row_elements
        )

        # Create random activations
        activations_f32 = np.random.randn(row_elements).astype(np.float32)

        # FP32 reference
        ref_dot = np.dot(weight_f32, activations_f32)

        # CK GEMV (gemv_q6_k uses FP32 activations)
        ck_out = np.zeros(1, dtype=np.float32)

        # Set up function signature
        try:
            libck.ck_test_gemv_q6_k.argtypes = [
                ctypes.c_void_p,                  # W
                ctypes.POINTER(ctypes.c_float),  # x
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int                      # cols
            ]
        except AttributeError:
            print("  ck_test_gemv_q6_k not found in libck_parity.so")
            return False

        libck.ck_test_gemv_q6_k(
            q6k_row,
            activations_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            row_elements  # cols
        )

        abs_err = abs(ref_dot - ck_out[0])
        rel_err = abs_err / (abs(ref_dot) + 1e-9)

        passed = rel_err < 0.05  # 5% tolerance for quantized
        status = "PASS" if passed else "FAIL"

        print(f"  FP32 ref: {ref_dot:.6f}")
        print(f"  CK GEMV:  {ck_out[0]:.6f}")
        print(f"  [{status}] abs_err={abs_err:.2e}, rel_err={rel_err:.4%}")

        if not passed:
            all_pass = False

    return all_pass


def test_dequant_q6k_layout():
    """Verify Q6_K memory layout understanding by manual dequantization."""
    print(f"\n{'='*60}")
    print("TEST: Q6_K layout verification (manual dequant)")
    print(f"{'='*60}")

    bump_path = BASE_DIR / "build/smollm/weights.bump"
    manifest_path = BASE_DIR / "build/smollm/weights_manifest.json"

    if not bump_path.exists():
        print(f"Bump file not found: {bump_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find first Q6_K entry
    q6k_entry = None
    for entry in manifest.get('entries', []):
        if entry.get('dtype', '').lower() == 'q6_k':
            q6k_entry = entry
            break

    if not q6k_entry:
        print("No Q6_K tensor found")
        return False

    print(f"Using: {q6k_entry['name']}")

    # Load one block
    with open(bump_path, 'rb') as f:
        f.seek(q6k_entry.get('file_offset', 0))
        block_data = f.read(BLOCK_Q6_K_SIZE)

    # Parse Q6_K block manually
    # Q6_K layout (210 bytes):
    #   ql[128]: low 4 bits
    #   qh[64]: high 2 bits
    #   scales[16]: int8 scales
    #   d: fp16 super-block scale

    ql = np.frombuffer(block_data[0:128], dtype=np.uint8)
    qh = np.frombuffer(block_data[128:192], dtype=np.uint8)
    scales = np.frombuffer(block_data[192:208], dtype=np.int8)
    d_fp16 = struct.unpack('<H', block_data[208:210])[0]

    # Convert d from fp16 to float32
    def fp16_to_fp32(h):
        sign = (h & 0x8000) << 16
        exp = (h >> 10) & 0x1F
        mant = h & 0x3FF
        if exp == 0:
            if mant == 0:
                return 0.0
            exp = 1
            while (mant & 0x400) == 0:
                mant <<= 1
                exp -= 1
            mant &= 0x3FF
            result = sign | ((exp + 127 - 15) << 23) | (mant << 13)
        elif exp == 31:
            result = sign | 0x7F800000 | (mant << 13)
        else:
            result = sign | ((exp + 127 - 15) << 23) | (mant << 13)
        return struct.unpack('<f', struct.pack('<I', result))[0]

    d = fp16_to_fp32(d_fp16)

    print(f"  d (scale): {d:.6f}")
    print(f"  scales[0:4]: {scales[0:4]}")
    print(f"  ql[0:8]: {ql[0:8]}")
    print(f"  qh[0:8]: {qh[0:8]}")

    # Manual dequantization following GGML formula
    manual_dequant = np.zeros(QK_K, dtype=np.float32)

    ql_ptr = 0
    qh_ptr = 0
    sc_ptr = 0
    out_ptr = 0

    for n in range(0, QK_K, 128):  # Two iterations for 256 elements
        for l in range(32):
            is_ = l // 16

            # Reconstruct 6-bit values from ql (4 bits) + qh (2 bits)
            q1 = ((ql[ql_ptr + l] & 0xF) | (((qh[qh_ptr + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[ql_ptr + l + 32] & 0xF) | (((qh[qh_ptr + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[ql_ptr + l] >> 4) | (((qh[qh_ptr + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[ql_ptr + l + 32] >> 4) | (((qh[qh_ptr + l] >> 6) & 3) << 4)) - 32

            manual_dequant[out_ptr + l + 0] = d * scales[sc_ptr + is_ + 0] * q1
            manual_dequant[out_ptr + l + 32] = d * scales[sc_ptr + is_ + 2] * q2
            manual_dequant[out_ptr + l + 64] = d * scales[sc_ptr + is_ + 4] * q3
            manual_dequant[out_ptr + l + 96] = d * scales[sc_ptr + is_ + 6] * q4

        out_ptr += 128
        ql_ptr += 64
        qh_ptr += 32
        sc_ptr += 8

    # Compare with GGML
    ggml_dequant = np.zeros(QK_K, dtype=np.float32)
    libggml.test_dequant_q6_k(
        block_data,
        ggml_dequant.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        QK_K
    )

    diff = np.abs(manual_dequant - ggml_dequant)
    max_diff = np.max(diff)

    passed = max_diff < 1e-6
    status = "PASS" if passed else "FAIL"

    print(f"\n  Manual vs GGML max_diff: {max_diff:.2e}")
    print(f"  [{status}] Q6_K layout understanding")

    if not passed:
        idx = np.argmax(diff)
        print(f"  First mismatch at {idx}: manual={manual_dequant[idx]:.6f}, ggml={ggml_dequant[idx]:.6f}")

    return passed


def main():
    print("=" * 70)
    print("Q6_K KERNEL PARITY TESTS")
    print("=" * 70)

    results = {}

    # Test 1: Q6_K layout verification
    results["q6k_layout"] = test_dequant_q6k_layout()

    # Test 2: Dequant with real weights
    results["dequant_q6k"] = test_dequant_q6k_real_weights()

    # Test 3: GEMV with real weights
    results["gemv_q6k"] = test_gemv_q6k_real_weights()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
