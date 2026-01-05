#!/usr/bin/env python3
"""
test_layer_weights_parity.py - Compare layer weights between GGUF and C-Kernel bump format

Tests dequantization for various quant types:
- Q8_0 (embedding, some attention V)
- Q5_0 (most attention weights)
- Q6_K (some FFN down)
- Q4_K (some FFN down)
- F32 (RMSNorm)

Usage:
  python unittest/test_layer_weights_parity.py \
    --gguf ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
    --bump ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump \
    --manifest ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights_manifest.json
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path

# Block sizes
Q8_0_BLOCK_SIZE = 32
Q5_0_BLOCK_SIZE = 32
Q4_K_BLOCK_SIZE = 256
Q6_K_BLOCK_SIZE = 256


def dequant_q8_0_block(block_data: bytes) -> np.ndarray:
    """Dequantize Q8_0 block: 2 bytes scale (fp16) + 32 bytes quants (int8)."""
    scale = struct.unpack('<e', block_data[0:2])[0]
    quants = np.frombuffer(block_data[2:34], dtype=np.int8)
    return quants.astype(np.float32) * scale


def dequant_q5_0_block(block_data: bytes) -> np.ndarray:
    """
    Dequantize Q5_0 block: 2 bytes scale (fp16) + 4 bytes high bits + 16 bytes low bits
    Total: 22 bytes -> 32 floats

    Layout:
      - d: fp16 scale (2 bytes)
      - qh: 4 bytes (32 bits, one per value - the 5th bit)
      - qs: 16 bytes (32 4-bit values packed as pairs)
    """
    if len(block_data) != 22:
        raise ValueError(f"Q5_0 block must be 22 bytes, got {len(block_data)}")

    scale = struct.unpack('<e', block_data[0:2])[0]

    # qh: 4 bytes = 32 bits (one high bit per value)
    qh = np.frombuffer(block_data[2:6], dtype=np.uint8)
    qh_bits = np.unpackbits(qh, bitorder='little')[:32]  # 32 high bits

    # qs: 16 bytes = 32 nibbles (low 4 bits each)
    qs = np.frombuffer(block_data[6:22], dtype=np.uint8)

    # Unpack: each byte has two 4-bit values
    qs_lo = (qs & 0x0F).astype(np.int32)  # lower nibble
    qs_hi = (qs >> 4).astype(np.int32)    # upper nibble

    # Interleave: positions 0,2,4,... get lo, positions 1,3,5,... get hi
    q5 = np.zeros(32, dtype=np.int32)
    q5[0::2] = qs_lo
    q5[1::2] = qs_hi

    # Add high bit (bit 4) from qh
    q5 = q5 | (qh_bits.astype(np.int32) << 4)

    # Convert to signed: subtract 16 (values are 0-31, center at 16)
    q5 = q5 - 16

    return q5.astype(np.float32) * scale


def dequant_q6_k_block(block_data: bytes) -> np.ndarray:
    """
    Dequantize Q6_K block: 256 values per block
    Layout (210 bytes total):
      - ql: 128 bytes (256 4-bit low parts, packed)
      - qh: 64 bytes (256 2-bit high parts, packed)
      - scales: 16 bytes (16 scales as int8)
      - d: 2 bytes (fp16 super-scale)
    """
    if len(block_data) != 210:
        raise ValueError(f"Q6_K block must be 210 bytes, got {len(block_data)}")

    ql = np.frombuffer(block_data[0:128], dtype=np.uint8)
    qh = np.frombuffer(block_data[128:192], dtype=np.uint8)
    scales = np.frombuffer(block_data[192:208], dtype=np.int8)
    d = struct.unpack('<e', block_data[208:210])[0]

    result = np.zeros(256, dtype=np.float32)

    for j in range(256):
        # Get 4-bit low part from ql
        byte_idx = j // 2
        if j % 2 == 0:
            q_lo = ql[byte_idx] & 0x0F
        else:
            q_lo = ql[byte_idx] >> 4

        # Get 2-bit high part from qh
        # qh packs 4 values per byte
        qh_byte_idx = j // 4
        qh_shift = (j % 4) * 2
        q_hi = (qh[qh_byte_idx] >> qh_shift) & 0x03

        # Combine to 6-bit value
        q = q_lo | (q_hi << 4)

        # Get scale for this group (16 values per scale)
        scale_idx = j // 16
        sc = scales[scale_idx]

        # Dequantize: (q - 32) * scale * d
        result[j] = (q - 32) * sc * d

    return result


def dequant_q4_k_block(block_data: bytes) -> np.ndarray:
    """
    Dequantize Q4_K block: 256 values per block
    Layout (144 bytes total):
      - d: 2 bytes (fp16 super-scale)
      - dmin: 2 bytes (fp16 min scale)
      - scales: 12 bytes (packed scales and mins)
      - qs: 128 bytes (256 4-bit values packed)
    """
    if len(block_data) != 144:
        raise ValueError(f"Q4_K block must be 144 bytes, got {len(block_data)}")

    d = struct.unpack('<e', block_data[0:2])[0]
    dmin = struct.unpack('<e', block_data[2:4])[0]
    scales_raw = np.frombuffer(block_data[4:16], dtype=np.uint8)
    qs = np.frombuffer(block_data[16:144], dtype=np.uint8)

    # Unpack scales (8 scale/min pairs packed in 12 bytes)
    # This is complex - simplified version
    scales = np.zeros(8, dtype=np.float32)
    mins = np.zeros(8, dtype=np.float32)

    # Decode the 12-byte packed scales
    # Each 6-bit scale and 6-bit min is packed
    for i in range(8):
        if i < 4:
            sc = scales_raw[i] & 0x3F
            m = scales_raw[i + 4] & 0x3F
        else:
            sc = ((scales_raw[i + 4] >> 6) | ((scales_raw[i - 4] >> 4) & 0x0C) |
                  ((scales_raw[i] >> 2) & 0x30))
            m = ((scales_raw[i + 4] >> 4) & 0x03) | ((scales_raw[i - 4] >> 2) & 0x0C) | (scales_raw[i] & 0x30)
        scales[i] = sc
        mins[i] = m

    result = np.zeros(256, dtype=np.float32)

    for j in range(256):
        byte_idx = j // 2
        if j % 2 == 0:
            q = qs[byte_idx] & 0x0F
        else:
            q = qs[byte_idx] >> 4

        # Get scale/min for this group (32 values per group)
        group = j // 32
        sc = scales[group]
        m = mins[group]

        # Dequantize
        result[j] = d * sc * q - dmin * m

    return result


def dequant_tensor(raw_data: bytes, dtype: str, n_elements: int) -> np.ndarray:
    """Dequantize a tensor based on its dtype."""
    result = []

    if dtype == "q8_0":
        bytes_per_block = 34
        n_blocks = (n_elements + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE
        for i in range(n_blocks):
            block = raw_data[i * bytes_per_block:(i + 1) * bytes_per_block]
            result.extend(dequant_q8_0_block(block))

    elif dtype == "q5_0":
        bytes_per_block = 22
        n_blocks = (n_elements + Q5_0_BLOCK_SIZE - 1) // Q5_0_BLOCK_SIZE
        for i in range(n_blocks):
            block = raw_data[i * bytes_per_block:(i + 1) * bytes_per_block]
            result.extend(dequant_q5_0_block(block))

    elif dtype == "q6_k":
        bytes_per_block = 210
        n_blocks = (n_elements + Q6_K_BLOCK_SIZE - 1) // Q6_K_BLOCK_SIZE
        for i in range(n_blocks):
            block = raw_data[i * bytes_per_block:(i + 1) * bytes_per_block]
            result.extend(dequant_q6_k_block(block))

    elif dtype == "q4_k":
        bytes_per_block = 144
        n_blocks = (n_elements + Q4_K_BLOCK_SIZE - 1) // Q4_K_BLOCK_SIZE
        for i in range(n_blocks):
            block = raw_data[i * bytes_per_block:(i + 1) * bytes_per_block]
            result.extend(dequant_q4_k_block(block))

    elif dtype in ("f32", "fp32"):
        result = np.frombuffer(raw_data[:n_elements * 4], dtype=np.float32).tolist()

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return np.array(result[:n_elements], dtype=np.float32)


def get_gguf_tensor(reader, name: str) -> tuple:
    """Get a tensor from GGUF reader."""
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    return None


def compare_weights(gguf_path: str, bump_path: str, manifest_path: str, tests: list) -> dict:
    """Compare specific weights between GGUF and bump files."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)

    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest_lookup = {e['name']: e for e in manifest['entries']}

    with open(bump_path, 'rb') as bump_file:
        results = {}

        for test in tests:
            gguf_name = test['gguf_name']
            bump_name = test['bump_name']
            n_elements = test.get('n_elements', 896)  # default to hidden_dim

            print(f"\n{'='*60}")
            print(f"Testing: {gguf_name} vs {bump_name}")
            print(f"{'='*60}")

            # Read GGUF
            tensor = get_gguf_tensor(reader, gguf_name)
            if tensor is None:
                print(f"  ERROR: {gguf_name} not found in GGUF")
                results[gguf_name] = {'error': 'not found in GGUF'}
                continue

            print(f"  GGUF: type={tensor.tensor_type}, shape={tensor.shape}")

            # Read bump
            if bump_name not in manifest_lookup:
                print(f"  ERROR: {bump_name} not found in manifest")
                results[gguf_name] = {'error': 'not found in manifest'}
                continue

            entry = manifest_lookup[bump_name]
            bump_file.seek(entry['file_offset'])
            bump_data = bump_file.read(entry['size'])

            print(f"  BUMP: dtype={entry['dtype']}, size={entry['size']}")

            # Dequantize GGUF
            gguf_data = bytes(tensor.data.flatten())
            # GGML quant type mapping from the gguf library
            gguf_dtype = {
                0: 'f32',     # F32
                1: 'f16',     # F16
                6: 'q5_0',    # Q5_0
                8: 'q8_0',    # Q8_0
                12: 'q4_k',   # Q4_K
                14: 'q6_k',   # Q6_K
            }
            dtype = gguf_dtype.get(int(tensor.tensor_type), 'unknown')
            print(f"  Inferred dtype: {dtype}")

            # Get actual n_elements from tensor shape
            actual_n_elements = int(np.prod(tensor.shape))
            print(f"  Elements: {actual_n_elements}")

            # Only test first row for large matrices
            if actual_n_elements > 10000:
                test_elements = test.get('test_row_elements', tensor.shape[0])
                print(f"  Testing first row only: {test_elements} elements")
            else:
                test_elements = actual_n_elements

            try:
                gguf_dequant = dequant_tensor(gguf_data, dtype, test_elements)
                bump_dequant = dequant_tensor(bump_data, entry['dtype'], test_elements)

                # Compare
                diff = np.abs(gguf_dequant - bump_dequant)
                max_diff = float(np.max(diff))
                mean_diff = float(np.mean(diff))

                results[gguf_name] = {
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'gguf_first5': gguf_dequant[:5].tolist(),
                    'bump_first5': bump_dequant[:5].tolist(),
                    'pass': max_diff < 1e-4
                }

                status = "PASS" if results[gguf_name]['pass'] else "FAIL"
                print(f"  Result: {status}")
                print(f"    Max diff:  {max_diff:.10f}")
                print(f"    Mean diff: {mean_diff:.10f}")
                print(f"    GGUF[:5]:  {gguf_dequant[:5]}")
                print(f"    BUMP[:5]:  {bump_dequant[:5]}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results[gguf_name] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Test layer weight parity")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--bump", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    # Tests to run: GGUF name -> bump name mapping
    tests = [
        # Layer 0 RMSNorm (F32)
        {'gguf_name': 'blk.0.attn_norm.weight', 'bump_name': 'layer.0.ln1_gamma'},

        # Layer 0 Q weight (Q5_0)
        {'gguf_name': 'blk.0.attn_q.weight', 'bump_name': 'layer.0.wq', 'test_row_elements': 896},

        # Layer 0 V weight (Q8_0)
        {'gguf_name': 'blk.0.attn_v.weight', 'bump_name': 'layer.0.wv', 'test_row_elements': 896},

        # Layer 0 FFN down (Q6_K)
        {'gguf_name': 'blk.0.ffn_down.weight', 'bump_name': 'layer.0.w2', 'test_row_elements': 4864},

        # Layer 2 FFN down (Q4_K - different from layer 0)
        {'gguf_name': 'blk.2.ffn_down.weight', 'bump_name': 'layer.2.w2', 'test_row_elements': 4864},
    ]

    results = compare_weights(args.gguf, args.bump, args.manifest, tests)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    errors = 0

    for name, result in results.items():
        if 'error' in result:
            print(f"  {name}: ERROR - {result['error']}")
            errors += 1
        elif result['pass']:
            print(f"  {name}: PASS")
            passed += 1
        else:
            print(f"  {name}: FAIL (max_diff={result['max_diff']:.6f})")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed, {errors} errors")

    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == "__main__":
    exit(main())
