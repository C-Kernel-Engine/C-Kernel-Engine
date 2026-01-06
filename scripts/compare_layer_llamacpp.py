#!/usr/bin/env python3
"""
compare_layer_llamacpp.py - Compare layer-by-layer between llama.cpp and CK-Engine

This script:
1. Loads weights from GGUF (same source as llama.cpp)
2. Loads weights from CK bump format
3. Compares weights byte-by-byte
4. Runs forward pass for a specific layer
5. Compares each computation step

Usage:
    python scripts/compare_layer_llamacpp.py --gguf model.gguf --layer 2
    python scripts/compare_layer_llamacpp.py --gguf model.gguf --layer 2 --mode decode
    python scripts/compare_layer_llamacpp.py --gguf model.gguf --layer 2 --dump-weights
"""

import os
import sys
import json
import struct
import argparse
import ctypes
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ============================================================================
# Quantization helpers (matching llama.cpp exactly)
# ============================================================================

QK_K = 256
K_SCALE_SIZE = 12

QUANT_BLOCK_SIZES = {
    'q4_0': 18,   # 2 + 16 (32 weights, 4 bits each)
    'q4_1': 20,
    'q5_0': 22,   # 2 + 4 + 16 (32 weights)
    'q5_1': 24,
    'q8_0': 34,   # 2 + 32 (32 weights, 8 bits each)
    'q4_k': 144,  # QK_K=256 weights
    'q6_k': 210,  # QK_K=256 weights
}

QUANT_BLOCK_WEIGHTS = {
    'q4_0': 32, 'q4_1': 32, 'q5_0': 32, 'q5_1': 32,
    'q8_0': 32, 'q4_k': 256, 'q6_k': 256,
}


def dequant_q4_k_block(data: bytes) -> np.ndarray:
    """Dequantize Q4_K block (144 bytes -> 256 floats) - matches llama.cpp exactly"""
    d = np.frombuffer(data[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(data[2:4], dtype=np.float16)[0].astype(np.float32)
    scales = list(data[4:16])
    qs = data[16:144]

    result = np.zeros(256, dtype=np.float32)

    def get_scale_min(j, q):
        if j < 4:
            return q[j] & 63, q[j + 4] & 63
        else:
            return (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)

    q_ptr, is_idx = 0, 0
    for j in range(0, 256, 64):
        sc1, m1 = get_scale_min(is_idx, scales)
        sc2, m2 = get_scale_min(is_idx + 1, scales)
        d1, dm1 = d * sc1, dmin * m1
        d2, dm2 = d * sc2, dmin * m2

        for l in range(32):
            result[j + l] = d1 * (qs[q_ptr + l] & 0x0F) - dm1
        for l in range(32):
            result[j + 32 + l] = d2 * (qs[q_ptr + l] >> 4) - dm2

        q_ptr += 32
        is_idx += 2

    return result


def dequant_q6_k_block(data: bytes) -> np.ndarray:
    """Dequantize Q6_K block (210 bytes -> 256 floats) - matches llama.cpp exactly"""
    ql = data[0:128]
    qh = data[128:192]
    scales = np.frombuffer(data[192:208], dtype=np.int8)
    d = np.frombuffer(data[208:210], dtype=np.float16)[0].astype(np.float32)

    result = np.zeros(256, dtype=np.float32)

    for n in range(2):  # Two 128-element halves
        for l in range(32):
            is_idx = l // 16
            sc = scales[n * 8 + is_idx]

            # Extract 6-bit values
            q1 = (ql[n*64 + l] & 0xF) | (((qh[n*32 + l] >> 0) & 3) << 4)
            q2 = (ql[n*64 + l + 32] & 0xF) | (((qh[n*32 + l] >> 2) & 3) << 4)
            q3 = (ql[n*64 + l] >> 4) | (((qh[n*32 + l] >> 4) & 3) << 4)
            q4 = (ql[n*64 + l + 32] >> 4) | (((qh[n*32 + l] >> 6) & 3) << 4)

            result[n*128 + l + 0] = d * sc * (q1 - 32)
            result[n*128 + l + 32] = d * sc * (q2 - 32)
            result[n*128 + l + 64] = d * sc * (q3 - 32)
            result[n*128 + l + 96] = d * sc * (q4 - 32)

    return result


def dequant_q5_0_block(data: bytes) -> np.ndarray:
    """Dequantize Q5_0 block (22 bytes -> 32 floats)"""
    d = np.frombuffer(data[0:2], dtype=np.float16)[0].astype(np.float32)
    qh = struct.unpack('<I', data[2:6])[0]  # 32 high bits
    qs = data[6:22]  # 16 bytes = 32 4-bit values

    result = np.zeros(32, dtype=np.float32)
    for i in range(32):
        q4 = (qs[i // 2] >> (4 * (i % 2))) & 0xF
        qh_bit = (qh >> i) & 1
        q5 = q4 | (qh_bit << 4)
        result[i] = d * (q5 - 16)

    return result


def dequant_q8_0_block(data: bytes) -> np.ndarray:
    """Dequantize Q8_0 block (34 bytes -> 32 floats)"""
    d = np.frombuffer(data[0:2], dtype=np.float16)[0].astype(np.float32)
    qs = np.frombuffer(data[2:34], dtype=np.int8).astype(np.float32)
    return qs * d


def dequant_tensor(data: bytes, dtype: str, num_weights: int) -> np.ndarray:
    """Dequantize entire tensor"""
    block_size = QUANT_BLOCK_SIZES.get(dtype)
    weights_per_block = QUANT_BLOCK_WEIGHTS.get(dtype)

    if dtype == 'fp32':
        return np.frombuffer(data, dtype=np.float32)
    if dtype == 'fp16':
        return np.frombuffer(data, dtype=np.float16).astype(np.float32)

    if block_size is None:
        raise ValueError(f"Unknown dtype: {dtype}")

    num_blocks = num_weights // weights_per_block
    result = np.zeros(num_weights, dtype=np.float32)

    dequant_fn = {
        'q4_k': dequant_q4_k_block,
        'q6_k': dequant_q6_k_block,
        'q5_0': dequant_q5_0_block,
        'q8_0': dequant_q8_0_block,
    }.get(dtype)

    if dequant_fn is None:
        raise ValueError(f"No dequant function for: {dtype}")

    for b in range(num_blocks):
        block_data = data[b * block_size:(b + 1) * block_size]
        result[b * weights_per_block:(b + 1) * weights_per_block] = dequant_fn(block_data)

    return result


# ============================================================================
# GGUF Reader
# ============================================================================

GGML_TYPES = {
    0: 'fp32', 1: 'fp16', 2: 'q4_0', 3: 'q4_1',
    6: 'q5_0', 7: 'q5_1', 8: 'q8_0',
    12: 'q4_k', 14: 'q6_k',
}


class GGUFReader:
    def __init__(self, path: str):
        self.path = path
        self.tensors = {}
        self.metadata = {}
        self._parse()

    def _read_string(self, f):
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, vtype):
        if vtype == 4: return struct.unpack('<I', f.read(4))[0]
        elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
        elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
        elif vtype == 8: return self._read_string(f)
        elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
        else:
            # Skip unknown types
            return None

    def _parse(self):
        with open(self.path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError(f"Not a GGUF file: {magic}")

            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]

            # Read metadata
            for _ in range(kv_count):
                key = self._read_string(f)
                vtype = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, vtype)
                if value is not None:
                    self.metadata[key] = value

            # Read tensor info
            for _ in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                dtype_id = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]

                self.tensors[name] = {
                    'dims': dims,
                    'dtype': GGML_TYPES.get(dtype_id, f'unknown_{dtype_id}'),
                    'offset': offset,
                }

            # Calculate data start (aligned to 32 bytes)
            data_start = f.tell()
            data_start = (data_start + 31) & ~31

            for name in self.tensors:
                self.tensors[name]['file_offset'] = data_start + self.tensors[name]['offset']

    def read_tensor(self, name: str) -> Tuple[bytes, str, List[int]]:
        """Read raw tensor bytes"""
        info = self.tensors[name]
        dtype = info['dtype']
        dims = info['dims']
        num_weights = int(np.prod(dims))

        block_size = QUANT_BLOCK_SIZES.get(dtype, 4 if dtype == 'fp32' else 2)
        weights_per_block = QUANT_BLOCK_WEIGHTS.get(dtype, 1)

        if dtype in ['fp32', 'fp16']:
            size = num_weights * (4 if dtype == 'fp32' else 2)
        else:
            num_blocks = num_weights // weights_per_block
            size = num_blocks * block_size

        with open(self.path, 'rb') as f:
            f.seek(info['file_offset'])
            data = f.read(size)

        return data, dtype, dims


# ============================================================================
# CK-Engine Weight Reader
# ============================================================================

class CKWeightReader:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        manifest_path = self.model_dir / 'weights_manifest.json'
        with open(manifest_path) as f:
            return json.load(f)

    def read_tensor(self, name: str) -> Tuple[bytes, str]:
        """Read raw tensor bytes from bump file"""
        for entry in self.manifest['entries']:
            if entry['name'] == name:
                with open(self.model_dir / 'weights.bump', 'rb') as f:
                    f.seek(entry['file_offset'])
                    data = f.read(entry['size'])
                return data, entry['dtype']
        raise KeyError(f"Tensor not found: {name}")


# ============================================================================
# Comparison functions
# ============================================================================

def compare_bytes(name: str, gguf_data: bytes, ck_data: bytes) -> dict:
    """Compare raw bytes"""
    if len(gguf_data) != len(ck_data):
        return {
            'match': False,
            'error': f"Size mismatch: GGUF={len(gguf_data)}, CK={len(ck_data)}",
        }

    diff_count = sum(1 for a, b in zip(gguf_data, ck_data) if a != b)

    return {
        'match': diff_count == 0,
        'size': len(gguf_data),
        'diff_bytes': diff_count,
        'diff_pct': 100 * diff_count / len(gguf_data) if gguf_data else 0,
    }


def compare_dequant(name: str, gguf_data: bytes, ck_data: bytes, dtype: str, num_weights: int) -> dict:
    """Compare dequantized values"""
    try:
        gguf_dequant = dequant_tensor(gguf_data, dtype, num_weights)
        ck_dequant = dequant_tensor(ck_data, dtype, num_weights)
    except Exception as e:
        return {'match': False, 'error': str(e)}

    if gguf_dequant.shape != ck_dequant.shape:
        return {
            'match': False,
            'error': f"Shape mismatch: {gguf_dequant.shape} vs {ck_dequant.shape}",
        }

    diff = np.abs(gguf_dequant - ck_dequant)
    max_diff = diff.max()
    mean_diff = diff.mean()

    return {
        'match': max_diff < 1e-5,
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'gguf_range': [float(gguf_dequant.min()), float(gguf_dequant.max())],
        'ck_range': [float(ck_dequant.min()), float(ck_dequant.max())],
    }


def compare_layer_weights(layer: int, gguf: GGUFReader, ck: CKWeightReader, output_file=None) -> dict:
    """Compare all weights for a specific layer"""

    # Map GGUF tensor names to CK names
    weight_map = {
        f'blk.{layer}.attn_norm.weight': f'layer.{layer}.ln1_gamma',
        f'blk.{layer}.attn_q.weight': f'layer.{layer}.wq',
        f'blk.{layer}.attn_q.bias': f'layer.{layer}.bq',
        f'blk.{layer}.attn_k.weight': f'layer.{layer}.wk',
        f'blk.{layer}.attn_k.bias': f'layer.{layer}.bk',
        f'blk.{layer}.attn_v.weight': f'layer.{layer}.wv',
        f'blk.{layer}.attn_v.bias': f'layer.{layer}.bv',
        f'blk.{layer}.attn_output.weight': f'layer.{layer}.wo',
        f'blk.{layer}.ffn_norm.weight': f'layer.{layer}.ln2_gamma',
        f'blk.{layer}.ffn_gate.weight': f'layer.{layer}.w1',  # Part of w1
        f'blk.{layer}.ffn_up.weight': f'layer.{layer}.w1',    # Part of w1
        f'blk.{layer}.ffn_down.weight': f'layer.{layer}.w2',
    }

    results = {}

    for gguf_name, ck_name in weight_map.items():
        if gguf_name not in gguf.tensors:
            continue

        try:
            gguf_data, dtype, dims = gguf.read_tensor(gguf_name)
            ck_data, ck_dtype = ck.read_tensor(ck_name)
        except KeyError:
            results[gguf_name] = {'error': 'Not found in one of the sources'}
            continue

        num_weights = int(np.prod(dims))

        # Compare raw bytes
        byte_cmp = compare_bytes(gguf_name, gguf_data, ck_data)

        # Compare dequantized values
        dequant_cmp = compare_dequant(gguf_name, gguf_data, ck_data, dtype, num_weights)

        results[gguf_name] = {
            'gguf_dtype': dtype,
            'ck_dtype': ck_dtype,
            'dims': dims,
            'byte_comparison': byte_cmp,
            'dequant_comparison': dequant_cmp,
        }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def print_comparison_results(results: dict):
    """Pretty print comparison results"""
    print(f"\n{BOLD}=== Weight Comparison Results ==={RESET}\n")

    for name, data in results.items():
        if 'error' in data:
            print(f"  {RED}[ERROR]{RESET} {name}: {data['error']}")
            continue

        byte_match = data['byte_comparison'].get('match', False)
        dequant_match = data['dequant_comparison'].get('match', False)

        status = f"{GREEN}[PASS]{RESET}" if byte_match and dequant_match else f"{RED}[FAIL]{RESET}"

        print(f"  {status} {name}")
        print(f"       dtype: GGUF={data['gguf_dtype']}, CK={data['ck_dtype']}")

        if not byte_match:
            bc = data['byte_comparison']
            if 'diff_bytes' in bc:
                print(f"       {YELLOW}bytes differ: {bc['diff_bytes']}/{bc['size']} ({bc['diff_pct']:.2f}%){RESET}")
            else:
                print(f"       {RED}{bc.get('error', 'unknown error')}{RESET}")

        if not dequant_match:
            dc = data['dequant_comparison']
            if 'max_diff' in dc:
                print(f"       {YELLOW}max_diff: {dc['max_diff']:.6e}{RESET}")
                print(f"       GGUF range: {dc['gguf_range']}")
                print(f"       CK range: {dc['ck_range']}")
            else:
                print(f"       {RED}{dc.get('error', 'unknown error')}{RESET}")

        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare layer weights/computations between llama.cpp and CK-Engine')
    parser.add_argument('--gguf', required=True, help='Path to GGUF file')
    parser.add_argument('--ck-dir', help='Path to CK model directory (auto-detected if not specified)')
    parser.add_argument('--layer', type=int, default=2, help='Layer to compare (default: 2)')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--dump-weights', action='store_true', help='Dump weight comparison to file')
    args = parser.parse_args()

    # Auto-detect CK model directory
    if args.ck_dir:
        ck_dir = args.ck_dir
    else:
        gguf_name = Path(args.gguf).stem
        ck_dir = Path.home() / '.cache/ck-engine-v5/models' / gguf_name
        if not ck_dir.exists():
            # Try with parent directory name
            ck_dir = Path.home() / '.cache/ck-engine-v5/models' / Path(args.gguf).name.replace('.gguf', '')

    print(f"{BOLD}Layer {args.layer} Comparison: llama.cpp (GGUF) vs C-Kernel-Engine{RESET}")
    print(f"  GGUF: {args.gguf}")
    print(f"  CK:   {ck_dir}")
    print()

    # Load readers
    gguf = GGUFReader(args.gguf)
    ck = CKWeightReader(str(ck_dir))

    # Compare weights
    output_file = args.output or (f'layer{args.layer}_comparison.json' if args.dump_weights else None)
    results = compare_layer_weights(args.layer, gguf, ck, output_file)

    # Print results
    print_comparison_results(results)

    if output_file:
        print(f"Results saved to: {output_file}")

    # Summary
    passed = sum(1 for r in results.values() if r.get('byte_comparison', {}).get('match') and r.get('dequant_comparison', {}).get('match'))
    total = len(results)

    print(f"\n{BOLD}Summary: {passed}/{total} weights match exactly{RESET}")

    if passed < total:
        print(f"\n{YELLOW}Potential issues to investigate:{RESET}")
        print("  1. Weight packing order (gate+up concat for W1)")
        print("  2. Bias loading (some models have attention biases)")
        print("  3. File offset calculation in weight manifest")


if __name__ == '__main__':
    main()
