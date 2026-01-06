#!/usr/bin/env python3
"""
Compare WQ weights between GGUF and CK bump file

The Q projection is 2.46x smaller than expected. This test checks if:
1. The raw Q4_K bytes are identical between GGUF and bump
2. The dequantized values match
"""

import ctypes
import numpy as np
from pathlib import Path
import sys
import json

BASE_DIR = Path("/home/antshiv/Workspace/C-Kernel-Engine")
MODEL_DIR = Path("/home/antshiv/.cache/ck-engine-v5/models/qwen2.5-3b-instruct-q4_k_m")
GGUF_PATH = BASE_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"

sys.path.insert(0, str(BASE_DIR / "scripts"))
import convert_gguf_to_bump as gguf

QK_K = 256
BLOCK_Q4_K_SIZE = 144

def main():
    print("=" * 70)
    print("WQ WEIGHT COMPARISON: GGUF vs CK Bump")
    print("=" * 70)

    # Load dequant function
    libggml = ctypes.CDLL(str(BASE_DIR / "llama.cpp/libggml_kernel_test.so"))
    libggml.test_dequant_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    # Read GGUF WQ using the gguf library
    print("\n[1] Reading WQ from GGUF...")
    with open(GGUF_PATH, "rb") as f:
        r = gguf.GGUFReader(f)
        magic = r._read_exact(4)
        if magic != b"GGUF":
            raise gguf.GGUFError("Invalid GGUF magic")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        # Skip KV pairs
        for _ in range(n_kv):
            key = r.key_str()
            vtype = r.u32()
            gguf._gguf_skip_value(r, vtype)

        # Read tensor infos
        tensors = {}
        for _ in range(n_tensors):
            name = r.key_str()
            n_dims = r.u32()
            dims = tuple(int(r.u64()) for _ in range(n_dims))
            ggml_type = r.u32()
            offset = r.u64()
            tensors[name] = gguf.TensorInfo(
                name=name,
                dims=dims,
                ggml_type=int(ggml_type),
                offset=int(offset),
            )

        # Find data start
        alignment = 32
        data_start = gguf.align_up(r.tell(), alignment)

        # Get WQ tensor
        wq_name = "blk.0.attn_q.weight"
        if wq_name not in tensors:
            raise ValueError(f"Tensor {wq_name} not found")

        wq_info = tensors[wq_name]
        print(f"GGUF tensor: {wq_name}")
        print(f"  dims: {wq_info.dims}")
        print(f"  ggml_type: {wq_info.ggml_type} ({gguf.ggml_type_name(wq_info.ggml_type)})")
        print(f"  offset: {wq_info.offset}")

        # Calculate size
        n_elements = wq_info.ne0 * wq_info.ne1
        n_blocks = n_elements // QK_K
        n_bytes = n_blocks * BLOCK_Q4_K_SIZE

        print(f"  elements: {n_elements}")
        print(f"  blocks: {n_blocks}")
        print(f"  bytes: {n_bytes}")

        # Read data
        f.seek(data_start + wq_info.offset)
        gguf_data = f.read(n_bytes)

    # Read bump WQ
    print("\n[2] Reading WQ from CK bump...")
    manifest_path = MODEL_DIR / "weights_manifest.json"
    with open(manifest_path) as mf:
        manifest = json.load(mf)

    wq_entry = None
    for e in manifest.get("entries", []):
        if e["name"] == "layer.0.wq":
            wq_entry = e
            break

    if wq_entry is None:
        raise ValueError("layer.0.wq not found in manifest")

    print(f"CK entry: layer.0.wq")
    print(f"  dtype: {wq_entry.get('dtype')}")
    print(f"  size: {wq_entry.get('size')}")
    print(f"  file_offset: {wq_entry.get('file_offset')}")

    bump_path = MODEL_DIR / "weights.bump"
    with open(bump_path, "rb") as bf:
        bf.seek(wq_entry["file_offset"])
        bump_data = bf.read(wq_entry["size"])

    # Compare raw bytes
    print("\n[3] Comparing raw Q4_K bytes...")
    if len(gguf_data) != len(bump_data):
        print(f"  SIZE MISMATCH: GGUF={len(gguf_data)}, bump={len(bump_data)}")
    else:
        print(f"  Size matches: {len(gguf_data)} bytes")
        if gguf_data == bump_data:
            print("  Raw bytes: IDENTICAL")
        else:
            # Find first difference
            for i, (a, b) in enumerate(zip(gguf_data, bump_data)):
                if a != b:
                    print(f"  First diff at byte {i}: GGUF=0x{a:02x}, bump=0x{b:02x}")
                    break

            # Count differences
            diffs = sum(1 for a, b in zip(gguf_data, bump_data) if a != b)
            print(f"  Total different bytes: {diffs} ({100*diffs/len(gguf_data):.2f}%)")

    # Dequantize and compare
    print("\n[4] Dequantizing and comparing...")

    gguf_dequant = np.zeros(n_elements, dtype=np.float32)
    bump_dequant = np.zeros(n_elements, dtype=np.float32)

    libggml.test_dequant_q4_k(
        gguf_data,
        gguf_dequant.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_elements
    )

    libggml.test_dequant_q4_k(
        bump_data,
        bump_dequant.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_elements
    )

    print(f"  GGUF dequant range: ({gguf_dequant.min():.6f}, {gguf_dequant.max():.6f})")
    print(f"  Bump dequant range: ({bump_dequant.min():.6f}, {bump_dequant.max():.6f})")

    diff = np.abs(gguf_dequant - bump_dequant)
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")

    if diff.max() < 1e-6:
        print("\n  DEQUANTIZED VALUES MATCH!")
    else:
        print("\n  DEQUANTIZED VALUES DIFFER!")

    return 0

if __name__ == "__main__":
    exit(main())
