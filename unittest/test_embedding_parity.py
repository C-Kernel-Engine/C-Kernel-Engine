#!/usr/bin/env python3
"""
test_embedding_parity.py - Compare embedding lookup between GGUF and C-Kernel bump format

Tests:
1. GGUF embedding weight dequantization (using gguf library)
2. C-Kernel bump file embedding lookup
3. Compares the two to verify parity

Usage:
  python unittest/test_embedding_parity.py \
    --gguf ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
    --bump ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump \
    --manifest ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights_manifest.json \
    --token 9707
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path

# Q8_0 block size: 32 values per block
Q8_0_BLOCK_SIZE = 32


def dequant_q8_0_block(block_data: bytes) -> np.ndarray:
    """
    Dequantize a single Q8_0 block (34 bytes -> 32 floats).
    Q8_0 format: 2 bytes scale (fp16) + 32 bytes quants (int8)
    """
    if len(block_data) != 34:
        raise ValueError(f"Q8_0 block must be 34 bytes, got {len(block_data)}")

    # Scale is fp16 (2 bytes)
    scale = struct.unpack('<e', block_data[0:2])[0]

    # Quants are int8 (32 bytes)
    quants = np.frombuffer(block_data[2:34], dtype=np.int8)

    # Dequantize: float = quant * scale
    return quants.astype(np.float32) * scale


def dequant_q8_0_row(quant_data: bytes, n_elements: int) -> np.ndarray:
    """Dequantize a row of Q8_0 data."""
    n_blocks = (n_elements + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE
    bytes_per_block = 34  # 2 (scale) + 32 (quants)

    result = []
    for i in range(n_blocks):
        block_start = i * bytes_per_block
        block_end = block_start + bytes_per_block
        if block_end > len(quant_data):
            break
        block = quant_data[block_start:block_end]
        result.extend(dequant_q8_0_block(block))

    return np.array(result[:n_elements], dtype=np.float32)


def read_gguf_embedding(gguf_path: str, token_id: int) -> np.ndarray:
    """Read and dequantize embedding for a token from GGUF file."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)

    # Find token_embd.weight tensor
    emb_tensor = None
    for tensor in reader.tensors:
        if tensor.name == "token_embd.weight":
            emb_tensor = tensor
            break

    if emb_tensor is None:
        raise ValueError("token_embd.weight not found in GGUF")

    print(f"[GGUF] token_embd.weight:")
    print(f"  - tensor_type: {emb_tensor.tensor_type}")
    print(f"  - shape: {emb_tensor.shape}")
    print(f"  - n_elements: {emb_tensor.n_elements}")

    # Shape is (hidden_dim, vocab_size) in GGUF metadata
    hidden_dim = int(emb_tensor.shape[0])
    vocab_size = int(emb_tensor.shape[1])
    print(f"  - hidden_dim: {hidden_dim}, vocab_size: {vocab_size}")

    if token_id >= vocab_size:
        raise ValueError(f"Token ID {token_id} >= vocab_size {vocab_size}")

    # The gguf library provides data as (vocab_size, bytes_per_row) shaped array
    raw_data = emb_tensor.data
    print(f"  - data shape: {raw_data.shape}")
    print(f"  - data dtype: {raw_data.dtype}")

    # For Q8_0: each row is hidden_dim elements packed into blocks
    n_blocks_per_row = (hidden_dim + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE
    bytes_per_row = n_blocks_per_row * 34

    print(f"  - bytes_per_row: {bytes_per_row}")

    # Extract row for token_id - the data is already in (vocab_size, bytes_per_row) shape
    row_data = raw_data[token_id]

    # Dequantize
    embedding = dequant_q8_0_row(bytes(row_data), hidden_dim)

    return embedding


def read_bump_embedding(bump_path: str, manifest_path: str, token_id: int) -> np.ndarray:
    """Read and dequantize embedding for a token from bump file."""

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Find token_emb entry
    emb_entry = None
    for entry in manifest['entries']:
        if entry['name'] == 'token_emb':
            emb_entry = entry
            break

    if emb_entry is None:
        raise ValueError("token_emb not found in manifest")

    print(f"[BUMP] token_emb:")
    print(f"  - dtype: {emb_entry['dtype']}")
    print(f"  - file_offset: {emb_entry['file_offset']}")
    print(f"  - size: {emb_entry['size']}")

    # Read bump file
    with open(bump_path, 'rb') as f:
        # Read header to get dimensions
        magic = f.read(4)
        if magic != b'BUMP':
            raise ValueError(f"Invalid bump magic: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        header_size = struct.unpack('<I', f.read(4))[0]

        print(f"  - bump version: {version}")
        print(f"  - header_size: {header_size}")

        # Seek to embedding data
        f.seek(emb_entry['file_offset'])
        emb_data = f.read(emb_entry['size'])

    # We need hidden_dim - let's get it from config
    config_path = Path(bump_path).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    hidden_dim = config.get('hidden_size', config.get('embedding_length', 896))
    vocab_size = config.get('vocab_size', 151936)

    print(f"  - hidden_dim: {hidden_dim}, vocab_size: {vocab_size}")

    # For Q8_0: bytes_per_row = (hidden_dim / 32) * 34
    n_blocks_per_row = (hidden_dim + Q8_0_BLOCK_SIZE - 1) // Q8_0_BLOCK_SIZE
    bytes_per_row = n_blocks_per_row * 34

    print(f"  - bytes_per_row: {bytes_per_row}")

    # Extract row for token_id
    row_start = token_id * bytes_per_row
    row_end = row_start + bytes_per_row
    row_data = emb_data[row_start:row_end]

    # Dequantize
    embedding = dequant_q8_0_row(row_data, hidden_dim)

    return embedding


def compare_embeddings(gguf_emb: np.ndarray, bump_emb: np.ndarray) -> dict:
    """Compare two embedding vectors and return statistics."""
    diff = np.abs(gguf_emb - bump_emb)

    return {
        'max_diff': float(np.max(diff)),
        'mean_diff': float(np.mean(diff)),
        'num_mismatches': int(np.sum(diff > 1e-6)),
        'gguf_norm': float(np.linalg.norm(gguf_emb)),
        'bump_norm': float(np.linalg.norm(bump_emb)),
        'gguf_first5': gguf_emb[:5].tolist(),
        'bump_first5': bump_emb[:5].tolist(),
        'gguf_last5': gguf_emb[-5:].tolist(),
        'bump_last5': bump_emb[-5:].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Test embedding parity between GGUF and bump")
    parser.add_argument("--gguf", required=True, help="Path to GGUF file")
    parser.add_argument("--bump", required=True, help="Path to bump weights file")
    parser.add_argument("--manifest", required=True, help="Path to weights manifest JSON")
    parser.add_argument("--token", type=int, default=9707, help="Token ID to test (default: 9707 = 'hello')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Embedding Parity Test")
    print(f"=" * 60)
    print(f"Token ID: {args.token}")
    print()

    # Read from GGUF
    print("Reading GGUF embedding...")
    gguf_emb = read_gguf_embedding(args.gguf, args.token)
    print(f"  Shape: {gguf_emb.shape}")
    print()

    # Read from bump
    print("Reading BUMP embedding...")
    bump_emb = read_bump_embedding(args.bump, args.manifest, args.token)
    print(f"  Shape: {bump_emb.shape}")
    print()

    # Compare
    print("Comparing embeddings...")
    stats = compare_embeddings(gguf_emb, bump_emb)

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"  Max difference:     {stats['max_diff']:.10f}")
    print(f"  Mean difference:    {stats['mean_diff']:.10f}")
    print(f"  Num mismatches:     {stats['num_mismatches']} / {len(gguf_emb)}")
    print(f"  GGUF norm:          {stats['gguf_norm']:.6f}")
    print(f"  BUMP norm:          {stats['bump_norm']:.6f}")
    print()
    print(f"  GGUF first 5:       {stats['gguf_first5']}")
    print(f"  BUMP first 5:       {stats['bump_first5']}")
    print()
    print(f"  GGUF last 5:        {stats['gguf_last5']}")
    print(f"  BUMP last 5:        {stats['bump_last5']}")

    # Pass/fail
    if stats['max_diff'] < 1e-5:
        print(f"\n{'='*60}")
        print("PASS: Embeddings match!")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("FAIL: Embedding mismatch detected!")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    exit(main())
