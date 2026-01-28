#!/usr/bin/env python3
"""
Layer-by-layer test for v6.6 to identify where divergence occurs.
Tests each step from token embedding through layer 0.

Usage:
    python test_layer_by_layer.py

Compares:
1. Token embedding output
2. RMSNorm layer 0 output
3. Q/K/V projections
4. Attention output
5. MLP output
"""

import ctypes
import json
import os
import struct
import subprocess
import sys
import numpy as np
from pathlib import Path

# Model cache directory
V66_DIR = Path("/home/antshiv/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
V65_DIR = Path("/home/antshiv/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")

def load_q8_0_embedding(bump_path, offset, token_id, embed_dim=896):
    """Load and dequantize Q8_0 embedding for a specific token."""
    # Q8_0 format: 32 values per block
    block_size = 32
    num_blocks = (embed_dim + 31) // 32
    bytes_per_block = 2 + 32  # fp16 scale + 32 int8 quants

    # Calculate row offset
    row_start = offset + token_id * num_blocks * bytes_per_block

    values = []
    with open(bump_path, 'rb') as f:
        for b in range(num_blocks):
            f.seek(row_start + b * bytes_per_block)
            block_data = f.read(bytes_per_block)

            # Parse fp16 scale
            scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
            # Parse int8 quants
            quants = np.frombuffer(block_data[2:], dtype=np.int8)
            # Dequantize
            dequant = quants.astype(np.float32) * float(scale)
            values.extend(dequant.tolist())

    return np.array(values[:embed_dim], dtype=np.float32)

def load_fp32_vector(bump_path, offset, count):
    """Load fp32 vector from bump file."""
    with open(bump_path, 'rb') as f:
        f.seek(offset)
        data = f.read(count * 4)
    return np.frombuffer(data, dtype=np.float32)

def rmsnorm(x, gamma, eps=1e-6):
    """Compute RMSNorm."""
    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)

    # RMS normalization
    variance = np.mean(x ** 2)
    x_norm = x / np.sqrt(variance + eps)

    return x_norm * gamma

def test_v66_embedding():
    """Test v6.6 embedding lookup."""
    print("\n" + "="*60)
    print("TEST 1: Token Embedding Lookup")
    print("="*60)

    # Load manifest
    manifest = json.load(open(V66_DIR / "weights_manifest.json"))
    entries = {e['name']: e for e in manifest.get('entries', [])}

    tok_entry = entries['token_emb']
    print(f"Token embedding: offset={tok_entry['file_offset']}, dtype={tok_entry['dtype']}")

    # Test token 0 (often <unk> or similar)
    test_tokens = [0, 1, 100, 1000]

    v66_bump = V66_DIR / "weights.bump"
    v65_bump = V65_DIR / "weights.bump"

    v65_manifest = json.load(open(V65_DIR / "weights_manifest.json"))
    v65_entries = {e['name']: e for e in v65_manifest.get('entries', [])}
    v65_tok = v65_entries['token_emb']

    for token_id in test_tokens:
        v66_emb = load_q8_0_embedding(v66_bump, tok_entry['file_offset'], token_id)
        v65_emb = load_q8_0_embedding(v65_bump, v65_tok['file_offset'], token_id)

        diff = np.abs(v66_emb - v65_emb).max()
        mean_diff = np.abs(v66_emb - v65_emb).mean()

        print(f"\nToken {token_id}:")
        print(f"  v6.6 first 5: {v66_emb[:5]}")
        print(f"  v6.5 first 5: {v65_emb[:5]}")
        print(f"  Max diff: {diff:.8f}, Mean diff: {mean_diff:.8f}")

        if diff < 1e-6:
            print("  ✓ MATCH")
        else:
            print("  ✗ MISMATCH")

def test_v66_rmsnorm():
    """Test v6.6 RMSNorm for layer 0."""
    print("\n" + "="*60)
    print("TEST 2: Layer 0 RMSNorm")
    print("="*60)

    # Load manifests
    v66_manifest = json.load(open(V66_DIR / "weights_manifest.json"))
    v65_manifest = json.load(open(V65_DIR / "weights_manifest.json"))

    v66_entries = {e['name']: e for e in v66_manifest.get('entries', [])}
    v65_entries = {e['name']: e for e in v65_manifest.get('entries', [])}

    # Load ln1_gamma for layer 0
    v66_ln1 = v66_entries['layer.0.ln1_gamma']
    v65_ln1 = v65_entries['layer.0.ln1_gamma']

    print(f"v6.6 ln1_gamma: offset={v66_ln1['file_offset']}")
    print(f"v6.5 ln1_gamma: offset={v65_ln1['file_offset']}")

    v66_gamma = load_fp32_vector(V66_DIR / "weights.bump", v66_ln1['file_offset'], 896)
    v65_gamma = load_fp32_vector(V65_DIR / "weights.bump", v65_ln1['file_offset'], 896)

    gamma_diff = np.abs(v66_gamma - v65_gamma).max()
    print(f"\nGamma comparison:")
    print(f"  v6.6 first 5: {v66_gamma[:5]}")
    print(f"  v6.5 first 5: {v65_gamma[:5]}")
    print(f"  Max diff: {gamma_diff:.8f}")

    if gamma_diff < 1e-6:
        print("  ✓ Gamma values MATCH")
    else:
        print("  ✗ Gamma values MISMATCH")

    # Now test RMSNorm computation
    # Use token 0's embedding as input
    tok_entry = v66_entries['token_emb']
    input_emb = load_q8_0_embedding(V66_DIR / "weights.bump", tok_entry['file_offset'], 100)

    output = rmsnorm(input_emb, v66_gamma)

    print(f"\nRMSNorm test (token 100 embedding -> layer 0 input):")
    print(f"  Input first 5: {input_emb[:5]}")
    print(f"  Output first 5: {output[:5]}")
    print(f"  Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

def test_v66_layer0_weights():
    """Compare all layer 0 weights between v6.5 and v6.6."""
    print("\n" + "="*60)
    print("TEST 3: Layer 0 All Weights Comparison")
    print("="*60)

    v66_manifest = json.load(open(V66_DIR / "weights_manifest.json"))
    v65_manifest = json.load(open(V65_DIR / "weights_manifest.json"))

    v66_entries = {e['name']: e for e in v66_manifest.get('entries', [])}
    v65_entries = {e['name']: e for e in v65_manifest.get('entries', [])}

    # List of layer 0 weights
    layer0_weights = [
        'layer.0.ln1_gamma', 'layer.0.ln2_gamma',
        'layer.0.wq', 'layer.0.wk', 'layer.0.wv', 'layer.0.wo',
        'layer.0.bq', 'layer.0.bk', 'layer.0.bv', 'layer.0.bo',
        'layer.0.w1', 'layer.0.w2',
        'layer.0.b1', 'layer.0.b2',
    ]

    v66_bump = V66_DIR / "weights.bump"
    v65_bump = V65_DIR / "weights.bump"

    all_match = True
    for name in layer0_weights:
        v66_e = v66_entries.get(name)
        v65_e = v65_entries.get(name)

        if not v66_e or not v65_e:
            print(f"{name}: MISSING (v66={v66_e is not None}, v65={v65_e is not None})")
            continue

        # Read first 64 bytes from each
        with open(v66_bump, 'rb') as f:
            f.seek(v66_e['file_offset'])
            v66_data = f.read(min(64, v66_e['size']))

        with open(v65_bump, 'rb') as f:
            f.seek(v65_e['file_offset'])
            v65_data = f.read(min(64, v65_e['size']))

        match = v66_data == v65_data
        status = "✓" if match else "✗"
        if not match:
            all_match = False

        print(f"{status} {name}: offset v66={v66_e['file_offset']}, v65={v65_e['file_offset']}, size={v66_e['size']}")
        if not match:
            print(f"    v66: {v66_data[:16].hex()}")
            print(f"    v65: {v65_data[:16].hex()}")

    if all_match:
        print("\n✓ All layer 0 weights MATCH between v6.5 and v6.6")
    else:
        print("\n✗ Some layer 0 weights DIFFER!")

def check_c_code_offsets():
    """Check if C code W_* defines match manifest file_offset."""
    print("\n" + "="*60)
    print("TEST 4: C Code Offset Verification")
    print("="*60)

    model_c = V66_DIR / "model_v6_6.c"
    manifest = json.load(open(V66_DIR / "weights_manifest.json"))
    entries = {e['name']: e for e in manifest.get('entries', [])}

    # Parse W_* defines from C code
    import re
    c_defines = {}
    with open(model_c) as f:
        for line in f:
            m = re.match(r'#define\s+(W_\w+)\s+(\d+)', line)
            if m:
                name = m.group(1)
                value = int(m.group(2))
                c_defines[name] = value

    print(f"Found {len(c_defines)} W_* defines in C code")

    # Map C define names to manifest names
    name_map = {
        'W_TOKEN_EMB': 'token_emb',
        'W_LAYER_0_LN1_GAMMA': 'layer.0.ln1_gamma',
        'W_LAYER_0_LN2_GAMMA': 'layer.0.ln2_gamma',
        'W_LAYER_0_WQ': 'layer.0.wq',
        'W_LAYER_0_WK': 'layer.0.wk',
        'W_LAYER_0_WV': 'layer.0.wv',
        'W_LAYER_0_WO': 'layer.0.wo',
        'W_LAYER_0_BQ': 'layer.0.bq',
        'W_LAYER_0_W1': 'layer.0.w1',
        'W_LAYER_0_W2': 'layer.0.w2',
    }

    all_match = True
    for c_name, manifest_name in name_map.items():
        c_offset = c_defines.get(c_name)
        m_entry = entries.get(manifest_name, {})
        m_offset = m_entry.get('file_offset')

        if c_offset is None:
            print(f"? {c_name}: not found in C code")
            continue
        if m_offset is None:
            print(f"? {c_name}: {manifest_name} not found in manifest")
            continue

        match = c_offset == m_offset
        status = "✓" if match else "✗"
        if not match:
            all_match = False

        print(f"{status} {c_name}: C={c_offset}, manifest={m_offset}")
        if not match:
            print(f"    Diff: {c_offset - m_offset}")

    if all_match:
        print("\n✓ All checked C defines match manifest offsets")
    else:
        print("\n✗ Some C defines differ from manifest!")

def main():
    print("="*60)
    print("V6.6 LAYER-BY-LAYER VERIFICATION")
    print("="*60)

    test_v66_embedding()
    test_v66_rmsnorm()
    test_v66_layer0_weights()
    check_c_code_offsets()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("If all tests pass, the bump file contents are correct.")
    print("The issue would then be in runtime loading or kernel execution.")

if __name__ == "__main__":
    main()
