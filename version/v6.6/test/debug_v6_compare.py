#!/usr/bin/env python3
"""
Debug script to compare v6.5 vs v6.6 at key checkpoints:
1. Token embedding weights
2. First layer weights (ln1_gamma)
3. Outputs of embedding forward
"""

import struct
import json
import sys

def read_bytes(path, offset, size):
    """Read bytes from file at given offset."""
    with open(path, 'rb') as f:
        f.seek(offset)
        return f.read(size)

def bytes_to_floats(data):
    """Convert bytes to list of floats."""
    n = len(data) // 4
    return struct.unpack(f'<{n}f', data)

def bytes_to_q8_0_block(data):
    """Parse a Q8_0 block: 1 half-float scale + 32 int8 values."""
    # Q8_0: 2 bytes scale (fp16) + 32 bytes (int8 quants) = 34 bytes per block
    if len(data) < 34:
        return None, None
    scale_bytes = data[:2]
    quants = data[2:34]
    # fp16 to float
    import numpy as np
    scale = np.frombuffer(scale_bytes, dtype=np.float16)[0]
    qvals = np.frombuffer(quants, dtype=np.int8)
    return float(scale), qvals.tolist()

def main():
    v65_dir = "/home/antshiv/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
    v66_dir = "/home/antshiv/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

    # Load manifests
    v65_manifest = json.load(open(f"{v65_dir}/weights_manifest.json"))
    v66_manifest = json.load(open(f"{v66_dir}/weights_manifest.json"))

    v65_entries = {e['name']: e for e in v65_manifest.get('entries', [])}
    v66_entries = {e['name']: e for e in v66_manifest.get('entries', [])}

    print("=" * 60)
    print("COMPARING v6.5 vs v6.6 BUMP FILES")
    print("=" * 60)

    # Check bump file headers
    v65_bump = f"{v65_dir}/weights.bump"
    v66_bump = f"{v66_dir}/weights.bump"

    print("\n=== BUMP FILE HEADERS ===")
    v65_magic = read_bytes(v65_bump, 0, 8)
    v66_magic = read_bytes(v66_bump, 0, 8)
    print(f"v6.5 magic: {v65_magic}")
    print(f"v6.6 magic: {v66_magic}")

    # Compare token_emb
    print("\n=== TOKEN EMBEDDING (first Q8_0 block = token_id 0) ===")

    v65_tok = v65_entries['token_emb']
    v66_tok = v66_entries['token_emb']

    print(f"v6.5 token_emb: offset={v65_tok['file_offset']}, size={v65_tok['size']}")
    print(f"v6.6 token_emb: offset={v66_tok['file_offset']}, size={v66_tok['size']}")

    # Read first 34 bytes (one Q8_0 block)
    v65_emb_data = read_bytes(v65_bump, v65_tok['file_offset'], 340)
    v66_emb_data = read_bytes(v66_bump, v66_tok['file_offset'], 340)

    print(f"\nv6.5 first 20 bytes: {v65_emb_data[:20].hex()}")
    print(f"v6.6 first 20 bytes: {v66_emb_data[:20].hex()}")

    # Parse Q8_0 blocks
    v65_scale, v65_quants = bytes_to_q8_0_block(v65_emb_data)
    v66_scale, v66_quants = bytes_to_q8_0_block(v66_emb_data)

    print(f"\nv6.5 Q8_0 block 0: scale={v65_scale:.6f}, quants[:8]={v65_quants[:8]}")
    print(f"v6.6 Q8_0 block 0: scale={v66_scale:.6f}, quants[:8]={v66_quants[:8]}")

    if v65_emb_data == v66_emb_data:
        print("\n✓ Token embedding data MATCHES between v6.5 and v6.6")
    else:
        print("\n✗ Token embedding data DIFFERS!")
        # Find first difference
        for i in range(min(len(v65_emb_data), len(v66_emb_data))):
            if v65_emb_data[i] != v66_emb_data[i]:
                print(f"  First difference at byte {i}: v6.5={v65_emb_data[i]:02x}, v6.6={v66_emb_data[i]:02x}")
                break

    # Compare layer 0 weights
    print("\n=== LAYER 0 LN1_GAMMA (first 8 floats) ===")

    v65_ln1 = v65_entries['layer.0.ln1_gamma']
    v66_ln1 = v66_entries['layer.0.ln1_gamma']

    print(f"v6.5 ln1_gamma: offset={v65_ln1['file_offset']}, size={v65_ln1['size']}")
    print(f"v6.6 ln1_gamma: offset={v66_ln1['file_offset']}, size={v66_ln1['size']}")

    v65_ln1_data = read_bytes(v65_bump, v65_ln1['file_offset'], 32)
    v66_ln1_data = read_bytes(v66_bump, v66_ln1['file_offset'], 32)

    v65_floats = bytes_to_floats(v65_ln1_data)
    v66_floats = bytes_to_floats(v66_ln1_data)

    print(f"v6.5 ln1_gamma[:8]: {[f'{x:.6f}' for x in v65_floats[:8]]}")
    print(f"v6.6 ln1_gamma[:8]: {[f'{x:.6f}' for x in v66_floats[:8]]}")

    if v65_ln1_data == v66_ln1_data:
        print("\n✓ Layer 0 ln1_gamma MATCHES")
    else:
        print("\n✗ Layer 0 ln1_gamma DIFFERS!")

    # Compare WQ weights
    print("\n=== LAYER 0 WQ (first Q5_0 block) ===")

    v65_wq = v65_entries['layer.0.wq']
    v66_wq = v66_entries['layer.0.wq']

    print(f"v6.5 wq: offset={v65_wq['file_offset']}, dtype={v65_wq['dtype']}, size={v65_wq['size']}")
    print(f"v6.6 wq: offset={v66_wq['file_offset']}, dtype={v66_wq['dtype']}, size={v66_wq['size']}")

    v65_wq_data = read_bytes(v65_bump, v65_wq['file_offset'], 64)
    v66_wq_data = read_bytes(v66_bump, v66_wq['file_offset'], 64)

    print(f"v6.5 wq first 32 bytes: {v65_wq_data[:32].hex()}")
    print(f"v6.6 wq first 32 bytes: {v66_wq_data[:32].hex()}")

    if v65_wq_data == v66_wq_data:
        print("\n✓ Layer 0 WQ MATCHES")
    else:
        print("\n✗ Layer 0 WQ DIFFERS!")

    # Check for pos_emb written to file
    print("\n=== POS_EMB CHECK ===")

    # In v6.5, pos_emb is written after vocab_merges
    v65_merges = v65_entries.get('vocab_merges', {})
    v65_pos_emb_expected = v65_merges.get('file_offset', 0) + v65_merges.get('size', 0)

    v66_merges = v66_entries.get('vocab_merges', {})
    v66_pos_emb_expected = v66_merges.get('file_offset', 0) + v66_merges.get('size', 0)

    print(f"v6.5 after vocab_merges: offset {v65_pos_emb_expected} (expected pos_emb start)")
    print(f"v6.6 after vocab_merges: offset {v66_pos_emb_expected} (expected layer.0 or pos_emb)")

    # Check what comes after vocab_merges
    v65_next_data = read_bytes(v65_bump, v65_pos_emb_expected, 32)
    v66_next_data = read_bytes(v66_bump, v66_pos_emb_expected, 32)

    v65_next_floats = bytes_to_floats(v65_next_data)
    v66_next_floats = bytes_to_floats(v66_next_data)

    print(f"v6.5 bytes after vocab_merges: {v65_next_data[:16].hex()}")
    print(f"v6.5 as floats: {[f'{x:.6f}' for x in v65_next_floats[:4]]}")

    print(f"v6.6 bytes after vocab_merges: {v66_next_data[:16].hex()}")
    print(f"v6.6 as floats: {[f'{x:.6f}' for x in v66_next_floats[:4]]}")

    # Check layer.0.ln1_gamma position vs expected
    print("\n=== OFFSET GAP ANALYSIS ===")
    v65_gap = v65_ln1['file_offset'] - v65_pos_emb_expected
    v66_gap = v66_ln1['file_offset'] - v66_pos_emb_expected

    print(f"v6.5 gap (vocab_merges end to layer.0.ln1): {v65_gap} bytes ({v65_gap / 1024 / 1024:.1f} MB)")
    print(f"v6.6 gap (vocab_merges end to layer.0.ln1): {v66_gap} bytes ({v66_gap / 1024 / 1024:.1f} MB)")

    # The gap difference tells us about pos_emb
    gap_diff = v65_gap - v66_gap
    print(f"Gap difference: {gap_diff} bytes ({gap_diff / 1024 / 1024:.1f} MB)")

    # This should be the pos_emb size if v6.5 has it and v6.6 doesn't
    context_len = 32768
    embed_dim = 896
    expected_pos_emb = context_len * embed_dim * 4  # fp32
    print(f"Expected pos_emb size (32768 * 896 * 4): {expected_pos_emb} bytes ({expected_pos_emb / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
