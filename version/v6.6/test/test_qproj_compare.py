#!/usr/bin/env python3
"""
Compare Q projection output between v6.5 and v6.6.
"""

import ctypes
import sys
import struct
import numpy as np
from pathlib import Path

V66_DIR = Path("/home/antshiv/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
V65_DIR = Path("/home/antshiv/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")

def load_fp32_from_bump(path, offset, count):
    """Load fp32 values from bump file."""
    with open(path, 'rb') as f:
        f.seek(offset)
        return np.frombuffer(f.read(count * 4), dtype=np.float32).copy()

def load_q5_0_weights(path, offset, rows, cols):
    """Load Q5_0 matrix and dequantize."""
    # Q5_0 format: 32 values per block
    # 2 bytes scale (fp16) + 4 bytes quants (5-bit packed) = 22 bytes per block
    block_size = 32
    bytes_per_block = 2 + 4 + 16  # scale + high_bits + quants = 22 bytes

    num_blocks_per_row = (cols + 31) // 32

    weights = []
    with open(path, 'rb') as f:
        for row in range(rows):
            row_values = []
            for b in range(num_blocks_per_row):
                f.seek(offset + (row * num_blocks_per_row + b) * bytes_per_block)
                block = f.read(bytes_per_block)
                if len(block) < bytes_per_block:
                    break

                # Parse scale (fp16)
                scale = np.frombuffer(block[:2], dtype=np.float16)[0]

                # Parse high bits (4 bytes = 32 bits, each is the 5th bit for one value)
                high_bits = np.frombuffer(block[2:6], dtype=np.uint8)
                high = np.zeros(32, dtype=np.int8)
                for i in range(4):
                    for j in range(8):
                        high[i*8 + j] = (high_bits[i] >> j) & 1

                # Parse quants (16 bytes = 32 4-bit values)
                quants_raw = np.frombuffer(block[6:22], dtype=np.uint8)
                low = np.zeros(32, dtype=np.int8)
                for i in range(16):
                    low[i*2] = quants_raw[i] & 0xF
                    low[i*2 + 1] = (quants_raw[i] >> 4) & 0xF

                # Combine: value = (high_bit << 4) | low_4bits - 16
                values = (high.astype(np.int32) << 4) | low.astype(np.int32)
                values = values - 16  # Center around 0
                dequant = values.astype(np.float32) * float(scale)
                row_values.extend(dequant.tolist())

            weights.append(row_values[:cols])

    return np.array(weights, dtype=np.float32)

def gemv_reference(W, x):
    """Reference GEMV: y = W @ x."""
    return W @ x

def test_qproj():
    """Test Q projection."""
    print("="*60)
    print("Q PROJECTION TEST")
    print("="*60)

    # First, get the input (layer_input after rmsnorm)
    # We'll compute this from the embedding
    import json

    # Load manifests
    v66_manifest = json.load(open(V66_DIR / "weights_manifest.json"))
    v65_manifest = json.load(open(V65_DIR / "weights_manifest.json"))

    v66_entries = {e['name']: e for e in v66_manifest.get('entries', [])}
    v65_entries = {e['name']: e for e in v65_manifest.get('entries', [])}

    # Load WQ weights (first 10 rows for quick test)
    print("\n=== Loading WQ weights ===")
    v66_wq_offset = v66_entries['layer.0.wq']['file_offset']
    v65_wq_offset = v65_entries['layer.0.wq']['file_offset']

    print(f"v6.6 WQ offset: {v66_wq_offset}")
    print(f"v6.5 WQ offset: {v65_wq_offset}")

    # Just compare the raw bytes first
    with open(V66_DIR / "weights.bump", 'rb') as f:
        f.seek(v66_wq_offset)
        v66_wq_bytes = f.read(64)

    with open(V65_DIR / "weights.bump", 'rb') as f:
        f.seek(v65_wq_offset)
        v65_wq_bytes = f.read(64)

    print(f"\nv6.6 WQ first 32 bytes: {v66_wq_bytes[:32].hex()}")
    print(f"v6.5 WQ first 32 bytes: {v65_wq_bytes[:32].hex()}")

    if v66_wq_bytes == v65_wq_bytes:
        print("✓ WQ raw bytes match")
    else:
        print("✗ WQ raw bytes differ!")
        return

    # Load BQ biases
    print("\n=== Loading BQ biases ===")
    v66_bq_offset = v66_entries['layer.0.bq']['file_offset']
    v65_bq_offset = v65_entries['layer.0.bq']['file_offset']

    v66_bq = load_fp32_from_bump(V66_DIR / "weights.bump", v66_bq_offset, 896)
    v65_bq = load_fp32_from_bump(V65_DIR / "weights.bump", v65_bq_offset, 896)

    print(f"v6.6 BQ first 5: {v66_bq[:5]}")
    print(f"v6.5 BQ first 5: {v65_bq[:5]}")

    if np.allclose(v66_bq, v65_bq):
        print("✓ BQ biases match")
    else:
        print("✗ BQ biases differ!")
        diff_idx = np.where(np.abs(v66_bq - v65_bq) > 1e-6)[0]
        print(f"  Differs at indices: {diff_idx[:10]}")

    # Now run actual inference and capture outputs
    print("\n=== Running inference for Q projection output ===")

    # Use the stop_seq feature to stop after Q projection
    # We need to load the model and run decode with stop_seq=3

    def run_model_with_stop(model_dir, stop_at):
        """Load model and run decode, stopping at specified step."""
        lib_path = model_dir / "ck-kernel-inference.so"
        if not lib_path.exists():
            lib_path = model_dir / "libmodel.so"

        lib = ctypes.CDLL(str(lib_path))

        # Setup signatures
        lib.ck_model_init.argtypes = [ctypes.c_char_p]
        lib.ck_model_init.restype = ctypes.c_int

        lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
        lib.ck_model_decode.restype = ctypes.c_int

        lib.ck_model_free.argtypes = []

        # Try to get internal state accessor
        try:
            lib.ck_model_get_activation.argtypes = [ctypes.c_char_p]
            lib.ck_model_get_activation.restype = ctypes.POINTER(ctypes.c_float)
            has_get_act = True
        except:
            has_get_act = False

        # Init
        weights_path = model_dir / "weights.bump"
        ret = lib.ck_model_init(str(weights_path).encode())
        if ret != 0:
            print(f"Failed to init model: {ret}")
            return None, None

        vocab_size = 151936
        logits = (ctypes.c_float * vocab_size)()

        # Run decode
        ret = lib.ck_model_decode(100, logits)

        result = np.array(logits)

        lib.ck_model_free()
        return result

    print("\nRunning v6.5 decode...")
    v65_logits = run_model_with_stop(V65_DIR, 0)

    print("Running v6.6 decode...")
    v66_logits = run_model_with_stop(V66_DIR, 0)

    if v65_logits is not None and v66_logits is not None:
        print("\n=== Final Logits Comparison ===")
        print(f"v6.5 logits: min={v65_logits.min():.4f}, max={v65_logits.max():.4f}")
        print(f"v6.6 logits: min={v66_logits.min():.4f}, max={v66_logits.max():.4f}")

        diff = np.abs(v65_logits - v66_logits)
        print(f"Max diff: {diff.max():.4f}")
        print(f"Mean diff: {diff.mean():.4f}")

        # Find where max diff is
        max_idx = np.argmax(diff)
        print(f"Max diff at index {max_idx}: v6.5={v65_logits[max_idx]:.4f}, v6.6={v66_logits[max_idx]:.4f}")

if __name__ == "__main__":
    test_qproj()
