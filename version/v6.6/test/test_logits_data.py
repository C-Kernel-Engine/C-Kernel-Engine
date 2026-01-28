#!/usr/bin/env python3
"""
Check the actual data used by the logits kernel (gemv_q8_0_q8_0).
- Token embedding weights (W_TOKEN_EMB)
- Quantized activation (A_EMBEDDED_INPUT after quantize_final_output)
"""

import ctypes
import numpy as np
import os
from pathlib import Path

cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
model_dirs = list(cache_dir.glob("*Qwen*")) + list(cache_dir.glob("*qwen*"))

if not model_dirs:
    print("Error: No Qwen model found in cache")
    exit(1)

model_dir = model_dirs[0]
lib_path = model_dir / "libmodel.so"
weights_path = model_dir / "weights.bump"

# Read layout
import json
layout_path = model_dir / "layout_decode.json"
with open(layout_path) as f:
    layout = json.load(f)

config = layout.get("config", {})
embed_dim = config.get("embed_dim", 896)
vocab_size = config.get("vocab_size", 151936)

# Get offsets
act_buffers = {}
for buf in layout.get("memory", {}).get("activations", {}).get("buffers", []):
    act_buffers[buf["name"]] = buf["abs_offset"]

weights_info = layout.get("memory", {}).get("weights", {})
weight_offsets = {}
for entry in weights_info.get("entries", []):
    weight_offsets[entry["name"]] = {
        "offset": entry.get("abs_offset", 0),
        "size": entry.get("size", 0),
        "dtype": entry.get("dtype", "unknown")
    }

print(f"Embed dim: {embed_dim}")
print(f"Vocab size: {vocab_size}")

# Q8_0 constants
QK8_0 = 32
BLOCK_Q8_0_SIZE = 34

# Load library
lib = ctypes.CDLL(str(lib_path))
lib.ck_model_init.argtypes = [ctypes.c_char_p]
lib.ck_model_init.restype = ctypes.c_int
lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
lib.ck_model_decode.restype = ctypes.c_int
lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
lib.ck_model_get_vocab_size.restype = ctypes.c_int

# Initialize
print("\n--- Initializing model ---")
ret = lib.ck_model_init(str(weights_path).encode())
print(f"Init result: {ret}")

base_ptr = lib.ck_model_get_base_ptr()

# Check token embedding weights
print("\n--- Checking W_TOKEN_EMB (token embeddings) ---")
if "token_emb" in weight_offsets:
    te = weight_offsets["token_emb"]
    print(f"Offset: {te['offset']}, Size: {te['size']}, Dtype: {te['dtype']}")

    # Read first few rows
    bytes_per_row = (embed_dim // QK8_0) * BLOCK_Q8_0_SIZE  # 28 * 34 = 952
    print(f"Bytes per row: {bytes_per_row}")

    # Check a few random token embeddings
    for token_id in [0, 100, 9707, 50000]:
        row_offset = te['offset'] + token_id * bytes_per_row
        ptr = ctypes.cast(base_ptr + row_offset, ctypes.POINTER(ctypes.c_uint8))
        raw = bytes([ptr[i] for i in range(BLOCK_Q8_0_SIZE)])

        scale = np.frombuffer(raw[:2], dtype=np.float16)[0]
        quants = np.frombuffer(raw[2:], dtype=np.int8)

        print(f"  Token {token_id}: scale={scale:.6f}, quants_range=[{quants.min()}, {quants.max()}]", end="")
        if np.isnan(scale) or np.isinf(scale):
            print(" [INVALID SCALE!]")
        else:
            print()

# Run decode and stop after quantize_final_output (op 626)
print("\n--- Running decode (stop at op 626 = after quantize_final_output) ---")
os.environ['CK_STOP_OP'] = '626'

vocab = lib.ck_model_get_vocab_size()
logits = (ctypes.c_float * vocab)()
lib.ck_model_decode(9707, logits)

del os.environ['CK_STOP_OP']

# Check A_EMBEDDED_INPUT (should contain Q8_0 quantized data after quantize_final_output)
print("\n--- Checking A_EMBEDDED_INPUT (quantized activation for logits) ---")
if "embedded_input" in act_buffers:
    emb_offset = act_buffers["embedded_input"]
    print(f"Offset: {emb_offset}")

    # The quantized data should be 952 bytes (28 blocks * 34 bytes)
    q8_size = (embed_dim // QK8_0) * BLOCK_Q8_0_SIZE
    ptr = ctypes.cast(base_ptr + emb_offset, ctypes.POINTER(ctypes.c_uint8))
    raw = bytes([ptr[i] for i in range(q8_size)])

    print(f"Reading {q8_size} bytes of Q8_0 data")

    # Parse each block
    num_blocks = embed_dim // QK8_0
    all_scales = []
    all_quants = []

    for b in range(num_blocks):
        offset = b * BLOCK_Q8_0_SIZE
        scale = np.frombuffer(raw[offset:offset+2], dtype=np.float16)[0]
        quants = np.frombuffer(raw[offset+2:offset+BLOCK_Q8_0_SIZE], dtype=np.int8)
        all_scales.append(float(scale))
        all_quants.extend(quants.tolist())

    all_scales = np.array(all_scales)
    all_quants = np.array(all_quants, dtype=np.int8)

    print(f"  Num blocks: {num_blocks}")
    print(f"  Scales: range=[{all_scales.min():.6f}, {all_scales.max():.6f}], mean={all_scales.mean():.6f}")
    print(f"  Scales NaN: {np.isnan(all_scales).sum()}, Inf: {np.isinf(all_scales).sum()}")
    print(f"  Quants: range=[{all_quants.min()}, {all_quants.max()}]")

    # Show first few blocks
    print("\n  First 3 blocks:")
    for b in range(min(3, num_blocks)):
        offset = b * BLOCK_Q8_0_SIZE
        scale = np.frombuffer(raw[offset:offset+2], dtype=np.float16)[0]
        quants = np.frombuffer(raw[offset+2:offset+BLOCK_Q8_0_SIZE], dtype=np.int8)
        print(f"    Block {b}: scale={scale:.6f}, quants={quants[:5]}...")

    # Also read as FP32 to see what was there BEFORE quantization
    # (The buffer might not have been overwritten correctly)
    print("\n  Interpreting same buffer as FP32 (first 32 values):")
    fp32_ptr = ctypes.cast(base_ptr + emb_offset, ctypes.POINTER(ctypes.c_float))
    fp32_arr = np.ctypeslib.as_array(fp32_ptr, shape=(32,))
    print(f"    FP32 values: {fp32_arr[:10]}")
    print(f"    FP32 NaN count: {np.isnan(fp32_arr).sum()}")

# Now also check A_LAYER_INPUT (input to quantize_final_output = rmsnorm output)
print("\n--- Checking A_LAYER_INPUT (rmsnorm output, input to quantize) ---")
if "layer_input" in act_buffers:
    li_offset = act_buffers["layer_input"]
    ptr = ctypes.cast(base_ptr + li_offset, ctypes.POINTER(ctypes.c_float))
    li_arr = np.ctypeslib.as_array(ptr, shape=(embed_dim,))

    print(f"  NaN count: {np.isnan(li_arr).sum()}")
    print(f"  Inf count: {np.isinf(li_arr).sum()}")
    print(f"  Range: [{li_arr.min():.6f}, {li_arr.max():.6f}]")
    print(f"  Mean: {li_arr.mean():.6f}")
    print(f"  First 10: {li_arr[:10]}")

print("\n--- Done ---")
