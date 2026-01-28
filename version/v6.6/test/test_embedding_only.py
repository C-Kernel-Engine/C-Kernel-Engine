#!/usr/bin/env python3
"""
Test just the embedding lookup to verify basic functionality.
"""

import ctypes
import numpy as np
from pathlib import Path

cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
model_dirs = list(cache_dir.glob("*Qwen*")) + list(cache_dir.glob("*qwen*"))

if not model_dirs:
    print("Error: No Qwen model found in cache")
    exit(1)

model_dir = model_dirs[0]
lib_path = model_dir / "libmodel.so"
weights_path = model_dir / "weights.bump"

# Read layout to get buffer offsets
import json
layout_path = model_dir / "layout_decode.json"
with open(layout_path) as f:
    layout = json.load(f)

config = layout.get("config", {})
embed_dim = config.get("embed_dim", 896)
vocab_size = config.get("vocab_size", 151936)

# Get activation buffer offsets
act_buffers = {}
for buf in layout.get("memory", {}).get("activations", {}).get("buffers", []):
    act_buffers[buf["name"]] = buf["abs_offset"]

print(f"Embed dim: {embed_dim}")
print(f"Vocab size: {vocab_size}")
print(f"Activation buffers: {list(act_buffers.keys())}")

# Get token embedding weight offset
weights_info = layout.get("memory", {}).get("weights", {})
token_emb_offset = None
for entry in weights_info.get("entries", []):
    if entry.get("name") == "token_emb":
        token_emb_offset = entry.get("abs_offset", 0)
        token_emb_dtype = entry.get("dtype", "unknown")
        token_emb_size = entry.get("size", 0)
        break

print(f"Token embedding: offset={token_emb_offset}, dtype={token_emb_dtype}, size={token_emb_size}")

# Load library
lib = ctypes.CDLL(str(lib_path))
lib.ck_model_init.argtypes = [ctypes.c_char_p]
lib.ck_model_init.restype = ctypes.c_int
lib.ck_model_get_base_ptr.restype = ctypes.c_void_p
lib.ck_model_get_vocab_size.restype = ctypes.c_int

# Initialize
print("\n--- Initializing model ---")
ret = lib.ck_model_init(str(weights_path).encode())
print(f"Init result: {ret}")

if ret != 0:
    print("Error: Model init failed")
    exit(1)

base_ptr = lib.ck_model_get_base_ptr()
print(f"Base ptr: {hex(base_ptr)}")

# Read first few values of token embedding weights
print("\n--- Checking token embedding weights ---")
if token_emb_offset is not None:
    # Token embeddings are Q8_0 quantized
    # Q8_0 block: 2 bytes (fp16 scale) + 32 bytes (int8 values) = 34 bytes per 32 elements
    # Let's read the raw bytes for token 9707
    token_id = 9707
    block_size = 34
    elements_per_block = 32
    blocks_per_row = embed_dim // elements_per_block  # 896 / 32 = 28 blocks
    bytes_per_row = blocks_per_row * block_size  # 28 * 34 = 952 bytes

    row_offset = token_emb_offset + token_id * bytes_per_row

    # Read the first block of this token's embedding
    ptr = ctypes.cast(base_ptr + row_offset, ctypes.POINTER(ctypes.c_uint8))
    raw_bytes = bytes([ptr[i] for i in range(block_size)])

    # Parse Q8_0 block: first 2 bytes are fp16 scale
    scale_bytes = raw_bytes[:2]
    scale = np.frombuffer(scale_bytes, dtype=np.float16)[0]
    quants = np.frombuffer(raw_bytes[2:], dtype=np.int8)

    print(f"Token {token_id} embedding (first block):")
    print(f"  Scale (fp16): {scale}")
    print(f"  Quants (int8): {quants[:8]}...")
    print(f"  Dequantized: {(quants[:8].astype(np.float32) * float(scale))}")

    if np.isnan(scale) or np.isinf(scale):
        print("  WARNING: Scale is NaN/Inf!")

# Read embedded_input buffer after a decode call
print("\n--- Running single decode and checking buffers ---")
import os
os.environ['CK_STOP_OP'] = '1'  # Stop after first op (embedding)

lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
lib.ck_model_decode.restype = ctypes.c_int

vocab = lib.ck_model_get_vocab_size()
logits = (ctypes.c_float * vocab)()
lib.ck_model_decode(9707, logits)

del os.environ['CK_STOP_OP']

# Check embedded_input buffer
if "embedded_input" in act_buffers:
    emb_offset = act_buffers["embedded_input"]
    ptr = ctypes.cast(base_ptr + emb_offset, ctypes.POINTER(ctypes.c_float))
    emb_arr = np.ctypeslib.as_array(ptr, shape=(embed_dim,))

    print(f"\nEmbedded input buffer (after embedding op):")
    print(f"  Offset: {emb_offset}")
    print(f"  Shape: {emb_arr.shape}")
    print(f"  NaN count: {np.isnan(emb_arr).sum()}")
    print(f"  Inf count: {np.isinf(emb_arr).sum()}")
    print(f"  Range: [{emb_arr.min():.6f}, {emb_arr.max():.6f}]")
    print(f"  Mean: {emb_arr.mean():.6f}")
    print(f"  First 10: {emb_arr[:10]}")

# Check layer_input buffer
if "layer_input" in act_buffers:
    li_offset = act_buffers["layer_input"]
    ptr = ctypes.cast(base_ptr + li_offset, ctypes.POINTER(ctypes.c_float))
    li_arr = np.ctypeslib.as_array(ptr, shape=(embed_dim,))

    print(f"\nLayer input buffer:")
    print(f"  Offset: {li_offset}")
    print(f"  NaN count: {np.isnan(li_arr).sum()}")
    print(f"  Range: [{li_arr.min():.6f}, {li_arr.max():.6f}]")
    print(f"  First 10: {li_arr[:10]}")

print("\n--- Done ---")
