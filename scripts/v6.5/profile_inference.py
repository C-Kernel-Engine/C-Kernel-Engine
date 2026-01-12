#!/usr/bin/env python3
"""
Profile v6.5 inference using the compiled shared library.
Run with: python scripts/v6.5/profile_inference.py
Then use perf: perf record -g python scripts/v6.5/profile_inference.py
"""

import ctypes
import os
import sys
import time

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

# Find the shared library
MODEL_DIR = os.path.expanduser("~/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
LIB_PATH = os.path.join(MODEL_DIR, "ck-kernel-inference.so")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.bump")

if not os.path.exists(LIB_PATH):
    flush_print(f"Error: Library not found: {LIB_PATH}")
    sys.exit(1)

flush_print(f"Loading library: {LIB_PATH}")
lib = ctypes.CDLL(LIB_PATH)

flush_print("Setting up function signatures...")
# Define function signatures - match ck_chat.py exactly
lib.ck_model_init.argtypes = [ctypes.c_char_p]
lib.ck_model_init.restype = ctypes.c_int

lib.ck_model_free.argtypes = []
lib.ck_model_free.restype = None

# forward takes pointer but we pass None - logits are stored internally
lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.ck_model_forward.restype = ctypes.c_int

# decode returns logits internally too
lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
lib.ck_model_decode.restype = ctypes.c_int

lib.ck_model_get_vocab_size.argtypes = []
lib.ck_model_get_vocab_size.restype = ctypes.c_int

lib.ck_model_get_logits.argtypes = []
lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

lib.ck_model_get_active_tokens.argtypes = []
lib.ck_model_get_active_tokens.restype = ctypes.c_int

lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
lib.ck_model_embed_tokens.restype = ctypes.c_int

lib.ck_model_kv_cache_enable.argtypes = [ctypes.c_int]
lib.ck_model_kv_cache_enable.restype = ctypes.c_int

lib.ck_model_kv_cache_reset.argtypes = []
lib.ck_model_kv_cache_reset.restype = None

# Initialize model
flush_print(f"Initializing model from: {WEIGHTS_PATH}")
ret = lib.ck_model_init(WEIGHTS_PATH.encode())
flush_print(f"  ck_model_init returned: {ret}")
if ret != 0:
    flush_print(f"Error: ck_model_init failed with code {ret}")
    sys.exit(1)

flush_print("Getting vocab size...")
vocab_size = lib.ck_model_get_vocab_size()
flush_print(f"Vocab size: {vocab_size}")

# Test tokens (simple prompt)
test_tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]  # System prompt tokens
tokens_array = (ctypes.c_int32 * len(test_tokens))(*test_tokens)

# Enable KV cache for decode
flush_print("Enabling KV cache...")
ret = lib.ck_model_kv_cache_enable(32768)
flush_print(f"  kv_cache_enable returned: {ret}")

flush_print("Resetting KV cache...")
lib.ck_model_kv_cache_reset()

# Embed tokens for prefill
flush_print(f"Embedding {len(test_tokens)} tokens...")
ret = lib.ck_model_embed_tokens(tokens_array, len(test_tokens))
flush_print(f"  embed_tokens returned: {ret}")
if ret != 0:
    flush_print(f"Warning: embed_tokens returned {ret}")

# Run prefill - pass None, logits are internal
flush_print("Running prefill forward pass...")
t0 = time.perf_counter()
ret = lib.ck_model_forward(None)  # Pass None, not a buffer!
t1 = time.perf_counter()
flush_print(f"  forward returned: {ret}")
flush_print(f"Prefill: {(t1-t0)*1000:.2f} ms ({len(test_tokens)} tokens, {len(test_tokens)/(t1-t0):.2f} tok/s)")

# Run decode iterations for profiling
NUM_DECODE = 10
flush_print(f"\nRunning {NUM_DECODE} decode iterations...")

decode_times = []
for i in range(NUM_DECODE):
    # Use a simple token (space = 220 in many tokenizers)
    token = ctypes.c_int32(220)

    t0 = time.perf_counter()
    ret = lib.ck_model_decode(token, None)  # Pass None for logits
    t1 = time.perf_counter()

    decode_times.append((t1 - t0) * 1000)

    if ret != 0:
        flush_print(f"  Decode {i} failed with code {ret}")
        break
    if (i + 1) % 10 == 0:
        flush_print(f"  Completed {i + 1}/{NUM_DECODE} decodes")

if decode_times:
    avg_decode = sum(decode_times) / len(decode_times)
    tok_per_sec = 1000.0 / avg_decode

    flush_print(f"\nDecode stats:")
    flush_print(f"  Average: {avg_decode:.2f} ms/token")
    flush_print(f"  Throughput: {tok_per_sec:.2f} tok/s")
    flush_print(f"  Min: {min(decode_times):.2f} ms")
    flush_print(f"  Max: {max(decode_times):.2f} ms")

# Cleanup
flush_print("\nCleaning up...")
lib.ck_model_free()
flush_print("Done.")
