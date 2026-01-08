#!/usr/bin/env python3
"""
Profile C-Kernel decode performance.

Usage:
    # Record with perf (run as root or with perf permissions)
    sudo perf record -g python scripts/profile_decode.py --model-dir ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF

    # Generate flamegraph
    sudo perf script | ./FlameGraph/stackcollapse-perf.pl | ./FlameGraph/flamegraph.pl > flame.svg

    # Or just time it
    python scripts/profile_decode.py --model-dir ~/.cache/ck-engine-v5/models/Qwen--Qwen2-0.5B-Instruct-GGUF
"""

import argparse
import ctypes
import time
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--decode-steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    lib_path = model_dir / "libmodel.so"
    weights_path = model_dir / "weights.bump"

    if not lib_path.exists():
        print(f"Error: {lib_path} not found")
        return 1

    # Load library
    lib = ctypes.CDLL(str(lib_path))

    # Setup function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int

    lib.ck_model_kv_cache_enable.argtypes = [ctypes.c_int]
    lib.ck_model_kv_cache_enable.restype = ctypes.c_int

    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int

    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_get_context_window.restype = ctypes.c_int

    # Initialize model
    print(f"Loading model from {weights_path}...")
    t0 = time.time()
    ret = lib.ck_model_init(str(weights_path).encode())
    load_time = time.time() - t0

    if ret != 0:
        print(f"Error: Failed to init model (code {ret})")
        return 1

    vocab_size = lib.ck_model_get_vocab_size()
    context_window = lib.ck_model_get_context_window()
    print(f"Loaded in {load_time*1000:.1f}ms. Vocab: {vocab_size}, Context: {context_window}")

    # Enable KV cache
    lib.ck_model_kv_cache_enable(context_window)

    # Prefill with a short prompt (token IDs for "Hello")
    prompt_tokens = [9707]  # "Hello" in Qwen
    tokens = (ctypes.c_int32 * len(prompt_tokens))(*prompt_tokens)

    print(f"\nPrefill ({len(prompt_tokens)} tokens)...")
    t0 = time.time()
    lib.ck_model_embed_tokens(tokens, len(prompt_tokens))
    lib.ck_model_forward(None)
    prefill_time = time.time() - t0
    print(f"  Prefill: {prefill_time*1000:.2f}ms ({len(prompt_tokens)/prefill_time:.1f} tok/s)")

    # Warmup decode
    print(f"\nWarmup ({args.warmup} decode steps)...")
    for i in range(args.warmup):
        lib.ck_model_decode(ctypes.c_int32(1000 + i), None)

    # Timed decode
    print(f"\nBenchmark ({args.decode_steps} decode steps)...")
    decode_times = []

    t_total = time.time()
    for i in range(args.decode_steps):
        t0 = time.time()
        lib.ck_model_decode(ctypes.c_int32(2000 + i), None)
        decode_times.append(time.time() - t0)
    total_decode = time.time() - t_total

    # Stats
    avg_ms = (total_decode / args.decode_steps) * 1000
    tok_per_sec = args.decode_steps / total_decode

    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"{'='*50}")
    print(f"  Total decode time: {total_decode*1000:.1f}ms")
    print(f"  Avg per token:     {avg_ms:.2f}ms")
    print(f"  Throughput:        {tok_per_sec:.1f} tok/s")
    print(f"  Min:               {min(decode_times)*1000:.2f}ms")
    print(f"  Max:               {max(decode_times)*1000:.2f}ms")
    print(f"{'='*50}")

    # Compare with llama.cpp typical performance
    print(f"\nFor reference, llama.cpp on similar hardware typically achieves:")
    print(f"  Q4_K_M decode: ~50-100 tok/s on CPU")
    print(f"  Our result:    {tok_per_sec:.1f} tok/s")

    return 0

if __name__ == "__main__":
    sys.exit(main())
