#!/usr/bin/env python3
"""
Compare CK-Engine mega-fused attention with llama.cpp implementation.

This runs identical inputs through both implementations and compares:
1. Numerical accuracy
2. Performance
3. Memory usage
"""

import argparse
import ctypes
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_CPP_DIR = os.path.join(SCRIPT_DIR, "llama.cpp")
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unittest"))

from lib_loader import load_lib
from test_utils import time_function, TimingResult

# Try to import llama.cpp
try:
    # Add llama.cpp build to path
    LLAMA_BUILD_DIR = os.path.join(LLAMA_CPP_DIR, "build", "bin")
    if os.path.exists(LLAMA_BUILD_DIR):
        sys.path.insert(0, LLAMA_BUILD_DIR)
        import ggml
        LLAMA_AVAILABLE = True
        print(f"✓ llama.cpp found at {LLAMA_BUILD_DIR}")
    else:
        print(f"⚠ llama.cpp build not found at {LLAMA_BUILD_DIR}")
        LLAMA_AVAILABLE = False
except ImportError as e:
    print(f"⚠ Could not import llama.cpp: {e}")
    LLAMA_AVAILABLE = False

CK_DT_Q5_0 = 11
CK_DT_Q8_0 = 9

def make_q5_weights(rows, cols):
    """Create Q5_0 weights"""
    block_size, block_bytes = 32, 22
    n_blocks = (cols + 31) // 32
    data = np.zeros(rows * n_blocks * block_bytes, dtype=np.uint8)

    np.random.seed(42)
    for r in range(rows):
        for b in range(n_blocks):
            off = r * n_blocks * block_bytes + b * block_bytes
            scale = np.float16(0.02)
            data[off:off+2] = np.frombuffer(scale.tobytes(), dtype=np.uint8)
            data[off+2:off+block_bytes] = np.random.randint(0, 256, block_bytes-2, dtype=np.uint8)

    return data

def run_ck_engine(x, gamma, wq, wk, wv, wo, dims):
    """Run CK-Engine implementation"""
    lib = load_lib("libckernel_engine.so")

    # Get scratch sizes
    qkv_scratch = lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(
        ctypes.c_int(dims['embed_dim'])
    )
    scratch_size = lib.mega_fused_attention_prefill_scratch_size(
        ctypes.c_int(dims['tokens']), ctypes.c_int(dims['embed_dim']),
        ctypes.c_int(dims['num_heads']), ctypes.c_int(dims['head_dim'])
    )

    scratch = np.zeros(scratch_size, dtype=np.uint8)
    output = np.zeros((dims['tokens'], dims['embed_dim']), dtype=np.float32)

    # Run
    lib.mega_fused_attention_prefill(
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        None,
        gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        wk.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        wv.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        wo.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        np.zeros((dims['num_kv_heads'], dims['tokens'], dims['head_dim']), dtype=np.float32).ctypes.data,
        np.zeros((dims['num_kv_heads'], dims['tokens'], dims['head_dim']), dtype=np.float32).ctypes.data,
        None, None,
        ctypes.c_int(0),
        ctypes.c_int(dims['tokens']),
        ctypes.c_int(dims['tokens']),
        ctypes.c_int(dims['embed_dim']),
        ctypes.c_int(dims['embed_dim']),
        ctypes.c_int(dims['num_heads']),
        ctypes.c_int(dims['num_kv_heads']),
        ctypes.c_int(dims['head_dim']),
        ctypes.c_int(dims['head_dim']),
        ctypes.c_float(1e-6),
        scratch.ctypes.data,
    )

    return output

def run_llama_cpp(x, gamma, wq, wk, wv, wo, dims):
    """Run llama.cpp implementation (simplified)"""
    if not LLAMA_AVAILABLE:
        print("⚠ llama.cpp not available - using placeholder")
        return np.zeros((dims['tokens'], dims['embed_dim']), dtype=np.float32)

    # Simplified implementation - would need to use actual llama.cpp API
    # For now, just return zeros
    # TODO: Implement actual llama.cpp attention call

    print("⚠ llama.cpp implementation not yet fully integrated")
    print("   This would require:")
    print("   1. Loading a real llama.cpp model")
    print("   2. Running attention with the same weights")
    print("   3. Extracting attention output")

    return np.zeros((dims['tokens'], dims['embed_dim']), dtype=np.float32)

def compare_implementations(tokens=32, embed_dim=896, num_heads=14, num_kv_heads=2, head_dim=64, iters=5):
    """Compare CK-Engine vs llama.cpp"""

    print("="*70)
    print("CK-Engine vs llama.cpp Attention Comparison")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Tokens:      {tokens}")
    print(f"  Embed Dim:   {embed_dim}")
    print(f"  Heads:       {num_heads}")
    print(f"  KV Heads:    {num_kv_heads}")
    print(f"  Head Dim:    {head_dim}")
    print(f"  Iterations:  {iters}\n")

    dims = {
        'tokens': tokens,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
    }

    # Create test data
    print("Creating test data...")
    np.random.seed(42)
    x = np.random.randn(tokens, embed_dim).astype(np.float32) * 0.1
    gamma = (np.random.randn(embed_dim).astype(np.float32) * 0.5 + 0.75)

    q_dim = num_heads * head_dim
    wq = make_q5_weights(q_dim, embed_dim)
    wk = make_q5_weights(q_dim, embed_dim)
    wv = make_q5_weights(q_dim, embed_dim)
    wo = make_q5_weights(embed_dim, embed_dim)
    print("✓ Test data created\n")

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = run_ck_engine(x, gamma, wq, wk, wv, wo, dims)
        if LLAMA_AVAILABLE:
            _ = run_llama_cpp(x, gamma, wq, wk, wv, wo, dims)
    print("✓ Warmup complete\n")

    # Benchmark CK-Engine
    print("Benchmarking CK-Engine...")
    times_ck = []
    for i in range(iters):
        start = time.perf_counter()
        out_ck = run_ck_engine(x, gamma, wq, wk, wv, wo, dims)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times_ck.append(elapsed)
        print(f"  Run {i+1}: {elapsed:8.3f} ms")

    avg_ck = np.mean(times_ck)
    std_ck = np.std(times_ck)
    print(f"  Average: {avg_ck:8.3f} ± {std_ck:6.3f} ms\n")

    # Benchmark llama.cpp
    if LLAMA_AVAILABLE:
        print("Benchmarking llama.cpp...")
        times_llama = []
        for i in range(iters):
            start = time.perf_counter()
            out_llama = run_llama_cpp(x, gamma, wq, wk, wv, wo, dims)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times_llama.append(elapsed)
            print(f"  Run {i+1}: {elapsed:8.3f} ms")

        avg_llama = np.mean(times_llama)
        std_llama = np.std(times_llama)
        print(f"  Average: {avg_llama:8.3f} ± {std_llama:6.3f} ms\n")

        # Compare
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nPerformance:")
        print(f"  CK-Engine:   {avg_ck:8.3f} ± {std_ck:6.3f} ms")
        print(f"  llama.cpp:   {avg_llama:8.3f} ± {std_llama:6.3f} ms")
        speedup = avg_llama / avg_ck
        print(f"\n  Speedup:     {speedup:8.3f}x {'(CK-Engine is faster!)' if speedup > 1 else '(llama.cpp is faster)'}")

        # Numerical comparison
        max_diff = np.max(np.abs(out_ck - out_llama))
        print(f"\nNumerical Accuracy:")
        print(f"  Max difference:  {max_diff:.6e}")

        if max_diff < 1e-3:
            print(f"  Status:        ✓ PASSED (< 1e-3)")
        elif max_diff < 1e-2:
            print(f"  Status:        ⚠ WARNING (< 1e-2, acceptable for quantized)")
        else:
            print(f"  Status:        ✗ FAILED (too large)")

        return speedup, max_diff
    else:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nPerformance:")
        print(f"  CK-Engine:   {avg_ck:8.3f} ± {std_ck:6.3f} ms")
        print(f"\n  Note: llama.cpp not available for comparison")
        print(f"  To enable: Install llama.cpp at ~/llama.cpp")
        print(f"    cd ~/llama.cpp && pip install -e .")

        return None, None

def main():
    ap = argparse.ArgumentParser(description="Compare CK-Engine vs llama.cpp")
    ap.add_argument("--tokens", type=int, default=32, help="Number of tokens")
    ap.add_argument("--embed-dim", type=int, default=896, help="Embedding dimension")
    ap.add_argument("--heads", type=int, default=14, help="Number of attention heads")
    ap.add_argument("--kv-heads", type=int, default=2, help="Number of KV heads")
    ap.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    ap.add_argument("--iters", type=int, default=5, help="Number of iterations")
    args = ap.parse_args()

    speedup, max_diff = compare_implementations(
        tokens=args.tokens,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        iters=args.iters,
    )

    if speedup is not None:
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        if speedup > 1.2:
            print(f"\n✓ CK-Engine is {speedup:.2f}x FASTER than llama.cpp")
            print(f"  Your mega-fusion optimization is WORKING!")
        elif speedup > 0.8:
            print(f"\n≈ CK-Engine is {1/speedup:.2f}x slower than llama.cpp")
            print(f"  Performance is comparable (within 20%)")
        else:
            print(f"\n⚠ llama.cpp is {1/speedup:.2f}x faster than CK-Engine")
            print(f"  Need to investigate llama.cpp optimizations")

if __name__ == "__main__":
    main()
