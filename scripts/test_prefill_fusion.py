#!/usr/bin/env python3
"""
Benchmark prefill fusion kernels (RMSNorm + QKV) to measure cache benefit.

This test compares:
1. Fused: RMSNorm+QKV tiled, intermediate stays in L2
2. Unfused: Separate RMSNorm then QKV, intermediate goes to DRAM

Expected result:
- For small seq_len (<256): Similar performance (intermediate fits in cache anyway)
- For large seq_len (>512): Fused should be faster (avoids DRAM spills)

Usage:
    python scripts/test_prefill_fusion.py
    python scripts/test_prefill_fusion.py --seq-lens 128,256,512,1024,2048
"""

import ctypes
import numpy as np
import time
import argparse
import os
import sys

def load_ck_library():
    """Load the CK-Engine shared library."""
    lib_paths = [
        "./build/libckernel_engine.so",
        "./libckernel_engine.so",
        "../build/libckernel_engine.so",
        "./build/libck_engine.so",
    ]

    for path in lib_paths:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                return lib
            except OSError as e:
                print(f"Warning: Could not load {path}: {e}")

    print("ERROR: Could not find libck_engine.so")
    print("Run 'make' first to build the library")
    sys.exit(1)

def setup_functions(lib):
    """Set up ctypes function signatures."""

    # fused_rmsnorm_qkv_prefill
    lib.fused_rmsnorm_qkv_prefill.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # gamma
        ctypes.POINTER(ctypes.c_float),  # Wq
        ctypes.POINTER(ctypes.c_float),  # Wk
        ctypes.POINTER(ctypes.c_float),  # Wv
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # K
        ctypes.POINTER(ctypes.c_float),  # V
        ctypes.c_int,                     # seq_len
        ctypes.c_int,                     # hidden
        ctypes.c_int,                     # q_dim
        ctypes.c_int,                     # kv_dim
        ctypes.c_float,                   # eps
        ctypes.POINTER(ctypes.c_float),  # scratch
    ]
    lib.fused_rmsnorm_qkv_prefill.restype = None

    # unfused_rmsnorm_qkv_prefill
    lib.unfused_rmsnorm_qkv_prefill.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # gamma
        ctypes.POINTER(ctypes.c_float),  # Wq
        ctypes.POINTER(ctypes.c_float),  # Wk
        ctypes.POINTER(ctypes.c_float),  # Wv
        ctypes.POINTER(ctypes.c_float),  # x_norm
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # K
        ctypes.POINTER(ctypes.c_float),  # V
        ctypes.c_int,                     # seq_len
        ctypes.c_int,                     # hidden
        ctypes.c_int,                     # q_dim
        ctypes.c_int,                     # kv_dim
        ctypes.c_float,                   # eps
    ]
    lib.unfused_rmsnorm_qkv_prefill.restype = None

    # fused_rmsnorm_qkv_scratch_size
    lib.fused_rmsnorm_qkv_scratch_size.argtypes = [ctypes.c_int]
    lib.fused_rmsnorm_qkv_scratch_size.restype = ctypes.c_size_t

    return lib

def np_to_ptr(arr):
    """Convert numpy array to ctypes pointer."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def benchmark_fusion(lib, seq_len, hidden, q_dim, kv_dim, num_iters=10, warmup=3):
    """
    Benchmark fused vs unfused RMSNorm+QKV.

    Returns: (fused_time_ms, unfused_time_ms, max_diff)
    """
    eps = 1e-5

    # Allocate buffers
    x = np.random.randn(seq_len, hidden).astype(np.float32)
    gamma = np.random.randn(hidden).astype(np.float32) * 0.1 + 1.0
    Wq = np.random.randn(q_dim, hidden).astype(np.float32) * 0.02
    Wk = np.random.randn(kv_dim, hidden).astype(np.float32) * 0.02
    Wv = np.random.randn(kv_dim, hidden).astype(np.float32) * 0.02

    # Output buffers
    Q_fused = np.zeros((seq_len, q_dim), dtype=np.float32)
    K_fused = np.zeros((seq_len, kv_dim), dtype=np.float32)
    V_fused = np.zeros((seq_len, kv_dim), dtype=np.float32)

    Q_unfused = np.zeros((seq_len, q_dim), dtype=np.float32)
    K_unfused = np.zeros((seq_len, kv_dim), dtype=np.float32)
    V_unfused = np.zeros((seq_len, kv_dim), dtype=np.float32)

    # Scratch buffers
    scratch_size = lib.fused_rmsnorm_qkv_scratch_size(hidden)
    scratch = np.zeros(scratch_size // 4, dtype=np.float32)  # size is in bytes
    x_norm = np.zeros((seq_len, hidden), dtype=np.float32)

    # Warmup
    for _ in range(warmup):
        lib.fused_rmsnorm_qkv_prefill(
            np_to_ptr(x), np_to_ptr(gamma),
            np_to_ptr(Wq), np_to_ptr(Wk), np_to_ptr(Wv),
            np_to_ptr(Q_fused), np_to_ptr(K_fused), np_to_ptr(V_fused),
            seq_len, hidden, q_dim, kv_dim, eps, np_to_ptr(scratch)
        )
        lib.unfused_rmsnorm_qkv_prefill(
            np_to_ptr(x), np_to_ptr(gamma),
            np_to_ptr(Wq), np_to_ptr(Wk), np_to_ptr(Wv),
            np_to_ptr(x_norm),
            np_to_ptr(Q_unfused), np_to_ptr(K_unfused), np_to_ptr(V_unfused),
            seq_len, hidden, q_dim, kv_dim, eps
        )

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(num_iters):
        lib.fused_rmsnorm_qkv_prefill(
            np_to_ptr(x), np_to_ptr(gamma),
            np_to_ptr(Wq), np_to_ptr(Wk), np_to_ptr(Wv),
            np_to_ptr(Q_fused), np_to_ptr(K_fused), np_to_ptr(V_fused),
            seq_len, hidden, q_dim, kv_dim, eps, np_to_ptr(scratch)
        )
    fused_time = (time.perf_counter() - start) / num_iters * 1000  # ms

    # Benchmark unfused
    start = time.perf_counter()
    for _ in range(num_iters):
        lib.unfused_rmsnorm_qkv_prefill(
            np_to_ptr(x), np_to_ptr(gamma),
            np_to_ptr(Wq), np_to_ptr(Wk), np_to_ptr(Wv),
            np_to_ptr(x_norm),
            np_to_ptr(Q_unfused), np_to_ptr(K_unfused), np_to_ptr(V_unfused),
            seq_len, hidden, q_dim, kv_dim, eps
        )
    unfused_time = (time.perf_counter() - start) / num_iters * 1000  # ms

    # Verify correctness
    max_diff = max(
        np.max(np.abs(Q_fused - Q_unfused)),
        np.max(np.abs(K_fused - K_unfused)),
        np.max(np.abs(V_fused - V_unfused)),
    )

    return fused_time, unfused_time, max_diff

def main():
    parser = argparse.ArgumentParser(description="Benchmark prefill fusion kernels")
    parser.add_argument("--seq-lens", type=str, default="64,128,256,512,1024,2048",
                        help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--hidden", type=int, default=896,
                        help="Hidden dimension (default: 896 for Qwen2-0.5B)")
    parser.add_argument("--num-heads", type=int, default=14,
                        help="Number of attention heads")
    parser.add_argument("--num-kv-heads", type=int, default=2,
                        help="Number of KV heads (GQA)")
    parser.add_argument("--head-dim", type=int, default=64,
                        help="Head dimension")
    parser.add_argument("--iters", type=int, default=10,
                        help="Number of iterations per benchmark")
    args = parser.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    hidden = args.hidden
    q_dim = args.num_heads * args.head_dim
    kv_dim = args.num_kv_heads * args.head_dim

    print("=" * 70)
    print("Prefill Fusion Benchmark: RMSNorm + QKV Projection")
    print("=" * 70)
    print(f"Config: hidden={hidden}, q_dim={q_dim}, kv_dim={kv_dim}")
    print(f"        num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}")
    print()

    # Load library
    lib = load_ck_library()
    lib = setup_functions(lib)

    print(f"{'Seq Len':>8} | {'x_norm Size':>12} | {'Fused (ms)':>10} | {'Unfused (ms)':>12} | {'Speedup':>8} | {'Max Diff':>10}")
    print("-" * 70)

    results = []
    for seq_len in seq_lens:
        x_norm_size_mb = seq_len * hidden * 4 / (1024 * 1024)

        try:
            fused_ms, unfused_ms, max_diff = benchmark_fusion(
                lib, seq_len, hidden, q_dim, kv_dim, num_iters=args.iters
            )
            speedup = unfused_ms / fused_ms if fused_ms > 0 else 0

            status = "PASS" if max_diff < 1e-4 else "FAIL"
            print(f"{seq_len:>8} | {x_norm_size_mb:>9.2f} MB | {fused_ms:>10.3f} | {unfused_ms:>12.3f} | {speedup:>7.2f}x | {max_diff:>10.2e} {status}")

            results.append({
                "seq_len": seq_len,
                "x_norm_mb": x_norm_size_mb,
                "fused_ms": fused_ms,
                "unfused_ms": unfused_ms,
                "speedup": speedup,
                "max_diff": max_diff,
            })
        except Exception as e:
            print(f"{seq_len:>8} | ERROR: {e}")

    print()
    print("Analysis:")
    print("-" * 70)

    # Find crossover point
    for r in results:
        if r["speedup"] > 1.05:
            print(f"  Fusion benefit starts at seq_len={r['seq_len']} "
                  f"(x_norm={r['x_norm_mb']:.1f}MB, speedup={r['speedup']:.2f}x)")
            break
    else:
        print("  No significant fusion benefit observed at tested sequence lengths")
        print("  This may indicate:")
        print("    - Your L3 cache is large enough for tested sizes")
        print("    - Try larger sequence lengths (4096, 8192)")

    # Memory traffic analysis
    print()
    print("Expected DRAM traffic per call:")
    for r in results:
        # Unfused: write x_norm + read x_norm + write Q,K,V
        # Fused: write Q,K,V only (x_norm stays in L2)
        q_kv_size_mb = (r["seq_len"] * q_dim + 2 * r["seq_len"] * kv_dim) * 4 / (1024 * 1024)
        unfused_traffic = r["x_norm_mb"] * 2 + q_kv_size_mb  # write + read + output
        fused_traffic = q_kv_size_mb  # only output
        savings = unfused_traffic - fused_traffic
        print(f"  seq_len={r['seq_len']:>4}: Unfused={unfused_traffic:.1f}MB, "
              f"Fused={fused_traffic:.1f}MB, Savings={savings:.1f}MB")

if __name__ == "__main__":
    main()
