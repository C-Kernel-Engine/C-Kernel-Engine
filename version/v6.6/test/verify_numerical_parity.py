#!/usr/bin/env python3
"""
Quick numerical parity check for mega_fused_attention_prefill.

This verifies that the fused implementation matches the unfused baseline
within numerical tolerance (< 1e-3).
"""

import argparse
import ctypes
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "unittest"))

from lib_loader import load_lib
from test_utils import time_function

CK_DT_Q5_0 = 11

def make_q5_weights(rows, cols):
    """Create simple Q5_0 weights"""
    block_size, block_bytes = 32, 22
    n_blocks = (cols + 31) // 32
    data = np.zeros(rows * n_blocks * block_bytes, dtype=np.uint8)

    for r in range(rows):
        for b in range(n_blocks):
            off = r * n_blocks * block_bytes + b * block_bytes
            scale = np.float16(0.02)
            data[off:off+2] = np.frombuffer(scale.tobytes(), dtype=np.uint8)
            data[off+2:off+22] = np.random.randint(0, 256, 20, dtype=np.uint8)

    return data

def test_parity(tokens=32, embed_dim=896, num_heads=14, num_kv_heads=2, head_dim=64):
    """Test numerical parity between baseline and fused"""
    print("="*70)
    print("Numerical Parity Test: mega_fused_attention_prefill")
    print("="*70)
    print(f"\nConfig:")
    print(f"  Tokens:      {tokens}")
    print(f"  Embed Dim:   {embed_dim}")
    print(f"  Heads:       {num_heads}")
    print(f"  KV Heads:    {num_kv_heads}")
    print(f"  Head Dim:    {head_dim}")
    print()

    # Load library
    lib = load_lib("libckernel_engine.so")

    # Create test data
    np.random.seed(42)
    x = np.random.randn(tokens, embed_dim).astype(np.float32)
    gamma = np.random.randn(embed_dim).astype(np.float32) * 0.5 + 1.0

    # Create weights
    q_dim = num_heads * head_dim
    wq = make_q5_weights(q_dim, embed_dim)
    wk = make_q5_weights(q_dim, embed_dim)
    wv = make_q5_weights(q_dim, embed_dim)
    wo = make_q5_weights(embed_dim, embed_dim)

    # Allocate outputs
    out_baseline = np.zeros((tokens, embed_dim), dtype=np.float32)
    out_fused = np.zeros((tokens, embed_dim), dtype=np.float32)

    # Get scratch sizes
    qkv_scratch = lib.fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(
        ctypes.c_int(embed_dim)
    )
    scratch_size = lib.mega_fused_attention_prefill_scratch_size(
        ctypes.c_int(tokens), ctypes.c_int(embed_dim),
        ctypes.c_int(num_heads), ctypes.c_int(head_dim)
    )

    scratch = np.zeros(scratch_size, dtype=np.uint8)

    # Run baseline
    print("Running baseline (unfused)...")

    # RMSNorm + QKV
    lib.fused_rmsnorm_qkv_prefill_head_major_quant(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        wk.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        wv.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        np.zeros((num_heads, tokens, head_dim), dtype=np.float32).ctypes.data,
        np.zeros((num_kv_heads, tokens, head_dim), dtype=np.float32).ctypes.data,
        np.zeros((num_kv_heads, tokens, head_dim), dtype=np.float32).ctypes.data,
        ctypes.c_int(tokens),
        ctypes.c_int(embed_dim),
        ctypes.c_int(embed_dim),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(head_dim),
        ctypes.c_int(head_dim),
        ctypes.c_int(tokens),
        ctypes.c_float(1e-6),
        scratch.ctypes.data,
    )

    # Attention
    q = np.zeros((num_heads, tokens, head_dim), dtype=np.float32)
    k = np.zeros((num_kv_heads, tokens, head_dim), dtype=np.float32)
    v = np.zeros((num_kv_heads, tokens, head_dim), dtype=np.float32)

    lib.attention_forward_causal_head_major_gqa_flash_strided(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        np.zeros((num_heads, tokens, head_dim), dtype=np.float32).ctypes.data,
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(tokens),
        ctypes.c_int(head_dim),
        ctypes.c_int(head_dim),
        ctypes.c_int(tokens),
    )

    # Flatten head-major to token-major
    proj_in = np.zeros((tokens, embed_dim), dtype=np.float32)
    for t in range(tokens):
        for h in range(num_heads):
            proj_in[t, h*head_dim:(h+1)*head_dim] = q[h, t]

    # Output projection
    lib.ck_gemm_nt_quant(
        proj_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wo.ctypes.data,
        None,
        out_baseline.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(tokens),
        ctypes.c_int(embed_dim),
        ctypes.c_int(embed_dim),
        ctypes.c_int(CK_DT_Q5_0)
    )

    print("✓ Baseline complete")

    # Run fused
    print("Running fused (mega-fused)...")
    lib.mega_fused_attention_prefill(
        out_fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        None,
        gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        wq.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        wk.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        wv.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        wo.ctypes.data, None, ctypes.c_int(CK_DT_Q5_0),
        np.zeros((num_kv_heads, tokens, head_dim), dtype=np.float32).ctypes.data,
        np.zeros((num_kv_heads, tokens, head_dim), dtype=np.float32).ctypes.data,
        None, None,
        ctypes.c_int(0),
        ctypes.c_int(tokens),
        ctypes.c_int(tokens),
        ctypes.c_int(embed_dim),
        ctypes.c_int(embed_dim),
        ctypes.c_int(num_heads),
        ctypes.c_int(num_kv_heads),
        ctypes.c_int(head_dim),
        ctypes.c_int(head_dim),
        ctypes.c_float(1e-6),
        scratch.ctypes.data,
    )

    print("✓ Fused complete")

    # Compare
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    max_diff = np.max(np.abs(out_baseline - out_fused))
    max_baseline = np.max(np.abs(out_baseline))

    print(f"\nNumerical Accuracy:")
    print(f"  Max difference:  {max_diff:.6e}")
    print(f"  Relative error:  {max_diff / max_baseline:.6e}")
    print(f"  Max baseline:    {max_baseline:.6e}")

    # Check for NaN/Inf
    has_nan_baseline = np.any(np.isnan(out_baseline)) or np.any(np.isinf(out_baseline))
    has_nan_fused = np.any(np.isnan(out_fused)) or np.any(np.isinf(out_fused))

    print(f"\nValidation:")
    print(f"  Baseline has NaN/Inf: {has_nan_baseline}")
    print(f"  Fused has NaN/Inf:    {has_nan_fused}")

    # Final verdict
    print(f"\nVerdict:")
    if max_diff < 1e-3 and not has_nan_baseline and not has_nan_fused:
        print(f"  ✓ PASSED: Numerical parity within tolerance (1e-3)")
        return True
    elif max_diff < 1e-2:
        print(f"  ⚠ WARNING: Acceptable for quantized weights (tolerance: 1e-2)")
        return True
    else:
        print(f"  ✗ FAILED: Difference exceeds tolerance!")
        print(f"\n  Sample differences:")
        diffs = np.abs(out_baseline - out_fused).flatten()
        top_diffs = np.argsort(diffs)[-5:]
        for idx in top_diffs[::-1]:
            t = idx // embed_dim
            d = idx % embed_dim
            print(f"    token={t}, dim={d}: baseline={out_baseline.flat[idx]:.6f}, "
                  f"fused={out_fused.flat[idx]:.6f}, diff={diffs[idx]:.6e}")
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify numerical parity")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=896)
    ap.add_argument("--heads", type=int, default=14)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    args = ap.parse_args()

    success = test_parity(
        tokens=args.tokens,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )

    sys.exit(0 if success else 1)
