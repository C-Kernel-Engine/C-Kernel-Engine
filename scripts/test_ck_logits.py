#!/usr/bin/env python3
"""
Test CK Engine logits output

This script:
1. Loads the CK model
2. Runs a single token forward pass
3. Examines the logits to see what's happening

Usage:
    python scripts/test_ck_logits.py --model-dir /path/to/ck/model --token 1
"""

import ctypes
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to CK model directory")
    parser.add_argument("--token", type=int, default=9707, help="Token ID to test")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    lib_path = model_dir / "libmodel.so"

    print("=" * 60)
    print("CK ENGINE LOGITS TEST")
    print("=" * 60)

    print(f"\nLoading {lib_path}...")
    lib = ctypes.CDLL(str(lib_path))

    # Set up function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int

    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    # Initialize model
    weights_path = str(model_dir / "weights.bump")
    ret = lib.ck_model_init(weights_path.encode())
    if ret != 0:
        print(f"Failed to initialize model: {ret}")
        return 1

    vocab_size = lib.ck_model_get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # Test single token
    test_token = args.token
    print(f"\nTesting token: {test_token}")

    # Create token array
    token_arr = (ctypes.c_int32 * 1)(test_token)

    # Embed tokens
    ret = lib.ck_model_embed_tokens(token_arr, 1)
    if ret != 0:
        print(f"Embed failed: {ret}")
        return 1
    print("Embedding done")

    # Run forward
    ret = lib.ck_model_forward(None)
    if ret != 0:
        print(f"Forward failed: {ret}")
        return 1
    print("Forward done")

    # Get logits
    logits_ptr = lib.ck_model_get_logits()
    logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()

    # Analyze logits
    print("\n" + "=" * 60)
    print("LOGITS ANALYSIS")
    print("=" * 60)

    print(f"\nShape: {logits.shape}")
    print(f"Min: {np.min(logits):.4f}")
    print(f"Max: {np.max(logits):.4f}")
    print(f"Mean: {np.mean(logits):.4f}")
    print(f"Std: {np.std(logits):.4f}")

    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(logits))
    inf_count = np.sum(np.isinf(logits))
    print(f"\nNaN count: {nan_count}")
    print(f"Inf count: {inf_count}")

    # Top-10 tokens
    top10_idx = np.argsort(logits)[-10:][::-1]
    print(f"\nTop 10 tokens:")
    for i, idx in enumerate(top10_idx):
        print(f"  {i+1}. Token {idx}: {logits[idx]:.4f}")

    # Check if all values are same
    unique_vals = np.unique(logits)
    if len(unique_vals) < 10:
        print(f"\nWARNING: Only {len(unique_vals)} unique values in logits!")
        print(f"Unique values: {unique_vals}")

    # Check if logits look sensible
    if np.std(logits) < 0.01:
        print("\nWARNING: Logits have very low variance - model may not be working!")
    elif np.max(np.abs(logits)) > 1000:
        print("\nWARNING: Logits have very large values - possible overflow!")

    # Check first/last values
    print(f"\nFirst 5 logits: {logits[:5]}")
    print(f"Last 5 logits: {logits[-5:]}")

    # Softmax to see probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    print(f"\nTop 5 probabilities:")
    top5_idx = np.argsort(probs)[-5:][::-1]
    for idx in top5_idx:
        print(f"  Token {idx}: {probs[idx]:.4%}")

    return 0

if __name__ == "__main__":
    exit(main())
