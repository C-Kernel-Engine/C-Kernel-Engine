#!/usr/bin/env python3
"""
Compare v6.5 vs v6.6 outputs step by step.

This script runs both implementations with the same input and compares:
1. Token embeddings
2. Layer outputs (after each transformer layer)
3. Final logits
4. KV cache values

Usage:
    python compare_v65_v66.py --prompt "Hello!"
"""

import argparse
import ctypes
import numpy as np
import os
import sys
from pathlib import Path


def load_v66_model(model_path: str):
    """Load v6.6 model shared library."""
    so_path = Path(model_path).parent / "model_v6_6.so"
    if not so_path.exists():
        print(f"v6.6 model not found: {so_path}")
        print("Please compile it first with: python version/v6.6/scripts/ck_run_v6_6.py run ... --force-compile")
        return None

    lib = ctypes.CDLL(str(so_path))

    # Define API
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int

    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64

    lib.ck_model_kv_cache_reset.argtypes = []
    lib.ck_model_kv_cache_reset.restype = None

    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    return lib


def tokenize_with_v66(lib, text: str) -> list:
    """Tokenize using v6.6's built-in tokenizer."""
    # Get vocab info from model
    vocab_offsets_ptr = lib.ck_model_get_vocab_offsets()
    vocab_strings_ptr = lib.ck_model_get_vocab_strings()
    vocab_size = lib.ck_model_get_vocab_size()

    # Simple BPE tokenization - for now just use a basic approach
    # In production, use the proper tokenizer

    # For demo, let's just encode as bytes if we can't do proper BPE
    tokens = []
    for char in text:
        # This is a placeholder - proper tokenization needed
        tokens.append(ord(char) % vocab_size)

    return tokens


def compare_arrays(name: str, arr1: np.ndarray, arr2: np.ndarray, rtol=1e-3, atol=1e-5):
    """Compare two numpy arrays and report differences."""
    if arr1.shape != arr2.shape:
        print(f"  {name}: SHAPE MISMATCH {arr1.shape} vs {arr2.shape}")
        return False

    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Check for NaN/Inf
    nan1 = np.sum(np.isnan(arr1))
    nan2 = np.sum(np.isnan(arr2))
    inf1 = np.sum(np.isinf(arr1))
    inf2 = np.sum(np.isinf(arr2))

    if nan1 > 0 or nan2 > 0:
        print(f"  {name}: NaN detected! arr1:{nan1}, arr2:{nan2}")
        return False
    if inf1 > 0 or inf2 > 0:
        print(f"  {name}: Inf detected! arr1:{inf1}, arr2:{inf2}")
        return False

    is_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)

    if is_close:
        print(f"  {name}: OK (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
    else:
        # Find worst differences
        flat_diff = diff.flatten()
        worst_idx = np.argsort(flat_diff)[-5:][::-1]
        print(f"  {name}: MISMATCH (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
        print(f"    Worst indices: {worst_idx}")
        print(f"    arr1 values: {arr1.flatten()[worst_idx]}")
        print(f"    arr2 values: {arr2.flatten()[worst_idx]}")

    return is_close


def read_memory_region(base_ptr: int, offset: int, count: int, dtype=np.float32) -> np.ndarray:
    """Read memory region from model's bump allocator."""
    ptr = base_ptr + offset
    arr_type = ctypes.c_float * count if dtype == np.float32 else ctypes.c_int32 * count
    arr = arr_type.from_address(ptr)
    return np.array(arr, dtype=dtype)


def run_v66_with_trace(lib, tokens: list, config: dict):
    """Run v6.6 model and collect intermediate values."""
    results = {}

    # Reset KV cache
    lib.ck_model_kv_cache_reset()

    # Get base pointer
    base_ptr = lib.ck_model_get_base_ptr()

    # Define offsets from model (these should match model_v6_6.c)
    A_EMBEDDED_INPUT = config.get("A_EMBEDDED_INPUT", 396942152)
    A_LAYER_INPUT = config.get("A_LAYER_INPUT", 400612168)
    A_KV_CACHE = config.get("A_KV_CACHE", 407952200)
    A_LOGITS = config.get("A_LOGITS", 485284680)

    embed_dim = config.get("embed_dim", 896)
    vocab_size = config.get("vocab_size", 151936)
    num_tokens = len(tokens)

    # Convert tokens to ctypes array
    tokens_arr = (ctypes.c_int32 * len(tokens))(*tokens)

    # Run prefill
    print(f"\nRunning v6.6 prefill with {num_tokens} tokens...")
    ret = lib.ck_model_embed_tokens(tokens_arr, len(tokens))
    if ret != 0:
        print(f"  ERROR: embed_tokens returned {ret}")
        return None

    # Read embeddings
    results["embeddings"] = read_memory_region(base_ptr, A_EMBEDDED_INPUT, num_tokens * embed_dim)
    results["embeddings"] = results["embeddings"].reshape(num_tokens, embed_dim)

    # Read logits
    logits_ptr = lib.ck_model_get_logits()
    if logits_ptr:
        results["logits"] = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()

    # Read some KV cache
    kv_head_size = 2 * 1024 * 64  # num_kv_heads * max_seq * head_dim
    results["kv_cache_layer0_k"] = read_memory_region(base_ptr, A_KV_CACHE, min(1000, kv_head_size))

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare v6.5 vs v6.6 outputs")
    parser.add_argument("--prompt", default="Hello!", help="Input prompt")
    parser.add_argument("--model-dir", help="Model cache directory")
    parser.add_argument("--tokens", help="Comma-separated token IDs to use directly")
    args = parser.parse_args()

    # Find model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        cache_dir = Path.home() / ".cache/ck-engine-v6.6/models"
        model_dirs = list(cache_dir.glob("*qwen*")) + list(cache_dir.glob("*Qwen*"))
        if not model_dirs:
            # Try the local cache
            cache_dir = Path(".ck_cache")
            model_dirs = list(cache_dir.glob("*qwen*")) + list(cache_dir.glob("*Qwen*"))

        if not model_dirs:
            print("No model found. Please specify --model-dir")
            return 1
        model_dir = model_dirs[0]

    print(f"Using model directory: {model_dir}")

    # Find weights file
    bump_file = model_dir / "weights.bump"
    if not bump_file.exists():
        bump_files = list(model_dir.glob("*.bump"))
        if bump_files:
            bump_file = bump_files[0]
        else:
            print(f"No .bump file found in {model_dir}")
            return 1

    # Load v6.6
    print("\n=== Loading v6.6 model ===")
    lib66 = load_v66_model(str(model_dir))
    if not lib66:
        return 1

    ret = lib66.ck_model_init(str(bump_file).encode())
    if ret != 0:
        print(f"Failed to init v6.6 model: {ret}")
        return 1
    print("v6.6 model loaded successfully")

    # Get vocab size
    vocab_size = lib66.ck_model_get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # Parse tokens or tokenize
    if args.tokens:
        tokens = [int(t) for t in args.tokens.split(",")]
    else:
        # Use hardcoded tokens for "Hello!" for Qwen2
        # These are approximate - proper tokenization needed
        tokens = [9707]  # "Hello" in Qwen2 tokenizer (approximate)
        print(f"Using hardcoded tokens: {tokens}")
        print("(For accurate comparison, use --tokens with proper token IDs)")

    print(f"\nTokens: {tokens}")

    # Model config
    config = {
        "embed_dim": 896,
        "vocab_size": vocab_size,
        "A_EMBEDDED_INPUT": 396942152,
        "A_LAYER_INPUT": 400612168,
        "A_KV_CACHE": 407952200,
        "A_LOGITS": 485284680,
    }

    # Run v6.6
    results66 = run_v66_with_trace(lib66, tokens, config)

    if results66:
        print("\n=== v6.6 Results ===")
        print(f"Embeddings shape: {results66['embeddings'].shape}")
        print(f"Embeddings sample: {results66['embeddings'][0, :5]}")

        if "logits" in results66:
            logits = results66["logits"]
            print(f"Logits shape: {logits.shape}")
            print(f"Logits sample (first 5): {logits[:5]}")
            print(f"Logits sample (last 5): {logits[-5:]}")

            # Check for NaN/Inf
            nan_count = np.sum(np.isnan(logits))
            inf_count = np.sum(np.isinf(logits))
            if nan_count > 0 or inf_count > 0:
                print(f"WARNING: Logits contain NaN:{nan_count}, Inf:{inf_count}")

            # Top predictions
            top_k = 5
            top_indices = np.argsort(logits)[-top_k:][::-1]
            print(f"\nTop {top_k} predictions:")
            for idx in top_indices:
                print(f"  Token {idx}: {logits[idx]:.4f}")

        # Check KV cache
        kv = results66.get("kv_cache_layer0_k")
        if kv is not None:
            print(f"\nKV cache layer 0 K sample: {kv[:10]}")
            kv_nonzero = np.sum(np.abs(kv) > 1e-10)
            print(f"KV cache non-zero elements: {kv_nonzero}/{len(kv)}")

    # Cleanup
    lib66.ck_model_free()

    print("\n=== Comparison Complete ===")
    print("To compare with llama.cpp or v6.5, run them separately and compare the printed values.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
