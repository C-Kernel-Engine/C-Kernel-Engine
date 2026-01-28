#!/usr/bin/env python3
"""
Compare CK-Engine logits with llama.cpp logits.

Steps to use:
1. First, get llama.cpp logits:
   ./llama.cpp/build/bin/llama-cli -m model.gguf -p "Hello" --logits-all -n 1 > /tmp/llama_logits.txt

2. Then run this script to compare with CK-Engine
"""

import ctypes
import numpy as np
import sys
from pathlib import Path

def load_ck_model(model_dir):
    """Load CK-Engine model."""
    model_dir = Path(model_dir)
    lib_path = model_dir / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = model_dir / "libmodel.so"

    lib = ctypes.CDLL(str(lib_path))

    # Setup function signatures
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    # Get base pointer for activation inspection
    try:
        lib.ck_model_get_base_ptr.argtypes = []
        lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
    except:
        pass

    # Initialize
    weights_path = model_dir / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"Error: Failed to init model, code {ret}")
        return None

    return lib

def get_ck_logits(lib, tokens):
    """Get logits from CK-Engine for given tokens."""
    vocab_size = lib.ck_model_get_vocab_size()

    if len(tokens) == 1:
        # Single token - use decode
        out = (ctypes.c_float * vocab_size)()
        ret = lib.ck_model_decode(tokens[0], out)
        if ret != 0:
            print(f"Decode failed: {ret}")
            return None
        return np.array(out)
    else:
        # Multiple tokens - use embed_tokens (prefill)
        token_arr = (ctypes.c_int32 * len(tokens))(*tokens)
        ret = lib.ck_model_embed_tokens(token_arr, len(tokens))
        if ret != 0:
            print(f"Embed failed: {ret}")
            return None

        # Get logits
        logits_ptr = lib.ck_model_get_logits()
        return np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))

def try_llama_cpp_python(gguf_path, tokens):
    """Try to get logits from llama-cpp-python."""
    try:
        from llama_cpp import Llama

        print("Loading model with llama-cpp-python...")
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=512,
            n_batch=512,
            logits_all=True,
            verbose=False
        )

        # Evaluate tokens
        llm.reset()
        llm.eval(tokens)

        # Get logits for last token
        logits = np.array(llm._scores[-1])
        return logits

    except ImportError:
        print("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        return None
    except Exception as e:
        print(f"Error with llama-cpp-python: {e}")
        return None

def compare_logits(ck_logits, llama_logits, top_k=10):
    """Compare two sets of logits."""
    print("\n" + "="*70)
    print("LOGITS COMPARISON")
    print("="*70)

    print(f"\nCK-Engine logits:")
    print(f"  Shape: {ck_logits.shape}")
    print(f"  Min: {ck_logits.min():.4f}, Max: {ck_logits.max():.4f}")
    print(f"  Mean: {ck_logits.mean():.4f}, Std: {ck_logits.std():.4f}")
    print(f"  NaN: {np.isnan(ck_logits).sum()}, Inf: {np.isinf(ck_logits).sum()}")

    print(f"\nllama.cpp logits:")
    print(f"  Shape: {llama_logits.shape}")
    print(f"  Min: {llama_logits.min():.4f}, Max: {llama_logits.max():.4f}")
    print(f"  Mean: {llama_logits.mean():.4f}, Std: {llama_logits.std():.4f}")
    print(f"  NaN: {np.isnan(llama_logits).sum()}, Inf: {np.isinf(llama_logits).sum()}")

    # Compare
    diff = ck_logits - llama_logits
    abs_diff = np.abs(diff)

    print(f"\nDifference:")
    print(f"  Max abs diff: {abs_diff.max():.4f}")
    print(f"  Mean abs diff: {abs_diff.mean():.4f}")
    print(f"  Std diff: {diff.std():.4f}")

    # Top-K predictions
    ck_top = np.argsort(ck_logits)[-top_k:][::-1]
    llama_top = np.argsort(llama_logits)[-top_k:][::-1]

    print(f"\nTop-{top_k} predictions:")
    print(f"  CK-Engine: {ck_top}")
    print(f"  llama.cpp: {llama_top}")

    # Check agreement
    agreement = len(set(ck_top) & set(llama_top))
    print(f"  Agreement: {agreement}/{top_k} tokens in top-{top_k}")

    if np.argmax(ck_logits) == np.argmax(llama_logits):
        print(f"\n✓ Same argmax: {np.argmax(ck_logits)}")
    else:
        print(f"\n✗ Different argmax: CK={np.argmax(ck_logits)}, llama={np.argmax(llama_logits)}")

    return abs_diff.max()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
    parser.add_argument("--tokens", type=int, nargs="+", default=[100])
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()

    print("="*70)
    print("CK-ENGINE vs LLAMA.CPP LOGITS COMPARISON")
    print("="*70)
    print(f"Model dir: {model_dir}")
    print(f"Tokens: {args.tokens}")

    # Find GGUF
    gguf_files = list(model_dir.glob("*.gguf"))
    if not gguf_files:
        print("Error: No GGUF file found")
        sys.exit(1)
    gguf_path = gguf_files[0]
    print(f"GGUF: {gguf_path}")

    # Load CK-Engine
    print("\n--- Loading CK-Engine ---")
    lib = load_ck_model(model_dir)
    if not lib:
        sys.exit(1)

    print(f"Vocab size: {lib.ck_model_get_vocab_size()}")

    # Get CK logits
    print("\n--- Running CK-Engine ---")
    ck_logits = get_ck_logits(lib, args.tokens)
    if ck_logits is None:
        sys.exit(1)

    # Get llama.cpp logits
    print("\n--- Running llama.cpp ---")
    llama_logits = try_llama_cpp_python(gguf_path, args.tokens)

    if llama_logits is not None:
        compare_logits(ck_logits, llama_logits)
    else:
        print("\nCK-Engine only results:")
        print(f"  Logits shape: {ck_logits.shape}")
        print(f"  Min: {ck_logits.min():.4f}, Max: {ck_logits.max():.4f}")
        print(f"  Argmax: {np.argmax(ck_logits)}")

        print("\nTo get llama.cpp reference, install: pip install llama-cpp-python")

    lib.ck_model_free()

if __name__ == "__main__":
    main()
