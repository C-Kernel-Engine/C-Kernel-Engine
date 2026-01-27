#!/usr/bin/env python3
"""
compare_v65_v66.py - Compare outputs between v6.5 and v6.6
"""

import ctypes
import numpy as np
import sys
from pathlib import Path


def load_model(model_dir: Path, weights_path: Path):
    """Load model and return library handle."""
    # Find library
    lib_path = None
    for name in ["ck-kernel-inference.so", "libmodel.so"]:
        p = model_dir / name
        if p.exists():
            lib_path = p
            break

    if not lib_path:
        raise FileNotFoundError(f"No .so found in {model_dir}")

    lib = ctypes.CDLL(str(lib_path))

    # Bind functions
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int

    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    # Initialize
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        raise RuntimeError(f"ck_model_init failed: {ret}")

    return lib


def run_forward(lib, tokens: list) -> np.ndarray:
    """Run forward pass and return logits."""
    vocab_size = lib.ck_model_get_vocab_size()
    n_tokens = len(tokens)

    # Embed tokens
    token_arr = (ctypes.c_int32 * n_tokens)(*tokens)
    ret = lib.ck_model_embed_tokens(token_arr, n_tokens)
    if ret != 0:
        raise RuntimeError(f"embed_tokens failed: {ret}")

    # Forward
    ret = lib.ck_model_forward(None)
    if ret != 0:
        raise RuntimeError(f"forward failed: {ret}")

    # Get logits
    logits_ptr = lib.ck_model_get_logits()
    logits = np.ctypeslib.as_array(logits_ptr, shape=(n_tokens * vocab_size,))

    # Return last token's logits
    return logits[-(vocab_size):].copy()


def main():
    v65_dir = Path.home() / ".cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
    v66_dir = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

    v65_weights = v65_dir / "weights.bump"
    v66_weights = v66_dir / "weights.bump"

    # Test tokens (simple sequence)
    test_tokens = [151643, 872, 198, 9707, 576, 279, 7290, 315, 9625, 30]  # "Hello what is the capital of France?"

    print("="*60)
    print("COMPARING v6.5 vs v6.6 OUTPUT")
    print("="*60)
    print(f"Test tokens: {test_tokens}")
    print()

    # Load v6.5
    print("Loading v6.5...")
    try:
        lib65 = load_model(v65_dir, v65_weights)
        print(f"  Vocab size: {lib65.ck_model_get_vocab_size()}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    # Load v6.6
    print("Loading v6.6...")
    try:
        lib66 = load_model(v66_dir, v66_weights)
        print(f"  Vocab size: {lib66.ck_model_get_vocab_size()}")
    except Exception as e:
        print(f"  ERROR: {e}")
        lib65.ck_model_free()
        return 1

    # Run forward passes
    print("\nRunning forward passes...")

    try:
        logits65 = run_forward(lib65, test_tokens)
        print(f"  v6.5: {logits65.shape}, min={logits65.min():.4f}, max={logits65.max():.4f}")
    except Exception as e:
        print(f"  v6.5 ERROR: {e}")
        logits65 = None

    try:
        logits66 = run_forward(lib66, test_tokens)
        print(f"  v6.6: {logits66.shape}, min={logits66.min():.4f}, max={logits66.max():.4f}")
    except Exception as e:
        print(f"  v6.6 ERROR: {e}")
        logits66 = None

    # Compare
    if logits65 is not None and logits66 is not None:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)

        diff = np.abs(logits65 - logits66)
        print(f"  Max absolute diff: {diff.max():.6f}")
        print(f"  Mean absolute diff: {diff.mean():.6f}")
        print(f"  Relative error (mean): {(diff / (np.abs(logits65) + 1e-8)).mean():.6f}")

        # Top predictions
        top65 = np.argsort(logits65)[-5:][::-1]
        top66 = np.argsort(logits66)[-5:][::-1]

        print(f"\n  v6.5 top-5 tokens: {list(top65)}")
        print(f"  v6.6 top-5 tokens: {list(top66)}")
        print(f"  Match: {list(top65) == list(top66)}")

        # Detailed comparison of top tokens
        print("\n  Top-5 logit comparison:")
        for i, (t65, t66) in enumerate(zip(top65, top66)):
            print(f"    {i+1}. v6.5: token {t65} = {logits65[t65]:.4f}  |  v6.6: token {t66} = {logits66[t66]:.4f}")

    # Cleanup
    lib65.ck_model_free()
    lib66.ck_model_free()

    return 0


if __name__ == "__main__":
    sys.exit(main())
