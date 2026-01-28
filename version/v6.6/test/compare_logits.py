#!/usr/bin/env python3
"""
Compare logits between CK-Engine and llama.cpp token by token.
"""
import ctypes
import numpy as np
import subprocess
import json
from pathlib import Path

def load_ck_engine():
    """Load CK-Engine model."""
    model_dir = Path.home() / '.cache/ck-engine-v6.6/models/qwen2-0_5b-instruct-q4_k_m'

    # Load kernel library first
    kernel_lib = ctypes.CDLL(str(model_dir / 'libckernel_engine.so'), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(model_dir / 'libmodel.so'))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    weights_path = model_dir / 'weights.bump'
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        raise RuntimeError(f"Failed to init CK-Engine: {ret}")

    return lib

def ck_decode(lib, token_id):
    """Run CK-Engine decode and return logits."""
    vocab_size = lib.ck_model_get_vocab_size()
    out = (ctypes.c_float * vocab_size)()
    lib.ck_model_decode(token_id, out)
    return np.array(out)

def main():
    print("=" * 70)
    print("CK-Engine vs llama.cpp Logits Comparison")
    print("=" * 70)

    # Test token: 9707 = "Hello"
    token_id = 9707

    print(f"\nTest token: {token_id}")

    # Load and run CK-Engine
    print("\n--- CK-Engine ---")
    lib = load_ck_engine()
    ck_logits = ck_decode(lib, token_id)

    print(f"Vocab size: {len(ck_logits)}")
    print(f"Logits [0:5]: {ck_logits[:5]}")
    print(f"Logits stats: min={ck_logits.min():.4f}, max={ck_logits.max():.4f}")
    print(f"NaN count: {np.isnan(ck_logits).sum()}")

    ck_top5_idx = np.argsort(ck_logits)[-5:][::-1]
    print(f"Top 5 tokens: {ck_top5_idx}")
    print(f"Top 5 logits: {ck_logits[ck_top5_idx]}")
    print(f"Argmax token: {np.argmax(ck_logits)} (logit={ck_logits.max():.4f})")

    lib.ck_model_free()

    # Compare with llama.cpp reference values (from dump-layer output)
    print("\n--- llama.cpp (reference) ---")
    llama_logits_sample = [13.911835, 1.539046, 4.859981, 1.760679, 2.370180]
    print(f"Logits [0:5]: {llama_logits_sample}")

    print("\n--- Comparison ---")
    diff = ck_logits[:5] - np.array(llama_logits_sample)
    print(f"Difference [0:5]: {diff}")
    print(f"Max abs diff: {np.abs(diff).max():.6f}")

    if np.argmax(ck_logits) == 0:  # Both should predict token 0 as highest
        print("\n[OK] Both predict same top token!")
    else:
        print(f"\n[WARN] Different top token: CK={np.argmax(ck_logits)}")

if __name__ == '__main__':
    main()
