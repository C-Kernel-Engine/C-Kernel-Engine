#!/usr/bin/env python3
"""
Compare attention output between v6.5 and v6.6.

Focus on layer 0 attention to find the divergence point.
"""

import ctypes
import numpy as np
import os
from pathlib import Path

V66_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
V65_DIR = Path.home() / ".cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

EMBED_DIM = 896
NUM_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
VOCAB_SIZE = 151936

# V6.6 activation offsets
V66_A_EMBEDDED_INPUT = 396942152
V66_A_LAYER_INPUT = 400612168
V66_A_RESIDUAL = 404282184
V66_A_Q_SCRATCH = 433380168
V66_A_K_SCRATCH = 437050184
V66_A_V_SCRATCH = 437574472
V66_A_ATTN_SCRATCH = 438098760


def run_v66(token: int, stop_seq: int = -1):
    """Run v6.6 with optional stop."""
    # Load fresh library
    engine_path = V66_DIR / "libckernel_engine.so"
    if engine_path.exists():
        ctypes.CDLL(str(engine_path), mode=ctypes.RTLD_GLOBAL)

    lib_path = V66_DIR / "ck-kernel-inference.so"
    if not lib_path.exists():
        lib_path = V66_DIR / "libmodel.so"

    lib = ctypes.CDLL(str(lib_path))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64

    weights_path = V66_DIR / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"v6.6 init failed: {ret}")
        return None

    base_ptr = lib.ck_model_get_base_ptr()

    if stop_seq >= 0:
        os.environ["CK_STOP_OP"] = str(stop_seq)

    output = (ctypes.c_float * VOCAB_SIZE)()
    ret = lib.ck_model_decode(token, output)

    if stop_seq >= 0:
        del os.environ["CK_STOP_OP"]

    def read_act(offset, size):
        ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(ptr, shape=(size,)).copy()

    result = {
        'logits': np.array(output[:], dtype=np.float32),
        'embedded': read_act(V66_A_EMBEDDED_INPUT, EMBED_DIM),
        'layer_input': read_act(V66_A_LAYER_INPUT, EMBED_DIM),
        'residual': read_act(V66_A_RESIDUAL, EMBED_DIM),
        'q': read_act(V66_A_Q_SCRATCH, NUM_HEADS * HEAD_DIM),
        'k': read_act(V66_A_K_SCRATCH, NUM_KV_HEADS * HEAD_DIM),
        'v': read_act(V66_A_V_SCRATCH, NUM_KV_HEADS * HEAD_DIM),
        'attn': read_act(V66_A_ATTN_SCRATCH, NUM_HEADS * HEAD_DIM),
    }

    lib.ck_model_free()
    return result


def print_stats(name, arr):
    print(f"  {name:20s}: min={arr.min():10.4f}, max={arr.max():10.4f}, mean={arr.mean():10.4f}")


def main():
    token = 100

    print("=" * 70)
    print("V6.6 LAYER 0 ATTENTION TRACE")
    print("=" * 70)

    # Test at each key stop point
    stops = [
        (1, "After rmsnorm (layer_input ready)"),
        (3, "After Q gemv"),
        (4, "After Q bias add"),
        (5, "After K gemv"),
        (6, "After K bias add"),
        (7, "After V gemv"),
        (8, "After V bias add"),
        (9, "After RoPE Q"),
        (10, "After RoPE K + KV store"),
        (11, "After attention"),
        (12, "After out_proj gemv"),
        (13, "After out_proj bias"),
        (14, "After residual add"),
        (15, "After ln2 rmsnorm"),
    ]

    for stop, desc in stops:
        print(f"\n--- Stop {stop}: {desc} ---")
        result = run_v66(token, stop)
        if result:
            print_stats("embedded", result['embedded'])
            print_stats("layer_input", result['layer_input'])
            print_stats("q", result['q'])
            print_stats("k", result['k'])
            print_stats("v", result['v'])
            print_stats("attn", result['attn'])

    # Full run
    print(f"\n--- Full decode ---")
    result = run_v66(token, -1)
    if result:
        print_stats("embedded", result['embedded'])
        print_stats("layer_input", result['layer_input'])
        print_stats("logits", result['logits'])
        print(f"  Argmax: {np.argmax(result['logits'])}")


if __name__ == "__main__":
    main()
