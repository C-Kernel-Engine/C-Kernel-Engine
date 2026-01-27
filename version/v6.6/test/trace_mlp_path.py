#!/usr/bin/env python3
"""
Trace MLP path in v6.6 layer 0.
"""

import ctypes
import numpy as np
import os
from pathlib import Path

V66_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

EMBED_DIM = 896
INTERMEDIATE_SIZE = 4864
VOCAB_SIZE = 151936

# V6.6 activation offsets
V66_A_EMBEDDED_INPUT = 396942152
V66_A_LAYER_INPUT = 400612168
V66_A_MLP_SCRATCH = 441768776
V66_A_LAYER_OUTPUT = 481614664


def run_v66(token: int, stop_seq: int = -1):
    """Run v6.6 with optional stop."""
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
        return None

    base_ptr = lib.ck_model_get_base_ptr()

    if stop_seq >= 0:
        os.environ["CK_STOP_OP"] = str(stop_seq)

    output = (ctypes.c_float * VOCAB_SIZE)()
    lib.ck_model_decode(token, output)

    if stop_seq >= 0:
        del os.environ["CK_STOP_OP"]

    def read_act(offset, size):
        ptr = ctypes.cast(base_ptr + offset, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(ptr, shape=(size,)).copy()

    result = {
        'logits': np.array(output[:], dtype=np.float32),
        'embedded': read_act(V66_A_EMBEDDED_INPUT, EMBED_DIM),
        'layer_input': read_act(V66_A_LAYER_INPUT, EMBED_DIM),
        'mlp_scratch': read_act(V66_A_MLP_SCRATCH, 2 * INTERMEDIATE_SIZE),  # gate + up
        'layer_output': read_act(V66_A_LAYER_OUTPUT, EMBED_DIM),
    }

    lib.ck_model_free()
    return result


def print_stats(name, arr):
    print(f"  {name:20s}: min={arr.min():10.4f}, max={arr.max():10.4f}, mean={arr.mean():10.4f}")


def main():
    token = 100

    print("=" * 70)
    print("V6.6 LAYER 0 MLP TRACE")
    print("=" * 70)

    # MLP path stops (layer 0)
    # Op 16: residual_save
    # Op 15: gemv_q5_0 (mlp_gate_up) - writes gate+up to mlp_scratch
    # Op 16: add_inplace_f32 (bias_add for gate_up)
    # Op 17: swiglu_forward
    # Op 18: gemv_q6_k (mlp_down)
    # Op 19: add_inplace_f32 (bias_add for down)
    # Op 20: residual_add -> layer_output

    stops = [
        (15, "After ln2 (embedded ready)"),
        (16, "After residual save"),
        (17, "After gate_up gemv"),
        (18, "After gate_up bias"),
        (19, "After swiglu"),
        (20, "After down gemv"),
        (21, "After down bias"),
        (22, "After MLP residual add"),
    ]

    for stop, desc in stops:
        print(f"\n--- Stop {stop}: {desc} ---")
        result = run_v66(token, stop)
        if result:
            print_stats("embedded", result['embedded'])
            print_stats("layer_input", result['layer_input'])
            print_stats("mlp_scratch", result['mlp_scratch'][:INTERMEDIATE_SIZE])  # gate only
            print_stats("mlp_up", result['mlp_scratch'][INTERMEDIATE_SIZE:])  # up only
            print_stats("layer_output", result['layer_output'])

    # Check after full layer 0
    print(f"\n--- Stop 24 (start of layer 1) ---")
    result = run_v66(token, 24)
    if result:
        print_stats("embedded", result['embedded'])
        print_stats("layer_input", result['layer_input'])


if __name__ == "__main__":
    main()
