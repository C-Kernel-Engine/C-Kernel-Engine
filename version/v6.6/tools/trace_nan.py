#!/usr/bin/env python3
"""
Trace NaN values by running model with stop points.
"""

import ctypes
import numpy as np
from pathlib import Path
import os
import sys

MODEL_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"

def load_model():
    lib_path = MODEL_DIR / "libmodel.so"
    lib = ctypes.CDLL(str(lib_path))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_prefill.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_prefill.restype = ctypes.c_int
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []

    weights_path = MODEL_DIR / "weights.bump"
    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"Failed to init model: {ret}")
        return None
    return lib


def trace_ops(lib, token_id=100, max_ops=50):
    """Run model stopping at each op to find where NaN starts."""
    vocab_size = lib.ck_model_get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    tokens = (ctypes.c_int32 * 1)(token_id)
    logits = (ctypes.c_float * vocab_size)()

    last_good = -1
    for stop_op in range(max_ops):
        os.environ["CK_STOP_OP"] = str(stop_op)

        # Reset model state (reinit)
        lib.ck_model_free()
        weights_path = MODEL_DIR / "weights.bump"
        lib.ck_model_init(str(weights_path).encode())

        # Run prefill
        lib.ck_model_prefill(tokens, 1, logits)

        # Check logits
        result = np.array(logits[:])
        nan_count = np.sum(np.isnan(result))
        inf_count = np.sum(np.isinf(result))

        if nan_count == 0 and inf_count == 0:
            min_v, max_v = np.min(result), np.max(result)
            if min_v != 0 or max_v != 0:
                print(f"Op {stop_op}: OK (range [{min_v:.4f}, {max_v:.4f}])")
                last_good = stop_op
            else:
                print(f"Op {stop_op}: All zeros (not computed yet)")
        else:
            print(f"Op {stop_op}: NaN={nan_count}, Inf={inf_count} - FIRST BAD OP!")
            break

    print(f"\nLast good op: {last_good}")
    print(f"First bad op: {stop_op}")


def main():
    lib = load_model()
    if lib:
        trace_ops(lib)
        lib.ck_model_free()


if __name__ == "__main__":
    main()
