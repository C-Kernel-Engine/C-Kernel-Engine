#!/usr/bin/env python3
"""
ck_model_smoke_v4.py
====================

Lightweight v4 runtime smoke test:
  - initialize model
  - prefill a short prompt
  - run a few decode steps
  - verify canaries (if available)
"""

import argparse
import ctypes
import random
import sys
from pathlib import Path


def bind_optional(lib, name, argtypes, restype):
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return None
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def main() -> int:
    ap = argparse.ArgumentParser(description="v4 runtime smoke test")
    ap.add_argument("--model-dir", required=True, help="Directory with libmodel.so")
    ap.add_argument("--weights", help="Path to weights.bump (default: model-dir/weights.bump)")
    ap.add_argument("--prompt-len", type=int, default=4, help="Prompt length")
    ap.add_argument("--decode-steps", type=int, default=2, help="Decode steps")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    lib_path = model_dir / "libmodel.so"
    if not lib_path.exists():
        print(f"Error: libmodel.so not found at {lib_path}", file=sys.stderr)
        return 1

    weights_path = Path(args.weights) if args.weights else model_dir / "weights.bump"
    if not weights_path.exists():
        print(f"Error: weights not found at {weights_path}", file=sys.stderr)
        return 1

    lib = ctypes.CDLL(str(lib_path))

    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int

    lib.ck_model_get_context_window.argtypes = []
    lib.ck_model_get_context_window.restype = ctypes.c_int

    lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.ck_model_embed_tokens.restype = ctypes.c_int

    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int

    ck_model_kv_cache_enable = bind_optional(
        lib, "ck_model_kv_cache_enable", [ctypes.c_int], ctypes.c_int
    )
    ck_model_decode = bind_optional(
        lib, "ck_model_decode", [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)], ctypes.c_int
    )
    ck_model_verify_canaries = bind_optional(
        lib, "ck_model_verify_canaries", [], ctypes.c_int
    )

    ret = lib.ck_model_init(str(weights_path).encode())
    if ret != 0:
        print(f"Error: ck_model_init failed (code {ret})", file=sys.stderr)
        return 1

    vocab_size = lib.ck_model_get_vocab_size()
    context = lib.ck_model_get_context_window()
    prompt_len = max(1, min(args.prompt_len, context))
    decode_steps = max(0, min(args.decode_steps, context - prompt_len))

    if ck_model_kv_cache_enable:
        ck_model_kv_cache_enable(context)

    random.seed(args.seed)
    tokens = [random.randrange(vocab_size) for _ in range(prompt_len)]
    token_arr = (ctypes.c_int32 * prompt_len)(*tokens)
    ret = lib.ck_model_embed_tokens(token_arr, ctypes.c_int(prompt_len))
    if ret != 0:
        print(f"Error: ck_model_embed_tokens failed (code {ret})", file=sys.stderr)
        lib.ck_model_free()
        return 1

    ret = lib.ck_model_forward(None)
    if ret != 0:
        print(f"Error: ck_model_forward failed (code {ret})", file=sys.stderr)
        lib.ck_model_free()
        return 1

    if ck_model_decode:
        for _ in range(decode_steps):
            token = random.randrange(vocab_size)
            ret = ck_model_decode(ctypes.c_int32(token), None)
            if ret != 0:
                print(f"Error: ck_model_decode failed (code {ret})", file=sys.stderr)
                lib.ck_model_free()
                return 1

    if ck_model_verify_canaries:
        errors = ck_model_verify_canaries()
        if errors != 0:
            print(f"Error: canary check failed ({errors} errors)", file=sys.stderr)
            lib.ck_model_free()
            return 1

    lib.ck_model_free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
