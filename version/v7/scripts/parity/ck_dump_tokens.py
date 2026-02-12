#!/usr/bin/env python3
"""
Dump token IDs produced by the model's built-in tokenizer.

Usage:
  python version/v7/scripts/parity/ck_dump_tokens.py --model-dir /path/to/ck_build --prompt "Hello"
"""

import argparse
import ctypes
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump CK token IDs for a prompt")
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="Path to ck_build directory (contains libmodel.so, weights.bump)")
    parser.add_argument("--prompt", required=True, type=str, help="Prompt text")
    args = parser.parse_args()

    model_dir = args.model_dir
    lib_path = model_dir / "libmodel.so"
    weights_path = model_dir / "weights.bump"
    if not lib_path.exists():
        raise SystemExit(f"Missing {lib_path}")
    if not weights_path.exists():
        raise SystemExit(f"Missing {weights_path}")

    # Ensure runtime deps are found
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    build_dir = Path(__file__).resolve().parents[4] / "build"
    os.environ["LD_LIBRARY_PATH"] = f"{build_dir}:{model_dir}:{ld_path}"

    lib = ctypes.CDLL(str(lib_path))
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_encode_text.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.ck_model_encode_text.restype = ctypes.c_int
    lib.ck_model_get_token_buffer.argtypes = []
    lib.ck_model_get_token_buffer.restype = ctypes.POINTER(ctypes.c_int32)
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None

    rc = lib.ck_model_init(str(weights_path).encode("utf-8"))
    if rc != 0:
        raise SystemExit(f"ck_model_init failed: {rc}")

    prompt_bytes = args.prompt.encode("utf-8")
    n = lib.ck_model_encode_text(prompt_bytes, -1)
    if n <= 0:
        lib.ck_model_free()
        raise SystemExit("encode_text returned 0 tokens")

    buf = lib.ck_model_get_token_buffer()
    ids = [buf[i] for i in range(n)]
    print(f"[CK] tokens ({n}): {ids}")

    lib.ck_model_free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
