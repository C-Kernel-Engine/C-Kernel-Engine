#!/usr/bin/env python3
"""
Compare token IDs between CK tokenizer (from ck_build/libmodel.so) and
HuggingFace tokenizer (transformers).

Usage:
  python version/v6.6/scripts/compare_tokens_hf_ck.py \
    --ck-model-dir ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build \
    --hf-model gg-hf-gm/gemma-3-270m-it \
    --prompt "Hello"
"""

import argparse
import ctypes
import os
from pathlib import Path


def ck_encode_tokens(model_dir: Path, prompt: str):
    lib_path = model_dir / "libmodel.so"
    weights_path = model_dir / "weights.bump"
    if not lib_path.exists():
        raise SystemExit(f"Missing {lib_path}")
    if not weights_path.exists():
        raise SystemExit(f"Missing {weights_path}")

    # Ensure runtime deps are found
    build_dir = Path(__file__).resolve().parents[2] / "build"
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
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

    prompt_bytes = prompt.encode("utf-8")
    n = lib.ck_model_encode_text(prompt_bytes, -1)
    if n <= 0:
        lib.ck_model_free()
        raise SystemExit("encode_text returned 0 tokens")

    buf = lib.ck_model_get_token_buffer()
    ids = [buf[i] for i in range(n)]

    lib.ck_model_free()
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare HF vs CK tokenization")
    parser.add_argument("--ck-model-dir", required=True, type=Path,
                        help="Path to ck_build directory (libmodel.so, weights.bump)")
    parser.add_argument("--hf-model", required=True, type=str,
                        help="HuggingFace model or tokenizer repo id/path")
    parser.add_argument("--prompt", required=True, type=str, help="Prompt text")
    parser.add_argument("--no-fast", action="store_true",
                        help="Use slow HF tokenizer (use_fast=False)")
    parser.add_argument("--no-bos", action="store_true",
                        help="Do not add BOS for HF even if available")
    args = parser.parse_args()

    # CK tokens
    ck_ids = ck_encode_tokens(args.ck_model_dir, args.prompt)
    print(f"[CK] tokens ({len(ck_ids)}): {ck_ids}")

    # HF tokens
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print("[HF] transformers not available:", e)
        print("Install with: pip install transformers")
        return 1

    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=not args.no_fast)
    except Exception as e:
        if args.no_fast:
            raise
        # Fast tokenizer failed (often due to tokenizers version mismatch).
        # Fall back to slow tokenizer (SentencePiece/Python) for robustness.
        print(f"[HF] fast tokenizer failed, retrying with use_fast=False: {e}")
        tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)

    hf_raw = tok.encode(args.prompt, add_special_tokens=False)
    print(f"[HF] raw tokens ({len(hf_raw)}): {hf_raw}")

    hf_add = tok.encode(args.prompt, add_special_tokens=True)
    print(f"[HF] add_special_tokens ({len(hf_add)}): {hf_add}")

    if tok.bos_token_id is not None and not args.no_bos:
        hf_bos = [tok.bos_token_id] + hf_raw
        print(f"[HF] raw + BOS ({len(hf_bos)}): {hf_bos}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
