#!/usr/bin/env python3
from __future__ import annotations

"""
Compare CK batched prefill logits against CK stepwise decode logits.

This is tokenizer-free and llama-free. It answers a narrower question than the
llama parity runner: for the same explicit token prefix, do CK's prefill path
and decode-state path produce the same next-token logits?
"""

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_first_token_logits_v8 import compare_logits, discover_ck_model_dir, parse_tokens_csv  # type: ignore


def _init_model(lib: ctypes.CDLL, model_dir: Path) -> str:
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int

    candidates = [model_dir / "weights.bump", model_dir]
    if model_dir.name in {".ck_build", "ck_build"}:
        candidates.extend([model_dir.parent / "weights.bump", model_dir.parent])

    tried: list[str] = []
    for cand in candidates:
        p = cand.resolve()
        if str(p) in tried:
            continue
        tried.append(str(p))
        if lib.ck_model_init(str(p).encode("utf-8")) == 0:
            return str(p)
    raise RuntimeError(f"ck_model_init failed for: {', '.join(tried)}")


def _extract_logits(lib: ctypes.CDLL, vocab: int, fallback_active: int) -> tuple[np.ndarray, int, int]:
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    ptr = lib.ck_model_get_logits()
    if not ptr:
        raise RuntimeError("ck_model_get_logits returned null")

    stride = 0
    active = fallback_active
    if hasattr(lib, "ck_model_get_logits_stride"):
        lib.ck_model_get_logits_stride.argtypes = []
        lib.ck_model_get_logits_stride.restype = ctypes.c_int
        stride = int(lib.ck_model_get_logits_stride())
    if hasattr(lib, "ck_model_get_active_tokens"):
        lib.ck_model_get_active_tokens.argtypes = []
        lib.ck_model_get_active_tokens.restype = ctypes.c_int
        active = int(lib.ck_model_get_active_tokens())

    if stride > 0 and active > 0:
        flat = np.ctypeslib.as_array(ptr, shape=(active * stride,))
        start = (active - 1) * stride
        logits = flat[start : start + vocab].astype(np.float32, copy=True)
    else:
        logits = np.ctypeslib.as_array(ptr, shape=(vocab,)).astype(np.float32, copy=True)
    return logits, stride, active


def load_ck_logits_mode(model_dir: Path, tokens: list[int], mode: str) -> dict[str, Any]:
    lib = ctypes.CDLL(str(model_dir / "libmodel.so"), mode=ctypes.RTLD_GLOBAL)
    init_dir = _init_model(lib, model_dir)

    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    vocab = int(lib.ck_model_get_vocab_size())
    if vocab <= 0:
        raise RuntimeError(f"invalid vocab size: {vocab}")

    has_free = hasattr(lib, "ck_model_free")
    if has_free:
        lib.ck_model_free.argtypes = []
        lib.ck_model_free.restype = None

    try:
        if hasattr(lib, "ck_model_kv_cache_reset"):
            lib.ck_model_kv_cache_reset.argtypes = []
            lib.ck_model_kv_cache_reset.restype = None
            lib.ck_model_kv_cache_reset()

        if mode == "decode":
            if not hasattr(lib, "ck_model_decode"):
                raise RuntimeError("ck_model_decode is unavailable")
            lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
            lib.ck_model_decode.restype = ctypes.c_int
            for tok in tokens:
                rc = lib.ck_model_decode(ctypes.c_int32(int(tok)), None)
                if rc != 0:
                    raise RuntimeError(f"ck_model_decode failed rc={rc}")
        elif mode == "prefill":
            lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
            lib.ck_model_embed_tokens.restype = ctypes.c_int
            lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
            lib.ck_model_forward.restype = ctypes.c_int
            arr = (ctypes.c_int32 * len(tokens))(*[int(t) for t in tokens])
            rc = lib.ck_model_embed_tokens(arr, len(tokens))
            if rc != 0:
                raise RuntimeError(f"ck_model_embed_tokens failed rc={rc}")
            rc = lib.ck_model_forward(None)
            if rc != 0:
                raise RuntimeError(f"ck_model_forward failed rc={rc}")
        else:
            raise ValueError(f"unknown mode: {mode}")

        logits, stride, active = _extract_logits(lib, vocab, len(tokens))
        return {
            "mode": mode,
            "init_dir": init_dir,
            "vocab": vocab,
            "stride": stride,
            "active_tokens": active,
            "logits": logits,
        }
    finally:
        if has_free:
            try:
                lib.ck_model_free()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare CK prefill logits vs CK decode logits for explicit tokens")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    model_dir = discover_ck_model_dir(args.model_dir)
    tokens = parse_tokens_csv(args.tokens)
    prefill = load_ck_logits_mode(model_dir, tokens, "prefill")
    decode = load_ck_logits_mode(model_dir, tokens, "decode")
    cmp = compare_logits(decode["logits"], prefill["logits"], int(args.top_k))

    report = {
        "status": "pass" if cmp["top1_match"] else "fail",
        "pass": bool(cmp["top1_match"]),
        "model_dir": str(model_dir),
        "tokens": tokens,
        "prefill": {k: v for k, v in prefill.items() if k != "logits"},
        "decode": {k: v for k, v in decode.items() if k != "logits"},
        "compare_decode_minus_prefill": cmp,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.summary:
        print(
            f"status={report['status']} decode_top1={cmp['top1_ck']} prefill_top1={cmp['top1_llama']} "
            f"cosine={cmp['cosine']:.6f} rmse={cmp['rmse']:.6f} "
            f"topk_overlap={cmp['topk_overlap_count']}/{args.top_k}"
        )
    else:
        print(json.dumps(report))
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
