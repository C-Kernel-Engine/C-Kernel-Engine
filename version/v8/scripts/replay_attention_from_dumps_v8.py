#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BUILD_DIR = REPO_ROOT / "build"
LLAMA_CPP_BIN = REPO_ROOT / "llama.cpp" / "build" / "bin"
V7_SCRIPTS = REPO_ROOT / "version" / "v7" / "scripts"

if str(V7_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V7_SCRIPTS))

import parity_test  # type: ignore  # noqa: E402


def _require_tensor_any(dumps: list[parity_test.ParityDump], layer: int, names: list[str]) -> np.ndarray:
    for name in names:
        for dump in dumps:
            if dump.layer_id == layer and dump.op_name == name:
                return np.asarray(dump.data, dtype=np.float32)
    raise KeyError(f"missing tensor layer={layer} names={names}")


def _infer_num_tokens(flat: np.ndarray, num_heads: int, head_dim: int) -> int:
    denom = num_heads * head_dim
    if denom <= 0 or flat.size % denom != 0:
        raise ValueError(
            f"cannot infer tokens from {flat.size} elems with heads={num_heads} head_dim={head_dim}"
        )
    return flat.size // denom


def _reshape_token_major_qhd(flat: np.ndarray, num_tokens: int, num_heads: int, head_dim: int) -> np.ndarray:
    expected = num_tokens * num_heads * head_dim
    if flat.size != expected:
        raise ValueError(f"expected {expected} elems, got {flat.size}")
    return flat.reshape(num_tokens, num_heads, head_dim)


def _to_head_major(flat: np.ndarray, num_tokens: int, num_heads: int, head_dim: int) -> np.ndarray:
    token_major = _reshape_token_major_qhd(flat, num_tokens, num_heads, head_dim)
    return np.transpose(token_major, (1, 0, 2)).copy()


def _to_token_major(flat_head_major: np.ndarray, num_tokens: int, num_heads: int, aligned_head_dim: int) -> np.ndarray:
    return np.transpose(flat_head_major.reshape(num_heads, num_tokens, aligned_head_dim), (1, 0, 2)).copy()


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.astype(np.float64) - b.astype(np.float64)
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    cosine = float(np.dot(a64.ravel(), b64.ravel()) / denom) if denom else 1.0
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cosine": cosine,
    }


def _resolve_attention_fn(lib: ctypes.CDLL, mode: str) -> Callable[..., None]:
    if mode == "ggml":
        fn = lib.attention_forward_full_head_major_gqa_ggml_strided
    elif mode == "exact":
        fn = lib.attention_forward_full_head_major_gqa_exact_strided
    elif mode == "flash":
        fn = lib.attention_forward_full_head_major_gqa_flash_strided
    elif mode == "tiled_f16kv":
        fn = lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided
    else:
        raise ValueError(f"unsupported mode: {mode}")

    float_p = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = [
        float_p,
        float_p,
        float_p,
        float_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = None
    return fn


def _default_ggml_cpu_so() -> str | None:
    for name in ("libggml-cpu.so.0.9.8", "libggml-cpu.so.0", "libggml-cpu.so"):
        path = LLAMA_CPP_BIN / name
        if path.exists():
            return str(path)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay full-attention kernels from parity dumps")
    ap.add_argument("--dump", type=Path, required=True, help="Path to dump.bin")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--mode",
        choices=("ggml", "exact", "flash", "tiled_f16kv"),
        default="ggml",
    )
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--disable-multihead-oracle", action="store_true")
    ap.add_argument("--disable-regular-oracle", action="store_true")
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--num-kv-heads", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=72)
    ap.add_argument("--aligned-head-dim", type=int, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--raw-output", type=Path, default=None)
    args = ap.parse_args(argv)

    dumps = parity_test.read_dump_file(args.dump)
    q_flat = _require_tensor_any(dumps, args.layer, ["Qcur_rope"])
    num_tokens = _infer_num_tokens(q_flat, args.num_heads, args.head_dim)
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else args.num_heads
    aligned_head_dim = args.aligned_head_dim if args.aligned_head_dim is not None else args.head_dim

    q = _to_head_major(q_flat, num_tokens, args.num_heads, args.head_dim)
    k = _to_head_major(_require_tensor_any(dumps, args.layer, ["Kcur_rope"]), num_tokens, num_kv_heads, args.head_dim)
    v = _to_head_major(_require_tensor_any(dumps, args.layer, ["v_proj", "Vcur"]), num_tokens, num_kv_heads, args.head_dim)
    ref = _reshape_token_major_qhd(_require_tensor_any(dumps, args.layer, ["kqv_out"]), num_tokens, args.num_heads, args.head_dim)

    lib = ctypes.CDLL(str(BUILD_DIR / "libckernel_engine.so"))
    lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
    lib.ck_set_strict_parity.restype = None
    lib.ck_set_strict_parity(1 if args.strict else 0)

    old_multi = os.environ.get("CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE")
    old_regular = os.environ.get("CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE")
    old_ggml_cpu = os.environ.get("CK_GGML_CPU_SO")
    try:
        if args.strict and old_ggml_cpu is None:
            ggml_cpu_so = _default_ggml_cpu_so()
            if ggml_cpu_so is not None:
                os.environ["CK_GGML_CPU_SO"] = ggml_cpu_so
        if args.disable_multihead_oracle:
            os.environ["CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE"] = "1"
        else:
            os.environ.pop("CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE", None)
        if args.disable_regular_oracle:
            os.environ["CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE"] = "1"
        else:
            os.environ.pop("CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE", None)

        fn = _resolve_attention_fn(lib, args.mode)
        out = np.zeros((args.num_heads, num_tokens, aligned_head_dim), dtype=np.float32)
        fn(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            args.num_heads,
            num_kv_heads,
            num_tokens,
            args.head_dim,
            aligned_head_dim,
            num_tokens,
        )
    finally:
        if old_multi is None:
            os.environ.pop("CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE", None)
        else:
            os.environ["CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE"] = old_multi
        if old_regular is None:
            os.environ.pop("CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE", None)
        else:
            os.environ["CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE"] = old_regular
        if old_ggml_cpu is None:
            os.environ.pop("CK_GGML_CPU_SO", None)
        else:
            os.environ["CK_GGML_CPU_SO"] = old_ggml_cpu
        lib.ck_set_strict_parity(0)

    token_major = _to_token_major(out, num_tokens, args.num_heads, aligned_head_dim)[..., : args.head_dim]
    metrics = _metrics(token_major, ref)
    bitwise_equal = token_major.view(np.uint32) == ref.view(np.uint32)
    worst_idx = np.unravel_index(np.argmax(np.abs(token_major - ref)), token_major.shape)
    report = {
        "mode": args.mode,
        "strict": args.strict,
        "num_tokens": num_tokens,
        "num_heads": args.num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": args.head_dim,
        "aligned_head_dim": aligned_head_dim,
        "disable_multihead_oracle": args.disable_multihead_oracle,
        "disable_regular_oracle": args.disable_regular_oracle,
        "metrics": metrics,
        "bitwise": {
            "equal": int(np.count_nonzero(bitwise_equal)),
            "different": int(bitwise_equal.size - np.count_nonzero(bitwise_equal)),
            "total": int(bitwise_equal.size),
        },
        "worst_index": {
            "query": int(worst_idx[0]),
            "head": int(worst_idx[1]),
            "dim": int(worst_idx[2]),
            "actual": float(token_major[worst_idx]),
            "expected": float(ref[worst_idx]),
        },
    }

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.raw_output is not None:
        np.save(args.raw_output, token_major)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
