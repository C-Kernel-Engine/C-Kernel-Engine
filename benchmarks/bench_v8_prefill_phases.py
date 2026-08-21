#!/usr/bin/env python3
"""Measure v8 model initialization and prefill as separate phases."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import resource
import socket
import statistics
import time
from pathlib import Path

import numpy as np


DEFAULT_PATTERN = "1,248045,846,198,9419,248046,198,248045,74455,198"


def _elapsed_ms(start_ns: int) -> float:
    return (time.monotonic_ns() - start_ns) / 1.0e6


def _tokens(pattern: str, count: int) -> np.ndarray:
    values = [int(value.strip()) for value in pattern.split(",") if value.strip()]
    if not values or count <= 0:
        raise ValueError("token pattern and count must be non-empty")
    repeats = (count + len(values) - 1) // len(values)
    return np.asarray((values * repeats)[:count], dtype=np.int32)


def _bind(lib: ctypes.CDLL) -> None:
    lib.ck_model_init.argtypes = [ctypes.c_char_p]
    lib.ck_model_init.restype = ctypes.c_int
    lib.ck_model_embed_tokens.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.ck_model_embed_tokens.restype = ctypes.c_int
    lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_forward.restype = ctypes.c_int
    lib.ck_model_get_logits.argtypes = []
    lib.ck_model_get_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_model_get_active_tokens.argtypes = []
    lib.ck_model_get_active_tokens.restype = ctypes.c_int
    lib.ck_model_get_logits_stride.argtypes = []
    lib.ck_model_get_logits_stride.restype = ctypes.c_int
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--token-count", type=int, default=4096)
    parser.add_argument("--token-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    tokens = np.ascontiguousarray(_tokens(args.token_pattern, args.token_count))
    token_hash = hashlib.sha256(tokens.tobytes()).hexdigest()

    started = time.monotonic_ns()
    lib = ctypes.CDLL(str(model_dir / "libmodel.so"), mode=ctypes.RTLD_GLOBAL)
    load_ms = _elapsed_ms(started)
    _bind(lib)

    rows: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        init_started = time.monotonic_ns()
        status = lib.ck_model_init(str(model_dir / "weights.bump").encode())
        init_ms = _elapsed_ms(init_started)
        if status != 0:
            raise RuntimeError(f"ck_model_init failed with status {status}")
        try:
            embed_started = time.monotonic_ns()
            status = lib.ck_model_embed_tokens(
                tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                int(tokens.size),
            )
            embed_ms = _elapsed_ms(embed_started)
            if status != 0:
                raise RuntimeError(f"ck_model_embed_tokens failed with status {status}")

            forward_started = time.monotonic_ns()
            status = lib.ck_model_forward(None)
            forward_ms = _elapsed_ms(forward_started)
            if status != 0:
                raise RuntimeError(f"ck_model_forward failed with status {status}")

            copy_started = time.monotonic_ns()
            vocab = int(lib.ck_model_get_vocab_size())
            active = int(lib.ck_model_get_active_tokens())
            stride = int(lib.ck_model_get_logits_stride())
            logits_ptr = lib.ck_model_get_logits()
            if (
                vocab <= 0
                or active != tokens.size
                or (stride > 0 and stride < vocab)
                or not logits_ptr
            ):
                raise RuntimeError(
                    f"invalid runtime result vocab={vocab} active={active} stride={stride}"
                )
            if stride > 0:
                flat = np.ctypeslib.as_array(logits_ptr, shape=(active * stride,))
                start = (active - 1) * stride
                logits = flat[start : start + vocab].copy()
            else:
                logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab,)).copy()
            copy_ms = _elapsed_ms(copy_started)
            rows.append(
                {
                    "repeat": repeat,
                    "init_ms": init_ms,
                    "embed_ms": embed_ms,
                    "forward_ms": forward_ms,
                    "logits_copy_ms": copy_ms,
                    "execution_ms": embed_ms + forward_ms,
                    "cold_total_ms": init_ms + embed_ms + forward_ms + copy_ms,
                    "active_tokens": active,
                    "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                }
            )
        finally:
            lib.ck_model_free()

    hashes = {str(row["logits_sha256"]) for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"runtime is not repeatable: {sorted(hashes)}")
    report = {
        "schema_version": 1,
        "host": socket.gethostname(),
        "model_dir": str(model_dir),
        "token_count": int(tokens.size),
        "token_sha256": token_hash,
        "library_load_ms": load_ms,
        "repeats": rows,
        "median_init_ms": statistics.median(float(row["init_ms"]) for row in rows),
        "median_execution_ms": statistics.median(
            float(row["execution_ms"]) for row in rows
        ),
        "median_forward_ms": statistics.median(
            float(row["forward_ms"]) for row in rows
        ),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
