#!/usr/bin/env python3
"""Benchmark planner-owned causal MLA prefill attention in isolation."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--qk-dim", type=int, default=64)
    parser.add_argument("--v-dim", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    if args.threads > 0:
        os.environ["CK_NUM_THREADS"] = str(args.threads)
    library = ctypes.CDLL(str(args.library.resolve()))
    fptr = ctypes.POINTER(ctypes.c_float)
    function = library.deepseek_mla_attention_f32_parallel_dispatch
    function.argtypes = [
        fptr, fptr, fptr, fptr,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
        fptr, ctypes.c_size_t,
    ]
    function.restype = None

    rng = np.random.default_rng(20260827)
    q = np.ascontiguousarray(
        rng.normal(scale=0.03, size=(args.tokens, args.heads, args.qk_dim)),
        dtype=np.float32,
    )
    k = np.ascontiguousarray(
        rng.normal(scale=0.03, size=(args.tokens, args.kv_heads, args.qk_dim)),
        dtype=np.float32,
    )
    v = np.ascontiguousarray(
        rng.normal(scale=0.03, size=(args.tokens, args.kv_heads, args.v_dim)),
        dtype=np.float32,
    )
    output = np.empty((args.tokens, args.heads, args.v_dim), dtype=np.float32)
    score_rows = max(1, args.threads or args.heads)
    scores = np.empty((score_rows, args.tokens), dtype=np.float32)
    ptr = lambda array: array.ctypes.data_as(fptr)

    def invoke() -> float:
        start = time.perf_counter()
        function(
            ptr(q), ptr(k), ptr(v), ptr(output),
            args.heads, args.kv_heads, args.tokens,
            args.qk_dim, args.v_dim,
            np.float32(1.0 / np.sqrt(args.qk_dim)),
            ptr(scores), scores.nbytes,
        )
        return time.perf_counter() - start

    invoke()
    timings = [invoke() for _ in range(args.repetitions)]
    report = {
        "library": str(args.library.resolve()),
        "tokens": args.tokens,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "qk_dim": args.qk_dim,
        "v_dim": args.v_dim,
        "threads": args.threads,
        "wall_seconds": timings,
        "median_seconds": statistics.median(timings),
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
