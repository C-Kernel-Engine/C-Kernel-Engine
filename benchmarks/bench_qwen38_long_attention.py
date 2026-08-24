#!/usr/bin/env python3
"""Benchmark exact Qwen3.8 long-prefill attention scheduling candidates."""

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


SCHEDULES = {
    "kv4": 0,
    "head": 1,
    "tile": 2,
    "kv4x": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=Path("build/libckernel_engine.so"))
    parser.add_argument("--contexts", type=int, nargs="+", default=[150_000, 250_000])
    parser.add_argument("--query-tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--schedules",
        choices=tuple(SCHEDULES),
        nargs="+",
        default=list(SCHEDULES),
    )
    return parser.parse_args()


def load_function(library: Path):
    lib = ctypes.CDLL(str(library.resolve()))
    function = lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_qtile64_schedule
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    return function


def ptr(array: np.ndarray, scalar):
    return array.ctypes.data_as(ctypes.POINTER(scalar))


def allocate_inputs(context: int, query_tokens: int):
    num_heads = 24
    num_kv_heads = 4
    head_dim = 256

    q = np.empty((num_heads, query_tokens, head_dim), dtype=np.float32)
    k = np.empty((num_kv_heads, context, head_dim), dtype=np.uint16)
    v = np.empty_like(k)
    output = np.empty_like(q)
    baseline = np.empty_like(q)

    q_pattern = np.linspace(-0.25, 0.25, head_dim, dtype=np.float32)
    k_pattern = np.linspace(-0.5, 0.5, head_dim, dtype=np.float16).view(np.uint16)
    v_pattern = np.linspace(0.375, -0.375, head_dim, dtype=np.float16).view(np.uint16)
    q[...] = q_pattern
    k[...] = k_pattern
    v[...] = v_pattern
    return q, k, v, output, baseline


def run_context(
    function,
    context: int,
    query_tokens: int,
    repeats: int,
    schedules: list[str],
):
    if context < query_tokens:
        raise ValueError(f"context {context} is smaller than query extent {query_tokens}")
    q, k, v, output, baseline = allocate_inputs(context, query_tokens)
    arguments = (
        ptr(q, ctypes.c_float),
        ptr(k, ctypes.c_uint16),
        ptr(v, ctypes.c_uint16),
        ptr(output, ctypes.c_float),
        24,
        4,
        query_tokens,
        context - query_tokens,
        context,
        256,
        256,
    )
    samples = {name: [] for name in schedules}
    baseline_ready = False
    baseline_hash = None

    for repeat in range(repeats):
        order = schedules if repeat % 2 == 0 else list(reversed(schedules))
        for name in order:
            started = time.perf_counter()
            status = function(*arguments, SCHEDULES[name])
            elapsed = time.perf_counter() - started
            if status != 0:
                raise RuntimeError(f"{name} failed with status {status}")
            samples[name].append(elapsed)

            if name == "kv4" and not baseline_ready:
                baseline[...] = output
                baseline_hash = hashlib.sha256(memoryview(baseline)).hexdigest()
                baseline_ready = True
            elif baseline_ready and not np.array_equal(output, baseline):
                actual_bits = output.ravel().view(np.uint32)
                expected_bits = baseline.ravel().view(np.uint32)
                mismatch = int(np.flatnonzero(actual_bits != expected_bits)[0])
                raise RuntimeError(f"{name} differs from kv4 at float index {mismatch}")

    if not baseline_ready:
        raise RuntimeError("kv4 must be included to certify candidate exactness")

    medians = {name: statistics.median(values) for name, values in samples.items()}
    baseline_seconds = medians["kv4"]
    return {
        "context_tokens": context,
        "query_tokens": query_tokens,
        "past_tokens": context - query_tokens,
        "num_query_heads": 24,
        "num_kv_heads": 4,
        "head_dim": 256,
        "threads_requested": int(os.environ.get("CK_NUM_THREADS", "0") or 0),
        "baseline_sha256": baseline_hash,
        "bit_exact": True,
        "schedules": {
            name: {
                "samples_seconds": values,
                "median_seconds": medians[name],
                "speedup_vs_kv4": baseline_seconds / medians[name],
            }
            for name, values in samples.items()
        },
    }


def main() -> int:
    args = parse_args()
    if "kv4" not in args.schedules:
        raise SystemExit("--schedules must include kv4 for exactness certification")
    schedules = ["kv4", *(name for name in args.schedules if name != "kv4")]
    function = load_function(args.library)
    report = {
        "benchmark": "qwen38_long_attention_schedule",
        "library": str(args.library.resolve()),
        "results": [
            run_context(function, context, args.query_tokens, args.repeats, schedules)
            for context in args.contexts
        ],
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
