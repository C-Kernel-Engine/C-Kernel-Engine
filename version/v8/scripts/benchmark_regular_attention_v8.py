#!/usr/bin/env python3
"""Benchmark the v8 llama-regular prefill attention provider."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np


FLOAT_P = ctypes.POINTER(ctypes.c_float)


class ThreadpoolProfile(ctypes.Structure):
    _fields_ = [
        ("dispatch_count", ctypes.c_uint64),
        ("dispatch_total_ns", ctypes.c_uint64),
        ("main_work_ns", ctypes.c_uint64),
        ("completion_wait_ns", ctypes.c_uint64),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-lib", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cpus", help="Process CPU list, for example 0,2,4-10")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        bounds = part.strip().split("-", 1)
        if len(bounds) == 1:
            cpus.add(int(bounds[0]))
        else:
            cpus.update(range(int(bounds[0]), int(bounds[1]) + 1))
    if not cpus:
        raise ValueError("CPU list is empty")
    return cpus


def ptr(values: np.ndarray) -> FLOAT_P:
    return values.ctypes.data_as(FLOAT_P)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> int:
    args = parse_args()
    if args.threads < 1 or args.heads < 1 or args.kv_heads < 1:
        raise SystemExit("threads, heads, and kv-heads must be positive")
    if args.tokens < 1 or args.head_dim < 1 or args.iterations < 1:
        raise SystemExit("tokens, head-dim, and iterations must be positive")
    if args.heads % args.kv_heads:
        raise SystemExit("heads must be divisible by kv-heads")
    if args.cpus:
        os.sched_setaffinity(0, parse_cpu_list(args.cpus))

    library_path = args.engine_lib.resolve()
    lib = ctypes.CDLL(str(library_path))
    kernel = lib.attention_forward_causal_head_major_gqa_llama_regular_strided_sliding_workspace
    kernel.argtypes = [
        FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        FLOAT_P, ctypes.c_size_t, FLOAT_P, ctypes.c_size_t,
        FLOAT_P, ctypes.c_size_t,
    ]
    lib.ck_set_num_threads.argtypes = [ctypes.c_int]
    lib.ck_threadpool_global.restype = ctypes.c_void_p
    lib.ck_threadpool_profile_reset.argtypes = [ctypes.c_void_p]
    lib.ck_threadpool_profile_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ThreadpoolProfile),
    ]
    lib.ck_threadpool_global_destroy.argtypes = []

    rng = np.random.default_rng(20260909)
    shape = (args.heads, args.tokens, args.head_dim)
    kv_shape = (args.kv_heads, args.tokens, args.head_dim)
    q = rng.standard_normal(shape, dtype=np.float32)
    k = rng.standard_normal(kv_shape, dtype=np.float32)
    v = rng.standard_normal(kv_shape, dtype=np.float32)
    output = np.empty(shape, dtype=np.float32)
    padded = max(256, ((args.tokens + 255) // 256) * 256)
    lanes = min(16, args.threads, args.heads * args.tokens)
    scores = np.empty((lanes, padded), dtype=np.float32)
    scaled = np.empty((lanes, padded), dtype=np.float32)
    values = np.empty((args.kv_heads, args.head_dim, padded), dtype=np.float32)

    def invoke() -> None:
        kernel(
            ptr(q), ptr(k), ptr(v), ptr(output),
            args.heads, args.kv_heads, args.tokens, args.head_dim,
            args.head_dim, args.tokens, args.sliding_window,
            ptr(scores), scores.nbytes, ptr(values), values.nbytes,
            ptr(scaled), scaled.nbytes,
        )

    lib.ck_threadpool_global_destroy()
    lib.ck_set_num_threads(args.threads)
    try:
        for _ in range(args.warmups):
            invoke()
        pool = lib.ck_threadpool_global()
        lib.ck_threadpool_profile_reset(pool)
        timings_ms = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            invoke()
            timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        profile = ThreadpoolProfile()
        lib.ck_threadpool_profile_snapshot(pool, ctypes.byref(profile))
    finally:
        lib.ck_threadpool_global_destroy()
        lib.ck_set_num_threads(0)

    result = {
        "schema_version": 1,
        "engine_library": str(library_path),
        "engine_sha256": sha256(library_path),
        "host": platform.node(),
        "cpu_model": cpu_model(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": args.threads,
        "shape": {
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "tokens": args.tokens,
            "head_dim": args.head_dim,
            "sliding_window": args.sliding_window,
        },
        "warmups": args.warmups,
        "iterations": args.iterations,
        "timings_ms": timings_ms,
        "median_ms": statistics.median(timings_ms),
        "minimum_ms": min(timings_ms),
        "maximum_ms": max(timings_ms),
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "threadpool": {
            "dispatch_count": profile.dispatch_count,
            "dispatch_total_ms": profile.dispatch_total_ns / 1_000_000.0,
            "main_work_ms": profile.main_work_ns / 1_000_000.0,
            "completion_wait_ms": profile.completion_wait_ns / 1_000_000.0,
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
