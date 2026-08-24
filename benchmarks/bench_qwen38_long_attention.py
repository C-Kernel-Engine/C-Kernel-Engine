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
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-auto",
        action="store_true",
        help="Benchmark the production map ABI in addition to research schedules.",
    )
    parser.add_argument("--reuse-query-tiles", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--reuse-concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--schedules",
        choices=tuple(SCHEDULES),
        nargs="+",
        default=list(SCHEDULES),
    )
    return parser.parse_args()


def load_functions(library: Path):
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
    reuse = lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_config
    reuse.argtypes = function.argtypes[:-1] + [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    reuse.restype = ctypes.c_int
    workspace_bytes = (
        lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_gqa_reuse_workspace_bytes
    )
    workspace_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    workspace_bytes.restype = ctypes.c_size_t
    auto = lib.attention_forward_causal_head_major_gqa_prefill_append_f16cache_auto_workspace
    auto.argtypes = function.argtypes[:-1] + [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    auto.restype = ctypes.c_int
    return function, reuse, workspace_bytes, auto


def ptr(array: np.ndarray, scalar):
    return array.ctypes.data_as(ctypes.POINTER(scalar))


def allocate_inputs(
    context: int,
    query_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
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
    reuse,
    reuse_workspace_bytes,
    auto,
    context: int,
    query_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    repeats: int,
    schedules: list[str],
    reuse_configs: list[tuple[int, int]],
    include_auto: bool,
):
    if context < query_tokens:
        raise ValueError(f"context {context} is smaller than query extent {query_tokens}")
    q, k, v, output, baseline = allocate_inputs(
        context, query_tokens, num_heads, num_kv_heads, head_dim
    )
    workers = int(os.environ.get("CK_NUM_THREADS", "0") or 0)
    if workers <= 0:
        workers = os.cpu_count() or 1
    required_workspace = max(
        reuse_workspace_bytes(
            num_heads,
            num_kv_heads,
            head_dim,
            workers,
            query_tile,
            concurrency,
        )
        for query_tile, concurrency in reuse_configs
    )
    workspace_storage = np.empty(required_workspace + 63, dtype=np.uint8)
    workspace_address = (workspace_storage.ctypes.data + 63) & ~63
    token_workspace = np.empty(2 * num_heads * head_dim, dtype=np.float32)
    arguments = (
        ptr(q, ctypes.c_float),
        ptr(k, ctypes.c_uint16),
        ptr(v, ctypes.c_uint16),
        ptr(output, ctypes.c_float),
        num_heads,
        num_kv_heads,
        query_tokens,
        context - query_tokens,
        context,
        head_dim,
        head_dim,
    )
    candidate_names = [
        f"reuse-q{query_tile}-c{concurrency}"
        for query_tile, concurrency in reuse_configs
    ]
    all_candidates = [*schedules]
    if include_auto:
        all_candidates.append("auto")
    all_candidates.extend(candidate_names)
    samples = {name: [] for name in all_candidates}
    baseline_ready = False
    baseline_hash = None

    for repeat in range(repeats):
        order = all_candidates if repeat % 2 == 0 else list(reversed(all_candidates))
        for name in order:
            started = time.perf_counter()
            if name in SCHEDULES:
                status = function(*arguments, SCHEDULES[name])
            elif name == "auto":
                status = auto(
                    *arguments,
                    3,
                    ptr(token_workspace, ctypes.c_float),
                    token_workspace.nbytes,
                    ctypes.c_void_p(workspace_address),
                    required_workspace,
                    24,
                    4,
                    256,
                    4096,
                    8192,
                    16,
                    128,
                    4,
                )
            else:
                _, query_text, concurrency_text = name.split("-")
                status = reuse(
                    *arguments,
                    int(query_text.removeprefix("q")),
                    int(concurrency_text.removeprefix("c")),
                    ctypes.c_void_p(workspace_address),
                    required_workspace,
                )
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
        "num_query_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
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
    if args.num_heads <= 0 or args.num_kv_heads <= 0:
        raise SystemExit("head counts must be positive")
    if args.num_heads % args.num_kv_heads:
        raise SystemExit("--num-heads must be divisible by --num-kv-heads")
    if "kv4" not in args.schedules:
        raise SystemExit("--schedules must include kv4 for exactness certification")
    schedules = ["kv4", *(name for name in args.schedules if name != "kv4")]
    function, reuse, reuse_workspace_bytes, auto = load_functions(args.library)
    reuse_configs = [
        (query_tile, concurrency)
        for query_tile in args.reuse_query_tiles
        for concurrency in args.reuse_concurrency
    ]
    report = {
        "benchmark": "qwen38_long_attention_schedule",
        "library": str(args.library.resolve()),
        "results": [
            run_context(
                function,
                reuse,
                reuse_workspace_bytes,
                auto,
                context,
                args.query_tokens,
                args.num_heads,
                args.num_kv_heads,
                args.head_dim,
                args.repeats,
                schedules,
                reuse_configs,
                args.include_auto,
            )
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
