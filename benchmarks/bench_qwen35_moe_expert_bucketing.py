#!/usr/bin/env python3
"""Benchmark exact Qwen3.5 routed-MoE scheduling on real BUMP weights."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import statistics
import time
from pathlib import Path

import numpy as np


FPTR = ctypes.POINTER(ctypes.c_float)
IPTR = ctypes.POINTER(ctypes.c_int)


def _fptr(array: np.ndarray) -> FPTR:
    return array.ctypes.data_as(FPTR)


def _iptr(array: np.ndarray) -> IPTR:
    return array.ctypes.data_as(IPTR)


def _manifest_entries(path: Path) -> dict[str, tuple[int, int]]:
    entries: dict[str, tuple[int, int]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 5:
            raise ValueError(f"invalid manifest row: {line}")
        entries[fields[0]] = (int(fields[2], 16), int(fields[3], 16))
    return entries


def _configure_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    call_args = [
        FPTR,
        IPTR,
        FPTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        FPTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.moe_swiglu_expert_q4k_q5k_workspace_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.moe_swiglu_expert_q4k_q5k_workspace_bytes.restype = ctypes.c_size_t
    lib.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes.restype = (
        ctypes.c_size_t
    )
    lib.moe_swiglu_expert_forward_q4k_q5k_parallel_workspace.argtypes = call_args
    lib.moe_swiglu_expert_forward_q4k_q5k_parallel_workspace.restype = ctypes.c_int
    lib.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace.argtypes = call_args
    lib.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace.restype = ctypes.c_int
    lib.moe_swiglu_expert_forward_q4k_q5k_bucketed_prepared_workspace.argtypes = [
        *call_args[:6],
        ctypes.c_void_p,
        ctypes.c_void_p,
        *call_args[6:],
    ]
    lib.moe_swiglu_expert_forward_q4k_q5k_bucketed_prepared_workspace.restype = (
        ctypes.c_int
    )
    lib.q4_k_packed_vnni_x8_block_size.restype = ctypes.c_size_t
    lib.ck_q4k_packed_vnni_x8_compact_order_available.restype = ctypes.c_int
    lib.pack_q4_k_to_packed_vnni_x8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    return lib


def _route_inputs(rows: int, experts: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    row_ids = np.arange(rows, dtype=np.int64)[:, None]
    slots = np.arange(top_k, dtype=np.int64)[None, :]
    indices = np.ascontiguousarray(
        ((row_ids * 17 + slots * 31) % experts).astype(np.int32)
    )
    weights = 1.0 / (slots.astype(np.float32) + 1.0)
    weights = np.broadcast_to(weights / np.sum(weights), (rows, top_k))
    return indices, np.ascontiguousarray(weights, dtype=np.float32)


def _weight_pointer(mapping: mmap.mmap, offset: int) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mapping, offset)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 128, 512, 4096])
    parser.add_argument(
        "--provider",
        choices=("both", "all", "row_parallel", "bucketed", "prepared"),
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    hidden_dim = 2048
    intermediate_dim = 512
    experts = 256
    top_k = 8
    lib = _configure_library(args.library.resolve())
    entries = _manifest_entries(args.manifest)
    names = {
        "gate": "layer.0.moe_expert_gate",
        "up": "layer.0.moe_expert_up",
        "down": "layer.0.moe_expert_down",
    }
    missing = [name for name in names.values() if name not in entries]
    if missing:
        raise SystemExit(f"missing manifest entries: {missing}")

    results: list[dict[str, object]] = []
    preparation: dict[str, object] | None = None
    with args.weights.open("r+b", buffering=0) as weights_file:
        weights_map = mmap.mmap(weights_file.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            weight_ptrs = {
                key: _weight_pointer(weights_map, entries[name][0])
                for key, name in names.items()
            }
            prepared_weights: dict[str, ctypes.Array[ctypes.c_char]] = {}
            if args.provider in {"all", "prepared"}:
                if not lib.ck_q4k_packed_vnni_x8_compact_order_available():
                    raise SystemExit(
                        "prepared provider requires AVX-512 VNNI compact-order support"
                    )
                packed_blocks = (
                    ((experts * intermediate_dim + 7) // 8)
                    * (hidden_dim // 256)
                )
                packed_bytes = (
                    packed_blocks * lib.q4_k_packed_vnni_x8_block_size()
                )
                prepared_weights = {
                    "gate": ctypes.create_string_buffer(packed_bytes),
                    "up": ctypes.create_string_buffer(packed_bytes),
                }
                prepare_start = time.perf_counter()
                for name in ("gate", "up"):
                    lib.pack_q4_k_to_packed_vnni_x8(
                        weight_ptrs[name],
                        prepared_weights[name],
                        experts * intermediate_dim,
                        hidden_dim,
                    )
                preparation = {
                    "seconds": time.perf_counter() - prepare_start,
                    "bytes": 2 * packed_bytes,
                }
            for rows in args.rows:
                if rows <= 0:
                    raise SystemExit("rows must be positive")
                rng = np.random.default_rng(1200 + rows)
                hidden = np.ascontiguousarray(
                    rng.normal(0.0, 0.2, size=(rows, hidden_dim)).astype(np.float32)
                )
                indices, routing = _route_inputs(rows, experts, top_k)
                outputs = {
                    "row_parallel": np.empty((rows, hidden_dim), dtype=np.float32),
                    "bucketed": np.empty((rows, hidden_dim), dtype=np.float32),
                    "prepared": np.empty((rows, hidden_dim), dtype=np.float32),
                }
                row_stride = lib.moe_swiglu_expert_q4k_q5k_workspace_bytes(
                    hidden_dim, intermediate_dim
                )
                active_threads = min(
                    rows,
                    int(os.environ.get("CK_NUM_THREADS", os.cpu_count() or 1)),
                    64,
                )
                bucketed_workspace_bytes = (
                    lib.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes(
                        rows, hidden_dim, intermediate_dim, experts, top_k
                    )
                )
                workspace_bytes = {
                    "row_parallel": row_stride * active_threads,
                    "bucketed": bucketed_workspace_bytes,
                    "prepared": bucketed_workspace_bytes,
                }
                workspaces = {
                    name: ctypes.create_string_buffer(size)
                    for name, size in workspace_bytes.items()
                }
                functions = {
                    "row_parallel": (
                        lib.moe_swiglu_expert_forward_q4k_q5k_parallel_workspace
                    ),
                    "bucketed": lib.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace,
                    "prepared": (
                        lib.moe_swiglu_expert_forward_q4k_q5k_bucketed_prepared_workspace
                    ),
                }
                if args.provider == "both":
                    selected = ["row_parallel", "bucketed"]
                elif args.provider == "all":
                    selected = ["row_parallel", "bucketed", "prepared"]
                else:
                    selected = [args.provider]

                row_result: dict[str, object] = {
                    "rows": rows,
                    "threads": active_threads,
                    "workspace_bytes": {
                        name: workspace_bytes[name] for name in selected
                    },
                    "providers": {},
                }
                samples: dict[str, list[float]] = {name: [] for name in selected}

                def run_provider(name: str) -> float:
                    common_args = (
                        _fptr(hidden),
                        _iptr(indices),
                        _fptr(routing),
                        weight_ptrs["gate"],
                        weight_ptrs["up"],
                        weight_ptrs["down"],
                    )
                    trailing_args = (
                        _fptr(outputs[name]),
                        rows,
                        hidden_dim,
                        intermediate_dim,
                        experts,
                        top_k,
                        workspaces[name],
                        workspace_bytes[name],
                    )
                    start = time.perf_counter()
                    if name == "prepared":
                        status = functions[name](
                            *common_args,
                            prepared_weights["gate"],
                            prepared_weights["up"],
                            *trailing_args,
                        )
                    else:
                        status = functions[name](*common_args, *trailing_args)
                    elapsed = time.perf_counter() - start
                    if status != 0:
                        raise RuntimeError(f"{name} returned {status}")
                    return elapsed

                for _ in range(args.warmup):
                    for name in selected:
                        run_provider(name)
                for iteration in range(args.repeats):
                    order = selected if iteration % 2 == 0 else list(reversed(selected))
                    for name in order:
                        samples[name].append(run_provider(name))

                for name in selected:
                    if not samples[name]:
                        raise RuntimeError("at least one repeat is required")
                    row_result["providers"][name] = {
                        "median_seconds": statistics.median(samples[name]),
                        "minimum_seconds": min(samples[name]),
                        "samples_seconds": samples[name],
                    }

                if len(selected) > 1:
                    reference = selected[0]
                    exact = all(
                        np.array_equal(
                            outputs[reference].view(np.uint32),
                            outputs[name].view(np.uint32),
                        )
                        for name in selected[1:]
                    )
                    row_result["bit_exact"] = exact
                    if not exact:
                        raise RuntimeError(
                            f"provider outputs differ for rows={rows}"
                        )
                    reference_time = row_result["providers"][reference][
                        "median_seconds"
                    ]
                    for name in selected[1:]:
                        provider_time = row_result["providers"][name][
                            "median_seconds"
                        ]
                        row_result[f"speedup_{name}_vs_{reference}"] = (
                            reference_time / provider_time
                        )
                results.append(row_result)
                print(json.dumps(row_result, sort_keys=True), flush=True)
        finally:
            weights_map.close()

    report = {
        "library": str(args.library.resolve()),
        "weights": str(args.weights.resolve()),
        "provider": args.provider,
        "preparation": preparation,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
