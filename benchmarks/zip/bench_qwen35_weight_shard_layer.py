#!/usr/bin/env python3
"""Measure one genuinely weight-sharded Qwen3.5 routed-MoE layer."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import socket
import struct
import time
from typing import Any

import numpy as np


QK_K = 256
Q4_K_BYTES = 144
Q5_K_BYTES = 176
HIDDEN = 2048
INTERMEDIATE = 512
EXPERTS = 256
SELECTED_EXPERTS = tuple(range(8))
TOP_K = len(SELECTED_EXPERTS)
HEADER = struct.Struct("!I")
FPTR = ctypes.POINTER(ctypes.c_float)
IPTR = ctypes.POINTER(ctypes.c_int)


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _fptr(array: np.ndarray) -> FPTR:
    return array.ctypes.data_as(FPTR)


def _iptr(array: np.ndarray) -> IPTR:
    return array.ctypes.data_as(IPTR)


def _recv_exact(connection: socket.socket, target: memoryview) -> None:
    received = 0
    while received < len(target):
        count = connection.recv_into(target[received:])
        if count == 0:
            raise ConnectionError("ZIP peer closed the connection")
        received += count


def _send_json(connection: socket.socket, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    connection.sendall(HEADER.pack(len(encoded)))
    connection.sendall(encoded)


def _recv_json(connection: socket.socket) -> dict[str, Any]:
    size_buffer = bytearray(HEADER.size)
    _recv_exact(connection, memoryview(size_buffer))
    (size,) = HEADER.unpack(size_buffer)
    payload = bytearray(size)
    _recv_exact(connection, memoryview(payload))
    return json.loads(payload)


def _load_manifest(model_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(
        (model_dir / "weights_manifest.json").read_text(encoding="utf-8")
    )
    entries = {str(entry["name"]): entry for entry in manifest["entries"]}
    expected = {
        "layer.0.moe_expert_gate": [EXPERTS, INTERMEDIATE, HIDDEN],
        "layer.0.moe_expert_up": [EXPERTS, INTERMEDIATE, HIDDEN],
        "layer.0.moe_expert_down": [EXPERTS, HIDDEN, INTERMEDIATE],
    }
    for name, shape in expected.items():
        entry = entries.get(name)
        if entry is None or entry.get("shape") != shape:
            raise ValueError(f"unexpected {name} contract: {entry}")
    return manifest, entries


def _bind_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    library.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes.restype = (
        ctypes.c_size_t
    )
    library.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace.argtypes = [
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
    library.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace.restype = (
        ctypes.c_int
    )
    return library


def _make_inputs(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((rows, HIDDEN), dtype=np.float32)
    indices = np.empty((rows, TOP_K), dtype=np.int32)
    base = np.arange(TOP_K, dtype=np.int32)
    for row in range(rows):
        indices[row] = np.roll(base, row % TOP_K)
    route = np.asarray(
        [0.22, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05],
        dtype=np.float32,
    )
    routing = np.broadcast_to(route, (rows, TOP_K)).copy()
    return hidden, indices, routing


class LayerWeights:
    def __init__(self, model_dir: Path, shard: str) -> None:
        self.model_dir = model_dir
        self.shard = shard
        _, entries = _load_manifest(model_dir)
        weights_path = model_dir / "weights.bump"
        self._fd = os.open(weights_path, os.O_RDONLY)
        self._mapping = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)

        if shard == "full":
            self.intermediate = INTERMEDIATE
            self.n_experts = EXPERTS
            self.indices_remap = False
            self.gate = np.frombuffer(
                self._mapping,
                dtype=np.uint8,
                count=EXPERTS * INTERMEDIATE * (HIDDEN // QK_K) * Q4_K_BYTES,
                offset=int(entries["layer.0.moe_expert_gate"]["file_offset"]),
            )
            self.up = np.frombuffer(
                self._mapping,
                dtype=np.uint8,
                count=EXPERTS * INTERMEDIATE * (HIDDEN // QK_K) * Q4_K_BYTES,
                offset=int(entries["layer.0.moe_expert_up"]["file_offset"]),
            )
            self.down = np.frombuffer(
                self._mapping,
                dtype=np.uint8,
                count=EXPERTS * HIDDEN * (INTERMEDIATE // QK_K) * Q5_K_BYTES,
                offset=int(entries["layer.0.moe_expert_down"]["file_offset"]),
            )
            self.weight_hash = "mapped-full-tensors"
            return

        rank = int(shard)
        if rank not in (0, 1):
            raise ValueError(f"unsupported shard: {shard}")
        self.intermediate = INTERMEDIATE // 2
        self.n_experts = TOP_K
        self.indices_remap = True
        begin = rank * self.intermediate
        q4_row_bytes = (HIDDEN // QK_K) * Q4_K_BYTES
        q5_full_row_bytes = (INTERMEDIATE // QK_K) * Q5_K_BYTES
        q5_shard_row_bytes = (self.intermediate // QK_K) * Q5_K_BYTES

        def gather_q4(name: str) -> np.ndarray:
            source = np.frombuffer(
                self._mapping,
                dtype=np.uint8,
                count=EXPERTS * INTERMEDIATE * q4_row_bytes,
                offset=int(entries[name]["file_offset"]),
            ).reshape(EXPERTS, INTERMEDIATE, q4_row_bytes)
            return np.ascontiguousarray(source[list(SELECTED_EXPERTS), begin : begin + self.intermediate])

        self.gate = gather_q4("layer.0.moe_expert_gate")
        self.up = gather_q4("layer.0.moe_expert_up")
        source_down = np.frombuffer(
            self._mapping,
            dtype=np.uint8,
            count=EXPERTS * HIDDEN * q5_full_row_bytes,
            offset=int(entries["layer.0.moe_expert_down"]["file_offset"]),
        ).reshape(EXPERTS, HIDDEN, q5_full_row_bytes)
        byte_begin = (begin // QK_K) * Q5_K_BYTES
        self.down = np.ascontiguousarray(
            source_down[
                list(SELECTED_EXPERTS),
                :,
                byte_begin : byte_begin + q5_shard_row_bytes,
            ]
        )
        digest = hashlib.sha256()
        digest.update(memoryview(self.gate).cast("B"))
        digest.update(memoryview(self.up).cast("B"))
        digest.update(memoryview(self.down).cast("B"))
        self.weight_hash = digest.hexdigest()

    def close(self) -> None:
        if self.shard == "full":
            del self.gate
            del self.up
            del self.down
        self._mapping.close()
        os.close(self._fd)


class LayerRunner:
    def __init__(
        self,
        model_dir: Path,
        library_path: Path,
        shard: str,
        rows: int,
        seed: int,
    ) -> None:
        started = time.perf_counter_ns()
        self.rows = rows
        self.library = _bind_library(library_path)
        self.weights = LayerWeights(model_dir, shard)
        self.hidden, self.indices, self.routing = _make_inputs(rows, seed)
        if self.weights.indices_remap:
            self.indices = np.ascontiguousarray(self.indices)
        workspace_bytes = int(
            self.library.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes(
                rows,
                HIDDEN,
                self.weights.intermediate,
                self.weights.n_experts,
                TOP_K,
            )
        )
        if workspace_bytes <= 0:
            raise RuntimeError("provider rejected the requested workspace shape")
        self.workspace = np.empty(workspace_bytes, dtype=np.uint8)
        self.output = np.empty((rows, HIDDEN), dtype=np.float32)
        self.setup_ms = (time.perf_counter_ns() - started) / 1.0e6

    def run_once(self) -> float:
        started = time.perf_counter_ns()
        rc = self.library.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace(
            _fptr(self.hidden),
            _iptr(self.indices),
            _fptr(self.routing),
            ctypes.c_void_p(self.weights.gate.ctypes.data),
            ctypes.c_void_p(self.weights.up.ctypes.data),
            ctypes.c_void_p(self.weights.down.ctypes.data),
            _fptr(self.output),
            self.rows,
            HIDDEN,
            self.weights.intermediate,
            self.weights.n_experts,
            TOP_K,
            ctypes.c_void_p(self.workspace.ctypes.data),
            self.workspace.nbytes,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
        if rc != 0:
            raise RuntimeError(f"MoE provider failed: rc={rc}")
        return elapsed_ms

    def benchmark(self, warmup: int, repeats: int) -> dict[str, Any]:
        for _ in range(warmup):
            self.run_once()
        samples = [self.run_once() for _ in range(repeats)]
        return {
            "samples_ms": samples,
            "median_ms": float(np.median(samples)),
            "minimum_ms": min(samples),
            "output_hash": _hash(self.output),
            "input_hash": _hash(self.hidden),
            "weight_hash": self.weights.weight_hash,
            "setup_ms": self.setup_ms,
            "workspace_bytes": self.workspace.nbytes,
        }


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    actual64 = actual.astype(np.float64).ravel()
    expected64 = expected.astype(np.float64).ravel()
    denominator = np.linalg.norm(actual64) * np.linalg.norm(expected64)
    return {
        "bit_exact": bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32))),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine": float(np.dot(actual64, expected64) / denominator),
    }


def _run_local(args: argparse.Namespace) -> int:
    runner = LayerRunner(args.model_dir, args.library, args.shard, args.rows, args.seed)
    result = runner.benchmark(args.warmup, args.repeats)
    result.update(
        {
            "mode": "local",
            "host": socket.gethostname(),
            "rows": args.rows,
            "hidden": HIDDEN,
            "intermediate": runner.weights.intermediate,
            "selected_experts": list(SELECTED_EXPERTS),
            "shard": args.shard,
            "threads": int(os.environ.get("CK_NUM_THREADS", "1")),
            "reduction_bytes": args.rows * HIDDEN * np.dtype(np.float32).itemsize,
        }
    )
    print(json.dumps(result, indent=2))
    return 0


def _run_worker(args: argparse.Namespace) -> int:
    runner = LayerRunner(args.model_dir, args.library, args.shard, args.rows, args.seed)
    for _ in range(args.warmup):
        runner.run_once()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.listen, args.port))
    listener.listen(1)
    print(json.dumps({"status": "ready", "host": socket.gethostname(), "port": args.port}), flush=True)
    connection, _ = listener.accept()
    with connection:
        command = bytearray(1)
        _recv_exact(connection, memoryview(command))
        if command != b"R":
            raise ValueError(f"invalid command: {bytes(command)!r}")
        compute_ms = runner.run_once()
        report = {
            "host": socket.gethostname(),
            "compute_ms": compute_ms,
            "input_hash": _hash(runner.hidden),
            "output_hash": _hash(runner.output),
            "weight_hash": runner.weights.weight_hash,
            "setup_ms": runner.setup_ms,
            "shard": args.shard,
        }
        _send_json(connection, report)
        connection.sendall(memoryview(runner.output).cast("B"))
    listener.close()
    return 0


def _run_coordinator(args: argparse.Namespace) -> int:
    local = LayerRunner(args.model_dir, args.library, args.shard, args.rows, args.seed)
    reference = LayerRunner(args.model_dir, args.library, "full", args.rows, args.seed)
    for _ in range(args.warmup):
        local.run_once()
    reference_report = reference.benchmark(args.warmup, args.repeats)

    with socket.create_connection((args.peer, args.port), timeout=30.0) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        wall_started = time.perf_counter_ns()
        connection.sendall(b"R")
        local_compute_ms = local.run_once()
        remote_report = _recv_json(connection)
        remote_output = np.empty_like(local.output)
        transfer_started = time.perf_counter_ns()
        _recv_exact(connection, memoryview(remote_output).cast("B"))
        transfer_ms = (time.perf_counter_ns() - transfer_started) / 1.0e6
        reduction_started = time.perf_counter_ns()
        combined = local.output + remote_output
        reduction_ms = (time.perf_counter_ns() - reduction_started) / 1.0e6
        distributed_wall_ms = (time.perf_counter_ns() - wall_started) / 1.0e6

    report = {
        "mode": "coordinator",
        "host": socket.gethostname(),
        "peer": args.peer,
        "rows": args.rows,
        "hidden": HIDDEN,
        "full_intermediate": INTERMEDIATE,
        "local_shard": args.shard,
        "remote_shard": remote_report["shard"],
        "selected_experts": list(SELECTED_EXPERTS),
        "local_compute_ms": local_compute_ms,
        "remote_compute_ms": remote_report["compute_ms"],
        "compute_critical_ms": max(local_compute_ms, remote_report["compute_ms"]),
        "transfer_ms": transfer_ms,
        "reduction_ms": reduction_ms,
        "distributed_wall_ms": distributed_wall_ms,
        "reduction_bytes": combined.nbytes,
        "reference": reference_report,
        "comparison": _comparison(combined, reference.output),
        "input_hash_match": _hash(local.hidden) == remote_report["input_hash"],
        "combined_output_hash": _hash(combined),
        "local_output_hash": _hash(local.output),
        "remote_output_hash": remote_report["output_hash"],
        "local_weight_hash": local.weights.weight_hash,
        "remote_weight_hash": remote_report["weight_hash"],
    }
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "worker", "coordinator"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--shard", choices=("full", "0", "1"), default="full")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=35035)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--peer")
    parser.add_argument("--port", type=int, default=29635)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.library = (
        args.library.expanduser().resolve()
        if args.library
        else args.model_dir / "libckernel_engine.so"
    )
    if args.rows <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("rows and repeats must be positive; warmup cannot be negative")
    if args.mode == "worker" and args.shard == "full":
        parser.error("worker requires --shard 0 or 1")
    if args.mode == "coordinator" and (args.shard == "full" or not args.peer):
        parser.error("coordinator requires --shard 0 or 1 and --peer")
    return {
        "local": _run_local,
        "worker": _run_worker,
        "coordinator": _run_coordinator,
    }[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
