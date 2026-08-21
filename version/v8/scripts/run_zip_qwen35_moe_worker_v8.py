#!/usr/bin/env python3
"""Serve Qwen3.5 MoE token rows from a local BUMP artifact."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import struct
import time
from typing import Any

import numpy as np


MAGIC = 0x434B5A50
VERSION = 1
REQUEST = 1
RESPONSE = 2
HEADER = struct.Struct("<12I4Q")
if HEADER.size != 80:
    raise RuntimeError("ZIP worker protocol header must remain 80 bytes")
PROT_READ = 0x1
MAP_PRIVATE = 0x02


def _recv_exact(sock: socket.socket, target: memoryview) -> int:
    received = 0
    while received < len(target):
        count = sock.recv_into(target[received:])
        if count == 0:
            raise ConnectionError("coordinator closed the ZIP connection")
        received += count
    return received


def _load_manifest(model_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(
        (model_dir / "weights_manifest.json").read_text(encoding="utf-8")
    )
    entries = {str(entry["name"]): entry for entry in manifest["entries"]}
    return manifest, entries


def _map_weights(path: Path) -> tuple[ctypes.CDLL, int, int, int]:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong,
    ]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    fd = os.open(path, os.O_RDONLY)
    size = path.stat().st_size
    address = libc.mmap(None, size, PROT_READ, MAP_PRIVATE, fd, 0)
    failed = ctypes.c_void_p(-1).value
    if address in (None, failed):
        error = ctypes.get_errno()
        os.close(fd)
        raise OSError(error, os.strerror(error), str(path))
    return libc, int(address), size, fd


def _bind_kernels(library_path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    library.moe_swiglu_expert_q4k_q5k_workspace_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.moe_swiglu_expert_q4k_q5k_workspace_bytes.restype = ctypes.c_size_t
    library.moe_swiglu_expert_forward_q4k_q5k_parallel_workspace.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.moe_swiglu_expert_forward_q4k_q5k_parallel_workspace.restype = ctypes.c_int
    library.moe_swiglu_shared_q8_0_gated_workspace_bytes.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.moe_swiglu_shared_q8_0_gated_workspace_bytes.restype = ctypes.c_size_t
    library.moe_swiglu_shared_forward_q8_0_gated_workspace.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.moe_swiglu_shared_forward_q8_0_gated_workspace.restype = ctypes.c_int
    return library


def _weight_pointer(
    entries: dict[str, dict[str, Any]], base: int, layer: int, suffix: str
) -> ctypes.c_void_p:
    name = f"layer.{layer}.{suffix}"
    if name not in entries:
        raise KeyError(f"missing worker weight: {name}")
    return ctypes.c_void_p(base + int(entries[name]["file_offset"]))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29535)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    manifest, entries = _load_manifest(model_dir)
    expected_layers = int(manifest.get("num_layers", args.layers))
    if args.layers != expected_layers:
        raise ValueError(
            f"worker layer count {args.layers} does not match manifest {expected_layers}"
        )
    library = _bind_kernels(model_dir / "libckernel_engine.so")
    libc, weight_base, weight_size, weight_fd = _map_weights(model_dir / "weights.bump")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.listen, args.port))
    listener.listen(1)
    print(
        json.dumps(
            {
                "status": "listening",
                "host": socket.gethostname(),
                "address": args.listen,
                "port": args.port,
                "layers": args.layers,
            }
        ),
        flush=True,
    )

    session_started = 0
    routed_ns = 0
    shared_ns = 0
    receive_ns = 0
    send_ns = 0
    bytes_received = 0
    bytes_sent = 0
    output_hash = hashlib.sha256()
    status = "pass"
    error: str | None = None
    connection: socket.socket | None = None
    try:
        connection, peer = listener.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        session_started = time.monotonic_ns()
        for layer in range(args.layers):
            header_buffer = bytearray(HEADER.size)
            receive_started = time.monotonic_ns()
            bytes_received += _recv_exact(connection, memoryview(header_buffer))
            fields = HEADER.unpack(header_buffer)
            (
                magic,
                version,
                sequence,
                total_rows,
                hidden_dim,
                intermediate_dim,
                n_experts,
                top_k,
                remote_begin,
                remote_rows,
                kind,
                _reserved,
                hidden_bytes,
                index_bytes,
                routing_bytes,
                output_bytes,
            ) = fields
            expected_hidden_bytes = remote_rows * hidden_dim * np.dtype(np.float32).itemsize
            expected_route_bytes = remote_rows * top_k * np.dtype(np.int32).itemsize
            if (
                magic != MAGIC
                or version != VERSION
                or sequence != layer
                or kind != REQUEST
                or remote_begin + remote_rows != total_rows
                or hidden_bytes != expected_hidden_bytes
                or index_bytes != expected_route_bytes
                or routing_bytes != expected_route_bytes
                or output_bytes != expected_hidden_bytes
            ):
                raise ValueError(f"invalid ZIP request contract at layer {layer}: {fields}")

            hidden = np.empty((remote_rows, hidden_dim), dtype=np.float32)
            indices = np.empty((remote_rows, top_k), dtype=np.int32)
            routing = np.empty((remote_rows, top_k), dtype=np.float32)
            bytes_received += _recv_exact(connection, memoryview(hidden).cast("B"))
            bytes_received += _recv_exact(connection, memoryview(indices).cast("B"))
            bytes_received += _recv_exact(connection, memoryview(routing).cast("B"))
            receive_ns += time.monotonic_ns() - receive_started

            routed = np.zeros((remote_rows, hidden_dim), dtype=np.float32)
            output = np.empty((remote_rows, hidden_dim), dtype=np.float32)
            routed_stride = int(
                library.moe_swiglu_expert_q4k_q5k_workspace_bytes(
                    hidden_dim, intermediate_dim
                )
            )
            worker_threads = max(1, int(os.environ.get("CK_NUM_THREADS", "1")))
            routed_workspace = np.empty(
                routed_stride * min(remote_rows, worker_threads), dtype=np.uint8
            )
            routed_started = time.monotonic_ns()
            rc = library.moe_swiglu_expert_forward_q4k_q5k_parallel_workspace(
                hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                routing.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                _weight_pointer(entries, weight_base, layer, "moe_expert_gate"),
                _weight_pointer(entries, weight_base, layer, "moe_expert_up"),
                _weight_pointer(entries, weight_base, layer, "moe_expert_down"),
                routed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                remote_rows,
                hidden_dim,
                intermediate_dim,
                n_experts,
                top_k,
                ctypes.c_void_p(routed_workspace.ctypes.data),
                routed_workspace.nbytes,
            )
            routed_ns += time.monotonic_ns() - routed_started
            if rc != 0:
                raise RuntimeError(f"routed provider failed at layer {layer}: rc={rc}")

            shared_bytes = int(
                library.moe_swiglu_shared_q8_0_gated_workspace_bytes(
                    hidden_dim, intermediate_dim
                )
            )
            shared_workspace = np.empty(shared_bytes, dtype=np.uint8)
            shared_started = time.monotonic_ns()
            rc = library.moe_swiglu_shared_forward_q8_0_gated_workspace(
                hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                routed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                _weight_pointer(entries, weight_base, layer, "moe_shared_gate"),
                _weight_pointer(entries, weight_base, layer, "moe_shared_up"),
                _weight_pointer(entries, weight_base, layer, "moe_shared_down"),
                ctypes.cast(
                    _weight_pointer(entries, weight_base, layer, "moe_shared_router"),
                    ctypes.POINTER(ctypes.c_float),
                ),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                remote_rows,
                hidden_dim,
                intermediate_dim,
                ctypes.c_void_p(shared_workspace.ctypes.data),
                shared_workspace.nbytes,
            )
            shared_ns += time.monotonic_ns() - shared_started
            if rc != 0:
                raise RuntimeError(f"shared provider failed at layer {layer}: rc={rc}")

            output_hash.update(output.tobytes())
            response = HEADER.pack(
                MAGIC,
                VERSION,
                layer,
                total_rows,
                hidden_dim,
                intermediate_dim,
                n_experts,
                top_k,
                remote_begin,
                remote_rows,
                RESPONSE,
                0,
                0,
                0,
                0,
                output.nbytes,
            )
            send_started = time.monotonic_ns()
            connection.sendall(response)
            connection.sendall(memoryview(output).cast("B"))
            send_ns += time.monotonic_ns() - send_started
            bytes_sent += len(response) + output.nbytes
    except Exception as exc:
        status = "fail"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed_ns = time.monotonic_ns() - session_started if session_started else 0
        report = {
            "schema_version": 1,
            "status": status,
            "error": error,
            "host": socket.gethostname(),
            "model_dir": str(model_dir),
            "layers": args.layers,
            "weight_bytes_mapped": weight_size,
            "wall_ms": elapsed_ns / 1.0e6,
            "routed_ms": routed_ns / 1.0e6,
            "shared_ms": shared_ns / 1.0e6,
            "receive_ms": receive_ns / 1.0e6,
            "send_ms": send_ns / 1.0e6,
            "bytes_received": bytes_received,
            "bytes_sent": bytes_sent,
            "output_sequence_sha256": output_hash.hexdigest(),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "threads": os.environ.get("CK_NUM_THREADS"),
        }
        _write_report(args.report, report)
        if connection is not None:
            connection.close()
        listener.close()
        libc.munmap(ctypes.c_void_p(weight_base), weight_size)
        os.close(weight_fd)

    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
