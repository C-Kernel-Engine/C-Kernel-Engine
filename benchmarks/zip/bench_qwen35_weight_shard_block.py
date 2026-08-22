#!/usr/bin/env python3
"""Measure one complete weight-sharded Qwen3.5 attention+MoE block."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np


LAYER = 3
HIDDEN = 2048
HEAD_DIM = 256
Q_HEADS = 16
KV_HEADS = 2
INTERMEDIATE = 512
EXPERTS = 256
TOP_K = 8
Q8_BLOCK = 32
Q8_BYTES = 34
Q4_BLOCK = 256
Q4_BYTES = 144
Q5_BLOCK = 256
Q5_BYTES = 176
EPS = np.float32(1.0e-6)
ATTENTION_REDUCTION = 3
FPTR = ctypes.POINTER(ctypes.c_float)
IPTR = ctypes.POINTER(ctypes.c_int)


def _load_transport_helpers() -> Any:
    path = Path(__file__).with_name("bench_qwen35_weight_shard_layer.py")
    spec = importlib.util.spec_from_file_location("cke_zip_moe_bench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import transport helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSPORT = _load_transport_helpers()


def _fptr(array: np.ndarray) -> FPTR:
    return array.ctypes.data_as(FPTR)


def _iptr(array: np.ndarray) -> IPTR:
    return array.ctypes.data_as(IPTR)


def _vptr(array: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(array.ctypes.data)


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    actual64 = actual.astype(np.float64).ravel()
    expected64 = expected.astype(np.float64).ravel()
    denominator = np.linalg.norm(actual64) * np.linalg.norm(expected64)
    return {
        "bit_exact": bool(
            np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
        ),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine": float(np.dot(actual64, expected64) / denominator),
    }


def _make_input(rows: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(
        (rows, HIDDEN), dtype=np.float32
    )


def _q8_row_bytes(width: int) -> int:
    if width % Q8_BLOCK:
        raise ValueError(f"Q8_0 width must be block aligned: {width}")
    return (width // Q8_BLOCK) * Q8_BYTES


class BlockWeights:
    """Map a complete block or materialize one block-aligned rank shard."""

    def __init__(self, model_dir: Path, shard: str) -> None:
        manifest = json.loads(
            (model_dir / "weights_manifest.json").read_text(encoding="utf-8")
        )
        self.entries = {str(entry["name"]): entry for entry in manifest["entries"]}
        self._validate_manifest()
        self._fd = os.open(model_dir / "weights.bump", os.O_RDONLY)
        self._mapping = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        self.shard = shard
        self.rank = None if shard == "full" else int(shard)
        if self.rank not in (None, 0, 1):
            raise ValueError(f"unsupported shard: {shard}")
        self.q_heads = Q_HEADS if self.rank is None else Q_HEADS // 2
        self.kv_heads = KV_HEADS if self.rank is None else KV_HEADS // 2
        self.intermediate = INTERMEDIATE if self.rank is None else INTERMEDIATE // 2

        self.attn_norm = self._fp32("attn_norm")
        self.post_attention_norm = self._fp32("post_attention_norm")
        self.q_norm = self._fp32("attn_q_norm")
        self.k_norm = self._fp32("attn_k_norm")
        self.router = self._fp32("moe_router")
        self.shared_router = self._fp32("moe_shared_router")

        if self.rank is None:
            self.q_gate = self._raw("attn_q_gate")
            self.k = self._raw("attn_k")
            self.v = self._raw("attn_v")
            self.attn_output = self._raw("attn_output")
            self.expert_gate = self._raw("moe_expert_gate")
            self.expert_up = self._raw("moe_expert_up")
            self.expert_down = self._raw("moe_expert_down")
            self.shared_gate = self._raw("moe_shared_gate")
            self.shared_up = self._raw("moe_shared_up")
            self.shared_down = self._raw("moe_shared_down")
            self.weight_hash = "mapped-full-layer-3"
        else:
            self._materialize_shard()

    def _entry(self, suffix: str) -> dict[str, Any]:
        return self.entries[f"layer.{LAYER}.{suffix}"]

    def _validate_manifest(self) -> None:
        expected = {
            "attn_norm": ([HIDDEN], "fp32"),
            "post_attention_norm": ([HIDDEN], "fp32"),
            "attn_q_gate": ([Q_HEADS * HEAD_DIM * 2, HIDDEN], "q8_0"),
            "attn_k": ([KV_HEADS * HEAD_DIM, HIDDEN], "q8_0"),
            "attn_v": ([KV_HEADS * HEAD_DIM, HIDDEN], "q8_0"),
            "attn_output": ([HIDDEN, Q_HEADS * HEAD_DIM], "q8_0"),
            "attn_q_norm": ([HEAD_DIM], "fp32"),
            "attn_k_norm": ([HEAD_DIM], "fp32"),
            "moe_router": ([EXPERTS, HIDDEN], "fp32"),
            "moe_expert_gate": ([EXPERTS, INTERMEDIATE, HIDDEN], "q4_k"),
            "moe_expert_up": ([EXPERTS, INTERMEDIATE, HIDDEN], "q4_k"),
            "moe_expert_down": ([EXPERTS, HIDDEN, INTERMEDIATE], "q5_k"),
            "moe_shared_router": ([HIDDEN], "fp32"),
            "moe_shared_gate": ([INTERMEDIATE, HIDDEN], "q8_0"),
            "moe_shared_up": ([INTERMEDIATE, HIDDEN], "q8_0"),
            "moe_shared_down": ([HIDDEN, INTERMEDIATE], "q8_0"),
        }
        for suffix, (shape, dtype) in expected.items():
            entry = self.entries.get(f"layer.{LAYER}.{suffix}")
            if entry is None or entry.get("shape") != shape or entry.get("dtype") != dtype:
                raise ValueError(f"unexpected layer.{LAYER}.{suffix} contract: {entry}")

    def _raw(self, suffix: str) -> np.ndarray:
        entry = self._entry(suffix)
        return np.frombuffer(
            self._mapping,
            dtype=np.uint8,
            count=int(entry["size"]),
            offset=int(entry["file_offset"]),
        )

    def _fp32(self, suffix: str) -> np.ndarray:
        entry = self._entry(suffix)
        return np.frombuffer(
            self._mapping,
            dtype=np.float32,
            count=int(entry["size"]) // np.dtype(np.float32).itemsize,
            offset=int(entry["file_offset"]),
        )

    def _q8_output_rows(self, suffix: str, rows: int, begin: int, count: int) -> np.ndarray:
        source = self._raw(suffix).reshape(rows, _q8_row_bytes(HIDDEN))
        return np.ascontiguousarray(source[begin : begin + count])

    def _q8_input_slice(
        self, suffix: str, rows: int, full_width: int, begin: int, width: int
    ) -> np.ndarray:
        source = self._raw(suffix).reshape(rows, _q8_row_bytes(full_width))
        byte_begin = _q8_row_bytes(begin)
        byte_width = _q8_row_bytes(width)
        return np.ascontiguousarray(source[:, byte_begin : byte_begin + byte_width])

    def _materialize_shard(self) -> None:
        assert self.rank is not None
        qg_rows = Q_HEADS * HEAD_DIM * 2
        qg_shard_rows = qg_rows // 2
        qg_begin = self.rank * qg_shard_rows
        kv_rows = KV_HEADS * HEAD_DIM
        kv_shard_rows = kv_rows // 2
        kv_begin = self.rank * kv_shard_rows
        attn_input_width = Q_HEADS * HEAD_DIM
        attn_shard_width = attn_input_width // 2
        attn_begin = self.rank * attn_shard_width

        self.q_gate = self._q8_output_rows(
            "attn_q_gate", qg_rows, qg_begin, qg_shard_rows
        )
        self.k = self._q8_output_rows("attn_k", kv_rows, kv_begin, kv_shard_rows)
        self.v = self._q8_output_rows("attn_v", kv_rows, kv_begin, kv_shard_rows)
        self.attn_output = self._q8_input_slice(
            "attn_output", HIDDEN, attn_input_width, attn_begin, attn_shard_width
        )

        q4_row_bytes = (HIDDEN // Q4_BLOCK) * Q4_BYTES
        intermediate_begin = self.rank * self.intermediate

        def expert_q4(suffix: str) -> np.ndarray:
            source = self._raw(suffix).reshape(EXPERTS, INTERMEDIATE, q4_row_bytes)
            return np.ascontiguousarray(
                source[:, intermediate_begin : intermediate_begin + self.intermediate]
            )

        self.expert_gate = expert_q4("moe_expert_gate")
        self.expert_up = expert_q4("moe_expert_up")

        q5_full_row_bytes = (INTERMEDIATE // Q5_BLOCK) * Q5_BYTES
        q5_shard_row_bytes = (self.intermediate // Q5_BLOCK) * Q5_BYTES
        q5_byte_begin = (intermediate_begin // Q5_BLOCK) * Q5_BYTES
        expert_down = self._raw("moe_expert_down").reshape(
            EXPERTS, HIDDEN, q5_full_row_bytes
        )
        self.expert_down = np.ascontiguousarray(
            expert_down[:, :, q5_byte_begin : q5_byte_begin + q5_shard_row_bytes]
        )

        self.shared_gate = self._q8_output_rows(
            "moe_shared_gate",
            INTERMEDIATE,
            intermediate_begin,
            self.intermediate,
        )
        self.shared_up = self._q8_output_rows(
            "moe_shared_up",
            INTERMEDIATE,
            intermediate_begin,
            self.intermediate,
        )
        self.shared_down = self._q8_input_slice(
            "moe_shared_down",
            HIDDEN,
            INTERMEDIATE,
            intermediate_begin,
            self.intermediate,
        )

        digest = hashlib.sha256()
        for value in (
            self.q_gate,
            self.k,
            self.v,
            self.attn_output,
            self.expert_gate,
            self.expert_up,
            self.expert_down,
            self.shared_gate,
            self.shared_up,
            self.shared_down,
        ):
            digest.update(memoryview(value).cast("B"))
        self.weight_hash = digest.hexdigest()


def _bind_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    library.rmsnorm_forward_llama_production.argtypes = [
        FPTR, FPTR, FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    library.gemm_nt_q8_0_q8_0_contract.argtypes = [
        FPTR, ctypes.c_void_p, FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.split_q_gate_forward.argtypes = [
        FPTR, FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.ck_layout_token_to_head_f32.argtypes = [
        FPTR, FPTR, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.ck_layout_head_to_token_f32.argtypes = [
        FPTR, FPTR, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.qk_norm_forward_llama_production.argtypes = [
        FPTR, FPTR, FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    library.mrope_qk_text.argtypes = [
        FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,
    ]
    library.kv_cache_store_batch_f16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract_workspace.argtypes = [
        FPTR, ctypes.c_void_p, ctypes.c_void_p, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        FPTR, ctypes.c_size_t,
    ]
    library.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract_workspace.restype = ctypes.c_int
    library.attn_gate_sigmoid_mul_forward.argtypes = [
        FPTR, FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.gemm_blocked_serial.argtypes = [
        FPTR, FPTR, FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.moe_softmax_topk_router_llama_f32_workspace.argtypes = [
        FPTR, IPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.moe_softmax_topk_router_llama_f32_workspace.restype = ctypes.c_int
    library.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes.restype = ctypes.c_size_t
    library.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace.argtypes = [
        FPTR, IPTR, FPTR,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace.restype = ctypes.c_int
    library.moe_swiglu_shared_q8_0_gated_workspace_bytes.argtypes = [
        ctypes.c_int, ctypes.c_int,
    ]
    library.moe_swiglu_shared_q8_0_gated_workspace_bytes.restype = ctypes.c_size_t
    library.moe_swiglu_shared_forward_q8_0_gated_parallel_workspace.argtypes = [
        FPTR, FPTR,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        FPTR, FPTR,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    library.moe_swiglu_shared_forward_q8_0_gated_parallel_workspace.restype = ctypes.c_int
    library.ck_residual_add_token_major.argtypes = [
        FPTR, FPTR, FPTR, ctypes.c_int, ctypes.c_int,
    ]
    return library


class BlockRunner:
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
        self.weights = BlockWeights(model_dir, shard)
        self.input = _make_input(rows, seed)
        q_dim = self.weights.q_heads * HEAD_DIM
        kv_dim = self.weights.kv_heads * HEAD_DIM

        self.attn_normed = np.empty((rows, HIDDEN), dtype=np.float32)
        self.q_gate_packed = np.empty((rows, q_dim * 2), dtype=np.float32)
        self.q_token = np.empty((rows, q_dim), dtype=np.float32)
        self.attn_gate = np.empty((rows, q_dim), dtype=np.float32)
        self.q_head = np.empty((self.weights.q_heads, rows, HEAD_DIM), dtype=np.float32)
        self.k_token = np.empty((rows, kv_dim), dtype=np.float32)
        self.v_token = np.empty((rows, kv_dim), dtype=np.float32)
        self.k_head = np.empty((self.weights.kv_heads, rows, HEAD_DIM), dtype=np.float32)
        self.v_head = np.empty((self.weights.kv_heads, rows, HEAD_DIM), dtype=np.float32)
        self.k_cache = np.empty((self.weights.kv_heads, rows, HEAD_DIM), dtype=np.uint16)
        self.v_cache = np.empty((self.weights.kv_heads, rows, HEAD_DIM), dtype=np.uint16)
        self.attn_head = np.empty_like(self.q_head)
        self.attn_token = np.empty((rows, q_dim), dtype=np.float32)
        self.attn_partial = np.empty((rows, HIDDEN), dtype=np.float32)
        self.after_attn = np.empty((rows, HIDDEN), dtype=np.float32)
        self.mlp_normed = np.empty((rows, HIDDEN), dtype=np.float32)
        self.router_logits = np.empty((rows, EXPERTS), dtype=np.float32)
        self.indices = np.empty((rows, TOP_K), dtype=np.int32)
        self.routing = np.empty((rows, TOP_K), dtype=np.float32)
        self.routed = np.empty((rows, HIDDEN), dtype=np.float32)
        self.mlp_partial = np.empty((rows, HIDDEN), dtype=np.float32)
        self.output = np.empty((rows, HIDDEN), dtype=np.float32)

        self.attn_workspace = np.empty(32768, dtype=np.uint8)
        self.router_workspace = np.empty(1024, dtype=np.uint8)
        expert_bytes = int(
            self.library.moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes(
                rows, HIDDEN, self.weights.intermediate, EXPERTS, TOP_K
            )
        )
        if expert_bytes <= 0:
            raise RuntimeError("routed expert provider rejected the block shape")
        self.expert_workspace = np.empty(expert_bytes, dtype=np.uint8)
        shared_stride = int(
            self.library.moe_swiglu_shared_q8_0_gated_workspace_bytes(
                HIDDEN, self.weights.intermediate
            )
        )
        if shared_stride <= 0:
            raise RuntimeError("shared expert provider rejected the block shape")
        self.shared_workspace = np.empty(shared_stride * 64, dtype=np.uint8)
        self.setup_ms = (time.perf_counter_ns() - started) / 1.0e6

    def run_attention(self) -> float:
        started = time.perf_counter_ns()
        w = self.weights
        q_dim = w.q_heads * HEAD_DIM
        kv_dim = w.kv_heads * HEAD_DIM
        self.library.rmsnorm_forward_llama_production(
            _fptr(self.input), _fptr(w.attn_norm), _fptr(self.attn_normed), None,
            self.rows, HIDDEN, HIDDEN, EPS,
        )
        self.library.gemm_nt_q8_0_q8_0_contract(
            _fptr(self.attn_normed), _vptr(w.q_gate), None,
            _fptr(self.q_gate_packed), self.rows, q_dim * 2, HIDDEN,
        )
        self.library.split_q_gate_forward(
            _fptr(self.q_gate_packed), _fptr(self.q_token), _fptr(self.attn_gate),
            self.rows, q_dim, q_dim, HEAD_DIM,
        )
        self.library.ck_layout_token_to_head_f32(
            _fptr(self.q_token), _fptr(self.q_head),
            self.rows, w.q_heads, HEAD_DIM,
        )
        self.library.gemm_nt_q8_0_q8_0_contract(
            _fptr(self.attn_normed), _vptr(w.k), None, _fptr(self.k_token),
            self.rows, kv_dim, HIDDEN,
        )
        self.library.ck_layout_token_to_head_f32(
            _fptr(self.k_token), _fptr(self.k_head),
            self.rows, w.kv_heads, HEAD_DIM,
        )
        self.library.gemm_nt_q8_0_q8_0_contract(
            _fptr(self.attn_normed), _vptr(w.v), None, _fptr(self.v_token),
            self.rows, kv_dim, HIDDEN,
        )
        self.library.ck_layout_token_to_head_f32(
            _fptr(self.v_token), _fptr(self.v_head),
            self.rows, w.kv_heads, HEAD_DIM,
        )
        self.library.qk_norm_forward_llama_production(
            _fptr(self.q_head), _fptr(self.k_head),
            _fptr(w.q_norm), _fptr(w.k_norm),
            w.q_heads, w.kv_heads, self.rows, HEAD_DIM, EPS,
        )
        self.library.mrope_qk_text(
            _fptr(self.q_head), _fptr(self.k_head),
            w.q_heads, w.kv_heads, self.rows, HEAD_DIM, HEAD_DIM,
            0, 64, 11, 11, 10, 0, 262144,
            ctypes.c_float(10000000.0), ctypes.c_float(1.0),
            ctypes.c_float(0.0), ctypes.c_float(1.0),
            ctypes.c_float(0.0), ctypes.c_float(0.0),
        )
        self.library.kv_cache_store_batch_f16(
            _vptr(self.k_cache), _vptr(self.v_cache),
            _fptr(self.k_head), _fptr(self.v_head),
            0, self.rows, w.kv_heads, HEAD_DIM, self.rows,
        )
        rc = self.library.attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract_workspace(
            _fptr(self.q_head), _vptr(self.k_cache), _vptr(self.v_cache),
            _fptr(self.attn_head),
            w.q_heads, w.kv_heads, self.rows, 0, self.rows,
            HEAD_DIM, HEAD_DIM, ATTENTION_REDUCTION,
            _fptr(self.attn_workspace), self.attn_workspace.nbytes,
        )
        if rc != 0:
            raise RuntimeError(f"attention provider failed: rc={rc}")
        self.library.ck_layout_head_to_token_f32(
            _fptr(self.attn_head), _fptr(self.attn_token),
            w.q_heads, self.rows, HEAD_DIM,
        )
        self.library.attn_gate_sigmoid_mul_forward(
            _fptr(self.attn_token), _fptr(self.attn_gate), _fptr(self.attn_token),
            self.rows, w.q_heads, HEAD_DIM,
        )
        self.library.gemm_nt_q8_0_q8_0_contract(
            _fptr(self.attn_token), _vptr(w.attn_output), None,
            _fptr(self.attn_partial), self.rows, HIDDEN, q_dim,
        )
        return (time.perf_counter_ns() - started) / 1.0e6

    def set_after_attention(self, combined: np.ndarray) -> float:
        started = time.perf_counter_ns()
        self.library.ck_residual_add_token_major(
            _fptr(combined), _fptr(self.input), _fptr(self.after_attn),
            self.rows, HIDDEN,
        )
        return (time.perf_counter_ns() - started) / 1.0e6

    def prepare_mlp(self, select_routes: bool) -> float:
        started = time.perf_counter_ns()
        w = self.weights
        self.library.rmsnorm_forward_llama_production(
            _fptr(self.after_attn), _fptr(w.post_attention_norm),
            _fptr(self.mlp_normed), None,
            self.rows, HIDDEN, HIDDEN, EPS,
        )
        if select_routes:
            self.library.gemm_blocked_serial(
                _fptr(self.mlp_normed), _fptr(w.router), None,
                _fptr(self.router_logits), self.rows, EXPERTS, HIDDEN,
            )
            rc = self.library.moe_softmax_topk_router_llama_f32_workspace(
                _fptr(self.router_logits), _iptr(self.indices), _fptr(self.routing),
                self.rows, EXPERTS, TOP_K, ctypes.c_float(1.0),
                _vptr(self.router_workspace), self.router_workspace.nbytes,
            )
            if rc != 0:
                raise RuntimeError(f"router provider failed: rc={rc}")
        return (time.perf_counter_ns() - started) / 1.0e6

    def run_mlp_experts(self) -> float:
        started = time.perf_counter_ns()
        w = self.weights
        rc = self.library.moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace(
            _fptr(self.mlp_normed), _iptr(self.indices), _fptr(self.routing),
            _vptr(w.expert_gate), _vptr(w.expert_up), _vptr(w.expert_down),
            _fptr(self.routed),
            self.rows, HIDDEN, w.intermediate, EXPERTS, TOP_K,
            _vptr(self.expert_workspace), self.expert_workspace.nbytes,
        )
        if rc != 0:
            raise RuntimeError(f"routed expert provider failed: rc={rc}")
        rc = self.library.moe_swiglu_shared_forward_q8_0_gated_parallel_workspace(
            _fptr(self.mlp_normed), _fptr(self.routed),
            _vptr(w.shared_gate), _vptr(w.shared_up), _vptr(w.shared_down),
            _fptr(w.shared_router), _fptr(self.mlp_partial),
            self.rows, HIDDEN, w.intermediate,
            _vptr(self.shared_workspace), self.shared_workspace.nbytes,
        )
        if rc != 0:
            raise RuntimeError(f"shared expert provider failed: rc={rc}")
        return (time.perf_counter_ns() - started) / 1.0e6

    def run_mlp(self) -> float:
        return self.prepare_mlp(select_routes=True) + self.run_mlp_experts()

    def finish_layer(self, combined: np.ndarray) -> float:
        started = time.perf_counter_ns()
        self.library.ck_residual_add_token_major(
            _fptr(combined), _fptr(self.after_attn), _fptr(self.output),
            self.rows, HIDDEN,
        )
        return (time.perf_counter_ns() - started) / 1.0e6

    def run_full(self) -> dict[str, float]:
        attention_ms = self.run_attention()
        attention_residual_ms = self.set_after_attention(self.attn_partial)
        mlp_ms = self.run_mlp()
        mlp_residual_ms = self.finish_layer(self.mlp_partial)
        return {
            "attention_ms": attention_ms,
            "attention_residual_ms": attention_residual_ms,
            "mlp_ms": mlp_ms,
            "mlp_residual_ms": mlp_residual_ms,
            "total_ms": attention_ms + attention_residual_ms + mlp_ms + mlp_residual_ms,
        }


def _run_verify(args: argparse.Namespace) -> int:
    canonical = BlockRunner(args.model_dir, args.library, "full", args.rows, args.seed)
    stage_reference = BlockRunner(args.model_dir, args.library, "full", args.rows, args.seed)
    rank0 = BlockRunner(args.model_dir, args.library, "0", args.rows, args.seed)
    rank1 = BlockRunner(args.model_dir, args.library, "1", args.rows, args.seed)

    canonical_timing = canonical.run_full()
    rank0_attention_ms = rank0.run_attention()
    rank1_attention_ms = rank1.run_attention()
    attention_combined = rank0.attn_partial + rank1.attn_partial
    attention_comparison = _comparison(attention_combined, canonical.attn_partial)

    rank0.set_after_attention(attention_combined)
    rank1.set_after_attention(attention_combined)
    stage_reference.set_after_attention(attention_combined)
    stage_reference_mlp_ms = stage_reference.run_mlp()
    rank0_prepare_ms = rank0.prepare_mlp(select_routes=False)
    rank1_prepare_ms = rank1.prepare_mlp(select_routes=False)
    rank0.indices[:] = stage_reference.indices
    rank0.routing[:] = stage_reference.routing
    rank1.indices[:] = stage_reference.indices
    rank1.routing[:] = stage_reference.routing
    rank0_mlp_ms = rank0_prepare_ms + rank0.run_mlp_experts()
    rank1_mlp_ms = rank1_prepare_ms + rank1.run_mlp_experts()
    mlp_combined = rank0.mlp_partial + rank1.mlp_partial
    mlp_comparison = _comparison(mlp_combined, stage_reference.mlp_partial)

    rank0.finish_layer(mlp_combined)
    rank1.finish_layer(mlp_combined)
    layer_comparison = _comparison(rank0.output, canonical.output)
    report = {
        "mode": "verify",
        "host": socket.gethostname(),
        "layer": LAYER,
        "rows": args.rows,
        "hidden": HIDDEN,
        "attention_shard": {"q_heads": 8, "kv_heads": 1},
        "moe_shard": {"intermediate": 256, "experts": EXPERTS},
        "collectives": 2,
        "payload_bytes_per_collective": args.rows * HIDDEN * 4,
        "canonical_timing": canonical_timing,
        "rank0_attention_ms": rank0_attention_ms,
        "rank1_attention_ms": rank1_attention_ms,
        "rank0_mlp_ms": rank0_mlp_ms,
        "rank1_mlp_ms": rank1_mlp_ms,
        "stage_reference_mlp_ms": stage_reference_mlp_ms,
        "attention_comparison": attention_comparison,
        "mlp_comparison": mlp_comparison,
        "layer_comparison": layer_comparison,
        "router_index_mismatches_before_sync": int(
            np.count_nonzero(stage_reference.indices != canonical.indices)
        ),
        "router_sync_bytes": args.rows * TOP_K * 8,
        "input_hash": _hash(canonical.input),
        "canonical_output_hash": _hash(canonical.output),
        "distributed_output_hash": _hash(rank0.output),
        "rank0_weight_hash": rank0.weights.weight_hash,
        "rank1_weight_hash": rank1.weights.weight_hash,
        "setup_ms": {
            "canonical": canonical.setup_ms,
            "stage_reference": stage_reference.setup_ms,
            "rank0": rank0.setup_ms,
            "rank1": rank1.setup_ms,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


def _send_array(connection: socket.socket, array: np.ndarray) -> float:
    started = time.perf_counter_ns()
    connection.sendall(memoryview(array).cast("B"))
    return (time.perf_counter_ns() - started) / 1.0e6


def _recv_array(connection: socket.socket, array: np.ndarray) -> float:
    started = time.perf_counter_ns()
    TRANSPORT._recv_exact(connection, memoryview(array).cast("B"))
    return (time.perf_counter_ns() - started) / 1.0e6


def _run_worker(args: argparse.Namespace) -> int:
    runner = BlockRunner(args.model_dir, args.library, args.shard, args.rows, args.seed)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.listen, args.port))
    listener.listen(1)
    print(
        json.dumps({"status": "ready", "host": socket.gethostname(), "port": args.port}),
        flush=True,
    )
    connection, _ = listener.accept()
    with connection:
        command = bytearray(1)
        TRANSPORT._recv_exact(connection, memoryview(command))
        if command != b"B":
            raise ValueError(f"invalid command: {bytes(command)!r}")

        attention_ms = runner.run_attention()
        TRANSPORT._send_json(
            connection,
            {
                "stage": "attention",
                "compute_ms": attention_ms,
                "input_hash": _hash(runner.input),
                "partial_hash": _hash(runner.attn_partial),
                "weight_hash": runner.weights.weight_hash,
            },
        )
        attention_send_ms = _send_array(connection, runner.attn_partial)
        attention_combined = np.empty_like(runner.attn_partial)
        attention_receive_ms = _recv_array(connection, attention_combined)
        runner.set_after_attention(attention_combined)
        mlp_prepare_ms = runner.prepare_mlp(select_routes=False)
        route_indices_receive_ms = _recv_array(connection, runner.indices)
        route_weights_receive_ms = _recv_array(connection, runner.routing)
        mlp_expert_ms = runner.run_mlp_experts()
        mlp_ms = mlp_prepare_ms + mlp_expert_ms
        TRANSPORT._send_json(
            connection,
            {
                "stage": "mlp",
                "compute_ms": mlp_ms,
                "prepare_ms": mlp_prepare_ms,
                "expert_ms": mlp_expert_ms,
                "router_hash": _hash(runner.indices),
                "partial_hash": _hash(runner.mlp_partial),
            },
        )
        mlp_send_ms = _send_array(connection, runner.mlp_partial)
        mlp_combined = np.empty_like(runner.mlp_partial)
        mlp_receive_ms = _recv_array(connection, mlp_combined)
        runner.finish_layer(mlp_combined)
        TRANSPORT._send_json(
            connection,
            {
                "stage": "complete",
                "output_hash": _hash(runner.output),
                "attention_send_ms": attention_send_ms,
                "attention_receive_ms": attention_receive_ms,
                "mlp_send_ms": mlp_send_ms,
                "mlp_receive_ms": mlp_receive_ms,
                "route_indices_receive_ms": route_indices_receive_ms,
                "route_weights_receive_ms": route_weights_receive_ms,
            },
        )
    listener.close()
    return 0


def _run_coordinator(args: argparse.Namespace) -> int:
    local = BlockRunner(args.model_dir, args.library, args.shard, args.rows, args.seed)
    canonical = BlockRunner(args.model_dir, args.library, "full", args.rows, args.seed)
    canonical_timing = canonical.run_full()
    payload_bytes = args.rows * HIDDEN * 4

    with socket.create_connection((args.peer, args.port), timeout=120.0) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        wall_started = time.perf_counter_ns()
        connection.sendall(b"B")

        local_attention_ms = local.run_attention()
        remote_attention = TRANSPORT._recv_json(connection)
        remote_attention_partial = np.empty_like(local.attn_partial)
        attention_receive_ms = _recv_array(connection, remote_attention_partial)
        reduce_started = time.perf_counter_ns()
        attention_combined = local.attn_partial + remote_attention_partial
        attention_reduce_ms = (time.perf_counter_ns() - reduce_started) / 1.0e6
        attention_broadcast_ms = _send_array(connection, attention_combined)
        local.set_after_attention(attention_combined)

        local_mlp_prepare_ms = local.prepare_mlp(select_routes=True)
        route_indices_broadcast_ms = _send_array(connection, local.indices)
        route_weights_broadcast_ms = _send_array(connection, local.routing)
        local_mlp_expert_ms = local.run_mlp_experts()
        local_mlp_ms = local_mlp_prepare_ms + local_mlp_expert_ms
        remote_mlp = TRANSPORT._recv_json(connection)
        remote_mlp_partial = np.empty_like(local.mlp_partial)
        mlp_receive_ms = _recv_array(connection, remote_mlp_partial)
        reduce_started = time.perf_counter_ns()
        mlp_combined = local.mlp_partial + remote_mlp_partial
        mlp_reduce_ms = (time.perf_counter_ns() - reduce_started) / 1.0e6
        mlp_broadcast_ms = _send_array(connection, mlp_combined)
        local.finish_layer(mlp_combined)
        remote_complete = TRANSPORT._recv_json(connection)
        distributed_wall_ms = (time.perf_counter_ns() - wall_started) / 1.0e6

    report = {
        "mode": "coordinator",
        "host": socket.gethostname(),
        "peer": args.peer,
        "layer": LAYER,
        "rows": args.rows,
        "hidden": HIDDEN,
        "collectives": 2,
        "payload_bytes_per_collective": payload_bytes,
        "sequential_tcp_bytes": payload_bytes * 4,
        "router_sync_bytes": args.rows * TOP_K * 8,
        "local_attention_ms": local_attention_ms,
        "remote_attention_ms": remote_attention["compute_ms"],
        "attention_compute_critical_ms": max(
            local_attention_ms, remote_attention["compute_ms"]
        ),
        "attention_receive_ms": attention_receive_ms,
        "attention_reduce_ms": attention_reduce_ms,
        "attention_broadcast_ms": attention_broadcast_ms,
        "local_mlp_ms": local_mlp_ms,
        "local_mlp_prepare_ms": local_mlp_prepare_ms,
        "local_mlp_expert_ms": local_mlp_expert_ms,
        "remote_mlp_ms": remote_mlp["compute_ms"],
        "remote_mlp_prepare_ms": remote_mlp["prepare_ms"],
        "remote_mlp_expert_ms": remote_mlp["expert_ms"],
        "mlp_compute_critical_ms": max(local_mlp_ms, remote_mlp["compute_ms"]),
        "mlp_receive_ms": mlp_receive_ms,
        "mlp_reduce_ms": mlp_reduce_ms,
        "mlp_broadcast_ms": mlp_broadcast_ms,
        "route_indices_broadcast_ms": route_indices_broadcast_ms,
        "route_weights_broadcast_ms": route_weights_broadcast_ms,
        "distributed_wall_ms": distributed_wall_ms,
        "canonical_timing": canonical_timing,
        "attention_comparison": _comparison(attention_combined, canonical.attn_partial),
        "layer_comparison": _comparison(local.output, canonical.output),
        "input_hash_match": _hash(local.input) == remote_attention["input_hash"],
        "router_hash_match": _hash(local.indices) == remote_mlp["router_hash"],
        "coordinator_output_hash": _hash(local.output),
        "worker_output_hash": remote_complete["output_hash"],
        "worker_transport_ms": {
            "attention_send": remote_complete["attention_send_ms"],
            "attention_receive": remote_complete["attention_receive_ms"],
            "mlp_send": remote_complete["mlp_send_ms"],
            "mlp_receive": remote_complete["mlp_receive_ms"],
            "route_indices_receive": remote_complete["route_indices_receive_ms"],
            "route_weights_receive": remote_complete["route_weights_receive_ms"],
        },
        "local_weight_hash": local.weights.weight_hash,
        "remote_weight_hash": remote_attention["weight_hash"],
    }
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "worker", "coordinator"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--shard", choices=("0", "1"), default="0")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3503503)
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--peer")
    parser.add_argument("--port", type=int, default=29645)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.library = (
        args.library.expanduser().resolve()
        if args.library
        else args.model_dir / "libckernel_engine.so"
    )
    if args.rows <= 0:
        parser.error("rows must be positive")
    if args.mode == "coordinator" and not args.peer:
        parser.error("coordinator requires --peer")
    return {
        "verify": _run_verify,
        "worker": _run_worker,
        "coordinator": _run_coordinator,
    }[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
