#!/usr/bin/env python3
"""Determinism and persistent-threadpool parity for mixed Q4_K/Q8_0 MoE."""

from __future__ import annotations

import ctypes
import struct
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build/libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)
QK_K = 256
Q4_K_BYTES = 144
Q8_0_BYTES = 34


def _f32(array: np.ndarray) -> F32P:
    return array.ctypes.data_as(F32P)


def _i32(array: np.ndarray) -> I32P:
    return array.ctypes.data_as(I32P)


def _q4_blocks(count: int, seed: int) -> ctypes.Array:
    rng = np.random.default_rng(seed)
    raw = bytearray(count * Q4_K_BYTES)
    for block in range(count):
        offset = block * Q4_K_BYTES
        struct.pack_into("<H", raw, offset, 0x211F + block % 7)
        struct.pack_into("<H", raw, offset + 2, 0x1800 + block % 5)
        raw[offset + 4 : offset + Q4_K_BYTES] = rng.integers(
            0, 256, size=Q4_K_BYTES - 4, dtype=np.uint8
        ).tobytes()
    return ctypes.create_string_buffer(bytes(raw), len(raw))


def _q8_blocks(count: int, seed: int) -> ctypes.Array:
    rng = np.random.default_rng(seed)
    raw = bytearray(count * Q8_0_BYTES)
    for block in range(count):
        offset = block * Q8_0_BYTES
        struct.pack_into("<H", raw, offset, 0x211F + block % 7)
        raw[offset + 2 : offset + Q8_0_BYTES] = rng.integers(
            -127, 128, size=32, dtype=np.int8
        ).tobytes()
    return ctypes.create_string_buffer(bytes(raw), len(raw))


class MixedQ4KQ80MoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        LIB.ck_set_num_threads.argtypes = [ctypes.c_int]
        LIB.ck_set_num_threads(4)
        LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes.restype = ctypes.c_size_t
        LIB.moe_swiglu_shared_q4k_q8_0_gated_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        LIB.moe_swiglu_shared_q4k_q8_0_gated_workspace_bytes.restype = ctypes.c_size_t
        cls.expert_args = [
            F32P, I32P, F32P, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
        ]
        cls.shared_args = [
            F32P, F32P, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        for name in (
            "moe_swiglu_expert_forward_q4k_q8_0_workspace",
            "moe_swiglu_expert_forward_q4k_q8_0_parallel_workspace",
        ):
            fn = getattr(LIB, name)
            fn.argtypes = cls.expert_args
            fn.restype = ctypes.c_int
        for name in (
            "moe_swiglu_shared_forward_q4k_q8_0_gated_workspace",
            "moe_swiglu_shared_forward_q4k_q8_0_gated_parallel_workspace",
        ):
            fn = getattr(LIB, name)
            fn.argtypes = cls.shared_args
            fn.restype = ctypes.c_int
        LIB.quantize_row_q8_k.argtypes = [F32P, ctypes.c_void_p, ctypes.c_int]
        LIB.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, F32P, F32P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        LIB.swiglu_forward_ggml_split.argtypes = [
            F32P, F32P, F32P, ctypes.c_int, ctypes.c_int,
        ]
        LIB.quantize_row_q8_0.argtypes = [F32P, ctypes.c_void_p, ctypes.c_int]
        LIB.gemv_q8_0_q8_0.argtypes = [
            F32P, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
        ]

    def setUp(self) -> None:
        self.rows = 7
        self.hidden = QK_K
        self.intermediate = 640
        self.experts = 4
        self.top_k = 3
        rng = np.random.default_rng(0x4380)
        self.x = np.ascontiguousarray(rng.normal(0, 0.2, (self.rows, self.hidden)), dtype=np.float32)
        self.routed = np.ascontiguousarray(rng.normal(0, 0.1, (self.rows, self.hidden)), dtype=np.float32)
        self.indices = np.ascontiguousarray(rng.integers(0, self.experts, (self.rows, self.top_k), dtype=np.int32))
        self.routing = np.ascontiguousarray(rng.uniform(0.1, 0.9, (self.rows, self.top_k)), dtype=np.float32)
        self.router = np.ascontiguousarray(rng.normal(0, 0.1, (1, self.hidden)), dtype=np.float32)
        self.expert_gate = _q4_blocks(self.experts * self.intermediate, 1)
        self.expert_up = _q4_blocks(self.experts * self.intermediate, 2)
        self.expert_down = _q8_blocks(self.experts * self.hidden * (self.intermediate // 32), 3)
        self.shared_gate = _q4_blocks(self.intermediate, 4)
        self.shared_up = _q4_blocks(self.intermediate, 5)
        self.shared_down = _q8_blocks(self.hidden * (self.intermediate // 32), 6)
        self.stride = int(LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes(self.hidden, self.intermediate))
        self.shared_stride = int(
            LIB.moe_swiglu_shared_q4k_q8_0_gated_workspace_bytes(
                self.hidden, self.intermediate,
            )
        )

    def test_routed_parallel_matches_serial_bit_exact(self) -> None:
        serial = LIB.moe_swiglu_expert_forward_q4k_q8_0_workspace
        parallel = LIB.moe_swiglu_expert_forward_q4k_q8_0_parallel_workspace
        for rows in (1, self.rows):
            expected = np.empty((rows, self.hidden), dtype=np.float32)
            actual = np.empty_like(expected)
            common = (_f32(self.x[:rows]), _i32(self.indices[:rows]), _f32(self.routing[:rows]), self.expert_gate, self.expert_up, self.expert_down)
            serial_workspace = ctypes.create_string_buffer(self.stride)
            parallel_workspace = ctypes.create_string_buffer(self.stride * 64)
            self.assertEqual(serial(*common, _f32(expected), rows, self.hidden, self.intermediate, self.experts, self.top_k, serial_workspace, self.stride), 0)
            self.assertEqual(parallel(*common, _f32(actual), rows, self.hidden, self.intermediate, self.experts, self.top_k, parallel_workspace, len(parallel_workspace)), 0)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_routed_parallel_scales_down_to_available_workspace(self) -> None:
        serial = LIB.moe_swiglu_expert_forward_q4k_q8_0_workspace
        parallel = LIB.moe_swiglu_expert_forward_q4k_q8_0_parallel_workspace
        expected = np.empty((self.rows, self.hidden), dtype=np.float32)
        actual = np.empty_like(expected)
        common = (
            _f32(self.x), _i32(self.indices), _f32(self.routing),
            self.expert_gate, self.expert_up, self.expert_down,
        )
        serial_workspace = ctypes.create_string_buffer(self.stride)
        constrained_workspace = ctypes.create_string_buffer(self.stride)
        self.assertEqual(
            serial(
                *common, _f32(expected), self.rows, self.hidden,
                self.intermediate, self.experts, self.top_k,
                serial_workspace, self.stride,
            ),
            0,
        )
        self.assertEqual(
            parallel(
                *common, _f32(actual), self.rows, self.hidden,
                self.intermediate, self.experts, self.top_k,
                constrained_workspace, self.stride,
            ),
            0,
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_gated_shared_parallel_matches_serial_bit_exact(self) -> None:
        serial = LIB.moe_swiglu_shared_forward_q4k_q8_0_gated_workspace
        parallel = LIB.moe_swiglu_shared_forward_q4k_q8_0_gated_parallel_workspace
        expected = np.empty_like(self.x)
        actual = np.empty_like(self.x)
        common = (_f32(self.x), _f32(self.routed), self.shared_gate, self.shared_up, self.shared_down, _f32(self.router))
        serial_workspace = ctypes.create_string_buffer(self.shared_stride)
        parallel_workspace = ctypes.create_string_buffer(self.shared_stride)
        self.assertEqual(serial(*common, _f32(expected), self.rows, self.hidden, self.intermediate, serial_workspace, self.shared_stride), 0)
        self.assertEqual(parallel(*common, _f32(actual), self.rows, self.hidden, self.intermediate, parallel_workspace, len(parallel_workspace)), 0)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_gated_shared_uses_four_row_prefill_reduction_groups(self) -> None:
        rows = self.rows
        q8_k_row_bytes = (self.hidden // QK_K) * 292
        q8_0_row_bytes = (self.intermediate // 32) * Q8_0_BYTES
        hidden_q8 = np.empty((rows, q8_k_row_bytes), dtype=np.uint8)
        gate = np.empty((rows, self.intermediate), dtype=np.float32)
        up = np.empty_like(gate)
        activation = np.empty_like(gate)
        activation_q8 = np.empty((rows, q8_0_row_bytes), dtype=np.uint8)
        shared = np.empty((rows, self.hidden), dtype=np.float32)

        for row in range(rows):
            LIB.quantize_row_q8_k(
                _f32(self.x[row]), ctypes.c_void_p(hidden_q8[row].ctypes.data),
                self.hidden,
            )
        for row0 in range(0, rows, 4):
            batch_rows = min(4, rows - row0)
            LIB.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
                ctypes.c_void_p(hidden_q8[row0:].ctypes.data),
                self.shared_gate, None, _f32(gate[row0:]),
                batch_rows, self.intermediate, self.hidden,
            )
            LIB.gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
                ctypes.c_void_p(hidden_q8[row0:].ctypes.data),
                self.shared_up, None, _f32(up[row0:]),
                batch_rows, self.intermediate, self.hidden,
            )
        LIB.swiglu_forward_ggml_split(
            _f32(gate), _f32(up), _f32(activation), rows, self.intermediate,
        )
        for row in range(rows):
            LIB.quantize_row_q8_0(
                _f32(activation[row]),
                ctypes.c_void_p(activation_q8[row].ctypes.data),
                self.intermediate,
            )
            LIB.gemv_q8_0_q8_0(
                _f32(shared[row]), self.shared_down,
                ctypes.c_void_p(activation_q8[row].ctypes.data),
                self.hidden, self.intermediate,
            )

        zeros = np.zeros_like(self.x)
        zero_router = np.zeros_like(self.router)
        actual = np.empty_like(self.x)
        workspace = ctypes.create_string_buffer(self.shared_stride)
        fn = LIB.moe_swiglu_shared_forward_q4k_q8_0_gated_workspace
        self.assertEqual(
            fn(
                _f32(self.x), _f32(zeros), self.shared_gate, self.shared_up,
                self.shared_down, _f32(zero_router), _f32(actual), rows,
                self.hidden, self.intermediate, workspace, self.shared_stride,
            ),
            0,
        )
        expected = np.ascontiguousarray(shared * np.float32(0.5))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
