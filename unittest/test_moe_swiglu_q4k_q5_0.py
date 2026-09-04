#!/usr/bin/env python3
"""Persistent-threadpool parity for mixed Q4_K/Q5_0 MoE providers."""

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
Q5_0_BYTES = 22


def _f32(array: np.ndarray) -> F32P:
    return array.ctypes.data_as(F32P)


def _i32(array: np.ndarray) -> I32P:
    return array.ctypes.data_as(I32P)


def _blocks(count: int, size: int, seed: int) -> ctypes.Array:
    rng = np.random.default_rng(seed)
    raw = bytearray(count * size)
    for block in range(count):
        offset = block * size
        struct.pack_into("<H", raw, offset, 0x211F + block % 7)
        raw[offset + 2 : offset + size] = rng.integers(
            0, 256, size=size - 2, dtype=np.uint8
        ).tobytes()
    return ctypes.create_string_buffer(bytes(raw), len(raw))


class MixedQ4KQ50MoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        LIB.ck_set_num_threads.argtypes = [ctypes.c_int]
        LIB.ck_set_num_threads(4)
        LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
        ]
        LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes.restype = ctypes.c_size_t
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
            "moe_swiglu_expert_forward_q4k_q5_0_workspace",
            "moe_swiglu_expert_forward_q4k_q5_0_parallel_workspace",
        ):
            function = getattr(LIB, name)
            function.argtypes = cls.expert_args
            function.restype = ctypes.c_int
        for name in (
            "moe_swiglu_shared_forward_q4k_q5_0_gated_workspace",
            "moe_swiglu_shared_forward_q4k_q5_0_gated_parallel_workspace",
        ):
            function = getattr(LIB, name)
            function.argtypes = cls.shared_args
            function.restype = ctypes.c_int

    def setUp(self) -> None:
        self.rows = 7
        self.hidden = QK_K
        self.intermediate = 640
        self.experts = 4
        self.top_k = 3
        rng = np.random.default_rng(0x4350)
        self.x = np.ascontiguousarray(
            rng.normal(0, 0.2, (self.rows, self.hidden)), dtype=np.float32
        )
        self.routed = np.ascontiguousarray(
            rng.normal(0, 0.1, (self.rows, self.hidden)), dtype=np.float32
        )
        self.indices = np.ascontiguousarray(
            rng.integers(
                0, self.experts, (self.rows, self.top_k), dtype=np.int32
            )
        )
        self.routing = np.ascontiguousarray(
            rng.uniform(0.1, 0.9, (self.rows, self.top_k)), dtype=np.float32
        )
        self.router = np.ascontiguousarray(
            rng.normal(0, 0.1, (1, self.hidden)), dtype=np.float32
        )
        self.expert_gate = _blocks(self.experts * self.intermediate, Q4_K_BYTES, 1)
        self.expert_up = _blocks(self.experts * self.intermediate, Q4_K_BYTES, 2)
        self.expert_down = _blocks(
            self.experts * self.hidden * (self.intermediate // 32), Q5_0_BYTES, 3
        )
        self.shared_gate = _blocks(self.intermediate, Q4_K_BYTES, 4)
        self.shared_up = _blocks(self.intermediate, Q4_K_BYTES, 5)
        self.shared_down = _blocks(
            self.hidden * (self.intermediate // 32), Q5_0_BYTES, 6
        )
        self.stride = int(
            LIB.moe_swiglu_expert_q4k_q8_0_workspace_bytes(
                self.hidden, self.intermediate
            )
        )

    def test_routed_parallel_matches_serial_bit_exact(self) -> None:
        serial = LIB.moe_swiglu_expert_forward_q4k_q5_0_workspace
        parallel = LIB.moe_swiglu_expert_forward_q4k_q5_0_parallel_workspace
        for rows in (1, self.rows):
            expected = np.empty((rows, self.hidden), dtype=np.float32)
            actual = np.empty_like(expected)
            common = (
                _f32(self.x[:rows]), _i32(self.indices[:rows]),
                _f32(self.routing[:rows]), self.expert_gate, self.expert_up,
                self.expert_down,
            )
            serial_workspace = ctypes.create_string_buffer(self.stride)
            parallel_workspace = ctypes.create_string_buffer(self.stride * 64)
            self.assertEqual(
                serial(
                    *common, _f32(expected), rows, self.hidden,
                    self.intermediate, self.experts, self.top_k,
                    serial_workspace, self.stride,
                ),
                0,
            )
            self.assertEqual(
                parallel(
                    *common, _f32(actual), rows, self.hidden,
                    self.intermediate, self.experts, self.top_k,
                    parallel_workspace, len(parallel_workspace),
                ),
                0,
            )
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_routed_parallel_scales_down_to_available_workspace(self) -> None:
        serial = LIB.moe_swiglu_expert_forward_q4k_q5_0_workspace
        parallel = LIB.moe_swiglu_expert_forward_q4k_q5_0_parallel_workspace
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
        serial = LIB.moe_swiglu_shared_forward_q4k_q5_0_gated_workspace
        parallel = LIB.moe_swiglu_shared_forward_q4k_q5_0_gated_parallel_workspace
        expected = np.empty_like(self.x)
        actual = np.empty_like(self.x)
        common = (
            _f32(self.x), _f32(self.routed), self.shared_gate, self.shared_up,
            self.shared_down, _f32(self.router),
        )
        serial_workspace = ctypes.create_string_buffer(self.stride)
        parallel_workspace = ctypes.create_string_buffer(self.stride * 64)
        self.assertEqual(
            serial(
                *common, _f32(expected), self.rows, self.hidden,
                self.intermediate, serial_workspace, self.stride,
            ),
            0,
        )
        self.assertEqual(
            parallel(
                *common, _f32(actual), self.rows, self.hidden,
                self.intermediate, parallel_workspace, len(parallel_workspace),
            ),
            0,
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
