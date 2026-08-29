#!/usr/bin/env python3
"""Parity contracts for Laguna's parallel compact MoE providers."""

from __future__ import annotations

import ctypes
import struct
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
FPTR = ctypes.POINTER(ctypes.c_float)
IPTR = ctypes.POINTER(ctypes.c_int)
QK_K = 256
Q4_K_BYTES = 144
Q6_K_BYTES = 210


def _fptr(array: np.ndarray) -> FPTR:
    return array.ctypes.data_as(FPTR)


def _iptr(array: np.ndarray) -> IPTR:
    return array.ctypes.data_as(IPTR)


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


def _q6_blocks(count: int, seed: int) -> ctypes.Array:
    rng = np.random.default_rng(seed)
    raw = bytearray(count * Q6_K_BYTES)
    for block in range(count):
        offset = block * Q6_K_BYTES
        raw[offset : offset + 192] = rng.integers(
            0, 256, size=192, dtype=np.uint8
        ).tobytes()
        raw[offset + 192 : offset + 208] = rng.integers(
            -16, 17, size=16, dtype=np.int8
        ).tobytes()
        struct.pack_into("<H", raw, offset + 208, 0x211F + block % 7)
    return ctypes.create_string_buffer(bytes(raw), len(raw))


class LagunaParallelMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        LIB.ck_set_num_threads.argtypes = [ctypes.c_int]
        LIB.ck_set_num_threads(4)
        LIB.moe_swiglu_expert_q4k_q5k_workspace_bytes.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
        ]
        LIB.moe_swiglu_expert_q4k_q5k_workspace_bytes.restype = ctypes.c_size_t

        cls.expert_args = [
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
        cls.shared_args = [
            FPTR,
            FPTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            FPTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        for suffix in ("q4k_q4k", "q4k_q6k"):
            serial = getattr(LIB, f"moe_swiglu_expert_forward_{suffix}_workspace")
            parallel = getattr(
                LIB, f"moe_swiglu_expert_forward_{suffix}_parallel_workspace"
            )
            serial.argtypes = cls.expert_args
            serial.restype = ctypes.c_int
            parallel.argtypes = cls.expert_args
            parallel.restype = ctypes.c_int

            serial_shared = getattr(
                LIB, f"moe_swiglu_shared_forward_{suffix}_workspace"
            )
            parallel_shared = getattr(
                LIB, f"moe_swiglu_shared_forward_{suffix}_parallel_workspace"
            )
            serial_shared.argtypes = cls.shared_args
            serial_shared.restype = ctypes.c_int
            parallel_shared.argtypes = cls.shared_args
            parallel_shared.restype = ctypes.c_int

    def setUp(self) -> None:
        self.rows = 7
        self.hidden = QK_K
        self.intermediate = QK_K
        self.experts = 4
        self.top_k = 3
        rng = np.random.default_rng(20260829)
        self.hidden_values = np.ascontiguousarray(
            rng.normal(0.0, 0.2, size=(self.rows, self.hidden)).astype(np.float32)
        )
        self.routed_values = np.ascontiguousarray(
            rng.normal(0.0, 0.1, size=(self.rows, self.hidden)).astype(np.float32)
        )
        self.indices = np.ascontiguousarray(
            rng.integers(
                0,
                self.experts,
                size=(self.rows, self.top_k),
                dtype=np.int32,
            )
        )
        self.routing = np.ascontiguousarray(
            rng.uniform(0.1, 0.9, size=(self.rows, self.top_k)).astype(np.float32)
        )
        gate_blocks = self.experts * self.intermediate * (self.hidden // QK_K)
        down_blocks = self.experts * self.hidden * (self.intermediate // QK_K)
        self.expert_gate = _q4_blocks(gate_blocks, 1)
        self.expert_up = _q4_blocks(gate_blocks, 2)
        self.expert_down_q4 = _q4_blocks(down_blocks, 3)
        self.expert_down_q6 = _q6_blocks(down_blocks, 4)

        shared_gate_blocks = self.intermediate * (self.hidden // QK_K)
        shared_down_blocks = self.hidden * (self.intermediate // QK_K)
        self.shared_gate = _q4_blocks(shared_gate_blocks, 5)
        self.shared_up = _q4_blocks(shared_gate_blocks, 6)
        self.shared_down_q4 = _q4_blocks(shared_down_blocks, 7)
        self.shared_down_q6 = _q6_blocks(shared_down_blocks, 8)

    def _workspace_stride(self) -> int:
        return int(
            LIB.moe_swiglu_expert_q4k_q5k_workspace_bytes(
                self.hidden, self.intermediate
            )
        )

    def _assert_expert_parity(self, suffix: str, down: ctypes.Array) -> None:
        serial = getattr(LIB, f"moe_swiglu_expert_forward_{suffix}_workspace")
        parallel = getattr(
            LIB, f"moe_swiglu_expert_forward_{suffix}_parallel_workspace"
        )
        stride = self._workspace_stride()
        for rows in (1, self.rows):
            with self.subTest(suffix=suffix, rows=rows):
                expected = np.empty((rows, self.hidden), dtype=np.float32)
                actual = np.empty_like(expected)
                serial_workspace = ctypes.create_string_buffer(stride)
                parallel_workspace = ctypes.create_string_buffer(stride * 64)
                common = (
                    _fptr(self.hidden_values[:rows]),
                    _iptr(self.indices[:rows]),
                    _fptr(self.routing[:rows]),
                    self.expert_gate,
                    self.expert_up,
                    down,
                )
                self.assertEqual(
                    serial(
                        *common,
                        _fptr(expected),
                        rows,
                        self.hidden,
                        self.intermediate,
                        self.experts,
                        self.top_k,
                        serial_workspace,
                        stride,
                    ),
                    0,
                )
                self.assertEqual(
                    parallel(
                        *common,
                        _fptr(actual),
                        rows,
                        self.hidden,
                        self.intermediate,
                        self.experts,
                        self.top_k,
                        parallel_workspace,
                        len(parallel_workspace),
                    ),
                    0,
                )
                np.testing.assert_array_equal(
                    actual.view(np.uint32), expected.view(np.uint32)
                )

    def test_q4_down_expert_parallel_matches_serial_bit_exact(self) -> None:
        self._assert_expert_parity("q4k_q4k", self.expert_down_q4)

    def test_q6_down_expert_parallel_matches_serial_bit_exact(self) -> None:
        self._assert_expert_parity("q4k_q6k", self.expert_down_q6)

    def _assert_shared_parity(self, suffix: str, down: ctypes.Array) -> None:
        serial = getattr(LIB, f"moe_swiglu_shared_forward_{suffix}_workspace")
        parallel = getattr(
            LIB, f"moe_swiglu_shared_forward_{suffix}_parallel_workspace"
        )
        stride = self._workspace_stride()
        expected = np.empty_like(self.hidden_values)
        actual = np.empty_like(expected)
        serial_workspace = ctypes.create_string_buffer(stride)
        parallel_workspace = ctypes.create_string_buffer(stride * 64)
        common = (
            _fptr(self.hidden_values),
            _fptr(self.routed_values),
            self.shared_gate,
            self.shared_up,
            down,
        )
        self.assertEqual(
            serial(
                *common,
                _fptr(expected),
                self.rows,
                self.hidden,
                self.intermediate,
                serial_workspace,
                stride,
            ),
            0,
        )
        self.assertEqual(
            parallel(
                *common,
                _fptr(actual),
                self.rows,
                self.hidden,
                self.intermediate,
                parallel_workspace,
                len(parallel_workspace),
            ),
            0,
        )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_q4_down_shared_parallel_matches_serial_bit_exact(self) -> None:
        self._assert_shared_parity("q4k_q4k", self.shared_down_q4)

    def test_q6_down_shared_parallel_matches_serial_bit_exact(self) -> None:
        self._assert_shared_parity("q4k_q6k", self.shared_down_q6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
