#!/usr/bin/env python3
"""Contract tests for compact Q4_K/Q5_K routed SwiGLU composition."""

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
Q5_K_BYTES = 176
Q8_K_BYTES = 292


def _fptr(array: np.ndarray) -> FPTR:
    return array.ctypes.data_as(FPTR)


def _iptr(array: np.ndarray) -> IPTR:
    return array.ctypes.data_as(IPTR)


def _weight_blocks(count: int, block_bytes: int, seed: int) -> ctypes.Array:
    rng = np.random.default_rng(seed)
    raw = bytearray(count * block_bytes)
    for block in range(count):
        offset = block * block_bytes
        struct.pack_into("<H", raw, offset, 0x211F + block % 7)
        struct.pack_into("<H", raw, offset + 2, 0x1800 + block % 5)
        raw[offset + 4 : offset + 16] = rng.integers(
            0, 256, size=12, dtype=np.uint8
        ).tobytes()
        raw[offset + 16 : offset + block_bytes] = rng.integers(
            0, 256, size=block_bytes - 16, dtype=np.uint8
        ).tobytes()
    return ctypes.create_string_buffer(bytes(raw), len(raw))


class CompactRoutedSwiGLUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        LIB.moe_swiglu_expert_q4k_q5k_workspace_bytes.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
        ]
        LIB.moe_swiglu_expert_q4k_q5k_workspace_bytes.restype = ctypes.c_size_t
        LIB.moe_swiglu_expert_forward_q4k_q5k_workspace.argtypes = [
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
        LIB.moe_swiglu_expert_forward_q4k_q5k_workspace.restype = ctypes.c_int
        LIB.quantize_row_q8_k.argtypes = [FPTR, ctypes.c_void_p, ctypes.c_int]
        LIB.gemv_q4_k_q8_k.argtypes = [
            FPTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        LIB.gemv_q5_k_q8_k.argtypes = LIB.gemv_q4_k_q8_k.argtypes
        LIB.swiglu_forward_ggml.argtypes = [FPTR, FPTR, ctypes.c_int, ctypes.c_int]
        LIB.axpy_f32.argtypes = [FPTR, FPTR, ctypes.c_float, ctypes.c_int]

    def setUp(self) -> None:
        self.rows = 2
        self.hidden = QK_K
        self.intermediate = QK_K
        self.experts = 3
        self.top_k = 2
        rng = np.random.default_rng(1234)
        self.x = np.ascontiguousarray(
            rng.normal(0.0, 0.2, size=(self.rows, self.hidden)).astype(np.float32)
        )
        self.indices = np.ascontiguousarray(
            np.array([[2, 0], [1, 2]], dtype=np.int32)
        )
        self.routing = np.ascontiguousarray(
            np.array([[0.65, 0.35], [0.55, 0.45]], dtype=np.float32)
        )
        blocks_up = self.experts * self.intermediate
        blocks_down = self.experts * self.hidden
        self.gate = _weight_blocks(blocks_up, Q4_K_BYTES, 1)
        self.up = _weight_blocks(blocks_up, Q4_K_BYTES, 2)
        self.down = _weight_blocks(blocks_down, Q5_K_BYTES, 3)

    def _run_composed_reference(self) -> np.ndarray:
        result = np.zeros((self.rows, self.hidden), dtype=np.float32)
        hidden_q8 = ctypes.create_string_buffer(Q8_K_BYTES)
        act_q8 = ctypes.create_string_buffer(Q8_K_BYTES)
        gate_up = np.empty(2 * self.intermediate, dtype=np.float32)
        expert_output = np.empty(self.hidden, dtype=np.float32)
        q4_expert_stride = self.intermediate * Q4_K_BYTES
        q5_expert_stride = self.hidden * Q5_K_BYTES

        for row in range(self.rows):
            LIB.quantize_row_q8_k(_fptr(self.x[row]), hidden_q8, self.hidden)
            for slot in range(self.top_k):
                expert = int(self.indices[row, slot])
                gate_ptr = ctypes.byref(self.gate, expert * q4_expert_stride)
                up_ptr = ctypes.byref(self.up, expert * q4_expert_stride)
                down_ptr = ctypes.byref(self.down, expert * q5_expert_stride)
                LIB.gemv_q4_k_q8_k(
                    _fptr(gate_up), gate_ptr, hidden_q8, self.intermediate, self.hidden
                )
                LIB.gemv_q4_k_q8_k(
                    _fptr(gate_up[self.intermediate :]),
                    up_ptr,
                    hidden_q8,
                    self.intermediate,
                    self.hidden,
                )
                LIB.swiglu_forward_ggml(
                    _fptr(gate_up), _fptr(gate_up), 1, self.intermediate
                )
                LIB.quantize_row_q8_k(_fptr(gate_up), act_q8, self.intermediate)
                LIB.gemv_q5_k_q8_k(
                    _fptr(expert_output),
                    down_ptr,
                    act_q8,
                    self.hidden,
                    self.intermediate,
                )
                LIB.axpy_f32(
                    _fptr(result[row]),
                    _fptr(expert_output),
                    float(self.routing[row, slot]),
                    self.hidden,
                )
        return result

    def test_matches_composed_primitives_bit_exact(self) -> None:
        expected = self._run_composed_reference()
        actual = np.full_like(expected, np.nan)
        workspace_bytes = LIB.moe_swiglu_expert_q4k_q5k_workspace_bytes(
            self.hidden, self.intermediate
        )
        workspace = ctypes.create_string_buffer(workspace_bytes)
        status = LIB.moe_swiglu_expert_forward_q4k_q5k_workspace(
            _fptr(self.x),
            _iptr(self.indices),
            _fptr(self.routing),
            self.gate,
            self.up,
            self.down,
            _fptr(actual),
            self.rows,
            self.hidden,
            self.intermediate,
            self.experts,
            self.top_k,
            workspace,
            workspace_bytes,
        )
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_workspace_and_indices_fail_closed(self) -> None:
        output = np.zeros((self.rows, self.hidden), dtype=np.float32)
        required = LIB.moe_swiglu_expert_q4k_q5k_workspace_bytes(
            self.hidden, self.intermediate
        )
        workspace = ctypes.create_string_buffer(required)
        status = LIB.moe_swiglu_expert_forward_q4k_q5k_workspace(
            _fptr(self.x), _iptr(self.indices), _fptr(self.routing),
            self.gate, self.up, self.down, _fptr(output),
            self.rows, self.hidden, self.intermediate, self.experts, self.top_k,
            workspace, required - 1,
        )
        self.assertEqual(status, -1)

        invalid = self.indices.copy()
        invalid[0, 0] = self.experts
        status = LIB.moe_swiglu_expert_forward_q4k_q5k_workspace(
            _fptr(self.x), _iptr(invalid), _fptr(self.routing),
            self.gate, self.up, self.down, _fptr(output),
            self.rows, self.hidden, self.intermediate, self.experts, self.top_k,
            workspace, required,
        )
        self.assertEqual(status, -2)


if __name__ == "__main__":
    unittest.main()
