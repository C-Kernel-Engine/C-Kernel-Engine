#!/usr/bin/env python3
"""Test the bounded-workspace Q8_0 gated shared SwiGLU composition."""

from __future__ import annotations

import ctypes
import os
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("CK_ENGINE_SO", ROOT / "build/libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)


def _f32(array: np.ndarray) -> F32P:
    return array.ctypes.data_as(F32P)


@unittest.skipUnless(LIBRARY.exists(), f"missing CKE library: {LIBRARY}")
class MoeSharedQ80Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIBRARY))
        cls.lib.moe_swiglu_shared_q8_0_gated_workspace_bytes.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.lib.moe_swiglu_shared_q8_0_gated_workspace_bytes.restype = ctypes.c_size_t
        cls.lib.moe_swiglu_shared_forward_q8_0_gated_workspace.argtypes = [
            F32P,
            F32P,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            F32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        cls.lib.moe_swiglu_shared_forward_q8_0_gated_workspace.restype = ctypes.c_int
        cls.lib.moe_swiglu_shared_forward_q8_0_gated_parallel_workspace.argtypes = [
            F32P,
            F32P,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            F32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        cls.lib.moe_swiglu_shared_forward_q8_0_gated_parallel_workspace.restype = ctypes.c_int
        cls.lib.quantize_row_q8_0.argtypes = [F32P, ctypes.c_void_p, ctypes.c_int]
        cls.lib.quantize_row_q8_0.restype = None
        cls.lib.gemv_q8_0_q8_0.argtypes = [
            F32P,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.lib.gemv_q8_0_q8_0.restype = None
        cls.lib.swiglu_forward_ggml.argtypes = [F32P, F32P, ctypes.c_int, ctypes.c_int]
        cls.lib.swiglu_forward_ggml.restype = None
        cls.lib.gemm_nt_f32_llama_production.argtypes = [
            F32P,
            F32P,
            F32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        cls.lib.gemm_nt_f32_llama_production.restype = None
        cls.lib.sigmoid_forward.argtypes = [F32P, F32P, ctypes.c_size_t]
        cls.lib.sigmoid_forward.restype = None

    @staticmethod
    def _q8_row_bytes(width: int) -> int:
        return (width // 32) * 34

    def _quantize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        packed = np.empty(matrix.shape[0] * self._q8_row_bytes(matrix.shape[1]), dtype=np.uint8)
        row_bytes = self._q8_row_bytes(matrix.shape[1])
        for row in range(matrix.shape[0]):
            self.lib.quantize_row_q8_0(
                _f32(matrix[row]),
                ctypes.c_void_p(packed.ctypes.data + row * row_bytes),
                matrix.shape[1],
            )
        return packed

    def test_composition_is_deterministic_and_bounded(self) -> None:
        rng = np.random.default_rng(0x3508)
        rows, hidden, intermediate = 3, 64, 32
        x = rng.normal(size=(rows, hidden)).astype(np.float32)
        routed = rng.normal(size=(rows, hidden)).astype(np.float32)
        gate = self._quantize_matrix(rng.normal(size=(intermediate, hidden)))
        up = self._quantize_matrix(rng.normal(size=(intermediate, hidden)))
        down = self._quantize_matrix(rng.normal(size=(hidden, intermediate)))
        scalar_gate = rng.normal(size=(1, hidden)).astype(np.float32)
        required = self.lib.moe_swiglu_shared_q8_0_gated_workspace_bytes(
            hidden, intermediate
        )
        self.assertEqual(required, 128 + 256 + 64 + 256)

        outputs = []
        for _ in range(2):
            output = np.empty_like(x)
            workspace = np.empty(required, dtype=np.uint8)
            rc = self.lib.moe_swiglu_shared_forward_q8_0_gated_workspace(
                _f32(x),
                _f32(routed),
                gate.ctypes.data_as(ctypes.c_void_p),
                up.ctypes.data_as(ctypes.c_void_p),
                down.ctypes.data_as(ctypes.c_void_p),
                _f32(scalar_gate),
                _f32(output),
                rows,
                hidden,
                intermediate,
                workspace.ctypes.data_as(ctypes.c_void_p),
                workspace.nbytes,
            )
            self.assertEqual(rc, 0)
            self.assertTrue(np.all(np.isfinite(output)))
            outputs.append(output.copy())
        np.testing.assert_array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32))

        expected = np.empty_like(x)
        for row in range(rows):
            hidden_q8 = np.empty(self._q8_row_bytes(hidden), dtype=np.uint8)
            gate_up = np.empty(2 * intermediate, dtype=np.float32)
            activation_q8 = np.empty(self._q8_row_bytes(intermediate), dtype=np.uint8)
            shared = np.empty(hidden, dtype=np.float32)
            self.lib.quantize_row_q8_0(
                _f32(x[row]), hidden_q8.ctypes.data_as(ctypes.c_void_p), hidden
            )
            self.lib.gemv_q8_0_q8_0(
                _f32(gate_up[:intermediate]),
                gate.ctypes.data_as(ctypes.c_void_p),
                hidden_q8.ctypes.data_as(ctypes.c_void_p),
                intermediate,
                hidden,
            )
            self.lib.gemv_q8_0_q8_0(
                _f32(gate_up[intermediate:]),
                up.ctypes.data_as(ctypes.c_void_p),
                hidden_q8.ctypes.data_as(ctypes.c_void_p),
                intermediate,
                hidden,
            )
            self.lib.swiglu_forward_ggml(_f32(gate_up), _f32(gate_up), 1, intermediate)
            self.lib.quantize_row_q8_0(
                _f32(gate_up[:intermediate]),
                activation_q8.ctypes.data_as(ctypes.c_void_p),
                intermediate,
            )
            self.lib.gemv_q8_0_q8_0(
                _f32(shared),
                down.ctypes.data_as(ctypes.c_void_p),
                activation_q8.ctypes.data_as(ctypes.c_void_p),
                hidden,
                intermediate,
            )
            scalar = np.empty(1, dtype=np.float32)
            self.lib.gemm_nt_f32_llama_production(
                _f32(x[row]), _f32(scalar_gate), F32P(), _f32(scalar), 1, 1, hidden
            )
            gate_scale = np.empty(1, dtype=np.float32)
            self.lib.sigmoid_forward(_f32(scalar), _f32(gate_scale), 1)
            expected[row] = routed[row] + shared * gate_scale[0]
        np.testing.assert_array_equal(
            outputs[0].view(np.uint32), expected.view(np.uint32)
        )

    def test_parallel_provider_is_bit_exact_with_serial_provider(self) -> None:
        rng = np.random.default_rng(0x3509)
        rows, hidden, intermediate = 67, 64, 32
        x = rng.normal(size=(rows, hidden)).astype(np.float32)
        routed = rng.normal(size=(rows, hidden)).astype(np.float32)
        gate = self._quantize_matrix(rng.normal(size=(intermediate, hidden)))
        up = self._quantize_matrix(rng.normal(size=(intermediate, hidden)))
        down = self._quantize_matrix(rng.normal(size=(hidden, intermediate)))
        scalar_gate = rng.normal(size=(1, hidden)).astype(np.float32)
        stride = self.lib.moe_swiglu_shared_q8_0_gated_workspace_bytes(
            hidden, intermediate
        )
        serial_output = np.empty_like(x)
        serial_workspace = np.empty(stride, dtype=np.uint8)
        self.assertEqual(
            self.lib.moe_swiglu_shared_forward_q8_0_gated_workspace(
                _f32(x),
                _f32(routed),
                gate.ctypes.data_as(ctypes.c_void_p),
                up.ctypes.data_as(ctypes.c_void_p),
                down.ctypes.data_as(ctypes.c_void_p),
                _f32(scalar_gate),
                _f32(serial_output),
                rows,
                hidden,
                intermediate,
                serial_workspace.ctypes.data_as(ctypes.c_void_p),
                serial_workspace.nbytes,
            ),
            0,
        )

        parallel_output = np.empty_like(x)
        parallel_workspace = np.empty(64 * stride, dtype=np.uint8)
        self.assertEqual(
            self.lib.moe_swiglu_shared_forward_q8_0_gated_parallel_workspace(
                _f32(x),
                _f32(routed),
                gate.ctypes.data_as(ctypes.c_void_p),
                up.ctypes.data_as(ctypes.c_void_p),
                down.ctypes.data_as(ctypes.c_void_p),
                _f32(scalar_gate),
                _f32(parallel_output),
                rows,
                hidden,
                intermediate,
                parallel_workspace.ctypes.data_as(ctypes.c_void_p),
                parallel_workspace.nbytes,
            ),
            0,
        )
        np.testing.assert_array_equal(
            parallel_output.view(np.uint32), serial_output.view(np.uint32)
        )

        self.assertEqual(
            self.lib.moe_swiglu_shared_forward_q8_0_gated_parallel_workspace(
                _f32(x),
                _f32(routed),
                gate.ctypes.data_as(ctypes.c_void_p),
                up.ctypes.data_as(ctypes.c_void_p),
                down.ctypes.data_as(ctypes.c_void_p),
                _f32(scalar_gate),
                _f32(parallel_output),
                rows,
                hidden,
                intermediate,
                parallel_workspace.ctypes.data_as(ctypes.c_void_p),
                stride - 1,
            ),
            -1,
        )

    def test_workspace_and_shapes_fail_closed(self) -> None:
        hidden, intermediate = 64, 32
        required = self.lib.moe_swiglu_shared_q8_0_gated_workspace_bytes(
            hidden, intermediate
        )
        x = np.zeros((1, hidden), dtype=np.float32)
        output = np.empty_like(x)
        scalar_gate = np.zeros((1, hidden), dtype=np.float32)
        q8_hidden = np.zeros(self._q8_row_bytes(hidden), dtype=np.uint8)
        q8_intermediate = np.zeros(self._q8_row_bytes(intermediate), dtype=np.uint8)
        workspace = np.empty(required, dtype=np.uint8)
        rc = self.lib.moe_swiglu_shared_forward_q8_0_gated_workspace(
            _f32(x),
            F32P(),
            q8_hidden.ctypes.data_as(ctypes.c_void_p),
            q8_hidden.ctypes.data_as(ctypes.c_void_p),
            q8_intermediate.ctypes.data_as(ctypes.c_void_p),
            _f32(scalar_gate),
            _f32(output),
            1,
            hidden,
            intermediate,
            workspace.ctypes.data_as(ctypes.c_void_p),
            required - 1,
        )
        self.assertEqual(rc, -1)
        self.assertEqual(
            self.lib.moe_swiglu_shared_q8_0_gated_workspace_bytes(63, intermediate),
            0,
        )


if __name__ == "__main__":
    unittest.main()
