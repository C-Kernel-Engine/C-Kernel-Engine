#!/usr/bin/env python3
"""Certify the allocation-free full-softmax MoE router against llama.cpp."""

from __future__ import annotations

import ctypes
import os
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CK_LIBRARY = Path(os.environ.get("CK_ENGINE_SO", ROOT / "build/libckernel_engine.so"))
LLAMA_ROOT = Path(os.environ.get("CK_LLAMA_CPP_ROOT", ROOT / "llama.cpp"))
LLAMA_LIBRARY = LLAMA_ROOT / "build/bin/libggml-cpu.so"

F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)


def _as_f32_pointer(array: np.ndarray) -> F32P:
    return array.ctypes.data_as(F32P)


def _as_i32_pointer(array: np.ndarray) -> I32P:
    return array.ctypes.data_as(I32P)


@unittest.skipUnless(CK_LIBRARY.exists(), f"missing CKE library: {CK_LIBRARY}")
@unittest.skipUnless(LLAMA_LIBRARY.exists(), f"missing llama.cpp library: {LLAMA_LIBRARY}")
class MoeSoftmaxTopkRouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cke = ctypes.CDLL(str(CK_LIBRARY))
        cls.llama = ctypes.CDLL(str(LLAMA_LIBRARY))

        cls.cke.moe_softmax_topk_router_workspace_bytes.argtypes = [ctypes.c_int]
        cls.cke.moe_softmax_topk_router_workspace_bytes.restype = ctypes.c_size_t
        cls.cke.moe_softmax_topk_router_llama_f32_workspace.argtypes = [
            F32P,
            I32P,
            F32P,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        cls.cke.moe_softmax_topk_router_llama_f32_workspace.restype = ctypes.c_int

        cls.llama.ggml_vec_soft_max_f32.argtypes = [
            ctypes.c_int,
            F32P,
            F32P,
            ctypes.c_float,
        ]
        cls.llama.ggml_vec_soft_max_f32.restype = ctypes.c_double

    def _run_cke(
        self, logits: np.ndarray, top_k: int, scale: float = 1.0
    ) -> tuple[int, np.ndarray, np.ndarray]:
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        rows, experts = logits.shape
        indices = np.empty((rows, top_k), dtype=np.int32)
        weights = np.empty((rows, top_k), dtype=np.float32)
        workspace_size = self.cke.moe_softmax_topk_router_workspace_bytes(experts)
        workspace = np.empty(workspace_size, dtype=np.uint8)
        rc = self.cke.moe_softmax_topk_router_llama_f32_workspace(
            _as_f32_pointer(logits),
            _as_i32_pointer(indices),
            _as_f32_pointer(weights),
            rows,
            experts,
            top_k,
            ctypes.c_float(scale),
            workspace.ctypes.data_as(ctypes.c_void_p),
            workspace.nbytes,
        )
        return rc, indices, weights

    def _llama_reference(
        self, logits: np.ndarray, top_k: int, scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, experts = logits.shape
        indices = np.empty((rows, top_k), dtype=np.int32)
        weights = np.empty((rows, top_k), dtype=np.float32)
        for row in range(rows):
            probabilities = np.empty(experts, dtype=np.float32)
            softmax_sum = self.llama.ggml_vec_soft_max_f32(
                experts,
                _as_f32_pointer(probabilities),
                _as_f32_pointer(logits[row]),
                ctypes.c_float(float(np.max(logits[row]))),
            )
            probabilities *= np.float32(1.0 / softmax_sum)
            selected = np.argsort(-probabilities, kind="stable")[:top_k]
            selected_weights = probabilities[selected].copy()
            selected_sum = np.float32(sum(float(value) for value in selected_weights))
            selected_sum = max(selected_sum, np.float32(6.103515625e-5))
            indices[row] = selected
            weights[row] = (selected_weights / selected_sum) * np.float32(scale)
        return indices, weights

    def test_qwen35_shape_is_bit_exact_with_llama_softmax(self) -> None:
        rng = np.random.default_rng(0x35A3B)
        for rows in (1, 7, 32):
            logits = rng.normal(0.0, 3.0, size=(rows, 256)).astype(np.float32)
            for scale in (1.0, 2.5):
                rc, actual_indices, actual_weights = self._run_cke(logits, 8, scale)
                expected_indices, expected_weights = self._llama_reference(
                    logits, 8, scale
                )
                self.assertEqual(rc, 0)
                np.testing.assert_array_equal(actual_indices, expected_indices)
                np.testing.assert_array_equal(
                    actual_weights.view(np.uint32), expected_weights.view(np.uint32)
                )

    def test_workspace_is_exact_and_fail_closed(self) -> None:
        experts = 256
        required = self.cke.moe_softmax_topk_router_workspace_bytes(experts)
        self.assertEqual(required, 1024)
        logits = np.arange(experts, dtype=np.float32).reshape(1, experts)
        indices = np.empty((1, 8), dtype=np.int32)
        weights = np.empty((1, 8), dtype=np.float32)
        workspace = np.empty(required, dtype=np.uint8)
        rc = self.cke.moe_softmax_topk_router_llama_f32_workspace(
            _as_f32_pointer(logits),
            _as_i32_pointer(indices),
            _as_f32_pointer(weights),
            1,
            experts,
            8,
            ctypes.c_float(1.0),
            workspace.ctypes.data_as(ctypes.c_void_p),
            required - 1,
        )
        self.assertEqual(rc, -1)

    def test_nonfinite_logits_are_rejected(self) -> None:
        logits = np.zeros((1, 256), dtype=np.float32)
        logits[0, 17] = np.nan
        rc, _, _ = self._run_cke(logits, 8)
        self.assertEqual(rc, -2)

    def test_equal_probabilities_are_deterministic(self) -> None:
        logits = np.zeros((1, 256), dtype=np.float32)
        rc, indices, weights = self._run_cke(logits, 8)
        self.assertEqual(rc, 0)
        np.testing.assert_array_equal(indices[0], np.arange(8, dtype=np.int32))
        np.testing.assert_array_equal(
            weights[0].view(np.uint32),
            np.full(8, np.float32(0.125), dtype=np.float32).view(np.uint32),
        )


if __name__ == "__main__":
    unittest.main()
