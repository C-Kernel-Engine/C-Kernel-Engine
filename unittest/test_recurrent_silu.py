#!/usr/bin/env python3
"""PyTorch parity test for recurrent_silu."""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    print(f"[SKIP] torch not available: {exc}")
    sys.exit(0)


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so")) if (ROOT / "build" / "libckernel_engine.so").exists() else None
if LIB is None:  # pragma: no cover
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)

LIB.recurrent_silu_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
LIB.recurrent_silu_forward_ggml.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
LIB.swiglu_forward_ggml.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
LIB.recurrent_silu_backward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
LIB.ck_set_num_threads.argtypes = [ctypes.c_int]
LIB.ck_set_num_threads.restype = None
LIB.ck_threadpool_global_destroy.argtypes = []
LIB.ck_threadpool_global_destroy.restype = None


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class TestRecurrentSilu(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def _run_case(self, rows: int, dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = (0.25 * rng.standard_normal((rows, dim))).astype(np.float32)
        d_out = (0.20 * rng.standard_normal((rows, dim))).astype(np.float32)
        ck_out = np.zeros_like(x)
        ck_dx = np.zeros_like(x)
        LIB.recurrent_silu_forward(_as_ptr(x), _as_ptr(ck_out), rows, dim)
        LIB.recurrent_silu_backward(_as_ptr(d_out), _as_ptr(x), _as_ptr(ck_dx), rows, dim)

        tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        tout = F.silu(tx)
        tout.backward(torch.tensor(d_out))

        np.testing.assert_allclose(ck_out, tout.detach().numpy(), atol=5e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dx, tx.grad.detach().numpy(), atol=5e-5, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(7, 64, 5)

    def test_medium_case(self) -> None:
        self._run_case(11, 128, 17)

    def _run_ggml_exact_case(self, rows: int, dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = (1.75 * rng.standard_normal((rows, dim))).astype(np.float32)
        ck_out = np.zeros_like(x)
        packed = np.concatenate([x, np.ones_like(x)], axis=1)
        oracle = np.zeros_like(x)
        LIB.recurrent_silu_forward_ggml(_as_ptr(x), _as_ptr(ck_out), rows, dim)
        LIB.swiglu_forward_ggml(_as_ptr(packed), _as_ptr(oracle), rows, dim)
        np.testing.assert_array_equal(ck_out, oracle)

    def test_llama_avx2_vector_and_scalar_tail_are_exact(self) -> None:
        self._run_ggml_exact_case(3, 131, 29)

    def test_llama_qwen35_production_width_is_exact(self) -> None:
        self._run_ggml_exact_case(1, 6144, 31)

    def test_llama_parallel_rows_are_bit_exact_and_repeatable(self) -> None:
        rows, dim = 257, 1027
        rng = np.random.default_rng(43)
        x = (1.75 * rng.standard_normal((rows, dim))).astype(np.float32)

        def run(threads: int, in_place: bool) -> np.ndarray:
            LIB.ck_threadpool_global_destroy()
            LIB.ck_set_num_threads(threads)
            output = x.copy() if in_place else np.empty_like(x)
            source = output if in_place else x
            LIB.recurrent_silu_forward_ggml(
                _as_ptr(source), _as_ptr(output), rows, dim
            )
            return output

        try:
            serial = run(1, False)
            for _ in range(8):
                np.testing.assert_array_equal(run(4, False), serial)
                np.testing.assert_array_equal(run(4, True), serial)
        finally:
            LIB.ck_threadpool_global_destroy()
            LIB.ck_set_num_threads(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
