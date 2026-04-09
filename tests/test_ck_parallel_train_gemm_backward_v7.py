#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNITTEST_DIR = ROOT / "unittest"


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _load_lib():
    if str(UNITTEST_DIR) not in sys.path:
        sys.path.insert(0, str(UNITTEST_DIR))
    from lib_loader import load_lib  # noqa: E402

    lib = load_lib("libckernel_engine.so")
    lib.ck_set_num_threads.argtypes = [ctypes.c_int]
    lib.ck_set_num_threads.restype = None
    lib.gemm_blocked_serial_train_parallel_dispatch.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A [M, K]
        ctypes.POINTER(ctypes.c_float),  # B [N, K]
        ctypes.POINTER(ctypes.c_float),  # bias [N] or NULL
        ctypes.POINTER(ctypes.c_float),  # C [M, N]
        ctypes.c_int,                    # M
        ctypes.c_int,                    # N
        ctypes.c_int,                    # K
    ]
    lib.gemm_blocked_serial_train_parallel_dispatch.restype = None
    for name in (
        "gemm_backward_f32_train_parallel_dispatch",
        "gemm_backward_f32_train_parallel_dispatch_v2",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # d_output [T, O]
            ctypes.POINTER(ctypes.c_float),  # input [T, I]
            ctypes.POINTER(ctypes.c_float),  # W [O, I]
            ctypes.POINTER(ctypes.c_float),  # d_input [T, I]
            ctypes.POINTER(ctypes.c_float),  # d_W [O, I]
            ctypes.POINTER(ctypes.c_float),  # d_b [O] or NULL
            ctypes.c_int,                    # T
            ctypes.c_int,                    # aligned_in
            ctypes.c_int,                    # aligned_out
            ctypes.c_int,                    # num_threads
        ]
        fn.restype = None
    lib.gradient_accumulate_multi_f32.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lib.gradient_accumulate_multi_f32.restype = None
    return lib


LIB = _load_lib()


class TrainParallelGemmBackwardTests(unittest.TestCase):
    def test_forward_dispatch_matches_numpy_reference(self) -> None:
        LIB.ck_set_num_threads(4)
        rng = np.random.default_rng(4321)
        shapes = [
            (4, 128, 256),
            (8, 256, 256),
            (8, 2048, 256),
        ]

        for (m, n, k) in shapes:
            for with_bias in (True, False):
                a = rng.standard_normal((m, k), dtype=np.float32)
                b = rng.standard_normal((n, k), dtype=np.float32)
                c = np.zeros((m, n), dtype=np.float32)
                bias = (rng.standard_normal(n, dtype=np.float32) * 0.01).astype(np.float32)

                bias_ptr = _ptr(bias) if with_bias else ctypes.POINTER(ctypes.c_float)()
                LIB.gemm_blocked_serial_train_parallel_dispatch(
                    _ptr(a.reshape(-1)),
                    _ptr(b.reshape(-1)),
                    bias_ptr,
                    _ptr(c.reshape(-1)),
                    ctypes.c_int(m),
                    ctypes.c_int(n),
                    ctypes.c_int(k),
                )

                c_ref = (a @ b.T).astype(np.float32)
                if with_bias:
                    c_ref = (c_ref + bias.reshape(1, n)).astype(np.float32)

                self.assertLessEqual(
                    float(np.max(np.abs(c - c_ref))),
                    2e-4,
                    msg=f"forward dispatch mismatch for {(m, n, k)}",
                )

    def test_dispatch_wrappers_match_numpy_reference(self) -> None:
        LIB.ck_set_num_threads(4)
        rng = np.random.default_rng(1234)
        shapes = [
            (2, 96, 64),
            (4, 128, 192),
            (8, 256, 128),
            (8, 256, 1024),
        ]

        for symbol in (
            "gemm_backward_f32_train_parallel_dispatch",
            "gemm_backward_f32_train_parallel_dispatch_v2",
        ):
            fn = getattr(LIB, symbol)
            for (t, aligned_in, aligned_out) in shapes:
                for with_bias in (True, False):
                    d_output = rng.standard_normal((t, aligned_out), dtype=np.float32)
                    input_act = rng.standard_normal((t, aligned_in), dtype=np.float32)
                    weights = rng.standard_normal((aligned_out, aligned_in), dtype=np.float32)

                    d_input = np.zeros((t, aligned_in), dtype=np.float32)
                    d_weight = np.zeros((aligned_out, aligned_in), dtype=np.float32)
                    d_bias_init = (rng.standard_normal(aligned_out, dtype=np.float32) * 0.01).astype(np.float32)
                    d_bias = d_bias_init.copy()

                    bias_ptr = _ptr(d_bias) if with_bias else ctypes.POINTER(ctypes.c_float)()
                    fn(
                        _ptr(d_output.reshape(-1)),
                        _ptr(input_act.reshape(-1)),
                        _ptr(weights.reshape(-1)),
                        _ptr(d_input.reshape(-1)),
                        _ptr(d_weight.reshape(-1)),
                        bias_ptr,
                        ctypes.c_int(t),
                        ctypes.c_int(aligned_in),
                        ctypes.c_int(aligned_out),
                        ctypes.c_int(4),
                    )

                    d_input_ref = (d_output @ weights).astype(np.float32)
                    d_weight_ref = (d_output.T @ input_act).astype(np.float32)
                    d_bias_ref = (d_bias_init + d_output.sum(axis=0)).astype(np.float32)

                    self.assertLessEqual(
                        float(np.max(np.abs(d_input - d_input_ref))),
                        2e-4,
                        msg=f"{symbol} d_input mismatch for {(t, aligned_in, aligned_out)}",
                    )
                    self.assertLessEqual(
                        float(np.max(np.abs(d_weight - d_weight_ref))),
                        2e-4,
                        msg=f"{symbol} d_weight mismatch for {(t, aligned_in, aligned_out)}",
                    )
                    if with_bias:
                        self.assertLessEqual(
                            float(np.max(np.abs(d_bias - d_bias_ref))),
                            2e-4,
                            msg=f"{symbol} d_bias mismatch for {(t, aligned_in, aligned_out)}",
                        )
                    else:
                        self.assertLessEqual(
                            float(np.max(np.abs(d_bias - d_bias_init))),
                            0.0,
                            msg=f"{symbol} unexpectedly wrote NULL bias for {(t, aligned_in, aligned_out)}",
                        )

    def test_gradient_accumulate_multi_matches_reference(self) -> None:
        LIB.ck_set_num_threads(4)
        rng = np.random.default_rng(777)
        dsts = [
            rng.standard_normal(2048, dtype=np.float32),
            rng.standard_normal(262144, dtype=np.float32),
            rng.standard_normal(32768, dtype=np.float32),
        ]
        srcs = [
            rng.standard_normal(arr.shape[0], dtype=np.float32)
            for arr in dsts
        ]
        refs = [dst.copy() + src for dst, src in zip(dsts, srcs)]

        dst_ptrs = (ctypes.POINTER(ctypes.c_float) * len(dsts))(*[_ptr(arr) for arr in dsts])
        src_ptrs = (ctypes.POINTER(ctypes.c_float) * len(srcs))(*[_ptr(arr) for arr in srcs])
        numels = (ctypes.c_size_t * len(dsts))(*[arr.shape[0] for arr in dsts])

        LIB.gradient_accumulate_multi_f32(
            dst_ptrs,
            src_ptrs,
            numels,
            ctypes.c_int(len(dsts)),
        )

        for got, ref in zip(dsts, refs):
            self.assertLessEqual(float(np.max(np.abs(got - ref))), 2e-4)


if __name__ == "__main__":
    unittest.main()
