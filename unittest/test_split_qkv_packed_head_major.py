#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIB_CANDIDATES = [
    ROOT / "build" / "libckernel_engine.so",
    ROOT / "libckernel_engine.so",
]


def _load_lib() -> ctypes.CDLL | None:
    for path in LIB_CANDIDATES:
        if not path.exists():
            continue
        lib = ctypes.CDLL(str(path))
        fn = lib.split_qkv_packed_head_major_forward
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        fn.restype = None
        return lib
    return None


LIB = _load_lib()
if LIB is None:  # pragma: no cover - dependency skip
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ref_split_qkv_head_major(
    packed: np.ndarray,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = packed.shape[0]
    q_head_dim = q_dim // num_heads
    k_head_dim = k_dim // num_kv_heads
    v_head_dim = v_dim // num_kv_heads

    q = packed[:, :q_dim].reshape(rows, num_heads, q_head_dim).transpose(1, 0, 2).copy()
    k = packed[:, q_dim:q_dim + k_dim].reshape(rows, num_kv_heads, k_head_dim).transpose(1, 0, 2).copy()
    v = packed[:, q_dim + k_dim:].reshape(rows, num_kv_heads, v_head_dim).transpose(1, 0, 2).copy()
    return q, k, v


class TestSplitQKVPackedHeadMajor(unittest.TestCase):
    def _run_case(
        self,
        *,
        rows: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        q_dim = num_heads * head_dim
        k_dim = num_kv_heads * head_dim
        v_dim = num_kv_heads * head_dim
        packed = (0.25 * rng.standard_normal((rows, q_dim + k_dim + v_dim))).astype(np.float32)

        ref_q, ref_k, ref_v = _ref_split_qkv_head_major(
            packed, q_dim, k_dim, v_dim, num_heads, num_kv_heads
        )

        out_q = np.zeros_like(ref_q)
        out_k = np.zeros_like(ref_k)
        out_v = np.zeros_like(ref_v)
        LIB.split_qkv_packed_head_major_forward(
            _as_ptr(packed),
            _as_ptr(out_q),
            _as_ptr(out_k),
            _as_ptr(out_v),
            rows,
            q_dim,
            k_dim,
            v_dim,
            num_heads,
            num_kv_heads,
        )

        np.testing.assert_allclose(out_q, ref_q, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(out_k, ref_k, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(out_v, ref_v, atol=0.0, rtol=0.0)

    def test_small_dense_case(self) -> None:
        self._run_case(rows=4, num_heads=2, num_kv_heads=2, head_dim=8, seed=7)

    def test_gqa_case(self) -> None:
        self._run_case(rows=9, num_heads=8, num_kv_heads=2, head_dim=16, seed=13)

    def test_qwen3vl_like_case(self) -> None:
        self._run_case(rows=17, num_heads=16, num_kv_heads=16, head_dim=72, seed=23)


if __name__ == "__main__":
    unittest.main(verbosity=2)
