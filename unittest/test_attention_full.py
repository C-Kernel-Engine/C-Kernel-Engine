"""
Full / bidirectional attention kernel tests.

Validates encoder-style prefill attention where every query token attends to
the full key/value sequence with no causal masking.
"""

import ctypes
import math
import unittest

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import max_diff, numpy_to_ptr


lib = load_lib("libckernel_engine.so", "libckernel_attention.so")

lib.attention_forward_full_head_major_gqa_flash.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
]
lib.attention_forward_full_head_major_gqa_flash.restype = None

lib.attention_forward_full_head_major_gqa_flash_strided.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # kv_stride_tokens
]
lib.attention_forward_full_head_major_gqa_flash_strided.restype = None

lib.attention_forward_causal_head_major_gqa_flash.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.attention_forward_causal_head_major_gqa_flash.restype = None


def full_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for full / bidirectional GQA attention."""
    num_heads, num_tokens, head_dim = q.shape
    num_kv_heads = k.shape[0]
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(head_dim)

    for h in range(num_heads):
        kv_head = (h * num_kv_heads) // num_heads
        scores = torch.matmul(q[h], k[kv_head].transpose(0, 1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out[h] = torch.matmul(weights, v[kv_head])

    return out


class TestAttentionFull(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(7)
        self.H = 6
        self.KV = 2
        self.T = 11
        self.D = 16
        self.aligned = 16

        self.q = np.random.randn(self.H, self.T, self.aligned).astype(np.float32)
        self.k = np.random.randn(self.KV, self.T, self.aligned).astype(np.float32)
        self.v = np.random.randn(self.KV, self.T, self.aligned).astype(np.float32)

        self.out_full = np.zeros((self.H, self.T, self.aligned), dtype=np.float32)
        self.out_causal = np.zeros_like(self.out_full)

    def test_full_flash_matches_pytorch_reference(self) -> None:
        lib.attention_forward_full_head_major_gqa_flash(
            numpy_to_ptr(self.q),
            numpy_to_ptr(self.k),
            numpy_to_ptr(self.v),
            numpy_to_ptr(self.out_full),
            ctypes.c_int(self.H),
            ctypes.c_int(self.KV),
            ctypes.c_int(self.T),
            ctypes.c_int(self.D),
            ctypes.c_int(self.aligned),
        )

        ref = full_attention_reference(
            torch.from_numpy(self.q[:, :, : self.D].copy()),
            torch.from_numpy(self.k[:, :, : self.D].copy()),
            torch.from_numpy(self.v[:, :, : self.D].copy()),
        )
        got = torch.from_numpy(self.out_full[:, :, : self.D].copy())
        diff = max_diff(got, ref)
        self.assertLessEqual(diff, 1e-5)

    def test_full_flash_strided_matches_reference(self) -> None:
        stride_tokens = 17
        k_strided = np.zeros((self.KV, stride_tokens, self.aligned), dtype=np.float32)
        v_strided = np.zeros_like(k_strided)
        k_strided[:, : self.T, :] = self.k
        v_strided[:, : self.T, :] = self.v

        lib.attention_forward_full_head_major_gqa_flash_strided(
            numpy_to_ptr(self.q),
            numpy_to_ptr(k_strided),
            numpy_to_ptr(v_strided),
            numpy_to_ptr(self.out_full),
            ctypes.c_int(self.H),
            ctypes.c_int(self.KV),
            ctypes.c_int(self.T),
            ctypes.c_int(self.D),
            ctypes.c_int(self.aligned),
            ctypes.c_int(stride_tokens),
        )

        ref = full_attention_reference(
            torch.from_numpy(self.q[:, :, : self.D].copy()),
            torch.from_numpy(self.k[:, :, : self.D].copy()),
            torch.from_numpy(self.v[:, :, : self.D].copy()),
        )
        got = torch.from_numpy(self.out_full[:, :, : self.D].copy())
        diff = max_diff(got, ref)
        self.assertLessEqual(diff, 1e-5)

    def test_full_attention_differs_from_causal_when_future_tokens_exist(self) -> None:
        lib.attention_forward_full_head_major_gqa_flash(
            numpy_to_ptr(self.q),
            numpy_to_ptr(self.k),
            numpy_to_ptr(self.v),
            numpy_to_ptr(self.out_full),
            ctypes.c_int(self.H),
            ctypes.c_int(self.KV),
            ctypes.c_int(self.T),
            ctypes.c_int(self.D),
            ctypes.c_int(self.aligned),
        )
        lib.attention_forward_causal_head_major_gqa_flash(
            numpy_to_ptr(self.q),
            numpy_to_ptr(self.k),
            numpy_to_ptr(self.v),
            numpy_to_ptr(self.out_causal),
            ctypes.c_int(self.H),
            ctypes.c_int(self.KV),
            ctypes.c_int(self.T),
            ctypes.c_int(self.D),
            ctypes.c_int(self.aligned),
        )

        diff = float(np.max(np.abs(self.out_full[:, :-1, : self.D] - self.out_causal[:, :-1, : self.D])))
        self.assertGreater(diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
