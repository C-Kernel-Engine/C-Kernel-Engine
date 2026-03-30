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

lib.attention_forward_full_head_major_gqa_exact_strided.argtypes = [
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
lib.attention_forward_full_head_major_gqa_exact_strided.restype = None

lib.attention_forward_full_head_major_gqa_ggml_strided.argtypes = [
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
lib.attention_forward_full_head_major_gqa_ggml_strided.restype = None

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


def full_attention_reference_kv_f16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ggml-like full attention reference with F32 Q and F16-rounded K/V."""
    num_heads, num_tokens, head_dim = q.shape
    num_kv_heads = k.shape[0]
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(head_dim)

    def round_f16(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float16).to(torch.float32)

    for h in range(num_heads):
        kv_head = (h * num_kv_heads) // num_heads
        k_head = round_f16(k[kv_head])
        v_head = round_f16(v[kv_head])

        scores = torch.matmul(q[h], k_head.transpose(0, 1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out[h] = torch.matmul(weights, v_head)

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

        ref = full_attention_reference_kv_f16(
            torch.from_numpy(self.q[:, :, : self.D].copy()),
            torch.from_numpy(self.k[:, :, : self.D].copy()),
            torch.from_numpy(self.v[:, :, : self.D].copy()),
        )
        got = torch.from_numpy(self.out_full[:, :, : self.D].copy())
        diff = max_diff(got, ref)
        self.assertLessEqual(diff, 5e-5)

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

        ref = full_attention_reference_kv_f16(
            torch.from_numpy(self.q[:, :, : self.D].copy()),
            torch.from_numpy(self.k[:, :, : self.D].copy()),
            torch.from_numpy(self.v[:, :, : self.D].copy()),
        )
        got = torch.from_numpy(self.out_full[:, :, : self.D].copy())
        diff = max_diff(got, ref)
        self.assertLessEqual(diff, 5e-5)

    def test_full_flash_matches_reference_qwen3vl_shape(self) -> None:
        rng = np.random.default_rng(11)
        heads = 16
        kv_heads = 16
        tokens = 19
        head_dim = 72
        aligned = 72

        q = rng.standard_normal((heads, tokens, aligned), dtype=np.float32)
        k = rng.standard_normal((kv_heads, tokens, aligned), dtype=np.float32)
        v = rng.standard_normal((kv_heads, tokens, aligned), dtype=np.float32)
        out = np.zeros((heads, tokens, aligned), dtype=np.float32)

        lib.attention_forward_full_head_major_gqa_flash_strided(
            numpy_to_ptr(q),
            numpy_to_ptr(k),
            numpy_to_ptr(v),
            numpy_to_ptr(out),
            ctypes.c_int(heads),
            ctypes.c_int(kv_heads),
            ctypes.c_int(tokens),
            ctypes.c_int(head_dim),
            ctypes.c_int(aligned),
            ctypes.c_int(tokens),
        )

        ref = full_attention_reference_kv_f16(
            torch.from_numpy(q[:, :, :head_dim].copy()),
            torch.from_numpy(k[:, :, :head_dim].copy()),
            torch.from_numpy(v[:, :, :head_dim].copy()),
        )
        got = torch.from_numpy(out[:, :, :head_dim].copy())
        diff = max_diff(got, ref)
        self.assertLessEqual(diff, 5e-5)

    def test_full_exact_strided_matches_pytorch_reference(self) -> None:
        lib.attention_forward_full_head_major_gqa_exact_strided(
            numpy_to_ptr(self.q),
            numpy_to_ptr(self.k),
            numpy_to_ptr(self.v),
            numpy_to_ptr(self.out_full),
            ctypes.c_int(self.H),
            ctypes.c_int(self.KV),
            ctypes.c_int(self.T),
            ctypes.c_int(self.D),
            ctypes.c_int(self.aligned),
            ctypes.c_int(self.T),
        )

        ref = full_attention_reference(
            torch.from_numpy(self.q[:, :, : self.D].copy()),
            torch.from_numpy(self.k[:, :, : self.D].copy()),
            torch.from_numpy(self.v[:, :, : self.D].copy()),
        )
        got = torch.from_numpy(self.out_full[:, :, : self.D].copy())
        diff = max_diff(got, ref)
        self.assertLessEqual(diff, 1e-5)

    def test_full_ggml_strided_matches_pytorch_reference(self) -> None:
        lib.attention_forward_full_head_major_gqa_ggml_strided(
            numpy_to_ptr(self.q),
            numpy_to_ptr(self.k),
            numpy_to_ptr(self.v),
            numpy_to_ptr(self.out_full),
            ctypes.c_int(self.H),
            ctypes.c_int(self.KV),
            ctypes.c_int(self.T),
            ctypes.c_int(self.D),
            ctypes.c_int(self.aligned),
            ctypes.c_int(self.T),
        )

        ref = full_attention_reference(
            torch.from_numpy(self.q[:, :, : self.D].copy()),
            torch.from_numpy(self.k[:, :, : self.D].copy()),
            torch.from_numpy(self.v[:, :, : self.D].copy()),
        )
        got = torch.from_numpy(self.out_full[:, :, : self.D].copy())
        diff = max_diff(got, ref)
        self.assertLessEqual(diff, 1e-4)

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
