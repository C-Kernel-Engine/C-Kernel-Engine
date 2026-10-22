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

lib.ck_set_num_threads.argtypes = [ctypes.c_int]
lib.ck_set_num_threads.restype = None
lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
lib.ck_set_strict_parity.restype = None

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

lib.attention_forward_full_head_major_gqa_flash_strided_noscale_f32kv.argtypes = list(
    lib.attention_forward_full_head_major_gqa_flash_strided.argtypes
)
lib.attention_forward_full_head_major_gqa_flash_strided_noscale_f32kv.restype = None

lib.attention_forward_full_head_major_gqa_flash_strided_gemma4.argtypes = list(
    lib.attention_forward_full_head_major_gqa_flash_strided.argtypes
)
lib.attention_forward_full_head_major_gqa_flash_strided_gemma4.restype = None

lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided.argtypes = [
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
lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided.restype = None

lib.attention_forward_full_head_major_gqa_tiled64_f16kv_fp32_strided.argtypes = list(
    lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided.argtypes
)
lib.attention_forward_full_head_major_gqa_tiled64_f16kv_fp32_strided.restype = None

lib.attention_forward_full_head_major_gqa_tiled336_f16kv_fp32_strided.argtypes = list(
    lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided.argtypes
)
lib.attention_forward_full_head_major_gqa_tiled336_f16kv_fp32_strided.restype = None

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
lib.attention_forward_full_head_major_gqa_ggml_strided_workspace.argtypes = [
    *lib.attention_forward_full_head_major_gqa_ggml_strided.argtypes,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
lib.attention_forward_full_head_major_gqa_ggml_strided_workspace.restype = None

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

    def test_full_noscale_f32kv_preserves_kv_precision(self) -> None:
        rng = np.random.default_rng(4701)
        heads, kv_heads, tokens, dim, stride = 4, 2, 9, 8, 13
        q = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
        k = np.zeros((kv_heads, stride, dim), dtype=np.float32)
        v = np.zeros_like(k)
        k[:, :tokens] = rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32)
        v[:, :tokens] = rng.standard_normal((kv_heads, tokens, dim), dtype=np.float32)
        output = np.zeros((heads * tokens * dim + 2,), dtype=np.float32)
        output[-2:] = [12345.0, -12345.0]
        rounded_output = np.zeros_like(output)

        lib.attention_forward_full_head_major_gqa_flash_strided_noscale_f32kv(
            numpy_to_ptr(q), numpy_to_ptr(k), numpy_to_ptr(v), numpy_to_ptr(output),
            heads, kv_heads, tokens, dim, dim, stride,
        )
        lib.attention_forward_full_head_major_gqa_flash_strided_gemma4(
            numpy_to_ptr(q), numpy_to_ptr(k), numpy_to_ptr(v), numpy_to_ptr(rounded_output),
            heads, kv_heads, tokens, dim, dim, stride,
        )

        q64 = torch.from_numpy(q).to(torch.float64)
        k64 = torch.from_numpy(k[:, :tokens]).to(torch.float64)
        v64 = torch.from_numpy(v[:, :tokens]).to(torch.float64)
        reference = torch.empty((heads, tokens, dim), dtype=torch.float64)
        rounded_reference = torch.empty_like(reference)
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            for query in range(tokens):
                scores = k64[kv_head] @ q64[head, query]
                reference[head, query] = torch.softmax(scores, dim=0) @ v64[kv_head]
                rounded_k = k64[kv_head].to(torch.float16).to(torch.float64)
                rounded_v = v64[kv_head].to(torch.float16).to(torch.float64)
                rounded_scores = rounded_k @ q64[head, query]
                rounded_reference[head, query] = torch.softmax(rounded_scores, dim=0) @ rounded_v

        actual = output[:-2].reshape(heads, tokens, dim)
        rounded_actual = rounded_output[:-2].reshape(heads, tokens, dim)
        self.assertTrue(np.array_equal(output[-2:], [12345.0, -12345.0]))
        self.assertLess(float(np.max(np.abs(actual - reference.numpy()))), 5e-5)
        self.assertGreater(float(np.max(np.abs(actual - rounded_reference.numpy()))), 1e-5)
        self.assertLess(float(np.max(np.abs(rounded_actual - rounded_reference.numpy()))), 5e-5)

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

    def test_full_flash_threaded_route_matches_reference(self) -> None:
        """Exercise the production threadpool route, not only the small serial route."""
        rng = np.random.default_rng(23)
        heads = 16
        kv_heads = 4
        tokens = 129
        head_dim = 32

        q = rng.standard_normal((heads, tokens, head_dim), dtype=np.float32)
        k = rng.standard_normal((kv_heads, tokens, head_dim), dtype=np.float32)
        v = rng.standard_normal((kv_heads, tokens, head_dim), dtype=np.float32)
        out = np.zeros_like(q)

        lib.attention_forward_full_head_major_gqa_flash(
            numpy_to_ptr(q),
            numpy_to_ptr(k),
            numpy_to_ptr(v),
            numpy_to_ptr(out),
            ctypes.c_int(heads),
            ctypes.c_int(kv_heads),
            ctypes.c_int(tokens),
            ctypes.c_int(head_dim),
            ctypes.c_int(head_dim),
        )

        ref = full_attention_reference_kv_f16(
            torch.from_numpy(q.copy()),
            torch.from_numpy(k.copy()),
            torch.from_numpy(v.copy()),
        )
        diff = max_diff(torch.from_numpy(out), ref)
        self.assertLessEqual(diff, 5e-5)

    def test_full_tiled_f16kv_contract_qwen3vl_multi_tile(self) -> None:
        """Exercise the contract-selected 64x64 path across key/query tiles."""
        rng = np.random.default_rng(29)
        heads = 16
        kv_heads = 16
        tokens = 129
        head_dim = 72
        stride_tokens = 137

        q = rng.standard_normal((heads, tokens, head_dim), dtype=np.float32)
        k = np.zeros((kv_heads, stride_tokens, head_dim), dtype=np.float32)
        v = np.zeros_like(k)
        k[:, :tokens] = rng.standard_normal((kv_heads, tokens, head_dim), dtype=np.float32)
        v[:, :tokens] = rng.standard_normal((kv_heads, tokens, head_dim), dtype=np.float32)
        out = np.zeros_like(q)
        repeated = np.zeros_like(q)

        args = (
            numpy_to_ptr(q), numpy_to_ptr(k), numpy_to_ptr(v),
            ctypes.c_int(heads), ctypes.c_int(kv_heads), ctypes.c_int(tokens),
            ctypes.c_int(head_dim), ctypes.c_int(head_dim), ctypes.c_int(stride_tokens),
        )
        lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided(
            args[0], args[1], args[2], numpy_to_ptr(out), *args[3:]
        )
        lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided(
            args[0], args[1], args[2], numpy_to_ptr(repeated), *args[3:]
        )

        ref = full_attention_reference_kv_f16(
            torch.from_numpy(q.copy()),
            torch.from_numpy(k[:, :tokens].copy()),
            torch.from_numpy(v[:, :tokens].copy()),
        )
        self.assertTrue(np.array_equal(out.view(np.uint32), repeated.view(np.uint32)))
        self.assertLessEqual(max_diff(torch.from_numpy(out), ref), 5e-5)

    def test_full_tiled336_preserves_tile64_output_across_crossover(self) -> None:
        """Query tiling may change batching and scratch use, never reduction math."""
        rng = np.random.default_rng(47)
        heads = 16
        kv_heads = 16
        head_dim = 72
        self.addCleanup(lib.ck_set_num_threads, ctypes.c_int(20))

        for tokens, threads in ((129, 1), (1536, 8), (4032, 20), (4032, 28)):
            with self.subTest(tokens=tokens, threads=threads):
                lib.ck_set_num_threads(ctypes.c_int(threads))
                shape = (heads, tokens, head_dim)
                q = rng.standard_normal(shape, dtype=np.float32)
                k = rng.standard_normal(shape, dtype=np.float32)
                v = rng.standard_normal(shape, dtype=np.float32)
                tile64 = np.zeros_like(q)
                tile336 = np.zeros_like(q)
                args = (
                    numpy_to_ptr(q), numpy_to_ptr(k), numpy_to_ptr(v),
                    ctypes.c_int(heads), ctypes.c_int(kv_heads), ctypes.c_int(tokens),
                    ctypes.c_int(head_dim), ctypes.c_int(head_dim), ctypes.c_int(tokens),
                )

                lib.attention_forward_full_head_major_gqa_tiled64_f16kv_fp32_strided(
                    args[0], args[1], args[2], numpy_to_ptr(tile64), *args[3:]
                )
                lib.attention_forward_full_head_major_gqa_tiled336_f16kv_fp32_strided(
                    args[0], args[1], args[2], numpy_to_ptr(tile336), *args[3:]
                )

                self.assertTrue(
                    np.array_equal(tile64.view(np.uint32), tile336.view(np.uint32)),
                    f"query-tile providers differ at T={tokens}",
                )
                production = np.zeros_like(q)
                lib.attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided(
                    args[0], args[1], args[2], numpy_to_ptr(production), *args[3:]
                )
                expected = tile336 if tokens >= 1536 else tile64
                self.assertTrue(
                    np.array_equal(production.view(np.uint32), expected.view(np.uint32)),
                    f"production query-tile threshold selected the wrong provider at T={tokens}",
                )

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

    def test_full_ggml_workspace_is_byte_exact_with_compatibility_entry(self) -> None:
        try:
            for strict in (0, 1):
                lib.ck_set_strict_parity(strict)
                expected = np.zeros_like(self.out_full)
                actual = np.zeros_like(self.out_full)
                lib.attention_forward_full_head_major_gqa_ggml_strided(
                    numpy_to_ptr(self.q), numpy_to_ptr(self.k), numpy_to_ptr(self.v),
                    numpy_to_ptr(expected), self.H, self.KV, self.T, self.D,
                    self.aligned, self.T,
                )
                score_rows = np.empty((self.H, self.T), dtype=np.float32)
                v_columns = np.empty((self.H, self.D, self.T), dtype=np.float32)
                probability_row = np.empty(self.T, dtype=np.float32)
                lib.attention_forward_full_head_major_gqa_ggml_strided_workspace(
                    numpy_to_ptr(self.q), numpy_to_ptr(self.k), numpy_to_ptr(self.v),
                    numpy_to_ptr(actual), self.H, self.KV, self.T, self.D,
                    self.aligned, self.T,
                    numpy_to_ptr(score_rows), score_rows.nbytes,
                    numpy_to_ptr(v_columns), v_columns.nbytes,
                    numpy_to_ptr(probability_row), probability_row.nbytes,
                )
                self.assertTrue(
                    np.array_equal(actual.view(np.uint32), expected.view(np.uint32)),
                    f"workspace drifted with strict={strict}",
                )
        finally:
            lib.ck_set_strict_parity(0)

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
