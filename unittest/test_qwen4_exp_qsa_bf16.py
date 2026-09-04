#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import math
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[1]
TORCH_CPU = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
if TORCH_CPU.exists():
    import os

    os.environ.setdefault("CK_MKL_LIBRARY", str(TORCH_CPU))
    os.environ.setdefault("CK_SLEEF_LIBRARY", str(TORCH_CPU))
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int32)
U16P = ctypes.POINTER(ctypes.c_uint16)

LIB.qwen4_qsa_index_select_bf16.argtypes = [
    F32P, F32P, F32P, F32P, F32P, F32P, F32P, F32P, F32P, I32P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
]
LIB.attention_forward_sparse_token_major_gqa_bf16cache_pytorch_cpu_flash_contract.argtypes = [
    F32P, U16P, U16P, F32P, F32P, F32P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
LIB.ck_attention_sparse_bf16_pytorch_gqa_available.argtypes = []
LIB.ck_attention_sparse_bf16_pytorch_gqa_available.restype = ctypes.c_int


def fptr(value: np.ndarray) -> F32P:
    return value.ctypes.data_as(F32P)


def i32ptr(value: np.ndarray) -> I32P:
    return value.ctypes.data_as(I32P)


def u16ptr(value: np.ndarray) -> U16P:
    return value.ctypes.data_as(U16P)


def bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


def bf16_bits(value: torch.Tensor) -> np.ndarray:
    return value.to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy()


def rmsnorm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + eps)
    return bf16(normalized * weight.float())


def rope_split(value: torch.Tensor, rotary_dim: int, position: int, theta: float) -> torch.Tensor:
    output = value.clone()
    half = rotary_dim // 2
    index = torch.arange(half, dtype=torch.float32)
    angle = position * theta ** (-2.0 * index / rotary_dim)
    first = value[..., :half].float()
    second = value[..., half:rotary_dim].float()
    output[..., :half] = bf16(first * angle.cos() - second * angle.sin())
    output[..., half:rotary_dim] = bf16(second * angle.cos() + first * angle.sin())
    return output


class TestQwen4ExpQsaBf16(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def test_index_selection_matches_block_oracle_and_is_chunk_stable(self) -> None:
        rows, query_heads, index_dim = 13, 3, 8
        budget, compress, rotary_dim, context = 8, 2, 4, 32
        selection_width = budget + compress - 1
        block_topk = budget // compress
        projected = bf16(torch.randn(rows, (query_heads + 1) * index_dim))
        q_weight = (1.0 + 0.1 * torch.randn(index_dim)).float()
        k_weight = (1.0 + 0.1 * torch.randn(index_dim)).float()
        theta, eps = 10000.0, 1e-6

        def run(start: int, end: int, cache: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            count = end - start
            selected = np.full((count, selection_width), -1.0, dtype=np.float32)
            output_cache = cache.copy()
            LIB.qwen4_qsa_index_select_bf16(
                fptr(projected[start:end].numpy().copy()), fptr(cache),
                fptr(q_weight.numpy().copy()), fptr(k_weight.numpy().copy()),
                fptr(selected), fptr(output_cache),
                fptr(np.zeros((query_heads, index_dim), dtype=np.float32)),
                fptr(np.zeros(index_dim, dtype=np.float32)),
                fptr(np.zeros(block_topk, dtype=np.float32)),
                i32ptr(np.zeros(block_topk, dtype=np.int32)),
                count, query_heads, index_dim, budget, compress, rotary_dim,
                context, start, ctypes.c_float(theta), ctypes.c_float(eps),
            )
            return selected, output_cache

        zero_cache = np.zeros((context, index_dim), dtype=np.float32)
        one_selected, one_cache = run(0, rows, zero_cache)
        first_selected, first_cache = run(0, 5, zero_cache)
        second_selected, split_cache = run(5, rows, first_cache)
        np.testing.assert_array_equal(np.concatenate([first_selected, second_selected]), one_selected)
        np.testing.assert_array_equal(split_cache[:rows], one_cache[:rows])

        raw_keys = projected[:, query_heads * index_dim :].reshape(rows, index_dim)
        for row in range(rows):
            query = rmsnorm(
                projected[row, : query_heads * index_dim].reshape(query_heads, index_dim),
                q_weight,
                eps,
            )
            query = rope_split(query, rotary_dim, row, theta)
            complete = (row + 1) // compress
            scores = []
            for block in range(complete):
                pooled = bf16(raw_keys[block * compress : (block + 1) * compress].float().mean(0))
                pooled = rmsnorm(pooled, k_weight, eps)
                pooled = rope_split(pooled, rotary_dim, block * compress, theta)
                score = torch.relu(query.float() @ pooled.float()).sum() / math.sqrt(index_dim)
                scores.append(float(score))
            chosen = sorted(
                sorted(range(complete), key=lambda index: scores[index], reverse=True)[:block_topk]
            )
            expected = [token for block in chosen for token in range(block * compress, (block + 1) * compress)]
            expected.extend(range(complete * compress, row + 1))
            actual = [int(item) for item in one_selected[row] if item >= 0]
            self.assertEqual(actual, expected)

    def test_sparse_attention_matches_pytorch_cpu_flash_contract_exactly(self) -> None:
        if not LIB.ck_attention_sparse_bf16_pytorch_gqa_available():
            self.skipTest("sparse BF16 CPU-flash provider requires AVX-512F")
        rows, query_heads, kv_heads, head_dim = 3, 4, 2, 8
        context, position, width = 12, 5, 7
        query = bf16(torch.randn(rows, query_heads, head_dim))
        key = bf16(torch.randn(kv_heads, context, head_dim))
        value = bf16(torch.randn(kv_heads, context, head_dim))
        selected = np.full((rows, width), -1.0, dtype=np.float32)
        selected[0, :4] = [0, 2, 4, 5]
        selected[1, :5] = [0, 1, 3, 5, 6]
        selected[2, :6] = [0, 2, 3, 4, 6, 7]
        output = np.zeros((rows, query_heads, head_dim), dtype=np.float32)
        LIB.attention_forward_sparse_token_major_gqa_bf16cache_pytorch_cpu_flash_contract(
            fptr(query.numpy().copy()), u16ptr(bf16_bits(key)), u16ptr(bf16_bits(value)),
            fptr(selected), fptr(output), fptr(np.zeros(width, dtype=np.float32)),
            rows, query_heads, kv_heads, head_dim, width, context, position,
        )

        expected = torch.zeros_like(query)
        groups = query_heads // kv_heads
        for row in range(rows):
            visible = position + row + 1
            indices = [int(item) for item in selected[row] if item >= 0]
            mask = torch.zeros((1, 1, 1, visible), dtype=torch.bool)
            mask[..., indices] = True
            repeated_key = key[:, :visible].repeat_interleave(groups, dim=0)
            repeated_value = value[:, :visible].repeat_interleave(groups, dim=0)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                expected[row] = torch.nn.functional.scaled_dot_product_attention(
                    query[row].unsqueeze(0).unsqueeze(2).to(torch.bfloat16),
                    repeated_key.unsqueeze(0).to(torch.bfloat16),
                    repeated_value.unsqueeze(0).to(torch.bfloat16),
                    attn_mask=mask,
                )[0, :, 0].float()
        np.testing.assert_array_equal(output, expected.float().numpy())

    def test_sparse_attention_preserves_full_width_flash_blocks(self) -> None:
        if not LIB.ck_attention_sparse_bf16_pytorch_gqa_available():
            self.skipTest("sparse BF16 CPU-flash provider requires AVX-512F")
        rows, query_heads, kv_heads, head_dim = 2, 8, 2, 256
        context, position, width = 1300, 1298, 8
        query = bf16(torch.randn(rows, query_heads, head_dim))
        key = bf16(torch.randn(kv_heads, context, head_dim))
        value = bf16(torch.randn(kv_heads, context, head_dim))
        selected = np.full((rows, width), -1.0, dtype=np.float32)
        selected[0, :5] = [0, 13, 510, 1027, 1298]
        selected[1, :6] = [0, 13, 510, 1027, 1298, 1299]
        output = np.zeros((rows, query_heads, head_dim), dtype=np.float32)
        LIB.attention_forward_sparse_token_major_gqa_bf16cache_pytorch_cpu_flash_contract(
            fptr(query.numpy().copy()), u16ptr(bf16_bits(key)), u16ptr(bf16_bits(value)),
            fptr(selected), fptr(output), fptr(np.zeros(width, dtype=np.float32)),
            rows, query_heads, kv_heads, head_dim, width, context, position,
        )

        expected = torch.zeros_like(query)
        groups = query_heads // kv_heads
        for row in range(rows):
            visible = position + row + 1
            indices = [int(item) for item in selected[row] if item >= 0]
            mask = torch.zeros((1, 1, 1, visible), dtype=torch.bool)
            mask[..., indices] = True
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                expected[row] = torch.nn.functional.scaled_dot_product_attention(
                    query[row].unsqueeze(0).unsqueeze(2).to(torch.bfloat16),
                    key[:, :visible].repeat_interleave(groups, 0).unsqueeze(0).to(torch.bfloat16),
                    value[:, :visible].repeat_interleave(groups, 0).unsqueeze(0).to(torch.bfloat16),
                    attn_mask=mask,
                )[0, :, 0].float()
        np.testing.assert_array_equal(output, expected.float().numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
