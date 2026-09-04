#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LIB = ctypes.CDLL(str(ROOT / "build" / "libckernel_engine.so"))
F32P = ctypes.POINTER(ctypes.c_float)
U16P = ctypes.POINTER(ctypes.c_uint16)

LIB.hyper_stream_expand_bf16.argtypes = [F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.hyper_connection_mix_bf16.argtypes = [
    F32P, F32P, U16P, U16P, U16P, F32P, F32P, F32P, F32P, F32P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
]
LIB.hyper_stream_inject_bf16.argtypes = [F32P, F32P, F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.hyper_stream_inject_f32.argtypes = [F32P, F32P, F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def fptr(value: np.ndarray) -> F32P:
    return value.ctypes.data_as(F32P)


def u16ptr(value: np.ndarray) -> U16P:
    return value.ctypes.data_as(U16P)


def bf16_bits(value: torch.Tensor) -> np.ndarray:
    return value.to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy()


class TestHyperConnectionBf16(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_expand_mix_and_inject(self) -> None:
        rows, streams, hidden, dynamic = 3, 4, 16, 20
        hyper_dim = streams * hidden
        source = torch.randn(rows, hidden, dtype=torch.float32).to(torch.bfloat16)
        hyper = source.repeat(1, streams)
        norm_weight = (1.0 + 0.1 * torch.randn(hyper_dim)).float()
        down = (0.1 * torch.randn(dynamic, hyper_dim)).to(torch.bfloat16)
        up = (0.1 * torch.randn(hyper_dim, dynamic)).to(torch.bfloat16)
        inject = (0.1 * torch.randn(streams, hyper_dim)).to(torch.bfloat16)

        source_np = source.float().numpy().copy()
        expanded_np = np.zeros((rows, hyper_dim), dtype=np.float32)
        LIB.hyper_stream_expand_bf16(fptr(source_np), fptr(expanded_np), rows, streams, hidden)
        np.testing.assert_array_equal(expanded_np, hyper.float().numpy())

        grouped = hyper.view(rows, streams, hidden)
        variance = grouped.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = (
            grouped.float()
            * torch.rsqrt(variance + 1e-6)
            * norm_weight.view(streams, hidden)
        ).to(torch.bfloat16)
        flat = normalized.view(rows, hyper_dim)
        dynamic_ref = (F.linear(flat, down) / streams).to(torch.bfloat16)
        dynamic_ref = F.silu(dynamic_ref).to(torch.bfloat16)
        mix = torch.sigmoid(F.linear(dynamic_ref, up)).to(torch.bfloat16)
        mixed_ref = (flat.view(rows, streams, hidden) * mix.view(rows, streams, hidden)).mean(dim=1).to(torch.bfloat16)
        injection_ref = (2.0 * torch.sigmoid((F.linear(flat, inject) / streams).to(torch.bfloat16))).to(torch.bfloat16)

        mixed_np = np.zeros((rows, hidden), dtype=np.float32)
        injection_np = np.zeros((rows, streams), dtype=np.float32)
        normalized_np = np.zeros((rows, hyper_dim), dtype=np.float32)
        dynamic_np = np.zeros((rows, dynamic), dtype=np.float32)
        mix_np = np.zeros((rows, hyper_dim), dtype=np.float32)
        LIB.hyper_connection_mix_bf16(
            fptr(expanded_np), fptr(norm_weight.numpy().copy()), u16ptr(bf16_bits(down)),
            u16ptr(bf16_bits(up)), u16ptr(bf16_bits(inject)), fptr(mixed_np),
            fptr(injection_np), fptr(normalized_np), fptr(dynamic_np), fptr(mix_np),
            rows, streams, hidden, dynamic, ctypes.c_float(1e-6), 1,
        )
        np.testing.assert_array_equal(normalized_np, flat.float().numpy())
        np.testing.assert_allclose(mixed_np, mixed_ref.float().numpy(), atol=0.08, rtol=0.0)
        np.testing.assert_allclose(injection_np, injection_ref.float().numpy(), atol=0.02, rtol=0.0)

        block = torch.randn(rows, hidden).to(torch.bfloat16)
        output_ref = (hyper.view(rows, streams, hidden) + block[:, None, :] * injection_ref[:, :, None]).to(torch.bfloat16)
        output_np = np.zeros((rows, hyper_dim), dtype=np.float32)
        LIB.hyper_stream_inject_bf16(
            fptr(expanded_np), fptr(block.float().numpy().copy()), fptr(injection_np),
            fptr(output_np), rows, streams, hidden,
        )
        np.testing.assert_allclose(output_np, output_ref.view(rows, hyper_dim).float().numpy(), atol=0.02, rtol=0.0)

    def test_fp32_inject_preserves_the_unrounded_residual_formula(self) -> None:
        rows, streams, hidden = 2, 4, 7
        rng = np.random.default_rng(47)
        hyper = rng.standard_normal((rows, streams * hidden), dtype=np.float32)
        block = rng.standard_normal((rows, hidden), dtype=np.float32)
        injection = rng.standard_normal((rows, streams), dtype=np.float32)
        expected = (
            hyper.reshape(rows, streams, hidden)
            + block[:, None, :] * injection[:, :, None]
        ).reshape(rows, streams * hidden)
        output = np.zeros_like(hyper)

        LIB.hyper_stream_inject_f32(
            fptr(hyper), fptr(block), fptr(injection), fptr(output),
            rows, streams, hidden,
        )

        np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
