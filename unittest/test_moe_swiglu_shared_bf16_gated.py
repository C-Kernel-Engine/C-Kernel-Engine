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

LIB.moe_swiglu_shared_forward_bf16_gated.argtypes = [
    F32P, F32P, U16P, U16P, U16P, U16P, F32P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]


def fptr(value: np.ndarray) -> F32P:
    return value.ctypes.data_as(F32P)


def u16ptr(value: np.ndarray) -> U16P:
    return value.ctypes.data_as(U16P)


def bf16_bits(value: torch.Tensor) -> np.ndarray:
    return value.to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy()


class TestMoeSwigluSharedBf16Gated(unittest.TestCase):
    def test_matches_bf16_graph_boundaries(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(67)
        rows, hidden, intermediate = 3, 16, 24
        x = torch.randn(rows, hidden).to(torch.bfloat16)
        routed = torch.randn(rows, hidden).to(torch.bfloat16)
        gate = (0.1 * torch.randn(intermediate, hidden)).to(torch.bfloat16)
        up = (0.1 * torch.randn(intermediate, hidden)).to(torch.bfloat16)
        down = (0.1 * torch.randn(hidden, intermediate)).to(torch.bfloat16)
        router = (0.1 * torch.randn(1, hidden)).to(torch.bfloat16)

        shared = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        expected = routed + torch.sigmoid(F.linear(x, router)) * shared

        output = np.zeros((rows, hidden), dtype=np.float32)
        LIB.moe_swiglu_shared_forward_bf16_gated(
            fptr(x.float().numpy().copy()),
            fptr(routed.float().numpy().copy()),
            u16ptr(bf16_bits(gate)),
            u16ptr(bf16_bits(up)),
            u16ptr(bf16_bits(down)),
            u16ptr(bf16_bits(router)),
            fptr(output),
            rows,
            hidden,
            intermediate,
        )
        np.testing.assert_array_equal(output, expected.float().numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
