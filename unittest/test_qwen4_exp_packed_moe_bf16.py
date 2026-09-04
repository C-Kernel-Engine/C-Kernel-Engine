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
I32P = ctypes.POINTER(ctypes.c_int32)
U16P = ctypes.POINTER(ctypes.c_uint16)

LIB.moe_swiglu_packed_expert_forward_bf16.argtypes = [
    F32P, I32P, F32P, U16P, U16P, F32P,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]


def fptr(value: np.ndarray) -> F32P:
    return value.ctypes.data_as(F32P)


def i32ptr(value: np.ndarray) -> I32P:
    return value.ctypes.data_as(I32P)


def u16ptr(value: np.ndarray) -> U16P:
    return value.ctypes.data_as(U16P)


def bf16_bits(value: torch.Tensor) -> np.ndarray:
    return value.to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy()


class TestQwen4ExpPackedMoeBf16(unittest.TestCase):
    def test_packed_gate_up_matches_split_pytorch_oracle(self) -> None:
        torch.manual_seed(53)
        rows, hidden, intermediate, experts, top_k = 3, 8, 12, 5, 2
        x = torch.randn(rows, hidden).to(torch.bfloat16)
        gate = torch.randn(experts, intermediate, hidden).to(torch.bfloat16)
        up = torch.randn(experts, intermediate, hidden).to(torch.bfloat16)
        packed = torch.cat([gate, up], dim=1)
        down = torch.randn(experts, hidden, intermediate).to(torch.bfloat16)
        indices = torch.tensor([[0, 2], [4, 1], [3, 0]], dtype=torch.int32)
        routes = torch.softmax(torch.randn(rows, top_k), dim=-1).to(torch.bfloat16)

        contributions = []
        for row in range(rows):
            row_contributions = []
            for slot in range(top_k):
                    expert = int(indices[row, slot])
                    activation = F.silu(F.linear(x[row], gate[expert])) * F.linear(
                        x[row], up[expert]
                    )
                    projected = F.linear(activation, down[expert])
                    row_contributions.append(routes[row, slot] * projected)
            contributions.append(torch.stack(row_contributions).sum(dim=0))
        expected = torch.stack(contributions).to(torch.bfloat16)

        output = np.zeros((rows, hidden), dtype=np.float32)
        LIB.moe_swiglu_packed_expert_forward_bf16(
            fptr(x.float().numpy().copy()), i32ptr(indices.numpy().copy()),
            fptr(routes.float().numpy().copy()), u16ptr(bf16_bits(packed)),
            u16ptr(bf16_bits(down)), fptr(output), rows, hidden,
            intermediate, experts, top_k,
        )
        np.testing.assert_array_equal(output, expected.float().numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
