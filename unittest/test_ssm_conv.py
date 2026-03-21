#!/usr/bin/env python3
"""
PyTorch parity test for the qwen3next/Qwen3.5 SSM causal depthwise convolution.

This covers both:
- forward parity: out
- backward parity: d_conv_x, d_kernel

Run directly:
    python3 unittest/test_ssm_conv.py
"""

from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - dependency skip
    print(f"[SKIP] torch not available: {exc}")
    sys.exit(0)


ROOT = Path(__file__).resolve().parents[1]
LIB_CANDIDATES = [
    ROOT / "build" / "libckernel_engine.so",
    ROOT / "libckernel_engine.so",
]


def _load_lib() -> ctypes.CDLL | None:
    for path in LIB_CANDIDATES:
        if path.exists():
            lib = ctypes.CDLL(str(path))
            fwd = lib.ssm_conv1d_forward
            fwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # conv_x
                ctypes.POINTER(ctypes.c_float),  # kernel
                ctypes.POINTER(ctypes.c_float),  # out
                ctypes.c_int,                    # kernel_size
                ctypes.c_int,                    # num_channels
                ctypes.c_int,                    # num_tokens
                ctypes.c_int,                    # num_seqs
            ]
            fwd.restype = None

            bwd = lib.ssm_conv1d_backward
            bwd.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # d_out
                ctypes.POINTER(ctypes.c_float),  # conv_x
                ctypes.POINTER(ctypes.c_float),  # kernel
                ctypes.POINTER(ctypes.c_float),  # d_conv_x
                ctypes.POINTER(ctypes.c_float),  # d_kernel
                ctypes.c_int,                    # kernel_size
                ctypes.c_int,                    # num_channels
                ctypes.c_int,                    # num_tokens
                ctypes.c_int,                    # num_seqs
            ]
            bwd.restype = None
            return lib
    return None


LIB = _load_lib()
if LIB is None:  # pragma: no cover - dependency skip
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)


def _as_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class TestSSMConvParity(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.atol_forward = 2e-5
        self.atol_backward = 5e-5

    def _run_case(self, kernel_size: int, num_channels: int, num_tokens: int, num_seqs: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        seq_width = kernel_size - 1 + num_tokens

        conv_x = (0.25 * rng.standard_normal((num_seqs, num_channels, seq_width))).astype(np.float32)
        kernel = (0.20 * rng.standard_normal((num_channels, kernel_size))).astype(np.float32)
        d_out = (0.30 * rng.standard_normal((num_seqs, num_tokens, num_channels))).astype(np.float32)

        ck_out = np.zeros((num_seqs, num_tokens, num_channels), dtype=np.float32)
        LIB.ssm_conv1d_forward(
            _as_ptr(conv_x),
            _as_ptr(kernel),
            _as_ptr(ck_out),
            kernel_size,
            num_channels,
            num_tokens,
            num_seqs,
        )

        ck_d_conv_x = np.zeros_like(conv_x)
        ck_d_kernel = np.zeros_like(kernel)
        LIB.ssm_conv1d_backward(
            _as_ptr(d_out),
            _as_ptr(conv_x),
            _as_ptr(kernel),
            _as_ptr(ck_d_conv_x),
            _as_ptr(ck_d_kernel),
            kernel_size,
            num_channels,
            num_tokens,
            num_seqs,
        )

        t_conv_x = torch.tensor(conv_x, dtype=torch.float32, requires_grad=True)
        t_kernel = torch.tensor(kernel[:, None, :], dtype=torch.float32, requires_grad=True)
        t_d_out = torch.tensor(np.transpose(d_out, (0, 2, 1)), dtype=torch.float32)

        t_out = F.conv1d(t_conv_x, t_kernel, bias=None, stride=1, padding=0, groups=num_channels)
        t_out.backward(t_d_out)

        torch_out = np.transpose(t_out.detach().numpy(), (0, 2, 1))
        torch_d_conv_x = t_conv_x.grad.detach().numpy()
        torch_d_kernel = t_kernel.grad.detach().numpy()[:, 0, :]

        np.testing.assert_allclose(ck_out, torch_out, atol=self.atol_forward, rtol=0.0)
        np.testing.assert_allclose(ck_d_conv_x, torch_d_conv_x, atol=self.atol_backward, rtol=0.0)
        np.testing.assert_allclose(ck_d_kernel, torch_d_kernel, atol=self.atol_backward, rtol=0.0)

    def test_small_case(self) -> None:
        self._run_case(kernel_size=4, num_channels=8, num_tokens=5, num_seqs=2, seed=7)

    def test_medium_case(self) -> None:
        self._run_case(kernel_size=4, num_channels=32, num_tokens=9, num_seqs=3, seed=11)

    def test_wider_case(self) -> None:
        self._run_case(kernel_size=4, num_channels=48, num_tokens=13, num_seqs=2, seed=19)


if __name__ == "__main__":
    unittest.main(verbosity=2)
