#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "build" / "libckernel_engine.so"
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int32)


@unittest.skipUnless(LIBRARY.exists(), f"missing CKE library: {LIBRARY}")
class TestMoeSoftmaxTopkRouterPytorchBf16(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = ctypes.CDLL(str(LIBRARY))
        cls.lib.moe_softmax_topk_router_workspace_bytes.argtypes = [ctypes.c_int]
        cls.lib.moe_softmax_topk_router_workspace_bytes.restype = ctypes.c_size_t
        cls.lib.moe_softmax_topk_router_pytorch_bf16_workspace.argtypes = [
            F32P, I32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t,
        ]
        cls.lib.moe_softmax_topk_router_pytorch_bf16_workspace.restype = ctypes.c_int

    def test_matches_pytorch_bf16_router_boundary(self) -> None:
        logits = torch.tensor(
            [[0.05541992, -0.21679688, 0.09423828, 0.04028320,
              -0.10693359, 0.375, -0.13964844, 0.2578125]],
            dtype=torch.bfloat16,
        )
        probabilities = torch.softmax(logits, dtype=torch.float32, dim=-1)
        expected_weights, expected_indices = torch.topk(probabilities, 4, dim=-1)
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
        expected_weights = expected_weights.to(torch.bfloat16).float().numpy()

        logits_f32 = logits.float().numpy()
        indices = np.empty((1, 4), dtype=np.int32)
        weights = np.empty((1, 4), dtype=np.float32)
        workspace_bytes = self.lib.moe_softmax_topk_router_workspace_bytes(8)
        workspace = np.empty(workspace_bytes, dtype=np.uint8)
        status = self.lib.moe_softmax_topk_router_pytorch_bf16_workspace(
            logits_f32.ctypes.data_as(F32P),
            indices.ctypes.data_as(I32P),
            weights.ctypes.data_as(F32P),
            1, 8, 4, ctypes.c_float(1.0),
            workspace.ctypes.data_as(ctypes.c_void_p), workspace.nbytes,
        )
        self.assertEqual(status, 0)
        np.testing.assert_array_equal(indices, expected_indices.numpy().astype(np.int32))
        np.testing.assert_array_equal(weights, expected_weights)


if __name__ == "__main__":
    unittest.main(verbosity=2)
