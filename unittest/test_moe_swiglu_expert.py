#!/usr/bin/env python3
"""PyTorch parity test for routed/shared SwiGLU MoE expert MLPs."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"[SKIP] torch not available: {exc}")
    sys.exit(0)

ROOT = Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "libckernel_engine.so"
if not LIB_PATH.exists():  # pragma: no cover
    print("[SKIP] libckernel_engine.so not found")
    sys.exit(0)

LIB = ctypes.CDLL(str(LIB_PATH))
fptr = ctypes.POINTER(ctypes.c_float)
iptr = ctypes.POINTER(ctypes.c_int)
u16ptr = ctypes.POINTER(ctypes.c_uint16)

LIB.moe_swiglu_expert_forward_f32.argtypes = [fptr, iptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_swiglu_expert_forward_f32.restype = None
LIB.moe_swiglu_expert_backward_f32.argtypes = [fptr, fptr, iptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_swiglu_expert_backward_f32.restype = None
LIB.moe_swiglu_shared_forward_f32.argtypes = [fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_swiglu_shared_forward_f32.restype = None
LIB.moe_swiglu_expert_forward_bf16.argtypes = [fptr, iptr, fptr, u16ptr, u16ptr, u16ptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_swiglu_expert_forward_bf16.restype = None
LIB.moe_swiglu_expert_forward_bf16_row_range.argtypes = [fptr, iptr, fptr, u16ptr, u16ptr, u16ptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_swiglu_expert_forward_bf16_row_range.restype = None
LIB.moe_swiglu_shared_forward_bf16.argtypes = [fptr, fptr, u16ptr, u16ptr, u16ptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_swiglu_shared_forward_bf16.restype = None
LIB.moe_swiglu_shared_forward_bf16_row_range.argtypes = [fptr, fptr, u16ptr, u16ptr, u16ptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_swiglu_shared_forward_bf16_row_range.restype = None
LIB.farskip_swiglu_shared_combine_bf16.argtypes = [
    fptr, fptr, fptr, u16ptr, u16ptr, u16ptr, fptr, fptr,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
LIB.farskip_swiglu_shared_combine_bf16.restype = None
LIB.farskip_swiglu_shared_combine_bf16_row_range.argtypes = [
    fptr, fptr, fptr, u16ptr, u16ptr, u16ptr, fptr, fptr,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
LIB.farskip_swiglu_shared_combine_bf16_row_range.restype = None
LIB.moe_swiglu_shared_backward_f32.argtypes = [fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.moe_swiglu_shared_backward_f32.restype = None


def _fptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return a.ctypes.data_as(fptr)


def _iptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_int):
    return a.ctypes.data_as(iptr)

def _u16ptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_uint16):
    return a.ctypes.data_as(u16ptr)


def _bf16_bits(a: np.ndarray) -> np.ndarray:
    values = torch.tensor(a, dtype=torch.float32).to(torch.bfloat16)
    return np.ascontiguousarray(values.view(torch.uint16).numpy())


def _bf16_float(a: np.ndarray) -> np.ndarray:
    bits = _bf16_bits(a)
    return np.ascontiguousarray((bits.astype(np.uint32) << 16).view(np.float32))


def torch_routed(hidden, indices, weights, gate, up, down):
    rows, hidden_dim = hidden.shape
    top_k = indices.shape[1]
    out = torch.zeros_like(hidden)
    for r in range(rows):
        for s in range(top_k):
            e = int(indices[r, s])
            g = torch.matmul(gate[e], hidden[r])
            u = torch.matmul(up[e], hidden[r])
            act = torch.nn.functional.silu(g) * u
            expert_out = torch.matmul(down[e], act)
            out[r] = out[r] + weights[r, s] * expert_out
    return out


def torch_shared(hidden, routed, gate, up, down):
    act = torch.nn.functional.silu(hidden @ gate.T) * (hidden @ up.T)
    out = act @ down.T
    return out + routed


class TestMoESwiGLUExpert(unittest.TestCase):
    def test_farskip_bf16_combine_matches_pytorch_two_stream_order(self) -> None:
        rows, hidden_dim, intermediate_dim = 3, 8, 7
        rng = np.random.default_rng(113)
        hidden_np = np.ascontiguousarray(
            (0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32)
        )
        routed_np = np.ascontiguousarray(
            (0.1 * rng.standard_normal((rows, hidden_dim))).astype(np.float32)
        )
        residual_np = np.ascontiguousarray(
            (0.3 * rng.standard_normal((rows, hidden_dim))).astype(np.float32)
        )
        gate_np = np.ascontiguousarray(
            (0.13 * rng.standard_normal((intermediate_dim, hidden_dim))).astype(np.float32)
        )
        up_np = np.ascontiguousarray(
            (0.11 * rng.standard_normal((intermediate_dim, hidden_dim))).astype(np.float32)
        )
        down_np = np.ascontiguousarray(
            (0.09 * rng.standard_normal((hidden_dim, intermediate_dim))).astype(np.float32)
        )
        gate_bits = _bf16_bits(gate_np)
        up_bits = _bf16_bits(up_np)
        down_bits = _bf16_bits(down_np)
        main = np.empty_like(hidden_np)
        routed_free = np.empty_like(hidden_np)

        LIB.farskip_swiglu_shared_combine_bf16(
            _fptr(hidden_np), _fptr(routed_np), _fptr(residual_np),
            _u16ptr(gate_bits), _u16ptr(up_bits), _u16ptr(down_bits),
            _fptr(main), _fptr(routed_free), rows, hidden_dim, intermediate_dim,
        )

        hidden = torch.tensor(hidden_np, dtype=torch.float32)
        gate = torch.tensor(_bf16_float(gate_np), dtype=torch.float32)
        up = torch.tensor(_bf16_float(up_np), dtype=torch.float32)
        down = torch.tensor(_bf16_float(down_np), dtype=torch.float32)
        shared = (
            torch.nn.functional.silu(hidden @ gate.T) * (hidden @ up.T)
        ) @ down.T
        mlp_output = torch.tensor(routed_np) + shared
        routed_free_ref = torch.tensor(residual_np) + shared
        main_ref = torch.tensor(residual_np) + mlp_output

        np.testing.assert_allclose(routed_free, routed_free_ref.numpy(), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(main, main_ref.numpy(), atol=2e-6, rtol=0.0)

    def test_bf16_weight_forward_matches_pytorch_bf16_values(self) -> None:
        rows, hidden_dim, intermediate_dim, n_experts, top_k = 3, 8, 6, 5, 2
        rng = np.random.default_rng(109)
        hidden_np = np.ascontiguousarray((0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))
        gate_np = np.ascontiguousarray((0.13 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32))
        up_np = np.ascontiguousarray((0.11 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32))
        down_np = np.ascontiguousarray((0.09 * rng.standard_normal((n_experts, hidden_dim, intermediate_dim))).astype(np.float32))
        shared_gate_np = np.ascontiguousarray(gate_np[0])
        shared_up_np = np.ascontiguousarray(up_np[0])
        shared_down_np = np.ascontiguousarray(down_np[0])
        idx_np = np.ascontiguousarray(np.array([[0, 2], [3, 1], [4, 0]], dtype=np.int32))
        weight_np = np.ascontiguousarray(np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=np.float32))
        routed_np = np.ascontiguousarray((0.1 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))

        gate_bits, up_bits, down_bits = _bf16_bits(gate_np), _bf16_bits(up_np), _bf16_bits(down_np)
        shared_gate_bits = _bf16_bits(shared_gate_np)
        shared_up_bits = _bf16_bits(shared_up_np)
        shared_down_bits = _bf16_bits(shared_down_np)

        routed_out = np.empty_like(hidden_np)
        shared_out = np.empty_like(hidden_np)
        LIB.moe_swiglu_expert_forward_bf16(
            _fptr(hidden_np), _iptr(idx_np), _fptr(weight_np),
            _u16ptr(gate_bits), _u16ptr(up_bits), _u16ptr(down_bits),
            _fptr(routed_out), rows, hidden_dim, intermediate_dim, n_experts, top_k,
        )
        LIB.moe_swiglu_shared_forward_bf16(
            _fptr(hidden_np), _fptr(routed_np),
            _u16ptr(shared_gate_bits), _u16ptr(shared_up_bits), _u16ptr(shared_down_bits),
            _fptr(shared_out), rows, hidden_dim, intermediate_dim,
        )

        hidden = torch.tensor(hidden_np, dtype=torch.float32)
        indices = torch.tensor(idx_np, dtype=torch.long)
        weights = torch.tensor(weight_np, dtype=torch.float32)
        routed_ref = torch_routed(
            hidden,
            indices,
            weights,
            torch.tensor(_bf16_float(gate_np)),
            torch.tensor(_bf16_float(up_np)),
            torch.tensor(_bf16_float(down_np)),
        )
        shared_ref = torch_shared(
            hidden,
            torch.tensor(routed_np),
            torch.tensor(_bf16_float(shared_gate_np)),
            torch.tensor(_bf16_float(shared_up_np)),
            torch.tensor(_bf16_float(shared_down_np)),
        )
        np.testing.assert_allclose(routed_out, routed_ref.numpy(), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(shared_out, shared_ref.numpy(), atol=2e-6, rtol=0.0)

    def test_bf16_row_ranges_are_byte_exact(self) -> None:
        rows, hidden_dim, intermediate_dim, n_experts, top_k = 7, 8, 6, 5, 2
        rng = np.random.default_rng(127)
        hidden = np.ascontiguousarray((0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))
        routed = np.ascontiguousarray((0.1 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))
        indices = np.ascontiguousarray(np.stack([
            rng.choice(n_experts, size=top_k, replace=False) for _ in range(rows)
        ]).astype(np.int32))
        raw_routes = rng.random((rows, top_k)).astype(np.float32)
        routes = np.ascontiguousarray(raw_routes / raw_routes.sum(axis=1, keepdims=True))
        gate = _bf16_bits(np.ascontiguousarray((0.13 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32)))
        up = _bf16_bits(np.ascontiguousarray((0.11 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32)))
        down = _bf16_bits(np.ascontiguousarray((0.09 * rng.standard_normal((n_experts, hidden_dim, intermediate_dim))).astype(np.float32)))

        expert_ref = np.empty_like(hidden)
        expert_split = np.empty_like(hidden)
        LIB.moe_swiglu_expert_forward_bf16(
            _fptr(hidden), _iptr(indices), _fptr(routes), _u16ptr(gate),
            _u16ptr(up), _u16ptr(down), _fptr(expert_ref), rows, hidden_dim,
            intermediate_dim, n_experts, top_k,
        )
        for begin, end in ((0, 2), (2, 5), (5, rows)):
            LIB.moe_swiglu_expert_forward_bf16_row_range(
                _fptr(hidden), _iptr(indices), _fptr(routes), _u16ptr(gate),
                _u16ptr(up), _u16ptr(down), _fptr(expert_split), rows,
                hidden_dim, intermediate_dim, n_experts, top_k, begin, end,
            )
        self.assertEqual(expert_ref.tobytes(), expert_split.tobytes())

        shared_gate, shared_up, shared_down = gate[0], up[0], down[0]
        shared_ref = np.empty_like(hidden)
        shared_split = np.empty_like(hidden)
        LIB.moe_swiglu_shared_forward_bf16(
            _fptr(hidden), _fptr(routed), _u16ptr(shared_gate),
            _u16ptr(shared_up), _u16ptr(shared_down), _fptr(shared_ref),
            rows, hidden_dim, intermediate_dim,
        )
        for begin, end in ((0, 3), (3, 6), (6, rows)):
            LIB.moe_swiglu_shared_forward_bf16_row_range(
                _fptr(hidden), _fptr(routed), _u16ptr(shared_gate),
                _u16ptr(shared_up), _u16ptr(shared_down), _fptr(shared_split),
                rows, hidden_dim, intermediate_dim, begin, end,
            )
        self.assertEqual(shared_ref.tobytes(), shared_split.tobytes())

        residual = np.ascontiguousarray(
            (0.3 * rng.standard_normal((rows, hidden_dim))).astype(np.float32)
        )
        farskip_main_ref = np.empty_like(hidden)
        farskip_free_ref = np.empty_like(hidden)
        farskip_main_split = np.empty_like(hidden)
        farskip_free_split = np.empty_like(hidden)
        LIB.farskip_swiglu_shared_combine_bf16(
            _fptr(hidden), _fptr(routed), _fptr(residual),
            _u16ptr(shared_gate), _u16ptr(shared_up), _u16ptr(shared_down),
            _fptr(farskip_main_ref), _fptr(farskip_free_ref), rows,
            hidden_dim, intermediate_dim,
        )
        for begin, end in ((0, 1), (1, 4), (4, rows)):
            LIB.farskip_swiglu_shared_combine_bf16_row_range(
                _fptr(hidden), _fptr(routed), _fptr(residual),
                _u16ptr(shared_gate), _u16ptr(shared_up),
                _u16ptr(shared_down), _fptr(farskip_main_split),
                _fptr(farskip_free_split), rows, hidden_dim,
                intermediate_dim, begin, end,
            )
        self.assertEqual(farskip_main_ref.tobytes(), farskip_main_split.tobytes())
        self.assertEqual(farskip_free_ref.tobytes(), farskip_free_split.tobytes())

    def test_routed_forward_backward(self) -> None:
        rows, hidden_dim, intermediate_dim, n_experts, top_k = 4, 9, 7, 6, 3
        rng = np.random.default_rng(101)
        hidden_np = np.ascontiguousarray((0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))
        gate_np = np.ascontiguousarray((0.13 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32))
        up_np = np.ascontiguousarray((0.11 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32))
        down_np = np.ascontiguousarray((0.09 * rng.standard_normal((n_experts, hidden_dim, intermediate_dim))).astype(np.float32))
        idx_np = np.empty((rows, top_k), dtype=np.int32)
        for r in range(rows):
            idx_np[r] = rng.choice(n_experts, size=top_k, replace=False).astype(np.int32)
        w_raw = rng.random((rows, top_k)).astype(np.float32)
        weights_np = np.ascontiguousarray(w_raw / w_raw.sum(axis=1, keepdims=True))
        d_out_np = np.ascontiguousarray((0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))

        ck_out = np.empty_like(hidden_np)
        LIB.moe_swiglu_expert_forward_f32(_fptr(hidden_np), _iptr(idx_np), _fptr(weights_np), _fptr(gate_np), _fptr(up_np), _fptr(down_np), _fptr(ck_out), rows, hidden_dim, intermediate_dim, n_experts, top_k)

        ck_dh = np.empty_like(hidden_np)
        ck_dw = np.empty_like(weights_np)
        ck_dg = np.empty_like(gate_np)
        ck_du = np.empty_like(up_np)
        ck_dd = np.empty_like(down_np)
        LIB.moe_swiglu_expert_backward_f32(_fptr(d_out_np), _fptr(hidden_np), _iptr(idx_np), _fptr(weights_np), _fptr(gate_np), _fptr(up_np), _fptr(down_np), _fptr(ck_dh), _fptr(ck_dw), _fptr(ck_dg), _fptr(ck_du), _fptr(ck_dd), rows, hidden_dim, intermediate_dim, n_experts, top_k)

        hidden = torch.tensor(hidden_np, dtype=torch.float32, requires_grad=True)
        gate = torch.tensor(gate_np, dtype=torch.float32, requires_grad=True)
        up = torch.tensor(up_np, dtype=torch.float32, requires_grad=True)
        down = torch.tensor(down_np, dtype=torch.float32, requires_grad=True)
        weights = torch.tensor(weights_np, dtype=torch.float32, requires_grad=True)
        indices = torch.tensor(idx_np, dtype=torch.long)
        ref = torch_routed(hidden, indices, weights, gate, up, down)
        ref.backward(torch.tensor(d_out_np, dtype=torch.float32))

        np.testing.assert_allclose(ck_out, ref.detach().numpy(), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dh, hidden.grad.detach().numpy(), atol=3e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dw, weights.grad.detach().numpy(), atol=3e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dg, gate.grad.detach().numpy(), atol=3e-6, rtol=0.0)
        np.testing.assert_allclose(ck_du, up.grad.detach().numpy(), atol=3e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dd, down.grad.detach().numpy(), atol=3e-6, rtol=0.0)

    def test_shared_forward_backward(self) -> None:
        rows, hidden_dim, intermediate_dim = 5, 8, 11
        rng = np.random.default_rng(103)
        hidden_np = np.ascontiguousarray((0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))
        routed_np = np.ascontiguousarray((0.1 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))
        gate_np = np.ascontiguousarray((0.13 * rng.standard_normal((intermediate_dim, hidden_dim))).astype(np.float32))
        up_np = np.ascontiguousarray((0.11 * rng.standard_normal((intermediate_dim, hidden_dim))).astype(np.float32))
        down_np = np.ascontiguousarray((0.09 * rng.standard_normal((hidden_dim, intermediate_dim))).astype(np.float32))
        d_out_np = np.ascontiguousarray((0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))

        ck_out = np.empty_like(hidden_np)
        LIB.moe_swiglu_shared_forward_f32(_fptr(hidden_np), _fptr(routed_np), _fptr(gate_np), _fptr(up_np), _fptr(down_np), _fptr(ck_out), rows, hidden_dim, intermediate_dim)

        ck_dh = np.empty_like(hidden_np)
        ck_dr = np.empty_like(routed_np)
        ck_dg = np.empty_like(gate_np)
        ck_du = np.empty_like(up_np)
        ck_dd = np.empty_like(down_np)
        LIB.moe_swiglu_shared_backward_f32(_fptr(d_out_np), _fptr(hidden_np), _fptr(gate_np), _fptr(up_np), _fptr(down_np), _fptr(ck_dh), _fptr(ck_dr), _fptr(ck_dg), _fptr(ck_du), _fptr(ck_dd), rows, hidden_dim, intermediate_dim)

        hidden = torch.tensor(hidden_np, dtype=torch.float32, requires_grad=True)
        routed = torch.tensor(routed_np, dtype=torch.float32, requires_grad=True)
        gate = torch.tensor(gate_np, dtype=torch.float32, requires_grad=True)
        up = torch.tensor(up_np, dtype=torch.float32, requires_grad=True)
        down = torch.tensor(down_np, dtype=torch.float32, requires_grad=True)
        ref = torch_shared(hidden, routed, gate, up, down)
        ref.backward(torch.tensor(d_out_np, dtype=torch.float32))

        np.testing.assert_allclose(ck_out, ref.detach().numpy(), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dh, hidden.grad.detach().numpy(), atol=3e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dr, routed.grad.detach().numpy(), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(ck_dg, gate.grad.detach().numpy(), atol=3e-6, rtol=0.0)
        np.testing.assert_allclose(ck_du, up.grad.detach().numpy(), atol=3e-6, rtol=0.0)
        np.testing.assert_allclose(ck_dd, down.grad.detach().numpy(), atol=3e-6, rtol=0.0)


def _time_us(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1.0e6 / max(1, iterations)


def run_benchmark() -> None:
    rows, hidden_dim, intermediate_dim, n_experts, top_k = 8, 64, 48, 16, 4
    rng = np.random.default_rng(107)
    hidden = np.ascontiguousarray((0.2 * rng.standard_normal((rows, hidden_dim))).astype(np.float32))
    gate = np.ascontiguousarray((0.13 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32))
    up = np.ascontiguousarray((0.11 * rng.standard_normal((n_experts, intermediate_dim, hidden_dim))).astype(np.float32))
    down = np.ascontiguousarray((0.09 * rng.standard_normal((n_experts, hidden_dim, intermediate_dim))).astype(np.float32))
    idx = np.empty((rows, top_k), dtype=np.int32)
    for r in range(rows):
        idx[r] = rng.choice(n_experts, size=top_k, replace=False).astype(np.int32)
    w = rng.random((rows, top_k)).astype(np.float32)
    weights = np.ascontiguousarray(w / w.sum(axis=1, keepdims=True))
    out = np.empty_like(hidden)

    th = torch.tensor(hidden, dtype=torch.float32)
    tg = torch.tensor(gate, dtype=torch.float32)
    tu = torch.tensor(up, dtype=torch.float32)
    td = torch.tensor(down, dtype=torch.float32)
    tw = torch.tensor(weights, dtype=torch.float32)
    ti = torch.tensor(idx, dtype=torch.long)

    def ck_step() -> None:
        LIB.moe_swiglu_expert_forward_f32(_fptr(hidden), _iptr(idx), _fptr(weights), _fptr(gate), _fptr(up), _fptr(down), _fptr(out), rows, hidden_dim, intermediate_dim, n_experts, top_k)

    def torch_step() -> None:
        torch_routed(th, ti, tw, tg, tu, td)

    ck_step(); torch_step()
    torch_us = _time_us(torch_step, 100)
    ck_us = _time_us(ck_step, 100)
    print("kernel                      pytorch_us      ck_us       speedup")
    print(f"moe_swiglu_expert_forward {torch_us:12.3f} {ck_us:10.3f} {torch_us / max(ck_us, 1e-12):8.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args, remaining = ap.parse_known_args()
    if args.benchmark:
        run_benchmark()
    else:
        sys.argv = [sys.argv[0], *remaining]
        unittest.main(verbosity=2)
