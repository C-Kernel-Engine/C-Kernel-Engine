#!/usr/bin/env python3
"""PyTorch parity tests for scalar Nemotron-H/Mamba2 reference kernels."""

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

LIB.mamba2_in_proj_split_f32.argtypes = [fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.mamba2_in_proj_split_f32.restype = None
LIB.mamba2_conv1d_decode_f32.argtypes = [fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.mamba2_conv1d_decode_f32.restype = None
LIB.mamba2_conv1d_f32_channel_range.argtypes = [
    fptr, fptr, fptr, fptr, fptr, fptr,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
LIB.mamba2_conv1d_f32_channel_range.restype = None
LIB.mamba2_dt_softplus_f32.argtypes = [fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
LIB.mamba2_dt_softplus_f32.restype = None
LIB.mamba2_selective_state_update_decode_f32.argtypes = [fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.mamba2_selective_state_update_decode_f32.restype = None
LIB.mamba2_selective_scan_f32.argtypes = [fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
LIB.mamba2_selective_scan_f32.restype = None
LIB.mamba2_selective_scan_f32_head_range.argtypes = [
    fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr, fptr,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
LIB.mamba2_selective_scan_f32_head_range.restype = None
LIB.mamba2_rmsnorm_gate_f32.argtypes = [fptr, fptr, fptr, fptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
LIB.mamba2_rmsnorm_gate_f32.restype = None


def _fptr(a: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return a.ctypes.data_as(fptr)


def _np(x: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(x.detach().cpu().numpy().astype(np.float32))


class TestMamba2Reference(unittest.TestCase):
    def test_in_proj_split_matches_torch(self) -> None:
        rows, d_mlp, inter, conv_dim, heads = 3, 5, 8, 14, 4
        torch.manual_seed(1)
        projected = torch.randn(rows, 2 * d_mlp + inter + conv_dim + heads, dtype=torch.float32)
        _, _, gate_ref, hidden_bc_ref, dt_ref = projected.split([d_mlp, d_mlp, inter, conv_dim, heads], dim=-1)
        projected_np = _np(projected)
        gate = np.empty((rows, inter), dtype=np.float32)
        hidden_bc = np.empty((rows, conv_dim), dtype=np.float32)
        dt = np.empty((rows, heads), dtype=np.float32)
        LIB.mamba2_in_proj_split_f32(_fptr(projected_np), _fptr(gate), _fptr(hidden_bc), _fptr(dt), rows, d_mlp, inter, conv_dim, heads)
        np.testing.assert_allclose(gate, _np(gate_ref), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(hidden_bc, _np(hidden_bc_ref), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(dt, _np(dt_ref), atol=0.0, rtol=0.0)

    def test_conv1d_decode_matches_torch(self) -> None:
        rows, conv_dim, kernel = 2, 11, 4
        torch.manual_seed(2)
        state = torch.randn(conv_dim, kernel, dtype=torch.float32) * 0.2
        x = torch.randn(rows, conv_dim, dtype=torch.float32) * 0.2
        weight = torch.randn(conv_dim, kernel, dtype=torch.float32) * 0.3
        bias = torch.randn(conv_dim, dtype=torch.float32) * 0.1
        state_ref = state.clone()
        conv_rows = []
        for row in range(rows):
            state_ref = torch.roll(state_ref, shifts=-1, dims=-1)
            state_ref[:, -1] = x[row]
            conv_rows.append(torch.nn.functional.silu((state_ref * weight).sum(dim=-1) + bias))
        conv_ref = torch.stack(conv_rows, dim=0)
        state_np, x_np, weight_np, bias_np = map(_np, (state, x, weight, bias))
        conv = np.empty((rows, conv_dim), dtype=np.float32)
        state_out = np.empty((conv_dim, kernel), dtype=np.float32)
        LIB.mamba2_conv1d_decode_f32(_fptr(state_np), _fptr(x_np), _fptr(weight_np), _fptr(bias_np), _fptr(conv), _fptr(state_out), rows, conv_dim, kernel)
        np.testing.assert_allclose(state_out, _np(state_ref), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(conv, _np(conv_ref), atol=1e-6, rtol=0.0)

    def test_conv1d_channel_ranges_are_byte_exact(self) -> None:
        rows, conv_dim, kernel = 9, 17, 4
        rng = np.random.default_rng(2026)
        state = rng.standard_normal((conv_dim, kernel), dtype=np.float32)
        x = rng.standard_normal((rows, conv_dim), dtype=np.float32)
        weight = rng.standard_normal((conv_dim, kernel), dtype=np.float32)
        bias = rng.standard_normal(conv_dim, dtype=np.float32)
        expected = np.empty((rows, conv_dim), dtype=np.float32)
        expected_state = np.empty((conv_dim, kernel), dtype=np.float32)
        actual = np.full((rows, conv_dim), np.nan, dtype=np.float32)
        actual_state = np.full((conv_dim, kernel), np.nan, dtype=np.float32)

        LIB.mamba2_conv1d_decode_f32(
            _fptr(state), _fptr(x), _fptr(weight), _fptr(bias),
            _fptr(expected), _fptr(expected_state), rows, conv_dim, kernel,
        )
        for begin, end in ((0, 3), (3, 11), (11, conv_dim)):
            LIB.mamba2_conv1d_f32_channel_range(
                _fptr(state), _fptr(x), _fptr(weight), _fptr(bias),
                _fptr(actual), _fptr(actual_state), rows, conv_dim, kernel,
                begin, end,
            )
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(actual_state.tobytes(), expected_state.tobytes())

    def test_dt_softplus_matches_torch(self) -> None:
        rows, heads = 4, 7
        torch.manual_seed(3)
        dt = torch.randn(rows, heads, dtype=torch.float32)
        bias = torch.randn(heads, dtype=torch.float32) * 0.2
        ref = torch.nn.functional.softplus(dt + bias).clamp(0.01, 2.0)
        dt_np, bias_np = map(_np, (dt, bias))
        out = np.empty((rows, heads), dtype=np.float32)
        LIB.mamba2_dt_softplus_f32(_fptr(dt_np), _fptr(bias_np), _fptr(out), rows, heads, ctypes.c_float(0.01), ctypes.c_float(2.0))
        np.testing.assert_allclose(out, _np(ref), atol=1e-6, rtol=0.0)

    def test_selective_state_update_decode_matches_torch(self) -> None:
        rows, heads, head_dim, state_dim, groups = 2, 6, 5, 4, 3
        torch.manual_seed(4)
        state = torch.randn(rows, heads, head_dim, state_dim, dtype=torch.float32) * 0.1
        x = torch.randn(rows, heads, head_dim, dtype=torch.float32) * 0.2
        dt = torch.rand(rows, heads, dtype=torch.float32) * 0.3 + 0.01
        a = -(torch.rand(heads, dtype=torch.float32) * 0.5 + 0.1)
        b = torch.randn(rows, groups, state_dim, dtype=torch.float32) * 0.2
        c = torch.randn(rows, groups, state_dim, dtype=torch.float32) * 0.2
        d = torch.randn(heads, dtype=torch.float32) * 0.05
        state_ref = torch.empty_like(state)
        y_ref = torch.empty_like(x)
        for r in range(rows):
            for h in range(heads):
                g = min(h // ((heads + groups - 1) // groups), groups - 1)
                decay = torch.exp(dt[r, h] * a[h])
                for hd in range(head_dim):
                    new_state = state[r, h, hd] * decay + dt[r, h] * b[r, g] * x[r, h, hd]
                    state_ref[r, h, hd] = new_state
                    y_ref[r, h, hd] = (new_state * c[r, g]).sum() + d[h] * x[r, h, hd]
        arrays = list(map(_np, (state, x, dt, a, b, c, d)))
        state_out = np.empty((rows, heads, head_dim, state_dim), dtype=np.float32)
        y = np.empty((rows, heads, head_dim), dtype=np.float32)
        LIB.mamba2_selective_state_update_decode_f32(*map(_fptr, arrays), _fptr(state_out), _fptr(y), rows, heads, head_dim, state_dim, groups)
        np.testing.assert_allclose(state_out, _np(state_ref), atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(y, _np(y_ref), atol=1e-6, rtol=0.0)


    def test_selective_state_update_decode_accepts_packed_xbc(self) -> None:
        rows, heads, head_dim, state_dim, groups = 2, 8, 3, 5, 2
        torch.manual_seed(44)
        state = torch.randn(rows, heads, head_dim, state_dim, dtype=torch.float32) * 0.1
        hidden = torch.randn(rows, heads, head_dim, dtype=torch.float32) * 0.2
        b = torch.randn(rows, groups, state_dim, dtype=torch.float32) * 0.2
        c = torch.randn(rows, groups, state_dim, dtype=torch.float32) * 0.2
        dt = torch.rand(rows, heads, dtype=torch.float32) * 0.3 + 0.01
        a = -(torch.rand(heads, dtype=torch.float32) * 0.5 + 0.1)
        d = torch.randn(heads, dtype=torch.float32) * 0.05
        packed = torch.cat([hidden.reshape(rows, -1), b.reshape(rows, -1), c.reshape(rows, -1)], dim=-1)

        state_ref = torch.empty_like(state)
        y_ref = torch.empty_like(hidden)
        for r in range(rows):
            for h in range(heads):
                g = min(h // ((heads + groups - 1) // groups), groups - 1)
                decay = torch.exp(dt[r, h] * a[h])
                for hd in range(head_dim):
                    new_state = state[r, h, hd] * decay + dt[r, h] * b[r, g] * hidden[r, h, hd]
                    state_ref[r, h, hd] = new_state
                    y_ref[r, h, hd] = (new_state * c[r, g]).sum() + d[h] * hidden[r, h, hd]

        state_np, packed_np, dt_np, a_np, d_np = map(_np, (state, packed, dt, a, d))
        state_out = np.empty((rows, heads, head_dim, state_dim), dtype=np.float32)
        y = np.empty((rows, heads, head_dim), dtype=np.float32)
        LIB.mamba2_selective_state_update_decode_f32(
            _fptr(state_np), _fptr(packed_np), _fptr(dt_np), _fptr(a_np),
            _fptr(packed_np), _fptr(packed_np), _fptr(d_np),
            _fptr(state_out), _fptr(y), rows, heads, head_dim, state_dim, groups,
        )
        np.testing.assert_allclose(state_out, _np(state_ref), atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(y, _np(y_ref), atol=1e-6, rtol=0.0)


    def test_selective_scan_matches_torch_sequence(self) -> None:
        batch, seq, heads, head_dim, state_dim, groups = 2, 5, 6, 4, 3, 3
        torch.manual_seed(6)
        state0 = torch.randn(batch, heads, head_dim, state_dim, dtype=torch.float32) * 0.1
        x = torch.randn(batch, seq, heads, head_dim, dtype=torch.float32) * 0.2
        dt = torch.rand(batch, seq, heads, dtype=torch.float32) * 0.3 + 0.01
        a = -(torch.rand(heads, dtype=torch.float32) * 0.5 + 0.1)
        b = torch.randn(batch, seq, groups, state_dim, dtype=torch.float32) * 0.2
        c = torch.randn(batch, seq, groups, state_dim, dtype=torch.float32) * 0.2
        d = torch.randn(heads, dtype=torch.float32) * 0.05

        state_ref = state0.clone()
        y_ref = torch.empty_like(x)
        for bs in range(batch):
            for t in range(seq):
                for h in range(heads):
                    g = min(h // ((heads + groups - 1) // groups), groups - 1)
                    decay = torch.exp(dt[bs, t, h] * a[h])
                    for hd in range(head_dim):
                        new_state = state_ref[bs, h, hd] * decay + dt[bs, t, h] * b[bs, t, g] * x[bs, t, h, hd]
                        state_ref[bs, h, hd] = new_state
                        y_ref[bs, t, h, hd] = (new_state * c[bs, t, g]).sum() + d[h] * x[bs, t, h, hd]

        arrays = list(map(_np, (state0, x, dt, a, b, c, d)))
        state_out = np.empty((batch, heads, head_dim, state_dim), dtype=np.float32)
        y = np.empty((batch, seq, heads, head_dim), dtype=np.float32)
        LIB.mamba2_selective_scan_f32(*map(_fptr, arrays), _fptr(state_out), _fptr(y), batch, seq, heads, head_dim, state_dim, groups)
        np.testing.assert_allclose(state_out, _np(state_ref), atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(y, _np(y_ref), atol=1e-6, rtol=0.0)

    def test_selective_scan_head_ranges_are_byte_exact(self) -> None:
        batch, seq, heads, head_dim, state_dim, groups = 1, 13, 9, 4, 3, 3
        rng = np.random.default_rng(3036)
        state = rng.standard_normal(
            (batch, heads, head_dim, state_dim), dtype=np.float32
        )
        x = rng.standard_normal(
            (batch, seq, heads, head_dim), dtype=np.float32
        )
        dt = np.abs(rng.standard_normal((batch, seq, heads), dtype=np.float32))
        a = -np.abs(rng.standard_normal(heads, dtype=np.float32))
        b = rng.standard_normal(
            (batch, seq, groups, state_dim), dtype=np.float32
        )
        c = rng.standard_normal(
            (batch, seq, groups, state_dim), dtype=np.float32
        )
        d = rng.standard_normal(heads, dtype=np.float32)
        expected_state = np.empty_like(state)
        expected_y = np.empty_like(x)
        actual_state = np.full_like(state, np.nan)
        actual_y = np.full_like(x, np.nan)
        common = tuple(map(_fptr, (state, x, dt, a, b, c, d)))

        LIB.mamba2_selective_scan_f32(
            *common, _fptr(expected_state), _fptr(expected_y),
            batch, seq, heads, head_dim, state_dim, groups,
        )
        for begin, end in ((0, 2), (2, 7), (7, heads)):
            LIB.mamba2_selective_scan_f32_head_range(
                *common, _fptr(actual_state), _fptr(actual_y),
                batch, seq, heads, head_dim, state_dim, groups, begin, end,
            )
        self.assertEqual(actual_state.tobytes(), expected_state.tobytes())
        self.assertEqual(actual_y.tobytes(), expected_y.tobytes())

    def test_selective_state_update_decode_matches_single_token_scan_mapping(self) -> None:
        # Nemotron-H prefill and decode must use the same repeating B/C
        # group map so full-prefix prefill and incremental decode are equivalent
        # for the same state, x, dt, A, B, C, and D inputs.
        batch, seq, heads, head_dim, state_dim, groups = 1, 1, 8, 2, 3, 2
        rng = np.random.default_rng(77)
        state0 = np.zeros((batch, heads, head_dim, state_dim), dtype=np.float32)
        x = np.ascontiguousarray((0.2 * rng.standard_normal((batch, seq, heads, head_dim))).astype(np.float32))
        dt = np.ascontiguousarray((0.01 + 0.3 * rng.random((batch, seq, heads))).astype(np.float32))
        a = np.ascontiguousarray((-(0.1 + 0.5 * rng.random(heads))).astype(np.float32))
        b = np.ascontiguousarray((0.2 * rng.standard_normal((batch, seq, groups, state_dim))).astype(np.float32))
        c = np.ascontiguousarray((0.2 * rng.standard_normal((batch, seq, groups, state_dim))).astype(np.float32))
        d = np.ascontiguousarray((0.05 * rng.standard_normal(heads)).astype(np.float32))
        scan_state = np.empty_like(state0)
        scan_y = np.empty_like(x)
        LIB.mamba2_selective_scan_f32(_fptr(state0), _fptr(x), _fptr(dt), _fptr(a), _fptr(b), _fptr(c), _fptr(d), _fptr(scan_state), _fptr(scan_y), batch, seq, heads, head_dim, state_dim, groups)

        decode_state = np.empty((seq, heads, head_dim, state_dim), dtype=np.float32)
        decode_y = np.empty((seq, heads, head_dim), dtype=np.float32)
        LIB.mamba2_selective_state_update_decode_f32(
            _fptr(state0.reshape(heads, head_dim, state_dim)),
            _fptr(x.reshape(seq, heads, head_dim)),
            _fptr(dt.reshape(seq, heads)),
            _fptr(a),
            _fptr(b.reshape(seq, groups, state_dim)),
            _fptr(c.reshape(seq, groups, state_dim)),
            _fptr(d),
            _fptr(decode_state),
            _fptr(decode_y),
            seq, heads, head_dim, state_dim, groups,
        )

        ref = np.empty_like(scan_y)
        for h in range(heads):
            g = min(h // ((heads + groups - 1) // groups), groups - 1)
            for hd in range(head_dim):
                new_state = dt[0, 0, h] * b[0, 0, g] * x[0, 0, h, hd]
                ref[0, 0, h, hd] = float(np.dot(new_state, c[0, 0, g]) + d[h] * x[0, 0, h, hd])
        np.testing.assert_allclose(scan_y, ref, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(decode_y.reshape(batch, seq, heads, head_dim), scan_y, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(decode_state.reshape(batch, heads, head_dim, state_dim), scan_state, atol=1e-6, rtol=0.0)

    def test_rmsnorm_gate_matches_torch_grouped_after_gate(self) -> None:
        rows, inner_dim, group_size = 3, 24, 6
        torch.manual_seed(5)
        x = torch.randn(rows, inner_dim, dtype=torch.float32) * 0.2
        gate = torch.randn(rows, inner_dim, dtype=torch.float32) * 0.2
        weight = torch.randn(inner_dim, dtype=torch.float32) * 0.2 + 1.0
        gated = x * torch.nn.functional.silu(gate)
        chunks = []
        for start in range(0, inner_dim, group_size):
            chunk = gated[:, start:start + group_size]
            inv = torch.rsqrt(chunk.square().mean(dim=-1, keepdim=True) + 1e-5)
            chunks.append(chunk * inv * weight[start:start + group_size])
        ref = torch.cat(chunks, dim=-1)
        x_np, gate_np, weight_np = map(_np, (x, gate, weight))
        out = np.empty((rows, inner_dim), dtype=np.float32)
        LIB.mamba2_rmsnorm_gate_f32(_fptr(x_np), _fptr(gate_np), _fptr(weight_np), _fptr(out), rows, inner_dim, group_size, ctypes.c_float(1e-5))
        np.testing.assert_allclose(out, _np(ref), atol=1e-6, rtol=0.0)


def _time_us(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1.0e6 / max(1, iterations)


def run_benchmark() -> None:
    rows, heads, head_dim, state_dim, groups = 4, 64, 64, 128, 8
    rng = np.random.default_rng(23)
    state = np.ascontiguousarray((0.02 * rng.standard_normal((rows, heads, head_dim, state_dim))).astype(np.float32))
    x = np.ascontiguousarray((0.02 * rng.standard_normal((rows, heads, head_dim))).astype(np.float32))
    dt = np.ascontiguousarray((0.01 + 0.1 * rng.random((rows, heads))).astype(np.float32))
    a = np.ascontiguousarray((-(0.1 + 0.5 * rng.random(heads))).astype(np.float32))
    b = np.ascontiguousarray((0.02 * rng.standard_normal((rows, groups, state_dim))).astype(np.float32))
    c = np.ascontiguousarray((0.02 * rng.standard_normal((rows, groups, state_dim))).astype(np.float32))
    d = np.ascontiguousarray((0.01 * rng.standard_normal(heads)).astype(np.float32))
    state_out = np.empty_like(state)
    y = np.empty_like(x)

    def ck_step() -> None:
        LIB.mamba2_selective_state_update_decode_f32(
            _fptr(state), _fptr(x), _fptr(dt), _fptr(a), _fptr(b), _fptr(c), _fptr(d),
            _fptr(state_out), _fptr(y), rows, heads, head_dim, state_dim, groups
        )

    ck_step()
    ck_us = _time_us(ck_step, 20)
    elems = rows * heads * head_dim * state_dim
    print("kernel                                  elems        ck_us")
    print(f"mamba2_selective_state_update_decode {elems:10d} {ck_us:10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args, remaining = ap.parse_known_args()
    if args.benchmark:
        run_benchmark()
    else:
        sys.argv = [sys.argv[0], *remaining]
        unittest.main(verbosity=2)
