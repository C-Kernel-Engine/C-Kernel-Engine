"""
LEGACY DeltaNet parity test.

This file reflects the older contract where the DeltaNet kernel performed its
own q/k normalization internally. The current qwen35/qwen3next path normalizes
q/k explicitly before the DeltaNet kernel, so the authoritative test lives at:

    tests/test_deltanet.py

Keep this file only as a historical reference while older notes/docs still
point at it; do not use it as the active parity gate.
"""
import argparse
import ctypes
import math

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import max_diff, numpy_to_ptr


torch.set_num_threads(1)

lib = load_lib("libckernel_engine.so")

lib.gated_deltanet_autoregressive_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
lib.gated_deltanet_autoregressive_forward.restype = None

lib.gated_deltanet_autoregressive_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_out
    ctypes.POINTER(ctypes.c_float),  # d_state_out
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # g
    ctypes.POINTER(ctypes.c_float),  # beta
    ctypes.POINTER(ctypes.c_float),  # state_in
    ctypes.POINTER(ctypes.c_float),  # state_out
    ctypes.POINTER(ctypes.c_float),  # d_q
    ctypes.POINTER(ctypes.c_float),  # d_k
    ctypes.POINTER(ctypes.c_float),  # d_v
    ctypes.POINTER(ctypes.c_float),  # d_g
    ctypes.POINTER(ctypes.c_float),  # d_beta
    ctypes.POINTER(ctypes.c_float),  # d_state_in
    ctypes.c_int, ctypes.c_int, ctypes.c_float,
]
lib.gated_deltanet_autoregressive_backward.restype = None


def deltanet_torch(q, k, v, g, beta, state_in, eps):
    dim = q.shape[-1]
    q_hat = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + eps) / math.sqrt(dim)
    k_hat = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + eps)
    beta_s = torch.sigmoid(beta).unsqueeze(-1)
    gate = torch.exp(g).view(-1, 1, 1)

    state_decay = state_in * gate
    kv_mem = torch.einsum("hij,hi->hj", state_decay, k_hat)
    delta = (v - kv_mem) * beta_s
    state_out = state_decay + torch.einsum("hi,hj->hij", k_hat, delta)
    out = torch.einsum("hij,hi->hj", state_out, q_hat)
    return state_out, out


def print_result(name, diff, tol):
    status = "PASS" if diff <= tol else "FAIL"
    print(f"{name:<22} max_diff={diff:.8e} tol={tol:.1e} [{status}]")
    return diff <= tol


def run_case(num_heads, state_dim, eps, seed):
    rng = np.random.default_rng(seed)

    q_np = rng.standard_normal((num_heads, state_dim), dtype=np.float32)
    k_np = rng.standard_normal((num_heads, state_dim), dtype=np.float32)
    v_np = rng.standard_normal((num_heads, state_dim), dtype=np.float32)
    g_np = (0.25 * rng.standard_normal(num_heads)).astype(np.float32)
    beta_np = rng.standard_normal(num_heads, dtype=np.float32)
    state_in_np = (0.25 * rng.standard_normal((num_heads, state_dim, state_dim))).astype(np.float32)
    d_out_np = rng.standard_normal((num_heads, state_dim), dtype=np.float32)
    d_state_out_np = (0.25 * rng.standard_normal((num_heads, state_dim, state_dim))).astype(np.float32)

    state_out_np = np.zeros_like(state_in_np)
    out_np = np.zeros_like(v_np)
    d_q_np = np.zeros_like(q_np)
    d_k_np = np.zeros_like(k_np)
    d_v_np = np.zeros_like(v_np)
    d_g_np = np.zeros_like(g_np)
    d_beta_np = np.zeros_like(beta_np)
    d_state_in_np = np.zeros_like(state_in_np)

    lib.gated_deltanet_autoregressive_forward(
        numpy_to_ptr(q_np),
        numpy_to_ptr(k_np),
        numpy_to_ptr(v_np),
        numpy_to_ptr(g_np),
        numpy_to_ptr(beta_np),
        numpy_to_ptr(state_in_np),
        numpy_to_ptr(state_out_np),
        numpy_to_ptr(out_np),
        ctypes.c_int(num_heads),
        ctypes.c_int(state_dim),
        ctypes.c_float(eps),
    )

    lib.gated_deltanet_autoregressive_backward(
        numpy_to_ptr(d_out_np),
        numpy_to_ptr(d_state_out_np),
        numpy_to_ptr(q_np),
        numpy_to_ptr(k_np),
        numpy_to_ptr(v_np),
        numpy_to_ptr(g_np),
        numpy_to_ptr(beta_np),
        numpy_to_ptr(state_in_np),
        numpy_to_ptr(state_out_np),
        numpy_to_ptr(d_q_np),
        numpy_to_ptr(d_k_np),
        numpy_to_ptr(d_v_np),
        numpy_to_ptr(d_g_np),
        numpy_to_ptr(d_beta_np),
        numpy_to_ptr(d_state_in_np),
        ctypes.c_int(num_heads),
        ctypes.c_int(state_dim),
        ctypes.c_float(eps),
    )

    q_t = torch.from_numpy(q_np.copy()).requires_grad_(True)
    k_t = torch.from_numpy(k_np.copy()).requires_grad_(True)
    v_t = torch.from_numpy(v_np.copy()).requires_grad_(True)
    g_t = torch.from_numpy(g_np.copy()).requires_grad_(True)
    beta_t = torch.from_numpy(beta_np.copy()).requires_grad_(True)
    state_in_t = torch.from_numpy(state_in_np.copy()).requires_grad_(True)
    d_out_t = torch.from_numpy(d_out_np.copy())
    d_state_out_t = torch.from_numpy(d_state_out_np.copy())

    state_out_t, out_t = deltanet_torch(q_t, k_t, v_t, g_t, beta_t, state_in_t, eps)
    loss = (out_t * d_out_t).sum() + (state_out_t * d_state_out_t).sum()
    loss.backward()

    checks = [
        ("forward_out", max_diff(torch.from_numpy(out_np.copy()), out_t.detach()), 1e-6),
        ("forward_state", max_diff(torch.from_numpy(state_out_np.copy()), state_out_t.detach()), 1e-6),
        ("d_q", max_diff(torch.from_numpy(d_q_np.copy()), q_t.grad), 1e-5),
        ("d_k", max_diff(torch.from_numpy(d_k_np.copy()), k_t.grad), 1e-5),
        ("d_v", max_diff(torch.from_numpy(d_v_np.copy()), v_t.grad), 1e-6),
        ("d_g", max_diff(torch.from_numpy(d_g_np.copy()), g_t.grad), 1e-5),
        ("d_beta", max_diff(torch.from_numpy(d_beta_np.copy()), beta_t.grad), 1e-6),
        ("d_state_in", max_diff(torch.from_numpy(d_state_in_np.copy()), state_in_t.grad), 1e-5),
    ]

    passed = True
    print(f"\nCase H={num_heads}, D={state_dim}")
    for name, diff, tol in checks:
        passed &= print_result(name, diff, tol)
    return passed


def main():
    parser = argparse.ArgumentParser(description="Gated DeltaNet PyTorch parity test")
    parser.add_argument("--quick", action="store_true", help="Run a smaller config set")
    parser.add_argument("--eps", type=float, default=1e-6, help="Normalization epsilon")
    args = parser.parse_args()

    configs = [(4, 16, 7), (8, 32, 13)]
    if not args.quick:
        configs.append((16, 64, 29))

    all_passed = True
    for num_heads, state_dim, seed in configs:
        all_passed &= run_case(num_heads, state_dim, args.eps, seed)

    if not all_passed:
        raise SystemExit(1)

    print("\nAll DeltaNet forward/backward parity checks passed.")


if __name__ == "__main__":
    main()
