"""
QK Norm kernel unit tests with PyTorch parity and timing.

Covers:
- qk_norm_forward parity + speed
- qk_norm_backward parity + speed
- wrapper consistency vs direct rmsnorm_forward calls

After changes:
  python unittest/test_qk_norm.py
  python unittest/test_qk_norm.py --quick
"""

import argparse
import ctypes
import sys
from typing import Dict, List

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, print_system_info, time_function
)


lib = load_lib("libckernel_engine.so")


lib.qk_norm_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # q_gamma
    ctypes.POINTER(ctypes.c_float),  # k_gamma
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_float,                  # eps
]
lib.qk_norm_forward.restype = None

lib.qk_norm_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_q_out
    ctypes.POINTER(ctypes.c_float),  # d_k_out
    ctypes.POINTER(ctypes.c_float),  # q_in
    ctypes.POINTER(ctypes.c_float),  # k_in
    ctypes.POINTER(ctypes.c_float),  # q_gamma
    ctypes.POINTER(ctypes.c_float),  # k_gamma
    ctypes.POINTER(ctypes.c_float),  # d_q_in
    ctypes.POINTER(ctypes.c_float),  # d_k_in
    ctypes.POINTER(ctypes.c_float),  # d_q_gamma
    ctypes.POINTER(ctypes.c_float),  # d_k_gamma
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_float,                  # eps
]
lib.qk_norm_backward.restype = None

lib.rmsnorm_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.POINTER(ctypes.c_float),  # rstd_cache
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # d_model
    ctypes.c_int,                    # aligned_embed_dim
    ctypes.c_float,                  # eps
]
lib.rmsnorm_forward.restype = None


def rmsnorm_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    return x * rstd * gamma


def qk_norm_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    eps: float,
):
    q_normed = rmsnorm_torch(q, q_gamma, eps)
    k_normed = rmsnorm_torch(k, k_gamma, eps)
    return q_normed, k_normed


def run_qk_norm_test(
    num_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    head_dim: int,
    eps: float = 1e-6,
    warmup: int = 10,
    iterations: int = 200,
    label: str = "",
):
    if not label:
        label = f"FWD H={num_heads},KV={num_kv_heads},T={num_tokens},D={head_dim}"

    np.random.seed(42)
    q_np = np.random.randn(num_heads * num_tokens * head_dim).astype(np.float32)
    k_np = np.random.randn(num_kv_heads * num_tokens * head_dim).astype(np.float32)
    q_gamma_np = np.random.randn(head_dim).astype(np.float32)
    k_gamma_np = np.random.randn(head_dim).astype(np.float32)

    q_c = q_np.copy()
    k_c = k_np.copy()
    q_ptr = numpy_to_ptr(q_c)
    k_ptr = numpy_to_ptr(k_c)
    q_gamma_ptr = numpy_to_ptr(q_gamma_np)
    k_gamma_ptr = numpy_to_ptr(k_gamma_np)

    q_torch = torch.from_numpy(q_np.copy()).reshape(num_heads, num_tokens, head_dim)
    k_torch = torch.from_numpy(k_np.copy()).reshape(num_kv_heads, num_tokens, head_dim)
    q_gamma_torch = torch.from_numpy(q_gamma_np.copy())
    k_gamma_torch = torch.from_numpy(k_gamma_np.copy())
    q_ref, k_ref = qk_norm_torch(q_torch, k_torch, q_gamma_torch, k_gamma_torch, eps)

    lib.qk_norm_forward(
        q_ptr, k_ptr, q_gamma_ptr, k_gamma_ptr,
        ctypes.c_int(num_heads), ctypes.c_int(num_kv_heads),
        ctypes.c_int(num_tokens), ctypes.c_int(head_dim), ctypes.c_float(eps)
    )

    q_out = torch.from_numpy(q_c.copy())
    k_out = torch.from_numpy(k_c.copy())
    q_diff = max_diff(q_out, q_ref.reshape(-1))
    k_diff = max_diff(k_out, k_ref.reshape(-1))

    def pytorch_qk_norm():
        return qk_norm_torch(q_torch, k_torch, q_gamma_torch, k_gamma_torch, eps)

    def c_qk_norm():
        np.copyto(q_c, q_np)
        np.copyto(k_c, k_np)
        lib.qk_norm_forward(
            q_ptr, k_ptr, q_gamma_ptr, k_gamma_ptr,
            ctypes.c_int(num_heads), ctypes.c_int(num_kv_heads),
            ctypes.c_int(num_tokens), ctypes.c_int(head_dim), ctypes.c_float(eps)
        )

    pt_time = time_function(pytorch_qk_norm, warmup=warmup, iterations=iterations, name="PyTorch FWD")
    c_time = time_function(c_qk_norm, warmup=warmup, iterations=iterations, name="C FWD")

    combined_diff = max(q_diff, k_diff)
    tolerance = 1e-5
    return TestResult(
        name=label,
        passed=combined_diff <= tolerance,
        max_diff=combined_diff,
        tolerance=tolerance,
        pytorch_time=pt_time,
        kernel_time=c_time,
    ), {"q": q_diff, "k": k_diff}


def run_qk_norm_backward_test(
    num_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    head_dim: int,
    eps: float = 1e-6,
    warmup: int = 10,
    iterations: int = 100,
    label: str = "",
):
    if not label:
        label = f"BWD H={num_heads},KV={num_kv_heads},T={num_tokens},D={head_dim}"

    np.random.seed(123)
    q_np = np.random.randn(num_heads, num_tokens, head_dim).astype(np.float32)
    k_np = np.random.randn(num_kv_heads, num_tokens, head_dim).astype(np.float32)
    q_gamma_np = np.random.randn(head_dim).astype(np.float32)
    k_gamma_np = np.random.randn(head_dim).astype(np.float32)
    d_q_out_np = np.random.randn(num_heads, num_tokens, head_dim).astype(np.float32)
    d_k_out_np = np.random.randn(num_kv_heads, num_tokens, head_dim).astype(np.float32)

    q_t = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
    k_t = torch.tensor(k_np, dtype=torch.float32, requires_grad=True)
    qg_t = torch.tensor(q_gamma_np, dtype=torch.float32, requires_grad=True)
    kg_t = torch.tensor(k_gamma_np, dtype=torch.float32, requires_grad=True)
    d_q_t = torch.tensor(d_q_out_np, dtype=torch.float32)
    d_k_t = torch.tensor(d_k_out_np, dtype=torch.float32)
    q_out_t, k_out_t = qk_norm_torch(q_t, k_t, qg_t, kg_t, eps)
    loss_t = (q_out_t * d_q_t).sum() + (k_out_t * d_k_t).sum()
    loss_t.backward()

    d_q_in_ref = q_t.grad.detach().cpu().numpy()
    d_k_in_ref = k_t.grad.detach().cpu().numpy()
    d_q_gamma_ref = qg_t.grad.detach().cpu().numpy()
    d_k_gamma_ref = kg_t.grad.detach().cpu().numpy()

    d_q_in = np.zeros_like(q_np, dtype=np.float32)
    d_k_in = np.zeros_like(k_np, dtype=np.float32)
    d_q_gamma = np.zeros_like(q_gamma_np, dtype=np.float32)
    d_k_gamma = np.zeros_like(k_gamma_np, dtype=np.float32)

    d_q_out_ptr = numpy_to_ptr(d_q_out_np.reshape(-1))
    d_k_out_ptr = numpy_to_ptr(d_k_out_np.reshape(-1))
    q_in_ptr = numpy_to_ptr(q_np.reshape(-1))
    k_in_ptr = numpy_to_ptr(k_np.reshape(-1))
    q_gamma_ptr = numpy_to_ptr(q_gamma_np)
    k_gamma_ptr = numpy_to_ptr(k_gamma_np)
    d_q_in_ptr = numpy_to_ptr(d_q_in.reshape(-1))
    d_k_in_ptr = numpy_to_ptr(d_k_in.reshape(-1))
    d_q_gamma_ptr = numpy_to_ptr(d_q_gamma)
    d_k_gamma_ptr = numpy_to_ptr(d_k_gamma)

    lib.qk_norm_backward(
        d_q_out_ptr, d_k_out_ptr, q_in_ptr, k_in_ptr, q_gamma_ptr, k_gamma_ptr,
        d_q_in_ptr, d_k_in_ptr, d_q_gamma_ptr, d_k_gamma_ptr,
        ctypes.c_int(num_heads), ctypes.c_int(num_kv_heads),
        ctypes.c_int(num_tokens), ctypes.c_int(head_dim), ctypes.c_float(eps)
    )

    diff_dq = float(np.max(np.abs(d_q_in - d_q_in_ref)))
    diff_dk = float(np.max(np.abs(d_k_in - d_k_in_ref)))
    diff_dqg = float(np.max(np.abs(d_q_gamma - d_q_gamma_ref)))
    diff_dkg = float(np.max(np.abs(d_k_gamma - d_k_gamma_ref)))
    combined_diff = max(diff_dq, diff_dk, diff_dqg, diff_dkg)
    tolerance = 1e-4

    def pytorch_qk_norm_backward():
        q_t_local = torch.tensor(q_np, dtype=torch.float32, requires_grad=True)
        k_t_local = torch.tensor(k_np, dtype=torch.float32, requires_grad=True)
        qg_t_local = torch.tensor(q_gamma_np, dtype=torch.float32, requires_grad=True)
        kg_t_local = torch.tensor(k_gamma_np, dtype=torch.float32, requires_grad=True)
        q_out_local, k_out_local = qk_norm_torch(q_t_local, k_t_local, qg_t_local, kg_t_local, eps)
        loss_local = (q_out_local * d_q_t).sum() + (k_out_local * d_k_t).sum()
        loss_local.backward()

    def c_qk_norm_backward():
        d_q_in.fill(0.0)
        d_k_in.fill(0.0)
        d_q_gamma.fill(0.0)
        d_k_gamma.fill(0.0)
        lib.qk_norm_backward(
            d_q_out_ptr, d_k_out_ptr, q_in_ptr, k_in_ptr, q_gamma_ptr, k_gamma_ptr,
            d_q_in_ptr, d_k_in_ptr, d_q_gamma_ptr, d_k_gamma_ptr,
            ctypes.c_int(num_heads), ctypes.c_int(num_kv_heads),
            ctypes.c_int(num_tokens), ctypes.c_int(head_dim), ctypes.c_float(eps)
        )

    pt_time = time_function(pytorch_qk_norm_backward, warmup=warmup, iterations=iterations, name="PyTorch BWD")
    c_time = time_function(c_qk_norm_backward, warmup=warmup, iterations=iterations, name="C BWD")

    return TestResult(
        name=label,
        passed=combined_diff <= tolerance,
        max_diff=combined_diff,
        tolerance=tolerance,
        pytorch_time=pt_time,
        kernel_time=c_time,
    ), {
        "d_q_in": diff_dq,
        "d_k_in": diff_dk,
        "d_q_gamma": diff_dqg,
        "d_k_gamma": diff_dkg,
    }


def run_consistency_test(num_heads: int, num_kv_heads: int, num_tokens: int, head_dim: int, eps: float = 1e-6):
    np.random.seed(7)
    q_np = np.random.randn(num_heads * num_tokens * head_dim).astype(np.float32)
    k_np = np.random.randn(num_kv_heads * num_tokens * head_dim).astype(np.float32)
    q_gamma_np = np.random.randn(head_dim).astype(np.float32)
    k_gamma_np = np.random.randn(head_dim).astype(np.float32)

    q1 = q_np.copy()
    k1 = k_np.copy()
    lib.qk_norm_forward(
        numpy_to_ptr(q1), numpy_to_ptr(k1),
        numpy_to_ptr(q_gamma_np), numpy_to_ptr(k_gamma_np),
        ctypes.c_int(num_heads), ctypes.c_int(num_kv_heads),
        ctypes.c_int(num_tokens), ctypes.c_int(head_dim), ctypes.c_float(eps)
    )

    q2 = q_np.copy()
    k2 = k_np.copy()
    lib.rmsnorm_forward(
        numpy_to_ptr(q2), numpy_to_ptr(q_gamma_np), numpy_to_ptr(q2), None,
        ctypes.c_int(num_heads * num_tokens), ctypes.c_int(head_dim),
        ctypes.c_int(head_dim), ctypes.c_float(eps)
    )
    lib.rmsnorm_forward(
        numpy_to_ptr(k2), numpy_to_ptr(k_gamma_np), numpy_to_ptr(k2), None,
        ctypes.c_int(num_kv_heads * num_tokens), ctypes.c_int(head_dim),
        ctypes.c_int(head_dim), ctypes.c_float(eps)
    )

    combined = max(float(np.max(np.abs(q1 - q2))), float(np.max(np.abs(k1 - k2))))
    return TestResult(
        name="FWD wrapper == 2x rmsnorm",
        passed=combined == 0.0,
        max_diff=combined,
        tolerance=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="QK norm parity + speed test (forward and backward).")
    parser.add_argument("--quick", action="store_true", help="Run lightweight configs for CI/make test.")
    args = parser.parse_args()

    print_system_info()
    report = TestReport(
        test_name="QK Norm (forward + backward)",
        dtype="fp32",
        shape="Various (Qwen-style configs)",
        cpu_info=get_cpu_info(),
    )

    report.add_result(run_consistency_test(num_heads=32, num_kv_heads=8, num_tokens=1, head_dim=128))

    if args.quick:
        fwd_cases: List[Dict] = [
            dict(num_heads=16, num_kv_heads=8, num_tokens=1, head_dim=64, warmup=2, iterations=30, label="FWD decode"),
            dict(num_heads=16, num_kv_heads=8, num_tokens=32, head_dim=64, warmup=2, iterations=20, label="FWD prefill"),
        ]
        bwd_cases: List[Dict] = [
            dict(num_heads=16, num_kv_heads=8, num_tokens=1, head_dim=64, warmup=2, iterations=20, label="BWD decode"),
            dict(num_heads=16, num_kv_heads=8, num_tokens=16, head_dim=64, warmup=2, iterations=15, label="BWD prefill"),
        ]
    else:
        fwd_cases = [
            dict(num_heads=32, num_kv_heads=8, num_tokens=1, head_dim=128, warmup=10, iterations=300, label="FWD Qwen3-8B decode"),
            dict(num_heads=32, num_kv_heads=8, num_tokens=128, head_dim=128, warmup=5, iterations=100, label="FWD Qwen3-8B prefill-128"),
            dict(num_heads=16, num_kv_heads=8, num_tokens=1, head_dim=64, warmup=10, iterations=300, label="FWD Qwen3-0.6B decode"),
            dict(num_heads=32, num_kv_heads=32, num_tokens=1, head_dim=128, warmup=10, iterations=300, label="FWD MHA decode"),
        ]
        bwd_cases = [
            dict(num_heads=32, num_kv_heads=8, num_tokens=1, head_dim=128, warmup=5, iterations=120, label="BWD Qwen3-8B decode"),
            dict(num_heads=32, num_kv_heads=8, num_tokens=64, head_dim=128, warmup=3, iterations=40, label="BWD Qwen3-8B prefill-64"),
            dict(num_heads=32, num_kv_heads=32, num_tokens=1, head_dim=128, warmup=5, iterations=120, label="BWD MHA decode"),
        ]

    for cfg in fwd_cases:
        result, details = run_qk_norm_test(**cfg)
        report.add_result(result)
        print(f"  [detail:{cfg['label']}] q={details['q']:.2e} k={details['k']:.2e}")

    for cfg in bwd_cases:
        result, details = run_qk_norm_backward_test(**cfg)
        report.add_result(result)
        print(
            f"  [detail:{cfg['label']}] "
            f"d_q_in={details['d_q_in']:.2e} d_k_in={details['d_k_in']:.2e} "
            f"d_q_gamma={details['d_q_gamma']:.2e} d_k_gamma={details['d_k_gamma']:.2e}"
        )

    report.print_report()
    return 0 if report.all_passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
