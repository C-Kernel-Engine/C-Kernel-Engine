"""
QK Norm kernel unit tests with PyTorch parity and performance metrics.

Tests qk_norm_forward (per-head RMSNorm on Q and K) against PyTorch reference.
Reports accuracy, timing, and system information.

Architecture context:
  QK norm normalizes Q and K after projection, before RoPE. This stabilizes
  the Q*K^T dot product before softmax, preventing attention collapse from
  large-magnitude vectors. V is NOT normalized because it doesn't participate
  in attention score computation -- it's linearly combined after softmax.

After changes: python unittest/test_qk_norm.py
"""
import ctypes
import sys

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_engine.so")


# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

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

# Also load rmsnorm_forward for direct comparison
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


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch reference implementation
# ═══════════════════════════════════════════════════════════════════════════════

def rmsnorm_torch(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference RMSNorm: x: [..., D], gamma: [D]"""
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    return x * rstd * gamma


def qk_norm_torch(q: torch.Tensor, k: torch.Tensor,
                   q_gamma: torch.Tensor, k_gamma: torch.Tensor,
                   eps: float):
    """
    PyTorch reference for per-head QK norm.

    Args:
        q: [num_heads, num_tokens, head_dim]
        k: [num_kv_heads, num_tokens, head_dim]
        q_gamma: [head_dim]
        k_gamma: [head_dim]
        eps: RMSNorm epsilon

    Returns:
        (q_normed, k_normed) with same shapes
    """
    # Apply RMSNorm independently to each head's vector
    # q shape: [H, T, D] -> normalize along D for each (h, t) pair
    q_normed = rmsnorm_torch(q, q_gamma, eps)
    k_normed = rmsnorm_torch(k, k_gamma, eps)
    return q_normed, k_normed


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_qk_norm_test(num_heads, num_kv_heads, num_tokens, head_dim, eps=1e-6,
                     warmup=10, iterations=1000, label=None):
    """Run QK norm test with accuracy and timing for a specific configuration."""
    if label is None:
        label = f"H={num_heads}, KV={num_kv_heads}, T={num_tokens}, D={head_dim}"

    np.random.seed(42)

    # Pre-allocate numpy arrays (contiguous, head-major layout)
    q_np = np.random.randn(num_heads * num_tokens * head_dim).astype(np.float32)
    k_np = np.random.randn(num_kv_heads * num_tokens * head_dim).astype(np.float32)
    q_gamma_np = np.random.randn(head_dim).astype(np.float32)
    k_gamma_np = np.random.randn(head_dim).astype(np.float32)

    # Keep copies for C kernel (operates in-place)
    q_c = q_np.copy()
    k_c = k_np.copy()

    # Get pointers
    q_ptr = numpy_to_ptr(q_c)
    k_ptr = numpy_to_ptr(k_c)
    q_gamma_ptr = numpy_to_ptr(q_gamma_np)
    k_gamma_ptr = numpy_to_ptr(k_gamma_np)

    # PyTorch reference
    q_torch = torch.from_numpy(q_np.copy()).reshape(num_heads, num_tokens, head_dim)
    k_torch = torch.from_numpy(k_np.copy()).reshape(num_kv_heads, num_tokens, head_dim)
    q_gamma_torch = torch.from_numpy(q_gamma_np.copy())
    k_gamma_torch = torch.from_numpy(k_gamma_np.copy())

    q_ref, k_ref = qk_norm_torch(q_torch, k_torch, q_gamma_torch, k_gamma_torch, eps)
    q_ref_flat = q_ref.reshape(-1)
    k_ref_flat = k_ref.reshape(-1)

    # C kernel (in-place)
    lib.qk_norm_forward(q_ptr, k_ptr, q_gamma_ptr, k_gamma_ptr,
                        ctypes.c_int(num_heads), ctypes.c_int(num_kv_heads),
                        ctypes.c_int(num_tokens), ctypes.c_int(head_dim),
                        ctypes.c_float(eps))

    q_out = torch.from_numpy(q_c.copy())
    k_out = torch.from_numpy(k_c.copy())
    q_diff = max_diff(q_out, q_ref_flat)
    k_diff = max_diff(k_out, k_ref_flat)

    # Timing: PyTorch
    def pytorch_qk_norm():
        return qk_norm_torch(q_torch, k_torch, q_gamma_torch, k_gamma_torch, eps)

    pt_time = time_function(pytorch_qk_norm, warmup=warmup, iterations=iterations,
                            name="PyTorch")

    # Timing: C kernel
    def c_qk_norm():
        # Reset buffers (in-place modifies them)
        np.copyto(q_c, q_np)
        np.copyto(k_c, k_np)
        lib.qk_norm_forward(q_ptr, k_ptr, q_gamma_ptr, k_gamma_ptr,
                            ctypes.c_int(num_heads), ctypes.c_int(num_kv_heads),
                            ctypes.c_int(num_tokens), ctypes.c_int(head_dim),
                            ctypes.c_float(eps))

    c_time = time_function(c_qk_norm, warmup=warmup, iterations=iterations,
                           name="C qk_norm")

    combined_diff = max(q_diff, k_diff)
    tolerance = 1e-5

    return TestResult(
        name=label,
        passed=combined_diff <= tolerance,
        max_diff=combined_diff,
        tolerance=tolerance,
        pytorch_time=pt_time,
        kernel_time=c_time,
    ), q_diff, k_diff


def run_consistency_test(num_heads, num_kv_heads, num_tokens, head_dim, eps=1e-6):
    """
    Verify qk_norm_forward matches calling rmsnorm_forward directly.
    This ensures the wrapper is correctly decomposing into two rmsnorm calls.
    """
    np.random.seed(123)

    q_np = np.random.randn(num_heads * num_tokens * head_dim).astype(np.float32)
    k_np = np.random.randn(num_kv_heads * num_tokens * head_dim).astype(np.float32)
    q_gamma_np = np.random.randn(head_dim).astype(np.float32)
    k_gamma_np = np.random.randn(head_dim).astype(np.float32)

    # Path 1: qk_norm_forward
    q1 = q_np.copy()
    k1 = k_np.copy()
    lib.qk_norm_forward(
        numpy_to_ptr(q1), numpy_to_ptr(k1),
        numpy_to_ptr(q_gamma_np), numpy_to_ptr(k_gamma_np),
        ctypes.c_int(num_heads), ctypes.c_int(num_kv_heads),
        ctypes.c_int(num_tokens), ctypes.c_int(head_dim),
        ctypes.c_float(eps))

    # Path 2: two direct rmsnorm_forward calls
    q2 = q_np.copy()
    k2 = k_np.copy()
    lib.rmsnorm_forward(
        numpy_to_ptr(q2), numpy_to_ptr(q_gamma_np), numpy_to_ptr(q2), None,
        ctypes.c_int(num_heads * num_tokens), ctypes.c_int(head_dim),
        ctypes.c_int(head_dim), ctypes.c_float(eps))
    lib.rmsnorm_forward(
        numpy_to_ptr(k2), numpy_to_ptr(k_gamma_np), numpy_to_ptr(k2), None,
        ctypes.c_int(num_kv_heads * num_tokens), ctypes.c_int(head_dim),
        ctypes.c_int(head_dim), ctypes.c_float(eps))

    q_diff = float(np.max(np.abs(q1 - q2)))
    k_diff = float(np.max(np.abs(k1 - k2)))
    combined = max(q_diff, k_diff)

    return TestResult(
        name="wrapper == 2x rmsnorm",
        passed=combined == 0.0,  # Should be bit-exact
        max_diff=combined,
        tolerance=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    report = TestReport(
        test_name="QK Norm (per-head RMSNorm on Q and K)",
        dtype="fp32",
        shape="Various (Qwen3 configs)",
        cpu_info=get_cpu_info()
    )

    # --- Consistency test: wrapper matches direct rmsnorm calls ---
    consistency = run_consistency_test(
        num_heads=32, num_kv_heads=8, num_tokens=1, head_dim=128)
    report.add_result(consistency)

    # --- Qwen3-8B decode (single token) ---
    result, qd, kd = run_qk_norm_test(
        num_heads=32, num_kv_heads=8, num_tokens=1, head_dim=128,
        label="Qwen3-8B decode (H=32,KV=8,T=1,D=128)")
    report.add_result(result)
    print(f"  [detail] Q max_diff={qd:.2e}, K max_diff={kd:.2e}")

    # --- Qwen3-8B prefill (128 tokens) ---
    result, qd, kd = run_qk_norm_test(
        num_heads=32, num_kv_heads=8, num_tokens=128, head_dim=128,
        label="Qwen3-8B prefill (H=32,KV=8,T=128,D=128)")
    report.add_result(result)
    print(f"  [detail] Q max_diff={qd:.2e}, K max_diff={kd:.2e}")

    # --- Qwen3-0.6B decode (smaller model) ---
    result, qd, kd = run_qk_norm_test(
        num_heads=16, num_kv_heads=8, num_tokens=1, head_dim=64,
        label="Qwen3-0.6B decode (H=16,KV=8,T=1,D=64)")
    report.add_result(result)
    print(f"  [detail] Q max_diff={qd:.2e}, K max_diff={kd:.2e}")

    # --- Qwen3-8B prefill (512 tokens, stress test) ---
    result, qd, kd = run_qk_norm_test(
        num_heads=32, num_kv_heads=8, num_tokens=512, head_dim=128,
        warmup=5, iterations=200,
        label="Qwen3-8B prefill (H=32,KV=8,T=512,D=128)")
    report.add_result(result)
    print(f"  [detail] Q max_diff={qd:.2e}, K max_diff={kd:.2e}")

    # --- MHA config (num_heads == num_kv_heads, no GQA) ---
    result, qd, kd = run_qk_norm_test(
        num_heads=32, num_kv_heads=32, num_tokens=1, head_dim=128,
        label="MHA decode (H=32,KV=32,T=1,D=128)")
    report.add_result(result)
    print(f"  [detail] Q max_diff={qd:.2e}, K max_diff={kd:.2e}")

    report.print_report()

    if not report.all_passed():
        sys.exit(1)
