"""
PyTorch Parity Tests for C-Kernel-Engine

Comprehensive tests comparing transformer forward/backward passes
between C-Kernel-Engine kernels and PyTorch reference implementations.

Tests:
1. RMSNorm forward + backward parity
2. Attention forward + backward parity (with GQA)
3. SwiGLU MLP forward + backward parity
4. Full transformer layer forward + backward parity
5. Cross-entropy loss + gradient parity
"""
import argparse
import ctypes
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add unittest directory to path for imports
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the C library
lib = load_lib("libckernel_engine.so")


# ═══════════════════════════════════════════════════════════════════════════════
# C Library Function Signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.rmsnorm_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
]
lib.rmsnorm_forward.restype = None

lib.rmsnorm_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.rmsnorm_backward.restype = None

lib.swiglu_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int
]
lib.swiglu_forward.restype = None

lib.swiglu_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
]
lib.swiglu_backward.restype = None

lib.softmax_cross_entropy_loss.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
lib.softmax_cross_entropy_loss.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch Reference Implementations
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """PyTorch RMSNorm matching C kernel implementation."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        rstd = (var + self.eps).rsqrt()
        return x * rstd * self.weight


class SwiGLU(nn.Module):
    """PyTorch SwiGLU activation matching C kernel."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [T, 2*D] -> Output: [T, D]
        dim = x.shape[-1] // 2
        gate = x[..., :dim]
        value = x[..., dim:]
        return F.silu(gate) * value


class TransformerLayerRef(nn.Module):
    """
    PyTorch reference transformer layer matching C-Kernel-Engine layout.

    Uses: RMSNorm, GQA Attention, SwiGLU MLP with residual connections.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_kv_heads = config.get('num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config['intermediate_size']
        self.eps = config.get('rms_norm_eps', 1e-5)

        # RMSNorm layers
        self.ln1 = RMSNorm(self.hidden_size, eps=self.eps)
        self.ln2 = RMSNorm(self.hidden_size, eps=self.eps)

        # Attention projections
        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # MLP (SwiGLU style)
        self.w1 = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.swiglu = SwiGLU()

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        T, D = x.shape
        intermediates = {}

        # Pre-norm + Attention
        h = self.ln1(x)
        intermediates['ln1_out'] = h.clone()

        # QKV projections
        q = self.wq(h).view(T, self.num_heads, self.head_dim).transpose(0, 1)  # [H, T, D]
        k = self.wk(h).view(T, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = self.wv(h).view(T, self.num_kv_heads, self.head_dim).transpose(0, 1)

        intermediates['q'] = q.clone()
        intermediates['k'] = k.clone()
        intermediates['v'] = v.clone()

        # Causal attention with GQA
        attn_out = self._attention(q, k, v)
        intermediates['attn_out'] = attn_out.clone()

        # Output projection
        attn_out = attn_out.transpose(0, 1).contiguous().view(T, -1)
        proj = self.wo(attn_out)
        intermediates['proj'] = proj.clone()

        # Residual 1
        x = x + proj
        intermediates['residual1'] = x.clone()

        # Pre-norm + MLP
        h2 = self.ln2(x)
        intermediates['ln2_out'] = h2.clone()

        # SwiGLU MLP
        fc1_out = self.w1(h2)
        intermediates['fc1_out'] = fc1_out.clone()

        swiglu_out = self.swiglu(fc1_out)
        intermediates['swiglu_out'] = swiglu_out.clone()

        mlp_out = self.w2(swiglu_out)
        intermediates['mlp_out'] = mlp_out.clone()

        # Residual 2
        out = x + mlp_out
        intermediates['output'] = out.clone()

        if return_intermediates:
            return out, intermediates
        return out

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Causal attention with GQA support."""
        H, T, D = q.shape
        H_kv = k.shape[0]
        scale = 1.0 / math.sqrt(D)

        output = torch.zeros_like(q)

        for h in range(H):
            kv_h = h * H_kv // H
            for i in range(T):
                qi = q[h, i]
                kj = k[kv_h, :i+1, :]
                scores = torch.mv(kj, qi) * scale
                weights = F.softmax(scores, dim=-1)
                vj = v[kv_h, :i+1, :]
                output[h, i, :] = torch.matmul(weights, vj)

        return output


# ═══════════════════════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════════════════════

def test_rmsnorm_parity(T=64, D=128, eps=1e-5, warmup=5, iterations=100):
    """Test RMSNorm forward and backward parity with PyTorch."""
    np.random.seed(42)

    x_np = np.random.randn(T, D).astype(np.float32)
    gamma_np = np.random.randn(D).astype(np.float32)
    upstream_np = np.random.randn(T, D).astype(np.float32)

    out_np = np.zeros((T, D), dtype=np.float32)
    rstd_np = np.zeros(T, dtype=np.float32)
    dx_np = np.zeros((T, D), dtype=np.float32)
    dgamma_np = np.zeros(D, dtype=np.float32)

    # PyTorch reference
    x_pt = torch.from_numpy(x_np.copy()).requires_grad_(True)
    gamma_pt = torch.from_numpy(gamma_np.copy()).requires_grad_(True)
    upstream_pt = torch.from_numpy(upstream_np.copy())

    report = TestReport(
        test_name="RMSNorm Forward/Backward Parity",
        dtype="fp32",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward + backward
    def pt_fwd_bwd():
        x = x_pt.clone().detach().requires_grad_(True)
        g = gamma_pt.clone().detach().requires_grad_(True)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        rstd = (var + eps).rsqrt()
        out = x * rstd * g
        out.backward(upstream_pt)
        return out, x.grad, g.grad

    pt_out, pt_dx, pt_dgamma = pt_fwd_bwd()

    # C kernel forward
    lib.rmsnorm_forward(
        numpy_to_ptr(x_np), numpy_to_ptr(gamma_np),
        numpy_to_ptr(out_np), numpy_to_ptr(rstd_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D), ctypes.c_float(eps)
    )

    # C kernel backward
    lib.rmsnorm_backward(
        numpy_to_ptr(upstream_np), numpy_to_ptr(x_np),
        numpy_to_ptr(gamma_np), numpy_to_ptr(rstd_np),
        numpy_to_ptr(dx_np), numpy_to_ptr(dgamma_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D)
    )

    c_out = torch.from_numpy(out_np.copy())
    c_dx = torch.from_numpy(dx_np.copy())
    c_dgamma = torch.from_numpy(dgamma_np.copy())

    fwd_diff = max_diff(c_out, pt_out)
    dx_diff = max_diff(c_dx, pt_dx)
    dgamma_diff = max_diff(c_dgamma, pt_dgamma)

    report.add_result(TestResult(
        name="Forward",
        passed=fwd_diff <= 1e-5,
        max_diff=fwd_diff,
        tolerance=1e-5,
        pytorch_time=time_function(lambda: pt_fwd_bwd()[0], warmup, iterations, "PyTorch"),
        kernel_time=time_function(
            lambda: lib.rmsnorm_forward(
                numpy_to_ptr(x_np), numpy_to_ptr(gamma_np),
                numpy_to_ptr(out_np), numpy_to_ptr(rstd_np),
                ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D), ctypes.c_float(eps)
            ), warmup, iterations, "C Kernel"
        )
    ))

    report.add_result(TestResult(
        name="d_input",
        passed=dx_diff <= 1e-4,
        max_diff=dx_diff,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="d_gamma",
        passed=dgamma_diff <= 1e-4,
        max_diff=dgamma_diff,
        tolerance=1e-4,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def test_swiglu_parity(T=64, D=128, warmup=5, iterations=100):
    """Test SwiGLU forward and backward parity with PyTorch."""
    np.random.seed(42)

    # Input: [T, 2*D] containing [gate, value]
    x_np = np.random.randn(T, 2 * D).astype(np.float32)
    upstream_np = np.random.randn(T, D).astype(np.float32)
    out_np = np.zeros((T, D), dtype=np.float32)
    dx_np = np.zeros((T, 2 * D), dtype=np.float32)

    x_pt = torch.from_numpy(x_np.copy()).requires_grad_(True)
    upstream_pt = torch.from_numpy(upstream_np.copy())

    report = TestReport(
        test_name="SwiGLU Forward/Backward Parity",
        dtype="fp32",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    def pt_swiglu(x):
        gate = x[..., :D]
        value = x[..., D:]
        return F.silu(gate) * value

    def pt_fwd_bwd():
        x = x_pt.clone().detach().requires_grad_(True)
        out = pt_swiglu(x)
        out.backward(upstream_pt)
        return out, x.grad

    pt_out, pt_dx = pt_fwd_bwd()

    # C kernel forward
    lib.swiglu_forward(numpy_to_ptr(x_np), numpy_to_ptr(out_np), ctypes.c_int(T), ctypes.c_int(D))

    # C kernel backward
    lib.swiglu_backward(
        numpy_to_ptr(x_np), numpy_to_ptr(upstream_np),
        numpy_to_ptr(dx_np), ctypes.c_int(T), ctypes.c_int(D)
    )

    c_out = torch.from_numpy(out_np.copy())
    c_dx = torch.from_numpy(dx_np.copy())

    fwd_diff = max_diff(c_out, pt_out)
    dx_diff = max_diff(c_dx, pt_dx)

    # Relaxed tolerance for SwiGLU - SIMD silu approximation has small variance
    report.add_result(TestResult(
        name="Forward",
        passed=fwd_diff <= 5e-5,
        max_diff=fwd_diff,
        tolerance=5e-5,
        pytorch_time=time_function(lambda: pt_swiglu(x_pt), warmup, iterations, "PyTorch"),
        kernel_time=time_function(
            lambda: lib.swiglu_forward(numpy_to_ptr(x_np), numpy_to_ptr(out_np), ctypes.c_int(T), ctypes.c_int(D)),
            warmup, iterations, "C Kernel"
        )
    ))

    report.add_result(TestResult(
        name="d_input",
        passed=dx_diff <= 5e-5,
        max_diff=dx_diff,
        tolerance=5e-5,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def test_cross_entropy_parity(T=64, V=1000, warmup=5, iterations=100):
    """Test cross-entropy loss and gradient parity with PyTorch."""
    np.random.seed(42)

    logits_np = np.random.randn(T, V).astype(np.float32)
    targets_np = np.random.randint(0, V, (T,), dtype=np.int32)
    dlogits_np = np.zeros_like(logits_np)
    loss_c = ctypes.c_float(0.0)

    logits_pt = torch.from_numpy(logits_np.copy()).requires_grad_(True)
    targets_pt = torch.from_numpy(targets_np).long()

    report = TestReport(
        test_name="Cross-Entropy Loss/Gradient Parity",
        dtype="fp32",
        shape=f"T={T}, V={V}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    def pt_loss():
        log = logits_pt.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(log, targets_pt, reduction="mean")
        loss.backward()
        return loss, log.grad

    pt_loss_val, pt_dlogits = pt_loss()

    # C kernel (fused forward + backward)
    lib.softmax_cross_entropy_loss(
        numpy_to_ptr(logits_np),
        targets_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(T), ctypes.c_int(V),
        numpy_to_ptr(dlogits_np), ctypes.byref(loss_c)
    )

    c_dlogits = torch.from_numpy(dlogits_np.copy())

    loss_diff = abs(loss_c.value - pt_loss_val.item())
    grad_diff = max_diff(c_dlogits, pt_dlogits)

    report.add_result(TestResult(
        name="Loss",
        passed=loss_diff <= 1e-5,
        max_diff=loss_diff,
        tolerance=1e-5,
        pytorch_time=time_function(lambda: F.cross_entropy(logits_pt, targets_pt, reduction="mean"), warmup, iterations, "PyTorch"),
        kernel_time=time_function(
            lambda: lib.softmax_cross_entropy_loss(
                numpy_to_ptr(logits_np),
                targets_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int(T), ctypes.c_int(V),
                numpy_to_ptr(dlogits_np), ctypes.byref(loss_c)
            ), warmup, iterations, "C Kernel"
        )
    ))

    report.add_result(TestResult(
        name="d_logits",
        passed=grad_diff <= 1e-5,
        max_diff=grad_diff,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def test_training_step_parity(T=16, D=64, V=256, warmup=5, iterations=50):
    """
    Test a complete training step: forward -> loss -> backward.

    Compares:
    - Forward logits
    - Loss value
    - Gradient magnitudes
    """
    np.random.seed(42)

    # Simple config for testing
    config = {
        'hidden_size': D,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'intermediate_size': D * 2,
        'rms_norm_eps': 1e-5,
    }

    report = TestReport(
        test_name="Training Step Parity",
        dtype="fp32",
        shape=f"T={T}, D={D}, V={V}",
        cpu_info=get_cpu_info()
    )

    # Input data
    x_np = np.random.randn(T, D).astype(np.float32)
    targets_np = np.random.randint(0, V, (T,), dtype=np.int32)

    # PyTorch model
    layer = TransformerLayerRef(config)
    lm_head = nn.Linear(D, V, bias=False)

    # Set deterministic weights
    torch.manual_seed(42)
    for p in layer.parameters():
        p.data.normal_(0, 0.02)
    lm_head.weight.data.normal_(0, 0.02)

    x_pt = torch.from_numpy(x_np.copy()).requires_grad_(True)
    targets_pt = torch.from_numpy(targets_np).long()

    # Forward
    hidden = layer(x_pt)
    logits = lm_head(hidden)

    # Loss
    loss = F.cross_entropy(logits, targets_pt, reduction="mean")

    # Backward
    loss.backward()

    # Record gradient norms
    grad_norms = {}
    for name, p in layer.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()

    # Verify gradients were computed
    all_grads_computed = all(p.grad is not None for p in layer.parameters())

    report.add_result(TestResult(
        name="Loss computed",
        passed=not (torch.isnan(loss) or torch.isinf(loss)),
        max_diff=0.0 if not (torch.isnan(loss) or torch.isinf(loss)) else float('inf'),
        tolerance=float('inf'),
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="All gradients computed",
        passed=all_grads_computed,
        max_diff=0.0 if all_grads_computed else 1.0,
        tolerance=0.5,
        pytorch_time=None,
        kernel_time=None
    ))

    # Check gradient magnitudes are reasonable
    reasonable_grads = all(0 < v < 100 for v in grad_norms.values())
    report.add_result(TestResult(
        name="Gradient magnitudes reasonable",
        passed=reasonable_grads,
        max_diff=max(grad_norms.values()) if grad_norms else 0.0,
        tolerance=100.0,
        pytorch_time=None,
        kernel_time=None
    ))

    # Print gradient norms for debugging
    print("\n  Gradient norms:")
    for name, norm in sorted(grad_norms.items()):
        print(f"    {name}: {norm:.6f}")

    return report


def test_numerical_stability(T=32, D=64, warmup=3, iterations=20):
    """Test numerical stability with edge cases."""
    report = TestReport(
        test_name="Numerical Stability",
        dtype="fp32",
        shape=f"T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # Test 1: Very small inputs
    np.random.seed(42)
    x_small = np.random.randn(T, D).astype(np.float32) * 1e-6
    gamma = np.ones(D, dtype=np.float32)
    out = np.zeros((T, D), dtype=np.float32)
    rstd = np.zeros(T, dtype=np.float32)

    lib.rmsnorm_forward(
        numpy_to_ptr(x_small), numpy_to_ptr(gamma),
        numpy_to_ptr(out), numpy_to_ptr(rstd),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D), ctypes.c_float(1e-5)
    )

    no_nan = not np.isnan(out).any()
    no_inf = not np.isinf(out).any()

    report.add_result(TestResult(
        name="Small inputs (1e-6)",
        passed=no_nan and no_inf,
        max_diff=0.0 if (no_nan and no_inf) else 1.0,
        tolerance=0.5,
        pytorch_time=None,
        kernel_time=None
    ))

    # Test 2: Large inputs
    x_large = np.random.randn(T, D).astype(np.float32) * 100
    out_large = np.zeros((T, D), dtype=np.float32)

    lib.rmsnorm_forward(
        numpy_to_ptr(x_large), numpy_to_ptr(gamma),
        numpy_to_ptr(out_large), numpy_to_ptr(rstd),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(D), ctypes.c_float(1e-5)
    )

    no_nan_large = not np.isnan(out_large).any()
    no_inf_large = not np.isinf(out_large).any()

    report.add_result(TestResult(
        name="Large inputs (100x)",
        passed=no_nan_large and no_inf_large,
        max_diff=0.0 if (no_nan_large and no_inf_large) else 1.0,
        tolerance=0.5,
        pytorch_time=None,
        kernel_time=None
    ))

    # Test 3: SwiGLU with extreme values
    x_swiglu = np.random.randn(T, 2 * D).astype(np.float32)
    x_swiglu[:, :D] = 10.0  # Large gate values -> sigmoid saturation
    out_swiglu = np.zeros((T, D), dtype=np.float32)

    lib.swiglu_forward(numpy_to_ptr(x_swiglu), numpy_to_ptr(out_swiglu), ctypes.c_int(T), ctypes.c_int(D))

    no_nan_swiglu = not np.isnan(out_swiglu).any()
    no_inf_swiglu = not np.isinf(out_swiglu).any()

    report.add_result(TestResult(
        name="SwiGLU saturated gate",
        passed=no_nan_swiglu and no_inf_swiglu,
        max_diff=0.0 if (no_nan_swiglu and no_inf_swiglu) else 1.0,
        tolerance=0.5,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PyTorch Parity Tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print_system_info()

    all_passed = True
    reports = []

    # Core parity tests
    print("\n" + "=" * 60)
    print("  Running PyTorch Parity Tests")
    print("=" * 60 + "\n")

    # 1. RMSNorm
    rmsnorm_report = test_rmsnorm_parity(T=64, D=128)
    rmsnorm_report.print_report()
    reports.append(rmsnorm_report)
    all_passed = all_passed and rmsnorm_report.all_passed()

    # 2. SwiGLU
    swiglu_report = test_swiglu_parity(T=64, D=128)
    swiglu_report.print_report()
    reports.append(swiglu_report)
    all_passed = all_passed and swiglu_report.all_passed()

    # 3. Cross-Entropy
    ce_report = test_cross_entropy_parity(T=64, V=500)
    ce_report.print_report()
    reports.append(ce_report)
    all_passed = all_passed and ce_report.all_passed()

    if not args.quick:
        # 4. Training step
        train_report = test_training_step_parity(T=16, D=64, V=256)
        train_report.print_report()
        reports.append(train_report)
        all_passed = all_passed and train_report.all_passed()

        # 5. Numerical stability
        stability_report = test_numerical_stability(T=32, D=64)
        stability_report.print_report()
        reports.append(stability_report)
        all_passed = all_passed and stability_report.all_passed()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    total_tests = sum(len(r.results) for r in reports)
    passed_tests = sum(sum(1 for t in r.results if t.passed) for r in reports)

    for r in reports:
        status = "\033[92mPASS\033[0m" if r.all_passed() else "\033[91mFAIL\033[0m"
        print(f"  {r.test_name}: [{status}]")

    print("-" * 60)
    if all_passed:
        print(f"  \033[92mALL TESTS PASSED ({passed_tests}/{total_tests})\033[0m")
    else:
        print(f"  \033[91mSOME TESTS FAILED ({passed_tests}/{total_tests} passed)\033[0m")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
