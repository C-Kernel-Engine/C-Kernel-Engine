"""
LayerNorm kernel unit tests with performance metrics.

Tests forward and backward passes against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    CPUInfo, TestReport, TestResult, get_cpu_info,
    max_diff, tensor_to_ptr, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.layernorm_naive_serial.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.layernorm_naive_serial.restype = None

lib.layernorm_naive_serial_matched_precision.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.layernorm_naive_serial_matched_precision.restype = None

lib.layernorm_forward_rolled_slice.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.layernorm_forward_rolled_slice.restype = None

lib.layernorm_forward_unrolled_slice.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.layernorm_forward_unrolled_slice.restype = None

lib.layernorm_backward_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_output
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # gamma
    ctypes.POINTER(ctypes.c_float),  # mean
    ctypes.POINTER(ctypes.c_float),  # rstd
    ctypes.POINTER(ctypes.c_float),  # d_input
    ctypes.POINTER(ctypes.c_float),  # d_gamma
    ctypes.POINTER(ctypes.c_float),  # d_beta
    ctypes.c_int,                    # tokens
    ctypes.c_int,                    # d_model
    ctypes.c_int,                    # aligned_embed_dim
]
lib.layernorm_backward_kernel.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def aligned_np(shape, dtype=np.float32, align=64):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-buf.ctypes.data) % align
    return buf[offset:offset + nbytes].view(dtype).reshape(shape)


def run_c_layernorm_naive(x, gamma, beta, eps=1e-5):
    T, D = x.shape
    aligned = D
    x_f = x.contiguous().float()
    g_f = gamma.contiguous().float()
    b_f = beta.contiguous().float()

    out = torch.empty_like(x_f)
    mean = torch.empty(T, dtype=torch.float32)
    rstd = torch.empty(T, dtype=torch.float32)

    lib.layernorm_naive_serial(
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(b_f),
        tensor_to_ptr(out),
        tensor_to_ptr(mean),
        tensor_to_ptr(rstd),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned),
        ctypes.c_float(eps),
    )
    return out, mean, rstd


def run_c_layernorm_rolled(x, gamma, beta, eps=1e-5):
    T, D = x.shape
    aligned = D
    x_f = x.contiguous().float()
    g_f = gamma.contiguous().float()
    b_f = beta.contiguous().float()

    out = torch.empty_like(x_f)
    mean = torch.empty(T, dtype=torch.float32)
    rstd = torch.empty(T, dtype=torch.float32)

    lib.layernorm_forward_rolled_slice(
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(b_f),
        tensor_to_ptr(out),
        tensor_to_ptr(mean),
        tensor_to_ptr(rstd),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(aligned),
        ctypes.c_float(eps),
    )
    return out, mean, rstd


def run_c_layernorm_unrolled(x, gamma, beta, eps=1e-5):
    T, D = x.shape
    x_f = x.contiguous().float()
    g_f = gamma.contiguous().float()
    b_f = beta.contiguous().float()

    out = torch.empty_like(x_f)
    mean = torch.empty(T, dtype=torch.float32)
    rstd = torch.empty(T, dtype=torch.float32)

    lib.layernorm_forward_unrolled_slice(
        tensor_to_ptr(x_f),
        tensor_to_ptr(g_f),
        tensor_to_ptr(b_f),
        tensor_to_ptr(out),
        tensor_to_ptr(mean),
        tensor_to_ptr(rstd),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_float(eps),
    )
    return out, mean, rstd


def run_c_layernorm_backward(upstream, x, gamma, mean, rstd):
    T, D = x.shape

    d_input = torch.zeros_like(x)
    d_gamma = torch.zeros_like(gamma)
    d_beta = torch.zeros_like(gamma)

    lib.layernorm_backward_kernel(
        tensor_to_ptr(upstream),
        tensor_to_ptr(x),
        tensor_to_ptr(gamma),
        tensor_to_ptr(mean),
        tensor_to_ptr(rstd),
        tensor_to_ptr(d_input),
        tensor_to_ptr(d_gamma),
        tensor_to_ptr(d_beta),
        ctypes.c_int(T),
        ctypes.c_int(D),
        ctypes.c_int(D),
    )
    return d_input, d_gamma, d_beta


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def ggml_layernorm_avx2_reference(x, gamma, beta, eps):
    """Reproduce ggml's ordered AVX2 LayerNorm arithmetic."""
    tokens, d_model = x.shape
    if d_model % 8:
        raise ValueError("the AVX2 oracle requires a width divisible by 8")

    sqrtf = ctypes.CDLL("libm.so.6").sqrtf
    sqrtf.argtypes = [ctypes.c_float]
    sqrtf.restype = ctypes.c_float
    output = np.empty_like(x)

    for token in range(tokens):
        sum_acc = 0.0
        for value in x[token]:
            sum_acc += float(value)
        mean = np.float32(np.float32(sum_acc) / np.float32(d_model))
        centered = np.subtract(x[token], mean, dtype=np.float32)

        variance_acc = 0.0
        for offset in range(0, d_model, 8):
            squared = np.multiply(
                centered[offset:offset + 8],
                centered[offset:offset + 8],
                dtype=np.float32,
            )
            lanes = np.add(squared[:4], squared[4:], dtype=np.float32)
            lanes = np.add(lanes[:2], lanes[2:], dtype=np.float32)
            chunk_sum = np.float32(lanes[0] + lanes[1])
            variance_acc += float(chunk_sum)

        variance = np.float32(variance_acc / float(d_model))
        variance_eps = np.float32(variance + np.float32(eps))
        scale = np.float32(
            np.float32(1.0) /
            np.float32(sqrtf(ctypes.c_float(variance_eps)))
        )
        normalized = np.multiply(centered, scale, dtype=np.float32)
        normalized = np.multiply(normalized, gamma, dtype=np.float32)
        output[token] = np.add(normalized, beta, dtype=np.float32)

    return output


def test_ggml_exact_layernorm_production_width():
    """Guard the exact provider at Qwen3-VL's production hidden width."""
    rng = np.random.default_rng(21)
    tokens, d_model = 32, 1152
    x = rng.standard_normal((tokens, d_model), dtype=np.float32)
    gamma = rng.standard_normal(d_model, dtype=np.float32)
    beta = rng.standard_normal(d_model, dtype=np.float32)
    output = np.empty_like(x)

    lib.layernorm_naive_serial_matched_precision(
        numpy_to_ptr(x),
        numpy_to_ptr(gamma),
        numpy_to_ptr(beta),
        numpy_to_ptr(output),
        None,
        None,
        tokens,
        d_model,
        ctypes.c_float(1e-6),
    )
    reference = ggml_layernorm_avx2_reference(x, gamma, beta, 1e-6)
    differing = int(np.count_nonzero(
        output.view(np.uint32) != reference.view(np.uint32)
    ))
    diff = float(np.max(np.abs(output - reference)))
    print(
        "GGML-exact LayerNorm 32x1152: "
        f"different={differing}/{output.size}, max_diff={diff:.2e}"
    )
    if differing:
        raise AssertionError(
            "matched LayerNorm changed ggml's ordered AVX2 arithmetic"
        )


def run_forward_tests(T=32, D=128, eps=1e-5, warmup=10, iterations=1000):
    """Run forward pass tests with accuracy and timing."""
    torch.manual_seed(0)

    # Pre-allocate numpy arrays for accurate timing (avoids ctypes overhead)
    x_np = aligned_np((T, D)); x_np[:] = np.random.randn(T, D).astype(np.float32)
    gamma_np = aligned_np((D,)); gamma_np[:] = np.random.randn(D).astype(np.float32)
    beta_np = aligned_np((D,)); beta_np[:] = np.random.randn(D).astype(np.float32)
    out_np = aligned_np((T, D)); out_np.fill(0.0)
    mean_np = aligned_np((T,)); mean_np.fill(0.0)
    rstd_np = aligned_np((T,)); rstd_np.fill(0.0)

    # Get pointers once
    x_ptr = numpy_to_ptr(x_np)
    g_ptr = numpy_to_ptr(gamma_np)
    b_ptr = numpy_to_ptr(beta_np)
    o_ptr = numpy_to_ptr(out_np)
    m_ptr = numpy_to_ptr(mean_np)
    r_ptr = numpy_to_ptr(rstd_np)

    # Torch tensors for PyTorch comparison
    x = torch.from_numpy(x_np)
    gamma = torch.from_numpy(gamma_np)
    beta = torch.from_numpy(beta_np)

    report = TestReport(
        test_name="LayerNorm Forward",
        dtype="fp32",
        shape=f"tokens={T}, d_model={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    ref = torch.layer_norm(x, (D,), gamma, beta, eps)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: torch.layer_norm(x, (D,), gamma, beta, eps),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # Test naive kernel (with pre-allocated buffers for timing)
    lib.layernorm_naive_serial(x_ptr, g_ptr, b_ptr, o_ptr, m_ptr, r_ptr, T, D, D, ctypes.c_float(eps))
    out_naive = torch.from_numpy(out_np.copy())
    diff_naive = max_diff(out_naive, ref)
    naive_time = time_function(
        lambda: lib.layernorm_naive_serial(x_ptr, g_ptr, b_ptr, o_ptr, m_ptr, r_ptr, T, D, D, ctypes.c_float(eps)),
        warmup=warmup, iterations=iterations, name="Naive"
    )
    report.add_result(TestResult(
        name="Naive (scalar)",
        passed=diff_naive <= 1e-5,
        max_diff=diff_naive,
        tolerance=1e-5,
        pytorch_time=pytorch_time,
        kernel_time=naive_time
    ))

    # Test rolled kernel
    lib.layernorm_forward_rolled_slice(x_ptr, g_ptr, b_ptr, o_ptr, m_ptr, r_ptr, T, D, D, ctypes.c_float(eps))
    out_rolled = torch.from_numpy(out_np.copy())
    diff_rolled = max_diff(out_rolled, ref)
    rolled_time = time_function(
        lambda: lib.layernorm_forward_rolled_slice(x_ptr, g_ptr, b_ptr, o_ptr, m_ptr, r_ptr, T, D, D, ctypes.c_float(eps)),
        warmup=warmup, iterations=iterations, name="Rolled"
    )
    report.add_result(TestResult(
        name="Rolled (SIMD)",
        passed=diff_rolled <= 1e-5,
        max_diff=diff_rolled,
        tolerance=1e-5,
        pytorch_time=pytorch_time,
        kernel_time=rolled_time
    ))

    # Test unrolled kernel
    lib.layernorm_forward_unrolled_slice(x_ptr, g_ptr, b_ptr, o_ptr, m_ptr, r_ptr, T, D, ctypes.c_float(eps))
    out_unrolled = torch.from_numpy(out_np.copy())
    diff_unrolled = max_diff(out_unrolled, ref)
    unrolled_time = time_function(
        lambda: lib.layernorm_forward_unrolled_slice(x_ptr, g_ptr, b_ptr, o_ptr, m_ptr, r_ptr, T, D, ctypes.c_float(eps)),
        warmup=warmup, iterations=iterations, name="Unrolled"
    )
    report.add_result(TestResult(
        name="Unrolled (SIMD)",
        passed=diff_unrolled <= 1e-5,
        max_diff=diff_unrolled,
        tolerance=1e-5,
        pytorch_time=pytorch_time,
        kernel_time=unrolled_time
    ))

    return report


def run_backward_tests(T=32, D=128, eps=1e-5, warmup=10, iterations=1000):
    """Run backward pass tests with accuracy and timing (forward, backward, and combined)."""
    np.random.seed(1)

    # Pre-allocate numpy arrays
    x_np = aligned_np((T, D)); x_np[:] = np.random.randn(T, D).astype(np.float32)
    gamma_np = aligned_np((D,)); gamma_np[:] = np.random.randn(D).astype(np.float32)
    beta_np = aligned_np((D,)); beta_np[:] = np.random.randn(D).astype(np.float32)
    upstream_np = aligned_np((T, D)); upstream_np[:] = np.random.randn(T, D).astype(np.float32)
    out_np = aligned_np((T, D)); out_np.fill(0.0)
    mean_np = aligned_np((T,)); mean_np.fill(0.0)
    rstd_np = aligned_np((T,)); rstd_np.fill(0.0)
    d_input_np = aligned_np((T, D)); d_input_np.fill(0.0)
    d_gamma_np = aligned_np((D,)); d_gamma_np.fill(0.0)
    d_beta_np = aligned_np((D,)); d_beta_np.fill(0.0)

    # Get pointers
    x_ptr = numpy_to_ptr(x_np)
    g_ptr = numpy_to_ptr(gamma_np)
    b_ptr = numpy_to_ptr(beta_np)
    o_ptr = numpy_to_ptr(out_np)
    m_ptr = numpy_to_ptr(mean_np)
    r_ptr = numpy_to_ptr(rstd_np)
    up_ptr = numpy_to_ptr(upstream_np)
    di_ptr = numpy_to_ptr(d_input_np)
    dg_ptr = numpy_to_ptr(d_gamma_np)
    db_ptr = numpy_to_ptr(d_beta_np)

    # Torch tensors
    x = torch.from_numpy(x_np)
    gamma = torch.from_numpy(gamma_np)
    beta = torch.from_numpy(beta_np)
    upstream = torch.from_numpy(upstream_np)

    report = TestReport(
        test_name="LayerNorm Backward",
        dtype="fp32",
        shape=f"tokens={T}, d_model={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch forward only
    def pytorch_forward():
        return torch.layer_norm(x, (D,), gamma, beta, eps)

    # PyTorch forward+backward
    def pytorch_fwd_bwd():
        x_ref = x.clone().detach().requires_grad_(True)
        gamma_ref = gamma.clone().detach().requires_grad_(True)
        beta_ref = beta.clone().detach().requires_grad_(True)
        y_ref = torch.layer_norm(x_ref, (D,), gamma_ref, beta_ref, eps)
        y_ref.backward(upstream)
        return x_ref.grad, gamma_ref.grad, beta_ref.grad

    # Get reference grads for accuracy check
    dx_ref, dgamma_ref, dbeta_ref = pytorch_fwd_bwd()

    # C forward
    def c_forward():
        lib.layernorm_naive_serial(x_ptr, g_ptr, b_ptr, o_ptr, m_ptr, r_ptr, T, D, D, ctypes.c_float(eps))

    # C backward only
    def c_backward():
        lib.layernorm_backward_kernel(up_ptr, x_ptr, g_ptr, m_ptr, r_ptr, di_ptr, dg_ptr, db_ptr, T, D, D)

    # C forward + backward
    def c_fwd_bwd():
        lib.layernorm_naive_serial(x_ptr, g_ptr, b_ptr, o_ptr, m_ptr, r_ptr, T, D, D, ctypes.c_float(eps))
        lib.layernorm_backward_kernel(up_ptr, x_ptr, g_ptr, m_ptr, r_ptr, di_ptr, dg_ptr, db_ptr, T, D, D)

    # Run once for accuracy
    c_forward()
    c_backward()
    d_input = torch.from_numpy(d_input_np.copy())
    d_gamma = torch.from_numpy(d_gamma_np.copy())
    d_beta = torch.from_numpy(d_beta_np.copy())

    # Timing
    pt_fwd_time = time_function(pytorch_forward, warmup=warmup, iterations=iterations, name="PyTorch Fwd")
    pt_fwd_bwd_time = time_function(pytorch_fwd_bwd, warmup=warmup, iterations=iterations, name="PyTorch Fwd+Bwd")
    c_fwd_time = time_function(c_forward, warmup=warmup, iterations=iterations, name="C Fwd")
    c_bwd_time = time_function(c_backward, warmup=warmup, iterations=iterations, name="C Bwd")
    c_fwd_bwd_time = time_function(c_fwd_bwd, warmup=warmup, iterations=iterations, name="C Fwd+Bwd")

    # PyTorch backward only (estimated by subtracting forward from fwd+bwd)
    pt_bwd_est = pt_fwd_bwd_time.mean_us - pt_fwd_time.mean_us

    # Accuracy results
    diff_d_input = max_diff(d_input, dx_ref)
    diff_d_gamma = max_diff(d_gamma, dgamma_ref)
    diff_d_beta = max_diff(d_beta, dbeta_ref)

    report.add_result(TestResult(
        name="d_input",
        passed=diff_d_input <= 5e-6,
        max_diff=diff_d_input,
        tolerance=5e-6,
        pytorch_time=None,
        kernel_time=None
    ))
    report.add_result(TestResult(
        name="d_gamma",
        passed=diff_d_gamma <= 5e-6,
        max_diff=diff_d_gamma,
        tolerance=5e-6,
        pytorch_time=None,
        kernel_time=None
    ))
    report.add_result(TestResult(
        name="d_beta",
        passed=diff_d_beta <= 5e-6,
        max_diff=diff_d_beta,
        tolerance=5e-6,
        pytorch_time=None,
        kernel_time=None
    ))

    # Store timing data for custom print
    report.timing_breakdown = {
        'pt_fwd': pt_fwd_time.mean_us,
        'pt_bwd_est': pt_bwd_est,
        'pt_fwd_bwd': pt_fwd_bwd_time.mean_us,
        'c_fwd': c_fwd_time.mean_us,
        'c_bwd': c_bwd_time.mean_us,
        'c_fwd_bwd': c_fwd_bwd_time.mean_us,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    test_ggml_exact_layernorm_production_width()

    # Forward tests
    fwd_report = run_forward_tests(T=32, D=128, warmup=10, iterations=1000)
    fwd_report.print_report()

    # Backward tests (same shape as forward for fair comparison)
    bwd_report = run_backward_tests(T=32, D=128, warmup=10, iterations=1000)
    bwd_report.print_report()

    # Print detailed timing breakdown
    if hasattr(bwd_report, 'timing_breakdown'):
        t = bwd_report.timing_breakdown
        print("  DETAILED TIMING BREAKDOWN (Forward vs Backward)")
        print("  " + "-" * 55)
        print(f"  {'Operation':<20} {'PyTorch (us)':<15} {'C Kernel (us)':<15} {'Speedup':<10}")
        print("  " + "-" * 55)
        print(f"  {'Forward':<20} {t['pt_fwd']:<15.1f} {t['c_fwd']:<15.1f} {t['pt_fwd']/t['c_fwd']:.2f}x")
        print(f"  {'Backward (est)':<20} {t['pt_bwd_est']:<15.1f} {t['c_bwd']:<15.1f} {t['pt_bwd_est']/t['c_bwd']:.2f}x")
        print(f"  {'Forward+Backward':<20} {t['pt_fwd_bwd']:<15.1f} {t['c_fwd_bwd']:<15.1f} {t['pt_fwd_bwd']/t['c_fwd_bwd']:.2f}x")
        print("  " + "-" * 55)
        print()

    # Exit with error if any tests failed
    if not fwd_report.all_passed() or not bwd_report.all_passed():
        exit(1)
