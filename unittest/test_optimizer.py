"""
Optimizer kernel unit tests with performance metrics.

Tests AdamW, SGD, and gradient operations against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np
import torch

from lib_loader import load_lib, find_lib, _root_dir
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info, TimingResult
)


# ═══════════════════════════════════════════════════════════════════════════════
# Library Compilation and Loading
# ═══════════════════════════════════════════════════════════════════════════════

def compile_optimizer_lib():
    """Compile optimizer kernels into a shared library."""
    root = _root_dir()
    src_dir = os.path.join(root, "src", "kernels")
    core_src_dir = os.path.join(root, "src")
    include_dir = os.path.join(root, "include")
    build_dir = os.path.join(root, "build")
    lib_path = os.path.join(build_dir, "libckernel_optimizer.so")

    os.makedirs(build_dir, exist_ok=True)

    sources = [
        os.path.join(src_dir, "optimizer_kernels.c"),
        os.path.join(src_dir, "optimizer_kernels_bf16.c"),
        # Provides ck_strict_parity_enabled() and ck_get_physical_cores().
        os.path.join(core_src_dir, "ckernel_strict.c"),
        # Optimizer kernels call ck_threadpool_global() for large tensors.
        os.path.join(core_src_dir, "ck_threadpool.c"),
    ]

    # Check sources exist
    for src in sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source not found: {src}")

    # Detect CPU features
    cpu_features = ""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpu_features = f.read()
    except:
        pass

    avx_flags = []
    # Only enable FMA if CPU supports it
    has_fma = " fma " in cpu_features or cpu_features.endswith(" fma")

    if "avx512f" in cpu_features:
        avx_flags = ["-mavx512f"]
        if has_fma:
            avx_flags.append("-mfma")
    elif "avx2" in cpu_features:
        avx_flags = ["-mavx2"]
        if has_fma:
            avx_flags.append("-mfma")
    elif "avx " in cpu_features or cpu_features.endswith("avx"):
        avx_flags = ["-mavx"]
        if has_fma:
            avx_flags.append("-mfma")

    cmd = [
        "gcc", "-O3", "-fPIC", "-shared", "-Wall",
    ] + avx_flags + [
        f"-I{include_dir}",
        "-o", lib_path,
    ] + sources + ["-lm", "-lpthread"]

    print(f"  Compiling optimizer library...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try without SIMD
        cmd = ["gcc", "-O3", "-fPIC", "-shared", f"-I{include_dir}", "-o", lib_path] + sources + ["-lm", "-lpthread"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")

    print(f"  Compiled: {lib_path}")
    return lib_path


def load_optimizer_lib():
    """Load the optimizer library, compiling if needed."""
    try:
        lib = load_lib("libckernel_optimizer.so")
    except (FileNotFoundError, OSError):
        compile_optimizer_lib()
        lib = load_lib("libckernel_optimizer.so")

    # Define function signatures

    # AdamW fp32
    lib.adamw_update_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # grad
        ctypes.POINTER(ctypes.c_float),  # weight
        ctypes.POINTER(ctypes.c_float),  # m
        ctypes.POINTER(ctypes.c_float),  # v
        ctypes.c_size_t,                  # numel
        ctypes.c_float,                   # lr
        ctypes.c_float,                   # beta1
        ctypes.c_float,                   # beta2
        ctypes.c_float,                   # eps
        ctypes.c_float,                   # weight_decay
        ctypes.c_int,                     # step
    ]
    lib.adamw_update_f32.restype = None

    # AdamW bf16
    lib.adamw_update_bf16.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),  # grad (bf16)
        ctypes.POINTER(ctypes.c_uint16),  # weight (bf16)
        ctypes.POINTER(ctypes.c_float),   # m (fp32)
        ctypes.POINTER(ctypes.c_float),   # v (fp32)
        ctypes.c_size_t,                   # numel
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_int,
    ]
    lib.adamw_update_bf16.restype = None

    # SGD momentum fp32
    lib.sgd_momentum_update_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # grad
        ctypes.POINTER(ctypes.c_float),  # weight
        ctypes.POINTER(ctypes.c_float),  # velocity
        ctypes.c_size_t,                  # numel
        ctypes.c_float,                   # lr
        ctypes.c_float,                   # momentum
        ctypes.c_float,                   # weight_decay
    ]
    lib.sgd_momentum_update_f32.restype = None

    # Gradient ops
    lib.zero_gradients_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.zero_gradients_f32.restype = None

    lib.gradient_accumulate_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.gradient_accumulate_f32.restype = None

    lib.gradient_scale_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    lib.gradient_scale_f32.restype = None

    lib.gradient_clip_norm_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    lib.gradient_clip_norm_f32.restype = ctypes.c_float

    return lib


# Load library at module level
lib = load_optimizer_lib()


# ═══════════════════════════════════════════════════════════════════════════════
# BF16 Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def float32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Convert float32 to bfloat16 (stored as uint16)."""
    x32 = x.astype(np.float32)
    x_int = x32.view(np.uint32)
    lsb = (x_int >> 16) & 1
    rounding = 0x7FFF + lsb
    x_int = x_int + rounding
    return (x_int >> 16).astype(np.uint16)


def bf16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) to float32."""
    x32 = x.astype(np.uint32) << 16
    return x32.view(np.float32)


def numpy_to_ptr_u16(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_uint16):
    """Convert numpy uint16 array to ctypes pointer."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch Reference Implementations
# ═══════════════════════════════════════════════════════════════════════════════

def pytorch_adamw_step(grad, weight, m, v, lr, beta1, beta2, eps, weight_decay, step):
    """Reference AdamW implementation."""
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad

    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2

    weight_new = weight - lr * (m_hat / (torch.sqrt(v_hat) + eps) + weight_decay * weight)

    return weight_new, m_new, v_new


def pytorch_sgd_momentum_step(grad, weight, velocity, lr, momentum, weight_decay):
    """Reference SGD with momentum implementation."""
    velocity_new = momentum * velocity + grad
    weight_new = weight - lr * (velocity_new + weight_decay * weight)
    return weight_new, velocity_new


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_adamw_tests(N=4096, warmup=10, iterations=100):
    """Run AdamW optimizer tests with accuracy and timing."""
    np.random.seed(42)

    report = TestReport(
        test_name="AdamW Optimizer",
        dtype="fp32 / bf16",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    # Hyperparameters
    lr, beta1, beta2, eps, weight_decay = 0.001, 0.9, 0.999, 1e-8, 0.01

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: AdamW fp32 single step
    # ─────────────────────────────────────────────────────────────────────────
    grad = np.random.randn(N).astype(np.float32) * 0.1
    weight = np.random.randn(N).astype(np.float32)
    m = np.zeros(N, dtype=np.float32)
    v = np.zeros(N, dtype=np.float32)
    step = 1

    weight_c = weight.copy()
    m_c, v_c = m.copy(), v.copy()

    def c_adamw_f32():
        lib.adamw_update_f32(
            numpy_to_ptr(grad), numpy_to_ptr(weight_c),
            numpy_to_ptr(m_c), numpy_to_ptr(v_c),
            N, lr, beta1, beta2, eps, weight_decay, step
        )

    # Reset and run C kernel
    weight_c[:] = weight
    m_c[:] = m
    v_c[:] = v
    c_adamw_f32()

    # PyTorch reference
    weight_ref, m_ref, v_ref = pytorch_adamw_step(
        torch.from_numpy(grad), torch.from_numpy(weight.copy()),
        torch.from_numpy(m.copy()), torch.from_numpy(v.copy()),
        lr, beta1, beta2, eps, weight_decay, step
    )

    diff_w = np.max(np.abs(weight_c - weight_ref.numpy()))
    diff_m = np.max(np.abs(m_c - m_ref.numpy()))
    diff_v = np.max(np.abs(v_c - v_ref.numpy()))

    # Timing
    def pytorch_adamw():
        pytorch_adamw_step(
            torch.from_numpy(grad), torch.from_numpy(weight.copy()),
            torch.from_numpy(m.copy()), torch.from_numpy(v.copy()),
            lr, beta1, beta2, eps, weight_decay, step
        )

    # Reset for timing
    weight_c[:] = weight
    m_c[:] = m
    v_c[:] = v

    pt_time = time_function(pytorch_adamw, warmup, iterations, "PyTorch")
    ck_time = time_function(c_adamw_f32, warmup, iterations, "C AdamW")

    report.add_result(TestResult(
        name="fp32 (step=1)",
        passed=diff_w < 1e-6 and diff_m < 1e-6 and diff_v < 1e-6,
        max_diff=max(diff_w, diff_m, diff_v),
        tolerance=1e-6,
        pytorch_time=pt_time,
        kernel_time=ck_time
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: AdamW fp32 multiple steps (accumulated error)
    # ─────────────────────────────────────────────────────────────────────────
    weight_c = np.random.randn(N).astype(np.float32)
    m_c = np.zeros(N, dtype=np.float32)
    v_c = np.zeros(N, dtype=np.float32)

    weight_t = torch.from_numpy(weight_c.copy())
    m_t = torch.zeros(N)
    v_t = torch.zeros(N)

    num_steps = 10
    for s in range(1, num_steps + 1):
        g = np.random.randn(N).astype(np.float32) * 0.1
        g_t = torch.from_numpy(g.copy())

        lib.adamw_update_f32(
            numpy_to_ptr(g), numpy_to_ptr(weight_c),
            numpy_to_ptr(m_c), numpy_to_ptr(v_c),
            N, lr, beta1, beta2, eps, weight_decay, s
        )

        weight_t, m_t, v_t = pytorch_adamw_step(
            g_t, weight_t, m_t, v_t,
            lr, beta1, beta2, eps, weight_decay, s
        )

    diff_multi = np.max(np.abs(weight_c - weight_t.numpy()))

    report.add_result(TestResult(
        name=f"fp32 ({num_steps} steps)",
        passed=diff_multi < 1e-5,
        max_diff=diff_multi,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=None
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: AdamW bf16
    # ─────────────────────────────────────────────────────────────────────────
    grad_f32 = np.random.randn(N).astype(np.float32) * 0.1
    weight_f32 = np.random.randn(N).astype(np.float32)

    grad_bf16 = float32_to_bf16(grad_f32)
    weight_bf16 = float32_to_bf16(weight_f32)
    m_bf16 = np.zeros(N, dtype=np.float32)
    v_bf16 = np.zeros(N, dtype=np.float32)

    lib.adamw_update_bf16(
        numpy_to_ptr_u16(grad_bf16), numpy_to_ptr_u16(weight_bf16),
        numpy_to_ptr(m_bf16), numpy_to_ptr(v_bf16),
        N, lr, beta1, beta2, eps, weight_decay, 1
    )

    weight_c_bf16 = bf16_to_float32(weight_bf16)

    # Reference using bf16-converted inputs
    grad_ref = bf16_to_float32(float32_to_bf16(grad_f32))
    weight_ref = bf16_to_float32(float32_to_bf16(weight_f32))
    weight_t_bf16, _, _ = pytorch_adamw_step(
        torch.from_numpy(grad_ref), torch.from_numpy(weight_ref),
        torch.zeros(N), torch.zeros(N),
        lr, beta1, beta2, eps, weight_decay, 1
    )
    weight_ref_bf16 = bf16_to_float32(float32_to_bf16(weight_t_bf16.numpy()))

    diff_bf16 = np.max(np.abs(weight_c_bf16 - weight_ref_bf16))

    report.add_result(TestResult(
        name="bf16 weights",
        passed=diff_bf16 < 1e-2,
        max_diff=diff_bf16,
        tolerance=1e-2,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def run_sgd_tests(N=4096, warmup=10, iterations=100):
    """Run SGD with momentum tests."""
    np.random.seed(43)

    report = TestReport(
        test_name="SGD Momentum Optimizer",
        dtype="fp32",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    lr, momentum, weight_decay = 0.01, 0.9, 0.0001

    grad = np.random.randn(N).astype(np.float32) * 0.1
    weight = np.random.randn(N).astype(np.float32)
    velocity = np.zeros(N, dtype=np.float32)

    weight_c = weight.copy()
    velocity_c = velocity.copy()

    lib.sgd_momentum_update_f32(
        numpy_to_ptr(grad), numpy_to_ptr(weight_c), numpy_to_ptr(velocity_c),
        N, lr, momentum, weight_decay
    )

    weight_ref, velocity_ref = pytorch_sgd_momentum_step(
        torch.from_numpy(grad), torch.from_numpy(weight.copy()),
        torch.from_numpy(velocity.copy()),
        lr, momentum, weight_decay
    )

    diff_w = np.max(np.abs(weight_c - weight_ref.numpy()))
    diff_v = np.max(np.abs(velocity_c - velocity_ref.numpy()))

    # Timing
    weight_c[:] = weight
    velocity_c[:] = velocity

    def c_sgd():
        lib.sgd_momentum_update_f32(
            numpy_to_ptr(grad), numpy_to_ptr(weight_c), numpy_to_ptr(velocity_c),
            N, lr, momentum, weight_decay
        )

    def pytorch_sgd():
        pytorch_sgd_momentum_step(
            torch.from_numpy(grad), torch.from_numpy(weight.copy()),
            torch.from_numpy(velocity.copy()),
            lr, momentum, weight_decay
        )

    pt_time = time_function(pytorch_sgd, warmup, iterations, "PyTorch")
    ck_time = time_function(c_sgd, warmup, iterations, "C SGD")

    report.add_result(TestResult(
        name="fp32 single step",
        passed=diff_w < 1e-6 and diff_v < 1e-6,
        max_diff=max(diff_w, diff_v),
        tolerance=1e-6,
        pytorch_time=pt_time,
        kernel_time=ck_time
    ))

    return report


def run_gradient_ops_tests(N=8192, warmup=10, iterations=100):
    """Run gradient operation tests."""
    np.random.seed(44)

    report = TestReport(
        test_name="Gradient Operations",
        dtype="fp32",
        shape=f"N={N}",
        cpu_info=get_cpu_info()
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Gradient Accumulation
    # ─────────────────────────────────────────────────────────────────────────
    dst = np.random.randn(N).astype(np.float32)
    src = np.random.randn(N).astype(np.float32)
    expected = dst + src

    dst_c = dst.copy()
    lib.gradient_accumulate_f32(numpy_to_ptr(dst_c), numpy_to_ptr(src), N)

    diff_acc = np.max(np.abs(dst_c - expected))

    # Timing
    dst_c[:] = dst
    def c_accum():
        lib.gradient_accumulate_f32(numpy_to_ptr(dst_c), numpy_to_ptr(src), N)

    def np_accum():
        dst_c[:] += src

    np_time = time_function(np_accum, warmup, iterations, "NumPy")
    ck_time = time_function(c_accum, warmup, iterations, "C Kernel")

    report.add_result(TestResult(
        name="accumulate (+=)",
        passed=diff_acc < 1e-7,
        max_diff=diff_acc,
        tolerance=1e-7,
        pytorch_time=np_time,
        kernel_time=ck_time
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Gradient Scaling
    # ─────────────────────────────────────────────────────────────────────────
    scale = 1.0 / 16.0
    grad = np.random.randn(N).astype(np.float32)
    expected = grad * scale

    grad_c = grad.copy()
    lib.gradient_scale_f32(numpy_to_ptr(grad_c), N, scale)

    diff_scale = np.max(np.abs(grad_c - expected))

    grad_c[:] = grad
    def c_scale():
        lib.gradient_scale_f32(numpy_to_ptr(grad_c), N, scale)

    def np_scale():
        grad_c[:] *= scale

    np_time = time_function(np_scale, warmup, iterations, "NumPy")
    ck_time = time_function(c_scale, warmup, iterations, "C Kernel")

    report.add_result(TestResult(
        name="scale (*=)",
        passed=diff_scale < 1e-7,
        max_diff=diff_scale,
        tolerance=1e-7,
        pytorch_time=np_time,
        kernel_time=ck_time
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Gradient Clipping
    # ─────────────────────────────────────────────────────────────────────────
    max_norm = 1.0
    grad = np.random.randn(N).astype(np.float32) * 2.0
    original_norm = np.linalg.norm(grad)

    grad_c = grad.copy()
    returned_norm = lib.gradient_clip_norm_f32(numpy_to_ptr(grad_c), N, max_norm)
    new_norm = np.linalg.norm(grad_c)

    # Check correctness
    norm_match = abs(returned_norm - original_norm) < 1e-3
    clipped_ok = new_norm <= max_norm + 1e-5

    # Timing
    grad_c[:] = grad
    def c_clip():
        lib.gradient_clip_norm_f32(numpy_to_ptr(grad_c), N, max_norm)

    def torch_clip():
        g = torch.from_numpy(grad.copy())
        torch.nn.utils.clip_grad_norm_([g], max_norm)

    pt_time = time_function(torch_clip, warmup, iterations, "PyTorch")
    ck_time = time_function(c_clip, warmup, iterations, "C Kernel")

    report.add_result(TestResult(
        name=f"clip norm ({original_norm:.1f}→{new_norm:.3f})",
        passed=norm_match and clipped_ok,
        max_diff=abs(new_norm - max_norm),
        tolerance=1e-5,
        pytorch_time=pt_time,
        kernel_time=ck_time
    ))

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    # Run all tests
    adamw_report = run_adamw_tests(N=4096, warmup=10, iterations=100)
    adamw_report.print_report()

    sgd_report = run_sgd_tests(N=4096, warmup=10, iterations=100)
    sgd_report.print_report()

    grad_report = run_gradient_ops_tests(N=8192, warmup=10, iterations=100)
    grad_report.print_report()

    # Summary
    all_passed = (
        adamw_report.all_passed() and
        sgd_report.all_passed() and
        grad_report.all_passed()
    )

    if not all_passed:
        sys.exit(1)
