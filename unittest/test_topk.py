"""
Top-K selection kernel unit tests with performance metrics.

Tests top-K and argmax operations against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes

import numpy as np
import torch
import torch.nn.functional as F

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_engine.so")

# =============================================================================
# Function signatures
# =============================================================================

lib.topk_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.c_int,                     # n
    ctypes.c_int,                     # k
    ctypes.POINTER(ctypes.c_int),    # indices
    ctypes.POINTER(ctypes.c_float),  # values (can be NULL)
]
lib.topk_f32.restype = None

lib.topk_softmax_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.c_int,                     # n
    ctypes.c_int,                     # k
    ctypes.POINTER(ctypes.c_int),    # indices
    ctypes.POINTER(ctypes.c_float),  # weights
]
lib.topk_softmax_f32.restype = None

lib.topk_batched_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # scores [num_tokens, n_experts]
    ctypes.c_int,                     # num_tokens
    ctypes.c_int,                     # n_experts
    ctypes.c_int,                     # k
    ctypes.POINTER(ctypes.c_int),    # indices [num_tokens, k]
    ctypes.POINTER(ctypes.c_float),  # weights [num_tokens, k] (can be NULL)
]
lib.topk_batched_f32.restype = None

lib.argmax_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # scores
    ctypes.c_int,                     # n
]
lib.argmax_f32.restype = ctypes.c_int


# =============================================================================
# Helper functions
# =============================================================================

def numpy_int_ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_int):
    """Convert numpy int32 array to ctypes int pointer."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


# =============================================================================
# Tests
# =============================================================================

def run_topk_tests(n=64, k=8, warmup=10, iterations=1000):
    """Run top-K selection tests with accuracy and timing."""
    np.random.seed(42)

    # Generate random scores (simulating router logits)
    scores_np = np.random.randn(n).astype(np.float32)
    scores_ptr = numpy_to_ptr(scores_np)

    # Output buffers
    indices_np = np.zeros(k, dtype=np.int32)
    values_np = np.zeros(k, dtype=np.float32)
    indices_ptr = numpy_int_ptr(indices_np)
    values_ptr = numpy_to_ptr(values_np)

    # PyTorch reference
    scores_torch = torch.from_numpy(scores_np.copy())

    report = TestReport(
        test_name="Top-K Selection",
        dtype="fp32",
        shape=f"n={n}, k={k}",
        cpu_info=get_cpu_info()
    )

    # === Test 1: topk_f32 ===
    def pytorch_topk():
        return torch.topk(scores_torch, k, dim=0, largest=True, sorted=True)

    def c_topk():
        lib.topk_f32(scores_ptr, n, k, indices_ptr, values_ptr)

    # Run once for accuracy
    c_topk()
    pt_values, pt_indices = pytorch_topk()

    # Check indices (may be in different order if values are equal)
    c_indices = indices_np.copy()
    c_values = values_np.copy()

    # Compare values (should match exactly since we're selecting same top-k)
    # Sort both by index for comparison
    pt_sorted = sorted(zip(pt_indices.numpy(), pt_values.numpy()), key=lambda x: x[0])
    c_sorted = sorted(zip(c_indices, c_values), key=lambda x: x[0])

    values_match = np.allclose([v for _, v in pt_sorted], [v for _, v in c_sorted], rtol=1e-5)
    indices_match = set(pt_indices.numpy().tolist()) == set(c_indices.tolist())

    # Timing
    pt_time = time_function(pytorch_topk, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_topk, warmup=warmup, iterations=iterations, name="C topk_f32")

    report.add_result(TestResult(
        name="topk_f32",
        passed=values_match and indices_match,
        max_diff=0.0 if values_match else 1.0,
        tolerance=1e-5,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    return report


def run_topk_softmax_tests(n=64, k=8, warmup=10, iterations=1000):
    """Run top-K with softmax normalization tests."""
    np.random.seed(43)

    scores_np = np.random.randn(n).astype(np.float32)
    scores_ptr = numpy_to_ptr(scores_np)

    indices_np = np.zeros(k, dtype=np.int32)
    weights_np = np.zeros(k, dtype=np.float32)
    indices_ptr = numpy_int_ptr(indices_np)
    weights_ptr = numpy_to_ptr(weights_np)

    scores_torch = torch.from_numpy(scores_np.copy())

    report = TestReport(
        test_name="Top-K Softmax",
        dtype="fp32",
        shape=f"n={n}, k={k}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference: topk then softmax
    def pytorch_topk_softmax():
        values, indices = torch.topk(scores_torch, k, dim=0, largest=True, sorted=True)
        weights = F.softmax(values, dim=0)
        return indices, weights

    def c_topk_softmax():
        lib.topk_softmax_f32(scores_ptr, n, k, indices_ptr, weights_ptr)

    # Run once for accuracy
    c_topk_softmax()
    pt_indices, pt_weights = pytorch_topk_softmax()

    c_indices = indices_np.copy()
    c_weights = weights_np.copy()

    # Check that weights sum to 1
    weight_sum = c_weights.sum()
    weights_normalized = abs(weight_sum - 1.0) < 1e-5

    # Check that we selected the same indices (as a set)
    indices_match = set(pt_indices.numpy().tolist()) == set(c_indices.tolist())

    # Check weights for matching indices
    weights_match = True
    if indices_match:
        # Compare weights for each index
        pt_idx_weight = {int(idx): float(w) for idx, w in zip(pt_indices, pt_weights)}
        c_idx_weight = {int(idx): float(w) for idx, w in zip(c_indices, c_weights)}
        for idx in pt_idx_weight:
            if abs(pt_idx_weight[idx] - c_idx_weight[idx]) > 1e-4:
                weights_match = False
                break

    # Timing
    pt_time = time_function(pytorch_topk_softmax, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_topk_softmax, warmup=warmup, iterations=iterations, name="C topk_softmax")

    max_diff_val = abs(weight_sum - 1.0) if not weights_normalized else 0.0

    report.add_result(TestResult(
        name="topk_softmax_f32",
        passed=weights_normalized and indices_match and weights_match,
        max_diff=max_diff_val,
        tolerance=1e-4,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    return report


def run_argmax_tests(n=1024, warmup=10, iterations=1000):
    """Run argmax tests with accuracy and timing."""
    np.random.seed(44)

    scores_np = np.random.randn(n).astype(np.float32)
    scores_ptr = numpy_to_ptr(scores_np)

    scores_torch = torch.from_numpy(scores_np.copy())

    report = TestReport(
        test_name="Argmax",
        dtype="fp32",
        shape=f"n={n}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    def pytorch_argmax():
        return torch.argmax(scores_torch).item()

    def c_argmax():
        return lib.argmax_f32(scores_ptr, n)

    # Check accuracy
    pt_result = pytorch_argmax()
    c_result = c_argmax()
    indices_match = pt_result == c_result

    # Timing
    pt_time = time_function(pytorch_argmax, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_argmax, warmup=warmup, iterations=iterations, name="C argmax")

    report.add_result(TestResult(
        name="argmax_f32",
        passed=indices_match,
        max_diff=0.0 if indices_match else 1.0,
        tolerance=0.0,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    return report


def run_batched_topk_tests(num_tokens=32, n_experts=8, k=2, warmup=10, iterations=1000):
    """Run batched top-K tests (MoE-style routing)."""
    np.random.seed(45)

    # Router logits: [num_tokens, n_experts]
    scores_np = np.random.randn(num_tokens, n_experts).astype(np.float32)
    scores_ptr = numpy_to_ptr(scores_np)

    indices_np = np.zeros((num_tokens, k), dtype=np.int32)
    weights_np = np.zeros((num_tokens, k), dtype=np.float32)
    indices_ptr = numpy_int_ptr(indices_np)
    weights_ptr = numpy_to_ptr(weights_np)

    scores_torch = torch.from_numpy(scores_np.copy())

    report = TestReport(
        test_name="Batched Top-K (MoE Router)",
        dtype="fp32",
        shape=f"tokens={num_tokens}, experts={n_experts}, k={k}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    def pytorch_batched_topk():
        values, indices = torch.topk(scores_torch, k, dim=1, largest=True, sorted=True)
        weights = F.softmax(values, dim=1)
        return indices, weights

    def c_batched_topk():
        lib.topk_batched_f32(scores_ptr, num_tokens, n_experts, k, indices_ptr, weights_ptr)

    # Run once for accuracy
    c_batched_topk()
    pt_indices, pt_weights = pytorch_batched_topk()

    c_indices = indices_np.copy().reshape(num_tokens, k)
    c_weights = weights_np.copy().reshape(num_tokens, k)

    # Check each token
    all_match = True
    max_weight_diff = 0.0
    for t in range(num_tokens):
        pt_set = set(pt_indices[t].numpy().tolist())
        c_set = set(c_indices[t].tolist())
        if pt_set != c_set:
            all_match = False
            break

        # Check weight differences for matching indices
        for i, idx in enumerate(c_indices[t]):
            pt_idx_pos = (pt_indices[t] == idx).nonzero(as_tuple=True)[0]
            if len(pt_idx_pos) > 0:
                pt_w = pt_weights[t, pt_idx_pos[0]].item()
                c_w = c_weights[t, i]
                diff = abs(pt_w - c_w)
                max_weight_diff = max(max_weight_diff, diff)

    passed = all_match and max_weight_diff < 1e-4

    # Timing
    pt_time = time_function(pytorch_batched_topk, warmup=warmup, iterations=iterations, name="PyTorch")
    c_time = time_function(c_batched_topk, warmup=warmup, iterations=iterations, name="C batched_topk")

    report.add_result(TestResult(
        name="topk_batched_f32",
        passed=passed,
        max_diff=max_weight_diff,
        tolerance=1e-4,
        pytorch_time=pt_time,
        kernel_time=c_time
    ))

    return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print_system_info()

    # Basic top-K tests
    topk_report = run_topk_tests(n=64, k=8, warmup=10, iterations=1000)
    topk_report.print_report()

    # Top-K with softmax
    softmax_report = run_topk_softmax_tests(n=64, k=8, warmup=10, iterations=1000)
    softmax_report.print_report()

    # Argmax
    argmax_report = run_argmax_tests(n=1024, warmup=10, iterations=1000)
    argmax_report.print_report()

    # Batched (MoE-style)
    batched_report = run_batched_topk_tests(num_tokens=32, n_experts=8, k=2, warmup=10, iterations=1000)
    batched_report.print_report()

    # Exit with error if any tests failed
    all_passed = (
        topk_report.all_passed() and
        softmax_report.all_passed() and
        argmax_report.all_passed() and
        batched_report.all_passed()
    )
    if not all_passed:
        exit(1)
