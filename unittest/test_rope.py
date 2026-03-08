"""
RoPE (Rotary Position Embedding) kernel unit tests with performance metrics.

Tests forward and backward passes against PyTorch reference.
Reports accuracy, timing, and system information.
"""
import ctypes

import numpy as np
import torch

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)


# Load the library
lib = load_lib("libckernel_rope.so", "libckernel_engine.so")

# ═══════════════════════════════════════════════════════════════════════════════
# Function signatures
# ═══════════════════════════════════════════════════════════════════════════════

lib.rope_precompute_cache.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,                    # max_seq_len
    ctypes.c_int,                    # head_dim
    ctypes.c_float,                  # base
    ctypes.c_int,                    # rotary_dim
    ctypes.c_char_p,                 # scaling_type
    ctypes.c_float,                  # scaling_factor
]
lib.rope_precompute_cache.restype = None

lib.rope_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x (in-place)
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # pos_offset
]
lib.rope_forward.restype = None

lib.rope_backward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # d_out
    ctypes.POINTER(ctypes.c_float),  # d_x (output)
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # pos_offset
]
lib.rope_backward.restype = None

lib.rope_forward_qk_with_rotary_dim.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q (in-place)
    ctypes.POINTER(ctypes.c_float),  # k (in-place)
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # pos_offset
    ctypes.c_int,                    # rotary_dim
]
lib.rope_forward_qk_with_rotary_dim.restype = None

lib.rope_forward_qk_pairwise_with_rotary_dim.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q (in-place)
    ctypes.POINTER(ctypes.c_float),  # k (in-place)
    ctypes.POINTER(ctypes.c_float),  # cos_cache
    ctypes.POINTER(ctypes.c_float),  # sin_cache
    ctypes.c_int,                    # num_heads
    ctypes.c_int,                    # num_kv_heads
    ctypes.c_int,                    # num_tokens
    ctypes.c_int,                    # head_dim
    ctypes.c_int,                    # aligned_head_dim
    ctypes.c_int,                    # pos_offset
    ctypes.c_int,                    # rotary_dim
]
lib.rope_forward_qk_pairwise_with_rotary_dim.restype = None


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementations
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_freqs_cis_pytorch(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """PyTorch reference: compute cos/sin cache for RoPE."""
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def rope_forward_pytorch(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, pos_offset: int = 0):
    """PyTorch reference RoPE forward (rotate-half)."""
    H, T, D = x.shape
    half_dim = D // 2

    x_out = x.clone()
    for h in range(H):
        for t in range(T):
            pos = pos_offset + t
            cos_row = cos_cache[pos]
            sin_row = sin_cache[pos]

            for i in range(half_dim):
                x0 = x[h, t, i]
                x1 = x[h, t, i + half_dim]
                c = cos_row[i]
                s = sin_row[i]

                x_out[h, t, i] = x0 * c - x1 * s
                x_out[h, t, i + half_dim] = x0 * s + x1 * c

    return x_out


def rope_forward_pytorch_vectorized(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, pos_offset: int = 0):
    """Vectorized PyTorch reference for fair timing comparison."""
    H, T, D = x.shape
    half_dim = D // 2

    # Get relevant cache slices
    cos = cos_cache[pos_offset:pos_offset + T]  # [T, half_dim]
    sin = sin_cache[pos_offset:pos_offset + T]  # [T, half_dim]

    # Split x into two halves
    x1 = x[..., :half_dim]  # [H, T, half_dim]
    x2 = x[..., half_dim:]  # [H, T, half_dim]

    # Apply rotation (broadcast cos/sin over heads)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)


def rope_backward_pytorch(d_out: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, pos_offset: int = 0):
    """PyTorch reference RoPE backward (inverse rotation)."""
    H, T, D = d_out.shape
    half_dim = D // 2

    d_x = torch.zeros_like(d_out)
    for h in range(H):
        for t in range(T):
            pos = pos_offset + t
            cos_row = cos_cache[pos]
            sin_row = sin_cache[pos]

            for i in range(half_dim):
                d0 = d_out[h, t, i]
                d1 = d_out[h, t, i + half_dim]
                c = cos_row[i]
                s = sin_row[i]

                d_x[h, t, i] = d0 * c + d1 * s
                d_x[h, t, i + half_dim] = -d0 * s + d1 * c

    return d_x


def rope_forward_pytorch_pairwise_vectorized(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    pos_offset: int = 0,
    rotary_dim: int | None = None,
):
    """Vectorized PyTorch reference for Llama-style even/odd RoPE."""
    H, T, D = x.shape
    rotary = D if rotary_dim is None else min(int(rotary_dim), D)
    rotary -= rotary % 2
    if rotary <= 0:
        return x.clone()

    half_dim = rotary // 2
    cos = cos_cache[pos_offset:pos_offset + T, :half_dim]
    sin = sin_cache[pos_offset:pos_offset + T, :half_dim]

    out = x.clone()
    even = x[..., 0:rotary:2]
    odd = x[..., 1:rotary:2]
    out[..., 0:rotary:2] = even * cos - odd * sin
    out[..., 1:rotary:2] = even * sin + odd * cos
    return out


def rope_forward_pairwise_numpy(
    x_np: np.ndarray,
    cos_np: np.ndarray,
    sin_np: np.ndarray,
    *,
    pos_offset: int = 0,
    rotary_dim: int | None = None,
) -> np.ndarray:
    x = torch.from_numpy(x_np.copy())
    cos_cache = torch.from_numpy(cos_np.copy())
    sin_cache = torch.from_numpy(sin_np.copy())
    return rope_forward_pytorch_pairwise_vectorized(
        x,
        cos_cache,
        sin_cache,
        pos_offset=pos_offset,
        rotary_dim=rotary_dim,
    ).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_cache_tests(max_seq_len=128, head_dim=64, warmup=10, iterations=1000):
    """Test cos/sin cache precomputation."""
    np.random.seed(0)
    half_dim = head_dim // 2

    # Pre-allocate numpy arrays
    cos_np = np.zeros((max_seq_len, half_dim), dtype=np.float32)
    sin_np = np.zeros((max_seq_len, half_dim), dtype=np.float32)

    cos_ptr = numpy_to_ptr(cos_np)
    sin_ptr = numpy_to_ptr(sin_np)

    report = TestReport(
        test_name="RoPE Cache Precompute",
        dtype="fp32",
        shape=f"max_seq_len={max_seq_len}, head_dim={head_dim}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    cos_ref, sin_ref = precompute_freqs_cis_pytorch(head_dim, max_seq_len)

    # C kernel
    def c_precompute():
        lib.rope_precompute_cache(
            cos_ptr, sin_ptr,
            ctypes.c_int(max_seq_len), ctypes.c_int(head_dim),
            ctypes.c_float(10000.0),
            ctypes.c_int(head_dim), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
        )

    c_precompute()
    cos_c = torch.from_numpy(cos_np.copy())
    sin_c = torch.from_numpy(sin_np.copy())

    diff_cos = max_diff(cos_c, cos_ref)
    diff_sin = max_diff(sin_c, sin_ref)

    cache_tolerance = 5e-6
    report.add_result(TestResult(
        name="cos_cache",
        passed=diff_cos <= cache_tolerance,
        max_diff=diff_cos,
        tolerance=cache_tolerance,
        pytorch_time=None,
        kernel_time=None
    ))

    report.add_result(TestResult(
        name="sin_cache",
        passed=diff_sin <= cache_tolerance,
        max_diff=diff_sin,
        tolerance=cache_tolerance,
        pytorch_time=None,
        kernel_time=None
    ))

    return report


def run_cache_tests_multi_theta():
    """
    Test cos/sin cache precomputation with different theta values.

    CRITICAL: Different models use different RoPE theta (base frequency):
      - Llama 2:     10,000
      - Llama 3:    500,000
      - Qwen2:    1,000,000
      - Mistral:     10,000

    This test ensures our kernel produces correct results for all variants.
    """
    report = TestReport(
        test_name="RoPE Cache Multi-Theta Parity",
        dtype="fp32",
        shape="Various theta values",
        cpu_info=get_cpu_info()
    )

    # Test configurations: (theta, model_name, head_dim, max_seq_len)
    test_configs = [
        (10000.0, "Llama2/Mistral", 64, 256),
        (10000.0, "Llama2/Mistral (128d)", 128, 256),
        (500000.0, "Llama3", 64, 256),
        (500000.0, "Llama3 (128d)", 128, 256),
        (1000000.0, "Qwen2", 64, 256),
        (1000000.0, "Qwen2 (128d)", 128, 256),
        # Edge cases
        (10000.0, "Small context", 64, 32),
        (1000000.0, "Large context", 64, 2048),
    ]

    for theta, model_name, head_dim, max_seq_len in test_configs:
        half_dim = head_dim // 2

        # Allocate buffers
        cos_np = np.zeros((max_seq_len, half_dim), dtype=np.float32)
        sin_np = np.zeros((max_seq_len, half_dim), dtype=np.float32)

        # C kernel
        lib.rope_precompute_cache(
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(max_seq_len), ctypes.c_int(head_dim),
            ctypes.c_float(theta),
            ctypes.c_int(head_dim), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
        )

        # PyTorch reference (high precision)
        cos_ref, sin_ref = precompute_freqs_cis_pytorch(head_dim, max_seq_len, base=theta)

        cos_c = torch.from_numpy(cos_np)
        sin_c = torch.from_numpy(sin_np)

        diff_cos = max_diff(cos_c, cos_ref)
        diff_sin = max_diff(sin_c, sin_ref)
        max_diff_val = max(diff_cos, diff_sin)

        # Tolerance: larger theta can have slightly more numerical error
        # due to the exp(-exponent * log_base) computation
        tolerance = 1e-5 if theta <= 100000 else 5e-5

        report.add_result(TestResult(
            name=f"{model_name} (θ={theta:.0f}, d={head_dim})",
            passed=max_diff_val <= tolerance,
            max_diff=max_diff_val,
            tolerance=tolerance,
            pytorch_time=None,
            kernel_time=None
        ))

        # Also verify specific positions have reasonable values
        # Position 0 should have cos=1, sin=0 for all frequencies
        if max_seq_len > 0:
            pos0_cos_diff = float(torch.max(torch.abs(cos_c[0] - 1.0)))
            pos0_sin_diff = float(torch.max(torch.abs(sin_c[0])))
            pos0_ok = pos0_cos_diff < 1e-6 and pos0_sin_diff < 1e-6
            if not pos0_ok:
                print(f"  WARNING: {model_name} pos=0 sanity check failed: cos_diff={pos0_cos_diff}, sin_diff={pos0_sin_diff}")

    return report


def run_rope_forward_multi_theta():
    """
    Test RoPE forward pass with different theta values.

    Ensures that applying RoPE with different base frequencies
    produces correct rotations compared to PyTorch reference.
    """
    report = TestReport(
        test_name="RoPE Forward Multi-Theta",
        dtype="fp32",
        shape="Various theta values",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (10000.0, "Llama2", 8, 32, 64),     # (theta, name, H, T, D)
        (500000.0, "Llama3", 8, 32, 64),
        (1000000.0, "Qwen2", 14, 32, 64),   # Qwen2-0.5B has 14 heads
        (1000000.0, "Qwen2-1.5B", 12, 32, 128),  # head_dim=128
    ]

    for theta, model_name, H, T, D in test_configs:
        np.random.seed(42)
        half_dim = D // 2

        # Random input
        x_np = np.random.randn(H, T, D).astype(np.float32)
        cos_np = np.zeros((T, half_dim), dtype=np.float32)
        sin_np = np.zeros((T, half_dim), dtype=np.float32)

        # Precompute cache with this theta
        lib.rope_precompute_cache(
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_float(theta),
            ctypes.c_int(D), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
        )

        # PyTorch reference
        x = torch.from_numpy(x_np.copy())
        cos_cache = torch.from_numpy(cos_np.copy())
        sin_cache = torch.from_numpy(sin_np.copy())
        ref = rope_forward_pytorch_vectorized(x, cos_cache, sin_cache)

        # C kernel (in-place)
        x_c_np = x_np.copy()
        lib.rope_forward(
            numpy_to_ptr(x_c_np), numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(0)
        )

        out = torch.from_numpy(x_c_np)
        diff = max_diff(out, ref)

        report.add_result(TestResult(
            name=f"{model_name} (θ={theta:.0f})",
            passed=diff <= 1e-5,
            max_diff=diff,
            tolerance=1e-5,
            pytorch_time=None,
            kernel_time=None
        ))

    return report


def run_rope_decode_positions_test():
    """
    Test RoPE at various decode positions (KV cache continuation).

    When decoding token N, we apply RoPE at position N, not position 0.
    This tests the pos_offset parameter works correctly.
    """
    report = TestReport(
        test_name="RoPE Decode Positions",
        dtype="fp32",
        shape="Various positions",
        cpu_info=get_cpu_info()
    )

    H, T, D = 8, 1, 64  # Single token decode
    half_dim = D // 2
    max_cache_len = 2048

    # Precompute full cache
    cos_np = np.zeros((max_cache_len, half_dim), dtype=np.float32)
    sin_np = np.zeros((max_cache_len, half_dim), dtype=np.float32)
    lib.rope_precompute_cache(
        numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
        ctypes.c_int(max_cache_len), ctypes.c_int(D), ctypes.c_float(10000.0),
        ctypes.c_int(D), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
    )

    cos_cache = torch.from_numpy(cos_np)
    sin_cache = torch.from_numpy(sin_np)

    # Test various positions
    test_positions = [0, 1, 10, 100, 500, 1000, 2000]

    for pos in test_positions:
        if pos >= max_cache_len:
            continue

        np.random.seed(pos)
        x_np = np.random.randn(H, T, D).astype(np.float32)

        # PyTorch reference
        x = torch.from_numpy(x_np.copy())
        ref = rope_forward_pytorch_vectorized(x, cos_cache, sin_cache, pos_offset=pos)

        # C kernel
        x_c_np = x_np.copy()
        lib.rope_forward(
            numpy_to_ptr(x_c_np), numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(pos)
        )

        out = torch.from_numpy(x_c_np)
        diff = max_diff(out, ref)

        report.add_result(TestResult(
            name=f"pos={pos}",
            passed=diff <= 1e-5,
            max_diff=diff,
            tolerance=1e-5,
            pytorch_time=None,
            kernel_time=None
        ))

    return report


def run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500):
    """Run forward pass tests with accuracy and timing."""
    np.random.seed(0)
    half_dim = D // 2

    # Pre-allocate numpy arrays
    x_np = np.random.randn(H, T, D).astype(np.float32)
    cos_np = np.zeros((T, half_dim), dtype=np.float32)
    sin_np = np.zeros((T, half_dim), dtype=np.float32)

    # Precompute cache
    lib.rope_precompute_cache(
        numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_float(10000.0),
        ctypes.c_int(D), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
    )

    # Torch tensors
    x = torch.from_numpy(x_np.copy())
    cos_cache = torch.from_numpy(cos_np.copy())
    sin_cache = torch.from_numpy(sin_np.copy())

    report = TestReport(
        test_name="RoPE Forward",
        dtype="fp32",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference (vectorized for fair comparison)
    ref = rope_forward_pytorch_vectorized(x, cos_cache, sin_cache)

    # Time PyTorch
    pytorch_time = time_function(
        lambda: rope_forward_pytorch_vectorized(x, cos_cache, sin_cache),
        warmup=warmup, iterations=iterations, name="PyTorch"
    )

    # C kernel (in-place)
    x_c_np = x_np.copy()
    x_c_ptr = numpy_to_ptr(x_c_np)

    def c_rope_forward():
        np.copyto(x_c_np, x_np)
        lib.rope_forward(
            x_c_ptr, numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(0)
        )

    c_rope_forward()
    out = torch.from_numpy(x_c_np.copy())
    diff = max_diff(out, ref)

    kernel_time = time_function(c_rope_forward, warmup=warmup, iterations=iterations, name="C RoPE")

    report.add_result(TestResult(
        name="RoPE Forward",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


def run_backward_tests(H=8, T=64, D=64, warmup=10, iterations=500):
    """Run backward pass tests with accuracy and timing."""
    np.random.seed(1)
    half_dim = D // 2

    # Pre-allocate numpy arrays
    d_out_np = np.random.randn(H, T, D).astype(np.float32)
    d_x_np = np.zeros((H, T, D), dtype=np.float32)
    cos_np = np.zeros((T, half_dim), dtype=np.float32)
    sin_np = np.zeros((T, half_dim), dtype=np.float32)

    # Precompute cache
    lib.rope_precompute_cache(
        numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_float(10000.0),
        ctypes.c_int(D), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
    )

    # Get pointers
    d_out_ptr = numpy_to_ptr(d_out_np)
    d_x_ptr = numpy_to_ptr(d_x_np)

    # Torch tensors
    d_out = torch.from_numpy(d_out_np.copy())
    cos_cache = torch.from_numpy(cos_np.copy())
    sin_cache = torch.from_numpy(sin_np.copy())

    report = TestReport(
        test_name="RoPE Backward",
        dtype="fp32",
        shape=f"H={H}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    # PyTorch reference
    d_x_ref = rope_backward_pytorch(d_out, cos_cache, sin_cache)

    # C kernel
    def c_rope_backward():
        lib.rope_backward(
            d_out_ptr, d_x_ptr,
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(0)
        )

    c_rope_backward()
    d_x_c = torch.from_numpy(d_x_np.copy())
    diff = max_diff(d_x_c, d_x_ref)

    kernel_time = time_function(c_rope_backward, warmup=warmup, iterations=iterations, name="C RoPE Bwd")

    report.add_result(TestResult(
        name="d_input",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=None,
        kernel_time=kernel_time
    ))

    return report


def run_accuracy_tests():
    """Run accuracy tests at various sizes."""
    report = TestReport(
        test_name="RoPE Accuracy (Various Sizes)",
        dtype="fp32",
        shape="Multiple configurations",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (2, 8, 16, "Tiny"),
        (4, 16, 32, "Small"),
        (8, 32, 64, "Medium"),
        (8, 64, 128, "Large"),
    ]

    for H, T, D, name in test_configs:
        np.random.seed(42)
        half_dim = D // 2

        x_np = np.random.randn(H, T, D).astype(np.float32)
        cos_np = np.zeros((T, half_dim), dtype=np.float32)
        sin_np = np.zeros((T, half_dim), dtype=np.float32)

        lib.rope_precompute_cache(
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(T), ctypes.c_int(D), ctypes.c_float(10000.0),
            ctypes.c_int(D), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
        )

        x = torch.from_numpy(x_np.copy())
        cos_cache = torch.from_numpy(cos_np.copy())
        sin_cache = torch.from_numpy(sin_np.copy())

        # Use loop reference for exact comparison
        ref = rope_forward_pytorch(x, cos_cache, sin_cache)

        x_c_np = x_np.copy()
        lib.rope_forward(
            numpy_to_ptr(x_c_np), numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(D),
            ctypes.c_int(D), ctypes.c_int(0)
        )

        out = torch.from_numpy(x_c_np)
        diff = max_diff(out, ref)

        report.add_result(TestResult(
            name=f"{name} (H={H},T={T},D={D})",
            passed=diff <= 1e-5,
            max_diff=diff,
            tolerance=1e-5,
            pytorch_time=None,
            kernel_time=None
        ))

    return report


def run_rope_pairwise_multi_theta():
    """
    Test Llama-style pairwise RoPE forward across model-specific theta values.

    This covers the layout used by Llama-family checkpoints, where pairs are
    consecutive channels: (0,1), (2,3), ...
    """
    report = TestReport(
        test_name="RoPE Pairwise Forward Multi-Theta",
        dtype="fp32",
        shape="Various theta values / Llama-family",
        cpu_info=get_cpu_info()
    )

    test_configs = [
        (10000.0, "Llama2", 8, 4, 16, 64),
        (500000.0, "Llama3", 8, 4, 16, 64),
        (70000000.0, "Nanbeige", 20, 4, 8, 128),
    ]

    for theta, model_name, q_heads, kv_heads, tokens, head_dim in test_configs:
        np.random.seed(int(theta) % 100000)
        half_dim = head_dim // 2
        q_np = np.random.randn(q_heads, tokens, head_dim).astype(np.float32)
        k_np = np.random.randn(kv_heads, tokens, head_dim).astype(np.float32)
        cos_np = np.zeros((tokens, half_dim), dtype=np.float32)
        sin_np = np.zeros((tokens, half_dim), dtype=np.float32)

        lib.rope_precompute_cache(
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(tokens), ctypes.c_int(head_dim), ctypes.c_float(theta),
            ctypes.c_int(head_dim), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
        )

        q_ref = rope_forward_pairwise_numpy(q_np, cos_np, sin_np)
        k_ref = rope_forward_pairwise_numpy(k_np, cos_np, sin_np)

        q_out = q_np.copy()
        k_out = k_np.copy()
        lib.rope_forward_qk_pairwise_with_rotary_dim(
            numpy_to_ptr(q_out), numpy_to_ptr(k_out),
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(q_heads), ctypes.c_int(kv_heads), ctypes.c_int(tokens),
            ctypes.c_int(head_dim), ctypes.c_int(head_dim), ctypes.c_int(0), ctypes.c_int(head_dim),
        )

        q_diff = max_diff(torch.from_numpy(q_out), torch.from_numpy(q_ref))
        k_diff = max_diff(torch.from_numpy(k_out), torch.from_numpy(k_ref))

        report.add_result(TestResult(
            name=f"{model_name} pairwise-q (θ={theta:.0f})",
            passed=q_diff <= 1e-5,
            max_diff=q_diff,
            tolerance=1e-5,
            pytorch_time=None,
            kernel_time=None
        ))
        report.add_result(TestResult(
            name=f"{model_name} pairwise-k (θ={theta:.0f})",
            passed=k_diff <= 1e-5,
            max_diff=k_diff,
            tolerance=1e-5,
            pytorch_time=None,
            kernel_time=None
        ))

    return report


def run_rope_pairwise_decode_positions_test():
    """Verify pairwise RoPE uses decode position offsets correctly."""
    report = TestReport(
        test_name="RoPE Pairwise Decode Positions",
        dtype="fp32",
        shape="Various positions / Llama-family",
        cpu_info=get_cpu_info()
    )

    q_heads, kv_heads, tokens, head_dim = 20, 4, 1, 128
    max_cache_len = 2048
    half_dim = head_dim // 2

    cos_np = np.zeros((max_cache_len, half_dim), dtype=np.float32)
    sin_np = np.zeros((max_cache_len, half_dim), dtype=np.float32)
    lib.rope_precompute_cache(
        numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
        ctypes.c_int(max_cache_len), ctypes.c_int(head_dim), ctypes.c_float(70000000.0),
        ctypes.c_int(head_dim), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
    )

    for pos in [0, 1, 10, 100, 500, 1000]:
        np.random.seed(pos + 7)
        q_np = np.random.randn(q_heads, tokens, head_dim).astype(np.float32)
        k_np = np.random.randn(kv_heads, tokens, head_dim).astype(np.float32)

        q_ref = rope_forward_pairwise_numpy(q_np, cos_np, sin_np, pos_offset=pos)
        k_ref = rope_forward_pairwise_numpy(k_np, cos_np, sin_np, pos_offset=pos)

        q_out = q_np.copy()
        k_out = k_np.copy()
        lib.rope_forward_qk_pairwise_with_rotary_dim(
            numpy_to_ptr(q_out), numpy_to_ptr(k_out),
            numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
            ctypes.c_int(q_heads), ctypes.c_int(kv_heads), ctypes.c_int(tokens),
            ctypes.c_int(head_dim), ctypes.c_int(head_dim), ctypes.c_int(pos), ctypes.c_int(head_dim),
        )

        q_diff = max_diff(torch.from_numpy(q_out), torch.from_numpy(q_ref))
        k_diff = max_diff(torch.from_numpy(k_out), torch.from_numpy(k_ref))
        diff = max(q_diff, k_diff)
        report.add_result(TestResult(
            name=f"pairwise-pos={pos}",
            passed=diff <= 1e-5,
            max_diff=diff,
            tolerance=1e-5,
            pytorch_time=None,
            kernel_time=None
        ))

    return report


def run_pairwise_forward_tests(q_heads=20, kv_heads=4, T=16, D=128, warmup=5, iterations=100):
    """Run pairwise forward pass tests with accuracy and timing."""
    np.random.seed(123)
    half_dim = D // 2

    q_np = np.random.randn(q_heads, T, D).astype(np.float32)
    k_np = np.random.randn(kv_heads, T, D).astype(np.float32)
    cos_np = np.zeros((T, half_dim), dtype=np.float32)
    sin_np = np.zeros((T, half_dim), dtype=np.float32)

    lib.rope_precompute_cache(
        numpy_to_ptr(cos_np), numpy_to_ptr(sin_np),
        ctypes.c_int(T), ctypes.c_int(D), ctypes.c_float(70000000.0),
        ctypes.c_int(D), ctypes.c_char_p(b"none"), ctypes.c_float(1.0),
    )

    q = torch.from_numpy(q_np.copy())
    k = torch.from_numpy(k_np.copy())
    cos_cache = torch.from_numpy(cos_np.copy())
    sin_cache = torch.from_numpy(sin_np.copy())

    report = TestReport(
        test_name="RoPE Pairwise Forward",
        dtype="fp32",
        shape=f"QH={q_heads}, KVH={kv_heads}, T={T}, D={D}",
        cpu_info=get_cpu_info()
    )

    def pytorch_pairwise():
        q_ref = rope_forward_pytorch_pairwise_vectorized(q, cos_cache, sin_cache)
        k_ref = rope_forward_pytorch_pairwise_vectorized(k, cos_cache, sin_cache)
        return torch.cat([q_ref.reshape(-1), k_ref.reshape(-1)], dim=0)

    q_out = q_np.copy()
    k_out = k_np.copy()
    q_ptr = numpy_to_ptr(q_out)
    k_ptr = numpy_to_ptr(k_out)
    cos_ptr = numpy_to_ptr(cos_np)
    sin_ptr = numpy_to_ptr(sin_np)

    def c_pairwise():
        np.copyto(q_out, q_np)
        np.copyto(k_out, k_np)
        lib.rope_forward_qk_pairwise_with_rotary_dim(
            q_ptr, k_ptr, cos_ptr, sin_ptr,
            ctypes.c_int(q_heads), ctypes.c_int(kv_heads), ctypes.c_int(T),
            ctypes.c_int(D), ctypes.c_int(D), ctypes.c_int(0), ctypes.c_int(D),
        )
        return torch.cat([torch.from_numpy(q_out.reshape(-1).copy()), torch.from_numpy(k_out.reshape(-1).copy())], dim=0)

    ref = pytorch_pairwise()
    out = c_pairwise()
    diff = max_diff(out, ref)
    pytorch_time = time_function(pytorch_pairwise, warmup=warmup, iterations=iterations, name="PyTorch Pairwise")
    kernel_time = time_function(c_pairwise, warmup=warmup, iterations=iterations, name="C RoPE Pairwise")

    report.add_result(TestResult(
        name="RoPE Pairwise QK",
        passed=diff <= 1e-5,
        max_diff=diff,
        tolerance=1e-5,
        pytorch_time=pytorch_time,
        kernel_time=kernel_time
    ))

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_system_info()

    all_reports = []

    # Cache tests (default theta=10000)
    cache_report = run_cache_tests()
    cache_report.print_report()
    all_reports.append(cache_report)

    # Multi-theta cache tests (Llama2, Llama3, Qwen2, Mistral)
    # CRITICAL: This tests that rope_precompute_cache works with different
    # model architectures that use different RoPE theta values.
    multi_theta_report = run_cache_tests_multi_theta()
    multi_theta_report.print_report()
    all_reports.append(multi_theta_report)

    # Multi-theta forward tests
    multi_theta_fwd_report = run_rope_forward_multi_theta()
    multi_theta_fwd_report.print_report()
    all_reports.append(multi_theta_fwd_report)

    # Decode position tests (KV cache continuation)
    decode_pos_report = run_rope_decode_positions_test()
    decode_pos_report.print_report()
    all_reports.append(decode_pos_report)

    # Pairwise Llama-family tests
    pairwise_multi_theta_report = run_rope_pairwise_multi_theta()
    pairwise_multi_theta_report.print_report()
    all_reports.append(pairwise_multi_theta_report)

    pairwise_decode_pos_report = run_rope_pairwise_decode_positions_test()
    pairwise_decode_pos_report.print_report()
    all_reports.append(pairwise_decode_pos_report)

    # Accuracy tests
    acc_report = run_accuracy_tests()
    acc_report.print_report()
    all_reports.append(acc_report)

    # Forward performance tests
    fwd_report = run_forward_tests(H=8, T=64, D=64, warmup=10, iterations=500)
    fwd_report.print_report()
    all_reports.append(fwd_report)

    pairwise_fwd_report = run_pairwise_forward_tests()
    pairwise_fwd_report.print_report()
    all_reports.append(pairwise_fwd_report)

    # Backward tests
    bwd_report = run_backward_tests(H=8, T=64, D=64, warmup=10, iterations=500)
    bwd_report.print_report()
    all_reports.append(bwd_report)

    # Exit with error if any tests failed
    all_passed = all(r.all_passed() for r in all_reports)
    if not all_passed:
        print("\n" + "="*60)
        print("SOME TESTS FAILED!")
        print("="*60)
        exit(1)
    else:
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
