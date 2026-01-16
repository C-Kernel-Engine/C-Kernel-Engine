"""
Benchmark Block 2 (MLP) fusion: v2 (SIMD + gate/up fusion) vs separate.

For Qwen2-0.5B:
- embed_dim = 896
- intermediate_dim = 4864 (approx 5.4x)

Intermediate buffers:
- gate_out: 4864 * 4 = 19.5 KB
- up_out: 4864 * 4 = 19.5 KB
- Combined: 39 KB > 32 KB L1D

This is where fusion SHOULD help because intermediates exceed L1 cache.
"""
import ctypes
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np

from lib_loader import load_lib
from test_utils import get_cpu_info, print_system_info

lib = load_lib("libckernel_engine.so")

# Function signatures
# v2: mlp_fused_fp32_v2
lib.mlp_fused_fp32_v2.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # hidden_in
    ctypes.POINTER(ctypes.c_float),  # rms_weight
    ctypes.c_float,                  # eps
    ctypes.POINTER(ctypes.c_float),  # w_gate
    ctypes.POINTER(ctypes.c_float),  # w_up
    ctypes.POINTER(ctypes.c_float),  # w_down
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # intermediate_dim
    ctypes.POINTER(ctypes.c_float),  # hidden_out
]
lib.mlp_fused_fp32_v2.restype = None

# v3: mlp_fused_fp32_v3 (SIMD with sequential weight access)
lib.mlp_fused_fp32_v3.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # hidden_in
    ctypes.POINTER(ctypes.c_float),  # rms_weight
    ctypes.c_float,                  # eps
    ctypes.POINTER(ctypes.c_float),  # w_gate
    ctypes.POINTER(ctypes.c_float),  # w_up
    ctypes.POINTER(ctypes.c_float),  # w_down
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # intermediate_dim
    ctypes.POINTER(ctypes.c_float),  # hidden_out
]
lib.mlp_fused_fp32_v3.restype = None

# separate: mlp_separate_fp32
lib.mlp_separate_fp32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # hidden_in
    ctypes.POINTER(ctypes.c_float),  # rms_weight
    ctypes.c_float,                  # eps
    ctypes.POINTER(ctypes.c_float),  # w_gate
    ctypes.POINTER(ctypes.c_float),  # w_up
    ctypes.POINTER(ctypes.c_float),  # w_down
    ctypes.POINTER(ctypes.c_float),  # normed_buf
    ctypes.POINTER(ctypes.c_float),  # gate_buf
    ctypes.POINTER(ctypes.c_float),  # up_buf
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # intermediate_dim
    ctypes.POINTER(ctypes.c_float),  # hidden_out
]
lib.mlp_separate_fp32.restype = None


def ptr(arr):
    """Get ctypes pointer to numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def benchmark(fn, iterations=500, warmup=50):
    """Benchmark a function, return time in microseconds."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1e6  # microseconds


def test_mlp_fusion(embed_dim=896, intermediate_dim=4864):
    """
    Test Block 2 (MLP) fusion.

    Default dimensions are for Qwen2-0.5B:
    - embed_dim = 896
    - intermediate_dim = 4864 (roughly 5.4x expansion)
    """
    np.random.seed(42)

    # Allocate arrays
    hidden_in = np.random.randn(embed_dim).astype(np.float32)
    rms_weight = np.abs(np.random.randn(embed_dim).astype(np.float32)) + 0.1
    w_gate = np.random.randn(intermediate_dim, embed_dim).astype(np.float32) * 0.02
    w_up = np.random.randn(intermediate_dim, embed_dim).astype(np.float32) * 0.02
    w_down = np.random.randn(embed_dim, intermediate_dim).astype(np.float32) * 0.02

    # Output arrays
    out_v2 = np.zeros(embed_dim, dtype=np.float32)
    out_v3 = np.zeros(embed_dim, dtype=np.float32)
    out_sep = np.zeros(embed_dim, dtype=np.float32)

    # Separate version buffers
    normed_buf = np.zeros(embed_dim, dtype=np.float32)
    gate_buf = np.zeros(intermediate_dim, dtype=np.float32)
    up_buf = np.zeros(intermediate_dim, dtype=np.float32)

    eps = 1e-6

    def run_v2():
        lib.mlp_fused_fp32_v2(
            ptr(hidden_in), ptr(rms_weight),
            ctypes.c_float(eps),
            ptr(w_gate), ptr(w_up), ptr(w_down),
            ctypes.c_int(embed_dim),
            ctypes.c_int(intermediate_dim),
            ptr(out_v2)
        )

    def run_v3():
        lib.mlp_fused_fp32_v3(
            ptr(hidden_in), ptr(rms_weight),
            ctypes.c_float(eps),
            ptr(w_gate), ptr(w_up), ptr(w_down),
            ctypes.c_int(embed_dim),
            ctypes.c_int(intermediate_dim),
            ptr(out_v3)
        )

    def run_separate():
        lib.mlp_separate_fp32(
            ptr(hidden_in), ptr(rms_weight),
            ctypes.c_float(eps),
            ptr(w_gate), ptr(w_up), ptr(w_down),
            ptr(normed_buf), ptr(gate_buf), ptr(up_buf),
            ctypes.c_int(embed_dim),
            ctypes.c_int(intermediate_dim),
            ptr(out_sep)
        )

    # Run once to populate outputs
    run_v2()
    run_v3()
    run_separate()

    # Correctness check
    diff_v2 = np.max(np.abs(out_v2 - out_sep))
    diff_v3 = np.max(np.abs(out_v3 - out_sep))
    correct = diff_v2 < 1e-4 and diff_v3 < 1e-4

    print(f"\n{'='*70}")
    print(f"Block 2: MLP Fusion Benchmark")
    print(f"{'='*70}")
    print(f"Dimensions: embed_dim={embed_dim}, intermediate_dim={intermediate_dim}")
    print(f"\nCorrectness check:")
    print(f"  v2 vs separate: {diff_v2:.2e} {'PASS' if diff_v2 < 1e-4 else 'FAIL'}")
    print(f"  v3 vs separate: {diff_v3:.2e} {'PASS' if diff_v3 < 1e-4 else 'FAIL'}")

    if not correct:
        print("ERROR: Correctness check failed!")
        return False

    # Benchmark
    print(f"\nBenchmark (500 iterations):")
    t_sep = benchmark(run_separate)
    t_v2 = benchmark(run_v2)
    t_v3 = benchmark(run_v3)

    speedup_v2 = t_sep / t_v2
    speedup_v3 = t_sep / t_v3

    print(f"  Separate (scalar):      {t_sep:8.1f} us")
    print(f"  Fused v2 (gate+up):     {t_v2:8.1f} us  ({speedup_v2:.2f}x)")
    print(f"  Fused v3 (sequential):  {t_v3:8.1f} us  ({speedup_v3:.2f}x)")

    # Analysis
    print(f"\n{'='*70}")
    print("Analysis:")
    print(f"{'='*70}")

    best = min(t_v2, t_v3)
    best_name = "v2" if t_v2 < t_v3 else "v3"
    best_speedup = t_sep / best

    print(f"  Best fused: {best_name} ({best:.1f} us, {best_speedup:.2f}x)")

    gate_up_bytes = 2 * intermediate_dim * 4
    print(f"  gate + up buffers: {gate_up_bytes} bytes ({gate_up_bytes/1024:.1f} KB)")
    print(f"  Typical L1D size:  32 KB")

    if gate_up_bytes > 32 * 1024:
        print(f"  Buffers EXCEED L1D - fusion should help!")
    else:
        print(f"  Buffers fit in L1D - fusion benefit minimal")

    if best_speedup >= 1.3:
        print(f"  SUCCESS: {best_name} achieves >= 1.3x speedup target ({best_speedup:.2f}x)")
    else:
        print(f"  BELOW TARGET: {best_name} speedup {best_speedup:.2f}x is below 1.3x target")

    return True


def test_multiple_sizes():
    """Test fusion across different model sizes."""
    print("\n" + "="*70)
    print("Testing across different model dimensions:")
    print("="*70)

    configs = [
        # (embed_dim, intermediate_dim, name)
        (256, 1024, "Small"),
        (512, 2048, "Medium"),
        (896, 4864, "Qwen2-0.5B"),
        (1024, 4096, "1B-class"),
        (2048, 8192, "2B-class"),
    ]

    results = []
    for embed_dim, intermediate_dim, name in configs:
        print(f"\n{name}: embed={embed_dim}, intermediate={intermediate_dim}")

        np.random.seed(42)
        hidden_in = np.random.randn(embed_dim).astype(np.float32)
        rms_weight = np.abs(np.random.randn(embed_dim).astype(np.float32)) + 0.1
        w_gate = np.random.randn(intermediate_dim, embed_dim).astype(np.float32) * 0.02
        w_up = np.random.randn(intermediate_dim, embed_dim).astype(np.float32) * 0.02
        w_down = np.random.randn(embed_dim, intermediate_dim).astype(np.float32) * 0.02

        out = np.zeros(embed_dim, dtype=np.float32)
        normed_buf = np.zeros(embed_dim, dtype=np.float32)
        gate_buf = np.zeros(intermediate_dim, dtype=np.float32)
        up_buf = np.zeros(intermediate_dim, dtype=np.float32)

        def run_sep():
            lib.mlp_separate_fp32(
                ptr(hidden_in), ptr(rms_weight), ctypes.c_float(1e-6),
                ptr(w_gate), ptr(w_up), ptr(w_down),
                ptr(normed_buf), ptr(gate_buf), ptr(up_buf),
                embed_dim, intermediate_dim, ptr(out))

        def run_v2():
            lib.mlp_fused_fp32_v2(
                ptr(hidden_in), ptr(rms_weight), ctypes.c_float(1e-6),
                ptr(w_gate), ptr(w_up), ptr(w_down),
                embed_dim, intermediate_dim, ptr(out))

        def run_v3():
            lib.mlp_fused_fp32_v3(
                ptr(hidden_in), ptr(rms_weight), ctypes.c_float(1e-6),
                ptr(w_gate), ptr(w_up), ptr(w_down),
                embed_dim, intermediate_dim, ptr(out))

        t_sep = benchmark(run_sep, iterations=200, warmup=20)
        t_v2 = benchmark(run_v2, iterations=200, warmup=20)
        t_v3 = benchmark(run_v3, iterations=200, warmup=20)

        speedup_v2 = t_sep / t_v2
        speedup_v3 = t_sep / t_v3
        gate_up_kb = 2 * intermediate_dim * 4 / 1024

        print(f"  Separate: {t_sep:.1f} us")
        print(f"  Fused v2: {t_v2:.1f} us ({speedup_v2:.2f}x)")
        print(f"  Fused v3: {t_v3:.1f} us ({speedup_v3:.2f}x)")
        print(f"  gate+up:  {gate_up_kb:.1f} KB {'(>L1!)' if gate_up_kb > 32 else ''}")

        results.append((name, speedup_v2, speedup_v3, gate_up_kb))

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"{'Model':<15} {'v2':<10} {'v3':<10} {'gate+up':<12} {'Best':<10}")
    print("-"*57)
    for name, sp_v2, sp_v3, gate_up_kb in results:
        best = max(sp_v2, sp_v3)
        best_name = "v2" if sp_v2 > sp_v3 else "v3"
        print(f"{name:<15} {sp_v2:.2f}x     {sp_v3:.2f}x     {gate_up_kb:.1f} KB      {best_name}={best:.2f}x")


if __name__ == "__main__":
    print_system_info()

    # Main benchmark
    passed = test_mlp_fusion()

    # Multi-size test
    test_multiple_sizes()

    if not passed:
        sys.exit(1)
