"""
Benchmark true SIMD fusion (v2) vs fake fusion (v1) vs separate kernels.

Tests Block 1: RMSNorm + QKV projection

Expected results:
- Separate:  Baseline (4x normed reads from cache)
- Fused v1:  ~1.1-1.2x speedup (function bundling only)
- Fused v2:  ~1.5-2x speedup (normed never leaves registers)
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

# Function signatures for all three variants
# Separate: rmsnorm_qkv_separate_fp32
lib.rmsnorm_qkv_separate_fp32.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # rms_weight
    ctypes.POINTER(ctypes.c_float),  # wq
    ctypes.POINTER(ctypes.c_float),  # wk
    ctypes.POINTER(ctypes.c_float),  # wv
    ctypes.POINTER(ctypes.c_float),  # normed (intermediate buffer)
    ctypes.POINTER(ctypes.c_float),  # q_out
    ctypes.POINTER(ctypes.c_float),  # k_out
    ctypes.POINTER(ctypes.c_float),  # v_out
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # q_dim
    ctypes.c_int,                    # kv_dim
    ctypes.c_float,                  # eps
]
lib.rmsnorm_qkv_separate_fp32.restype = None

# Fused v1: rmsnorm_qkv_fp32_fused
lib.rmsnorm_qkv_fp32_fused.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # rms_weight
    ctypes.POINTER(ctypes.c_float),  # wq
    ctypes.POINTER(ctypes.c_float),  # wk
    ctypes.POINTER(ctypes.c_float),  # wv
    ctypes.POINTER(ctypes.c_float),  # q_out
    ctypes.POINTER(ctypes.c_float),  # k_out
    ctypes.POINTER(ctypes.c_float),  # v_out
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # q_dim
    ctypes.c_int,                    # kv_dim
    ctypes.c_float,                  # eps
]
lib.rmsnorm_qkv_fp32_fused.restype = None

# Fused v2: rmsnorm_qkv_fp32_fused_v2 (TRUE SIMD fusion - 8 outputs at a time)
lib.rmsnorm_qkv_fp32_fused_v2.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # rms_weight
    ctypes.POINTER(ctypes.c_float),  # wq
    ctypes.POINTER(ctypes.c_float),  # wk
    ctypes.POINTER(ctypes.c_float),  # wv
    ctypes.POINTER(ctypes.c_float),  # q_out
    ctypes.POINTER(ctypes.c_float),  # k_out
    ctypes.POINTER(ctypes.c_float),  # v_out
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # q_dim
    ctypes.c_int,                    # kv_dim
    ctypes.c_float,                  # eps
]
lib.rmsnorm_qkv_fp32_fused_v2.restype = None

# Fused v3: rmsnorm_qkv_fp32_fused_v3 (TRUE fusion - Q,K,V simultaneously)
lib.rmsnorm_qkv_fp32_fused_v3.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # rms_weight
    ctypes.POINTER(ctypes.c_float),  # wq
    ctypes.POINTER(ctypes.c_float),  # wk
    ctypes.POINTER(ctypes.c_float),  # wv
    ctypes.POINTER(ctypes.c_float),  # q_out
    ctypes.POINTER(ctypes.c_float),  # k_out
    ctypes.POINTER(ctypes.c_float),  # v_out
    ctypes.c_int,                    # embed_dim
    ctypes.c_int,                    # q_dim
    ctypes.c_int,                    # kv_dim
    ctypes.c_float,                  # eps
]
lib.rmsnorm_qkv_fp32_fused_v3.restype = None


def ptr(arr):
    """Get ctypes pointer to numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def benchmark(fn, iterations=1000, warmup=100):
    """Benchmark a function, return time in microseconds."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1e6  # microseconds


def test_rmsnorm_qkv_fusion(embed_dim=896, q_dim=896, kv_dim=128):
    """
    Test Block 1 fusion: RMSNorm + QKV projection.

    Default dimensions are for Qwen2-0.5B:
    - embed_dim = 896
    - q_dim = num_heads * head_dim = 14 * 64 = 896
    - kv_dim = num_kv_heads * head_dim = 2 * 64 = 128
    """
    np.random.seed(42)

    # Allocate arrays
    x = np.random.randn(embed_dim).astype(np.float32)
    rms_weight = np.abs(np.random.randn(embed_dim).astype(np.float32)) + 0.1
    wq = np.random.randn(q_dim, embed_dim).astype(np.float32) * 0.02
    wk = np.random.randn(kv_dim, embed_dim).astype(np.float32) * 0.02
    wv = np.random.randn(kv_dim, embed_dim).astype(np.float32) * 0.02

    # Output arrays (separate for each variant to avoid caching effects)
    q_sep = np.zeros(q_dim, dtype=np.float32)
    k_sep = np.zeros(kv_dim, dtype=np.float32)
    v_sep = np.zeros(kv_dim, dtype=np.float32)
    normed_buf = np.zeros(embed_dim, dtype=np.float32)

    q_v1 = np.zeros(q_dim, dtype=np.float32)
    k_v1 = np.zeros(kv_dim, dtype=np.float32)
    v_v1 = np.zeros(kv_dim, dtype=np.float32)

    q_v2 = np.zeros(q_dim, dtype=np.float32)
    k_v2 = np.zeros(kv_dim, dtype=np.float32)
    v_v2 = np.zeros(kv_dim, dtype=np.float32)

    q_v3 = np.zeros(q_dim, dtype=np.float32)
    k_v3 = np.zeros(kv_dim, dtype=np.float32)
    v_v3 = np.zeros(kv_dim, dtype=np.float32)

    eps = 1e-6

    # Define benchmark functions
    def run_separate():
        lib.rmsnorm_qkv_separate_fp32(
            ptr(x), ptr(rms_weight),
            ptr(wq), ptr(wk), ptr(wv),
            ptr(normed_buf),
            ptr(q_sep), ptr(k_sep), ptr(v_sep),
            ctypes.c_int(embed_dim),
            ctypes.c_int(q_dim),
            ctypes.c_int(kv_dim),
            ctypes.c_float(eps)
        )

    def run_v1():
        lib.rmsnorm_qkv_fp32_fused(
            ptr(x), ptr(rms_weight),
            ptr(wq), ptr(wk), ptr(wv),
            ptr(q_v1), ptr(k_v1), ptr(v_v1),
            ctypes.c_int(embed_dim),
            ctypes.c_int(q_dim),
            ctypes.c_int(kv_dim),
            ctypes.c_float(eps)
        )

    def run_v2():
        lib.rmsnorm_qkv_fp32_fused_v2(
            ptr(x), ptr(rms_weight),
            ptr(wq), ptr(wk), ptr(wv),
            ptr(q_v2), ptr(k_v2), ptr(v_v2),
            ctypes.c_int(embed_dim),
            ctypes.c_int(q_dim),
            ctypes.c_int(kv_dim),
            ctypes.c_float(eps)
        )

    def run_v3():
        lib.rmsnorm_qkv_fp32_fused_v3(
            ptr(x), ptr(rms_weight),
            ptr(wq), ptr(wk), ptr(wv),
            ptr(q_v3), ptr(k_v3), ptr(v_v3),
            ctypes.c_int(embed_dim),
            ctypes.c_int(q_dim),
            ctypes.c_int(kv_dim),
            ctypes.c_float(eps)
        )

    # Run once to populate output arrays for correctness check
    run_separate()
    run_v1()
    run_v2()
    run_v3()

    # Correctness check
    q_ref = q_sep.copy()
    k_ref = k_sep.copy()
    v_ref = v_sep.copy()

    def check_correctness(name, q, k, v):
        q_diff = np.max(np.abs(q - q_ref))
        k_diff = np.max(np.abs(k - k_ref))
        v_diff = np.max(np.abs(v - v_ref))
        passed = q_diff < 1e-4 and k_diff < 1e-4 and v_diff < 1e-4
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status} (max diff: Q={q_diff:.2e}, K={k_diff:.2e}, V={v_diff:.2e})")
        return passed

    print(f"\n{'='*70}")
    print(f"Block 1: RMSNorm + QKV Projection Benchmark")
    print(f"{'='*70}")
    print(f"Dimensions: embed_dim={embed_dim}, q_dim={q_dim}, kv_dim={kv_dim}")
    print(f"\nCorrectness checks:")
    v1_correct = check_correctness("Fused v1", q_v1, k_v1, v_v1)
    v2_correct = check_correctness("Fused v2", q_v2, k_v2, v_v2)
    v3_correct = check_correctness("Fused v3", q_v3, k_v3, v_v3)

    if not (v1_correct and v2_correct and v3_correct):
        print("\nERROR: Correctness check failed!")
        return False

    # Benchmark
    print(f"\nBenchmark (1000 iterations):")
    t_sep = benchmark(run_separate)
    t_v1 = benchmark(run_v1)
    t_v2 = benchmark(run_v2)
    t_v3 = benchmark(run_v3)

    print(f"  Separate (baseline):     {t_sep:7.1f} us")
    print(f"  Fused v1 (fake):         {t_v1:7.1f} us  (speedup: {t_sep/t_v1:.2f}x)")
    print(f"  Fused v2 (8-at-a-time):  {t_v2:7.1f} us  (speedup: {t_sep/t_v2:.2f}x)")
    print(f"  Fused v3 (Q,K,V simult): {t_v3:7.1f} us  (speedup: {t_sep/t_v3:.2f}x)")
    print(f"\n  v3 vs v1:                {t_v1/t_v3:.2f}x faster")
    print(f"  v3 vs separate:          {t_sep/t_v3:.2f}x faster")

    # Analysis
    print(f"\n{'='*70}")
    print("Analysis:")
    print(f"{'='*70}")

    best = min(t_v1, t_v2, t_v3)
    best_name = "v1" if best == t_v1 else ("v2" if best == t_v2 else "v3")
    print(f"  Best fused version: {best_name} ({best:.1f} us)")

    if t_sep/best >= 1.3:
        print(f"  SUCCESS: {best_name} achieves >= 1.3x speedup target ({t_sep/best:.2f}x)")
    else:
        print(f"  BELOW TARGET: {best_name} speedup {t_sep/best:.2f}x is below 1.3x target")

    # Memory traffic analysis
    normed_bytes = embed_dim * 4  # float32
    print(f"\n  Memory traffic analysis:")
    print(f"    normed[] size: {normed_bytes} bytes ({normed_bytes/1024:.1f} KB)")
    print(f"    Separate: normed written + read 3x = {normed_bytes * 4} bytes extra")
    print(f"    Fused v2: normed NEVER stored = 0 bytes extra")

    return True


def test_multiple_sizes():
    """Test fusion across different model sizes."""
    print("\n" + "="*70)
    print("Testing across different model dimensions:")
    print("="*70)

    configs = [
        # (embed_dim, q_dim, kv_dim, name)
        (256, 256, 64, "Small"),
        (512, 512, 64, "Medium"),
        (896, 896, 128, "Qwen2-0.5B"),
        (1024, 1024, 128, "1B-class"),
        (2048, 2048, 256, "2B-class"),
    ]

    results = []
    for embed_dim, q_dim, kv_dim, name in configs:
        print(f"\n{name}: embed={embed_dim}, q={q_dim}, kv={kv_dim}")

        np.random.seed(42)
        x = np.random.randn(embed_dim).astype(np.float32)
        rms_weight = np.abs(np.random.randn(embed_dim).astype(np.float32)) + 0.1
        wq = np.random.randn(q_dim, embed_dim).astype(np.float32) * 0.02
        wk = np.random.randn(kv_dim, embed_dim).astype(np.float32) * 0.02
        wv = np.random.randn(kv_dim, embed_dim).astype(np.float32) * 0.02

        q_out = np.zeros(q_dim, dtype=np.float32)
        k_out = np.zeros(kv_dim, dtype=np.float32)
        v_out = np.zeros(kv_dim, dtype=np.float32)
        normed = np.zeros(embed_dim, dtype=np.float32)

        def run_sep():
            lib.rmsnorm_qkv_separate_fp32(
                ptr(x), ptr(rms_weight), ptr(wq), ptr(wk), ptr(wv), ptr(normed),
                ptr(q_out), ptr(k_out), ptr(v_out),
                embed_dim, q_dim, kv_dim, ctypes.c_float(1e-6))

        def run_v1():
            lib.rmsnorm_qkv_fp32_fused(
                ptr(x), ptr(rms_weight), ptr(wq), ptr(wk), ptr(wv),
                ptr(q_out), ptr(k_out), ptr(v_out),
                embed_dim, q_dim, kv_dim, ctypes.c_float(1e-6))

        def run_v2():
            lib.rmsnorm_qkv_fp32_fused_v2(
                ptr(x), ptr(rms_weight), ptr(wq), ptr(wk), ptr(wv),
                ptr(q_out), ptr(k_out), ptr(v_out),
                embed_dim, q_dim, kv_dim, ctypes.c_float(1e-6))

        def run_v3():
            lib.rmsnorm_qkv_fp32_fused_v3(
                ptr(x), ptr(rms_weight), ptr(wq), ptr(wk), ptr(wv),
                ptr(q_out), ptr(k_out), ptr(v_out),
                embed_dim, q_dim, kv_dim, ctypes.c_float(1e-6))

        t_sep = benchmark(run_sep, iterations=500, warmup=50)
        t_v1 = benchmark(run_v1, iterations=500, warmup=50)
        t_v2 = benchmark(run_v2, iterations=500, warmup=50)
        t_v3 = benchmark(run_v3, iterations=500, warmup=50)

        speedup_v1 = t_sep / t_v1
        speedup_v2 = t_sep / t_v2
        speedup_v3 = t_sep / t_v3

        print(f"  Separate: {t_sep:.1f} us")
        print(f"  Fused v1: {t_v1:.1f} us ({speedup_v1:.2f}x)")
        print(f"  Fused v2: {t_v2:.1f} us ({speedup_v2:.2f}x)")
        print(f"  Fused v3: {t_v3:.1f} us ({speedup_v3:.2f}x)")

        results.append((name, speedup_v1, speedup_v2, speedup_v3))

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"{'Model':<15} {'v1':<10} {'v2':<10} {'v3':<10} {'Best':<10}")
    print("-"*55)
    for name, sp_v1, sp_v2, sp_v3 in results:
        best = max(sp_v1, sp_v2, sp_v3)
        best_name = "v1" if best == sp_v1 else ("v2" if best == sp_v2 else "v3")
        print(f"{name:<15} {sp_v1:.2f}x     {sp_v2:.2f}x     {sp_v3:.2f}x     {best_name}={best:.2f}x")


if __name__ == "__main__":
    print_system_info()

    # Main benchmark
    passed = test_rmsnorm_qkv_fusion()

    # Multi-size test
    test_multiple_sizes()

    if not passed:
        sys.exit(1)
