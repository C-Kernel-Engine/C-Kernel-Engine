#!/usr/bin/env python3
"""
test_avx512_parity.py - AVX-512 Parity Test for Q6_K x Q8_K Kernel

This script tests the AVX-512 implementation of the Q6_K x Q8_K kernel
against:
1. Python reference implementation
2. SSE/AVX2 builds (if available)
3. llama.cpp parity test (if llama.cpp is built)

Usage:
    python scripts/test_avx512_parity.py [--full] [--llamacpp]

    --full     Run extensive tests with multiple block sizes
    --llamacpp Also test against llama.cpp reference (requires build)

Requirements:
    - AVX-512 capable CPU for full testing
    - gcc with AVX-512 support (gcc >= 7)
"""

import argparse
import ctypes
import numpy as np
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Block sizes (must match ggml)
QK_K = 256
BLOCK_Q6_K_SIZE = 210  # 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
BLOCK_Q8_K_SIZE = 292  # 4 (d) + 256 (qs) + 32 (bsums)

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

PROJECT_ROOT = Path(__file__).parent.parent


def check_avx512_support() -> Tuple[bool, str]:
    """Check if CPU supports AVX-512"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()

        has_avx512f = 'avx512f' in cpuinfo
        has_avx512bw = 'avx512bw' in cpuinfo
        has_avx512vbmi = 'avx512vbmi' in cpuinfo

        if has_avx512f and has_avx512bw:
            if has_avx512vbmi:
                return True, "AVX-512 VBMI (Ice Lake+)"
            else:
                return True, "AVX-512 (Skylake-X)"
        elif has_avx512f:
            return True, "AVX-512F only"
        else:
            return False, "No AVX-512"
    except:
        return False, "Cannot detect"


def fp16_to_fp32(h):
    """Convert FP16 (uint16) to FP32"""
    return np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0].astype(np.float32)


def fp32_to_fp16_bytes(f):
    """Convert FP32 to FP16 bytes"""
    h = np.array([f], dtype=np.float16).tobytes()
    return h[0], h[1]


def create_q6k_block(d_scale, scales, values):
    """Create a Q6_K block matching ggml layout"""
    assert len(scales) == 16
    assert len(values) == 256

    block = bytearray(BLOCK_Q6_K_SIZE)
    values = np.clip(values, 0, 63).astype(np.uint8)

    # ql: low 4 bits, packed as pairs
    for i in range(128):
        lo = values[i * 2] & 0x0F
        hi = values[i * 2 + 1] & 0x0F
        block[i] = lo | (hi << 4)

    # qh: high 2 bits, packed as quads
    for i in range(64):
        qh_val = 0
        for j in range(4):
            qh_val |= ((values[i * 4 + j] >> 4) & 0x03) << (j * 2)
        block[128 + i] = qh_val

    # scales
    for i, s in enumerate(scales):
        block[192 + i] = np.uint8(np.int8(s).view(np.uint8))

    # d (FP16)
    d_lo, d_hi = fp32_to_fp16_bytes(d_scale)
    block[208] = d_lo
    block[209] = d_hi

    return bytes(block)


def create_q8k_block(d_scale, values):
    """Create a Q8_K block matching ggml layout"""
    assert len(values) == 256

    block = bytearray(BLOCK_Q8_K_SIZE)

    # d (FP32)
    d_bytes = struct.pack('<f', d_scale)
    block[0:4] = d_bytes

    # qs (int8)
    values = np.clip(values, -128, 127).astype(np.int8)
    block[4:260] = values.tobytes()

    # bsums (int16) - sum of each 16-element sub-block
    bsums = []
    for i in range(16):
        bsum = int(np.sum(values[i*16:(i+1)*16]))
        bsums.append(np.int16(bsum))
    block[260:292] = np.array(bsums, dtype=np.int16).tobytes()

    return bytes(block)


def dot_q6k_q8k_reference(w_block, x_block) -> float:
    """
    Pure Python reference implementation of Q6_K x Q8_K dot product.
    This matches the algorithm in ggml/llama.cpp exactly.
    """
    # Extract Q6_K components
    ql = np.frombuffer(w_block[0:128], dtype=np.uint8)
    qh = np.frombuffer(w_block[128:192], dtype=np.uint8)
    scales = np.frombuffer(w_block[192:208], dtype=np.int8)
    d_w = fp16_to_fp32(struct.unpack('<H', w_block[208:210])[0])

    # Extract Q8_K components
    d_x = struct.unpack('<f', x_block[0:4])[0]
    q8 = np.frombuffer(x_block[4:260], dtype=np.int8)

    d = d_w * d_x
    sumf = 0.0

    q8_idx = 0
    ql_idx = 0
    qh_idx = 0
    sc_idx = 0

    for n in range(QK_K // 128):  # 2 iterations
        ql_slice = ql[ql_idx:ql_idx+64]
        qh_slice = qh[qh_idx:qh_idx+32]
        sc_slice = scales[sc_idx:sc_idx+8]
        q8_slice = q8[q8_idx:q8_idx+128]

        for l in range(32):
            is_val = l // 16

            q1 = int((ql_slice[l] & 0x0F) | (((qh_slice[l] >> 0) & 3) << 4)) - 32
            q2 = int((ql_slice[l + 32] & 0x0F) | (((qh_slice[l] >> 2) & 3) << 4)) - 32
            q3 = int((ql_slice[l] >> 4) | (((qh_slice[l] >> 4) & 3) << 4)) - 32
            q4 = int((ql_slice[l + 32] >> 4) | (((qh_slice[l] >> 6) & 3) << 4)) - 32

            sumf += d * float(sc_slice[is_val + 0]) * float(q1) * float(q8_slice[l + 0])
            sumf += d * float(sc_slice[is_val + 2]) * float(q2) * float(q8_slice[l + 32])
            sumf += d * float(sc_slice[is_val + 4]) * float(q3) * float(q8_slice[l + 64])
            sumf += d * float(sc_slice[is_val + 6]) * float(q4) * float(q8_slice[l + 96])

        q8_idx += 128
        ql_idx += 64
        qh_idx += 32
        sc_idx += 8

    return sumf


def compile_test_lib(march: str, output_name: str) -> Optional[Path]:
    """Compile test library with specific architecture"""

    src_files = [
        "src/kernels/gemm_kernels_q6k_q8k.c",
        "src/kernels/gemm_kernels_q6k.c",
        "src/kernels/gemm_kernels_q4k_q8k.c",
        "src/kernels/gemm_kernels_q4k_q8k_avx2.c",
        "src/kernels/gemm_kernels_q4k_sse.c",
        "src/kernels/quantize_row_q8_k_sse.c",
        "src/cpu_features.c",
    ]

    include_dir = PROJECT_ROOT / "include"
    build_dir = PROJECT_ROOT / "build"
    build_dir.mkdir(exist_ok=True)

    output_path = build_dir / output_name

    src_paths = ' '.join(str(PROJECT_ROOT / f) for f in src_files)

    cmd = f"gcc -O3 {march} -fPIC -shared -I{include_dir} {src_paths} -o {output_path} -lm 2>&1"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"{RED}Compilation failed for {march}:{RESET}")
        print(result.stderr[:500])
        return None

    return output_path


def load_lib(path: Path):
    """Load shared library and set up function signatures"""
    lib = ctypes.CDLL(str(path))

    lib.vec_dot_q6_k_q8_k.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p, ctypes.c_void_p
    ]
    lib.vec_dot_q6_k_q8_k.restype = None

    return lib


def test_single_dot_product(lib, num_blocks: int, test_name: str) -> Tuple[bool, float, float]:
    """
    Test single Q6_K x Q8_K dot product.
    Returns (passed, lib_result, ref_result)
    """
    np.random.seed(42 + num_blocks)

    K = QK_K * num_blocks

    # Create test data
    w_blocks = []
    x_blocks = []

    for _ in range(num_blocks):
        d_w = np.random.rand() * 0.1 + 0.01  # Non-zero scale
        scales = np.random.randint(-8, 8, size=16).astype(np.int8)
        values = np.random.randint(0, 64, size=256).astype(np.uint8)
        w_blocks.append(create_q6k_block(d_w, scales, values))

        d_x = np.random.rand() * 0.1 + 0.01
        x_vals = np.random.randint(-50, 50, size=256).astype(np.int8)
        x_blocks.append(create_q8k_block(d_x, x_vals))

    w_buffer = b''.join(w_blocks)
    x_buffer = b''.join(x_blocks)

    # Reference calculation
    ref_result = 0.0
    for i in range(num_blocks):
        ref_result += dot_q6k_q8k_reference(w_blocks[i], x_blocks[i])

    # Library calculation
    lib_result = ctypes.c_float(0.0)
    lib.vec_dot_q6_k_q8_k(
        K,
        ctypes.byref(lib_result),
        (ctypes.c_ubyte * len(w_buffer)).from_buffer_copy(w_buffer),
        (ctypes.c_ubyte * len(x_buffer)).from_buffer_copy(x_buffer)
    )

    diff = abs(lib_result.value - ref_result)
    rel_err = diff / (abs(ref_result) + 1e-10)

    passed = rel_err < 1e-4  # Allow small numerical differences

    return passed, lib_result.value, ref_result


def run_parity_test(march_flags: str, arch_name: str, full: bool = False):
    """Run parity test for a specific architecture"""

    print(f"\n{CYAN}Testing {arch_name}...{RESET}")

    lib_name = f"libq6k_test_{arch_name.replace(' ', '_').replace('+', '')}.so"
    lib_path = compile_test_lib(march_flags, lib_name)

    if lib_path is None:
        print(f"  {YELLOW}SKIP - Compilation failed{RESET}")
        return None

    try:
        lib = load_lib(lib_path)
    except Exception as e:
        print(f"  {YELLOW}SKIP - Cannot load library: {e}{RESET}")
        return None

    # Test configurations
    if full:
        test_configs = [1, 2, 4, 8, 16, 32, 64, 128]
    else:
        test_configs = [1, 4, 16]

    results = []

    for num_blocks in test_configs:
        test_name = f"{num_blocks} blocks ({num_blocks * QK_K} elements)"
        passed, lib_val, ref_val = test_single_dot_product(lib, num_blocks, test_name)

        diff = abs(lib_val - ref_val)
        rel_err = diff / (abs(ref_val) + 1e-10)

        if passed:
            print(f"  {GREEN}PASS{RESET} {test_name}: lib={lib_val:.6f} ref={ref_val:.6f} (err={rel_err:.2e})")
        else:
            print(f"  {RED}FAIL{RESET} {test_name}: lib={lib_val:.6f} ref={ref_val:.6f} (err={rel_err:.2e})")

        results.append(passed)

    return all(results)


def run_cross_architecture_test(full: bool = False):
    """Test that different architectures produce the same results"""

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Cross-Architecture Parity Test{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    # Test architectures
    archs = [
        ("-march=x86-64", "SSE2 baseline"),
        ("-march=core2 -mssse3", "SSSE3"),
        ("-march=haswell", "AVX2"),
    ]

    # Add AVX-512 if supported
    has_avx512, avx512_type = check_avx512_support()
    if has_avx512:
        archs.append(("-march=skylake-avx512", "AVX-512"))
        if "VBMI" in avx512_type:
            archs.append(("-march=icelake-client", "AVX-512 VBMI"))

    libs = {}

    # Compile all libraries
    print("\nCompiling test libraries...")
    for march, name in archs:
        lib_name = f"libq6k_test_{name.replace(' ', '_').replace('-', '')}.so"
        lib_path = compile_test_lib(march, lib_name)
        if lib_path:
            try:
                libs[name] = load_lib(lib_path)
                print(f"  {GREEN}OK{RESET} {name}")
            except:
                print(f"  {YELLOW}SKIP{RESET} {name} (cannot load)")
        else:
            print(f"  {YELLOW}SKIP{RESET} {name} (cannot compile)")

    if len(libs) < 2:
        print(f"\n{YELLOW}Need at least 2 architectures for cross-test{RESET}")
        return False

    # Run cross-architecture tests
    print(f"\nComparing outputs across architectures...")

    if full:
        test_configs = [1, 4, 16, 64, 256]
    else:
        test_configs = [1, 16]

    all_passed = True

    for num_blocks in test_configs:
        np.random.seed(42 + num_blocks)

        K = QK_K * num_blocks

        # Create test data
        w_blocks = []
        x_blocks = []

        for _ in range(num_blocks):
            d_w = np.random.rand() * 0.1 + 0.01
            scales = np.random.randint(-8, 8, size=16).astype(np.int8)
            values = np.random.randint(0, 64, size=256).astype(np.uint8)
            w_blocks.append(create_q6k_block(d_w, scales, values))

            d_x = np.random.rand() * 0.1 + 0.01
            x_vals = np.random.randint(-50, 50, size=256).astype(np.int8)
            x_blocks.append(create_q8k_block(d_x, x_vals))

        w_buffer = b''.join(w_blocks)
        x_buffer = b''.join(x_blocks)

        # Calculate with each library
        results = {}
        for name, lib in libs.items():
            result = ctypes.c_float(0.0)
            lib.vec_dot_q6_k_q8_k(
                K,
                ctypes.byref(result),
                (ctypes.c_ubyte * len(w_buffer)).from_buffer_copy(w_buffer),
                (ctypes.c_ubyte * len(x_buffer)).from_buffer_copy(x_buffer)
            )
            results[name] = result.value

        # Compare all pairs
        arch_names = list(results.keys())
        ref_name = arch_names[0]
        ref_val = results[ref_name]

        max_diff = 0.0
        for name in arch_names[1:]:
            diff = abs(results[name] - ref_val)
            max_diff = max(max_diff, diff)

        passed = max_diff < 1e-4

        if passed:
            print(f"  {GREEN}PASS{RESET} {num_blocks} blocks: max_diff={max_diff:.2e}")
        else:
            print(f"  {RED}FAIL{RESET} {num_blocks} blocks: max_diff={max_diff:.2e}")
            for name, val in results.items():
                print(f"       {name}: {val:.6f}")
            all_passed = False

    return all_passed


def run_llamacpp_parity(full: bool = False):
    """Test against llama.cpp reference (if available)"""

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}llama.cpp Parity Test{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    llamacpp_dir = PROJECT_ROOT / "llama.cpp"
    parity_test = llamacpp_dir / "build" / "test-kernel-parity"

    if not parity_test.exists():
        print(f"\n{YELLOW}llama.cpp parity test not built{RESET}")
        print(f"To build:")
        print(f"  cd {llamacpp_dir}")
        print(f"  mkdir -p build && cd build")
        print(f"  cmake .. -DGGML_AVX512=ON")
        print(f"  make test-kernel-parity")
        return None

    # Run llama.cpp parity test
    print("\nRunning llama.cpp parity test...")

    result = subprocess.run(
        [str(parity_test)],
        capture_output=True,
        text=True,
        cwd=llamacpp_dir / "build"
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="AVX-512 Parity Test for Q6_K x Q8_K Kernel")
    parser.add_argument("--full", action="store_true", help="Run extensive tests")
    parser.add_argument("--llamacpp", action="store_true", help="Also test against llama.cpp")
    parser.add_argument("--cross", action="store_true", help="Run cross-architecture tests")
    args = parser.parse_args()

    print(f"{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}AVX-512 Parity Test for Q6_K x Q8_K Kernel{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    # Check CPU features
    has_avx512, avx512_type = check_avx512_support()
    print(f"\nCPU: {avx512_type}")

    if not has_avx512:
        print(f"\n{YELLOW}WARNING: No AVX-512 support detected{RESET}")
        print("Tests will use SSE/AVX2 paths only")

    results = {}

    # Test 1: Python reference parity
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Test 1: Python Reference Parity{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    # Test native build
    result = run_parity_test("-march=native", "native", args.full)
    if result is not None:
        results["native"] = result

    # Test 2: Cross-architecture (if requested or AVX-512 available)
    if args.cross or has_avx512:
        result = run_cross_architecture_test(args.full)
        results["cross-arch"] = result

    # Test 3: llama.cpp parity (if requested)
    if args.llamacpp:
        result = run_llamacpp_parity(args.full)
        if result is not None:
            results["llamacpp"] = result

    # Summary
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}Summary{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    all_passed = True
    for name, passed in results.items():
        if passed is None:
            status = f"{YELLOW}SKIP{RESET}"
        elif passed:
            status = f"{GREEN}PASS{RESET}"
        else:
            status = f"{RED}FAIL{RESET}"
            all_passed = False
        print(f"  {name}: {status}")

    if all_passed and results:
        print(f"\n{GREEN}All tests passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}Some tests failed or skipped{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
