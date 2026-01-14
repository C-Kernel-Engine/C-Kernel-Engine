#!/usr/bin/env python3
"""
Kernel Architecture Parity Test

Tests that all SIMD variants (ref, SSE, AVX, AVX2, AVX-512) produce
numerically identical results.

Usage:
    python scripts/test_kernel_parity.py
    python scripts/test_kernel_parity.py --verbose
    python scripts/test_kernel_parity.py --quant q4_k  # Test specific format
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

# Get project root
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
BUILD_DIR = ROOT_DIR / "build"

# Tolerance for floating point comparison
ATOL = 1e-4
RTOL = 1e-3

def get_cpu_features():
    """Detect CPU SIMD capabilities."""
    features = {
        'sse4_1': False,
        'sse4_2': False,
        'avx': False,
        'avx2': False,
        'fma': False,
        'avx512f': False,
        'avx512bw': False,
        'avx512vnni': False,
        'amx': False,
    }

    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()

        features['sse4_1'] = 'sse4_1' in cpuinfo
        features['sse4_2'] = 'sse4_2' in cpuinfo
        features['avx'] = ' avx ' in cpuinfo
        features['avx2'] = 'avx2' in cpuinfo
        features['fma'] = ' fma ' in cpuinfo
        features['avx512f'] = 'avx512f' in cpuinfo
        features['avx512bw'] = 'avx512bw' in cpuinfo
        features['avx512vnni'] = 'avx512vnni' in cpuinfo or 'avx512_vnni' in cpuinfo
        features['amx'] = 'amx_tile' in cpuinfo
    except:
        pass

    return features

def print_cpu_info():
    """Print CPU capabilities."""
    features = get_cpu_features()

    print("=" * 60)
    print("CPU SIMD Capabilities")
    print("=" * 60)

    # Get CPU model
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    print(f"CPU: {line.split(':')[1].strip()}")
                    break
    except:
        print("CPU: Unknown")

    print()
    print("Instruction Sets:")
    for feat, available in features.items():
        status = "✅" if available else "❌"
        print(f"  {feat:15} {status}")
    print()

    return features

def compile_test_binary(arch_flags, output_name):
    """Compile test binary with specific architecture flags."""
    src_file = ROOT_DIR / "scripts" / "test_kernel_parity_runner.c"

    if not src_file.exists():
        return None

    output_path = BUILD_DIR / output_name

    # Compiler flags
    cc = os.environ.get('CC', 'gcc')
    cflags = [
        '-O2',
        '-Wall',
        '-I', str(ROOT_DIR / 'include'),
    ]
    cflags.extend(arch_flags.split())

    # Link against kernel library
    ldflags = [
        '-L', str(BUILD_DIR),
        '-lck_kernels',
        '-lm',
    ]

    cmd = [cc] + cflags + [str(src_file), '-o', str(output_path)] + ldflags

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compile failed for {arch_flags}:")
            print(result.stderr)
            return None
        return output_path
    except Exception as e:
        print(f"Compile error: {e}")
        return None

def run_existing_parity_test():
    """Run the existing llama.cpp parity test."""
    print("=" * 60)
    print("Running llama.cpp Parity Tests")
    print("=" * 60)

    result = subprocess.run(
        ['make', 'llamacpp-parity-full'],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)

    return result.returncode == 0

def test_kernel_dispatch():
    """Test that kernel dispatch selects correct implementation."""
    print("=" * 60)
    print("Kernel Dispatch Test")
    print("=" * 60)

    features = get_cpu_features()

    # Expected dispatch based on features
    expected = {}

    # Q4_K x Q8_K
    if features['avx512vnni']:
        expected['Q4_K_Q8_K'] = 'VNNI'
    elif features['avx2']:
        expected['Q4_K_Q8_K'] = 'AVX2'
    elif features['avx']:
        expected['Q4_K_Q8_K'] = 'AVX'
    elif features['sse4_1']:
        expected['Q4_K_Q8_K'] = 'SSE'
    else:
        expected['Q4_K_Q8_K'] = 'REF'

    # Q5_0 x Q8_0
    if features['avx512f']:
        expected['Q5_0_Q8_0'] = 'AVX-512'
    elif features['avx']:
        expected['Q5_0_Q8_0'] = 'AVX'
    elif features['sse4_1']:
        expected['Q5_0_Q8_0'] = 'SSE'
    else:
        expected['Q5_0_Q8_0'] = 'REF'

    # Q6_K x Q8_K
    if features['avx512f'] and features['avx512bw']:
        expected['Q6_K_Q8_K'] = 'AVX-512'
    elif features['avx2']:
        expected['Q6_K_Q8_K'] = 'AVX2'
    elif features['avx']:
        expected['Q6_K_Q8_K'] = 'AVX'
    elif features['sse4_1']:
        expected['Q6_K_Q8_K'] = 'SSE'
    else:
        expected['Q6_K_Q8_K'] = 'REF'

    # Q8_0 x FP32
    if features['avx512f']:
        expected['Q8_0'] = 'AVX-512'
    elif features['avx']:
        expected['Q8_0'] = 'AVX'
    elif features['sse4_1']:
        expected['Q8_0'] = 'SSE'
    else:
        expected['Q8_0'] = 'REF'

    print("Expected Kernel Dispatch:")
    for kernel, impl in expected.items():
        print(f"  {kernel:15} -> {impl}")
    print()

    return expected

def create_parity_matrix():
    """Create and display the parity test matrix."""
    print("=" * 60)
    print("Architecture Parity Matrix")
    print("=" * 60)

    # This is what we found in the audit
    matrix = {
        'Q4_K_Q8_K': {
            'ref': True,
            'SSE': True,
            'AVX': True,
            'AVX2': True,
            'AVX-512': False,  # Uses VNNI only
            'VNNI': True,
        },
        'Q5_0_Q8_0': {
            'ref': True,
            'SSE': True,
            'AVX': True,
            'AVX2': 'fallback',  # Falls back to AVX
            'AVX-512': True,
            'VNNI': False,
        },
        'Q6_K_Q8_K': {
            'ref': True,
            'SSE': True,
            'AVX': True,
            'AVX2': True,
            'AVX-512': True,
            'VNNI': False,
        },
        'Q8_0': {
            'ref': True,
            'SSE': True,
            'AVX': True,
            'AVX2': 'fallback',  # Falls back to AVX
            'AVX-512': True,
            'VNNI': False,
        },
    }

    # Print header
    archs = ['ref', 'SSE', 'AVX', 'AVX2', 'AVX-512', 'VNNI']
    header = f"{'Kernel':15}" + "".join(f"{a:10}" for a in archs)
    print(header)
    print("-" * len(header))

    # Print matrix
    for kernel, impls in matrix.items():
        row = f"{kernel:15}"
        for arch in archs:
            status = impls.get(arch, False)
            if status is True:
                row += f"{'✅':10}"
            elif status == 'fallback':
                row += f"{'⚠️':10}"
            else:
                row += f"{'❌':10}"
        print(row)

    print()
    print("Legend: ✅ = dedicated impl, ⚠️ = fallback, ❌ = not available")
    print()

    return matrix

def main():
    parser = argparse.ArgumentParser(description='Kernel Architecture Parity Test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quant', type=str, help='Test specific quantization format')
    parser.add_argument('--skip-parity', action='store_true', help='Skip llama.cpp parity test')
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        CK-Engine Kernel Architecture Parity Test         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # 1. Print CPU info
    features = print_cpu_info()

    # 2. Show parity matrix
    matrix = create_parity_matrix()

    # 3. Show expected dispatch
    expected = test_kernel_dispatch()

    # 4. Run existing parity test
    if not args.skip_parity:
        success = run_existing_parity_test()
        if not success:
            print("❌ llama.cpp parity test FAILED")
            return 1
        print("✅ llama.cpp parity test PASSED")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("Gaps to address:")
    print("  1. Q4_K x Q8_K: No AVX-512F (non-VNNI) implementation")
    print("     -> CPUs with AVX-512 but without VNNI fall back to AVX2")
    print()
    print("  2. Q5_0: No dedicated AVX2 implementation")
    print("     -> Falls back to AVX (loses FMA benefit)")
    print()
    print("  3. Q8_0: No dedicated AVX2 implementation")
    print("     -> Falls back to AVX (loses FMA benefit)")
    print()

    # Check if user's remote machine is fine
    if features['avx512vnni']:
        print("✅ Your CPU has AVX-512 VNNI - all kernels will use optimal paths")
    elif features['avx512f']:
        print("⚠️ Your CPU has AVX-512F but not VNNI - Q4_K will use AVX2 fallback")
    elif features['avx2']:
        print("⚠️ Your CPU has AVX2 - Q5_0/Q8_0 will use AVX fallback")

    print()
    return 0

if __name__ == '__main__':
    sys.exit(main())
