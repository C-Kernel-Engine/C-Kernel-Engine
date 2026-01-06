#!/usr/bin/env python3
"""
test_kernels_vs_llamacpp.py - Kernel-level parity tests

Tests individual CK kernels against llama.cpp's ggml implementations.
Similar to test_pytorch_parity.sh but for quantized kernels.

Usage:
    python scripts/test_kernels_vs_llamacpp.py --all
    python scripts/test_kernels_vs_llamacpp.py --kernel dequant_q4k
    python scripts/test_kernels_vs_llamacpp.py --tol 1e-4

Prerequisites:
    1. Build llama.cpp kernel test library:
       cd llama.cpp && make libggml_kernel_test.so

    2. Build CK parity library:
       make libck_parity.so
"""

import ctypes
import numpy as np
import argparse
import struct
import sys
from pathlib import Path

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Block sizes from GGML (must match)
QK_K = 256  # Elements per K-quant super-block
QK4_0 = 32  # Elements per Q4_0 block
BLOCK_Q4_K_SIZE = 144  # bytes per Q4_K block
BLOCK_Q6_K_SIZE = 210  # bytes per Q6_K block
BLOCK_Q8_K_SIZE = 292  # bytes per Q8_K block
BLOCK_Q4_0_SIZE = 18   # bytes per Q4_0 block


def load_libraries():
    """Load both test libraries."""
    base_dir = Path(__file__).parent.parent

    # Try to load llama.cpp kernel test library
    llama_paths = [
        base_dir / "llama.cpp" / "libggml_kernel_test.so",
        base_dir / "llama.cpp" / "build" / "libggml_kernel_test.so",
    ]

    libggml = None
    for p in llama_paths:
        if p.exists():
            try:
                libggml = ctypes.CDLL(str(p))
                print(f"Loaded llama.cpp library: {p}")
                break
            except OSError as e:
                print(f"Failed to load {p}: {e}")

    # Try to load CK parity library
    ck_paths = [
        base_dir / "build" / "libck_parity.so",
        base_dir / "libck_parity.so",
    ]

    libck = None
    for p in ck_paths:
        if p.exists():
            try:
                libck = ctypes.CDLL(str(p))
                print(f"Loaded CK library: {p}")
                break
            except OSError as e:
                print(f"Failed to load {p}: {e}")

    return libggml, libck


def fp16_to_bytes(val: float) -> bytes:
    """Convert float to FP16 bytes."""
    return struct.pack('<e', val)


def random_q4k_block() -> bytes:
    """Generate a random Q4_K block (144 bytes)."""
    data = bytearray()

    # d (fp16 scale): 2 bytes
    d = np.random.uniform(0.01, 0.1)
    data.extend(fp16_to_bytes(d))

    # dmin (fp16 min): 2 bytes
    dmin = np.random.uniform(0.001, 0.05)
    data.extend(fp16_to_bytes(dmin))

    # scales (6-bit packed): 12 bytes
    scales = np.random.randint(0, 64, size=12, dtype=np.uint8)
    data.extend(scales.tobytes())

    # qs (4-bit weights): 128 bytes
    qs = np.random.randint(0, 256, size=128, dtype=np.uint8)
    data.extend(qs.tobytes())

    assert len(data) == BLOCK_Q4_K_SIZE, f"Q4_K block size mismatch: {len(data)} != {BLOCK_Q4_K_SIZE}"
    return bytes(data)


def random_q4k_weights(n_elements: int) -> bytes:
    """Generate random Q4_K quantized weights."""
    assert n_elements % QK_K == 0, f"n_elements must be multiple of {QK_K}"
    n_blocks = n_elements // QK_K

    data = bytearray()
    for _ in range(n_blocks):
        data.extend(random_q4k_block())

    return bytes(data)


def random_q4_0_block() -> bytes:
    """Generate a random Q4_0 block (18 bytes)."""
    data = bytearray()

    # d (fp16 scale): 2 bytes
    d = np.random.uniform(0.01, 0.1)
    data.extend(fp16_to_bytes(d))

    # qs (4-bit weights): 16 bytes (32 weights, 2 per byte)
    qs = np.random.randint(0, 256, size=16, dtype=np.uint8)
    data.extend(qs.tobytes())

    assert len(data) == BLOCK_Q4_0_SIZE
    return bytes(data)


def random_q4_0_weights(n_elements: int) -> bytes:
    """Generate random Q4_0 quantized weights."""
    assert n_elements % QK4_0 == 0
    n_blocks = n_elements // QK4_0

    data = bytearray()
    for _ in range(n_blocks):
        data.extend(random_q4_0_block())

    return bytes(data)


class KernelTester:
    def __init__(self, libggml, libck, tol=1e-3):
        self.libggml = libggml
        self.libck = libck
        self.tol = tol
        self.results = []

        # Set up function signatures for llama.cpp library
        if libggml:
            self._setup_ggml_signatures()

        # Set up function signatures for CK library
        if libck:
            self._setup_ck_signatures()

    def _setup_ggml_signatures(self):
        """Set up ctypes signatures for llama.cpp functions."""
        lib = self.libggml

        # Dequantization
        lib.test_dequant_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.test_dequant_q4_k.restype = None

        lib.test_dequant_q4_0.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.test_dequant_q4_0.restype = None

        # Quantization
        lib.test_quantize_q8_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int]
        lib.test_quantize_q8_k.restype = None

        # GEMV
        lib.test_gemv_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.test_gemv_q4_k.restype = None

        # GEMM
        lib.test_gemm_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.test_gemm_q4_k.restype = None

        # RMSNorm
        lib.test_rmsnorm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.test_rmsnorm.restype = None

        # RoPE
        lib.test_rope.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_float]
        lib.test_rope.restype = None

        # SwiGLU
        lib.test_swiglu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                     ctypes.c_int, ctypes.c_int]
        lib.test_swiglu.restype = None

        # Softmax
        lib.test_softmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.test_softmax.restype = None

    def _setup_ck_signatures(self):
        """Set up ctypes signatures for CK functions."""
        lib = self.libck

        # Dequantization
        lib.ck_test_dequant_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_test_dequant_q4_k.restype = None

        lib.ck_test_dequant_q4_0.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_test_dequant_q4_0.restype = None

        # Quantization
        lib.ck_test_quantize_q8_k.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int]
        lib.ck_test_quantize_q8_k.restype = None

        # GEMV
        lib.ck_test_gemv_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_test_gemv_q4_k.restype = None

        # GEMM
        lib.ck_test_gemm_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                           ctypes.POINTER(ctypes.c_float),
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.ck_test_gemm_q4_k.restype = None

        # RMSNorm
        lib.ck_test_rmsnorm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_float),
                                         ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.ck_test_rmsnorm.restype = None

        # RoPE (interleaved version for llama.cpp compatibility)
        lib.ck_test_rope_interleaved.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_float]
        lib.ck_test_rope_interleaved.restype = None

        # SwiGLU
        lib.ck_test_swiglu.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_int, ctypes.c_int]
        lib.ck_test_swiglu.restype = None

        # Softmax
        lib.ck_test_softmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.ck_test_softmax.restype = None

    def compare(self, name: str, ggml_out: np.ndarray, ck_out: np.ndarray) -> bool:
        """Compare two outputs and record result."""
        diff = np.abs(ggml_out - ck_out)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        # Check for NaN
        has_nan = np.isnan(ggml_out).any() or np.isnan(ck_out).any()
        passed = max_diff < self.tol and not has_nan

        self.results.append((name, passed, max_diff, mean_diff))

        status = "PASS" if passed else "FAIL"
        color = GREEN if passed else RED

        print(f"[{color}{status}{RESET}] {name}: max_diff={max_diff:.2e}, mean={mean_diff:.2e}")

        if not passed:
            # Show worst case
            idx = np.argmax(diff)
            print(f"       ggml[{idx}]={ggml_out.flat[idx]:.6f}, ck[{idx}]={ck_out.flat[idx]:.6f}")
            if has_nan:
                print(f"       {RED}WARNING: NaN detected!{RESET}")

        return passed

    def test_dequant_q4k(self, size: int = 256):
        """Test Q4_K dequantization."""
        print(f"\n--- test_dequant_q4k (size={size}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        # Generate random Q4_K data
        q4k_data = random_q4k_weights(size)

        # GGML dequant
        ggml_out = np.zeros(size, dtype=np.float32)
        self.libggml.test_dequant_q4_k(q4k_data, ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size)

        # CK dequant
        ck_out = np.zeros(size, dtype=np.float32)
        self.libck.ck_test_dequant_q4_k(q4k_data, ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size)

        return self.compare("dequant_q4_k", ggml_out, ck_out)

    def test_dequant_q4_0(self, size: int = 32):
        """Test Q4_0 dequantization."""
        print(f"\n--- test_dequant_q4_0 (size={size}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        # Generate random Q4_0 data
        q4_0_data = random_q4_0_weights(size)

        # GGML dequant
        ggml_out = np.zeros(size, dtype=np.float32)
        self.libggml.test_dequant_q4_0(q4_0_data, ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size)

        # CK dequant
        ck_out = np.zeros(size, dtype=np.float32)
        self.libck.ck_test_dequant_q4_0(q4_0_data, ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size)

        return self.compare("dequant_q4_0", ggml_out, ck_out)

    def test_gemv_q4k(self, cols: int = 256):
        """Test Q4_K GEMV (matrix-vector multiply).

        Since ggml_vec_dot_q4_K_q8_K has issues in our test harness,
        we use dequant+FP32 as reference (dequant already verified to match).
        """
        print(f"\n--- test_gemv_q4k (cols={cols}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        # Generate random weights and input
        q4k_weights = random_q4k_weights(cols)
        input_f32 = np.random.randn(cols).astype(np.float32)

        # Reference: Dequantize weights to FP32 (using ggml dequant which matches CK)
        # then compute FP32 dot product
        dequant_weights = np.zeros(cols, dtype=np.float32)
        self.libggml.test_dequant_q4_k(q4k_weights,
                                        dequant_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        cols)
        ref_out = np.array([np.dot(dequant_weights, input_f32)], dtype=np.float32)

        # CK gemv (quantized)
        ck_out = np.zeros(1, dtype=np.float32)
        self.libck.ck_test_gemv_q4_k(q4k_weights,
                                      input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      cols)

        # Compare CK quantized GEMV against FP32 reference
        # Note: some error expected due to Q8_K quantization of activations
        return self.compare("gemv_q4_k", ref_out, ck_out)

    def test_gemm_q4k(self, rows: int = 64, cols: int = 256, n_tokens: int = 4):
        """Test Q4_K GEMM (batched matrix multiply).

        Since ggml_vec_dot_q4_K_q8_K has issues in our test harness,
        we use dequant+FP32 as reference (dequant already verified to match).
        """
        print(f"\n--- test_gemm_q4k (rows={rows}, cols={cols}, tokens={n_tokens}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        q4k_weights = random_q4k_weights(rows * cols)
        input_f32 = np.random.randn(n_tokens, cols).astype(np.float32)

        # Reference: Dequantize weights to FP32, then compute FP32 GEMM
        dequant_weights = np.zeros(rows * cols, dtype=np.float32)
        self.libggml.test_dequant_q4_k(q4k_weights,
                                        dequant_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        rows * cols)
        weight_matrix = dequant_weights.reshape(rows, cols)
        ref_out = np.dot(input_f32, weight_matrix.T).astype(np.float32)

        # CK GEMM (quantized)
        ck_out = np.zeros((n_tokens, rows), dtype=np.float32)
        self.libck.ck_test_gemm_q4_k(q4k_weights,
                                      input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      rows, cols, n_tokens)

        # Compare CK quantized GEMM against FP32 reference
        # Note: some error expected due to Q8_K quantization of activations
        return self.compare("gemm_q4_k", ref_out, ck_out)

    def test_rmsnorm(self, n_tokens: int = 4, dim: int = 256):
        """Test RMSNorm."""
        print(f"\n--- test_rmsnorm (tokens={n_tokens}, dim={dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        input_f32 = np.random.randn(n_tokens, dim).astype(np.float32)
        weight = np.random.randn(dim).astype(np.float32)
        eps = 1e-6

        ggml_out = np.zeros((n_tokens, dim), dtype=np.float32)
        ck_out = np.zeros((n_tokens, dim), dtype=np.float32)

        self.libggml.test_rmsnorm(input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   n_tokens, dim, ctypes.c_float(eps))

        self.libck.ck_test_rmsnorm(input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                    weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                    ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                    n_tokens, dim, ctypes.c_float(eps))

        return self.compare("rmsnorm", ggml_out, ck_out)

    def test_rope(self, n_tokens: int = 4, n_heads: int = 8, head_dim: int = 64):
        """Test RoPE (Rotary Position Embedding)."""
        print(f"\n--- test_rope (tokens={n_tokens}, heads={n_heads}, dim={head_dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        n_heads_kv = n_heads  # Assume no GQA for simplicity

        q = np.random.randn(n_tokens, n_heads * head_dim).astype(np.float32)
        k = np.random.randn(n_tokens, n_heads_kv * head_dim).astype(np.float32)

        q_ggml, k_ggml = q.copy(), k.copy()
        q_ck, k_ck = q.copy(), k.copy()

        theta = 10000.0

        self.libggml.test_rope(q_ggml.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                k_ggml.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                n_tokens, n_heads, n_heads_kv, head_dim,
                                0, ctypes.c_float(theta))

        # Use interleaved version for llama.cpp compatibility
        self.libck.ck_test_rope_interleaved(q_ck.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                             k_ck.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                             n_tokens, n_heads, n_heads_kv, head_dim,
                                             0, ctypes.c_float(theta))

        self.compare("rope_q", q_ggml, q_ck)
        return self.compare("rope_k", k_ggml, k_ck)

    def test_swiglu(self, n_tokens: int = 4, intermediate_dim: int = 256):
        """Test SwiGLU activation."""
        print(f"\n--- test_swiglu (tokens={n_tokens}, inter={intermediate_dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        gate_up = np.random.randn(n_tokens, 2 * intermediate_dim).astype(np.float32)

        ggml_out = np.zeros((n_tokens, intermediate_dim), dtype=np.float32)
        ck_out = np.zeros((n_tokens, intermediate_dim), dtype=np.float32)

        self.libggml.test_swiglu(gate_up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  n_tokens, intermediate_dim)

        self.libck.ck_test_swiglu(gate_up.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   n_tokens, intermediate_dim)

        return self.compare("swiglu", ggml_out, ck_out)

    def test_softmax(self, n: int = 128):
        """Test softmax."""
        print(f"\n--- test_softmax (n={n}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        input_f32 = np.random.randn(n).astype(np.float32)

        ggml_out = np.zeros(n, dtype=np.float32)
        ck_out = np.zeros(n, dtype=np.float32)

        self.libggml.test_softmax(input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   n)

        self.libck.ck_test_softmax(input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                    ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                    n)

        return self.compare("softmax", ggml_out, ck_out)

    def run_all(self):
        """Run all kernel tests."""
        print("=" * 70)
        print(f"{BOLD}KERNEL PARITY TESTS: C-Kernel-Engine vs llama.cpp/ggml{RESET}")
        print("=" * 70)

        # Dequantization kernels
        self.test_dequant_q4k(256)
        self.test_dequant_q4k(512)
        self.test_dequant_q4_0(32)
        self.test_dequant_q4_0(64)

        # Quantized GEMV/GEMM
        self.test_gemv_q4k(256)
        self.test_gemv_q4k(512)
        self.test_gemm_q4k(64, 256, 4)
        self.test_gemm_q4k(256, 512, 8)

        # Activation kernels
        self.test_rmsnorm(4, 256)
        self.test_rmsnorm(1, 2048)

        self.test_rope(4, 8, 64)
        self.test_rope(1, 32, 128)

        self.test_swiglu(4, 256)
        self.test_swiglu(1, 1024)

        self.test_softmax(128)
        self.test_softmax(512)

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print(f"{BOLD}KERNEL TEST SUMMARY{RESET}")
        print("=" * 70)

        if not self.results:
            print(f"{YELLOW}No tests were run. Check library availability.{RESET}")
            return False

        passed = sum(1 for r in self.results if r[1])
        total = len(self.results)

        print(f"Passed: {passed}/{total}")

        if passed < total:
            print(f"\n{RED}Failed tests:{RESET}")
            for name, ok, max_diff, _ in self.results:
                if not ok:
                    print(f"  - {name}: max_diff={max_diff:.2e}")
            return False
        else:
            print(f"\n{GREEN}All kernels match llama.cpp/ggml!{RESET}")
            return True


def main():
    parser = argparse.ArgumentParser(description="Kernel-level parity tests: CK vs llama.cpp")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--kernel", type=str, help="Test specific kernel")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance (default: 1e-3)")
    args = parser.parse_args()

    libggml, libck = load_libraries()

    if not libggml:
        print(f"\n{RED}ERROR: Could not load llama.cpp kernel test library.{RESET}")
        print("Build it with:")
        print("  cd llama.cpp")
        print("  g++ -shared -fPIC -o libggml_kernel_test.so \\")
        print("      tests/test-kernel-parity.cpp \\")
        print("      -I ggml/include -I ggml/src \\")
        print("      -L build -lggml -lm -lpthread")
        sys.exit(1)

    if not libck:
        print(f"\n{RED}ERROR: Could not load CK parity library.{RESET}")
        print("Build it with:")
        print("  make libck_parity.so")
        sys.exit(1)

    tester = KernelTester(libggml, libck, tol=args.tol)

    if args.kernel:
        # Run specific test
        test_method = f"test_{args.kernel}"
        if hasattr(tester, test_method):
            getattr(tester, test_method)()
            tester.print_summary()
        else:
            print(f"{RED}Unknown kernel: {args.kernel}{RESET}")
            print("Available kernels:")
            for attr in dir(tester):
                if attr.startswith("test_") and callable(getattr(tester, attr)):
                    print(f"  - {attr[5:]}")
            sys.exit(1)
    else:
        # Run all tests
        success = tester.run_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
