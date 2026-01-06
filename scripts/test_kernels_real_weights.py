#!/usr/bin/env python3
"""
test_kernels_real_weights.py - Kernel parity tests using real model weights

Tests quantized kernels (Q4_K, Q6_K, Q8_K) using actual GGUF weights for
llama.cpp and corresponding bump weights for CK-Kernel-Engine.

Usage:
    python scripts/test_kernels_real_weights.py \
        --gguf qwen2.5-3b-instruct-q4_k_m.gguf \
        --bump weights.bump \
        --manifest weights_manifest.json

Prerequisites:
    1. Convert GGUF to bump format:
       python scripts/convert_gguf_to_bump.py \
           --gguf model.gguf --output weights.bump --verify

    2. Build parity libraries:
       make parity-libs
"""

import argparse
import ctypes
import json
import mmap
import numpy as np
import struct
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Quantization block sizes
QK_K = 256  # Elements per K-quant super-block
BLOCK_Q4_K_SIZE = 144
BLOCK_Q6_K_SIZE = 210
BLOCK_Q8_K_SIZE = 292


def load_libraries():
    """Load parity test libraries."""
    base_dir = Path(__file__).parent.parent

    libggml = None
    libck = None

    # Load llama.cpp kernel test library
    ggml_path = base_dir / "llama.cpp" / "libggml_kernel_test.so"
    if ggml_path.exists():
        try:
            libggml = ctypes.CDLL(str(ggml_path))
            print(f"Loaded llama.cpp library: {ggml_path}")
        except OSError as e:
            print(f"{RED}Failed to load llama.cpp lib: {e}{RESET}")

    # Load CK parity library
    ck_path = base_dir / "build" / "libck_parity.so"
    if ck_path.exists():
        try:
            libck = ctypes.CDLL(str(ck_path))
            print(f"Loaded CK library: {ck_path}")
        except OSError as e:
            print(f"{RED}Failed to load CK lib: {e}{RESET}")

    return libggml, libck


def setup_signatures(libggml, libck):
    """Set up ctypes function signatures."""
    if libggml:
        # Dequantization
        libggml.test_dequant_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        libggml.test_dequant_q4_k.restype = None
        libggml.test_dequant_q6_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        libggml.test_dequant_q6_k.restype = None

    if libck:
        # Dequantization
        libck.ck_test_dequant_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        libck.ck_test_dequant_q4_k.restype = None
        libck.ck_test_dequant_q6_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        libck.ck_test_dequant_q6_k.restype = None

        # GEMV/GEMM
        libck.ck_test_gemv_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        libck.ck_test_gemv_q4_k.restype = None
        libck.ck_test_gemm_q4_k.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float),
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
        libck.ck_test_gemm_q4_k.restype = None


class WeightLoader:
    """Load weights from GGUF and bump files."""

    def __init__(self, gguf_path: str, bump_path: str, manifest_path: str):
        self.gguf_path = Path(gguf_path)
        self.bump_path = Path(bump_path)
        self.manifest_path = Path(manifest_path)

        # Load manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Open bump file for memory mapping
        self.bump_file = open(bump_path, 'rb')
        self.bump_mmap = mmap.mmap(self.bump_file.fileno(), 0, access=mmap.ACCESS_READ)

        # Build lookup by CK name
        self.entries = {}
        for entry in self.manifest.get('entries', []):
            self.entries[entry['name']] = entry

        # Get model config
        self.n_layers = self.manifest.get('n_layers', 24)
        print(f"Model has {self.n_layers} layers")

    def close(self):
        """Close file handles."""
        self.bump_mmap.close()
        self.bump_file.close()

    def get_entry(self, ck_name: str) -> Optional[dict]:
        """Get manifest entry by CK name."""
        return self.entries.get(ck_name)

    def load_bump_tensor(self, ck_name: str) -> Tuple[bytes, dict]:
        """Load raw tensor data from bump file."""
        entry = self.entries.get(ck_name)
        if not entry:
            raise KeyError(f"Tensor {ck_name} not found in manifest")

        # Use file_offset field from manifest
        offset = entry.get('file_offset', entry.get('offset', 0))
        size = entry['size']
        self.bump_mmap.seek(offset)
        data = self.bump_mmap.read(size)
        return data, entry

    def list_tensors_by_type(self, dtype: str) -> list:
        """List all tensors of a specific quantization type."""
        # Handle both uppercase and lowercase dtype
        dtype_lower = dtype.lower()
        return [e for e in self.manifest.get('entries', [])
                if e.get('dtype', '').lower() == dtype_lower]

    def get_test_layers(self) -> list:
        """Get layer indices to test (first, middle, last)."""
        if self.n_layers <= 3:
            return list(range(self.n_layers))
        return [0, self.n_layers // 2, self.n_layers - 1]


class KernelParityTester:
    """Test quantized kernels using real weights."""

    def __init__(self, libggml, libck, weight_loader: WeightLoader, tol: float = 1e-3):
        self.libggml = libggml
        self.libck = libck
        self.loader = weight_loader
        self.tol = tol
        self.results = []

    def compare(self, name: str, ref: np.ndarray, test: np.ndarray) -> bool:
        """Compare two arrays and record result."""
        diff = np.abs(ref - test)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        # Calculate relative error
        ref_max = float(np.max(np.abs(ref)))
        rel_error = max_diff / (ref_max + 1e-9)

        passed = max_diff < self.tol
        self.results.append((name, passed, max_diff, mean_diff, rel_error))

        status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
        print(f"{status} {name}: max_diff={max_diff:.2e}, rel_err={rel_error:.2%}")

        if not passed:
            idx = np.argmax(diff)
            print(f"       ref[{idx}]={ref.flat[idx]:.6f}, test[{idx}]={test.flat[idx]:.6f}")

        return passed

    def test_dequant_q4k(self, ck_name: str) -> bool:
        """Test Q4_K dequantization with real weights."""
        print(f"\n--- test_dequant_q4k: {ck_name} ---")

        entry = self.loader.get_entry(ck_name)
        if not entry or entry.get('dtype', '').lower() != 'q4_k':
            print(f"{YELLOW}[SKIP] Not a Q4_K tensor (dtype={entry.get('dtype') if entry else 'N/A'}){RESET}")
            return False

        # Load from bump file (same format as GGUF for quantized weights)
        bump_data, _ = self.loader.load_bump_tensor(ck_name)

        # Calculate number of elements
        n_blocks = len(bump_data) // BLOCK_Q4_K_SIZE
        n_elements = n_blocks * QK_K

        print(f"  Tensor: {ck_name}")
        print(f"  Size: {len(bump_data)} bytes = {n_blocks} blocks = {n_elements} elements")

        # Dequantize with llama.cpp (ggml)
        ggml_out = np.zeros(n_elements, dtype=np.float32)
        self.libggml.test_dequant_q4_k(bump_data,
                                        ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        n_elements)

        # Dequantize with CK
        ck_out = np.zeros(n_elements, dtype=np.float32)
        self.libck.ck_test_dequant_q4_k(bump_data,
                                         ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         n_elements)

        # Show sample values
        print(f"  GGML first 5: {ggml_out[:5]}")
        print(f"  CK first 5:   {ck_out[:5]}")

        return self.compare(f"dequant_q4k({ck_name})", ggml_out, ck_out)

    def test_gemv_q4k(self, ck_name: str, use_first_row: bool = True) -> bool:
        """Test Q4_K GEMV with real weights."""
        print(f"\n--- test_gemv_q4k: {ck_name} ---")

        entry = self.loader.get_entry(ck_name)
        if not entry or entry.get('dtype', '').lower() != 'q4_k':
            print(f"{YELLOW}[SKIP] Not a Q4_K tensor (dtype={entry.get('dtype') if entry else 'N/A'}){RESET}")
            return False

        # Load weights from bump (CK format)
        bump_data, _ = self.loader.load_bump_tensor(ck_name)

        # Calculate dimensions from data size
        n_blocks = len(bump_data) // BLOCK_Q4_K_SIZE
        n_elements = n_blocks * QK_K

        # Use 2048 elements for GEMV test (8 blocks)
        test_cols = min(2048, n_elements)
        test_blocks = test_cols // QK_K
        test_bytes = test_blocks * BLOCK_Q4_K_SIZE

        print(f"  Total: {n_elements} elements, testing GEMV with {test_cols} cols")

        # Use first row for GEMV test
        weight_row = bump_data[:test_bytes]
        cols = test_cols

        # Generate random input
        np.random.seed(42)  # Reproducible
        input_f32 = np.random.randn(cols).astype(np.float32) * 0.1

        # FP32 reference: dequantize + dot product
        dequant = np.zeros(cols, dtype=np.float32)
        self.libggml.test_dequant_q4_k(weight_row,
                                        dequant.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        cols)
        ref_out = np.array([np.dot(dequant, input_f32)], dtype=np.float32)

        # CK quantized GEMV
        ck_out = np.zeros(1, dtype=np.float32)
        self.libck.ck_test_gemv_q4_k(weight_row,
                                      input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      cols)

        print(f"  FP32 reference: {ref_out[0]:.6f}")
        print(f"  CK GEMV:        {ck_out[0]:.6f}")

        # Use larger tolerance for quantized GEMV (Q8_K activation quantization adds error)
        return self.compare(f"gemv_q4k({ck_name})", ref_out, ck_out)

    def test_gemm_q4k(self, ck_name: str, n_tokens: int = 4) -> bool:
        """Test Q4_K GEMM with real weights."""
        print(f"\n--- test_gemm_q4k: {ck_name} (tokens={n_tokens}) ---")

        entry = self.loader.get_entry(ck_name)
        if not entry or entry.get('dtype', '').lower() != 'q4_k':
            print(f"{YELLOW}[SKIP] Not a Q4_K tensor{RESET}")
            return False

        # Load weights from bump
        bump_data, _ = self.loader.load_bump_tensor(ck_name)

        # Calculate dimensions from data size
        n_blocks = len(bump_data) // BLOCK_Q4_K_SIZE
        n_elements = n_blocks * QK_K

        # For testing, assume a reasonable matrix size
        # Use first 2048x2048 = 4M elements or less
        test_elements = min(2048 * 2048, n_elements)
        test_cols = 2048
        test_rows = test_elements // test_cols

        print(f"  Total: {n_elements} elements, testing: [{test_rows}, {test_cols}], tokens={n_tokens}")

        # Generate random input
        np.random.seed(42)
        input_f32 = np.random.randn(n_tokens, test_cols).astype(np.float32) * 0.1

        # FP32 reference
        dequant = np.zeros(test_rows * test_cols, dtype=np.float32)
        test_bytes = (test_rows * test_cols // QK_K) * BLOCK_Q4_K_SIZE
        self.libggml.test_dequant_q4_k(bump_data[:test_bytes],
                                        dequant.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        test_rows * test_cols)
        weight_matrix = dequant.reshape(test_rows, test_cols)
        ref_out = np.dot(input_f32, weight_matrix.T).astype(np.float32)

        # CK quantized GEMM
        ck_out = np.zeros((n_tokens, test_rows), dtype=np.float32)
        self.libck.ck_test_gemm_q4_k(bump_data[:test_bytes],
                                      input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      test_rows, test_cols, n_tokens)

        return self.compare(f"gemm_q4k({ck_name})", ref_out, ck_out)

    def run_all_tests(self):
        """Run tests on sample weights from different layers."""
        print("=" * 70)
        print(f"{BOLD}KERNEL PARITY TESTS WITH REAL WEIGHTS{RESET}")
        print("=" * 70)

        # Find Q4_K tensors
        q4k_tensors = self.loader.list_tensors_by_type('Q4_K')
        print(f"\nFound {len(q4k_tensors)} Q4_K tensors")

        # Get test layers based on model size
        test_layers = self.loader.get_test_layers()
        print(f"Testing layers: {test_layers}")

        # Select test points: wq, wk, w1 from each test layer
        test_tensors = []
        tensor_types = ['wq', 'wk', 'w1']

        for layer in test_layers:
            for tensor_type in tensor_types:
                tensor_name = f"layer.{layer}.{tensor_type}"
                if self.loader.get_entry(tensor_name):
                    entry = self.loader.get_entry(tensor_name)
                    if entry.get('dtype', '').lower() == 'q4_k':
                        test_tensors.append(tensor_name)

        print(f"Test tensors: {test_tensors}")

        # Run dequant tests
        print(f"\n{'='*70}")
        print(f"{BOLD}Dequantization Tests{RESET}")
        print(f"{'='*70}")

        for name in test_tensors:
            self.test_dequant_q4k(name)

        # Run GEMV tests
        print(f"\n{'='*70}")
        print(f"{BOLD}GEMV Tests (quantized){RESET}")
        print(f"{'='*70}")

        for name in test_tensors[:3]:  # Fewer GEMV tests
            self.test_gemv_q4k(name)

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*70}")
        print(f"{BOLD}TEST SUMMARY{RESET}")
        print(f"{'='*70}")

        passed = sum(1 for r in self.results if r[1])
        total = len(self.results)

        for name, ok, max_diff, mean_diff, rel_err in self.results:
            status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
            print(f"  [{status}] {name}: max={max_diff:.2e}, rel={rel_err:.2%}")

        print(f"\nOverall: {passed}/{total} passed")

        if passed == total:
            print(f"\n{GREEN}All tests passed!{RESET}")
        else:
            print(f"\n{RED}Some tests failed - check implementations{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Kernel parity tests with real weights")
    parser.add_argument("--gguf", required=True, help="Path to GGUF model file")
    parser.add_argument("--bump", required=True, help="Path to bump weights file")
    parser.add_argument("--manifest", help="Path to weights manifest JSON")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for comparisons")
    parser.add_argument("--tensor", help="Test specific tensor by CK name")
    args = parser.parse_args()

    # Derive manifest path if not provided
    if not args.manifest:
        bump_path = Path(args.bump)
        args.manifest = str(bump_path.parent / "weights_manifest.json")

    # Load libraries
    libggml, libck = load_libraries()
    if not libggml or not libck:
        print(f"{RED}Failed to load libraries{RESET}")
        sys.exit(1)

    setup_signatures(libggml, libck)

    # Load weights
    loader = WeightLoader(args.gguf, args.bump, args.manifest)

    # Run tests
    tester = KernelParityTester(libggml, libck, loader, tol=args.tol)

    if args.tensor:
        # Test specific tensor
        tester.test_dequant_q4k(args.tensor)
        tester.test_gemv_q4k(args.tensor)
    else:
        # Run all tests
        tester.run_all_tests()

    loader.close()


if __name__ == "__main__":
    main()
