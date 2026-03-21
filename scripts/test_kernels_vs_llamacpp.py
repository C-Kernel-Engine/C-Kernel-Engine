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
import time
import os
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
        self.ggml_has_deltanet = False
        self.ck_has_deltanet = False
        self.ggml_has_ssm_conv = False
        self.ck_has_ssm_conv = False
        self.ggml_has_split_q_gate = False
        self.ck_has_split_q_gate = False
        self.ggml_has_attn_gate_sigmoid_mul = False
        self.ck_has_attn_gate_sigmoid_mul = False
        self.ggml_has_recurrent_split_qkv = False
        self.ck_has_recurrent_split_qkv = False
        self.ggml_has_recurrent_dt_gate = False
        self.ck_has_recurrent_dt_gate = False
        self.ggml_has_recurrent_conv_state_update = False
        self.ck_has_recurrent_conv_state_update = False
        self.ggml_has_recurrent_silu = False
        self.ck_has_recurrent_silu = False
        self.ggml_has_recurrent_split_conv_qkv = False
        self.ck_has_recurrent_split_conv_qkv = False
        self.ggml_has_recurrent_qk_l2_norm = False
        self.ck_has_recurrent_qk_l2_norm = False
        self.ggml_has_recurrent_norm_gate = False
        self.ck_has_recurrent_norm_gate = False

        # Set up function signatures for llama.cpp library
        if libggml:
            self._setup_ggml_signatures()

        # Set up function signatures for CK library
        if libck:
            self._setup_ck_signatures()

    def _setup_ggml_signatures(self):
        """Set up ctypes signatures for llama.cpp functions."""
        lib = self.libggml

        # Init
        lib.test_init.argtypes = []
        lib.test_init.restype = None
        lib.test_init()

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

        if hasattr(lib, "test_split_q_gate"):
            lib.test_split_q_gate.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.test_split_q_gate.restype = None
            self.ggml_has_split_q_gate = True

        if hasattr(lib, "test_recurrent_split_qkv"):
            lib.test_recurrent_split_qkv.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.test_recurrent_split_qkv.restype = None
            self.ggml_has_recurrent_split_qkv = True

        if hasattr(lib, "test_recurrent_dt_gate"):
            lib.test_recurrent_dt_gate.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.test_recurrent_dt_gate.restype = None
            self.ggml_has_recurrent_dt_gate = True

        if hasattr(lib, "test_recurrent_conv_state_update"):
            lib.test_recurrent_conv_state_update.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.test_recurrent_conv_state_update.restype = None
            self.ggml_has_recurrent_conv_state_update = True

        if hasattr(lib, "test_recurrent_silu"):
            lib.test_recurrent_silu.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.test_recurrent_silu.restype = None
            self.ggml_has_recurrent_silu = True

        if hasattr(lib, "test_recurrent_split_conv_qkv"):
            lib.test_recurrent_split_conv_qkv.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.test_recurrent_split_conv_qkv.restype = None
            self.ggml_has_recurrent_split_conv_qkv = True

        if hasattr(lib, "test_recurrent_qk_l2_norm"):
            lib.test_recurrent_qk_l2_norm.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            lib.test_recurrent_qk_l2_norm.restype = None
            self.ggml_has_recurrent_qk_l2_norm = True

        if hasattr(lib, "test_recurrent_norm_gate"):
            lib.test_recurrent_norm_gate.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            lib.test_recurrent_norm_gate.restype = None
            self.ggml_has_recurrent_norm_gate = True

        if hasattr(lib, "test_attn_gate_sigmoid_mul"):
            lib.test_attn_gate_sigmoid_mul.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.test_attn_gate_sigmoid_mul.restype = None
            self.ggml_has_attn_gate_sigmoid_mul = True

        if hasattr(lib, "test_ssm_conv1d"):
            lib.test_ssm_conv1d.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.test_ssm_conv1d.restype = None
            self.ggml_has_ssm_conv = True

        if hasattr(lib, "test_gated_deltanet_autoregressive"):
            lib.test_gated_deltanet_autoregressive.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
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
            lib.test_gated_deltanet_autoregressive.restype = None
            self.ggml_has_deltanet = True

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

        if hasattr(lib, "ck_test_split_q_gate"):
            lib.ck_test_split_q_gate.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ck_test_split_q_gate.restype = None
            self.ck_has_split_q_gate = True

        if hasattr(lib, "ck_test_recurrent_split_qkv"):
            lib.ck_test_recurrent_split_qkv.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ck_test_recurrent_split_qkv.restype = None
            self.ck_has_recurrent_split_qkv = True

        if hasattr(lib, "ck_test_recurrent_dt_gate"):
            lib.ck_test_recurrent_dt_gate.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ck_test_recurrent_dt_gate.restype = None
            self.ck_has_recurrent_dt_gate = True

        if hasattr(lib, "ck_test_recurrent_conv_state_update"):
            lib.ck_test_recurrent_conv_state_update.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ck_test_recurrent_conv_state_update.restype = None
            self.ck_has_recurrent_conv_state_update = True

        if hasattr(lib, "ck_test_recurrent_silu"):
            lib.ck_test_recurrent_silu.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ck_test_recurrent_silu.restype = None
            self.ck_has_recurrent_silu = True

        if hasattr(lib, "ck_test_recurrent_split_conv_qkv"):
            lib.ck_test_recurrent_split_conv_qkv.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ck_test_recurrent_split_conv_qkv.restype = None
            self.ck_has_recurrent_split_conv_qkv = True

        if hasattr(lib, "ck_test_recurrent_qk_l2_norm"):
            lib.ck_test_recurrent_qk_l2_norm.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            lib.ck_test_recurrent_qk_l2_norm.restype = None
            self.ck_has_recurrent_qk_l2_norm = True

        if hasattr(lib, "ck_test_recurrent_norm_gate"):
            lib.ck_test_recurrent_norm_gate.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            lib.ck_test_recurrent_norm_gate.restype = None
            self.ck_has_recurrent_norm_gate = True

        if hasattr(lib, "ck_test_attn_gate_sigmoid_mul"):
            lib.ck_test_attn_gate_sigmoid_mul.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ck_test_attn_gate_sigmoid_mul.restype = None
            self.ck_has_attn_gate_sigmoid_mul = True

        if hasattr(lib, "ck_test_ssm_conv1d"):
            lib.ck_test_ssm_conv1d.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ck_test_ssm_conv1d.restype = None
            self.ck_has_ssm_conv = True

        if hasattr(lib, "ck_test_gated_deltanet_autoregressive"):
            lib.ck_test_gated_deltanet_autoregressive.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
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
            lib.ck_test_gated_deltanet_autoregressive.restype = None
            self.ck_has_deltanet = True

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
        """Test Q4_K GEMV (matrix-vector multiply)."""
        print(f"\n--- test_gemv_q4k (cols={cols}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        # Generate random weights and input
        q4k_weights = random_q4k_weights(cols)
        input_f32 = np.random.randn(cols).astype(np.float32)

        # GGML gemv (reference)
        ggml_out = np.zeros(1, dtype=np.float32)
        self.libggml.test_gemv_q4_k(q4k_weights,
                                     input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     cols)

        # CK gemv (quantized)
        ck_out = np.zeros(1, dtype=np.float32)
        self.libck.ck_test_gemv_q4_k(q4k_weights,
                                      input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      cols)

        return self.compare("gemv_q4_k", ggml_out, ck_out)

    def test_gemm_q4k(self, rows: int = 64, cols: int = 256, n_tokens: int = 4):
        """Test Q4_K GEMM (batched matrix multiply)."""
        print(f"\n--- test_gemm_q4k (rows={rows}, cols={cols}, tokens={n_tokens}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False

        q4k_weights = random_q4k_weights(rows * cols)
        input_f32 = np.random.randn(n_tokens, cols).astype(np.float32)

        # GGML GEMM (reference)
        ggml_out = np.zeros((n_tokens, rows), dtype=np.float32)
        self.libggml.test_gemm_q4_k(q4k_weights,
                                     input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     rows, cols, n_tokens)

        # CK GEMM (quantized)
        ck_out = np.zeros((n_tokens, rows), dtype=np.float32)
        self.libck.ck_test_gemm_q4_k(q4k_weights,
                                      input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      rows, cols, n_tokens)

        return self.compare("gemm_q4_k", ggml_out, ck_out)

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

    def test_recurrent_split_qkv(self, rows: int = 5, q_dim: int = 32, k_dim: int = 32, v_dim: int = 64):
        """Test packed recurrent QKV split."""
        print(f"\n--- test_recurrent_split_qkv (rows={rows}, q={q_dim}, k={k_dim}, v={v_dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_recurrent_split_qkv or not self.ck_has_recurrent_split_qkv:
            print(f"{RED}[FAIL]{RESET} Missing recurrent split parity symbol in helper library")
            self.results.append(("recurrent_split_qkv_symbol", False, float("inf"), float("inf")))
            return False

        packed = (0.25 * np.random.randn(rows, q_dim + k_dim + v_dim)).astype(np.float32)

        ggml_q = np.zeros((rows, q_dim), dtype=np.float32)
        ggml_k = np.zeros((rows, k_dim), dtype=np.float32)
        ggml_v = np.zeros((rows, v_dim), dtype=np.float32)
        ck_q = np.zeros_like(ggml_q)
        ck_k = np.zeros_like(ggml_k)
        ck_v = np.zeros_like(ggml_v)

        self.libggml.test_recurrent_split_qkv(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            q_dim,
            k_dim,
            v_dim,
        )

        self.libck.ck_test_recurrent_split_qkv(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            q_dim,
            k_dim,
            v_dim,
        )

        q_ok = self.compare("recurrent_split_q", ggml_q, ck_q)
        k_ok = self.compare("recurrent_split_k", ggml_k, ck_k)
        v_ok = self.compare("recurrent_split_v", ggml_v, ck_v)
        return q_ok and k_ok and v_ok

    def test_split_q_gate(self, rows: int = 5, q_dim: int = 64, gate_dim: int = 64, group_dim: int | None = None):
        """Test packed full-attention Q+gate split."""
        print(f"\n--- test_split_q_gate (rows={rows}, q={q_dim}, gate={gate_dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_split_q_gate or not self.ck_has_split_q_gate:
            print(f"{RED}[FAIL]{RESET} Missing split_q_gate parity symbol in helper library")
            self.results.append(("split_q_gate_symbol", False, float('inf'), float('inf')))
            return False

        if group_dim is None or group_dim <= 0:
            group_dim = q_dim
        num_groups = max(1, q_dim // group_dim)
        gate_group_dim = gate_dim // num_groups
        q_src = (0.25 * np.random.randn(rows, q_dim)).astype(np.float32)
        gate_src = (0.25 * np.random.randn(rows, gate_dim)).astype(np.float32)
        packed = np.zeros((rows, q_dim + gate_dim), dtype=np.float32)
        for group in range(num_groups):
            q0 = group * group_dim
            g0 = group * gate_group_dim
            dst0 = group * (group_dim + gate_group_dim)
            packed[:, dst0:dst0 + group_dim] = q_src[:, q0:q0 + group_dim]
            packed[:, dst0 + group_dim:dst0 + group_dim + gate_group_dim] = gate_src[:, g0:g0 + gate_group_dim]
        ggml_q = np.zeros((rows, q_dim), dtype=np.float32)
        ggml_gate = np.zeros((rows, gate_dim), dtype=np.float32)
        ck_q = np.zeros_like(ggml_q)
        ck_gate = np.zeros_like(ggml_gate)

        self.libggml.test_split_q_gate(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            q_dim,
            gate_dim,
            group_dim,
        )
        self.libck.ck_test_split_q_gate(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            q_dim,
            gate_dim,
            group_dim,
        )

        q_ok = self.compare("split_q_gate_q", ggml_q, ck_q)
        gate_ok = self.compare("split_q_gate_gate", ggml_gate, ck_gate)
        return q_ok and gate_ok

    def test_recurrent_dt_gate(self, rows: int = 7, dim: int = 16):
        """Test recurrent dt-gate transform."""
        print(f"\n--- test_recurrent_dt_gate (rows={rows}, dim={dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_recurrent_dt_gate or not self.ck_has_recurrent_dt_gate:
            print(f"{RED}[FAIL]{RESET} Missing recurrent dt gate parity symbol in helper library")
            self.results.append(("recurrent_dt_gate_symbol", False, float("inf"), float("inf")))
            return False

        alpha = (0.30 * np.random.randn(rows, dim)).astype(np.float32)
        dt_bias = (0.20 * np.random.randn(dim)).astype(np.float32)
        a = (0.25 * np.random.randn(dim)).astype(np.float32)

        ggml_gate = np.zeros((rows, dim), dtype=np.float32)
        ck_gate = np.zeros_like(ggml_gate)

        self.libggml.test_recurrent_dt_gate(
            alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dt_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            dim,
        )

        self.libck.ck_test_recurrent_dt_gate(
            alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dt_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            dim,
        )

        return self.compare("recurrent_dt_gate", ggml_gate, ck_gate)

    def test_recurrent_conv_state_update(self, history_len: int = 3, num_seqs: int = 2, num_tokens: int = 4,
                                         q_dim: int = 16, k_dim: int = 16, v_dim: int = 24):
        """Test recurrent convolution state window construction."""
        print(
            f"\n--- test_recurrent_conv_state_update (history={history_len}, seqs={num_seqs}, "
            f"tokens={num_tokens}, q={q_dim}, k={k_dim}, v={v_dim}) ---"
        )

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_recurrent_conv_state_update or not self.ck_has_recurrent_conv_state_update:
            print(f"{RED}[FAIL]{RESET} Missing recurrent conv-state parity symbol in helper library")
            self.results.append(("recurrent_conv_state_update_symbol", False, float("inf"), float("inf")))
            return False

        channels = q_dim + k_dim + v_dim
        state_in = (0.20 * np.random.randn(num_seqs, channels, history_len)).astype(np.float32)
        q = (0.25 * np.random.randn(num_seqs * num_tokens, q_dim)).astype(np.float32)
        k = (0.25 * np.random.randn(num_seqs * num_tokens, k_dim)).astype(np.float32)
        v = (0.25 * np.random.randn(num_seqs * num_tokens, v_dim)).astype(np.float32)

        ggml_conv_x = np.zeros((num_seqs, channels, history_len + num_tokens), dtype=np.float32)
        ck_conv_x = np.zeros_like(ggml_conv_x)
        ggml_state_out = np.zeros((num_seqs, channels, history_len), dtype=np.float32)
        ck_state_out = np.zeros_like(ggml_state_out)

        self.libggml.test_recurrent_conv_state_update(
            state_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_conv_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_state_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            history_len,
            num_seqs,
            num_tokens,
            q_dim,
            k_dim,
            v_dim,
        )

        self.libck.ck_test_recurrent_conv_state_update(
            state_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_conv_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_state_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            history_len,
            num_seqs,
            num_tokens,
            q_dim,
            k_dim,
            v_dim,
        )

        conv_ok = self.compare("recurrent_conv_state_update_conv_x", ggml_conv_x, ck_conv_x)
        state_ok = self.compare("recurrent_conv_state_update_state_out", ggml_state_out, ck_state_out)
        return conv_ok and state_ok

    def test_recurrent_silu(self, rows: int = 7, dim: int = 64):
        """Test recurrent SiLU activation."""
        print(f"\n--- test_recurrent_silu (rows={rows}, dim={dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_recurrent_silu or not self.ck_has_recurrent_silu:
            print(f"{RED}[FAIL]{RESET} Missing recurrent SiLU parity symbol in helper library")
            self.results.append(("recurrent_silu_symbol", False, float("inf"), float("inf")))
            return False

        x = (0.25 * np.random.randn(rows, dim)).astype(np.float32)
        ggml_out = np.zeros((rows, dim), dtype=np.float32)
        ck_out = np.zeros_like(ggml_out)

        self.libggml.test_recurrent_silu(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            dim,
        )
        self.libck.ck_test_recurrent_silu(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            dim,
        )

        return self.compare("recurrent_silu", ggml_out, ck_out)

    def test_recurrent_split_conv_qkv(self, rows: int = 5, q_dim: int = 64, k_dim: int = 64, v_dim: int = 128):
        """Test post-convolution recurrent QKV split."""
        print(f"\n--- test_recurrent_split_conv_qkv (rows={rows}, q={q_dim}, k={k_dim}, v={v_dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_recurrent_split_conv_qkv or not self.ck_has_recurrent_split_conv_qkv:
            print(f"{RED}[FAIL]{RESET} Missing recurrent post-conv split parity symbol in helper library")
            self.results.append(("recurrent_split_conv_qkv_symbol", False, float("inf"), float("inf")))
            return False

        packed = (0.25 * np.random.randn(rows, q_dim + k_dim + v_dim)).astype(np.float32)
        ggml_q = np.zeros((rows, q_dim), dtype=np.float32)
        ggml_k = np.zeros((rows, k_dim), dtype=np.float32)
        ggml_v = np.zeros((rows, v_dim), dtype=np.float32)
        ck_q = np.zeros_like(ggml_q)
        ck_k = np.zeros_like(ggml_k)
        ck_v = np.zeros_like(ggml_v)

        self.libggml.test_recurrent_split_conv_qkv(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            q_dim,
            k_dim,
            v_dim,
        )
        self.libck.ck_test_recurrent_split_conv_qkv(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            q_dim,
            k_dim,
            v_dim,
        )

        q_ok = self.compare("recurrent_split_conv_q", ggml_q, ck_q)
        k_ok = self.compare("recurrent_split_conv_k", ggml_k, ck_k)
        v_ok = self.compare("recurrent_split_conv_v", ggml_v, ck_v)
        return q_ok and k_ok and v_ok

    def test_recurrent_qk_l2_norm(self, rows: int = 5, q_dim: int = 64, k_dim: int = 64, head_dim: int = 16):
        """Test recurrent per-head L2 normalization for Q/K."""
        print(f"\n--- test_recurrent_qk_l2_norm (rows={rows}, q={q_dim}, k={k_dim}, head_dim={head_dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_recurrent_qk_l2_norm or not self.ck_has_recurrent_qk_l2_norm:
            print(f"{RED}[FAIL]{RESET} Missing recurrent Q/K L2 norm parity symbol in helper library")
            self.results.append(("recurrent_qk_l2_norm_symbol", False, float('inf'), float('inf')))
            return False

        q = (0.20 * np.random.randn(rows, q_dim)).astype(np.float32)
        k = (0.20 * np.random.randn(rows, k_dim)).astype(np.float32)
        ggml_q = q.copy()
        ggml_k = k.copy()
        ck_q = q.copy()
        ck_k = k.copy()
        eps = ctypes.c_float(1e-5)

        self.libggml.test_recurrent_qk_l2_norm(
            ggml_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            q_dim,
            k_dim,
            head_dim,
            eps,
        )
        self.libck.ck_test_recurrent_qk_l2_norm(
            ck_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            q_dim,
            k_dim,
            head_dim,
            eps,
        )

        q_ok = self.compare("recurrent_qk_l2_norm_q", ggml_q, ck_q)
        k_ok = self.compare("recurrent_qk_l2_norm_k", ggml_k, ck_k)
        return q_ok and k_ok

    def test_recurrent_norm_gate(self, rows: int = 4, num_heads: int = 8, head_dim: int = 16):
        """Test recurrent per-head RMSNorm followed by SiLU(gate)."""
        print(f"\n--- test_recurrent_norm_gate (rows={rows}, heads={num_heads}, head_dim={head_dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_recurrent_norm_gate or not self.ck_has_recurrent_norm_gate:
            print(f"{RED}[FAIL]{RESET} Missing recurrent norm-gate parity symbol in helper library")
            self.results.append(("recurrent_norm_gate_symbol", False, float("inf"), float("inf")))
            return False

        dim = num_heads * head_dim
        x = (0.25 * np.random.randn(rows, dim)).astype(np.float32)
        gate = (0.25 * np.random.randn(rows, dim)).astype(np.float32)
        weight = (0.20 * np.random.randn(head_dim)).astype(np.float32)
        ggml_out = np.zeros((rows, dim), dtype=np.float32)
        ck_out = np.zeros_like(ggml_out)

        self.libggml.test_recurrent_norm_gate(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            num_heads,
            head_dim,
            ctypes.c_float(1e-6),
        )
        self.libck.ck_test_recurrent_norm_gate(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            num_heads,
            head_dim,
            ctypes.c_float(1e-6),
        )

        return self.compare("recurrent_norm_gate", ggml_out, ck_out)

    def test_attn_gate_sigmoid_mul(self, rows: int = 6, dim: int = 128):
        """Test full-attention gate sigmoid multiply."""
        print(f"\n--- test_attn_gate_sigmoid_mul (rows={rows}, dim={dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_attn_gate_sigmoid_mul or not self.ck_has_attn_gate_sigmoid_mul:
            print(f"{RED}[FAIL]{RESET} Missing attn_gate_sigmoid_mul parity symbol in helper library")
            self.results.append(("attn_gate_sigmoid_mul_symbol", False, float('inf'), float('inf')))
            return False

        x = (0.25 * np.random.randn(rows, dim)).astype(np.float32)
        gate = (0.25 * np.random.randn(rows, dim)).astype(np.float32)
        ggml_out = np.zeros((rows, dim), dtype=np.float32)
        ck_out = np.zeros_like(ggml_out)

        self.libggml.test_attn_gate_sigmoid_mul(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            dim,
        )
        self.libck.ck_test_attn_gate_sigmoid_mul(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            dim,
        )

        return self.compare("attn_gate_sigmoid_mul", ggml_out, ck_out)

    def test_ssm_conv1d(self, kernel_size: int = 4, num_channels: int = 48, num_tokens: int = 7, num_seqs: int = 2):
        """Test qwen3next/Qwen3.5 SSM causal depthwise convolution."""
        print(
            f"\n--- test_ssm_conv1d (kernel={kernel_size}, channels={num_channels}, "
            f"tokens={num_tokens}, seqs={num_seqs}) ---"
        )

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_ssm_conv or not self.ck_has_ssm_conv:
            print(f"{RED}[FAIL]{RESET} Missing SSM conv parity symbol in helper library")
            self.results.append(("ssm_conv1d_symbol", False, float("inf"), float("inf")))
            return False

        seq_width = kernel_size - 1 + num_tokens
        conv_x = (0.25 * np.random.randn(num_seqs, num_channels, seq_width)).astype(np.float32)
        kernel = (0.20 * np.random.randn(num_channels, kernel_size)).astype(np.float32)

        ggml_out = np.zeros((num_seqs, num_tokens, num_channels), dtype=np.float32)
        ck_out = np.zeros((num_seqs, num_tokens, num_channels), dtype=np.float32)

        self.libggml.test_ssm_conv1d(
            conv_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kernel_size,
            num_channels,
            num_tokens,
            num_seqs,
        )

        self.libck.ck_test_ssm_conv1d(
            conv_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kernel_size,
            num_channels,
            num_tokens,
            num_seqs,
        )

        return self.compare("ssm_conv1d", ggml_out, ck_out)

    def test_gated_deltanet_autoregressive(self, num_heads: int = 4, state_dim: int = 16):
        """Test Qwen3.5-style single-token Gated DeltaNet state update."""
        print(f"\n--- test_gated_deltanet_autoregressive (heads={num_heads}, dim={state_dim}) ---")

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available{RESET}")
            return False
        if not self.ggml_has_deltanet or not self.ck_has_deltanet:
            print(f"{RED}[FAIL]{RESET} Missing Gated DeltaNet parity symbol in helper library")
            self.results.append(("gated_deltanet_symbol", False, float("inf"), float("inf")))
            return False

        q = (0.25 * np.random.randn(num_heads, state_dim)).astype(np.float32)
        k = (0.25 * np.random.randn(num_heads, state_dim)).astype(np.float32)
        v = (0.25 * np.random.randn(num_heads, state_dim)).astype(np.float32)
        g = (0.1 * np.random.randn(num_heads)).astype(np.float32)
        beta = (0.5 * np.random.randn(num_heads)).astype(np.float32)
        state_in = (0.2 * np.random.randn(num_heads, state_dim, state_dim)).astype(np.float32)
        norm_eps = 1e-6

        ggml_state = np.zeros_like(state_in)
        ggml_out = np.zeros((num_heads, state_dim), dtype=np.float32)
        ck_state = np.zeros_like(state_in)
        ck_out = np.zeros((num_heads, state_dim), dtype=np.float32)

        self.libggml.test_gated_deltanet_autoregressive(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            state_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_heads,
            state_dim,
            ctypes.c_float(norm_eps),
        )

        self.libck.ck_test_gated_deltanet_autoregressive(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            state_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_heads,
            state_dim,
            ctypes.c_float(norm_eps),
        )

        out_ok = self.compare("gated_deltanet_out", ggml_out, ck_out)
        state_ok = self.compare("gated_deltanet_state", ggml_state, ck_state)
        return out_ok and state_ok

    def run_all(self, quick=False):
        """Run all kernel tests."""
        print("=" * 80)
        print(f"{BOLD}KERNEL PARITY TESTS: C-Kernel-Engine vs llama.cpp/ggml{RESET}")
        print("=" * 80)
        print(f"""
{YELLOW}Purpose:{RESET}  Verify CK kernels produce identical outputs to llama.cpp/ggml
{YELLOW}Method:{RESET}   Run both implementations with same inputs, compare outputs
{YELLOW}Tolerance:{RESET} max_diff < {self.tol:.0e} for all elements

{YELLOW}Kernels Tested:{RESET}
  - Dequantization: Q4_K, Q4_0 (convert quantized weights to FP32)
  - GEMV:          Q4_K matrix-vector multiply (decode path)
  - GEMM:          Q4_K matrix-matrix multiply (prefill path)
  - Activations:   RMSNorm, RoPE, SwiGLU, Softmax
  - Full-attn prep: split_q_gate, sigmoid gate * attention output
  - Recurrent split: Packed QKV -> Q/K/V
  - Recurrent gate: softplus(alpha + dt_bias) * a
  - Recurrent prep: conv-state update, SSM conv, SiLU, packed splits, Q/K L2 norm
  - Recurrent finish: per-head gated RMSNorm
  - Recurrent:     Gated DeltaNet autoregressive state update
""")

        # Dequantization kernels
        self.test_dequant_q4k(256)
        if not quick:
            self.test_dequant_q4k(512)
        self.test_dequant_q4_0(32)
        if not quick:
            self.test_dequant_q4_0(64)

        # Quantized GEMV/GEMM
        self.test_gemv_q4k(256)
        if not quick:
            self.test_gemv_q4k(512)
        self.test_gemm_q4k(64, 256, 4)
        if not quick:
            self.test_gemm_q4k(256, 512, 8)

        # Activation kernels
        self.test_rmsnorm(4, 256)
        if not quick:
            self.test_rmsnorm(1, 2048)

        self.test_rope(4, 8, 64)
        if not quick:
            self.test_rope(1, 32, 128)

        self.test_swiglu(4, 256)
        if not quick:
            self.test_swiglu(1, 1024)

        self.test_softmax(128)
        if not quick:
            self.test_softmax(512)

        self.test_split_q_gate(5, 64, 64, 64)
        if not quick:
            self.test_split_q_gate(7, 256, 256, 64)
        self.test_recurrent_split_qkv(5, 32, 32, 64)
        if not quick:
            self.test_recurrent_split_qkv(7, 128, 128, 256)
        self.test_recurrent_dt_gate(7, 16)
        if not quick:
            self.test_recurrent_dt_gate(11, 32)
        self.test_recurrent_conv_state_update(3, 2, 4, 16, 16, 24)
        if not quick:
            self.test_recurrent_conv_state_update(4, 3, 5, 32, 32, 48)
        self.test_recurrent_silu(7, 64)
        if not quick:
            self.test_recurrent_silu(11, 128)
        self.test_recurrent_split_conv_qkv(5, 64, 64, 128)
        if not quick:
            self.test_recurrent_split_conv_qkv(9, 128, 128, 256)
        self.test_recurrent_qk_l2_norm(5, 64, 64, 16)
        if not quick:
            self.test_recurrent_qk_l2_norm(7, 2048, 2048, 128)
        self.test_recurrent_norm_gate(4, 8, 16)
        if not quick:
            self.test_recurrent_norm_gate(7, 16, 32)
        self.test_attn_gate_sigmoid_mul(6, 128)
        if not quick:
            self.test_attn_gate_sigmoid_mul(9, 256)

        self.test_ssm_conv1d(4, 48, 7, 2)
        if not quick:
            self.test_ssm_conv1d(4, 96, 13, 3)

        self.test_gated_deltanet_autoregressive(4, 16)
        if not quick:
            self.test_gated_deltanet_autoregressive(8, 32)
            self.test_gated_deltanet_autoregressive(16, 64)

        # Summary
        return self.print_summary()

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

    def benchmark_gemv_q4k(self, rows: int, cols: int, warmup: int = 5, iters: int = 50):
        """Benchmark Q4_K GEMV performance for both CK and llama.cpp."""
        if not self.libggml or not self.libck:
            return None, None

        # Generate test data
        q4k_weights = random_q4k_weights(rows * cols)
        input_f32 = np.random.randn(cols).astype(np.float32)

        ggml_out = np.zeros(rows, dtype=np.float32)
        ck_out = np.zeros(rows, dtype=np.float32)

        # Warmup llama.cpp
        for _ in range(warmup):
            self.libggml.test_gemv_q4_k(q4k_weights,
                                         input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         cols)

        # Time llama.cpp
        t0 = time.perf_counter()
        for _ in range(iters):
            self.libggml.test_gemv_q4_k(q4k_weights,
                                         input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         cols)
        t1 = time.perf_counter()
        ggml_time_ms = (t1 - t0) / iters * 1000

        # Warmup CK
        for _ in range(warmup):
            self.libck.ck_test_gemv_q4_k(q4k_weights,
                                          input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          cols)

        # Time CK
        t0 = time.perf_counter()
        for _ in range(iters):
            self.libck.ck_test_gemv_q4_k(q4k_weights,
                                          input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          cols)
        t1 = time.perf_counter()
        ck_time_ms = (t1 - t0) / iters * 1000

        return ggml_time_ms, ck_time_ms

    def benchmark_gemm_q4k(self, rows: int, cols: int, tokens: int, warmup: int = 5, iters: int = 50):
        """Benchmark Q4_K GEMM performance for both CK and llama.cpp."""
        if not self.libggml or not self.libck:
            return None, None

        q4k_weights = random_q4k_weights(rows * cols)
        input_f32 = np.random.randn(tokens, cols).astype(np.float32)

        ggml_out = np.zeros((tokens, rows), dtype=np.float32)
        ck_out = np.zeros((tokens, rows), dtype=np.float32)

        # Warmup llama.cpp
        for _ in range(warmup):
            self.libggml.test_gemm_q4_k(q4k_weights,
                                         input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         rows, cols, tokens)

        # Time llama.cpp
        t0 = time.perf_counter()
        for _ in range(iters):
            self.libggml.test_gemm_q4_k(q4k_weights,
                                         input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         ggml_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         rows, cols, tokens)
        t1 = time.perf_counter()
        ggml_time_ms = (t1 - t0) / iters * 1000

        # Warmup CK
        for _ in range(warmup):
            self.libck.ck_test_gemm_q4_k(q4k_weights,
                                          input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          rows, cols, tokens)

        # Time CK
        t0 = time.perf_counter()
        for _ in range(iters):
            self.libck.ck_test_gemm_q4_k(q4k_weights,
                                          input_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          ck_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          rows, cols, tokens)
        t1 = time.perf_counter()
        ck_time_ms = (t1 - t0) / iters * 1000

        return ggml_time_ms, ck_time_ms

    def run_performance_benchmarks(self, large=False):
        """Run performance benchmarks comparing CK vs llama.cpp."""
        print("\n" + "=" * 80)
        print(f"{BOLD}KERNEL PERFORMANCE BENCHMARKS: C-Kernel-Engine vs llama.cpp{RESET}")
        print("=" * 80)

        if not self.libggml or not self.libck:
            print(f"{YELLOW}[SKIP] Libraries not available for performance testing{RESET}")
            return

        # Test configurations based on model sizes
        # Note: Q4_K block size is QK_K=256, so dimensions must be multiples of 256
        if large:
            # 7B model dimensions (rounded to 256 for Q4_K alignment)
            # Original LLaMA 7B: hidden=4096, ff=11008
            # Rounded: ff=11264 (44x256)
            configs = [
                ("GEMV qkv_proj_7b", 4096, 4096, 1),
                ("GEMV mlp_up_7b", 11264, 4096, 1),
                ("GEMV mlp_down_7b", 4096, 11264, 1),
                ("GEMV embed_7b", 32000, 4096, 1),
                ("GEMM prefill_7b", 4096, 4096, 128),
            ]
        else:
            # 0.5B model dimensions (rounded to 256 for Q4_K alignment)
            # Original Qwen 0.5B: hidden=896, ff=4864
            # Rounded for Q4_K: hidden=768 (3x256), ff=4864 (19x256)
            configs = [
                ("GEMV qkv_proj", 768, 768, 1),
                ("GEMV mlp_up", 4864, 768, 1),
                ("GEMV mlp_down", 768, 4864, 1),
                ("GEMM prefill", 768, 768, 64),
                ("GEMM large_prefill", 4864, 768, 64),
            ]

        print(f"\n{'Config':<25} {'M':>8} {'K':>8} {'T':>4} {'llama.cpp':>12} {'CK':>12} {'Speedup':>10} {'CK GFLOPS':>12}")
        print("-" * 95)

        perf_results = []
        for name, rows, cols, tokens in configs:
            if tokens == 1:
                ggml_ms, ck_ms = self.benchmark_gemv_q4k(rows, cols)
            else:
                ggml_ms, ck_ms = self.benchmark_gemm_q4k(rows, cols, tokens)

            if ggml_ms is None or ck_ms is None:
                print(f"{name:<25} {'SKIP':>8}")
                continue

            # Calculate GFLOPS: 2*M*K*T FLOPs (multiply-add)
            flops = 2.0 * rows * cols * tokens
            ck_gflops = (flops / 1e9) / (ck_ms / 1000)
            ggml_gflops = (flops / 1e9) / (ggml_ms / 1000)

            speedup = ggml_ms / ck_ms if ck_ms > 0 else 0
            speedup_str = f"{speedup:.2f}x" if speedup >= 1.0 else f"{speedup:.2f}x"
            color = GREEN if speedup >= 0.9 else (YELLOW if speedup >= 0.5 else RED)

            print(f"{name:<25} {rows:>8} {cols:>8} {tokens:>4} {ggml_ms:>10.3f}ms {ck_ms:>10.3f}ms {color}{speedup_str:>10}{RESET} {ck_gflops:>10.2f}")
            perf_results.append((name, rows, cols, tokens, ggml_ms, ck_ms, speedup, ck_gflops))

        print("-" * 95)

        # Calculate average speedup and GFLOPS
        if perf_results:
            avg_speedup = sum(r[6] for r in perf_results) / len(perf_results)
            avg_gflops = sum(r[7] for r in perf_results) / len(perf_results)
            avg_ggml_gflops = sum((2.0 * r[1] * r[2] * r[3] / 1e9) / (r[4] / 1000) for r in perf_results) / len(perf_results)

            print(f"\n{'Average Speedup (CK/llama.cpp):':<35} {avg_speedup:.2f}x")
            print(f"{'Average CK GFLOPS:':<35} {avg_gflops:.2f}")
            print(f"{'Average llama.cpp GFLOPS:':<35} {avg_ggml_gflops:.2f}")

            # Estimate decode throughput for Qwen 0.5B
            if not large:
                # 324 Q5_0 GEMVs + 36 Q4_K GEMVs per token
                # Use mlp_down (896x4864) as representative GEMV time
                gemv_time_ms = next((r[5] for r in perf_results if "mlp_down" in r[0] and r[3] == 1), None)
                if gemv_time_ms:
                    gemv_per_token = 324 + 36  # approximate
                    ms_per_token = gemv_time_ms * gemv_per_token
                    tok_per_s = 1000.0 / ms_per_token

                    print(f"\n{'='*60}")
                    print(f"{BOLD}DECODE THROUGHPUT ESTIMATE (Qwen 0.5B){RESET}")
                    print(f"{'='*60}")
                    print(f"  Representative GEMV time:   {gemv_time_ms:.3f} ms")
                    print(f"  GEMVs per token:            ~{gemv_per_token}")
                    print(f"  Estimated CK decode:        {tok_per_s:.2f} tok/s")
                    print(f"  llama.cpp reference:        ~35 tok/s")
                    print(f"  Gap:                        {35.0 / tok_per_s:.1f}x")
                    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Kernel-level parity tests: CK vs llama.cpp")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--kernel", type=str, help="Test specific kernel")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance (default: 1e-3)")
    parser.add_argument("--perf", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--perf-large", action="store_true", help="Run performance benchmarks with 7B dimensions")
    args = parser.parse_args()

    libggml, libck = load_libraries()

    # Allow graceful skip on CI when libraries aren't available
    skip_if_missing = os.environ.get("CK_SKIP_IF_MISSING", "0") == "1"

    if not libggml:
        print(f"\n{YELLOW}WARNING: Could not load llama.cpp kernel test library.{RESET}")
        print("Build it with:")
        print("  cd llama.cpp")
        print("  g++ -shared -fPIC -o libggml_kernel_test.so \\")
        print("      tests/test-kernel-parity.cpp \\")
        print("      -I ggml/include -I ggml/src \\")
        print("      -L build -lggml -lm -lpthread")
        if skip_if_missing:
            print(f"\n{YELLOW}SKIP: CK_SKIP_IF_MISSING=1, skipping kernel tests{RESET}")
            sys.exit(0)
        sys.exit(1)

    if not libck:
        print(f"\n{YELLOW}WARNING: Could not load CK parity library.{RESET}")
        print("Build it with:")
        print("  make libck_parity.so")
        if skip_if_missing:
            print(f"\n{YELLOW}SKIP: CK_SKIP_IF_MISSING=1, skipping kernel tests{RESET}")
            sys.exit(0)
        sys.exit(1)

    tester = KernelTester(libggml, libck, tol=args.tol)

    if args.perf or args.perf_large:
        # Run performance benchmarks
        tester.run_performance_benchmarks(large=args.perf_large)
        sys.exit(0)
    elif args.kernel:
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
        success = tester.run_all(quick=args.quick)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
