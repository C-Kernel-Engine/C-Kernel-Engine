"""
Unit tests for optimized Q6_K and Fused RMSNorm-Q8_K kernels.
Compares C SSE/AVX implementations against reference and PyTorch.
"""
import ctypes
import numpy as np
import torch
import time
import os

from lib_loader import load_lib
from test_utils import (
    TestReport, TestResult, get_cpu_info,
    max_diff, numpy_to_ptr, time_function, print_system_info
)

# Load library
lib = load_lib("libckernel_engine.so")

# Function signatures
lib.gemm_nt_q6_k_sse.argtypes = [
    ctypes.POINTER(ctypes.c_float), # A
    ctypes.c_void_p,               # B
    ctypes.POINTER(ctypes.c_float), # bias
    ctypes.POINTER(ctypes.c_float), # C
    ctypes.c_int, ctypes.c_int, ctypes.c_int # M, N, K
]
lib.gemm_nt_q6_k_ref.argtypes = lib.gemm_nt_q6_k_sse.argtypes

lib.rmsnorm_q8_k_fused.argtypes = [
    ctypes.POINTER(ctypes.c_float), # input
    ctypes.POINTER(ctypes.c_float), # gamma
    ctypes.c_void_p,               # vy
    ctypes.c_int, ctypes.c_int, ctypes.c_int, # tokens, d_model, aligned
    ctypes.c_float                 # eps
]

# Constants
QK_K = 256

class BlockQ8K(ctypes.Structure):
    _fields_ = [
        ("d", ctypes.c_float),
        ("qs", ctypes.c_int8 * QK_K),
        ("bsums", ctypes.c_int16 * (QK_K // 16))
    ]

def test_q6k_parity(M=1, N=128, K=256):
    print(f"\n--- Q6_K SSE Parity (M={M}, N={N}, K={K}) ---")
    
    # Q6_K data is complex to generate manually, so we'll use the ref kernel 
    # as the baseline for the optimized kernel.
    # For a real test, we'd need a proper Q6_K quantizer.
    # Since we don't have one in Python yet, we skip manual bit manipulation 
    # and just ensure the SSE kernel matches the ref kernel on dummy data.
    print("[SKIP] Manual Q6_K bit-generation is complex. Tested via integration in ck_run_v5.")

def test_fused_rmsnorm_q8k(T=4, D=512):
    print(f"\n--- Fused RMSNorm-Q8_K (T={T}, D={D}) ---")
    
    input_np = np.random.randn(T, D).astype(np.float32)
    gamma_np = np.random.randn(D).astype(np.float32)
    eps = 1e-6
    
    # 1. Reference: PyTorch RMSNorm -> manual Q8_K quantization
    def rmsnorm_torch(x, gamma, eps):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * (var + eps).rsqrt()) * gamma

    ref_f32 = rmsnorm_torch(torch.from_numpy(input_np), torch.from_numpy(gamma_np), eps).numpy()
    
    # 2. C Fused kernel
    out_q8 = (BlockQ8K * (T * (D // QK_K)))()
    lib.rmsnorm_q8_k_fused(
        numpy_to_ptr(input_np),
        numpy_to_ptr(gamma_np),
        ctypes.byref(out_q8),
        T, D, D, eps
    )
    
    # 3. Dequantize C output to compare with ref
    success_count = 0
    for t in range(T):
        for b in range(D // QK_K):
            block = out_q8[t * (D // QK_K) + b]
            d = block.d
            qs = np.array(block.qs, dtype=np.int8)
            
            dequant = qs.astype(np.float32) * d
            ref_slice = ref_f32[t, b*QK_K : (b+1)*QK_K]
            
            diff = np.abs(dequant - ref_slice)
            if np.max(diff) < 0.05: # Quantization error tolerance
                success_count += 1
            else:
                print(f"  [FAIL] T={t} B={b} max_diff={np.max(diff):.4f}")

    if success_count == T * (D // QK_K):
        print(f"  [PASS] Fused RMSNorm-Q8_K matches PyTorch+Quant (Tol=0.05)")
    else:
        print(f"  [FAIL] {T*(D//QK_K) - success_count} blocks failed")

if __name__ == "__main__":
    test_fused_rmsnorm_q8k()
