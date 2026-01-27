#!/usr/bin/env python3
"""
Test gemv_q5_0_q8_0 kernel directly from inference library.
Compare timing with parity test.
"""
import ctypes
import time
import numpy as np

# Load inference library
LIB_PATH = "/home/antshiv/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.so"
lib = ctypes.CDLL(LIB_PATH)

# Load parity library for comparison
PARITY_LIB = "/home/antshiv/Workspace/C-Kernel-Engine/build/libck_parity.so"
parity_lib = ctypes.CDLL(PARITY_LIB)

# Setup function signatures
lib.gemv_q5_0_q8_0.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # y
    ctypes.c_void_p,                  # W (Q5_0)
    ctypes.c_void_p,                  # x (Q8_0)
    ctypes.c_int,                     # M
    ctypes.c_int,                     # K
]
lib.gemv_q5_0_q8_0.restype = None

parity_lib.gemv_q5_0_q8_0.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]
parity_lib.gemv_q5_0_q8_0.restype = None

# Test dimensions (Qwen QKV: 896x896)
M = 896
K = 896

# Block sizes
QK5_0 = 32
QK8_0 = 32
BLOCK_Q5_0_SIZE = 22  # 2 (d) + 4 (qh) + 16 (qs)
BLOCK_Q8_0_SIZE = 34  # 2 (d) + 32 (qs)

# Allocate buffers
n_blocks_w = (M * K) // QK5_0
n_blocks_x = K // QK8_0

weights = (ctypes.c_uint8 * (n_blocks_w * BLOCK_Q5_0_SIZE))()
x_q8 = (ctypes.c_uint8 * (n_blocks_x * BLOCK_Q8_0_SIZE))()
y_infer = (ctypes.c_float * M)()
y_parity = (ctypes.c_float * M)()

# Initialize with random data
import random
for i in range(len(weights)):
    weights[i] = random.randint(0, 255)
for i in range(len(x_q8)):
    x_q8[i] = random.randint(0, 255)

# Warmup
for _ in range(5):
    lib.gemv_q5_0_q8_0(y_infer, weights, x_q8, M, K)
    parity_lib.gemv_q5_0_q8_0(y_parity, weights, x_q8, M, K)

# Time inference library
n_iter = 100
t0 = time.perf_counter()
for _ in range(n_iter):
    lib.gemv_q5_0_q8_0(y_infer, weights, x_q8, M, K)
t1 = time.perf_counter()
infer_time = (t1 - t0) / n_iter * 1000

# Time parity library
t0 = time.perf_counter()
for _ in range(n_iter):
    parity_lib.gemv_q5_0_q8_0(y_parity, weights, x_q8, M, K)
t1 = time.perf_counter()
parity_time = (t1 - t0) / n_iter * 1000

print(f"Test: gemv_q5_0_q8_0 ({M}x{K})")
print(f"  Inference lib: {infer_time:.4f} ms")
print(f"  Parity lib:    {parity_time:.4f} ms")
print(f"  Ratio:         {parity_time/infer_time:.2f}x")
print()

# Check if results match
match = True
for i in range(M):
    if abs(y_infer[i] - y_parity[i]) > 1e-5:
        match = False
        break
print(f"  Results match: {match}")

# Calculate expected full decode time
n_gemv_per_token = 216
expected_decode_ms = n_gemv_per_token * infer_time
print(f"\n  If {n_gemv_per_token} GEMVs/token @ {infer_time:.4f}ms each:")
print(f"    Expected: {expected_decode_ms:.2f} ms/token ({1000/expected_decode_ms:.1f} tok/s)")
print(f"    Actual:   ~600 ms/token (1.4 tok/s)")
