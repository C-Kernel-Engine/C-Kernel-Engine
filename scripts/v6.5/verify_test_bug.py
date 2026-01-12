#!/usr/bin/env python3
"""Verify the parity test bug - it only tests 1 row, not M rows."""
import ctypes
import time

PARITY_LIB = "/home/antshiv/Workspace/C-Kernel-Engine/build/libck_parity.so"
lib = ctypes.CDLL(PARITY_LIB)

# Test dimensions
M_full = 896
M_single = 1
K = 896

# Block sizes
QK5_0 = 32
BLOCK_Q5_0_SIZE = 22
BLOCK_Q8_0_SIZE = 34

# Allocate
n_blocks_w = (M_full * K) // QK5_0
weights = (ctypes.c_uint8 * (n_blocks_w * BLOCK_Q5_0_SIZE))()
input_f32 = (ctypes.c_float * K)()
output_full = (ctypes.c_float * M_full)()
output_single = (ctypes.c_float * M_single)()

# Setup function
lib.ck_test_gemv_q5_0_q8_0.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int
]
lib.ck_test_gemv_q5_0_q8_0.restype = None

# Warmup
for _ in range(5):
    lib.ck_test_gemv_q5_0_q8_0(weights, input_f32, output_single, M_single, K)
    lib.ck_test_gemv_q5_0_q8_0(weights, input_f32, output_full, M_full, K)

n_iter = 50

# Time M=1 (what parity test actually does)
t0 = time.perf_counter()
for _ in range(n_iter):
    lib.ck_test_gemv_q5_0_q8_0(weights, input_f32, output_single, M_single, K)
t1 = time.perf_counter()
single_time = (t1 - t0) / n_iter * 1000

# Time M=896 (what we need for inference)
t0 = time.perf_counter()
for _ in range(n_iter):
    lib.ck_test_gemv_q5_0_q8_0(weights, input_f32, output_full, M_full, K)
t1 = time.perf_counter()
full_time = (t1 - t0) / n_iter * 1000

print("Q5_0 GEMV timing comparison:")
print(f"  M=1 (what test does):     {single_time:.4f} ms")
print(f"  M=896 (what we need):     {full_time:.4f} ms")
print(f"  Ratio:                    {full_time/single_time:.1f}x")
print()
print("  Parity test reports '896x896' but only tests 1 row!")
print("  This explains why test shows 0.004ms but inference is slow.")
