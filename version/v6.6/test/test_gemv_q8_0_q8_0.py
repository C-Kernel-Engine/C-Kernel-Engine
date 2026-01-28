#!/usr/bin/env python3
"""
Test gemv_q8_0_q8_0 kernel directly to see if it produces NaN.
"""

import ctypes
import numpy as np
from pathlib import Path

# Load the library
lib_path = Path("/home/antshiv/Workspace/C-Kernel-Engine/build/libckernel_engine.so")
if not lib_path.exists():
    print(f"Error: {lib_path} not found")
    exit(1)

lib = ctypes.CDLL(str(lib_path))

# Q8_0 block structure: 2 bytes (fp16 scale) + 32 bytes (int8 quants) = 34 bytes per 32 elements
QK8_0 = 32
BLOCK_Q8_0_SIZE = 34

def quantize_to_q8_0(x: np.ndarray) -> bytes:
    """Quantize FP32 array to Q8_0 format."""
    assert len(x) % QK8_0 == 0, f"Length {len(x)} must be multiple of {QK8_0}"

    num_blocks = len(x) // QK8_0
    output = bytearray()

    for b in range(num_blocks):
        block = x[b * QK8_0 : (b + 1) * QK8_0]

        # Find max absolute value for scale
        amax = np.abs(block).max()
        if amax == 0:
            scale = np.float16(0.0)
            quants = np.zeros(QK8_0, dtype=np.int8)
        else:
            scale = np.float16(amax / 127.0)
            # Quantize
            quants = np.round(block / float(scale)).astype(np.int8)
            quants = np.clip(quants, -128, 127).astype(np.int8)

        # Write block: scale (fp16) + quants (int8 * 32)
        output.extend(scale.tobytes())
        output.extend(quants.tobytes())

    return bytes(output)

def dequantize_q8_0(data: bytes, length: int) -> np.ndarray:
    """Dequantize Q8_0 data to FP32."""
    num_blocks = length // QK8_0
    output = np.zeros(length, dtype=np.float32)

    for b in range(num_blocks):
        offset = b * BLOCK_Q8_0_SIZE
        scale = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        quants = np.frombuffer(data[offset+2:offset+BLOCK_Q8_0_SIZE], dtype=np.int8)
        output[b * QK8_0 : (b + 1) * QK8_0] = quants.astype(np.float32) * float(scale)

    return output

# Test parameters
M = 1000  # Small vocab for testing (not 151936)
K = 896   # Embed dim

print(f"Testing gemv_q8_0_q8_0 with M={M}, K={K}")

# Create random test data
np.random.seed(42)
x_fp32 = np.random.randn(K).astype(np.float32) * 0.1  # Input vector
W_fp32 = np.random.randn(M, K).astype(np.float32) * 0.1  # Weight matrix

print(f"Input x: range=[{x_fp32.min():.4f}, {x_fp32.max():.4f}], mean={x_fp32.mean():.4f}")
print(f"Weight W: range=[{W_fp32.min():.4f}, {W_fp32.max():.4f}], mean={W_fp32.mean():.4f}")

# Quantize input
x_q8_bytes = quantize_to_q8_0(x_fp32)
x_q8 = np.frombuffer(x_q8_bytes, dtype=np.uint8)
print(f"Quantized x: {len(x_q8)} bytes")

# Quantize weight matrix (row by row)
W_q8_bytes = b''
for row in range(M):
    W_q8_bytes += quantize_to_q8_0(W_fp32[row])
W_q8 = np.frombuffer(W_q8_bytes, dtype=np.uint8)
print(f"Quantized W: {len(W_q8)} bytes")

# Verify dequantization
x_deq = dequantize_q8_0(x_q8_bytes, K)
print(f"Dequantized x error: max={np.abs(x_fp32 - x_deq).max():.6f}")

# Compute expected output (FP32 reference)
y_expected = W_fp32 @ x_fp32
print(f"Expected y: range=[{y_expected.min():.4f}, {y_expected.max():.4f}]")

# Allocate output buffer
y_out = np.zeros(M, dtype=np.float32)

# Call the kernel
lib.gemv_q8_0_q8_0.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # y (output)
    ctypes.c_void_p,                  # W (Q8_0 weight)
    ctypes.c_void_p,                  # x (Q8_0 input)
    ctypes.c_int,                     # M
    ctypes.c_int                      # K
]

print("\nCalling gemv_q8_0_q8_0...")
lib.gemv_q8_0_q8_0(
    y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    W_q8.ctypes.data_as(ctypes.c_void_p),
    x_q8.ctypes.data_as(ctypes.c_void_p),
    M,
    K
)

# Check output
nan_count = np.isnan(y_out).sum()
inf_count = np.isinf(y_out).sum()

print(f"\nOutput y:")
print(f"  NaN count: {nan_count}/{M}")
print(f"  Inf count: {inf_count}/{M}")

if nan_count == 0 and inf_count == 0:
    print(f"  Range: [{y_out.min():.4f}, {y_out.max():.4f}]")
    print(f"  Mean: {y_out.mean():.4f}")

    # Compare with expected
    diff = np.abs(y_out - y_expected)
    print(f"\nComparison with FP32 reference:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Correlation: {np.corrcoef(y_out, y_expected)[0,1]:.6f}")

    print("\n✓ KERNEL TEST PASSED")
else:
    print("\n✗ KERNEL TEST FAILED - Contains NaN/Inf")

    # Check first few values
    print(f"\nFirst 10 output values: {y_out[:10]}")
