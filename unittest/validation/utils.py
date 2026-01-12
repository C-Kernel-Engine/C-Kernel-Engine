"""
Utility functions for kernel validation.
"""

import os
import struct
import numpy as np
from typing import Optional, Tuple


def fmt_size(size: int) -> str:
    """Format size in human-readable format"""
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024**3):.2f} GB"
    elif size >= 1024 * 1024:
        return f"{size / (1024**2):.2f} MB"
    elif size >= 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} bytes"


def fmt_diff(diff: float) -> str:
    """Format numerical difference for display"""
    if diff == 0:
        return "0"
    elif diff < 1e-10:
        return f"{diff:.2e}"
    elif diff < 1e-6:
        return f"{diff:.3e}"
    elif diff < 1e-3:
        return f"{diff:.6f}"
    else:
        return f"{diff:.4f}"


def compare_tensors(
    a: np.ndarray,
    b: np.ndarray,
    name: str = "tensor"
) -> Tuple[float, float, bool, str]:
    """
    Compare two tensors and return (max_diff, mean_diff, passed, message).

    Returns:
        max_diff: Maximum absolute difference
        mean_diff: Mean absolute difference
        passed: True if tensors match within tolerance
        message: Description of result
    """
    # Check for shape mismatch
    if a.shape != b.shape:
        if a.size == b.size:
            b = b.reshape(a.shape)
        else:
            return float('inf'), float('inf'), False, f"Shape mismatch: {a.shape} vs {b.shape}"

    # Check for NaN/Inf
    if np.any(np.isnan(b)):
        nan_count = np.sum(np.isnan(b))
        return float('inf'), float('inf'), False, f"Output has {nan_count} NaN values"

    if np.any(np.isinf(b)):
        inf_count = np.sum(np.isinf(b))
        return float('inf'), float('inf'), False, f"Output has {inf_count} Inf values"

    # Compute differences
    diff = np.abs(a - b)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    # Find first divergence point
    if max_diff > 0:
        first_diff_idx = np.argmax(diff.flatten() > 1e-6)
        message = f"max_diff={fmt_diff(max_diff)} at index {first_diff_idx}"
    else:
        message = "exact match"

    return max_diff, mean_diff, True, message


def load_binary_tensor(path: str, dtype: np.dtype = np.float32) -> Optional[np.ndarray]:
    """Load a tensor from binary file"""
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=dtype)


def save_binary_tensor(tensor: np.ndarray, path: str):
    """Save a tensor to binary file"""
    tensor.astype(np.float32).tofile(path)


def fp16_to_fp32(h: int) -> float:
    """Convert FP16 (uint16) to FP32"""
    return np.frombuffer(struct.pack('<H', h), dtype=np.float16)[0].astype(np.float32)


def align_up(n: int, a: int) -> int:
    """Align n up to multiple of a"""
    return ((n + a - 1) // a) * a


def get_quantized_size(num_elements: int, quant_type: str) -> int:
    """Calculate size in bytes for quantized tensor"""
    BLOCK_CONFIGS = {
        'q4_k': (256, 144),
        'q6_k': (256, 210),
        'q5_0': (32, 22),
        'q4_0': (32, 18),
        'q8_0': (32, 34),
        'q8_k': (256, 292),
        'f32': (1, 4),
        'f16': (1, 2),
        'bf16': (1, 2),
    }

    config = BLOCK_CONFIGS.get(quant_type.lower())
    if not config:
        raise ValueError(f"Unknown quant type: {quant_type}")

    elems_per_block, bytes_per_block = config
    num_blocks = (num_elements + elems_per_block - 1) // elems_per_block
    return num_blocks * bytes_per_block


def print_comparison_table(results: list):
    """Print a formatted comparison table"""
    print(f"\n{'Operation':<30} {'Status':<8} {'Max Diff':<12} {'Notes'}")
    print("-" * 70)
    for r in results:
        name = r.get('name', 'Unknown')[:30]
        status = r.get('status', 'UNKNOWN')
        max_diff = r.get('max_diff', None)

        if status == 'PASS':
            status_str = '\033[92mPASS\033[0m'
        elif status == 'FAIL':
            status_str = '\033[91mFAIL\033[0m'
        else:
            status_str = '\033[93mSKIP\033[0m'

        diff_str = fmt_diff(max_diff) if max_diff is not None else "-"
        notes = r.get('message', '')[:20]

        print(f"{name:<30} {status_str:<8} {diff_str:<12} {notes}")
