"""
Kernel specification registry for extensible validation.

Each kernel type (Q4_K, Q6_K, Q5_0, etc.) has a spec that defines:
  - Block structure (size, elements per block)
  - Dequantization function references
  - GEMV/GEMM function references
  - Tolerance thresholds
"""

from dataclasses import dataclass
from typing import Dict, Callable, Optional


@dataclass
class KernelSpec:
    """Specification for a quantization kernel type"""
    name: str                    # e.g., "q4_k"
    block_size_bytes: int        # e.g., 144 for Q4_K
    elements_per_block: int      # e.g., 256 for K-quants, 32 for simple quants
    ggml_type: int               # GGML_TYPE_* enum value
    ck_dtype: int                # CK_DT_* enum value

    # Tolerance for numerical comparison
    dequant_tolerance: float = 1e-5
    gemv_tolerance: float = 1e-4
    layer_tolerance: float = 1e-3

    # C function names (for ctypes loading)
    dequant_func: Optional[str] = None   # e.g., "ck_test_dequant_q4_k"
    gemv_func: Optional[str] = None      # e.g., "ck_test_gemv_q4_k"
    gemm_func: Optional[str] = None      # e.g., "ck_test_gemm_q4_k"

    def validate_alignment(self, dim: int) -> bool:
        """Check if dimension is properly aligned for this quant type"""
        return dim % self.elements_per_block == 0


# GGML type constants (from ggml.h)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15

# CK dtype constants (from ckernel_dtype.h)
CK_DT_FP32 = 0
CK_DT_BF16 = 1
CK_DT_FP16 = 2
CK_DT_Q4_0 = 3
CK_DT_Q4_1 = 4
CK_DT_Q5_0 = 11
CK_DT_Q5_1 = 12
CK_DT_Q4_K = 7
CK_DT_Q6_K = 8
CK_DT_Q8_0 = 9
CK_DT_Q8_K = 10


# Global kernel registry
KERNEL_REGISTRY: Dict[str, KernelSpec] = {}


def register_kernel(spec: KernelSpec):
    """Register a kernel specification"""
    KERNEL_REGISTRY[spec.name.lower()] = spec


def get_kernel_spec(name: str) -> Optional[KernelSpec]:
    """Get kernel spec by name"""
    return KERNEL_REGISTRY.get(name.lower())


def get_spec_by_ggml_type(ggml_type: int) -> Optional[KernelSpec]:
    """Get kernel spec by GGML type"""
    for spec in KERNEL_REGISTRY.values():
        if spec.ggml_type == ggml_type:
            return spec
    return None


# Register built-in kernel types
register_kernel(KernelSpec(
    name="q4_k",
    block_size_bytes=144,
    elements_per_block=256,
    ggml_type=GGML_TYPE_Q4_K,
    ck_dtype=CK_DT_Q4_K,
    dequant_func="ck_test_dequant_q4_k",
    gemv_func="ck_test_gemv_q4_k",
    gemm_func="ck_test_gemm_q4_k",
))

register_kernel(KernelSpec(
    name="q6_k",
    block_size_bytes=210,
    elements_per_block=256,
    ggml_type=GGML_TYPE_Q6_K,
    ck_dtype=CK_DT_Q6_K,
    dequant_func="ck_test_dequant_q6_k",
    gemv_func="ck_test_gemv_q6_k",
))

register_kernel(KernelSpec(
    name="q5_0",
    block_size_bytes=22,
    elements_per_block=32,
    ggml_type=GGML_TYPE_Q5_0,
    ck_dtype=CK_DT_Q5_0,
    dequant_func="ck_test_dequant_q5_0",
    gemv_func="ck_test_gemv_q5_0",
))

register_kernel(KernelSpec(
    name="q4_0",
    block_size_bytes=18,
    elements_per_block=32,
    ggml_type=GGML_TYPE_Q4_0,
    ck_dtype=CK_DT_Q4_0,
    dequant_func="ck_test_dequant_q4_0",
))

register_kernel(KernelSpec(
    name="q8_0",
    block_size_bytes=34,
    elements_per_block=32,
    ggml_type=GGML_TYPE_Q8_0,
    ck_dtype=CK_DT_Q8_0,
    gemv_func="ck_test_gemv_q8_0",
))

register_kernel(KernelSpec(
    name="q8_k",
    block_size_bytes=292,
    elements_per_block=256,
    ggml_type=GGML_TYPE_Q8_K,
    ck_dtype=CK_DT_Q8_K,
))

register_kernel(KernelSpec(
    name="f32",
    block_size_bytes=4,
    elements_per_block=1,
    ggml_type=GGML_TYPE_F32,
    ck_dtype=CK_DT_FP32,
    dequant_tolerance=0,  # Exact match expected
))

register_kernel(KernelSpec(
    name="f16",
    block_size_bytes=2,
    elements_per_block=1,
    ggml_type=GGML_TYPE_F16,
    ck_dtype=CK_DT_FP16,
    dequant_tolerance=1e-3,  # FP16 has less precision
))


def list_kernels() -> list:
    """List all registered kernel types"""
    return list(KERNEL_REGISTRY.keys())


def get_block_size(quant_type: str) -> int:
    """Get block size in bytes for a quantization type"""
    spec = get_kernel_spec(quant_type)
    if spec:
        return spec.block_size_bytes
    raise ValueError(f"Unknown quantization type: {quant_type}")


def get_elements_per_block(quant_type: str) -> int:
    """Get number of elements per block for a quantization type"""
    spec = get_kernel_spec(quant_type)
    if spec:
        return spec.elements_per_block
    raise ValueError(f"Unknown quantization type: {quant_type}")
