#!/usr/bin/env python3
"""
quant_types.py - Quantization type registry for IR v4

Defines quantization formats matching llama.cpp/GGML for weight-only
quantization in LLM inference. Provides:
  - Block structure sizes matching llama.cpp
  - Kernel selection for quantized operations
  - Buffer size calculations for quantized tensors

Key formats:
  Q4_0:  Simple 4-bit (32 weights/block, 1 FP16 scale) = 4.5 bits/weight
  Q4_K:  K-quant 4-bit (256 weights/block, nested scales) = 4.5 bits/weight
  Q6_K:  K-quant 6-bit (256 weights/block, per-16 scales) = 6.5 bits/weight
  Q8_K:  K-quant 8-bit (256 weights/block, for activations) = 9.1 bits/weight

llama.cpp compatibility:
  - Block structures match ggml-common.h exactly
  - QK_K = 256 (super-block size for K-quants)
  - K_SCALE_SIZE = 12 (packed 6-bit scales for Q4_K)
  - Dequant-inside-kernel pattern for GEMV/GEMM
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants matching llama.cpp/ggml-common.h
# ---------------------------------------------------------------------------

QK_K = 256        # Super-block size for K-quants
K_SCALE_SIZE = 12  # Packed scale array size for Q4_K/Q5_K

QK4_0 = 32        # Block size for Q4_0
QK8_0 = 32        # Block size for Q8_0


@dataclass
class QuantType:
    """Quantization type specification."""
    name: str           # e.g., "q4_k", "q6_k"
    ggml_name: str      # e.g., "GGML_TYPE_Q4_K"
    block_size: int     # Elements per block (32 or 256)
    block_bytes: int    # Bytes per block
    bits_per_weight: float
    has_min: bool       # Whether format stores min (Q4_K) vs zero-centered
    description: str


# Quantization type registry matching llama.cpp exactly
QUANT_TYPES: Dict[str, QuantType] = {
    # Simple formats (QK = 32)
    "q4_0": QuantType(
        name="q4_0",
        ggml_name="GGML_TYPE_Q4_0",
        block_size=32,
        block_bytes=18,  # 2 (FP16 scale) + 16 (32 x 4-bit packed)
        bits_per_weight=4.5,
        has_min=False,
        description="4-bit, 32/block, 1 FP16 scale, zero-centered",
    ),
    "q4_1": QuantType(
        name="q4_1",
        ggml_name="GGML_TYPE_Q4_1",
        block_size=32,
        block_bytes=20,  # 2 (FP16 scale) + 2 (FP16 min) + 16 (32 x 4-bit packed)
        bits_per_weight=5.0,
        has_min=True,
        description="4-bit, 32/block, FP16 scale + min",
    ),
    "q5_0": QuantType(
        name="q5_0",
        ggml_name="GGML_TYPE_Q5_0",
        block_size=32,
        block_bytes=22,  # 2 (FP16 scale) + 4 (high bits) + 16 (low 4-bit)
        bits_per_weight=5.5,
        has_min=False,
        description="5-bit, 32/block, 1 FP16 scale, zero-centered",
    ),
    "q5_1": QuantType(
        name="q5_1",
        ggml_name="GGML_TYPE_Q5_1",
        block_size=32,
        block_bytes=24,  # 2 (FP16 scale) + 2 (FP16 min) + 4 (high bits) + 16 (low 4-bit)
        bits_per_weight=6.0,
        has_min=True,
        description="5-bit, 32/block, FP16 scale + min",
    ),
    "q8_0": QuantType(
        name="q8_0",
        ggml_name="GGML_TYPE_Q8_0",
        block_size=32,
        block_bytes=34,  # 2 (FP16 scale) + 32 (32 x 8-bit)
        bits_per_weight=8.5,
        has_min=False,
        description="8-bit, 32/block, 1 FP16 scale, for simple quantization",
    ),
    # K-quant formats (QK_K = 256)
    "q4_k": QuantType(
        name="q4_k",
        ggml_name="GGML_TYPE_Q4_K",
        block_size=256,
        block_bytes=144,  # 2+2 (d,dmin) + 12 (scales) + 128 (256 x 4-bit)
        bits_per_weight=4.5,
        has_min=True,
        description="4-bit K-quant, 256/block, nested 6-bit scales+mins (Q4_K_M)",
    ),
    "q6_k": QuantType(
        name="q6_k",
        ggml_name="GGML_TYPE_Q6_K",
        block_size=256,
        block_bytes=210,  # 2 (d) + 16 (int8 scales) + 128 (low 4-bit) + 64 (high 2-bit)
        bits_per_weight=6.5625,
        has_min=False,
        description="6-bit K-quant, 256/block, per-16 int8 scales",
    ),
    "q8_k": QuantType(
        name="q8_k",
        ggml_name="GGML_TYPE_Q8_K",
        block_size=256,
        block_bytes=292,  # 4 (FP32 d) + 256 (int8) + 32 (bsums)
        bits_per_weight=9.125,
        has_min=False,
        description="8-bit K-quant, 256/block, FP32 scale + bsums (for activations)",
    ),
}


# ---------------------------------------------------------------------------
# Quantized kernel registry
# ---------------------------------------------------------------------------

# Maps (op, weight_dtype, activation_dtype) -> kernel name
# For inference: weights are quantized, activations can be FP32 or Q8_K
QUANTIZED_KERNELS = {
    # GEMV/GEMM with Q4_K weights, FP32 activations (dequant-inside)
    ("linear", "q4_k", "f32"): "gemv_q4_k_f32",
    ("linear", "q4_k", "bf16"): "gemv_q4_k_bf16",

    # GEMV with Q4_K weights, Q8_K activations (fully quantized dot product)
    ("linear", "q4_k", "q8_k"): "gemv_q4_k_q8_k",

    # GEMV with Q6_K weights
    ("linear", "q6_k", "f32"): "gemv_q6_k_f32",
    ("linear", "q6_k", "bf16"): "gemv_q6_k_bf16",
    ("linear", "q6_k", "q8_k"): "gemv_q6_k_q8_k",

    # GEMV with Q4_0 weights (simpler format)
    ("linear", "q4_0", "f32"): "gemv_q4_0",
    ("linear", "q4_0", "bf16"): "gemv_q4_0",

    # GEMV with Q4_1 weights (4-bit with min)
    ("linear", "q4_1", "f32"): "gemv_q4_1",
    ("linear", "q4_1", "bf16"): "gemv_q4_1",

    # GEMV with Q5_0 weights (5-bit simple format)
    ("linear", "q5_0", "f32"): "gemv_q5_0",
    ("linear", "q5_0", "bf16"): "gemv_q5_0",

    # GEMV with Q5_1 weights (5-bit with min)
    ("linear", "q5_1", "f32"): "gemv_q5_1",
    ("linear", "q5_1", "bf16"): "gemv_q5_1",

    # GEMV with Q8_0 weights (8-bit simple format)
    ("linear", "q8_0", "f32"): "gemv_q8_0",
    ("linear", "q8_0", "bf16"): "gemv_q8_0",

    # Fused MLP kernels (gate + up + SwiGLU + down) - decode mode
    ("fused_mlp", "q4_k", "f32"): "mlp_fused_decode_q4_k_f32",
    ("fused_mlp", "q4_k", "q8_k"): "mlp_fused_decode_q4_k_q8_k",

    # Dequantization kernels (for prefill where we need full FP)
    ("dequant", "q4_k", "f32"): "dequant_q4_k_row",
    ("dequant", "q6_k", "f32"): "dequant_q6_k_row",
    ("dequant", "q4_0", "f32"): "dequant_q4_0_row",
    ("dequant", "q4_1", "f32"): "dequant_q4_1_row",
    ("dequant", "q5_0", "f32"): "dequant_q5_0_row",
    ("dequant", "q5_1", "f32"): "dequant_q5_1_row",
    ("dequant", "q8_0", "f32"): "dequant_q8_0_row",

    # Quantization kernels (for on-the-fly activation quantization)
    ("quantize", "f32", "q8_k"): "quantize_row_q8_k",
}

# Fallback kernel mappings for decode mode (prefer fused when available)
DECODE_FUSED_PATTERNS = {
    # Pattern: (sequence of ops) -> fused kernel
    ("mlp_up", "swiglu", "mlp_down"): "fused_mlp",
}


def get_quantized_kernel(
    op: str,
    weight_dtype: str,
    activation_dtype: str,
    mode: str = "decode"
) -> Optional[str]:
    """Get kernel name for quantized operation.

    Args:
        op: Operation name ("linear", "fused_mlp", etc.)
        weight_dtype: Weight data type ("q4_k", "q6_k", "f32", etc.)
        activation_dtype: Activation data type ("f32", "bf16", "q8_k")
        mode: Execution mode ("prefill", "decode")

    Returns:
        Kernel function name, or None if no quantized kernel exists.
    """
    key = (op, weight_dtype.lower(), activation_dtype.lower())
    return QUANTIZED_KERNELS.get(key)


def calculate_quantized_size(dtype: str, n_elements: int) -> int:
    """Calculate byte size for quantized tensor.

    Args:
        dtype: Quantization type ("q4_k", "q6_k", etc.)
        n_elements: Number of elements (weights)

    Returns:
        Size in bytes
    """
    dtype_lower = dtype.lower()
    if dtype_lower in QUANT_TYPES:
        qt = QUANT_TYPES[dtype_lower]
        n_blocks = (n_elements + qt.block_size - 1) // qt.block_size
        return n_blocks * qt.block_bytes

    # Non-quantized fallback
    elem_bytes = {"f32": 4, "bf16": 2, "f16": 2, "i32": 4, "i8": 1}
    return n_elements * elem_bytes.get(dtype_lower, 4)


def is_quantized_dtype(dtype: str) -> bool:
    """Check if dtype is a quantization format."""
    return dtype.lower() in QUANT_TYPES


def get_block_size(dtype: str) -> int:
    """Get the block size for a quantization format."""
    dtype_lower = dtype.lower()
    if dtype_lower in QUANT_TYPES:
        return QUANT_TYPES[dtype_lower].block_size
    return 1  # Non-quantized: 1 element per "block"


# ---------------------------------------------------------------------------
# GGUF type mapping (for loading llama.cpp models)
# ---------------------------------------------------------------------------

# Maps GGUF tensor type to our dtype string
GGUF_TYPE_MAP = {
    0: "f32",      # GGML_TYPE_F32
    1: "f16",      # GGML_TYPE_F16
    2: "q4_0",     # GGML_TYPE_Q4_0
    3: "q4_1",     # GGML_TYPE_Q4_1
    6: "q5_0",     # GGML_TYPE_Q5_0
    7: "q5_1",     # GGML_TYPE_Q5_1
    8: "q8_0",     # GGML_TYPE_Q8_0
    12: "q4_k",    # GGML_TYPE_Q4_K
    14: "q6_k",    # GGML_TYPE_Q6_K
    15: "q8_k",    # GGML_TYPE_Q8_K
    30: "bf16",    # GGML_TYPE_BF16
}


def gguf_type_to_dtype(gguf_type: int) -> str:
    """Convert GGUF tensor type to dtype string."""
    return GGUF_TYPE_MAP.get(gguf_type, "f32")


# ---------------------------------------------------------------------------
# Inference strategy selection
# ---------------------------------------------------------------------------

@dataclass
class InferenceStrategy:
    """Strategy for running quantized inference."""
    weight_dtype: str       # e.g., "q4_k"
    activation_dtype: str   # e.g., "f32" or "q8_k"
    use_activation_quant: bool
    fused_mlp: bool
    decode_kernel_style: str  # "dequant_gemm" or "quant_dot"


def select_inference_strategy(
    weight_dtype: str,
    mode: str,
    use_activation_quant: bool = True,
    prefer_fused: bool = True
) -> InferenceStrategy:
    """Select inference strategy based on weight format and preferences.

    Args:
        weight_dtype: Weight quantization format
        mode: "prefill" or "decode"
        use_activation_quant: Whether to quantize activations to Q8_K
        prefer_fused: Whether to prefer fused kernels

    Returns:
        InferenceStrategy describing how to run inference
    """
    weight_dtype = weight_dtype.lower()

    # Prefill: use FP32 activations (need full precision for long sequences)
    if mode == "prefill":
        return InferenceStrategy(
            weight_dtype=weight_dtype,
            activation_dtype="f32",
            use_activation_quant=False,
            fused_mlp=False,  # Prefill uses batched GEMM
            decode_kernel_style="dequant_gemm",
        )

    # Decode: can use Q8_K activations for memory efficiency
    if mode == "decode":
        if use_activation_quant and weight_dtype in ("q4_k", "q6_k"):
            # Fully quantized: Q4_K weights x Q8_K activations
            return InferenceStrategy(
                weight_dtype=weight_dtype,
                activation_dtype="q8_k",
                use_activation_quant=True,
                fused_mlp=prefer_fused,
                decode_kernel_style="quant_dot",
            )
        else:
            # Dequant-inside-kernel: Q4_K weights x FP32 activations
            return InferenceStrategy(
                weight_dtype=weight_dtype,
                activation_dtype="f32",
                use_activation_quant=False,
                fused_mlp=prefer_fused,
                decode_kernel_style="dequant_gemm",
            )

    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Weight memory estimation
# ---------------------------------------------------------------------------

def estimate_quantized_model_memory(
    num_layers: int,
    embed_dim: int,
    intermediate_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    weight_dtype: str = "q4_k",
    include_kv_cache: bool = True,
    max_seq_len: int = 4096,
    kv_dtype: str = "bf16",
) -> Dict[str, int]:
    """Estimate memory usage for a quantized model.

    Returns dict with:
      - weights: Total weight memory in bytes
      - kv_cache: KV cache memory in bytes (if include_kv_cache)
      - per_layer: Breakdown by component
    """
    E, I, H, KV, D, V = embed_dim, intermediate_dim, num_heads, num_kv_heads, head_dim, vocab_size

    # Per-layer weights
    wq_elems = H * D * E  # [H, D, E]
    wk_elems = KV * D * E
    wv_elems = KV * D * E
    wo_elems = H * E * D  # [H, E, D] = [H, D, E] transposed
    w1_elems = 2 * I * E  # Gate + Up concatenated
    w2_elems = E * I
    ln_elems = E * 2  # ln1 + ln2

    layer_elems = wq_elems + wk_elems + wv_elems + wo_elems + w1_elems + w2_elems + ln_elems
    layer_bytes = calculate_quantized_size(weight_dtype, layer_elems)

    # Global weights
    embed_bytes = calculate_quantized_size(weight_dtype, V * E)
    final_ln_bytes = calculate_quantized_size("f32", E)  # LN always FP32
    lm_head_bytes = calculate_quantized_size(weight_dtype, V * E)  # May be tied

    total_weights = (num_layers * layer_bytes) + embed_bytes + final_ln_bytes + lm_head_bytes

    result = {
        "weights": total_weights,
        "embed": embed_bytes,
        "layers": num_layers * layer_bytes,
        "per_layer": layer_bytes,
        "lm_head": lm_head_bytes,
    }

    if include_kv_cache:
        kv_elem_bytes = {"bf16": 2, "f16": 2, "f32": 4}.get(kv_dtype, 2)
        kv_per_layer = 2 * KV * max_seq_len * D * kv_elem_bytes  # K + V
        result["kv_cache"] = num_layers * kv_per_layer
        result["total"] = total_weights + result["kv_cache"]
    else:
        result["total"] = total_weights

    return result


if __name__ == "__main__":
    # Print quantization type info
    print("Quantization Types (llama.cpp compatible):")
    print("-" * 70)
    for name, qt in QUANT_TYPES.items():
        print(f"  {name:6s}: {qt.block_size:3d}/block, {qt.block_bytes:3d} bytes, "
              f"{qt.bits_per_weight:.2f} bpw - {qt.description[:40]}")

    # Example memory estimation for Qwen2-0.5B
    print("\nMemory Estimate (Qwen2-0.5B with Q4_K):")
    mem = estimate_quantized_model_memory(
        num_layers=24,
        embed_dim=896,
        intermediate_dim=4864,
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        vocab_size=151936,
        weight_dtype="q4_k",
        max_seq_len=4096,
    )
    print(f"  Weights: {mem['weights'] / 1e6:.1f} MB")
    print(f"  KV Cache: {mem['kv_cache'] / 1e6:.1f} MB")
    print(f"  Total: {mem['total'] / 1e6:.1f} MB")
