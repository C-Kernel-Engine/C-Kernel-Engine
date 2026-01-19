#!/usr/bin/env python3
"""
Kernel Registry - Central definition of all kernels and their buffer requirements.

This is the SINGLE SOURCE OF TRUTH for:
1. Memory planner - knows what buffers each kernel needs
2. Fusion pass - knows what kernels a fused kernel replaces
3. Codegen - knows what arguments to pass to each kernel
4. Validation - can verify all required buffers are allocated

Each kernel entry specifies:
- id: unique identifier (the C function name)
- inputs: list of input buffer specs
- outputs: list of output buffer specs
- scratch: list of scratch buffer specs (temporary, can be reused between layers)
- dims: dimension parameters the kernel expects
- fuses: (optional) list of kernel ids this fused kernel replaces

Buffer specs use symbolic dimensions that get resolved at planning time:
- T: number of tokens (seq_len for prefill, 1 for decode)
- E: embed_dim
- AE: aligned_embed_dim
- H: num_heads
- KV: num_kv_heads
- D: head_dim
- AD: aligned_head_dim
- I: intermediate_dim
- AI: aligned_intermediate_dim
- max_T: max_context_len (for KV cache allocation)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# =============================================================================
# QUANTIZATION FORMAT INFO (for buffer size calculation)
# =============================================================================

QUANT_BLOCK_INFO = {
    # Format: (block_size, bytes_per_block)
    "fp32": (1, 4),
    "fp16": (1, 2),
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q8_0": (32, 34),
    "q4_k": (256, 144),
    "q6_k": (256, 210),
    "q8_k": (256, 292),
}


# =============================================================================
# KERNEL REGISTRY
# =============================================================================

KERNEL_REGISTRY: Dict[str, Dict[str, Any]] = {

    # =========================================================================
    # NORMALIZATION KERNELS
    # =========================================================================

    "rmsnorm_forward": {
        "id": "rmsnorm_forward",
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "gamma", "dtype": "fp32", "shape": ["AE"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "scratch": [
            {"name": "rstd", "dtype": "fp32", "shape": ["T"], "optional": True},
        ],
        "dims": ["T", "E", "AE"],
        "description": "RMSNorm: output = gamma * (input / rms(input))",
    },

    # =========================================================================
    # EMBEDDING KERNELS
    # =========================================================================

    "embedding_forward": {
        "id": "embedding_forward",
        "inputs": [
            {"name": "tokens", "dtype": "int32", "shape": ["T"]},
            {"name": "embed_table", "dtype": "fp32", "shape": ["V", "AE"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "scratch": [],
        "dims": ["T", "V", "AE"],
        "description": "Token embedding lookup (FP32)",
    },

    "embedding_forward_q4_k": {
        "id": "embedding_forward_q4_k",
        "inputs": [
            {"name": "tokens", "dtype": "int32", "shape": ["T"]},
            {"name": "embed_table", "dtype": "q4_k", "shape": ["V", "AE"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "scratch": [],
        "dims": ["T", "V", "AE"],
        "description": "Token embedding lookup (Q4_K quantized)",
    },

    # =========================================================================
    # GEMM KERNELS (FP32 activation)
    # =========================================================================

    "gemm_nt_q4_k": {
        "id": "gemm_nt_q4_k",
        "inputs": [
            {"name": "A", "dtype": "fp32", "shape": ["M", "K"]},
            {"name": "B", "dtype": "q4_k", "shape": ["N", "K"]},  # Transposed
            {"name": "bias", "dtype": "fp32", "shape": ["N"], "optional": True},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["M", "N"]},
        ],
        "scratch": [],
        "dims": ["M", "N", "K"],
        "description": "GEMM: C = A @ B.T + bias (Q4_K weights, FP32 activations)",
    },

    "gemm_nt_q5_0": {
        "id": "gemm_nt_q5_0",
        "inputs": [
            {"name": "A", "dtype": "fp32", "shape": ["M", "K"]},
            {"name": "B", "dtype": "q5_0", "shape": ["N", "K"]},
            {"name": "bias", "dtype": "fp32", "shape": ["N"], "optional": True},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["M", "N"]},
        ],
        "scratch": [],
        "dims": ["M", "N", "K"],
        "description": "GEMM: C = A @ B.T + bias (Q5_0 weights, FP32 activations)",
    },

    "gemm_nt_q6_k": {
        "id": "gemm_nt_q6_k",
        "inputs": [
            {"name": "A", "dtype": "fp32", "shape": ["M", "K"]},
            {"name": "B", "dtype": "q6_k", "shape": ["N", "K"]},
            {"name": "bias", "dtype": "fp32", "shape": ["N"], "optional": True},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["M", "N"]},
        ],
        "scratch": [],
        "dims": ["M", "N", "K"],
        "description": "GEMM: C = A @ B.T + bias (Q6_K weights, FP32 activations)",
    },

    "gemm_nt_q8_0": {
        "id": "gemm_nt_q8_0",
        "inputs": [
            {"name": "A", "dtype": "fp32", "shape": ["M", "K"]},
            {"name": "B", "dtype": "q8_0", "shape": ["N", "K"]},
            {"name": "bias", "dtype": "fp32", "shape": ["N"], "optional": True},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["M", "N"]},
        ],
        "scratch": [],
        "dims": ["M", "N", "K"],
        "description": "GEMM: C = A @ B.T + bias (Q8_0 weights, FP32 activations)",
    },

    # =========================================================================
    # GEMM KERNELS (INT8 activation - quantize input first)
    # =========================================================================

    "gemm_nt_q4_k_q8_k": {
        "id": "gemm_nt_q4_k_q8_k",
        "inputs": [
            {"name": "A", "dtype": "q8_k", "shape": ["M", "K"]},  # Quantized activation
            {"name": "B", "dtype": "q4_k", "shape": ["N", "K"]},
            {"name": "bias", "dtype": "fp32", "shape": ["N"], "optional": True},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["M", "N"]},
        ],
        "scratch": [],
        "dims": ["M", "N", "K"],
        "requires_quantized_input": "q8_k",
        "description": "GEMM: C = A @ B.T + bias (Q4_K weights, Q8_K activations)",
    },

    "gemm_nt_q5_0_q8_0": {
        "id": "gemm_nt_q5_0_q8_0",
        "inputs": [
            {"name": "A", "dtype": "q8_0", "shape": ["M", "K"]},
            {"name": "B", "dtype": "q5_0", "shape": ["N", "K"]},
            {"name": "bias", "dtype": "fp32", "shape": ["N"], "optional": True},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["M", "N"]},
        ],
        "scratch": [],
        "dims": ["M", "N", "K"],
        "requires_quantized_input": "q8_0",
        "description": "GEMM: C = A @ B.T + bias (Q5_0 weights, Q8_0 activations)",
    },

    "gemm_nt_q6_k_q8_k": {
        "id": "gemm_nt_q6_k_q8_k",
        "inputs": [
            {"name": "A", "dtype": "q8_k", "shape": ["M", "K"]},
            {"name": "B", "dtype": "q6_k", "shape": ["N", "K"]},
            {"name": "bias", "dtype": "fp32", "shape": ["N"], "optional": True},
        ],
        "outputs": [
            {"name": "C", "dtype": "fp32", "shape": ["M", "N"]},
        ],
        "scratch": [],
        "dims": ["M", "N", "K"],
        "requires_quantized_input": "q8_k",
        "description": "GEMM: C = A @ B.T + bias (Q6_K weights, Q8_K activations)",
    },

    # =========================================================================
    # GEMV KERNELS (single token, INT8 activation)
    # =========================================================================

    "gemv_q4_k_q8_k": {
        "id": "gemv_q4_k_q8_k",
        "inputs": [
            {"name": "x", "dtype": "q8_k", "shape": ["K"]},
            {"name": "W", "dtype": "q4_k", "shape": ["N", "K"]},
        ],
        "outputs": [
            {"name": "y", "dtype": "fp32", "shape": ["N"]},
        ],
        "scratch": [],
        "dims": ["N", "K"],
        "requires_quantized_input": "q8_k",
        "description": "GEMV: y = W @ x (Q4_K weights, Q8_K activation, single token)",
    },

    "gemv_q5_0_q8_0": {
        "id": "gemv_q5_0_q8_0",
        "inputs": [
            {"name": "x", "dtype": "q8_0", "shape": ["K"]},
            {"name": "W", "dtype": "q5_0", "shape": ["N", "K"]},
        ],
        "outputs": [
            {"name": "y", "dtype": "fp32", "shape": ["N"]},
        ],
        "scratch": [],
        "dims": ["N", "K"],
        "requires_quantized_input": "q8_0",
        "description": "GEMV: y = W @ x (Q5_0 weights, Q8_0 activation, single token)",
    },

    "gemv_q6_k_q8_k": {
        "id": "gemv_q6_k_q8_k",
        "inputs": [
            {"name": "x", "dtype": "q8_k", "shape": ["K"]},
            {"name": "W", "dtype": "q6_k", "shape": ["N", "K"]},
        ],
        "outputs": [
            {"name": "y", "dtype": "fp32", "shape": ["N"]},
        ],
        "scratch": [],
        "dims": ["N", "K"],
        "requires_quantized_input": "q8_k",
        "description": "GEMV: y = W @ x (Q6_K weights, Q8_K activation, single token)",
    },

    # =========================================================================
    # QUANTIZATION KERNELS
    # =========================================================================

    "quantize_row_q8_0": {
        "id": "quantize_row_q8_0",
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["K"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "q8_0", "shape": ["K"]},
        ],
        "scratch": [],
        "dims": ["K"],
        "description": "Quantize FP32 row to Q8_0 format",
    },

    "quantize_row_q8_k": {
        "id": "quantize_row_q8_k",
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["K"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "q8_k", "shape": ["K"]},
        ],
        "scratch": [],
        "dims": ["K"],
        "description": "Quantize FP32 row to Q8_K format",
    },

    # =========================================================================
    # ATTENTION KERNELS
    # =========================================================================

    "attention_forward_causal_head_major_gqa_flash_strided": {
        "id": "attention_forward_causal_head_major_gqa_flash_strided",
        "inputs": [
            {"name": "Q", "dtype": "fp32", "shape": ["H", "T", "AD"]},
            {"name": "K_cache", "dtype": "fp32", "shape": ["KV", "max_T", "AD"]},
            {"name": "V_cache", "dtype": "fp32", "shape": ["KV", "max_T", "AD"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["H", "T", "AD"]},
        ],
        "scratch": [],
        "dims": ["H", "KV", "T", "D", "AD", "max_T"],
        "description": "Flash attention with GQA support, causal mask, head-major layout",
    },

    # =========================================================================
    # ACTIVATION KERNELS
    # =========================================================================

    "swiglu_forward": {
        "id": "swiglu_forward",
        "inputs": [
            {"name": "gate_up", "dtype": "fp32", "shape": ["T", "2*AI"]},  # Packed gate+up
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AI"]},
        ],
        "scratch": [],
        "dims": ["T", "AI"],
        "description": "SwiGLU: output = silu(gate) * up",
    },

    # =========================================================================
    # RESIDUAL KERNELS
    # =========================================================================

    "ck_residual_add_token_major": {
        "id": "ck_residual_add_token_major",
        "inputs": [
            {"name": "a", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "b", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "scratch": [],
        "dims": ["T", "AE"],
        "description": "Residual add: output = a + b",
    },

    # =========================================================================
    # ROPE KERNELS
    # =========================================================================

    "rope_forward": {
        "id": "rope_forward",
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["H", "T", "AD"]},
            {"name": "cos", "dtype": "fp32", "shape": ["max_T", "AD"]},
            {"name": "sin", "dtype": "fp32", "shape": ["max_T", "AD"]},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["H", "T", "AD"]},
        ],
        "scratch": [],
        "dims": ["H", "T", "AD", "max_T"],
        "description": "RoPE positional encoding",
    },

    # =========================================================================
    # FUSED KERNELS
    # =========================================================================

    "mega_fused_attention_prefill": {
        "id": "mega_fused_attention_prefill",
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "residual", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "ln1_gamma", "dtype": "fp32", "shape": ["AE"]},
            {"name": "wq", "dtype": "q5_0|q8_0", "shape": ["H*AD", "AE"]},
            {"name": "bq", "dtype": "fp32", "shape": ["H*AD"], "optional": True},
            {"name": "wk", "dtype": "q5_0|q8_0", "shape": ["KV*AD", "AE"]},
            {"name": "bk", "dtype": "fp32", "shape": ["KV*AD"], "optional": True},
            {"name": "wv", "dtype": "q5_0|q8_0", "shape": ["KV*AD", "AE"]},
            {"name": "bv", "dtype": "fp32", "shape": ["KV*AD"], "optional": True},
            {"name": "wo", "dtype": "q5_0|q8_0", "shape": ["AE", "H*AD"]},
            {"name": "bo", "dtype": "fp32", "shape": ["AE"], "optional": True},
            {"name": "rope_cos", "dtype": "fp32", "shape": ["max_T", "AD"], "optional": True},
            {"name": "rope_sin", "dtype": "fp32", "shape": ["max_T", "AD"], "optional": True},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "scratch": [
            {"name": "q", "dtype": "fp32", "shape": ["H", "T", "AD"]},
            {"name": "kv_cache_k", "dtype": "fp32", "shape": ["KV", "max_T", "AD"]},
            {"name": "kv_cache_v", "dtype": "fp32", "shape": ["KV", "max_T", "AD"]},
            {"name": "attn_out", "dtype": "fp32", "shape": ["H", "T", "AD"]},
            {"name": "q8_scratch", "dtype": "q8_0", "shape": ["T", "AE"]},
        ],
        "dims": ["T", "E", "AE", "H", "KV", "D", "AD", "max_T"],
        "fuses": [
            "rmsnorm_forward",
            "gemm_nt_q5_0",  # or gemm_nt_q8_0 for Q/K/V
            "attention_forward_causal_head_major_gqa_flash_strided",
            "ck_residual_add_token_major",
        ],
        "speedup_estimate": 1.45,
        "description": "Fused: RMSNorm + QKV projection + Flash Attention + Out-proj + Residual",
    },

    "mega_fused_attention_prefill_q8_0": {
        "id": "mega_fused_attention_prefill_q8_0",
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "residual", "dtype": "fp32", "shape": ["T", "AE"], "optional": True},
            {"name": "ln1_gamma", "dtype": "fp32", "shape": ["AE"]},
            {"name": "wq", "dtype": "q8_0", "shape": ["H*AD", "AE"]},
            {"name": "bq", "dtype": "fp32", "shape": ["H*AD"], "optional": True},
            {"name": "wk", "dtype": "q8_0", "shape": ["KV*AD", "AE"]},
            {"name": "bk", "dtype": "fp32", "shape": ["KV*AD"], "optional": True},
            {"name": "wv", "dtype": "q8_0", "shape": ["KV*AD", "AE"]},
            {"name": "bv", "dtype": "fp32", "shape": ["KV*AD"], "optional": True},
            {"name": "wo", "dtype": "q8_0", "shape": ["AE", "H*AD"]},
            {"name": "bo", "dtype": "fp32", "shape": ["AE"], "optional": True},
            {"name": "rope_cos", "dtype": "fp32", "shape": ["max_T", "AD"], "optional": True},
            {"name": "rope_sin", "dtype": "fp32", "shape": ["max_T", "AD"], "optional": True},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "scratch": [
            {"name": "q", "dtype": "fp32", "shape": ["H", "T", "AD"]},
            {"name": "kv_cache_k", "dtype": "fp32", "shape": ["KV", "max_T", "AD"]},
            {"name": "kv_cache_v", "dtype": "fp32", "shape": ["KV", "max_T", "AD"]},
            {"name": "attn_out", "dtype": "fp32", "shape": ["H", "T", "AD"]},
            {"name": "q8_scratch", "dtype": "q8_0", "shape": ["T", "AE"]},
        ],
        "dims": ["T", "E", "AE", "H", "KV", "D", "AD", "max_T"],
        "fuses": [
            "rmsnorm_forward",
            "gemm_nt_q8_0",
            "attention_forward_causal_head_major_gqa_flash_strided",
            "ck_residual_add_token_major",
        ],
        "speedup_estimate": 1.45,
        "description": "Fused attention with Q8_0 weights (higher precision)",
    },

    "mega_fused_outproj_mlp_prefill": {
        "id": "mega_fused_outproj_mlp_prefill",
        "inputs": [
            {"name": "attn_out", "dtype": "fp32", "shape": ["H", "T", "AD"]},  # Head-major
            {"name": "residual", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "ln2_gamma", "dtype": "fp32", "shape": ["AE"]},
            {"name": "wo", "dtype": "q5_0", "shape": ["AE", "H*AD"]},
            {"name": "bo", "dtype": "fp32", "shape": ["AE"], "optional": True},
            {"name": "w1", "dtype": "q5_0", "shape": ["2*AI", "AE"]},  # Gate+Up packed
            {"name": "b1", "dtype": "fp32", "shape": ["2*AI"], "optional": True},
            {"name": "w2", "dtype": "q4_k|q6_k", "shape": ["AE", "AI"]},
            {"name": "b2", "dtype": "fp32", "shape": ["AE"], "optional": True},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "scratch": [
            {"name": "proj_out", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "ln2_out", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "fc1_out", "dtype": "fp32", "shape": ["T", "2*AI"]},
            {"name": "swiglu_out", "dtype": "fp32", "shape": ["T", "AI"]},
            {"name": "q8_scratch", "dtype": "q8_0", "shape": ["T", "AE"]},
        ],
        "dims": ["T", "E", "AE", "H", "D", "AD", "I", "AI"],
        "fuses": [
            "gemm_nt_q5_0",      # Out-projection
            "ck_residual_add_token_major",
            "rmsnorm_forward",
            "gemm_nt_q5_0",      # W1 (gate+up)
            "swiglu_forward",
            "gemm_nt_q4_k",      # W2 (down) - or q6_k
            "ck_residual_add_token_major",
        ],
        "speedup_estimate": 1.10,
        "description": "Fused: Out-proj + Residual + RMSNorm + MLP(SwiGLU) + Residual",
    },

    "fused_rmsnorm_qkv_prefill_head_major_quant": {
        "id": "fused_rmsnorm_qkv_prefill_head_major_quant",
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "gamma", "dtype": "fp32", "shape": ["AE"]},
            {"name": "wq", "dtype": "q5_0|q8_0", "shape": ["H*AD", "AE"]},
            {"name": "bq", "dtype": "fp32", "shape": ["H*AD"], "optional": True},
            {"name": "wk", "dtype": "q5_0|q8_0", "shape": ["KV*AD", "AE"]},
            {"name": "bk", "dtype": "fp32", "shape": ["KV*AD"], "optional": True},
            {"name": "wv", "dtype": "q5_0|q8_0", "shape": ["KV*AD", "AE"]},
            {"name": "bv", "dtype": "fp32", "shape": ["KV*AD"], "optional": True},
        ],
        "outputs": [
            {"name": "Q", "dtype": "fp32", "shape": ["H", "T", "AD"]},
            {"name": "K", "dtype": "fp32", "shape": ["KV", "T", "AD"]},
            {"name": "V", "dtype": "fp32", "shape": ["KV", "T", "AD"]},
        ],
        "scratch": [
            {"name": "q8_scratch", "dtype": "q8_0", "shape": ["T", "AE"]},
        ],
        "dims": ["T", "E", "AE", "H", "KV", "D", "AD"],
        "fuses": [
            "rmsnorm_forward",
            "gemm_nt_q5_0",  # Q projection
            "gemm_nt_q5_0",  # K projection
            "gemm_nt_q5_0",  # V projection
        ],
        "description": "Fused: RMSNorm + QKV projection (head-major output)",
    },

    "fused_mlp_swiglu_prefill_w1w2_quant": {
        "id": "fused_mlp_swiglu_prefill_w1w2_quant",
        "inputs": [
            {"name": "input", "dtype": "fp32", "shape": ["T", "AE"]},
            {"name": "w1", "dtype": "q5_0", "shape": ["2*AI", "AE"]},
            {"name": "b1", "dtype": "fp32", "shape": ["2*AI"], "optional": True},
            {"name": "w2", "dtype": "q4_k|q6_k", "shape": ["AE", "AI"]},
            {"name": "b2", "dtype": "fp32", "shape": ["AE"], "optional": True},
        ],
        "outputs": [
            {"name": "output", "dtype": "fp32", "shape": ["T", "AE"]},
        ],
        "scratch": [
            {"name": "fc1_out", "dtype": "fp32", "shape": ["T", "2*AI"]},
            {"name": "swiglu_out", "dtype": "fp32", "shape": ["T", "AI"]},
            {"name": "q8_scratch", "dtype": "q8_0", "shape": ["T", "AE"]},
        ],
        "dims": ["T", "E", "AE", "I", "AI"],
        "fuses": [
            "gemm_nt_q5_0",
            "swiglu_forward",
            "gemm_nt_q4_k",
        ],
        "description": "Fused MLP: W1(gate+up) + SwiGLU + W2(down)",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_kernel(kernel_id: str) -> Optional[Dict[str, Any]]:
    """Get kernel spec by ID."""
    return KERNEL_REGISTRY.get(kernel_id)


def kernel_exists(kernel_id: str) -> bool:
    """Check if kernel exists in registry."""
    return kernel_id in KERNEL_REGISTRY


def get_fused_kernels() -> List[str]:
    """Get list of all fused kernel IDs."""
    return [k for k, v in KERNEL_REGISTRY.items() if "fuses" in v]


def get_kernels_that_fuse(kernel_id: str) -> List[str]:
    """Find fused kernels that can replace a given kernel."""
    result = []
    for k, v in KERNEL_REGISTRY.items():
        if "fuses" in v and kernel_id in v["fuses"]:
            result.append(k)
    return result


def compute_buffer_size(shape: List[str], dtype: str, dims: Dict[str, int]) -> int:
    """
    Compute buffer size in bytes given symbolic shape and concrete dimensions.

    Args:
        shape: e.g., ["T", "AE"] or ["H", "T", "AD"]
        dtype: e.g., "fp32", "q4_k", "q8_0"
        dims: e.g., {"T": 512, "AE": 2048, "H": 32, "AD": 64}

    Returns:
        Size in bytes
    """
    # Resolve shape to concrete dimensions
    elements = 1
    for s in shape:
        if isinstance(s, int):
            elements *= s
        elif isinstance(s, str):
            # Handle expressions like "H*AD", "2*AI"
            expr = s
            for dim_name, dim_val in sorted(dims.items(), key=lambda x: -len(x[0])):
                expr = expr.replace(dim_name, str(dim_val))
            elements *= eval(expr)

    # Compute size based on dtype
    dtype_base = dtype.split("|")[0]  # Handle "q5_0|q8_0" -> "q5_0"
    if dtype_base in QUANT_BLOCK_INFO:
        block_size, block_bytes = QUANT_BLOCK_INFO[dtype_base]
        n_blocks = (elements + block_size - 1) // block_size
        return n_blocks * block_bytes
    else:
        # Default to FP32
        return elements * 4


def get_kernel_memory_requirements(kernel_id: str, dims: Dict[str, int]) -> Dict[str, Any]:
    """
    Get total memory requirements for a kernel.

    Args:
        kernel_id: Kernel ID
        dims: Concrete dimension values

    Returns:
        Dict with input/output/scratch sizes and total
    """
    spec = KERNEL_REGISTRY.get(kernel_id)
    if not spec:
        raise ValueError(f"Unknown kernel: {kernel_id}")

    result = {
        "inputs": [],
        "outputs": [],
        "scratch": [],
        "total_output_bytes": 0,
        "total_scratch_bytes": 0,
    }

    for category in ["inputs", "outputs", "scratch"]:
        for buf in spec.get(category, []):
            if buf.get("optional"):
                continue
            size = compute_buffer_size(buf["shape"], buf["dtype"], dims)
            result[category].append({
                "name": buf["name"],
                "dtype": buf["dtype"],
                "shape": buf["shape"],
                "size": size,
            })
            if category == "outputs":
                result["total_output_bytes"] += size
            elif category == "scratch":
                result["total_scratch_bytes"] += size

    return result


def list_all_kernels() -> List[str]:
    """List all kernel IDs."""
    return list(KERNEL_REGISTRY.keys())


def print_kernel_summary():
    """Print summary of all kernels."""
    print("=" * 70)
    print("KERNEL REGISTRY SUMMARY")
    print("=" * 70)

    basic = [k for k in KERNEL_REGISTRY if "fuses" not in KERNEL_REGISTRY[k]]
    fused = [k for k in KERNEL_REGISTRY if "fuses" in KERNEL_REGISTRY[k]]

    print(f"\nBasic kernels ({len(basic)}):")
    for k in sorted(basic):
        desc = KERNEL_REGISTRY[k].get("description", "")[:50]
        print(f"  {k}: {desc}")

    print(f"\nFused kernels ({len(fused)}):")
    for k in sorted(fused):
        spec = KERNEL_REGISTRY[k]
        fuses_count = len(spec.get("fuses", []))
        speedup = spec.get("speedup_estimate", "?")
        print(f"  {k}: fuses {fuses_count} ops, ~{speedup}x speedup")


if __name__ == "__main__":
    print_kernel_summary()

    # Example: compute memory for mega_fused_attention_prefill
    print("\n" + "=" * 70)
    print("EXAMPLE: mega_fused_attention_prefill memory requirements")
    print("=" * 70)

    dims = {
        "T": 512,       # Prefill tokens
        "E": 2048,      # embed_dim
        "AE": 2048,     # aligned_embed_dim
        "H": 32,        # num_heads
        "KV": 8,        # num_kv_heads (GQA)
        "D": 64,        # head_dim
        "AD": 64,       # aligned_head_dim
        "I": 8192,      # intermediate_dim
        "AI": 8192,     # aligned_intermediate_dim
        "max_T": 2048,  # max context length
    }

    reqs = get_kernel_memory_requirements("mega_fused_attention_prefill", dims)

    print(f"\nDimensions: {dims}")
    print(f"\nOutputs ({reqs['total_output_bytes'] / 1024 / 1024:.1f} MB):")
    for buf in reqs["outputs"]:
        print(f"  {buf['name']}: {buf['size'] / 1024 / 1024:.2f} MB")

    print(f"\nScratch ({reqs['total_scratch_bytes'] / 1024 / 1024:.1f} MB):")
    for buf in reqs["scratch"]:
        print(f"  {buf['name']}: {buf['size'] / 1024 / 1024:.2f} MB")
