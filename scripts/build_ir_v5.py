#!/usr/bin/env python3
"""
build_ir_v5.py - IR v5 pipeline (standalone, manifest-first)

config.json + weights_manifest.json -> graph IR -> lowered IR -> layout -> generated C

v5 differences from v4:
  - Requires weights manifest (manifest-first approach)
  - Uses explicit unrolled codegen (codegen_v5) by default
  - Proper mixed-quant support via per-tensor dtypes from manifest
  - Standalone implementation (not a wrapper)
"""

import copy
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import build_ir_v3 as v3
import codegen_v4
import codegen_v5
import fusion_patterns as fp
import parallel_planner as pp
import quant_types as qt
import training_config as tc

# ---------------------------------------------------------------------------
# Presets (local configs for quick tests)
# ---------------------------------------------------------------------------

PRESETS = {
    "qwen2-0.5b": {
        "config": "qwen2_0.5.json",
        "name": "qwen2_0.5b",
        "hf": "Qwen/Qwen2-0.5B",
    },
    "smollm-135": {
        "config": "smolLM-135.json",
        "name": "smollm_135",
        "hf": "HuggingFaceTB/SmolLM-135M",
    },
}

HOST_OPS = {"embedding", "rope_precompute"}

WEIGHT_MAP_V4 = [
    {"hf": "model.embed_tokens.weight", "ck": "token_emb"},
    {"hf": "model.layers.{layer}.input_layernorm.weight", "ck": "layer.{L}.ln1_gamma"},
    {"hf": "model.layers.{layer}.self_attn.q_proj.weight", "ck": "layer.{L}.wq"},
    {"hf": "model.layers.{layer}.self_attn.k_proj.weight", "ck": "layer.{L}.wk"},
    {"hf": "model.layers.{layer}.self_attn.v_proj.weight", "ck": "layer.{L}.wv"},
    {"hf": "model.layers.{layer}.self_attn.o_proj.weight", "ck": "layer.{L}.wo"},
    {"hf": "model.layers.{layer}.post_attention_layernorm.weight", "ck": "layer.{L}.ln2_gamma"},
    {"hf": "model.layers.{layer}.mlp.gate_proj.weight", "ck": "layer.{L}.w1", "pack": "concat", "axis": 0, "part": "gate"},
    {"hf": "model.layers.{layer}.mlp.up_proj.weight", "ck": "layer.{L}.w1", "pack": "concat", "axis": 0, "part": "up"},
    {"hf": "model.layers.{layer}.mlp.down_proj.weight", "ck": "layer.{L}.w2"},
    {"hf": "model.norm.weight", "ck": "final_ln_weight"},
    {"hf": "lm_head.weight", "ck": "lm_head_weight", "optional": True},
]

# ---------------------------------------------------------------------------
# Kernel selection (baseline)
# ---------------------------------------------------------------------------

KERNELS = {
    # Forward kernels
    "embedding": {"bf16": "embedding_bf16", "f32": "embedding_f32"},
    "rmsnorm": {"bf16": "rmsnorm_bf16", "f32": "rmsnorm_f32"},
    "linear": {"bf16": "gemm_bf16", "f32": "gemm_f32"},
    "rope": {"bf16": "rope_bf16", "f32": "rope_f32"},
    "attention_prefill": {"bf16": "attention_prefill_bf16", "f32": "attention_prefill_f32"},
    "attention_decode": {"bf16": "attention_decode_bf16", "f32": "attention_decode_f32"},
    "swiglu": {"bf16": "swiglu_bf16", "f32": "swiglu_f32"},
    "add": {"bf16": "add_bf16", "f32": "add_f32"},
    "kv_cache_update": {"bf16": "kv_cache_update", "f32": "kv_cache_update"},
    # Fused kernels (forward)
    "fused_mlp": {"bf16": "mlp_fused_decode_bf16", "f32": "mlp_fused_decode_f32"},
    "gemm_swiglu": {"bf16": "gemm_swiglu_fused_bf16", "f32": "gemm_swiglu_fused"},
    "residual_rmsnorm": {"bf16": "residual_rmsnorm_fused_bf16", "f32": "residual_rmsnorm_fused_f32"},
    "gemm_gelu": {"bf16": "gemm_bias_gelu_fused_bf16", "f32": "gemm_bias_gelu_fused"},
    "gemm_relu": {"bf16": "gemm_bias_relu_fused_bf16", "f32": "gemm_bias_relu_fused"},
    "attention_proj_fused": {"bf16": "attention_proj_fused_decode_bf16", "f32": "attention_proj_fused_decode_f32"},
    # Backward kernels
    "embedding_backward": {"bf16": "embedding_backward_bf16", "f32": "embedding_backward"},
    "rmsnorm_backward": {"bf16": "rmsnorm_backward_bf16", "f32": "rmsnorm_backward"},
    "gemm_backward": {"bf16": "gemm_backward_bf16", "f32": "gemm_backward"},
    "rope_backward": {"bf16": "rope_backward_bf16", "f32": "rope_backward"},
    "attention_backward": {"bf16": "attention_backward_causal_head_major_gqa_bf16", "f32": "attention_backward_causal_head_major_gqa"},
    "softmax_backward": {"bf16": "backward_causal_softmax_head_major_bf16", "f32": "backward_causal_softmax_head_major"},
    "swiglu_backward": {"bf16": "swiglu_backward_bf16", "f32": "swiglu_backward"},
    "gelu_backward": {"bf16": "gelu_backward_fast_bf16", "f32": "gelu_backward_fast"},
    "relu_backward": {"bf16": "relu_backward_bf16", "f32": "relu_backward"},
    "sigmoid_backward": {"bf16": "sigmoid_backward_bf16", "f32": "sigmoid_backward"},
    "add_backward": {"bf16": "add_backward_bf16", "f32": "add_backward"},
    "layernorm_backward": {"bf16": "layernorm_backward_kernel_bf16", "f32": "layernorm_backward_kernel"},
    "fc1_backward": {"bf16": "fc1_backward_kernel_bf16", "f32": "fc1_backward_kernel"},
    "fc2_backward": {"bf16": "fc2_backward_kernel_bf16", "f32": "fc2_backward_kernel"},
    # Loss kernels
    "cross_entropy_loss": {"bf16": "cross_entropy_loss_bf16", "f32": "cross_entropy_loss"},
}

# Quantized kernels: (weight_dtype, activation_dtype) -> kernel name
# Used for inference with quantized weights (llama.cpp compatible)
QUANT_KERNELS = {
    # GEMV with quantized weights (decode mode - single token)
    ("linear", "q4_k", "f32"): "gemv_q4_k",
    ("linear", "q4_k", "q8_k"): "gemv_q4_k_q8_k",
    ("linear", "q6_k", "f32"): "gemv_q6_k",
    ("linear", "q4_0", "f32"): "gemv_q4_0",
    ("linear", "q4_1", "f32"): "gemv_q4_1",
    ("linear", "q5_0", "f32"): "gemv_q5_0",
    ("linear", "q5_1", "f32"): "gemv_q5_1",
    ("linear", "q8_0", "f32"): "gemv_q8_0",
    # GEMM for prefill (batched, quantized weights)
    ("linear_prefill", "q4_k", "f32"): "gemm_nt_q4_k",
    ("linear_prefill", "q6_k", "f32"): "gemm_nt_q6_k",
    ("linear_prefill", "q4_0", "f32"): "gemm_q4_0",
    ("linear_prefill", "q4_1", "f32"): "gemm_nt_q4_1",
    ("linear_prefill", "q5_0", "f32"): "gemm_nt_q5_0",
    ("linear_prefill", "q5_1", "f32"): "gemm_nt_q5_1",
    ("linear_prefill", "q8_0", "f32"): "gemm_q8_0",
    ("linear_prefill", "q4_k", "q8_k"): "gemm_nt_q4_k_q8_k",
    # Dequantization (for explicit dequant ops)
    ("dequant", "q4_k", "f32"): "dequant_q4_k_row",
    ("dequant", "q6_k", "f32"): "dequant_q6_k_row",
    ("dequant", "q4_0", "f32"): "dequant_q4_0_row",
    ("dequant", "q4_1", "f32"): "dequant_q4_1_row",
    ("dequant", "q5_0", "f32"): "dequant_q5_0_row",
    ("dequant", "q5_1", "f32"): "dequant_q5_1_row",
    ("dequant", "q8_0", "f32"): "dequant_q8_0_row",
    # On-the-fly activation quantization
    ("quantize", "f32", "q8_k"): "quantize_row_q8_k",
}

# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

QK_K = 256

def align_up_bytes(n: int, alignment: int) -> int:
    return (n + alignment - 1) & ~(alignment - 1)

def align_up_elems(elems: int, elem_bytes: int, alignment: int) -> int:
    return align_up_bytes(elems * elem_bytes, alignment) // elem_bytes


# ---------------------------------------------------------------------------
# Graph IR v5 (kernel-aligned, manifest-first)
# ---------------------------------------------------------------------------

def build_graph_ir_v5(config: Dict, model_name: str, alignment_bytes: int = 64) -> Dict:
    dtype = config["dtype"]
    elem_bytes = v3.DTYPE_BYTES.get(dtype, 4)
    weight_dtype = str(config.get("weight_dtype", "")).lower()
    use_k_align = weight_dtype in ("q4_k", "q6_k", "q8_k")
    qk_align_bytes = QK_K * elem_bytes

    E = config["embed_dim"]
    H = config["num_heads"]
    KV = config["num_kv_heads"]
    D = config["head_dim"]
    I = config["intermediate_dim"]
    T = config["max_seq_len"]
    V = config["vocab_size"]

    AE = align_up_elems(E, elem_bytes, qk_align_bytes if use_k_align else alignment_bytes)
    AD = align_up_elems(D, elem_bytes, alignment_bytes)
    AI = align_up_elems(I, elem_bytes, qk_align_bytes if use_k_align else alignment_bytes)
    AC = align_up_elems(T, elem_bytes, alignment_bytes)
    config["aligned_embed"] = AE
    config["aligned_head"] = AD
    config["aligned_intermediate"] = AI
    config["aligned_context"] = AC

    symbols = {
        "E": {"name": "embed_dim", "value": E},
        "AE": {"name": "aligned_embed", "value": AE},
        "H": {"name": "num_heads", "value": H},
        "KV": {"name": "num_kv_heads", "value": KV},
        "D": {"name": "head_dim", "value": D},
        "AD": {"name": "aligned_head", "value": AD},
        "I": {"name": "intermediate_dim", "value": I},
        "AI": {"name": "aligned_intermediate", "value": AI},
        "T": {"name": "max_seq_len", "value": T},
        "AC": {"name": "aligned_context", "value": AC},
        "S": {"name": "tokens", "value": T},
        "V": {"name": "vocab_size", "value": V},
        # Training-specific symbols (set during lowering)
        "B": {"name": "batch_size", "value": 1},
        "MB": {"name": "micro_batch_size", "value": 1},
        "ACCUM": {"name": "accumulation_steps", "value": 1},
    }
    sym_values = {k: v["value"] for k, v in symbols.items()}

    def buf(name: str,
            role: str,
            shape_expr: List[str],
            when: Optional[List[str]] = None,
            tied_to: Optional[str] = None,
            buf_dtype: Optional[str] = None) -> Dict:
        resolved = v3.resolve_shape_expr(shape_expr, sym_values)
        out = {
            "name": name,
            "role": role,
            "dtype": buf_dtype or dtype,
            "shape": shape_expr,
            "resolved_shape": resolved,
        }
        if tied_to:
            out["tied_to"] = tied_to
        if when:
            out["when"] = when
        return out

    globals_buffers = []
    if config.get("rope_theta", 0) > 0:
        globals_buffers.append(buf("rope_cos_cache", "precomputed", ["T", "D/2"]))
        globals_buffers.append(buf("rope_sin_cache", "precomputed", ["T", "D/2"]))

    header_buffers = [
        buf("token_emb", "weight", ["V", "AE"]),
        buf("embedded_input", "activation", ["S", "AE"]),
        # Backward gradients
        buf("d_embedded_input", "gradient", ["S", "AE"], when=["backward", "training"]),
        buf("d_token_emb", "weight_grad", ["V", "AE"], when=["backward", "training"]),
        # Adam optimizer state for token_emb (fp32 for numerical stability)
        buf("m_token_emb", "optimizer_state", ["V", "AE"], when=["training"], buf_dtype="f32"),
        buf("v_token_emb", "optimizer_state", ["V", "AE"], when=["training"], buf_dtype="f32"),
    ]

    layer_buffers = [
        # Forward activations
        buf("layer.{L}.input", "activation", ["S", "AE"]),
        buf("layer.{L}.ln1_gamma", "weight", ["AE"]),
        buf("layer.{L}.ln1_out", "activation", ["S", "AE"]),
        buf("layer.{L}.ln1_rstd", "activation", ["S"], when=["backward", "training"]),
        buf("layer.{L}.wq", "weight", ["H", "AD", "AE"]),
        buf("layer.{L}.bq", "weight", ["H", "AD"]),  # Attention Q bias (Qwen2-style)
        buf("layer.{L}.wk", "weight", ["KV", "AD", "AE"]),
        buf("layer.{L}.bk", "weight", ["KV", "AD"]),  # Attention K bias (Qwen2-style)
        buf("layer.{L}.wv", "weight", ["KV", "AD", "AE"]),
        buf("layer.{L}.bv", "weight", ["KV", "AD"]),  # Attention V bias (Qwen2-style)
        buf("layer.{L}.q", "activation", ["H", "S", "AD"]),
        buf("layer.{L}.k", "activation", ["KV", "AC", "AD"]),
        buf("layer.{L}.v", "activation", ["KV", "AC", "AD"]),
        buf("layer.{L}.scores", "activation", ["H", "AC", "AC"], when=["prefill", "backward"]),
        buf("layer.{L}.attn_out", "activation", ["H", "S", "AD"]),
        buf("layer.{L}.wo", "weight", ["H", "AE", "AD"]),
        buf("layer.{L}.bo", "weight", ["AE"]),  # Attention output bias (zeros placeholder)
        buf("layer.{L}.proj_tmp", "activation", ["S", "AE"]),
        buf("layer.{L}.proj_scratch", "scratch", ["S", "AE"]),
        buf("layer.{L}.residual1", "activation", ["S", "AE"]),
        buf("layer.{L}.ln2_gamma", "weight", ["AE"]),
        buf("layer.{L}.ln2_out", "activation", ["S", "AE"]),
        buf("layer.{L}.ln2_rstd", "activation", ["S"], when=["backward", "training"]),
        buf("layer.{L}.w1", "weight", ["2*AI", "AE"]),
        buf("layer.{L}.b1", "weight", ["2*AI"]),  # FFN bias (zeros placeholder)
        buf("layer.{L}.fc1_out", "activation", ["S", "2*AI"]),
        buf("layer.{L}.swiglu_out", "activation", ["S", "AI"]),
        buf("layer.{L}.w2", "weight", ["AE", "AI"]),
        buf("layer.{L}.b2", "weight", ["AE"]),  # FFN output bias (zeros placeholder)
        buf("layer.{L}.mlp_out", "activation", ["S", "AE"]),
        buf("layer.{L}.output", "activation", ["S", "AE"]),
        # Backward gradients (d_x = gradient w.r.t. x)
        buf("layer.{L}.d_output", "gradient", ["S", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_mlp_out", "gradient", ["S", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_swiglu_out", "gradient", ["S", "AI"], when=["backward", "training"]),
        buf("layer.{L}.d_fc1_out", "gradient", ["S", "2*AI"], when=["backward", "training"]),
        buf("layer.{L}.d_ln2_out", "gradient", ["S", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_residual1", "gradient", ["S", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_proj_tmp", "gradient", ["S", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_attn_out", "gradient", ["H", "S", "AD"], when=["backward", "training"]),
        buf("layer.{L}.d_q", "gradient", ["H", "S", "AD"], when=["backward", "training"]),
        buf("layer.{L}.d_k", "gradient", ["KV", "S", "AD"], when=["backward", "training"]),
        buf("layer.{L}.d_v", "gradient", ["KV", "S", "AD"], when=["backward", "training"]),
        buf("layer.{L}.d_ln1_out", "gradient", ["S", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_input", "gradient", ["S", "AE"], when=["backward", "training"]),
        # Weight gradients
        buf("layer.{L}.d_ln1_gamma", "weight_grad", ["AE"], when=["backward", "training"]),
        buf("layer.{L}.d_wq", "weight_grad", ["H", "AD", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_wk", "weight_grad", ["KV", "AD", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_wv", "weight_grad", ["KV", "AD", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_wo", "weight_grad", ["H", "AE", "AD"], when=["backward", "training"]),
        buf("layer.{L}.d_ln2_gamma", "weight_grad", ["AE"], when=["backward", "training"]),
        buf("layer.{L}.d_w1", "weight_grad", ["2*AI", "AE"], when=["backward", "training"]),
        buf("layer.{L}.d_w2", "weight_grad", ["AE", "AI"], when=["backward", "training"]),
        # Adam optimizer state: m (momentum), v (variance) - stored in fp32
        buf("layer.{L}.m_ln1_gamma", "optimizer_state", ["AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.v_ln1_gamma", "optimizer_state", ["AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.m_wq", "optimizer_state", ["H", "AD", "AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.v_wq", "optimizer_state", ["H", "AD", "AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.m_wk", "optimizer_state", ["KV", "AD", "AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.v_wk", "optimizer_state", ["KV", "AD", "AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.m_wv", "optimizer_state", ["KV", "AD", "AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.v_wv", "optimizer_state", ["KV", "AD", "AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.m_wo", "optimizer_state", ["H", "AE", "AD"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.v_wo", "optimizer_state", ["H", "AE", "AD"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.m_ln2_gamma", "optimizer_state", ["AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.v_ln2_gamma", "optimizer_state", ["AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.m_w1", "optimizer_state", ["2*AI", "AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.v_w1", "optimizer_state", ["2*AI", "AE"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.m_w2", "optimizer_state", ["AE", "AI"], when=["training"], buf_dtype="f32"),
        buf("layer.{L}.v_w2", "optimizer_state", ["AE", "AI"], when=["training"], buf_dtype="f32"),
    ]

    tie_embeddings = bool(config.get("tie_word_embeddings", True))

    footer_buffers = [
        buf("final_ln_weight", "weight", ["AE"]),
        buf("final_output", "activation", ["S", "AE"]),
        buf("final_ln_rstd", "activation", ["S"], when=["backward", "training"]),
        buf("lm_head_weight", "weight", ["V", "AE"], tied_to="token_emb" if tie_embeddings else None),
        buf("logits", "activation", ["S", "V"]),
        # Training: labels input and loss output
        buf("labels", "input", ["S"], when=["training"], buf_dtype="i32"),
        buf("loss", "output", [1], when=["training"], buf_dtype="f32"),
        # Backward gradients (loss gradient from cross-entropy)
        buf("d_logits", "gradient", ["S", "V"], when=["backward", "training"]),
        buf("d_final_output", "gradient", ["S", "AE"], when=["backward", "training"]),
        buf("d_final_ln_weight", "weight_grad", ["AE"], when=["backward", "training"]),
        buf("d_lm_head_weight", "weight_grad", ["V", "AE"], when=["backward", "training"]),
        # Adam optimizer state for final_ln_weight
        buf("m_final_ln_weight", "optimizer_state", ["AE"], when=["training"], buf_dtype="f32"),
        buf("v_final_ln_weight", "optimizer_state", ["AE"], when=["training"], buf_dtype="f32"),
        # lm_head optimizer state (tied to token_emb if tie_embeddings)
        buf("m_lm_head_weight", "optimizer_state", ["V", "AE"], when=["training"], buf_dtype="f32"),
        buf("v_lm_head_weight", "optimizer_state", ["V", "AE"], when=["training"], buf_dtype="f32"),
        # Training hyperparameters (scalars)
        buf("learning_rate", "hyperparameter", [1], when=["training"], buf_dtype="f32"),
        buf("beta1", "hyperparameter", [1], when=["training"], buf_dtype="f32"),
        buf("beta2", "hyperparameter", [1], when=["training"], buf_dtype="f32"),
        buf("epsilon", "hyperparameter", [1], when=["training"], buf_dtype="f32"),
        buf("weight_decay", "hyperparameter", [1], when=["training"], buf_dtype="f32"),
        buf("step_count", "state", [1], when=["training"], buf_dtype="i32"),
    ]

    header_ops = [
        {
            "op": "embedding",
            "name": "token_embed",
            "inputs": ["tokens"],
            "weights": ["token_emb"],
            "outputs": ["embedded_input"],
        },
    ]
    if globals_buffers:
        header_ops.append({
            "op": "rope_precompute",
            "name": "rope_precompute",
            "inputs": [],
            "outputs": ["rope_cos_cache", "rope_sin_cache"],
            "attrs": {"theta": config.get("rope_theta", 10000.0)},
        })

    body_ops = [
        {
            "op": "rmsnorm",
            "name": "ln1",
            "inputs": ["input"],
            "weights": ["layer.{L}.ln1_gamma"],
            "outputs": ["layer.{L}.ln1_out"],
        },
        {
            "op": "qkv_project",
            "name": "qkv_project",
            "inputs": ["layer.{L}.ln1_out"],
            "weights": ["layer.{L}.wq", "layer.{L}.wk", "layer.{L}.wv"],
            "biases": ["layer.{L}.bq", "layer.{L}.bk", "layer.{L}.bv"],
            "outputs": ["layer.{L}.q", "layer.{L}.k", "layer.{L}.v"],
        },
        {
            "op": "attention",
            "name": "attention",
            "inputs": ["layer.{L}.q", "layer.{L}.k", "layer.{L}.v"],
            "outputs": ["layer.{L}.attn_out"],
            "scratch": ["layer.{L}.scores"],
        },
        {
            "op": "attn_proj",
            "name": "attn_proj",
            "inputs": ["layer.{L}.attn_out"],
            "weights": ["layer.{L}.wo"],
            "outputs": ["layer.{L}.proj_tmp"],
            "scratch": ["layer.{L}.proj_scratch"],
        },
        {
            "op": "residual_add",
            "name": "residual1",
            "inputs": ["input", "layer.{L}.proj_tmp"],
            "outputs": ["layer.{L}.residual1"],
        },
        {
            "op": "rmsnorm",
            "name": "ln2",
            "inputs": ["layer.{L}.residual1"],
            "weights": ["layer.{L}.ln2_gamma"],
            "outputs": ["layer.{L}.ln2_out"],
        },
        {
            "op": "mlp_up",
            "name": "mlp_up",
            "inputs": ["layer.{L}.ln2_out"],
            "weights": ["layer.{L}.w1"],
            "outputs": ["layer.{L}.fc1_out"],
        },
        {
            "op": "swiglu",
            "name": "swiglu",
            "inputs": ["layer.{L}.fc1_out"],
            "outputs": ["layer.{L}.swiglu_out"],
        },
        {
            "op": "mlp_down",
            "name": "mlp_down",
            "inputs": ["layer.{L}.swiglu_out"],
            "weights": ["layer.{L}.w2"],
            "outputs": ["layer.{L}.mlp_out"],
        },
        {
            "op": "residual_add",
            "name": "residual2",
            "inputs": ["layer.{L}.residual1", "layer.{L}.mlp_out"],
            "outputs": ["layer.{L}.output"],
        },
    ]

    if globals_buffers:
        body_ops.insert(2, {
            "op": "rope",
            "name": "rope",
            "inputs": ["layer.{L}.q", "layer.{L}.k", "rope_cos_cache", "rope_sin_cache"],
            "outputs": ["layer.{L}.q", "layer.{L}.k"],
        })

    # Backward ops (reverse order of forward)
    # Used by both 'backward' mode (gradient-only) and 'training' mode (forward+backward)
    backward_body_ops = [
        # residual2 backward: d_output splits to d_residual1 and d_mlp_out
        {
            "op": "add_backward",
            "name": "residual2_backward",
            "inputs": ["d_output"],
            "outputs": ["layer.{L}.d_residual1", "layer.{L}.d_mlp_out"],
            "when": ["backward", "training"],
        },
        # mlp_down backward: d_mlp_out -> d_swiglu_out, d_w2
        {
            "op": "gemm_backward",
            "name": "mlp_down_backward",
            "inputs": ["layer.{L}.d_mlp_out", "layer.{L}.swiglu_out", "layer.{L}.w2"],
            "outputs": ["layer.{L}.d_swiglu_out"],
            "weight_grads": ["layer.{L}.d_w2"],
            "when": ["backward", "training"],
        },
        # swiglu backward: d_swiglu_out -> d_fc1_out
        {
            "op": "swiglu_backward",
            "name": "swiglu_backward",
            "inputs": ["layer.{L}.d_swiglu_out", "layer.{L}.fc1_out"],
            "outputs": ["layer.{L}.d_fc1_out"],
            "when": ["backward", "training"],
        },
        # mlp_up backward: d_fc1_out -> d_ln2_out, d_w1
        {
            "op": "gemm_backward",
            "name": "mlp_up_backward",
            "inputs": ["layer.{L}.d_fc1_out", "layer.{L}.ln2_out", "layer.{L}.w1"],
            "outputs": ["layer.{L}.d_ln2_out"],
            "weight_grads": ["layer.{L}.d_w1"],
            "when": ["backward", "training"],
        },
        # ln2 backward: d_ln2_out -> d_residual1_ln2, d_ln2_gamma
        {
            "op": "rmsnorm_backward",
            "name": "ln2_backward",
            "inputs": ["layer.{L}.d_ln2_out", "layer.{L}.residual1", "layer.{L}.ln2_gamma", "layer.{L}.ln2_rstd"],
            "outputs": ["layer.{L}.d_residual1"],  # Accumulates with residual path
            "weight_grads": ["layer.{L}.d_ln2_gamma"],
            "when": ["backward", "training"],
        },
        # residual1 backward: d_residual1 splits to d_input and d_proj_tmp
        {
            "op": "add_backward",
            "name": "residual1_backward",
            "inputs": ["layer.{L}.d_residual1"],
            "outputs": ["layer.{L}.d_input", "layer.{L}.d_proj_tmp"],
            "when": ["backward", "training"],
        },
        # attn_proj backward: d_proj_tmp -> d_attn_out, d_wo
        {
            "op": "gemm_backward",
            "name": "attn_proj_backward",
            "inputs": ["layer.{L}.d_proj_tmp", "layer.{L}.attn_out", "layer.{L}.wo"],
            "outputs": ["layer.{L}.d_attn_out"],
            "weight_grads": ["layer.{L}.d_wo"],
            "when": ["backward", "training"],
        },
        # attention backward: d_attn_out -> d_q, d_k, d_v
        {
            "op": "attention_backward",
            "name": "attention_backward",
            "inputs": ["layer.{L}.d_attn_out", "layer.{L}.q", "layer.{L}.k", "layer.{L}.v", "layer.{L}.scores"],
            "outputs": ["layer.{L}.d_q", "layer.{L}.d_k", "layer.{L}.d_v"],
            "when": ["backward", "training"],
        },
        # qkv_project backward: d_q,d_k,d_v -> d_ln1_out, d_wq,d_wk,d_wv
        {
            "op": "qkv_backward",
            "name": "qkv_backward",
            "inputs": ["layer.{L}.d_q", "layer.{L}.d_k", "layer.{L}.d_v", "layer.{L}.ln1_out"],
            "weights": ["layer.{L}.wq", "layer.{L}.wk", "layer.{L}.wv"],
            "outputs": ["layer.{L}.d_ln1_out"],
            "weight_grads": ["layer.{L}.d_wq", "layer.{L}.d_wk", "layer.{L}.d_wv"],
            "when": ["backward", "training"],
        },
        # ln1 backward: d_ln1_out -> d_input, d_ln1_gamma
        {
            "op": "rmsnorm_backward",
            "name": "ln1_backward",
            "inputs": ["layer.{L}.d_ln1_out", "input", "layer.{L}.ln1_gamma", "layer.{L}.ln1_rstd"],
            "outputs": ["layer.{L}.d_input"],  # Accumulates with residual path
            "weight_grads": ["layer.{L}.d_ln1_gamma"],
            "when": ["backward", "training"],
        },
    ]

    # Insert rope backward if rope is used
    if globals_buffers:
        # Find attention_backward index and insert rope_backward before it
        for i, op in enumerate(backward_body_ops):
            if op["name"] == "attention_backward":
                backward_body_ops.insert(i + 1, {
                    "op": "rope_backward",
                    "name": "rope_backward",
                    "inputs": ["layer.{L}.d_q", "layer.{L}.d_k", "rope_cos_cache", "rope_sin_cache"],
                    "outputs": ["layer.{L}.d_q", "layer.{L}.d_k"],
                    "when": ["backward", "training"],
                })
                break

    footer_ops = [
        {
            "op": "rmsnorm",
            "name": "final_ln",
            "inputs": ["last_layer_output"],
            "weights": ["final_ln_weight"],
            "outputs": ["final_output"],
            "cache": ["final_ln_rstd"],  # Cache rstd for backward
        },
        {
            "op": "lm_head",
            "name": "lm_head",
            "inputs": ["final_output"],
            "weights": ["lm_head_weight"],
            "outputs": ["logits"],
        },
        # Training: cross-entropy loss (logits, labels) -> loss, d_logits
        {
            "op": "cross_entropy_loss",
            "name": "cross_entropy",
            "inputs": ["logits", "labels"],
            "outputs": ["loss", "d_logits"],
            "when": ["training"],
            "description": "Cross-entropy loss with fused softmax and gradient computation",
        },
    ]

    # Footer backward ops (start from d_logits)
    # For 'backward' mode: d_logits is assumed to be provided externally
    # For 'training' mode: d_logits comes from cross_entropy_loss above
    footer_backward_ops = [
        # lm_head backward: d_logits -> d_final_output, d_lm_head_weight
        {
            "op": "gemm_backward",
            "name": "lm_head_backward",
            "inputs": ["d_logits", "final_output", "lm_head_weight"],
            "outputs": ["d_final_output"],
            "weight_grads": ["d_lm_head_weight"],
            "when": ["backward", "training"],
        },
        # final_ln backward: d_final_output -> d_last_layer_output, d_final_ln_weight
        {
            "op": "rmsnorm_backward",
            "name": "final_ln_backward",
            "inputs": ["d_final_output", "last_layer_output", "final_ln_weight", "final_ln_rstd"],
            "outputs": ["d_last_layer_output"],
            "weight_grads": ["d_final_ln_weight"],
            "when": ["backward", "training"],
        },
    ]

    # Header backward ops
    header_backward_ops = [
        # embedding backward: d_embedded_input -> d_token_emb (scatter add)
        {
            "op": "embedding_backward",
            "name": "embedding_backward",
            "inputs": ["d_embedded_input", "tokens"],
            "outputs": ["d_token_emb"],
            "when": ["backward", "training"],
        },
    ]

    # Optimizer step ops (AdamW update for each weight)
    # These run after backward pass, updating weights using gradients
    optimizer_layer_ops = [
        # ln1_gamma
        {
            "op": "adamw_update",
            "name": "adamw_ln1_gamma",
            "inputs": ["layer.{L}.d_ln1_gamma", "layer.{L}.ln1_gamma",
                       "layer.{L}.m_ln1_gamma", "layer.{L}.v_ln1_gamma",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["layer.{L}.ln1_gamma", "layer.{L}.m_ln1_gamma", "layer.{L}.v_ln1_gamma"],
            "when": ["training"],
        },
        # wq
        {
            "op": "adamw_update",
            "name": "adamw_wq",
            "inputs": ["layer.{L}.d_wq", "layer.{L}.wq",
                       "layer.{L}.m_wq", "layer.{L}.v_wq",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["layer.{L}.wq", "layer.{L}.m_wq", "layer.{L}.v_wq"],
            "when": ["training"],
        },
        # wk
        {
            "op": "adamw_update",
            "name": "adamw_wk",
            "inputs": ["layer.{L}.d_wk", "layer.{L}.wk",
                       "layer.{L}.m_wk", "layer.{L}.v_wk",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["layer.{L}.wk", "layer.{L}.m_wk", "layer.{L}.v_wk"],
            "when": ["training"],
        },
        # wv
        {
            "op": "adamw_update",
            "name": "adamw_wv",
            "inputs": ["layer.{L}.d_wv", "layer.{L}.wv",
                       "layer.{L}.m_wv", "layer.{L}.v_wv",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["layer.{L}.wv", "layer.{L}.m_wv", "layer.{L}.v_wv"],
            "when": ["training"],
        },
        # wo
        {
            "op": "adamw_update",
            "name": "adamw_wo",
            "inputs": ["layer.{L}.d_wo", "layer.{L}.wo",
                       "layer.{L}.m_wo", "layer.{L}.v_wo",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["layer.{L}.wo", "layer.{L}.m_wo", "layer.{L}.v_wo"],
            "when": ["training"],
        },
        # ln2_gamma
        {
            "op": "adamw_update",
            "name": "adamw_ln2_gamma",
            "inputs": ["layer.{L}.d_ln2_gamma", "layer.{L}.ln2_gamma",
                       "layer.{L}.m_ln2_gamma", "layer.{L}.v_ln2_gamma",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["layer.{L}.ln2_gamma", "layer.{L}.m_ln2_gamma", "layer.{L}.v_ln2_gamma"],
            "when": ["training"],
        },
        # w1 (gate + up)
        {
            "op": "adamw_update",
            "name": "adamw_w1",
            "inputs": ["layer.{L}.d_w1", "layer.{L}.w1",
                       "layer.{L}.m_w1", "layer.{L}.v_w1",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["layer.{L}.w1", "layer.{L}.m_w1", "layer.{L}.v_w1"],
            "when": ["training"],
        },
        # w2 (down)
        {
            "op": "adamw_update",
            "name": "adamw_w2",
            "inputs": ["layer.{L}.d_w2", "layer.{L}.w2",
                       "layer.{L}.m_w2", "layer.{L}.v_w2",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["layer.{L}.w2", "layer.{L}.m_w2", "layer.{L}.v_w2"],
            "when": ["training"],
        },
    ]

    # Header optimizer ops (token_emb)
    optimizer_header_ops = [
        {
            "op": "adamw_update",
            "name": "adamw_token_emb",
            "inputs": ["d_token_emb", "token_emb",
                       "m_token_emb", "v_token_emb",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["token_emb", "m_token_emb", "v_token_emb"],
            "when": ["training"],
        },
    ]

    # Footer optimizer ops (final_ln_weight, lm_head_weight)
    optimizer_footer_ops = [
        {
            "op": "adamw_update",
            "name": "adamw_final_ln_weight",
            "inputs": ["d_final_ln_weight", "final_ln_weight",
                       "m_final_ln_weight", "v_final_ln_weight",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["final_ln_weight", "m_final_ln_weight", "v_final_ln_weight"],
            "when": ["training"],
        },
        # lm_head update (skipped if tied to token_emb)
        {
            "op": "adamw_update",
            "name": "adamw_lm_head_weight",
            "inputs": ["d_lm_head_weight", "lm_head_weight",
                       "m_lm_head_weight", "v_lm_head_weight",
                       "learning_rate", "beta1", "beta2", "epsilon", "weight_decay", "step_count"],
            "outputs": ["lm_head_weight", "m_lm_head_weight", "v_lm_head_weight"],
            "when": ["training"],
            "skip_if_tied": True,  # Skip if lm_head tied to token_emb
        },
        # Increment step count
        {
            "op": "increment",
            "name": "increment_step",
            "inputs": ["step_count"],
            "outputs": ["step_count"],
            "when": ["training"],
        },
    ]

    # Gradient reduction ops (for data parallel training)
    gradient_reduction_ops = [
        # AllReduce all weight gradients (sum across workers, then average)
        {
            "op": "allreduce",
            "name": "allreduce_gradients",
            "inputs": ["all_weight_gradients"],
            "outputs": ["all_weight_gradients"],
            "attrs": {
                "reduce_op": "sum",
                "scale": "1/data_parallel_size",
            },
            "when": ["training"],
            "condition": "data_parallel_size > 1",
        },
    ]

    section = {
        "id": 0,
        "name": "text_decoder",
        "inputs": [
            {"name": "tokens", "dtype": "i32", "shape": ["S"]},
        ],
        "globals": globals_buffers,
        "buffers": {
            "header": header_buffers,
            "layer": layer_buffers,
            "footer": footer_buffers,
        },
        "header": {
            "ops": header_ops,
            "backward_ops": header_backward_ops,
            "optimizer_ops": optimizer_header_ops,
            "outputs": ["embedded_input"],
        },
        "body": {
            "repeat": "num_layers",
            "layer_var": "L",
            "bindings": {
                "input": {
                    "first_layer": "embedded_input",
                    "next_layer": "layer.{L-1}.output",
                },
                # Backward bindings (gradient flows from output to input)
                "d_output": {
                    "last_layer": "d_last_layer_output",  # From footer backward
                    "prev_layer": "layer.{L+1}.d_input",  # From next layer's backward
                },
            },
            "ops": body_ops,
            "backward_ops": backward_body_ops,
            "optimizer_ops": optimizer_layer_ops,
            "outputs": ["layer.{L}.output"],
        },
        "footer": {
            "bindings": {
                "last_layer_output": "layer.{L-1}.output",
            },
            "ops": footer_ops,
            "backward_ops": footer_backward_ops,
            "optimizer_ops": optimizer_footer_ops,
            "outputs": ["logits"],
        },
        "gradient_reduction": gradient_reduction_ops,
    }

    if config.get("model_type") in {"llama", "qwen2", "mistral"}:
        weight_map = WEIGHT_MAP_V4
    else:
        weight_map = []

    return {
        "version": 4,
        "kind": "graph",
        "generated": datetime.utcnow().isoformat() + "Z",
        "model": model_name,
        "config": config,
        "symbols": symbols,
        "sections": [section],
        "weight_map": weight_map,
    }


# ---------------------------------------------------------------------------
# Weights metadata (safetensors header / index)
# ---------------------------------------------------------------------------

def read_safetensors_header(path: str) -> Dict:
    """Read safetensors header without loading weights."""
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header_json = f.read(header_len).decode("utf-8")
    return json.loads(header_json)


def read_weights_index(path: str) -> Dict:
    """Read model.safetensors.index.json (names only)."""
    with open(path, "r") as f:
        return json.load(f)


def extract_weight_names(weights_meta: Dict) -> List[str]:
    names = set()
    header = weights_meta.get("header", {})
    for key in header.keys():
        if key != "__metadata__":
            names.add(key)
    index = weights_meta.get("index", {})
    weight_map = index.get("weight_map", {})
    names.update(weight_map.keys())
    return sorted(names)


def extract_kernel_names_from_c(path: str) -> List[str]:
    """Extract kernel spec names from src/ckernel_kernel_specs.c."""
    kernels = []
    in_table = False
    with open(path, "r") as f:
        for line in f:
            if "const CKKernelSpec ck_kernel_specs[]" in line:
                in_table = True
                continue
            if in_table and line.strip().startswith("};"):
                break
            if not in_table:
                continue
            m = re.search(r'\{\s*"([^"]+)"\s*,', line)
            if m:
                kernels.append(m.group(1))
    return kernels


def load_kernel_registry(path: Optional[str] = None) -> Dict[str, Dict]:
    if path:
        with open(path, "r") as f:
            data = json.load(f)
        kernels = data.get("kernels", [])
        return {k["name"]: k for k in kernels}

    c_path = os.path.join("src", "ckernel_kernel_specs.c")
    if os.path.exists(c_path):
        names = extract_kernel_names_from_c(c_path)
        return {name: {"name": name} for name in names}
    return {}


# ---------------------------------------------------------------------------
# Lowering helpers
# ---------------------------------------------------------------------------

def expand_layer_name(name: str, layer_id: int) -> str:
    """Expand layer placeholders in a buffer/op name."""
    if "{L-1}" in name:
        name = name.replace("{L-1}", str(layer_id - 1))
    if "{L}" in name:
        name = name.replace("{L}", str(layer_id))
    return name


def normalize_layer_template(name: str) -> str:
    """Normalize layer.N.* into layer.{L}.*"""
    return re.sub(r"layer\.[0-9]+\.", "layer.{L}.", name)


def op_enabled(op: Dict, mode: str) -> bool:
    when = op.get("when")
    if not when:
        return True
    return mode in when


def select_kernel(op: Dict, dtype: str, mode: str, registry: Dict[str, Dict],
                  weight_dtype: Optional[str] = None) -> Optional[str]:
    """Select kernel for an operation.

    Args:
        op: Operation dict with at least "op" key
        dtype: Default/activation dtype ("f32", "bf16", etc.)
        mode: Execution mode ("prefill", "decode", "training")
        registry: Optional kernel registry for custom kernels
        weight_dtype: Optional weight dtype for quantized inference ("q4_k", "q6_k", etc.)

    Returns:
        Kernel function name, or None for host-side ops
    """
    op_name = op["op"]
    if op_name in HOST_OPS or op_name == "lm_head":
        return None
    if op_name == "attention":
        key = "attention" if mode in {"prefill", "decode"} else op_name
    else:
        key = op_name

    # Normalize dtype
    dtype_key = dtype
    if dtype_key == "fp32":
        dtype_key = "f32"
    elif dtype_key == "fp16":
        dtype_key = "f16"

    # Check for quantized weight operations
    w_dtype = weight_dtype or op.get("weight_dtype")
    if w_dtype and qt.is_quantized_dtype(w_dtype):
        w_dtype_key = w_dtype.lower()
        # Try QUANT_KERNELS first
        quant_key = (key, w_dtype_key, dtype_key)
        if quant_key in QUANT_KERNELS:
            return QUANT_KERNELS[quant_key]
        # For prefill mode, try linear_prefill variant
        if mode == "prefill" and key == "linear":
            prefill_key = ("linear_prefill", w_dtype_key, dtype_key)
            if prefill_key in QUANT_KERNELS:
                return QUANT_KERNELS[prefill_key]

    if registry and key in registry:
        return key

    candidates = KERNELS.get(key, {})
    return candidates.get(dtype_key, f"{key}_{dtype_key}")


def lower_graph_ir(graph: Dict, mode: str, tokens: int, registry: Dict[str, Dict],
                   training_cfg: Optional["tc.TrainingConfig"] = None,
                   weights_manifest: Optional[Dict] = None,
                   weight_dtype: Optional[str] = None) -> Dict:
    """Lower graph IR into a per-mode expanded program.

    Modes:
      - prefill: Forward pass only (parallel attention)
      - decode: Forward pass only (single-token attention)
      - backward: Backward pass only (assumes activations cached)
      - training: Forward + loss + backward (complete training step)

    Args:
      training_cfg: Optional TrainingConfig with batch size, optimizer settings, etc.
    """
    config = graph["config"]
    symbols = graph["symbols"].copy()

    # Override tokens (S) while keeping max seq (T)
    symbols["S"] = {"name": "tokens", "value": tokens}

    # For training mode, set batch and accumulation parameters
    if mode == "training" and training_cfg:
        symbols["B"] = {"name": "batch_size", "value": training_cfg.batch_size}
        symbols["MB"] = {"name": "micro_batch_size", "value": training_cfg.micro_batch_size}
        symbols["ACCUM"] = {"name": "accumulation_steps", "value": training_cfg.accumulation_steps}
    else:
        symbols["B"] = {"name": "batch_size", "value": 1}
        symbols["MB"] = {"name": "micro_batch_size", "value": 1}
        symbols["ACCUM"] = {"name": "accumulation_steps", "value": 1}

    sym_values = {k: v["value"] for k, v in symbols.items()}

    section = graph["sections"][0]
    num_layers = config["num_layers"]
    manifest_entries = {}
    if weights_manifest and isinstance(weights_manifest, dict):
        for entry in weights_manifest.get("entries", []):
            if "name" in entry:
                manifest_entries[entry["name"]] = entry

    # Templates
    header_templates = section["buffers"]["header"]
    layer_templates = section["buffers"]["layer"]
    footer_templates = section["buffers"]["footer"]
    globals_templates = section.get("globals", [])

    header_template_map = {b["name"]: b for b in header_templates}
    footer_template_map = {b["name"]: b for b in footer_templates}
    globals_template_map = {b["name"]: b for b in globals_templates}
    layer_template_map = {b["name"]: b for b in layer_templates}

    inputs = []
    for buf in section.get("inputs", []):
        resolved = v3.resolve_shape_expr(buf["shape"], sym_values)
        inputs.append({**buf, "resolved_shape": resolved})
    input_names = {b["name"] for b in inputs}

    def buffer_spec(name: str, mode_override: Optional[str] = None) -> Optional[Dict]:
        check_mode = mode_override or mode
        if name in header_template_map:
            tmpl = header_template_map[name]
        elif name in footer_template_map:
            tmpl = footer_template_map[name]
        elif name in globals_template_map:
            tmpl = globals_template_map[name]
        elif name.startswith("layer."):
            tmpl_name = normalize_layer_template(name)
            tmpl = layer_template_map.get(tmpl_name)
            if tmpl is None:
                raise KeyError(f"Missing layer template for: {name}")
        else:
            raise KeyError(f"Missing template for: {name}")

        # Check when clause - for training mode, accept both forward and backward buffers
        when = tmpl.get("when")
        if when:
            if check_mode == "training":
                # Training mode needs all buffers (forward + backward)
                pass  # Accept all
            elif check_mode not in when:
                return None

        # Get buffer role and shape
        role = tmpl.get("role", "activation")
        shape = list(tmpl["shape"])  # Copy to avoid modifying template

        # For CPU training: buffers stay 2D (no batch dimension)
        # Batch is simulated via sequential accumulation loop, not 3D tensor ops
        # AMX only supports 2D tile operations

        resolved = v3.resolve_shape_expr(shape, sym_values)
        dtype = tmpl.get("dtype", config["dtype"])
        if role == "weight":
            manifest = manifest_entries.get(name)
            if manifest and "dtype" in manifest:
                dtype = manifest["dtype"]
            elif weight_dtype:
                dtype = weight_dtype

        out = {
            "name": name,
            "role": role,
            "dtype": dtype,
            "shape": shape,
            "resolved_shape": resolved,
        }
        if tmpl.get("tied_to"):
            out["tied_to"] = tmpl["tied_to"]
        if role == "weight":
            manifest = manifest_entries.get(name)
            if manifest and "size" in manifest:
                out["file_size"] = int(manifest["size"])
        return out

    def resolve_names(names: List[str], layer_id: int, bindings: Dict[str, str]) -> List[str]:
        out = []
        for n in names:
            if n in bindings:
                n = bindings[n]
            n = expand_layer_name(n, layer_id)
            out.append(n)
        return out

    def process_ops(ops_source: List[Dict], layer_id: int, bindings: Dict[str, str],
                    used_bufs: Dict[str, set], skip_registry_check: bool = False) -> List[Dict]:
        """Process a list of ops, expanding names and selecting kernels."""
        result = []
        for op in ops_source:
            if not op_enabled(op, mode):
                continue
            op_out = dict(op)
            op_weight_dtype = None
            op_weight_names = []
            if "weights" in op_out:
                op_weight_names = resolve_names(op_out["weights"], layer_id, bindings)
                for w_name in op_weight_names:
                    entry = manifest_entries.get(w_name)
                    w_dtype = None
                    if entry and "dtype" in entry:
                        w_dtype = entry["dtype"]
                    elif weight_dtype:
                        w_dtype = weight_dtype
                    if w_dtype:
                        if op_weight_dtype is None:
                            op_weight_dtype = w_dtype
                        elif op_weight_dtype != w_dtype:
                            op_weight_dtype = None
                            break
            if op_weight_dtype:
                op_out["weight_dtype"] = op_weight_dtype
            op_out["kernel"] = select_kernel(op, config["dtype"], mode, registry,
                                             weight_dtype=op_weight_dtype)
            # Don't require backward kernels to be in registry
            if (op_out["kernel"] and registry and op_out["kernel"] not in registry and
                    not skip_registry_check and not (op_weight_dtype and qt.is_quantized_dtype(op_weight_dtype))):
                raise KeyError(f"Unknown kernel: {op_out['kernel']}")
            if op_out["kernel"]:
                op_out["kernel_dtype"] = config["dtype"]

            for key in ("inputs", "outputs", "weights", "biases", "scratch", "weight_grads", "cache"):
                if key in op_out:
                    names = resolve_names(op_out[key], layer_id, bindings)
                    op_out[key] = names
                    for name in names:
                        if name in input_names:
                            continue
                        if name.startswith("layer."):
                            used_bufs["layer"].add(name)
                        elif name in globals_template_map:
                            used_bufs["globals"].add(name)
                        elif name in header_template_map:
                            used_bufs["header"].add(name)
                        elif name in footer_template_map:
                            used_bufs["footer"].add(name)

            # Special case: attention in decode mode doesn't need scratch
            if "scratch" in op_out and op_out["op"] == "attention" and mode == "decode":
                op_out["scratch"] = []

            result.append(op_out)
        return result

    # Determine which ops to process based on mode
    is_backward = mode == "backward"
    is_training = mode == "training"
    skip_registry = is_backward or is_training  # Backward kernels may not be registered

    # For training mode, we produce a two-phase schedule: forward + backward
    if is_training:
        return _lower_training_mode(
            graph, section, config, symbols, sym_values, num_layers,
            header_templates, layer_templates, footer_templates, globals_templates,
            header_template_map, footer_template_map, globals_template_map, layer_template_map,
            inputs, input_names, buffer_spec, resolve_names, process_ops, registry
        )

    # For backward mode, we process backward_ops; for forward modes, we process ops
    if is_backward:
        header_ops_source = section["header"].get("backward_ops", [])
        body_ops_source = section["body"].get("backward_ops", [])
        footer_ops_source = section["footer"].get("backward_ops", [])
    else:
        # Forward ops (for prefill, decode)
        header_ops_source = section["header"]["ops"]
        body_ops_source = section["body"]["ops"]
        footer_ops_source = section["footer"]["ops"]

    # Track used buffers
    used_bufs = {"header": set(), "footer": set(), "globals": set(), "layer": set()}

    # Header ops (no layer expansion)
    header_ops = process_ops(header_ops_source, 0, {}, used_bufs, skip_registry)

    # Body ops (expanded per layer)
    # For backward mode, process layers in reverse order
    layers_out = []
    layer_order = range(num_layers - 1, -1, -1) if is_backward else range(num_layers)

    for layer_id in layer_order:
        used_bufs["layer"] = set()  # Reset per layer

        bindings = {}
        for key, spec in section["body"].get("bindings", {}).items():
            if isinstance(spec, dict):
                # Forward bindings
                if "first_layer" in spec and "next_layer" in spec:
                    if layer_id == 0:
                        bindings[key] = spec["first_layer"]
                    else:
                        bindings[key] = expand_layer_name(spec["next_layer"], layer_id)
                # Backward bindings
                elif "last_layer" in spec and "prev_layer" in spec:
                    if layer_id == num_layers - 1:
                        bindings[key] = spec["last_layer"]
                    else:
                        bindings[key] = expand_layer_name(spec["prev_layer"], layer_id)
            else:
                bindings[key] = spec

        layer_ops = process_ops(body_ops_source, layer_id, bindings, used_bufs, skip_registry)

        # Build layer buffers in template order
        layer_buffers = []
        for tmpl in layer_templates:
            name = expand_layer_name(tmpl["name"], layer_id)
            if name in used_bufs["layer"]:
                spec = buffer_spec(name)
                if spec:
                    layer_buffers.append(spec)

        layers_out.append({
            "id": layer_id,
            "ops": layer_ops,
            "buffers": layer_buffers,
        })

    # Footer ops
    footer_bindings = {}
    for key, spec in section["footer"].get("bindings", {}).items():
        if isinstance(spec, dict) and "first_layer" in spec and "next_layer" in spec:
            if num_layers == 0:
                footer_bindings[key] = spec.get("first_layer", spec.get("next_layer", ""))
            else:
                footer_bindings[key] = expand_layer_name(spec.get("next_layer", ""), num_layers)
        else:
            footer_bindings[key] = expand_layer_name(spec, num_layers)

    footer_ops = process_ops(footer_ops_source, num_layers, footer_bindings, used_bufs, skip_registry)

    # Build header/footer/global buffers in template order
    header_buffers = []
    for tmpl in header_templates:
        name = tmpl["name"]
        if name in used_bufs["header"]:
            spec = buffer_spec(name)
            if spec:
                header_buffers.append(spec)

    footer_buffers = []
    for tmpl in footer_templates:
        name = tmpl["name"]
        if name in used_bufs["footer"]:
            spec = buffer_spec(name)
            if spec:
                footer_buffers.append(spec)

    # Resolve tied_to references: if a buffer references another, add the target
    header_names = {b["name"] for b in header_buffers}
    for buf in footer_buffers:
        tied_to = buf.get("tied_to")
        if tied_to and tied_to not in header_names:
            spec = buffer_spec(tied_to)
            if spec:
                header_buffers.insert(0, spec)
                header_names.add(tied_to)

    globals_buffers = []
    for tmpl in globals_templates:
        name = tmpl["name"]
        if name in used_bufs["globals"]:
            spec = buffer_spec(name)
            if spec:
                globals_buffers.append(spec)

    return {
        "version": 4,
        "kind": "lowered",
        "mode": mode,
        "generated": datetime.utcnow().isoformat() + "Z",
        "model": graph["model"],
        "config": config,
        "symbols": symbols,
        "sections": [
            {
                "id": section["id"],
                "name": section["name"],
                "inputs": inputs,
                "globals": globals_buffers,
                "header": {"ops": header_ops, "buffers": header_buffers},
                "layers": layers_out,
                "footer": {"ops": footer_ops, "buffers": footer_buffers},
            }
        ],
    }


def _lower_training_mode(
    graph: Dict, section: Dict, config: Dict, symbols: Dict, sym_values: Dict,
    num_layers: int, header_templates: List, layer_templates: List,
    footer_templates: List, globals_templates: List,
    header_template_map: Dict, footer_template_map: Dict,
    globals_template_map: Dict, layer_template_map: Dict,
    inputs: List, input_names: set,
    buffer_spec, resolve_names, process_ops, registry: Dict
) -> Dict:
    """
    Lower graph IR for training mode: forward pass + loss + backward pass.

    Produces a structure with explicit forward and backward phases:
    {
        "forward_pass": { header, layers[0..N], footer (with loss) },
        "backward_pass": { footer, layers[N..0], header }
    }
    """
    mode = "training"

    # Track all used buffers
    all_used = {"header": set(), "footer": set(), "globals": set()}

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    # Forward header ops
    fwd_header_ops = process_ops(
        section["header"]["ops"], 0, {},
        {"header": all_used["header"], "footer": all_used["footer"],
         "globals": all_used["globals"], "layer": set()},
        skip_registry_check=True
    )

    # Forward body ops (layers 0 → N-1)
    fwd_layers = []
    for layer_id in range(num_layers):
        layer_used = set()
        used_bufs = {"header": all_used["header"], "footer": all_used["footer"],
                     "globals": all_used["globals"], "layer": layer_used}

        # Build forward bindings
        bindings = {}
        for key, spec in section["body"].get("bindings", {}).items():
            if isinstance(spec, dict):
                if "first_layer" in spec and "next_layer" in spec:
                    if layer_id == 0:
                        bindings[key] = spec["first_layer"]
                    else:
                        bindings[key] = expand_layer_name(spec["next_layer"], layer_id)
            elif not isinstance(spec, dict):
                bindings[key] = spec

        layer_ops = process_ops(
            section["body"]["ops"], layer_id, bindings, used_bufs,
            skip_registry_check=True
        )

        # Collect all layer buffer names used in forward
        fwd_layers.append({
            "id": layer_id,
            "ops": layer_ops,
            "_used": layer_used,
        })

    # Forward footer ops (includes lm_head and cross_entropy_loss)
    footer_bindings = {}
    for key, spec in section["footer"].get("bindings", {}).items():
        if isinstance(spec, dict):
            continue
        footer_bindings[key] = expand_layer_name(spec, num_layers)

    fwd_footer_ops = process_ops(
        section["footer"]["ops"], num_layers, footer_bindings,
        {"header": all_used["header"], "footer": all_used["footer"],
         "globals": all_used["globals"], "layer": set()},
        skip_registry_check=True
    )

    # =========================================================================
    # BACKWARD PASS
    # =========================================================================

    # Backward footer ops (lm_head_backward, final_ln_backward)
    bwd_footer_ops = process_ops(
        section["footer"].get("backward_ops", []), num_layers, footer_bindings,
        {"header": all_used["header"], "footer": all_used["footer"],
         "globals": all_used["globals"], "layer": set()},
        skip_registry_check=True
    )

    # Backward body ops (layers N-1 → 0, reverse order)
    bwd_layers = []
    for layer_id in range(num_layers - 1, -1, -1):
        layer_used = set()
        used_bufs = {"header": all_used["header"], "footer": all_used["footer"],
                     "globals": all_used["globals"], "layer": layer_used}

        # Build backward bindings
        bindings = {}
        for key, spec in section["body"].get("bindings", {}).items():
            if isinstance(spec, dict):
                # Backward bindings (d_output comes from next layer or footer)
                if "last_layer" in spec and "prev_layer" in spec:
                    if layer_id == num_layers - 1:
                        bindings[key] = spec["last_layer"]
                    else:
                        bindings[key] = expand_layer_name(spec["prev_layer"], layer_id)
                # Forward bindings (for accessing cached activations)
                elif "first_layer" in spec and "next_layer" in spec:
                    if layer_id == 0:
                        bindings[key] = spec["first_layer"]
                    else:
                        bindings[key] = expand_layer_name(spec["next_layer"], layer_id)
            elif not isinstance(spec, dict):
                bindings[key] = spec

        layer_ops = process_ops(
            section["body"].get("backward_ops", []), layer_id, bindings, used_bufs,
            skip_registry_check=True
        )

        # Merge used buffers with forward pass for this layer
        fwd_layer = fwd_layers[layer_id]
        combined_used = fwd_layer["_used"] | layer_used

        bwd_layers.append({
            "id": layer_id,
            "ops": layer_ops,
            "_used": combined_used,
        })

    # Backward header ops (embedding_backward)
    bwd_header_ops = process_ops(
        section["header"].get("backward_ops", []), 0, {},
        {"header": all_used["header"], "footer": all_used["footer"],
         "globals": all_used["globals"], "layer": set()},
        skip_registry_check=True
    )

    # =========================================================================
    # GRADIENT REDUCTION (for data parallel training)
    # =========================================================================

    gradient_reduction = section.get("gradient_reduction", [])
    reduction_ops = process_ops(
        gradient_reduction, 0, {},
        {"header": all_used["header"], "footer": all_used["footer"],
         "globals": all_used["globals"], "layer": set()},
        skip_registry_check=True
    )

    # =========================================================================
    # OPTIMIZER PASS
    # =========================================================================

    # Optimizer header ops (token_emb update)
    opt_header_ops = process_ops(
        section["header"].get("optimizer_ops", []), 0, {},
        {"header": all_used["header"], "footer": all_used["footer"],
         "globals": all_used["globals"], "layer": set()},
        skip_registry_check=True
    )

    # Optimizer layer ops (per-layer weight updates)
    opt_layers = []
    for layer_id in range(num_layers):
        layer_used = set()
        used_bufs = {"header": all_used["header"], "footer": all_used["footer"],
                     "globals": all_used["globals"], "layer": layer_used}

        # No special bindings needed for optimizer
        bindings = {}

        layer_ops = process_ops(
            section["body"].get("optimizer_ops", []), layer_id, bindings, used_bufs,
            skip_registry_check=True
        )

        # Add optimizer buffer usage to layer
        fwd_layers[layer_id]["_used"] |= layer_used

        opt_layers.append({
            "id": layer_id,
            "ops": layer_ops,
        })

    # Optimizer footer ops (final_ln_weight, lm_head_weight updates)
    opt_footer_ops = process_ops(
        section["footer"].get("optimizer_ops", []), num_layers, footer_bindings,
        {"header": all_used["header"], "footer": all_used["footer"],
         "globals": all_used["globals"], "layer": set()},
        skip_registry_check=True
    )

    # =========================================================================
    # BUILD BUFFERS
    # =========================================================================

    # Build layer buffers (need both forward and backward buffers)
    final_layers = []
    for layer_id in range(num_layers):
        fwd_layer = fwd_layers[layer_id]
        bwd_layer = bwd_layers[num_layers - 1 - layer_id]  # Backward is reversed
        combined_used = fwd_layer["_used"] | bwd_layer["_used"]

        layer_buffers = []
        for tmpl in layer_templates:
            name = expand_layer_name(tmpl["name"], layer_id)
            if name in combined_used:
                spec = buffer_spec(name, mode_override="training")
                if spec:
                    layer_buffers.append(spec)

        final_layers.append({
            "id": layer_id,
            "forward_ops": fwd_layer["ops"],
            "backward_ops": bwd_layer["ops"],
            "buffers": layer_buffers,
        })

    # Header buffers
    header_buffers = []
    for tmpl in header_templates:
        name = tmpl["name"]
        if name in all_used["header"]:
            spec = buffer_spec(name, mode_override="training")
            if spec:
                header_buffers.append(spec)

    # Footer buffers
    footer_buffers = []
    for tmpl in footer_templates:
        name = tmpl["name"]
        if name in all_used["footer"]:
            spec = buffer_spec(name, mode_override="training")
            if spec:
                footer_buffers.append(spec)

    # Resolve tied_to references
    header_names = {b["name"] for b in header_buffers}
    for buf in footer_buffers:
        tied_to = buf.get("tied_to")
        if tied_to and tied_to not in header_names:
            spec = buffer_spec(tied_to, mode_override="training")
            if spec:
                header_buffers.insert(0, spec)
                header_names.add(tied_to)

    # Globals buffers
    globals_buffers = []
    for tmpl in globals_templates:
        name = tmpl["name"]
        if name in all_used["globals"]:
            spec = buffer_spec(name, mode_override="training")
            if spec:
                globals_buffers.append(spec)

    # Add training-specific inputs (labels - same shape as tokens, no batch dim)
    # Batch is simulated via accumulation loop, not tensor dimension
    training_inputs = list(inputs)
    labels_spec = buffer_spec("labels", mode_override="training")
    if labels_spec:
        training_inputs.append({
            "name": "labels",
            "dtype": "i32",
            "shape": ["S"],
            "resolved_shape": [sym_values["S"]],
        })

    # Compute batch simulation parameters
    effective_batch = sym_values.get("B", 1)
    context_length = sym_values.get("S", 128)
    tokens_per_batch = effective_batch * context_length

    return {
        "version": 4,
        "kind": "lowered",
        "mode": "training",
        "generated": datetime.utcnow().isoformat() + "Z",
        "model": graph["model"],
        "config": config,
        "symbols": symbols,
        # CPU batch simulation: no 3D tensor ops, use sequential accumulation
        "batch_simulation": {
            "strategy": "sequential_accumulate",
            "effective_batch_size": effective_batch,
            "accumulation_steps": effective_batch,
            "samples_per_step": 1,
            "context_length": context_length,
            "tokens_per_batch": tokens_per_batch,
            "description": (
                f"Simulate batch={effective_batch} by processing {effective_batch} samples "
                f"sequentially with same weights, accumulating gradients, then updating once. "
                f"Each sample has {context_length} tokens. "
                f"Mathematically equivalent to GPU parallel batch."
            ),
        },
        # Execution order shows the accumulation loop explicitly
        "execution_order": {
            "description": "Training step structure for CPU batch simulation",
            "phases": [
                {
                    "name": "zero_gradients",
                    "description": "Initialize all weight gradients to zero",
                },
                {
                    "name": "accumulation_loop",
                    "loop_var": "sample_idx",
                    "loop_count": effective_batch,
                    "description": f"Process {effective_batch} samples with frozen weights",
                    "body": [
                        "load_sample[sample_idx]",
                        "forward_pass.header",
                        "forward_pass.layers[0..N]",
                        "forward_pass.footer",
                        "backward_pass.footer",
                        "backward_pass.layers[N..0]",
                        "backward_pass.header",
                        "gradient_accumulate",  # d_weight += d_weight_sample
                    ],
                },
                {
                    "name": "gradient_average",
                    "description": f"Average gradients: d_weight /= {effective_batch}",
                },
                {
                    "name": "gradient_reduction",
                    "condition": "data_parallel_size > 1",
                    "description": "AllReduce gradients across data parallel workers",
                },
                {
                    "name": "optimizer_step",
                    "description": "Update weights once using averaged gradients",
                    "body": [
                        "optimizer_pass.header",
                        "optimizer_pass.layers[0..N]",
                        "optimizer_pass.footer",
                    ],
                },
            ],
        },
        "sections": [
            {
                "id": section["id"],
                "name": section["name"],
                "inputs": training_inputs,
                "globals": globals_buffers,
                "forward_pass": {
                    "header": {"ops": fwd_header_ops, "buffers": header_buffers},
                    "layers": [{"id": l["id"], "ops": l["forward_ops"]} for l in final_layers],
                    "footer": {"ops": fwd_footer_ops, "buffers": footer_buffers},
                },
                "backward_pass": {
                    "footer": {"ops": bwd_footer_ops},
                    "layers": [{"id": l["id"], "ops": l["backward_ops"]} for l in reversed(final_layers)],
                    "header": {"ops": bwd_header_ops},
                },
                "gradient_ops": {
                    "zero": {
                        "op": "zero_gradients",
                        "description": "Set all d_weight buffers to zero",
                        "targets": "all weight_grad buffers",
                    },
                    "accumulate": {
                        "op": "gradient_accumulate",
                        "description": "d_weight += d_weight_sample (after each backward)",
                    },
                    "average": {
                        "op": "gradient_average",
                        "description": f"d_weight /= {effective_batch} (before optimizer)",
                        "divisor": effective_batch,
                    },
                },
                "gradient_reduction": {
                    "ops": reduction_ops,
                    "condition": "data_parallel_size > 1",
                    "description": "AllReduce gradients across data parallel workers",
                },
                "optimizer_pass": {
                    "header": {"ops": opt_header_ops},
                    "layers": [{"id": l["id"], "ops": l["ops"]} for l in opt_layers],
                    "footer": {"ops": opt_footer_ops},
                },
                "buffers": {
                    "header": header_buffers,
                    "layers": [{"id": l["id"], "buffers": l["buffers"]} for l in final_layers],
                    "footer": footer_buffers,
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Fusion Optimization Pass
# ---------------------------------------------------------------------------

def find_fusion_candidates(ops: List[Dict], mode: str) -> List[Dict]:
    """
    Scan a list of ops for fusible sequences.
    Returns candidates sorted by priority (highest first).
    """
    patterns = fp.get_patterns_for_mode(mode)
    candidates = []

    for pattern in patterns:
        seq = pattern["sequence"]
        seq_len = len(seq)

        # Sliding window match
        for i in range(len(ops) - seq_len + 1):
            window = ops[i:i + seq_len]
            if fp.ops_match_sequence(window, seq):
                # Validate data flow
                if fp.validate_data_flow(window):
                    candidates.append({
                        "start_idx": i,
                        "end_idx": i + seq_len,
                        "pattern": pattern,
                        "matched_ops": window,
                    })

    return candidates


def apply_fusions_to_ops(ops: List[Dict], candidates: List[Dict], dtype: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply non-overlapping fusions to ops list.
    Returns (new_ops, applied_fusions).
    """
    if not candidates:
        return ops, []

    # Sort by priority (already sorted) then by start index
    # Take highest priority non-overlapping fusions
    applied = []
    used_indices = set()

    for cand in candidates:
        start, end = cand["start_idx"], cand["end_idx"]
        # Check for overlap with already applied fusions
        if any(i in used_indices for i in range(start, end)):
            continue

        # Mark indices as used
        for i in range(start, end):
            used_indices.add(i)
        applied.append(cand)

    if not applied:
        return ops, []

    # Build new ops list
    new_ops = []
    i = 0
    while i < len(ops):
        # Check if this index starts a fusion
        fusion = next((f for f in applied if f["start_idx"] == i), None)
        if fusion:
            # Create fused op
            fused_op = fp.merge_op_ios(
                fusion["matched_ops"],
                fusion["pattern"],
                dtype
            )
            new_ops.append(fused_op)
            i = fusion["end_idx"]
        else:
            new_ops.append(ops[i])
            i += 1

    return new_ops, applied


def filter_unused_buffers(buffers: List[Dict], ops: List[Dict], removed_patterns: List[str]) -> List[Dict]:
    """
    Remove buffers that are no longer used after fusion.
    """
    # Collect all buffer names still referenced by ops
    used_names = set()
    for op in ops:
        for key in ("inputs", "outputs", "weights", "scratch"):
            for name in op.get(key, []):
                used_names.add(name)

    # Keep buffers that are either:
    # 1. Still used by some op
    # 2. Not in the removed patterns list
    filtered = []
    for buf in buffers:
        name = buf["name"]
        is_removed = any(name.endswith(p) for p in removed_patterns)
        if name in used_names or not is_removed:
            filtered.append(buf)

    return filtered


def apply_fusion_pass(lowered: Dict, mode: str, config: Dict) -> Tuple[Dict, fp.FusionStats]:
    """
    Apply fusion optimizations to lowered IR.

    Args:
        lowered: Lowered IR dict
        mode: Execution mode (prefill/decode)
        config: Fusion configuration

    Returns:
        (optimized_ir, fusion_stats)
    """
    if not config.get("enable_fusion", True):
        return lowered, fp.FusionStats()

    stats = fp.FusionStats()
    optimized = copy.deepcopy(lowered)
    dtype = optimized["config"]["dtype"]

    # Collect all patterns' removed buffers
    all_removed_patterns = []
    for pattern in fp.FUSION_PATTERNS:
        all_removed_patterns.extend(pattern.get("remove_buffers", []))

    # Process each layer
    for layer in optimized["sections"][0]["layers"]:
        layer_id = layer["id"]
        ops = layer["ops"]

        # Find and apply fusions
        candidates = find_fusion_candidates(ops, mode)
        new_ops, applied = apply_fusions_to_ops(ops, candidates, dtype)

        if applied:
            layer["ops"] = new_ops

            # Track removed buffers for this layer
            removed_buffers = []
            for fusion in applied:
                pattern = fusion["pattern"]
                for suffix in pattern.get("remove_buffers", []):
                    # Find matching buffer names
                    for buf in layer["buffers"]:
                        if buf["name"].endswith(suffix):
                            removed_buffers.append(buf["name"])

            # Filter unused buffers
            layer["buffers"] = filter_unused_buffers(
                layer["buffers"],
                new_ops,
                all_removed_patterns
            )

            # Record stats
            for fusion in applied:
                pattern = fusion["pattern"]
                ops_count = len(fusion["matched_ops"])
                stats.record_fusion(layer_id, pattern, ops_count, removed_buffers)

    # Also check header/footer ops (less common but possible)
    section = optimized["sections"][0]
    for part in ["header", "footer"]:
        if part in section and "ops" in section[part]:
            ops = section[part]["ops"]
            candidates = find_fusion_candidates(ops, mode)
            new_ops, applied = apply_fusions_to_ops(ops, candidates, dtype)
            if applied:
                section[part]["ops"] = new_ops

    return optimized, stats


def emit_fusion_report(stats: fp.FusionStats, mode: str, path: str) -> None:
    """Emit fusion report JSON."""
    report = {
        "mode": mode,
        "generated": datetime.utcnow().isoformat() + "Z",
        **stats.to_dict(),
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[FUSION] Written: {path}")


# ---------------------------------------------------------------------------
# Layout from lowered IR
# ---------------------------------------------------------------------------

def build_layout_from_lowered(lowered: Dict, model_name: str) -> v3.ModelLayout:
    """Compute deterministic layout from lowered IR."""
    allocator = v3.BumpAllocator(start_offset=64)

    section = lowered["sections"][0]
    mode = lowered.get("mode", "prefill")
    # Keep per-buffer canaries to detect intra-layer overruns in tests and production.
    guard_buffers = bool(lowered.get("config", {}).get("guard_buffers", True))

    canaries: List[v3.Canary] = []

    header_canary_start = allocator.alloc_canary("header_start")
    canaries.append(header_canary_start)
    header_buffers = []
    name_to_buffer = {}

    def alloc_buffer(spec: Dict) -> v3.Buffer:
        name = spec["name"]
        tied_to = spec.get("tied_to")
        if tied_to:
            target = name_to_buffer.get(tied_to)
            if not target:
                raise KeyError(f"tied_to target not found: {tied_to}")
            buf = v3.Buffer(
                name=name,
                shape=spec["resolved_shape"],
                dtype=spec["dtype"],
                role=spec["role"],
                offset=target.offset,
                size=0,
                tied_to=tied_to,
            )
            name_to_buffer[name] = buf
            return buf

        if "file_size" in spec:
            size = align_up_bytes(int(spec["file_size"]), v3.CACHE_LINE)
        elif qt.is_quantized_dtype(spec["dtype"]):
            elements = 1
            for dim in spec["resolved_shape"]:
                elements *= int(dim)
            size = align_up_bytes(qt.calculate_quantized_size(spec["dtype"], elements),
                                  v3.CACHE_LINE)
        else:
            size = v3.aligned_size(spec["resolved_shape"], spec["dtype"], v3.CACHE_LINE)
        offset = allocator.alloc(name, size)
        buf = v3.Buffer(
            name=name,
            shape=spec["resolved_shape"],
            dtype=spec["dtype"],
            role=spec["role"],
            offset=offset,
            size=size,
            tied_to=spec.get("tied_to"),
        )
        name_to_buffer[name] = buf
        if guard_buffers:
            canary = allocator.alloc_canary(f"{name}_end")
            canaries.append(canary)
        return buf

    # Training mode has a different structure: buffers are under "buffers" key
    if mode == "training":
        buffers_section = section.get("buffers", {})
        header_buf_specs = buffers_section.get("header", [])
        layer_buf_specs = buffers_section.get("layers", [])
        footer_buf_specs = buffers_section.get("footer", [])
    else:
        header_buf_specs = section["header"]["buffers"]
        layer_buf_specs = section["layers"]
        footer_buf_specs = section["footer"]["buffers"]

    for spec in header_buf_specs:
        header_buffers.append(alloc_buffer(spec))
    header_canary_end = allocator.alloc_canary("header_end")
    canaries.append(header_canary_end)

    layers = []
    if mode == "training":
        # Training mode: layers are in buffers.layers with {id, buffers}
        for layer_entry in layer_buf_specs:
            layer_id = layer_entry["id"]
            canary_start = allocator.alloc_canary(f"layer_{layer_id}_start")
            canaries.append(canary_start)
            buffers = [alloc_buffer(spec) for spec in layer_entry.get("buffers", [])]
            canary_end = allocator.alloc_canary(f"layer_{layer_id}_end")
            canaries.append(canary_end)
            start_offset = canary_start.offset
            end_offset = allocator.offset
            layers.append(
                v3.LayerLayout(
                    layer_id=layer_id,
                    canary_start=canary_start,
                    buffers=buffers,
                    canary_end=canary_end,
                    total_bytes=end_offset - start_offset,
                )
            )
    else:
        for layer in layer_buf_specs:
            layer_id = layer["id"]
            canary_start = allocator.alloc_canary(f"layer_{layer_id}_start")
            canaries.append(canary_start)
            buffers = [alloc_buffer(spec) for spec in layer["buffers"]]
            canary_end = allocator.alloc_canary(f"layer_{layer_id}_end")
            canaries.append(canary_end)
            start_offset = canary_start.offset
            end_offset = allocator.offset
            layers.append(
                v3.LayerLayout(
                    layer_id=layer_id,
                    canary_start=canary_start,
                    buffers=buffers,
                    canary_end=canary_end,
                    total_bytes=end_offset - start_offset,
                )
            )

    footer_canary_start = allocator.alloc_canary("footer_start")
    canaries.append(footer_canary_start)
    footer_buffers = [alloc_buffer(spec) for spec in footer_buf_specs]
    footer_canary_end = allocator.alloc_canary("footer_end")
    canaries.append(footer_canary_end)

    globals_buffers = [alloc_buffer(spec) for spec in section.get("globals", [])]

    # Count totals
    weight_bytes = 0
    activation_bytes = 0
    def count_buffers(buffers: List[v3.Buffer]):
        nonlocal weight_bytes, activation_bytes
        for buf in buffers:
            if buf.tied_to:
                continue
            if buf.role == "weight":
                weight_bytes += buf.size
            else:
                activation_bytes += buf.size

    count_buffers(header_buffers)
    count_buffers(footer_buffers)
    count_buffers(globals_buffers)
    for layer in layers:
        count_buffers(layer.buffers)

    canary_count = len(canaries)

    section_layout = v3.SectionLayout(
        name=section["name"],
        section_id=section["id"],
        config=lowered["config"],
        header_canary_start=header_canary_start,
        header_buffers=header_buffers,
        header_canary_end=header_canary_end,
        layers=layers,
        footer_canary_start=footer_canary_start,
        footer_buffers=footer_buffers,
        footer_canary_end=footer_canary_end,
        globals=globals_buffers,
        total_bytes=allocator.offset,
    )

    return v3.ModelLayout(
        name=model_name,
        config=lowered["config"],
        sections=[section_layout],
        magic_header_size=64,
        total_bytes=allocator.offset,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        canary_count=canary_count,
        canaries=canaries,
    )


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def emit_lowered_ir(lowered: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(lowered, f, indent=2)
    print(f"[LOWERED] Written: {path}")


def expected_hf_shape(ck_name: str, config: Dict) -> Optional[List[int]]:
    E = config["embed_dim"]
    H = config["num_heads"]
    KV = config["num_kv_heads"]
    D = config["head_dim"]
    I = config["intermediate_dim"]
    V = config["vocab_size"]

    if ck_name in {"token_emb", "lm_head_weight"}:
        return [V, E]
    if ck_name.endswith("ln1_gamma") or ck_name.endswith("ln2_gamma") or ck_name == "final_ln_weight":
        return [E]
    if ck_name.endswith(".wq"):
        return [H * D, E]
    if ck_name.endswith(".wk") or ck_name.endswith(".wv"):
        return [KV * D, E]
    if ck_name.endswith(".wo"):
        return [E, H * D]
    if ck_name.endswith(".w1"):
        return [2 * I, E]
    if ck_name.endswith(".w2"):
        return [E, I]
    return None


def build_weight_map_report(graph: Dict, weights_meta: Dict) -> Dict:
    config = graph["config"]
    num_layers = config["num_layers"]

    header = weights_meta.get("header", {})
    weight_names = set(extract_weight_names(weights_meta))

    # Build a map of buffer specs for target shape lookup
    buffer_specs = {}
    section = graph["sections"][0]
    for group in ("header", "layer", "footer"):
        for buf in section["buffers"][group]:
            buffer_specs[buf["name"]] = buf
    for buf in section.get("globals", []):
        buffer_specs[buf["name"]] = buf

    entries = []
    missing = []
    unmapped = []
    seen = set()

    weight_map = graph.get("weight_map", [])

    def lookup_buffer_spec(name: str) -> Optional[Dict]:
        if name in buffer_specs:
            return buffer_specs[name]
        if name.startswith("layer."):
            tmpl = normalize_layer_template(name)
            return buffer_specs.get(tmpl)
        return None

    def add_entry(hf_name: str, ck_name: str, meta: Dict) -> None:
        entry = {
            "hf_name": hf_name,
            "ck_name": ck_name,
            "optional": meta.get("optional", False),
        }
        if meta.get("pack"):
            entry["pack"] = {
                "type": meta["pack"],
                "axis": meta.get("axis", 0),
                "part": meta.get("part", ""),
            }

        spec = lookup_buffer_spec(ck_name)
        if spec:
            entry["target_shape"] = spec["resolved_shape"]

        expected = expected_hf_shape(ck_name, config)
        if expected:
            entry["expected_hf_shape"] = expected

        if hf_name in weight_names:
            seen.add(hf_name)
            entry["status"] = "ok"
            if hf_name in header:
                entry["hf_shape"] = header[hf_name].get("shape")
                if expected and not meta.get("pack"):
                    if entry["hf_shape"] == expected:
                        entry["shape_ok"] = True
                    elif entry["hf_shape"] == list(reversed(expected)):
                        entry["shape_ok"] = True
                        entry["transpose"] = True
                    else:
                        entry["shape_ok"] = False
        else:
            entry["status"] = "missing"
            if not entry["optional"]:
                missing.append(hf_name)
        entries.append(entry)

    for mapping in weight_map:
        hf = mapping["hf"]
        ck = mapping["ck"]
        if "{layer}" in hf:
            for layer_id in range(num_layers):
                hf_name = hf.replace("{layer}", str(layer_id))
                ck_name = ck.replace("{L}", str(layer_id))
                add_entry(hf_name, ck_name, mapping)
        else:
            add_entry(hf, ck, mapping)

    for name in weight_names:
        if name not in seen:
            unmapped.append(name)

    return {
        "model": graph["model"],
        "generated": datetime.utcnow().isoformat() + "Z",
        "missing": missing,
        "unmapped": unmapped,
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> Dict:
    def normalize_args(argv: List[str]) -> List[str]:
        value_flags = {
            "--config",
            "--name",
            "--prefix",
            "--weights-header",
            "--weights-index",
            "--weights-manifest",
            "--kernel-specs",
            "--emit",
            "--tokens",
            "--dtype",
            "--weight-dtype",
            "--modes",
            "--preset",
            "--fusion",
            "--parallel",
            "--memory",
            "--batch-size",
            "--micro-batch-size",
            "--context-length",
            "--max-layers",
            "--optimizer",
            "--learning-rate",
            "--weight-decay",
            "--data-parallel",
            "--tensor-parallel",
            "--codegen",
        }
        normalized: List[str] = []
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg in value_flags:
                if i + 1 >= len(argv):
                    raise ValueError(f"Expected value after {arg}")
                value = argv[i + 1]
                if value.startswith("--"):
                    raise ValueError(f"Expected value after {arg}, got: {value}")
                normalized.append(f"{arg}={value}")
                i += 2
                continue
            normalized.append(arg)
            i += 1
        return normalized

    argv = normalize_args(argv)
    args = {
        "model": None,
        "config": None,
        "name": None,
        "prefix": None,
        "weights_header": None,
        "weights_index": None,
        "weights_manifest": None,
        "tokens": None,
        "dtype": None,
        "weight_dtype": None,  # Quantized weight dtype (q4_k, q6_k, etc.)
        "modes": ["prefill", "decode"],
        "kernel_specs": None,
        "preset": None,
        "emit": "exe",
        "max_layers": None,
        # Fusion options
        "fusion": "auto",  # on/off/auto
        "fusion_verbose": False,
        # Parallel planning options
        "parallel": "on",  # on/off
        "parallel_verbose": False,
        # Debug options
        "debug": False,  # Emit debug prints in generated C code
        "parity": False,  # Emit buffer saves for parity comparison with PyTorch
        "codegen": "v5",  # Codegen version: v5 (explicit unrolled, default) or v4 (loop-based)
        # Training options
        "memory": None,  # Available memory in GB (auto-detect if None)
        "batch_size": None,  # Target batch size
        "micro_batch_size": None,  # Micro-batch size for gradient accumulation
        "context_length": None,  # Context length for training
        "optimizer": "adamw",  # adamw or sgd
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "data_parallel": 1,  # Data parallel size
        "tensor_parallel": 1,  # Tensor parallel size
    }

    for arg in argv:
        if arg in ("--help", "-h"):
            args["help"] = True
            continue
        if arg.startswith("--config="):
            args["config"] = arg.split("=", 1)[1]
        elif arg.startswith("--name="):
            args["name"] = arg.split("=", 1)[1]
        elif arg.startswith("--prefix="):
            args["prefix"] = arg.split("=", 1)[1]
        elif arg.startswith("--weights-header="):
            args["weights_header"] = arg.split("=", 1)[1]
        elif arg.startswith("--weights-index="):
            args["weights_index"] = arg.split("=", 1)[1]
        elif arg.startswith("--weights-manifest="):
            args["weights_manifest"] = arg.split("=", 1)[1]
        elif arg.startswith("--kernel-specs="):
            args["kernel_specs"] = arg.split("=", 1)[1]
        elif arg == "--emit-lib":
            args["emit"] = "lib"
        elif arg == "--emit-exe":
            args["emit"] = "exe"
        elif arg.startswith("--emit="):
            emit_val = arg.split("=", 1)[1].lower()
            if emit_val not in ("lib", "exe"):
                raise ValueError(f"--emit must be lib|exe, got: {emit_val}")
            args["emit"] = emit_val
        elif arg.startswith("--tokens="):
            args["tokens"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--dtype="):
            args["dtype"] = arg.split("=", 1)[1].lower()
        elif arg.startswith("--weight-dtype="):
            w_dtype = arg.split("=", 1)[1].lower()
            # Normalize aliases
            if w_dtype not in ("q4_0", "q4_k", "q4_k_m", "q6_k", "q8_0", "q8_k", "f32", "bf16"):
                raise ValueError(f"--weight-dtype must be q4_0/q4_k/q4_k_m/q6_k/q8_0/q8_k/f32/bf16, got: {w_dtype}")
            args["weight_dtype"] = w_dtype
        elif arg.startswith("--modes="):
            modes = arg.split("=", 1)[1]
            args["modes"] = [m.strip() for m in modes.split(",") if m.strip()]
        elif arg.startswith("--max-layers="):
            args["max_layers"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--preset="):
            args["preset"] = arg.split("=", 1)[1]
        elif arg.startswith("--fusion="):
            fusion_val = arg.split("=", 1)[1].lower()
            if fusion_val not in ("on", "off", "auto"):
                raise ValueError(f"--fusion must be on/off/auto, got: {fusion_val}")
            args["fusion"] = fusion_val
        elif arg == "--fusion-verbose":
            args["fusion_verbose"] = True
        elif arg.startswith("--parallel="):
            parallel_val = arg.split("=", 1)[1].lower()
            if parallel_val not in ("on", "off"):
                raise ValueError(f"--parallel must be on/off, got: {parallel_val}")
            args["parallel"] = parallel_val
        elif arg == "--parallel-verbose":
            args["parallel_verbose"] = True
        elif arg == "--debug":
            args["debug"] = True
        elif arg == "--parity":
            args["parity"] = True
        elif arg.startswith("--codegen="):
            codegen_val = arg.split("=", 1)[1].lower()
            if codegen_val not in ("v4", "v5"):
                raise ValueError(f"--codegen must be v4/v5, got: {codegen_val}")
            args["codegen"] = codegen_val
        # Training options
        elif arg.startswith("--memory="):
            args["memory"] = float(arg.split("=", 1)[1])
        elif arg.startswith("--batch-size="):
            args["batch_size"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--micro-batch-size="):
            args["micro_batch_size"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--context-length="):
            args["context_length"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--optimizer="):
            opt = arg.split("=", 1)[1].lower()
            if opt not in ("adamw", "sgd"):
                raise ValueError(f"--optimizer must be adamw/sgd, got: {opt}")
            args["optimizer"] = opt
        elif arg.startswith("--learning-rate="):
            args["learning_rate"] = float(arg.split("=", 1)[1])
        elif arg.startswith("--weight-decay="):
            args["weight_decay"] = float(arg.split("=", 1)[1])
        elif arg.startswith("--data-parallel="):
            args["data_parallel"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--tensor-parallel="):
            args["tensor_parallel"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--"):
            raise ValueError(f"Unknown option: {arg}")
        else:
            args["model"] = arg

    if not args.get("help") and not args["model"] and not args["config"] and not args["preset"]:
        raise ValueError("Must provide model ID/URL or --config=FILE or --preset=NAME")

    return args


def print_usage():
    print("Usage:")
    print("  python scripts/build_ir_v5.py MODEL [OPTIONS]")
    print("  python scripts/build_ir_v5.py --config=FILE [OPTIONS]")
    print("  python scripts/build_ir_v5.py --preset=NAME [OPTIONS]")
    print()
    print("Options:")
    print("  --config=FILE           Use local config.json")
    print("  --preset=NAME           Use local preset (qwen2-0.5b, smollm-135)")
    print("  --weights-header=FILE   Safetensors header for weight mapping")
    print("  --weights-index=FILE    model.safetensors.index.json")
    print("  --weights-manifest=FILE Weights manifest (from convert_*_to_bump.py)")
    print("  --kernel-specs=FILE     Kernel registry JSON (optional)")
    print("  --prefix=DIR            Output directory")
    print("  --tokens=N              Tokens for prefill/backward (default: max_seq_len)")
    print("  --dtype=fp32|bf16       Override dtype for activations (default: config dtype)")
    print("  --weight-dtype=TYPE     Weight dtype for quantized inference (q4_k, q6_k, etc.)")
    print("  --modes=MODE[,MODE...]  Modes to emit (default: prefill,decode)")
    print("  --emit=lib|exe          Emit shared-library C (lib) or standalone main (exe)")
    print("  --emit-lib              Shorthand for --emit=lib")
    print("  --emit-exe              Shorthand for --emit=exe")
    print("  --max-layers=N          Limit layers for quick parity tests")
    print()
    print("Available Modes:")
    print("  prefill                 Forward pass for prompt processing (S=tokens)")
    print("  decode                  Forward pass for token generation (S=1)")
    print("  backward                Backward pass only (assumes activations cached)")
    print("  training                Full training step: forward + loss + backward")
    print()
    print("Fusion Options:")
    print("  --fusion=on|off|auto    Enable/disable fusion pass (default: auto)")
    print("  --fusion-verbose        Print fusion decisions")
    print()
    print("Parallel Options:")
    print("  --parallel=on|off       Enable/disable parallel planning (default: on)")
    print("  --parallel-verbose      Print parallel strategy decisions")
    print()
    print("Codegen Options:")
    print("  --codegen=v4|v5         Codegen version (default: v5)")
    print("                          v5: Explicit unrolled (each layer separate, explicit kernels)")
    print("                          v4: Loop-based with runtime dtype dispatch (legacy)")
    print("  --debug                 Emit debug prints in generated C code")
    print("  --parity                Emit buffer saves for PyTorch comparison")
    print()
    print("Training Options (for --modes=training):")
    print("  --memory=GB             Available memory in GB (auto-detect if not set)")
    print("  --batch-size=N          Target effective batch size")
    print("  --micro-batch-size=N    Micro-batch size (gradient accumulation)")
    print("  --context-length=N      Training context length")
    print("  --optimizer=adamw|sgd   Optimizer (default: adamw)")
    print("  --learning-rate=LR      Learning rate (default: 1e-4)")
    print("  --weight-decay=WD       Weight decay for AdamW (default: 0.01)")
    print("  --data-parallel=N       Data parallel size (default: 1)")
    print("  --tensor-parallel=N     Tensor parallel size (default: 1)")
    print()
    print("Quantization Options (llama.cpp compatible):")
    print("  --weight-dtype=q4_k     Q4_K: 4-bit K-quant (4.5 bits/weight)")
    print("  --weight-dtype=q4_k_m   Mixed GGUF; uses per-weight manifest dtypes")
    print("  --weight-dtype=q6_k     Q6_K: 6-bit K-quant (6.5 bits/weight)")
    print("  --weight-dtype=q4_0     Q4_0: Simple 4-bit (4.5 bits/weight)")
    print("  --weight-dtype=q8_0     Q8_0: Simple 8-bit (8.5 bits/weight)")
    print()
    print("Notes:")
    print("  Quantized inference uses --weight-dtype for weights, --dtype for activations.")
    print("  q4_k_m requires --weights-manifest and does not force global K-alignment.")
    print("  Block structures match llama.cpp/GGML for GGUF model compatibility.")
    print("  Training mode auto-detects system memory and computes optimal config.")
    print()
    print("Examples:")
    print("  # Generate prefill and decode schedules")
    print("  python scripts/build_ir_v5.py --preset=qwen2-0.5b")
    print()
    print("  # Generate training schedule (auto-detect memory, compute optimal batch)")
    print("  python scripts/build_ir_v5.py --preset=qwen2-0.5b --modes=training")
    print()
    print("  # Training with specific memory budget and batch size")
    print("  python scripts/build_ir_v5.py --preset=qwen2-0.5b --modes=training --memory=16 --batch-size=32")
    print()
    print("  # Generate backward only (for memory-efficient training)")
    print("  python scripts/build_ir_v5.py --preset=qwen2-0.5b --modes=backward --fusion=on")
    print()
    print("  # Use custom config file")
    print("  python scripts/build_ir_v5.py --config=smolLM-135.json --modes=prefill,decode")
    print()
    print("  # Quantized inference with Q4_K weights (llama.cpp compatible)")
    print("  python scripts/build_ir_v5.py --preset=qwen2-0.5b --weight-dtype=q4_k --dtype=f32")
    print()
    print("  # Verbose output for debugging")
    print("  python scripts/build_ir_v5.py --preset=qwen2-0.5b --fusion-verbose --parallel-verbose")


def main(argv: List[str]) -> int:
    try:
        args = parse_args(argv)
    except ValueError as e:
        print(f"Error: {e}")
        print_usage()
        return 1

    if args.get("help"):
        print_usage()
        return 0

    if args.get("preset"):
        preset = PRESETS.get(args["preset"])
        if not preset:
            print(f"Error: Unknown preset '{args['preset']}'")
            print_usage()
            return 1
        if not args["config"]:
            args["config"] = preset["config"]
        if not args["name"]:
            args["name"] = preset["name"]

    if args["config"]:
        config_path = args["config"]
        if not os.path.exists(config_path):
            preset = PRESETS.get(args.get("preset") or "")
            hf_id = preset.get("hf") if preset else None
            if hf_id:
                print(f"[CONFIG] Local preset missing, fetching HF config: {hf_id}")
                _, cached_path = v3.download_hf_config(hf_id)
                config_path = cached_path
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"[CONFIG] Reading local: {config_path}")
        config = v3.parse_config(config_path)
        model_name = args["name"] or config.get("model_type", "model")
    else:
        model_id = v3.parse_hf_model_id(args["model"])
        raw_config, cached_path = v3.download_hf_config(model_id)
        config = v3.parse_config(cached_path)
        model_name = args["name"] or v3.model_id_to_name(model_id)

    if args["dtype"]:
        dtype = args["dtype"]
        if dtype == "f32":
            dtype = "fp32"
        elif dtype == "f16":
            dtype = "fp16"
        if dtype not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"--dtype must be fp32|bf16|fp16, got: {dtype}")
        config["dtype"] = dtype

    if args.get("max_layers") is not None:
        max_layers = int(args["max_layers"])
        if max_layers <= 0:
            raise ValueError(f"--max-layers must be >= 1 (got {max_layers})")
        if max_layers < config["num_layers"]:
            print(f"[CONFIG] Limiting layers: {max_layers}/{config['num_layers']}")
            config["num_layers"] = max_layers
        elif max_layers > config["num_layers"]:
            print(f"[CONFIG] max-layers={max_layers} > num_layers={config['num_layers']} (using {config['num_layers']})")

    if args["tokens"]:
        tokens = args["tokens"]
    else:
        tokens = config["max_seq_len"]

    if args["prefix"]:
        output_dir = args["prefix"]
    else:
        safe_name = model_name.replace("-", "_").replace(".", "_")
        output_dir = os.path.join("build", f"{safe_name}_v5")

    print(f"[MODEL]  {model_name}")
    print(f"[OUTPUT] {output_dir}/")

    # Kernel registry
    registry = load_kernel_registry(args.get("kernel_specs"))
    if registry:
        print(f"[KERNELS] Loaded {len(registry)} kernel specs")
    else:
        print("[KERNELS] Warning: no kernel registry found; kernel validation disabled")

    # Weights metadata
    weights_meta = {}
    if args["weights_header"]:
        print(f"[WEIGHTS] Reading safetensors header: {args['weights_header']}")
        weights_meta["header"] = read_safetensors_header(args["weights_header"])
    if args["weights_index"]:
        print(f"[WEIGHTS] Reading index: {args['weights_index']}")
        weights_meta["index"] = read_weights_index(args["weights_index"])
    weights_manifest = None
    if args.get("weights_manifest"):
        print(f"[WEIGHTS] Reading manifest: {args['weights_manifest']}")
        with open(args["weights_manifest"], "r") as f:
            weights_manifest = json.load(f)
    manifest_weight_dtype = None
    if weights_manifest and isinstance(weights_manifest, dict):
        dtype_set = {
            entry.get("dtype")
            for entry in weights_manifest.get("entries", [])
            if entry.get("dtype")
        }
        non_fp = {
            dtype for dtype in dtype_set
            if dtype not in ("fp32", "f32", "bf16", "fp16")
        }
        if len(non_fp) == 1:
            manifest_weight_dtype = next(iter(non_fp))
        elif len(dtype_set) == 1:
            manifest_weight_dtype = next(iter(dtype_set))

    if args.get("weight_dtype") == "q4_k_m":
        if weights_manifest:
            print("[WEIGHTS] q4_k_m is mixed; using manifest dtypes for weights")
            args["weight_dtype"] = None
        else:
            raise ValueError("--weight-dtype=q4_k_m requires --weights-manifest")

    if args.get("weight_dtype"):
        config["weight_dtype"] = args["weight_dtype"]
    elif manifest_weight_dtype:
        config["weight_dtype"] = manifest_weight_dtype

    # Graph IR
    graph = build_graph_ir_v5(config, model_name)
    if weights_meta:
        graph["weights"] = weights_meta

    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, "graph.json")
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"[GRAPH] Written: {graph_path}")

    if weights_meta:
        weights_report = build_weight_map_report(graph, weights_meta)
        weights_path = os.path.join(output_dir, "weights_map.json")
        with open(weights_path, "w") as f:
            json.dump(weights_report, f, indent=2)
        print(f"[WEIGHTS] Written: {weights_path}")

    def emit_weights_manifest(layout: v3.ModelLayout, manifest: Dict, out_dir: str) -> None:
        entries_in = {e["name"]: e for e in manifest.get("entries", [])}
        missing = []
        merged = []

        section = layout.sections[0]
        buffers = []
        buffers.extend(section.header_buffers)
        for layer in section.layers:
            buffers.extend(layer.buffers)
        buffers.extend(section.footer_buffers)

        for buf in buffers:
            if buf.role != "weight":
                continue
            if buf.tied_to:
                continue
            entry = entries_in.get(buf.name)
            if not entry:
                missing.append(buf.name)
                continue
            merged.append(
                {
                    "name": buf.name,
                    "dtype": entry.get("dtype", buf.dtype),
                    "file_offset": entry.get("file_offset", 0),
                    "size": entry.get("size", 0),
                    "runtime_offset": buf.offset,
                }
            )

        manifest_out = {
            "format": "ck-bumpwgt4-merged-v1",
            "generated": datetime.utcnow().isoformat() + "Z",
            "model": layout.name,
            "has_attention_biases": manifest.get("has_attention_biases", False),
            "missing": missing,
            "entries": merged,
        }

        json_path = os.path.join(out_dir, "weights_manifest.json")
        with open(json_path, "w") as f:
            json.dump(manifest_out, f, indent=2)
        print(f"[WEIGHTS] Written: {json_path}")

        map_path = os.path.join(out_dir, "weights_manifest.map")
        with open(map_path, "w") as f:
            f.write("# ck-bumpwgt4-manifest-map v1\n")
            f.write("# name|dtype|file_offset|size|runtime_offset\n")
            for e in merged:
                f.write(
                    f"{e['name']}|{e['dtype']}|0x{e['file_offset']:016X}|0x{e['size']:016X}|0x{e['runtime_offset']:016X}\n"
                )
        print(f"[WEIGHTS] Written: {map_path}")

    # Fusion configuration
    fusion_mode = args.get("fusion", "auto")
    fusion_verbose = args.get("fusion_verbose", False)

    # Parallel planning configuration
    parallel_enabled = args.get("parallel", "on") == "on"
    parallel_verbose = args.get("parallel_verbose", False)

    # Determine if fusion is enabled
    # auto: enable for decode mode only (highest benefit)
    # on: enable for all modes
    # off: disable fusion
    def should_fuse(mode: str) -> bool:
        if fusion_mode == "off":
            return False
        if fusion_mode == "on":
            return True
        # auto: fuse for decode (best benefit), skip for prefill
        return mode == "decode"

    # Training configuration (computed once if training mode requested)
    training_cfg = None
    if "training" in args["modes"]:
        # Build training config from model
        training_cfg = tc.TrainingConfig.from_model_config(config)
        training_cfg.optimizer = args["optimizer"]
        training_cfg.learning_rate = args["learning_rate"]
        training_cfg.weight_decay = args["weight_decay"]
        training_cfg.data_parallel_size = args["data_parallel"]
        training_cfg.tensor_parallel_size = args["tensor_parallel"]

        # Get available memory
        if args["memory"]:
            available_memory = int(args["memory"] * 1024**3)
            print(f"[TRAINING] Using specified memory: {args['memory']:.1f} GB")
        else:
            sys_mem = tc.get_system_memory()
            available_memory = sys_mem["ram_bytes"]
            print(f"[TRAINING] Auto-detected RAM: {available_memory / 1024**3:.1f} GB")
            if sys_mem["gpu_vram_bytes"] > 0:
                print(f"[TRAINING] Detected {sys_mem['gpu_count']} GPU(s): "
                      f"{sys_mem['gpu_vram_bytes'] / 1024**3:.1f} GB VRAM")

        # Compute optimal configuration
        breakdown, recommendations = tc.find_optimal_config(
            training_cfg,
            available_memory,
            target_batch_size=args["batch_size"],
            target_context_length=args["context_length"] or args["tokens"],
            optimizer=args["optimizer"],
        )

        if breakdown is None:
            print(f"[TRAINING] ERROR: {recommendations.get('error', 'Unknown error')}")
            return 1

        # Store computed values
        training_cfg.batch_size = breakdown.batch_size
        training_cfg.micro_batch_size = breakdown.micro_batch_size
        training_cfg.context_length = breakdown.context_length
        training_cfg.memory_breakdown = breakdown

        # Print summary
        tc.print_memory_summary(breakdown, recommendations)

        # Compute reduction strategy
        reduction = tc.compute_reduction_strategy(
            training_cfg, args["data_parallel"], args["tensor_parallel"]
        )

        # Emit training config
        training_config_path = os.path.join(output_dir, "training_config.json")
        tc.emit_training_config(training_cfg, breakdown, recommendations, reduction, training_config_path)

    # Lower + layout per mode
    for mode in args["modes"]:
        mode_tokens = 1 if mode == "decode" else tokens

        # For training mode, use computed context length
        if mode == "training" and training_cfg:
            mode_tokens = training_cfg.context_length

        lowered = lower_graph_ir(graph, mode, mode_tokens, registry, training_cfg,
                                 weights_manifest=weights_manifest,
                                 weight_dtype=args.get("weight_dtype"))

        def emit_mode_outputs(enable_fusion: bool) -> None:
            fusion_config = {"enable_fusion": enable_fusion}
            optimized, fusion_stats = apply_fusion_pass(lowered, mode, fusion_config)

            if fusion_stats.fusions_applied:
                print(f"[FUSION] {mode}: {len(fusion_stats.fusions_applied)} fusions applied, "
                      f"{fusion_stats.ops_removed} ops removed, "
                      f"{fusion_stats.buffers_removed} buffers removed")
                if fusion_verbose:
                    for f in fusion_stats.fusions_applied:
                        print(f"  Layer {f['layer']}: {f['pattern']} ({f['ops_fused']} ops)")

                fusion_report_path = os.path.join(output_dir, f"fusion_{mode}.json")
                emit_fusion_report(fusion_stats, mode, fusion_report_path)
            elif enable_fusion:
                print(f"[FUSION] {mode}: no fusion patterns matched")

            # Apply parallel planning pass
            if parallel_enabled:
                optimized, parallel_stats = pp.apply_parallel_planning(optimized, mode)
                parallelized = parallel_stats["parallelized_ops"]
                total = parallel_stats["total_ops"]
                strategies = parallel_stats["strategies"]

                print(f"[PARALLEL] {mode}: {parallelized}/{total} ops parallelized")
                if parallel_verbose and strategies:
                    for strat, count in sorted(strategies.items()):
                        print(f"  {strat}: {count} ops")

                parallel_report_path = os.path.join(output_dir, f"parallel_{mode}.json")
                pp.emit_parallel_report(parallel_stats, mode, parallel_report_path)

            lowered_path = os.path.join(output_dir, f"lowered_{mode}.json")
            emit_lowered_ir(optimized, lowered_path)

            layout_name = f"{model_name}_{mode}"
            layout = build_layout_from_lowered(optimized, layout_name)

            layout_json_path = os.path.join(output_dir, f"layout_{mode}.json")
            layout_map = os.path.join(output_dir, f"layout_{mode}.map")

            v3.emit_layout_json(layout, layout_json_path)
            v3.emit_layout_map(layout, layout_map)

            if weights_manifest and mode in ("prefill", "decode"):
                emit_weights_manifest(layout, weights_manifest, output_dir)

            if parallel_enabled:
                with open(layout_json_path, "r") as f:
                    layout_dict = json.load(f)

                layout_with_parallel = pp.annotate_layout_buffers(
                    layout_dict, config, mode
                )

                schedule_path = os.path.join(output_dir, f"schedule_{mode}.json")
                schedule = {
                    "version": 4,
                    "kind": "schedule",
                    "mode": mode,
                    "generated": datetime.utcnow().isoformat() + "Z",
                    "model": model_name,
                    "layout": layout_with_parallel,
                    "ops": [],
                }

                section = optimized["sections"][0]
                for layer in section.get("layers", []):
                    for op in layer.get("ops", []):
                        schedule["ops"].append({
                            "layer": layer["id"],
                            "op": op.get("op"),
                            "kernel": op.get("kernel"),
                            "parallel": op.get("parallel", {}),
                        })

                with open(schedule_path, "w") as f:
                    json.dump(schedule, f, indent=2)
                print(f"[SCHEDULE] Written: {schedule_path}")

            safe_name = layout_name.replace("-", "_").replace(".", "_")
            safe_name_upper = safe_name.upper()
            header_name = f"generated_{safe_name}.h"
            extra_api = None
            if mode == "decode":
                extra_api = [
                    f"void {safe_name.lower()}_decode({safe_name_upper}Model *model, const int *token, int token_index);"
                ]
            v3.emit_c_header(layout, os.path.join(output_dir, header_name), extra_api=extra_api)
            if mode in ("prefill", "decode"):
                if config["dtype"] != "fp32":
                    print("[WARN] v4/v5 codegen currently emits fp32 activations only. Use --dtype=fp32 for runnable C.")
                codegen_version = args.get("codegen", "v5")
                if codegen_version == "v5":
                    print(f"[CODEGEN] Using v5 (explicit unrolled) for {mode}")
                    codegen_v5.emit_c_source_v5(
                        layout,
                        os.path.join(output_dir, f"generated_{safe_name}.c"),
                        header_name,
                        mode,
                        emit_main=(args.get("emit") == "exe"),
                        emit_debug=args.get("debug", False),
                        emit_parity=args.get("parity", False),
                        weights_manifest=weights_manifest,
                    )
                else:
                    codegen_v4.emit_c_source_v4(
                        layout,
                        os.path.join(output_dir, f"generated_{safe_name}.c"),
                        header_name,
                        mode,
                        emit_main=(args.get("emit") == "exe"),
                        emit_debug=args.get("debug", False),
                        emit_parity=args.get("parity", False),
                    )
            else:
                v3.emit_c_source(
                    layout,
                    os.path.join(output_dir, f"generated_{safe_name}.c"),
                    header_name,
                    emit_main=(args.get("emit") == "exe"),
                )

        try:
            emit_mode_outputs(should_fuse(mode))
        except ValueError as e:
            if mode in ("prefill", "decode") and "needs unfused buffers" in str(e):
                print(f"[FUSION] {mode}: codegen needs unfused buffers; regenerating with --fusion=off")
                emit_mode_outputs(False)
            else:
                raise

    print("[DONE] IR v5 pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
