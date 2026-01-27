#!/usr/bin/env python3
"""
build_ir_v6_6.py - IR v6.6 pipeline (standalone, manifest-first, registry-driven)

config.json + weights_manifest.json + KERNEL_REGISTRY.json -> graph IR -> lowered IR -> layout -> generated C

v6.6 improvements over v6:
  - INT8 activations enabled by default (5-15x speedup over FP32)
  - Uses gemv_q5_0_q8_0, gemv_q8_0_q8_0 for Q5_0/Q8_0 weights
  - Uses gemv_q4_k_q8_k for Q4_K weights
  - Proper mixed-quant support via per-tensor dtypes from manifest
  - Consistent v6.6 naming throughout
  - REGISTRY-DRIVEN: All kernel selection from KERNEL_REGISTRY.json (mandatory)

REQUIREMENTS:
  1. weights_manifest.json (from convert_*_to_bump_v6_6.py)
  2. KERNEL_REGISTRY.json (from gen_kernel_registry_from_maps.py)
     Run: python version/v6.6/scripts/gen_kernel_registry_from_maps.py

USAGE:
  python build_ir_v6_6.py --weights-manifest=weights_manifest.json --modes=decode

For FP32 activations (slower but precise):
  python codegen_v6_6.py --activations=fp32 ...
"""

import copy
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# v6.6 imports - compat layer handles legacy v3/v4 functions
# Note: ir_core_v6_6.py imports from ir_types_v6_6.py for v6 types
import ir_core_v6_6 as v3
import v6_6_ir_lowering as v6_low  # v6.6 IR lowering with per-layer buffers
import codegen_v6_6 as codegen_v6
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
# Alignment helpers
# ---------------------------------------------------------------------------

QK_K = 256

def align_up_bytes(n: int, alignment: int) -> int:
    return (n + alignment - 1) & ~(alignment - 1)

def align_up_elems(elems: int, elem_bytes: int, alignment: int) -> int:
    return align_up_bytes(elems * elem_bytes, alignment) // elem_bytes


# ---------------------------------------------------------------------------
# Graph IR v6 (kernel-aligned, manifest-first)
# ---------------------------------------------------------------------------
def map_template_op_to_internal(template_op: str, layer_id: int, op_index: int,
                                prev_outputs: List[str],
                                rmsnorm_count: int = 0, residual_count: int = 0,
                                template_flags: Optional[Dict] = None) -> Tuple[Optional[Dict], int, int]:
    """
    Map a template op to internal IR op format.

    Args:
        template_op: The template op name (e.g., "rmsnorm", "qkv_proj")
        layer_id: The layer index
        op_index: Position of this op in the sequence (0-based)
        prev_outputs: List of output tensor names from previous ops in the sequence
        rmsnorm_count: Number of rmsnorm ops already seen in this layer
        residual_count: Number of residual_add ops already seen in this layer
        template_flags: Template flags (use_qkv_bias, etc.)

    Returns:
        Tuple of (mapped_op_dict, updated_rmsnorm_count, updated_residual_count)
        Returns (None, rmsnorm_count, residual_count) if op should be skipped
    """
    template_flags = template_flags or {}
    has_bias = template_flags.get("use_qkv_bias") == True

    # Mapping from template ops to internal ops
    # Base mappings don't include layer-specific details
    op_mappings = {
        "rmsnorm": {
            "op": "rmsnorm",
            "name": "ln{ln_num}",
            "inputs": ["input"],
            "weights": ["layer.{L}.ln{ln_num}_gamma"],
            "outputs": ["layer.{L}.ln{ln_num}_out"],
        },
        "qkv_proj": {
            "op": "qkv_project",
            "name": "qkv_project",
            "inputs": ["layer.{L}.ln{ln_num}_out"],
            "weights": ["layer.{L}.wq", "layer.{L}.wk", "layer.{L}.wv"],
            "outputs": ["layer.{L}.q", "layer.{L}.k", "layer.{L}.v"],
        },
        "rope_qk": {
            "op": "rope",
            "name": "rope",
            "inputs": ["layer.{L}.q", "layer.{L}.k", "rope_cos_cache", "rope_sin_cache"],
            "outputs": ["layer.{L}.q", "layer.{L}.k"],
        },
        "attn": {
            "op": "attention",
            "name": "attention",
            "inputs": ["layer.{L}.q", "layer.{L}.k", "layer.{L}.v"],
            "outputs": ["layer.{L}.attn_out"],
            "scratch": ["layer.{L}.scores"],
        },
        "out_proj": {
            "op": "attn_proj",
            "name": "attn_proj",
            "inputs": ["layer.{L}.attn_out"],
            "weights": ["layer.{L}.wo"],
            "outputs": ["layer.{L}.proj_tmp"],
            "scratch": ["layer.{L}.proj_scratch"],
        },
        "residual_add": {
            "op": "residual_add",
            "name": "residual{residual_num}",
            "inputs": [],
            "outputs": ["layer.{L}.residual{residual_num}"],
        },
        "mlp_gate_up": {
            "op": "mlp_up",
            "name": "mlp_up",
            "inputs": ["layer.{L}.ln{ln_num}_out"],
            "weights": ["layer.{L}.w1"],
            "outputs": ["layer.{L}.fc1_out"],
        },
        "silu_mul": {
            "op": "swiglu",
            "name": "swiglu",
            "inputs": ["layer.{L}.fc1_out"],
            "outputs": ["layer.{L}.swiglu_out"],
        },
        "mlp_down": {
            "op": "mlp_down",
            "name": "mlp_down",
            "inputs": ["layer.{L}.swiglu_out"],
            "weights": ["layer.{L}.w2"],
            "outputs": ["layer.{L}.mlp_out"],
        },
    }

    if template_op not in op_mappings:
        print(f"[TEMPLATE] Warning: unknown op '{template_op}', skipping")
        return None, rmsnorm_count, residual_count

    # Get the base mapping
    mapped = copy.deepcopy(op_mappings[template_op])

    # Replace layer placeholder
    def replace_layer_placeholder(obj):
        if isinstance(obj, str):
            return obj.replace("{L}", str(layer_id))
        elif isinstance(obj, list):
            return [replace_layer_placeholder(item) for item in obj]
        return obj

    # Handle biases based on template flags
    # Note: biases are always included in the template; they'll be zeros if not in weights
    if template_op == "qkv_proj":
        mapped["biases"] = ["layer.{L}.bq", "layer.{L}.bk", "layer.{L}.bv"]
    elif template_op == "out_proj":
        mapped["biases"] = ["layer.{L}.bo"]
    elif template_op == "mlp_gate_up":
        mapped["biases"] = ["layer.{L}.b1"]
    elif template_op == "mlp_down":
        mapped["biases"] = ["layer.{L}.b2"]

    # Sequence-aware determination based on counts (passed in, not function attrs)
    if template_op == "rmsnorm":
        # First rmsnorm is ln1 (index 0), second is ln2 (index 1)
        ln_num = 1 if rmsnorm_count == 0 else 2
        rmsnorm_count += 1
        mapped["name"] = mapped["name"].replace("{ln_num}", str(ln_num))
        mapped["weights"] = [w.replace("{ln_num}", str(ln_num)) for w in mapped["weights"]]
        mapped["outputs"] = [o.replace("{ln_num}", str(ln_num)) for o in mapped["outputs"]]
        # Input for ln1 is "input" (from previous layer or residual)
        # Input for ln2 is residual1 (from first residual_add)
        if ln_num == 2:
            # Find residual1 output from previous ops
            residual1_found = False
            for prev_out in prev_outputs:
                if f"layer.{layer_id}.residual1" in prev_out:
                    mapped["inputs"] = [prev_out]
                    residual1_found = True
                    break
            if not residual1_found:
                # Fallback - use the first output that looks like a residual
                for prev_out in prev_outputs:
                    if "residual" in prev_out and "out" in prev_out:
                        mapped["inputs"] = [prev_out]
                        residual1_found = True
                        break
                if not residual1_found:
                    mapped["inputs"] = [f"layer.{layer_id}.residual1"]
    elif template_op == "residual_add":
        # First residual_add → residual1 (adds: input + proj_tmp)
        # Second residual_add → residual2 (adds: residual1_out + mlp_out)
        residual_num = 1 if residual_count == 0 else 2
        residual_count += 1

        # Determine inputs based on which residual_add this is
        if residual_num == 1:
            # First residual: add layer input to attention output
            # Find proj_tmp from previous ops
            proj_tmp = None
            for prev_out in prev_outputs:
                if f"layer.{layer_id}.proj_tmp" in prev_out:
                    proj_tmp = prev_out
                    break
            if proj_tmp:
                mapped["inputs"] = ["input", proj_tmp]
            else:
                mapped["inputs"] = ["input", f"layer.{layer_id}.proj_tmp"]
            mapped["outputs"] = [f"layer.{layer_id}.residual1"]
        else:
            # Second residual: add residual1 to mlp output
            # Find residual1 output and mlp_out from previous ops
            residual1_out = None
            mlp_out = None
            for prev_out in prev_outputs:
                if f"layer.{layer_id}.residual1" in prev_out:
                    residual1_out = prev_out
                elif f"layer.{layer_id}.mlp_out" in prev_out:
                    mlp_out = prev_out

            if residual1_out and mlp_out:
                mapped["inputs"] = [residual1_out, mlp_out]
            else:
                # Fallback
                mapped["inputs"] = [f"layer.{layer_id}.residual1", f"layer.{layer_id}.mlp_out"]
            mapped["outputs"] = [f"layer.{layer_id}.output"]

        mapped["name"] = mapped["name"].replace("{residual_num}", str(residual_num))
        mapped["outputs"] = [o.replace("{residual_num}", str(residual_num)) for o in mapped["outputs"]]
    elif template_op == "mlp_gate_up":
        # MLP gate_up should take input from ln2 output (residual1_out after residual1)
        # Find ln2 output from previous ops
        ln2_out = None
        for prev_out in prev_outputs:
            if f"layer.{layer_id}.ln2_out" in prev_out:
                ln2_out = prev_out
                break

        if ln2_out:
            mapped["inputs"] = [ln2_out]
        else:
            # Fallback to residual1
            mapped["inputs"] = [f"layer.{layer_id}.residual1"]

        # Update outputs to use ln_num for consistency
        mapped["outputs"] = [o.replace("{ln_num}", "2") for o in mapped["outputs"]]
        mapped["weights"] = [w.replace("{ln_num}", "2") for w in mapped["weights"]]
    else:
        # For other ops, just replace {ln_num} if present
        # Most ops after attention use ln2
        if "{ln_num}" in mapped["name"]:
            mapped["name"] = mapped["name"].replace("{ln_num}", "2")
        if "{ln_num}" in str(mapped.get("weights", [])):
            mapped["weights"] = [w.replace("{ln_num}", "2") for w in mapped["weights"]]
        if "{ln_num}" in str(mapped.get("outputs", [])):
            mapped["outputs"] = [o.replace("{ln_num}", "2") for o in mapped["outputs"]]

    return replace_layer_placeholder(mapped), rmsnorm_count, residual_count


def build_graph_ir_v6(config: Dict, model_name: str, alignment_bytes: int = 64, template: Optional[Dict] = None) -> Dict:
    dtype = config.get("dtype", "fp32")  # Default to fp32 if not specified
    elem_bytes = v3.DTYPE_BYTES.get(dtype, 4)
    weight_dtype = str(config.get("weight_dtype", "")).lower()
    use_k_align = weight_dtype in ("q4_k", "q6_k", "q8_k")
    qk_align_bytes = QK_K * elem_bytes

    # Support both internal naming and HuggingFace naming conventions
    E = config.get("embed_dim", config.get("hidden_size", 0))
    H = config.get("num_heads", config.get("num_attention_heads", 0))
    KV = config.get("num_kv_heads", config.get("num_key_value_heads", H))
    D = config.get("head_dim", E // H if H else 64)
    I = config.get("intermediate_dim", config.get("intermediate_size", 0))
    T = config.get("max_seq_len", config.get("max_position_embeddings", 32768))
    V = config.get("vocab_size", 0)

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
        "NUM_MERGES": {"name": "num_merges", "value": config.get("num_merges", 0)},
        "VOCAB_BYTES": {"name": "total_vocab_bytes", "value": config.get("total_vocab_bytes", 0)},
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
        # Vocabulary binary data (part of model weights)
        buf("vocab_offsets", "weight", ["V"], buf_dtype="i32"),
        buf("vocab_strings", "weight", ["VOCAB_BYTES"], buf_dtype="u8"),
        buf("vocab_merges", "weight", ["NUM_MERGES", "3"], buf_dtype="i32"),
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

    # Build body ops from template (required for v6.6)
    if not template or "block_types" not in template:
        raise ValueError(
            "Template is required (from weights_manifest or BUMP metadata). "
            "idecarun convert_*_to_bump_v6_6.py and pass --weights-manifest or --bump."
        )

    print(f"[TEMPLATE] Building graph from template: {template.get('name', 'unknown')}")

    # Read and use template flags
    template_flags = template.get("flags", {})
    print(f"[TEMPLATE] Flags:")
    for key, value in template_flags.items():
        print(f"  - {key}: {value}")
        # Store flags in config for later use
        config[f"template_{key}"] = value

    # Use template to build ops
    body_ops = []
    # For now, use the first (or default) block type
    # In the future, we can implement layer_map overrides
    default_block = template["block_types"].get("dense")
    if not default_block or "ops" not in default_block:
        raise ValueError("Template missing block_types.dense.ops")

    # Sequence-aware mapping: track op index and previous outputs
    prev_outputs = []  # Track outputs from previous ops in the sequence
    rmsnorm_count = 0
    residual_count = 0
    for op_index, template_op in enumerate(default_block["ops"]):
        mapped_op, rmsnorm_count, residual_count = map_template_op_to_internal(
            template_op,
            1,  # layer_id will be filled per layer (using 1 for template building)
            op_index,
            prev_outputs,
            rmsnorm_count,
            residual_count,
            template_flags
        )
        if mapped_op:
            body_ops.append(mapped_op)
            # Add outputs to prev_outputs for next iteration
            if "outputs" in mapped_op:
                prev_outputs.extend(mapped_op["outputs"])
    print(f"[TEMPLATE] Generated {len(body_ops)} ops from template")

    # Handle rope insertion
    if globals_buffers:
        # Check if template already includes rope_qk
        template_has_rope = False
        if template and "block_types" in template:
            default_block = template["block_types"].get("dense")
            if default_block and "ops" in default_block:
                template_has_rope = "rope_qk" in default_block["ops"]

        if not template_has_rope:
            # Insert rope op after qkv_project (at index 2)
            body_ops.insert(2, {
                "op": "rope",
                "name": "rope",
                "inputs": ["layer.{L}.q", "layer.{L}.k", "rope_cos_cache", "rope_sin_cache"],
                "outputs": ["layer.{L}.q", "layer.{L}.k"],
            })
            print("[TEMPLATE] Inserted rope op (template didn't include rope_qk)")

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
        "template": template,
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


def load_kernel_registry() -> Dict[str, Dict]:
    registry_path = v3.DEFAULT_KERNEL_REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Kernel registry not found: {registry_path}. "
            "Run version/v6.6/scripts/gen_kernel_registry_from_maps.py first."
        )
    with registry_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return v3.normalize_kernel_registry(data)


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


def _get_activation_dtype_for_weight(weight_dtype: str) -> str:
    """Determine activation dtype based on weight dtype.

    K-type quants (q4_k, q6_k) use Q8_K activations.
    Standard quants (q5_0, q8_0) use Q8_0 activations.
    """
    weight_dtype = weight_dtype.lower()
    if weight_dtype in ("q4_k", "q6_k", "q4_k_m"):
        return "q8_k"
    elif weight_dtype in ("q5_0", "q8_0", "q4_0", "q4_1", "q5_1"):
        return "q8_0"
    else:
        return "q8_0"  # Default to Q8_0


def select_kernel(op: Dict, dtype: str, mode: str, registry: Dict[str, Dict],
                  weight_dtype: Optional[str] = None,
                  allow_missing: bool = False) -> Optional[str]:
    """Select EXACT kernel for an operation from registry.

    v6.6 kernel selection is data-driven:
    - Kernel names are exact (e.g., "gemv_q5_0_q8_0", "gemm_nt_q4_k_q8_k")
    - Mode determines GEMV (decode) vs GEMM (prefill)
    - Weight dtype and activation dtype determine the exact kernel variant

    Args:
        op: Operation dict with at least "op" key
        dtype: Default/activation dtype ("f32", "bf16", etc.)
        mode: Execution mode ("prefill", "decode", "training")
        registry: Kernel registry from kernel maps (required in v6.6)
        weight_dtype: Weight dtype for quantized inference ("q4_k", "q6_k", "q5_0", etc.)

    Returns:
        Exact kernel function name, or None for host-side ops
    """
    op_name = op["op"]
    if op_name in HOST_OPS or op_name == "lm_head":
        return None

    # Get weight dtype (from arg or op)
    w_dtype = weight_dtype or op.get("weight_dtype")
    is_quant = bool(w_dtype) and (w_dtype == "mixed" or qt.is_quantized_dtype(w_dtype))

    # Determine activation dtype based on weight dtype
    act_dtype = _get_activation_dtype_for_weight(w_dtype) if is_quant else None

    if not registry:
        raise ValueError("v6.6 requires kernel registry for kernel selection")

    # -----------------------------------------------------------------
    # EXACT kernel selection based on op type, mode, and dtypes
    # -----------------------------------------------------------------

    # For linear projections (QKV, attention output, MLP)
    # decode -> GEMV (matrix-vector), prefill -> GEMM (matrix-matrix)
    projection_ops = {"qkv_project", "attn_proj", "mlp_up", "mlp_down", "mlp_gate"}
    is_projection = op_name in projection_ops

    if is_projection and is_quant:
        w_key = w_dtype.lower()

        if mode == "decode":
            # Decode uses GEMV: gemv_{weight}_{activation}
            exact_kernel = f"gemv_{w_key}_{act_dtype}"
        else:
            # Prefill uses GEMM: gemm_nt_{weight}_{activation}
            exact_kernel = f"gemm_nt_{w_key}_{act_dtype}"

        # Check registry for exact kernel
        if exact_kernel in registry:
            kernel_entry = registry[exact_kernel]
            kernel_fn = kernel_entry.get("impl", {}).get("function", exact_kernel)
            print(f"[KERNEL] {op_name} ({mode}) -> {kernel_fn}")
            return kernel_fn

        # Fallback: search by matching op/quant fields
        print(f"[KERNEL] Exact kernel '{exact_kernel}' not in registry, searching...")

    # For non-projection or non-quantized ops, search registry by op type
    matching_kernel = None

    def quant_weight_matches(weight_q: str, target: str) -> bool:
        if not weight_q or not target:
            return False
        weight_q = weight_q.strip().lower()
        target = target.strip().lower()
        if weight_q == "mixed":
            return True
        if "|" in weight_q:
            return target in [part.strip().lower() for part in weight_q.split("|")]
        return weight_q == target

    # Map op names to registry op types
    op_to_registry_op = {
        "attention": "attention",
        "rmsnorm": "rmsnorm",
        "rope": "rope",
        "swiglu": "swiglu",
        "embedding": "embedding",
        "quantize": "quantize",
        "residual_add": "residual_add",
        "attn_proj": "attention_projection",
        "qkv_project": "qkv_projection",
        "mlp_up": "gemm",
        "mlp_down": "gemm",
        "mlp_gate": "gemm",
    }

    registry_op = op_to_registry_op.get(op_name, op_name)

    # Ops that don't use quantized weights (FP32 only)
    non_quantized_weight_ops = {"rmsnorm", "swiglu", "attention", "rope", "residual_add"}
    is_non_quantized_op = op_name in non_quantized_weight_ops or registry_op in non_quantized_weight_ops

    # Search registry for best match
    for kernel_id, kernel_entry in registry.items():
        kernel_op = kernel_entry.get("op", "")
        kernel_quant = kernel_entry.get("quant", {})

        # Match by op type
        if kernel_op != registry_op and kernel_op != op_name:
            continue

        # For quantized ops that actually use quantized weights, match weight dtype
        if is_quant and not is_non_quantized_op:
            weight_q = kernel_quant.get("weight", "").lower()
            if not quant_weight_matches(weight_q, w_dtype):
                continue

            # For projections, also check activation dtype and op type (gemv vs gemm)
            if is_projection:
                act_q = kernel_quant.get("activation", "").lower()
                if act_q != act_dtype:
                    continue

                # Check gemv vs gemm based on mode
                if mode == "decode" and kernel_op != "gemv":
                    continue
                if mode == "prefill" and kernel_op != "gemm":
                    continue

        # Found a match
        kernel_fn = kernel_entry.get("impl", {}).get("function")
        if not kernel_fn:
            kernel_fn = kernel_entry.get("name", kernel_entry.get("id", kernel_id))
        matching_kernel = kernel_fn
        print(f"[KERNEL] {op_name} ({mode}) -> {matching_kernel}")
        break

    if matching_kernel:
        return matching_kernel

    if allow_missing:
        return op.get("kernel")

    raise KeyError(
        f"No kernel in registry for op='{op_name}' mode='{mode}' "
        f"weight_dtype='{w_dtype or 'none'}' activation_dtype='{act_dtype or 'none'}'"
    )


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
    template = graph.get("template")
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
    num_layers = config.get("num_layers", config.get("num_hidden_layers", 0))
    manifest_entries = {}
    if weights_manifest and isinstance(weights_manifest, dict):
        for entry in weights_manifest.get("entries", []):
            if "name" in entry:
                manifest_entries[entry["name"]] = entry

    quant_summary = graph.get("quant_summary", {})

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
        dtype = tmpl.get("dtype", config.get("dtype", "fp32"))
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
            op_weight_dtypes = None
            op_weight_names = []

            # Prefer quant_summary for per-op dtypes (layer-aware)
            if isinstance(quant_summary, dict) and quant_summary:
                op_weight_dtypes = get_op_weight_dtypes(quant_summary, layer_id, op_out.get("op", ""))
                if op_weight_dtypes:
                    unique = {dt for dt in op_weight_dtypes.values() if dt}
                    if unique:
                        op_weight_dtype = next(iter(unique)) if len(unique) == 1 else "mixed"

            # Fallback to manifest entries if quant_summary missing or incomplete
            if op_weight_dtype is None and "weights" in op_out:
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
                            op_weight_dtype = "mixed"
                            break

            if op_weight_dtypes:
                op_out["weight_dtypes"] = op_weight_dtypes
            if op_weight_dtype:
                op_out["weight_dtype"] = op_weight_dtype
            activation_dtype = config.get("dtype", "fp32")
            op_out["kernel"] = select_kernel(op, activation_dtype, mode, registry,
                                             weight_dtype=op_weight_dtype,
                                             allow_missing=skip_registry_check)
            # Don't require backward kernels to be in registry
            is_quant = op_weight_dtype and (op_weight_dtype == "mixed" or qt.is_quantized_dtype(op_weight_dtype))
            if (op_out["kernel"] and registry and op_out["kernel"] not in registry and
                    not skip_registry_check and not is_quant):
                raise KeyError(f"Unknown kernel: {op_out['kernel']}")
            if op_out["kernel"]:
                op_out["kernel_dtype"] = activation_dtype

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
    used_bufs = {"header": {"vocab_offsets", "vocab_strings", "vocab_merges"}, "footer": set(), "globals": set(), "layer": set()}

    # Header ops (no layer expansion)
    header_ops = process_ops(header_ops_source, 0, {}, used_bufs, skip_registry)

    # Body ops (expanded per layer)
    # For backward mode, process layers in reverse order
    layers_out = []
    layer_order = range(num_layers - 1, -1, -1) if is_backward else range(num_layers)

    # Check if template has layer_map
    template_has_layer_map = template and "layer_map" in template
    layer_map = template.get("layer_map", {}) if template_has_layer_map else {}

    for layer_id in layer_order:
        used_bufs["layer"] = set()  # Reset per layer

        # Determine which block type to use for this layer
        # Check overrides first, then use default
        block_type = layer_map.get("default", "dense")
        if "overrides" in layer_map and str(layer_id) in layer_map["overrides"]:
            block_type = layer_map["overrides"][str(layer_id)]
            if layer_id < 10:  # Only print for first few layers to avoid spam
                print(f"[TEMPLATE] Layer {layer_id}: using block '{block_type}' (override)")

        # Get the appropriate ops for this block type
        layer_ops_source = body_ops_source  # Default
        if template_has_layer_map and block_type in template["block_types"]:
            # Build ops for this specific block type
            block_def = template["block_types"][block_type]
            if "ops" in block_def:
                # Build ops using template mapping
                block_ops = []
                prev_outputs = []
                # Initialize sequence counters for this layer
                rmsnorm_count = 0
                residual_count = 0
                # Get template flags
                template_flags = template.get("flags", {})
                for op_index, template_op in enumerate(block_def["ops"]):
                    mapped_op, rmsnorm_count, residual_count = map_template_op_to_internal(
                        template_op,
                        layer_id,
                        op_index,
                        prev_outputs,
                        rmsnorm_count,
                        residual_count,
                        template_flags
                    )
                    if mapped_op:
                        block_ops.append(mapped_op)
                        if "outputs" in mapped_op:
                            prev_outputs.extend(mapped_op["outputs"])
                layer_ops_source = block_ops
                if layer_id < 10:  # Only print for first few layers
                    print(f"[TEMPLATE] Layer {layer_id}: built {len(block_ops)} ops from block '{block_type}'")

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

        layer_ops = process_ops(layer_ops_source, layer_id, bindings, used_bufs, skip_registry)

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

def find_fusion_candidates(ops: List[Dict], mode: str, registry_path: Optional[Path] = None) -> List[Dict]:
    """
    Scan a list of ops for fusible sequences.
    Returns candidates sorted by priority (highest first).

    Supports both:
      - Manual patterns: Match by op names (from graph IR)
      - Registry patterns: Match by kernel IDs (from lowered IR)

    Args:
        ops: List of ops from lowered IR
        mode: Execution mode (prefill/decode)
        registry_path: Optional path to kernel registry
    """
    # Try to use registry-driven patterns if registry is available
    try:
        patterns = fp.get_registry_driven_patterns(mode, registry_path)
        print(f"[FUSION] Loaded {len(patterns)} patterns (registry + manual)")
    except (FileNotFoundError, ImportError):
        # Fallback to manual patterns only
        patterns = fp.get_patterns_for_mode(mode)
        print(f"[FUSION] Loaded {len(patterns)} patterns (manual only)")

    candidates = []

    for pattern in patterns:
        seq = pattern["sequence"]
        seq_len = len(seq)

        # Sliding window match
        for i in range(len(ops) - seq_len + 1):
            window = ops[i:i + seq_len]

            # Check if this is a registry pattern (kernel IDs) or manual pattern (op names)
            # Registry patterns have kernel IDs in ops, manual patterns have op names
            if "kernel" in window[0]:
                # Registry pattern: match by kernel IDs
                kernel_seq = [op.get("kernel") for op in window]
                if kernel_seq == seq:
                    candidates.append({
                        "start_idx": i,
                        "end_idx": i + seq_len,
                        "pattern": pattern,
                        "matched_ops": window,
                    })
            else:
                # Manual pattern: match by op names
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


def apply_fusion_pass(lowered: Dict, mode: str, config: Dict, registry_path: Optional[Path] = None) -> Tuple[Dict, fp.FusionStats]:
    """
    Apply fusion optimizations to lowered IR.

    Args:
        lowered: Lowered IR dict
        mode: Execution mode (prefill/decode)
        config: Fusion configuration
        registry_path: Optional path to kernel registry

    Returns:
        (optimized_ir, fusion_stats)
    """
    if not config.get("enable_fusion", True):
        return lowered, fp.FusionStats()

    stats = fp.FusionStats()
    optimized = copy.deepcopy(lowered)
    dtype = optimized["config"].get("dtype", "fp32")

    # Get fusion patterns (registry-driven if available)
    try:
        patterns = fp.get_registry_driven_patterns(mode, registry_path)
    except (FileNotFoundError, ImportError):
        # Fallback to manual patterns only
        patterns = fp.get_patterns_for_mode(mode)

    # Collect all patterns' removed buffers
    all_removed_patterns = []
    for pattern in patterns:
        all_removed_patterns.extend(pattern.get("remove_buffers", []))

    # Process each layer
    for layer in optimized["sections"][0]["layers"]:
        layer_id = layer["id"]
        ops = layer["ops"]

        # Find and apply fusions
        candidates = find_fusion_candidates(ops, mode, registry_path)
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
        offset = allocator.alloc(size)
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


def build_layout_v6_native(config: Dict, model_name: str,
                           include_training: bool = False,
                           include_decode_scratch: bool = True) -> v6_low.ModelLayout:
    """Build layout using v6_ir_lowering (per-layer buffers, training-compatible).

    This is the v6 native layout builder that provides:
    1. Per-layer activation buffers (no sharing between layers for backprop)
    2. Per-layer decode scratch buffers in arena (not stack)
    3. Complete training support with gradient and optimizer state buffers

    Use this for training/backprop-compatible layouts. For inference-only
    with shared buffers (smaller memory), use build_layout_from_lowered().

    Args:
        config: Model configuration dict
        model_name: Model identifier
        include_training: If True, include gradient and optimizer state buffers
        include_decode_scratch: If True, include per-layer decode scratch buffers

    Returns:
        v6_low.ModelLayout with computed offsets
    """
    return v6_low.build_model_layout(
        config, model_name,
        include_training=include_training,
        include_decode_scratch=include_decode_scratch,
    )


def emit_layout_v6_native(layout: v6_low.ModelLayout, output_dir: str):
    """Emit v6 native layout files (JSON and human-readable map)."""
    import os
    json_path = os.path.join(output_dir, "layout_v6_native.json")
    map_path = os.path.join(output_dir, "layout_v6_native.map")

    v6_low.emit_layout_json(layout, json_path)
    v6_low.emit_layout_map(layout, map_path)


def get_decode_buffer_offset(layout: v6_low.ModelLayout, layer_id: int, buffer_name: str) -> Optional[int]:
    """Get the arena offset for a per-layer decode scratch buffer.

    This is used by codegen to emit arena pointers instead of stack arrays.

    Args:
        layout: v6 native ModelLayout
        layer_id: Layer index
        buffer_name: One of: q_token, k_token, v_token, attn_out, fc1_out, swiglu_out

    Returns:
        Offset in bytes from arena base, or None if not found
    """
    decode_bufs = v6_low.get_layer_decode_buffers(layout, layer_id)
    if not decode_bufs:
        return None

    buf_map = {
        "q_token": decode_bufs.q_token,
        "k_token": decode_bufs.k_token,
        "v_token": decode_bufs.v_token,
        "attn_out": decode_bufs.attn_out,
        "fc1_out": decode_bufs.fc1_out,
        "swiglu_out": decode_bufs.swiglu_out,
    }
    buf = buf_map.get(buffer_name)
    return buf.offset if buf else None


def compute_decode_scratch_offsets(layout: v6_low.ModelLayout, layer_id: int) -> Dict[str, int]:
    """Get all decode scratch buffer offsets for a layer.

    Returns:
        Dict mapping buffer names to arena offsets
    """
    decode_bufs = v6_low.get_layer_decode_buffers(layout, layer_id)
    if not decode_bufs:
        return {}

    return {
        "q_token": decode_bufs.q_token.offset,
        "k_token": decode_bufs.k_token.offset,
        "v_token": decode_bufs.v_token.offset,
        "attn_out": decode_bufs.attn_out.offset,
        "fc1_out": decode_bufs.fc1_out.offset,
        "swiglu_out": decode_bufs.swiglu_out.offset,
    }


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
    num_layers = config.get("num_layers", config.get("num_hidden_layers", 0))

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
            "--template",
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
            "--bump",
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
        "template": None,
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
        "int8": True,  # Use INT8 activations for faster decode (5-15x speedup)
        "codegen": "v6",  # Codegen version: v6 (explicit unrolled, default) or v4 (loop-based)
        "decode_layout": "prefill",  # Use prefill or decode layout for decode codegen
        "skip_manifest_validation": False,  # Skip manifest validation (debug/testing)
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
        # BUMPWGT5 weights file
        "bump": None,  # Path to BUMPWGT5 weights file (reads metadata from EOF footer)
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
        elif arg.startswith("--template="):
            args["template"] = arg.split("=", 1)[1]
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
        elif arg.startswith("--decode-layout="):
            layout_val = arg.split("=", 1)[1].lower()
            if layout_val not in ("prefill", "decode"):
                raise ValueError(f"--decode-layout must be prefill|decode, got: {layout_val}")
            args["decode_layout"] = layout_val
        elif arg.startswith("--weight-dtype="):
            w_dtype = arg.split("=", 1)[1].lower()
            # Normalize aliases
            valid_dtypes = ("q4_0", "q4_1", "q5_0", "q5_1", "q4_k", "q4_k_m", "q6_k", "q8_0", "q8_k", "f32", "bf16")
            if w_dtype not in valid_dtypes:
                raise ValueError(f"--weight-dtype must be one of {'/'.join(valid_dtypes)}, got: {w_dtype}")
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
        elif arg == "--int8":
            args["int8"] = True
        elif arg == "--no-int8":
            args["int8"] = False
        elif arg.startswith("--codegen="):
            codegen_val = arg.split("=", 1)[1].lower()
            if codegen_val not in ("v4", "v6"):
                raise ValueError(f"--codegen must be v4/v6, got: {codegen_val}")
            args["codegen"] = codegen_val
        elif arg == "--skip-manifest-validation":
            args["skip_manifest_validation"] = True
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
        elif arg.startswith("--bump="):
            args["bump"] = arg.split("=", 1)[1]
        elif arg.startswith("--"):
            raise ValueError(f"Unknown option: {arg}")
        else:
            args["model"] = arg

    if (not args.get("help") and not args["model"] and not args["config"] and
            not args["preset"] and not args.get("weights_manifest") and not args.get("bump")):
        raise ValueError("Must provide model ID/URL or --config=FILE or --preset=NAME")

    return args


def print_usage():
    print("Usage:")
    print("  python scripts/v6.6/build_ir_v6_6.py MODEL [OPTIONS]")
    print("  python scripts/v6.6/build_ir_v6_6.py --config=FILE [OPTIONS]")
    print("  python scripts/v6.6/build_ir_v6_6.py --preset=NAME [OPTIONS]")
    print()
    print("v6.6 Features:")
    print("  - INT8 activations ENABLED by default (5-15x faster than FP32)")
    print("  - Uses quantized kernels: gemv_q5_0_q8_0, gemv_q8_0_q8_0, gemv_q4_k_q8_k")
    print("  - Mixed quantization support (Q5_0, Q8_0, Q4_K, Q6_K per layer)")
    print()
    print("Options:")
    print("  --config=FILE           Use local config.json")
    print("  --preset=NAME           Use local preset (qwen2-0.5b, smollm-135)")
    print("  --weights-header=FILE   Safetensors header for weight mapping")
    print("  --weights-index=FILE    model.safetensors.index.json")
    print("  --weights-manifest=FILE Weights manifest (from convert_*_to_bump.py)")
    print("  --bump=FILE             BUMPWGT5 weights file (reads template/config from footer)")
    print("  --template=NAME         Template name (e.g., qwen2, llama) or path")
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
    print("  --codegen=v4|v6         Codegen version (default: v6)")
    print("                          v6: Explicit unrolled (each layer separate, explicit kernels)")
    print("                          v4: Loop-based with runtime dtype dispatch (legacy)")
    print("  --decode-layout=prefill|decode  Layout to use for decode codegen (default: prefill)")
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
    print("  --weight-dtype=q6_k     Q6_K: 6-bit K-quant (6.6 bits/weight)")
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
    print("  python scripts/v6/build_ir_v6.py --preset=qwen2-0.5b")
    print()
    print("  # Generate training schedule (auto-detect memory, compute optimal batch)")
    print("  python scripts/v6/build_ir_v6.py --preset=qwen2-0.5b --modes=training")
    print()
    print("  # Training with specific memory budget and batch size")
    print("  python scripts/v6/build_ir_v6.py --preset=qwen2-0.5b --modes=training --memory=16 --batch-size=32")
    print()
    print("  # Generate backward only (for memory-efficient training)")
    print("  python scripts/v6/build_ir_v6.py --preset=qwen2-0.5b --modes=backward --fusion=on")
    print()
    print("  # Use custom config file")
    print("  python scripts/v6/build_ir_v6.py --config=smolLM-135.json --modes=prefill,decode")
    print()
    print("  # Quantized inference with Q4_K weights (llama.cpp compatible)")
    print("  python scripts/v6/build_ir_v6.py --preset=qwen2-0.5b --weight-dtype=q4_k --dtype=f32")
    print()
    print("  # Verbose output for debugging")
    print("  python scripts/v6/build_ir_v6.py --preset=qwen2-0.5b --fusion-verbose --parallel-verbose")


# ============================================================================
# BUMPWGT5 METADATA READING (EOF FOOTER)
# ============================================================================
#
# BUMPWGT5 FILE STRUCTURE:
# ========================
#   [weights binary data...] + [metadata JSON] + [footer (48 bytes)]
#
# FOOTER FORMAT (little-endian, 48 bytes total):
# +--------+--------+--------+----------------------------------------+
# | Offset | Size   | Type   | Description                            |
# +--------+--------+--------+----------------------------------------+
# | 0      | 8      | u8[8]  | Magic: "BUMPV5MD" (8 bytes)            |
# | 8      | 8      | uint64 | meta_size: size of metadata JSON bytes |
# | 16     | 32     | u8[32] | SHA-256 hash of metadata JSON bytes    |
# +--------+--------+--------+----------------------------------------+
#
# METADATA JSON CONTENTS:
# =======================
# {
#   "version": 5,                           # BUMP format version
#   "template": {                           # Full template (same as --template files)
#     "name": "qwen2",
#     "block_types": { "dense": { "ops": [...] } },
#     "layer_map": { "default": "dense" },
#     "flags": { "use_qkv_bias": "from_weights", ... }
#   },
#   "config": {                             # Model configuration
#     "model_type": "qwen2",
#     "embed_dim": 1024,
#     "num_heads": 16,
#     "num_kv_heads": 2,
#     "head_dim": 64,
#     "intermediate_size": 2816,
#     "max_seq_len": 32768,
#     "vocab_size": 151936,
#     "num_layers": 24
#   },
#   "quant_summary": {                      # Per-tensor quantization info
#     "model.layers.0.attn.wq": "q4_k",
#     "model.layers.0.attn.wk": "q4_k",
#     "model.layers.0.mlp.w1": "q5_0",
#     ...
#   },
#   "manifest_hash": "abc123..."            # SHA-256 of weights_manifest.json
# }
#
# WHY BUMP METADATA?
# ==================
# 1. Single source of truth: template, config, and weights are guaranteed to match
# 2. No need for separate --template, --config, --weights-manifest arguments
# 3. Enables validation: manifest_hash confirms manifest matches baked weights
# 4. Self-describing: any BUMP file can be loaded without external files
#
# USAGE:
# ======
#   python build_ir_v6_6.py --bump=model.bump
#
# This will:
#   1. Read template from BUMP metadata
#   2. Apply config overrides from metadata
#   3. Use quant_summary for kernel selection
#   4. Validate manifest_hash if --weights-manifest is provided
# ============================================================================

import struct
import hashlib

BUMP_FOOTER_SIZE = 48  # Footer is always 48 bytes: 8 + 8 + 32
BUMP_FOOTER_MAGIC = b"BUMPV5MD"  # Magic bytes to identify BUMPWGT5 format


def read_bump_metadata(bump_path: str) -> Dict[str, Any]:
    """
    Read metadata from BUMPWGT5 EOF footer.

    This function reads the metadata embedded at the end of BUMPWGT5 files.
    The metadata contains the template, model config, quantization info,
    and a hash of the weights manifest for validation.

    Args:
        bump_path: Path to BUMPWGT5 weights file (e.g., "model.bump")

    Returns:
        Dict containing:
        - version: BUMP format version (5)
        - template: Full template JSON (same format as --template files)
          - name: Template name (e.g., "qwen2", "llama")
          - block_types: Dict of block definitions with "ops" lists
          - layer_map: Dict mapping layer indices to block types
          - flags: Architecture flags (use_qkv_bias, activation, rope)
        - config: Model configuration dict
          - embed_dim: Embedding dimension
          - num_heads: Number of attention heads
          - num_kv_heads: Number of key/value heads (for GQA)
          - head_dim: Dimension per attention head
          - intermediate_dim: MLP intermediate dimension
          - max_seq_len: Maximum sequence length
          - vocab_size: Vocabulary size
          - num_layers: Number of transformer layers
        - quant_summary: {tensor_name: dtype} map for quantized weights
          - Maps full tensor paths to quantization types
          - Used for kernel selection (q4_k, q5_0, q8_0, etc.)
        - manifest_hash: SHA-256 hex string of canonical weights_manifest.json
          - Used to validate that external manifest matches BUMP file

    Raises:
        ValueError: If file is too small, invalid magic, corrupted metadata,
                    or SHA256 verification fails

    Example:
        >>> metadata = read_bump_metadata("model.bump")
        >>> print(metadata["template"]["name"])
        "qwen2"
        >>> print(metadata["config"]["num_layers"])
        24
    """
    with open(bump_path, "rb") as f:
        # Get file size by seeking to end
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

        # Seek to start of footer (last 48 bytes)
        f.seek(-BUMP_FOOTER_SIZE, os.SEEK_END)

        footer = f.read(BUMP_FOOTER_SIZE)
        if len(footer) != BUMP_FOOTER_SIZE:
            raise ValueError(
                f"BUMP file too small: cannot read footer "
                f"(file is {file_size} bytes, need {BUMP_FOOTER_SIZE})"
            )

        # Parse footer: 8-byte magic + 8-byte size (little-endian) + 32-byte hash
        # Format: "<8sQ32s" = little-endian, 8-byte string, unsigned long long, 32-byte string
        magic, meta_size, expected_hash = struct.unpack("<8sQ32s", footer)

        # Validate magic bytes
        if magic != BUMP_FOOTER_MAGIC:
            raise ValueError(
                f"Not a BUMPWGT5 file: invalid magic {magic!r} "
                f"(expected {BUMP_FOOTER_MAGIC!r})"
            )

        # Calculate offset to metadata JSON
        # Metadata starts at: file_size - footer_size - meta_size
        meta_offset = file_size - BUMP_FOOTER_SIZE - meta_size

        if meta_offset < 0:
            raise ValueError(
                f"Invalid metadata: meta_size={meta_size} exceeds file bounds "
                f"(file is {file_size} bytes)"
            )

        # Read metadata JSON bytes
        f.seek(meta_offset)
        meta_bytes = f.read(meta_size)
        if len(meta_bytes) != meta_size:
            raise ValueError(
                f"Failed to read metadata: expected {meta_size} bytes, "
                f"got {len(meta_bytes)}"
            )

        # Verify SHA256 of raw bytes (CRITICAL: don't re-serialize!)
        # We hash the raw bytes we read, not json.dumps() output
        actual_hash = hashlib.sha256(meta_bytes).digest()
        if actual_hash != expected_hash:
            # Show first 16 chars of hashes for debugging
            raise ValueError(
                f"Metadata SHA256 mismatch: file may be corrupted or tampered with "
                f"(expected {expected_hash[:16].hex()}..., got {actual_hash[:16].hex()}...)"
            )

        # Parse metadata JSON
        metadata = json.loads(meta_bytes.decode("utf-8"))

        # Diagnostic output
        config_meta = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
        model_name = config_meta.get("model_type") or config_meta.get("model") or "unknown"
        template_name = metadata.get("template", {}).get("name", "unknown") if isinstance(metadata.get("template"), dict) else "unknown"
        quant_count = len(metadata.get("quant_summary", {}))

        print(f"[BUMP] Read metadata from {bump_path}")
        print(f"[BUMP]   Template: {template_name}")
        print(f"[BUMP]   Config: {model_name}")
        print(f"[BUMP]   Quant types: {quant_count} tensors")

        return metadata


def canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    """
    Encode JSON with stable ordering for hashing.

    This produces canonical JSON that can be hashed and compared across systems.
    Key properties:
    - sort_keys=True: Keys are sorted alphabetically
    - separators=(",", ":"): No spaces after commas/colons
    - ensure_ascii=True: ASCII output (no unicode escapes needed)

    Args:
        payload: Dict to serialize to canonical JSON bytes

    Returns:
        UTF-8 encoded JSON bytes with stable ordering

    Example:
        >>> canonical_json_bytes({"b": 1, "a": 2})
        b'{"a":2,"b":1}'
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def normalize_bump_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a BUMP/manifest config dict into build_ir config keys.

    Accepts both BUMPWGT5 field names (intermediate_size, context_length, model)
    and build_ir names (intermediate_dim, max_seq_len, model_type), as well as
    HuggingFace naming conventions (hidden_size, num_attention_heads, etc.).
    """
    # Map various naming conventions to canonical internal keys
    # Priority: first match wins (so internal keys take precedence over HF keys)
    bump_to_config = {
        # Internal / BUMP keys
        "embed_dim": "embed_dim",
        "num_heads": "num_heads",
        "num_kv_heads": "num_kv_heads",
        "head_dim": "head_dim",
        "intermediate_dim": "intermediate_dim",
        "max_seq_len": "max_seq_len",
        "num_layers": "num_layers",
        "rms_eps": "rms_eps",
        # HuggingFace keys
        "hidden_size": "embed_dim",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_kv_heads",
        "intermediate_size": "intermediate_dim",
        "max_position_embeddings": "max_seq_len",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "rms_eps",
        # BUMP legacy keys
        "context_length": "max_seq_len",
        # Common keys
        "vocab_size": "vocab_size",
        "model": "model_type",
        "model_type": "model_type",
        "activation_fn": "activation_fn",
        "rope_theta": "rope_theta",
    }
    normalized: Dict[str, Any] = {}
    if not isinstance(raw_config, dict):
        return normalized
    for bump_key, cfg_key in bump_to_config.items():
        if bump_key in raw_config and cfg_key not in normalized:
            # Only set if not already set (internal keys take precedence)
            normalized[cfg_key] = raw_config[bump_key]
    return normalized


def extract_config_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a config dict from a weights manifest.

    Supports either:
    - manifest["config"] (preferred, full config dict), or
    - top-level fields like embed_dim/num_heads/etc (legacy).
    """
    if not isinstance(manifest, dict):
        return {}

    if isinstance(manifest.get("config"), dict):
        return normalize_bump_config(manifest["config"])

    legacy_keys = (
        "model",
        "model_type",
        "embed_dim",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "intermediate_size",
        "intermediate_dim",
        "context_length",
        "max_seq_len",
        "vocab_size",
        "rms_eps",
    )
    legacy_cfg = {key: manifest[key] for key in legacy_keys if key in manifest}
    return normalize_bump_config(legacy_cfg)


def extract_quant_summary(weights_manifest: Optional[Dict[str, Any]],
                          bump_metadata: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Extract per-layer quant summary from manifest or BUMP metadata."""
    if isinstance(weights_manifest, dict):
        qs = weights_manifest.get("quant_summary")
        if isinstance(qs, dict):
            return qs
    if isinstance(bump_metadata, dict):
        qs = bump_metadata.get("quant_summary")
        if isinstance(qs, dict):
            return qs
    return {}


def lookup_quant_dtype(quant_summary: Dict[str, Dict[str, str]],
                       layer_id: int,
                       weight_key: str) -> Optional[str]:
    """Lookup dtype from quant_summary for a given layer + weight key."""
    if not quant_summary:
        return None
    layer_key = f"layer.{layer_id}"
    layer_entry = quant_summary.get(layer_key)
    if isinstance(layer_entry, dict):
        dtype = layer_entry.get(weight_key)
        if dtype:
            return dtype

    # Fallback: quant_summary might be a flat map with full tensor names
    candidates = [
        f"layer.{layer_id}.{weight_key}",
        f"model.layer.{layer_id}.{weight_key}",
        f"model.layers.{layer_id}.{weight_key}",
    ]
    for name in candidates:
        dtype = quant_summary.get(name)
        if dtype:
            return dtype
    return None


def get_op_weight_dtypes(quant_summary: Dict[str, Dict[str, str]],
                         layer_id: int,
                         op_name: str) -> Optional[Dict[str, str]]:
    """Return per-op weight dtypes from quant_summary (if available)."""
    if not quant_summary:
        return None
    if op_name == "qkv_project":
        return {
            "wq": lookup_quant_dtype(quant_summary, layer_id, "wq"),
            "wk": lookup_quant_dtype(quant_summary, layer_id, "wk"),
            "wv": lookup_quant_dtype(quant_summary, layer_id, "wv"),
        }
    if op_name == "attn_proj":
        return {"wo": lookup_quant_dtype(quant_summary, layer_id, "wo")}
    if op_name == "mlp_up":
        return {"w1": lookup_quant_dtype(quant_summary, layer_id, "w1")}
    if op_name == "mlp_down":
        return {"w2": lookup_quant_dtype(quant_summary, layer_id, "w2")}
    return None


# =============================================================================
# PREFLIGHT VALIDATION FUNCTIONS
# =============================================================================
# These functions perform explicit checks before IR generation to catch
# configuration errors early with clear error messages.
# =============================================================================

def validate_sidecar_template(weights_manifest: Optional[Dict],
                               bump_metadata: Optional[Dict],
                               model_type: Optional[str] = None) -> Tuple[bool, str]:
    """Validate that template exists in sidecar or templates directory.

    Returns:
        (ok, message) tuple. ok=True if template found.
    """
    manifest_template = weights_manifest.get("template") if isinstance(weights_manifest, dict) else None
    bump_template = bump_metadata.get("template") if isinstance(bump_metadata, dict) else None

    if isinstance(manifest_template, dict) and manifest_template:
        name = manifest_template.get("name", "unknown")
        return True, f"Template found in manifest: {name}"

    if isinstance(bump_template, dict) and bump_template:
        name = bump_template.get("name", "unknown")
        return True, f"Template found in BUMP metadata: {name}"

    # Check templates directory as fallback
    if model_type:
        templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
        template_path = os.path.join(templates_dir, f"{model_type}.json")
        if os.path.exists(template_path):
            return True, f"Template found in templates/{model_type}.json"

    return False, "No template found in sidecar or templates directory"


def validate_sidecar_quant(weights_manifest: Optional[Dict],
                           bump_metadata: Optional[Dict],
                           weight_dtype: Optional[str] = None) -> Tuple[bool, str]:
    """Validate that quant_summary exists in sidecar or weight_dtype is specified.

    Returns:
        (ok, message) tuple. ok=True if quant_summary found or weight_dtype specified.
    """
    quant_summary = extract_quant_summary(weights_manifest, bump_metadata)

    if not quant_summary:
        # Allow fallback to global weight_dtype if specified
        if weight_dtype:
            return True, f"Using global weight_dtype={weight_dtype} (no per-layer quant_summary)"
        return False, "No quant_summary found in sidecar (use --weight-dtype for uniform weights)"

    # Check structure: should have layer.N entries with wq, wk, wv, wo, w1, w2
    layer_count = 0
    valid_layers = 0
    expected_keys = {"wq", "wk", "wv", "wo", "w1", "w2"}

    for key, value in quant_summary.items():
        if key.startswith("layer."):
            layer_count += 1
            if isinstance(value, dict):
                found_keys = set(value.keys())
                if found_keys & expected_keys:
                    valid_layers += 1

    if layer_count == 0:
        return False, "quant_summary has no layer entries (expected layer.0, layer.1, ...)"

    if valid_layers == 0:
        return False, f"quant_summary has {layer_count} layers but none have valid weight dtypes"

    return True, f"quant_summary valid: {valid_layers}/{layer_count} layers with per-op dtypes"


def validate_kernel_registry(registry: Dict[str, Dict]) -> Tuple[bool, str]:
    """Validate that kernel registry is loaded and has kernels.

    Returns:
        (ok, message) tuple.
    """
    if not registry:
        return False, "Kernel registry is empty or not loaded"

    kernel_count = len(registry)
    if kernel_count == 0:
        return False, "Kernel registry has no kernels"

    # Count by op type
    ops = {}
    for k, v in registry.items():
        op = v.get("op", "unknown")
        ops[op] = ops.get(op, 0) + 1

    op_summary = ", ".join(f"{op}:{count}" for op, count in sorted(ops.items()))
    return True, f"Kernel registry valid: {kernel_count} kernels ({op_summary})"


def run_preflight_checks(weights_manifest: Optional[Dict],
                         bump_metadata: Optional[Dict],
                         registry: Dict[str, Dict],
                         quant_summary: Dict[str, Dict[str, str]],
                         graph: Dict,
                         modes: List[str],
                         model_type: Optional[str] = None,
                         weight_dtype: Optional[str] = None) -> bool:
    """Run all preflight validation checks.

    Prints clear status for each check. Returns False if any check fails.
    """
    print("\n" + "=" * 70)
    print("PREFLIGHT CHECKS (v6.6)")
    print("=" * 70)

    all_passed = True

    # Extract model_type and weight_dtype from graph config if not provided
    if graph:
        config = graph.get("config", {})
        if not model_type:
            model_type = config.get("model_type", config.get("model", None))
        if not weight_dtype:
            weight_dtype = config.get("weight_dtype", None)

    # Check 1: Sidecar template
    ok, msg = validate_sidecar_template(weights_manifest, bump_metadata, model_type)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Template: {msg}")
    if not ok:
        all_passed = False

    # Check 2: Sidecar quant_summary or weight_dtype
    ok, msg = validate_sidecar_quant(weights_manifest, bump_metadata, weight_dtype)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Weight info: {msg}")
    if not ok:
        all_passed = False

    # Check 3: Kernel registry
    ok, msg = validate_kernel_registry(registry)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Kernel registry: {msg}")
    if not ok:
        all_passed = False

    # Check 4: Kernel coverage for each mode
    for mode in modes:
        missing = check_kernel_coverage(graph, registry, quant_summary, mode, weight_dtype)
        if missing:
            status = "FAIL"
            all_passed = False
            print(f"[{status}] Kernel coverage ({mode}): {len(missing)} missing kernels")
            for m in missing[:10]:  # Show first 10
                print(f"       - {m}")
            if len(missing) > 10:
                print(f"       ... and {len(missing) - 10} more")
        else:
            print(f"[PASS] Kernel coverage ({mode}): All kernels available")

    print("=" * 70)
    if all_passed:
        print("All preflight checks PASSED - proceeding with IR generation")
    else:
        print("PREFLIGHT FAILED - Cannot generate IR without fixing above issues")
    print("=" * 70 + "\n")

    return all_passed


def check_kernel_coverage(graph: Dict,
                          registry: Dict[str, Dict],
                          quant_summary: Dict[str, Dict[str, str]],
                          mode: str,
                          fallback_weight_dtype: Optional[str] = None) -> List[str]:
    """Check that all required kernels exist in registry.

    Args:
        fallback_weight_dtype: Used when quant_summary is empty (global dtype)

    Returns list of missing kernels (empty = all present).
    """
    missing = []
    config = graph["config"]
    section = graph["sections"][0]
    num_layers = config.get("num_layers", config.get("num_hidden_layers", 0))

    # Use fallback if no quant_summary
    if not quant_summary:
        fallback_weight_dtype = fallback_weight_dtype or config.get("weight_dtype")

    # Ops that use weights (need weight dtype)
    weighted_ops = {"qkv_project", "qkv_proj", "attn_proj", "out_proj", "mlp_up", "mlp_down", "mlp_gate", "mlp_gate_up"}

    # Check header ops
    for op in section.get("header", {}).get("ops", []):
        op_name = op["op"]
        if op_name in HOST_OPS or op_name == "lm_head":
            continue

        # Determine weight dtype
        weight_dtypes = get_op_weight_dtypes(quant_summary, 0, op_name)
        if weight_dtypes:
            # Multi-weight op (e.g., qkv has wq, wk, wv)
            for weight_name, weight_dtype in weight_dtypes.items():
                dtype = weight_dtype or fallback_weight_dtype
                if dtype:
                    if not kernel_exists_in_registry(op_name, dtype, mode, registry):
                        missing.append(
                            f"{op_name}[{weight_name}] weight={dtype} mode={mode}"
                        )
        elif op_name in weighted_ops and fallback_weight_dtype:
            # Use fallback for weighted ops
            if not kernel_exists_in_registry(op_name, fallback_weight_dtype, mode, registry):
                missing.append(f"{op_name} weight={fallback_weight_dtype} mode={mode}")
        else:
            # Non-weighted op
            if not kernel_exists_in_registry(op_name, None, mode, registry):
                missing.append(f"{op_name} mode={mode}")

    # Check layer ops
    for layer_id in range(num_layers):
        # Use ops from template structure
        template = graph.get("template", {})
        if template and "block_types" in template:
            default_block = template["block_types"].get("dense", {})
            if "ops" in default_block:
                # Template ops are strings like "rmsnorm", not dicts
                for op_name in default_block["ops"]:
                    if not op_name or op_name in HOST_OPS:
                        continue

                    weight_dtypes = get_op_weight_dtypes(quant_summary, layer_id, op_name)
                    if weight_dtypes:
                        for weight_name, weight_dtype in weight_dtypes.items():
                            dtype = weight_dtype or fallback_weight_dtype
                            if dtype:
                                if not kernel_exists_in_registry(op_name, dtype, mode, registry):
                                    missing.append(
                                        f"layer.{layer_id}.{op_name}[{weight_name}] weight={dtype} mode={mode}"
                                    )
                    elif op_name in weighted_ops and fallback_weight_dtype:
                        # Use fallback for weighted ops
                        if not kernel_exists_in_registry(op_name, fallback_weight_dtype, mode, registry):
                            missing.append(f"layer.{layer_id}.{op_name} weight={fallback_weight_dtype} mode={mode}")
                    else:
                        # Non-weighted op
                        if not kernel_exists_in_registry(op_name, None, mode, registry):
                            missing.append(f"layer.{layer_id}.{op_name} mode={mode}")

    return missing


def kernel_exists_in_registry(op_name: str,
                              weight_dtype: Optional[str],
                              mode: str,
                              registry: Dict[str, Dict]) -> bool:
    """Check if kernel exists in registry for (op, weight_dtype, mode).

    For projection ops (qkv_project, attn_proj, mlp_up, mlp_down):
      - decode mode -> look for GEMV kernel
      - prefill mode -> look for GEMM kernel

    For non-weight ops (rmsnorm, rope, swiglu, attention):
      - Look for kernel by op name directly
    """
    if not registry:
        return False

    # Build registry index if not done
    if not hasattr(kernel_exists_in_registry, "_index"):
        kernel_exists_in_registry._index = {
            k: v for k, v in registry.items()
        }

    registry_index = kernel_exists_in_registry._index

    # Projection ops use GEMV (decode) or GEMM (prefill)
    # Projection ops use GEMV/GEMM kernels
    projection_ops = {
        "qkv_project", "qkv_proj",
        "attn_proj", "out_proj",  # attention output projection
        "mlp_up", "mlp_down", "mlp_gate", "mlp_gate_up"
    }

    if op_name in projection_ops and weight_dtype:
        # For projections, check for exact GEMV/GEMM kernel
        act_dtype = _get_activation_dtype_for_weight(weight_dtype)

        if mode == "decode":
            # Look for GEMV kernel: gemv_{weight}_{activation}
            target_kernel = f"gemv_{weight_dtype.lower()}_{act_dtype}"
        else:
            # Look for GEMM kernel: gemm_nt_{weight}_{activation}
            target_kernel = f"gemm_nt_{weight_dtype.lower()}_{act_dtype}"

        # Check if exact kernel exists
        if target_kernel in registry_index:
            return True

        # Also check by searching for matching quant
        for kernel_id, entry in registry_index.items():
            kernel_op = entry.get("op", "")

            # Check op type matches mode
            if mode == "decode" and kernel_op != "gemv":
                continue
            if mode != "decode" and kernel_op != "gemm":
                continue

            quant = entry.get("quant", {})
            weight_q = quant.get("weight", "").lower()
            act_q = quant.get("activation", "").lower()

            if weight_q == weight_dtype.lower() and act_q == act_dtype:
                return True

        return False

    # Non-projection ops: map to registry op name
    op_map = {
        "residual_add": "residual_add",
        "rmsnorm": "rmsnorm",
        "ln1": "rmsnorm",
        "ln2": "rmsnorm",
        "attention": "attention",
        "attn": "attention",
        "rope_qk": "rope",
        "rope": "rope",
        "swiglu": "swiglu",
        "silu_mul": "swiglu",
        "embedding": "embedding",
    }

    registry_op = op_map.get(op_name, op_name)

    # Find kernels matching this op
    for kernel_id, entry in registry_index.items():
        if entry.get("op") != registry_op:
            continue

        # Check dtype compatibility (for ops with weights like embedding)
        quant = entry.get("quant", {})
        weight_support = quant.get("weight", "")

        if weight_dtype:
            if "mixed" in weight_support.lower():
                return True
            if weight_dtype.lower() in weight_support.lower():
                return True
            # Also check exact match
            if weight_dtype.lower() == weight_support.lower():
                return True
        else:
            # No weight dtype constraint - kernel exists for non-weighted ops
            return True

    return False


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

    bootstrap_manifest = None
    if args.get("weights_manifest"):
        with open(args["weights_manifest"], "r") as f:
            bootstrap_manifest = json.load(f)

    bootstrap_bump_metadata = None
    if args.get("bump"):
        bootstrap_bump_metadata = read_bump_metadata(args["bump"])

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
        raw_config = v3.parse_config(config_path)
        # Normalize config keys (HF uses different names than BUMP)
        config = normalize_bump_config(raw_config)
        try:
            config["num_merges"] = int(raw_config.get("num_merges", config.get("num_merges", 0)) or 0)
            config["total_vocab_bytes"] = int(raw_config.get("total_vocab_bytes", config.get("total_vocab_bytes", 0)) or 0)
        except Exception:
            pass
        model_name = args["name"] or config.get("model_type", "model")
    elif args.get("model"):
        model_id = v3.parse_hf_model_id(args["model"])
        raw_config, cached_path = v3.download_hf_config(model_id)
        raw_config = v3.parse_config(cached_path)
        config = normalize_bump_config(raw_config)
        model_name = args["name"] or v3.model_id_to_name(model_id)
    else:
        manifest_cfg = extract_config_from_manifest(bootstrap_manifest)
        bump_cfg = normalize_bump_config(
            bootstrap_bump_metadata.get("config", {}) if isinstance(bootstrap_bump_metadata, dict) else {}
        )
        config = manifest_cfg or bump_cfg
        if not config:
            raise ValueError("No config found in manifest or BUMP metadata; pass --config or --preset")
        if "dtype" not in config:
            config["dtype"] = "fp32"
        model_name = args["name"] or config.get("model_type", "model")

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
        output_dir = os.path.join("build", f"{safe_name}_v6")

    print(f"[MODEL]  {model_name}")
    print(f"[OUTPUT] {output_dir}/")

    # =========================================================================
    # KERNEL REGISTRY
    # =========================================================================
    # The kernel registry maps kernel names to their buffer specifications,
    # constraints, and fusion relationships. It's used for:
    # 1. Validating that required buffers are allocated
    # 2. Selecting appropriate kernels for a given operation
    # 3. Fusion pass (finding fused kernels that replace multiple ops)
    registry = load_kernel_registry()
    print(f"[KERNELS] Loaded {len(registry)} kernel specs")

    # =========================================================================
    # TEMPLATE LOADING
    # =========================================================================
    # The template defines the IR graph structure:
    # - block_types: Named operation sequences (e.g., "dense", "moe")
    # - layer_map: Which block type to use per layer (with overrides)
    # - flags: Architecture flags (use_qkv_bias, activation, rope, etc.)
    #
    # Priority for template source (highest to lowest):
    # 1. --template=NAME/FILE: Explicit override from file system
    # 2. --weights-manifest=FILE: Use embedded template (sidecar source of truth)
    # 3. --bump=FILE: Read template from BUMPWGT5 EOF metadata
    # 4. Default: Use hardcoded _get_default_body_ops()
    #
    # The weights manifest is the preferred source because it is produced
    # alongside the weights and is used by the loader/runtime mapping.
    # =========================================================================
    template = None
    bump_metadata = bootstrap_bump_metadata

    # -------------------------------------------------------------------------
    # STEP 1: If --bump is provided, read metadata from BUMPWGT5 EOF footer
    # -------------------------------------------------------------------------
    if args.get("bump") and bump_metadata is None:
        print(f"[BUMP] Reading metadata from: {args['bump']}")
        bump_metadata = read_bump_metadata(args["bump"])

    # -------------------------------------------------------------------------
    # STEP 2: Read weights manifest early (drives template/config when present)
    # -------------------------------------------------------------------------
    weights_manifest = bootstrap_manifest
    if args.get("weights_manifest") and weights_manifest is None:
        print(f"[WEIGHTS] Reading manifest: {args['weights_manifest']}")
        with open(args["weights_manifest"], "r") as f:
            weights_manifest = json.load(f)

        # Validate manifest hash against BUMP metadata (if available)
        if bump_metadata and bump_metadata.get("manifest_hash"):
            expected_hash = bump_metadata["manifest_hash"]
            actual_hash = hashlib.sha256(canonical_json_bytes(weights_manifest)).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    "weights_manifest.json hash mismatch for BUMPWGT5 metadata "
                    f"(expected {expected_hash}, got {actual_hash})"
                )
            print("[WEIGHTS] Manifest hash matches BUMP metadata")

    # -------------------------------------------------------------------------
    # STEP 3: Select template (CLI > manifest > BUMP > default)
    # -------------------------------------------------------------------------
    manifest_template = weights_manifest.get("template") if isinstance(weights_manifest, dict) else None
    bump_template = bump_metadata.get("template") if isinstance(bump_metadata, dict) else None

    if args.get("template"):
        template_path = args["template"]
        if not os.path.exists(template_path):
            template_path = os.path.join("version", "v6.6", "templates", f"{args['template']}.json")
        if os.path.exists(template_path):
            print(f"[TEMPLATE] Loading: {template_path}")
            with open(template_path, "r") as f:
                template = json.load(f)
            print(f"[TEMPLATE] Loaded template: {template.get('name', 'unknown')}")
        else:
            print(f"[TEMPLATE] Warning: template not found: {template_path}")
    elif isinstance(manifest_template, dict):
        template = manifest_template
        print(f"[TEMPLATE] Using template from weights manifest: {template.get('name', 'unknown')}")
    elif isinstance(bump_template, dict):
        template = bump_template
        print(f"[TEMPLATE] Using template from BUMP metadata: {template.get('name', 'unknown')}")
    else:
        # v6.6: Try to load template from templates/ directory based on model_type
        model_type = config.get("model_type", "unknown")
        templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
        template_path = os.path.join(templates_dir, f"{model_type}.json")

        if os.path.exists(template_path):
            print(f"[TEMPLATE] Loading from templates directory: {template_path}")
            with open(template_path, "r") as f:
                template = json.load(f)
            print(f"[TEMPLATE] Loaded template: {template.get('name', model_type)}")
        else:
            # No template found anywhere
            raise ValueError(
                f"No template found for model_type '{model_type}'. "
                f"v6.6 requires template in weights_manifest, BUMP metadata, or templates/{model_type}.json. "
                f"Options:\n"
                f"  1. Re-run convert_gguf_to_bump_v6_6.py with --bump-version=5 to embed template\n"
                f"  2. Provide --template=PATH to a template JSON file\n"
                f"  3. Add 'template' section to weights_manifest.json\n"
                f"  4. Create version/v6.6/templates/{model_type}.json"
            )

    if isinstance(manifest_template, dict) and isinstance(bump_template, dict):
        manifest_name = manifest_template.get("name")
        bump_name = bump_template.get("name")
        if manifest_name and bump_name and manifest_name != bump_name:
            raise ValueError(
                f"Template mismatch: manifest '{manifest_name}' vs BUMP '{bump_name}'"
            )

    # -------------------------------------------------------------------------
    # STEP 4: Apply config overrides (manifest takes precedence)
    # -------------------------------------------------------------------------
    manifest_cfg = normalize_bump_config(weights_manifest.get("config", {})) if isinstance(weights_manifest, dict) else {}
    bump_cfg = normalize_bump_config(bump_metadata.get("config", {})) if isinstance(bump_metadata, dict) else {}

    if manifest_cfg and bump_cfg:
        mismatches = []
        for key in sorted(set(manifest_cfg) & set(bump_cfg)):
            if manifest_cfg[key] != bump_cfg[key]:
                mismatches.append((key, manifest_cfg[key], bump_cfg[key]))
        if mismatches:
            msg = ", ".join(f"{k}={a} vs {b}" for k, a, b in mismatches)
            raise ValueError(f"Config mismatch between manifest and BUMP metadata: {msg}")

    if manifest_cfg:
        print(f"[CONFIG] Applying {len(manifest_cfg)} fields from weights manifest")
        for key, value in manifest_cfg.items():
            if value != config.get(key):
                print(f"[CONFIG]   Override {key}: {config.get(key)} -> {value}")
                config[key] = value

    if bump_cfg:
        print(f"[CONFIG] Applying {len(bump_cfg)} fields from BUMP metadata")
        for key, value in bump_cfg.items():
            if key not in manifest_cfg and value != config.get(key):
                print(f"[CONFIG]   Override {key}: {config.get(key)} -> {value}")
                config[key] = value

    # =========================================================================
    # WEIGHTS METADATA
    # =========================================================================
    # Weights can come from multiple sources:
    #   - Safetensors header (.safetensors)
    #   - Weights index (model.safetensors.index.json)
    #   - Weights manifest (weights_manifest.json from convert_*_to_bump.py)
    #
    # The manifest is critical for quantized inference:
    #   - Maps tensor names to file offsets
    #   - Specifies dtype per tensor (q4_k, q5_0, q8_0, etc.)
    #   - Used by the weight loader to map tensors to memory
    #
    # If BUMP metadata is provided, we validate the manifest hash matches
    # the canonical JSON hash stored in the BUMP footer.
    # =========================================================================
    weights_meta = {}
    if args["weights_header"]:
        print(f"[WEIGHTS] Reading safetensors header: {args['weights_header']}")
        weights_meta["header"] = read_safetensors_header(args["weights_header"])
    if args["weights_index"]:
        print(f"[WEIGHTS] Reading index: {args['weights_index']}")
        weights_meta["index"] = read_weights_index(args["weights_index"])
    # Determine weight dtype from manifest if not explicitly specified
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

    quant_summary = extract_quant_summary(weights_manifest, bump_metadata)
    if quant_summary:
        print(f"[QUANT] Loaded {len(quant_summary)} per-layer dtype entries")

    # Graph IR
    graph = build_graph_ir_v6(config, model_name, template=template)
    if quant_summary:
        graph["quant_summary"] = quant_summary
    if weights_meta:
        graph["weights"] = weights_meta

    # =========================================================================
    # PREFLIGHT CHECKS (v6.6)
    # =========================================================================
    # Before generating IR, validate:
    #   1. Sidecar has template (sequence of ops)
    #   2. Sidecar has quant_summary (per-layer, per-op dtypes)
    #   3. Kernel registry is loaded
    #   4. All required kernels exist for requested modes
    #
    # If any check fails, stop with clear error - cannot generate IR without
    # proper kernel coverage.
    # =========================================================================
    modes_arg = args.get("modes", ["prefill", "decode"])
    requested_modes = modes_arg if isinstance(modes_arg, list) else modes_arg.split(",")

    preflight_passed = run_preflight_checks(
        weights_manifest=weights_manifest,
        bump_metadata=bump_metadata,
        registry=registry,
        quant_summary=quant_summary,
        graph=graph,
        modes=requested_modes,
    )

    if not preflight_passed:
        raise RuntimeError(
            "Preflight checks failed. Cannot generate IR without:\n"
            "  - Template in sidecar (weights_manifest.json or BUMP metadata)\n"
            "  - quant_summary in sidecar (per-layer weight dtypes)\n"
            "  - All required kernels in KERNEL_REGISTRY.json\n"
            "\n"
            "Fix the above issues and re-run. To add missing kernels:\n"
            "  1. Create kernel map JSON in version/v6.6/kernel_maps/\n"
            "  2. Run: python3 version/v6.6/scripts/gen_kernel_registry_from_maps.py"
        )

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
    prefill_layout = None
    for mode in args["modes"]:
        mode_tokens = 1 if mode == "decode" else tokens

        # For training mode, use computed context length
        if mode == "training" and training_cfg:
            mode_tokens = training_cfg.context_length

        lowered = lower_graph_ir(graph, mode, mode_tokens, registry, training_cfg,
                                 weights_manifest=weights_manifest,
                                 weight_dtype=args.get("weight_dtype"))

        def emit_mode_outputs(enable_fusion: bool) -> None:
            nonlocal prefill_layout
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

            if mode == "prefill":
                prefill_layout = layout

            layout_for_codegen = layout
            if mode == "decode" and args.get("decode_layout") == "prefill":
                if prefill_layout is None:
                    print("[WARN] decode-layout=prefill requested but prefill layout not available; using decode layout.")
                else:
                    layout_for_codegen = copy.deepcopy(prefill_layout)
                    layout_for_codegen.name = layout_name

            layout_json_path = os.path.join(output_dir, f"layout_{mode}.json")
            layout_map = os.path.join(output_dir, f"layout_{mode}.map")

            v3.emit_layout_json(layout_for_codegen, layout_json_path)
            v3.emit_layout_map(layout_for_codegen, layout_map)

            if weights_manifest and mode in ("prefill", "decode"):
                emit_weights_manifest(layout_for_codegen, weights_manifest, output_dir)

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
            kernel_base = {
                "decode": "ck-kernel-inference",
                "prefill": "ck-kernel-prefill",
                "backward": "ck-kernel-backprop",
                "training": "ck-kernel-backprop",
            }.get(mode, f"ck-kernel-{mode}")
            header_name = f"{kernel_base}.h"
            source_name = f"{kernel_base}.c"
            extra_api = None
            if mode == "decode":
                extra_api = [
                    f"/* Model struct - shared between model.c and kernel implementation */",
                    f"typedef struct {{",
                    f"    void *weights_base;",
                    f"    size_t weights_size;",
                    f"    size_t total_bytes;",
                    f"    uint64_t vocab_offsets;",
                    f"    uint64_t vocab_strings;",
                    f"    uint64_t vocab_merges;",
                    f"    uint64_t logits;",
                    f"}} {safe_name_upper}Model;",
                    f"",
                    f"/* Kernel API */",
                    f"int {safe_name.lower()}_model_allocate({safe_name_upper}Model *model);",
                    f"void {safe_name.lower()}_model_free({safe_name_upper}Model *model);",
                    f"void {safe_name.lower()}_precompute_rope({safe_name_upper}Model *model);",
                    f"void {safe_name.lower()}_forward({safe_name_upper}Model *model, const int *tokens, int num_tokens);",
                    f"void {safe_name.lower()}_decode({safe_name_upper}Model *model, const int *token, int token_index);",
                    f"int {safe_name.lower()}_verify_canaries({safe_name_upper}Model *model);",
                ]
            v3.emit_c_header(layout_for_codegen, os.path.join(output_dir, header_name), extra_api=extra_api)
            if mode in ("prefill", "decode"):
                codegen_version = args.get("codegen", "v6")
                int8_activations = args.get("int8", True)
                if codegen_version == "v6":
                    if int8_activations:
                        print(f"[CODEGEN] Using v6.6 (explicit unrolled + INT8 activations) for {mode}")
                        print("[CODEGEN] INT8 activations enabled (Q5_0 x Q8_0, Q8_0 x Q8_0, Q4_K x Q8_K)")
                    else:
                        print(f"[CODEGEN] Using v6.6 (explicit unrolled + FP32 activations) for {mode}")
                        print("[CODEGEN] INT8 disabled, using FP32 GEMM kernels")
                    codegen_v6.emit_c_source_v6(
                        layout_for_codegen,
                        os.path.join(output_dir, source_name),
                        header_name,
                        mode,
                        emit_main=(args.get("emit") == "exe"),
                        emit_debug=args.get("debug", False),
                        emit_parity=args.get("parity", False),
                        weights_manifest=weights_manifest if not args.get("skip_manifest_validation") else None,
                        int8_activations=int8_activations,
                    )
                else:
                    # Legacy v4 codegen (requires compat module)
                    # codegen_v4.emit_c_source_v4(...)
                    print("[ERROR] v4 codegen not available in this standalone build")
                    raise NotImplementedError("v4 codegen requires compat_codegen_v4_v6_6 module")
            else:
                v3.emit_c_source(
                    layout,
                    os.path.join(output_dir, source_name),
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

    print("[DONE] IR v6 pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
