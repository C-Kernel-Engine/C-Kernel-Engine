#!/usr/bin/env python3
"""
=============================================================================
EXPERIMENTAL/FUTURE - NOT USED BY CURRENT v6.6 PIPELINE
=============================================================================
This file is part of the experimental op_builders system for future IR2 work.
It is NOT called by ck_run_v6_6.py or the current build pipeline.

Related files: parse_template_v2.py, op_builders_auto.py, op_builders_hybrid_v6_6.py
Current pipeline uses: build_ir_v6_6.py with kernel maps directly.
=============================================================================

op_builders_v6_6.py - Op Builders for Template v2

Maps template operations (OpNode) to GraphIR Op objects with symbolic tensor names.

USAGE:
    from op_builders_v6_6 import build_op_from_template, OpContext

    ctx = OpContext(config)
    for op_node in template_ops:
        ir_op = build_op_from_template(op_node, ctx, manifest)

DESIGN:
    - OpNode (from template) → Op (GraphIR with symbolic names)
    - OpContext tracks state (previous outputs, counters)
    - Each builder knows how to wire one op type
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ir_types_v6_6 import Op
from parse_template_v2 import OpNode


# =============================================================================
# OP CONTEXT: State Tracker
# =============================================================================

class OpContext:
    """
    Tracks state during op building.

    Maintains:
    - Previous op outputs (for input chaining)
    - Op counters (for unique naming)
    - Current tensor (for residual connections)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prev_output: Optional[str] = None
        self.residual_input: Optional[str] = None

        # Per-layer counters (reset each layer)
        self.rmsnorm_count = 0
        self.residual_count = 0

    def set_output(self, output: str):
        """Update previous output for next op."""
        self.prev_output = output

    def save_residual(self):
        """Save current output for residual connection."""
        self.residual_input = self.prev_output

    def reset_layer(self):
        """Reset counters for new layer."""
        self.rmsnorm_count = 0
        self.residual_count = 0
        self.residual_input = None


# =============================================================================
# TENSOR NAMING HELPERS
# =============================================================================

def make_tensor_name(op_node: OpNode, suffix: str) -> str:
    """Generate symbolic tensor name."""
    if op_node.phase == "body" and op_node.layer_index is not None:
        return f"layer.{op_node.layer_index}.{suffix}"
    else:
        # Header/footer: global names
        return suffix


def make_weight_name(op_node: OpNode, weight: str) -> str:
    """Generate weight tensor name."""
    if op_node.phase == "body" and op_node.layer_index is not None:
        return f"layer.{op_node.layer_index}.{weight}"
    else:
        return weight


# =============================================================================
# OP BUILDERS: Template Op → GraphIR Op
# =============================================================================

def build_tokenizer_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build tokenizer op (header)."""
    op = Op(
        op="tokenizer",
        name="tokenizer",
        inputs=["input_ids"],
        outputs=["input_tokens"],
        params={"vocab_size": ctx.config.get("vocab_size", 32000)}
    )
    ctx.set_output("input_tokens")
    return op


def build_embedding_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build embedding lookup (header)."""
    output = make_tensor_name(op_node, "embed_out")

    op = Op(
        op="embedding",  # Matches kernel registry
        name="embedding",
        inputs=[ctx.prev_output or "input_tokens"],
        outputs=[output],
        weights=["token_emb"],
        params={
            "vocab_size": ctx.config.get("vocab_size", 32000),
            "embed_dim": ctx.config.get("embed_dim", 512)
        }
    )
    ctx.set_output(output)
    return op


def build_rmsnorm_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build RMSNorm op."""
    ctx.rmsnorm_count += 1
    ln_id = ctx.rmsnorm_count

    input_name = ctx.prev_output or make_tensor_name(op_node, "input")
    output_name = make_tensor_name(op_node, f"ln{ln_id}_out")
    weight_name = make_weight_name(op_node, f"ln{ln_id}_gamma")

    # Save input for first residual
    if ctx.rmsnorm_count == 1:
        ctx.save_residual()

    op = Op(
        op="rmsnorm",
        name=make_tensor_name(op_node, f"ln{ln_id}"),
        inputs=[input_name],
        outputs=[output_name],
        weights=[weight_name],
        params={"eps": ctx.config.get("rms_eps", 1e-5)}
    )
    ctx.set_output(output_name)
    return op


def build_qkv_proj_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build QKV projection."""
    input_name = ctx.prev_output
    q_out = make_tensor_name(op_node, "q")
    k_out = make_tensor_name(op_node, "k")
    v_out = make_tensor_name(op_node, "v")

    wq = make_weight_name(op_node, "wq")
    wk = make_weight_name(op_node, "wk")
    wv = make_weight_name(op_node, "wv")

    op = Op(
        op="qkv_projection",  # Matches kernel registry
        name=make_tensor_name(op_node, "qkv"),
        inputs=[input_name],
        outputs=[q_out, k_out, v_out],
        weights=[wq, wk, wv],
        params={
            "num_heads": ctx.config.get("num_heads", 8),
            "num_kv_heads": ctx.config.get("num_kv_heads", 8),
            "head_dim": ctx.config.get("head_dim", 64)
        }
    )
    ctx.set_output(q_out)
    return op


def build_rope_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build RoPE (in-place on Q/K)."""
    q_name = make_tensor_name(op_node, "q")
    k_name = make_tensor_name(op_node, "k")

    op = Op(
        op="rope",
        name=make_tensor_name(op_node, "rope"),
        inputs=[q_name, k_name, "rope_cos_cache", "rope_sin_cache"],
        outputs=[q_name, k_name],  # In-place
        params={
            "rope_theta": ctx.config.get("rope_theta", 10000.0),
            "head_dim": ctx.config.get("head_dim", 64)
        }
    )
    return op


def build_attention_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build attention op."""
    q_name = make_tensor_name(op_node, "q")
    k_name = make_tensor_name(op_node, "k")
    v_name = make_tensor_name(op_node, "v")
    output_name = make_tensor_name(op_node, "attn_out")
    scores_name = make_tensor_name(op_node, "scores")

    op = Op(
        op="attention",
        name=make_tensor_name(op_node, "attn"),
        inputs=[q_name, k_name, v_name],
        outputs=[output_name],
        scratch=[scores_name],
        params={
            "num_heads": ctx.config.get("num_heads", 8),
            "num_kv_heads": ctx.config.get("num_kv_heads", 8),
            "head_dim": ctx.config.get("head_dim", 64)
        }
    )
    ctx.set_output(output_name)
    return op


def build_out_proj_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build attention output projection."""
    input_name = ctx.prev_output
    output_name = make_tensor_name(op_node, "attn_proj_out")
    weight_name = make_weight_name(op_node, "wo")

    op = Op(
        op="gemv",  # Linear projection - matches kernel registry
        name=make_tensor_name(op_node, "out_proj"),
        inputs=[input_name],
        outputs=[output_name],
        weights=[weight_name],
        params={"embed_dim": ctx.config.get("embed_dim", 512)}
    )
    ctx.set_output(output_name)
    return op


def build_residual_add_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build residual add."""
    ctx.residual_count += 1

    input1 = ctx.residual_input
    input2 = ctx.prev_output
    output_name = make_tensor_name(op_node, f"residual{ctx.residual_count}_out")

    op = Op(
        op="residual_add",  # Matches kernel registry
        name=make_tensor_name(op_node, f"residual{ctx.residual_count}"),
        inputs=[input1, input2],
        outputs=[output_name]
    )
    ctx.set_output(output_name)

    # Save for next residual
    if ctx.residual_count == 1:
        ctx.save_residual()

    return op


def build_mlp_gate_up_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build MLP gate+up projection (fused)."""
    input_name = ctx.prev_output
    output_name = make_tensor_name(op_node, "mlp_gate_up_out")
    weight_name = make_weight_name(op_node, "w1")

    op = Op(
        op="fused_mlp_block",  # Matches kernel registry (fused gate+up)
        name=make_tensor_name(op_node, "mlp_gate_up"),
        inputs=[input_name],
        outputs=[output_name],
        weights=[weight_name],
        params={
            "embed_dim": ctx.config.get("embed_dim", 512),
            "intermediate_size": ctx.config.get("intermediate_size", 2048)
        }
    )
    ctx.set_output(output_name)
    return op


def build_silu_mul_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build SiLU + multiply (SwiGLU)."""
    input_name = ctx.prev_output
    output_name = make_tensor_name(op_node, "mlp_act_out")

    op = Op(
        op="swiglu",  # Matches kernel registry (SiLU + multiply)
        name=make_tensor_name(op_node, "silu_mul"),
        inputs=[input_name],
        outputs=[output_name]
    )
    ctx.set_output(output_name)
    return op


def build_mlp_down_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build MLP down projection."""
    input_name = ctx.prev_output
    output_name = make_tensor_name(op_node, "mlp_out")
    weight_name = make_weight_name(op_node, "w2")

    op = Op(
        op="gemv",  # Linear projection - matches kernel registry
        name=make_tensor_name(op_node, "mlp_down"),
        inputs=[input_name],
        outputs=[output_name],
        weights=[weight_name],
        params={
            "embed_dim": ctx.config.get("embed_dim", 512),
            "intermediate_size": ctx.config.get("intermediate_size", 2048)
        }
    )
    ctx.set_output(output_name)
    return op


def build_weight_tying_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build weight tying (metadata op)."""
    return Op(
        op="weight_tying",
        name="weight_tying",
        inputs=[],
        outputs=[],
        params={"tied": ctx.config.get("tie_word_embeddings", True)}
    )


def build_logits_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build final logits projection (footer)."""
    input_name = ctx.prev_output

    op = Op(
        op="gemv",  # Linear projection - matches kernel registry
        name="lm_head",
        inputs=[input_name],
        outputs=["logits"],
        weights=["lm_head_weight"],
        params={
            "embed_dim": ctx.config.get("embed_dim", 512),
            "vocab_size": ctx.config.get("vocab_size", 32000)
        }
    )
    ctx.set_output("logits")
    return op


# =============================================================================
# OP BUILDER REGISTRY
# =============================================================================

OP_BUILDERS = {
    "tokenizer": build_tokenizer_op,
    "dense_embedding_lookup": build_embedding_op,
    "rmsnorm": build_rmsnorm_op,
    "qkv_proj": build_qkv_proj_op,
    "rope_qk": build_rope_op,
    "attn": build_attention_op,
    "out_proj": build_out_proj_op,
    "residual_add": build_residual_add_op,
    "mlp_gate_up": build_mlp_gate_up_op,
    "silu_mul": build_silu_mul_op,
    "mlp_down": build_mlp_down_op,
    "weight_tying": build_weight_tying_op,
    "logits": build_logits_op,
}


# =============================================================================
# PUBLIC API
# =============================================================================

def build_op_from_template(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Optional[Op]:
    """
    Build GraphIR Op from template OpNode.

    Args:
        op_node: OpNode from parse_template_v2
        ctx: OpContext for state tracking
        manifest: Weights manifest

    Returns:
        Op object or None if op should be skipped

    Raises:
        ValueError: If op_id has no builder
    """
    if op_node.op_id not in OP_BUILDERS:
        raise ValueError(
            f"No builder for template op: '{op_node.op_id}'. "
            f"Supported ops: {', '.join(sorted(OP_BUILDERS.keys()))}"
        )

    builder = OP_BUILDERS[op_node.op_id]
    return builder(op_node, ctx, manifest)


def get_supported_ops() -> List[str]:
    """Get list of supported template op IDs."""
    return sorted(OP_BUILDERS.keys())


def check_template_ops(template_ops: List[str]) -> List[str]:
    """
    Check which template ops are missing builders.

    Args:
        template_ops: List of op IDs from template

    Returns:
        List of unsupported op IDs
    """
    supported = set(OP_BUILDERS.keys())
    return [op for op in template_ops if op not in supported]
