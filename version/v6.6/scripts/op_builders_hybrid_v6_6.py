#!/usr/bin/env python3
"""
op_builders_hybrid_v6_6.py - Hybrid Op Builders (Auto + Manual)

DESIGN:
    - Auto-generated builders from kernel registry (guaranteed to match kernels)
    - Manual builders for special/metadata ops (tokenizer, weight_tying)
    - Template op mapping to kernel ops

USAGE:
    from op_builders_hybrid_v6_6 import build_op_from_template, OpContext
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent to path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Import auto-generated builders
from op_builders_auto import (
    OpContext,
    make_tensor_name,
    make_weight_name,
    OP_BUILDERS as AUTO_BUILDERS
)
from ir_types_v6_6 import Op
from parse_template_v2 import OpNode


# =============================================================================
# TEMPLATE OP → KERNEL OP MAPPING
# =============================================================================

TEMPLATE_TO_KERNEL_MAP = {
    # Header ops
    "tokenizer": "tokenizer",  # Special: metadata only
    "dense_embedding_lookup": "embedding",

    # Attention block
    "rmsnorm": "rmsnorm",
    "qkv_proj": "qkv_projection",
    "rope_qk": "rope",
    "attn": "attention",
    "out_proj": "gemv",  # Linear projection

    # Residual
    "residual_add": "residual_add",

    # MLP block
    "mlp_gate_up": "fused_mlp_block",
    "silu_mul": "swiglu",
    "mlp_down": "gemv",  # Linear projection

    # Footer ops
    "weight_tying": "weight_tying",  # Special: metadata only
    "logits": "gemv",  # Linear projection
}


# =============================================================================
# MANUAL BUILDERS FOR SPECIAL OPS
# =============================================================================

def build_tokenizer_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build tokenizer op (metadata only)."""
    op = Op(
        op="tokenizer",
        name="tokenizer",
        inputs=["input_ids"],
        outputs=["input_tokens"],
        params={"vocab_size": ctx.config.get("vocab_size", 32000)}
    )
    ctx.set_output("input_tokens")
    return op


def build_weight_tying_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build weight tying (metadata only)."""
    return Op(
        op="weight_tying",
        name="weight_tying",
        inputs=[],
        outputs=[],
        params={"tied": ctx.config.get("tie_word_embeddings", True)}
    )


# =============================================================================
# CONTEXT-AWARE WRAPPERS
# =============================================================================

def build_rmsnorm_op_wrapper(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Wrapper for rmsnorm with counter tracking."""
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


def build_residual_add_op_wrapper(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Wrapper for residual_add with counter tracking."""
    ctx.residual_count += 1

    input1 = ctx.residual_input
    input2 = ctx.prev_output
    output_name = make_tensor_name(op_node, f"residual{ctx.residual_count}_out")

    op = Op(
        op="residual_add",
        name=make_tensor_name(op_node, f"residual{ctx.residual_count}"),
        inputs=[input1, input2],
        outputs=[output_name]
    )
    ctx.set_output(output_name)

    # Save for next residual
    if ctx.residual_count == 1:
        ctx.save_residual()

    return op


# =============================================================================
# UNIFIED OP BUILDER REGISTRY
# =============================================================================

# Start with auto-generated builders
OP_BUILDERS = AUTO_BUILDERS.copy()

# Override with manual/wrapper builders for special cases
OP_BUILDERS.update({
    "tokenizer": build_tokenizer_op,
    "weight_tying": build_weight_tying_op,
    "rmsnorm": build_rmsnorm_op_wrapper,
    "residual_add": build_residual_add_op_wrapper,
})


# =============================================================================
# PUBLIC API
# =============================================================================

def build_op_from_template(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Optional[Op]:
    """
    Build GraphIR Op from template OpNode.

    Maps template op IDs to kernel op types and uses appropriate builder.

    Args:
        op_node: OpNode from parse_template_v2
        ctx: OpContext for state tracking
        manifest: Weights manifest

    Returns:
        Op object or None if op should be skipped

    Raises:
        ValueError: If op_id has no mapping or builder
    """
    template_op = op_node.op_id

    # Map template op to kernel op
    if template_op not in TEMPLATE_TO_KERNEL_MAP:
        raise ValueError(
            f"No mapping for template op: '{template_op}'. "
            f"Supported template ops: {', '.join(sorted(TEMPLATE_TO_KERNEL_MAP.keys()))}"
        )

    kernel_op = TEMPLATE_TO_KERNEL_MAP[template_op]

    # Find builder
    if kernel_op not in OP_BUILDERS:
        raise ValueError(
            f"No builder for kernel op: '{kernel_op}' (from template op '{template_op}'). "
            f"Available builders: {', '.join(sorted(OP_BUILDERS.keys()))}"
        )

    # Call builder
    builder = OP_BUILDERS[kernel_op]
    op = builder(op_node, ctx, manifest)

    # Ensure op.op matches kernel registry
    if op and op.op != kernel_op:
        # Fix op type if builder didn't set it correctly
        op.op = kernel_op

    return op


def get_supported_template_ops() -> List[str]:
    """Get list of supported template op IDs."""
    return sorted(TEMPLATE_TO_KERNEL_MAP.keys())


def get_supported_kernel_ops() -> List[str]:
    """Get list of supported kernel op types."""
    return sorted(OP_BUILDERS.keys())


def check_template_ops(template_ops: List[str]) -> List[str]:
    """
    Check which template ops are missing mappings or builders.

    Args:
        template_ops: List of op IDs from template

    Returns:
        List of unsupported op IDs
    """
    missing = []
    for op in template_ops:
        if op not in TEMPLATE_TO_KERNEL_MAP:
            missing.append(op)
        else:
            kernel_op = TEMPLATE_TO_KERNEL_MAP[op]
            if kernel_op not in OP_BUILDERS:
                missing.append(f"{op} → {kernel_op} (no builder)")

    return missing


def get_kernel_coverage_report() -> str:
    """Generate report of template→kernel op coverage."""
    lines = []
    lines.append("Template Op → Kernel Op Mapping:")
    lines.append("=" * 60)

    for t_op in sorted(TEMPLATE_TO_KERNEL_MAP.keys()):
        k_op = TEMPLATE_TO_KERNEL_MAP[t_op]
        has_builder = k_op in OP_BUILDERS

        if has_builder:
            if k_op in AUTO_BUILDERS:
                status = "✓ (auto)"
            else:
                status = "✓ (manual)"
        else:
            status = "✗ NO BUILDER"

        lines.append(f"  {status} {t_op:30s} → {k_op}")

    lines.append("")
    lines.append(f"Total template ops: {len(TEMPLATE_TO_KERNEL_MAP)}")
    lines.append(f"Auto-generated: {len(AUTO_BUILDERS)}")
    lines.append(f"Manual: {len([k for k in OP_BUILDERS if k not in AUTO_BUILDERS])}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Print coverage report when run directly
    print(get_kernel_coverage_report())
