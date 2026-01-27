#!/usr/bin/env python3
"""
codegen_prefill_v6_6.py - Generate C code for PREFILL mode from lowered IR.

This generates ck_prefill() which processes multiple tokens at once.
The IR (lowered_prefill_call.json) already has function names and expressions.
We just substitute num_tokens for const:1 sources.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def emit_prefill_op(op: Dict, seq_idx: int, config: Dict) -> str:
    """Emit a single op call for prefill mode.

    The IR already provides:
      - function: kernel function name
      - args[]: each with name, source, expr

    We just substitute num_tokens for const:1 and fix memcpy size.
    """
    func = op.get("function", "unknown")
    op_type = op.get("op", "unknown")
    layer = op.get("layer", -1)
    args_list = op.get("args", [])

    embed_dim = config.get("embed_dim", 896)

    lines = []
    lines.append(f"    /* Op {seq_idx}: {func} ({op_type}) layer={layer} */")

    # Build argument list with substitutions
    args = []
    for arg in args_list:
        expr = arg.get("expr", "0")
        source = arg.get("source", "")
        name = arg.get("name", "")

        # Substitute num_tokens for token count parameters
        if source == "const:1":
            expr = "num_tokens"
        elif source == "dim:seq_len":
            expr = "num_tokens"
        # For memcpy size, compute dynamically
        elif source == "dim:_memcpy_bytes" and op_type == "residual_save":
            expr = f"(size_t)num_tokens * {embed_dim} * sizeof(float)"
        # For GEMM M dimension (batch size), use num_tokens
        elif source == "dim:_m" and name == "M":
            expr = "num_tokens"

        args.append(expr)

    # For quantize ops: use batch versions which output row-major Q8 data
    # quantize_row_q8_0(x, y, k) -> quantize_batch_q8_0(x, y, num_tokens, k)
    if func == "quantize_row_q8_0":
        func = "quantize_batch_q8_0"
        # Insert num_tokens as 3rd argument (before k)
        args.insert(2, "num_tokens")
    elif func == "quantize_row_q8_k":
        func = "quantize_batch_q8_k"
        args.insert(2, "num_tokens")

    # Format the function call
    if len(args) <= 3:
        # Short call on one line
        lines.append(f"    {func}({', '.join(args)});")
    else:
        # Multi-line for readability
        lines.append(f"    {func}(")
        for i, arg in enumerate(args):
            comma = "," if i < len(args) - 1 else ""
            lines.append(f"        {arg}{comma}")
        lines.append(f"    );")

    return "\n".join(lines)


def emit_prefill_function(ops: List[Dict], config: Dict) -> str:
    """Emit the prefill function with all ops unrolled."""
    lines = []
    lines.append("""
/* ============================================================================
 * PREFILL - Batched processing from IR Lower (prefill mode)
 * ============================================================================ */
static void ck_prefill(CKModel *model, const int32_t *tokens, int num_tokens) {
    if (!model || !tokens || num_tokens <= 0) return;

    /* Clamp to max context */
    if (num_tokens > MAX_SEQ_LEN) num_tokens = MAX_SEQ_LEN;

    const char *stop_env = getenv("CK_STOP_OP");
    int stop_seq = stop_env ? atoi(stop_env) : -1;

    /* Copy input tokens to activation buffer (follow same pattern as decode) */
    memcpy((void*)(model->bump + A_TOKEN_IDS), tokens, (size_t)num_tokens * sizeof(int32_t));
""")

    for seq_idx, op in enumerate(ops):
        lines.append(emit_prefill_op(op, seq_idx, config))
        lines.append(f"    if (stop_seq == {seq_idx}) return;")
        lines.append("")

    lines.append("    model->pos = num_tokens;")
    lines.append("}")
    return "\n".join(lines)


def generate_prefill(ir_path: Path, layout_path: Path = None) -> str:
    """Generate prefill C code from IR.

    The IR already contains everything we need - just read and emit.
    """
    ir = json.load(open(ir_path))

    ops = ir.get("operations", [])
    config = ir.get("config", {})

    parts = []

    # Header comment
    parts.append(f'''/*
 * Auto-generated PREFILL code by codegen_prefill_v6_6.py
 * Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
 * Model: {config.get("model", "unknown")}
 * Mode: prefill
 * Ops: {len(ops)}
 */
''')

    parts.append(emit_prefill_function(ops, config))

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate prefill C code from lowered IR")
    parser.add_argument("--ir", required=True, help="Lowered prefill IR JSON (lowered_prefill_call.json)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()

    code = generate_prefill(Path(args.ir))

    if args.output:
        Path(args.output).write_text(code)
        print(f"Generated: {args.output}")
    else:
        print(code)

    return 0


if __name__ == "__main__":
    sys.exit(main())
