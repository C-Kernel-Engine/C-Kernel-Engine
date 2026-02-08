#!/usr/bin/env python3
"""
codegen_prefill_v6_6.py - Generate C code for PREFILL mode from lowered IR.

This generates ck_prefill() which processes multiple tokens at once.
The IR (lowered_prefill_call.json) already has function names and expressions.
We just substitute num_tokens for const:1 sources.

=============================================================================
IMPORTANT: CODEGEN IS DUMB - NO PARALLELIZATION LOGIC HERE
=============================================================================

When you look at this code, you'll see many `for` loops that LOOK like they
could be parallelized with `#pragma omp parallel for`. You might be tempted
to add pragmas here. DON'T.

WHY NOT?

1. Codegen has NO global view of the computation graph
2. Adding pragmas here could cause FALSE SHARING between ops
3. Two adjacent ops might both parallelize the same buffer = cache thrashing
4. Thread over-subscription if multiple ops spawn threads

WHERE DOES PARALLELIZATION COME FROM?

The parallel_pass.py runs BEFORE codegen and makes centralized decisions:
- Analyzes the full op graph
- Detects false sharing risks
- Decides which ops to parallelize
- Writes op["parallel"]["pragma"] with the EXACT pragma to emit

WHAT CODEGEN DOES:

Codegen BLINDLY reads op["parallel"]["pragma"] and emits it.
No intelligence. No decisions. Just emit what IR says.

If you need to change parallelization strategy, modify parallel_pass.py,
NOT this file.
=============================================================================
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def get_parallel_pragma(op: Dict) -> str:
    """
    Get OpenMP pragma from op's parallel annotation.

    This function does NOT make decisions - it just reads what parallel_pass.py
    wrote to the IR. If no pragma exists, returns empty string.
    """
    parallel = op.get("parallel", {})
    if not parallel.get("enabled", False):
        return ""
    pragma = parallel.get("pragma", "")
    if pragma and not pragma.startswith("//"):
        return pragma
    return ""


def _q8_0_row_bytes(embed_dim: int) -> Optional[int]:
    if embed_dim % 32 != 0:
        return None
    return (embed_dim // 32) * 34


def emit_prefill_op(op: Dict, seq_idx: int, config: Dict, profile: bool = False) -> str:
    """Emit a single op call for prefill mode.

    The IR already provides:
      - function: kernel function name
      - args[]: each with name, source, expr

    We just substitute num_tokens for const:1 and fix memcpy size.
    If profile=True, emit CK_PROFILE_BEGIN/END timing wrappers.
    """
    func = op.get("function", "unknown")
    op_type = op.get("op", "unknown")
    layer = op.get("layer", -1)
    args_list = op.get("args", [])

    # Handle special auto-inserted ops
    if op_type == "copy_last_logits":
        vocab_size = config.get("vocab_size", 151936)
        return f"""    /* Op {seq_idx}: copy_last_logits (prefill fixup) */
    /* Copy last token's logits to start of buffer for ck_model_forward() */
    if (num_tokens > 1) {{
        memmove(
            (void*)(model->bump + A_LOGITS),
            (const void*)(model->bump + A_LOGITS + (size_t)(num_tokens - 1) * {vocab_size} * sizeof(float)),
            {vocab_size} * sizeof(float)
        );
    }}"""

    # If logits layout is last-only, emit a GEMV on the last token only.
    if op_type == "logits" and str(config.get("logits_layout", "auto")).lower() == "last":
        vocab_size = int(config.get("vocab_size", 151936))
        embed_dim = int(config.get("embed_dim", 0))
        row_bytes = _q8_0_row_bytes(embed_dim)
        if row_bytes is not None:
            gemv_func = func
            if gemv_func.startswith("gemm_nt_"):
                gemv_func = "gemv_" + gemv_func[len("gemm_nt_"):]
            elif gemv_func.startswith("gemm_"):
                gemv_func = "gemv_" + gemv_func[len("gemm_"):]
            return f"""    /* Op {seq_idx}: logits (last-only) */
    {gemv_func}(
        (float*)(model->bump + A_LOGITS),
        (const void*)(model->bump + W_TOKEN_EMB),
        (void*)(model->bump + A_LAYER_INPUT + (size_t)(num_tokens - 1) * {row_bytes}),
        {vocab_size},
        {embed_dim}
    );"""

    if op_type == "kv_cache_batch_copy":
        # Copy K/V from scratch (head-major after transpose) to KV cache
        # Scratch layout: [num_kv_heads, num_tokens, head_dim] (compact, head-major)
        # KV cache layout: [num_kv_heads, max_seq_len, head_dim] (with stride, head-major)
        layer = op.get("layer", 0)
        num_kv_heads = config.get("num_kv_heads", 2)
        head_dim = config.get("head_dim", 64)
        context_len = config.get("context_len", config.get("context_length", 1024))
        return f"""    /* Op {seq_idx}: kv_cache_batch_copy layer={layer} */
    /* Copy K/V from head-major scratch to KV cache for subsequent decode */
    {{
        const int Hkv = {num_kv_heads};
        const int D = {head_dim};
        const int cache_stride = {context_len};
        float *k_scratch = (float*)(model->bump + A_K_SCRATCH);
        float *v_scratch = (float*)(model->bump + A_V_SCRATCH);
        float *kv_cache = (float*)model->kv_cache;
        for (int h = 0; h < Hkv; h++) {{
            /* K: copy from scratch[h, 0:num_tokens, :] to cache[h, 0:num_tokens, :] */
            /* Scratch is compact: stride = num_tokens, Cache has stride = cache_stride */
            memcpy(
                kv_cache + ({layer}*2)*Hkv*cache_stride*D + h*cache_stride*D,
                k_scratch + h*num_tokens*D,
                (size_t)num_tokens * D * sizeof(float)
            );
            /* V: copy from scratch[h, 0:num_tokens, :] to cache[h, 0:num_tokens, :] */
            memcpy(
                kv_cache + ({layer}*2+1)*Hkv*cache_stride*D + h*cache_stride*D,
                v_scratch + h*num_tokens*D,
                (size_t)num_tokens * D * sizeof(float)
            );
        }}
    }}"""

    # Handle transpose_kv_to_head_major: convert from [T, Hkv*D] to [Hkv, T, D]
    if op_type == "transpose_kv_to_head_major":
        num_kv_heads = config.get("num_kv_heads", 2)
        head_dim = config.get("head_dim", 64)
        # _is_k is set by emit_prefill_function preprocessing
        is_k = op.get("_is_k", True)
        scratch_name = "A_K_SCRATCH" if is_k else "A_V_SCRATCH"
        max_tokens = config.get("context_len", config.get("context_length", 1024))
        omp_pragma = get_parallel_pragma(op)
        if omp_pragma:
            omp_pragma = f"\n        {omp_pragma}"
        return f"""    /* Op {seq_idx}: transpose_{("k" if is_k else "v")}_to_head_major layer={layer} */
    /* Transpose from [T, Hkv*D] (token-major GEMM output) to [Hkv, T, D] (head-major for attention) */
    {{
        const int Hkv = {num_kv_heads};
        const int D = {head_dim};
        float *buf = (float*)(model->bump + {scratch_name});
        /* Use temp buffer for out-of-place transpose (safe, no aliasing) */
        static float _temp_buf[{num_kv_heads * max_tokens * head_dim}];
        /* Copy with transpose: src[t, h*D+d] -> dst[h, t, d] */{omp_pragma}
        for (int t = 0; t < num_tokens; t++) {{
            for (int h = 0; h < Hkv; h++) {{
                memcpy(_temp_buf + h * num_tokens * D + t * D,
                       buf + t * Hkv * D + h * D,
                       D * sizeof(float));
            }}
        }}
        /* Copy back */
        memcpy(buf, _temp_buf, (size_t)Hkv * num_tokens * D * sizeof(float));
    }}"""

    # Handle transpose_qkv_to_head_major for Q: convert from [T, H*D] to [H, T, D]
    if op_type == "transpose_qkv_to_head_major":
        qkv_type = op.get("_qkv_type", "q")
        if qkv_type == "q":
            num_heads = config.get("num_heads", 14)
            head_dim = config.get("head_dim", 64)
            scratch_name = "A_Q_SCRATCH"
            max_tokens = config.get("context_len", config.get("context_length", 1024))
            omp_pragma = get_parallel_pragma(op)
            if omp_pragma:
                omp_pragma = f"\n        {omp_pragma}"
            return f"""    /* Op {seq_idx}: transpose_q_to_head_major layer={layer} */
    /* Transpose from [T, H*D] (token-major GEMM output) to [H, T, D] (head-major for attention) */
    {{
        const int H = {num_heads};
        const int D = {head_dim};
        float *buf = (float*)(model->bump + {scratch_name});
        /* Use temp buffer for out-of-place transpose */
        static float _temp_buf[{num_heads * max_tokens * head_dim}];
        /* Copy with transpose: src[t, h*D+d] -> dst[h, t, d] */{omp_pragma}
        for (int t = 0; t < num_tokens; t++) {{
            for (int h = 0; h < H; h++) {{
                memcpy(_temp_buf + h * num_tokens * D + t * D,
                       buf + t * H * D + h * D,
                       D * sizeof(float));
            }}
        }}
        /* Copy back */
        memcpy(buf, _temp_buf, (size_t)H * num_tokens * D * sizeof(float));
    }}"""

    # Handle transpose_attn_out_to_token_major: convert from [H, T, D] to [T, H*D]
    # This is the reverse of the Q transpose - needed after attention before out_proj
    if op_type == "transpose_attn_out_to_token_major":
        num_heads = config.get("num_heads", 14)
        head_dim = config.get("head_dim", 64)
        max_tokens = config.get("context_len", config.get("context_length", 1024))
        # Parallelize over heads (outer loop)
        omp_pragma = get_parallel_pragma(op)
        if omp_pragma:
            omp_pragma = f"\n        {omp_pragma}"
        return f"""    /* Op {seq_idx}: transpose_attn_out_to_token_major layer={layer} */
    /* Transpose from [H, T, D] (head-major attention output) to [T, H*D] (token-major for out_proj) */
    {{
        const int H = {num_heads};
        const int D = {head_dim};
        float *buf = (float*)(model->bump + A_ATTN_SCRATCH);
        /* Use temp buffer for out-of-place transpose */
        static float _temp_buf[{num_heads * max_tokens * head_dim}];
        /* Copy with transpose: src[h, t, d] -> dst[t, h*D+d] */{omp_pragma}
        for (int h = 0; h < H; h++) {{
            for (int t = 0; t < num_tokens; t++) {{
                memcpy(_temp_buf + t * H * D + h * D,
                       buf + h * num_tokens * D + t * D,
                       D * sizeof(float));
            }}
        }}
        /* Copy back */
        memcpy(buf, _temp_buf, (size_t)num_tokens * H * D * sizeof(float));
    }}"""

    embed_dim = config.get("embed_dim", 896)

    lines = []
    lines.append(f"    /* Op {seq_idx}: {func} ({op_type}) layer={layer} */")
    if profile:
        lines.append(f"    CK_PROFILE_BEGIN();")

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
        # For prefill attention kernels, K/V scratch is compact head-major:
        # [Hkv, num_tokens, D]. The stride in tokens must be num_tokens, not
        # MAX_SEQ_LEN/context length. Apply to both regular and sliding attention.
        elif name == "kv_stride_tokens" and op_type in ("attn", "attn_sliding"):
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
    if profile:
        lines.append(f'    CK_PROFILE_END("prefill", "{func}", "{op_type}", {layer});')

    return "\n".join(lines)


def emit_prefill_function(ops: List[Dict], config: Dict, profile: bool = False) -> str:
    """Emit the prefill function with all ops unrolled."""
    lines = []
    scale_embeddings_sqrt_dim = bool(config.get("scale_embeddings_sqrt_dim", False))
    embed_scale_emitted = False
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

    if profile:
        lines.append("    CK_PROFILE_VARS();")
        lines.append("")

    # Preprocess ops to determine K vs V for transpose_kv_to_head_major
    # Within each layer, the first transpose_kv is K, the second is V
    layer_kv_count: Dict[int, int] = {}
    for op in ops:
        if op.get("op") == "transpose_kv_to_head_major":
            layer = op.get("layer", 0)
            count = layer_kv_count.get(layer, 0)
            op["_is_k"] = (count == 0)  # First is K, second is V
            layer_kv_count[layer] = count + 1

    for seq_idx, op in enumerate(ops):
        lines.append(emit_prefill_op(op, seq_idx, config, profile=profile))
        lines.append(f"    if (stop_seq == {seq_idx}) return;")
        if (scale_embeddings_sqrt_dim
                and not embed_scale_emitted
                and op.get("op") == "dense_embedding_lookup"
                and int(op.get("layer", -1)) == -1):
            lines.append("""    /* Gemma embedding contract:
     * llama.cpp applies inp_scaled = inp_embd * sqrt(n_embd) before layer-0.
     * Without this, residual path parity diverges at sa_out even if q/k/v look close.
     */
    {
        const float emb_scale = sqrtf((float)EMBED_DIM);
        float *emb = (float*)(model->bump + A_EMBEDDED_INPUT);
        const int n = num_tokens * EMBED_DIM;
        for (int i = 0; i < n; ++i) {
            emb[i] *= emb_scale;
        }
    }
    #ifdef CK_PARITY_DUMP
    ck_dump_tensor((float*)(model->bump + A_EMBEDDED_INPUT), -1, "inp_scaled", num_tokens * EMBED_DIM);
    #endif""")
            embed_scale_emitted = True
        lines.append("")

    lines.append("    model->pos = num_tokens;")
    lines.append("}")
    return "\n".join(lines)


def generate_prefill(ir_path: Path, layout_path: Path = None, profile: bool = False) -> str:
    """Generate prefill C code from IR.

    The IR already contains everything we need - just read and emit.
    If profile=True, emit CK_PROFILE timing wrappers around each kernel call.
    """
    ir = json.load(open(ir_path))

    ops = ir.get("operations", [])
    config = ir.get("config", {})

    parts = []

    # Header comment
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f'''/*
 * Auto-generated PREFILL code by codegen_prefill_v6_6.py
 * Generated: {now}
 * Model: {config.get("model", "unknown")}
 * Mode: prefill
 * Ops: {len(ops)}
 */
''')

    parts.append(emit_prefill_function(ops, config, profile=profile))

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
