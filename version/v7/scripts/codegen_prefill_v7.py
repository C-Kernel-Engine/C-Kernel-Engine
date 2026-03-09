#!/usr/bin/env python3
"""
codegen_prefill_v7.py - Generate C code for PREFILL mode from lowered IR.

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


def _find_arg_expr(
    args_list: List[Dict],
    *,
    source_prefix: Optional[str] = None,
    arg_name: Optional[str] = None,
) -> Optional[str]:
    for item in args_list:
        if not isinstance(item, dict):
            continue
        if arg_name is not None and str(item.get("name", "")) != arg_name:
            continue
        source = str(item.get("source", ""))
        if source_prefix is not None and not source.startswith(source_prefix):
            continue
        expr = str(item.get("expr", "")).strip()
        if expr:
            return expr
    return None


def _last_token_row_offset_expr(func_name: str, embed_dim: int) -> Optional[str]:
    """Return byte-offset expression for token-major activation row stride."""
    if embed_dim <= 0:
        return None
    fn = str(func_name or "").lower()
    if "q8_k" in fn:
        return f"(size_t)(num_tokens - 1) * (size_t)({embed_dim} / QK_K) * sizeof(block_q8_K)"
    if "q8_0" in fn:
        return f"(size_t)(num_tokens - 1) * (size_t)({embed_dim} / QK8_0) * sizeof(block_q8_0)"
    if "fp32" in fn or "f32" in fn:
        return f"(size_t)(num_tokens - 1) * (size_t){embed_dim} * sizeof(float)"
    # Conservative default for q8_0-style activation packing.
    row_bytes = _q8_0_row_bytes(embed_dim)
    if row_bytes is None:
        return None
    return f"(size_t)(num_tokens - 1) * (size_t){row_bytes}"


def emit_prefill_op(op: Dict, seq_idx: int, config: Dict, profile: bool = False, dump: bool = False) -> str:
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
        if embed_dim > 0:
            gemv_func = func
            if gemv_func.startswith("gemm_nt_"):
                gemv_func = "gemv_" + gemv_func[len("gemm_nt_"):]
            elif gemv_func.startswith("gemm_"):
                gemv_func = "gemv_" + gemv_func[len("gemm_"):]
            weight_expr = _find_arg_expr(args_list, source_prefix="weight:", arg_name="B") or _find_arg_expr(
                args_list, source_prefix="weight:"
            )
            input_expr = _find_arg_expr(args_list, source_prefix="activation:", arg_name="A") or _find_arg_expr(
                args_list, source_prefix="activation:"
            )
            output_expr = _find_arg_expr(args_list, source_prefix="output:", arg_name="C") or _find_arg_expr(
                args_list, source_prefix="output:"
            )
            row_offset_expr = _last_token_row_offset_expr(gemv_func, embed_dim)
            if not weight_expr:
                raise RuntimeError("prefill logits(last): missing weight arg expression in lowered call IR")
            if not input_expr:
                raise RuntimeError("prefill logits(last): missing activation arg expression in lowered call IR")
            if not output_expr:
                raise RuntimeError("prefill logits(last): missing output arg expression in lowered call IR")
            if not row_offset_expr:
                raise RuntimeError(
                    f"prefill logits(last): unable to derive row stride for func={gemv_func} embed_dim={embed_dim}"
                )
            return f"""    /* Op {seq_idx}: logits (last-only) */
    {gemv_func}(
        {output_expr},
        {weight_expr},
        (const void*)(((const uint8_t*)({input_expr})) + {row_offset_expr}),
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
        /* Reuse activation scratch to avoid huge per-op static BSS allocations. */
        float *_temp_buf = (float*)(model->bump + A_LAYER_OUTPUT);
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
        /* Reuse activation scratch to avoid huge per-op static BSS allocations. */
        float *_temp_buf = (float*)(model->bump + A_LAYER_OUTPUT);
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
        /* Reuse activation scratch to avoid huge per-op static BSS allocations. */
        float *_temp_buf = (float*)(model->bump + A_LAYER_OUTPUT);
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
    arg_expr_by_name: Dict[str, str] = {}
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
        name_key = str(name).lower()
        if name_key and expr and name_key not in arg_expr_by_name:
            arg_expr_by_name[name_key] = expr

    # Prefill quantization must preserve token-major row layout exactly for
    # downstream GEMM kernels. Emit explicit per-token row quantization loops
    # instead of batch helper calls to avoid ABI/dispatch ambiguity.
    batch_quant_kind = None
    if func in ("quantize_row_q8_0", "quantize_row_q8_k"):
        batch_quant_kind = func

    if op_type == "out_proj" and func in ("gemm_nt_q4_k_q8_k", "gemm_nt_q6_k_q8_k"):
        a_expr = arg_expr_by_name.get("a")
        b_expr = arg_expr_by_name.get("b")
        bias_expr = arg_expr_by_name.get("bias", "NULL")
        c_expr = arg_expr_by_name.get("c")
        m_expr = arg_expr_by_name.get("m", "num_tokens")
        n_expr = arg_expr_by_name.get("n")
        k_expr = arg_expr_by_name.get("k")
        fp32_func = "gemm_nt_q4_k" if func == "gemm_nt_q4_k_q8_k" else "gemm_nt_q6_k"
        if a_expr and b_expr and c_expr and n_expr and k_expr:
            lines.append("    if (debug_outproj_fp32 && ck_debug_outproj_fp32_input != NULL) {")
            lines.append(f"        {fp32_func}(")
            lines.append("            ck_debug_outproj_fp32_input,")
            lines.append(f"            {b_expr},")
            lines.append(f"            {bias_expr},")
            lines.append(f"            {c_expr},")
            lines.append(f"            {m_expr},")
            lines.append(f"            {n_expr},")
            lines.append(f"            {k_expr}")
            lines.append("        );")
            lines.append("    } else {")
            lines.append(f"        {func}(")
            lines.append(f"            {a_expr},")
            lines.append(f"            {b_expr},")
            lines.append(f"            {bias_expr},")
            lines.append(f"            {c_expr},")
            lines.append(f"            {m_expr},")
            lines.append(f"            {n_expr},")
            lines.append(f"            {k_expr}")
            lines.append("        );")
            lines.append("    }")
            if profile:
                lines.append(f'    CK_PROFILE_END("prefill", "{func}", "{op_type}", {layer});')

            if dump:
                raw_expr = c_expr.replace("(float*)", "").replace("(void*)", "").strip()
                size_expr = f"({m_expr}) * ({n_expr})"
                lines.append("    #ifdef CK_PARITY_DUMP")
                lines.append(f'    ck_dump_tensor((float*){raw_expr}, {layer}, "attn_output", {size_expr});')
                lines.append("    #endif")
            return "\n".join(lines)

    # Format the function call / quantization loop
    if batch_quant_kind and len(args) >= 3:
        x_expr = args[0]
        y_expr = args[1]
        k_expr = args[2]
        if batch_quant_kind == "quantize_row_q8_k":
            row_bytes_expr = "(size_t)(_k / QK_K) * sizeof(block_q8_K)"
        else:
            row_bytes_expr = "(size_t)(_k / QK8_0) * sizeof(block_q8_0)"
        if op_type == "quantize_out_proj_input":
            lines.append(f"    ck_debug_outproj_fp32_input = (const float*)({x_expr});")
            lines.append("    if (!debug_outproj_fp32) {")
            lines.append(f"        const float *_x_base = (const float*)({x_expr});")
            lines.append(f"        uint8_t *_y_base = (uint8_t*)({y_expr});")
            lines.append(f"        const int _k = (int)({k_expr});")
            lines.append(f"        const size_t _row_bytes = {row_bytes_expr};")
            lines.append("        for (int _t = 0; _t < num_tokens; ++_t) {")
            lines.append(
                f"            {batch_quant_kind}("
                "_x_base + (size_t)_t * (size_t)_k, "
                "(void*)(_y_base + (size_t)_t * _row_bytes), "
                "_k);"
            )
            lines.append("        }")
            lines.append("    }")
        else:
            lines.append("    {")
            lines.append(f"        const float *_x_base = (const float*)({x_expr});")
            lines.append(f"        uint8_t *_y_base = (uint8_t*)({y_expr});")
            lines.append(f"        const int _k = (int)({k_expr});")
            lines.append(f"        const size_t _row_bytes = {row_bytes_expr};")
            lines.append("        for (int _t = 0; _t < num_tokens; ++_t) {")
            lines.append(
                f"            {batch_quant_kind}("
                "_x_base + (size_t)_t * (size_t)_k, "
                "(void*)(_y_base + (size_t)_t * _row_bytes), "
                "_k);"
            )
            lines.append("        }")
            lines.append("    }")
    else:
        if len(args) <= 3:
            # Short call on one line
            lines.append(f"    {func}({', '.join(args)});")
        else:
            # Multi-line for readability
            lines.append(f"    {func}(")
            for i, arg in enumerate(args):
                comma = "," if i < len(args) - 1 else ""
                lines.append(f"        {arg}{comma}")
            lines.append("    );")
    if profile:
        lines.append(f'    CK_PROFILE_END("prefill", "{func}", "{op_type}", {layer});')

    if dump:
        dump_op_map = {
            "dense_embedding_lookup": "token_embedding",
            "attn_norm": "attn_norm",
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "attn": "attn_output",
            "attn_sliding": "attn_output",
            "out_proj": "attn_output",
            "post_attention_norm": "attn_post_norm",
            "ffn_norm": "ffn_norm",
            "mlp_gate_up": "ffn_gate_up",
            "geglu": "ffn_gate_par",
            "mlp_down": "down_proj",
            "post_ffn_norm": "ffn_post_norm",
            "final_rmsnorm": "final_norm",
            "logits": "logits",
        }
        dump_name = dump_op_map.get(op_type)

        def _get_arg(*names: str) -> Optional[str]:
            for nm in names:
                ex = arg_expr_by_name.get(nm.lower())
                if ex:
                    return ex
            return None

        def _mul_expr(*terms: Optional[str]) -> Optional[str]:
            used = [f"({t})" for t in terms if t]
            if not used:
                return None
            return " * ".join(used)

        def _emit_dump(expr: Optional[str], name: str, size_expr: Optional[str]) -> None:
            if not expr or not size_expr:
                return
            raw_expr = expr.replace("(float*)", "").replace("(void*)", "").strip()
            lines.append("    #ifdef CK_PARITY_DUMP")
            lines.append(f'    ck_dump_tensor((float*){raw_expr}, {layer}, "{name}", {size_expr});')
            lines.append("    #endif")

        tokens = "num_tokens"
        embed_dim_expr = _get_arg("aligned_embed_dim", "d_model", "embed_dim") or str(embed_dim)
        m_dim = _get_arg("m")
        n_dim = _get_arg("n")
        num_heads = _get_arg("num_heads") or "NUM_HEADS"
        num_kv_heads = _get_arg("num_kv_heads") or "NUM_KV_HEADS"
        head_dim = _get_arg("aligned_head_dim", "head_dim") or "HEAD_DIM"

        if op_type == "qk_norm":
            q_expr = _get_arg("q")
            k_expr = _get_arg("k")
            q_size = _mul_expr(tokens, num_heads, head_dim)
            k_size = _mul_expr(tokens, num_kv_heads, head_dim)
            _emit_dump(q_expr, "qcur_normed", q_size)
            _emit_dump(k_expr, "kcur_normed", k_size)
        elif dump_name:
            out_expr = _get_arg("output", "out", "c", "y", "out_token")
            size_expr = None

            if op_type in ("dense_embedding_lookup", "attn_norm", "post_attention_norm", "ffn_norm", "post_ffn_norm", "residual_add"):
                size_expr = _mul_expr(tokens, embed_dim_expr)
            elif op_type in ("q_proj", "k_proj", "v_proj", "out_proj", "mlp_gate_up", "mlp_down", "logits"):
                if func.startswith("gemm_") and n_dim:
                    size_expr = _mul_expr(tokens, n_dim)
                else:
                    size_expr = _mul_expr(tokens, n_dim or m_dim)
            elif op_type in ("attn", "attn_sliding"):
                size_expr = _mul_expr(tokens, num_heads, head_dim)
            elif op_type == "geglu":
                dim = _get_arg("dim")
                size_expr = _mul_expr(tokens, dim) if dim else None

            _emit_dump(out_expr, dump_name, size_expr)

    return "\n".join(lines)


def emit_prefill_function(ops: List[Dict], config: Dict, profile: bool = False, dump: bool = False) -> str:
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
    const char *debug_outproj_env = getenv("CK_V7_DEBUG_OUTPROJ_FP32");
    int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : 0;
    const float *ck_debug_outproj_fp32_input = NULL;

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
        lines.append(emit_prefill_op(op, seq_idx, config, profile=profile, dump=dump))
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


def generate_prefill(ir_path: Path, layout_path: Path = None, profile: bool = False, dump: bool = False) -> str:
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
 * Auto-generated PREFILL code by codegen_prefill_v7.py
 * Generated: {now}
 * Model: {config.get("model", "unknown")}
 * Mode: prefill
 * Ops: {len(ops)}
 */
''')

    parts.append(emit_prefill_function(ops, config, profile=profile, dump=dump))

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
