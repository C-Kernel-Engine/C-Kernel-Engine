#!/usr/bin/env python3
"""
codegen_prefill_v8.py - Generate C code for PREFILL mode from lowered IR.

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


def _annotate_kv_transpose_roles(ops: List[Dict]) -> None:
    """Mark each transpose_kv_to_head_major op as K or V within its layer."""
    layer_kv_count: Dict[int, int] = {}
    for op in ops:
        if op.get("op") != "transpose_kv_to_head_major":
            continue
        layer = int(op.get("layer", 0))
        count = layer_kv_count.get(layer, 0)
        op["_is_k"] = (count == 0)
        layer_kv_count[layer] = count + 1


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
    section = op.get("section", "")
    op_instance_idx = int(op.get("op_instance_idx", op.get("instance", 0)) or 0)
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
                kv_cache + (1ULL*({layer}*2)*Hkv*cache_stride*D) + (size_t)h*cache_stride*D,
                k_scratch + h*num_tokens*D,
                (size_t)num_tokens * D * sizeof(float)
            );
            /* V: copy from scratch[h, 0:num_tokens, :] to cache[h, 0:num_tokens, :] */
            memcpy(
                kv_cache + (1ULL*({layer}*2+1)*Hkv*cache_stride*D) + (size_t)h*cache_stride*D,
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
        elif op_type in ("rmsnorm", "layernorm"):
            dump_label = None
            if str(section or "") == "footer":
                dump_label = "final_norm"
            elif op_instance_idx == 0:
                dump_label = "attn_norm"
            elif op_instance_idx == 1:
                dump_label = "ffn_norm"
            if dump_label is not None:
                _emit_dump(_get_arg("output", "out", "x", "y"), dump_label, _mul_expr(tokens, embed_dim_expr))
        elif op_type == "residual_add":
            dump_label = "ffn_inp" if op_instance_idx == 0 else "layer_out" if op_instance_idx == 1 else None
            if dump_label is not None:
                _emit_dump(_get_arg("output", "out", "c", "y"), dump_label, _mul_expr(tokens, embed_dim_expr))
        elif dump_name:
            out_expr = _get_arg("output", "out", "c", "y", "out_token")
            size_expr = None

            if op_type in ("dense_embedding_lookup", "attn_norm", "post_attention_norm", "ffn_norm", "post_ffn_norm"):
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

    _annotate_kv_transpose_roles(ops)
    residual_add_count_total = 0

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
    lines.append("    model->rope_pos = num_tokens;")
    lines.append("}")
    return "\n".join(lines)


def _find_embedding_header_op(ops: List[Dict]) -> Optional[Dict]:
    for op in ops:
        if str(op.get("op", "")) == "dense_embedding_lookup":
            return op
    return None


def _emit_embedding_scale_block(num_tokens_expr: str, dump: bool) -> str:
    lines: List[str] = [
        """    /* Gemma embedding contract:
     * llama.cpp applies inp_scaled = inp_embd * sqrt(n_embd) before layer-0.
     * Without this, residual path parity diverges at sa_out even if q/k/v look close.
     */
    {
        const float emb_scale = sqrtf((float)EMBED_DIM);
        float *emb = (float*)(model->bump + A_EMBEDDED_INPUT);
        const int n = """
        + num_tokens_expr
        + """ * EMBED_DIM;
        for (int i = 0; i < n; ++i) {
            emb[i] *= emb_scale;
        }
    }"""
    ]
    if dump:
        lines.extend(
            [
                "    #ifdef CK_PARITY_DUMP",
                f'    ck_dump_tensor((float*)(model->bump + A_EMBEDDED_INPUT), -1, "inp_scaled", {num_tokens_expr} * EMBED_DIM);',
                "    #endif",
            ]
        )
    return "\n".join(lines)


def _is_qwen3vl_multimodal_config(config: Dict) -> bool:
    model = str(config.get("model", config.get("name", ""))).lower()
    return model == "qwen3vl"


def _emit_qwen3vl_prefill_bridge_helpers(config: Dict) -> str:
    embed_dim = int(config.get("embed_dim", 0) or 0)
    num_deepstack_layers = int(config.get("num_deepstack_layers", 0) or 0)
    if embed_dim <= 0 or num_deepstack_layers <= 0:
        return ""

    return f"""
static int g_qwen3vl_prefill_bridge_active = 0;
static int g_qwen3vl_prefill_text_pos = 0;
static int g_qwen3vl_prefill_num_tokens = 0;
static int32_t *g_qwen3vl_prefill_positions = NULL;
static const float *g_qwen3vl_prefill_rows = NULL;
static int g_qwen3vl_prefill_row_dim = 0;

static void ck_qwen3vl_prefill_bridge_clear(void) {{
    g_qwen3vl_prefill_bridge_active = 0;
    g_qwen3vl_prefill_text_pos = 0;
    g_qwen3vl_prefill_num_tokens = 0;
    free(g_qwen3vl_prefill_positions);
    g_qwen3vl_prefill_positions = NULL;
    g_qwen3vl_prefill_rows = NULL;
    g_qwen3vl_prefill_row_dim = 0;
}}

static void ck_qwen3vl_prefill_bridge_free(void) {{
    ck_qwen3vl_prefill_bridge_clear();
}}

static int ck_qwen3vl_prefill_bridge_text_pos(void) {{
    return g_qwen3vl_prefill_text_pos;
}}

static int ck_qwen3vl_prefill_bridge_is_active(void) {{
    return g_qwen3vl_prefill_bridge_active;
}}

static int ck_qwen3vl_prefill_bridge_prepare(const float *rows, int num_tokens, int row_dim, int grid_x, int grid_y, int text_pos) {{
    ck_qwen3vl_prefill_bridge_clear();
    if (!rows || num_tokens <= 0 || row_dim < {embed_dim}) return -1;
    if (grid_x <= 0 || grid_y <= 0) return -2;
    if (grid_x * grid_y != num_tokens) return -3;

    g_qwen3vl_prefill_positions = (int32_t*)malloc((size_t)4 * (size_t)num_tokens * sizeof(int32_t));
    if (!g_qwen3vl_prefill_positions) {{
        ck_qwen3vl_prefill_bridge_clear();
        return -4;
    }}

    for (int tok = 0; tok < num_tokens; ++tok) {{
        const int x = tok % grid_x;
        const int y = tok / grid_x;
        g_qwen3vl_prefill_positions[tok] = 0;
        g_qwen3vl_prefill_positions[tok + num_tokens] = y;
        g_qwen3vl_prefill_positions[tok + 2 * num_tokens] = x;
        g_qwen3vl_prefill_positions[tok + 3 * num_tokens] = 0;
    }}

    g_qwen3vl_prefill_rows = rows;
    g_qwen3vl_prefill_row_dim = row_dim;
    g_qwen3vl_prefill_num_tokens = num_tokens;
    g_qwen3vl_prefill_text_pos = text_pos > 0 ? text_pos : (grid_x > grid_y ? grid_x : grid_y);
    g_qwen3vl_prefill_bridge_active = 1;
    return 0;
}}

static void ck_qwen3vl_prefill_mrope_qk(float *q, float *k, int num_heads, int num_kv_heads, int num_tokens, int head_dim, int aligned_head_dim, int pos_offset, int n_dims, int section_0, int section_1, int section_2, int section_3, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {{
    if (g_qwen3vl_prefill_bridge_active && g_qwen3vl_prefill_positions && g_qwen3vl_prefill_num_tokens == num_tokens) {{
        mrope_qk_vision(q, k, g_qwen3vl_prefill_positions, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        return;
    }}
    mrope_qk_text(q, k, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset, n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}}

static void ck_qwen3vl_prefill_deepstack_add(CKModel *model, int layer, int num_tokens) {{
    if (!model || !g_qwen3vl_prefill_bridge_active || !g_qwen3vl_prefill_rows) return;
    if (layer < 0 || layer >= {num_deepstack_layers}) return;
    const size_t slice_offset = (size_t){embed_dim} + (size_t)layer * (size_t){embed_dim};
    const size_t need = slice_offset + (size_t){embed_dim};
    if ((size_t)g_qwen3vl_prefill_row_dim < need) return;
    float *dst = (float*)(model->bump + A_EMBEDDED_INPUT);
    for (int tok = 0; tok < num_tokens; ++tok) {{
        const float *src = g_qwen3vl_prefill_rows + (size_t)tok * (size_t)g_qwen3vl_prefill_row_dim + slice_offset;
        float *dst_row = dst + (size_t)tok * (size_t){embed_dim};
        for (int i = 0; i < {embed_dim}; ++i) {{
            dst_row[i] += src[i];
        }}
    }}
}}
"""


def _emit_prefill_quant_debug_override(op: Dict, seq_idx: int, config: Dict, *, debug_flag_name: str, debug_input_name: str, profile: bool = False) -> str:
    func = str(op.get("function", "") or "")
    if func not in {"quantize_row_q8_0", "quantize_row_q8_k"}:
        raise RuntimeError(f"unsupported prefill quant override func={func}")

    args_list = op.get("args", [])
    x_expr = _find_arg_expr(args_list, arg_name="x") or _find_arg_expr(args_list, arg_name="x_q8")
    y_expr = _find_arg_expr(args_list, arg_name="y")
    k_expr = _find_arg_expr(args_list, arg_name="k")
    if not x_expr or not y_expr or not k_expr:
        raise RuntimeError("prefill quant override missing x/y/k args")

    row_bytes_expr = "(size_t)(_k / QK_K) * sizeof(block_q8_K)" if func == "quantize_row_q8_k" else "(size_t)(_k / QK8_0) * sizeof(block_q8_0)"
    lines = [
        f"    /* Op {seq_idx}: {func} ({op.get('op', 'unknown')}) layer={op.get('layer', -1)} */",
        f"    {debug_input_name} = (const float*)({x_expr});",
        f"    if (!{debug_flag_name}) {{",
    ]
    if profile:
        lines.append("        CK_PROFILE_BEGIN();")
    lines.extend(
        [
            f"        const float *_x_base = (const float*)({x_expr});",
            f"        uint8_t *_y_base = (uint8_t*)({y_expr});",
            f"        const int _k = (int)({k_expr});",
            f"        const size_t _row_bytes = {row_bytes_expr};",
            "        for (int _t = 0; _t < num_tokens; ++_t) {",
            f"            {func}(_x_base + (size_t)_t * (size_t)_k, (void*)(_y_base + (size_t)_t * _row_bytes), _k);",
            "        }",
        ]
    )
    if profile:
        lines.append(f'        CK_PROFILE_END("prefill", "{func}", "{op.get("op", "unknown")}", {int(op.get("layer", -1) or -1)});')
    lines.append("    }")
    return "\n".join(lines)


def _emit_prefill_gemm_fp32_override(op: Dict, seq_idx: int, *, debug_flag_name: str, debug_input_name: str, profile: bool = False, dump: bool = False) -> str:
    func = str(op.get("function", "") or "")
    if func not in {"gemm_nt_q4_k_q8_k", "gemm_nt_q6_k_q8_k"}:
        raise RuntimeError(f"unsupported prefill fp32 override func={func}")

    args_list = op.get("args", [])
    m_arg = next(
        (
            arg
            for arg in args_list
            if isinstance(arg, dict) and str(arg.get("name", "")).lower() == "m"
        ),
        None,
    )
    a_expr = _find_arg_expr(args_list, arg_name="A") or _find_arg_expr(args_list, arg_name="a")
    b_expr = _find_arg_expr(args_list, arg_name="B") or _find_arg_expr(args_list, arg_name="b")
    bias_expr = _find_arg_expr(args_list, arg_name="bias") or "NULL"
    c_expr = _find_arg_expr(args_list, arg_name="C") or _find_arg_expr(args_list, arg_name="c")
    m_expr = _find_arg_expr(args_list, arg_name="M") or _find_arg_expr(args_list, arg_name="m") or "num_tokens"
    if isinstance(m_arg, dict) and str(m_arg.get("source", "")).lower() == "dim:_m":
        # Embedded-prefix prefill only replays the active prefix rows, not the
        # full static context length baked into the prefill IR.
        m_expr = "num_tokens"
    n_expr = _find_arg_expr(args_list, arg_name="N") or _find_arg_expr(args_list, arg_name="n")
    k_expr = _find_arg_expr(args_list, arg_name="K") or _find_arg_expr(args_list, arg_name="k")
    if not a_expr or not b_expr or not c_expr or not n_expr or not k_expr:
        raise RuntimeError("prefill fp32 override missing GEMM args")

    fp32_func = "gemm_nt_q4_k" if func == "gemm_nt_q4_k_q8_k" else "gemm_nt_q6_k"
    layer_value = op.get("layer", -1)
    layer = int(layer_value) if layer_value is not None else -1
    lines = [
        f"    /* Op {seq_idx}: {func} ({op.get('op', 'unknown')}) layer={layer} */",
        f"    if ({debug_flag_name} && {debug_input_name} != NULL) {{",
    ]
    if profile:
        lines.append("        CK_PROFILE_BEGIN();")
    lines.extend(
        [
            f"        {fp32_func}(",
            f"            {debug_input_name},",
            f"            {b_expr},",
            f"            {bias_expr},",
            f"            {c_expr},",
            f"            {m_expr},",
            f"            {n_expr},",
            f"            {k_expr}",
            "        );",
        ]
    )
    if profile:
        lines.append(f'        CK_PROFILE_END("prefill", "{fp32_func}", "{op.get("op", "unknown")}", {layer});')
    lines.append("    } else {")
    if profile:
        lines.append("        CK_PROFILE_BEGIN();")
    lines.extend(
        [
            f"        {func}(",
            f"            {a_expr},",
            f"            {b_expr},",
            f"            {bias_expr},",
            f"            {c_expr},",
            f"            {m_expr},",
            f"            {n_expr},",
            f"            {k_expr}",
            "        );",
        ]
    )
    if profile:
        lines.append(f'        CK_PROFILE_END("prefill", "{func}", "{op.get("op", "unknown")}", {layer});')
    lines.append("    }")
    if dump:
        raw_expr = c_expr.replace("(float*)", "").replace("(void*)", "").strip()
        lines.append("    #ifdef CK_PARITY_DUMP")
        lines.append(f'    ck_dump_tensor((float*){raw_expr}, {layer}, "down_proj", ({m_expr}) * ({n_expr}));')
        lines.append("    #endif")
    return "\n".join(lines)


def emit_prefill_from_embedded_function(
    ops: List[Dict],
    config: Dict,
    profile: bool = False,
    dump: bool = False,
) -> str:
    """Emit a prefill entrypoint that assumes embedded_input is already populated."""
    lines = []
    scale_embeddings_sqrt_dim = bool(config.get("scale_embeddings_sqrt_dim", False))
    embed_scale_emitted = False
    is_qwen3vl = _is_qwen3vl_multimodal_config(config)
    num_deepstack_layers = int(config.get("num_deepstack_layers", 0) or 0) if is_qwen3vl else 0
    helper_block = _emit_qwen3vl_prefill_bridge_helpers(config) if is_qwen3vl else ""
    if helper_block:
        lines.append(helper_block)

    lines.append(
        """
/* ============================================================================
 * PREFILL FROM EMBEDDED INPUT - Multimodal/orchestrated prefill path
 * ============================================================================
 * Assumes:
 *   - A_EMBEDDED_INPUT already contains the first num_tokens rows
 *   - token rows after num_tokens are don't-care
 *   - dense_embedding_lookup must be skipped to preserve external prefixes
 * ============================================================================ */
static void ck_prefill_from_embedded(CKModel *model, int num_tokens) {
    if (!model || num_tokens <= 0) return;

    /* Clamp to max context */
    if (num_tokens > MAX_SEQ_LEN) num_tokens = MAX_SEQ_LEN;

    const char *stop_env = getenv("CK_STOP_OP");
    int stop_seq = stop_env ? atoi(stop_env) : -1;
    const char *debug_outproj_env = getenv("CK_V7_DEBUG_OUTPROJ_FP32");
    int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : 0;
    const float *ck_debug_outproj_fp32_input = NULL;
    const char *debug_mlp_down_env = getenv("CK_V7_DEBUG_MLP_DOWN_FP32");
    int debug_mlp_down_fp32 = debug_mlp_down_env ? (atoi(debug_mlp_down_env) != 0) : 0;
    const float *ck_debug_mlp_down_fp32_input = NULL;
"""
    )
    if is_qwen3vl:
        lines.append(
            """    if (ck_qwen3vl_prefill_bridge_is_active()) {
        debug_outproj_fp32 = 1;
        debug_mlp_down_fp32 = 1;
    }
"""
        )

    if profile:
        lines.append("    CK_PROFILE_VARS();")
        lines.append("")

    _annotate_kv_transpose_roles(ops)
    residual_add_count_total = 0

    for seq_idx, op in enumerate(ops):
        op_type = str(op.get("op", ""))
        if op_type == "dense_embedding_lookup":
            lines.append(f"    if (stop_seq == {seq_idx}) return;")
            if (
                scale_embeddings_sqrt_dim
                and not embed_scale_emitted
                and int(op.get("layer", -1)) == -1
            ):
                lines.append(_emit_embedding_scale_block("num_tokens", dump))
                embed_scale_emitted = True
                lines.append("")
            continue

        if is_qwen3vl and op_type == "quantize_mlp_down_input" and str(op.get("function", "")) in {"quantize_row_q8_0", "quantize_row_q8_k"}:
            op_code = _emit_prefill_quant_debug_override(
                op,
                seq_idx,
                config,
                debug_flag_name="debug_mlp_down_fp32",
                debug_input_name="ck_debug_mlp_down_fp32_input",
                profile=profile,
            )
        elif is_qwen3vl and op_type == "mlp_down" and str(op.get("function", "")) in {"gemm_nt_q4_k_q8_k", "gemm_nt_q6_k_q8_k"}:
            op_code = _emit_prefill_gemm_fp32_override(
                op,
                seq_idx,
                debug_flag_name="debug_mlp_down_fp32",
                debug_input_name="ck_debug_mlp_down_fp32_input",
                profile=profile,
                dump=dump,
            )
        else:
            op_code = emit_prefill_op(op, seq_idx, config, profile=profile, dump=dump)
            if is_qwen3vl and op_type == "rope_qk":
                op_code = op_code.replace("mrope_qk_text(", "ck_qwen3vl_prefill_mrope_qk(")
        lines.append(op_code)
        lines.append(f"    if (stop_seq == {seq_idx}) return;")
        if is_qwen3vl and op_type == "residual_add":
            residual_add_count_total += 1
            if residual_add_count_total % 2 == 0:
                deepstack_layer = residual_add_count_total // 2 - 1
                if 0 <= deepstack_layer < num_deepstack_layers:
                    lines.append(f"    ck_qwen3vl_prefill_deepstack_add(model, {deepstack_layer}, num_tokens);")
        lines.append("")

    lines.append("    model->pos = num_tokens;")
    if is_qwen3vl:
        lines.append("    model->rope_pos = ck_qwen3vl_prefill_bridge_is_active() ? ck_qwen3vl_prefill_bridge_text_pos() : num_tokens;")
        lines.append("    ck_qwen3vl_prefill_bridge_clear();")
    else:
        lines.append("    model->rope_pos = num_tokens;")
    lines.append("}")
    return "\n".join(lines)


def emit_multimodal_bridge_api(ops: List[Dict], config: Dict | None = None) -> str:
    """Emit small helpers for encoder->decoder stitched prefill."""
    embedding_op = _find_embedding_header_op(ops)
    if not embedding_op:
        return ""

    args_list = embedding_op.get("args", [])
    if not isinstance(args_list, list) or not args_list:
        return ""

    func = str(embedding_op.get("function", "") or "").strip()
    if not func:
        return ""

    config = dict(config or {})
    is_qwen3vl = _is_qwen3vl_multimodal_config(config)

    token_ids_expr = _find_arg_expr(args_list, arg_name="token_ids") or "(int32_t*)(model->bump + A_TOKEN_IDS)"
    token_embeddings_expr = _find_arg_expr(args_list, arg_name="token_embeddings")
    pos_embeddings_expr = _find_arg_expr(args_list, arg_name="pos_embeddings") or "NULL"
    output_expr = _find_arg_expr(args_list, arg_name="output") or "(float*)(model->bump + A_EMBEDDED_INPUT)"
    vocab_size_expr = _find_arg_expr(args_list, arg_name="vocab_size") or "VOCAB_SIZE"
    embed_dim_expr = _find_arg_expr(args_list, arg_name="embed_dim") or "EMBED_DIM"
    aligned_embed_dim_expr = _find_arg_expr(args_list, arg_name="aligned_embed_dim") or embed_dim_expr
    context_window_expr = _find_arg_expr(args_list, arg_name="context_window") or "MAX_SEQ_LEN"
    add_pos_expr = _find_arg_expr(args_list, arg_name="add_pos") or "0"

    if not token_embeddings_expr:
        return ""

    qwen_prepare_block = ""
    if is_qwen3vl:
        qwen_prepare_block = """
        if (prefix_embed_dim > aligned_embed_dim) {
            if (prefix_grid_x > 0 && prefix_grid_y > 0) {
                int prep_rc = ck_qwen3vl_prefill_bridge_prepare(
                    prefix_embeddings,
                    prefix_tokens,
                    prefix_embed_dim,
                    prefix_grid_x,
                    prefix_grid_y,
                    prefix_text_pos
                );
                if (prep_rc != 0) return prep_rc;
            } else {
                const int side = (int)(sqrt((double)prefix_tokens) + 0.5);
                if (side > 0 && side * side == prefix_tokens) {
                    int prep_rc = ck_qwen3vl_prefill_bridge_prepare(prefix_embeddings, prefix_tokens, prefix_embed_dim, side, side, side);
                    if (prep_rc != 0) return prep_rc;
                }
            }
        }
"""

    return f"""
/* ============================================================================
 * MULTIMODAL BRIDGE HELPERS
 * ============================================================================
 * These helpers keep the stable token-only API intact, but allow an
 * orchestrator to:
 *   1. write encoder-produced prefix embeddings into A_EMBEDDED_INPUT
 *   2. embed text tokens after that prefix using the model's own embedding op
 *   3. run the normal decoder body/footer from the prepared embedding buffer
 * ============================================================================ */
static int ck_write_embeddings_at(CKModel *model, const float *embeddings, int count, int start_pos) {{
    if (!model || !embeddings || count <= 0) return -1;
    if (start_pos < 0 || start_pos >= ({context_window_expr})) return -2;
    if (count > ({context_window_expr}) - start_pos) {{
        count = ({context_window_expr}) - start_pos;
    }}

    int32_t *token_base = {token_ids_expr};
    float *out_base = {output_expr};
    const int aligned_embed_dim = ({aligned_embed_dim_expr});

    memset(token_base + (size_t)start_pos, 0, (size_t)count * sizeof(int32_t));
    memcpy(out_base + (size_t)start_pos * (size_t)aligned_embed_dim,
           embeddings,
           (size_t)count * (size_t)aligned_embed_dim * sizeof(float));
    return count;
}}

static int ck_write_embeddings_at_ex(CKModel *model, const float *embeddings, int count, int row_dim, int start_pos) {{
    if (!model || !embeddings || count <= 0) return -1;
    if (row_dim <= 0) row_dim = ({aligned_embed_dim_expr});
    if (row_dim < ({embed_dim_expr})) return -3;
    if (start_pos < 0 || start_pos >= ({context_window_expr})) return -2;
    if (count > ({context_window_expr}) - start_pos) {{
        count = ({context_window_expr}) - start_pos;
    }}

    int32_t *token_base = {token_ids_expr};
    float *out_base = {output_expr};
    const int aligned_embed_dim = ({aligned_embed_dim_expr});
    for (int i = 0; i < count; ++i) {{
        const float *src = embeddings + (size_t)i * (size_t)row_dim;
        float *dst = out_base + (size_t)(start_pos + i) * (size_t)aligned_embed_dim;
        memcpy(dst, src, (size_t)aligned_embed_dim * sizeof(float));
        token_base[start_pos + i] = 0;
    }}
    return count;
}}

static int ck_embed_tokens_at(CKModel *model, const int32_t *tokens, int count, int start_pos) {{
    if (!model || !tokens || count <= 0) return -1;
    if (start_pos < 0 || start_pos >= ({context_window_expr})) return -2;
    if (count > ({context_window_expr}) - start_pos) {{
        count = ({context_window_expr}) - start_pos;
    }}

    int32_t *token_base = {token_ids_expr};
    float *out_base = {output_expr};
    const float *pos_base = {pos_embeddings_expr};
    const int aligned_embed_dim = ({aligned_embed_dim_expr});
    const float *pos_slice = pos_base ? (pos_base + (size_t)start_pos * (size_t)aligned_embed_dim) : NULL;

    memcpy(token_base + (size_t)start_pos, tokens, (size_t)count * sizeof(int32_t));
    {func}(
        tokens,
        count,
        ({vocab_size_expr}),
        {token_embeddings_expr},
        pos_slice,
        out_base + (size_t)start_pos * (size_t)aligned_embed_dim,
        ({embed_dim_expr}),
        aligned_embed_dim,
        ({context_window_expr}) - start_pos,
        ({add_pos_expr})
    );
    return count;
}}

CK_EXPORT int ck_model_write_embeddings(const float *embeddings, int count, int start_pos) {{
    return ck_write_embeddings_at(g_model, embeddings, count, start_pos);
}}

CK_EXPORT int ck_model_write_embeddings_ex(const float *embeddings, int count, int row_dim, int start_pos) {{
    return ck_write_embeddings_at_ex(g_model, embeddings, count, row_dim, start_pos);
}}

CK_EXPORT int ck_model_embed_tokens_at(const int32_t *tokens, int count, int start_pos) {{
    return ck_embed_tokens_at(g_model, tokens, count, start_pos);
}}

CK_EXPORT int ck_model_forward_from_embeddings(int total_tokens, float *output) {{
    if (!g_model) return -1;
    if (total_tokens <= 0) return -2;
    if (total_tokens > ({context_window_expr})) {{
        total_tokens = ({context_window_expr});
    }}
    g_model->pos = 0;
    g_model->rope_pos = 0;
    g_model->bridge_has_explicit_positions = 0;
    ck_prefill_from_embedded(g_model, total_tokens);
    if (output) memcpy(output, g_model->logits, VOCAB_SIZE * sizeof(float));
    return 0;
}}

CK_EXPORT int ck_model_forward_mixed_grid_ex(const float *prefix_embeddings,
                                             int prefix_tokens,
                                             int prefix_embed_dim,
                                             int prefix_grid_x,
                                             int prefix_grid_y,
                                             int prefix_text_pos,
                                             const int32_t *tokens,
                                             int token_count,
                                             float *output) {{
    if (!g_model) return -1;
    if (prefix_tokens < 0 || token_count < 0) return -2;
    if (prefix_tokens + token_count <= 0) return -3;
    if (prefix_tokens + token_count > ({context_window_expr})) return -4;
    if (prefix_tokens > 0 && !prefix_embeddings) return -5;
    if (token_count > 0 && !tokens) return -6;
    if ((prefix_grid_x > 0) != (prefix_grid_y > 0)) return -7;
    if (prefix_grid_x < 0 || prefix_grid_y < 0) return -8;
    if ((prefix_grid_x > 0 || prefix_grid_y > 0) && prefix_tokens <= 0) return -9;
    if (prefix_grid_x > 0 && prefix_grid_y > 0 && prefix_grid_x * prefix_grid_y != prefix_tokens) return -10;

    memset(g_model->kv_cache, 0, KV_CACHE_SIZE);
    g_model->pos = 0;
    g_model->rope_pos = 0;
    g_model->bridge_has_explicit_positions = 0;

    const int aligned_embed_dim = ({aligned_embed_dim_expr});
    if (prefix_tokens > 0) {{
        if (prefix_embed_dim <= 0) prefix_embed_dim = aligned_embed_dim;
        if (prefix_embed_dim < ({embed_dim_expr})) return -11;
{qwen_prepare_block}        int rc = ck_write_embeddings_at_ex(g_model, prefix_embeddings, prefix_tokens, prefix_embed_dim, 0);
        if (rc < 0) return rc;
        ck_prefill_from_embedded(g_model, prefix_tokens);
    }}

    for (int i = 0; i < token_count; ++i) {{
        ck_decode(g_model, tokens[i]);
    }}

    if (output) memcpy(output, g_model->logits, VOCAB_SIZE * sizeof(float));
    return 0;
}}

CK_EXPORT int ck_model_forward_mixed_ex(const float *prefix_embeddings,
                                        int prefix_tokens,
                                        int prefix_embed_dim,
                                        const int32_t *tokens,
                                        int token_count,
                                        float *output) {{
    return ck_model_forward_mixed_grid_ex(
        prefix_embeddings,
        prefix_tokens,
        prefix_embed_dim,
        0,
        0,
        0,
        tokens,
        token_count,
        output
    );
}}

CK_EXPORT int ck_model_forward_mixed(const float *prefix_embeddings,
                                     int prefix_tokens,
                                     const int32_t *tokens,
                                     int token_count,
                                     float *output) {{
    return ck_model_forward_mixed_ex(prefix_embeddings, prefix_tokens, ({aligned_embed_dim_expr}), tokens, token_count, output);
}}
"""


def generate_prefill(ir_path: Path, layout_path: Path = None, profile: bool = False, dump: bool = False) -> str:
    """Generate prefill C code from IR.

    The IR already contains everything we need - just read and emit.
    If profile=True, emit CK_PROFILE timing wrappers around each kernel call.
    """
    with open(ir_path, "r", encoding="utf-8") as f:
        ir = json.load(f)

    ops = ir.get("operations", [])
    config = ir.get("config", {})

    parts = []

    # Header comment
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    parts.append(f'''/*
 * Auto-generated PREFILL code by codegen_prefill_v8.py
 * Generated: {now}
 * Model: {config.get("model", "unknown")}
 * Mode: prefill
 * Ops: {len(ops)}
 */
''')

    parts.append(emit_prefill_function(ops, config, profile=profile, dump=dump))
    parts.append(emit_prefill_from_embedded_function(ops, config, profile=profile, dump=dump))
    bridge_api = emit_multimodal_bridge_api(ops, config)
    if bridge_api:
        parts.append(bridge_api)

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
