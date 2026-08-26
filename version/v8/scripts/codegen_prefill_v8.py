#!/usr/bin/env python3
from __future__ import annotations

"""
codegen_prefill_v8.py - Generate C code for PREFILL mode from lowered IR.

This generates ck_prefill() which processes multiple tokens at once.
The IR (lowered_prefill_call.json) already has function names and expressions.
Token-count arguments are selected from their semantic source or name.

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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from codegen_capabilities_v8 import (
    activation_quantized_row_bytes_expr,
    is_q4_q6_q8_linear,
    resolved_activation_quantization_emission,
    resolved_quantized_linear_emission,
)


def _annotate_kv_transpose_roles(ops: List[Dict]) -> None:
    """Mark synthetic transpose ops with K/V role and per-layer head geometry."""

    def _arg_expr(op: Dict, name: str) -> Optional[str]:
        target = name.lower()
        for arg in op.get("args", []) or []:
            if str(arg.get("name", "")).lower() == target:
                expr = str(arg.get("expr", "")).strip()
                return expr or None
        return None

    layer_dims: Dict[int, Dict[str, str]] = {}
    for op in ops:
        if op.get("op") != "qk_norm":
            continue
        layer = int(op.get("layer", 0))
        num_heads = _arg_expr(op, "num_heads")
        num_kv_heads = _arg_expr(op, "num_kv_heads")
        head_dim = _arg_expr(op, "head_dim")
        if num_heads and num_kv_heads and head_dim:
            layer_dims[layer] = {
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }

    layer_kv_count: Dict[int, int] = {}
    for op in ops:
        op_name = op.get("op")
        if op_name not in {
            "transpose_qkv_to_head_major",
            "transpose_kv_to_head_major",
            "transpose_attn_out_to_token_major",
            "transpose_cross_q_to_head_major",
            "transpose_cross_kv_to_head_major",
            "transpose_cross_attn_out_to_token_major",
            "kv_cache_batch_copy",
            "kv_cache_store_batch_bf16",
            "kv_cache_store_batch_f16",
        }:
            continue
        layer = int(op.get("layer", 0))
        dims = layer_dims.get(layer)
        if dims:
            op["_num_heads"] = dims["num_heads"]
            op["_num_kv_heads"] = dims["num_kv_heads"]
            op["_head_dim"] = dims["head_dim"]
        if op_name == "transpose_kv_to_head_major":
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
    if "q8_0_q8_0" in fn:
        return f"(size_t)(num_tokens - 1) * (size_t)({embed_dim} / QK8_0) * sizeof(block_q8_0)"
    if "fp32" in fn or "f32" in fn or "bf16" in fn:
        return f"(size_t)(num_tokens - 1) * (size_t){embed_dim} * sizeof(float)"
    return f"(size_t)(num_tokens - 1) * (size_t){embed_dim} * sizeof(float)"


def emit_prefill_op(
    op: Dict,
    seq_idx: int,
    config: Dict,
    profile: bool = False,
    dump: bool = False,
    *,
    segment_plan_available: bool = False,
) -> str:
    """Emit a single op call for prefill mode.

    The IR already provides:
      - function: kernel function name
      - args[]: each with name, source, expr

    We substitute num_tokens for token-count arguments and fix memcpy size.
    If profile=True, emit CK_PROFILE_BEGIN/END timing wrappers.
    """
    func = op.get("function", "unknown")
    physical_execution = op.get("resolved_physical_execution")
    emitted_func = (
        str(physical_execution.get("function", "") or "")
        if isinstance(physical_execution, dict)
        else ""
    ) or func
    op_type = op.get("op", "unknown")
    layer = op.get("layer", -1)
    section = op.get("section", "")
    op_instance_idx = int(op.get("op_instance_idx", op.get("instance", 0)) or 0)
    args_list = op.get("args", [])
    linear_emission = resolved_quantized_linear_emission(op)
    segmented_row_function = (
        _segmented_row_provider(op, config) if segment_plan_available else ""
    )
    segmented_attention_provider = (
        _segmented_attention_provider(op) if segment_plan_available else {}
    )
    segmented_attention_function = (
        str(segmented_attention_provider.get("function", ""))
        if isinstance(segmented_attention_provider, dict)
        else ""
    )
    quantization_emission = resolved_activation_quantization_emission(op)
    if op_type.startswith("quantize_") and quantization_emission is None:
        raise RuntimeError(
            f"quantization op {op_type!r} requires resolved map-owned codegen capability"
        )

    # Handle special auto-inserted ops
    if op_type == "final_logit_softcap" and str(config.get("logits_layout", "auto")).lower() == "last":
        vocab_size = int(config.get("vocab_size", 151936))
        cap = float(config.get("final_logit_softcapping", 0.0) or 0.0)
        return f"""    /* Op {seq_idx}: {func} ({op_type}) layer={layer} */
    {func}(
        (float*)(model->bump + A_LOGITS),
        1,
        {vocab_size},
    {cap}
    );"""

    if op_type == "final_logit_scale" and str(config.get("logits_layout", "auto")).lower() == "last":
        vocab_size = int(config.get("vocab_size", 151936))
        scale = float(config.get("logit_scale", 1.0) or 1.0)
        return f"""    /* Op {seq_idx}: {func} ({op_type}) layer={layer} */
    {func}(
        (float*)(model->bump + A_LOGITS),
        1,
        {vocab_size},
        {scale}
    );"""

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
            gemm_a_expr = _find_arg_expr(
                args_list, source_prefix="activation:", arg_name="A"
            )
            gemm_b_expr = _find_arg_expr(
                args_list, source_prefix="weight:", arg_name="B"
            )
            gemm_bias_expr = _find_arg_expr(
                args_list, source_prefix="weight_f:", arg_name="bias"
            )
            gemm_c_expr = _find_arg_expr(
                args_list, source_prefix="output:", arg_name="C"
            )
            # Some exact numerical contracts require the resolved GEMM provider
            # even for M=1. Lower 3 carries this map-owned policy explicitly.
            # Quantized providers omit it and use their registered row GEMV.
            preserve_provider = (
                (op.get("call_abi") or {}).get("last_token_dispatch")
                == "preserve_provider"
            )
            if preserve_provider and gemm_a_expr and gemm_b_expr and gemm_c_expr:
                dump_code = ""
                if dump:
                    raw_output = gemm_c_expr.replace("(float*)", "").replace("(void*)", "").strip()
                    dump_code = f'''\n    #ifdef CK_PARITY_DUMP
    ck_dump_tensor((float*){raw_output}, -1, "logits", {vocab_size});
    #endif'''
                return f"""    /* Op {seq_idx}: logits (last-only exact GEMM contract) */
    {func}(
        (const float*)({gemm_a_expr}) + (size_t)(num_tokens - 1) * {embed_dim},
        {gemm_b_expr},
        {gemm_bias_expr or "NULL"},
        {gemm_c_expr},
        1,
        {vocab_size},
        {embed_dim}
    );
    ck_debug_export_hidden(model, -1, "logits", (const float*){gemm_c_expr}, VOCAB_SIZE);{dump_code}"""
            if linear_emission is None:
                call_abi = op.get("call_abi") or {}
                if call_abi.get("owner") != "legacy_compatibility":
                    raise RuntimeError(
                        "prefill logits(last): resolved GEMM provider must declare "
                        "a map-owned row-quantized provider or preserve the provider"
                    )
                # Legacy families still carry version-0 ABI bindings. Keep their
                # established row-provider mapping bounded to that explicit owner.
                gemv_func = func
                if gemv_func.startswith("gemm_nt_"):
                    gemv_func = "gemv_" + gemv_func[len("gemm_nt_"):]
                elif gemv_func.startswith("gemm_"):
                    gemv_func = "gemv_" + gemv_func[len("gemm_"):]
                else:
                    raise RuntimeError(
                        "prefill logits(last): legacy GEMM provider has no "
                        "registered row-provider naming contract"
                    )
            else:
                gemv_func = linear_emission.get("row_quantized_function")
                if not gemv_func:
                    raise RuntimeError(
                        "prefill logits(last): resolved quantized linear capability "
                        "has no row_quantized provider"
                    )
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
            dump_code = ""
            if dump:
                raw_output = output_expr.replace("(float*)", "").replace("(void*)", "").strip()
                dump_code = f'''\n    #ifdef CK_PARITY_DUMP
    ck_dump_tensor((float*){raw_output}, -1, "logits", {vocab_size});
    #endif'''
            return f"""    /* Op {seq_idx}: logits (last-only) */
    {gemv_func}(
        {output_expr},
        {weight_expr},
        (const void*)(((const uint8_t*)({input_expr})) + {row_offset_expr}),
        {vocab_size},
        {embed_dim}
    );
    ck_debug_export_hidden(model, -1, "logits", (const float*){output_expr}, VOCAB_SIZE);{dump_code}"""

    if op_type == "kv_cache_batch_copy":
        # Copy K/V from scratch (head-major after transpose) to KV cache
        # Scratch layout: [num_kv_heads, num_tokens, head_dim] (compact, head-major)
        # KV cache layout: [num_kv_heads, max_seq_len, head_dim] (with stride, head-major)
        layer = op.get("layer", 0)
        num_kv_heads = op.get("_num_kv_heads", config.get("num_kv_heads", 2))
        head_dim = op.get("_head_dim", config.get("head_dim", 64))
        context_len = config.get("context_len", config.get("context_length", 1024))
        decode_kv_cache_dtype = str(config.get("decode_kv_cache_dtype", "fp32") or "fp32").strip().lower()
        decode_uses_fp16_kv = decode_kv_cache_dtype in {"fp16", "f16"}
        k_offsets = config.get("layer_k_cache_offset") or []
        v_offsets = config.get("layer_v_cache_offset") or []
        compact_kv_layout = bool(k_offsets or v_offsets)
        if (
            isinstance(k_offsets, list)
            and layer < len(k_offsets)
            and k_offsets[layer] is not None
            and int(k_offsets[layer]) >= 0
        ):
            k_base_expr = f"kv_cache + ({int(k_offsets[layer])}ULL*cache_stride)"
        else:
            if compact_kv_layout:
                raise RuntimeError(
                    f"kv_cache_batch_copy layer {layer} does not own compact K storage"
                )
            k_base_expr = f"kv_cache + (1ULL*({layer}*2)*Hkv*cache_stride*D)"
        if (
            isinstance(v_offsets, list)
            and layer < len(v_offsets)
            and v_offsets[layer] is not None
            and int(v_offsets[layer]) >= 0
        ):
            v_base_expr = f"kv_cache + ({int(v_offsets[layer])}ULL*cache_stride)"
        else:
            if compact_kv_layout:
                raise RuntimeError(
                    f"kv_cache_batch_copy layer {layer} does not own compact V storage"
                )
            v_base_expr = f"kv_cache + (1ULL*({layer}*2+1)*Hkv*cache_stride*D)"
        if decode_uses_fp16_kv:
            return f"""    /* Op {seq_idx}: kv_cache_batch_copy layer={layer} */
    /* Copy K/V from head-major scratch to FP16 KV cache for subsequent decode */
    {{
        const int Hkv = {num_kv_heads};
        const int D = {head_dim};
        const int cache_stride = {context_len};
        float *k_scratch = (float*)(model->bump + A_K_SCRATCH);
        float *v_scratch = (float*)(model->bump + A_V_SCRATCH);
        uint16_t *kv_cache = (uint16_t*)model->kv_cache_f16;
        for (int h = 0; h < Hkv; h++) {{
            uint16_t *k_dst = {k_base_expr} + ((size_t)h*cache_stride + (size_t)prefill_start_pos)*D;
            uint16_t *v_dst = {v_base_expr} + ((size_t)h*cache_stride + (size_t)prefill_start_pos)*D;
            const float *k_src = k_scratch + h*num_tokens*D;
            const float *v_src = v_scratch + h*num_tokens*D;
            for (int t = 0; t < num_tokens; ++t) {{
                uint16_t *kd = k_dst + (size_t)t*D;
                uint16_t *vd = v_dst + (size_t)t*D;
                const float *ks = k_src + (size_t)t*D;
                const float *vs = v_src + (size_t)t*D;
                for (int d = 0; d < D; ++d) {{
                    kd[d] = ck_fp32_to_fp16_soft(ks[d]);
                    vd[d] = ck_fp32_to_fp16_soft(vs[d]);
                }}
            }}
        }}
    }}"""
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
                {k_base_expr} + ((size_t)h*cache_stride + (size_t)prefill_start_pos)*D,
                k_scratch + h*num_tokens*D,
                (size_t)num_tokens * D * sizeof(float)
            );
            /* V: copy from scratch[h, 0:num_tokens, :] to cache[h, 0:num_tokens, :] */
            memcpy(
                {v_base_expr} + ((size_t)h*cache_stride + (size_t)prefill_start_pos)*D,
                v_scratch + h*num_tokens*D,
                (size_t)num_tokens * D * sizeof(float)
            );
        }}
    }}"""

    # Handle transpose_kv_to_head_major: convert from [T, Hkv*D] to [Hkv, T, D]
    if op_type == "transpose_cross_q_to_head_major":
        num_heads = config.get("num_heads", 1)
        head_dim = config.get("head_dim", 1)
        return f"""    /* Op {seq_idx}: transpose_cross_q_to_head_major layer={layer} */
    {{
        const int H = {num_heads};
        const int D = {head_dim};
        float *buf = (float*)(model->bump + A_CROSS_Q_SCRATCH);
        float *tmp = (float*)(model->bump + A_CROSS_LAYOUT_SCRATCH);
        for (int t = 0; t < num_tokens; ++t) {{
            for (int h = 0; h < H; ++h) {{
                memcpy(tmp + ((size_t)h*num_tokens + t)*D,
                       buf + ((size_t)t*H + h)*D,
                       (size_t)D*sizeof(float));
            }}
        }}
        memcpy(buf, tmp, (size_t)H*num_tokens*D*sizeof(float));
    }}"""

    if op_type == "transpose_cross_kv_to_head_major":
        num_heads = config.get("num_heads", 1)
        head_dim = config.get("head_dim", 1)
        encoder_tokens = config.get("encoder_memory_length", 0)
        kind = str(op.get("_cross_kv_kind", "key"))
        cache_name = "A_CROSS_K_CACHE" if kind == "key" else "A_CROSS_V_CACHE"
        return f"""    /* Op {seq_idx}: transpose_cross_{kind}_to_head_major layer={layer} */
    {{
        const int H = {num_heads};
        const int T = {encoder_tokens};
        const int D = {head_dim};
        const size_t layer_stride = (size_t)H * (size_t)T * (size_t)D;
        float *buf = (float*)(model->bump + {cache_name}) + (size_t){layer} * layer_stride;
        float *tmp = (float*)(model->bump + A_CROSS_LAYOUT_SCRATCH);
        for (int t = 0; t < T; ++t) {{
            for (int h = 0; h < H; ++h) {{
                memcpy(tmp + ((size_t)h*T + t)*D,
                       buf + ((size_t)t*H + h)*D,
                       (size_t)D*sizeof(float));
            }}
        }}
        memcpy(buf, tmp, (size_t)H*T*D*sizeof(float));
    }}"""

    if op_type == "transpose_cross_attn_out_to_token_major":
        num_heads = config.get("num_heads", 1)
        head_dim = config.get("head_dim", 1)
        return f"""    /* Op {seq_idx}: transpose_cross_attn_out_to_token_major layer={layer} */
    {{
        const int H = {num_heads};
        const int D = {head_dim};
        float *buf = (float*)(model->bump + A_CROSS_ATTN_SCRATCH);
        float *tmp = (float*)(model->bump + A_CROSS_LAYOUT_SCRATCH);
        for (int h = 0; h < H; ++h) {{
            for (int t = 0; t < num_tokens; ++t) {{
                memcpy(tmp + ((size_t)t*H + h)*D,
                       buf + ((size_t)h*num_tokens + t)*D,
                       (size_t)D*sizeof(float));
            }}
        }}
        memcpy(buf, tmp, (size_t)H*num_tokens*D*sizeof(float));
    }}"""

    # Handle transpose_kv_to_head_major: convert from [T, Hkv*D] to [Hkv, T, D]
    if op_type == "transpose_kv_to_head_major":
        num_kv_heads = op.get("_num_kv_heads", config.get("num_kv_heads", 2))
        head_dim = op.get("_head_dim", config.get("head_dim", 64))
        # _is_k is set by emit_prefill_function preprocessing
        is_k = op.get("_is_k", True)
        scratch_name = "A_K_SCRATCH" if is_k else "A_V_SCRATCH"
        max_tokens = config.get("context_len", config.get("context_length", 1024))
        return f"""    /* Op {seq_idx}: transpose_{("k" if is_k else "v")}_to_head_major layer={layer} */
    /* Transpose from [T, Hkv*D] (token-major GEMM output) to [Hkv, T, D] (head-major for attention) */
    {{
        const int Hkv = {num_kv_heads};
        const int D = {head_dim};
        float *buf = (float*)(model->bump + {scratch_name});
        /* Reuse activation scratch to avoid huge per-op static BSS allocations. */
        float *_temp_buf = (float*)(model->bump + A_LAYER_OUTPUT);
        ck_layout_token_to_head_f32(buf, _temp_buf, num_tokens, Hkv, D);
        /* Copy back */
        memcpy(buf, _temp_buf, (size_t)Hkv * num_tokens * D * sizeof(float));
        ck_debug_export_hidden(
            model,
            {layer},
            "{("k" if is_k else "v")}_head_major",
            (const float*)buf,
            Hkv * num_tokens * D
        );
    }}"""

    # Handle transpose_qkv_to_head_major for Q: convert from [T, H*D] to [H, T, D]
    if op_type == "transpose_qkv_to_head_major":
        qkv_type = op.get("_qkv_type", "q")
        if qkv_type == "q":
            num_heads = op.get("_num_heads", config.get("num_heads", 14))
            head_dim = op.get("_head_dim", config.get("head_dim", 64))
            scratch_name = "A_Q_SCRATCH"
            max_tokens = config.get("context_len", config.get("context_length", 1024))
            return f"""    /* Op {seq_idx}: transpose_q_to_head_major layer={layer} */
    /* Transpose from [T, H*D] (token-major GEMM output) to [H, T, D] (head-major for attention) */
    {{
        const int H = {num_heads};
        const int D = {head_dim};
        float *buf = (float*)(model->bump + {scratch_name});
        /* Reuse activation scratch to avoid huge per-op static BSS allocations. */
        float *_temp_buf = (float*)(model->bump + A_LAYER_OUTPUT);
        ck_layout_token_to_head_f32(buf, _temp_buf, num_tokens, H, D);
        /* Copy back */
        memcpy(buf, _temp_buf, (size_t)H * num_tokens * D * sizeof(float));
    }}"""

    # Handle transpose_attn_out_to_token_major: convert from [H, T, D] to [T, H*D]
    # This is the reverse of the Q transpose - needed after attention before out_proj
    if op_type == "transpose_attn_out_to_token_major":
        num_heads = op.get("_num_heads", config.get("num_heads", 14))
        head_dim = op.get("_head_dim", config.get("head_dim", 64))
        max_tokens = config.get("context_len", config.get("context_length", 1024))
        dump_block = ""
        if dump:
            dump_block = f"""
    #ifdef CK_PARITY_DUMP
    ck_dump_tensor((float*)(model->bump + A_ATTN_SCRATCH), {layer}, "kqv_out", (num_tokens) * ({num_heads}) * ({head_dim}));
    #endif"""
        return f"""    /* Op {seq_idx}: transpose_attn_out_to_token_major layer={layer} */
    /* Transpose from [H, T, D] (head-major attention output) to [T, H*D] (token-major for out_proj) */
    {{
        const int H = {num_heads};
        const int D = {head_dim};
        float *buf = (float*)(model->bump + A_ATTN_SCRATCH);
        /* Reuse activation scratch to avoid huge per-op static BSS allocations. */
        float *_temp_buf = (float*)(model->bump + A_LAYER_OUTPUT);
        ck_layout_head_to_token_f32(buf, _temp_buf, H, num_tokens, D);
        /* Copy back */
        memcpy(buf, _temp_buf, (size_t)num_tokens * H * D * sizeof(float));
    }}{dump_block}"""

    embed_dim = config.get("embed_dim", 896)

    lines = []
    lines.append(f"    /* Op {seq_idx}: {func} ({op_type}) layer={layer} */")
    if profile:
        lines.append(f"    CK_PROFILE_BEGIN();")

    # Build argument list with substitutions
    args = []
    arg_expr_by_name: Dict[str, str] = {}
    shared_kv_source_layer = layer
    layer_kv_source = config.get("layer_kv_source")
    if op_type in ("attn", "attn_sliding") and isinstance(layer_kv_source, list):
        try:
            if 0 <= int(layer) < len(layer_kv_source):
                shared_kv_source_layer = int(layer_kv_source[int(layer)])
        except (TypeError, ValueError):
            shared_kv_source_layer = layer
    use_prefill_shared_kv_cache = (
        op_type in ("attn", "attn_sliding")
        and isinstance(shared_kv_source_layer, int)
        and int(shared_kv_source_layer) != int(layer)
    )

    def _prefill_kv_cache_expr(which: str, source_layer: int) -> str:
        offsets_key = "layer_k_cache_offset" if which == "k" else "layer_v_cache_offset"
        offsets = config.get(offsets_key)
        if (
            isinstance(offsets, list)
            and 0 <= int(source_layer) < len(offsets)
            and int(offsets[int(source_layer)]) >= 0
        ):
            return f"(model->kv_cache + {int(offsets[int(source_layer)])}ULL*MAX_SEQ_LEN)"
        if isinstance(offsets, list) and offsets:
            raise RuntimeError(
                f"prefill {which.upper()} cache source layer {source_layer} "
                "does not own compact KV storage"
            )
        try:
            kv_cache_head_dim = int(config.get("kv_cache_head_dim", config.get("head_dim", 1)) or config.get("head_dim", 1) or 1)
        except Exception:
            kv_cache_head_dim = int(config.get("head_dim", 1) or 1)
        if kv_cache_head_dim <= 0:
            kv_cache_head_dim = 1
        term = f"({int(source_layer)}*2)" if which == "k" else f"({int(source_layer)}*2+1)"
        return f"(model->kv_cache + {term}*NUM_KV_HEADS*MAX_SEQ_LEN*{kv_cache_head_dim})"

    dynamic_token_arg_names = {
        "seq_len",
        "num_tokens",
        "token_count",
        "tokens",
        "rows",
        "m",
    }
    for arg in args_list:
        expr = arg.get("expr", "0")
        source = arg.get("source", "")
        name = arg.get("name", "")
        name_lc = str(name).lower()

        # Substitute num_tokens for token count parameters. Mamba selective
        # scan is single-batch prompt prefill: batch stays 1, seq_len is the
        # runtime token count.
        if source == "dim:seq_len":
            expr = "num_tokens"
        elif source in ("param:seq_len", "runtime:seq_len"):
            expr = "num_tokens"
        elif source == "runtime:query_tokens":
            expr = "num_tokens"
        elif source == "runtime:prefill_start_pos":
            expr = "prefill_start_pos"
        elif name_lc in dynamic_token_arg_names and source in {"dim:_m", "param:_m", "runtime:kv_tokens", "runtime:cache_len"}:
            expr = "num_tokens"
        elif name_lc in {"seq_len", "num_tokens", "token_count", "tokens", "rows"} and str(expr).isdigit():
            expr = "num_tokens"
        elif use_prefill_shared_kv_cache and name_lc in {"k", "k_cache"}:
            expr = f"(const float*){_prefill_kv_cache_expr('k', int(shared_kv_source_layer))}"
        elif use_prefill_shared_kv_cache and name_lc in {"v", "v_cache"}:
            expr = f"(const float*){_prefill_kv_cache_expr('v', int(shared_kv_source_layer))}"
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
            expr = "MAX_SEQ_LEN" if use_prefill_shared_kv_cache else "num_tokens"

        args.append(expr)
        name_key = str(name).lower()
        if name_key and expr and name_key not in arg_expr_by_name:
            arg_expr_by_name[name_key] = expr

    if op_type == "recurrent_core":
        state_in = arg_expr_by_name.get("state_in")
        num_heads = arg_expr_by_name.get("num_heads") or arg_expr_by_name.get("groups")
        state_dim = arg_expr_by_name.get("state_dim") or arg_expr_by_name.get("head_dim")
        if state_in and num_heads and state_dim:
            raw_state_in = (
                state_in.replace("(const float*)", "")
                .replace("(float*)", "")
                .replace("(void*)", "")
                .strip()
            )
            lines.append(
                f'    ck_debug_export_hidden(model, {layer}, "state_predelta", '
                f'(const float*){raw_state_in}, ({num_heads}) * ({state_dim}) * ({state_dim}));'
            )

    # Prefill quantization preserves token-major row storage. A kernel map may
    # additionally declare a grouped arithmetic provider whose output keeps
    # that same storage ABI.
    batch_quant_kind = quantization_emission

    if batch_quant_kind:
        x_expr = arg_expr_by_name.get("x") or arg_expr_by_name.get("input")
        y_expr = arg_expr_by_name.get("y") or arg_expr_by_name.get("output")
        k_expr = arg_expr_by_name.get("k") or str(config.get("embed_dim", "EMBED_DIM"))
        if x_expr and y_expr and k_expr:
            row_bytes_expr = activation_quantized_row_bytes_expr(batch_quant_kind, "_k")
            if op_type == "quantize_out_proj_input":
                lines.append(f"    ck_debug_outproj_fp32_input = (const float*)({x_expr});")
                lines.append(
                    f'    if (num_tokens > 1) ck_debug_export_hidden(model, {layer}, "attn_out_last", '
                    f'(const float*)(((const float*)({x_expr})) + (((size_t)num_tokens - 1u) * (size_t)({k_expr}))), '
                    f'(int)({k_expr}));'
                )
            lines.append("    {")
            if profile:
                lines.append("        CK_PROFILE_BEGIN();")
            lines.extend([
                f"        const float *_x_base = (const float*)({x_expr});",
                f"        uint8_t *_y_base = (uint8_t*)({y_expr});",
                f"        const int _k = (int)({k_expr});",
                f"        const size_t _row_bytes = {row_bytes_expr};",
            ])
            prefill_batch = batch_quant_kind.get("prefill_batch")
            if prefill_batch:
                lines.append(
                    f"        {prefill_batch['function']}("
                    "_x_base, (void*)_y_base, num_tokens, _k);"
                )
            else:
                lines.extend([
                    "        for (int _t = 0; _t < num_tokens; ++_t) {",
                    f"            {func}(_x_base + (size_t)_t * (size_t)_k, (void*)(_y_base + (size_t)_t * _row_bytes), _k);",
                    "        }",
                ])
            if profile:
                lines.append(f'        CK_PROFILE_END("prefill", "{func}", "{op_type}", {layer});')
            lines.append("    }")
            return "\n".join(lines)

    if op_type == "out_proj" and is_q4_q6_q8_linear(linear_emission):
        a_expr = arg_expr_by_name.get("a")
        b_expr = arg_expr_by_name.get("b")
        bias_expr = arg_expr_by_name.get("bias", "NULL")
        c_expr = arg_expr_by_name.get("c")
        m_expr = arg_expr_by_name.get("m", "num_tokens")
        n_expr = arg_expr_by_name.get("n")
        k_expr = arg_expr_by_name.get("k")
        fp32_func = linear_emission["fp32_activation_function"]
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

            raw_expr = c_expr.replace("(float*)", "").replace("(void*)", "").strip()
            lines.append(
                f'    ck_debug_export_hidden(model, {layer}, "out_proj", '
                f'(const float*){raw_expr}, ({m_expr}) * ({n_expr}));'
            )
            lines.append(
                f'    if (num_tokens > 1) ck_debug_export_hidden(model, {layer}, "out_proj_last", '
                f'(const float*)(((const float*){raw_expr}) + (((size_t)num_tokens - 1u) * (size_t)(EMBED_DIM))), '
                f'EMBED_DIM);'
            )

            if dump:
                raw_expr = c_expr.replace("(float*)", "").replace("(void*)", "").strip()
                size_expr = f"({m_expr}) * ({n_expr})"
                lines.append("    #ifdef CK_PARITY_DUMP")
                lines.append(f'    ck_dump_tensor((float*){raw_expr}, {layer}, "attn_out", {size_expr});')
                lines.append("    #endif")
            return "\n".join(lines)

    if op_type == "mlp_gate_up" and is_q4_q6_q8_linear(linear_emission):
        a_expr = arg_expr_by_name.get("a")
        b_expr = arg_expr_by_name.get("b")
        bias_expr = arg_expr_by_name.get("bias", "NULL")
        c_expr = arg_expr_by_name.get("c")
        m_expr = arg_expr_by_name.get("m", "num_tokens")
        n_expr = arg_expr_by_name.get("n")
        k_expr = arg_expr_by_name.get("k")
        row_gemv_func = linear_emission["row_quantized_function"]
        weight_row_bytes_expr = (
            f"(((size_t)({k_expr}) / {linear_emission['weight_block_elements']}u) * "
            f"{linear_emission['weight_block_bytes']}u)"
        )
        if a_expr and b_expr and c_expr and n_expr and k_expr:
            lines.append(f"    if (debug_prefill_mlp_gate_up_row_gemv || debug_prefill_mlp_gate_up_row_gemv_layer == {layer}) {{")
            if profile:
                lines.append("        CK_PROFILE_BEGIN();")
            lines.extend([
                f"        const int _ck_m = (int)({m_expr});",
                f"        const int _ck_n = (int)({n_expr});",
                f"        const int _ck_k = (int)({k_expr});",
                "        const int _ck_half = _ck_n / 2;",
                "        const size_t _ck_a_row_bytes = (size_t)(_ck_k / QK_K) * sizeof(block_q8_K);",
                f"        const size_t _ck_w_row_bytes = {weight_row_bytes_expr};",
                f"        const uint8_t *_ck_a_base = (const uint8_t*)({a_expr});",
                f"        const uint8_t *_ck_w_base = (const uint8_t*)({b_expr});",
                f"        const float *_ck_bias = (const float*)({bias_expr});",
                f"        float *_ck_c_base = (float*)({c_expr});",
                "        for (int _ck_t = 0; _ck_t < _ck_m; ++_ck_t) {",
                "            const void *_ck_xq8 = (const void*)(_ck_a_base + (size_t)_ck_t * _ck_a_row_bytes);",
                "            float *_ck_y = _ck_c_base + (size_t)_ck_t * (size_t)_ck_n;",
                f"            {row_gemv_func}(_ck_y, (const void*)_ck_w_base, _ck_xq8, _ck_half, _ck_k);",
                f"            {row_gemv_func}(_ck_y + _ck_half, (const void*)(_ck_w_base + (size_t)_ck_half * _ck_w_row_bytes), _ck_xq8, _ck_half, _ck_k);",
                "            if (_ck_bias != NULL) {",
                "                for (int _ck_i = 0; _ck_i < _ck_n; ++_ck_i) _ck_y[_ck_i] += _ck_bias[_ck_i];",
                "            }",
                "        }",
            ])
            if profile:
                lines.append(f'        CK_PROFILE_END("prefill", "{row_gemv_func}", "{op_type}_row_gemv", {layer});')
            fp32_func = linear_emission["fp32_activation_function"]
            lines.append(f"    }} else if ((debug_mlp_gate_up_fp32 || debug_mlp_gate_up_fp32_layer == {layer}) && ck_debug_mlp_gate_up_fp32_input != NULL) {{")
            if profile:
                lines.append("        CK_PROFILE_BEGIN();")
            lines.extend([
                f"        {fp32_func}(",
                "            ck_debug_mlp_gate_up_fp32_input,",
                f"            {b_expr},",
                f"            {bias_expr},",
                f"            {c_expr},",
                f"            {m_expr},",
                f"            {n_expr},",
                f"            {k_expr}",
                "        );",
            ])
            if profile:
                lines.append(f'        CK_PROFILE_END("prefill", "{fp32_func}", "{op_type}", {layer});')
            lines.append("    } else {")
            if profile:
                lines.append("        CK_PROFILE_BEGIN();")
            lines.extend([
                f"        {func}(",
                f"            {a_expr},",
                f"            {b_expr},",
                f"            {bias_expr},",
                f"            {c_expr},",
                f"            {m_expr},",
                f"            {n_expr},",
                f"            {k_expr}",
                "        );",
            ])
            if profile:
                lines.append(f'        CK_PROFILE_END("prefill", "{func}", "{op_type}", {layer});')
            lines.append("    }")
            raw_expr = c_expr.replace("(float*)", "").replace("(void*)", "").strip()
            lines.append(
                f'    ck_debug_export_hidden(model, {layer}, "mlp_gate_up", '
                f'(const float*){raw_expr}, ({m_expr}) * ({n_expr}));'
            )
            lines.append(
                f'    if (num_tokens > 1) ck_debug_export_hidden(model, {layer}, "mlp_gate_last", '
                f'(const float*)(((const float*){raw_expr}) + (((size_t)num_tokens - 1u) * (size_t)({n_expr}))), '
                f'(({n_expr}) / 2));'
            )
            lines.append(
                f'    if (num_tokens > 1) ck_debug_export_hidden(model, {layer}, "mlp_up_last", '
                f'(const float*)(((const float*){raw_expr}) + (((size_t)num_tokens - 1u) * (size_t)({n_expr})) + (size_t)((({n_expr}) / 2))), '
                f'(({n_expr}) / 2));'
            )
            return "\n".join(lines)

    if op_type == "mamba_in_proj" and func == "gemm_nt_q5_0_q8_0":
        a_expr = arg_expr_by_name.get("a")
        b_expr = arg_expr_by_name.get("b")
        bias_expr = arg_expr_by_name.get("bias", "NULL")
        c_expr = arg_expr_by_name.get("c")
        m_expr = arg_expr_by_name.get("m", "num_tokens")
        n_expr = arg_expr_by_name.get("n")
        k_expr = arg_expr_by_name.get("k")
        if a_expr and b_expr and c_expr and n_expr and k_expr:
            lines.append("    if (debug_prefill_mamba_in_proj_row_gemv) {")
            if profile:
                lines.append("        CK_PROFILE_BEGIN();")
            lines.extend([
                f"        const int _ck_m = (int)({m_expr});",
                f"        const int _ck_n = (int)({n_expr});",
                f"        const int _ck_k = (int)({k_expr});",
                "        const size_t _ck_a_row_bytes = (size_t)(_ck_k / QK8_0) * sizeof(block_q8_0);",
                f"        const uint8_t *_ck_a_base = (const uint8_t*)({a_expr});",
                f"        const uint8_t *_ck_w_base = (const uint8_t*)({b_expr});",
                f"        const float *_ck_bias = (const float*)({bias_expr});",
                f"        float *_ck_c_base = (float*)({c_expr});",
                "        for (int _ck_t = 0; _ck_t < _ck_m; ++_ck_t) {",
                "            const void *_ck_xq8 = (const void*)(_ck_a_base + (size_t)_ck_t * _ck_a_row_bytes);",
                "            float *_ck_y = _ck_c_base + (size_t)_ck_t * (size_t)_ck_n;",
                "            gemv_q5_0_q8_0(_ck_y, (const void*)_ck_w_base, _ck_xq8, _ck_n, _ck_k);",
                "            if (_ck_bias != NULL) {",
                "                for (int _ck_i = 0; _ck_i < _ck_n; ++_ck_i) _ck_y[_ck_i] += _ck_bias[_ck_i];",
                "            }",
                "        }",
            ])
            if profile:
                lines.append(f'        CK_PROFILE_END("prefill", "gemv_q5_0_q8_0", "{op_type}_row_gemv", {layer});')
            lines.append("    } else {")
            if profile:
                lines.append("        CK_PROFILE_BEGIN();")
            lines.extend([
                f"        {func}(",
                f"            {a_expr},",
                f"            {b_expr},",
                f"            {bias_expr},",
                f"            {c_expr},",
                f"            {m_expr},",
                f"            {n_expr},",
                f"            {k_expr}",
                "        );",
            ])
            if profile:
                lines.append(f'        CK_PROFILE_END("prefill", "{func}", "{op_type}", {layer});')
            lines.append("    }")
            raw_expr = c_expr.replace("(float*)", "").replace("(void*)", "").strip()
            lines.append(
                f'    if (num_tokens > 1) ck_debug_export_hidden(model, {layer}, "mamba_in_proj_last", '
                f'(const float*)(((const float*){raw_expr}) + (((size_t)num_tokens - 1u) * (size_t)({n_expr}))), '
                f'(int)({n_expr}));'
            )
            return "\n".join(lines)

    # Format the function call / quantization loop
    if batch_quant_kind and len(args) >= 3:
        x_expr = args[0]
        y_expr = args[1]
        k_expr = args[2]
        row_bytes_expr = activation_quantized_row_bytes_expr(batch_quant_kind, "_k")
        prefill_batch = batch_quant_kind.get("prefill_batch")
        if op_type == "quantize_input_2":
            lines.append(f"    ck_debug_mlp_gate_up_fp32_input = (const float*)({x_expr});")
        if op_type == "quantize_out_proj_input":
            lines.append(f"    ck_debug_outproj_fp32_input = (const float*)({x_expr});")
            lines.append(
                f'    if (num_tokens > 1) ck_debug_export_hidden(model, {layer}, "attn_out_last", '
                f'(const float*)(((const float*)({x_expr})) + (((size_t)num_tokens - 1u) * (size_t)({k_expr}))), '
                f'(int)({k_expr}));'
            )
            lines.append("    if (!debug_outproj_fp32) {")
            lines.append(f"        const float *_x_base = (const float*)({x_expr});")
            lines.append(f"        uint8_t *_y_base = (uint8_t*)({y_expr});")
            lines.append(f"        const int _k = (int)({k_expr});")
            lines.append(f"        const size_t _row_bytes = {row_bytes_expr};")
            if prefill_batch:
                lines.append(
                    f"        {prefill_batch['function']}("
                    "_x_base, (void*)_y_base, num_tokens, _k);"
                )
            else:
                lines.append("        for (int _t = 0; _t < num_tokens; ++_t) {")
                lines.append(
                    f"            {func}("
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
            if prefill_batch:
                lines.append(
                    f"        {prefill_batch['function']}("
                    "_x_base, (void*)_y_base, num_tokens, _k);"
                )
            else:
                lines.append("        for (int _t = 0; _t < num_tokens; ++_t) {")
                lines.append(
                    f"            {func}("
                    "_x_base + (size_t)_t * (size_t)_k, "
                    "(void*)(_y_base + (size_t)_t * _row_bytes), "
                    "_k);"
                )
                lines.append("        }")
            lines.append("    }")
    elif segmented_attention_function:
        segmented_args = []
        for arg_name in segmented_attention_provider["base_args"]:
            arg = arg_expr_by_name.get(str(arg_name).lower())
            if arg is None:
                raise RuntimeError(
                    f"segmented-query provider requires missing base argument {arg_name!r}"
                )
            segmented_args.append(arg)
        lines.append("    if (ck_multimodal_prefill_has_segment_plan(num_tokens)) {")
        lines.append(f"        {segmented_attention_function}(")
        for arg in segmented_args:
            lines.append(f"            {arg},")
        lines.append("            g_multimodal_prefill_segment_lengths,")
        lines.append("            g_multimodal_prefill_num_segments")
        lines.append("        );")
        lines.append("    } else {")
        lines.append(f"        {emitted_func}(")
        for i, arg in enumerate(args):
            comma = "," if i < len(args) - 1 else ""
            lines.append(f"            {arg}{comma}")
        lines.append("        );")
        lines.append("    }")
    elif segmented_row_function and len(args) == 7:
        lines.append("    if (ck_multimodal_prefill_has_segment_plan(num_tokens)) {")
        lines.append(f"        {segmented_row_function}(")
        for arg in args:
            lines.append(f"            {arg},")
        lines.append("            g_multimodal_prefill_segment_lengths,")
        lines.append("            g_multimodal_prefill_num_segments")
        lines.append("        );")
        lines.append("    } else {")
        lines.append(f"        {emitted_func}(")
        for i, arg in enumerate(args):
            comma = "," if i < len(args) - 1 else ""
            lines.append(f"            {arg}{comma}")
        lines.append("        );")
        lines.append("    }")
    elif (
        op_type in {"attn", "attn_sliding"}
        and isinstance(physical_execution, dict)
        and str(physical_execution.get("mixed_visual_chunk_function", "") or "").strip()
    ):
        mixed_func = str(physical_execution["mixed_visual_chunk_function"]).strip()
        mixed_args = args[:10]
        lines.append("    if (bridge_noncausal_visual_chunk && bridge_visual_start >= 0 && bridge_visual_tokens > 0) {")
        lines.append(f"        {mixed_func}(")
        for i, arg in enumerate(mixed_args):
            lines.append(f"            {arg},")
        lines.append("            bridge_visual_start,")
        lines.append("            bridge_visual_tokens")
        lines.append("        );")
        lines.append("    } else {")
        if len(args) <= 3:
            lines.append(f"        {emitted_func}({', '.join(args)});")
        else:
            lines.append(f"        {emitted_func}(")
            for i, arg in enumerate(args):
                comma = "," if i < len(args) - 1 else ""
                lines.append(f"            {arg}{comma}")
            lines.append("        );")
        lines.append("    }")
    else:
        if len(args) <= 3:
            # Short call on one line
            lines.append(f"    {emitted_func}({', '.join(args)});")
        else:
            # Multi-line for readability
            lines.append(f"    {emitted_func}(")
            for i, arg in enumerate(args):
                comma = "," if i < len(args) - 1 else ""
                lines.append(f"        {arg}{comma}")
            lines.append("    );")
    if profile:
        lines.append(f'    CK_PROFILE_END("prefill", "{emitted_func}", "{op_type}", {layer});')

    def _hidden_arg(*names: str) -> Optional[str]:
        for nm in names:
            ex = arg_expr_by_name.get(nm.lower())
            if ex:
                return ex
        return None

    def _hidden_raw(expr: Optional[str]) -> Optional[str]:
        if not expr:
            return None
        return expr.replace("(float*)", "").replace("(void*)", "").strip()

    def _hidden_mul(*terms: Optional[str]) -> Optional[str]:
        used = [f"({term})" for term in terms if term]
        return " * ".join(used) if used else None

    def _emit_hidden_full(expr: Optional[str], label: str, count_expr: Optional[str]) -> None:
        raw = _hidden_raw(expr)
        if not raw or not count_expr:
            return
        lines.append(
            f'    ck_debug_export_hidden(model, {layer}, "{label}", '
            f'(const float*){raw}, {count_expr});'
        )

    def _emit_hidden_last(
        expr: Optional[str],
        label: str,
        width_expr: Optional[str],
        stride_expr: Optional[str] = None,
    ) -> None:
        raw = _hidden_raw(expr)
        if not raw or not width_expr:
            return
        row_stride = stride_expr or width_expr
        lines.append(
            f'    if (num_tokens > 1) ck_debug_export_hidden(model, {layer}, "{label}_last", '
            f'(const float*)(((const float*){raw}) + (((size_t)num_tokens - 1u) * (size_t)({row_stride}))), '
            f'{width_expr});'
        )

    def _emit_head_major_last(expr: Optional[str], label: str, heads_expr: Optional[str], dim_expr: Optional[str]) -> None:
        raw = _hidden_raw(expr)
        if not raw or not heads_expr or not dim_expr:
            return
        safe_label = label.replace("-", "_").replace(".", "_")
        lines.append("    if (num_tokens > 1) {")
        lines.append(f"        const int _ck_{safe_label}_heads = (int)({heads_expr});")
        lines.append(f"        const int _ck_{safe_label}_dim = (int)({dim_expr});")
        lines.append(f"        const float *_ck_{safe_label}_src = (const float*){raw};")
        lines.append(f"        float _ck_{safe_label}_last[(size_t)_ck_{safe_label}_heads * (size_t)_ck_{safe_label}_dim];")
        lines.append(f"        for (int _ck_h = 0; _ck_h < _ck_{safe_label}_heads; ++_ck_h) {{")
        lines.append(f"            const float *_ck_s = _ck_{safe_label}_src + ((size_t)_ck_h * (size_t)num_tokens + ((size_t)num_tokens - 1u)) * (size_t)_ck_{safe_label}_dim;")
        lines.append(f"            memcpy(_ck_{safe_label}_last + (size_t)_ck_h * (size_t)_ck_{safe_label}_dim, _ck_s, (size_t)_ck_{safe_label}_dim * sizeof(float));")
        lines.append("        }")
        lines.append(f'        ck_debug_export_hidden(model, {layer}, "{label}_last", _ck_{safe_label}_last, _ck_{safe_label}_heads * _ck_{safe_label}_dim);')
        lines.append("    }")

    if op_type == "residual_save":
        checkpoint = "layer_input" if op_instance_idx == 0 else "after_attn"
        _emit_hidden_full(
            _hidden_arg("src", "input", "x"),
            checkpoint,
            _hidden_mul("num_tokens", "EMBED_DIM"),
        )
    elif op_type == "attn_norm":
        _emit_hidden_full(
            _hidden_arg("output", "out", "x", "y"),
            "attn_norm",
            _hidden_mul("num_tokens", "EMBED_DIM"),
        )
    elif op_type == "recurrent_qkv_proj":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "linear_attn_qkv_mixed",
            _hidden_mul("num_tokens", _hidden_arg("N", "n")),
        )
    elif op_type == "recurrent_gate_proj":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "z",
            _hidden_mul("num_tokens", _hidden_arg("N", "n")),
        )
    elif op_type == "recurrent_alpha_proj":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "alpha",
            _hidden_mul("num_tokens", _hidden_arg("N", "n")),
        )
    elif op_type == "recurrent_beta_proj":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "beta",
            _hidden_mul("num_tokens", _hidden_arg("N", "n")),
        )
    elif op_type == "recurrent_dt_gate":
        _emit_hidden_full(
            _hidden_arg("gate", "output", "out"),
            "gate",
            _hidden_mul(
                _hidden_arg("rows", "num_tokens"),
                _hidden_arg("num_heads"),
                _hidden_arg("state_dim"),
            ),
        )
    elif op_type == "recurrent_ssm_conv":
        _emit_hidden_full(
            _hidden_arg("out", "output"),
            "conv_output_raw",
            _hidden_mul(_hidden_arg("num_tokens", "rows"), _hidden_arg("num_channels", "dim")),
        )
    elif op_type == "recurrent_silu":
        _emit_hidden_full(
            _hidden_arg("out", "output"),
            "conv_output_silu",
            _hidden_mul(_hidden_arg("rows", "num_tokens"), _hidden_arg("dim", "num_channels")),
        )
    elif op_type == "recurrent_split_conv_qkv":
        rows_expr = _hidden_arg("rows", "num_tokens")
        _emit_hidden_full(_hidden_arg("q"), "q_conv", _hidden_mul(rows_expr, _hidden_arg("q_dim")))
        _emit_hidden_full(_hidden_arg("k"), "k_conv", _hidden_mul(rows_expr, _hidden_arg("k_dim")))
        _emit_hidden_full(_hidden_arg("v"), "v_conv_predelta", _hidden_mul(rows_expr, _hidden_arg("v_dim")))
    elif op_type == "recurrent_qk_l2_norm":
        rows_expr = _hidden_arg("rows", "num_tokens")
        _emit_hidden_full(_hidden_arg("q"), "q_conv_predelta", _hidden_mul(rows_expr, _hidden_arg("q_dim")))
        _emit_hidden_full(_hidden_arg("k"), "k_conv_predelta", _hidden_mul(rows_expr, _hidden_arg("k_dim")))
    elif op_type == "recurrent_core":
        rows_expr = _hidden_arg("rows", "num_tokens", "tokens")
        heads_expr = _hidden_arg("num_heads", "groups")
        state_expr = _hidden_arg("state_dim", "head_dim")
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "attn_output",
            _hidden_mul(rows_expr, heads_expr, state_expr),
        )
        _emit_hidden_full(
            _hidden_arg("state_out"),
            "new_state",
            _hidden_mul(heads_expr, state_expr, state_expr),
        )
    elif op_type == "recurrent_norm_gate":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "final_output",
            _hidden_mul(
                _hidden_arg("rows", "num_tokens", "tokens"),
                _hidden_arg("num_heads", "groups"),
                _hidden_arg("head_dim", "state_dim"),
            ),
        )
    elif op_type == "recurrent_out_proj":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "linear_attn_out",
            _hidden_mul("num_tokens", _hidden_arg("N", "n") or "EMBED_DIM"),
        )
    elif op_type == "gemma4_per_layer_prepare":
        _emit_hidden_last(_hidden_arg("output", "out", "per_layer_input", "y"), "gemma4_per_layer_prepare", "NUM_LAYERS * 256")
    elif op_type in ("q_proj", "q_gate_proj"):
        q_width = _hidden_arg("n") or "NUM_HEADS * HEAD_DIM"
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "q_proj",
            _hidden_mul(_hidden_arg("m", "rows", "num_tokens") or "num_tokens", q_width),
        )
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "q_proj_post_bias",
            _hidden_mul(_hidden_arg("m", "rows", "num_tokens") or "num_tokens", q_width),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "c", "y"), "q_proj", q_width)
    elif op_type == "kv_a_proj":
        width = _hidden_arg("n", "out_dim") or "KV_LORA_RANK + ROTARY_DIM"
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "mla_kv_a",
            _hidden_mul(_hidden_arg("m", "rows", "num_tokens") or "num_tokens", width),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "c", "y"), "mla_kv_a", width)
    elif op_type == "kv_a_layernorm":
        width = _hidden_arg("d_model", "kv_lora_rank")
        _emit_hidden_full(
            _hidden_arg("output", "out", "y"),
            "mla_kv_norm",
            _hidden_mul(_hidden_arg("tokens", "rows", "num_tokens") or "num_tokens", width),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "y"), "mla_kv_norm", width)
    elif op_type == "kv_lora_decompress":
        rows = _hidden_arg("tokens", "rows", "num_tokens") or "num_tokens"
        heads = _hidden_arg("heads", "num_heads") or "NUM_HEADS"
        k_width = _hidden_arg("qk_nope_dim")
        v_width = _hidden_arg("v_dim", "v_head_dim")
        _emit_hidden_full(
            _hidden_arg("k_nope"), "mla_k_nope",
            _hidden_mul(rows, heads, k_width),
        )
        _emit_hidden_full(
            _hidden_arg("value", "v"), "mla_value",
            _hidden_mul(rows, heads, v_width),
        )
    elif op_type == "partial_rope_concat":
        rows = _hidden_arg("tokens", "rows", "num_tokens") or "num_tokens"
        heads = _hidden_arg("heads", "num_heads") or "NUM_HEADS"
        width = f"({_hidden_arg('qk_nope_dim')} + {_hidden_arg('qk_rope_dim')})"
        _emit_hidden_full(_hidden_arg("query", "q"), "mla_query", _hidden_mul(rows, heads, width))
        _emit_hidden_full(_hidden_arg("key", "k"), "mla_key", _hidden_mul(rows, heads, width))
    elif op_type == "mla_attention":
        _emit_hidden_full(
            _hidden_arg("output", "out"),
            "mla_context",
            _hidden_mul(
                _hidden_arg("num_tokens", "rows", "tokens") or "num_tokens",
                _hidden_arg("num_heads", "heads") or "NUM_HEADS",
                _hidden_arg("v_head_dim", "v_dim"),
            ),
        )
    elif op_type == "k_proj":
        k_width = _hidden_arg("n") or "NUM_KV_HEADS * HEAD_DIM"
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "k_proj",
            _hidden_mul(_hidden_arg("m", "rows", "num_tokens") or "num_tokens", k_width),
        )
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "k_proj_post_bias",
            _hidden_mul(_hidden_arg("m", "rows", "num_tokens") or "num_tokens", k_width),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "c", "y"), "k_proj", k_width)
    elif op_type == "v_proj":
        v_width = _hidden_arg("n") or "NUM_KV_HEADS * HEAD_DIM"
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "v_proj",
            _hidden_mul(_hidden_arg("m", "rows", "num_tokens") or "num_tokens", v_width),
        )
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "v_proj_post_bias",
            _hidden_mul(_hidden_arg("m", "rows", "num_tokens") or "num_tokens", v_width),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "c", "y"), "v_proj", v_width)
    elif op_type == "attention_gate_projection":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "attn_gate",
            _hidden_mul(
                _hidden_arg("rows", "num_tokens") or "num_tokens",
                _hidden_arg("N", "n", "gate_dim"),
            ),
        )
    elif op_type == "split_q_gate":
        _emit_hidden_full(
            _hidden_arg("gate"),
            "attn_gate",
            _hidden_mul(_hidden_arg("rows", "num_tokens"), _hidden_arg("gate_dim")),
        )
    elif op_type == "qk_norm":
        _emit_hidden_full(
            _hidden_arg("q"), "qk_norm_q",
            _hidden_mul(_hidden_arg("num_heads") or "NUM_HEADS", _hidden_arg("rows", "num_tokens") or "num_tokens", _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM"),
        )
        _emit_hidden_full(
            _hidden_arg("k"), "qk_norm_k",
            _hidden_mul(_hidden_arg("num_kv_heads") or "NUM_KV_HEADS", _hidden_arg("rows", "num_tokens") or "num_tokens", _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM"),
        )
        _emit_head_major_last(_hidden_arg("q"), "qk_norm_q", _hidden_arg("num_heads") or "NUM_HEADS", _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM")
        _emit_head_major_last(_hidden_arg("k"), "qk_norm_k", _hidden_arg("num_kv_heads") or "NUM_KV_HEADS", _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM")
    elif op_type == "rope_qk":
        _emit_hidden_full(
            _hidden_arg("q"), "rope_q",
            _hidden_mul(_hidden_arg("num_heads") or "NUM_HEADS", _hidden_arg("num_tokens", "rows") or "num_tokens", _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM"),
        )
        _emit_hidden_full(
            _hidden_arg("k"), "rope_k",
            _hidden_mul(_hidden_arg("num_kv_heads") or "NUM_KV_HEADS", _hidden_arg("num_tokens", "rows") or "num_tokens", _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM"),
        )
        _emit_head_major_last(_hidden_arg("q"), "rope_q", _hidden_arg("num_heads") or "NUM_HEADS", _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM")
        _emit_head_major_last(_hidden_arg("k"), "rope_k", _hidden_arg("num_kv_heads") or "NUM_KV_HEADS", _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM")
    elif op_type == "attn":
        _emit_hidden_full(
            _hidden_arg("out_token", "output", "out", "c", "y"),
            "attn_pregate",
            _hidden_mul(
                _hidden_arg("rows", "num_tokens") or "num_tokens",
                _hidden_arg("num_heads") or "NUM_HEADS",
                _hidden_arg("aligned_head_dim", "head_dim") or "HEAD_DIM",
            ),
        )
    elif op_type in ("attn_gate_sigmoid_mul", "attn_gate_softplus_mul"):
        _emit_hidden_full(
            _hidden_arg("output", "out", "y"),
            "attn_out",
            _hidden_mul(
                _hidden_arg("num_heads") or "NUM_HEADS",
                _hidden_arg("rows", "num_tokens") or "num_tokens",
                _hidden_arg("state_dim", "aligned_head_dim", "head_dim") or "HEAD_DIM",
            ),
        )
    elif op_type == "out_proj":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "out_proj",
            _hidden_mul("num_tokens", _hidden_arg("N", "n", "out_dim") or "EMBED_DIM"),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "c", "y"), "out_proj", "EMBED_DIM")
    elif op_type in ("rmsnorm", "layernorm"):
        if str(section or "") == "footer" or int(layer) < 0:
            _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), "final_hidden", "EMBED_DIM")
        elif op_instance_idx == 0:
            _emit_hidden_full(
                _hidden_arg("output", "out", "x", "y"),
                "block_rmsnorm",
                _hidden_mul(_hidden_arg("rows", "num_tokens") or "num_tokens", "EMBED_DIM"),
            )
            _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), "block_rmsnorm", "EMBED_DIM")
        elif op_instance_idx == 1:
            _emit_hidden_full(
                _hidden_arg("output", "out", "x", "y"),
                "ffn_norm",
                _hidden_mul(_hidden_arg("rows", "num_tokens") or "num_tokens", "EMBED_DIM"),
            )
            _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), "ffn_norm", "EMBED_DIM")
        else:
            _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), "rmsnorm", "EMBED_DIM")
    elif op_type == "block_rmsnorm":
        label = "block_rmsnorm" if op_instance_idx == 0 else "ffn_norm"
        if op_instance_idx == 1:
            _emit_hidden_full(
                _hidden_arg("input", "x"),
                "ffn_input",
                _hidden_mul(_hidden_arg("rows", "num_tokens") or "num_tokens", "EMBED_DIM"),
            )
            _emit_hidden_last(_hidden_arg("input", "x"), "ffn_input", "EMBED_DIM")
        _emit_hidden_full(
            _hidden_arg("output", "out", "x", "y"),
            label,
            _hidden_mul(_hidden_arg("rows", "num_tokens") or "num_tokens", "EMBED_DIM"),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), label, "EMBED_DIM")
    elif op_type == "post_attention_norm":
        _emit_hidden_full(
            _hidden_arg("output", "out", "x", "y"),
            "post_attn_norm",
            _hidden_mul("num_tokens", _hidden_arg("d_model", "aligned_embed_dim", "embed_dim", "dim") or "EMBED_DIM"),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), "post_attn_norm", "EMBED_DIM")
    elif op_type == "ffn_norm":
        _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), "ffn_norm", "EMBED_DIM")
    elif op_type == "post_ffn_norm":
        _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), "post_ffn_norm", "EMBED_DIM")
    elif op_type == "moe_router":
        width = _hidden_arg("N", "n", "n_experts") or "N_ROUTED_EXPERTS"
        output = _hidden_arg("C", "output", "out")
        _emit_hidden_full(output, "moe_router_logits", _hidden_mul("num_tokens", width))
        _emit_hidden_last(output, "moe_router_logits", width)
    elif op_type in ("group_limited_topk_router", "full_softmax_topk_router"):
        width = _hidden_arg("top_k") or "NUM_EXPERTS_PER_TOK"
        indices = _hidden_raw(_hidden_arg("indices"))
        weights = _hidden_arg("weights")
        if indices:
            lines.append(
                f'    ck_debug_export_hidden_i32(model, {layer}, "moe_selected_experts", '
                f'(const int32_t*){indices}, {_hidden_mul("num_tokens", width)});'
            )
        _emit_hidden_full(weights, "moe_routing_weights", _hidden_mul("num_tokens", width))
        _emit_hidden_last(weights, "moe_routing_weights", width)
    elif op_type == "moe_swiglu_expert_mlp":
        output = _hidden_arg("output", "out")
        _emit_hidden_full(output, "moe_routed_output", _hidden_mul("num_tokens", "EMBED_DIM"))
        _emit_hidden_last(output, "moe_routed_output", "EMBED_DIM")
    elif op_type in ("shared_swiglu_expert_mlp", "gated_shared_swiglu_expert_mlp"):
        output = _hidden_arg("output", "out")
        _emit_hidden_full(output, "moe_combined_output", _hidden_mul("num_tokens", "EMBED_DIM"))
        _emit_hidden_last(output, "moe_combined_output", "EMBED_DIM")
    elif op_type == "farskip_routed_shared_combine":
        main_output = _hidden_arg("main_output")
        routed_free_output = _hidden_arg("routed_free_output")
        _emit_hidden_full(main_output, "layer_out", _hidden_mul("num_tokens", "EMBED_DIM"))
        _emit_hidden_last(main_output, "layer_out", "EMBED_DIM")
        _emit_hidden_full(routed_free_output, "routed_free_out", _hidden_mul("num_tokens", "EMBED_DIM"))
        _emit_hidden_last(routed_free_output, "routed_free_out", "EMBED_DIM")
    elif op_type == "residual_add":
        if op_instance_idx == 0:
            _emit_hidden_full(_hidden_arg("output", "out", "c", "y"), "after_attn", _hidden_mul("num_tokens", "EMBED_DIM"))
            _emit_hidden_last(_hidden_arg("b"), "after_attn_residual", "EMBED_DIM")
            _emit_hidden_last(_hidden_arg("output", "out", "c", "y"), "after_attn", "EMBED_DIM")
        elif op_instance_idx == 1:
            _emit_hidden_full(_hidden_arg("output", "out", "c", "y"), "layer_out", _hidden_mul("num_tokens", "EMBED_DIM"))
            _emit_hidden_last(_hidden_arg("b"), "ffn_residual", "EMBED_DIM")
            _emit_hidden_last(_hidden_arg("output", "out", "c", "y"), "layer_out", "EMBED_DIM")
    elif op_type == "mlp_gate_up":
        out_expr = _hidden_arg("output", "out", "c", "y")
        total_expr = _hidden_arg("n") or "(2 * INTERMEDIATE_DIM)"
        half_expr = f"(({total_expr}) / 2)"
        _emit_hidden_last(out_expr, "mlp_gate", half_expr, total_expr)
        raw = _hidden_raw(out_expr)
        if raw:
            lines.append(
                f'    if (num_tokens > 1) ck_debug_export_hidden(model, {layer}, "mlp_up_last", '
                f'(const float*)(((const float*){raw}) + (((size_t)num_tokens - 1u) * (size_t)({total_expr})) + (size_t)({half_expr})), '
                f'{half_expr});'
            )
    elif op_type == "geglu":
        _emit_hidden_last(_hidden_arg("output", "out", "x", "y", "data"), "mlp_geglu", _hidden_arg("dim") or "INTERMEDIATE_DIM")
    elif op_type in ("silu_mul", "swiglu"):
        width_expr = _hidden_arg("dim", "n", "intermediate_dim") or "INTERMEDIATE_DIM"
        _emit_hidden_full(
            _hidden_arg("output", "out", "x", "y", "data"),
            "mlp_swiglu",
            _hidden_mul("num_tokens", width_expr),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "x", "y", "data"), "mlp_swiglu", width_expr)
    elif op_type == "mlp_up":
        _emit_hidden_last(
            _hidden_arg("output", "out", "c", "y"),
            "mlp_up",
            _hidden_arg("n", "out_dim") or "INTERMEDIATE_SIZE",
        )
    elif op_type == "relu2":
        _emit_hidden_last(
            _hidden_arg("output", "out", "x", "y", "data"),
            "relu2",
            _hidden_arg("dim", "intermediate_dim") or "INTERMEDIATE_SIZE",
        )
    elif op_type == "mlp_down":
        _emit_hidden_full(
            _hidden_arg("output", "out", "c", "y"),
            "mlp_down",
            _hidden_mul("num_tokens", _hidden_arg("N", "n", "out_dim") or "EMBED_DIM"),
        )
        _emit_hidden_last(_hidden_arg("output", "out", "c", "y"), "mlp_down", "EMBED_DIM")
    elif op_type == "mamba_in_proj":
        _emit_hidden_last(_hidden_arg("output", "out", "c", "y", "C"), "mamba_in_proj", _hidden_arg("N") or _hidden_arg("m", "M", "rows", "out_dim") or "0")
    elif op_type == "mamba_in_proj_split":
        _emit_hidden_last(_hidden_arg("gate", "z"), "mamba_gate", _hidden_arg("intermediate_dim", "inner_dim") or "INTERMEDIATE_SIZE")
        _emit_hidden_last(_hidden_arg("hidden_bc", "conv_qkv", "x"), "mamba_hidden_bc", _hidden_arg("conv_dim") or "0")
        _emit_hidden_last(_hidden_arg("dt"), "mamba_dt_raw", _hidden_arg("num_heads") or "0")
    elif op_type == "mamba_conv1d_silu":
        _emit_hidden_last(_hidden_arg("conv_out", "out", "y"), "mamba_conv", _hidden_arg("conv_dim") or "0")
    elif op_type == "mamba_dt_softplus":
        _emit_hidden_last(_hidden_arg("dt_out", "out", "y"), "mamba_dt", _hidden_arg("num_heads") or "0")
    elif op_type == "mamba_selective_scan":
        _emit_hidden_last(_hidden_arg("y", "out"), "mamba_scan_y", f"(({_hidden_arg('num_heads') or '0'}) * ({_hidden_arg('head_dim') or '0'}))")
    elif op_type == "mamba_rmsnorm_gate":
        _emit_hidden_last(_hidden_arg("out", "y"), "mamba_normed", _hidden_arg("inner_dim", "intermediate_dim") or "INTERMEDIATE_SIZE")
    elif op_type == "mamba_out_proj":
        _emit_hidden_last(_hidden_arg("output", "out", "c", "y", "C"), "mamba_out_proj", "EMBED_DIM")
    elif op_type == "gemma4_per_layer_embed":
        _emit_hidden_last(_hidden_arg("hidden", "output", "out", "x", "y"), "gemma4_per_layer_embed", "EMBED_DIM")
    elif op_type == "final_rmsnorm":
        _emit_hidden_last(_hidden_arg("output", "out", "x", "y"), "final_hidden", "EMBED_DIM")

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

        def _emit_head_major_dump(
            expr: Optional[str],
            name: str,
            heads_expr: Optional[str],
            tokens_expr: Optional[str],
            head_dim_expr: Optional[str],
        ) -> None:
            if not expr or not heads_expr or not tokens_expr or not head_dim_expr:
                return
            raw_expr = expr.replace("(float*)", "").replace("(void*)", "").strip()
            lines.append("    #ifdef CK_PARITY_DUMP")
            lines.append(
                f'    ck_dump_tensor_head_major_token_major((float*){raw_expr}, {layer}, "{name}", '
                f"{heads_expr}, {tokens_expr}, {head_dim_expr});"
            )
            lines.append("    #endif")

        tokens = "num_tokens"
        embed_dim_expr = _get_arg("aligned_embed_dim", "d_model", "embed_dim") or str(embed_dim)
        m_dim = _get_arg("m")
        n_dim = _get_arg("n")
        num_heads = _get_arg("num_heads") or "NUM_HEADS"
        num_kv_heads = _get_arg("num_kv_heads") or "NUM_KV_HEADS"
        head_dim = _get_arg("aligned_head_dim", "head_dim") or "HEAD_DIM"

        if op_type == "residual_save" and op_instance_idx == 0:
            _emit_dump(
                _get_arg("src", "input", "x"),
                "layer_input",
                _mul_expr(tokens, embed_dim_expr),
            )
        elif op_type == "qk_norm":
            q_expr = _get_arg("q")
            k_expr = _get_arg("k")
            q_size = _mul_expr(tokens, num_heads, head_dim)
            k_size = _mul_expr(tokens, num_kv_heads, head_dim)
            _emit_dump(q_expr, "qcur_normed", q_size)
            _emit_dump(k_expr, "kcur_normed", k_size)
        elif op_type == "rope_qk":
            q_expr = _get_arg("q")
            k_expr = _get_arg("k")
            _emit_head_major_dump(q_expr, "Qcur_rope", num_heads, tokens, head_dim)
            _emit_head_major_dump(k_expr, "Kcur_rope", num_kv_heads, tokens, head_dim)
        elif op_type == "final_rmsnorm":
            _emit_dump(
                _get_arg("output", "out", "x", "y"),
                "final_norm",
                _mul_expr(tokens, embed_dim_expr),
            )
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
    context_capacity = int(config.get("context_length", config.get("context_len", 0)) or 0)
    prefill_chunk_length = int(config.get("prefill_chunk_length", context_capacity) or context_capacity)
    if context_capacity <= 0 or prefill_chunk_length <= 0:
        raise RuntimeError(
            "prefill codegen requires positive context_length and prefill_chunk_length"
        )
    prefill_chunk_length = min(prefill_chunk_length, context_capacity)
    scale_embeddings_sqrt_dim = bool(config.get("scale_embeddings_sqrt_dim", False))
    outproj_policy = str(config.get("out_proj_input_policy") or "").strip().lower()
    debug_outproj_default = 1 if outproj_policy in {"fp32", "fp32_input", "force_fp32"} else 0
    segment_preserving_ops = _segment_preserving_projection_operations(config)
    q4_gateup_swiglu_x16_default = int(
        bool(config.get("prefill_gateup_swiglu_fusion_default", True))
        and "mlp_gate_up" not in segment_preserving_ops
    )
    embed_scale_emitted = False
    prologue = """
/* ============================================================================
 * PREFILL - Batched processing from IR Lower (prefill mode)
 * ============================================================================ */
static void ck_prefill_range(CKModel *model, const int32_t *tokens, int num_tokens, int prefill_start_pos) {
    if (!model || !tokens || num_tokens <= 0) return;
    if (prefill_start_pos < 0 || prefill_start_pos >= MAX_SEQ_LEN) return;

    /* Clamp this chunk to the persistent context capacity. */
    if (num_tokens > MAX_SEQ_LEN - prefill_start_pos) {
        num_tokens = MAX_SEQ_LEN - prefill_start_pos;
    }

    const char *stop_env = getenv("CK_STOP_OP");
    int stop_seq = stop_env ? atoi(stop_env) : -1;
    const char *bridge_noncausal_env = getenv("CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK");
    int bridge_noncausal_visual_chunk = bridge_noncausal_env ? (atoi(bridge_noncausal_env) != 0) : 0;
    const char *bridge_visual_start_env = getenv("CK_BRIDGE_VISUAL_START");
    int bridge_visual_start = bridge_visual_start_env ? atoi(bridge_visual_start_env) : -1;
    const char *bridge_visual_tokens_env = getenv("CK_BRIDGE_VISUAL_TOKENS");
    int bridge_visual_tokens = bridge_visual_tokens_env ? atoi(bridge_visual_tokens_env) : 0;
    const char *debug_outproj_env = getenv("CK_V7_DEBUG_OUTPROJ_FP32");
    int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : 0;
    const float *ck_debug_outproj_fp32_input = NULL;
    const char *debug_mlp_down_env = getenv("CK_V7_DEBUG_MLP_DOWN_FP32");
    int debug_mlp_down_fp32 = debug_mlp_down_env ? (atoi(debug_mlp_down_env) != 0) : 0;
    const char *debug_mlp_down_layer_env = getenv("CK_V7_DEBUG_MLP_DOWN_FP32_LAYER");
    int debug_mlp_down_fp32_layer = debug_mlp_down_layer_env ? atoi(debug_mlp_down_layer_env) : -1;
    const char *debug_mlp_gate_up_env = getenv("CK_V8_DEBUG_MLP_GATE_UP_FP32");
    int debug_mlp_gate_up_fp32 = debug_mlp_gate_up_env ? (atoi(debug_mlp_gate_up_env) != 0) : 0;
    const char *debug_mlp_gate_up_layer_env = getenv("CK_V8_DEBUG_MLP_GATE_UP_FP32_LAYER");
    int debug_mlp_gate_up_fp32_layer = debug_mlp_gate_up_layer_env ? atoi(debug_mlp_gate_up_layer_env) : -1;
    const float *ck_debug_mlp_gate_up_fp32_input = NULL;
    const char *debug_prefill_mlp_gate_up_row_gemv_env = getenv("CK_V8_DEBUG_PREFILL_MLP_GATE_UP_ROW_GEMV");
    int debug_prefill_mlp_gate_up_row_gemv = debug_prefill_mlp_gate_up_row_gemv_env ? (atoi(debug_prefill_mlp_gate_up_row_gemv_env) != 0) : 0;
    const char *debug_prefill_mlp_gate_up_row_gemv_layer_env = getenv("CK_V8_DEBUG_PREFILL_MLP_GATE_UP_ROW_GEMV_LAYER");
    int debug_prefill_mlp_gate_up_row_gemv_layer = debug_prefill_mlp_gate_up_row_gemv_layer_env ? atoi(debug_prefill_mlp_gate_up_row_gemv_layer_env) : -1;
    const char *debug_prefill_mamba_in_proj_row_gemv_env = getenv("CK_V8_DEBUG_PREFILL_MAMBA_IN_PROJ_ROW_GEMV");
    int debug_prefill_mamba_in_proj_row_gemv = debug_prefill_mamba_in_proj_row_gemv_env ? (atoi(debug_prefill_mamba_in_proj_row_gemv_env) != 0) : 0;
    const char *ck_enable_q4k_gateup_swiglu_x16_env = getenv("CK_ENABLE_Q4K_GATEUP_SWIGLU_X16");
    const char *ck_disable_q4k_gateup_swiglu_x16_env = getenv("CK_DISABLE_Q4K_GATEUP_SWIGLU_X16");
    int ck_enable_q4k_gateup_swiglu_x16 = CK_Q4_GATEUP_SWIGLU_X16_DEFAULT;
    if (ck_enable_q4k_gateup_swiglu_x16_env &&
        ck_enable_q4k_gateup_swiglu_x16_env[0] &&
        strcmp(ck_enable_q4k_gateup_swiglu_x16_env, "0") != 0) {
        ck_enable_q4k_gateup_swiglu_x16 = 1;
    }
    if (ck_disable_q4k_gateup_swiglu_x16_env &&
        ck_disable_q4k_gateup_swiglu_x16_env[0] &&
        strcmp(ck_disable_q4k_gateup_swiglu_x16_env, "0") != 0) {
        ck_enable_q4k_gateup_swiglu_x16 = 0;
    }
    int ck_enable_swiglu_q8k_fusion = ck_env_truthy_or_qwen3vl_ocr_profile("CK_ENABLE_SWIGLU_Q8K_FUSION");

    /* Copy input tokens to activation buffer (follow same pattern as decode) */
    memcpy((void*)(model->bump + A_TOKEN_IDS), tokens, (size_t)num_tokens * sizeof(int32_t));
"""
    prologue = prologue.replace(
        "int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : 0;",
        f"int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : {debug_outproj_default};",
    )
    prologue = prologue.replace("CK_Q4_GATEUP_SWIGLU_X16_DEFAULT", str(q4_gateup_swiglu_x16_default))
    lines.append(prologue)

    if profile:
        lines.append("    CK_PROFILE_VARS();")
        lines.append("")

    _annotate_kv_transpose_roles(ops)
    residual_add_count_total = 0
    residual_add_counts_by_layer: Dict[int, int] = {}
    residual_save_counts_by_layer: Dict[int, int] = {}
    rmsnorm_counts_by_layer: Dict[int, int] = {}

    swiglu_q8k_fusion_guard: Optional[str] = None
    swiglu_q8k_fusion_call: Optional[str] = None

    for seq_idx, op in enumerate(ops):
        op_type_for_instance = str(op.get("op", ""))
        quantization_emission = resolved_activation_quantization_emission(op)
        if swiglu_q8k_fusion_guard and op_type_for_instance != "quantize_mlp_down_input":
            swiglu_q8k_fusion_guard = None
            swiglu_q8k_fusion_call = None
        if op_type_for_instance == "residual_add" and "op_instance_idx" not in op and "instance" not in op:
            layer_for_instance = int(op.get("layer", -1))
            inst = residual_add_counts_by_layer.get(layer_for_instance, 0)
            op = dict(op)
            op["op_instance_idx"] = inst
            residual_add_counts_by_layer[layer_for_instance] = inst + 1
        elif op_type_for_instance == "residual_save" and "op_instance_idx" not in op and "instance" not in op:
            layer_for_instance = int(op.get("layer", -1))
            inst = residual_save_counts_by_layer.get(layer_for_instance, 0)
            op = dict(op)
            op["op_instance_idx"] = inst
            residual_save_counts_by_layer[layer_for_instance] = inst + 1
        elif op_type_for_instance in {"rmsnorm", "layernorm", "block_rmsnorm"} and "op_instance_idx" not in op and "instance" not in op:
            layer_for_instance = int(op.get("layer", -1))
            inst = rmsnorm_counts_by_layer.get(layer_for_instance, 0)
            op = dict(op)
            op["op_instance_idx"] = inst
            rmsnorm_counts_by_layer[layer_for_instance] = inst + 1
        op_code = emit_prefill_op(op, seq_idx, config, profile=profile, dump=dump)
        if (
            swiglu_q8k_fusion_guard
            and swiglu_q8k_fusion_call
            and op_type_for_instance == "quantize_mlp_down_input"
            and quantization_emission is not None
            and quantization_emission["format"] == "q8_k"
        ):
            op_code = (
                f"    if ({swiglu_q8k_fusion_guard}) {{\n"
                f"{swiglu_q8k_fusion_call}\n"
                "    } else {\n"
                + "\n".join("    " + line if line else line for line in op_code.splitlines())
                + "\n    }"
            )
            swiglu_q8k_fusion_guard = None
            swiglu_q8k_fusion_call = None
        elif op_type_for_instance in {"silu_mul", "swiglu"}:
            next_op = ops[seq_idx + 1] if seq_idx + 1 < len(ops) else None
            next_quantization = (
                resolved_activation_quantization_emission(next_op)
                if isinstance(next_op, dict)
                else None
            )
            if (
                isinstance(next_op, dict)
                and str(op.get("function", "")) == "swiglu_forward"
                and str(next_op.get("op", "")) == "quantize_mlp_down_input"
                and next_quantization is not None
                and next_quantization["format"] == "q8_k"
            ):
                swiglu_args = op.get("args", [])
                quant_args = next_op.get("args", [])
                swiglu_input = _find_arg_expr(swiglu_args, arg_name="input")
                quant_output = _find_arg_expr(quant_args, arg_name="y") or _find_arg_expr(quant_args, arg_name="output")
                dim_expr = _find_arg_expr(swiglu_args, arg_name="dim") or _find_arg_expr(quant_args, arg_name="k")
                layer_value = op.get("layer", -1)
                layer = int(layer_value) if layer_value is not None else -1
                if swiglu_input and quant_output and dim_expr:
                    guard = f"ck_swiglu_q8k_fused_{seq_idx}"
                    swiglu_q8k_fusion_guard = guard
                    fused_lines = []
                    if profile:
                        fused_lines.append("        CK_PROFILE_BEGIN();")
                    fused_lines.append(f"        swiglu_forward_q8_k((const float*)({swiglu_input}), (void*)({quant_output}), num_tokens, (int)({dim_expr}));")
                    if profile:
                        fused_lines.append(f'        CK_PROFILE_END("prefill", "swiglu_forward_q8_k", "silu_mul_quantize_mlp_down_input", {layer});')
                    swiglu_q8k_fusion_call = "\n".join(fused_lines)
                    op_code = (
                        f"    const int {guard} = ck_enable_swiglu_q8k_fusion && !(debug_mlp_down_fp32 || debug_mlp_down_fp32_layer == {layer});\n"
                        f"    if (!{guard}) {{\n"
                        + "\n".join("    " + line if line else line for line in op_code.splitlines())
                        + "\n    }"
                    )
        lines.append(op_code)
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
    #endif
    if (num_tokens > 1) ck_debug_export_hidden(model, -1, "embedding_scaled_last",
        (const float*)(((const float*)(model->bump + A_EMBEDDED_INPUT)) + (((size_t)num_tokens - 1u) * (size_t)EMBED_DIM)),
        EMBED_DIM);""")
            embed_scale_emitted = True
        lines.append("")

    lines.append("    model->pos = prefill_start_pos + num_tokens;")
    lines.append("    model->rope_pos = prefill_start_pos + num_tokens;")
    if bool(config.get("_template_uses_persistent_cross_kv_cache", False)):
        lines.append("    model->encoder_kv_ready = 1;")
    lines.append("}")
    lines.append("")
    lines.append("static void ck_prefill(CKModel *model, const int32_t *tokens, int num_tokens) {")
    lines.append("    if (!model || !tokens || num_tokens <= 0) return;")
    lines.append("    int consumed = 0;")
    lines.append("    int start_pos = model->pos;")
    lines.append("    if (start_pos < 0 || start_pos >= MAX_SEQ_LEN) return;")
    lines.append("    if (num_tokens > MAX_SEQ_LEN - start_pos) num_tokens = MAX_SEQ_LEN - start_pos;")
    lines.append("    while (consumed < num_tokens) {")
    lines.append(f"        int chunk = num_tokens - consumed > {prefill_chunk_length} ? {prefill_chunk_length} : num_tokens - consumed;")
    lines.append("        if (chunk == 1 && consumed > 0) {")
    lines.append("            ck_decode(model, tokens[consumed]);")
    lines.append("        } else {")
    lines.append("            ck_prefill_range(model, tokens + consumed, chunk, start_pos + consumed);")
    lines.append("        }")
    lines.append("        consumed += chunk;")
    lines.append("    }")
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
     * llama.cpp scales token embeddings by sqrt(n_embd), but raw multimodal
     * embedding chunks are not scaled.  In the mixed bridge path, the visual
     * segment is marked by CK_BRIDGE_VISUAL_START/TOKENS.
     */
    {
        const float emb_scale = sqrtf((float)EMBED_DIM);
        float *emb = (float*)(model->bump + A_EMBEDDED_INPUT);
        const int rows = """
        + num_tokens_expr
        + """;
        const int visual_begin = (bridge_noncausal_visual_chunk && bridge_visual_start >= 0 && bridge_visual_tokens > 0)
            ? bridge_visual_start
            : -1;
        const int visual_end = (visual_begin >= 0) ? (visual_begin + bridge_visual_tokens) : -1;
        for (int row = 0; row < rows; ++row) {
            if (visual_begin >= 0 && row >= visual_begin && row < visual_end) {
                continue;
            }
            float *dst = emb + (size_t)row * (size_t)EMBED_DIM;
            for (int i = 0; i < EMBED_DIM; ++i) {
                dst[i] *= emb_scale;
            }
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


def _has_multimodal_bridge_contract(config: Dict) -> bool:
    bridge = config.get("multimodal_bridge_contract")
    if not isinstance(bridge, dict):
        return False
    return bool(str(bridge.get("prefix_policy", "") or "").strip())


def _has_segmented_append_prefill_contract(config: Dict) -> bool:
    bridge = config.get("multimodal_bridge_contract")
    if not isinstance(bridge, dict):
        return False
    schedule = bridge.get("prefill_schedule")
    if not isinstance(schedule, dict):
        return False
    return (
        schedule.get("segments") == ["text_before", "visual", "text_after"]
        and schedule.get("cache_transition") == "append_preserve"
        and schedule.get("position_transition") == "segment_defined"
    )


def _has_unified_mixed_prefill_contract(config: Dict) -> bool:
    bridge = config.get("multimodal_bridge_contract")
    if not isinstance(bridge, dict):
        return False
    schedule = bridge.get("prefill_schedule")
    if not isinstance(schedule, dict):
        return False
    return (
        bridge.get("prefill_batching") == "unified_mixed"
        and schedule.get("segments") == ["text_before", "visual", "text_after"]
        and schedule.get("cache_transition") == "single_pass"
        and schedule.get("position_transition") == "explicit_full_sequence"
    )


def _segment_preserving_projection_operations(config: Dict) -> set[str]:
    """Return circuit-owned operations whose row groups restart per segment."""
    bridge = config.get("multimodal_bridge_contract")
    if not isinstance(bridge, dict):
        return set()
    schedule = bridge.get("prefill_schedule")
    if not isinstance(schedule, dict):
        return set()
    boundaries = schedule.get("projection_row_group_boundaries")
    if not isinstance(boundaries, dict):
        return set()
    if boundaries.get("policy") != "restart_each_segment":
        return set()
    operations = boundaries.get("operations")
    if not isinstance(operations, list) or not all(
        isinstance(operation, str) and operation for operation in operations
    ):
        raise RuntimeError(
            "segment-preserving projection contract requires a non-empty operation list"
        )
    return set(operations)


def _segmented_row_provider(op: Dict, config: Dict) -> str:
    if str(op.get("op", "")) not in _segment_preserving_projection_operations(config):
        return ""
    linear = resolved_quantized_linear_emission(op)
    if not linear:
        return ""
    provider = linear.get("segmented_row_provider")
    if not isinstance(provider, dict):
        raise RuntimeError(
            f"operation {op.get('op')!r} requires a map-owned segmented-row provider"
        )
    if provider.get("boundary_semantics") != "restart_row_group_at_each_segment":
        raise RuntimeError("segmented-row provider has incompatible boundary semantics")
    return str(provider["function"])


def _segmented_attention_provider(op: Dict) -> Dict:
    if str(op.get("op", "")) not in {"attn", "attn_sliding"}:
        return {}
    execution = op.get("resolved_execution")
    implementation = execution.get("implementation") if isinstance(execution, dict) else None
    provider = (
        implementation.get("segmented_query_provider")
        if isinstance(implementation, dict)
        else None
    )
    if provider is None:
        return {}
    if not isinstance(provider, dict):
        raise RuntimeError("segmented-query attention provider metadata must be an object")
    required = {
        "function",
        "base_args",
        "segment_lengths_dtype",
        "boundary_semantics",
        "fallback",
    }
    if set(provider) != required:
        raise RuntimeError("segmented-query provider must define the exact ABI contract")
    if provider["segment_lengths_dtype"] != "i32":
        raise RuntimeError("segmented-query provider requires i32 segment lengths")
    if provider["boundary_semantics"] != "restart_query_tile_policy_at_each_segment":
        raise RuntimeError("segmented-query provider has incompatible boundary semantics")
    base_args = provider["base_args"]
    if not isinstance(base_args, list) or not base_args or not all(
        isinstance(name, str) and name for name in base_args
    ):
        raise RuntimeError("segmented-query provider requires exact base argument names")
    return provider


def _emit_multimodal_prefill_bridge_helpers(config: Dict, text_mrope_function: str) -> str:
    embed_dim = int(config.get("embed_dim", 0) or 0)
    num_deepstack_layers = int(config.get("num_deepstack_layers", 0) or 0)
    if embed_dim <= 0:
        return ""
    bridge = config.get("multimodal_bridge_contract")
    schedule = bridge.get("prefill_schedule") if isinstance(bridge, dict) else None
    position_transform = (
        schedule.get("position_transform") if isinstance(schedule, dict) else None
    )
    providers = bridge.get("providers") if isinstance(bridge, dict) else None
    if not isinstance(position_transform, dict) and isinstance(providers, dict):
        position_transform = providers.get("position_transform")
    positions_mrope_function = str(
        (position_transform or {}).get("resolved_function", "") or ""
    ).strip()
    deepstack_injection = (
        schedule.get("deepstack_injection") if isinstance(schedule, dict) else None
    )
    deepstack_add_function = str(
        (deepstack_injection or {}).get("resolved_function", "") or ""
    ).strip()
    position_builder = providers.get("position_builder") if isinstance(providers, dict) else None
    position_builder_function = str(
        (position_builder or {}).get("resolved_function", "") or ""
    ).strip()
    if _has_unified_mixed_prefill_contract(config) and not positions_mrope_function:
        raise RuntimeError(
            "unified mixed-prefill requires a resolved positions-aware M-RoPE provider"
        )
    if _has_unified_mixed_prefill_contract(config) and not position_builder_function:
        raise RuntimeError(
            "unified mixed-prefill requires a resolved M-RoPE position-builder provider"
        )
    if not position_builder_function:
        position_builder_function = "ck_multimodal_mrope_positions_2d"
    if (
        _has_unified_mixed_prefill_contract(config)
        and num_deepstack_layers > 0
        and not deepstack_add_function
    ):
        raise RuntimeError(
            "unified mixed-prefill with deepstack requires a resolved injection provider"
        )
    if _has_unified_mixed_prefill_contract(config):
        position_call = (
            f"{positions_mrope_function}(q, k, g_multimodal_prefill_positions, "
            "num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, "
            "n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, "
            "freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);"
        )
        deepstack_call = (
            f"{deepstack_add_function}(dst_row, src, dst_row, 1, {embed_dim});"
        )
    else:
        # Segmented append retains the established FP32 bridge contract. Its
        # segments are positioned independently, so it does not consume the
        # unified schedule's BF16 full-sequence providers.
        position_call = (
            "mrope_qk_imrope_positions(q, k, g_multimodal_prefill_positions, "
            "num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, "
            "n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, "
            "freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);"
        )
        deepstack_call = (
            f"for (int i = 0; i < {embed_dim}; ++i) {{\n"
            "            dst_row[i] += src[i];\n"
            "        }"
        )

    return f"""
static int g_multimodal_prefill_bridge_active = 0;
static int g_multimodal_prefill_text_pos = 0;
static int g_multimodal_prefill_total_tokens = 0;
static int g_multimodal_prefill_prefix_start = 0;
static int g_multimodal_prefill_prefix_tokens = 0;
static int32_t *g_multimodal_prefill_positions = NULL;
static const float *g_multimodal_prefill_rows = NULL;
static int g_multimodal_prefill_row_dim = 0;
static int g_multimodal_prefill_segment_lengths[3] = {{0, 0, 0}};
static int g_multimodal_prefill_num_segments = 0;

static void ck_multimodal_prefill_bridge_clear(void) {{
    g_multimodal_prefill_bridge_active = 0;
    g_multimodal_prefill_text_pos = 0;
    g_multimodal_prefill_total_tokens = 0;
    g_multimodal_prefill_prefix_start = 0;
    g_multimodal_prefill_prefix_tokens = 0;
    free(g_multimodal_prefill_positions);
    g_multimodal_prefill_positions = NULL;
    g_multimodal_prefill_rows = NULL;
    g_multimodal_prefill_row_dim = 0;
    g_multimodal_prefill_segment_lengths[0] = 0;
    g_multimodal_prefill_segment_lengths[1] = 0;
    g_multimodal_prefill_segment_lengths[2] = 0;
    g_multimodal_prefill_num_segments = 0;
}}

static void ck_multimodal_prefill_bridge_free(void) {{
    ck_multimodal_prefill_bridge_clear();
}}

static int ck_multimodal_prefill_bridge_text_pos(void) {{
    return g_multimodal_prefill_text_pos;
}}

static int ck_multimodal_prefill_bridge_next_text_pos(void) {{
    if (!g_multimodal_prefill_bridge_active || g_multimodal_prefill_prefix_tokens <= 0) return g_multimodal_prefill_text_pos;
    const int prefix_end = g_multimodal_prefill_prefix_start + g_multimodal_prefill_prefix_tokens;
    const int suffix_tokens = g_multimodal_prefill_total_tokens > prefix_end ? (g_multimodal_prefill_total_tokens - prefix_end) : 0;
    return g_multimodal_prefill_text_pos + suffix_tokens;
}}

static int ck_multimodal_prefill_bridge_is_active(void) {{
    return g_multimodal_prefill_bridge_active;
}}

static int ck_multimodal_prefill_has_segment_plan(int total_tokens) {{
    if (!g_multimodal_prefill_bridge_active ||
        g_multimodal_prefill_num_segments <= 0 ||
        total_tokens != g_multimodal_prefill_total_tokens) return 0;
    int rows = 0;
    for (int i = 0; i < g_multimodal_prefill_num_segments; ++i) {{
        rows += g_multimodal_prefill_segment_lengths[i];
    }}
    return rows == total_tokens;
}}

static int ck_multimodal_prefill_bridge_prepare(const float *rows,
                                             int total_tokens,
                                             int prefix_start,
                                             int position_base,
                                             int prefix_tokens,
                                             int row_dim,
                                             int grid_x,
                                             int grid_y,
                                             int text_pos) {{
    ck_multimodal_prefill_bridge_clear();
    if (!rows || total_tokens <= 0 || prefix_tokens <= 0 || row_dim < {embed_dim}) return -1;
    if (grid_x <= 0 || grid_y <= 0) return -2;
    if (grid_x * grid_y != prefix_tokens) return -3;
    if (prefix_start < 0 || prefix_start > total_tokens) return -4;
    if (prefix_tokens > total_tokens - prefix_start) return -5;

    g_multimodal_prefill_positions = (int32_t*)malloc((size_t)4 * (size_t)total_tokens * sizeof(int32_t));
    if (!g_multimodal_prefill_positions) {{
        ck_multimodal_prefill_bridge_clear();
        return -6;
    }}

    const int resolved_text_pos = {position_builder_function}(
        g_multimodal_prefill_positions,
        total_tokens,
        prefix_start,
        position_base,
        prefix_tokens,
        grid_x,
        grid_y,
        text_pos
    );
    if (resolved_text_pos < 0) {{
        ck_multimodal_prefill_bridge_clear();
        return resolved_text_pos;
    }}

    g_multimodal_prefill_rows = rows;
    g_multimodal_prefill_row_dim = row_dim;
    g_multimodal_prefill_total_tokens = total_tokens;
    g_multimodal_prefill_prefix_start = prefix_start;
    g_multimodal_prefill_prefix_tokens = prefix_tokens;
    g_multimodal_prefill_text_pos = resolved_text_pos;
    g_multimodal_prefill_segment_lengths[0] = prefix_start;
    g_multimodal_prefill_segment_lengths[1] = prefix_tokens;
    g_multimodal_prefill_segment_lengths[2] =
        total_tokens - prefix_start - prefix_tokens;
    g_multimodal_prefill_num_segments = 3;
    g_multimodal_prefill_bridge_active = 1;
    return 0;
}}

static void ck_multimodal_prefill_mrope_qk(float *q, float *k, int num_heads, int num_kv_heads, int num_tokens, int head_dim, int aligned_head_dim, int pos_offset, int n_dims, int section_0, int section_1, int section_2, int section_3, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {{
    if (g_multimodal_prefill_bridge_active && g_multimodal_prefill_positions && g_multimodal_prefill_total_tokens == num_tokens) {{
        {position_call}
        return;
    }}
    {text_mrope_function}(q, k, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset, n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}}

static void ck_multimodal_prefill_deepstack_add(CKModel *model, int layer, int num_tokens) {{
    const char *disable_env = getenv("CK_QWEN3VL_DISABLE_PREFILL_DEEPSTACK");
    if (disable_env && atoi(disable_env) != 0) return;
    if (!model || !g_multimodal_prefill_bridge_active || !g_multimodal_prefill_rows) return;
    if (layer < 0 || layer >= {num_deepstack_layers}) return;
    if (g_multimodal_prefill_prefix_tokens <= 0) return;
    const size_t slice_offset = (size_t){embed_dim} + (size_t)layer * (size_t){embed_dim};
    const size_t need = slice_offset + (size_t){embed_dim};
    if ((size_t)g_multimodal_prefill_row_dim < need) return;
    float *dst = (float*)(model->bump + A_EMBEDDED_INPUT);
    for (int tok = 0; tok < g_multimodal_prefill_prefix_tokens; ++tok) {{
        const float *src = g_multimodal_prefill_rows + (size_t)tok * (size_t)g_multimodal_prefill_row_dim + slice_offset;
        float *dst_row = dst + (size_t)(g_multimodal_prefill_prefix_start + tok) * (size_t){embed_dim};
        {deepstack_call}
    }}
}}
"""


def _emit_prefill_quant_rows(op: Dict, seq_idx: int, *, debug_inputs: list[str] | None = None, profile: bool = False) -> str:
    func = str(op.get("function", "") or "")
    quantization_emission = resolved_activation_quantization_emission(op)
    if quantization_emission is None:
        raise RuntimeError("prefill quant rows require a resolved activation quantization capability")
    args_list = op.get("args", [])
    x_expr = _find_arg_expr(args_list, arg_name="x") or _find_arg_expr(args_list, arg_name="input")
    y_expr = _find_arg_expr(args_list, arg_name="y") or _find_arg_expr(args_list, arg_name="output")
    k_expr = _find_arg_expr(args_list, arg_name="k")
    if not x_expr or not y_expr or not k_expr:
        raise RuntimeError(f"prefill quant rows missing x/y/k args for op={op.get('op')}")
    row_bytes_expr = activation_quantized_row_bytes_expr(quantization_emission, "_k")
    lines = [
        f"    /* Op {seq_idx}: {func} ({op.get('op', 'unknown')}) layer={op.get('layer', -1)} section={op.get('section', 'body')} */",
    ]
    for name in debug_inputs or []:
        lines.append(f"    {name} = (const float*)({x_expr});")
    lines.append("    {")
    if profile:
        lines.append("        CK_PROFILE_BEGIN();")
    lines.extend([
        f"        const float *_x_base = (const float*)({x_expr});",
        f"        uint8_t *_y_base = (uint8_t*)({y_expr});",
        f"        const int _k = (int)({k_expr});",
        f"        const size_t _row_bytes = {row_bytes_expr};",
    ])
    prefill_batch = quantization_emission.get("prefill_batch")
    if prefill_batch:
        lines.append(
            f"        {prefill_batch['function']}("
            "_x_base, (void*)_y_base, num_tokens, _k);"
        )
    else:
        lines.extend([
            "        for (int _t = 0; _t < num_tokens; ++_t) {",
            f"            {func}(_x_base + (size_t)_t * (size_t)_k, (void*)(_y_base + (size_t)_t * _row_bytes), _k);",
            "        }",
        ])
    if profile:
        lines.append(f'        CK_PROFILE_END("prefill", "{func}", "{op.get("op", "unknown")}", {int(op.get("layer", -1) or -1)});')
    lines.append("    }")
    return "\n".join(lines)


def _emit_prefill_quant_debug_override(op: Dict, seq_idx: int, config: Dict, *, debug_flag_name: str, debug_input_name: str, profile: bool = False) -> str:
    func = str(op.get("function", "") or "")
    quantization_emission = resolved_activation_quantization_emission(op)
    if quantization_emission is None:
        raise RuntimeError("prefill quant override requires a resolved activation quantization capability")

    args_list = op.get("args", [])
    x_expr = _find_arg_expr(args_list, arg_name="x") or _find_arg_expr(args_list, arg_name="x_q8")
    y_expr = _find_arg_expr(args_list, arg_name="y")
    k_expr = _find_arg_expr(args_list, arg_name="k")
    if not x_expr or not y_expr or not k_expr:
        raise RuntimeError("prefill quant override missing x/y/k args")

    row_bytes_expr = activation_quantized_row_bytes_expr(quantization_emission, "_k")
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
        ]
    )
    prefill_batch = quantization_emission.get("prefill_batch")
    if prefill_batch:
        lines.append(
            f"        {prefill_batch['function']}("
            "_x_base, (void*)_y_base, num_tokens, _k);"
        )
    else:
        lines.extend(
            [
                "        for (int _t = 0; _t < num_tokens; ++_t) {",
                f"            {func}(_x_base + (size_t)_t * (size_t)_k, (void*)(_y_base + (size_t)_t * _row_bytes), _k);",
                "        }",
            ]
        )
    if profile:
        lines.append(f'        CK_PROFILE_END("prefill", "{func}", "{op.get("op", "unknown")}", {int(op.get("layer", -1) or -1)});')
    lines.append("    }")
    return "\n".join(lines)


def _emit_prefill_gemm_fp32_override(op: Dict, seq_idx: int, config: Dict, *, debug_flag_name: str, debug_input_name: str, profile: bool = False, dump: bool = False) -> str:
    func = str(op.get("function", "") or "")
    linear_emission = resolved_quantized_linear_emission(op)
    if not is_q4_q6_q8_linear(linear_emission):
        raise RuntimeError("prefill fp32 override requires resolved Q4/Q6 x Q8 execution metadata")

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

    fp32_func = linear_emission["fp32_activation_function"]
    segmented_row_function = (
        _segmented_row_provider(op, config)
        if _has_multimodal_bridge_contract(config)
        else ""
    )
    layer_value = op.get("layer", -1)
    layer = int(layer_value) if layer_value is not None else -1
    row_gemv_func = linear_emission["row_quantized_function"]
    weight_row_bytes_expr = (
        f"(((size_t)({k_expr}) / {linear_emission['weight_block_elements']}u) * "
        f"{linear_emission['weight_block_bytes']}u)"
    )

    row_gemv_allowed = 1 if str(op.get("op", "")) == "mlp_gate_up" else 0
    lines = [
        f"    /* Op {seq_idx}: {func} ({op.get('op', 'unknown')}) layer={layer} */",
        f"    if ({row_gemv_allowed} && (debug_prefill_mlp_gate_up_row_gemv || debug_prefill_mlp_gate_up_row_gemv_layer == {layer})) {{",
    ]
    if profile:
        lines.append("        CK_PROFILE_BEGIN();")
    lines.extend(
        [
            f"        const int _ck_m = (int)({m_expr});",
            f"        const int _ck_n = (int)({n_expr});",
            f"        const int _ck_k = (int)({k_expr});",
            "        const int _ck_half = _ck_n / 2;",
            f"        const size_t _ck_a_row_bytes = (size_t)(_ck_k / QK_K) * sizeof(block_q8_K);",
            f"        const size_t _ck_w_row_bytes = {weight_row_bytes_expr};",
            f"        const uint8_t *_ck_a_base = (const uint8_t*)({a_expr});",
            f"        const uint8_t *_ck_w_base = (const uint8_t*)({b_expr});",
            f"        const float *_ck_bias = (const float*)({bias_expr});",
            f"        float *_ck_c_base = (float*)({c_expr});",
            "        for (int _ck_t = 0; _ck_t < _ck_m; ++_ck_t) {",
            "            const void *_ck_xq8 = (const void*)(_ck_a_base + (size_t)_ck_t * _ck_a_row_bytes);",
            "            float *_ck_y = _ck_c_base + (size_t)_ck_t * (size_t)_ck_n;",
            f"            {row_gemv_func}(_ck_y, (const void*)_ck_w_base, _ck_xq8, _ck_half, _ck_k);",
            f"            {row_gemv_func}(_ck_y + _ck_half, (const void*)(_ck_w_base + (size_t)_ck_half * _ck_w_row_bytes), _ck_xq8, _ck_half, _ck_k);",
            "            if (_ck_bias != NULL) {",
            "                for (int _ck_i = 0; _ck_i < _ck_n; ++_ck_i) _ck_y[_ck_i] += _ck_bias[_ck_i];",
            "            }",
            "        }",
        ]
    )
    if profile:
        lines.append(f'        CK_PROFILE_END("prefill", "{row_gemv_func}", "{op.get("op", "unknown")}_row_gemv", {layer});')
    lines.append(f"    }} else if ({debug_flag_name} && {debug_input_name} != NULL) {{")
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
    if segmented_row_function:
        lines.extend(
            [
                "        if (ck_multimodal_prefill_has_segment_plan(num_tokens)) {",
                f"            {segmented_row_function}(",
                f"                {a_expr},",
                f"                {b_expr},",
                f"                {bias_expr},",
                f"                {c_expr},",
                f"                {m_expr},",
                f"                {n_expr},",
                f"                {k_expr},",
                "                g_multimodal_prefill_segment_lengths,",
                "                g_multimodal_prefill_num_segments",
                "            );",
                "        } else {",
                f"            {func}(",
                f"                {a_expr},",
                f"                {b_expr},",
                f"                {bias_expr},",
                f"                {c_expr},",
                f"                {m_expr},",
                f"                {n_expr},",
                f"                {k_expr}",
                "            );",
                "        }",
            ]
        )
    else:
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



def _emit_prefill_q4_gateup_swiglu_x16(
    gate_op: Dict,
    swiglu_op: Dict,
    seq_idx: int,
    fused_var: str,
    config: Dict,
    *,
    debug_flag_name: str,
    debug_input_name: str,
    profile: bool = False,
) -> str:
    args_list = gate_op.get("args", [])
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
    m_expr = _find_arg_expr(args_list, arg_name="M") or _find_arg_expr(args_list, arg_name="m") or "num_tokens"
    if isinstance(m_arg, dict) and str(m_arg.get("source", "")).lower() == "dim:_m":
        # Embedded-prefix prefill only replays the active prefix rows.
        m_expr = "num_tokens"
    n_expr = _find_arg_expr(args_list, arg_name="N") or _find_arg_expr(args_list, arg_name="n")
    k_expr = _find_arg_expr(args_list, arg_name="K") or _find_arg_expr(args_list, arg_name="k")
    swiglu_args = swiglu_op.get("args", [])
    out_expr = (
        _find_arg_expr(swiglu_args, arg_name="output")
        or _find_arg_expr(swiglu_args, arg_name="out")
        or _find_arg_expr(swiglu_args, arg_name="y")
        or _find_arg_expr(swiglu_args, arg_name="C")
        or _find_arg_expr(swiglu_args, arg_name="c")
        or _find_arg_expr(swiglu_args, arg_name="data")
    )
    if not a_expr or not b_expr or not out_expr or not n_expr or not k_expr:
        raise RuntimeError("prefill q4 gate/up swiglu fusion missing args")

    layer_value = gate_op.get("layer", -1)
    layer = int(layer_value) if layer_value is not None else -1
    d_expr = f"(({n_expr}) / 2)"
    lines = [
        f"    /* Op {seq_idx}: fused gemm_nt_q4_k_q8_k + swiglu (mlp_gate_up) layer={layer} */",
        f"    int {fused_var} = 0;",
        f"    if (ck_enable_q4k_gateup_swiglu_x16 && !{debug_flag_name}) {{",
    ]
    if profile:
        lines.append("        CK_PROFILE_BEGIN();")
    lines.extend([
        "        gemm_nt_q4_k_q8_k_gateup_swiglu_x16_parallel_dispatch(",
        f"            {a_expr},",
        f"            {b_expr},",
        f"            {bias_expr},",
        f"            {out_expr},",
        f"            {m_expr},",
        f"            {d_expr},",
        f"            {k_expr}",
        "        );",
        f"        {fused_var} = 1;",
    ])
    if profile:
        lines.append(f'        CK_PROFILE_END("prefill", "gemm_nt_q4_k_q8_k_gateup_swiglu_x16", "mlp_gate_up_swiglu", {layer});')
    lines.append("    } else {")
    old_code = _emit_prefill_gemm_fp32_override(
        gate_op,
        seq_idx,
        config,
        debug_flag_name=debug_flag_name,
        debug_input_name=debug_input_name,
        profile=profile,
        dump=False,
    )
    lines.extend("    " + line if line else line for line in old_code.splitlines())
    lines.append("    }")
    raw_out = out_expr.replace("(float*)", "").replace("(void*)", "").strip()
    lines.append(
        f'    if ({fused_var}) ck_debug_export_hidden(model, {layer}, "mlp_swiglu", '
        f'(const float*){raw_out}, ({m_expr}) * ({d_expr}));'
    )
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
    outproj_policy = str(config.get("out_proj_input_policy") or "").strip().lower()
    debug_outproj_default = 1 if outproj_policy in {"fp32", "fp32_input", "force_fp32"} else 0
    embed_scale_emitted = False
    has_multimodal_bridge = _has_multimodal_bridge_contract(config)
    segment_preserving_ops = _segment_preserving_projection_operations(config)
    q4_gateup_swiglu_x16_default = int(
        bool(config.get("prefill_gateup_swiglu_fusion_default", True))
        and "mlp_gate_up" not in segment_preserving_ops
    )
    text_mrope_ops = [
        op
        for op in ops
        if str(op.get("op", "")) == "rope_qk"
        and (
            ((op.get("resolved_contract") or {}).get("semantics") or {}).get(
                "operator_family"
            )
            == "text_mrope"
        )
    ]
    rope_functions = {
        str(op.get("function", "") or "").strip() for op in text_mrope_ops
    }
    rope_functions.discard("")
    if text_mrope_ops and len(rope_functions) != 1:
        raise ValueError(
            "multimodal prefill requires exactly one resolved rope_qk function; "
            f"got {sorted(rope_functions)}"
        )
    text_mrope_function = next(iter(rope_functions), "")
    has_decoder_multimodal_bridge = has_multimodal_bridge and bool(text_mrope_function)
    num_deepstack_layers = (
        int(config.get("num_deepstack_layers", 0) or 0)
        if has_decoder_multimodal_bridge
        else 0
    )
    helper_block = (
        _emit_multimodal_prefill_bridge_helpers(config, text_mrope_function)
        if has_decoder_multimodal_bridge
        else ""
    )
    if helper_block:
        lines.append(helper_block)

    prologue = """
/* ============================================================================
 * PREFILL FROM EMBEDDED INPUT - Multimodal/orchestrated prefill path
 * ============================================================================
 * Assumes:
 *   - A_EMBEDDED_INPUT already contains the first num_tokens rows
 *   - token rows after num_tokens are don't-care
 *   - dense_embedding_lookup must be skipped to preserve external prefixes
 * ============================================================================ */
static void ck_prefill_from_embedded_range(CKModel *model, int num_tokens, int prefill_start_pos) {
    if (!model || num_tokens <= 0) return;
    if (prefill_start_pos < 0 || prefill_start_pos >= MAX_SEQ_LEN) return;

    /* Clamp to max context */
    if (num_tokens > MAX_SEQ_LEN - prefill_start_pos) num_tokens = MAX_SEQ_LEN - prefill_start_pos;
    const int prefill_rope_start_pos = model->rope_pos;

    const char *stop_env = getenv("CK_STOP_OP");
    int stop_seq = stop_env ? atoi(stop_env) : -1;
    const char *bridge_noncausal_env = getenv("CK_BRIDGE_NONCAUSAL_VISUAL_CHUNK");
    int bridge_noncausal_visual_chunk = bridge_noncausal_env ? (atoi(bridge_noncausal_env) != 0) : 0;
    const char *bridge_visual_start_env = getenv("CK_BRIDGE_VISUAL_START");
    int bridge_visual_start = bridge_visual_start_env ? atoi(bridge_visual_start_env) : -1;
    const char *bridge_visual_tokens_env = getenv("CK_BRIDGE_VISUAL_TOKENS");
    int bridge_visual_tokens = bridge_visual_tokens_env ? atoi(bridge_visual_tokens_env) : 0;
    const char *debug_outproj_env = getenv("CK_V7_DEBUG_OUTPROJ_FP32");
    int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : 0;
    const float *ck_debug_outproj_fp32_input = NULL;
    const char *debug_mlp_down_env = getenv("CK_V7_DEBUG_MLP_DOWN_FP32");
    int debug_mlp_down_fp32 = debug_mlp_down_env ? (atoi(debug_mlp_down_env) != 0) : 0;
    const char *debug_mlp_down_layer_env = getenv("CK_V7_DEBUG_MLP_DOWN_FP32_LAYER");
    int debug_mlp_down_fp32_layer = debug_mlp_down_layer_env ? atoi(debug_mlp_down_layer_env) : -1;
    const float *ck_debug_mlp_down_fp32_input = NULL;
    const char *debug_mlp_gate_up_env = getenv("CK_V8_DEBUG_MLP_GATE_UP_FP32");
    int debug_mlp_gate_up_fp32 = debug_mlp_gate_up_env ? (atoi(debug_mlp_gate_up_env) != 0) : 0;
    const char *debug_prefill_mlp_gate_up_row_gemv_env = getenv("CK_V8_DEBUG_PREFILL_MLP_GATE_UP_ROW_GEMV");
    int debug_prefill_mlp_gate_up_row_gemv = debug_prefill_mlp_gate_up_row_gemv_env ? (atoi(debug_prefill_mlp_gate_up_row_gemv_env) != 0) : 0;
    const char *debug_prefill_mlp_gate_up_row_gemv_layer_env = getenv("CK_V8_DEBUG_PREFILL_MLP_GATE_UP_ROW_GEMV_LAYER");
    int debug_prefill_mlp_gate_up_row_gemv_layer = debug_prefill_mlp_gate_up_row_gemv_layer_env ? atoi(debug_prefill_mlp_gate_up_row_gemv_layer_env) : -1;
    const char *debug_prefill_mamba_in_proj_row_gemv_env = getenv("CK_V8_DEBUG_PREFILL_MAMBA_IN_PROJ_ROW_GEMV");
    int debug_prefill_mamba_in_proj_row_gemv = debug_prefill_mamba_in_proj_row_gemv_env ? (atoi(debug_prefill_mamba_in_proj_row_gemv_env) != 0) : 0;
    const char *ck_enable_q4k_gateup_swiglu_x16_env = getenv("CK_ENABLE_Q4K_GATEUP_SWIGLU_X16");
    const char *ck_disable_q4k_gateup_swiglu_x16_env = getenv("CK_DISABLE_Q4K_GATEUP_SWIGLU_X16");
    int ck_enable_q4k_gateup_swiglu_x16 = CK_Q4_GATEUP_SWIGLU_X16_DEFAULT;
    if (ck_enable_q4k_gateup_swiglu_x16_env &&
        ck_enable_q4k_gateup_swiglu_x16_env[0] &&
        strcmp(ck_enable_q4k_gateup_swiglu_x16_env, "0") != 0) {
        ck_enable_q4k_gateup_swiglu_x16 = 1;
    }
    if (ck_disable_q4k_gateup_swiglu_x16_env &&
        ck_disable_q4k_gateup_swiglu_x16_env[0] &&
        strcmp(ck_disable_q4k_gateup_swiglu_x16_env, "0") != 0) {
        ck_enable_q4k_gateup_swiglu_x16 = 0;
    }
    int ck_enable_swiglu_q8k_fusion = ck_env_truthy_or_qwen3vl_ocr_profile("CK_ENABLE_SWIGLU_Q8K_FUSION");
    const float *ck_debug_mlp_gate_up_fp32_input = NULL;
"""
    prologue = prologue.replace(
        "int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : 0;",
        f"int debug_outproj_fp32 = debug_outproj_env ? (atoi(debug_outproj_env) != 0) : {debug_outproj_default};",
    )
    prologue = prologue.replace("CK_Q4_GATEUP_SWIGLU_X16_DEFAULT", str(q4_gateup_swiglu_x16_default))
    lines.append(prologue)
    if has_decoder_multimodal_bridge:
        lines.append(
            """    const char *bridge_fp32_env = getenv("CK_V8_MULTIMODAL_PREFILL_FP32");
    int bridge_force_fp32 = bridge_fp32_env ? (atoi(bridge_fp32_env) != 0) : 0;
    if (ck_multimodal_prefill_bridge_is_active() && bridge_force_fp32) {
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
    residual_add_counts_by_layer: Dict[int, int] = {}
    residual_save_counts_by_layer: Dict[int, int] = {}
    rmsnorm_counts_by_layer: Dict[int, int] = {}

    skip_swiglu_guard: Optional[str] = None
    swiglu_q8k_fusion_guard: Optional[str] = None
    swiglu_q8k_fusion_call: Optional[str] = None
    for seq_idx, op in enumerate(ops):
        op_type = str(op.get("op", ""))
        if skip_swiglu_guard and op_type not in {"silu_mul", "swiglu"}:
            skip_swiglu_guard = None
        if swiglu_q8k_fusion_guard and op_type != "quantize_mlp_down_input":
            swiglu_q8k_fusion_guard = None
            swiglu_q8k_fusion_call = None
        if op_type == "residual_add" and "op_instance_idx" not in op and "instance" not in op:
            layer_for_instance = int(op.get("layer", -1))
            inst = residual_add_counts_by_layer.get(layer_for_instance, 0)
            op = dict(op)
            op["op_instance_idx"] = inst
            residual_add_counts_by_layer[layer_for_instance] = inst + 1
        if op_type == "residual_save" and "op_instance_idx" not in op and "instance" not in op:
            layer_for_instance = int(op.get("layer", -1))
            inst = residual_save_counts_by_layer.get(layer_for_instance, 0)
            op = dict(op)
            op["op_instance_idx"] = inst
            residual_save_counts_by_layer[layer_for_instance] = inst + 1
        if op_type in {"rmsnorm", "layernorm", "block_rmsnorm"} and "op_instance_idx" not in op and "instance" not in op:
            layer_for_instance = int(op.get("layer", -1))
            inst = rmsnorm_counts_by_layer.get(layer_for_instance, 0)
            op = dict(op)
            op["op_instance_idx"] = inst
            rmsnorm_counts_by_layer[layer_for_instance] = inst + 1
        linear_emission = resolved_quantized_linear_emission(op)
        quantization_emission = resolved_activation_quantization_emission(op)
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

        if op_type == "quantize_input_0" and quantization_emission:
            op_code = _emit_prefill_quant_rows(
                op,
                seq_idx,
                profile=profile,
            )
        elif op_type == "quantize_input_2" and quantization_emission:
            op_code = _emit_prefill_quant_debug_override(
                op,
                seq_idx,
                config,
                debug_flag_name="debug_mlp_gate_up_fp32",
                debug_input_name="ck_debug_mlp_gate_up_fp32_input",
                profile=profile,
            )
        elif op_type == "mlp_gate_up" and is_q4_q6_q8_linear(linear_emission):
            next_op = ops[seq_idx + 1] if seq_idx + 1 < len(ops) else None
            if (
                linear_emission["weight_format"] == "q4_k"
                and isinstance(next_op, dict)
                and str(next_op.get("op", "")) in {"silu_mul", "swiglu"}
            ):
                fused_var = f"ck_q4_gateup_swiglu_x16_fused_{seq_idx}"
                op_code = _emit_prefill_q4_gateup_swiglu_x16(
                    op,
                    next_op,
                    seq_idx,
                    fused_var,
                    config,
                    debug_flag_name="debug_mlp_gate_up_fp32",
                    debug_input_name="ck_debug_mlp_gate_up_fp32_input",
                    profile=profile,
                )
                skip_swiglu_guard = fused_var
            else:
                op_code = _emit_prefill_gemm_fp32_override(
                    op,
                    seq_idx,
                    config,
                    debug_flag_name="debug_mlp_gate_up_fp32",
                    debug_input_name="ck_debug_mlp_gate_up_fp32_input",
                    profile=profile,
                    dump=dump,
                )
        elif op_type == "quantize_mlp_down_input" and quantization_emission:
            if swiglu_q8k_fusion_guard and swiglu_q8k_fusion_call and quantization_emission["format"] == "q8_k":
                base_code = _emit_prefill_quant_debug_override(
                    op,
                    seq_idx,
                    config,
                    debug_flag_name="debug_mlp_down_fp32",
                    debug_input_name="ck_debug_mlp_down_fp32_input",
                    profile=profile,
                )
                op_code = (
                    f"    if ({swiglu_q8k_fusion_guard}) {{\n"
                    f"{swiglu_q8k_fusion_call}\n"
                    "    } else {\n"
                    + "\n".join("    " + line if line else line for line in base_code.splitlines())
                    + "\n    }"
                )
                swiglu_q8k_fusion_guard = None
                swiglu_q8k_fusion_call = None
            else:
                op_code = _emit_prefill_quant_debug_override(
                    op,
                    seq_idx,
                    config,
                    debug_flag_name="debug_mlp_down_fp32",
                    debug_input_name="ck_debug_mlp_down_fp32_input",
                    profile=profile,
                )
        elif has_multimodal_bridge and op_type == "mlp_down" and is_q4_q6_q8_linear(linear_emission):
            op_code = _emit_prefill_gemm_fp32_override(
                op,
                seq_idx,
                config,
                debug_flag_name="debug_mlp_down_fp32",
                debug_input_name="ck_debug_mlp_down_fp32_input",
                profile=profile,
                dump=dump,
            )
            args_list = op.get("args", [])
            m_arg = next(
                (
                    arg
                    for arg in args_list
                    if isinstance(arg, dict) and str(arg.get("name", "")).lower() == "m"
                ),
                None,
            )
            out_expr = (
                _find_arg_expr(args_list, arg_name="output")
                or _find_arg_expr(args_list, arg_name="out")
                or _find_arg_expr(args_list, arg_name="C")
                or _find_arg_expr(args_list, arg_name="c")
                or _find_arg_expr(args_list, arg_name="y")
            )
            m_expr = _find_arg_expr(args_list, arg_name="M") or _find_arg_expr(args_list, arg_name="m") or "num_tokens"
            if isinstance(m_arg, dict) and str(m_arg.get("source", "")).lower() == "dim:_m":
                m_expr = "num_tokens"
            n_expr = _find_arg_expr(args_list, arg_name="N") or _find_arg_expr(args_list, arg_name="n") or "EMBED_DIM"
            if out_expr:
                raw_out = out_expr.replace("(float*)", "").replace("(void*)", "").strip()
                op_code += (
                    f'\n    ck_debug_export_hidden(model, {int(op.get("layer", -1))}, "mlp_down", '
                    f'(const float*){raw_out}, ({m_expr}) * ({n_expr}));'
                )
        else:
            op_code = emit_prefill_op(
                op,
                seq_idx,
                config,
                profile=profile,
                dump=dump,
                segment_plan_available=has_decoder_multimodal_bridge,
            )
            if has_decoder_multimodal_bridge and op_type == "rope_qk":
                resolved_rope_function = str(op.get("function", "") or "").strip()
                op_code = op_code.replace(
                    f"{resolved_rope_function}(",
                    "ck_multimodal_prefill_mrope_qk(",
                )
        swiglu_block_guard = skip_swiglu_guard if op_type in {"silu_mul", "swiglu"} else None
        if skip_swiglu_guard and op_type in {"silu_mul", "swiglu"}:
            op_code = f"    if (!{skip_swiglu_guard}) {{\n" + "\n".join("    " + line if line else line for line in op_code.splitlines()) + "\n    }"
            skip_swiglu_guard = None
        if op_type in {"silu_mul", "swiglu"}:
            next_op = ops[seq_idx + 1] if seq_idx + 1 < len(ops) else None
            next_quantization = (
                resolved_activation_quantization_emission(next_op)
                if isinstance(next_op, dict)
                else None
            )
            if (
                isinstance(next_op, dict)
                and str(op.get("function", "")) == "swiglu_forward"
                and str(next_op.get("op", "")) == "quantize_mlp_down_input"
                and next_quantization is not None
                and next_quantization["format"] == "q8_k"
            ):
                swiglu_args = op.get("args", [])
                quant_args = next_op.get("args", [])
                swiglu_input = _find_arg_expr(swiglu_args, arg_name="input")
                quant_output = _find_arg_expr(quant_args, arg_name="y") or _find_arg_expr(quant_args, arg_name="output")
                dim_expr = _find_arg_expr(swiglu_args, arg_name="dim") or _find_arg_expr(quant_args, arg_name="k")
                layer_value = op.get("layer", -1)
                layer = int(layer_value) if layer_value is not None else -1
                if swiglu_input and quant_output and dim_expr:
                    guard = f"ck_swiglu_q8k_fused_{seq_idx}"
                    swiglu_q8k_fusion_guard = guard
                    fused_lines = []
                    if profile:
                        fused_lines.append("        CK_PROFILE_BEGIN();")
                    fused_lines.append(f"        swiglu_forward_q8_k((const float*)({swiglu_input}), (void*)({quant_output}), num_tokens, (int)({dim_expr}));")
                    if profile:
                        fused_lines.append(f'        CK_PROFILE_END("prefill", "swiglu_forward_q8_k", "silu_mul_quantize_mlp_down_input", {layer});')
                    swiglu_q8k_fusion_call = "\n".join(fused_lines)
                    block_guard_expr = f" && !{swiglu_block_guard}" if swiglu_block_guard else ""
                    op_code = (
                        f"    const int {guard} = ck_enable_swiglu_q8k_fusion{block_guard_expr} && !(debug_mlp_down_fp32 || debug_mlp_down_fp32_layer == {layer});\n"
                        f"    if (!{guard}) {{\n"
                        + "\n".join("    " + line if line else line for line in op_code.splitlines())
                        + "\n    }"
                    )
        lines.append(op_code)
        lines.append(f"    if (stop_seq == {seq_idx}) return;")
        if has_decoder_multimodal_bridge and op_type == "residual_add":
            residual_add_count_total += 1
            if residual_add_count_total % 2 == 0:
                deepstack_layer = residual_add_count_total // 2 - 1
                if 0 <= deepstack_layer < num_deepstack_layers:
                    lines.append(f"    ck_multimodal_prefill_deepstack_add(model, {deepstack_layer}, num_tokens);")
        lines.append("")

    lines.append("    model->pos = prefill_start_pos + num_tokens;")
    if has_decoder_multimodal_bridge:
        lines.append("    model->rope_pos = ck_multimodal_prefill_bridge_is_active() ? ck_multimodal_prefill_bridge_next_text_pos() : prefill_rope_start_pos + num_tokens;")
        lines.append("    ck_multimodal_prefill_bridge_clear();")
    else:
        lines.append("    model->rope_pos = num_tokens;")
    if bool(config.get("_template_uses_persistent_cross_kv_cache", False)):
        lines.append("    model->encoder_kv_ready = 1;")
    lines.append("}")
    lines.append("")
    lines.append("static void ck_prefill_from_embedded(CKModel *model, int num_tokens) {")
    lines.append("    if (model) model->rope_pos = 0;")
    lines.append("    ck_prefill_from_embedded_range(model, num_tokens, 0);")
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
    has_multimodal_bridge = _has_multimodal_bridge_contract(config)
    has_segmented_append = _has_segmented_append_prefill_contract(config)
    has_unified_mixed = _has_unified_mixed_prefill_contract(config)
    if has_multimodal_bridge and not (has_segmented_append or has_unified_mixed):
        raise RuntimeError(
            "multimodal bridge must resolve exactly one supported prefill schedule"
        )
    bridge = config.get("multimodal_bridge_contract")
    providers = bridge.get("providers") if isinstance(bridge, dict) else None
    prefix_insert = providers.get("prefix_insert") if isinstance(providers, dict) else None
    prefix_insert_function = str(
        (prefix_insert or {}).get("resolved_function", "") or ""
    ).strip()
    if has_multimodal_bridge and not prefix_insert_function:
        raise RuntimeError("multimodal bridge requires a resolved prefix-insert provider")
    if not prefix_insert_function:
        prefix_insert_function = "ck_multimodal_prefix_insert_f32"

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

    bridge_prepare_block_mixed = ""
    bridge_prepare_block_segments = ""
    bridge_prepare_block_visual_segment = ""
    if has_multimodal_bridge:
        bridge_prepare_block_mixed = """
        if (prefix_tokens > 0) {
            if (prefix_grid_x > 0 && prefix_grid_y > 0) {
                int prep_rc = ck_multimodal_prefill_bridge_prepare(
                    prefix_embeddings,
                    prefix_tokens + token_count,
                    0,
                    0,
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
                    int prep_rc = ck_multimodal_prefill_bridge_prepare(
                        prefix_embeddings,
                        prefix_tokens + token_count,
                        0,
                        0,
                        prefix_tokens,
                        prefix_embed_dim,
                        side,
                        side,
                        side
                    );
                    if (prep_rc != 0) return prep_rc;
                }
            }
        }
"""
        bridge_prepare_block_segments = """
        if (prefix_tokens > 0) {
            if (prefix_grid_x > 0 && prefix_grid_y > 0) {
                int prep_rc = ck_multimodal_prefill_bridge_prepare(
                    prefix_embeddings,
                    total_tokens,
                    tokens_before_count,
                    tokens_before_count,
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
                    int prep_rc = ck_multimodal_prefill_bridge_prepare(
                        prefix_embeddings,
                        total_tokens,
                        tokens_before_count,
                        tokens_before_count,
                        prefix_tokens,
                        prefix_embed_dim,
                        side,
                        side,
                        tokens_before_count + side
                    );
                    if (prep_rc != 0) return prep_rc;
                }
            }
        }
"""
        bridge_prepare_block_visual_segment = """
        if (prefix_tokens > 0) {
            if (prefix_grid_x > 0 && prefix_grid_y > 0) {
                int prep_rc = ck_multimodal_prefill_bridge_prepare(
                    prefix_embeddings,
                    prefix_tokens,
                    0,
                    tokens_before_count,
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
                    int prep_rc = ck_multimodal_prefill_bridge_prepare(
                        prefix_embeddings,
                        prefix_tokens,
                        0,
                        tokens_before_count,
                        prefix_tokens,
                        prefix_embed_dim,
                        side,
                        side,
                        tokens_before_count + side
                    );
                    if (prep_rc != 0) return prep_rc;
                }
            }
        }
"""

    if has_segmented_append:
        segments_execute_block = f"""
    const int aligned_embed_dim = ({aligned_embed_dim_expr});
    if (tokens_before_count > 0) {{
        int rc = ck_embed_tokens_at(g_model, tokens_before, tokens_before_count, 0);
        if (rc < 0) return rc;
        g_model->rope_pos = 0;
        ck_prefill_from_embedded_range(g_model, tokens_before_count, 0);
    }}
    if (prefix_tokens > 0) {{
        if (prefix_embed_dim <= 0) prefix_embed_dim = aligned_embed_dim;
        if (prefix_embed_dim < ({embed_dim_expr})) return -13;
        int rc = ck_write_embeddings_at_ex(g_model, prefix_embeddings, prefix_tokens, prefix_embed_dim, 0);
        if (rc < 0) return rc;
{bridge_prepare_block_visual_segment}        g_model->rope_pos = tokens_before_count;
        ck_prefill_from_embedded_range(g_model, prefix_tokens, tokens_before_count);
    }}
    if (tokens_after_count > 0) {{
        int rc = ck_embed_tokens_at(g_model, tokens_after, tokens_after_count, 0);
        if (rc < 0) return rc;
        g_model->rope_pos = prefix_text_pos;
        ck_prefill_from_embedded_range(g_model, tokens_after_count, tokens_before_count + prefix_tokens);
    }}
"""
    else:
        segments_execute_block = f"""
    const int aligned_embed_dim = ({aligned_embed_dim_expr});
    if (tokens_before_count > 0) {{
        int rc = ck_embed_tokens_at(g_model, tokens_before, tokens_before_count, 0);
        if (rc < 0) return rc;
    }}
    if (prefix_tokens > 0) {{
        if (prefix_embed_dim <= 0) prefix_embed_dim = aligned_embed_dim;
        if (prefix_embed_dim < ({embed_dim_expr})) return -13;
        int rc = ck_write_embeddings_at_ex(g_model, prefix_embeddings, prefix_tokens, prefix_embed_dim, tokens_before_count);
        if (rc < 0) return rc;
{bridge_prepare_block_segments}    }}
    if (tokens_after_count > 0) {{
        int rc = ck_embed_tokens_at(g_model, tokens_after, tokens_after_count, tokens_before_count + prefix_tokens);
        if (rc < 0) return rc;
    }}

    ck_prefill_from_embedded(g_model, total_tokens);
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
    int32_t *token_base = {token_ids_expr};
    float *out_base = {output_expr};
    const int aligned_embed_dim = ({aligned_embed_dim_expr});
    return {prefix_insert_function}(
        embeddings,
        token_base,
        out_base,
        count,
        row_dim,
        aligned_embed_dim,
        aligned_embed_dim,
        start_pos,
        ({context_window_expr})
    );
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
#ifdef CK_HAS_PREFILL
    ck_parallel_prefill_release_transient_caches();
#endif
    if (output) memcpy(output, g_model->logits, VOCAB_SIZE * sizeof(float));
    return 0;
}}

CK_EXPORT int ck_model_forward_segments_grid_ex(const int32_t *tokens_before,
                                                int tokens_before_count,
                                                const float *prefix_embeddings,
                                                int prefix_tokens,
                                                int prefix_embed_dim,
                                                int prefix_grid_x,
                                                int prefix_grid_y,
                                                int prefix_text_pos,
                                                const int32_t *tokens_after,
                                                int tokens_after_count,
                                                float *output) {{
    if (!g_model) return -1;
    if (tokens_before_count < 0 || prefix_tokens < 0 || tokens_after_count < 0) return -2;
    if (tokens_before_count > 0 && !tokens_before) return -3;
    if (prefix_tokens > 0 && !prefix_embeddings) return -4;
    if (tokens_after_count > 0 && !tokens_after) return -5;
    if ((prefix_grid_x > 0) != (prefix_grid_y > 0)) return -7;
    if (prefix_grid_x < 0 || prefix_grid_y < 0) return -8;
    if ((prefix_grid_x > 0 || prefix_grid_y > 0) && prefix_tokens <= 0) return -9;
    if (prefix_grid_x > 0 && prefix_grid_y > 0 && prefix_grid_x * prefix_grid_y != prefix_tokens) return -10;

    const int total_tokens = tokens_before_count + prefix_tokens + tokens_after_count;
    if (total_tokens <= 0) return -11;
    if (total_tokens > ({context_window_expr})) return -12;

    memset(g_model->kv_cache, 0, KV_CACHE_SIZE);
    g_model->pos = 0;
    g_model->rope_pos = 0;
    g_model->bridge_has_explicit_positions = 0;

{segments_execute_block}
#ifdef CK_HAS_PREFILL
    ck_parallel_prefill_release_transient_caches();
#endif
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
{bridge_prepare_block_mixed}        int rc = ck_write_embeddings_at_ex(g_model, prefix_embeddings, prefix_tokens, prefix_embed_dim, 0);
        if (rc < 0) return rc;
    }}
    if (token_count > 0) {{
        int rc = ck_embed_tokens_at(g_model, tokens, token_count, prefix_tokens);
        if (rc < 0) return rc;
    }}

    ck_prefill_from_embedded(g_model, prefix_tokens + token_count);
#ifdef CK_HAS_PREFILL
    ck_parallel_prefill_release_transient_caches();
#endif
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


def emit_prefill_weight_prepare_function(ops: List[Dict]) -> str:
    """Emit load-time preparation for prefill-only packed weight providers.

    The provider owns ISA, shape, and environment eligibility. Codegen supplies
    only unique map-resolved weight pointers and dimensions.
    """
    weights: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    mapped_groups: dict[tuple[str, int, int], list[tuple[list[str], str]]] = {}
    mapped_seen: set[tuple[str, tuple[str, ...]]] = set()
    for op in ops:
        preparation = (op.get("call_abi") or {}).get("weight_preparation")
        if isinstance(preparation, dict):
            function = str(preparation.get("function", "") or "").strip()
            argument_map = preparation.get("arguments") or {}
            resolved_args: dict[str, str] = {}
            for symbol, arg_name in argument_map.items():
                expr = _find_arg_expr(op.get("args", []) or [], arg_name=str(arg_name))
                if expr:
                    resolved_args[str(symbol)] = expr
            ordered_symbols = list(argument_map.keys())
            if function and len(resolved_args) == len(ordered_symbols):
                call_args = [resolved_args[str(symbol)] for symbol in ordered_symbols]
                mapped_key = (function, tuple(call_args))
                if mapped_key not in mapped_seen:
                    mapped_seen.add(mapped_key)
                    bytes_expr = str(preparation.get("prepared_bytes", "0") or "0")
                    for symbol, expr in resolved_args.items():
                        bytes_expr = re.sub(rf"\b{re.escape(symbol)}\b", f"({expr})", bytes_expr)
                    max_bytes = int(preparation.get("max_total_bytes", 0) or 0)
                    min_remaining_bytes = int(
                        preparation.get("min_remaining_memory_bytes", 0) or 0
                    )
                    mapped_groups.setdefault(
                        (function, max_bytes, min_remaining_bytes), []
                    ).append((call_args, bytes_expr))
        if "q4_k_q8_k" not in str(op.get("function", "")):
            continue
        args = op.get("args", []) or []
        b_expr = _find_arg_expr(args, arg_name="B")
        n_expr = _find_arg_expr(args, arg_name="N")
        k_expr = _find_arg_expr(args, arg_name="K")
        if not b_expr or not n_expr or not k_expr:
            continue
        key = (b_expr, n_expr, k_expr)
        if key not in seen:
            seen.add(key)
            weights.append(key)

    lines = [
        "static void ck_prepare_prefill_weights(CKModel *model) {",
        "    if (!model) return;",
        "    int prepared = 0;",
    ]
    for group_idx, (
        (function, max_bytes, min_remaining_bytes),
        entries,
    ) in enumerate(mapped_groups.items()):
        lines.append(
            f"    const size_t mapped_prepared_budget_{group_idx} = "
            f"ck_model_preparation_budget((size_t){max_bytes}, "
            f"(size_t){min_remaining_bytes});"
        )
        lines.append(f"    size_t mapped_prepared_bytes_{group_idx} = 0;")
        lines.append(f"    int mapped_prepared_skipped_{group_idx} = 0;")
        for entry_idx, (call_args, bytes_expr) in enumerate(entries):
            lines.append(
                f"    const size_t mapped_prepared_item_{group_idx}_{entry_idx} = "
                f"(size_t)({bytes_expr});"
            )
            if max_bytes <= 0:
                condition = (
                    f"SIZE_MAX - mapped_prepared_bytes_{group_idx} >= "
                    f"mapped_prepared_item_{group_idx}_{entry_idx}"
                )
            else:
                condition = (
                    f"mapped_prepared_item_{group_idx}_{entry_idx} <= "
                    f"mapped_prepared_budget_{group_idx} && "
                    f"mapped_prepared_bytes_{group_idx} <= "
                    f"mapped_prepared_budget_{group_idx} - "
                    f"mapped_prepared_item_{group_idx}_{entry_idx}"
                )
            lines.append(f"    if ({condition}) {{")
            lines.append(
                f"        const int mapped_prepared_result_{group_idx}_{entry_idx} = "
                f"{function}({', '.join(call_args)});"
            )
            lines.append(
                f"        prepared += mapped_prepared_result_{group_idx}_{entry_idx};"
            )
            lines.append(f"        if (mapped_prepared_result_{group_idx}_{entry_idx} > 0) {{")
            lines.append(
                f"            mapped_prepared_bytes_{group_idx} += "
                f"mapped_prepared_item_{group_idx}_{entry_idx};"
            )
            lines.append("        } else {")
            lines.append(f"            mapped_prepared_skipped_{group_idx} += 1;")
            lines.append("        }")
            lines.append("    } else {")
            lines.append(f"        mapped_prepared_skipped_{group_idx} += 1;")
            lines.append("    }")
        lines.append(
            f"    if (mapped_prepared_bytes_{group_idx} > 0 || "
            f"mapped_prepared_skipped_{group_idx} > 0) {{"
        )
        lines.append(
            f'        fprintf(stderr, "[CK parallel prefill] {function}: prepared %zu bytes within runtime budget %zu (map max {max_bytes}, reserve {min_remaining_bytes}); skipped %d weight(s)\\n", mapped_prepared_bytes_{group_idx}, mapped_prepared_budget_{group_idx}, mapped_prepared_skipped_{group_idx});'
        )
        lines.append("    }")
    for b_expr, n_expr, k_expr in weights:
        lines.append(
            "    prepared += ck_q4k_prepare_vnni_x16_weight("
            f"{b_expr}, {n_expr}, {k_expr});"
        )
    lines.extend(
        [
            "    if (prepared > 0) {",
            '        fprintf(stderr, "[CK parallel prefill] Prepared %d model-owned prefill weights at load time\\n", prepared);',
            "    }",
            "}",
        ]
    )
    return "\n".join(lines)


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

    parts.append(emit_prefill_weight_prepare_function(ops))
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
