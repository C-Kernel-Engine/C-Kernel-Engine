#!/usr/bin/env python3
from __future__ import annotations

"""
v8 codegen wrapper.

v8 keeps its own vendored emitter copy so multimodal bring-up can evolve
without reaching back into version/v7. For encoder-only vision graphs we inject
minimal safe defaults and then delegate to the local v8 emitter copy.
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import codegen_core_v8  # type: ignore  # noqa: E402
import codegen_prefill_v8  # type: ignore  # noqa: E402
from vision_bridge_runtime_v8 import resolve_vision_bridge_contract  # type: ignore  # noqa: E402


def _patch_codegen_config(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    cfg = dict(out.get("config", {}) or {})
    if "vocab_size" not in cfg and "n_vocab" not in cfg:
        cfg["vocab_size"] = 1
        cfg["n_vocab"] = 1
        cfg["_v8_codegen_default_vocab"] = True
    out["config"] = cfg
    return out


def _inject_vision_only_fallbacks(code: str, layout_obj: Dict[str, Any]) -> str:
    act_buffers = (layout_obj.get("memory", {}) or {}).get("activations", {}).get("buffers", []) or []
    present = {str(buf.get("name", "")) for buf in act_buffers}
    fallback_defs = []
    for name in ("kv_cache", "rope_cache", "logits"):
        if name not in present:
            macro = f"A_{name.upper()}"
            fallback_defs.append(
                f"#ifndef {macro}\n#define {macro} 0\n#endif"
            )
    if not fallback_defs:
        return code

    block = "/* v8 vision-only fallback macros */\n" + "\n".join(fallback_defs) + "\n"
    insert_after = "#include <math.h>"
    if insert_after in code:
        return code.replace(insert_after, insert_after + "\n\n" + block, 1)
    return block + "\n" + code


def _inject_strict_vision_encoder_oracle(code: str, layout_obj: Dict[str, Any]) -> str:
    act_buffers = (layout_obj.get("memory", {}) or {}).get("activations", {}).get("buffers", []) or []
    by_name = {str(buf.get("name", "")): buf for buf in act_buffers}
    image_buf = by_name.get("image_input")
    bridge = resolve_vision_bridge_contract(layout_obj, by_name)
    target_name = str(bridge.get("fallback_buffer_name", "vision_output"))
    output_buf = by_name.get(target_name)
    if not image_buf or not output_buf:
        return code

    cfg = dict(layout_obj.get("config", {}) or {})
    image_h = int(cfg.get("image_size", 0) or 0)
    image_w = int(cfg.get("image_size", 0) or 0)
    if image_h <= 0 or image_w <= 0:
        return code

    image_elems = int(image_buf.get("size_bytes", image_buf.get("size", 0)) or 0) // 4
    output_elems = int(bridge.get("used_nbytes", 0) or 0) // 4
    pixel_count = image_h * image_w
    if pixel_count <= 0 or image_elems <= 0 or output_elems <= 0:
        return code
    if image_elems % pixel_count != 0:
        return code

    channels = image_elems // pixel_count
    if channels <= 0:
        return code

    target_macro = f"A_{target_name.upper()}"
    block = f"""    {{
        const char *strict_mtmd_oracle_env = getenv("CK_STRICT_MTMD_CLIP_ORACLE");
        int strict_mtmd_oracle = strict_mtmd_oracle_env ? (atoi(strict_mtmd_oracle_env) != 0) : 0;
        if (strict_mtmd_oracle && ck_strict_parity_enabled()) {{
            if (ck_strict_mtmd_clip_encode_planar_f32(
                    (const float*)(MEM + A_IMAGE_INPUT),
                    {channels},
                    {image_h},
                    {image_w},
                    (float*)(MEM + {target_macro}),
                    {output_elems})) {{
                model->pos++;
                return;
            }}
        }}
    }}
"""

    token_store_pat = re.compile(
        r"(    /\* Store token at offset [^\n]+\n"
        r"    \*\(int32_t\*\)\([^\n]+\) = token;\n)"
    )
    return token_store_pat.sub(r"\1\n" + block + "\n", code, count=1)


def _inject_activation_lookup_api(code: str, layout_obj: Dict[str, Any]) -> str:
    act_buffers = (layout_obj.get("memory", {}) or {}).get("activations", {}).get("buffers", []) or []
    if not act_buffers:
        return code

    weights = (layout_obj.get("memory", {}) or {}).get("weights", {}) or {}
    activation_base = int(weights.get("base_offset", 0) or 0) + int(weights.get("size", 0) or 0)
    by_name = {str(buf.get("name", "")): buf for buf in act_buffers}
    bridge = resolve_vision_bridge_contract(layout_obj, by_name)

    cases = []
    for buf in act_buffers:
        name = str(buf.get("name", "") or "")
        if not name:
            continue
        runtime_offset = activation_base + int(buf.get("offset", 0) or 0)
        size_bytes = int(buf.get("size_bytes", buf.get("size", 0)) or 0)
        c_name = json.dumps(name)
        cases.append(
            "    if (strcmp(name, {name}) == 0) {{\n"
            "        if (offset_out) *offset_out = (size_t){offset};\n"
            "        if (size_out) *size_out = (size_t){size};\n"
            "        return 1;\n"
            "    }}".format(name=c_name, offset=runtime_offset, size=size_bytes)
        )
    bridge_name = str(bridge.get("named_activation") or "")
    fallback_name = str(bridge.get("fallback_buffer_name") or "")
    fallback_buf = by_name.get(fallback_name)
    bridge_size = int(bridge.get("used_nbytes", 0) or 0)
    if bridge_name and fallback_buf and bridge_size > 0 and bridge_name not in by_name:
        runtime_offset = activation_base + int(fallback_buf.get("offset", 0) or 0)
        c_name = json.dumps(bridge_name)
        cases.append(
            "    if (strcmp(name, {name}) == 0) {{\n"
            "        if (offset_out) *offset_out = (size_t){offset};\n"
            "        if (size_out) *size_out = (size_t){size};\n"
            "        return 1;\n"
            "    }}".format(name=c_name, offset=runtime_offset, size=bridge_size)
        )
    if not cases:
        return code

    block = """/* v8 activation lookup helpers for external hosts */
static int ck_lookup_named_activation_info(const char *name, size_t *offset_out, size_t *size_out) {
    if (!name) return 0;
{cases}
    return 0;
}

CK_EXPORT intptr_t ck_model_get_named_activation_runtime_offset(const char *name) {
    size_t offset = 0;
    if (!ck_lookup_named_activation_info(name, &offset, NULL)) return -1;
    return (intptr_t)offset;
}

CK_EXPORT intptr_t ck_model_get_named_activation_nbytes(const char *name) {
    size_t size = 0;
    if (!ck_lookup_named_activation_info(name, NULL, &size)) return -1;
    return (intptr_t)size;
}

CK_EXPORT uintptr_t ck_model_get_named_activation_ptr(const char *name) {
    size_t offset = 0;
    if (!g_model) return (uintptr_t)0;
    if (!ck_lookup_named_activation_info(name, &offset, NULL)) return (uintptr_t)0;
    return (uintptr_t)(g_model->bump + offset);
}
""".replace("{cases}", "\n".join(cases))
    return code + "\n\n" + block


def _inject_missing_rope_init(
    code: str,
    layout_obj: Dict[str, Any],
    init_call_obj: Dict[str, Any] | None,
) -> str:
    if init_call_obj is not None:
        return code
    if "rope_precompute_cache(" in code:
        return code

    act_buffers = (layout_obj.get("memory", {}) or {}).get("activations", {}).get("buffers", []) or []
    present = {str(buf.get("name", "")) for buf in act_buffers}
    if "rope_cache" not in present:
        return code

    cfg = dict(layout_obj.get("config", {}) or {})
    if not bool(cfg.get("_template_uses_rope")) and "rope_theta" not in cfg:
        return code

    rope_theta = float(cfg.get("rope_theta", 0.0) or 0.0)
    rotary_dim = int(cfg.get("rotary_dim", cfg.get("head_dim", 0)) or 0)
    rope_scaling_type = json.dumps(str(cfg.get("rope_scaling_type", "none")))
    rope_scaling_factor = float(cfg.get("rope_scaling_factor", 1.0) or 1.0)
    if rope_theta <= 0.0 or rotary_dim <= 0:
        return code

    init_block = f"""    /* v8 fallback: precompute RoPE cache when init_call.json was not provided */
    rope_precompute_cache(
        (float*)(g_model->bump + A_ROPE_CACHE),
        (float*)(g_model->bump + A_ROPE_CACHE) + MAX_SEQ_LEN * ROTARY_DIM / 2,
        MAX_SEQ_LEN,
        HEAD_DIM,
        {rope_theta}f,
        ROTARY_DIM,
        {rope_scaling_type},
        {rope_scaling_factor}f
    );"""

    placeholder = "    /* No pre-weights init ops */"
    if placeholder in code:
        return code.replace(placeholder, init_block, 1)

    do_init = _extract_c_function(code, "static int do_init(void) {")
    if do_init is None:
        return code
    start, end, src = do_init
    needle = "    return 0;"
    if needle not in src:
        return code
    patched = src.replace(needle, init_block + "\n\n" + needle, 1)
    return code[:start] + patched + code[end:]


def _normalize_prefill_for_decode_layout(
    prefill_obj: Dict[str, Any] | None,
    layout_obj: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Align appended prefill code with the target decode runtime layout.

    v8 appends a prefill entrypoint into the decode runtime C file. That means
    the prefill code must obey the decode runtime's activation layout, not the
    standalone prefill lowering defaults. In particular, decode runtimes often
    expose last-only logits storage `[1, vocab]`, while standalone prefill IR
    may request full `[T, vocab]` logits plus a `copy_last_logits` fixup.
    Emitting that full logits GEMM into the decode layout overruns A_LOGITS.
    """
    if prefill_obj is None:
        return None

    out = dict(prefill_obj)
    cfg = dict(out.get("config", {}) or {})
    layout_cfg = dict(layout_obj.get("config", {}) or {})
    decode_logits_layout = str(layout_cfg.get("logits_layout", cfg.get("logits_layout", "auto"))).lower()
    if decode_logits_layout != "last":
        return out

    cfg["logits_layout"] = "last"
    out["config"] = cfg

    ops = out.get("operations", [])
    if isinstance(ops, list):
        out["operations"] = [
            op for op in ops
            if str((op or {}).get("op", "")) != "copy_last_logits"
        ]
    return out


def _build_hybrid_decode_prefill_layout(
    decode_layout_obj: Dict[str, Any],
    prefill_layout_obj: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Use prefill-sized activations with decode-runtime config/contracts."""
    if prefill_layout_obj is None:
        return decode_layout_obj

    out = json.loads(json.dumps(prefill_layout_obj))
    out["mode"] = decode_layout_obj.get("mode", out.get("mode", "decode"))

    out_cfg = dict(out.get("config", {}) or {})
    decode_cfg = dict(decode_layout_obj.get("config", {}) or {})
    out_cfg.update(decode_cfg)
    if "logits_layout" not in decode_cfg:
        out_cfg["logits_layout"] = str(out_cfg.get("logits_layout", "last")).lower()
    out["config"] = out_cfg

    decode_memory = dict(decode_layout_obj.get("memory", {}) or {})
    out_memory = dict(out.get("memory", {}) or {})
    if decode_memory.get("weights"):
        out_memory["weights"] = decode_memory["weights"]
    out["memory"] = out_memory
    return out


def _extract_c_function(code: str, signature: str) -> tuple[int, int, str] | None:
    start = code.find(signature)
    if start < 0:
        return None
    brace = code.find("{", start)
    if brace < 0:
        return None
    depth = 0
    end = -1
    for idx in range(brace, len(code)):
        ch = code[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end < 0:
        return None
    return start, end, code[start:end]


def _inject_decode_runtime_multimodal_fallback(code: str, layout_obj: Dict[str, Any]) -> str:
    cfg = dict(layout_obj.get("config", {}) or {})
    if str(cfg.get("logits_layout", "")).lower() != "last":
        return code

    embed_dim = int(cfg.get("embed_dim", 0) or 0)
    if embed_dim <= 0:
        return code

    is_qwen3vl = str(cfg.get("model", cfg.get("name", ""))).lower() == "qwen3vl"
    num_deepstack_layers = int(cfg.get("num_deepstack_layers", 0) or 0) if is_qwen3vl else 0
    input_embed_dim = int(cfg.get("input_embed_dim", 0) or 0)
    if input_embed_dim <= 0 and embed_dim > 0 and num_deepstack_layers > 0:
        input_embed_dim = embed_dim * (1 + num_deepstack_layers)
    if input_embed_dim <= 0:
        input_embed_dim = embed_dim
    deepstack_elems = max(1, num_deepstack_layers * embed_dim)

    def _inject_qwen3vl_deepstack_residuals(src: str) -> str:
        if not (is_qwen3vl and num_deepstack_layers > 0):
            return src

        comment_pat = re.compile(
            r"^    /\* Op (?P<op>\d+): ck_residual_add_token_major \(residual_add\) layer=(?P<layer>\d+) section=body \*/$"
        )
        counts: Dict[int, int] = {}
        lines = src.splitlines(keepends=True)
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            match = comment_pat.match(line.rstrip("\n"))
            if match is None:
                out.append(line)
                i += 1
                continue

            op = int(match.group("op"))
            layer = int(match.group("layer"))
            counts[layer] = counts.get(layer, 0) + 1

            out.append(line)
            i += 1
            while i < len(lines):
                out.append(lines[i])
                if lines[i] == f"    if (stop_seq == {op}) return;\n":
                    i += 1
                    break
                i += 1

            if counts[layer] == 2 and layer < num_deepstack_layers:
                offset = layer * embed_dim
                out.append(
                    f"""    if (g_bridge_deepstack_active) {{
        ck_residual_add_token_major(
            (float*)(model->bump + A_EMBEDDED_INPUT),
            g_bridge_deepstack_slices + {offset},
            (float*)(model->bump + A_EMBEDDED_INPUT),
            1,
            {embed_dim}
        );
    }}
"""
                )

        return "".join(out)

    decode_sig = "static void ck_decode(CKModel *model, int32_t token) {"
    if "static int g_bridge_prefix_tokens;\n" not in code:
        code = code.replace(decode_sig, "static int g_bridge_prefix_tokens;\n\n" + decode_sig, 1)
    decode_fn = _extract_c_function(code, decode_sig)
    if decode_fn is None:
        raise RuntimeError("unable to locate ck_decode for decode-runtime multimodal fallback")
    _, decode_end, decode_src = decode_fn
    decode_src = _inject_qwen3vl_deepstack_residuals(decode_src)
    embedded_decode = decode_src.replace(
        "static void ck_decode(CKModel *model, int32_t token) {",
        "static void ck_decode_embedded(CKModel *model) {",
        1,
    )
    if is_qwen3vl and num_deepstack_layers > 0:
        embedded_decode = (
            f"static int g_bridge_deepstack_active;\n"
            f"static float g_bridge_deepstack_slices[{deepstack_elems}];\n\n"
            + embedded_decode
        )
    embedded_decode, removed = re.subn(
        r"\n    /\* Store token at offset [^\n]*\n"
        r"    \*\(int32_t\*\)\([^\n]*\) = token;\n\n"
        r"    /\* Op 0: [\s\S]*?if \(stop_seq == 0\) return;\n",
        "\n",
        embedded_decode,
        count=1,
    )
    if removed != 1:
        raise RuntimeError("unable to derive ck_decode_embedded from ck_decode")
    code = code[:decode_end] + "\n\n" + embedded_decode + code[decode_end:]

    if is_qwen3vl:
        rope_wrapper = """static void ck_qwen3vl_runtime_mrope_qk(CKModel *model, float *q, float *k, int num_heads, int num_kv_heads, int num_tokens, int head_dim, int aligned_head_dim, int pos_offset, int n_dims, int section_0, int section_1, int section_2, int section_3, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (model && model->bridge_has_explicit_positions) {
        mrope_qk_imrope_positions(q, k, model->bridge_positions, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        return;
    }
    mrope_qk_text(q, k, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset, n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}
"""
        code = code.replace(decode_sig, rope_wrapper + "\n" + decode_sig, 1)
        code = code.replace("mrope_qk_text(", "ck_qwen3vl_runtime_mrope_qk(model, ")
        code = code.replace("    ck_qwen3vl_runtime_mrope_qk(model, q, k, ", "    mrope_qk_text(q, k, ", 1)
        prefill_helper = _extract_c_function(code, "static void ck_qwen3vl_prefill_mrope_qk(")
        if prefill_helper is not None:
            helper_start, helper_end, helper_src = prefill_helper
            helper_patched = helper_src.replace(
                "    ck_qwen3vl_runtime_mrope_qk(model, q, k, ",
                "    mrope_qk_text(q, k, ",
                1,
            )
            code = code[:helper_start] + helper_patched + code[helper_end:]

    decode_decl = "static void ck_decode(CKModel *model, int32_t token);\n"
    if decode_decl in code and "static void ck_bridge_free(void);" not in code:
        code = code.replace(
            decode_decl,
            decode_decl + "static void ck_bridge_free(void);\n",
            1,
        )

    embed_tokens_pat = re.compile(
        r"CK_EXPORT int ck_model_embed_tokens\(const int32_t \*tokens, int count\) \{\n"
        r"[\s\S]*?\n\}\n\n/\* Forward pass \(after embed_tokens\) \*/",
        re.S,
    )
    embed_tokens_repl = """CK_EXPORT int ck_model_embed_tokens(const int32_t *tokens, int count) {
    if (!g_model || !tokens || count <= 0) return -1;

    /* Decode-layout runtimes append prefill support, but their activations are
     * still sized for a single-token decode step. Preserve correctness by
     * replaying prompt tokens through the one-token decode path. */
    for (int i = 0; i < count; i++) {
        ck_decode(g_model, tokens[i]);
    }
    return 0;
}

/* Forward pass (after embed_tokens) */"""
    code, replaced = embed_tokens_pat.subn(embed_tokens_repl, code, count=1)
    if replaced != 1:
        raise RuntimeError("unable to patch ck_model_embed_tokens for decode-runtime multimodal fallback")

    free_marker = "    if (g_manifest) {\n"
    if free_marker in code and "    ck_bridge_free();\n" not in code:
        code = code.replace(free_marker, "    ck_bridge_free();\n" + free_marker, 1)

    bridge_pat = re.compile(
        r"/\* =+\n"
        r" \* MULTIMODAL BRIDGE HELPERS\n"
        r"[\s\S]*?CK_EXPORT int ck_model_forward_mixed\([^\n]*\n"
        r"[\s\S]*?\n\}\n",
        re.S,
    )
    bridge_block = f"""/* ============================================================================
 * MULTIMODAL BRIDGE HELPERS
 * ============================================================================
 * Decode-layout runtimes keep only single-token activations live. For
 * multimodal orchestration and long text prompts, stage rows out-of-band and
 * replay them through the single-token decode path.
 * ============================================================================ */
enum {{
    CK_BRIDGE_ROW_NONE = 0,
    CK_BRIDGE_ROW_EMBED = 1,
    CK_BRIDGE_ROW_TOKEN = 2,
}};

static float *g_bridge_embedding_rows = NULL;
static int g_bridge_embedding_dim = {embed_dim};
static int32_t *g_bridge_token_rows = NULL;
static uint8_t *g_bridge_row_kind = NULL;
static int g_bridge_row_capacity = 0;
static int g_bridge_prefix_start_pos = 0;
static int g_bridge_prefix_tokens = 0;
static int g_bridge_prefix_grid_x = 0;
static int g_bridge_prefix_grid_y = 0;
static int g_bridge_prefix_text_pos = 0;
static int g_bridge_deepstack_active = 0;
static float g_bridge_deepstack_slices[{deepstack_elems}] = {{0}};

static void ck_bridge_free(void) {{
    free(g_bridge_embedding_rows);
    free(g_bridge_token_rows);
    free(g_bridge_row_kind);
    g_bridge_embedding_rows = NULL;
    g_bridge_token_rows = NULL;
    g_bridge_row_kind = NULL;
    g_bridge_row_capacity = 0;
    g_bridge_embedding_dim = {embed_dim};
    g_bridge_deepstack_active = 0;
    memset(g_bridge_deepstack_slices, 0, sizeof(g_bridge_deepstack_slices));
}}

static void ck_bridge_clear_rows(void) {{
    if (g_bridge_row_kind && g_bridge_row_capacity > 0) {{
        memset(g_bridge_row_kind, 0, (size_t)g_bridge_row_capacity * sizeof(uint8_t));
    }}
    g_bridge_prefix_start_pos = 0;
    g_bridge_prefix_tokens = 0;
    g_bridge_prefix_grid_x = 0;
    g_bridge_prefix_grid_y = 0;
    g_bridge_prefix_text_pos = 0;
    g_bridge_embedding_dim = {embed_dim};
    g_bridge_deepstack_active = 0;
    memset(g_bridge_deepstack_slices, 0, sizeof(g_bridge_deepstack_slices));
}}

static int ck_bridge_ensure_capacity(int rows, int row_dim) {{
    if (row_dim < {embed_dim}) return -9;
    if (rows <= g_bridge_row_capacity && row_dim == g_bridge_embedding_dim) return 0;

    size_t old_cap = (size_t)(g_bridge_row_capacity > 0 ? g_bridge_row_capacity : 0);
    size_t new_cap = (size_t)((rows > g_bridge_row_capacity) ? rows : g_bridge_row_capacity);
    if (new_cap == 0) new_cap = (size_t)rows;
    float *new_embeddings = (float*)malloc(new_cap * (size_t)row_dim * sizeof(float));
    int32_t *new_tokens = (int32_t*)malloc(new_cap * sizeof(int32_t));
    uint8_t *new_kind = (uint8_t*)malloc(new_cap * sizeof(uint8_t));
    if (!new_embeddings || !new_tokens || !new_kind) {{
        free(new_embeddings);
        free(new_tokens);
        free(new_kind);
        return -7;
    }}
    memset(new_embeddings, 0, new_cap * (size_t)row_dim * sizeof(float));
    memset(new_tokens, 0, new_cap * sizeof(int32_t));
    memset(new_kind, 0, new_cap * sizeof(uint8_t));
    if (old_cap > 0 && g_bridge_embedding_rows) {{
        size_t copy_cap = old_cap < new_cap ? old_cap : new_cap;
        size_t copy_dim = (size_t)(g_bridge_embedding_dim < row_dim ? g_bridge_embedding_dim : row_dim);
        for (size_t i = 0; i < copy_cap; ++i) {{
            memcpy(
                new_embeddings + i * (size_t)row_dim,
                g_bridge_embedding_rows + i * (size_t)g_bridge_embedding_dim,
                copy_dim * sizeof(float)
            );
        }}
        memcpy(new_tokens, g_bridge_token_rows, copy_cap * sizeof(int32_t));
        memcpy(new_kind, g_bridge_row_kind, copy_cap * sizeof(uint8_t));
    }}
    free(g_bridge_embedding_rows);
    free(g_bridge_token_rows);
    free(g_bridge_row_kind);
    g_bridge_embedding_rows = new_embeddings;
    g_bridge_token_rows = new_tokens;
    g_bridge_row_kind = new_kind;
    g_bridge_embedding_dim = row_dim;
    g_bridge_row_capacity = (int)new_cap;
    return 0;
}}

static int ck_bridge_stage_embeddings(const float *embeddings, int count, int start_pos, int row_dim) {{
    if (!embeddings || count <= 0) return -1;
    if (start_pos < 0 || start_pos >= MAX_SEQ_LEN) return -2;
    if (count > MAX_SEQ_LEN - start_pos) {{
        count = MAX_SEQ_LEN - start_pos;
    }}
    int rc = ck_bridge_ensure_capacity(start_pos + count, row_dim);
    if (rc != 0) return rc;
    memcpy(
        g_bridge_embedding_rows + (size_t)start_pos * (size_t)row_dim,
        embeddings,
        (size_t)count * (size_t)row_dim * sizeof(float)
    );
    memset(g_bridge_token_rows + start_pos, 0, (size_t)count * sizeof(int32_t));
    memset(g_bridge_row_kind + start_pos, CK_BRIDGE_ROW_EMBED, (size_t)count * sizeof(uint8_t));
    return count;
}}

static int ck_bridge_stage_tokens(const int32_t *tokens, int count, int start_pos) {{
    if (!tokens || count <= 0) return -1;
    if (start_pos < 0 || start_pos >= MAX_SEQ_LEN) return -2;
    if (count > MAX_SEQ_LEN - start_pos) {{
        count = MAX_SEQ_LEN - start_pos;
    }}
    int rc = ck_bridge_ensure_capacity(start_pos + count, g_bridge_embedding_dim > 0 ? g_bridge_embedding_dim : {embed_dim});
    if (rc != 0) return rc;
    memcpy(g_bridge_token_rows + start_pos, tokens, (size_t)count * sizeof(int32_t));
    memset(g_bridge_row_kind + start_pos, CK_BRIDGE_ROW_TOKEN, (size_t)count * sizeof(uint8_t));
    return count;
}}

static int ck_bridge_forward_staged(CKModel *model, int total_tokens) {{
    if (!model) return -1;
    if (total_tokens <= 0) return -2;
    if (total_tokens > MAX_SEQ_LEN) {{
        total_tokens = MAX_SEQ_LEN;
    }}

    memset(model->kv_cache, 0, KV_CACHE_SIZE);
    model->pos = 0;
    model->rope_pos = 0;
    model->bridge_has_explicit_positions = 0;
    g_bridge_deepstack_active = 0;
    memset(g_bridge_deepstack_slices, 0, sizeof(g_bridge_deepstack_slices));

    float *embedded_out = (float*)(model->bump + A_EMBEDDED_INPUT);
    int32_t *token_ids = (int32_t*)(model->bump + A_TOKEN_IDS);
    const int prefix_start = g_bridge_prefix_start_pos;
    const int prefix_end = prefix_start + g_bridge_prefix_tokens;
    const int use_explicit_prefix_positions = g_bridge_prefix_tokens > 0 && g_bridge_prefix_grid_x > 0 && g_bridge_prefix_grid_y > 0;
    for (int i = 0; i < total_tokens; ++i) {{
        int kind = (i < g_bridge_row_capacity) ? (int)g_bridge_row_kind[i] : CK_BRIDGE_ROW_NONE;
        if (kind == CK_BRIDGE_ROW_EMBED) {{
            const float *row = g_bridge_embedding_rows + (size_t)i * (size_t)g_bridge_embedding_dim;
            if (use_explicit_prefix_positions && i >= prefix_start && i < prefix_end) {{
                const int local_idx = i - prefix_start;
                const int x = local_idx % g_bridge_prefix_grid_x;
                const int y = local_idx / g_bridge_prefix_grid_x;
                model->bridge_positions[0] = prefix_start;
                model->bridge_positions[1] = prefix_start + y;
                model->bridge_positions[2] = prefix_start + x;
                model->bridge_positions[3] = 0;
                model->bridge_has_explicit_positions = 1;
            }} else {{
                model->bridge_has_explicit_positions = 0;
            }}
            memcpy(
                embedded_out,
                row,
                (size_t)({embed_dim}) * sizeof(float)
            );
            g_bridge_deepstack_active = 0;
            memset(g_bridge_deepstack_slices, 0, sizeof(g_bridge_deepstack_slices));
            if ({num_deepstack_layers} > 0 && g_bridge_embedding_dim > {embed_dim}) {{
                size_t extra_floats = (size_t)(g_bridge_embedding_dim - {embed_dim});
                size_t copy_floats = extra_floats < (size_t){num_deepstack_layers * embed_dim} ? extra_floats : (size_t){num_deepstack_layers * embed_dim};
                if (copy_floats > 0) {{
                    memcpy(g_bridge_deepstack_slices, row + {embed_dim}, copy_floats * sizeof(float));
                    g_bridge_deepstack_active = 1;
                }}
            }}
            token_ids[0] = g_bridge_token_rows[i];
            ck_decode_embedded(model);
            model->bridge_has_explicit_positions = 0;
            g_bridge_deepstack_active = 0;
            memset(g_bridge_deepstack_slices, 0, sizeof(g_bridge_deepstack_slices));
            if (i + 1 == prefix_end) {{
                model->rope_pos = g_bridge_prefix_text_pos;
            }}
            continue;
        }}
        if (kind == CK_BRIDGE_ROW_TOKEN) {{
            g_bridge_deepstack_active = 0;
            memset(g_bridge_deepstack_slices, 0, sizeof(g_bridge_deepstack_slices));
            if (i == prefix_end && model->rope_pos < g_bridge_prefix_text_pos) {{
                model->rope_pos = g_bridge_prefix_text_pos;
            }}
            ck_decode(model, g_bridge_token_rows[i]);
            continue;
        }}
        return -8;
    }}
    return 0;
}}

CK_EXPORT int ck_model_write_embeddings(const float *embeddings, int count, int start_pos) {{
    return ck_bridge_stage_embeddings(embeddings, count, start_pos, {embed_dim});
}}

CK_EXPORT int ck_model_write_embeddings_ex(const float *embeddings, int count, int row_dim, int start_pos) {{
    return ck_bridge_stage_embeddings(embeddings, count, start_pos, row_dim);
}}

CK_EXPORT int ck_model_embed_tokens_at(const int32_t *tokens, int count, int start_pos) {{
    return ck_bridge_stage_tokens(tokens, count, start_pos);
}}

CK_EXPORT int ck_model_forward_from_embeddings(int total_tokens, float *output) {{
    if (!g_model) return -1;
    int rc = ck_bridge_forward_staged(g_model, total_tokens);
    if (rc != 0) return rc;
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
    if (prefix_tokens > 0 && prefix_embed_dim < {embed_dim}) return -6;
    if ((prefix_grid_x > 0) != (prefix_grid_y > 0)) return -7;
    if (prefix_grid_x < 0 || prefix_grid_y < 0) return -8;
    if ((prefix_grid_x > 0 || prefix_grid_y > 0) && prefix_tokens <= 0) return -9;
    if (prefix_grid_x > 0 && prefix_grid_y > 0 && prefix_grid_x * prefix_grid_y != prefix_tokens) return -10;

    const int total_tokens = tokens_before_count + prefix_tokens + tokens_after_count;
    if (total_tokens <= 0) return -11;
    if (total_tokens > MAX_SEQ_LEN) return -12;

    ck_bridge_clear_rows();
    g_bridge_prefix_start_pos = tokens_before_count;
    g_bridge_prefix_tokens = prefix_tokens;
    g_bridge_prefix_text_pos = tokens_before_count + prefix_tokens;
    if (prefix_tokens > 0) {{
        if (prefix_grid_x > 0 && prefix_grid_y > 0) {{
            g_bridge_prefix_grid_x = prefix_grid_x;
            g_bridge_prefix_grid_y = prefix_grid_y;
            g_bridge_prefix_text_pos = prefix_text_pos > 0
                ? prefix_text_pos
                : (tokens_before_count + (prefix_grid_x > prefix_grid_y ? prefix_grid_x : prefix_grid_y));
        }} else {{
            g_bridge_prefix_text_pos = prefix_text_pos > 0
                ? prefix_text_pos
                : (tokens_before_count + prefix_tokens);
        }}
    }}
    if (tokens_before_count > 0) {{
        int rc = ck_bridge_stage_tokens(tokens_before, tokens_before_count, 0);
        if (rc < 0) return rc;
    }}
    if (prefix_tokens > 0) {{
        int rc = ck_bridge_stage_embeddings(
            prefix_embeddings,
            prefix_tokens,
            tokens_before_count,
            prefix_embed_dim > 0 ? prefix_embed_dim : {embed_dim}
        );
        if (rc < 0) return rc;
    }}
    if (tokens_after_count > 0) {{
        int rc = ck_bridge_stage_tokens(tokens_after, tokens_after_count, tokens_before_count + prefix_tokens);
        if (rc < 0) return rc;
    }}
    int rc = ck_bridge_forward_staged(g_model, total_tokens);
    if (rc != 0) return rc;
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
    return ck_model_forward_segments_grid_ex(
        NULL,
        0,
        prefix_embeddings,
        prefix_tokens,
        prefix_embed_dim,
        prefix_grid_x,
        prefix_grid_y,
        prefix_text_pos,
        tokens,
        token_count,
        output
    );
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
    return ck_model_forward_mixed_ex(prefix_embeddings, prefix_tokens, {embed_dim}, tokens, token_count, output);
}}
"""
    code, bridge_replaced = bridge_pat.subn(bridge_block, code, count=1)
    if bridge_replaced != 1:
        raise RuntimeError("unable to replace multimodal bridge block for decode-runtime fallback")

    return code

def _inject_prefill_multimodal_bridge(
    code: str,
    ir_obj: Dict[str, Any],
    *,
    profile: bool = False,
    dump: bool = False,
) -> str:
    if "ck_model_forward_mixed(" in code or "ck_prefill_from_embedded(" in code:
        return code

    ops = ir_obj.get("operations", [])
    config = ir_obj.get("config", {})
    if not isinstance(ops, list) or not isinstance(config, dict):
        return code

    embedded_prefill = codegen_prefill_v8.emit_prefill_from_embedded_function(
        ops,
        config,
        profile=profile,
        dump=dump,
    )
    bridge_api = codegen_prefill_v8.emit_multimodal_bridge_api(ops, config)
    if not embedded_prefill and not bridge_api:
        return code

    extra_parts = []
    if embedded_prefill:
        extra_parts.append(embedded_prefill)
    if bridge_api:
        extra_parts.append(bridge_api)
    return code + "\n\n" + "\n\n".join(extra_parts)


def _patch_standalone_prefill_runtime(code: str, layout_obj: Dict[str, Any]) -> str:
    if str(layout_obj.get("mode", "")).lower() != "prefill":
        return code

    helper_sig = "static void kv_cache_batch_copy("
    if "kv_cache_batch_copy(" in code and helper_sig not in code:
        helper_block = """
/* v8 standalone prefill compat: generic codegen leaves kv_cache_batch_copy as
 * a pseudo-op call. The real multimodal prefill path uses the explicit helper
 * emitted later in this file; this shim only makes the standalone runtime
 * self-contained enough to compile. */
static void kv_cache_batch_copy(void *k_dst, const void *k_src, void *v_dst, const void *v_src, size_t nbytes) {
    if (k_dst && k_src && nbytes > 0) memcpy(k_dst, k_src, nbytes);
    if (v_dst && v_src && nbytes > 0) memcpy(v_dst, v_src, nbytes);
}
"""
        insert_after = "#include <math.h>"
        if insert_after in code:
            code = code.replace(insert_after, insert_after + "\n\n" + helper_block.strip(), 1)
        else:
            code = helper_block.strip() + "\n\n" + code

    code = code.replace("vocab_size * sizeof(float)", "VOCAB_SIZE * sizeof(float)")

    bad_copy_pat = re.compile(
        r"(    /\* Op \d+: memmove \(copy_last_logits\) layer=-1 section=footer \*/\n)"
        r"    memmove\(\n"
        r"        \(void\*\)\(model->bump \+ A_EMBEDDED_INPUT\),\n"
        r"        \(const void\*\)\(model->bump \+ A_LAYER_INPUT\),\n"
        r"        VOCAB_SIZE \* sizeof\(float\)\n"
        r"    \);\n"
    )
    code = bad_copy_pat.sub(
        r"\1"
        "    /* v8 standalone prefill compat: skip invalid generic copy_last_logits.\n"
        "     * The real prefill bridge emits a correct logits fixup later in the file. */\n"
        "    (void)0;\n",
        code,
        count=1,
    )

    return code


def _inject_decode_attention_parity_dumps(code: str, layout_obj: Dict[str, Any]) -> str:
    cfg = dict(layout_obj.get("config", {}) or {})
    if str(cfg.get("logits_layout", "")).lower() != "last":
        return code

    attn_pat = re.compile(
        r"(/\* Op \d+: attention_forward_decode_head_major_gqa_[A-Za-z0-9_]+ \(attn\) layer=(\d+) section=body \*/\n"
        r"(?:    [^\n]*\n)+?"
        r"    \);\n)"
        r"(    if \(stop_seq == \d+\) return;\n)",
        re.M,
    )

    def repl(match: re.Match[str]) -> str:
        call_block = match.group(1)
        layer = int(match.group(2))
        stop_line = match.group(3)
        dump_block = (
            "    #ifdef CK_PARITY_DUMP\n"
            f'    ck_dump_tensor((float*)(model->bump + A_ATTN_SCRATCH), {layer}, "kqv_out", NUM_HEADS * HEAD_DIM);\n'
            "    #endif\n"
        )
        return call_block + dump_block + stop_line

    return attn_pat.sub(repl, code)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v8 codegen wrapper over the stable v7 emitter")
    ap.add_argument("--ir", type=Path, required=True, help="Call-ready IR JSON (IR Lower 3)")
    ap.add_argument("--layout", type=Path, required=True, help="Memory layout JSON")
    ap.add_argument("--output", type=Path, required=True, help="Output C file")
    ap.add_argument("--prefill", type=Path, default=None, help="Optional lowered prefill IR for decoder runtimes")
    ap.add_argument("--prefill-layout", type=Path, default=None, help="Optional prefill layout JSON for hybrid decode+prefill runtimes")
    ap.add_argument("--init", type=Path, default=None, help="Optional init_call.json")
    ap.add_argument("--debug", action="store_true", help="Emit debug dumps")
    ap.add_argument("--profile", action="store_true", help="Emit profiling wrappers")
    ap.add_argument("--parity-dump", action="store_true", help="Emit parity dump helpers")
    ap.add_argument("--strict-contracts", action="store_true", help="Fail on strict contract/codegen errors")
    args = ap.parse_args(argv)

    with open(args.ir, "r", encoding="utf-8") as f:
        ir_obj = _patch_codegen_config(json.load(f))
    with open(args.layout, "r", encoding="utf-8") as f:
        layout_obj = _patch_codegen_config(json.load(f))
    prefill_obj = None
    prefill_layout_obj = None
    if args.prefill_layout is not None:
        with open(args.prefill_layout, "r", encoding="utf-8") as f:
            prefill_layout_obj = _patch_codegen_config(json.load(f))
        layout_obj = _build_hybrid_decode_prefill_layout(layout_obj, prefill_layout_obj)
    if args.prefill is not None:
        with open(args.prefill, "r", encoding="utf-8") as f:
            prefill_obj = _patch_codegen_config(json.load(f))
        prefill_obj = _normalize_prefill_for_decode_layout(prefill_obj, layout_obj)

    init_call_obj = None
    init_path = args.init if args.init is not None else args.ir.parent / "init_call.json"
    if init_path.exists():
        with open(init_path, "r", encoding="utf-8") as f:
            init_call_obj = _patch_codegen_config(json.load(f))

    with tempfile.TemporaryDirectory(prefix="codegen_v8_") as td:
        td_path = Path(td)
        ir_path = td_path / "call.v8.json"
        layout_path = td_path / "layout.v8.json"
        ir_path.write_text(json.dumps(ir_obj, indent=2), encoding="utf-8")
        layout_path.write_text(json.dumps(layout_obj, indent=2), encoding="utf-8")
        prefill_code = ""
        if prefill_obj is not None:
            prefill_path = td_path / "prefill.v8.json"
            prefill_path.write_text(json.dumps(prefill_obj, indent=2), encoding="utf-8")
            prefill_code = codegen_prefill_v8.generate_prefill(
                prefill_path,
                profile=args.profile,
                dump=args.parity_dump,
            )
        code = codegen_core_v8.generate(
            ir_path,
            layout_path,
            debug=args.debug,
            init_call=init_call_obj,
            profile=args.profile,
            dump=args.parity_dump,
            strict_contracts=args.strict_contracts,
        )
        if prefill_code:
            insert_marker = "#include <math.h>"
            if insert_marker in code:
                code = code.replace(
                    insert_marker,
                    insert_marker + "\n\n/* Prefill support enabled */\n#define CK_HAS_PREFILL 1",
                    1,
                )
            code = code + "\n\n" + prefill_code
            if args.prefill_layout is None:
                code = _inject_decode_runtime_multimodal_fallback(code, layout_obj)
        elif str(layout_obj.get("mode", "")).lower() == "prefill":
            code = _inject_prefill_multimodal_bridge(
                code,
                ir_obj,
                profile=args.profile,
                dump=args.parity_dump,
            )
        if args.parity_dump:
            code = _inject_decode_attention_parity_dumps(code, layout_obj)
        code = _inject_vision_only_fallbacks(code, layout_obj)
        code = _patch_standalone_prefill_runtime(code, layout_obj)
        code = _inject_missing_rope_init(code, layout_obj, init_call_obj)
        code = _inject_strict_vision_encoder_oracle(code, layout_obj)
        code = _inject_activation_lookup_api(code, layout_obj)

    args.output.write_text(code, encoding="utf-8")
    print(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
