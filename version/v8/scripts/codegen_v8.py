#!/usr/bin/env python3
from __future__ import annotations

"""
v8 codegen wrapper.

v8 still reuses the stable v7 emitter, but vision-only bring-up needs a small
normalization layer because the v7 generator assumes every runtime has decoder
fields such as vocab_size. For encoder-only vision graphs we inject minimal
safe defaults and then delegate to the stable emitter.
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
V7_SCRIPTS = REPO_ROOT / "version" / "v7" / "scripts"

if str(V7_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V7_SCRIPTS))

import codegen_v7  # type: ignore  # noqa: E402
import codegen_prefill_v7  # type: ignore  # noqa: E402
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

    decode_sig = "static void ck_decode(CKModel *model, int32_t token) {"
    decode_fn = _extract_c_function(code, decode_sig)
    if decode_fn is None:
        raise RuntimeError("unable to locate ck_decode for decode-runtime multimodal fallback")
    _, decode_end, decode_src = decode_fn

    embedded_decode = decode_src.replace(
        "static void ck_decode(CKModel *model, int32_t token) {",
        "static void ck_decode_embedded(CKModel *model) {",
        1,
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
static int32_t *g_bridge_token_rows = NULL;
static uint8_t *g_bridge_row_kind = NULL;
static int g_bridge_row_capacity = 0;

static void ck_bridge_free(void) {{
    free(g_bridge_embedding_rows);
    free(g_bridge_token_rows);
    free(g_bridge_row_kind);
    g_bridge_embedding_rows = NULL;
    g_bridge_token_rows = NULL;
    g_bridge_row_kind = NULL;
    g_bridge_row_capacity = 0;
}}

static void ck_bridge_clear_rows(void) {{
    if (g_bridge_row_kind && g_bridge_row_capacity > 0) {{
        memset(g_bridge_row_kind, 0, (size_t)g_bridge_row_capacity * sizeof(uint8_t));
    }}
}}

static int ck_bridge_ensure_capacity(int rows) {{
    if (rows <= g_bridge_row_capacity) return 0;

    size_t old_cap = (size_t)(g_bridge_row_capacity > 0 ? g_bridge_row_capacity : 0);
    size_t new_cap = (size_t)rows;
    float *new_embeddings = (float*)malloc(new_cap * (size_t)({embed_dim}) * sizeof(float));
    int32_t *new_tokens = (int32_t*)malloc(new_cap * sizeof(int32_t));
    uint8_t *new_kind = (uint8_t*)malloc(new_cap * sizeof(uint8_t));
    if (!new_embeddings || !new_tokens || !new_kind) {{
        free(new_embeddings);
        free(new_tokens);
        free(new_kind);
        return -7;
    }}
    if (old_cap > 0) {{
        memcpy(new_embeddings, g_bridge_embedding_rows, old_cap * (size_t)({embed_dim}) * sizeof(float));
        memcpy(new_tokens, g_bridge_token_rows, old_cap * sizeof(int32_t));
        memcpy(new_kind, g_bridge_row_kind, old_cap * sizeof(uint8_t));
    }}
    if (new_cap > old_cap) {{
        memset(new_embeddings + old_cap * (size_t)({embed_dim}), 0, (new_cap - old_cap) * (size_t)({embed_dim}) * sizeof(float));
        memset(new_tokens + old_cap, 0, (new_cap - old_cap) * sizeof(int32_t));
        memset(new_kind + old_cap, 0, (new_cap - old_cap) * sizeof(uint8_t));
    }}
    free(g_bridge_embedding_rows);
    free(g_bridge_token_rows);
    free(g_bridge_row_kind);
    g_bridge_embedding_rows = new_embeddings;
    g_bridge_token_rows = new_tokens;
    g_bridge_row_kind = new_kind;

    g_bridge_row_capacity = (int)new_cap;
    return 0;
}}

static int ck_bridge_stage_embeddings(const float *embeddings, int count, int start_pos) {{
    if (!embeddings || count <= 0) return -1;
    if (start_pos < 0 || start_pos >= MAX_SEQ_LEN) return -2;
    if (count > MAX_SEQ_LEN - start_pos) {{
        count = MAX_SEQ_LEN - start_pos;
    }}
    int rc = ck_bridge_ensure_capacity(start_pos + count);
    if (rc != 0) return rc;
    memcpy(
        g_bridge_embedding_rows + (size_t)start_pos * (size_t)({embed_dim}),
        embeddings,
        (size_t)count * (size_t)({embed_dim}) * sizeof(float)
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
    int rc = ck_bridge_ensure_capacity(start_pos + count);
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

    float *embedded_out = (float*)(model->bump + A_EMBEDDED_INPUT);
    int32_t *token_ids = (int32_t*)(model->bump + A_TOKEN_IDS);
    for (int i = 0; i < total_tokens; ++i) {{
        int kind = (i < g_bridge_row_capacity) ? (int)g_bridge_row_kind[i] : CK_BRIDGE_ROW_NONE;
        if (kind == CK_BRIDGE_ROW_EMBED) {{
            memcpy(
                embedded_out,
                g_bridge_embedding_rows + (size_t)i * (size_t)({embed_dim}),
                (size_t)({embed_dim}) * sizeof(float)
            );
            token_ids[0] = g_bridge_token_rows[i];
            ck_decode_embedded(model);
            continue;
        }}
        if (kind == CK_BRIDGE_ROW_TOKEN) {{
            ck_decode(model, g_bridge_token_rows[i]);
            continue;
        }}
        return -8;
    }}
    return 0;
}}

CK_EXPORT int ck_model_write_embeddings(const float *embeddings, int count, int start_pos) {{
    return ck_bridge_stage_embeddings(embeddings, count, start_pos);
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

CK_EXPORT int ck_model_forward_mixed(const float *prefix_embeddings,
                                     int prefix_tokens,
                                     const int32_t *tokens,
                                     int token_count,
                                     float *output) {{
    if (!g_model) return -1;
    if (prefix_tokens < 0 || token_count < 0) return -2;
    if (prefix_tokens + token_count <= 0) return -3;
    if (prefix_tokens + token_count > MAX_SEQ_LEN) return -4;
    if (prefix_tokens > 0 && !prefix_embeddings) return -5;
    if (token_count > 0 && !tokens) return -6;

    ck_bridge_clear_rows();
    if (prefix_tokens > 0) {{
        int rc = ck_bridge_stage_embeddings(prefix_embeddings, prefix_tokens, 0);
        if (rc < 0) return rc;
    }}
    if (token_count > 0) {{
        int rc = ck_bridge_stage_tokens(tokens, token_count, prefix_tokens);
        if (rc < 0) return rc;
    }}
    int rc = ck_bridge_forward_staged(g_model, prefix_tokens + token_count);
    if (rc != 0) return rc;
    if (output) memcpy(output, g_model->logits, VOCAB_SIZE * sizeof(float));
    return 0;
}}
"""
    code, bridge_replaced = bridge_pat.subn(bridge_block, code, count=1)
    if bridge_replaced != 1:
        raise RuntimeError("unable to replace multimodal bridge block for decode-runtime fallback")

    return code


def _inject_decode_attention_parity_dumps(code: str, layout_obj: Dict[str, Any]) -> str:
    cfg = dict(layout_obj.get("config", {}) or {})
    if str(cfg.get("logits_layout", "")).lower() != "last":
        return code

    attn_pat = re.compile(
        r"(/\* Op \d+: attention_forward_decode_head_major_gqa_flash \(attn\) layer=(\d+) section=body \*/\n"
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
            prefill_code = codegen_prefill_v7.generate_prefill(
                prefill_path,
                profile=args.profile,
                dump=args.parity_dump,
            )
        code = codegen_v7.generate(
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
            code = _inject_decode_runtime_multimodal_fallback(code, layout_obj)
        if args.parity_dump:
            code = _inject_decode_attention_parity_dumps(code, layout_obj)
        code = _inject_vision_only_fallbacks(code, layout_obj)
        code = _inject_missing_rope_init(code, layout_obj, init_call_obj)
        code = _inject_strict_vision_encoder_oracle(code, layout_obj)
        code = _inject_activation_lookup_api(code, layout_obj)

    args.output.write_text(code, encoding="utf-8")
    print(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
