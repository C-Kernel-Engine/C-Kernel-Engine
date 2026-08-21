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


_AUDIO_FRONTEND_OPS = {
    "audio_wav_decode",
    "audio_resample",
    "audio_pad_or_truncate",
    "audio_stft_tables",
    "audio_stft",
    "audio_mel_filters",
    "audio_log_mel",
    "audio_feature_window",
}


def _init_has_tokenizer_api(init_call_obj: Dict[str, Any] | None) -> bool:
    if not isinstance(init_call_obj, dict):
        return False
    for op in init_call_obj.get("operations", []):
        if not isinstance(op, dict):
            continue
        c_code = op.get("c_code")
        if isinstance(c_code, dict) and "ck_model_has_tokenizer" in str(
            c_code.get("api_functions") or ""
        ):
            return True
    return False


def _init_has_chat_contract(init_call_obj: Dict[str, Any] | None) -> bool:
    if not isinstance(init_call_obj, dict):
        return False
    contract = init_call_obj.get("chat_contract")
    if not isinstance(contract, dict):
        contract = (init_call_obj.get("config") or {}).get("chat_contract")
    if not isinstance(contract, dict):
        for op in init_call_obj.get("operations", []):
            if not isinstance(op, dict):
                continue
            c_code = op.get("c_code")
            if isinstance(c_code, dict) and "ck_model_format_chat" in str(
                c_code.get("api_functions") or ""
            ):
                return True
    return bool(
        isinstance(contract, dict)
        and contract.get("turn_prefix")
        and contract.get("assistant_generation_prefix")
    )


def _emit_runtime_capability_api(
    ir_obj: Dict[str, Any],
    layout_obj: Dict[str, Any],
    init_call_obj: Dict[str, Any] | None,
    generation_config: Dict[str, Any] | None = None,
    *,
    has_mixed_prefill: bool = False,
) -> str:
    """Emit a versioned descriptor from declared IR/configuration semantics."""
    config = dict(ir_obj.get("config", {}) or {})
    layout_config = dict(layout_obj.get("config", {}) or {})
    for key, value in layout_config.items():
        config.setdefault(key, value)
    operations = [op for op in ir_obj.get("operations", []) if isinstance(op, dict)]
    op_names = {str(op.get("op", "")) for op in operations}
    buffers = (
        (layout_obj.get("memory", {}) or {})
        .get("activations", {})
        .get("buffers", [])
        or []
    )
    buffer_names = {str(row.get("name", "")) for row in buffers if isinstance(row, dict)}
    by_name = {
        str(row.get("name", "")): row for row in buffers if isinstance(row, dict)
    }
    bridge = resolve_vision_bridge_contract(layout_obj, by_name)

    scope = str(config.get("artifact_scope") or "").strip().lower()
    has_encoder_input = _AUDIO_FRONTEND_OPS.issubset(op_names) or "image_input" in buffer_names
    has_decoder_state = bool(
        {"logits", "kv_cache"}.intersection(buffer_names)
        or {"logits", "kv_cache_store"}.intersection(op_names)
    )
    encoder_only = scope == "encoder_only" or (
        not scope and has_encoder_input and not has_decoder_state
    )
    decoder_only = scope == "decoder_only" or (
        not scope and has_decoder_state and not has_encoder_input
    )
    if encoder_only:
        role = "CK_MODEL_ROLE_ENCODER"
    elif decoder_only:
        role = "CK_MODEL_ROLE_DECODER"
    else:
        role = "CK_MODEL_ROLE_COMBINED"

    capabilities = ["CK_MODEL_CAP_INIT"]
    if buffers:
        capabilities.append("CK_MODEL_CAP_NAMED_ACTIVATIONS")
    if not encoder_only:
        capabilities.extend(
            ["CK_MODEL_CAP_AUTOREGRESSIVE_DECODE", "CK_MODEL_CAP_XRAY_KV"]
        )
    if _init_has_tokenizer_api(init_call_obj):
        capabilities.extend(
            ["CK_MODEL_CAP_TEXT_ENCODE", "CK_MODEL_CAP_TOKEN_DECODE"]
        )
    if _init_has_chat_contract(init_call_obj):
        capabilities.append("CK_MODEL_CAP_CHAT_FORMAT")
    special_tokens = (
        init_call_obj.get("special_tokens", {})
        if isinstance(init_call_obj, dict)
        else {}
    )
    if isinstance(special_tokens, dict):
        try:
            eos_token_id = int(special_tokens.get("eos_token_id", -1))
        except (TypeError, ValueError):
            eos_token_id = -1
        if eos_token_id >= 0:
            capabilities.append("CK_MODEL_CAP_STOP_TOKENS")
    if has_mixed_prefill and not encoder_only:
        capabilities.append("CK_MODEL_CAP_MIXED_EMBEDDING_PREFILL")
    if bool(config.get("uses_cross_attention")):
        capabilities.append("CK_MODEL_CAP_ENCODER_MEMORY")
    encoder_output_name = str(
        bridge.get("named_activation") or bridge.get("fallback_buffer_name") or ""
    )
    encoder_output_tokens = int(bridge.get("prefix_tokens", 0) or 0)
    encoder_output_dim = int(bridge.get("embed_dim", 0) or 0)
    if encoder_only and encoder_output_name and encoder_output_tokens > 0 and encoder_output_dim > 0:
        capabilities.append("CK_MODEL_CAP_ENCODER_OUTPUT")
    if _AUDIO_FRONTEND_OPS.issubset(op_names):
        capabilities.append("CK_MODEL_CAP_AUDIO_WAV_ENCODER")
    image_tensor_api = ""
    image_buf = by_name.get("image_input")
    image_height = int(config.get("image_height", config.get("image_size", 0)) or 0)
    image_width = int(config.get("image_width", config.get("image_size", 0)) or 0)
    image_nbytes = int(
        (image_buf or {}).get("size_bytes", (image_buf or {}).get("size", 0)) or 0
    )
    image_pixels = image_height * image_width
    image_channels = (
        image_nbytes // (image_pixels * 4)
        if image_pixels > 0 and image_nbytes > 0 and image_nbytes % (image_pixels * 4) == 0
        else 0
    )
    if encoder_only and image_buf and image_channels > 0:
        # This is deliberately not RAW_IMAGE_ENCODER: the generated runtime
        # currently accepts the circuit's normalized FP32 image tensor.
        capabilities.append("CK_MODEL_CAP_IMAGE_TENSOR_ENCODER")
        image_tensor_api = f"""
CK_EXPORT int ck_model_get_image_tensor_shape(
    int *channels,
    int *height,
    int *width) {{
    if (!channels || !height || !width) return -1;
    *channels = {image_channels};
    *height = {image_height};
    *width = {image_width};
    return 0;
}}

CK_EXPORT int ck_model_run_image_tensor_f32(
    const float *data,
    int channels,
    int height,
    int width) {{
    if (!data) return -1;
    if (channels != {image_channels} || height != {image_height} || width != {image_width}) return -2;
    const uintptr_t pointer = ck_model_get_named_activation_ptr("image_input");
    if (!pointer) return -3;
    memcpy((void *)pointer, data, (size_t){image_nbytes});
    return ck_model_decode(0, NULL);
}}
"""
    if not encoder_only and isinstance(generation_config, dict) and generation_config.get(
        "decoder_start_token_id"
    ) is not None:
        capabilities.append("CK_MODEL_CAP_GENERATION_POLICY")

    context_length = int(config.get("context_length", config.get("max_seq_len", 0)) or 0)
    vocab_size = int(config.get("vocab_size", config.get("n_vocab", 0)) or 0)
    encoder_tokens = int(config.get("encoder_memory_length", 0) or 0)
    encoder_dim = int(config.get("embed_dim", 0) or 0) if encoder_tokens > 0 else 0
    primary_tokens = encoder_output_tokens if encoder_only else 0
    primary_dim = encoder_output_dim if encoder_only else 0
    caps_expr = " |\n        ".join(dict.fromkeys(capabilities))
    encoder_output_api = ""
    if encoder_only and encoder_output_name and primary_tokens > 0 and primary_dim > 0:
        encoder_output_api = f"""
CK_EXPORT int ck_model_get_encoder_output(
    const float **data,
    int *tokens,
    int *dim) {{
    if (!data || !tokens || !dim) return -1;
    const uintptr_t pointer = ck_model_get_named_activation_ptr({json.dumps(encoder_output_name)});
    if (!pointer) return -2;
    *data = (const float *)pointer;
    *tokens = {primary_tokens};
    *dim = {primary_dim};
    return 0;
}}
"""

    return f"""
/* Generated capability declaration. Hosts route by this descriptor and never
 * infer modality or execution policy from model names. */
static const CKModelRuntimeDescriptorV8 g_ck_runtime_descriptor_v8 = {{
    sizeof(CKModelRuntimeDescriptorV8),
    CK_MODEL_ABI_V8_VERSION,
    {caps_expr},
    {role},
    0,
    {context_length},
    {vocab_size},
    {encoder_tokens},
    {encoder_dim},
    {primary_tokens},
    {primary_dim},
    {{ 0, 0, 0, 0, 0, 0, 0, 0 }}
}};

CK_EXPORT uint32_t ck_model_get_abi_version(void) {{
    return CK_MODEL_ABI_V8_VERSION;
}}

CK_EXPORT uint64_t ck_model_get_capabilities(void) {{
    return g_ck_runtime_descriptor_v8.capabilities;
}}

CK_EXPORT int ck_model_get_runtime_descriptor(
    CKModelRuntimeDescriptorV8 *descriptor,
    size_t descriptor_size) {{
    if (!descriptor || descriptor_size < sizeof(CKModelRuntimeDescriptorV8)) return -1;
    memcpy(descriptor, &g_ck_runtime_descriptor_v8, sizeof(g_ck_runtime_descriptor_v8));
    return 0;
}}
{encoder_output_api}
{image_tensor_api}
"""



def _emit_generation_policy_api(config: Dict[str, Any] | None) -> str:
    if not isinstance(config, dict) or config.get("decoder_start_token_id") is None:
        return ""
    start = int(config["decoder_start_token_id"])
    no_timestamps_raw = config.get("no_timestamps_token_id", -1)
    eos_raw = config.get("eos_token_id", -1)
    no_timestamps = int(no_timestamps_raw) if no_timestamps_raw is not None else -1
    eos = int(eos_raw) if eos_raw is not None else -1
    language_rows = []
    for marker, token_id in sorted((config.get("lang_to_id") or {}).items()):
        language = str(marker)
        if language.startswith("<|") and language.endswith("|>"):
            language = language[2:-2]
        language_rows.append((language, int(token_id)))
    task_rows = [
        (str(name), int(token_id))
        for name, token_id in sorted((config.get("task_to_id") or {}).items())
    ]
    suppress = [int(value) for value in (config.get("suppress_tokens") or [])]
    begin_suppress = [
        int(value) for value in (config.get("begin_suppress_tokens") or [])
    ]

    def lookup_lines(rows: list[tuple[str, int]], variable: str) -> str:
        return "\n".join(
            f"    if (strcmp({variable}, {json.dumps(name)}) == 0) return {token_id};"
            for name, token_id in rows
        )

    def int_array(name: str, values: list[int]) -> str:
        payload = ", ".join(str(value) for value in values) if values else "-1"
        return f"static const int32_t {name}[] = {{ {payload} }};"

    return f"""
/* Decoder generation policy generated from generation_config.json. */
{int_array("g_ck_suppress_tokens", suppress)}
{int_array("g_ck_begin_suppress_tokens", begin_suppress)}

static int32_t ck_generation_language_token(const char *language) {{
    if (!language) return -1;
{lookup_lines(language_rows, "language")}
    return -1;
}}

static int32_t ck_generation_task_token(const char *task) {{
    if (!task) return -1;
{lookup_lines(task_rows, "task")}
    return -1;
}}

CK_EXPORT int ck_model_build_generation_prefix(
    const char *language,
    const char *task,
    uint32_t flags,
    int32_t *tokens,
    int capacity) {{
    const int32_t language_token = ck_generation_language_token(language);
    const int32_t task_token = ck_generation_task_token(task);
    if (flags & CK_GENERATION_FLAG_TIMESTAMPS) return -3;
    const int required = 4;
    if (language_token < 0 || task_token < 0 || {no_timestamps} < 0) return -2;
    if (!tokens || capacity < required) return required;
    tokens[0] = {start};
    tokens[1] = language_token;
    tokens[2] = task_token;
    tokens[3] = {no_timestamps};
    return required;
}}

CK_EXPORT int ck_model_apply_generation_policy(
    float *logits,
    int vocab_size,
    const int32_t *generated_tokens,
    int generated_count,
    int step,
    uint32_t flags) {{
    (void)generated_tokens;
    (void)generated_count;
    if (!logits || vocab_size <= 0 || step < 0) return -1;
    if (flags & CK_GENERATION_FLAG_TIMESTAMPS) return -3;
    for (int i = 0; i < {len(suppress)}; ++i) {{
        const int token = g_ck_suppress_tokens[i];
        if (token >= 0 && token < vocab_size) logits[token] = -INFINITY;
    }}
    if (step == 0) {{
        for (int i = 0; i < {len(begin_suppress)}; ++i) {{
            const int token = g_ck_begin_suppress_tokens[i];
            if (token >= 0 && token < vocab_size) logits[token] = -INFINITY;
        }}
    }}
    if (!(flags & CK_GENERATION_FLAG_TIMESTAMPS) && {no_timestamps} >= 0) {{
        for (int token = {no_timestamps}; token < vocab_size; ++token) {{
            logits[token] = -INFINITY;
        }}
    }}
    return 0;
}}

CK_EXPORT int32_t ck_model_get_generation_eos_token(void) {{ return {eos}; }}
"""


def _audio_call_expression(op: Dict[str, Any]) -> str:
    function = str(op.get("function", "") or "").strip()
    args = op.get("args")
    if not function or not isinstance(args, list) or op.get("errors"):
        raise RuntimeError(
            "audio frontend codegen requires error-free call IR with a resolved function"
        )
    expressions = []
    for arg in args:
        if not isinstance(arg, dict) or not str(arg.get("expr", "") or "").strip():
            raise RuntimeError(
                f"audio frontend call IR for {function} has an incomplete argument"
            )
        expressions.append(str(arg["expr"]))
    return f"{function}({', '.join(expressions)})"


def _emit_audio_wav_entrypoint(
    ops: list[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    by_op = {
        str(op.get("op", "")): op
        for op in ops
        if str(op.get("op", "")) in _AUDIO_FRONTEND_OPS
    }
    if not by_op:
        return ""
    missing = sorted(_AUDIO_FRONTEND_OPS - set(by_op))
    if missing:
        raise RuntimeError(
            "audio frontend circuit did not lower every required operation: "
            + ", ".join(missing)
        )
    sample_rate = int(config.get("audio_sample_rate", 0) or 0)
    sample_extent = int(config.get("audio_sample_extent", 0) or 0)
    max_source_frames = int(config.get("audio_max_source_frames", 0) or 0)
    if min(sample_rate, sample_extent, max_source_frames) <= 0:
        raise RuntimeError("audio frontend codegen requires explicit positive extents")

    calls = {name: _audio_call_expression(by_op[name]) for name in _AUDIO_FRONTEND_OPS}
    return f"""
/* Generated from the resolved audio frontend call IR. Python must not select
 * or invoke individual frontend kernels. */
CK_EXPORT int ck_model_prepare_audio_wav_window(
    const uint8_t *audio_wav_bytes,
    size_t audio_wav_byte_count,
    int audio_window_start_frame,
    CKAudioWavInfo *audio_metadata) {{
    if (!g_model || !audio_wav_bytes || audio_wav_byte_count == 0) return -1;
    if (audio_window_start_frame < 0) return -1;
    CKModel *model = g_model;
    CKAudioWavInfo local_info;
    CKAudioWavInfo *audio_wav_info = audio_metadata ? audio_metadata : &local_info;
    float *audio_mono = (float*)(g_model->bump + A_AUDIO_SAMPLES);
    const int audio_mono_capacity = {max_source_frames};
    int audio_source_frames = {calls["audio_wav_decode"]};
    if (audio_source_frames <= 0) return -2;
    const int audio_source_rate = audio_wav_info->sample_rate;
    if (audio_window_start_frame != 0 && audio_source_rate != {sample_rate}) return -10;
    int audio_resampled_frames = audio_resampled_frame_count(
        audio_source_frames, audio_source_rate, {sample_rate});
    if (audio_resampled_frames <= 0 || audio_resampled_frames > {max_source_frames}) return -3;
    float *audio_resampled = audio_mono;
    if (audio_source_rate != {sample_rate}) {{
        audio_resampled = (float*)(g_model->bump + A_AUDIO_RESAMPLED);
        if ({calls["audio_resample"]} != 0) return -4;
    }}
    if ({calls["audio_stft_tables"]} != 0) return -6;
    if ({calls["audio_mel_filters"]} != 0) return -8;
    if (audio_wav_info->frames > {sample_extent}) {{
        if ({calls["audio_feature_window"]} <= 0) return -11;
    }} else {{
        if ({calls["audio_pad_or_truncate"]} < 0) return -5;
        if ({calls["audio_stft"]} != 0) return -7;
        if ({calls["audio_log_mel"]} != 0) return -9;
    }}
    return 0;
}}

CK_EXPORT int ck_model_run_audio_wav_window(const uint8_t *audio_wav_bytes,
                                            size_t audio_wav_byte_count,
                                            int audio_window_start_frame,
                                            CKAudioWavInfo *audio_metadata) {{
    const int status = ck_model_prepare_audio_wav_window(
        audio_wav_bytes,
        audio_wav_byte_count,
        audio_window_start_frame,
        audio_metadata);
    if (status != 0) return status;
    ck_prefill_from_embedded(g_model, {int(config.get("context_length", 0) or 0)});
    return 0;
}}

CK_EXPORT int ck_model_run_audio_wav(const uint8_t *audio_wav_bytes,
                                     size_t audio_wav_byte_count,
                                     CKAudioWavInfo *audio_metadata) {{
    return ck_model_run_audio_wav_window(
        audio_wav_bytes, audio_wav_byte_count, 0, audio_metadata);
}}
"""


def _patch_codegen_config(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    cfg = dict(out.get("config", {}) or {})
    if "vocab_size" not in cfg and "n_vocab" not in cfg:
        cfg["vocab_size"] = 1
        cfg["n_vocab"] = 1
        cfg["_v8_codegen_default_vocab"] = True
    out["config"] = cfg
    return out


def _granular_cutpoint_report(ir_obj: Dict[str, Any], layout_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Describe generated stop points without making model-specific assumptions.

    codegen_core_v8 emits `if (stop_seq == <op index>) return;` after most
    operations. The parity harness uses this metadata to turn a layer-level
    request into the concrete CK_STOP_OP index used by generated C.
    """
    ops = ir_obj.get("operations", [])
    if not isinstance(ops, list):
        ops = []
    cutpoints = []
    for index, op in enumerate(ops):
        if not isinstance(op, dict):
            continue
        try:
            layer = int(op.get("layer", -1) if op.get("layer", -1) is not None else -1)
        except (TypeError, ValueError):
            layer = -1
        cutpoints.append(
            {
                "index": int(index),
                "layer": layer,
                "op": str(op.get("op") or op.get("name") or op.get("type") or ""),
                "function": str(op.get("function") or op.get("kernel") or op.get("kernel_fn") or ""),
                "section": str(op.get("section") or ""),
            }
        )
    return {
        "schema": "ck.v8.granular_codegen.v1",
        "mode": str(ir_obj.get("mode") or layout_obj.get("mode") or ""),
        "model": str((ir_obj.get("config") or {}).get("model") or (layout_obj.get("config") or {}).get("model") or ""),
        "cutpoints": cutpoints,
    }


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

    memory = layout_obj.get("memory", {}) or {}
    weights = memory.get("weights", {}) or {}
    arena = memory.get("arena", {}) or {}
    activation_base = int(
        arena.get(
            "activations_base",
            int(weights.get("base_offset", 0) or 0) + int(weights.get("size", 0) or 0),
        )
    )
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
    """Select one fail-closed activation layout for the combined runtime."""
    if prefill_layout_obj is None:
        return decode_layout_obj

    out = codegen_core_v8._select_combined_runtime_layout(
        decode_layout_obj,
        prefill_layout_obj,
    )
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


def _uses_generated_batched_prefill(ir_obj: Dict[str, Any]) -> bool:
    """Return whether the runtime contract permits the generated prefill path."""
    policy = str((ir_obj.get("config") or {}).get("prefill_policy") or "").strip().lower()
    return policy not in {"sequential_decode", "decode"}


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


def _resolved_text_mrope_function(ir_obj: Dict[str, Any]) -> str:
    matches: list[tuple[str, str, str]] = []
    for op in list(ir_obj.get("operations") or []):
        if str(op.get("op", "")) != "rope_qk":
            continue
        resolved = op.get("resolved_contract") or {}
        semantics = resolved.get("semantics") or {}
        if semantics.get("operator_family") != "text_mrope":
            continue
        matches.append(
            (
                str(op.get("function", "") or "").strip(),
                str(resolved.get("function", "") or "").strip(),
                str(resolved.get("kernel_id", "") or "").strip(),
            )
        )
    unique = set(matches)
    if len(unique) != 1:
        raise RuntimeError(
            "multimodal decode requires exactly one resolved text_mrope provider; "
            f"got {sorted(unique)}"
        )
    function, contract_function, kernel_id = next(iter(unique))
    if not function or function != contract_function:
        raise RuntimeError(
            "text_mrope call IR function does not match its resolved contract: "
            f"call={function!r} contract={contract_function!r} kernel={kernel_id!r}"
        )
    return function


def _inject_decode_runtime_multimodal_fallback(
    code: str,
    layout_obj: Dict[str, Any],
    ir_obj: Dict[str, Any],
) -> str:
    cfg = dict(layout_obj.get("config", {}) or {})
    if str(cfg.get("logits_layout", "")).lower() != "last":
        return code

    embed_dim = int(cfg.get("embed_dim", 0) or 0)
    if embed_dim <= 0:
        return code

    bridge_contract = cfg.get("multimodal_bridge_contract")
    has_multimodal_bridge = isinstance(bridge_contract, dict) and bool(
        str(bridge_contract.get("prefix_policy", "") or "").strip()
    )
    num_deepstack_layers = int(cfg.get("num_deepstack_layers", 0) or 0) if has_multimodal_bridge else 0
    input_embed_dim = int(cfg.get("input_embed_dim", 0) or 0)
    if input_embed_dim <= 0 and embed_dim > 0 and num_deepstack_layers > 0:
        input_embed_dim = embed_dim * (1 + num_deepstack_layers)
    if input_embed_dim <= 0:
        input_embed_dim = embed_dim
    deepstack_elems = max(1, num_deepstack_layers * embed_dim)

    def _inject_multimodal_deepstack_residuals(src: str) -> str:
        if not (has_multimodal_bridge and num_deepstack_layers > 0):
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
    decode_src = _inject_multimodal_deepstack_residuals(decode_src)
    embedded_decode = decode_src.replace(
        "static void ck_decode(CKModel *model, int32_t token) {",
        "static void ck_decode_embedded(CKModel *model) {",
        1,
    )
    if has_multimodal_bridge and num_deepstack_layers > 0:
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

    if has_multimodal_bridge:
        text_mrope_function = _resolved_text_mrope_function(ir_obj)
        rope_wrapper = f"""static void ck_multimodal_runtime_mrope_qk(CKModel *model, float *q, float *k, int num_heads, int num_kv_heads, int num_tokens, int head_dim, int aligned_head_dim, int pos_offset, int n_dims, int section_0, int section_1, int section_2, int section_3, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {{
    if (model && model->bridge_has_explicit_positions) {{
        mrope_qk_imrope_positions(q, k, model->bridge_positions, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        return;
    }}
    {text_mrope_function}(q, k, num_heads, num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset, n_dims, section_0, section_1, section_2, section_3, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}}
"""
        code = code.replace(decode_sig, rope_wrapper + "\n" + decode_sig, 1)
        code = code.replace(
            f"{text_mrope_function}(",
            "ck_multimodal_runtime_mrope_qk(model, ",
        )
        code = code.replace(
            "    ck_multimodal_runtime_mrope_qk(model, q, k, ",
            f"    {text_mrope_function}(q, k, ",
            1,
        )
        prefill_helper = _extract_c_function(code, "static void ck_multimodal_prefill_mrope_qk(")
        if prefill_helper is not None:
            helper_start, helper_end, helper_src = prefill_helper
            helper_patched = helper_src.replace(
                "    ck_multimodal_runtime_mrope_qk(model, q, k, ",
                f"    {text_mrope_function}(q, k, ",
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

    encoder_ops = [
        op for op in ops if str(op.get("op", "")) not in _AUDIO_FRONTEND_OPS
    ]
    embedded_prefill = codegen_prefill_v8.emit_prefill_from_embedded_function(
        encoder_ops,
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
        if str(config.get("artifact_scope", "")).strip().lower() == "encoder_only":
            encoder_tokens = int(
                config.get(
                    "context_length",
                    config.get("audio_conv2_output_frames", 0),
                )
                or 0
            )
            if encoder_tokens <= 0:
                raise RuntimeError(
                    "encoder_only artifact requires a positive context_length execution extent"
                )
            extra_parts.append(
                f"""
CK_EXPORT int ck_model_run_encoder(void) {{
    if (!g_model) return -1;
    ck_prefill_from_embedded(g_model, {encoder_tokens});
    return 0;
}}
"""
            )
            audio_entrypoint = _emit_audio_wav_entrypoint(ops, config)
            if audio_entrypoint:
                extra_parts.append(audio_entrypoint)
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
    ap.add_argument("--generation-config", type=Path, default=None, help="Optional generation_config.json")
    ap.add_argument("--debug", action="store_true", help="Emit debug dumps")
    ap.add_argument("--profile", action="store_true", help="Emit profiling wrappers")
    ap.add_argument("--parity-dump", action="store_true", help="Emit parity dump helpers")
    ap.add_argument("--granular-test", action="store_true", help="Emit parity dumps and a generated-op cutpoint map for stitched parity debugging")
    ap.add_argument("--granular-report", type=Path, default=None, help="Optional JSON output for --granular-test cutpoints")
    ap.add_argument("--strict-contracts", action="store_true", help="Fail on strict contract/codegen errors")
    args = ap.parse_args(argv)
    emit_parity_dumps = bool(args.parity_dump or args.granular_test)

    with open(args.ir, "r", encoding="utf-8") as f:
        ir_obj = _patch_codegen_config(json.load(f))
    with open(args.layout, "r", encoding="utf-8") as f:
        layout_obj = _patch_codegen_config(json.load(f))
    uses_generated_batched_prefill = _uses_generated_batched_prefill(ir_obj)
    prefill_obj = None
    prefill_layout_obj = None
    if args.prefill_layout is not None and uses_generated_batched_prefill:
        with open(args.prefill_layout, "r", encoding="utf-8") as f:
            prefill_layout_obj = _patch_codegen_config(json.load(f))
        layout_obj = _build_hybrid_decode_prefill_layout(layout_obj, prefill_layout_obj)
    if args.prefill is not None and uses_generated_batched_prefill:
        with open(args.prefill, "r", encoding="utf-8") as f:
            prefill_obj = _patch_codegen_config(json.load(f))
        prefill_obj = _normalize_prefill_for_decode_layout(prefill_obj, layout_obj)

    init_call_obj = None
    init_path = args.init if args.init is not None else args.ir.parent / "init_call.json"
    if init_path.exists():
        with open(init_path, "r", encoding="utf-8") as f:
            init_call_obj = _patch_codegen_config(json.load(f))
    generation_config_obj = None
    generation_path = args.generation_config
    if generation_path is None:
        candidate = args.ir.parent / "generation_config.json"
        if candidate.exists():
            generation_path = candidate
    if generation_path is not None and generation_path.exists():
        with open(generation_path, "r", encoding="utf-8") as f:
            generation_config_obj = json.load(f)

    with tempfile.TemporaryDirectory(prefix="codegen_v8_") as td:
        td_path = Path(td)
        ir_path = td_path / "call.v8.json"
        layout_path = td_path / "layout.v8.json"
        core_ir_obj = ir_obj
        if str((ir_obj.get("config") or {}).get("artifact_scope", "")).strip().lower() == "encoder_only":
            core_ir_obj = dict(ir_obj)
            core_ir_obj["operations"] = [
                op
                for op in ir_obj.get("operations", [])
                if str(op.get("op", "")) not in _AUDIO_FRONTEND_OPS
            ]
        ir_path.write_text(json.dumps(core_ir_obj, indent=2), encoding="utf-8")
        layout_path.write_text(json.dumps(layout_obj, indent=2), encoding="utf-8")
        prefill_code = ""
        if prefill_obj is not None:
            prefill_path = td_path / "prefill.v8.json"
            prefill_path.write_text(json.dumps(prefill_obj, indent=2), encoding="utf-8")
            prefill_code = codegen_prefill_v8.generate_prefill(
                prefill_path,
                profile=args.profile,
                dump=emit_parity_dumps,
            )
        code = codegen_core_v8.generate(
            ir_path,
            layout_path,
            debug=args.debug,
            init_call=init_call_obj,
            profile=args.profile,
            dump=emit_parity_dumps,
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
                code = _inject_decode_runtime_multimodal_fallback(code, layout_obj, ir_obj)
        elif str(layout_obj.get("mode", "")).lower() == "prefill":
            code = _inject_prefill_multimodal_bridge(
                code,
                ir_obj,
                profile=args.profile,
                dump=emit_parity_dumps,
            )
        if emit_parity_dumps:
            code = _inject_decode_attention_parity_dumps(code, layout_obj)
        code = _inject_vision_only_fallbacks(code, layout_obj)
        code = _patch_standalone_prefill_runtime(code, layout_obj)
        code = _inject_missing_rope_init(code, layout_obj, init_call_obj)
        code = _inject_strict_vision_encoder_oracle(code, layout_obj)
        code = _inject_activation_lookup_api(code, layout_obj)
        include_marker = "#include <math.h>"
        abi_include = '#include "ck_model_abi_v8.h"'
        if abi_include not in code:
            if include_marker in code:
                code = code.replace(include_marker, include_marker + "\n" + abi_include, 1)
            else:
                code = abi_include + "\n" + code
        code += "\n\n" + _emit_runtime_capability_api(
            ir_obj,
            layout_obj,
            init_call_obj,
            generation_config_obj,
            has_mixed_prefill=bool(prefill_code) or
            str(layout_obj.get("mode", "")).lower() == "prefill",
        )
        generation_policy_api = ""
        if str((ir_obj.get("config") or {}).get("artifact_scope") or "").lower() != "encoder_only":
            generation_policy_api = _emit_generation_policy_api(generation_config_obj)
        if generation_policy_api:
            code += "\n\n" + generation_policy_api

    args.output.write_text(code, encoding="utf-8")
    if args.granular_report is not None:
        args.granular_report.parent.mkdir(parents=True, exist_ok=True)
        args.granular_report.write_text(
            json.dumps(_granular_cutpoint_report(ir_obj, layout_obj), indent=2),
            encoding="utf-8",
        )
    print(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
