from __future__ import annotations

import ctypes
from typing import Any, Mapping


def _config_int(config: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        try:
            value = int(config.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def buffer_nbytes(buf: Mapping[str, Any] | None) -> int:
    if not isinstance(buf, Mapping):
        return 0
    try:
        return int(buf.get("size_bytes", buf.get("size", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def resolve_vision_bridge_contract(
    layout_obj: Mapping[str, Any],
    activation_buffers: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    prefer_total_output: bool = False,
) -> dict[str, Any]:
    cfg = dict(layout_obj.get("config", {}) or {})
    by_name = dict(activation_buffers or {})

    bridge_embed_dim = _config_int(cfg, "projection_dim", "projector_out_dim")
    projector_out_dim = _config_int(cfg, "projector_out_dim", "projection_dim", "embed_dim")
    projector_total_out_dim = _config_int(cfg, "projector_total_out_dim")
    merged_tokens = _config_int(cfg, "vision_merged_tokens", "vision_num_patches")

    if bridge_embed_dim <= 0:
        bridge_embed_dim = projector_out_dim
    if projector_out_dim <= 0:
        projector_out_dim = bridge_embed_dim
    if projector_total_out_dim <= 0:
        projector_total_out_dim = projector_out_dim or bridge_embed_dim

    named_activation: str | None = None
    fallback_buffer_name = "vision_output" if "vision_output" in by_name else "embedded_input"
    reason = "vision_output"

    prefer_projector_output = (
        merged_tokens > 0
        and bridge_embed_dim > 0
        and "embedded_input" in by_name
        and (
            "vision_output" not in by_name
            or projector_total_out_dim != bridge_embed_dim
        )
    )
    if prefer_total_output and "vision_output" in by_name and projector_total_out_dim > 0:
        bridge_embed_dim = projector_total_out_dim
        named_activation = "vision_output"
        fallback_buffer_name = "vision_output"
        reason = "vision_output_total"
    elif prefer_projector_output:
        named_activation = "vision_bridge_output"
        fallback_buffer_name = "embedded_input"
        reason = "projector_output"
    elif fallback_buffer_name == "embedded_input":
        reason = "embedded_input_fallback"

    used_nbytes = 0
    if merged_tokens > 0 and bridge_embed_dim > 0:
        used_nbytes = merged_tokens * bridge_embed_dim * ctypes.sizeof(ctypes.c_float)
    else:
        fallback_nbytes = buffer_nbytes(by_name.get(fallback_buffer_name))
        used_nbytes = fallback_nbytes
        if fallback_nbytes > 0 and bridge_embed_dim > 0:
            elems = fallback_nbytes // ctypes.sizeof(ctypes.c_float)
            if elems % bridge_embed_dim == 0:
                merged_tokens = elems // bridge_embed_dim
        elif fallback_nbytes > 0 and bridge_embed_dim <= 0:
            bridge_embed_dim = fallback_nbytes // ctypes.sizeof(ctypes.c_float)
            merged_tokens = 1 if bridge_embed_dim > 0 else 0

    fallback_capacity = buffer_nbytes(by_name.get(fallback_buffer_name))
    if fallback_capacity > 0 and used_nbytes > fallback_capacity:
        raise RuntimeError(
            f"vision bridge contract overruns {fallback_buffer_name}: {used_nbytes} > {fallback_capacity}"
        )

    if bridge_embed_dim <= 0 and merged_tokens > 0 and used_nbytes > 0:
        bridge_embed_dim = used_nbytes // (merged_tokens * ctypes.sizeof(ctypes.c_float))

    return {
        "named_activation": named_activation,
        "fallback_buffer_name": fallback_buffer_name,
        "embed_dim": int(bridge_embed_dim),
        "prefix_tokens": int(merged_tokens),
        "used_nbytes": int(used_nbytes),
        "reason": reason,
        "projector_out_dim": int(projector_out_dim),
        "projector_total_out_dim": int(projector_total_out_dim),
    }


def declare_named_activation_api(lib: ctypes.CDLL) -> None:
    for name, restype in (
        ("ck_model_get_named_activation_runtime_offset", ctypes.c_int64),
        ("ck_model_get_named_activation_nbytes", ctypes.c_int64),
        ("ck_model_get_named_activation_ptr", ctypes.c_uint64),
    ):
        try:
            fn = getattr(lib, name)
        except AttributeError:
            continue
        fn.argtypes = [ctypes.c_char_p]
        fn.restype = restype


def try_named_activation_view(lib: ctypes.CDLL, name: str) -> tuple[int, int] | None:
    try:
        get_ptr = getattr(lib, "ck_model_get_named_activation_ptr")
        get_nbytes = getattr(lib, "ck_model_get_named_activation_nbytes")
    except AttributeError:
        return None

    ptr = int(get_ptr(name.encode()))
    nbytes = int(get_nbytes(name.encode()))
    if ptr == 0 or nbytes <= 0:
        return None
    return ptr, nbytes
