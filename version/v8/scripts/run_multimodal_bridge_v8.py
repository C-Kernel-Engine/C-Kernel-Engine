#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import heapq
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from array import array
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is optional at import time.
    Image = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BUILD_DIR = REPO_ROOT / "build"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gguf_tokenizer import GGUFTokenizer  # type: ignore  # noqa: E402
from vision_bridge_runtime_v8 import (  # type: ignore  # noqa: E402
    declare_named_activation_api,
    resolve_vision_bridge_contract,
    try_named_activation_view,
)


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


convert_gguf_to_bump_v8 = _load_module("convert_gguf_to_bump_v8_bridge", SCRIPT_DIR / "convert_gguf_to_bump_v8.py")
build_ir_v8 = _load_module("build_ir_v8_bridge", SCRIPT_DIR / "build_ir_v8.py")
codegen_v8 = _load_module("codegen_v8_bridge", SCRIPT_DIR / "codegen_v8.py")


_CHAT_TEMPLATE_ALIASES = {
    "qwen": "qwen2",
    "gemma": "gemma3",
}


def _parse_activation_preference_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in list(values or []):
        item = str(raw or "").strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"invalid activation preference override '{item}'; expected op=dtype"
            )
        op_name, pref = item.split("=", 1)
        op_name = op_name.strip()
        pref = pref.strip().lower()
        if not op_name or not pref:
            raise ValueError(
                f"invalid activation preference override '{item}'; expected op=dtype"
            )
        overrides[op_name] = pref
    return overrides


def _apply_activation_preference_overrides(
    config: dict[str, Any],
    overrides: dict[str, str] | None,
) -> dict[str, Any]:
    if not overrides:
        return config
    updated = dict(config)
    prefs = updated.get("activation_preference_by_op")
    if not isinstance(prefs, dict):
        prefs = {}
    else:
        prefs = dict(prefs)
    for op_name, pref in overrides.items():
        prefs[str(op_name)] = str(pref)
    updated["activation_preference_by_op"] = prefs
    return updated


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _log_progress(message: str) -> None:
    print(f"[v8-bridge] {message}", file=sys.stderr, flush=True)


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == encoded:
                return
        except Exception:
            pass
    path.write_text(encoded, encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _payload_ops(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    ops = payload.get("ops")
    if isinstance(ops, list):
        return [op for op in ops if isinstance(op, dict)]
    ops = payload.get("operations")
    if isinstance(ops, list):
        return [op for op in ops if isinstance(op, dict)]
    return []


def _int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _stage_layer_identity(stage_key: str, stage_label: str, section: str, layer: int) -> tuple[str, str]:
    sec = str(section or "body").strip().lower() or "body"
    if sec == "header":
        return (f"{stage_key}:header", f"{stage_label} / Header")
    if sec == "footer":
        return (f"{stage_key}:footer", f"{stage_label} / Footer")
    if sec == "bridge":
        return (f"{stage_key}:bridge", f"{stage_label} / Prefix Stitch")
    if layer >= 0:
        return (f"{stage_key}:layer:{layer}", f"{stage_label} / Layer {layer}")
    return (f"{stage_key}:misc", f"{stage_label} / Misc")


def _decorate_unified_op(op: dict[str, Any], *, stage_key: str, stage_label: str) -> None:
    section = str(op.get("section", "body") or "body")
    stage_layer = _int_or(op.get("layer"), -1)
    layer_key, layer_label = _stage_layer_identity(stage_key, stage_label, section, stage_layer)
    op["network_stage"] = stage_key
    op["network_stage_label"] = stage_label
    op["stage_layer"] = stage_layer
    op["unified_layer_key"] = layer_key
    op["unified_layer_label"] = layer_label


def _append_ir_segment(
    merged_ops: list[dict[str, Any]],
    raw_ops: list[dict[str, Any]],
    *,
    stage_key: str,
    stage_label: str,
    forced_from_ops: dict[int, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    local_ops: list[dict[str, Any]] = []
    op_id_map: dict[int, int] = {}
    base = len(merged_ops)
    for local_idx, raw in enumerate(raw_ops):
        op = copy.deepcopy(raw)
        old_id = _int_or(op.get("op_id"), local_idx)
        new_id = base + local_idx
        op_id_map[old_id] = new_id
        op["source_op_id"] = old_id
        op["op_id"] = new_id
        _decorate_unified_op(op, stage_key=stage_key, stage_label=stage_label)
        local_ops.append(op)

    forced = dict(forced_from_ops or {})
    for op in local_ops:
        dataflow = op.get("dataflow")
        if not isinstance(dataflow, dict):
            continue
        inputs = dataflow.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for info in inputs.values():
            if not isinstance(info, dict) or "from_op" not in info:
                continue
            old_from = _int_or(info.get("from_op"), -1)
            if old_from in forced:
                info["from_op"] = int(forced[old_from])
            elif old_from in op_id_map:
                info["from_op"] = int(op_id_map[old_from])

    merged_ops.extend(local_ops)
    return local_ops, op_id_map


def _append_call_segment(
    merged_ops: list[dict[str, Any]],
    raw_ops: list[dict[str, Any]],
    *,
    stage_key: str,
    stage_label: str,
) -> list[dict[str, Any]]:
    local_ops: list[dict[str, Any]] = []
    base = len(merged_ops)
    for local_idx, raw in enumerate(raw_ops):
        op = copy.deepcopy(raw)
        old_idx = _int_or(op.get("idx"), local_idx)
        op["source_idx"] = old_idx
        op["idx"] = base + local_idx
        _decorate_unified_op(op, stage_key=stage_key, stage_label=stage_label)
        local_ops.append(op)
    merged_ops.extend(local_ops)
    return local_ops


def _count_nonnegative_layers(ops: list[dict[str, Any]]) -> int:
    layers = {
        _int_or(op.get("layer"), -1)
        for op in ops
        if _int_or(op.get("layer"), -1) >= 0
    }
    return len(layers)


def _find_output_ref(
    ops: list[dict[str, Any]],
    *,
    preferred_names: list[str] | None = None,
) -> dict[str, Any] | None:
    preferred = {str(name).strip().lower() for name in list(preferred_names or []) if str(name).strip()}
    fallback: dict[str, Any] | None = None
    for op in reversed(ops):
        outputs = ((op.get("dataflow") or {}).get("outputs") or {})
        if not isinstance(outputs, dict):
            continue
        for output_name, output_info in outputs.items():
            if not isinstance(output_info, dict):
                continue
            entry = {
                "op_id": _int_or(op.get("op_id"), -1),
                "output_name": str(output_name),
                "slot": str(output_info.get("slot") or output_name),
                "tensor": str(output_info.get("tensor") or output_info.get("slot") or output_name),
                "dtype": str(output_info.get("dtype") or "fp32"),
            }
            if fallback is None:
                fallback = entry
            candidates = {
                str(output_name).strip().lower(),
                str(output_info.get("slot") or "").strip().lower(),
                str(output_info.get("tensor") or "").strip().lower(),
            }
            if preferred and any(candidate in preferred for candidate in candidates if candidate):
                return entry
    return fallback


def _build_unified_bridge_ir_op(
    bridge_op_id: int,
    *,
    bridge_report: dict[str, Any],
    encoder_output: dict[str, Any] | None,
    decoder_header_output: dict[str, Any] | None,
) -> dict[str, Any]:
    prefix_source = str(bridge_report.get("prefix_source") or "none")
    output_slot = str((decoder_header_output or {}).get("slot") or "main_stream")
    bridge_op = {
        "op_id": int(bridge_op_id),
        "kernel": "encoder_decoder_bridge",
        "op": "multimodal_bridge",
        "template_op_id": "multimodal_bridge",
        "section": "bridge",
        "layer": -1,
        "instance": 0,
        "params": {
            "prefix_source": prefix_source,
            "prefix_tokens": _int_or(bridge_report.get("prefix_tokens"), 0),
            "prefix_embed_dim": _int_or(bridge_report.get("prefix_embed_dim"), 0),
            "decoder_embed_dim": _int_or(bridge_report.get("decoder_embed_dim"), 0),
            "decoder_input_embed_dim": _int_or(bridge_report.get("decoder_input_embed_dim"), 0),
            "prompt_tokens_before_image": len(list(bridge_report.get("prompt_tokens_before_image") or [])),
            "prompt_tokens_after_image": len(list(bridge_report.get("prompt_tokens_after_image") or [])),
        },
        "dataflow": {
            "inputs": {},
            "outputs": {
                "out": {
                    "dtype": "fp32",
                    "slot": output_slot,
                    "tensor": "bridge.mixed_embeddings",
                }
            },
        },
        "weights": {},
    }
    inputs = bridge_op["dataflow"]["inputs"]
    if isinstance(decoder_header_output, dict) and _int_or(decoder_header_output.get("op_id"), -1) >= 0:
        inputs["text_embeddings"] = {
            "from_op": _int_or(decoder_header_output.get("op_id"), -1),
            "from_output": str(decoder_header_output.get("output_name") or "out"),
            "dtype": str(decoder_header_output.get("dtype") or "fp32"),
            "tensor": str(decoder_header_output.get("tensor") or "decoder.prefill.text_embeddings"),
            "slot": str(decoder_header_output.get("slot") or "main_stream"),
        }
    else:
        inputs["text_embeddings"] = {
            "from": "external:text_embeddings",
            "dtype": "fp32",
            "tensor": "external:text_embeddings",
            "slot": "external:text_embeddings",
        }
    if isinstance(encoder_output, dict) and _int_or(encoder_output.get("op_id"), -1) >= 0:
        inputs["vision_prefix"] = {
            "from_op": _int_or(encoder_output.get("op_id"), -1),
            "from_output": str(encoder_output.get("output_name") or "out"),
            "dtype": str(encoder_output.get("dtype") or "fp32"),
            "tensor": str(encoder_output.get("tensor") or "encoder.vision_output"),
            "slot": str(encoder_output.get("slot") or "vision_output"),
        }
    else:
        inputs["vision_prefix"] = {
            "from": f"external:{prefix_source or 'prefix'}",
            "dtype": "fp32",
            "tensor": f"external:{prefix_source or 'prefix'}",
            "slot": f"external:{prefix_source or 'prefix'}",
        }
    _decorate_unified_op(bridge_op, stage_key="bridge", stage_label="Bridge")
    return bridge_op


def _build_unified_bridge_call_op(
    bridge_idx: int,
    *,
    bridge_report: dict[str, Any],
    encoder_bridge_name: str,
    decoder_output_buffer: str,
) -> dict[str, Any]:
    op = {
        "idx": int(bridge_idx),
        "function": "encoder_decoder_bridge",
        "op": "multimodal_bridge",
        "layer": -1,
        "section": "bridge",
        "args": [
            {
                "name": "vision_prefix",
                "source": "bridge:encoder_prefix",
                "expr": encoder_bridge_name or "external:prefix_embeddings",
                "buffer_ref": encoder_bridge_name or "external_prefix",
            },
            {
                "name": "text_embeddings",
                "source": "bridge:text_embeddings",
                "expr": decoder_output_buffer or "decoder_prefill::embedded_input",
                "buffer_ref": decoder_output_buffer or "embedded_input",
            },
            {
                "name": "output",
                "source": "output:mixed",
                "expr": decoder_output_buffer or "decoder_prefill::embedded_input",
                "buffer_ref": decoder_output_buffer or "embedded_input",
            },
            {
                "name": "prefix_tokens",
                "source": "dim:prefix_tokens",
                "expr": str(_int_or(bridge_report.get("prefix_tokens"), 0)),
            },
            {
                "name": "prefix_embed_dim",
                "source": "dim:prefix_embed_dim",
                "expr": str(_int_or(bridge_report.get("prefix_embed_dim"), 0)),
            },
            {
                "name": "decoder_input_embed_dim",
                "source": "dim:decoder_input_embed_dim",
                "expr": str(_int_or(bridge_report.get("decoder_input_embed_dim"), 0)),
            },
        ],
        "warnings": [],
        "errors": [],
    }
    _decorate_unified_op(op, stage_key="bridge", stage_label="Bridge")
    return op


def _prefixed_name(stage_key: str, name: Any) -> str:
    raw = str(name or "").strip()
    return f"{stage_key}::{raw}" if raw else stage_key


def _merge_layout_segment_records(
    entries: list[dict[str, Any]],
    *,
    stage_key: str,
    stage_label: str,
    offset_shift: int,
    record_kind: str,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        item = copy.deepcopy(raw)
        item["network_stage"] = stage_key
        item["network_stage_label"] = stage_label
        item["name"] = _prefixed_name(stage_key, item.get("name"))
        if isinstance(item.get("define"), str) and str(item["define"]).strip():
            item["define"] = f"{stage_key.upper()}__{str(item['define']).strip()}"
        item["offset"] = _int_or(item.get("offset"), 0) + int(offset_shift)
        if "abs_offset" in item:
            item["abs_offset"] = _int_or(item.get("abs_offset"), _int_or(item.get("offset"), 0)) + int(offset_shift)
        elif record_kind == "activations":
            item["abs_offset"] = int(item["offset"])
        if record_kind == "activations":
            usage = str(item.get("usage") or "").strip()
            item["usage"] = f"{stage_label} / {usage}" if usage else stage_label
        merged.append(item)
    return merged


def _build_full_network_layout(
    encoder_layout: dict[str, Any] | None,
    decoder_layout: dict[str, Any] | None,
    *,
    bridge_report: dict[str, Any],
) -> dict[str, Any]:
    merged_weight_entries: list[dict[str, Any]] = []
    merged_activation_buffers: list[dict[str, Any]] = []
    weight_offset = 0
    activation_offset = 0
    segments: list[dict[str, Any]] = []

    def add_segment(stage_key: str, stage_label: str, payload: dict[str, Any] | None) -> None:
        nonlocal weight_offset, activation_offset
        if not isinstance(payload, dict):
            return
        memory = payload.get("memory")
        if not isinstance(memory, dict):
            return
        weights = memory.get("weights") if isinstance(memory.get("weights"), dict) else {}
        activations = memory.get("activations") if isinstance(memory.get("activations"), dict) else {}
        weight_entries = _merge_layout_segment_records(
            list(weights.get("entries") or []),
            stage_key=stage_key,
            stage_label=stage_label,
            offset_shift=weight_offset,
            record_kind="weights",
        )
        activation_buffers = _merge_layout_segment_records(
            list(activations.get("buffers") or []),
            stage_key=stage_key,
            stage_label=stage_label,
            offset_shift=activation_offset,
            record_kind="activations",
        )
        weight_size = _int_or(weights.get("size"), 0)
        activation_size = _int_or(activations.get("size"), 0)
        merged_weight_entries.extend(weight_entries)
        merged_activation_buffers.extend(activation_buffers)
        segments.append(
            {
                "stage": stage_key,
                "label": stage_label,
                "weight_offset": int(weight_offset),
                "weight_size": int(weight_size),
                "activation_offset": int(activation_offset),
                "activation_size": int(activation_size),
                "weight_entries": len(weight_entries),
                "activation_buffers": len(activation_buffers),
            }
        )
        weight_offset += weight_size
        activation_offset += activation_size

    add_segment("encoder", "Encoder", encoder_layout)
    add_segment("decoder_prefill", "Decoder Prefill", decoder_layout)

    decoder_cfg = dict((decoder_layout or {}).get("config", {}) or {})
    encoder_cfg = dict((encoder_layout or {}).get("config", {}) or {})
    prefix_tokens = _int_or(bridge_report.get("prefix_tokens"), 0)
    prefix_embed_dim = _int_or(bridge_report.get("prefix_embed_dim"), 0)
    bridge_external_bytes = max(0, prefix_tokens * prefix_embed_dim * ctypes.sizeof(ctypes.c_float))
    config = {
        "model": "multimodal_full_network",
        "arch": "bridge_full_network",
        "mode": "prefill",
        "embed_dim": _int_or(bridge_report.get("decoder_embed_dim"), _int_or(decoder_cfg.get("embed_dim"), 0)),
        "input_embed_dim": _int_or(
            bridge_report.get("decoder_input_embed_dim"),
            _int_or(decoder_cfg.get("input_embed_dim"), 0),
        ),
        "num_layers": _int_or(decoder_cfg.get("num_layers"), 0),
        "num_heads": _int_or(decoder_cfg.get("num_heads"), 0),
        "num_kv_heads": _int_or(decoder_cfg.get("num_kv_heads"), 0),
        "head_dim": _int_or(decoder_cfg.get("head_dim"), 0),
        "intermediate_size": _int_or(decoder_cfg.get("intermediate_size"), 0),
        "context_length": _int_or(bridge_report.get("decoder_context_len"), _int_or(decoder_cfg.get("context_length"), 0)),
        "vocab_size": _int_or(decoder_cfg.get("vocab_size"), 0),
        "vision_embed_dim": _int_or((bridge_report.get("encoder_report") or {}).get("embed_dim"), _int_or(encoder_cfg.get("embed_dim"), 0)),
        "vision_prefix_tokens": prefix_tokens,
        "vision_prefix_embed_dim": prefix_embed_dim,
        "vision_grid_x": _int_or(bridge_report.get("prefix_grid_x"), 0),
        "vision_grid_y": _int_or(bridge_report.get("prefix_grid_y"), 0),
        "bridge_external_prefix_bytes": bridge_external_bytes,
        "encoder_layers": _int_or(encoder_cfg.get("num_layers"), _int_or(encoder_cfg.get("num_hidden_layers"), 0)),
        "decoder_prefill_layers": _int_or(decoder_cfg.get("num_layers"), 0),
    }
    return {
        "format": "ck.layout.multimodal_full_network.v1",
        "version": 1,
        "mode": "prefill",
        "config": config,
        "memory": {
            "weights": {
                "size": int(weight_offset),
                "bump_size": int(weight_offset),
                "base_offset": 0,
                "entries": merged_weight_entries,
            },
            "activations": {
                "size": int(activation_offset),
                "buffers": merged_activation_buffers,
            },
            "arena": {
                "mode": "segmented",
                "weights_base": 0,
                "activations_base": int(weight_offset),
                "total_size": int(weight_offset + activation_offset),
            },
            "segments": segments,
            "bridge": {
                "prefix_source": str(bridge_report.get("prefix_source") or "none"),
                "prefix_tokens": prefix_tokens,
                "prefix_embed_dim": prefix_embed_dim,
                "external_prefix_bytes": bridge_external_bytes,
            },
        },
    }


def _build_full_network_graph(
    *,
    workdir: Path,
    bridge_report: dict[str, Any],
    encoder_dir: Path,
    decoder_dir: Path,
) -> dict[str, Any]:
    encoder_ir1 = _json_load(encoder_dir / "ir1.json")
    encoder_call = _json_load(encoder_dir / "call.json")
    encoder_layout = _json_load(encoder_dir / "layout.json")
    decoder_ir1 = _json_load(decoder_dir / "ir1_prefill.json")
    decoder_call = _json_load(decoder_dir / "call_prefill.json")
    decoder_layout = _json_load(decoder_dir / "layout_prefill.json")

    encoder_ir_ops = _payload_ops(encoder_ir1)
    decoder_ir_ops = _payload_ops(decoder_ir1)
    encoder_call_ops = _payload_ops(encoder_call)
    decoder_call_ops = _payload_ops(decoder_call)
    decoder_header_ir = [op for op in decoder_ir_ops if str(op.get("section") or "").lower() == "header"]
    decoder_tail_ir = [op for op in decoder_ir_ops if str(op.get("section") or "").lower() != "header"]
    decoder_header_call = [op for op in decoder_call_ops if str(op.get("section") or "").lower() == "header"]
    decoder_tail_call = [op for op in decoder_call_ops if str(op.get("section") or "").lower() != "header"]

    merged_ir_ops: list[dict[str, Any]] = []
    merged_call_ops: list[dict[str, Any]] = []

    encoder_ir_segment, _ = _append_ir_segment(
        merged_ir_ops,
        encoder_ir_ops,
        stage_key="encoder",
        stage_label="Encoder",
    )
    decoder_header_segment, decoder_header_map = _append_ir_segment(
        merged_ir_ops,
        decoder_header_ir,
        stage_key="decoder_prefill",
        stage_label="Decoder Prefill",
    )
    encoder_bridge_name = str(((bridge_report.get("encoder_report") or {}).get("bridge_activation")) or "vision_output")
    encoder_output = _find_output_ref(
        encoder_ir_segment,
        preferred_names=[
            encoder_bridge_name,
            "vision_output",
            "vision_bridge_output",
            "embedded_input",
        ],
    )
    decoder_header_output = _find_output_ref(decoder_header_segment)
    decoder_header_terminal_old_id = None
    if decoder_header_ir:
        decoder_header_terminal_old_id = max(_int_or(op.get("op_id"), -1) for op in decoder_header_ir)

    bridge_op_id = len(merged_ir_ops)
    bridge_ir_op = _build_unified_bridge_ir_op(
        bridge_op_id,
        bridge_report=bridge_report,
        encoder_output=encoder_output,
        decoder_header_output=decoder_header_output,
    )
    merged_ir_ops.append(bridge_ir_op)

    forced_from_ops = (
        {int(decoder_header_terminal_old_id): int(bridge_op_id)}
        if decoder_header_terminal_old_id is not None and decoder_header_terminal_old_id >= 0
        else {}
    )
    _append_ir_segment(
        merged_ir_ops,
        decoder_tail_ir,
        stage_key="decoder_prefill",
        stage_label="Decoder Prefill",
        forced_from_ops=forced_from_ops,
    )

    _append_call_segment(
        merged_call_ops,
        encoder_call_ops,
        stage_key="encoder",
        stage_label="Encoder",
    )
    decoder_header_call_segment = _append_call_segment(
        merged_call_ops,
        decoder_header_call,
        stage_key="decoder_prefill",
        stage_label="Decoder Prefill",
    )
    decoder_output_buffer = "decoder_prefill::embedded_input"
    for op in reversed(decoder_header_call_segment):
        args = op.get("args")
        if not isinstance(args, list):
            continue
        preferred_buffer = None
        for arg in args:
            if not isinstance(arg, dict):
                continue
            buffer_ref = str(arg.get("buffer_ref") or "").strip()
            if not buffer_ref:
                continue
            source = str(arg.get("source") or "").lower()
            name = str(arg.get("name") or "").lower()
            if source.startswith("output:") or name in {"output", "out", "c", "dst"}:
                preferred_buffer = buffer_ref
                break
            if preferred_buffer is None:
                preferred_buffer = buffer_ref
        if preferred_buffer:
            decoder_output_buffer = f"decoder_prefill::{preferred_buffer}"
            break
    merged_call_ops.append(
        _build_unified_bridge_call_op(
            len(merged_call_ops),
            bridge_report=bridge_report,
            encoder_bridge_name=f"encoder::{encoder_bridge_name}",
            decoder_output_buffer=decoder_output_buffer,
        )
    )
    _append_call_segment(
        merged_call_ops,
        decoder_tail_call,
        stage_key="decoder_prefill",
        stage_label="Decoder Prefill",
    )

    merged_layout = _build_full_network_layout(
        encoder_layout,
        decoder_layout,
        bridge_report=bridge_report,
    )
    merged_ir1 = {
        "format": "ck.ir1.multimodal_full_network.v1",
        "version": _int_or((decoder_ir1 or {}).get("version"), _int_or((encoder_ir1 or {}).get("version"), 1)),
        "mode": "prefill",
        "ops": merged_ir_ops,
        "segments": [
            {
                "stage": "encoder",
                "label": "Encoder",
                "ops": len(encoder_ir_segment),
                "body_layers": _count_nonnegative_layers(encoder_ir_segment),
            },
            {
                "stage": "bridge",
                "label": "Bridge",
                "ops": 1,
                "body_layers": 0,
            },
            {
                "stage": "decoder_prefill",
                "label": "Decoder Prefill",
                "ops": len(decoder_header_segment) + len(decoder_tail_ir),
                "body_layers": _count_nonnegative_layers(decoder_tail_ir),
            },
        ],
        "bridge": {
            "prefix_source": str(bridge_report.get("prefix_source") or "none"),
            "prefix_tokens": _int_or(bridge_report.get("prefix_tokens"), 0),
            "prefix_embed_dim": _int_or(bridge_report.get("prefix_embed_dim"), 0),
        },
    }
    merged_call = {
        "format": "ck.call.multimodal_full_network.v1",
        "version": 1,
        "mode": "prefill",
        "config": dict(merged_layout.get("config", {}) or {}),
        "memory": dict((merged_layout.get("memory") or {})),
        "operations": merged_call_ops,
        "errors": [],
    }
    return {
        "format": "ck.full_network_graph.v1",
        "version": 1,
        "mode": "prefill",
        "bridge_mode": "encoder_decoder" if encoder_ir_ops else "decoder_only",
        "config": dict(merged_layout.get("config", {}) or {}),
        "layout": merged_layout,
        "call": merged_call,
        "ir1": merged_ir1,
        "sources": {
            "bridge_report": str(workdir / "bridge_report.json"),
            "encoder_ir1": str(encoder_dir / "ir1.json"),
            "encoder_call": str(encoder_dir / "call.json"),
            "encoder_layout": str(encoder_dir / "layout.json"),
            "decoder_ir1_prefill": str(decoder_dir / "ir1_prefill.json"),
            "decoder_call_prefill": str(decoder_dir / "call_prefill.json"),
            "decoder_layout_prefill": str(decoder_dir / "layout_prefill.json"),
        },
    }


def _json_read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_identity(path: Path, *, hash_content: bool = False) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        payload["sha256"] = _sha256_path(path)
    return payload


def _artifacts_match_fingerprint(
    stamp_path: Path,
    fingerprint: dict[str, Any],
    required_paths: list[Path],
) -> bool:
    if not all(path.exists() for path in required_paths):
        return False
    return _json_read(stamp_path) == fingerprint


def _converter_fingerprint(gguf_path: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "gguf": _path_identity(gguf_path),
        "converter_script": _path_identity(SCRIPT_DIR / "convert_gguf_to_bump_v8.py", hash_content=True),
    }


def _runtime_fingerprint(
    *,
    manifest_path: Path,
    mode: str,
    context_override: int | None = None,
    parity_dump: bool = False,
) -> dict[str, Any]:
    return {
        "version": 1,
        "mode": str(mode),
        "manifest": _path_identity(manifest_path, hash_content=True),
        "context_override": int(context_override) if context_override is not None else None,
        "parity_dump": bool(parity_dump),
        "build_ir_script": _path_identity(SCRIPT_DIR / "build_ir_v8.py", hash_content=True),
        "codegen_script": _path_identity(SCRIPT_DIR / "codegen_v8.py", hash_content=True),
    }


def _normalize_chat_template_choice(mode: str | None) -> str:
    value = str(mode or "auto").strip().lower()
    if not value:
        return "auto"
    return _CHAT_TEMPLATE_ALIASES.get(value, value)


def _read_gguf_metadata(gguf_path: Path, wanted_keys: set[str]) -> dict[str, Any]:
    if not wanted_keys:
        return {}
    try:
        with open(gguf_path, "rb") as f:
            r = convert_gguf_to_bump_v8.GGUFReader(f)
            magic = r._read_exact(4)
            if magic != b"GGUF":
                return {}
            version = r.u32()
            if version >= 2:
                _ = r.u64()
                n_kv = r.u64()
            else:
                _ = r.u32()
                n_kv = r.u32()

            data: dict[str, Any] = {}
            for _ in range(int(n_kv)):
                key = r.key_str()
                vtype = r.u32()
                if key in wanted_keys:
                    data[key] = convert_gguf_to_bump_v8._gguf_read_value(r, vtype)
                else:
                    convert_gguf_to_bump_v8._gguf_skip_value(r, vtype)
            return data
    except Exception:
        return {}


def _load_builtin_chat_contract(template_name: str | None) -> dict[str, Any] | None:
    doc = build_ir_v8._load_builtin_template_doc(_normalize_chat_template_choice(template_name))
    if not isinstance(doc, dict):
        return None
    contract_doc = doc.get("contract") if isinstance(doc.get("contract"), dict) else None
    if not isinstance(contract_doc, dict):
        return None
    chat_contract = contract_doc.get("chat_contract")
    if not isinstance(chat_contract, dict):
        return None
    return chat_contract


def _fallback_chat_contract_from_template_text(chat_template: str | None) -> dict[str, Any] | None:
    template = str(chat_template or "")
    if "<|im_start|>" in template and "<|im_end|>" in template:
        template_markers = ["<|im_start|>", "<|im_end|>"]
        image_begin_marker = ""
        image_end_marker = ""
        if "<|vision_start|>" in template and "<|vision_end|>" in template:
            image_begin_marker = "<|vision_start|>"
            image_end_marker = "<|vision_end|>"
            template_markers.extend(["<|vision_start|>", "<|vision_end|>"])
        return {
            "version": 1,
            "name": "chatml_auto",
            "raw_prompt_allowed": False,
            "turn_prefix": "<|im_start|>{role}\n",
            "turn_suffix": "<|im_end|>\n",
            "assistant_generation_prefix": "<|im_start|>assistant\n",
            "role_labels": {"system": "system", "user": "user", "assistant": "assistant"},
            "system_prompt_mode": "dedicated_turn",
            "system_prompt_separator": "\n\n",
            "default_system_prompt": "",
            "inject_default_system_prompt": False,
            "force_bos_text_if_tokenizer_add_bos_false": "",
            "last_user_prefix": "",
            "last_user_prefix_suppression_markers": [],
            "thinking_mode_default": "",
            "assistant_generation_prefix_by_thinking_mode": {},
            "last_user_prefix_by_thinking_mode": {},
            "stop_text_markers": ["<|im_end|>"],
            "token_stop_markers": ["<|im_end|>"],
            "image_begin_marker": image_begin_marker,
            "image_end_marker": image_end_marker,
            "template_markers": template_markers,
            "min_response_tokens": 8,
        }
    if "<start_of_turn>" in template and "<end_of_turn>" in template:
        return {
            "version": 1,
            "name": "gemma_auto",
            "raw_prompt_allowed": False,
            "turn_prefix": "<start_of_turn>{role}\n",
            "turn_suffix": "<end_of_turn>\n",
            "assistant_generation_prefix": "<start_of_turn>model\n",
            "role_labels": {"system": "system", "user": "user", "assistant": "model"},
            "system_prompt_mode": "dedicated_turn",
            "system_prompt_separator": "\n\n",
            "default_system_prompt": "",
            "inject_default_system_prompt": False,
            "force_bos_text_if_tokenizer_add_bos_false": "",
            "last_user_prefix": "",
            "last_user_prefix_suppression_markers": [],
            "thinking_mode_default": "",
            "assistant_generation_prefix_by_thinking_mode": {},
            "last_user_prefix_by_thinking_mode": {},
            "stop_text_markers": ["<end_of_turn>"],
            "token_stop_markers": ["<end_of_turn>"],
            "template_markers": ["<start_of_turn>", "<end_of_turn>"],
            "min_response_tokens": 8,
        }
    return None


def _resolve_decoder_chat_contract(
    decoder_gguf: Path,
    *,
    chat_template_mode: str = "auto",
) -> dict[str, Any] | None:
    resolved_mode = _normalize_chat_template_choice(chat_template_mode)
    if resolved_mode == "none":
        return None

    if resolved_mode != "auto":
        explicit = _load_builtin_chat_contract(resolved_mode)
        if explicit is not None:
            return explicit

    meta = _read_gguf_metadata(
        decoder_gguf,
        {"general.architecture", "tokenizer.chat_template"},
    )
    gguf_template_contract = _fallback_chat_contract_from_template_text(meta.get("tokenizer.chat_template"))
    if resolved_mode == "auto" and gguf_template_contract is not None:
        return gguf_template_contract
    arch = str(meta.get("general.architecture") or "").strip().lower()
    if arch:
        contract = _load_builtin_chat_contract(arch)
        if contract is not None:
            return contract

    return gguf_template_contract


def _resolve_contract_thinking_overrides(
    contract: dict[str, Any],
    thinking_mode: str | None,
) -> tuple[str, str]:
    assistant_generation_prefix = str(contract.get("assistant_generation_prefix") or "")
    last_user_prefix = str(contract.get("last_user_prefix") or "")
    requested_mode = str(thinking_mode or "auto").strip().lower()
    default_mode = str(contract.get("thinking_mode_default") or "").strip().lower()
    resolved_mode = default_mode if requested_mode in {"", "auto"} else requested_mode

    assistant_by_mode = contract.get("assistant_generation_prefix_by_thinking_mode")
    if isinstance(assistant_by_mode, dict):
        override = assistant_by_mode.get(resolved_mode)
        if isinstance(override, str):
            assistant_generation_prefix = override

    last_user_prefix_by_mode = contract.get("last_user_prefix_by_thinking_mode")
    if isinstance(last_user_prefix_by_mode, dict):
        override = last_user_prefix_by_mode.get(resolved_mode)
        if isinstance(override, str):
            last_user_prefix = override

    return assistant_generation_prefix, last_user_prefix


def _format_prompt_with_chat_contract(
    prompt: str,
    contract: dict[str, Any] | None,
    *,
    thinking_mode: str = "auto",
    system_prompt: str | None = None,
) -> str:
    if not isinstance(contract, dict):
        return str(prompt or "")

    role_labels = contract.get("role_labels") if isinstance(contract.get("role_labels"), dict) else {}
    turn_prefix = str(contract.get("turn_prefix") or "")
    turn_suffix = str(contract.get("turn_suffix") or "")
    system_prompt_mode = str(contract.get("system_prompt_mode") or "disabled").strip().lower()
    system_prompt_separator = str(contract.get("system_prompt_separator") or "\n\n")
    default_system_prompt = str(contract.get("default_system_prompt") or "")
    inject_default_system_prompt = bool(contract.get("inject_default_system_prompt"))
    bos_prefix = str(contract.get("force_bos_text_if_tokenizer_add_bos_false") or "")
    suppression_markers = [
        str(marker).lower()
        for marker in list(contract.get("last_user_prefix_suppression_markers") or [])
        if str(marker or "").strip()
    ]
    assistant_generation_prefix, last_user_prefix = _resolve_contract_thinking_overrides(contract, thinking_mode)

    user_text = str(prompt or "")
    if last_user_prefix:
        lowered = user_text.lower()
        if last_user_prefix.lower() not in lowered and not any(marker in lowered for marker in suppression_markers):
            user_text = f"{last_user_prefix}{user_text}"

    system_text = str(system_prompt or "")
    if not system_text and inject_default_system_prompt:
        system_text = default_system_prompt

    if system_text and system_prompt_mode == "prepend_first_user":
        user_text = f"{system_text}{system_prompt_separator}{user_text}" if user_text else system_text
        system_text = ""

    def _render_turn(role: str, content: str) -> str:
        label = str(role_labels.get(role) or role)
        prefix = turn_prefix.replace("{role}", label)
        return f"{prefix}{content}{turn_suffix}"

    formatted = ""
    if bos_prefix:
        formatted += bos_prefix
    if system_text and system_prompt_mode == "dedicated_turn":
        formatted += _render_turn("system", system_text)
    formatted += _render_turn("user", user_text)
    formatted += assistant_generation_prefix
    return formatted if formatted else user_text


def _resolve_multimodal_image_markers(contract: dict[str, Any] | None) -> tuple[str, str] | None:
    if not isinstance(contract, dict):
        return None
    image_begin = str(contract.get("image_begin_marker") or "")
    image_end = str(contract.get("image_end_marker") or "")
    if image_begin and image_end:
        return image_begin, image_end
    return None


def _format_multimodal_prompt_segments(
    prompt: str,
    contract: dict[str, Any] | None,
    *,
    include_image: bool,
    thinking_mode: str = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    if not include_image:
        formatted = _format_prompt_with_chat_contract(
            prompt,
            contract,
            thinking_mode=thinking_mode,
            system_prompt=system_prompt,
        )
        return {
            "formatted_prompt": formatted,
            "before_text": "",
            "after_text": formatted,
            "uses_image_chunks": False,
            "image_begin_marker": "",
            "image_end_marker": "",
        }

    markers = _resolve_multimodal_image_markers(contract)
    if markers is None:
        formatted = _format_prompt_with_chat_contract(
            prompt,
            contract,
            thinking_mode=thinking_mode,
            system_prompt=system_prompt,
        )
        return {
            "formatted_prompt": formatted,
            "before_text": "",
            "after_text": formatted,
            "uses_image_chunks": False,
            "image_begin_marker": "",
            "image_end_marker": "",
        }

    image_begin, image_end = markers
    sentinel = "<<CK_IMAGE_EMBED_CHUNK>>"
    if sentinel in str(prompt or ""):
        raise ValueError("prompt contains reserved multimodal bridge sentinel")
    formatted = _format_prompt_with_chat_contract(
        f"{image_begin}{sentinel}{image_end}{str(prompt or '')}",
        contract,
        thinking_mode=thinking_mode,
        system_prompt=system_prompt,
    )
    if sentinel not in formatted:
        return {
            "formatted_prompt": formatted,
            "before_text": "",
            "after_text": formatted,
            "uses_image_chunks": False,
            "image_begin_marker": image_begin,
            "image_end_marker": image_end,
        }
    before_text, after_text = formatted.split(sentinel, 1)
    return {
        "formatted_prompt": formatted.replace(sentinel, "<image_embeds>"),
        "before_text": before_text,
        "after_text": after_text,
        "uses_image_chunks": True,
        "image_begin_marker": image_begin,
        "image_end_marker": image_end,
    }


def _ensure_engine_lib(openmp: bool = False) -> None:
    cmd = ["make"]
    if openmp:
        # BUILD_STAMP tracks compiler flags, so toggling CK_ENABLE_OPENMP only
        # rebuilds when the cached engine library was built for the wrong mode.
        cmd.append("CK_ENABLE_OPENMP=1")
    cmd.append("build/libckernel_engine.so")
    _run(cmd)


def _run_converter(
    gguf_path: Path,
    output_dir: Path,
    context_override: int | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bump_path = output_dir / "weights.bump"
    manifest_path = output_dir / "weights_manifest.json"
    config_path = output_dir / "config.json"
    stamp_path = output_dir / "convert.cache.json"
    fingerprint = _converter_fingerprint(gguf_path)

    if _artifacts_match_fingerprint(stamp_path, fingerprint, [bump_path, manifest_path, config_path]):
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest, manifest_path, bump_path, config_path

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SCRIPT_DIR / "convert_gguf_to_bump_v8.py"),
            "--gguf",
            str(gguf_path),
            "--output",
            str(bump_path),
            "--manifest-out",
            str(manifest_path),
            "--config-out",
            str(config_path),
        ]
        # Keep conversion artifact metadata model-native. Decoder context overrides
        # are enforced later during IR/layout generation, not by mutating config.json.
        _ = context_override
        convert_gguf_to_bump_v8.main()
    finally:
        sys.argv = old_argv

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    _json_write(stamp_path, fingerprint)
    return manifest, manifest_path, bump_path, config_path


def _round_by_factor(x: float, factor: int) -> int:
    return int(round(float(x) / float(factor))) * int(factor)


def _ceil_by_factor(x: float, factor: int) -> int:
    return int(math.ceil(float(x) / float(factor))) * int(factor)


def _floor_by_factor(x: float, factor: int) -> int:
    return int(math.floor(float(x) / float(factor))) * int(factor)


def _calc_qwen_vl_smart_resize(width: int, height: int, align_size: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or align_size <= 0:
        raise ValueError(f"invalid smart-resize inputs width={width} height={height} align={align_size}")
    w_bar = max(align_size, _round_by_factor(width, align_size))
    h_bar = max(align_size, _round_by_factor(height, align_size))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(float(width * height) / float(max_pixels))
        w_bar = max(align_size, _floor_by_factor(width / beta, align_size))
        h_bar = max(align_size, _floor_by_factor(height / beta, align_size))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(float(min_pixels) / float(width * height))
        w_bar = _ceil_by_factor(width * beta, align_size)
        h_bar = _ceil_by_factor(height * beta, align_size)
    return int(w_bar), int(h_bar)


def _coerce_float_triplet(values: Any, default: list[float]) -> list[float]:
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        out: list[float] = []
        for item in list(values)[:3]:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                return [float(v) for v in default]
        return out
    return [float(v) for v in default]


def _ppm_skip_ws_and_comments(data: bytes, idx: int) -> int:
    n = len(data)
    while idx < n:
        byte = data[idx]
        if byte in b" \t\r\n":
            idx += 1
            continue
        if byte == ord("#"):
            idx += 1
            while idx < n and data[idx] not in b"\r\n":
                idx += 1
            continue
        break
    return idx


def _ppm_next_token(data: bytes, idx: int) -> tuple[str, int]:
    idx = _ppm_skip_ws_and_comments(data, idx)
    start = idx
    n = len(data)
    while idx < n and data[idx] not in b" \t\r\n#":
        idx += 1
    if start == idx:
        raise RuntimeError("invalid PPM header: unexpected end of token stream")
    return data[start:idx].decode("ascii"), idx


def _read_ppm_rgb8(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    magic, idx = _ppm_next_token(data, 0)
    width_s, idx = _ppm_next_token(data, idx)
    height_s, idx = _ppm_next_token(data, idx)
    maxval_s, idx = _ppm_next_token(data, idx)
    try:
        width = int(width_s)
        height = int(height_s)
        maxval = int(maxval_s)
    except ValueError as exc:
        raise RuntimeError(f"invalid PPM header in {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid PPM dimensions in {path}: {width}x{height}")
    if maxval <= 0 or maxval > 255:
        raise RuntimeError(f"unsupported PPM maxval in {path}: {maxval}")
    sample_count = width * height * 3
    if magic == "P6":
        if idx >= len(data) or data[idx] not in b" \t\r\n":
            raise RuntimeError(f"invalid PPM header terminator in {path}")
        idx += 1
        payload = data[idx:]
        if len(payload) != sample_count:
            raise RuntimeError(
                f"invalid PPM payload size in {path}: expected {sample_count} bytes, got {len(payload)}"
            )
        return width, height, payload
    if magic == "P3":
        samples: list[int] = []
        while len(samples) < sample_count:
            token, idx = _ppm_next_token(data, idx)
            try:
                value = int(token)
            except ValueError as exc:
                raise RuntimeError(f"invalid PPM sample in {path}: {exc}") from exc
            if value < 0 or value > maxval:
                raise RuntimeError(f"PPM sample out of range in {path}: {value}")
            samples.append(int(round((float(value) / float(maxval)) * 255.0)))
        return width, height, bytes(samples)
    raise RuntimeError(f"unsupported PPM magic in {path}: {magic}")


def _image_source_size(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        width, height, _ = _read_ppm_rgb8(path)
        return width, height
    if Image is None:
        raise RuntimeError(f"Pillow is required for non-PPM image input: {path.name}")
    with Image.open(path) as src:
        width, height = src.size
    return int(width), int(height)


def _resize_rgb8_bilinear(
    src_rgb: bytes,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> list[tuple[int, int, int]]:
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        raise RuntimeError(
            f"invalid resize dimensions src={src_width}x{src_height} dst={dst_width}x{dst_height}"
        )
    if src_width == dst_width and src_height == dst_height:
        return [
            (
                int(src_rgb[i]),
                int(src_rgb[i + 1]),
                int(src_rgb[i + 2]),
            )
            for i in range(0, len(src_rgb), 3)
        ]

    out: list[tuple[int, int, int]] = []
    scale_x = float(src_width) / float(dst_width)
    scale_y = float(src_height) / float(dst_height)
    for y in range(dst_height):
        src_y = ((float(y) + 0.5) * scale_y) - 0.5
        y0 = max(0, min(src_height - 1, int(math.floor(src_y))))
        y1 = max(0, min(src_height - 1, y0 + 1))
        wy = max(0.0, min(1.0, src_y - float(y0)))
        for x in range(dst_width):
            src_x = ((float(x) + 0.5) * scale_x) - 0.5
            x0 = max(0, min(src_width - 1, int(math.floor(src_x))))
            x1 = max(0, min(src_width - 1, x0 + 1))
            wx = max(0.0, min(1.0, src_x - float(x0)))

            def _pix(px: int, py: int) -> tuple[float, float, float]:
                base = (py * src_width + px) * 3
                return (
                    float(src_rgb[base]),
                    float(src_rgb[base + 1]),
                    float(src_rgb[base + 2]),
                )

            p00 = _pix(x0, y0)
            p10 = _pix(x1, y0)
            p01 = _pix(x0, y1)
            p11 = _pix(x1, y1)

            def _blend(c00: float, c10: float, c01: float, c11: float) -> int:
                top = c00 * (1.0 - wx) + c10 * wx
                bot = c01 * (1.0 - wx) + c11 * wx
                value = top * (1.0 - wy) + bot * wy
                return int(max(0, min(255, round(value))))

            out.append(
                (
                    _blend(p00[0], p10[0], p01[0], p11[0]),
                    _blend(p00[1], p10[1], p01[1], p11[1]),
                    _blend(p00[2], p10[2], p01[2], p11[2]),
                )
            )
    return out


def _qwen3vl_geometry_overrides(config: dict[str, Any], image_path: Path) -> dict[str, Any]:
    source_width, source_height = _image_source_size(image_path)
    patch_size = int(config.get("patch_size", 0) or 0)
    merge_size = int(config.get("spatial_merge_size", 2) or 2)
    align_size = patch_size * merge_size
    if patch_size <= 0 or align_size <= 0:
        raise RuntimeError(f"invalid Qwen3-VL patch/merge config patch={patch_size} merge={merge_size}")
    patch_area = patch_size * patch_size * merge_size * merge_size
    min_pixels = int(config.get("image_min_pixels", 8 * patch_area) or (8 * patch_area))
    max_pixels = int(config.get("image_max_pixels", 4096 * patch_area) or (4096 * patch_area))
    image_width, image_height = _calc_qwen_vl_smart_resize(
        int(source_width),
        int(source_height),
        int(align_size),
        int(min_pixels),
        int(max_pixels),
    )
    if image_width % patch_size != 0 or image_height % patch_size != 0:
        raise RuntimeError(
            f"Qwen3-VL smart-resize produced non-divisible size {image_width}x{image_height} for patch_size={patch_size}"
        )
    vision_grid_w = image_width // patch_size
    vision_grid_h = image_height // patch_size
    if vision_grid_w % merge_size != 0 or vision_grid_h % merge_size != 0:
        raise RuntimeError(
            f"Qwen3-VL grid not divisible by merge_size: grid={vision_grid_w}x{vision_grid_h} merge={merge_size}"
        )
    vision_num_patches = vision_grid_w * vision_grid_h
    spatial_merge_factor = merge_size * merge_size
    vision_merged_tokens = vision_num_patches // spatial_merge_factor
    merged_grid_x = vision_grid_w // merge_size
    merged_grid_y = vision_grid_h // merge_size
    return {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "vision_grid_w": int(vision_grid_w),
        "vision_grid_h": int(vision_grid_h),
        "vision_num_patches": int(vision_num_patches),
        "vision_merged_tokens": int(vision_merged_tokens),
        "context_length": int(vision_num_patches),
        "max_seq_len": int(vision_num_patches),
        "merged_grid_x": int(merged_grid_x),
        "merged_grid_y": int(merged_grid_y),
        "image_source_width": int(source_width),
        "image_source_height": int(source_height),
    }


def _compile_generated_model(c_path: Path, so_path: Path) -> Path:
    stamp_path = so_path.with_suffix(so_path.suffix + ".build.json")
    source_hash = hashlib.sha256(c_path.read_bytes()).hexdigest()
    source_size = int(c_path.stat().st_size)
    build_fingerprint = {
        "version": 1,
        "source_path": str(c_path.resolve()),
        "source_sha256": source_hash,
        "source_size": source_size,
    }
    if so_path.exists():
        cached = _json_read(stamp_path)
        if cached == build_fingerprint:
            return so_path
    cmd = [
        "cc",
        "-shared",
        "-fPIC",
        "-O3",
        "-fopenmp",
        "-Iinclude",
        "-Iversion/v8/src",
        str(c_path),
        "version/v8/src/ckernel_model_load_v8.c",
        "version/v8/src/ck_parallel_decode_v8.c",
        "version/v8/src/ck_parallel_prefill_v8.c",
        "-Lbuild",
        "-lckernel_engine",
        f"-Wl,-rpath,{BUILD_DIR}",
        "-o",
        str(so_path),
        "-lm",
        "-lpthread",
    ]
    _run(cmd)
    _json_write(stamp_path, build_fingerprint)
    return so_path


def _load_layout(layout_path: Path) -> dict[str, Any]:
    with layout_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_activation_offsets(layout_path: Path) -> dict[str, dict[str, Any]]:
    layout = _load_layout(layout_path)
    return {
        str(buf["name"]): buf
        for buf in layout["memory"]["activations"]["buffers"]
    }


def _activation_runtime_base(layout: dict[str, Any]) -> int:
    weights = layout.get("memory", {}).get("weights", {})
    return int(weights.get("base_offset", 0)) + int(weights.get("size", 0))


def _activation_runtime_offset(layout: dict[str, Any], buf: dict[str, Any]) -> int:
    return _activation_runtime_base(layout) + int(buf["offset"])


def _buffer_nbytes(buf: dict[str, Any]) -> int:
    return int(buf.get("size_bytes", buf.get("size", 0)))


def _build_test_image(height: int, width: int, mode: str) -> tuple[list[float], list[float]]:
    interleaved = [0.0] * (height * width * 3)
    planar = [0.0] * (height * width * 3)
    for y in range(height):
        yf = y / max(1, height - 1)
        for x in range(width):
            xf = x / max(1, width - 1)
            idx = y * width + x
            if mode == "gray":
                rgb = (0.5, 0.5, 0.5)
            elif mode == "checker":
                v = 0.8 if ((x // 32) + (y // 32)) % 2 == 0 else 0.2
                rgb = (v, 1.0 - v * 0.5, 0.3 + 0.4 * yf)
            else:
                rgb = (
                    0.15 + 0.7 * xf,
                    0.10 + 0.6 * yf,
                    0.05 + 0.45 * (0.5 * xf + 0.5 * yf),
                )
            base_i = idx * 3
            interleaved[base_i + 0] = rgb[0]
            interleaved[base_i + 1] = rgb[1]
            interleaved[base_i + 2] = rgb[2]
            planar[idx] = rgb[0]
            planar[height * width + idx] = rgb[1]
            planar[2 * height * width + idx] = rgb[2]
    return interleaved, planar


def _load_image_file(
    image_path: Path,
    height: int,
    width: int,
    *,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"image file not found: {image_path}")
    mean = _coerce_float_triplet(image_mean, [0.5, 0.5, 0.5])
    std = _coerce_float_triplet(image_std, [0.5, 0.5, 0.5])

    suffix = image_path.suffix.lower()
    if suffix == ".ppm":
        source_width, source_height, src_rgb = _read_ppm_rgb8(image_path)
        pixels = _resize_rgb8_bilinear(src_rgb, source_width, source_height, width, height)
        preprocess_prefix = "ppm_rgb_bilinear_resize"
    else:
        if Image is None:
            raise RuntimeError("Pillow is required for non-PPM --image-path support")
        with Image.open(image_path) as src:
            source_width, source_height = src.size
            rgb = src.convert("RGB")
            if rgb.size != (width, height):
                if hasattr(Image, "Resampling"):
                    rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
                else:  # pragma: no cover - compatibility with older Pillow.
                    rgb = rgb.resize((width, height), Image.BILINEAR)
            pixels = list(rgb.getdata())
        preprocess_prefix = "rgb_bilinear_resize"

    interleaved = [0.0] * (height * width * 3)
    planar = [0.0] * (height * width * 3)
    for idx, (r, g, b) in enumerate(pixels):
        rf = (float(r) / 255.0 - mean[0]) / max(std[0], 1.0e-12)
        gf = (float(g) / 255.0 - mean[1]) / max(std[1], 1.0e-12)
        bf = (float(b) / 255.0 - mean[2]) / max(std[2], 1.0e-12)
        base_i = idx * 3
        interleaved[base_i + 0] = rf
        interleaved[base_i + 1] = gf
        interleaved[base_i + 2] = bf
        planar[idx] = rf
        planar[height * width + idx] = gf
        planar[2 * height * width + idx] = bf
    return {
        "interleaved": interleaved,
        "planar": planar,
        "image_source": "file",
        "image_path": str(image_path.resolve()),
        "source_image_size": [source_width, source_height],
        "preprocess": f"{preprocess_prefix}_{width}x{height}_normalize_mean_std",
    }


def _prepare_encoder_runtime(
    gguf_path: Path,
    output_dir: Path,
    image_path: Path | None = None,
    *,
    activation_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest, manifest_path, bump_path, config_path = _run_converter(gguf_path, output_dir)
    config = dict(manifest.get("config", {}) or {})
    if str(config.get("model", config.get("arch", ""))).lower() == "qwen3_vl_vision":
        base_image = int(config.get("image_size", 0) or 0)
        config.setdefault("image_height", base_image)
        config.setdefault("image_width", base_image)
        if image_path is not None:
            config.update(_qwen3vl_geometry_overrides(config, image_path))
    config = _apply_activation_preference_overrides(config, activation_overrides)
    manifest["config"] = config
    _json_write(manifest_path, manifest)
    _json_write(config_path, config)
    layout_path = output_dir / "layout.json"
    call_path = output_dir / "call.json"
    lowered_path = output_dir / "lowered.json"
    ir1_path = output_dir / "ir1.json"
    manifest_map = output_dir / "weights_manifest.map"
    c_path = output_dir / "encoder_v8.c"
    so_path = output_dir / "libencoder_v8.so"
    runtime_stamp = output_dir / "encoder_runtime.cache.json"
    runtime_fingerprint = _runtime_fingerprint(
        manifest_path=manifest_path,
        mode="encoder_prefill",
    )
    reusable_outputs = [
        manifest_path,
        config_path,
        bump_path,
        layout_path,
        call_path,
        lowered_path,
        ir1_path,
        c_path,
    ]

    if _artifacts_match_fingerprint(runtime_stamp, runtime_fingerprint, reusable_outputs):
        _log_progress(f"encoder runtime reuse workdir={output_dir}")
        _compile_generated_model(c_path, so_path)
        layout = _load_layout(layout_path)
        return {
            "gguf": str(gguf_path),
            "manifest": manifest,
            "weights_bump": bump_path,
            "manifest_map": manifest_map,
            "config_path": config_path,
            "layout_path": layout_path,
            "c_path": c_path,
            "so_path": so_path,
            "embed_dim": int(layout.get("config", {}).get("embed_dim", 0)),
        }

    rc = build_ir_v8.main(
        [
            "--manifest",
            str(manifest_path),
            "--mode",
            "prefill",
            "--output",
            str(ir1_path),
            "--layout-output",
            str(layout_path),
            "--lowered-output",
            str(lowered_path),
            "--call-output",
            str(call_path),
            "--manifest-map-output",
            str(manifest_map),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"build_ir_v8 encoder failed with rc={rc}")

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SCRIPT_DIR / "codegen_v8.py"),
            "--ir",
            str(call_path),
            "--layout",
            str(layout_path),
            "--output",
            str(c_path),
        ]
        codegen_rc = codegen_v8.main()
    finally:
        sys.argv = old_argv
    if codegen_rc != 0:
        raise RuntimeError(f"codegen_v8 encoder failed with rc={codegen_rc}")

    _compile_generated_model(c_path, so_path)
    _json_write(runtime_stamp, runtime_fingerprint)
    layout = _load_layout(layout_path)
    return {
        "gguf": str(gguf_path),
        "manifest": manifest,
        "weights_bump": bump_path,
        "manifest_map": manifest_map,
        "config_path": config_path,
        "layout_path": layout_path,
        "c_path": c_path,
        "so_path": so_path,
        "embed_dim": int(layout.get("config", {}).get("embed_dim", 0)),
    }


def _prepare_decoder_runtime(
    gguf_path: Path,
    output_dir: Path,
    parity_dump: bool = False,
    context_override: int | None = None,
) -> dict[str, Any]:
    manifest, manifest_path, bump_path, config_path = _run_converter(
        gguf_path,
        output_dir,
        context_override=context_override,
    )
    prefill_ir1 = output_dir / "ir1_prefill.json"
    prefill_layout = output_dir / "layout_prefill.json"
    prefill_lowered = output_dir / "lowered_prefill.json"
    prefill_call = output_dir / "call_prefill.json"
    decode_ir1 = output_dir / "ir1_decode.json"
    decode_layout = output_dir / "layout_decode.json"
    decode_lowered = output_dir / "lowered_decode.json"
    decode_call = output_dir / "call_decode.json"
    manifest_map = output_dir / "weights_manifest.map"
    suffix = "_parity_dump" if parity_dump else ""
    prefill_c_path = output_dir / f"decoder_v8_prefill{suffix}.c"
    prefill_so_path = output_dir / f"libdecoder_v8_prefill{suffix}.so"
    c_path = output_dir / f"decoder_v8{suffix}.c"
    so_path = output_dir / f"libdecoder_v8{suffix}.so"
    runtime_stamp = output_dir / f"decoder_runtime{suffix}.cache.json"
    runtime_fingerprint = _runtime_fingerprint(
        manifest_path=manifest_path,
        mode="decoder_hybrid",
        context_override=context_override,
        parity_dump=parity_dump,
    )
    reusable_outputs = [
        manifest_path,
        config_path,
        bump_path,
        prefill_ir1,
        prefill_layout,
        prefill_lowered,
        prefill_call,
        decode_ir1,
        decode_layout,
        decode_lowered,
        decode_call,
        manifest_map,
        prefill_c_path,
        c_path,
    ]

    if _artifacts_match_fingerprint(runtime_stamp, runtime_fingerprint, reusable_outputs):
        _log_progress(
            f"decoder runtime reuse workdir={output_dir} parity_dump={int(bool(parity_dump))}"
        )
        _compile_generated_model(c_path, so_path)
        _compile_generated_model(prefill_c_path, prefill_so_path)
        layout = _load_layout(decode_layout)
        if context_override is not None:
            decode_cfg = layout.get("config", {}) if isinstance(layout.get("config"), dict) else {}
            effective_context = int(decode_cfg.get("context_length", decode_cfg.get("context_len", 0)) or 0)
            if effective_context != int(context_override):
                raise RuntimeError(
                    "decoder runtime layout context mismatch: "
                    f"requested={int(context_override)} effective={effective_context}"
                )
        cfg = dict(layout.get("config", {}) or {})
        embed_dim = int(cfg.get("embed_dim", 0) or 0)
        num_deepstack_layers = int(cfg.get("num_deepstack_layers", 0) or 0)
        input_embed_dim = int(cfg.get("input_embed_dim", 0) or 0)
        if input_embed_dim <= 0 and embed_dim > 0 and num_deepstack_layers > 0:
            input_embed_dim = embed_dim * (1 + num_deepstack_layers)
        if input_embed_dim <= 0:
            input_embed_dim = embed_dim
        return {
            "gguf": str(gguf_path),
            "manifest": manifest,
            "weights_bump": bump_path,
            "manifest_map": manifest_map,
            "config_path": config_path,
            "prefill_layout_path": prefill_layout,
            "decode_layout_path": decode_layout,
            "prefill_c_path": prefill_c_path,
            "prefill_so_path": prefill_so_path,
            "c_path": c_path,
            "so_path": so_path,
            "embed_dim": embed_dim,
            "input_embed_dim": input_embed_dim,
            "num_deepstack_layers": num_deepstack_layers,
            "context_length": effective_context,
            "vocab_size": int(cfg.get("vocab_size", 0)),
        }

    prefill_args = [
        "--manifest",
        str(manifest_path),
        "--mode",
        "prefill",
        "--output",
        str(prefill_ir1),
        "--layout-output",
        str(prefill_layout),
        "--lowered-output",
        str(prefill_lowered),
        "--call-output",
        str(prefill_call),
    ]
    if context_override is not None:
        prefill_args.extend(["--context-len", str(int(context_override))])
    rc = build_ir_v8.main(prefill_args)
    if rc != 0:
        raise RuntimeError(f"build_ir_v8 decoder prefill failed with rc={rc}")

    decode_args = [
        "--manifest",
        str(manifest_path),
        "--mode",
        "decode",
        "--output",
        str(decode_ir1),
        "--layout-output",
        str(decode_layout),
        "--lowered-output",
        str(decode_lowered),
        "--call-output",
        str(decode_call),
        "--manifest-map-output",
        str(manifest_map),
    ]
    if context_override is not None:
        decode_args.extend(["--context-len", str(int(context_override))])
    rc = build_ir_v8.main(decode_args)
    if rc != 0:
        raise RuntimeError(f"build_ir_v8 decoder decode failed with rc={rc}")

    old_argv = sys.argv[:]
    try:
        # Build the staged decoder runtime on the decode layout. This keeps
        # multimodal replay on the faster decode path and allows decode-only KV
        # cache contracts to evolve without coupling them to prefill layout.
        sys.argv = [
            str(SCRIPT_DIR / "codegen_v8.py"),
            "--ir",
            str(decode_call),
            "--prefill",
            str(prefill_call),
            "--layout",
            str(decode_layout),
            "--output",
            str(c_path),
        ]
        if parity_dump:
            sys.argv.append("--parity-dump")
        codegen_rc = codegen_v8.main()
    finally:
        sys.argv = old_argv
    if codegen_rc != 0:
        raise RuntimeError(f"codegen_v8 decoder failed with rc={codegen_rc}")

    _compile_generated_model(c_path, so_path)
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SCRIPT_DIR / "codegen_v8.py"),
            "--ir",
            str(prefill_call),
            "--layout",
            str(prefill_layout),
            "--output",
            str(prefill_c_path),
        ]
        if parity_dump:
            sys.argv.append("--parity-dump")
        prefill_codegen_rc = codegen_v8.main()
    finally:
        sys.argv = old_argv
    if prefill_codegen_rc != 0:
        raise RuntimeError(f"codegen_v8 decoder prefill bridge failed with rc={prefill_codegen_rc}")

    _compile_generated_model(prefill_c_path, prefill_so_path)
    _json_write(runtime_stamp, runtime_fingerprint)
    layout = _load_layout(decode_layout)
    if context_override is not None:
        decode_cfg = layout.get("config", {}) if isinstance(layout.get("config"), dict) else {}
        effective_context = int(decode_cfg.get("context_length", decode_cfg.get("context_len", 0)) or 0)
        if effective_context != int(context_override):
            raise RuntimeError(
                "decoder runtime layout context mismatch: "
                f"requested={int(context_override)} effective={effective_context}"
            )
    cfg = dict(layout.get("config", {}) or {})
    embed_dim = int(cfg.get("embed_dim", 0) or 0)
    num_deepstack_layers = int(cfg.get("num_deepstack_layers", 0) or 0)
    input_embed_dim = int(cfg.get("input_embed_dim", 0) or 0)
    if input_embed_dim <= 0 and embed_dim > 0 and num_deepstack_layers > 0:
        input_embed_dim = embed_dim * (1 + num_deepstack_layers)
    if input_embed_dim <= 0:
        input_embed_dim = embed_dim
    return {
        "gguf": str(gguf_path),
        "manifest": manifest,
        "weights_bump": bump_path,
        "manifest_map": manifest_map,
        "config_path": config_path,
        "prefill_layout_path": prefill_layout,
        "decode_layout_path": decode_layout,
        "prefill_c_path": prefill_c_path,
        "prefill_so_path": prefill_so_path,
        "c_path": c_path,
        "so_path": so_path,
        "embed_dim": embed_dim,
        "input_embed_dim": input_embed_dim,
        "num_deepstack_layers": num_deepstack_layers,
        "context_length": int(cfg.get("context_length", cfg.get("context_len", 0)) or 0),
        "vocab_size": int(cfg.get("vocab_size", 0)),
    }


def _load_encoder_lib(model_so: Path) -> ctypes.CDLL:
    ctypes.CDLL(str(BUILD_DIR / "libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(model_so))
    lib.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.ck_model_init_with_manifest.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.c_void_p]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_get_base_ptr.argtypes = []
    lib.ck_model_get_base_ptr.restype = ctypes.c_uint64
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None
    declare_named_activation_api(lib)
    return lib


def _load_decoder_lib(model_so: Path) -> ctypes.CDLL:
    ctypes.CDLL(str(BUILD_DIR / "libckernel_engine.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(model_so))
    lib.ck_model_init_with_manifest.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.ck_model_init_with_manifest.restype = ctypes.c_int
    lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
    lib.ck_model_decode.restype = ctypes.c_int
    lib.ck_model_forward_mixed.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ck_model_forward_mixed.restype = ctypes.c_int
    try:
        lib.ck_model_forward_mixed_ex.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.ck_model_forward_mixed_ex.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        lib.ck_model_forward_mixed_grid_ex.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.ck_model_forward_mixed_grid_ex.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        lib.ck_model_forward_segments_grid_ex.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.ck_model_forward_segments_grid_ex.restype = ctypes.c_int
    except AttributeError:
        pass
    lib.ck_model_get_vocab_size.argtypes = []
    lib.ck_model_get_vocab_size.restype = ctypes.c_int
    lib.ck_set_strict_parity.argtypes = [ctypes.c_int]
    lib.ck_set_strict_parity.restype = None
    lib.ck_model_free.argtypes = []
    lib.ck_model_free.restype = None
    declare_named_activation_api(lib)
    return lib


def _run_encoder(runtime: dict[str, Any], image_mode: str, image_path: Path | None = None) -> dict[str, Any]:
    lib = _load_encoder_lib(runtime["so_path"])
    _log_progress("encoder: init start")
    rc = lib.ck_model_init_with_manifest(
        str(runtime["weights_bump"]).encode(),
        str(runtime["manifest_map"]).encode(),
    )
    if rc != 0:
        raise RuntimeError(f"encoder init failed with rc={rc}")
    try:
        _log_progress("encoder: init done")
        layout = _load_layout(runtime["layout_path"])
        offsets = _load_activation_offsets(runtime["layout_path"])
        bridge = resolve_vision_bridge_contract(layout, offsets, prefer_total_output=True)
        image_buf = offsets["image_input"]
        base_ptr = int(lib.ck_model_get_base_ptr())
        if base_ptr == 0:
            raise RuntimeError("encoder base ptr is null")
        layout_cfg = dict(layout.get("config", {}) or {})
        image_height = int(layout_cfg.get("image_height", layout_cfg.get("image_size", 0)) or 0)
        image_width = int(layout_cfg.get("image_width", layout_cfg.get("image_size", 0)) or 0)
        if image_height <= 0 or image_width <= 0:
            raise RuntimeError("encoder image dimensions missing from layout")
        image_mean = _coerce_float_triplet(layout_cfg.get("image_mean"), [0.5, 0.5, 0.5])
        image_std = _coerce_float_triplet(layout_cfg.get("image_std"), [0.5, 0.5, 0.5])
        merge_size = int(layout_cfg.get("spatial_merge_size", 1) or 1)
        merged_grid_x = int(layout_cfg.get("merged_grid_x", 0) or 0)
        merged_grid_y = int(layout_cfg.get("merged_grid_y", 0) or 0)

        if image_path is not None:
            image_report = _load_image_file(
                image_path,
                image_height,
                image_width,
                image_mean=image_mean,
                image_std=image_std,
            )
            interleaved = image_report["interleaved"]
            planar = image_report["planar"]
        else:
            interleaved, planar = _build_test_image(image_height, image_width, image_mode)
            image_report = {
                "image_source": "synthetic",
                "image_mode": image_mode,
                "image_path": None,
                "source_image_size": [image_width, image_height],
                "preprocess": "synthetic_generator",
            }
        image_len = _buffer_nbytes(image_buf) // ctypes.sizeof(ctypes.c_float)
        if len(planar) != image_len:
            raise RuntimeError(f"encoder planar image length mismatch: {len(planar)} != {image_len}")

        image_arr = (ctypes.c_float * image_len).from_address(
            base_ptr + _activation_runtime_offset(layout, image_buf)
        )
        image_arr[:] = planar
        _log_progress(
            f"encoder: decode start image={image_width}x{image_height} prefix_activation={bridge.get('fallback_buffer_name')}"
        )
        decode_t0 = time.perf_counter()
        rc = lib.ck_model_decode(0, None)
        if rc != 0:
            raise RuntimeError(f"encoder decode failed with rc={rc}")
        _log_progress(f"encoder: decode done elapsed={time.perf_counter() - decode_t0:.2f}s")

        embed_dim = int(bridge["embed_dim"])
        if embed_dim <= 0:
            raise RuntimeError("encoder bridge embed_dim is not available")

        output_ptr = 0
        output_nbytes = 0
        bridge_name = str(bridge.get("named_activation") or "")
        if bridge_name:
            named_view = try_named_activation_view(lib, bridge_name)
            if named_view is not None:
                output_ptr, output_nbytes = named_view
        if output_ptr == 0 or output_nbytes <= 0:
            output_buf = offsets[str(bridge["fallback_buffer_name"])]
            output_ptr = base_ptr + _activation_runtime_offset(layout, output_buf)
            output_nbytes = int(bridge["used_nbytes"])

        output_len = output_nbytes // ctypes.sizeof(ctypes.c_float)
        if output_ptr == 0 or output_len <= 0 or output_len % embed_dim != 0:
            raise RuntimeError(f"invalid encoder bridge output: output_len={output_len} embed_dim={embed_dim}")
        output_arr = (ctypes.c_float * output_len).from_address(output_ptr)
        return {
            "embed_dim": embed_dim,
            "prefix_tokens": output_len // embed_dim,
            "embeddings": array("f", output_arr),
            "prefix_grid_x": int(merged_grid_x or max(1, int(layout_cfg.get("vision_grid_w", 0) or 0) // max(1, merge_size))),
            "prefix_grid_y": int(merged_grid_y or max(1, int(layout_cfg.get("vision_grid_h", 0) or 0) // max(1, merge_size))),
            "prefix_text_pos": int(max(
                int(merged_grid_x or max(1, int(layout_cfg.get("vision_grid_w", 0) or 0) // max(1, merge_size))),
                int(merged_grid_y or max(1, int(layout_cfg.get("vision_grid_h", 0) or 0) // max(1, merge_size))),
            )),
            "bridge_activation": bridge_name or str(bridge["fallback_buffer_name"]),
            "bridge_reason": str(bridge["reason"]),
            "image_source": str(image_report["image_source"]),
            "image_mode": image_report.get("image_mode"),
            "image_path": image_report.get("image_path"),
            "source_image_size": image_report.get("source_image_size"),
            "preprocess": str(image_report["preprocess"]),
            "image_height": image_height,
            "image_width": image_width,
            "interleaved_image": interleaved,
        }
    finally:
        lib.ck_model_free()


def _run_decoder(
    runtime: dict[str, Any],
    prefix_embeddings: array,
    prefix_tokens: int,
    token_ids: list[int],
    *,
    tokens_before: list[int] | None = None,
    prefix_embed_dim: int | None = None,
    prefix_grid: tuple[int, int] | None = None,
    prefix_text_pos: int | None = None,
    strict_parity: bool = False,
    tokenizer: GGUFTokenizer | Any | None = None,
    stop_token_ids: list[int] | None = None,
    max_tokens: int = 0,
    temperature: float = 0.0,
    sample_top_k: int = 40,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repeat_penalty: float = 1.0,
    repeat_last_n: int = 64,
    min_response_tokens: int = 0,
    stream_output: bool = False,
    generation_progress_every: int = 0,
) -> dict[str, Any]:
    # Keep mixed prefix replay on the decode-layout runtime.
    #
    # The standalone prefill runtime is useful for batched prefill entrypoints,
    # but its generated ck_decode() still follows the prefill layout and can
    # replay full-sequence GEMM/transposes when used for continuation. That
    # breaks token 2+ after an external prefix. The decode-layout runtime's
    # ck_model_forward_mixed already stages prefix rows through the true
    # single-token decode path, which is the correctness-first contract we need
    # for multimodal bridge parity.
    model_so = Path(runtime["so_path"])
    lib = _load_decoder_lib(model_so)
    _log_progress("decoder: init start")
    rc = lib.ck_model_init_with_manifest(
        str(runtime["weights_bump"]).encode(),
        str(runtime["manifest_map"]).encode(),
    )
    if rc != 0:
        raise RuntimeError(f"decoder init failed with rc={rc}")
    try:
        _log_progress("decoder: init done")
        if hasattr(lib, "ck_set_strict_parity"):
            lib.ck_set_strict_parity(1 if strict_parity else 0)
        vocab_size = int(lib.ck_model_get_vocab_size())
        if vocab_size <= 0:
            vocab_size = int(runtime["vocab_size"])
        logits = (ctypes.c_float * vocab_size)()
        before_token_ids = [int(tok) for tok in list(tokens_before or [])]
        after_token_ids = [int(tok) for tok in list(token_ids or [])]
        resolved_prefix_dim = int(prefix_embed_dim or runtime.get("embed_dim", 0) or 0)
        prefix_ptr: ctypes.Array[ctypes.c_float] | None
        if prefix_tokens > 0:
            if resolved_prefix_dim <= 0 and len(prefix_embeddings) % prefix_tokens == 0:
                resolved_prefix_dim = len(prefix_embeddings) // prefix_tokens
            if resolved_prefix_dim <= 0:
                raise RuntimeError("decoder prefix embed_dim must be positive when prefix rows are present")
            expected = prefix_tokens * resolved_prefix_dim
            if len(prefix_embeddings) != expected:
                raise RuntimeError(
                    f"prefix float count mismatch: got={len(prefix_embeddings)} expected={expected} "
                    f"(tokens={prefix_tokens} row_dim={resolved_prefix_dim})"
                )
            prefix_ptr = (ctypes.c_float * len(prefix_embeddings))(*prefix_embeddings)
        else:
            prefix_ptr = None
        before_token_arr = (ctypes.c_int32 * len(before_token_ids))(*before_token_ids) if before_token_ids else None
        after_token_arr = (ctypes.c_int32 * len(after_token_ids))(*after_token_ids) if after_token_ids else None
        _log_progress(
            "decoder: forward_mixed start "
            f"prefix_tokens={prefix_tokens} prefix_dim={resolved_prefix_dim} "
            f"prompt_tokens_before={len(before_token_ids)} prompt_tokens_after={len(after_token_ids)} "
            f"grid={prefix_grid}"
        )
        forward_t0 = time.perf_counter()
        if before_token_ids and hasattr(lib, "ck_model_forward_segments_grid_ex"):
            grid_x, grid_y = prefix_grid if prefix_grid is not None else (0, 0)
            default_text_pos = (
                len(before_token_ids) + max(int(grid_x), int(grid_y))
                if prefix_grid is not None
                else len(before_token_ids) + max(0, int(prefix_tokens))
            )
            resolved_text_pos = int(prefix_text_pos or default_text_pos)
            rc = lib.ck_model_forward_segments_grid_ex(
                before_token_arr,
                len(before_token_ids),
                prefix_ptr,
                prefix_tokens,
                resolved_prefix_dim,
                int(grid_x),
                int(grid_y),
                resolved_text_pos,
                after_token_arr,
                len(after_token_ids),
                logits,
            )
        elif hasattr(lib, "ck_model_forward_mixed_grid_ex") and prefix_grid is not None:
            grid_x, grid_y = prefix_grid
            resolved_text_pos = int(prefix_text_pos or max(int(grid_x), int(grid_y), int(prefix_tokens)))
            rc = lib.ck_model_forward_mixed_grid_ex(
                prefix_ptr,
                prefix_tokens,
                resolved_prefix_dim,
                int(grid_x),
                int(grid_y),
                resolved_text_pos,
                after_token_arr,
                len(after_token_ids),
                logits,
            )
        elif hasattr(lib, "ck_model_forward_mixed_ex"):
            rc = lib.ck_model_forward_mixed_ex(prefix_ptr, prefix_tokens, resolved_prefix_dim, after_token_arr, len(after_token_ids), logits)
        else:
            rc = lib.ck_model_forward_mixed(prefix_ptr, prefix_tokens, after_token_arr, len(after_token_ids), logits)
        if rc != 0:
            raise RuntimeError(f"decoder forward_mixed failed with rc={rc}")
        _log_progress(f"decoder: forward_mixed done elapsed={time.perf_counter() - forward_t0:.2f}s")
        logits_arr = array("f", logits)
        stop_ids = {int(token_id) for token_id in list(stop_token_ids or [])}
        generated_token_ids: list[int] = []
        generation_stop_reason = "disabled"
        if max_tokens > 0:
            runtime_context_len = int(runtime.get("context_length", 0) or 0)
            prefill_tokens_total = int(len(before_token_ids)) + int(prefix_tokens) + int(len(after_token_ids))
            available_generation_budget = max(0, runtime_context_len - prefill_tokens_total) if runtime_context_len > 0 else int(max_tokens)
            effective_max_tokens = min(int(max_tokens), int(available_generation_budget))
            if effective_max_tokens < int(max_tokens):
                _log_progress(
                    "decoder: generation clamp "
                    f"requested={int(max_tokens)} effective={int(effective_max_tokens)} "
                    f"context={runtime_context_len} prefill_tokens={prefill_tokens_total}"
                )
            _log_progress(
                "decoder: generation start "
                f"max_tokens={int(effective_max_tokens)} stop_ids={sorted(stop_ids)} "
                f"temp={float(temperature):.3f} sample_top_k={int(sample_top_k)} "
                f"top_p={float(top_p):.3f} min_p={float(min_p):.3f} "
                f"repeat_penalty={float(repeat_penalty):.3f} repeat_last_n={int(repeat_last_n)}"
            )
            gen_t0 = time.perf_counter()
            generation_stop_reason = "max_tokens"
            current_logits: array | ctypes.Array[Any] = logits_arr
            streamed_text = ""
            for step in range(int(effective_max_tokens)):
                banned_ids = stop_ids if len(generated_token_ids) < int(min_response_tokens) else set()
                next_token = _sample_next_token(
                    current_logits,
                    temperature=float(temperature),
                    sample_top_k=int(sample_top_k),
                    top_p=float(top_p),
                    min_p=float(min_p),
                    recent_tokens=generated_token_ids,
                    repeat_penalty=float(repeat_penalty),
                    repeat_last_n=int(repeat_last_n),
                    banned_ids=banned_ids,
                )
                if int(next_token) in stop_ids:
                    generation_stop_reason = "stop_token"
                    break
                generated_token_ids.append(int(next_token))
                if stream_output:
                    streamed_text = _emit_stream_delta(tokenizer, generated_token_ids, streamed_text)
                if int(generation_progress_every) > 0 and (step + 1) % int(generation_progress_every) == 0:
                    _log_progress(f"decoder: generation progress tokens={len(generated_token_ids)}")
                if step + 1 >= int(effective_max_tokens):
                    break
                rc = lib.ck_model_decode(ctypes.c_int32(int(next_token)), logits)
                if rc != 0:
                    raise RuntimeError(f"decoder decode failed with rc={rc} at generated_step={step}")
                current_logits = logits
            if stream_output and generated_token_ids:
                print("", flush=True)
            _log_progress(
                "decoder: generation done "
                f"elapsed={time.perf_counter() - gen_t0:.2f}s generated_tokens={len(generated_token_ids)} "
                f"stop={generation_stop_reason}"
            )
        generated_text = ""
        generated_text_raw = ""
        if tokenizer is not None and generated_token_ids:
            generated_text = str(tokenizer.decode(generated_token_ids, skip_special=True))
            generated_text_raw = str(tokenizer.decode(generated_token_ids, skip_special=False))
        return {
            "vocab_size": vocab_size,
            "logits": logits_arr,
            "runtime_mode": "decode",
            "generated_token_ids": generated_token_ids,
            "generated_text": generated_text,
            "generated_text_raw": generated_text_raw,
            "generation_stop_reason": generation_stop_reason,
            "streamed_output": bool(stream_output and tokenizer is not None and max_tokens > 0),
        }
    finally:
        if hasattr(lib, "ck_set_strict_parity"):
            lib.ck_set_strict_parity(0)
        lib.ck_model_free()


def _topk(values: array, k: int) -> list[tuple[int, float]]:
    return heapq.nlargest(k, enumerate(values), key=lambda item: float(item[1]))


def _argmax(values: array | ctypes.Array[Any]) -> int:
    if len(values) <= 0:
        raise ValueError("cannot argmax empty logits")
    best_idx = 0
    best_val = float(values[0])
    for idx in range(1, len(values)):
        cur = float(values[idx])
        if cur > best_val:
            best_idx = idx
            best_val = cur
    return int(best_idx)


def _masked_argmax(values: array | ctypes.Array[Any] | list[float], banned_ids: set[int] | None = None) -> int:
    if len(values) <= 0:
        raise ValueError("cannot argmax empty logits")
    banned = banned_ids or set()
    best_idx = -1
    best_val = float("-inf")
    for idx in range(len(values)):
        if idx in banned:
            continue
        cur = float(values[idx])
        if best_idx < 0 or cur > best_val:
            best_idx = idx
            best_val = cur
    if best_idx >= 0:
        return int(best_idx)
    return _argmax(values)


def _sample_next_token(
    logits: array | ctypes.Array[Any],
    *,
    temperature: float,
    sample_top_k: int,
    top_p: float,
    min_p: float,
    recent_tokens: list[int] | None = None,
    repeat_penalty: float = 1.0,
    repeat_last_n: int = 64,
    banned_ids: set[int] | None = None,
) -> int:
    if len(logits) <= 0:
        raise ValueError("cannot sample empty logits")

    adjusted = [float(v) for v in logits]
    banned = banned_ids or set()

    history = list(recent_tokens or [])
    if repeat_penalty > 1.0 and history:
        if repeat_last_n > 0:
            history = history[-int(repeat_last_n):]
        for token_id in set(int(tok) for tok in history):
            if 0 <= token_id < len(adjusted):
                value = adjusted[token_id]
                adjusted[token_id] = value / repeat_penalty if value > 0.0 else value * repeat_penalty

    if temperature <= 0.0:
        return _masked_argmax(adjusted, banned)

    live_indices = [idx for idx in range(len(adjusted)) if idx not in banned]
    if not live_indices:
        return _argmax(logits)

    max_logit = max(adjusted[idx] for idx in live_indices)
    probs: list[tuple[int, float]] = []
    inv_temp = 1.0 / max(float(temperature), 1.0e-6)
    for idx in live_indices:
        probs.append((idx, math.exp((adjusted[idx] - max_logit) * inv_temp)))
    probs.sort(key=lambda row: row[1], reverse=True)

    if sample_top_k > 0 and sample_top_k < len(probs):
        probs = probs[: int(sample_top_k)]

    if probs and min_p > 0.0:
        threshold = probs[0][1] * float(min_p)
        filtered = [row for row in probs if row[1] >= threshold]
        if filtered:
            probs = filtered

    if probs and 0.0 < top_p < 1.0:
        total = sum(prob for _, prob in probs)
        if total > 0.0:
            nucleus: list[tuple[int, float]] = []
            cumulative = 0.0
            for token_id, prob in probs:
                nucleus.append((token_id, prob))
                cumulative += prob / total
                if cumulative >= top_p:
                    break
            if nucleus:
                probs = nucleus

    total = sum(prob for _, prob in probs)
    if total <= 0.0:
        return _masked_argmax(adjusted, banned)

    draw = random.random() * total
    cumulative = 0.0
    for token_id, prob in probs:
        cumulative += prob
        if cumulative >= draw:
            return int(token_id)
    return int(probs[-1][0])


def _emit_stream_delta(
    tokenizer: GGUFTokenizer | Any | None,
    generated_token_ids: list[int],
    rendered_text: str,
) -> str:
    if tokenizer is None or not generated_token_ids:
        return rendered_text
    current_text = str(tokenizer.decode(generated_token_ids, skip_special=True))
    if not current_text:
        return rendered_text
    if current_text.startswith(rendered_text):
        delta = current_text[len(rendered_text):]
    else:
        delta = current_text
    if delta:
        print(delta, end="", flush=True)
    return current_text


def _lookup_single_token_id(tokenizer: GGUFTokenizer | Any, text: str) -> int:
    value = str(text or "")
    if not value:
        return -1

    token_to_id = getattr(tokenizer, "token_to_id", None)
    if isinstance(token_to_id, dict) and value in token_to_id:
        try:
            return int(token_to_id[value])
        except Exception:
            pass

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:
            vocab = None
        if isinstance(vocab, dict) and value in vocab:
            try:
                return int(vocab[value])
            except Exception:
                pass

    lookup = getattr(tokenizer, "lookup_token_id", None)
    if callable(lookup):
        try:
            token_id = int(lookup(value))
        except Exception:
            token_id = -1
        if token_id >= 0:
            return token_id
    return -1


def _resolve_stop_token_policy(
    tokenizer: GGUFTokenizer | Any,
    chat_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    stop_ids: set[int] = set()
    stop_markers: list[str] = []

    eos_id = int(getattr(tokenizer, "eos_id", -1) or -1)
    if eos_id >= 0:
        stop_ids.add(eos_id)

    if isinstance(chat_contract, dict):
        for marker in list(chat_contract.get("token_stop_markers") or []):
            marker_text = str(marker or "")
            if not marker_text:
                continue
            token_id = _lookup_single_token_id(tokenizer, marker_text)
            if token_id >= 0:
                stop_ids.add(int(token_id))
                stop_markers.append(marker_text)

    return {
        "eos_id": eos_id,
        "stop_ids": sorted(stop_ids),
        "stop_markers": stop_markers,
    }


def _derive_decoder_context_len(
    prompt_token_count: int,
    prefix_tokens: int,
    requested: int | None = None,
    slack_tokens: int = 16,
    minimum_context: int = 32,
) -> int:
    needed = max(1, int(prompt_token_count) + max(0, int(prefix_tokens)))
    if requested is not None and int(requested) > 0:
        return max(int(requested), needed)
    return max(minimum_context, needed + max(1, int(slack_tokens)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Explicit v8 multimodal bridge runner")
    ap.add_argument("--decoder-gguf", type=Path, required=True, help="Decoder GGUF to lower/codegen")
    ap.add_argument("--encoder-gguf", type=Path, default=None, help="Optional vision encoder/mmproj GGUF")
    ap.add_argument("--workdir", type=Path, required=True, help="Artifact/output directory")
    ap.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt text for decoder tokenization")
    ap.add_argument("--chat-template", choices=["auto", "none", "qwen", "qwen2", "qwen3", "qwen35", "qwen3vl", "gemma", "gemma3"], default="auto")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--allow-raw-prompt", action="store_true", help="Acknowledge raw prompt formatting when chat templates are disabled")
    ap.add_argument("--thinking-mode", choices=["auto", "visible", "suppressed"], default="auto")
    ap.add_argument("--image-mode", choices=["checker", "gradient", "gray"], default="checker", help="Synthetic image generator to use when --image-path is not provided")
    ap.add_argument("--image-path", type=Path, default=None, help="Optional real image path for encoder input; overrides --image-mode")
    ap.add_argument("--synthetic-prefix-tokens", type=int, default=0, help="Use zero prefix embeddings when a real encoder bridge is unavailable")
    ap.add_argument("--decoder-context-len", type=int, default=None, help="Override decoder context length; default is prompt+prefix budget with small headroom")
    ap.add_argument("--dump-prefix-f32", type=Path, default=None, help="Optional output path for resolved float32 prefix embeddings")
    ap.add_argument("--max-tokens", type=int, default=0, help="Generate up to N tokens after multimodal prefill; 0 reports first-token logits only")
    ap.add_argument("--report-top-k", "--top-k", dest="report_top_k", type=int, default=8, help="How many top logits to report in bridge_report.json")
    ap.add_argument("--print-json-report", action="store_true", help="Print the full bridge_report.json payload to stdout instead of concise operator output")
    ap.add_argument("--generation-progress-every", type=int, default=0, help="Emit generation heartbeat every N tokens; 0 disables periodic progress logs")
    ap.add_argument("--no-stream-output", action="store_true", help="Disable live text streaming during generation and print only the final response")
    ap.add_argument("--temperature", type=float, default=0.0, help="Decode temperature; 0 keeps greedy generation")
    ap.add_argument("--sample-top-k", type=int, default=40, help="Top-k sampling cutoff used during bridge generation")
    ap.add_argument("--top-p", type=float, default=1.0, help="Nucleus top-p used during bridge generation")
    ap.add_argument("--min-p", type=float, default=0.0, help="Relative min-p floor used during bridge generation")
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="Penalty applied to recently generated tokens")
    ap.add_argument("--repeat-last-n", type=int, default=64, help="How many recent tokens the repetition penalty should inspect")
    ap.add_argument(
        "--vision-activation-pref",
        action="append",
        default=[],
        help="Optional vision encoder activation override(s) in op=dtype form, e.g. out_proj=q8",
    )
    args = ap.parse_args(argv)

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    encoder_dir = workdir / "encoder"
    decoder_dir = workdir / "decoder"
    vision_activation_overrides = _parse_activation_preference_overrides(args.vision_activation_pref)

    _log_progress(f"start workdir={workdir}")
    _ensure_engine_lib(openmp=args.encoder_gguf is not None)
    _log_progress(f"engine ready openmp={'on' if args.encoder_gguf is not None else 'off'}")

    tokenizer = GGUFTokenizer.from_gguf(str(args.decoder_gguf.resolve()))
    chat_template_mode = "none" if args.no_chat_template else args.chat_template
    chat_contract = _resolve_decoder_chat_contract(args.decoder_gguf.resolve(), chat_template_mode=chat_template_mode)
    include_image_chunks = bool(args.encoder_gguf is not None or int(args.synthetic_prefix_tokens) > 0)
    prompt_segments = _format_multimodal_prompt_segments(
        args.prompt,
        chat_contract,
        include_image=include_image_chunks,
        thinking_mode=args.thinking_mode,
    )
    formatted_prompt = str(prompt_segments["formatted_prompt"])
    prompt_prefix_token_ids = tokenizer.encode(str(prompt_segments["before_text"])) if str(prompt_segments["before_text"]) else []
    token_ids = tokenizer.encode(str(prompt_segments["after_text"])) if str(prompt_segments["after_text"]) else []
    total_text_prompt_tokens = len(prompt_prefix_token_ids) + len(token_ids)
    contract_name = str((chat_contract or {}).get("name") or "none")
    _log_progress(
        "tokenizer ready "
        f"prompt_tokens_before={len(prompt_prefix_token_ids)} prompt_tokens_after={len(token_ids)} "
        f"chat_template={chat_template_mode} contract={contract_name}"
    )

    prefix_source = "none"
    prefix_tokens = 0
    prefix_embeddings = array("f")
    prefix_grid: tuple[int, int] | None = None
    prefix_text_pos: int | None = None
    encoder_report: dict[str, Any] | None = None
    dim_mismatch: dict[str, int] | None = None

    if args.encoder_gguf is not None:
        _log_progress(f"encoder runtime prepare start gguf={args.encoder_gguf.resolve()}")
        encoder_prep_t0 = time.perf_counter()
        encoder_runtime = _prepare_encoder_runtime(
            args.encoder_gguf.resolve(),
            encoder_dir,
            image_path=args.image_path.resolve() if args.image_path is not None else None,
            activation_overrides=vision_activation_overrides,
        )
        _log_progress(f"encoder runtime prepare done elapsed={time.perf_counter() - encoder_prep_t0:.2f}s")
        _log_progress("encoder execution start")
        encoder_report = _run_encoder(
            encoder_runtime,
            args.image_mode,
            image_path=args.image_path.resolve() if args.image_path is not None else None,
        )
        _log_progress(
            "encoder execution done "
            f"prefix_tokens={int(encoder_report['prefix_tokens'])} embed_dim={int(encoder_report['embed_dim'])}"
        )

    decoder_prefix_budget = max(
        int(encoder_report["prefix_tokens"]) if encoder_report is not None else 0,
        max(0, int(args.synthetic_prefix_tokens)),
    )
    decoder_context_len = _derive_decoder_context_len(
        prompt_token_count=total_text_prompt_tokens,
        prefix_tokens=decoder_prefix_budget,
        requested=args.decoder_context_len,
        slack_tokens=max(16, int(args.max_tokens or 0)),
    )
    _log_progress(
        "decoder runtime prepare start "
        f"gguf={args.decoder_gguf.resolve()} context={decoder_context_len} prefix_budget={decoder_prefix_budget}"
    )
    decoder_prep_t0 = time.perf_counter()
    decoder_runtime = _prepare_decoder_runtime(
        args.decoder_gguf.resolve(),
        decoder_dir,
        context_override=decoder_context_len,
    )
    _log_progress(f"decoder runtime prepare done elapsed={time.perf_counter() - decoder_prep_t0:.2f}s")

    prefix_embed_dim = int(decoder_runtime["embed_dim"])
    if encoder_report is not None:
        encoder_embed_dim = int(encoder_report["embed_dim"])
        decoder_input_embed_dim = int(decoder_runtime.get("input_embed_dim", decoder_runtime["embed_dim"]))
        if encoder_embed_dim in {int(decoder_runtime["embed_dim"]), decoder_input_embed_dim}:
            prefix_source = "encoder"
            prefix_tokens = int(encoder_report["prefix_tokens"])
            prefix_embeddings = encoder_report["embeddings"]
            prefix_embed_dim = encoder_embed_dim
            grid_x = int(encoder_report.get("prefix_grid_x", 0) or 0)
            grid_y = int(encoder_report.get("prefix_grid_y", 0) or 0)
            if grid_x > 0 and grid_y > 0:
                prefix_grid = (grid_x, grid_y)
                local_prefix_text_pos = int(
                    encoder_report.get("prefix_text_pos", max(grid_x, grid_y)) or max(grid_x, grid_y)
                )
                prefix_text_pos = len(prompt_prefix_token_ids) + local_prefix_text_pos
        else:
            dim_mismatch = {
                "encoder_embed_dim": encoder_embed_dim,
                "decoder_embed_dim": int(decoder_runtime["embed_dim"]),
                "decoder_input_embed_dim": decoder_input_embed_dim,
            }

    if prefix_source != "encoder" and args.synthetic_prefix_tokens > 0:
        prefix_source = "synthetic_zero"
        prefix_tokens = args.synthetic_prefix_tokens
        prefix_embed_dim = int(decoder_runtime.get("input_embed_dim", decoder_runtime["embed_dim"]) or decoder_runtime["embed_dim"])
        prefix_embeddings = array("f", [0.0] * (prefix_tokens * prefix_embed_dim))
        side = int(math.isqrt(int(prefix_tokens)))
        if prefix_tokens > 0 and side > 0 and side * side == int(prefix_tokens):
            prefix_grid = (side, side)
            prefix_text_pos = len(prompt_prefix_token_ids) + side

    if prefix_source == "none":
        raise SystemExit(
            "No usable prefix source: encoder/decode dims do not match and no --synthetic-prefix-tokens was provided"
        )
    _log_progress(
        f"bridge ready prefix_source={prefix_source} prefix_tokens={prefix_tokens} prefix_dim={prefix_embed_dim}"
    )
    stop_policy = _resolve_stop_token_policy(tokenizer, chat_contract)
    min_response_tokens = max(0, int((chat_contract or {}).get("min_response_tokens", 0) or 0))

    dumped_prefix_path: str | None = None
    if args.dump_prefix_f32 is not None:
        dump_path = args.dump_prefix_f32.resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_bytes(prefix_embeddings.tobytes())
        dumped_prefix_path = str(dump_path)

    decoder_report = _run_decoder(
        decoder_runtime,
        prefix_embeddings,
        prefix_tokens,
        token_ids,
        tokens_before=prompt_prefix_token_ids,
        prefix_embed_dim=prefix_embed_dim,
        prefix_grid=prefix_grid,
        prefix_text_pos=prefix_text_pos,
        tokenizer=tokenizer,
        stop_token_ids=list(stop_policy["stop_ids"]),
        max_tokens=max(0, int(args.max_tokens)),
        temperature=float(args.temperature),
        sample_top_k=int(args.sample_top_k),
        top_p=float(args.top_p),
        min_p=float(args.min_p),
        repeat_penalty=float(args.repeat_penalty),
        repeat_last_n=int(args.repeat_last_n),
        min_response_tokens=min_response_tokens,
        stream_output=bool(args.max_tokens) and not bool(args.no_stream_output),
        generation_progress_every=int(args.generation_progress_every),
    )
    _log_progress("report assembly start")
    top = _topk(decoder_report["logits"], max(1, args.report_top_k))
    top_tokens = [
        {
            "token_id": int(tok_id),
            "logit": float(logit),
            "token_text": tokenizer.decode([int(tok_id)], skip_special=False),
        }
        for tok_id, logit in top
    ]

    report = {
        "status": "ok",
        "prefix_source": prefix_source,
        "prompt": args.prompt,
        "formatted_prompt": formatted_prompt,
        "chat_template_mode": chat_template_mode,
        "chat_contract_name": None if chat_contract is None else str(chat_contract.get("name") or ""),
        "eos_token_id": int(stop_policy["eos_id"]),
        "stop_token_ids": [int(tok) for tok in stop_policy["stop_ids"]],
        "stop_token_markers": [str(marker) for marker in stop_policy["stop_markers"]],
        "prompt_token_count": total_text_prompt_tokens,
        "prompt_tokens": [int(tok) for tok in (prompt_prefix_token_ids + token_ids)],
        "prompt_tokens_before_image": [int(tok) for tok in prompt_prefix_token_ids],
        "prompt_tokens_after_image": [int(tok) for tok in token_ids],
        "multimodal_prompt_segmented": bool(prompt_segments["uses_image_chunks"]),
        "decoder_embed_dim": int(decoder_runtime["embed_dim"]),
        "decoder_input_embed_dim": int(decoder_runtime.get("input_embed_dim", decoder_runtime["embed_dim"])),
        "decoder_context_len": int(decoder_context_len),
        "prefix_tokens": prefix_tokens,
        "prefix_embed_dim": int(prefix_embed_dim),
        "prefix_grid_x": None if prefix_grid is None else int(prefix_grid[0]),
        "prefix_grid_y": None if prefix_grid is None else int(prefix_grid[1]),
        "prefix_text_pos": None if prefix_text_pos is None else int(prefix_text_pos),
        "prefix_dump_path": dumped_prefix_path,
        "total_prefill_tokens": len(prompt_prefix_token_ids) + prefix_tokens + len(token_ids),
        "decoder_runtime": {
            "gguf": str(args.decoder_gguf.resolve()),
            "workdir": str(decoder_dir),
            "so_path": str(decoder_runtime["so_path"]),
            "c_path": str(decoder_runtime["c_path"]),
        },
        "encoder_runtime": {
            "gguf": str(args.encoder_gguf.resolve()),
            "workdir": str(encoder_dir),
        } if args.encoder_gguf is not None else None,
        "encoder_report": {
            "embed_dim": int(encoder_report["embed_dim"]),
            "prefix_tokens": int(encoder_report["prefix_tokens"]),
            "bridge_activation": str(encoder_report["bridge_activation"]),
            "bridge_reason": str(encoder_report["bridge_reason"]),
            "image_source": str(encoder_report["image_source"]),
            "image_mode": None if encoder_report["image_mode"] is None else str(encoder_report["image_mode"]),
            "image_path": encoder_report["image_path"],
            "source_image_size": encoder_report["source_image_size"],
            "preprocess": str(encoder_report["preprocess"]),
            "image_size": int(
                encoder_report.get(
                    "image_size",
                    max(
                        int(encoder_report.get("image_height", 0) or 0),
                        int(encoder_report.get("image_width", 0) or 0),
                    ),
                )
                or 0
            ),
            "image_height": int(encoder_report.get("image_height", 0) or 0),
            "image_width": int(encoder_report.get("image_width", 0) or 0),
            "prefix_grid_x": int(encoder_report.get("prefix_grid_x", 0) or 0),
            "prefix_grid_y": int(encoder_report.get("prefix_grid_y", 0) or 0),
            "prefix_text_pos": int(encoder_report.get("prefix_text_pos", 0) or 0),
            "activation_preference_overrides": dict(vision_activation_overrides),
        } if encoder_report is not None else None,
        "dim_mismatch": dim_mismatch,
        "top_logits": top_tokens,
        "generated_token_ids": [int(tok) for tok in decoder_report.get("generated_token_ids", [])],
        "generated_token_count": int(len(decoder_report.get("generated_token_ids", []) or [])),
        "generated_text": str(decoder_report.get("generated_text") or ""),
        "generated_text_raw": str(decoder_report.get("generated_text_raw") or ""),
        "generation_stop_reason": str(decoder_report.get("generation_stop_reason") or "disabled"),
        "generation_config": {
            "temperature": float(args.temperature),
            "sample_top_k": int(args.sample_top_k),
            "top_p": float(args.top_p),
            "min_p": float(args.min_p),
            "repeat_penalty": float(args.repeat_penalty),
            "repeat_last_n": int(args.repeat_last_n),
            "min_response_tokens": int(min_response_tokens),
        },
        "notes": [
            "This runner keeps the encoder->decoder bridge in orchestration instead of baking a multimodal special-case into templates.",
            "Synthetic-prefix mode only validates the decoder bridge seam; it is not a substitute for real multimodal parity.",
            "Real-image mode follows the Qwen3-VL smart-resize contract and replays explicit merged-grid positions into the decoder bridge.",
        ],
    }
    report_path = workdir / "bridge_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    full_network_graph = _build_full_network_graph(
        workdir=workdir,
        bridge_report=report,
        encoder_dir=encoder_dir,
        decoder_dir=decoder_dir,
    )
    full_network_path = workdir / "full_network_graph.json"
    _json_write(full_network_path, full_network_graph)
    _log_progress(f"done report={report_path}")
    if args.print_json_report:
        print(
            json.dumps(
                {
                    **report,
                    "full_network_graph": str(full_network_path),
                },
                indent=2,
            )
        )
    else:
        generated_text = str(report.get("generated_text") or "").strip()
        streamed_output = bool(decoder_report.get("streamed_output"))
        if generated_text and not streamed_output:
            print(generated_text)
        else:
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "top_logits": report["top_logits"],
                        "report_path": str(report_path),
                        "full_network_graph": str(full_network_path),
                    },
                    indent=2,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
