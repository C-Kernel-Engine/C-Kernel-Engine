#!/usr/bin/env python3
"""Shared chat-contract helpers for conversion and runtime chat formatting."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "version" / "v7" / "templates"

PRESET_ALIASES = {
    "qwen": "qwen2",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "qwen35": "qwen35",
    "gemma": "gemma3",
    "gemma3": "gemma3",
}

KNOWN_TEMPLATE_MARKERS = (
    "<start_of_turn>",
    "<end_of_turn>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
)


def extract_static_default_system_prompt(chat_template: str) -> Optional[str]:
    """Best-effort extraction of a static fallback system prompt from a Jinja template."""
    template = str(chat_template or "").replace("\\n", "\n")
    if not template:
        return None

    candidates: list[str] = []
    patterns = (
        r"<\|im_start\|>system\n(?P<prompt>.+?)<\|im_end\|>\n",
        r"<start_of_turn>system\n(?P<prompt>.+?)<end_of_turn>\n",
    )
    reject_substrings = (
        "{{",
        "{%",
        "messages[",
        "message.",
        "loop_messages",
        "tools",
        "tool_call",
        "bos_token",
        "render_content",
        "content +",
        "+ content",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, template, flags=re.S):
            prompt = str(match.group("prompt") or "").strip()
            if not prompt:
                continue
            if any(token in prompt for token in reject_substrings):
                continue
            if "{" in prompt or "}" in prompt or "+" in prompt:
                continue
            candidates.append(prompt)
    if not candidates:
        return None
    return min(candidates, key=len)


def looks_like_instruction_chat_model(
    *,
    chat_template: Optional[str] = None,
    finetune: Optional[str] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
) -> bool:
    template = str(chat_template or "")
    finetune_lc = str(finetune or "").lower()
    model_name_lc = str(model_name or "").lower()
    model_type_lc = str(model_type or "").lower()
    return (
        "<|im_start|>" in template
        or "<start_of_turn>" in template
        or "instruct" in finetune_lc
        or "chat" in finetune_lc
        or finetune_lc == "it"
        or "instruct" in model_name_lc
        or model_name_lc.endswith("-it")
        or model_type_lc.startswith("gemma")
        or model_type_lc.startswith("qwen")
    )


def normalize_chat_contract(contract: Any) -> Optional[dict[str, Any]]:
    if not isinstance(contract, dict):
        return None

    out = copy.deepcopy(contract)
    out.setdefault("version", 1)
    out.setdefault("name", "")
    out.setdefault("raw_prompt_allowed", False)
    out.setdefault("turn_prefix", "")
    out.setdefault("turn_suffix", "")
    out.setdefault("assistant_generation_prefix", "")
    out.setdefault("system_prompt_mode", "disabled")
    out.setdefault("system_prompt_separator", "\n\n")
    out.setdefault("default_system_prompt", "")
    out.setdefault("inject_default_system_prompt", False)
    out.setdefault("force_bos_text_if_tokenizer_add_bos_false", "")
    out.setdefault("last_user_prefix", "")
    out.setdefault("last_user_prefix_suppression_markers", [])
    out.setdefault("thinking_mode_default", "")
    out.setdefault("assistant_generation_prefix_by_thinking_mode", {})
    out.setdefault("last_user_prefix_by_thinking_mode", {})
    out.setdefault("stop_text_markers", [])
    out.setdefault("token_stop_markers", list(out.get("stop_text_markers") or []))
    out.setdefault("template_markers", [])
    out.setdefault("min_response_tokens", 0)
    role_labels = out.get("role_labels")
    if not isinstance(role_labels, dict):
        out["role_labels"] = {}
    for dict_key in (
        "assistant_generation_prefix_by_thinking_mode",
        "last_user_prefix_by_thinking_mode",
    ):
        value = out.get(dict_key)
        if not isinstance(value, dict):
            out[dict_key] = {}
    for list_key in (
        "last_user_prefix_suppression_markers",
        "stop_text_markers",
        "token_stop_markers",
        "template_markers",
    ):
        value = out.get(list_key)
        if not isinstance(value, list):
            out[list_key] = []
    try:
        out["min_response_tokens"] = max(0, int(out.get("min_response_tokens") or 0))
    except Exception:
        out["min_response_tokens"] = 0
    return out


def load_template_chat_contract(name: str) -> Optional[dict[str, Any]]:
    requested = str(name or "").strip().lower()
    alias = PRESET_ALIASES.get(requested)
    if not alias:
        return None
    path = TEMPLATE_DIR / f"{alias}.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    contract = doc.get("contract", {}).get("chat_contract")
    normalized = normalize_chat_contract(contract)
    if normalized is None:
        return None
    normalized["name"] = requested or str(normalized.get("name") or alias)
    return normalized


def hydrate_chat_contract(contract: Any) -> Optional[dict[str, Any]]:
    if not isinstance(contract, dict):
        return normalize_chat_contract(contract)
    base = load_template_chat_contract(str(contract.get("name") or ""))
    if base is None:
        return normalize_chat_contract(contract)
    merged = copy.deepcopy(base)
    merged.update(copy.deepcopy(contract))
    normalized = normalize_chat_contract(merged)
    if normalized is None:
        return None
    return normalized


def _collect_template_markers(chat_template: Optional[str]) -> list[str]:
    template = str(chat_template or "")
    markers: list[str] = []
    for marker in KNOWN_TEMPLATE_MARKERS:
        if marker in template:
            markers.append(marker)
    return markers


def _build_llama_chatml_contract(chat_template: Optional[str]) -> dict[str, Any]:
    markers = _collect_template_markers(chat_template)
    if "<|im_start|>" not in markers:
        markers.insert(0, "<|im_start|>")
    if "<|im_end|>" not in markers:
        insert_at = 1 if markers and markers[0] == "<|im_start|>" else len(markers)
        markers.insert(insert_at, "<|im_end|>")
    return {
        "version": 1,
        "name": "llama_chatml",
        "raw_prompt_allowed": False,
        "turn_prefix": "<|im_start|>{role}\n",
        "turn_suffix": "<|im_end|>\n",
        "assistant_generation_prefix": "<|im_start|>assistant\n",
        "role_labels": {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        },
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
        "template_markers": markers,
        "min_response_tokens": 8,
    }


def _infer_explicit_contract_name(
    *,
    chat_template: Optional[str],
    model_type: Optional[str],
    template_name: Optional[str],
) -> Optional[str]:
    template = str(chat_template or "")
    model_type_lc = str(model_type or "").lower()
    template_name_lc = str(template_name or "").lower()

    if "<start_of_turn>" in template and "<end_of_turn>" in template:
        return "gemma"
    if "<|im_start|>" in template and "<|im_end|>" in template:
        if model_type_lc == "qwen35" or template_name_lc == "qwen35":
            return "qwen35"
        if model_type_lc == "qwen3" or template_name_lc == "qwen3":
            return "qwen3"
        if model_type_lc == "qwen2" or template_name_lc == "qwen2":
            return "qwen2"
        if model_type_lc == "llama" or template_name_lc == "llama":
            return "llama_chatml"
        return "qwen"
    return None


def build_chat_contract(
    *,
    template_data: Optional[dict[str, Any]] = None,
    chat_template: Optional[str] = None,
    finetune: Optional[str] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    template_name = ""
    if isinstance(template_data, dict):
        template_name = str(template_data.get("name") or "")

    contract = normalize_chat_contract(
        (template_data or {}).get("contract", {}).get("chat_contract")
        if isinstance(template_data, dict)
        else None
    )

    explicit_name = _infer_explicit_contract_name(
        chat_template=chat_template,
        model_type=model_type,
        template_name=template_name,
    )
    if explicit_name:
        explicit_contract = (
            _build_llama_chatml_contract(chat_template)
            if explicit_name == "llama_chatml"
            else load_template_chat_contract(explicit_name)
        )
        if explicit_contract is not None:
            contract = explicit_contract

    if contract is None:
        return None

    if explicit_name is None and not looks_like_instruction_chat_model(
        chat_template=chat_template,
        finetune=finetune,
        model_name=model_name,
        model_type=model_type,
    ):
        return None

    extracted_system = extract_static_default_system_prompt(str(chat_template or ""))
    if extracted_system and not str(contract.get("default_system_prompt") or "").strip():
        contract["default_system_prompt"] = extracted_system
        contract["inject_default_system_prompt"] = True

    detected_markers = _collect_template_markers(chat_template)
    if detected_markers:
        merged_markers = list(contract.get("template_markers") or [])
        for marker in detected_markers:
            if marker not in merged_markers:
                merged_markers.append(marker)
        contract["template_markers"] = merged_markers

    return normalize_chat_contract(contract)


def resolve_chat_template_mode(
    mode: str,
    fallback_contract: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    mode_lc = str(mode or "auto").strip().lower()
    if mode_lc == "none":
        return None
    if mode_lc == "auto":
        return normalize_chat_contract(fallback_contract)
    return load_template_chat_contract(mode_lc)
