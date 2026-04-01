#!/usr/bin/env python3
"""Shared syntax-only pre-repair for compact scene DSL token streams."""

from __future__ import annotations

import re
from typing import Any


_BRACKET_TOKEN_RE = re.compile(r"\[[^\[\]\n]+\]")


def _split_token(token: str) -> tuple[str, str]:
    body = str(token or "").strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    if ":" not in body:
        return body.strip().lower(), ""
    name, payload = body.split(":", 1)
    return name.strip().lower(), payload.strip()


def _extract_bracket_tokens(text: str) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    while "][" in cleaned:
        cleaned = cleaned.replace("][", "] [")
    return [tok.strip() for tok in _BRACKET_TOKEN_RE.findall(cleaned) if tok.strip()]


def repair_compact_scene_text(
    text: Any,
    *,
    singleton_names: set[str] | None = None,
    prompt_only_names: set[str] | None = None,
) -> dict[str, Any]:
    """Repair recoverable syntax-only issues in compact scene DSL text.

    This helper is intentionally narrow. It may:
    - recover bracket-token streams without spaces
    - keep only the first [scene] ... [/scene] region
    - append [/scene] if missing
    - drop prompt-only control tags copied into the answer
    - dedupe singleton tags by name

    It does not invent family-specific components or semantic refs.
    """

    singletons = {str(name or "").strip().lower() for name in (singleton_names or set()) if str(name or "").strip()}
    prompt_only = {str(name or "").strip().lower() for name in (prompt_only_names or set()) if str(name or "").strip()}

    tokens = _extract_bracket_tokens(str(text or ""))
    if not tokens:
        return {
            "repaired_text": str(text or "").strip(),
            "changed": False,
            "diag": {"reason": "no_bracket_tokens"},
        }

    try:
        start_idx = next(idx for idx, token in enumerate(tokens) if token == "[scene]")
    except StopIteration:
        start_idx = -1
    if start_idx >= 0:
        tokens = tokens[start_idx:]

    repaired: list[str] = []
    seen_singletons: set[str] = set()
    saw_scene = False
    saw_end = False
    dropped_prompt_only = 0
    dropped_duplicate_singletons = 0
    dropped_duplicate_scene_markers = 0

    for token in tokens:
        name, _payload = _split_token(token)
        if token == "[scene]":
            if saw_scene:
                dropped_duplicate_scene_markers += 1
                continue
            saw_scene = True
            repaired.append(token)
            continue
        if not saw_scene:
            continue
        if token == "[/scene]":
            repaired.append(token)
            saw_end = True
            break
        if name in prompt_only:
            dropped_prompt_only += 1
            continue
        if name in singletons:
            if name in seen_singletons:
                dropped_duplicate_singletons += 1
                continue
            seen_singletons.add(name)
        repaired.append(token)

    if saw_scene and not saw_end:
        repaired.append("[/scene]")
        saw_end = True

    repaired_text = " ".join(repaired).strip()
    original_norm = " ".join(str(text or "").strip().split())
    changed = repaired_text != original_norm
    return {
        "repaired_text": repaired_text,
        "changed": changed,
        "diag": {
            "saw_scene": saw_scene,
            "saw_end": saw_end,
            "dropped_prompt_only": dropped_prompt_only,
            "dropped_duplicate_singletons": dropped_duplicate_singletons,
            "dropped_duplicate_scene_markers": dropped_duplicate_scene_markers,
            "singleton_names": sorted(singletons),
        },
    }

