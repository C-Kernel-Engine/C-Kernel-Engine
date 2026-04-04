#!/usr/bin/env python3
"""Tokenizer policy helpers for v7 structured-training lines."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BANNED_SPECIAL_TOKENS = {
    "[bundle]...[/bundle]",
}


def sanitize_tokenizer_doc(tokenizer_doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop known placeholder special tokens while preserving vocab ids."""
    doc = deepcopy(tokenizer_doc)
    removed: list[str] = []
    added_tokens = doc.get("added_tokens")
    if isinstance(added_tokens, list):
        kept: list[dict[str, Any]] = []
        for row in added_tokens:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            content = str(row.get("content") or "")
            if content in BANNED_SPECIAL_TOKENS:
                removed.append(content)
                continue
            kept.append(row)
        doc["added_tokens"] = kept
    return doc, removed


def visible_special_tokens(tokenizer_doc: dict[str, Any]) -> list[str]:
    added_tokens = tokenizer_doc.get("added_tokens")
    if not isinstance(added_tokens, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for row in added_tokens:
        if not isinstance(row, dict) or row.get("special") is not True:
            continue
        content = str(row.get("content") or "")
        if not content or content in seen:
            continue
        seen.add(content)
        out.append(content)
    return out


def normalize_tokenizer_for_warmstart(tokenizer_doc: dict[str, Any]) -> dict[str, Any]:
    """Normalize tokenizer docs for warm-start compatibility checks."""
    sanitized, _removed = sanitize_tokenizer_doc(tokenizer_doc)
    model = sanitized.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    merges = model.get("merges") if isinstance(model, dict) else None
    return {
        "added_tokens": [
            {
                "id": int(row.get("id")),
                "content": str(row.get("content") or ""),
                "special": bool(row.get("special") is True),
            }
            for row in (sanitized.get("added_tokens") or [])
            if isinstance(row, dict) and isinstance(row.get("id"), int)
        ],
        "vocab": vocab if isinstance(vocab, dict) else {},
        "merges": merges if isinstance(merges, list) else [],
    }
