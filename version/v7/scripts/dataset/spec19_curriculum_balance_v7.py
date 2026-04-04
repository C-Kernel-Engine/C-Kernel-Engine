#!/usr/bin/env python3
"""Surface-balance helpers for small-model cumulative spec19 curricula."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


POLICY_VERSION = "ck.spec19_curriculum_balance.v1"
DEFAULT_SOFT_SURFACES = (
    "routebook_paraphrase",
    "routebook_direct_hint",
    "style_topology_bridge",
)


def _surface_counts(meta_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in meta_rows:
        surface = str(item.get("prompt_surface") or "").strip()
        if surface:
            counts[surface] += 1
    return dict(sorted(counts.items()))


def balance_delta_rows(
    *,
    stage: str,
    base_train_rows: int,
    base_surface_counts: dict[str, int],
    proposed_rows: list[str],
    proposed_meta: list[dict[str, Any]],
    max_surface_growth_fraction: float = 0.20,
    dominant_surface: str = "clean_stop_anchor",
    soft_surfaces: tuple[str, ...] = DEFAULT_SOFT_SURFACES,
    dominant_to_soft_ratio_max: float = 1.5,
    new_surface_cap: int = 6,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    if len(proposed_rows) != len(proposed_meta):
        raise ValueError("proposed_rows and proposed_meta must have the same length")
    if base_train_rows < 0:
        raise ValueError("base_train_rows must be non-negative")

    proposed_surface_counts = _surface_counts(proposed_meta)
    capped_surface_counts: dict[str, int] = {}

    for surface, proposed_count in proposed_surface_counts.items():
        base_count = int(base_surface_counts.get(surface, 0) or 0)
        if base_count > 0:
            cap = max(1, int(math.ceil(base_count * float(max_surface_growth_fraction))))
        else:
            cap = max(1, int(new_surface_cap))
        capped_surface_counts[surface] = min(int(proposed_count), cap)

    soft_total = sum(capped_surface_counts.get(surface, 0) for surface in soft_surfaces)
    dominant_cap_before_ratio = capped_surface_counts.get(dominant_surface)
    dominant_ratio_applied = False
    if dominant_cap_before_ratio is not None and soft_total > 0:
        dominant_ratio_cap = max(1, int(math.floor(float(dominant_to_soft_ratio_max) * float(soft_total))))
        if dominant_cap_before_ratio > dominant_ratio_cap:
            capped_surface_counts[dominant_surface] = dominant_ratio_cap
            dominant_ratio_applied = True

    remaining = dict(capped_surface_counts)
    kept_rows: list[str] = []
    kept_meta: list[dict[str, Any]] = []
    for row, meta in zip(proposed_rows, proposed_meta):
        surface = str(meta.get("prompt_surface") or "").strip()
        if not surface:
            continue
        if remaining.get(surface, 0) <= 0:
            continue
        kept_rows.append(row)
        kept_meta.append(meta)
        remaining[surface] -= 1

    kept_surface_counts = _surface_counts(kept_meta)
    dropped_surface_counts = {
        surface: int(proposed_surface_counts.get(surface, 0) - kept_surface_counts.get(surface, 0))
        for surface in sorted(proposed_surface_counts)
    }

    report = {
        "policy": POLICY_VERSION,
        "stage": str(stage),
        "base_train_rows": int(base_train_rows),
        "base_surface_counts": {str(k): int(v) for k, v in sorted((base_surface_counts or {}).items())},
        "proposed_rows": int(len(proposed_rows)),
        "kept_rows": int(len(kept_rows)),
        "dropped_rows": int(len(proposed_rows) - len(kept_rows)),
        "proposed_surface_counts": proposed_surface_counts,
        "capped_surface_counts": {str(k): int(v) for k, v in sorted(capped_surface_counts.items())},
        "kept_surface_counts": kept_surface_counts,
        "dropped_surface_counts": dropped_surface_counts,
        "thresholds": {
            "max_surface_growth_fraction": float(max_surface_growth_fraction),
            "dominant_surface": str(dominant_surface),
            "soft_surfaces": list(soft_surfaces),
            "dominant_to_soft_ratio_max": float(dominant_to_soft_ratio_max),
            "new_surface_cap": int(new_surface_cap),
        },
        "notes": [
            "Small-model safeguard: keep cumulative replay dominant and constrain corrective deltas by winner-line surface counts.",
            "Treat this as a default balance policy, not a universal law for larger-capacity models.",
        ],
        "dominant_ratio_applied": bool(dominant_ratio_applied),
    }
    return kept_rows, kept_meta, report
