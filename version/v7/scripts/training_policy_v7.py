#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PAPER_GUIDANCE = [
    {
        "paper": "Chinchilla (Hoffmann et al., 2022)",
        "gate": "Do not trust a pilot or canary budget unless selected tokens and actual processed tokens match.",
        "suggestion": "Make token-budget accounting explicit before promoting a branch.",
    },
    {
        "paper": "phi-1 (Gunasekar et al., 2023)",
        "gate": "Prefer clean, dense, compiler-validated coverage over noisy growth or repair prose.",
        "suggestion": "Add new rows only when they add validated unique coverage.",
    },
    {
        "paper": "Deduplication of Training Data Makes Language Models Better (Lee et al., 2022)",
        "gate": "Do not shrink repetition unless you replace it with new useful coverage.",
        "suggestion": "Show replacement coverage before accepting a reduced curriculum mass.",
    },
    {
        "paper": "HumanEval / Codex (Chen et al., 2021)",
        "gate": "When outputs are mostly renderable or mechanically repairable, prefer decode/repair over more CE training.",
        "suggestion": "Route by executable or compilable correctness, not by loss alone.",
    },
    {
        "paper": "Grokking (Power et al., 2022)",
        "gate": "If loss improves while exactness regresses, stop repeating the same training style.",
        "suggestion": "Treat that pattern as a block on more same-style repair rungs.",
    },
]


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def infer_family_from_row(row: dict[str, Any], families: tuple[str, ...]) -> str:
    texts = [
        str(row.get("prompt") or ""),
        str(row.get("expected_output") or ""),
        str(row.get("actual_output") or ""),
        str(row.get("label") or ""),
    ]
    for family in families:
        for text in texts:
            if f"[layout:{family}]" in text or f"[family:{family}]" in text:
                return family
    return "unknown"


def probe_metrics_from_doc(
    doc: dict[str, Any],
    *,
    families: tuple[str, ...],
    hidden_splits: tuple[str, ...] = ("hidden_train", "hidden_test"),
) -> dict[str, Any]:
    results = [row for row in (doc.get("results") or []) if isinstance(row, dict)]
    family_exact: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    hidden_exact: Counter[str] = Counter()
    hidden_total: Counter[str] = Counter()
    split_exact: Counter[str] = Counter()
    split_renderable: Counter[str] = Counter()
    split_total: Counter[str] = Counter()
    exact_total = 0
    renderable_total = 0
    repairable_tail = 0
    malformed = 0

    for row in results:
        family = infer_family_from_row(row, families)
        split = str(row.get("split") or "")
        family_total[family] += 1
        split_total[split] += 1
        if bool(row.get("exact_match")):
            exact_total += 1
            family_exact[family] += 1
            split_exact[split] += 1
        if bool(row.get("renderable")):
            renderable_total += 1
            split_renderable[split] += 1
        if split in hidden_splits:
            hidden_total[split] += 1
            if bool(row.get("exact_match")):
                hidden_exact[split] += 1
        if not bool(row.get("exact_match")):
            if bool(row.get("renderable")):
                repairable_tail += 1
            else:
                malformed += 1

    return {
        "n_results": len(results),
        "overall_exact": rate(exact_total, len(results)),
        "renderable": rate(renderable_total, len(results)),
        "family_exact_counts": {family: int(family_exact.get(family, 0)) for family in families},
        "family_totals": {family: int(family_total.get(family, 0)) for family in families},
        "family_exact_rates": {
            family: rate(family_exact.get(family, 0), family_total.get(family, 0))
            for family in families
        },
        "hidden_exact_rates": {
            split: rate(hidden_exact.get(split, 0), hidden_total.get(split, 0))
            for split in hidden_splits
        },
        "split_exact_rates": {
            split: rate(split_exact.get(split, 0), split_total.get(split, 0))
            for split in sorted(split_total)
        },
        "split_renderable_rates": {
            split: rate(split_renderable.get(split, 0), split_total.get(split, 0))
            for split in sorted(split_total)
        },
        "repairable_nonexact_rows": repairable_tail,
        "malformed_nonexact_rows": malformed,
    }


def build_pilot_gate_payload(
    *,
    spec: str,
    baseline_probe: Path,
    current_probe: Path,
    baseline_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    families: tuple[str, ...],
    hidden_splits: tuple[str, ...] = ("hidden_train", "hidden_test"),
    require_hidden_non_regression: bool = True,
    require_family_non_regression: bool = True,
    require_overall_non_regression: bool = True,
    improve_families: tuple[str, ...] = (),
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    if require_hidden_non_regression:
        hidden_non_regression = all(
            current_metrics["hidden_exact_rates"].get(split, 0.0)
            >= baseline_metrics["hidden_exact_rates"].get(split, 0.0)
            for split in hidden_splits
        )
        checks["hidden_non_regression"] = hidden_non_regression
        if not hidden_non_regression:
            reasons.append("pilot regressed hidden exactness relative to the frozen baseline")

    if require_family_non_regression:
        family_non_regression = all(
            current_metrics["family_exact_rates"].get(family, 0.0)
            >= baseline_metrics["family_exact_rates"].get(family, 0.0)
            for family in families
        )
        checks["family_non_regression"] = family_non_regression
        if not family_non_regression:
            reasons.append("pilot regressed at least one family relative to the frozen baseline")

    if require_overall_non_regression:
        overall_exact_non_regression = current_metrics["overall_exact"] >= baseline_metrics["overall_exact"]
        checks["overall_exact_non_regression"] = overall_exact_non_regression
        if not overall_exact_non_regression:
            reasons.append("pilot regressed overall exactness relative to the frozen baseline")

    if improve_families:
        for family in improve_families:
            improved = current_metrics["family_exact_rates"].get(family, 0.0) > baseline_metrics["family_exact_rates"].get(family, 0.0)
            checks[f"{family}_improved"] = improved
            if not improved:
                reasons.append(f"pilot did not improve {family} exactness over the frozen baseline")

    return {
        "schema": "ck.training_pilot_gate.v1",
        "spec": spec,
        "baseline_probe": str(baseline_probe),
        "current_probe": str(current_probe),
        "checks": checks,
        "clears_gate": all(checks.values()) if checks else True,
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "reasons": reasons,
    }


def render_pilot_gate_md(payload: dict[str, Any], *, families: tuple[str, ...]) -> str:
    checks = payload["checks"]
    verdict = "PASS" if payload["clears_gate"] else "BLOCK"
    lines = [
        "# Training Pilot Gate",
        "",
        f"- spec: `{payload.get('spec')}`",
        f"- verdict: **{verdict}**",
        f"- baseline: `{payload['baseline_probe']}`",
        f"- current: `{payload['current_probe']}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Metrics", ""])
    lines.append(f"- baseline overall exact: `{payload['baseline_metrics']['overall_exact']:.4f}`")
    lines.append(f"- current overall exact: `{payload['current_metrics']['overall_exact']:.4f}`")
    lines.append(f"- baseline renderable: `{payload['baseline_metrics']['renderable']:.4f}`")
    lines.append(f"- current renderable: `{payload['current_metrics']['renderable']:.4f}`")
    lines.extend(["", "## Family exact counts", ""])
    for family in families:
        base = payload["baseline_metrics"]["family_exact_counts"].get(family, 0)
        cur = payload["current_metrics"]["family_exact_counts"].get(family, 0)
        total = payload["current_metrics"]["family_totals"].get(family, 0)
        lines.append(f"- {family}: `{cur}/{total}` vs baseline `{base}/{total}`")
    if payload["reasons"]:
        lines.extend(["", "## Gate blockers", ""])
        lines.extend(f"- {reason}" for reason in payload["reasons"])
    return "\n".join(lines) + "\n"


def build_training_decision_payload(
    *,
    spec: str,
    frozen_run: Path,
    frozen_metrics: dict[str, Any],
    descendants: list[dict[str, Any]],
    hidden_splits: tuple[str, ...] = ("hidden_train", "hidden_test"),
    strong_overall_exact_threshold: float = 0.80,
    strong_hidden_exact_threshold: float = 1.0,
    blocked_descendant_threshold: int = 2,
    default_allowed_action: str = "pilot_train",
    blocked_default_action: str = "decode_repair",
    suggested_next_training_branch: str = "capacity_branch",
    training_reenable_conditions: list[str] | None = None,
    banned_training_patterns: list[str] | None = None,
    paper_guidance: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    blocked_descendants = [
        row for row in descendants
        if (not bool(row.get("beats_frozen"))) or (row.get("pilot_gate_clears") is False)
    ]
    strong_hidden = all(
        frozen_metrics["hidden_exact_rates"].get(split, 0.0) >= strong_hidden_exact_threshold
        for split in hidden_splits
    )
    strong_frozen = frozen_metrics["overall_exact"] >= strong_overall_exact_threshold and strong_hidden
    repairable_bias = frozen_metrics["repairable_nonexact_rows"] >= frozen_metrics["malformed_nonexact_rows"]

    training_allowed = not (strong_frozen and len(blocked_descendants) >= blocked_descendant_threshold)
    default_action = default_allowed_action if training_allowed else blocked_default_action

    reasons: list[str] = []
    if strong_frozen:
        reasons.append("frozen raw baseline already demonstrates the target capability")
    if len(blocked_descendants) >= blocked_descendant_threshold:
        reasons.append("multiple post-baseline runs failed to beat or even preserve the frozen winner")
    if repairable_bias:
        reasons.append("the remaining error mass is closer to decode/repair work than fresh capability acquisition")
    if descendants and descendants[-1].get("pilot_gate_clears") is False:
        reasons.append("the latest pilot failed the frozen-baseline gate and should not promote into another raw rung")

    unblock_requirements = [
        "complete an autopsy of the latest blocked rung before changing training data again",
        "verify pilot or canary budget integrity so selected tokens match actual processed tokens",
        "prove the next training idea is a new branch, not another narrow raw-repair continuation",
    ]

    return {
        "schema": "ck.training_decision.v1",
        "spec": spec,
        "frozen_raw_winner": str(frozen_run),
        "frozen_metrics": frozen_metrics,
        "descendant_runs": descendants,
        "paper_guidance": list(paper_guidance or DEFAULT_PAPER_GUIDANCE),
        "training_allowed": training_allowed,
        "default_action": default_action,
        "allowed_actions": ["decode_repair", "measurement_autopsy"] + ([default_allowed_action] if training_allowed else []),
        "suggested_next_training_branch": suggested_next_training_branch,
        "block_raw_repair_rungs": not training_allowed,
        "reasons": reasons,
        "unblock_requirements": unblock_requirements,
        "training_reenable_conditions": list(training_reenable_conditions or []),
        "banned_training_patterns": list(banned_training_patterns or []),
    }


def render_training_decision_md(payload: dict[str, Any]) -> str:
    verdict = "ALLOW TRAINING" if payload["training_allowed"] else "BLOCK RAW TRAINING"
    lines = [
        "# Training Decision",
        "",
        f"- spec: `{payload.get('spec')}`",
        f"- verdict: **{verdict}**",
        f"- frozen raw winner: `{payload['frozen_raw_winner']}`",
        f"- default action: `{payload['default_action']}`",
        f"- suggested next training branch: `{payload['suggested_next_training_branch']}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("reasons") or [])
    lines.extend(["", "## Unblock Requirements", ""])
    lines.extend(f"- {item}" for item in payload.get("unblock_requirements") or [])
    if payload.get("training_reenable_conditions"):
        lines.extend(["", "## Training Re-Enable Conditions", ""])
        lines.extend(f"- {item}" for item in payload["training_reenable_conditions"])
    if payload.get("banned_training_patterns"):
        lines.extend(["", "## Banned Training Patterns", ""])
        lines.extend(f"- {item}" for item in payload["banned_training_patterns"])
    lines.extend(["", "## Paper Guidance", ""])
    for item in payload.get("paper_guidance") or []:
        lines.append(f"- {item['paper']}: {item['gate']} Suggestion: {item['suggestion']}")
    if payload.get("descendant_runs"):
        lines.extend(["", "## Descendant Runs", ""])
        for row in payload["descendant_runs"]:
            lines.append(
                f"- `{row['run_name']}`: exact `{row['overall_exact']:.4f}`, renderable `{row['renderable']:.4f}`, "
                f"beats frozen `{row['beats_frozen']}`, pilot gate `{row['pilot_gate_clears']}`"
            )
    return "\n".join(lines) + "\n"
