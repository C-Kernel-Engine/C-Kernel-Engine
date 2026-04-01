#!/usr/bin/env python3
"""Build a persistent spec16 training decision artifact from the frozen winner and recent runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


FAMILIES = ("memory_map", "timeline", "system_diagram")
RUN_RE = re.compile(r"_r(\d+)$")


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _family_from_prompt(prompt: str) -> str:
    text = str(prompt or "")
    for family in FAMILIES:
        if f"[layout:{family}]" in text:
            return family
    return "unknown"


def _rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _probe_metrics_from_doc(doc: dict[str, Any]) -> dict[str, Any]:
    results = [row for row in (doc.get("results") or []) if isinstance(row, dict)]
    family_exact: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    hidden_exact: Counter[str] = Counter()
    hidden_total: Counter[str] = Counter()
    exact_total = 0
    renderable_total = 0
    repairable_tail = 0
    malformed = 0
    for row in results:
        family = _family_from_prompt(str(row.get("prompt") or ""))
        family_total[family] += 1
        if bool(row.get("exact_match")):
            exact_total += 1
            family_exact[family] += 1
        if bool(row.get("renderable")):
            renderable_total += 1
        if row.get("split") in {"hidden_train", "hidden_test"}:
            hidden_total[str(row["split"])] += 1
            if bool(row.get("exact_match")):
                hidden_exact[str(row["split"])] += 1
        if not bool(row.get("exact_match")):
            if bool(row.get("renderable")):
                repairable_tail += 1
            else:
                malformed += 1
    return {
        "n_results": len(results),
        "overall_exact": _rate(exact_total, len(results)),
        "renderable": _rate(renderable_total, len(results)),
        "family_exact_counts": {family: int(family_exact.get(family, 0)) for family in FAMILIES},
        "family_totals": {family: int(family_total.get(family, 0)) for family in FAMILIES},
        "family_exact_rates": {
            family: _rate(family_exact.get(family, 0), family_total.get(family, 0))
            for family in FAMILIES
        },
        "hidden_exact_rates": {
            split: _rate(hidden_exact.get(split, 0), hidden_total.get(split, 0))
            for split in ("hidden_train", "hidden_test")
        },
        "repairable_nonexact_rows": repairable_tail,
        "malformed_nonexact_rows": malformed,
    }


def _rung_number(path: Path) -> int:
    match = RUN_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _candidate_run_paths(root: Path, frozen_run: Path) -> list[Path]:
    parent = frozen_run.parent if frozen_run.parent.exists() else root
    candidates = []
    for run_dir in sorted(parent.glob("spec16_scene_bundle_l3_d192_h384_ctx768_r*"), key=_rung_number):
        if not run_dir.is_dir():
            continue
        if run_dir.resolve() == frozen_run.resolve():
            continue
        if (run_dir / "spec16_probe_report.json").exists():
            candidates.append(run_dir.resolve())
    return candidates


def _paper_guidance() -> list[dict[str, str]]:
    return [
        {
            "paper": "Chinchilla (Hoffmann et al., 2022)",
            "gate": "Do not trust a pilot budget unless selected tokens and actual processed tokens match.",
            "suggestion": "Block new raw training until token-budget accounting is verified end to end.",
        },
        {
            "paper": "phi-1 (Gunasekar et al., 2023)",
            "gate": "Prefer clean, compiler-validated coverage over more repair prose.",
            "suggestion": "Only add new training rows when they add validated unique coverage, not just new warning language.",
        },
        {
            "paper": "Deduplication of Training Data Makes Language Models Better (Lee et al., 2022)",
            "gate": "Do not remove shortcut repetition unless it is replaced with new useful coverage.",
            "suggestion": "Any dedup or curriculum shrink must show replacement coverage before training is allowed.",
        },
        {
            "paper": "HumanEval / Codex (Chen et al., 2021)",
            "gate": "When outputs are mostly renderable or mechanically repairable, prefer decode/repair over more CE training.",
            "suggestion": "Use executable or compilable correctness as the route selector, not train loss.",
        },
        {
            "paper": "Grokking (Power et al., 2022)",
            "gate": "If loss improves while exactness regresses, stop repeating the same objective and data style.",
            "suggestion": "Treat that pattern as a block on more same-family repair rungs.",
        },
    ]


def _training_reenable_conditions() -> list[str]:
    return [
        "No new raw spec16 rung may be a narrow repair-prose continuation. The next allowed training branch must be a clean capacity branch.",
        "A capacity branch must reuse the frozen r9 recipe and contract as the baseline surface. Do not reuse the r10-r12 repair-heavy prompt surfaces as the default raw recipe.",
        "Before launch, record both selected stage budgets and expected actual processed tokens, and verify the pilot fraction against the frozen baseline on actual processed tokens.",
        "A pilot may proceed to a full rung only if it improves system_diagram while showing zero family regression and zero hidden regression relative to frozen r9.",
        "If a candidate branch reuses the same unique row set as the previous blocked rung, treat it as a compute/order experiment only. It does not count as a new training idea.",
        "If deterministic repair on the frozen baseline is still the main source of gains, keep the line on decode/repair. Do not reopen raw training just because loss can still move.",
    ]


def _banned_training_patterns() -> list[str]:
    return [
        "Do not launch another raw rung whose main change is more warning-language rows about stopping, singleton tags, wrapper junk, or control markers.",
        "Do not treat the r11/r12 repair-heavy row surface as a safe pilot substrate for reduced-compute experiments.",
        "Do not approve a pilot that only changes row ordering or token budget on an otherwise unchanged brittle repair curriculum.",
        "Do not use low loss or improved renderability by itself as evidence that a raw branch is promotable.",
    ]


def _build_decision_payload(
    *,
    frozen_run: Path,
    frozen_metrics: dict[str, Any],
    descendants: list[dict[str, Any]],
) -> dict[str, Any]:
    blocked_descendants = [
        row for row in descendants
        if (not bool(row.get("beats_frozen"))) or (row.get("pilot_gate_clears") is False)
    ]
    strong_frozen = (
        frozen_metrics["overall_exact"] >= 0.80
        and frozen_metrics["hidden_exact_rates"]["hidden_train"] >= 1.0
        and frozen_metrics["hidden_exact_rates"]["hidden_test"] >= 1.0
    )
    repairable_bias = frozen_metrics["repairable_nonexact_rows"] >= frozen_metrics["malformed_nonexact_rows"]

    training_allowed = not (strong_frozen and len(blocked_descendants) >= 2)
    default_action = "pilot_train" if training_allowed else "decode_repair"
    suggested_next_training_branch = "capacity_branch" if not training_allowed else "pilot_train"
    reasons: list[str] = []
    if strong_frozen:
        reasons.append("frozen raw baseline already demonstrates the spec16 bundle capability")
    if len(blocked_descendants) >= 2:
        reasons.append("multiple post-baseline rungs failed to beat or even preserve the frozen winner")
    if repairable_bias:
        reasons.append("the remaining error mass is closer to decode/repair work than fresh capability acquisition")
    if descendants and descendants[-1].get("pilot_gate_clears") is False:
        reasons.append("the latest pilot failed the frozen-baseline gate and should not promote into another raw rung")

    unblock_requirements = [
        "complete an autopsy of the latest blocked rung before changing training data again",
        "verify pilot budget integrity so selected tokens match actual processed tokens",
        "prove the next training idea is a new branch, not another narrow raw-repair rung",
    ]
    if not training_allowed:
        unblock_requirements.append(
            "if more raw margin is still needed after decode/repair, branch to a capacity test on the frozen r9 recipe"
        )

    return {
        "schema": "ck.spec16_training_decision.v1",
        "spec": "spec16",
        "frozen_raw_winner": str(frozen_run),
        "frozen_metrics": frozen_metrics,
        "descendant_runs": descendants,
        "paper_guidance": _paper_guidance(),
        "training_allowed": training_allowed,
        "default_action": default_action,
        "allowed_actions": ["decode_repair", "measurement_autopsy"] + (["pilot_train"] if training_allowed else []),
        "suggested_next_training_branch": suggested_next_training_branch,
        "block_raw_repair_rungs": not training_allowed,
        "reasons": reasons,
        "unblock_requirements": unblock_requirements,
        "training_reenable_conditions": _training_reenable_conditions(),
        "banned_training_patterns": _banned_training_patterns(),
    }


def _render_md(payload: dict[str, Any]) -> str:
    verdict = "ALLOW TRAINING" if payload["training_allowed"] else "BLOCK RAW TRAINING"
    lines = [
        "# Spec16 Training Decision",
        "",
        f"- Verdict: **{verdict}**",
        f"- Frozen raw winner: `{payload['frozen_raw_winner']}`",
        f"- Default action: `{payload['default_action']}`",
        f"- Suggested next training branch: `{payload['suggested_next_training_branch']}`",
        "",
        "## Reasons",
        "",
        *[f"- {item}" for item in payload["reasons"]],
        "",
        "## Unblock Requirements",
        "",
        *[f"- {item}" for item in payload["unblock_requirements"]],
        "",
        "## Training Re-Enable Conditions",
        "",
        *[f"- {item}" for item in payload["training_reenable_conditions"]],
        "",
        "## Banned Training Patterns",
        "",
        *[f"- {item}" for item in payload["banned_training_patterns"]],
        "",
        "## Paper Guidance",
        "",
    ]
    for item in payload["paper_guidance"]:
        lines.append(f"- {item['paper']}: {item['gate']} Suggestion: {item['suggestion']}")
    if payload["descendant_runs"]:
        lines.extend(["", "## Descendant Runs", ""])
        for row in payload["descendant_runs"]:
            lines.append(
                f"- `{row['run_name']}`: exact `{row['overall_exact']:.4f}`, renderable `{row['renderable']:.4f}`, "
                f"beats frozen `{row['beats_frozen']}`, pilot gate `{row['pilot_gate_clears']}`"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen-run", required=True, help="Frozen raw winner run directory")
    ap.add_argument("--runs-root", default="version/v7/runs", help="Runs root for auto-discovery")
    ap.add_argument("--run", action="append", default=[], help="Optional explicit descendant run dirs")
    ap.add_argument("--json-out", required=True, help="Output decision JSON")
    ap.add_argument("--md-out", required=True, help="Output decision markdown")
    args = ap.parse_args()

    frozen_run = Path(args.frozen_run).expanduser().resolve()
    frozen_probe = frozen_run / "spec16_probe_report.json"
    frozen_doc = _load_json(frozen_probe)
    if not isinstance(frozen_doc, dict):
        raise SystemExit(f"failed to load frozen probe report: {frozen_probe}")
    frozen_metrics = _probe_metrics_from_doc(frozen_doc)

    descendant_paths = [Path(item).expanduser().resolve() for item in args.run] if args.run else _candidate_run_paths(Path(args.runs_root).expanduser().resolve(), frozen_run)
    descendants: list[dict[str, Any]] = []
    for run_dir in descendant_paths:
        probe_path = run_dir / "spec16_probe_report.json"
        probe_doc = _load_json(probe_path)
        if not isinstance(probe_doc, dict):
            continue
        metrics = _probe_metrics_from_doc(probe_doc)
        gate_doc = _load_json(run_dir / "spec16_pilot_gate.json")
        pilot_gate_clears = gate_doc.get("clears_gate") if isinstance(gate_doc, dict) else None
        descendants.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "overall_exact": metrics["overall_exact"],
                "renderable": metrics["renderable"],
                "family_exact_rates": metrics["family_exact_rates"],
                "hidden_exact_rates": metrics["hidden_exact_rates"],
                "beats_frozen": metrics["overall_exact"] > frozen_metrics["overall_exact"],
                "pilot_gate_clears": pilot_gate_clears,
            }
        )

    payload = _build_decision_payload(
        frozen_run=frozen_run,
        frozen_metrics=frozen_metrics,
        descendants=descendants,
    )
    json_out = Path(args.json_out).expanduser().resolve()
    md_out = Path(args.md_out).expanduser().resolve()
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(_render_md(payload), encoding="utf-8")
    print(json.dumps({"training_allowed": payload["training_allowed"], "default_action": payload["default_action"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
