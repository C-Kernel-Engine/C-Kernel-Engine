#!/usr/bin/env python3
"""Compare a spec16 pilot probe report against a frozen baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FAMILIES = ("memory_map", "timeline", "system_diagram")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _family_from_prompt(prompt: str) -> str:
    text = str(prompt or "")
    for family in FAMILIES:
        if f"[layout:{family}]" in text:
            return family
    return "unknown"


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _metrics(doc: dict[str, Any]) -> dict[str, Any]:
    results = [row for row in (doc.get("results") or []) if isinstance(row, dict)]
    family_exact: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    hidden_exact: Counter[str] = Counter()
    hidden_total: Counter[str] = Counter()
    exact_total = 0
    renderable_total = 0

    for row in results:
        family = _family_from_prompt(str(row.get("prompt") or ""))
        family_total[family] += 1
        if bool(row.get("exact_match")):
            exact_total += 1
            family_exact[family] += 1
        if bool(row.get("renderable")):
            renderable_total += 1
        split = str(row.get("split") or "")
        if split in {"hidden_train", "hidden_test"}:
            hidden_total[split] += 1
            if bool(row.get("exact_match")):
                hidden_exact[split] += 1

    family_rates = {
        family: _rate(family_exact.get(family, 0), family_total.get(family, 0))
        for family in FAMILIES
    }
    hidden_rates = {
        split: _rate(hidden_exact.get(split, 0), hidden_total.get(split, 0))
        for split in ("hidden_train", "hidden_test")
    }
    return {
        "n_results": len(results),
        "overall_exact": _rate(exact_total, len(results)),
        "renderable": _rate(renderable_total, len(results)),
        "family_exact_counts": {family: int(family_exact.get(family, 0)) for family in FAMILIES},
        "family_totals": {family: int(family_total.get(family, 0)) for family in FAMILIES},
        "family_exact_rates": family_rates,
        "hidden_exact_counts": {split: int(hidden_exact.get(split, 0)) for split in ("hidden_train", "hidden_test")},
        "hidden_totals": {split: int(hidden_total.get(split, 0)) for split in ("hidden_train", "hidden_test")},
        "hidden_exact_rates": hidden_rates,
    }


def _render_md(payload: dict[str, Any]) -> str:
    checks = payload["checks"]
    verdict = "PASS" if payload["clears_gate"] else "BLOCK"
    lines = [
        "# Spec16 Pilot Gate",
        "",
        f"- Verdict: **{verdict}**",
        f"- Baseline: `{payload['baseline_probe']}`",
        f"- Current: `{payload['current_probe']}`",
        "",
        "## Checks",
        "",
        f"- Hidden non-regression: `{checks['hidden_non_regression']}`",
        f"- Family non-regression: `{checks['family_non_regression']}`",
        f"- System-diagram improved: `{checks['system_diagram_improved']}`",
        f"- Overall exact non-regression: `{checks['overall_exact_non_regression']}`",
        "",
        "## Metrics",
        "",
        f"- Baseline overall exact: `{payload['baseline_metrics']['overall_exact']:.4f}`",
        f"- Current overall exact: `{payload['current_metrics']['overall_exact']:.4f}`",
        f"- Baseline renderable: `{payload['baseline_metrics']['renderable']:.4f}`",
        f"- Current renderable: `{payload['current_metrics']['renderable']:.4f}`",
        "",
        "## Family exact counts",
        "",
    ]
    for family in FAMILIES:
        base = payload["baseline_metrics"]["family_exact_counts"][family]
        cur = payload["current_metrics"]["family_exact_counts"][family]
        total = payload["current_metrics"]["family_totals"][family]
        lines.append(f"- {family}: `{cur}/{total}` vs baseline `{base}/{total}`")
    if payload["reasons"]:
        lines.extend(["", "## Gate blockers", ""])
        lines.extend(f"- {reason}" for reason in payload["reasons"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current", required=True, help="Current spec16 probe report JSON")
    ap.add_argument("--baseline", required=True, help="Frozen-baseline spec16 probe report JSON")
    ap.add_argument("--json-out", required=True, help="Output JSON path")
    ap.add_argument("--md-out", required=True, help="Output markdown path")
    args = ap.parse_args()

    current_path = Path(args.current).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    current_metrics = _metrics(_load_json(current_path))
    baseline_metrics = _metrics(_load_json(baseline_path))

    family_non_regression = all(
        current_metrics["family_exact_rates"][family] >= baseline_metrics["family_exact_rates"][family]
        for family in FAMILIES
    )
    system_diagram_improved = (
        current_metrics["family_exact_rates"]["system_diagram"]
        > baseline_metrics["family_exact_rates"]["system_diagram"]
    )
    hidden_non_regression = all(
        current_metrics["hidden_exact_rates"][split] >= baseline_metrics["hidden_exact_rates"][split]
        for split in ("hidden_train", "hidden_test")
    )
    overall_exact_non_regression = current_metrics["overall_exact"] >= baseline_metrics["overall_exact"]

    checks = {
        "hidden_non_regression": hidden_non_regression,
        "family_non_regression": family_non_regression,
        "system_diagram_improved": system_diagram_improved,
        "overall_exact_non_regression": overall_exact_non_regression,
    }
    reasons: list[str] = []
    if not hidden_non_regression:
        reasons.append("pilot regressed hidden exactness relative to the frozen baseline")
    if not family_non_regression:
        reasons.append("pilot regressed at least one family relative to the frozen baseline")
    if not system_diagram_improved:
        reasons.append("pilot did not improve system_diagram exactness over the frozen baseline")
    if not overall_exact_non_regression:
        reasons.append("pilot regressed overall exactness relative to the frozen baseline")

    payload = {
        "schema": "ck.spec16_pilot_gate.v1",
        "baseline_probe": str(baseline_path),
        "current_probe": str(current_path),
        "checks": checks,
        "clears_gate": all(checks.values()),
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "reasons": reasons,
    }

    json_out = Path(args.json_out).expanduser().resolve()
    md_out = Path(args.md_out).expanduser().resolve()
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(_render_md(payload), encoding="utf-8")
    print(json.dumps({"clears_gate": payload["clears_gate"], "checks": checks, "reasons": reasons}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
