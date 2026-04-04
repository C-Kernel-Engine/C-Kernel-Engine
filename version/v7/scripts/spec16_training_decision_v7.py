#!/usr/bin/env python3
"""Build a persistent spec16 training decision artifact from the frozen winner and recent runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training_policy_v7 import build_training_decision_payload, load_json, probe_metrics_from_doc, render_training_decision_md


FAMILIES = ("memory_map", "timeline", "system_diagram")
HIDDEN_SPLITS = ("hidden_train", "hidden_test")
RUN_RE = re.compile(r"_r(\d+)$")


def _probe_metrics_from_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return probe_metrics_from_doc(doc, families=FAMILIES, hidden_splits=HIDDEN_SPLITS)


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
    payload = build_training_decision_payload(
        spec="spec16",
        frozen_run=Path("/tmp/spec16"),
        frozen_metrics={
            "overall_exact": 0.0,
            "hidden_exact_rates": {"hidden_train": 0.0, "hidden_test": 0.0},
            "repairable_nonexact_rows": 0,
            "malformed_nonexact_rows": 0,
        },
        descendants=[],
    )
    return list(payload["paper_guidance"])


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
    payload = build_training_decision_payload(
        spec="spec16",
        frozen_run=frozen_run,
        frozen_metrics=frozen_metrics,
        descendants=descendants,
        hidden_splits=HIDDEN_SPLITS,
        strong_overall_exact_threshold=0.80,
        strong_hidden_exact_threshold=1.0,
        blocked_descendant_threshold=2,
        default_allowed_action="pilot_train",
        blocked_default_action="decode_repair",
        suggested_next_training_branch="capacity_branch",
        training_reenable_conditions=_training_reenable_conditions(),
        banned_training_patterns=_banned_training_patterns(),
        paper_guidance=_paper_guidance(),
    )
    payload["schema"] = "ck.spec16_training_decision.v1"
    return payload


def _render_md(payload: dict[str, Any]) -> str:
    return render_training_decision_md(payload).replace(
        "# Training Decision",
        "# Spec16 Training Decision",
        1,
    )


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
    frozen_doc = load_json(frozen_probe)
    if not isinstance(frozen_doc, dict):
        raise SystemExit(f"failed to load frozen probe report: {frozen_probe}")
    frozen_metrics = _probe_metrics_from_doc(frozen_doc)

    descendant_paths = [Path(item).expanduser().resolve() for item in args.run] if args.run else _candidate_run_paths(Path(args.runs_root).expanduser().resolve(), frozen_run)
    descendants: list[dict[str, Any]] = []
    for run_dir in descendant_paths:
        probe_path = run_dir / "spec16_probe_report.json"
        probe_doc = load_json(probe_path)
        if not isinstance(probe_doc, dict):
            continue
        metrics = _probe_metrics_from_doc(probe_doc)
        gate_doc = load_json(run_dir / "spec16_pilot_gate.json")
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
