#!/usr/bin/env python3
"""Build a persistent generic training-decision artifact from a frozen winner and recent runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from training_policy_v7 import build_training_decision_payload, load_json, probe_metrics_from_doc, render_training_decision_md


RUN_RE = re.compile(r"_r(\d+)$")


def _rung_number(path: Path) -> int:
    match = RUN_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _candidate_run_paths(runs_root: Path, prefix_glob: str, frozen_run: Path) -> list[Path]:
    candidates = []
    for run_dir in sorted(runs_root.glob(prefix_glob), key=_rung_number):
        if not run_dir.is_dir():
            continue
        if run_dir.resolve() == frozen_run.resolve():
            continue
        candidates.append(run_dir.resolve())
    return candidates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="Spec id")
    ap.add_argument("--frozen-run", required=True, help="Frozen raw winner run directory")
    ap.add_argument("--baseline-probe-name", default=None, help="Probe report filename under the frozen run")
    ap.add_argument("--runs-root", default="version/v7/runs", help="Runs root for auto-discovery")
    ap.add_argument("--run-glob", required=True, help="Glob used to discover descendant run directories")
    ap.add_argument("--run", action="append", default=[], help="Optional explicit descendant run dirs")
    ap.add_argument("--probe-name", required=True, help="Probe report filename under descendant runs")
    ap.add_argument("--pilot-gate-name", default=None, help="Optional pilot gate filename under descendant runs")
    ap.add_argument("--families", nargs="+", required=True, help="Known family names")
    ap.add_argument("--hidden-split", action="append", default=["hidden_train", "hidden_test"], help="Hidden split ids")
    ap.add_argument("--strong-overall-threshold", type=float, default=0.80)
    ap.add_argument("--strong-hidden-threshold", type=float, default=1.0)
    ap.add_argument("--blocked-descendant-threshold", type=int, default=2)
    ap.add_argument("--default-allowed-action", default="pilot_train")
    ap.add_argument("--blocked-default-action", default="decode_repair")
    ap.add_argument("--suggested-next-training-branch", default="capacity_branch")
    ap.add_argument("--training-reenable-condition", action="append", default=[])
    ap.add_argument("--banned-pattern", action="append", default=[])
    ap.add_argument("--json-out", required=True, help="Output decision JSON")
    ap.add_argument("--md-out", required=True, help="Output decision markdown")
    args = ap.parse_args()

    spec = str(args.spec).strip()
    families = tuple(str(item).strip() for item in args.families if str(item).strip())
    hidden_splits = tuple(str(item).strip() for item in args.hidden_split if str(item).strip())
    frozen_run = Path(args.frozen_run).expanduser().resolve()
    baseline_probe_name = str(args.baseline_probe_name or args.probe_name).strip()
    frozen_probe = frozen_run / baseline_probe_name
    frozen_doc = load_json(frozen_probe)
    if not isinstance(frozen_doc, dict):
        raise SystemExit(f"failed to load frozen probe report: {frozen_probe}")
    frozen_metrics = probe_metrics_from_doc(frozen_doc, families=families, hidden_splits=hidden_splits)

    descendant_paths = (
        [Path(item).expanduser().resolve() for item in args.run]
        if args.run
        else _candidate_run_paths(Path(args.runs_root).expanduser().resolve(), str(args.run_glob), frozen_run)
    )
    descendants: list[dict[str, object]] = []
    for run_dir in descendant_paths:
        probe_path = run_dir / str(args.probe_name)
        probe_doc = load_json(probe_path)
        if not isinstance(probe_doc, dict):
            continue
        metrics = probe_metrics_from_doc(probe_doc, families=families, hidden_splits=hidden_splits)
        gate_doc = load_json(run_dir / str(args.pilot_gate_name)) if args.pilot_gate_name else None
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

    payload = build_training_decision_payload(
        spec=spec,
        frozen_run=frozen_run,
        frozen_metrics=frozen_metrics,
        descendants=descendants,
        hidden_splits=hidden_splits,
        strong_overall_exact_threshold=float(args.strong_overall_threshold),
        strong_hidden_exact_threshold=float(args.strong_hidden_threshold),
        blocked_descendant_threshold=int(args.blocked_descendant_threshold),
        default_allowed_action=str(args.default_allowed_action),
        blocked_default_action=str(args.blocked_default_action),
        suggested_next_training_branch=str(args.suggested_next_training_branch),
        training_reenable_conditions=[str(item).strip() for item in args.training_reenable_condition if str(item).strip()],
        banned_training_patterns=[str(item).strip() for item in args.banned_pattern if str(item).strip()],
    )

    json_out = Path(args.json_out).expanduser().resolve()
    md_out = Path(args.md_out).expanduser().resolve()
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(render_training_decision_md(payload), encoding="utf-8")
    print(json.dumps({"training_allowed": payload["training_allowed"], "default_action": payload["default_action"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
