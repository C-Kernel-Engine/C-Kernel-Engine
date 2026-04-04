#!/usr/bin/env python3
"""Compare a pilot or canary probe report against a frozen baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_policy_v7 import (
    build_pilot_gate_payload,
    load_json,
    probe_metrics_from_doc,
    render_pilot_gate_md,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="Spec id")
    ap.add_argument("--current", required=True, help="Current probe report JSON")
    ap.add_argument("--baseline", required=True, help="Frozen baseline probe report JSON")
    ap.add_argument("--families", nargs="+", required=True, help="Known family names")
    ap.add_argument("--improve-family", action="append", default=[], help="Family that must strictly improve")
    ap.add_argument("--hidden-split", action="append", default=["hidden_train", "hidden_test"], help="Hidden split ids")
    ap.add_argument("--no-hidden-non-regression", action="store_true")
    ap.add_argument("--no-family-non-regression", action="store_true")
    ap.add_argument("--no-overall-non-regression", action="store_true")
    ap.add_argument("--json-out", required=True, help="Output JSON path")
    ap.add_argument("--md-out", required=True, help="Output markdown path")
    args = ap.parse_args()

    families = tuple(str(item).strip() for item in args.families if str(item).strip())
    hidden_splits = tuple(str(item).strip() for item in args.hidden_split if str(item).strip())
    current_path = Path(args.current).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()

    current_doc = load_json(current_path)
    baseline_doc = load_json(baseline_path)
    if not isinstance(current_doc, dict):
        raise SystemExit(f"failed to load current probe report: {current_path}")
    if not isinstance(baseline_doc, dict):
        raise SystemExit(f"failed to load baseline probe report: {baseline_path}")

    current_metrics = probe_metrics_from_doc(current_doc, families=families, hidden_splits=hidden_splits)
    baseline_metrics = probe_metrics_from_doc(baseline_doc, families=families, hidden_splits=hidden_splits)
    payload = build_pilot_gate_payload(
        spec=str(args.spec),
        baseline_probe=baseline_path,
        current_probe=current_path,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        families=families,
        hidden_splits=hidden_splits,
        require_hidden_non_regression=not bool(args.no_hidden_non_regression),
        require_family_non_regression=not bool(args.no_family_non_regression),
        require_overall_non_regression=not bool(args.no_overall_non_regression),
        improve_families=tuple(str(item).strip() for item in args.improve_family if str(item).strip()),
    )

    json_out = Path(args.json_out).expanduser().resolve()
    md_out = Path(args.md_out).expanduser().resolve()
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(render_pilot_gate_md(payload, families=families), encoding="utf-8")
    print(json.dumps({"clears_gate": payload["clears_gate"], "checks": payload["checks"], "reasons": payload["reasons"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
