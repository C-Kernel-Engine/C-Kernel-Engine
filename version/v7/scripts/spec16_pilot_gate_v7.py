#!/usr/bin/env python3
"""Compare a spec16 pilot probe report against a frozen baseline."""

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


FAMILIES = ("memory_map", "timeline", "system_diagram")
HIDDEN_SPLITS = ("hidden_train", "hidden_test")


def _render_md(payload: dict[str, object]) -> str:
    return render_pilot_gate_md(payload, families=FAMILIES).replace(
        "# Training Pilot Gate",
        "# Spec16 Pilot Gate",
        1,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current", required=True, help="Current spec16 probe report JSON")
    ap.add_argument("--baseline", required=True, help="Frozen-baseline spec16 probe report JSON")
    ap.add_argument("--json-out", required=True, help="Output JSON path")
    ap.add_argument("--md-out", required=True, help="Output markdown path")
    args = ap.parse_args()

    current_path = Path(args.current).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    current_doc = load_json(current_path)
    baseline_doc = load_json(baseline_path)
    if not isinstance(current_doc, dict):
        raise SystemExit(f"failed to load current probe report: {current_path}")
    if not isinstance(baseline_doc, dict):
        raise SystemExit(f"failed to load baseline probe report: {baseline_path}")

    current_metrics = probe_metrics_from_doc(current_doc, families=FAMILIES, hidden_splits=HIDDEN_SPLITS)
    baseline_metrics = probe_metrics_from_doc(baseline_doc, families=FAMILIES, hidden_splits=HIDDEN_SPLITS)
    payload = build_pilot_gate_payload(
        spec="spec16",
        baseline_probe=baseline_path,
        current_probe=current_path,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        families=FAMILIES,
        hidden_splits=HIDDEN_SPLITS,
        require_hidden_non_regression=True,
        require_family_non_regression=True,
        require_overall_non_regression=True,
        improve_families=("system_diagram",),
    )
    payload["schema"] = "ck.spec16_pilot_gate.v1"

    json_out = Path(args.json_out).expanduser().resolve()
    md_out = Path(args.md_out).expanduser().resolve()
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(_render_md(payload), encoding="utf-8")
    print(json.dumps({"clears_gate": payload["clears_gate"], "checks": payload["checks"], "reasons": payload["reasons"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
