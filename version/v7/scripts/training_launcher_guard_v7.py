#!/usr/bin/env python3
"""Block training launch when required metadata or policy artifacts are missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CANONICAL_TRAIN_ROOT = (Path.home() / ".cache" / "ck-engine-v7" / "models" / "train").resolve()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _check_training_plan(plan: dict[str, Any], *, require_run_policy: bool, require_token_budget: bool, require_canary_metadata: bool) -> list[str]:
    reasons: list[str] = []
    run_policy = plan.get("run_policy") if isinstance(plan.get("run_policy"), dict) else None
    token_budget = plan.get("token_budget") if isinstance(plan.get("token_budget"), dict) else None

    if require_run_policy and run_policy is None:
        reasons.append("training_plan.json is missing run_policy")
    if require_token_budget and token_budget is None:
        reasons.append("training_plan.json is missing token_budget")

    if require_canary_metadata:
        if run_policy is None:
            reasons.append("canary metadata is missing because run_policy is absent")
        else:
            mode = str(run_policy.get("mode") or "").strip().lower()
            if mode not in {"pilot", "canary", "full"}:
                reasons.append("run_policy.mode must be one of pilot, canary, or full")
        if token_budget is None:
            reasons.append("canary metadata is missing because token_budget is absent")
        else:
            required_budget_keys = (
                "recommended_pretrain_total_tokens",
                "selected_pretrain_total_tokens",
                "recommended_midtrain_total_tokens",
                "selected_midtrain_total_tokens",
            )
            for key in required_budget_keys:
                if key not in token_budget:
                    reasons.append(f"token_budget is missing {key}")
            if "canary_token_fraction" not in token_budget:
                pilot_fraction = run_policy.get("pilot_token_fraction") if isinstance(run_policy, dict) else None
                if not isinstance(pilot_fraction, dict) or "numerator" not in pilot_fraction or "denominator" not in pilot_fraction:
                    reasons.append("canary metadata is missing token fraction information")
    return reasons


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="Run directory")
    ap.add_argument("--preflight", default=None, help="Required preflight JSON")
    ap.add_argument("--blueprint", default=None, help="Required blueprint JSON")
    ap.add_argument("--blueprint-audit", default=None, help="Required blueprint audit JSON")
    ap.add_argument("--decision-artifact", default=None, help="Optional decision artifact JSON")
    ap.add_argument("--allow-decision-override", action="store_true")
    ap.add_argument("--require-run-scope", action="store_true")
    ap.add_argument("--require-run-policy", action="store_true")
    ap.add_argument("--require-token-budget", action="store_true")
    ap.add_argument("--require-canary-metadata", action="store_true")
    ap.add_argument(
        "--allow-non-cache-run-dir",
        action="store_true",
        help="Permit run dirs outside the canonical ~/.cache/ck-engine-v7/models/train root.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    reasons: list[str] = []

    if not bool(args.allow_non_cache_run_dir) and not _path_is_within(run_dir, CANONICAL_TRAIN_ROOT):
        reasons.append(
            "run dir must live under the canonical cache train root: "
            f"{CANONICAL_TRAIN_ROOT}"
        )

    training_plan_path = run_dir / "training_plan.json"
    training_plan = _load_json(training_plan_path)
    if training_plan is None:
        reasons.append(f"missing or unreadable training_plan.json at {training_plan_path}")
    else:
        reasons.extend(
            _check_training_plan(
                training_plan,
                require_run_policy=bool(args.require_run_policy),
                require_token_budget=bool(args.require_token_budget),
                require_canary_metadata=bool(args.require_canary_metadata),
            )
        )
        if bool(args.require_run_scope):
            run_scope = training_plan.get("run_scope")
            if not isinstance(run_scope, dict):
                reasons.append("training_plan.json is missing embedded run_scope")

    if bool(args.require_run_scope):
        run_scope_path = run_dir / "run_scope.json"
        if _load_json(run_scope_path) is None:
            reasons.append(f"missing or unreadable run_scope.json at {run_scope_path}")

    if args.preflight:
        preflight_path = Path(args.preflight).expanduser().resolve()
        preflight = _load_json(preflight_path)
        if preflight is None:
            reasons.append(f"missing or unreadable preflight JSON at {preflight_path}")
        else:
            if "canary" not in preflight:
                reasons.append("preflight JSON is missing canary metadata")

    if args.blueprint:
        blueprint_path = Path(args.blueprint).expanduser().resolve()
        blueprint = _load_json(blueprint_path)
        if blueprint is None:
            reasons.append(f"missing or unreadable blueprint JSON at {blueprint_path}")

    if args.blueprint_audit:
        audit_path = Path(args.blueprint_audit).expanduser().resolve()
        audit = _load_json(audit_path)
        if audit is None:
            reasons.append(f"missing or unreadable blueprint audit JSON at {audit_path}")
        else:
            verdict = str(audit.get("verdict") or "").strip().lower()
            if verdict != "pass":
                reasons.append(f"blueprint audit verdict must be pass, found {verdict or '<empty>'}")

    if args.decision_artifact and not bool(args.allow_decision_override):
        decision_path = Path(args.decision_artifact).expanduser().resolve()
        decision = _load_json(decision_path)
        if decision is None:
            reasons.append(f"missing or unreadable decision artifact at {decision_path}")
        elif not bool(decision.get("training_allowed")):
            reasons.append(
                "training blocked by decision artifact: "
                f"default_action={decision.get('default_action')} reasons={decision.get('reasons')}"
            )

    payload = {
        "ok": not reasons,
        "run_dir": str(run_dir),
        "reasons": reasons,
    }
    print(json.dumps(payload, separators=(",", ":")))
    if reasons:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
