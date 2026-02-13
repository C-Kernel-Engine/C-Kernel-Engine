#!/usr/bin/env python3
"""
v7 performance gate evaluator.

Reads profile/perf artifacts and validates model-family budgets.
Writes perf_gate_report.json and exits non-zero on budget regressions.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_model_dir_from_input(model_input: str) -> Optional[Path]:
    try:
        from ck_run_v7 import CACHE_DIR, detect_input_type  # type: ignore
    except Exception:
        return None

    input_type, info = detect_input_type(model_input)
    if input_type == "hf_gguf":
        return CACHE_DIR / info["repo_id"].replace("/", "--")
    if input_type == "hf_id":
        return CACHE_DIR / info["model_id"].replace("/", "--")
    if input_type == "gguf":
        return CACHE_DIR / info["path"].stem
    if input_type == "local_dir":
        local = Path(info["path"]).resolve()
        return local.parent if local.name == ".ck_build" else local
    if input_type == "local_config":
        cfg_parent = Path(info["path"]).resolve().parent
        return cfg_parent.parent if cfg_parent.name == ".ck_build" else cfg_parent
    return None


def _artifact_candidates(model_dir: Path, name: str) -> list[Path]:
    return [
        model_dir / name,
        model_dir / ".ck_build" / name,
    ]


def _resolve_artifact_path(model_dir: Path, name: str, explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in _artifact_candidates(model_dir, name):
        if candidate.exists():
            return candidate
    return model_dir / name


def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def classify_model_family(model_dir: Path, explicit_family: Optional[str]) -> str:
    if explicit_family:
        return explicit_family.strip().lower()
    name = model_dir.name.lower()
    if "qwen3" in name:
        return "qwen3"
    if "qwen2" in name:
        return "qwen2"
    if "gemma" in name:
        return "gemma"
    return "default"


def parse_env_float(*keys: str) -> Optional[float]:
    for key in keys:
        raw = os.environ.get(key)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw.strip())
        except ValueError:
            pass
    return None


def resolve_budgets(family: str) -> Dict[str, float]:
    base = {
        "min_decode_tok_s": 8.0,
        "min_ipc": 0.6,
        "max_cache_miss_rate": 0.25,
        "max_branch_miss_rate": 0.08,
    }
    family_defaults = {
        "qwen2": {"min_decode_tok_s": 8.0},
        "qwen3": {"min_decode_tok_s": 8.0},
        "gemma": {"min_decode_tok_s": 8.0},
    }
    if family in family_defaults:
        base.update(family_defaults[family])

    env_family = family.upper().replace(".", "_").replace("-", "_")
    overrides = {
        "min_decode_tok_s": parse_env_float(
            f"CK_V7_PERF_{env_family}_MIN_DECODE_TOK_S",
            "CK_V7_PERF_MIN_DECODE_TOK_S",
        ),
        "min_ipc": parse_env_float(
            f"CK_V7_PERF_{env_family}_MIN_IPC",
            "CK_V7_PERF_MIN_IPC",
        ),
        "max_cache_miss_rate": parse_env_float(
            f"CK_V7_PERF_{env_family}_MAX_CACHE_MISS_RATE",
            "CK_V7_PERF_MAX_CACHE_MISS_RATE",
        ),
        "max_branch_miss_rate": parse_env_float(
            f"CK_V7_PERF_{env_family}_MAX_BRANCH_MISS_RATE",
            "CK_V7_PERF_MAX_BRANCH_MISS_RATE",
        ),
    }
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
    return base


def compare_ge(value: Optional[float], threshold: float) -> Tuple[bool, str]:
    if value is None:
        return False, "missing"
    return value >= threshold, "ok" if value >= threshold else "below_threshold"


def compare_le(value: Optional[float], threshold: float) -> Tuple[bool, str]:
    if value is None:
        return False, "missing"
    return value <= threshold, "ok" if value <= threshold else "above_threshold"


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_decode_tok_s(profile: Dict) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Compute decode tokens/s from profile_summary.

    Preferred source:
      - profile entries where mode == "decode"
      - decode token count from unique token_id values in decode entries
      - decode time from sum(time_us) over decode entries

    Legacy fallback:
      - decode_tok_s = 1000 / total_ms (old profile_summary semantics)
    """
    diagnostics: Dict[str, float] = {}

    entries = profile.get("entries")
    if isinstance(entries, list) and entries:
        decode_entries = [e for e in entries if str(e.get("mode", "")) == "decode"]
        if decode_entries:
            total_decode_us = sum(_safe_float(e.get("time_us")) for e in decode_entries)
            token_ids = set()
            for e in decode_entries:
                try:
                    token_ids.add(int(e.get("token_id", 0)))
                except (TypeError, ValueError):
                    continue
            decode_tokens = len(token_ids)
            diagnostics["decode_total_us"] = total_decode_us
            diagnostics["decode_tokens"] = float(decode_tokens)
            if total_decode_us > 0 and decode_tokens > 0:
                tok_s = decode_tokens * 1_000_000.0 / total_decode_us
                diagnostics["decode_tok_s_source"] = 1.0  # entries-based
                return tok_s, diagnostics

    decode_total_ms = profile.get("total_ms")
    if isinstance(decode_total_ms, (int, float)) and float(decode_total_ms) > 0:
        diagnostics["legacy_total_ms"] = float(decode_total_ms)
        diagnostics["decode_tok_s_source"] = 0.0  # legacy fallback
        return 1000.0 / float(decode_total_ms), diagnostics

    return None, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate v7 perf budgets")
    parser.add_argument("--model-dir", type=Path, help="Model output directory")
    parser.add_argument("--model-input", type=str, help="Model input string (same as ck_run_v7.py)")
    parser.add_argument("--profile-json", type=Path, help="Path to profile_summary.json")
    parser.add_argument("--perf-stat-json", type=Path, help="Path to perf_stat_summary.json")
    parser.add_argument("--family", type=str, help="Model family override (qwen2/qwen3/gemma/default)")
    parser.add_argument("--output", type=Path, help="Output report path (default: <model-dir>/perf_gate_report.json)")
    parser.add_argument(
        "--allow-missing-metrics",
        action="store_true",
        help="Do not fail when a metric is missing; mark check as warning",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None and args.model_input:
        model_dir = detect_model_dir_from_input(args.model_input)
    if model_dir is None:
        parser.error("Provide --model-dir or --model-input")
    model_dir = model_dir.resolve()

    profile_json = _resolve_artifact_path(model_dir, "profile_summary.json", args.profile_json)
    perf_stat_json = _resolve_artifact_path(model_dir, "perf_stat_summary.json", args.perf_stat_json)
    if not profile_json.exists():
        print(f"[perf-gate] missing profile summary: {profile_json}")
        return 2
    if not perf_stat_json.exists():
        print(f"[perf-gate] missing perf stat summary: {perf_stat_json}")
        return 2

    profile = load_json(profile_json)
    perf = load_json(perf_stat_json)

    family = classify_model_family(model_dir, args.family)
    budgets = resolve_budgets(family)

    decode_tok_s, decode_diag = compute_decode_tok_s(profile)

    derived = perf.get("derived", {}) if isinstance(perf, dict) else {}
    ipc = float(derived["ipc"]) if isinstance(derived, dict) and "ipc" in derived else None
    cache_miss_rate = (
        float(derived["cache_miss_rate"])
        if isinstance(derived, dict) and "cache_miss_rate" in derived
        else None
    )
    branch_miss_rate = (
        float(derived["branch_miss_rate"])
        if isinstance(derived, dict) and "branch_miss_rate" in derived
        else None
    )

    checks = []
    ge_metrics = [
        ("decode_tok_s", decode_tok_s, budgets["min_decode_tok_s"]),
        ("ipc", ipc, budgets["min_ipc"]),
    ]
    le_metrics = [
        ("cache_miss_rate", cache_miss_rate, budgets["max_cache_miss_rate"]),
        ("branch_miss_rate", branch_miss_rate, budgets["max_branch_miss_rate"]),
    ]

    for metric, value, threshold in ge_metrics:
        ok, reason = compare_ge(value, threshold)
        checks.append(
            {
                "metric": metric,
                "value": value,
                "comparison": ">=",
                "threshold": threshold,
                "passed": ok if reason != "missing" else bool(args.allow_missing_metrics),
                "reason": reason,
            }
        )
    for metric, value, threshold in le_metrics:
        ok, reason = compare_le(value, threshold)
        checks.append(
            {
                "metric": metric,
                "value": value,
                "comparison": "<=",
                "threshold": threshold,
                "passed": ok if reason != "missing" else bool(args.allow_missing_metrics),
                "reason": reason,
            }
        )

    passed = all(bool(c["passed"]) for c in checks)
    report = {
        "generated_at": utc_now_iso(),
        "model_dir": str(model_dir),
        "family": family,
        "budgets": budgets,
        "sources": {
            "profile_summary": str(profile_json),
            "perf_stat_summary": str(perf_stat_json),
        },
        "metrics": {
            "decode_tok_s": decode_tok_s,
            "ipc": ipc,
            "cache_miss_rate": cache_miss_rate,
            "branch_miss_rate": branch_miss_rate,
            "decode_diagnostics": decode_diag,
        },
        "checks": checks,
        "passed": passed,
    }

    out_path = args.output or (model_dir / "perf_gate_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Keep root and .ck_build copies in sync for mixed legacy/new readers.
    for mirror in _artifact_candidates(model_dir, "perf_gate_report.json"):
        if mirror.resolve() == out_path.resolve():
            continue
        mirror.parent.mkdir(parents=True, exist_ok=True)
        with open(mirror, "w") as f:
            json.dump(report, f, indent=2)

    print(f"[perf-gate] model={model_dir.name} family={family}")
    print(f"[perf-gate] decode_tok_s={decode_tok_s if decode_tok_s is not None else 'NA'}")
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        value = "NA" if c["value"] is None else f"{float(c['value']):.6g}"
        print(f"  [{status}] {c['metric']} {c['comparison']} {c['threshold']:.6g} (value={value}, reason={c['reason']})")
    print(f"[perf-gate] wrote {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
