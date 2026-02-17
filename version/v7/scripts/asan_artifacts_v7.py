#!/usr/bin/env python3
"""
Generate asan_summary.json for v7 IR visualizer.

Consumes memory_verification_latest.json (PR3.7 report) and emits a compact
summary suitable for dashboard cards.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_if_exists(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return None


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_check_summary(verify_payload: dict) -> Dict[str, object]:
    checks = verify_payload.get("checks") if isinstance(verify_payload, dict) else {}
    if not isinstance(checks, dict):
        checks = {}
    norm: Dict[str, dict] = {}
    passed = 0
    total = 0
    failed = []
    for name, row in checks.items():
        if not isinstance(row, dict):
            continue
        ok = bool(row.get("ok"))
        total += 1
        if ok:
            passed += 1
        else:
            failed.append(str(name))
        norm[str(name)] = {
            "ok": ok,
            "summary": str(row.get("summary", "")) if row.get("summary") is not None else "",
        }
    return {
        "checks": norm,
        "passed_checks": passed,
        "total_checks": total,
        "failed_checks": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate asan_summary.json for v7")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--verify-report", type=Path, help="memory_verification_latest.json path")
    parser.add_argument("--memory-diagnostic", type=Path, help="memory_diagnostic_latest.json path")
    args = parser.parse_args()

    verify = read_json_if_exists(args.verify_report)
    diag = read_json_if_exists(args.memory_diagnostic)
    check_summary = build_check_summary(verify or {})

    artifacts = []
    if args.verify_report and args.verify_report.exists():
        artifacts.append({"label": "memory verification", "path": str(args.verify_report)})
    if args.memory_diagnostic and args.memory_diagnostic.exists():
        artifacts.append({"label": "memory diagnostic", "path": str(args.memory_diagnostic)})

    payload: Dict[str, object] = {
        "generated_at": utc_now_iso(),
        "verify_report_path": str(args.verify_report) if args.verify_report else None,
        "memory_diagnostic_path": str(args.memory_diagnostic) if args.memory_diagnostic else None,
        "overall_ok": bool((verify or {}).get("ok")) if isinstance(verify, dict) else False,
        "checks": check_summary.get("checks", {}),
        "passed_checks": check_summary.get("passed_checks", 0),
        "total_checks": check_summary.get("total_checks", 0),
        "failed_checks": check_summary.get("failed_checks", []),
        "diagnostic": (diag or {}).get("diagnostic") if isinstance(diag, dict) else None,
        "artifacts": artifacts,
    }

    out_path = args.out_dir / "asan_summary.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
