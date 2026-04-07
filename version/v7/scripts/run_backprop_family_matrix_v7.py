#!/usr/bin/env python3
"""
run_backprop_family_matrix_v7.py

Operator-facing v7 text backprop family matrix runner.

This wraps run_training_parity_regimen_v7.py and writes one consolidated report
for the active text-family surface.

Modes:
  - fast: qwen2, qwen3, gemma, nanbeige
  - full: qwen2, qwen3, gemma, nanbeige, qwen35
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
REGIMEN = SCRIPT_DIR / "run_training_parity_regimen_v7.py"

FAST_FAMILIES: tuple[str, ...] = ("qwen2", "qwen3", "gemma", "nanbeige")
FULL_FAMILIES: tuple[str, ...] = FAST_FAMILIES + ("qwen35",)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_report_root() -> Path:
    first = None
    for raw in (os.environ.get("CK_V7_REPORT_DIR"), os.environ.get("CK_CACHE_DIR")):
        if raw:
            first = Path(raw).expanduser()
            break
    if first is not None:
        base = first
        if base.name == "train":
            return base.parent / "reports" / "backprop_family_matrix"
        if base.name == "models":
            return base / "reports" / "backprop_family_matrix"
        if base.name == "reports":
            return base / "backprop_family_matrix"
        return base / "reports" / "backprop_family_matrix"
    return Path.home() / ".cache" / "ck-engine-v7" / "models" / "reports" / "backprop_family_matrix"


def _pick_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _families_for_mode(mode: str) -> list[str]:
    text = str(mode).strip().lower()
    if text == "fast":
        return list(FAST_FAMILIES)
    if text == "full":
        return list(FULL_FAMILIES)
    raise ValueError(f"unsupported mode: {mode}")


def _parse_families(raw: str | None, *, mode: str) -> list[str]:
    if raw:
        out = [str(tok).strip() for tok in str(raw).split(",") if str(tok).strip()]
        if not out:
            raise ValueError("family override is empty")
        return out
    return _families_for_mode(mode)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def _run(cmd: Sequence[str], *, env: Dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _family_report_dir(report_root: Path, family: str) -> Path:
    return report_root / family


def _family_result_from_report(family: str, report_path: Path) -> Dict[str, Any]:
    if not report_path.exists():
        return {
            "family": family,
            "status": "FAIL",
            "passed": False,
            "summary": {"failed_stage_ids": ["missing_report"]},
            "report_json": str(report_path),
        }
    doc = _load_json(report_path)
    summary = doc.get("summary") if isinstance(doc.get("summary"), dict) else {}
    return {
        "family": family,
        "status": "PASS" if bool(summary.get("passed", False)) else "FAIL",
        "passed": bool(summary.get("passed", False)),
        "summary": summary,
        "report_json": str(report_path),
        "report_md": str(report_path.with_suffix(".md")),
        "logs_dir": str(report_path.parent / "training_parity_regimen_logs"),
        "runtime_gate_profile": (doc.get("config") or {}).get("runtime_gate_profile"),
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# v7 Backprop Family Matrix")
    lines.append("")
    lines.append(f"- generated_at: `{payload['generated_at']}`")
    lines.append(f"- mode: `{payload['mode']}`")
    lines.append(f"- passed: `{payload['summary']['passed']}`")
    lines.append(f"- families: `{', '.join(payload['families'])}`")
    lines.append("")
    lines.append("| Family | Status | Failed Stages | Report |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["results"]:
        failed = row.get("summary", {}).get("failed_stage_ids", [])
        failed_text = ", ".join(str(x) for x in failed) if failed else "-"
        lines.append(
            "| %s | %s | %s | `%s` |"
            % (
                row["family"],
                row["status"],
                failed_text,
                row.get("report_json", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the v7 text backprop family matrix.")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--families", type=str, default=None, help="Optional comma list overriding --mode.")
    ap.add_argument("--python-exec", type=str, default=None)
    ap.add_argument("--report-root", type=Path, default=None, help="Per-family artifact root.")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None, help="Optional CK_CACHE_DIR for child runs.")
    ap.add_argument("--extended-checks", action="store_true", help="Enable regimen G/H/I stages for each family.")
    ap.set_defaults(memory_check=True)
    ap.add_argument("--no-memory-check", dest="memory_check", action="store_false",
                    help="Skip live host memory headroom preflight for each family regimen.")
    ap.add_argument("--memory-min-available-gb", type=float, default=6.0,
                    help="Minimum live MemAvailable GiB floor forwarded to each family regimen.")
    ap.add_argument("--extended-memory-min-available-gb", type=float, default=8.0,
                    help="Minimum live MemAvailable GiB floor used by extended family regimen runs.")
    ap.add_argument("--memory-min-available-ratio", type=float, default=0.20,
                    help="Minimum live MemAvailable / MemTotal ratio forwarded to each family regimen.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    families = _parse_families(args.families, mode=args.mode)
    python_exec = _pick_python(args.python_exec)
    if not REGIMEN.exists():
        raise SystemExit(f"missing script: {REGIMEN}")

    report_root = args.report_root.resolve() if args.report_root is not None else _default_report_root()
    json_out = args.json_out.resolve() if args.json_out is not None else (report_root / "v7_backprop_family_matrix_latest.json")
    md_out = args.md_out.resolve() if args.md_out is not None else (report_root / "v7_backprop_family_matrix_latest.md")
    report_root.mkdir(parents=True, exist_ok=True)

    child_env = dict(os.environ)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        child_env["CK_CACHE_DIR"] = str(args.cache_dir.resolve())

    results: list[Dict[str, Any]] = []
    for family in families:
        family_dir = _family_report_dir(report_root, family)
        family_dir.mkdir(parents=True, exist_ok=True)
        family_json = family_dir / "training_parity_regimen_latest.json"
        family_md = family_dir / "training_parity_regimen_latest.md"
        family_logs = family_dir / "training_parity_regimen_logs"
        cmd: list[str] = [
            python_exec,
            str(REGIMEN),
            "--family",
            str(family),
            "--python-exec",
            str(python_exec),
            "--json-out",
            str(family_json),
            "--md-out",
            str(family_md),
            "--logs-dir",
            str(family_logs),
            "--memory-min-available-gb",
            str(float(args.memory_min_available_gb)),
            "--extended-memory-min-available-gb",
            str(float(args.extended_memory_min_available_gb)),
            "--memory-min-available-ratio",
            str(float(args.memory_min_available_ratio)),
        ]
        if args.extended_checks:
            cmd.append("--extended-checks")
        if not bool(args.memory_check):
            cmd.append("--no-memory-check")
        if args.force:
            cmd.append("--force")
        print(f"[family] {family}: {' '.join(cmd)}")
        proc = _run(cmd, env=child_env)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        row = _family_result_from_report(family, family_json)
        row["rc"] = int(proc.returncode)
        results.append(row)
        if args.stop_on_fail and not row["passed"]:
            break

    failed = [row["family"] for row in results if not row["passed"]]
    payload = {
        "generated_at": _utc_now_iso(),
        "mode": str(args.mode),
        "families": list(families),
        "report_root": str(report_root),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir is not None else None,
        "extended_checks": bool(args.extended_checks),
        "summary": {
            "passed": len(failed) == 0 and len(results) == len(families),
            "failed_families": failed,
            "total_families": len(families),
            "completed_families": len(results),
            "passed_families": sum(1 for row in results if row["passed"]),
        },
        "results": results,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_out.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"[done] json={json_out}")
    print(f"[done] md={md_out}")
    return 0 if payload["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
