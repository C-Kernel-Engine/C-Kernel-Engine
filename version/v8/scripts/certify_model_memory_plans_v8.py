#!/usr/bin/env python3
"""Certify model memory plans without compiling or executing model code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
V8_ROOT = SCRIPT_DIR.parent
DEFAULT_FAMILIES = V8_ROOT / "regression" / "families.json"
DEFAULT_REPORT = V8_ROOT / ".cache" / "reports" / "model_memory_plans_latest.json"
DEFAULT_BASELINE = V8_ROOT / "contracts" / "model_memory_plan_baseline.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import ck_run_v8  # noqa: E402


OPTIONAL_MODELS = (
    ("gemma4", "Gemma4", "V8_GEMMA4_MODEL"),
    ("glm4", "GLM-4", "V8_GLM4_MODEL"),
    ("kimi", "Kimi", "V8_KIMI_MODEL"),
    ("qwen36", "Qwen3.6", "V8_QWEN36_MODEL"),
    ("instella", "Instella-MoE", "V8_INSTELLA_MODEL"),
    ("qwen35_moe", "Qwen3.5 MoE", "V8_QWEN35_MOE_MODEL"),
)


def _load_models(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "id": str(row["id"]),
            "label": str(row.get("label") or row["id"]),
            "model": str(row["model"]),
            "required": True,
        }
        for row in payload.get("families", [])
        if isinstance(row, dict) and row.get("enabled", True) and row.get("model")
    ]
    for model_id, label, env_name in OPTIONAL_MODELS:
        model = os.environ.get(env_name, "").strip()
        rows.append(
            {
                "id": model_id,
                "label": label,
                "model": model,
                "required": False,
                "model_env": env_name,
            }
        )
    return rows


def _run_dir(model: str) -> Path:
    input_type, info = ck_run_v8.detect_input_type(model)
    return ck_run_v8._resolve_run_dir(model, input_type, info, None)


def _phase_evidence(path: Path, lowered_path: Path, phase: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lowered = json.loads(lowered_path.read_text(encoding="utf-8"))
    validation = (payload.get("validation") or {}).get("activation_memory") or {}
    writes = validation.get("writes") if isinstance(validation.get("writes"), list) else []
    writable_operations = sum(
        bool(operation.get("outputs"))
        for operation in lowered.get("operations", [])
        if isinstance(operation, dict)
    )
    for write in writes:
        required = int(write.get("required_bytes", -1))
        available = int(write.get("available_bytes", -1))
        if required < 0 or available < 0 or required > available:
            raise RuntimeError(
                f"{phase} planner evidence contains an invalid write extent: {write}"
            )
    if validation.get("status") != "PASS":
        raise RuntimeError(f"{phase} activation-memory validation did not pass")
    return {
        "phase": phase,
        "status": "PASS",
        "arena_bytes": int(validation.get("arena_bytes", 0) or 0),
        "activation_buffer_count": int(validation.get("activation_buffer_count", 0) or 0),
        "writable_operation_count": writable_operations,
        "extent_validated_write_count": len(writes),
        "extent_coverage_percent": (
            round(100.0 * len(writes) / writable_operations, 2)
            if writable_operations
            else 100.0
        ),
        "max_required_bytes": max(
            (int(write.get("required_bytes", 0) or 0) for write in writes),
            default=0,
        ),
        "min_write_headroom_bytes": min(
            (
                int(write.get("available_bytes", 0) or 0)
                - int(write.get("required_bytes", 0) or 0)
                for write in writes
            ),
            default=0,
        ),
    }


def _certify(model: dict[str, Any], context_len: int) -> dict[str, Any]:
    model_spec = str(model.get("model") or "")
    if not model_spec:
        return {
            "model_id": model["id"],
            "label": model["label"],
            "context_len": context_len,
            "status": "SKIP",
            "reason": f"optional model is unset: {model.get('model_env')}",
        }

    command = [
        sys.executable,
        str(SCRIPT_DIR / "ck_run_v8.py"),
        "run",
        model_spec,
        "--plan-only",
        "--context-len",
        str(context_len),
        "--logits-layout",
        "last",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        output_tail = "\n".join(completed.stdout.splitlines()[-20:])
        if ck_run_v8.DOWNLOAD_ERROR_MARKER in completed.stdout:
            return {
                "model_id": model["id"],
                "label": model["label"],
                "context_len": context_len,
                "status": "SKIP",
                "reason": "model download unavailable",
                "output_tail": output_tail,
            }
        return {
            "model_id": model["id"],
            "label": model["label"],
            "context_len": context_len,
            "status": "FAIL",
            "reason": f"planner exited {completed.returncode}",
            "output_tail": output_tail,
        }

    run_dir = _run_dir(model_spec)
    try:
        phases = [
            _phase_evidence(
                run_dir / "layout_prefill.json",
                run_dir / "lowered_prefill.json",
                "prefill",
            ),
            _phase_evidence(
                run_dir / "layout_decode.json",
                run_dir / "lowered_decode.json",
                "decode",
            ),
        ]
    except (OSError, ValueError, RuntimeError) as exc:
        return {
            "model_id": model["id"],
            "label": model["label"],
            "context_len": context_len,
            "status": "FAIL",
            "reason": str(exc),
        }
    return {
        "model_id": model["id"],
        "label": model["label"],
        "context_len": context_len,
        "status": "PASS",
        "run_dir": str(run_dir),
        "phases": phases,
    }


def _apply_baseline(rows: list[dict[str, Any]], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    models = baseline.get("models") if isinstance(baseline.get("models"), dict) else {}
    for row in rows:
        if row.get("status") != "PASS":
            continue
        minimums = models.get(row.get("model_id"))
        if not isinstance(minimums, dict):
            continue
        failures = []
        for phase in row.get("phases", []):
            name = str(phase.get("phase"))
            minimum = int(minimums.get(name, 0) or 0)
            observed = int(phase.get("extent_validated_write_count", 0) or 0)
            if observed < minimum:
                failures.append(f"{name} writes {observed} < baseline {minimum}")
        if failures:
            row["status"] = "FAIL"
            row["reason"] = "extent coverage regressed: " + "; ".join(failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, default=DEFAULT_FAMILIES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--contexts", default=os.environ.get("CK_V8_MEMORY_PLAN_CONTEXTS", "128,1024,8192"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    contexts = sorted({int(value) for value in args.contexts.split(",") if value.strip()})
    if not contexts or any(value <= 0 for value in contexts):
        parser.error("--contexts must contain positive comma-separated integers")

    rows = [
        _certify(model, context_len)
        for model in _load_models(args.families)
        for context_len in contexts
    ]
    _apply_baseline(rows, args.baseline)
    summary = {
        "status": "FAIL" if any(row["status"] == "FAIL" for row in rows) else "PASS",
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] == "FAIL" for row in rows),
        "skipped": sum(row["status"] == "SKIP" for row in rows),
        "total": len(rows),
        "extent_validated_writes": sum(
            phase.get("extent_validated_write_count", 0)
            for row in rows
            for phase in row.get("phases", [])
        ),
        "writable_operations": sum(
            phase.get("writable_operation_count", 0)
            for row in rows
            for phase in row.get("phases", [])
        ),
    }
    report = {
        "schema": "cke.v8.model_memory_plan_matrix",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contexts": contexts,
        "summary": summary,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "model memory plans: "
        f"status={summary['status']} pass={summary['passed']} fail={summary['failed']} "
        f"skip={summary['skipped']} writes={summary['extent_validated_writes']}/"
        f"{summary['writable_operations']}"
    )
    for row in rows:
        print(f"  {row['status']:4} {row['model_id']} context={row['context_len']} {row.get('reason', '')}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
