#!/usr/bin/env python3
"""
v6.6 Memory sign-off wrapper.

Runs:
  - test_memory_planner.py
  - advanced_memory_validator.py
  - test_kv_cache.py

and writes one normalized memory_signoff.json artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
V66_ROOT = SCRIPT_DIR.parent
TEST_DIR = V66_ROOT / "test"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_model_dir_from_input(model_input: str) -> Optional[Path]:
    try:
        from ck_run_v6_6 import CACHE_DIR, detect_input_type  # type: ignore
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
        return Path(info["path"]) / ".ck_build"
    if input_type == "local_config":
        return Path(info["path"]).parent / ".ck_build"
    return None


def parse_trailing_json_blob(output: str) -> Optional[Dict]:
    s = output.strip()
    # Try direct parse first.
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: find last JSON object in mixed stdout.
    idx = s.rfind("{")
    while idx >= 0:
        candidate = s[idx:]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            idx = s.rfind("{", 0, idx)
            continue
    return None


def run_cmd_json(cmd: list[str], cwd: Path) -> Tuple[int, str, Optional[Dict]]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    payload = parse_trailing_json_blob(proc.stdout)
    return proc.returncode, proc.stdout, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v6.6 memory sign-off and emit memory_signoff.json")
    parser.add_argument("--model-dir", type=Path, help="Model directory containing layout/lowered artifacts")
    parser.add_argument("--model-input", type=str, help="Model input string (same as ck_run_v6_6.py)")
    parser.add_argument("--json-out", type=Path, help="Output file path (default: <model-dir>/memory_signoff.json)")
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None and args.model_input:
        model_dir = detect_model_dir_from_input(args.model_input)
    if model_dir is None:
        parser.error("Provide --model-dir or --model-input")
    model_dir = model_dir.resolve()

    layout_path = model_dir / "layout_decode.json"
    ir_path = model_dir / "lowered_decode_call.json"
    if not ir_path.exists():
        ir_path = model_dir / "lowered_decode.json"
    if not layout_path.exists() or not ir_path.exists():
        print(f"Missing required artifacts in {model_dir}")
        print("Expected: layout_decode.json + lowered_decode_call.json (or lowered_decode.json)")
        return 2

    planner_cmd = [
        sys.executable,
        str(TEST_DIR / "test_memory_planner.py"),
        "--layout",
        str(layout_path),
        "--ir",
        str(ir_path),
        "--quiet",
        "--json",
    ]
    adv_cmd = [
        sys.executable,
        str(TEST_DIR / "advanced_memory_validator.py"),
        "--model-dir",
        str(model_dir),
        "--json",
    ]
    kv_cmd = [
        sys.executable,
        str(TEST_DIR / "test_kv_cache.py"),
        "--model-dir",
        str(model_dir),
        "--json",
    ]

    print(f"[memory-signoff] model_dir={model_dir}")
    planner_rc, planner_out, planner = run_cmd_json(planner_cmd, TEST_DIR)
    adv_rc, adv_out, adv = run_cmd_json(adv_cmd, TEST_DIR)
    kv_rc, kv_out, kv = run_cmd_json(kv_cmd, TEST_DIR)

    checks = {
        "memory_planner": planner or {
            "passed": planner_rc == 0,
            "errors": [f"Failed to parse JSON output (rc={planner_rc})"],
            "warnings": [],
            "raw_output": planner_out[-2000:],
        },
        "advanced_memory_validator": adv or {
            "passed": adv_rc == 0,
            "errors": [f"Failed to parse JSON output (rc={adv_rc})"],
            "warnings": [],
            "info": [],
            "raw_output": adv_out[-2000:],
        },
        "kv_cache": kv or {
            "passed": kv_rc == 0,
            "errors": [f"Failed to parse JSON output (rc={kv_rc})"],
            "warnings": [],
            "results": [],
            "raw_output": kv_out[-2000:],
        },
    }

    err_count = 0
    warn_count = 0
    for payload in checks.values():
        errs = payload.get("errors", []) if isinstance(payload, dict) else []
        warns = payload.get("warnings", []) if isinstance(payload, dict) else []
        err_count += len(errs)
        warn_count += len(warns)

    passed = all(bool(payload.get("passed")) for payload in checks.values())
    report = {
        "generated_at": utc_now_iso(),
        "model_dir": str(model_dir),
        "passed": passed,
        "error_count": err_count,
        "warning_count": warn_count,
        "checks": checks,
    }

    out_path = args.json_out or (model_dir / "memory_signoff.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    status = "PASS" if passed else "FAIL"
    print(f"[memory-signoff] {status}: errors={err_count} warnings={warn_count}")
    print(f"[memory-signoff] wrote {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

