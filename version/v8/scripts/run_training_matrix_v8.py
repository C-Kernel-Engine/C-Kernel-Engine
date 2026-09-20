#!/usr/bin/env python3
"""Run the generated-C/PyTorch training contract across model depths.

Each case owns its generated runtime, tokenizer artifacts, checkpoints, export,
and IR report. A case is accepted only when the underlying end-to-end workflow
passes and reports the requested circuit and tokenizer identities.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "version" / "v8" / "scripts" / "run_training_workflow_v8.py"
DEFAULT_ROOT = ROOT / "version" / "v8" / ".cache" / "training_matrix"
DEFAULT_REPORT = ROOT / "version" / "v8" / ".cache" / "reports" / "training_matrix_latest.json"

PROFILES: dict[str, dict[str, int]] = {
    "4l_dense": {"layers": 4, "d_model": 32, "hidden": 64, "heads": 4, "kv_heads": 4, "seq_len": 32},
    "6l_gqa": {"layers": 6, "d_model": 32, "hidden": 64, "heads": 4, "kv_heads": 2, "seq_len": 32},
    "10l_gqa": {"layers": 10, "d_model": 64, "hidden": 128, "heads": 8, "kv_heads": 2, "seq_len": 64},
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for name in args.profile:
        profile = PROFILES[name]
        run_dir = args.run_root / name
        report_path = run_dir / "training_workflow.json"
        command = [
            args.python, str(WORKFLOW), "--run-dir", str(run_dir), "--json-out", str(report_path),
            "--layers", str(profile["layers"]), "--d-model", str(profile["d_model"]),
            "--hidden", str(profile["hidden"]), "--num-heads", str(profile["heads"]),
            "--num-kv-heads", str(profile["kv_heads"]), "--seq-len", str(profile["seq_len"]),
            "--tokenizer", "bpe", "--vocab-size", str(args.vocab_size),
            "--epochs", str(args.epochs), "--grad-accum", str(args.grad_accum),
            "--max-train-tokens", str(args.max_train_tokens),
            "--max-validation-tokens", str(args.max_validation_tokens),
        ]
        case_started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        document: dict[str, Any] = {}
        if report_path.is_file():
            document = json.loads(report_path.read_text(encoding="utf-8"))
        config = document.get("configuration") if isinstance(document.get("configuration"), dict) else {}
        identity_passed = bool(
            config.get("layers") == profile["layers"]
            and config.get("heads") == profile["heads"]
            and config.get("kv_heads") == profile["kv_heads"]
            and config.get("vocab_size") == args.vocab_size
            and config.get("tokenizer") == "bpe"
        )
        passed = completed.returncode == 0 and document.get("passed") is True and identity_passed
        rows.append({
            "profile": name, "passed": passed, "returncode": completed.returncode,
            "requested": profile, "identity_passed": identity_passed,
            "report": str(report_path), "run_dir": str(run_dir),
            "wall_seconds": time.perf_counter() - case_started, "stdout_tail": completed.stdout[-4000:],
        })
        if not passed and args.fail_fast:
            break
    result = {
        "schema": "cke.v8.training_matrix.v1", "status": "PASS" if len(rows) == len(args.profile) and all(r["passed"] for r in rows) else "FAIL",
        "passed": len(rows) == len(args.profile) and all(r["passed"] for r in rows),
        "profiles_requested": args.profile, "tokenizer": "cke_true_bpe_v1", "vocab_size": args.vocab_size,
        "epochs": args.epochs, "max_train_tokens": args.max_train_tokens,
        "max_validation_tokens": args.max_validation_tokens, "cases": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(args.report, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Certify 4/6/10-layer generated-C training with a CKE BPE tokenizer")
    parser.add_argument("--profile", action="append", choices=tuple(PROFILES), help="Repeat to select cases; default is all")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json-out", dest="report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").is_file() else sys.executable)
    parser.add_argument("--vocab-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-train-tokens", type=int, default=1024)
    parser.add_argument("--max-validation-tokens", type=int, default=256)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    args.profile = args.profile or list(PROFILES)
    args.run_root = args.run_root.resolve(); args.report = args.report.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "report": str(args.report)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
