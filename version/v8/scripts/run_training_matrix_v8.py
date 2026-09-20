#!/usr/bin/env python3
"""Run the generated-C/PyTorch training contract across model depths."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "version" / "v8" / "scripts" / "run_training_workflow_v8.py"
CORPUS = ROOT / "version" / "v8" / "training" / "english_byte_v1.json"
DEFAULT_ROOT = ROOT / "version" / "v8" / ".cache" / "training_matrix"
DEFAULT_REPORT = ROOT / "version" / "v8" / ".cache" / "reports" / "training_matrix_latest.json"

PROFILES: dict[str, dict[str, int]] = {
    "4l_dense": {"layers": 4, "d_model": 32, "hidden": 64, "heads": 4, "kv_heads": 4, "seq_len": 32},
    "6l_gqa": {"layers": 6, "d_model": 32, "hidden": 64, "heads": 4, "kv_heads": 2, "seq_len": 32},
    "10l_gqa": {"layers": 10, "d_model": 64, "hidden": 128, "heads": 8, "kv_heads": 2, "seq_len": 64},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _execute(command: list[str], timeout: float) -> tuple[int, str]:
    process = subprocess.Popen(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate(process)
        remainder, _ = process.communicate()
        captured = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        captured += remainder or ""
        raise TimeoutError(f"case exceeded {timeout:g}s timeout; output={captured[-1000:]}") from exc
    return int(process.returncode), stdout or ""


def _expected_config(profile: Mapping[str, int], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "architecture": "qwen3_style_dense_reduced", "dtype": "fp32",
        "layers": profile["layers"], "d_model": profile["d_model"], "hidden": profile["hidden"],
        "heads": profile["heads"], "kv_heads": profile["kv_heads"], "seq_len": profile["seq_len"],
        "vocab_size": args.vocab_size, "tokenizer": "bpe", "epochs": args.epochs,
        "grad_accum": args.grad_accum, "optimizer": "generated_c_adamw",
        "lr": 3e-4, "beta1": .9, "beta2": .999, "eps": 1e-8, "weight_decay": .01,
    }


def _validate_case_report(
    document: Mapping[str, Any], *, expected: Mapping[str, Any], run_dir: Path,
    report_path: Path, matrix_run_id: str, case_id: str, git_commit: str,
    corpus_sha256: str, max_train_tokens: int, max_validation_tokens: int,
    not_before_ns: int = 0,
) -> list[str]:
    errors: list[str] = []
    if document.get("schema") != "cke.v8.training_workflow.v1":
        errors.append("schema")
    if document.get("status") != "PASS" or document.get("passed") is not True:
        errors.append("workflow_status")
    config = document.get("configuration") if isinstance(document.get("configuration"), Mapping) else {}
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"configuration.{key}")
    identity = document.get("matrix_identity") if isinstance(document.get("matrix_identity"), Mapping) else {}
    if identity.get("run_id") != matrix_run_id:
        errors.append("matrix_identity.run_id")
    if identity.get("case_id") != case_id:
        errors.append("matrix_identity.case_id")
    if identity.get("profile") != case_id:
        errors.append("matrix_identity.profile")
    if identity.get("run_dir") != str(run_dir):
        errors.append("matrix_identity.run_dir")
    if identity.get("report") != str(report_path):
        errors.append("matrix_identity.report")
    execution = document.get("execution") if isinstance(document.get("execution"), Mapping) else {}
    if execution.get("git_commit") != git_commit:
        errors.append("execution.git_commit")
    corpus = document.get("corpus") if isinstance(document.get("corpus"), Mapping) else {}
    if corpus.get("spec_sha256") != corpus_sha256:
        errors.append("corpus.spec_sha256")
    tokenizer = corpus.get("tokenizer") if isinstance(corpus.get("tokenizer"), Mapping) else {}
    for key, value in {
        "mode": "bpe", "name": "cke_true_bpe_v1", "vocab_size": expected["vocab_size"],
        "training_split_only": True,
    }.items():
        if tokenizer.get(key) != value:
            errors.append(f"corpus.tokenizer.{key}")
    splits = corpus.get("splits") if isinstance(corpus.get("splits"), Mapping) else {}
    for split, limit in (("train", max_train_tokens), ("validation", max_validation_tokens)):
        row = splits.get(split) if isinstance(splits.get(split), Mapping) else {}
        if limit > 0 and row.get("tokens") != limit:
            errors.append(f"corpus.splits.{split}.tokens")
        token_path = Path(str(row.get("token_ids") or ""))
        try:
            token_path.resolve().relative_to((run_dir / "dataset").resolve())
        except (ValueError, OSError):
            errors.append(f"corpus.splits.{split}.token_ids_path")
        if not token_path.is_file() or row.get("token_ids_sha256") != _sha256(token_path):
            errors.append(f"corpus.splits.{split}.token_ids_identity")
    if not report_path.is_file():
        errors.append("report_missing")
    elif report_path.stat().st_mtime_ns < not_before_ns:
        errors.append("report_stale_mtime")
    return sorted(set(errors))


def _result(
    args: argparse.Namespace, *, rows: list[dict[str, Any]], run_id: str,
    git_commit: str, started_at: str, started: float,
) -> dict[str, Any]:
    passed = len(rows) == len(args.profile) and all(row.get("passed") is True for row in rows)
    return {
        "schema": "cke.v8.training_matrix.v2", "status": "PASS" if passed else "FAIL", "passed": passed,
        "execution": {"run_id": run_id, "git_commit": git_commit, "started_at": started_at},
        "profiles_requested": list(args.profile), "tokenizer": "cke_true_bpe_v1", "vocab_size": args.vocab_size,
        "epochs": args.epochs, "grad_accum": args.grad_accum,
        "max_train_tokens": args.max_train_tokens, "max_validation_tokens": args.max_validation_tokens,
        "case_timeout_seconds": args.case_timeout, "cases": rows,
        "wall_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = str(uuid.uuid4())
    git_commit = _git_commit()
    corpus_sha256 = _sha256(args.corpus)
    rows: list[dict[str, Any]] = []
    _write_json(args.report, _result(
        args, rows=rows, run_id=run_id, git_commit=git_commit,
        started_at=started_at, started=started,
    ))
    for name in args.profile:
        profile = PROFILES[name]
        run_dir = args.run_root / name
        report_path = run_dir / "training_workflow.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            report_path.unlink()
        case_started_ns = time.time_ns()
        command = [
            args.python, str(args.workflow), "--run-dir", str(run_dir), "--json-out", str(report_path),
            "--corpus", str(args.corpus), "--matrix-run-id", run_id, "--matrix-case-id", name,
            "--layers", str(profile["layers"]), "--d-model", str(profile["d_model"]),
            "--hidden", str(profile["hidden"]), "--num-heads", str(profile["heads"]),
            "--num-kv-heads", str(profile["kv_heads"]), "--seq-len", str(profile["seq_len"]),
            "--tokenizer", "bpe", "--vocab-size", str(args.vocab_size), "--epochs", str(args.epochs),
            "--grad-accum", str(args.grad_accum), "--max-train-tokens", str(args.max_train_tokens),
            "--max-validation-tokens", str(args.max_validation_tokens),
        ]
        case_started = time.perf_counter()
        row: dict[str, Any] = {
            "profile": name, "case_id": name, "passed": False,
            "requested": _expected_config(profile, args), "report": str(report_path),
            "run_dir": str(run_dir), "command": command,
        }
        try:
            returncode, stdout = _execute(command, float(args.case_timeout))
            row.update({"returncode": returncode, "stdout_tail": stdout[-4000:]})
            try:
                parsed = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("report root is not an object")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                row.update({"failure_kind": "malformed_report", "error": str(exc)})
            else:
                errors = _validate_case_report(
                    parsed, expected=row["requested"], run_dir=run_dir, report_path=report_path,
                    matrix_run_id=run_id, case_id=name, git_commit=git_commit,
                    corpus_sha256=corpus_sha256, max_train_tokens=args.max_train_tokens,
                    max_validation_tokens=args.max_validation_tokens, not_before_ns=case_started_ns,
                )
                row["identity_errors"] = errors
                row["report_sha256"] = _sha256(report_path)
                row["passed"] = returncode == 0 and not errors
                if errors:
                    row["failure_kind"] = (
                        "stale_report" if any(error.startswith("matrix_identity") for error in errors)
                        else "identity_mismatch"
                    )
                elif returncode != 0:
                    row["failure_kind"] = "workflow_failure"
        except TimeoutError as exc:
            row.update({"failure_kind": "timeout", "error": str(exc), "returncode": None})
        except OSError as exc:
            row.update({"failure_kind": "launch_error", "error": str(exc), "returncode": None})
        except Exception as exc:
            row.update({"failure_kind": "runner_error", "error": f"{type(exc).__name__}: {exc}", "returncode": None})
        row["wall_seconds"] = time.perf_counter() - case_started
        rows.append(row)
        _write_json(args.report, _result(
            args, rows=rows, run_id=run_id, git_commit=git_commit,
            started_at=started_at, started=started,
        ))
    return _result(
        args, rows=rows, run_id=run_id, git_commit=git_commit,
        started_at=started_at, started=started,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Certify reduced 4/6/10-layer Qwen3-style generated-C training")
    parser.add_argument("--profile", action="append", choices=tuple(PROFILES), help="Repeat to select cases; default is all")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json-out", dest="report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").is_file() else sys.executable)
    parser.add_argument("--workflow", type=Path, default=WORKFLOW, help=argparse.SUPPRESS)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--vocab-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-train-tokens", type=int, default=1024)
    parser.add_argument("--max-validation-tokens", type=int, default=256)
    parser.add_argument("--case-timeout", type=float, default=1800.0)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    args.profile = args.profile or list(PROFILES)
    args.run_root = args.run_root.resolve()
    args.report = args.report.resolve()
    args.workflow = args.workflow.resolve()
    args.corpus = args.corpus.resolve()
    if args.case_timeout <= 0:
        parser.error("--case-timeout must be positive")
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "schema": "cke.v8.training_matrix.v2", "status": "FAIL", "passed": False,
            "fatal_error": f"{type(exc).__name__}: {exc}", "cases": [],
        }
    _write_json(args.report, result)
    print(json.dumps({"status": result["status"], "report": str(args.report)}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
