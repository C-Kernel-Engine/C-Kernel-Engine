#!/usr/bin/env python3
"""Run the artifact-backed Gemma4 nightly certification lane."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODEL = "hf://unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
DEFAULT_TOKENS = "2,46762,786,496,9813,2591,529,565,3393,236761"

sys.path.insert(0, str(SCRIPT_DIR))
import run_regression_v8 as regression  # noqa: E402
from verify_gemma4_runtime_contract_v8 import verify_runtime  # noqa: E402


def _run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line)
        print(line, end="", flush=True)
    return subprocess.CompletedProcess(command, process.wait(), "".join(output), "")


def _latest_family_report(report_root: Path) -> Path:
    reports = sorted(report_root.glob("*/gemma4/family_summary.json"))
    if not reports:
        raise RuntimeError(f"Gemma4 regression did not publish a family report under {report_root}")
    return reports[-1]


def _resolve_runtime_dir(run_dir: Path) -> Path:
    runtime_dir = regression._resolve_runtime_dir(run_dir)
    if not (runtime_dir / "libmodel.so").exists():
        raise RuntimeError(f"Gemma4 generated runtime is missing under {run_dir}")
    return runtime_dir


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _write_family_manifest(path: Path, context_len: int) -> None:
    source = ROOT / "version/v8/regression/families_gemma4_certification.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    families = payload.get("families") or []
    if len(families) != 1 or families[0].get("id") != "gemma4":
        raise RuntimeError("Gemma4 family manifest must contain exactly one Gemma4 entry")
    families[0]["context_len"] = int(context_len)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _host_metadata() -> dict[str, Any]:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    model_name = ""
    vendor_id = ""
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip() == "model name" and not model_name:
            model_name = value.strip()
        elif key.strip() == "vendor_id" and not vendor_id:
            vendor_id = value.strip()
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "processor": model_name,
        "vendor_id": vendor_id,
        "logical_cpus": os.cpu_count(),
    }


def _available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("V8_GEMMA4_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CK_GEMMA4_CERTIFICATION_ROOT",
                Path.home() / ".cache/ck-engine-v8/nightly/gemma4-certification",
            )
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "build" / "v8_gemma4_certification" / "summary.json",
    )
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeatability-runs", type=int, default=3)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--min-memory-gb", type=int, default=50)
    args = parser.parse_args()

    if (
        args.context_len <= 0
        or args.threads <= 0
        or args.repeatability_runs <= 0
        or args.min_memory_gb < 0
    ):
        parser.error(
            "context length, threads, and repeatability runs must be positive; "
            "minimum memory cannot be negative"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    work_root = args.work_root.expanduser().resolve() / run_id
    run_root = work_root / "runtime"
    quality_root = work_root / "quality"
    run_dir = run_root / "gemma4"
    family_manifest = work_root / "families_gemma4_certification.json"
    env = os.environ.copy()
    env.update(
        {
            "V8_GEMMA4_MODEL": str(args.model),
            "CK_NUM_THREADS": str(args.threads),
            "OMP_NUM_THREADS": "1",
        }
    )

    host = _host_metadata()
    available_memory = _available_memory_bytes()
    required_memory = int(args.min_memory_gb) * (1 << 30)
    report: dict[str, Any] = {
        "schema": "cke.v8.gemma4_certification",
        "schema_version": 1,
        "status": "FAIL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "threads": int(args.threads),
        "context_len": int(args.context_len),
        "work_root": str(work_root),
        "host": host,
        "memory_preflight": {
            "available_bytes": available_memory,
            "required_bytes": required_memory,
        },
    }
    _write_summary(args.report, report)
    if available_memory < required_memory:
        report["status"] = "SKIP"
        report["reason"] = (
            f"Gemma4 certification needs {args.min_memory_gb} GiB MemAvailable; "
            f"found {available_memory / (1 << 30):.1f} GiB"
        )
        _write_summary(args.report, report)
        print(f"SKIP: {report['reason']}")
        return 0

    _write_family_manifest(family_manifest, int(args.context_len))

    quality_command = [
        sys.executable,
        str(SCRIPT_DIR / "run_regression_v8.py"),
        "--mode",
        "full",
        "--family",
        "gemma4",
        "--families-manifest",
        str(family_manifest),
        "--run-root",
        str(run_root),
        "--report-root",
        str(quality_root),
        "--force-rebuild",
    ]
    quality = _run(quality_command, env)
    report["quality_command"] = quality_command
    report["quality_returncode"] = quality.returncode
    if quality.returncode != 0:
        report["reason"] = "Gemma4 quality/repeatability regression failed"
        _write_summary(args.report, report)
        return quality.returncode

    family_report_path = _latest_family_report(quality_root)
    family_report = json.loads(family_report_path.read_text(encoding="utf-8"))
    report["quality_report"] = str(family_report_path)
    report["quality_status"] = family_report.get("status")
    if family_report.get("status") != "PASS":
        report["reason"] = f"Gemma4 quality gate reported {family_report.get('status')}"
        _write_summary(args.report, report)
        return 2

    runtime_dir = _resolve_runtime_dir(run_dir)
    contract_path = work_root / "runtime_contract.json"
    try:
        contract = verify_runtime(runtime_dir)
    except (OSError, ValueError) as exc:
        report["reason"] = f"Gemma4 runtime contract failed: {exc}"
        _write_summary(args.report, report)
        return 2
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    report["runtime_dir"] = str(runtime_dir)
    report["runtime_contract"] = str(contract_path)
    report["memory_plan"] = {
        phase: contract["phases"][phase]["memory"]
        for phase in ("prefill", "decode")
    }

    gguf = regression._resolve_gguf_path(str(args.model))
    if gguf is None:
        report["reason"] = "unable to resolve the cached Gemma4 GGUF after conversion"
        _write_summary(args.report, report)
        return 2

    logits_path = work_root / "first_token_parity.json"
    logits_command = [
        sys.executable,
        str(SCRIPT_DIR / "compare_first_token_logits_v8.py"),
        "--model-dir",
        str(runtime_dir),
        "--gguf",
        str(gguf),
        "--tokens",
        str(args.tokens),
        "--ctx-len",
        str(args.context_len),
        "--threads",
        str(args.threads),
        "--top-k",
        "16",
        "--min-topk-overlap",
        "0.5",
        "--ck-repeatability-runs",
        str(args.repeatability_runs),
        "--json-out",
        str(logits_path),
    ]
    logits = _run(logits_command, env)
    report["first_token_command"] = logits_command
    report["first_token_returncode"] = logits.returncode
    report["first_token_report"] = str(logits_path)
    if logits.returncode != 0:
        report["reason"] = "Gemma4 first-token parity/repeatability failed"
        _write_summary(args.report, report)
        return logits.returncode

    logits_report = json.loads(logits_path.read_text(encoding="utf-8"))
    report["first_token"] = {
        "status": logits_report.get("status"),
        "ck_logits_sha256": (logits_report.get("ck") or {}).get("logits_sha256"),
        "repeatability": (logits_report.get("ck") or {}).get("repeatability"),
        "compare": logits_report.get("compare"),
    }
    report["status"] = "PASS"
    _write_summary(args.report, report)
    print(
        "Gemma4 nightly: PASS "
        f"quality={family_report.get('status')} "
        f"widths={contract.get('attention_output_widths')} "
        f"logits={report['first_token']['ck_logits_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
