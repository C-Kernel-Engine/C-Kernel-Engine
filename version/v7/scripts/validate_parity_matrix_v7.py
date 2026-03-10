#!/usr/bin/env python3
"""
Strict v7 parity matrix validator (Gemma/Qwen2/Qwen3).

This validator enforces all-layer parity by running detailed parity analysis
per model and then applying strict criteria:
1) parity prefill/decode return codes must be zero
2) parity summaries must not contain FAIL/ERROR
3) first_issue for prefill/decode must be null
4) optional coverage check: no missing CK dump points
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "version" / "v7" / "scripts"
DETAIL = SCRIPTS / "detailed_parity_analysis.py"


DEFAULT_MODELS = [
    {
        "name": "qwen2-0.5b",
        "family": "qwen2",
        "uri": "hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf",
    },
    {
        "name": "qwen3-0.6b",
        "family": "qwen3",
        "uri": "hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
    },
    {
        "name": "gemma3-270m",
        "family": "gemma",
        "uri": "hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf",
    },
]


@dataclass
class ParityRow:
    model: str
    family: str
    cached: str
    runtime: str
    parity: str
    coverage: str
    overall: str
    note: str
    work_dir: str
    seconds: float


def _cache_dir() -> Path:
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        path = Path(env).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    default = Path.home() / ".cache" / "ck-engine-v7" / "models"
    default.mkdir(parents=True, exist_ok=True)
    return default


def _cache_candidates() -> list[Path]:
    out = [_cache_dir()]
    fallback = ROOT / ".ck_cache"
    if fallback not in out:
        out.append(fallback)
    return out


def _parse_hf_gguf(uri: str) -> tuple[str, str] | None:
    if not (uri.startswith("hf://") and uri.endswith(".gguf")):
        return None
    body = uri[len("hf://") :]
    parts = body.split("/")
    if len(parts) < 3:
        return None
    repo = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    return repo, filename


def _cached_gguf_path(uri: str) -> Path | None:
    parsed = _parse_hf_gguf(uri)
    if parsed is None:
        p = Path(uri).expanduser()
        return p if p.exists() else None
    repo, filename = parsed
    rel = Path(repo.replace("/", "--")) / Path(filename).name
    for base in _cache_candidates():
        p = base / rel
        if p.exists():
            return p
    return None


def _is_cached(uri: str) -> bool:
    return _cached_gguf_path(uri) is not None


def _work_dir_for_uri(uri: str) -> Path:
    parsed = _parse_hf_gguf(uri)
    if parsed is not None:
        repo, _ = parsed
        return _cache_dir() / repo.replace("/", "--") / "ck_build"
    p = Path(uri).expanduser()
    if p.is_file() and p.suffix.lower() == ".gguf":
        return _cache_dir() / p.stem / "ck_build"
    if p.is_dir():
        return p / "ck_build"
    return _cache_dir() / "unknown" / "ck_build"


def _find_llama_runtime() -> Path | None:
    candidates = [
        ROOT / "build" / "llama-parity",
        ROOT / "llama.cpp" / "build" / "bin" / "llama-completion",
        ROOT / "llama.cpp" / "build" / "bin" / "llama-cli",
        ROOT / "llama.cpp" / "main",
        ROOT / "llama.cpp" / "build" / "bin" / "main",
    ]
    return next((p for p in candidates if p.exists()), None)


def _have_llama_runtime() -> bool:
    return _find_llama_runtime() is not None


def _status_order(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "SKIP": 1, "FAIL": 2}.get(status, 2)


def _join_status(*statuses: str) -> str:
    worst = "PASS"
    for s in statuses:
        if _status_order(s) > _status_order(worst):
            worst = s
    return worst


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=None if timeout <= 0 else timeout,
    )


def _last_error_line(proc: subprocess.CompletedProcess[str]) -> str:
    text = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
    if not text:
        return "no-output"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "Error:" in ln or "FAIL" in ln or "failed" in ln.lower():
            return ln[:220]
    return lines[-1][:220]


def _int(d: dict, key: str) -> int:
    try:
        return int(d.get(key, 0))
    except Exception:
        return 0


def _run_model(model: dict[str, str], args: argparse.Namespace, runtime_ok: bool) -> ParityRow:
    uri = model["uri"]
    name = model["name"]
    family = model["family"]
    run_input = uri
    work_dir = _work_dir_for_uri(uri)
    cached = _is_cached(uri)
    cached_status = "YES" if cached else "NO"

    if not runtime_ok:
        return ParityRow(
            model=name,
            family=family,
            cached=cached_status,
            runtime="SKIP",
            parity="SKIP",
            coverage="SKIP",
            overall="SKIP",
            note="llama runtime missing (build/llama-parity, llama.cpp/build/bin/llama-cli, llama.cpp/build/bin/llama-completion, or llama.cpp/main)",
            work_dir=str(work_dir),
            seconds=0.0,
        )

    if (not cached) and (not args.allow_download):
        return ParityRow(
            model=name,
            family=family,
            cached=cached_status,
            runtime="PASS",
            parity="SKIP",
            coverage="SKIP",
            overall="SKIP",
            note="not-cached (use --allow-download)",
            work_dir=str(work_dir),
            seconds=0.0,
        )

    local_gguf = _cached_gguf_path(uri)
    if local_gguf is not None and local_gguf.suffix.lower() == ".gguf":
        run_input = str(local_gguf)
        work_dir = _cache_dir() / local_gguf.stem / "ck_build"

    cmd = [
        sys.executable,
        str(DETAIL),
        "--model-uri",
        run_input,
        "--family",
        family,
        "--output-dir",
        str(work_dir),
        "--prompt",
        args.prompt,
        "--context-len",
        str(args.context_len),
        "--max-tokens",
        str(args.max_tokens),
        "--report-prefix",
        args.report_prefix,
    ]
    if args.force_compile:
        cmd.append("--force-compile")
    if args.llama_timeout > 0:
        cmd.extend(["--llama-timeout", str(args.llama_timeout)])

    t0 = time.time()
    try:
        proc = _run(cmd, timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        return ParityRow(
            model=name,
            family=family,
            cached=cached_status,
            runtime="PASS",
            parity="FAIL",
            coverage="N/A",
            overall="FAIL",
            note=f"timeout>{args.timeout_sec}s",
            work_dir=str(work_dir),
            seconds=time.time() - t0,
        )

    report_json = work_dir / f"{args.report_prefix}.json"
    if not report_json.exists():
        return ParityRow(
            model=name,
            family=family,
            cached=cached_status,
            runtime="PASS",
            parity="FAIL",
            coverage="N/A",
            overall="FAIL",
            note=f"missing report: {report_json.name}; {_last_error_line(proc)}",
            work_dir=str(work_dir),
            seconds=time.time() - t0,
        )

    try:
        report = json.loads(report_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return ParityRow(
            model=name,
            family=family,
            cached=cached_status,
            runtime="PASS",
            parity="FAIL",
            coverage="N/A",
            overall="FAIL",
            note=f"invalid report json: {exc}",
            work_dir=str(work_dir),
            seconds=time.time() - t0,
        )

    parity = report.get("parity", {}) if isinstance(report, dict) else {}
    p_sum = parity.get("prefill_summary", {}) if isinstance(parity, dict) else {}
    d_sum = parity.get("decode_summary", {}) if isinstance(parity, dict) else {}
    prefill_rc = _int(parity, "prefill_rc")
    decode_rc = _int(parity, "decode_rc")
    prefill_issue = parity.get("prefill_first_issue")
    decode_issue = parity.get("decode_first_issue")

    fail_err = _int(p_sum, "FAIL") + _int(p_sum, "ERROR") + _int(d_sum, "FAIL") + _int(d_sum, "ERROR")
    warn_count = _int(p_sum, "WARN") + _int(d_sum, "WARN")
    parity_ok = (
        proc.returncode == 0
        and prefill_rc == 0
        and decode_rc == 0
        and fail_err == 0
        and prefill_issue is None
        and decode_issue is None
    )
    if args.fail_on_warn and warn_count > 0:
        parity_ok = False
    parity_status = "PASS" if parity_ok else "FAIL"

    coverage = report.get("ck_coverage", {}) if isinstance(report, dict) else {}
    missing = _int(coverage, "missing_count")
    expected_pts = _int(coverage, "expected_points")
    coverage_status = "PASS"
    if args.require_coverage and expected_pts > 0 and missing > 0:
        coverage_status = "FAIL"

    note_parts = []
    if run_input != uri:
        note_parts.append("offline-run=local-gguf")
    if parity_status != "PASS":
        note_parts.append(
            f"prefill_rc={prefill_rc},decode_rc={decode_rc},fail_err={fail_err},warn={warn_count}"
        )
        if prefill_issue is not None:
            note_parts.append("prefill_first_issue")
        if decode_issue is not None:
            note_parts.append("decode_first_issue")
    if coverage_status != "PASS":
        note_parts.append(f"coverage-missing={missing}")
    if proc.returncode != 0:
        note_parts.append(_last_error_line(proc))
    if not note_parts:
        note_parts.append("strict-parity-ok")

    overall = _join_status("PASS", parity_status, coverage_status)
    return ParityRow(
        model=name,
        family=family,
        cached=cached_status,
        runtime="PASS",
        parity=parity_status,
        coverage=coverage_status,
        overall=overall,
        note="; ".join(note_parts),
        work_dir=str(work_dir),
        seconds=time.time() - t0,
    )


def _table(rows: list[ParityRow]) -> str:
    headers = ["Model", "Family", "Cached", "Runtime", "Parity", "Coverage", "Overall", "Sec", "Note"]
    vals = [headers]
    for r in rows:
        vals.append([
            r.model,
            r.family,
            r.cached,
            r.runtime,
            r.parity,
            r.coverage,
            r.overall,
            f"{r.seconds:.1f}",
            r.note,
        ])
    widths = [max(len(str(row[i])) for row in vals) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt(headers), sep]
    for row in vals[1:]:
        out.append(fmt(row))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict v7 parity matrix validator")
    ap.add_argument("--allow-download", action="store_true", help="Allow HF download when cache is missing")
    ap.add_argument("--force-compile", action="store_true", help="Force regenerate/recompile before parity")
    ap.add_argument("--prompt", default="Hello", help="Prompt seed for parity dump")
    ap.add_argument("--context-len", type=int, default=512, help="Context len for parity run")
    ap.add_argument("--max-tokens", type=int, default=1, help="Decode tokens for parity dump")
    ap.add_argument("--timeout-sec", type=int, default=3600, help="Per-model timeout")
    ap.add_argument("--llama-timeout", type=int, default=0, help="Optional llama exhaustive timeout")
    ap.add_argument("--require-runtime", action="store_true", help="Fail if llama runtime is missing")
    ap.add_argument("--require-all", action="store_true", help="Fail if any model is skipped")
    ap.add_argument("--require-coverage", action="store_true", default=True, help="Require full CK dump coverage")
    ap.add_argument("--no-require-coverage", dest="require_coverage", action="store_false")
    ap.add_argument("--fail-on-warn", action="store_true", help="Treat WARN parity statuses as fail")
    ap.add_argument("--report-prefix", default="detailed_parity_analysis")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path")
    args = ap.parse_args()

    runtime_ok = _have_llama_runtime()
    if args.require_runtime and not runtime_ok:
        print(
            "ERROR: llama runtime missing "
            "(need build/llama-parity, llama.cpp/build/bin/llama-cli, "
            "llama.cpp/build/bin/llama-completion, or llama.cpp/main)"
        )
        return 1

    rows = [_run_model(m, args, runtime_ok) for m in DEFAULT_MODELS]

    print("=" * 156)
    print("v7 STRICT PARITY MATRIX REPORT")
    print("=" * 156)
    print(f"Runtime available: {'YES' if runtime_ok else 'NO'}")
    print(f"Models: {', '.join(m['name'] for m in DEFAULT_MODELS)}")
    print(_table(rows))
    print("=" * 156)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.overall] = counts.get(r.overall, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Summary: {summary if summary else 'none'}")

    if args.json_out is not None:
        payload = {
            "runtime_available": runtime_ok,
            "rows": [
                {
                    "model": r.model,
                    "family": r.family,
                    "cached": r.cached,
                    "runtime": r.runtime,
                    "parity": r.parity,
                    "coverage": r.coverage,
                    "overall": r.overall,
                    "note": r.note,
                    "work_dir": r.work_dir,
                    "seconds": r.seconds,
                }
                for r in rows
            ],
            "summary": counts,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    failed = any(r.overall == "FAIL" for r in rows)
    skipped = any(r.overall == "SKIP" for r in rows)

    if failed:
        return 1
    if args.require_all and skipped:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
