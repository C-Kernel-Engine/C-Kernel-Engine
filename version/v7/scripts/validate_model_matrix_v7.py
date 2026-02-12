#!/usr/bin/env python3
"""
Dynamic v7 model-matrix validator.

Runs build-phase validation across Gemma/Qwen2/Qwen3:
1) optional static tooling-contract preflight
2) per-model generate+compile (ck_run_v7.py --generate-only)
3) generated artifact presence checks
4) optional smoke checks (--test)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "version" / "v7" / "scripts"
CK_RUN = SCRIPTS / "ck_run_v7.py"
STATIC_CONTRACTS = SCRIPTS / "validate_tooling_contracts.py"


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
class MatrixRow:
    model: str
    family: str
    cached: str
    build: str
    artifacts: str
    sliding: str
    smoke: str
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
    try:
        default.mkdir(parents=True, exist_ok=True)
        probe = default / ".ck_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return default
    except Exception:
        fallback = ROOT / ".ck_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _cache_candidates() -> list[Path]:
    out: list[Path] = []
    primary = _cache_dir()
    out.append(primary)
    default = Path.home() / ".cache" / "ck-engine-v7" / "models"
    fallback = ROOT / ".ck_cache"
    for p in (default, fallback):
        if p not in out:
            out.append(p)
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


def _work_dir_for_uri(uri: str) -> Path:
    parsed = _parse_hf_gguf(uri)
    if parsed is not None:
        repo, _ = parsed
        return _cache_dir() / repo.replace("/", "--")
    p = Path(uri).expanduser()
    if p.is_file() and p.suffix.lower() == ".gguf":
        return _cache_dir() / p.stem
    if p.is_dir():
        return p / ".ck_build"
    return _cache_dir() / "unknown"


def _is_cached(uri: str) -> bool:
    return _cached_gguf_path(uri) is not None


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


def _status_order(status: str) -> int:
    # Higher is worse.
    return {
        "PASS": 0,
        "WARN": 1,
        "SKIP": 1,
        "N/A": 1,
        "FAIL": 2,
    }.get(status, 2)


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


def _artifact_status(work_dir: Path) -> tuple[str, str]:
    required = [
        "weights.bump",
        "weights_manifest.json",
        "layout_decode.json",
        "lowered_decode_call.json",
        "model_v7.c",
        "libmodel.so",
    ]
    missing = [name for name in required if not (work_dir / name).exists()]
    if missing:
        head = ",".join(missing[:3])
        extra = len(missing) - min(3, len(missing))
        suffix = "" if extra <= 0 else f",+{extra}"
        return "FAIL", f"missing={head}{suffix}"
    return "PASS", "required-artifacts-present"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _sliding_contract_status(work_dir: Path, family: str) -> tuple[str, str, str]:
    """
    Return (display_status, gate_status, note).

    display_status is shown in table (PASS/FAIL/N/A),
    gate_status is used for overall PASS/FAIL joining.
    """
    if family != "gemma":
        return "N/A", "PASS", "not-required"

    decode_path = work_dir / "lowered_decode_call.json"
    prefill_path = work_dir / "lowered_prefill_call.json"
    if not decode_path.exists() or not prefill_path.exists():
        return "FAIL", "FAIL", "missing-lowered-call-json"

    decode = _load_json(decode_path)
    prefill = _load_json(prefill_path)
    if decode is None or prefill is None:
        return "FAIL", "FAIL", "invalid-lowered-call-json"

    def _check(doc: dict[str, Any], fn_name: str) -> tuple[int, int]:
        ops = doc.get("operations")
        if not isinstance(ops, list):
            return 0, 0
        total = 0
        with_window = 0
        for op in ops:
            if not isinstance(op, dict):
                continue
            if op.get("op") != "attn_sliding":
                continue
            if op.get("function") != fn_name:
                continue
            total += 1
            args = op.get("args", [])
            if not isinstance(args, list):
                continue
            for arg in args:
                if not isinstance(arg, dict):
                    continue
                if arg.get("name") != "sliding_window":
                    continue
                expr = str(arg.get("expr", "")).strip()
                try:
                    val = int(float(expr))
                except Exception:
                    val = 0
                if val > 0:
                    with_window += 1
                break
        return total, with_window

    dec_total, dec_with_window = _check(
        decode, "attention_forward_decode_head_major_gqa_flash_sliding"
    )
    pre_total, pre_with_window = _check(
        prefill, "attention_forward_causal_head_major_gqa_flash_strided_sliding"
    )

    config = decode.get("config", {})
    cfg_window = 0
    if isinstance(config, dict):
        try:
            cfg_window = int(config.get("sliding_window", 0))
        except Exception:
            cfg_window = 0

    note = (
        f"cfg_window={cfg_window},"
        f"decode={dec_total}/{dec_with_window},"
        f"prefill={pre_total}/{pre_with_window}"
    )

    ok = (
        cfg_window > 0
        and dec_total > 0
        and pre_total > 0
        and dec_total == dec_with_window
        and pre_total == pre_with_window
    )
    if ok:
        return "PASS", "PASS", note
    return "FAIL", "FAIL", note


def _last_error_line(proc: subprocess.CompletedProcess[str]) -> str:
    text = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
    if not text:
        return "no-output"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "Error:" in ln or "failed" in ln.lower() or "FAIL" in ln:
            return ln[:180]
    return lines[-1][:180]


def _run_static_contracts(timeout: int) -> tuple[str, str]:
    proc = _run([sys.executable, str(STATIC_CONTRACTS)], timeout=timeout)
    if proc.returncode != 0:
        return "FAIL", _last_error_line(proc)
    m = re.search(r"Summary:\s*PASS=(\d+)\s*WARN=(\d+)\s*FAIL=(\d+)", proc.stdout or "")
    if not m:
        return "WARN", "summary-not-found"
    warn = int(m.group(2))
    fail = int(m.group(3))
    if fail > 0:
        return "FAIL", f"warn={warn},fail={fail}"
    if warn > 0:
        return "WARN", f"warn={warn}"
    return "PASS", "static-contracts-clean"


def _run_model(model: dict[str, str], args: argparse.Namespace) -> MatrixRow:
    uri = model["uri"]
    name = model["name"]
    family = model["family"]
    run_input = uri
    work_dir = _work_dir_for_uri(uri)
    cached = _is_cached(uri)
    cached_status = "YES" if cached else "NO"

    # For cached hf:// GGUF models, run from local GGUF path to avoid network/tokenizer fetches.
    local_gguf = _cached_gguf_path(uri)
    if local_gguf is not None and local_gguf.suffix.lower() == ".gguf":
        run_input = str(local_gguf)
        work_dir = _cache_dir() / local_gguf.stem

    if (not cached) and (not args.allow_download):
        return MatrixRow(
            model=name,
            family=family,
            cached=cached_status,
            build="SKIP",
            artifacts="SKIP",
            sliding="N/A",
            smoke="N/A",
            overall="SKIP",
            note="not-cached (use --allow-download)",
            work_dir=str(work_dir),
            seconds=0.0,
        )

    cmd = [
        sys.executable,
        str(CK_RUN),
        "run",
        run_input,
        "--generate-only",
        "--prompt",
        args.prompt,
        "--max-tokens",
        "1",
        "--context-len",
        str(args.context_len),
        "--no-chat-template",
    ]
    if args.force_compile:
        cmd.append("--force-compile")
    if args.with_smoke:
        cmd.append("--test")
    if args.reverse_test:
        cmd.append("--reverse-test")

    t0 = time.time()
    try:
        proc = _run(cmd, timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        return MatrixRow(
            model=name,
            family=family,
            cached=cached_status,
            build="FAIL",
            artifacts="N/A",
            sliding="N/A",
            smoke="N/A",
            overall="FAIL",
            note=f"timeout>{args.timeout_sec}s",
            work_dir=str(work_dir),
            seconds=time.time() - t0,
        )

    build_status = "PASS" if proc.returncode == 0 else "FAIL"
    artifact_status, artifact_note = _artifact_status(work_dir)
    sliding_display, sliding_gate, sliding_note = _sliding_contract_status(work_dir, family)
    smoke_status = "PASS" if args.with_smoke and proc.returncode == 0 else ("N/A" if not args.with_smoke else "FAIL")

    note_parts = []
    if run_input != uri:
        note_parts.append("offline-run=local-gguf")
    if proc.returncode != 0:
        note_parts.append(_last_error_line(proc))
    note_parts.append(artifact_note)
    if family == "gemma":
        note_parts.append(sliding_note)
    note = "; ".join(note_parts)

    overall = _join_status(
        build_status,
        artifact_status,
        sliding_gate,
        smoke_status if args.with_smoke else "PASS",
    )
    return MatrixRow(
        model=name,
        family=family,
        cached=cached_status,
        build=build_status,
        artifacts=artifact_status,
        sliding=sliding_display,
        smoke=smoke_status,
        overall=overall,
        note=note,
        work_dir=str(work_dir),
        seconds=time.time() - t0,
    )


def _table(rows: list[MatrixRow]) -> str:
    headers = ["Model", "Family", "Cached", "Build", "Artifacts", "Sliding", "Smoke", "Overall", "Sec", "Note"]
    vals = [headers]
    for r in rows:
        vals.append([
            r.model,
            r.family,
            r.cached,
            r.build,
            r.artifacts,
            r.sliding,
            r.smoke,
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
    ap = argparse.ArgumentParser(description="Dynamic v7 model matrix validator")
    ap.add_argument("--allow-download", action="store_true", help="Allow model download when cache is missing")
    ap.add_argument("--with-smoke", action="store_true", help="Run ck_run --test in addition to generate/compile")
    ap.add_argument("--force-compile", action="store_true", help="Force regeneration/recompile per model")
    ap.add_argument("--reverse-test", action="store_true", help="Run IR reverse validator during model build")
    ap.add_argument("--skip-static-contracts", action="store_true", help="Skip static tooling contract preflight")
    ap.add_argument("--context-len", type=int, default=128, help="Context length used during build phase")
    ap.add_argument("--prompt", default="Hello", help="Prompt seed used by ck_run")
    ap.add_argument("--timeout-sec", type=int, default=1800, help="Per-model command timeout in seconds")
    ap.add_argument("--require-all", action="store_true", help="Fail if any model is skipped")
    ap.add_argument("--strict", action="store_true", help="Fail on WARN statuses as well")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path")
    args = ap.parse_args()

    static_status = "N/A"
    static_note = "skipped"
    if not args.skip_static_contracts:
        static_status, static_note = _run_static_contracts(timeout=max(60, min(600, args.timeout_sec)))

    rows = [_run_model(m, args) for m in DEFAULT_MODELS]

    print("=" * 140)
    print("v7 DYNAMIC MODEL MATRIX REPORT")
    print("=" * 140)
    print(f"Static contracts: {static_status} ({static_note})")
    print(f"Models: {', '.join(m['name'] for m in DEFAULT_MODELS)}")
    print(_table(rows))
    print("=" * 140)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.overall] = counts.get(r.overall, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Summary: {summary if summary else 'none'}")

    if args.json_out is not None:
        payload = {
            "static_contracts": {"status": static_status, "note": static_note},
            "rows": [
                {
                    "model": r.model,
                    "family": r.family,
                    "cached": r.cached,
                    "build": r.build,
                    "artifacts": r.artifacts,
                    "sliding": r.sliding,
                    "smoke": r.smoke,
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

    failed = any(r.overall == "FAIL" for r in rows) or static_status == "FAIL"
    warned = any(r.overall == "WARN" for r in rows) or static_status == "WARN"
    skipped = any(r.overall == "SKIP" for r in rows)

    if failed:
        return 1
    if args.require_all and skipped:
        return 2
    if args.strict and warned:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
