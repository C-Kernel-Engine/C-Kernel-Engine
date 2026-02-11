#!/usr/bin/env python3
"""
Long-decode stability validator for v6.6 model matrix.

Checks each model with longer generation to catch:
- hangs/timeouts
- crashes/assertions
- OOM/mmap failures
- severe decode under-run
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


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "version" / "v6.6" / "scripts"
CK_RUN = SCRIPTS / "ck_run_v6_6.py"


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
class StabilityRow:
    model: str
    family: str
    cached: str
    decode_runs: int
    overall: str
    note: str
    seconds: float


def _cache_dir() -> Path:
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        path = Path(env).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    default = Path.home() / ".cache" / "ck-engine-v6.6" / "models"
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
        if "Error:" in ln or "failed" in ln.lower() or "Traceback" in ln:
            return ln[:220]
    return lines[-1][:220]


def _decode_runs(text: str) -> int:
    m = re.search(r"decode:\s+[0-9.]+\s+ms\s*/\s*(\d+)\s+runs", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\bGenerated\s+(\d+)\s+tokens\b", text)
    if m:
        return int(m.group(1))
    return 0


def _has_fatal_output(text: str) -> str | None:
    checks = [
        (r"segfault|SIGSEGV|assertion failed|abort\(", "crash"),
        (r"Traceback", "python-traceback"),
        (r"mmap failed|Cannot allocate memory|out of memory|ENOMEM", "oom"),
        (r"\bnan\b", "nan"),
    ]
    lowered = text.lower()
    for pat, label in checks:
        if re.search(pat, lowered):
            return label
    return None


def _run_model(model: dict[str, str], args: argparse.Namespace) -> StabilityRow:
    uri = model["uri"]
    run_input = uri
    cached = _is_cached(uri)
    cached_status = "YES" if cached else "NO"
    if (not cached) and (not args.allow_download):
        return StabilityRow(
            model=model["name"],
            family=model["family"],
            cached=cached_status,
            decode_runs=0,
            overall="SKIP",
            note="not-cached (use --allow-download)",
            seconds=0.0,
        )
    local_gguf = _cached_gguf_path(uri)
    if local_gguf is not None and local_gguf.suffix.lower() == ".gguf":
        run_input = str(local_gguf)

    cmd = [
        sys.executable,
        str(CK_RUN),
        "run",
        run_input,
        "--prompt",
        args.prompt,
        "--context-len",
        str(args.context_len),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--no-chat-template",
    ]
    if args.force_compile:
        cmd.append("--force-compile")

    t0 = time.time()
    try:
        proc = _run(cmd, timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        return StabilityRow(
            model=model["name"],
            family=model["family"],
            cached=cached_status,
            decode_runs=0,
            overall="FAIL",
            note=f"timeout>{args.timeout_sec}s",
            seconds=time.time() - t0,
        )

    text = ((proc.stdout or "") + "\n" + (proc.stderr or ""))
    fatal = _has_fatal_output(text)
    runs = _decode_runs(text)
    if proc.returncode != 0:
        return StabilityRow(
            model=model["name"],
            family=model["family"],
            cached=cached_status,
            decode_runs=runs,
            overall="FAIL",
            note=_last_error_line(proc),
            seconds=time.time() - t0,
        )
    if fatal is not None:
        return StabilityRow(
            model=model["name"],
            family=model["family"],
            cached=cached_status,
            decode_runs=runs,
            overall="FAIL",
            note=f"fatal-output={fatal}",
            seconds=time.time() - t0,
        )
    if runs < args.min_decode_runs:
        return StabilityRow(
            model=model["name"],
            family=model["family"],
            cached=cached_status,
            decode_runs=runs,
            overall="FAIL",
            note=f"decode-runs={runs}<min={args.min_decode_runs}",
            seconds=time.time() - t0,
        )
    return StabilityRow(
        model=model["name"],
        family=model["family"],
        cached=cached_status,
        decode_runs=runs,
        overall="PASS",
        note="long-decode-stable",
        seconds=time.time() - t0,
    )


def _table(rows: list[StabilityRow]) -> str:
    headers = ["Model", "Family", "Cached", "DecodeRuns", "Overall", "Sec", "Note"]
    vals = [headers]
    for r in rows:
        vals.append([r.model, r.family, r.cached, str(r.decode_runs), r.overall, f"{r.seconds:.1f}", r.note])
    widths = [max(len(str(row[i])) for row in vals) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt(headers), sep]
    for row in vals[1:]:
        out.append(fmt(row))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="v6.6 long decode stability matrix")
    ap.add_argument("--allow-download", action="store_true", help="Allow HF download when cache is missing")
    ap.add_argument("--force-compile", action="store_true", help="Force regenerate/recompile")
    ap.add_argument("--prompt", default="Write 20 short bullet points about systems programming.", help="Prompt seed")
    ap.add_argument("--context-len", type=int, default=1024, help="Context length")
    ap.add_argument("--max-tokens", type=int, default=256, help="Requested decode tokens")
    ap.add_argument("--min-decode-runs", type=int, default=64, help="Minimum decode runs required for pass")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--timeout-sec", type=int, default=1800, help="Per-model timeout")
    ap.add_argument("--require-all", action="store_true", help="Fail if any model is skipped")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path")
    args = ap.parse_args()

    rows = [_run_model(m, args) for m in DEFAULT_MODELS]

    print("=" * 120)
    print("v6.6 LONG DECODE STABILITY REPORT")
    print("=" * 120)
    print(f"Prompt: {args.prompt}")
    print(f"Settings: ctx={args.context_len}, max_tokens={args.max_tokens}, min_decode_runs={args.min_decode_runs}")
    print(_table(rows))
    print("=" * 120)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.overall] = counts.get(r.overall, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Summary: {summary if summary else 'none'}")

    if args.json_out is not None:
        payload = {
            "rows": [
                {
                    "model": r.model,
                    "family": r.family,
                    "cached": r.cached,
                    "decode_runs": r.decode_runs,
                    "overall": r.overall,
                    "note": r.note,
                    "seconds": r.seconds,
                }
                for r in rows
            ],
            "summary": counts,
            "settings": {
                "context_len": args.context_len,
                "max_tokens": args.max_tokens,
                "min_decode_runs": args.min_decode_runs,
                "temperature": args.temperature,
                "prompt": args.prompt,
            },
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

