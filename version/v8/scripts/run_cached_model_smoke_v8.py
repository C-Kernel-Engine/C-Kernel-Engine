#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = Path("/opt/app-root/src/.cache/ck-engine-v8/models")


@dataclass(frozen=True)
class SmokeTarget:
    name: str
    dir_patterns: tuple[str, ...]
    gguf_patterns: tuple[str, ...]
    context_len: int
    thinking_mode: str = "suppressed"


TARGETS: tuple[SmokeTarget, ...] = (
    SmokeTarget(
        name="qwen2",
        dir_patterns=("Qwen--Qwen2*", "*Qwen2*"),
        gguf_patterns=("*.gguf",),
        context_len=1024,
    ),
    SmokeTarget(
        name="qwen3",
        dir_patterns=("Qwen--Qwen3-*", "*Qwen3-0.6B*"),
        gguf_patterns=("*.gguf",),
        context_len=1024,
    ),
    SmokeTarget(
        name="qwen35",
        dir_patterns=("unsloth--Qwen3.5-*", "Qwen3.5-*"),
        gguf_patterns=("*.gguf",),
        context_len=1034,
    ),
    SmokeTarget(
        name="gemma3",
        dir_patterns=("unsloth--gemma-3-*", "gemma-3-*"),
        gguf_patterns=("*.gguf",),
        context_len=1024,
    ),
)


def _quality_issues(text: str) -> list[str]:
    issues: list[str] = []
    if "\ufffd" in text or "\u2581" in text:
        issues.append("decoded tokenizer marker/replacement char")
    for needle in ("int main()", "Pythonic Approach", "Conclusion", "This is a complete example"):
        if text.count(needle) >= 4:
            issues.append(f"repeated phrase: {needle!r}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8:
        repeated_lines = sum(1 for i in range(1, len(lines)) if lines[i] == lines[i - 1])
        if repeated_lines >= 3:
            issues.append("adjacent repeated lines")
    return issues


def _extract_response(output: str) -> str:
    if "Response:" not in output:
        return ""
    response = output.split("Response:", 1)[1]
    for marker in ("\x1b[90mprompt eval:", "prompt eval:", "\nstop:"):
        if marker in response:
            response = response.split(marker, 1)[0]
    return response.strip()


def _first_existing_dir(cache_root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        for path in sorted(cache_root.glob(pattern)):
            if path.is_dir() and (path / "config.json").exists():
                return path
    return None


def _first_existing_gguf(run_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file():
                return path
    return None


def _run_target(target: SmokeTarget, *, cache_root: Path, prompt: str, max_tokens: int) -> dict[str, Any]:
    run_dir = _first_existing_dir(cache_root, target.dir_patterns)
    if run_dir is None:
        return {"name": target.name, "status": "skipped", "reason": "not cached"}

    gguf = _first_existing_gguf(run_dir, target.gguf_patterns)
    if gguf is None:
        return {
            "name": target.name,
            "status": "skipped",
            "run_dir": str(run_dir),
            "reason": "cached runtime has no colocated GGUF",
        }

    cmd = [
        sys.executable,
        str(ROOT / "version/v8/scripts/ck_run_v8.py"),
        "run",
        str(gguf),
        "--run",
        str(run_dir),
        "--prompt",
        prompt,
        "--max-tokens",
        str(int(max_tokens)),
        "--context-len",
        str(int(target.context_len)),
        "--thinking-mode",
        target.thinking_mode,
    ]
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - start
    output = proc.stdout or ""
    response = _extract_response(output)
    quality_issues = _quality_issues(response)
    status = "pass" if proc.returncode == 0 and "Response:" in output and not quality_issues else "fail"
    return {
        "name": target.name,
        "status": status,
        "returncode": int(proc.returncode),
        "elapsed_sec": round(elapsed, 3),
        "run_dir": str(run_dir),
        "gguf": str(gguf),
        "quality_issues": quality_issues,
        "response_preview": response[:240],
        "tail": "\n".join(output.strip().splitlines()[-16:]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cached v8 smoke prompts across supported model families.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--prompt", default="Hello!")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = [
        _run_target(target, cache_root=args.cache_root, prompt=args.prompt, max_tokens=args.max_tokens)
        for target in TARGETS
    ]
    report = {"cache_root": str(args.cache_root), "prompt": args.prompt, "max_tokens": args.max_tokens, "results": rows}
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if any(row["status"] == "fail" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
