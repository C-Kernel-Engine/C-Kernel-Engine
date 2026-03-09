#!/usr/bin/env python3
"""Run the semantic SVG toy model and render its IR output to an SVG preview."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

from render_svg_semantic_ir_v7 import render_ir


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_NAME = "toy_svg_semantic_shapes_ctx512_d64_h128"
DEFAULT_RUN_ROOT = Path.home() / ".cache" / "ck-engine-v7" / "models" / "train"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _python_bin() -> str:
    venv = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _extract_response(stdout: str) -> str:
    clean_stdout = ANSI_RE.sub("", stdout)
    lines = clean_stdout.splitlines()
    chunks: list[str] = []
    capture = False
    timing_prefixes = ("prompt eval:", "decode:", "sample:", "total:")
    for raw in lines:
        line = raw.strip()
        if line.startswith("Response:"):
            capture = True
            tail = line[len("Response:") :].strip()
            if tail:
                chunks.append(tail)
            continue
        if not capture:
            continue
        if line.startswith(timing_prefixes):
            break
        if line:
            chunks.append(line)
    result = " ".join(part.strip() for part in chunks if part.strip()).strip()
    for prefix in timing_prefixes:
        if prefix in result:
            result = result.split(prefix, 1)[0].strip()
    if "<|eos|>" in result:
        result = result.split("<|eos|>", 1)[0].strip()
    if "[/svg]" in result:
        result = result.split("[/svg]", 1)[0].strip() + " [/svg]"
    return result


def _resolve_model_dir(run_name: str | None, run_dir: str | None, model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir).expanduser().resolve()
    if run_dir:
        return Path(run_dir).expanduser().resolve() / ".ck_build"
    if not run_name:
        run_name = DEFAULT_RUN_NAME
    return (DEFAULT_RUN_ROOT / run_name / ".ck_build").resolve()


def _write_preview(out_dir: Path, prompt: str, ir_text: str, svg_text: str) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ir_path = out_dir / "preview.ir.txt"
    svg_path = out_dir / "preview.svg"
    html_path = out_dir / "preview.html"
    ir_path.write_text(ir_text + "\n", encoding="utf-8")
    svg_path.write_text(svg_text + "\n", encoding="utf-8")
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>Semantic SVG Preview</title>",
                "<style>body{font-family:ui-monospace,monospace;background:#111827;color:#e5e7eb;padding:24px}pre{white-space:pre-wrap;background:#0b1220;padding:12px;border-radius:8px}section{margin-bottom:20px}.frame{background:#fff;border-radius:8px;padding:16px;display:inline-block}</style>",
                "</head><body>",
                "<section><h2>Prompt</h2>",
                f"<pre>{html.escape(prompt)}</pre></section>",
                "<section><h2>IR</h2>",
                f"<pre>{html.escape(ir_text)}</pre></section>",
                "<section><h2>SVG</h2><div class='frame'>",
                svg_text,
                "</div></section>",
                "</body></html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return ir_path, svg_path, html_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the semantic SVG toy model and render a preview")
    ap.add_argument("--prompt", required=True, help="Prompt text passed to ck_chat.py")
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Cache-backed run name")
    ap.add_argument("--run-dir", default=None, help="Explicit run directory; .ck_build is used beneath it")
    ap.add_argument("--model-dir", default=None, help="Explicit .ck_build model directory")
    ap.add_argument("--max-tokens", type=int, default=32, help="Decode length")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--stop-on-text", default="<|eos|>", help="Stop string for ck_chat.py")
    ap.add_argument("--out-dir", default="/tmp/ckv7_svg_semantic_preview", help="Preview output directory")
    args = ap.parse_args()

    model_dir = _resolve_model_dir(args.run_name, args.run_dir, args.model_dir)
    if not model_dir.exists():
        raise SystemExit(
            f"model dir not found: {model_dir}\n"
            "Build it first with: python3 version/v7/scripts/ck_run_v7.py run <run-dir> --context-len 512 --max-tokens 32 --force-compile --generate-only"
        )

    cmd = [
        _python_bin(),
        str(PROJECT_ROOT / "scripts" / "ck_chat.py"),
        "--model-dir",
        str(model_dir),
        "--python-tokenizer",
        "--chat-template",
        "none",
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--stop-on-text",
        args.stop_on_text,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=True)
    ir_text = _extract_response(proc.stdout)
    if not ir_text:
        raise SystemExit(f"failed to extract response from ck_chat.py output:\n{proc.stdout}")

    svg_text = render_ir(ir_text)
    out_dir = Path(args.out_dir).expanduser().resolve()
    ir_path, svg_path, html_path = _write_preview(out_dir, args.prompt, ir_text, svg_text)
    print(f"model_dir: {model_dir}")
    print(f"ir: {ir_text}")
    print(f"ir_file: {ir_path}")
    print(f"svg_file: {svg_path}")
    print(f"html_file: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
