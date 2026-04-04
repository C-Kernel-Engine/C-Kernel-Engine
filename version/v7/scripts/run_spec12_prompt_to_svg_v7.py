#!/usr/bin/env python3
"""Run a spec12 prompt through CK inference and compile the first scene DSL to SVG."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from probe_report_adapters_v7 import apply_output_adapter, extract_response_text


ROOT = Path(__file__).resolve().parents[3]
CK_CHAT = ROOT / "scripts" / "ck_chat.py"
DEFAULT_STOP_MARKERS = ("[task:svg]",)
_TAG_RE = re.compile(r"\[([a-z_]+):([^\]]+)\]")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _python_bin() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _render_catalog_path(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "dataset" / "tokenizer").glob("*_render_catalog.json"))
    if not candidates:
        candidates = sorted((run_dir / "tokenizer").glob("*_render_catalog.json"))
    if not candidates:
        raise SystemExit(f"no *_render_catalog.json found under {run_dir}")
    return candidates[0]


def _parse_prompt_tags(prompt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _TAG_RE.finditer(str(prompt or "")):
        out[str(match.group(1) or "").strip()] = str(match.group(2) or "").strip()
    return out


def _norm_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").replace("-", " ").strip().lower())


def _row_score(row: dict[str, Any], *, prompt: str, tags: dict[str, str], case_id: str | None) -> int:
    score = 0
    if str(row.get("prompt") or "").strip() == str(prompt).strip():
        score += 1000

    row_case = str(row.get("case_id") or "").strip()
    if case_id:
        if row_case != case_id:
            return -10_000
        score += 500

    for key in ("layout", "topic", "theme", "tone", "density"):
        want = str(tags.get(key) or "").strip()
        have = str(row.get(key) or "").strip()
        if not want:
            continue
        if want == have:
            score += 80 if key in {"layout", "topic"} else 30
        else:
            score -= 40 if key in {"layout", "topic"} else 10

    prompt_norm = _norm_phrase(prompt)
    for field, weight in (("prompt_topic", 140), ("case_id", 120), ("source_asset", 60)):
        phrase = _norm_phrase(str(row.get(field) or ""))
        if phrase and phrase in prompt_norm:
            score += weight

    if bool(row.get("training_prompt")):
        score += 5
    if str(row.get("prompt_surface") or "") == "tag_canonical":
        score += 3
    return score


def _pick_catalog_row(catalog_rows: list[dict[str, Any]], *, prompt: str, case_id: str | None) -> dict[str, Any]:
    tags = _parse_prompt_tags(prompt)
    if not catalog_rows:
        raise SystemExit("render catalog is empty")
    ranked = sorted(
        catalog_rows,
        key=lambda row: (
            _row_score(row, prompt=prompt, tags=tags, case_id=case_id),
            str(row.get("case_id") or ""),
            str(row.get("prompt_surface") or ""),
        ),
        reverse=True,
    )
    best = ranked[0]
    if _row_score(best, prompt=prompt, tags=tags, case_id=case_id) < 0:
        raise SystemExit("no compatible content row found for prompt and case selection")
    return best


def _run_ck_chat(
    *,
    model_dir: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repeat_penalty: float,
    stop_markers: list[str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        _python_bin(),
        str(CK_CHAT),
        "--model-dir",
        str(model_dir),
        "--chat-template",
        "none",
        "--allow-raw-prompt",
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--top-p",
        str(top_p),
        "--repeat-penalty",
        str(repeat_penalty),
        "--no-stats",
        "--prompt",
        prompt,
    ]
    for marker in stop_markers:
        cmd.extend(["--stop-on-text", marker])
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)


def _build_preview_html(payload: dict[str, Any]) -> str:
    svg_markup = str(payload.get("svg") or "").strip()
    if not svg_markup:
        svg_markup = '<div class="empty">No renderable SVG was produced.</div>'
    content_json = payload.get("content_json")
    content_block = json.dumps(content_json, indent=2, ensure_ascii=False) if isinstance(content_json, dict) else "—"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8"/>',
            f"  <title>{html.escape(str(payload.get('title') or 'Spec12 Prompt Demo'))}</title>",
            "  <style>",
            "    :root { --bg:#0c111b; --panel:#121a28; --line:#273247; --ink:#ebf1f7; --muted:#99a6b8; --accent:#67e8f9; }",
            "    body { margin:0; padding:24px; background:linear-gradient(180deg,#0b1118,#0f1724); color:var(--ink); font-family:'IBM Plex Sans','Segoe UI',sans-serif; }",
            "    main { max-width: 1500px; margin: 0 auto; }",
            "    h1,h2 { margin:0 0 10px; }",
            "    .lede { margin:0 0 22px; color:var(--muted); }",
            "    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }",
            "    .panel { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px; box-shadow:0 12px 26px rgba(0,0,0,0.22); }",
            "    .label { color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; font-size:12px; font-weight:700; margin-bottom:10px; }",
            "    pre { margin:0; white-space:pre-wrap; word-break:break-word; background:#0b1220; border:1px solid #1e2a3d; border-radius:12px; padding:12px; overflow:auto; }",
            "    .meta { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-bottom:16px; }",
            "    .chip { background:#0b1220; border:1px solid #1e2a3d; border-radius:999px; padding:8px 12px; font-size:13px; }",
            "    .svg-frame { min-height:320px; background:white; border-radius:12px; border:1px solid #d7dde6; padding:10px; overflow:auto; }",
            "    .empty { color:#44556e; font-size:14px; }",
            "    @media (max-width: 980px) { .grid { grid-template-columns:1fr; } }",
            "  </style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{html.escape(str(payload.get('title') or 'Spec12 Prompt Demo'))}</h1>",
            "<p class=\"lede\">Prompt -> CK-generated scene DSL -> compiled SVG using the selected external content payload.</p>",
            "<section class=\"panel\">",
            "<div class=\"label\">Selected Case</div>",
            "<div class=\"meta\">",
            f"<div class=\"chip\">Run: <strong>{html.escape(str(payload.get('run_dir') or ''))}</strong></div>",
            f"<div class=\"chip\">Case: <strong>{html.escape(str(payload.get('case_id') or ''))}</strong></div>",
            f"<div class=\"chip\">Prompt topic: <strong>{html.escape(str(payload.get('prompt_topic') or ''))}</strong></div>",
            f"<div class=\"chip\">Layout: <strong>{html.escape(str(payload.get('layout') or ''))}</strong></div>",
            f"<div class=\"chip\">Topic token: <strong>{html.escape(str(payload.get('topic') or ''))}</strong></div>",
            f"<div class=\"chip\">Source asset: <strong>{html.escape(str(payload.get('source_asset') or ''))}</strong></div>",
            "</div>",
            "</section>",
            "<div class=\"grid\">",
            "<section class=\"panel\">",
            "<div class=\"label\">Prompt</div>",
            f"<pre>{html.escape(str(payload.get('prompt') or ''))}</pre>",
            "</section>",
            "<section class=\"panel\">",
            "<div class=\"label\">Generated Scene DSL</div>",
            f"<pre>{html.escape(str(payload.get('scene_dsl') or ''))}</pre>",
            "</section>",
            "</div>",
            "<div class=\"grid\" style=\"margin-top:16px;\">",
            "<section class=\"panel\">",
            "<div class=\"label\">Compiled SVG</div>",
            f"<div class=\"svg-frame\">{svg_markup}</div>",
            "</section>",
            "<section class=\"panel\">",
            "<div class=\"label\">content_json</div>",
            f"<pre>{html.escape(content_block)}</pre>",
            "</section>",
            "</div>",
            "<div class=\"grid\" style=\"margin-top:16px;\">",
            "<section class=\"panel\">",
            "<div class=\"label\">Raw CK Output</div>",
            f"<pre>{html.escape(str(payload.get('raw_response') or ''))}</pre>",
            "</section>",
            "<section class=\"panel\">",
            "<div class=\"label\">ck_chat stdout</div>",
            f"<pre>{html.escape(str(payload.get('ck_stdout') or ''))}</pre>",
            "</section>",
            "</div>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path, help="Training run directory, e.g. ~/.cache/.../spec12..._r17")
    ap.add_argument("--prompt", required=True, help="Prompt to run through the CK inference path")
    ap.add_argument("--case-id", default=None, help="Optional explicit case override from the render catalog")
    ap.add_argument("--content-json", type=Path, default=None, help="Optional explicit content JSON override")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/spec12_prompt_svg_demo"), help="Directory to write artifacts")
    ap.add_argument("--name", default="spec12_demo", help="Artifact basename without extension")
    ap.add_argument("--max-tokens", type=int, default=160, help="Decode token cap")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--top-k", type=int, default=1, help="Top-k sampling")
    ap.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="Repeat penalty")
    ap.add_argument("--stop-on-text", action="append", default=[], help="Additional decoded text stop markers")
    ap.add_argument("--list-cases", action="store_true", help="List known case_ids for the run and exit")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    model_dir = (run_dir / ".ck_build").resolve() if (run_dir / ".ck_build").exists() else run_dir
    render_catalog = _load_json(_render_catalog_path(run_dir))
    if not isinstance(render_catalog, list):
        raise SystemExit("expected render catalog JSON list")
    catalog_rows = [row for row in render_catalog if isinstance(row, dict)]
    if args.list_cases:
        case_ids = sorted({str(row.get("case_id") or "").strip() for row in catalog_rows if str(row.get("case_id") or "").strip()})
        for item in case_ids:
            print(item)
        return 0

    selected = _pick_catalog_row(catalog_rows, prompt=args.prompt, case_id=args.case_id)
    selected_tags = _parse_prompt_tags(str(selected.get("prompt") or args.prompt))
    selected_case_id = (
        str(selected.get("case_id") or "").strip()
        or str(selected.get("prompt_topic") or "").strip()
        or str(selected.get("topic") or "").strip()
        or str(selected_tags.get("topic") or "").strip()
    )
    selected_prompt_topic = (
        str(selected.get("prompt_topic") or "").strip()
        or str(selected.get("topic") or "").strip()
        or str(selected_tags.get("topic") or "").strip()
    )
    content_json = (
        _load_json(args.content_json.expanduser().resolve())
        if args.content_json
        else selected.get("content_json")
    )
    if not isinstance(content_json, dict):
        raise SystemExit("selected content_json is missing or invalid")

    stop_markers = list(DEFAULT_STOP_MARKERS)
    for marker in args.stop_on_text:
        text = str(marker or "").strip()
        if text and text not in stop_markers:
            stop_markers.append(text)

    proc = _run_ck_chat(
        model_dir=model_dir,
        prompt=args.prompt,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repeat_penalty=float(args.repeat_penalty),
        stop_markers=stop_markers,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or f"ck_chat failed with code {proc.returncode}")

    response_text = extract_response_text(proc.stdout, args.prompt)
    adapted = apply_output_adapter(
        "text_renderer",
        response_text,
        {
            "stop_markers": ["[/scene]"],
            "renderer": "structured_svg_scene_spec12.v1",
            "preview_mime": "image/svg+xml",
            "content_json": content_json,
        },
    )
    scene_dsl = str(adapted.get("parsed_output") or "").strip()
    svg = str(adapted.get("materialized_output") or "").strip()
    renderable = bool(adapted.get("renderable")) and bool(svg)
    render_error = str(adapted.get("render_error") or "").strip() or None

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    raw_path = base.with_suffix(".raw.txt")
    scene_path = base.with_suffix(".scene.dsl")
    svg_path = base.with_suffix(".svg")
    html_path = base.with_suffix(".html")
    json_path = base.with_suffix(".json")

    raw_path.write_text(proc.stdout, encoding="utf-8")
    scene_path.write_text(scene_dsl + ("\n" if scene_dsl else ""), encoding="utf-8")
    if svg:
        svg_path.write_text(svg + "\n", encoding="utf-8")

    result = {
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "prompt": args.prompt,
        "selected_prompt": selected.get("prompt"),
        "case_id": selected_case_id or None,
        "prompt_topic": selected_prompt_topic or None,
        "layout": selected.get("layout") or selected_tags.get("layout"),
        "topic": selected.get("topic") or selected_tags.get("topic"),
        "theme": selected.get("theme") or selected_tags.get("theme"),
        "tone": selected.get("tone") or selected_tags.get("tone"),
        "density": selected.get("density") or selected_tags.get("density"),
        "source_asset": selected.get("source_asset"),
        "content_json": content_json,
        "scene_dsl": scene_dsl,
        "svg": svg if svg else None,
        "renderable": renderable,
        "render_error": render_error,
        "raw_response": response_text,
        "ck_stdout": proc.stdout,
        "ck_stderr": proc.stderr,
        "artifacts": {
            "raw_stdout": str(raw_path),
            "scene_dsl": str(scene_path),
            "svg": str(svg_path) if svg else None,
            "html": str(html_path),
            "json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path.write_text(
        _build_preview_html(
            {
                **result,
                "title": f"Spec12 Prompt Demo: {args.name}",
            }
        ),
        encoding="utf-8",
    )

    print(f"[OK] scene DSL: {scene_path}")
    if svg:
        print(f"[OK] svg: {svg_path}")
    print(f"[OK] html: {html_path}")
    print(f"[OK] json: {json_path}")
    if not renderable:
        raise SystemExit(render_error or "scene did not render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
