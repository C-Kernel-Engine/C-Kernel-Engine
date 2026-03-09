#!/usr/bin/env python3
"""Build a standalone HTML probe report for the semantic SVG toy run."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from realtime_svg_semantic_preview_v7 import _extract_response, _python_bin
from render_svg_semantic_ir_v7 import render_ir


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_NAME = "toy_svg_semantic_shapes_ctx512_d64_h128"
DEFAULT_RUN_ROOT = Path.home() / ".cache" / "ck-engine-v7" / "models" / "train"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _html(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _resolve_run_dir(run_name: str | None, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    return (DEFAULT_RUN_ROOT / (run_name or DEFAULT_RUN_NAME)).resolve()


def _choose_prompt_sets(expected_map: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    prompts = [
        {"category": "seen", "label": "Seen: warm circle lg", "prompt": "[task:svg] [shape:circle] [palette:warm] [size:lg] [OUT]"},
        {"category": "seen", "label": "Seen: cool rect md", "prompt": "[task:svg] [shape:rect] [palette:cool] [size:md] [OUT]"},
        {"category": "seen", "label": "Seen: mono circle sm", "prompt": "[task:svg] [shape:circle] [palette:mono] [size:sm] [OUT]"},
        {"category": "seen", "label": "Seen: signal triangle xl", "prompt": "[task:svg] [shape:triangle] [palette:signal] [size:xl] [OUT]"},
        {"category": "holdout", "label": "Holdout: warm circle xl", "prompt": "[task:svg] [shape:circle] [palette:warm] [size:xl] [OUT]"},
        {"category": "holdout", "label": "Holdout: mono circle md", "prompt": "[task:svg] [shape:circle] [palette:mono] [size:md] [OUT]"},
        {"category": "holdout", "label": "Holdout: cool rect xs", "prompt": "[task:svg] [shape:rect] [palette:cool] [size:xs] [OUT]"},
        {"category": "holdout", "label": "Holdout: warm triangle sm", "prompt": "[task:svg] [shape:triangle] [palette:warm] [size:sm] [OUT]"},
        {"category": "future", "label": "Future: dark card bullets", "prompt": "[task:card] [theme:dark] [accent:cool] [text:title] [text:bullet3] [OUT]"},
        {"category": "future", "label": "Future: light card compact", "prompt": "[task:card] [theme:light] [accent:mono] [text:title] [text:bullet2] [OUT]"},
        {"category": "future", "label": "Future: bar chart up", "prompt": "[task:chart] [chart:bar] [bars:3] [trend:up] [OUT]"},
        {"category": "future", "label": "Future: line chart down", "prompt": "[task:chart] [chart:line] [bars:3] [trend:down] [OUT]"},
        {"category": "future", "label": "Future: quad curve", "prompt": "[task:plot] [curve:quad-up] [OUT]"},
        {"category": "future", "label": "Future: s-curve", "prompt": "[task:plot] [curve:s-curve] [OUT]"},
    ]
    return [item for item in prompts if item["prompt"] in expected_map]


def _run_prompt(model_dir: Path, prompt: str, max_tokens: int) -> dict[str, Any]:
    cmd = [
        _python_bin(),
        str(ROOT / "scripts" / "ck_chat.py"),
        "--model-dir",
        str(model_dir),
        "--python-tokenizer",
        "--chat-template",
        "none",
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0.0",
        "--stop-on-text",
        "<|eos|>",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), check=True)
    raw_ir = _extract_response(proc.stdout)
    first_ir = raw_ir
    tail_drift = False
    if "[/svg]" in raw_ir:
        cropped = raw_ir.split("[/svg]", 1)[0].strip() + " [/svg]"
        tail_drift = cropped != raw_ir
        first_ir = cropped
    render_error = None
    rendered_svg = None
    try:
        rendered_svg = render_ir(first_ir)
    except BaseException as exc:  # pragma: no cover - defensive surface for report capture
        render_error = str(exc)
    return {
        "raw_ir": raw_ir,
        "ir": first_ir,
        "tail_drift": tail_drift,
        "rendered_svg": rendered_svg,
        "render_error": render_error,
    }


def _metric_class(ok: bool | None) -> str:
    if ok is True:
        return "good"
    if ok is False:
        return "bad"
    return "mid"


def _build_probe_card(result: dict[str, Any]) -> str:
    expected_svg = result.get("expected_svg")
    rendered_svg = result.get("rendered_svg")
    exact = result.get("exact_match")
    renderable = bool(rendered_svg)
    chips = [
        f'<span class="chip { _metric_class(exact) }">{"Exact" if exact else "Drift"}</span>' if exact is not None else "",
        f'<span class="chip { _metric_class(renderable) }">{"Renderable" if renderable else "Render fail"}</span>',
        '<span class="chip mid">Tail drift</span>' if result.get("tail_drift") else "",
        f'<span class="chip stage">{_html(result.get("stage") or "unknown")}</span>',
        f'<span class="chip split">{_html(result.get("split") or "n/a")}</span>',
    ]
    expected_block = (
        expected_svg
        if expected_svg
        else '<div class="empty-preview">No expected SVG captured.</div>'
    )
    actual_block = (
        rendered_svg
        if rendered_svg
        else f'<div class="empty-preview">Render failed: {_html(result.get("render_error") or "unknown")}</div>'
    )
    return (
        '<article class="probe-card">'
        f'<div class="probe-top"><div><div class="probe-label">{_html(result.get("label"))}</div>'
        f'<div class="probe-prompt">{_html(result.get("prompt"))}</div></div>'
        f'<div class="chips">{"".join(chips)}</div></div>'
        '<div class="preview-grid">'
        '<div class="preview-cell"><div class="preview-title">Actual SVG</div>'
        f'<div class="svg-frame">{actual_block}</div></div>'
        '<div class="preview-cell"><div class="preview-title">Expected target</div>'
        f'<div class="svg-frame">{expected_block}</div></div>'
        '</div>'
        '<div class="text-grid">'
        f'<div><div class="text-title">Generated IR</div><pre>{_html(result.get("ir"))}</pre></div>'
        f'<div><div class="text-title">Expected IR</div><pre>{_html(result.get("expected_ir") or "—")}</pre></div>'
        '</div>'
        '</article>'
    )


def _category_section(title: str, subtitle: str, results: list[dict[str, Any]]) -> str:
    cards = "".join(_build_probe_card(item) for item in results)
    return (
        '<section class="panel">'
        f'<h2>{_html(title)}</h2>'
        f'<p class="muted">{_html(subtitle)}</p>'
        f'<div class="probe-stack">{cards}</div>'
        '</section>'
    )


def _summary_value(results: list[dict[str, Any]], predicate) -> tuple[int, int]:
    total = len(results)
    good = sum(1 for item in results if predicate(item))
    return good, total


def _pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "0%"
    return f"{(100.0 * numer / denom):.0f}%"


def _build_html(run_dir: Path, training: dict[str, Any], results: list[dict[str, Any]]) -> str:
    seen = [r for r in results if r["category"] == "seen"]
    holdout = [r for r in results if r["category"] == "holdout"]
    future = [r for r in results if r["category"] == "future"]
    exact_all = _summary_value([r for r in results if r.get("expected_ir")], lambda x: bool(x.get("exact_match")))
    renderable_all = _summary_value(results, lambda x: bool(x.get("rendered_svg")))
    seen_exact = _summary_value(seen, lambda x: bool(x.get("exact_match")))
    holdout_exact = _summary_value(holdout, lambda x: bool(x.get("exact_match")))
    future_renderable = _summary_value(future, lambda x: bool(x.get("rendered_svg")))
    ck_loss = training.get("ck_loss") or {}
    hero = (
        '<section class="hero">'
        '<div>'
        '<span class="eyebrow">Toy Semantic SVG</span>'
        '<h1>Probe Report</h1>'
        '<p class="subhead">Prompt probes captured from the semantic SVG toy run. '
        'This report shows what the model actually emits, how the first valid IR renders, '
        'and where it matches or drifts from expected targets.</p>'
        f'<div class="meta">Run: {_html(run_dir.name)} | Steps: {_html(ck_loss.get("steps"))} | '
        f'Loss: {_html(ck_loss.get("first"))} → {_html(ck_loss.get("final"))}</div>'
        '</div>'
        '<div class="hero-metrics">'
        f'<div class="hero-card"><div class="k">Exact Match</div><div class="v">{_pct(*exact_all)}</div><div class="s">{exact_all[0]} / {exact_all[1]} expected probes</div></div>'
        f'<div class="hero-card"><div class="k">Renderable</div><div class="v">{_pct(*renderable_all)}</div><div class="s">{renderable_all[0]} / {renderable_all[1]} probes</div></div>'
        f'<div class="hero-card"><div class="k">Seen Exact</div><div class="v">{_pct(*seen_exact)}</div><div class="s">{seen_exact[0]} / {seen_exact[1]} seen prompts</div></div>'
        f'<div class="hero-card"><div class="k">Holdout Exact</div><div class="v">{_pct(*holdout_exact)}</div><div class="s">{holdout_exact[0]} / {holdout_exact[1]} holdouts</div></div>'
        f'<div class="hero-card"><div class="k">Future Renderable</div><div class="v">{_pct(*future_renderable)}</div><div class="s">{future_renderable[0]} / {future_renderable[1]} future-stage prompts</div></div>'
        f'<div class="hero-card"><div class="k">Best Loss</div><div class="v">{_html(f"{float(ck_loss.get('min', 0.0)):.4f}" if ck_loss.get("min") is not None else "-")}</div><div class="s">step {_html(ck_loss.get("min_step"))}</div></div>'
        '</div>'
        '</section>'
    )
    sections = [
        _category_section("Seen Prompts", "In-distribution prompts from the stage-1 training set.", seen),
        _category_section("Holdout Prompts", "Held-out shape/palette/size combinations that were never in the stage-1 train rows.", holdout),
        _category_section("Future-Stage Prompts", "Cards, charts, and curves are in the tokenizer and expected catalog, but this run was not trained on them yet.", future),
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Toy Semantic SVG Probe Report</title>
  <style>
    :root {{
      --bg: #0f1116;
      --bg-2: #171a22;
      --panel: rgba(255,255,255,0.05);
      --border: rgba(255,255,255,0.10);
      --text: #eef2f7;
      --muted: #98a2b3;
      --good: #39d98a;
      --bad: #ff7b72;
      --mid: #ffb020;
      --accent: #7aa2ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(57,217,138,0.10), transparent 24%),
        linear-gradient(180deg, #11141b 0%, #0b0d12 100%);
    }}
    .page {{
      width: min(1520px, calc(100vw - 44px));
      margin: 24px auto 40px;
    }}
    .hero, .panel {{
      border: 1px solid var(--border);
      border-radius: 20px;
      background: var(--panel);
      box-shadow: 0 24px 60px rgba(0,0,0,0.32);
      backdrop-filter: blur(8px);
    }}
    .hero {{
      padding: 28px 30px;
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 22px;
      margin-bottom: 22px;
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(122,162,255,0.16);
      color: #bfd1ff;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 12px 0 10px; font-size: 38px; line-height: 1.04; }}
    h2 {{ margin: 0 0 8px; font-size: 24px; }}
    .subhead, .muted, .meta {{ color: var(--muted); line-height: 1.6; }}
    .meta {{ margin-top: 10px; font-size: 14px; }}
    .hero-metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .hero-card {{
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      border-radius: 16px;
      padding: 16px;
    }}
    .hero-card .k {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 12px;
      font-weight: 700;
    }}
    .hero-card .v {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 800;
    }}
    .hero-card .s {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .panel {{
      padding: 24px 26px;
      margin-bottom: 20px;
    }}
    .probe-stack {{
      display: grid;
      gap: 18px;
      margin-top: 18px;
    }}
    .probe-card {{
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      background: rgba(255,255,255,0.035);
      padding: 18px;
    }}
    .probe-top {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .probe-label {{
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 6px;
    }}
    .probe-prompt {{
      color: #d4def5;
      font-family: ui-monospace, monospace;
      font-size: 13px;
      line-height: 1.6;
      word-break: break-word;
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .chip {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .chip.good {{ background: rgba(57,217,138,0.16); color: #b7f3d4; }}
    .chip.bad {{ background: rgba(255,123,114,0.16); color: #ffc1bb; }}
    .chip.mid {{ background: rgba(255,176,32,0.16); color: #ffd48f; }}
    .chip.stage, .chip.split {{ background: rgba(122,162,255,0.16); color: #bfd1ff; }}
    .preview-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .preview-cell {{
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      border-radius: 14px;
      padding: 12px;
    }}
    .preview-title {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
      font-weight: 700;
    }}
    .svg-frame {{
      min-height: 168px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 12px;
      background: #f8fafc;
      overflow: hidden;
      padding: 10px;
    }}
    .svg-frame svg {{
      width: 100%;
      height: auto;
      max-width: 360px;
      display: block;
    }}
    .empty-preview {{
      color: #475569;
      font-size: 13px;
      line-height: 1.5;
      text-align: center;
    }}
    .text-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .text-title {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
      font-weight: 700;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 96px;
      border-radius: 12px;
      padding: 12px;
      background: #0b1220;
      color: #d8e4ff;
      font-family: ui-monospace, monospace;
      font-size: 12px;
      line-height: 1.55;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    @media (max-width: 980px) {{
      .hero, .preview-grid, .text-grid {{ grid-template-columns: 1fr; }}
      .probe-top {{ flex-direction: column; }}
      .chips {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    {hero}
    {"".join(sections)}
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a styled probe report for the semantic SVG toy run")
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Cache-backed run name")
    ap.add_argument("--run-dir", default=None, help="Explicit run dir")
    ap.add_argument(
        "--catalog",
        default="version/v7/data/generated/toy_svg_semantic_shapes_render_catalog.json",
        help="Expected render catalog JSON",
    )
    ap.add_argument("--max-tokens", type=int, default=32, help="Decode length for probes")
    ap.add_argument(
        "--output",
        default=None,
        help="Optional output HTML path (default: <run-dir>/toy_svg_semantic_probe_report.html)",
    )
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_name, args.run_dir)
    model_dir = run_dir / ".ck_build"
    if not model_dir.exists():
        raise SystemExit(f"missing compiled model dir: {model_dir}")

    train_report = _load_json(run_dir / "train_semantic_shapes_stage_a.json")
    catalog = _load_json(Path(args.catalog).expanduser().resolve())
    expected_map = {entry["prompt"]: entry for entry in catalog}

    prompts = _choose_prompt_sets(expected_map)
    results: list[dict[str, Any]] = []
    for item in prompts:
        expected = expected_map.get(item["prompt"]) or {}
        got = _run_prompt(model_dir, item["prompt"], args.max_tokens)
        exact_match = None
        if expected.get("output_ir") is not None:
            exact_match = got["ir"] == expected.get("output_ir")
        results.append(
            {
                "category": item["category"],
                "label": item["label"],
                "prompt": item["prompt"],
                "stage": expected.get("stage"),
                "split": expected.get("split"),
                "expected_ir": expected.get("output_ir"),
                "expected_svg": expected.get("svg_xml"),
                "exact_match": exact_match,
                **got,
            }
        )

    html_doc = _build_html(run_dir, train_report, results)
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "toy_svg_semantic_probe_report.html"
    compat_output = run_dir / "svg_training_report_card.html"
    output.write_text(html_doc, encoding="utf-8")
    if output != compat_output:
        compat_output.write_text(html_doc, encoding="utf-8")
    (output.with_suffix(".json")).write_text(json.dumps({"run_dir": str(run_dir), "results": results}, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_html": str(output),
                "compat_output_html": str(compat_output),
                "output_json": str(output.with_suffix(".json")),
                "probe_count": len(results),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
