#!/usr/bin/env python3
"""Build a standalone HTML probe report for the structured SVG atoms toy run."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path
from typing import Any

from realtime_svg_semantic_preview_v7 import _extract_response, _python_bin


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_NAME = "toy_svg_structured_atoms_ctx512_d64_h128"
DEFAULT_RUN_ROOT = Path.home() / ".cache" / "ck-engine-v7" / "models" / "train"


def _html(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(run_name: str | None, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    return (DEFAULT_RUN_ROOT / (run_name or DEFAULT_RUN_NAME)).resolve()


def render_structured_ir(text: str) -> str:
    tokens = [tok.strip() for tok in text.split() if tok.strip()]
    if "[/svg]" in tokens:
        tokens = tokens[: tokens.index("[/svg]") + 1]
    width = "128"
    height = "128"
    fill = "red"
    stroke = "black"
    stroke_width = "2"
    if "[w:128]" in tokens:
        width = "128"
    if "[h:128]" in tokens:
        height = "128"
    for tok in tokens:
        if tok.startswith("[fill:") and tok.endswith("]"):
            fill = tok[len("[fill:") : -1]
        if tok.startswith("[stroke:") and tok.endswith("]"):
            stroke = tok[len("[stroke:") : -1]
        if tok.startswith("[sw:") and tok.endswith("]"):
            stroke_width = tok[len("[sw:") : -1]
    if "[circle]" in tokens:
        attrs = {
            "cx": "64",
            "cy": "64",
            "r": "18",
        }
        for tok in tokens:
            if tok.startswith("[cx:") and tok.endswith("]"):
                attrs["cx"] = tok[len("[cx:") : -1]
            if tok.startswith("[cy:") and tok.endswith("]"):
                attrs["cy"] = tok[len("[cy:") : -1]
            if tok.startswith("[r:") and tok.endswith("]"):
                attrs["r"] = tok[len("[r:") : -1]
        body = f'<circle cx="{attrs["cx"]}" cy="{attrs["cy"]}" r="{attrs["r"]}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    elif "[rect]" in tokens:
        attrs = {"x": "42", "y": "48", "width": "44", "height": "32", "rx": "6"}
        for key in list(attrs):
            prefix = f"[{key}:"
            for tok in tokens:
                if tok.startswith(prefix) and tok.endswith("]"):
                    attrs[key] = tok[len(prefix) : -1]
        body = (
            f'<rect x="{attrs["x"]}" y="{attrs["y"]}" width="{attrs["width"]}" '
            f'height="{attrs["height"]}" rx="{attrs["rx"]}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    elif "[polygon]" in tokens:
        points = "64,34 36,86 92,86"
        for tok in tokens:
            if tok.startswith("[points:") and tok.endswith("]"):
                points = tok[len("[points:") : -1].replace("|", " ")
        body = f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    else:
        raise SystemExit("unsupported structured IR")
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


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
    raw = _extract_response(proc.stdout)
    ir = raw
    if "[/svg]" in raw:
        ir = raw.split("[/svg]", 1)[0].strip() + " [/svg]"
    render_error = None
    rendered_svg = None
    try:
        rendered_svg = render_structured_ir(ir)
    except BaseException as exc:  # pragma: no cover
        render_error = str(exc)
    return {
        "raw_ir": raw,
        "ir": ir,
        "rendered_svg": rendered_svg,
        "render_error": render_error,
        "tail_drift": raw != ir,
    }


def _build_probe_card(row: dict[str, Any]) -> str:
    exact = row.get("exact_match")
    renderable = bool(row.get("rendered_svg"))
    actual_block = row.get("rendered_svg") if renderable else f'<div class="empty-preview">{_html(row.get("render_error") or "Not renderable")}</div>'
    chips = [
        f'<span class="chip {"good" if exact else "bad"}">{"Exact" if exact else "Drift"}</span>',
        f'<span class="chip {"good" if renderable else "bad"}">{"Renderable" if renderable else "Render fail"}</span>',
        f'<span class="chip stage">{_html(row.get("split") or "n/a")}</span>',
    ]
    if row.get("tail_drift"):
        chips.append('<span class="chip mid">Tail drift</span>')
    return (
        '<article class="probe-card">'
        f'<div class="probe-top"><div><div class="probe-label">{_html(row.get("label"))}</div>'
        f'<div class="probe-prompt">{_html(row.get("prompt"))}</div></div><div class="chips">{"".join(chips)}</div></div>'
        '<div class="preview-grid">'
        '<div class="preview-cell"><div class="preview-title">Actual SVG</div>'
        f'<div class="svg-frame">{actual_block}</div></div>'
        '<div class="preview-cell"><div class="preview-title">Expected target</div>'
        f'<div class="svg-frame">{row.get("expected_svg")}</div></div>'
        '</div>'
        '<div class="text-grid">'
        f'<div><div class="text-title">Generated IR</div><pre>{_html(row.get("ir"))}</pre></div>'
        f'<div><div class="text-title">Expected IR</div><pre>{_html(row.get("expected_ir"))}</pre></div>'
        '</div>'
        '</article>'
    )


def _build_html(run_dir: Path, loss: dict[str, Any], results: list[dict[str, Any]]) -> str:
    exact = sum(1 for row in results if row.get("exact_match"))
    renderable = sum(1 for row in results if row.get("rendered_svg"))
    holdouts = [row for row in results if row.get("split") == "holdout"]
    holdout_exact = sum(1 for row in holdouts if row.get("exact_match"))
    cards = "".join(_build_probe_card(row) for row in results)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Toy Structured SVG Probe Report</title>
  <style>
    :root {{
      --text: #eef2f7;
      --muted: #98a2b3;
      --panel: rgba(255,255,255,0.05);
      --border: rgba(255,255,255,0.10);
      --good: #39d98a;
      --bad: #ff7b72;
      --mid: #ffb020;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(57,217,138,0.10), transparent 24%),
        linear-gradient(180deg, #11141b 0%, #0b0d12 100%);
    }}
    .page {{ width: min(1500px, calc(100vw - 44px)); margin: 24px auto 40px; }}
    .hero, .panel {{ border: 1px solid var(--border); border-radius: 20px; background: var(--panel); box-shadow: 0 24px 60px rgba(0,0,0,0.32); backdrop-filter: blur(8px); }}
    .hero {{ padding: 28px 30px; display: grid; grid-template-columns: 1.3fr 1fr; gap: 22px; margin-bottom: 22px; }}
    .eyebrow {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: rgba(122,162,255,0.16); color: #bfd1ff; font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }}
    h1 {{ margin: 12px 0 10px; font-size: 38px; line-height: 1.04; }}
    h2 {{ margin: 0 0 8px; font-size: 24px; }}
    .subhead, .muted, .meta {{ color: var(--muted); line-height: 1.6; }}
    .meta {{ margin-top: 10px; font-size: 14px; }}
    .hero-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .hero-card {{ border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.04); border-radius: 16px; padding: 16px; }}
    .hero-card .k {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 12px; font-weight: 700; }}
    .hero-card .v {{ margin-top: 8px; font-size: 28px; font-weight: 800; }}
    .hero-card .s {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .panel {{ padding: 24px 26px; }}
    .probe-stack {{ display: grid; gap: 18px; margin-top: 18px; }}
    .probe-card {{ border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; background: rgba(255,255,255,0.035); padding: 18px; }}
    .probe-top {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 14px; }}
    .probe-label {{ font-size: 18px; font-weight: 800; margin-bottom: 6px; }}
    .probe-prompt {{ color: #d4def5; font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.6; word-break: break-word; }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .chip {{ display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }}
    .chip.good {{ background: rgba(57,217,138,0.16); color: #b7f3d4; }}
    .chip.bad {{ background: rgba(255,123,114,0.16); color: #ffc1bb; }}
    .chip.mid {{ background: rgba(255,176,32,0.16); color: #ffd48f; }}
    .chip.stage {{ background: rgba(122,162,255,0.16); color: #bfd1ff; }}
    .preview-grid, .text-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .preview-grid {{ margin-bottom: 14px; }}
    .preview-cell {{ border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); border-radius: 14px; padding: 12px; }}
    .preview-title, .text-title {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; font-weight: 700; }}
    .svg-frame {{ min-height: 160px; display: flex; align-items: center; justify-content: center; border-radius: 12px; background: #f8fafc; overflow: hidden; padding: 10px; }}
    .svg-frame svg {{ width: 100%; height: auto; max-width: 320px; display: block; }}
    .empty-preview {{ color: #475569; font-size: 13px; line-height: 1.5; text-align: center; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; min-height: 96px; border-radius: 12px; padding: 12px; background: #0b1220; color: #d8e4ff; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.55; border: 1px solid rgba(255,255,255,0.08); }}
    @media (max-width: 980px) {{
      .hero, .preview-grid, .text-grid {{ grid-template-columns: 1fr; }}
      .probe-top {{ flex-direction: column; }}
      .chips {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <span class="eyebrow">Toy Structured SVG</span>
        <h1>Probe Report</h1>
        <p class="subhead">Structured SVG IR run with a fixed no-merge tokenizer. This report probes seen and holdout prompts, renders the generated IR, and compares it to the expected target.</p>
        <div class="meta">Run: {_html(run_dir.name)} | Steps: {_html(loss.get("steps"))} | Loss: {_html(loss.get("first"))} → {_html(loss.get("final"))}</div>
      </div>
      <div class="hero-metrics">
        <div class="hero-card"><div class="k">Exact Match</div><div class="v">{exact}/{len(results)}</div><div class="s">{(100.0 * exact / max(len(results), 1)):.0f}% of probes</div></div>
        <div class="hero-card"><div class="k">Renderable</div><div class="v">{renderable}/{len(results)}</div><div class="s">{(100.0 * renderable / max(len(results), 1)):.0f}% render to SVG</div></div>
        <div class="hero-card"><div class="k">Holdout Exact</div><div class="v">{holdout_exact}/{len(holdouts)}</div><div class="s">generalization probes</div></div>
        <div class="hero-card"><div class="k">Best Loss</div><div class="v">{float(loss.get("min", 0.0)):.4f}</div><div class="s">step {_html(loss.get("min_step"))}</div></div>
      </div>
    </section>
    <section class="panel">
      <h2>Prompt Probes</h2>
      <p class="muted">These probes are drawn from the structured toy catalog. Because the output is symbolic IR, exact match is a better signal here than it was for the raw XML toy.</p>
      <div class="probe-stack">{cards}</div>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a probe report for the structured SVG toy run")
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Cache-backed run name")
    ap.add_argument("--run-dir", default=None, help="Explicit run dir")
    ap.add_argument("--max-tokens", type=int, default=24, help="Decode length for probes")
    ap.add_argument("--output", default=None, help="Optional output HTML path")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_name, args.run_dir)
    model_dir = run_dir / ".ck_build"
    if not model_dir.exists():
        raise SystemExit(f"missing compiled model dir: {model_dir}")

    catalog = _load_json(ROOT / "version/v7/data/generated/toy_svg_structured_atoms_render_catalog.json")
    catalog_by_prompt = {row["prompt"]: row for row in catalog}
    prompts = [
        ("Seen: red circle small", "[task:svg] [shape:circle] [color:red] [size:small] [OUT]"),
        ("Seen: green circle big", "[task:svg] [shape:circle] [color:green] [size:big] [OUT]"),
        ("Seen: blue rect small", "[task:svg] [shape:rect] [color:blue] [size:small] [OUT]"),
        ("Seen: red triangle small", "[task:svg] [shape:triangle] [color:red] [size:small] [OUT]"),
        ("Holdout: red circle big", "[task:svg] [shape:circle] [color:red] [size:big] [OUT]"),
        ("Holdout: blue circle small", "[task:svg] [shape:circle] [color:blue] [size:small] [OUT]"),
        ("Holdout: green rect big", "[task:svg] [shape:rect] [color:green] [size:big] [OUT]"),
        ("Holdout: blue triangle big", "[task:svg] [shape:triangle] [color:blue] [size:big] [OUT]"),
    ]

    loss = (_load_json(run_dir / "train_structured_svg_atoms_stage_a.json").get("ck_loss") or {})
    results: list[dict[str, Any]] = []
    for label, prompt in prompts:
        expected = catalog_by_prompt[prompt]
        got = _run_prompt(model_dir, prompt, args.max_tokens)
        results.append(
            {
                "label": label,
                "prompt": prompt,
                "split": expected.get("split"),
                "expected_ir": expected.get("output_tokens"),
                "expected_svg": expected.get("svg_xml"),
                "exact_match": got.get("ir") == expected.get("output_tokens"),
                **got,
            }
        )

    html_doc = _build_html(run_dir, loss, results)
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "toy_svg_structured_probe_report.html"
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
