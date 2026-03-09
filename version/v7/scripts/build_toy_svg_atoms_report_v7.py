#!/usr/bin/env python3
"""Build a standalone HTML probe report for the original raw-SVG toy run."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from realtime_svg_semantic_preview_v7 import _extract_response, _python_bin


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_NAME = "toy_svg_atoms_ctx512_d64_h128"
DEFAULT_RUN_ROOT = Path.home() / ".cache" / "ck-engine-v7" / "models" / "train"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _html(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def _resolve_run_dir(run_name: str | None, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    return (DEFAULT_RUN_ROOT / (run_name or DEFAULT_RUN_NAME)).resolve()


def _parse_expected_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        start = raw.find("<svg")
        end = raw.rfind("</svg>")
        if start == -1 or end == -1:
            continue
        prompt = raw[:start]
        svg = raw[start:end + 6]
        rows[prompt] = {"prompt": prompt, "svg": svg}
    return rows


def _normalize_svg(svg: str) -> str:
    return re.sub(r"\s+", " ", svg.strip())


def _extract_svg_parts(text: str) -> tuple[str, str, str]:
    start = text.find("<svg")
    end = text.find("</svg>")
    if start == -1 or end == -1:
        return text.strip(), "", ""
    end += 6
    return text[:start].strip(), text[start:end], text[end:].strip()


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
        "<eos>",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), check=True)
    raw = _extract_response(proc.stdout)
    prefix, svg, suffix = _extract_svg_parts(raw)
    return {
        "raw_response": raw,
        "prefix": prefix,
        "svg": svg,
        "suffix": suffix,
        "renderable": bool(svg),
    }


def _probe_cards(results: list[dict[str, Any]]) -> str:
    out = []
    for row in results:
        exact = row.get("exact_match")
        renderable = bool(row.get("svg"))
        actual_block = row.get("svg") if row.get("svg") else '<div class="empty-preview">No SVG extracted.</div>'
        chips = []
        chips.append(f'<span class="chip {"good" if exact else "bad"}">{"Exact" if exact else "Drift"}</span>')
        chips.append(f'<span class="chip {"good" if renderable else "bad"}">{"Renderable" if renderable else "No SVG"}</span>')
        if row.get("prefix"):
            chips.append('<span class="chip mid">Prefix drift</span>')
        if row.get("suffix"):
            chips.append('<span class="chip mid">Tail drift</span>')
        chips.append(f'<span class="chip stage">{_html(row.get("split"))}</span>')
        out.append(
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
            f'<div><div class="text-title">Prefix / drift</div><pre>{_html(row.get("prefix") or "—")}</pre></div>'
            f'<div><div class="text-title">Tail / extra output</div><pre>{_html(row.get("suffix") or "—")}</pre></div>'
            '</div>'
            '</article>'
        )
    return "".join(out)


def _build_html(run_dir: Path, metrics: dict[str, Any], results: list[dict[str, Any]]) -> str:
    exact = sum(1 for row in results if row.get("exact_match"))
    renderable = sum(1 for row in results if row.get("svg"))
    holdout = [row for row in results if row.get("split") == "holdout"]
    holdout_exact = sum(1 for row in holdout if row.get("exact_match"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Toy SVG Atoms Probe Report</title>
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
    .svg-frame {{ min-height: 150px; display: flex; align-items: center; justify-content: center; border-radius: 12px; background: #f8fafc; overflow: hidden; padding: 10px; }}
    .svg-frame svg {{ width: 100%; height: auto; max-width: 320px; display: block; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; min-height: 96px; border-radius: 12px; padding: 12px; background: #0b1220; color: #d8e4ff; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.55; border: 1px solid rgba(255,255,255,0.08); }}
    .empty-preview {{ color: #475569; font-size: 13px; line-height: 1.5; text-align: center; }}
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
        <span class="eyebrow">Toy SVG Atoms</span>
        <h1>Probe Report</h1>
        <p class="subhead">Original raw-SVG toy run. This report shows the first extracted SVG from each prompt, compares it to the expected target, and highlights prefix or tail drift.</p>
        <div class="meta">Run: {_html(run_dir.name)} | Final loss: {_html(metrics.get("final_loss"))}</div>
      </div>
      <div class="hero-metrics">
        <div class="hero-card"><div class="k">Exact Match</div><div class="v">{exact}/{len(results)}</div><div class="s">{(100.0 * exact / max(len(results), 1)):.0f}% of probes</div></div>
        <div class="hero-card"><div class="k">Renderable</div><div class="v">{renderable}/{len(results)}</div><div class="s">{(100.0 * renderable / max(len(results), 1)):.0f}% with extracted SVG</div></div>
        <div class="hero-card"><div class="k">Holdout Exact</div><div class="v">{holdout_exact}/{len(holdout)}</div><div class="s">generalization probes</div></div>
        <div class="hero-card"><div class="k">Known Weakness</div><div class="v">Raw XML</div><div class="s">prompt/output boundary drift is expected here</div></div>
      </div>
    </section>
    <section class="panel">
      <h2>Prompt Probes</h2>
      <p class="muted">Seen prompts and held-out combinations from the original DSL. This run predates the structured semantic IR, so exact XML match is a higher bar.</p>
      <div class="probe-stack">{_probe_cards(results)}</div>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a probe report for the original toy SVG atoms run")
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Cache-backed run name")
    ap.add_argument("--run-dir", default=None, help="Explicit run dir")
    ap.add_argument("--max-tokens", type=int, default=96, help="Decode length for probes")
    ap.add_argument("--output", default=None, help="Optional output HTML path")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_name, args.run_dir)
    model_dir = run_dir / ".ck_build"
    if not model_dir.exists():
        raise SystemExit(f"missing compiled model dir: {model_dir}")

    expected_map = _parse_expected_rows(ROOT / "version/v7/data/generated/toy_svg_atoms_all.txt")
    seen = (ROOT / "version/v7/data/generated/toy_svg_atoms_seen_prompts.txt").read_text(encoding="utf-8").splitlines()
    holdout = (ROOT / "version/v7/data/generated/toy_svg_atoms_holdout_prompts.txt").read_text(encoding="utf-8").splitlines()
    prompt_defs = [
        ("seen", "Seen: red circle small", "[shape:circle][color:red][size:small]"),
        ("seen", "Seen: green circle big", "[shape:circle][color:green][size:big]"),
        ("seen", "Seen: blue rect big", "[shape:rect][color:blue][size:big]"),
        ("seen", "Seen: red triangle big", "[shape:triangle][color:red][size:big]"),
    ] + [("holdout", f"Holdout {idx+1}", prompt) for idx, prompt in enumerate(holdout)]

    metrics = {
        "final_loss": None,
    }
    overfit_path = run_dir / "train_toy_svg_atoms_stage_a_overfit.json"
    if overfit_path.exists():
        doc = _load_json(overfit_path)
        metrics["final_loss"] = ((doc.get("ck_loss") or {}).get("final"))

    results: list[dict[str, Any]] = []
    for split, label, prompt in prompt_defs:
        expected = expected_map.get(prompt, {})
        got = _run_prompt(model_dir, prompt, args.max_tokens)
        exact_match = bool(got.get("svg")) and _normalize_svg(got.get("svg", "")) == _normalize_svg(expected.get("svg", ""))
        results.append(
            {
                "split": split,
                "label": label,
                "prompt": prompt,
                "expected_svg": expected.get("svg", ""),
                "exact_match": exact_match,
                **got,
            }
        )

    html_doc = _build_html(run_dir, metrics, results)
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "toy_svg_atoms_probe_report.html"
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
