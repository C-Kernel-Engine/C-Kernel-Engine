#!/usr/bin/env python3
"""Build a smoke report for the spec_broader_1 comparison gold pack."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from bootstrap_spec_broader_1_comparison_gold_pack_v7 import CASES, OUT_DIR as GOLD_DIR, STATUS_JSON, write_gold_pack
from render_svg_structured_scene_spec09_v7 import render_structured_scene_spec09_svg


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "version" / "v7" / "reports"
OUT_DIR = REPORTS / "spec_broader_1_smoke" / "comparison_span_chart"


def _rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start=base.resolve())).as_posix()


def _text_present(svg: str, value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text in svg:
        return True
    tokens = [part for part in text.replace("/", " ").split() if part]
    if len(tokens) >= 2:
        return all(token in svg for token in tokens[:2])
    return tokens[0] in svg if tokens else False


def _metrics(svg: str, content: dict[str, Any]) -> dict[str, Any]:
    title = content["title"]["headline"]
    labels = [
        str(content["bars"]["primary"]["label"]),
        str(content["bars"]["secondary"]["label"]),
    ]
    values = [
        str(content["bars"]["primary"]["value"]),
        str(content["bars"]["secondary"]["value"]),
    ]
    return {
        "compile_ok": svg.startswith("<svg") and svg.endswith("</svg>"),
        "headline_present": _text_present(svg, title),
        "label_matches": sum(1 for label in labels if _text_present(svg, label)),
        "value_matches": sum(1 for value in values if _text_present(svg, value)),
        "rects": svg.count("<rect "),
    }


def _build_rows() -> list[dict[str, Any]]:
    write_gold_pack()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in CASES:
        scene_path = GOLD_DIR / f"{case.case_id}.scene.compact.dsl"
        content_path = GOLD_DIR / f"{case.case_id}.content.json"
        scene_text = scene_path.read_text(encoding="utf-8")
        content = json.loads(content_path.read_text(encoding="utf-8"))
        svg = render_structured_scene_spec09_svg(scene_text, content=content)
        svg_path = OUT_DIR / f"{case.case_id}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        rows.append(
            {
                "case_id": case.case_id,
                "asset": case.asset,
                "scene_path": str(scene_path),
                "content_path": str(content_path),
                "svg_path": str(svg_path),
                "rationale": case.rationale,
                "metrics": _metrics(svg, content),
            }
        )
    return rows


def _html(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        metrics = row["metrics"]
        cards.append(
            f"""
            <section class="card">
              <div class="meta">
                <div>
                  <h2>{html.escape(row['asset'])}</h2>
                  <p class="sub">{html.escape(row['rationale'])}</p>
                </div>
                <div class="pill">comparison_span_chart</div>
              </div>
              <div class="checks">
                <span class="ok">compile_ok={str(metrics['compile_ok']).lower()}</span>
                <span>headline={str(metrics['headline_present']).lower()}</span>
                <span>labels={metrics['label_matches']}/2</span>
                <span>values={metrics['value_matches']}/2</span>
                <span>rects={metrics['rects']}</span>
              </div>
              <div class="paths">
                <a href="{html.escape(_rel(Path(row['scene_path']), OUT_DIR))}">scene DSL</a>
                <a href="{html.escape(_rel(Path(row['content_path']), OUT_DIR))}">content JSON</a>
                <a href="{html.escape(_rel(Path(row['svg_path']), OUT_DIR))}">compiled SVG</a>
              </div>
              <div class="frame">
                <img src="{html.escape(_rel(Path(row['svg_path']), OUT_DIR))}" alt="{html.escape(row['case_id'])}"/>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Spec Broader 1 Comparison Smoke</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --ink: #172430;
      --muted: #5a6874;
      --card: rgba(255,255,255,0.9);
      --line: rgba(23,36,48,0.10);
      --accent: #0c6a83;
      --ok: #1f6d42;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Iowan Old Style", Georgia, serif; background: linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%); color: var(--ink); }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 48px 24px 64px; }}
    h1 {{ margin: 0 0 10px; font-size: 44px; line-height: 1.05; }}
    .lede {{ max-width: 920px; color: var(--muted); font-size: 18px; line-height: 1.6; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 28px 0 36px; }}
    .summary .box, .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 20px; box-shadow: 0 16px 40px rgba(34,48,58,0.08); }}
    .summary .box {{ padding: 18px 20px; }}
    .summary .k {{ font: 700 28px/1.1 "IBM Plex Sans", "Segoe UI", sans-serif; }}
    .summary .v {{ color: var(--muted); margin-top: 6px; font: 500 13px/1.4 "IBM Plex Sans", "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
    .card {{ padding: 20px; }}
    .meta {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
    .meta h2 {{ margin: 0 0 6px; font-size: 28px; }}
    .sub {{ margin: 0; color: var(--muted); font-size: 15px; line-height: 1.5; }}
    .pill {{ padding: 8px 12px; border-radius: 999px; background: rgba(12,106,131,0.10); color: var(--accent); font: 700 12px/1 "IBM Plex Sans", "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.08em; }}
    .checks, .paths {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; font: 600 12px/1.4 "IBM Plex Sans", "Segoe UI", sans-serif; }}
    .checks span, .paths a {{ padding: 7px 10px; border-radius: 999px; background: rgba(23,36,48,0.06); color: var(--ink); text-decoration: none; }}
    .checks .ok {{ background: rgba(31,109,66,0.12); color: var(--ok); }}
    .frame {{ margin-top: 18px; border-radius: 16px; overflow: hidden; border: 1px solid var(--line); background: #fff; min-height: 220px; }}
    .frame img {{ display: block; width: 100%; height: auto; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Spec Broader 1 Comparison Smoke</h1>
      <p class="lede">
        This is the first compiler-first gold pack for the broader branch. The assets stay inside the existing
        <code>comparison_span_chart</code> renderer, but all visible copy and values are externalized into
        <code>content.json</code> so the DSL remains structural.
      </p>
    </header>
    <section class="summary">
      <div class="box"><div class="k">{len(rows)}</div><div class="v">Gold Cases</div></div>
      <div class="box"><div class="k">{sum(1 for row in rows if row['metrics']['compile_ok'])}</div><div class="v">Compile OK</div></div>
      <div class="box"><div class="k">{sum(row['metrics']['headline_present'] for row in rows)}/{len(rows)}</div><div class="v">Headline Present</div></div>
      <div class="box"><div class="k">{sum(row['metrics']['label_matches'] for row in rows)}/{2 * len(rows)}</div><div class="v">Bar Labels Present</div></div>
      <div class="box"><div class="k">{sum(row['metrics']['value_matches'] for row in rows)}/{2 * len(rows)}</div><div class="v">Bar Values Present</div></div>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    rows = _build_rows()
    report_json = REPORTS / "spec_broader_1_smoke" / "comparison_span_chart_compiler_smoke_report.json"
    report_html = REPORTS / "spec_broader_1_smoke" / "comparison_span_chart_compiler_smoke_report.html"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    report_html.write_text(_html(rows), encoding="utf-8")
    print(f"[OK] wrote: {STATUS_JSON}")
    print(f"[OK] wrote: {report_json}")
    print(f"[OK] wrote: {report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
