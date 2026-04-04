#!/usr/bin/env python3
"""Build a compiler smoke report for the spec14a comparison-board gold scenes."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec14a_v7 import render_structured_scene_spec14a_svg


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "version" / "v7" / "reports"
SPEC14A_GOLD = REPORTS / "spec14a_gold_mappings"
OUT_DIR = REPORTS / "spec14a_smoke" / "gold_compiled"


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    scene_path: Path
    content_path: Path
    source_asset: str
    rationale: str


def _cases() -> list[SmokeCase]:
    return [
        SmokeCase(
            case_id="tokenizer_performance_comparison",
            scene_path=SPEC14A_GOLD / "tokenizer-performance-comparison.scene.compact.dsl",
            content_path=SPEC14A_GOLD / "tokenizer-performance-comparison.content.json",
            source_asset="tokenizer-performance-comparison.svg",
            rationale="Performance comparison board with three columns, four metric cards, and two operator callouts.",
        ),
        SmokeCase(
            case_id="tokenizer_algorithms_comparison",
            scene_path=SPEC14A_GOLD / "sentencepiece-vs-bpe-wordpiece.scene.compact.dsl",
            content_path=SPEC14A_GOLD / "sentencepiece-vs-bpe-wordpiece.content.json",
            source_asset="sentencepiece-vs-bpe-wordpiece.svg",
            rationale="Algorithm comparison board proving the family can compare three approaches without topic ids in the DSL.",
        ),
        SmokeCase(
            case_id="rope_layouts_comparison",
            scene_path=SPEC14A_GOLD / "rope-layouts-compared.scene.compact.dsl",
            content_path=SPEC14A_GOLD / "rope-layouts-compared.content.json",
            source_asset="rope-layouts-compared.svg",
            rationale="Compatibility-oriented board with three lanes and three action callouts.",
        ),
        SmokeCase(
            case_id="compute_bandwidth_chasm",
            scene_path=SPEC14A_GOLD / "compute-bandwidth-chasm.scene.compact.dsl",
            content_path=SPEC14A_GOLD / "compute-bandwidth-chasm.content.json",
            source_asset="compute-bandwidth-chasm.svg",
            rationale="Systems comparison board comparing regimes rather than only objects.",
        ),
        SmokeCase(
            case_id="quantization_formats_comparison",
            scene_path=SPEC14A_GOLD / "quantization-formats.scene.compact.dsl",
            content_path=SPEC14A_GOLD / "quantization-formats.content.json",
            source_asset="quantization-formats.svg",
            rationale="Dense technical board testing whether the renderer can keep heavier comparison payloads readable.",
        ),
    ]


def _svg_rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), start=base.resolve())).as_posix()


def _text_present(svg: str, value: str) -> bool:
    return bool(value) and value in svg


def _column_titles(content: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    columns = content.get("columns") if isinstance(content.get("columns"), dict) else {}
    for payload in columns.values():
        if isinstance(payload, dict):
            title = str(payload.get("title") or "").strip()
            if title:
                titles.append(title)
    return titles


def _metric_values(content: dict[str, Any]) -> list[str]:
    values: list[str] = []
    metrics = content.get("metrics") if isinstance(content.get("metrics"), dict) else {}
    for payload in metrics.values():
        if isinstance(payload, dict):
            value = str(payload.get("value") or "").strip()
            if value:
                values.append(value)
    return values


def _metrics(svg: str, content: dict[str, Any]) -> dict[str, Any]:
    headline = str(((content.get("header") or {}).get("headline")) or "").strip()
    column_titles = _column_titles(content)
    metric_values = _metric_values(content)
    matched_columns = [title for title in column_titles if _text_present(svg, title)]
    matched_metrics = [value for value in metric_values if _text_present(svg, value)]
    return {
        "compile_ok": svg.startswith("<svg") and svg.endswith("</svg>"),
        "headline_present": _text_present(svg, headline),
        "column_title_matches": len(matched_columns),
        "column_title_total": len(column_titles),
        "metric_value_matches": len(matched_metrics),
        "metric_value_total": len(metric_values),
        "matched_columns": matched_columns,
        "matched_metrics": matched_metrics,
        "rects": svg.count("<rect "),
    }


def _build_rows() -> list[dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in _cases():
        scene_text = case.scene_path.read_text(encoding="utf-8").strip()
        content = json.loads(case.content_path.read_text(encoding="utf-8"))
        svg = render_structured_scene_spec14a_svg(scene_text, content=content)
        svg_path = OUT_DIR / f"{case.case_id}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        metrics = _metrics(svg, content)
        rows.append(
            {
                "case_id": case.case_id,
                "scene_path": str(case.scene_path),
                "content_path": str(case.content_path),
                "svg_path": str(svg_path),
                "source_asset": case.source_asset,
                "rationale": case.rationale,
                "headline": str(((content.get("header") or {}).get("headline")) or ""),
                "metrics": metrics,
            }
        )
    return rows


def _html(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        metrics = row["metrics"]
        svg_rel = _svg_rel(Path(row["svg_path"]), OUT_DIR)
        cards.append(
            f"""
            <section class="card">
              <div class="meta">
                <div>
                  <h2>{html.escape(row['headline'] or row['case_id'])}</h2>
                  <p class="sub">{html.escape(row['rationale'])}</p>
                </div>
                <div class="pill">comparison_board</div>
              </div>
              <div class="checks">
                <span class="ok">compile_ok={str(metrics['compile_ok']).lower()}</span>
                <span>headline={str(metrics['headline_present']).lower()}</span>
                <span>columns={metrics['column_title_matches']}/{metrics['column_title_total']}</span>
                <span>metrics={metrics['metric_value_matches']}/{metrics['metric_value_total']}</span>
                <span>rects={metrics['rects']}</span>
              </div>
              <div class="paths">
                <a href="{html.escape(_svg_rel(Path(row['scene_path']), OUT_DIR))}">scene DSL</a>
                <a href="{html.escape(_svg_rel(Path(row['content_path']), OUT_DIR))}">content JSON</a>
                <a href="{html.escape(svg_rel)}">compiled SVG</a>
                <span>asset target: {html.escape(row['source_asset'])}</span>
              </div>
              <div class="frame">
                <img src="{html.escape(svg_rel)}" alt="{html.escape(row['case_id'])}"/>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Spec14a Compiler Smoke Report</title>
  <style>
    :root {{
      --bg: #f2efe8;
      --ink: #172430;
      --muted: #5a6874;
      --card: rgba(255,255,255,0.9);
      --line: rgba(23,36,48,0.10);
      --accent: #0c6a83;
      --ok: #1f6d42;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      background:
        radial-gradient(circle at top left, rgba(12,106,131,0.10), transparent 30%),
        linear-gradient(180deg, #f8f5ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 48px 24px 64px; }}
    header {{ margin-bottom: 28px; }}
    h1 {{ margin: 0 0 10px; font-size: 44px; line-height: 1.05; }}
    .lede {{ max-width: 920px; color: var(--muted); font-size: 18px; line-height: 1.6; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 28px 0 36px;
    }}
    .summary .box, .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 16px 40px rgba(34, 48, 58, 0.08);
      backdrop-filter: blur(8px);
    }}
    .summary .box {{ padding: 18px 20px; }}
    .summary .k {{ font: 700 28px/1.1 "IBM Plex Sans", "Segoe UI", sans-serif; }}
    .summary .v {{ color: var(--muted); margin-top: 6px; font: 500 13px/1.4 "IBM Plex Sans", "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 20px;
    }}
    .card {{ padding: 20px; }}
    .meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .meta h2 {{ margin: 0 0 6px; font-size: 28px; }}
    .sub {{ margin: 0; color: var(--muted); font-size: 15px; line-height: 1.5; }}
    .pill {{
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(12,106,131,0.1);
      color: var(--accent);
      font: 700 12px/1 "IBM Plex Sans", "Segoe UI", sans-serif;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      white-space: nowrap;
    }}
    .checks, .paths {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
      font: 600 12px/1.4 "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .checks span, .paths a, .paths span {{
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(23,36,48,0.06);
      color: var(--ink);
      text-decoration: none;
    }}
    .checks .ok {{ background: rgba(31,109,66,0.12); color: var(--ok); }}
    .frame {{
      margin-top: 18px;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #fff;
      min-height: 220px;
    }}
    .frame img {{ display: block; width: 100%; height: auto; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Spec14a Compiler Smoke Report</h1>
      <p class="lede">
        This report checks whether the successor comparison-board family is visually viable before any tokenizer rebuild
        or training rung starts. The board family keeps payload facts outside the scene DSL and uses only reusable
        columns, metrics, and callouts inside the model output contract.
      </p>
    </header>
    <section class="summary">
      <div class="box"><div class="k">{len(rows)}</div><div class="v">Gold Board Cases</div></div>
      <div class="box"><div class="k">{sum(1 for row in rows if row['metrics']['compile_ok'])}</div><div class="v">Compile OK</div></div>
      <div class="box"><div class="k">{sum(row['metrics']['column_title_matches'] for row in rows)}/{sum(row['metrics']['column_title_total'] for row in rows)}</div><div class="v">Column Titles Present</div></div>
      <div class="box"><div class="k">{sum(row['metrics']['metric_value_matches'] for row in rows)}/{sum(row['metrics']['metric_value_total'] for row in rows)}</div><div class="v">Metric Values Present</div></div>
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
    report_json = REPORTS / "spec14a_smoke" / "compiler_smoke_report.json"
    report_html = REPORTS / "spec14a_smoke" / "compiler_smoke_report.html"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_html.write_text(_html(rows), encoding="utf-8")
    print(f"[OK] wrote: {report_json}")
    print(f"[OK] wrote: {report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
