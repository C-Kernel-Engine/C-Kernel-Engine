#!/usr/bin/env python3
"""Build a compiler smoke report for the current spec15b system_diagram gold scenes."""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec15b_v7 import render_structured_scene_spec15b_svg


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "version" / "v7" / "reports"
SPEC15B_GOLD = REPORTS / "spec15b_gold_mappings"
OUT_DIR = REPORTS / "spec15b_smoke" / "gold_compiled"


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    scene_path: Path
    compact_scene_path: Path
    content_path: Path
    source_asset: str
    rationale: str


def _cases() -> list[SmokeCase]:
    return [
        SmokeCase(
            case_id="pipeline_overview_system",
            scene_path=SPEC15B_GOLD / "pipeline-overview-system.scene.dsl",
            compact_scene_path=SPEC15B_GOLD / "pipeline-overview-system.scene.compact.dsl",
            content_path=SPEC15B_GOLD / "pipeline-overview-system.content.json",
            source_asset="pipeline-overview.svg",
            rationale="Prompt-to-SVG system pipeline using four ordered stages and one terminal result panel.",
        ),
        SmokeCase(
            case_id="ir_pipeline_flow_system",
            scene_path=SPEC15B_GOLD / "ir-pipeline-flow-system.scene.dsl",
            compact_scene_path=SPEC15B_GOLD / "ir-pipeline-flow-system.scene.compact.dsl",
            content_path=SPEC15B_GOLD / "ir-pipeline-flow-system.content.json",
            source_asset="ir-pipeline-flow.svg",
            rationale="Compiler/runtime system flow under the same bounded stage-link-terminal contract.",
        ),
        SmokeCase(
            case_id="kernel_registry_flow_system",
            scene_path=SPEC15B_GOLD / "kernel-registry-flow-system.scene.dsl",
            compact_scene_path=SPEC15B_GOLD / "kernel-registry-flow-system.scene.compact.dsl",
            content_path=SPEC15B_GOLD / "kernel-registry-flow-system.content.json",
            source_asset="kernel-registry-flow.svg",
            rationale="Kernel-resolution system flow showing that the family is generic across payload topics.",
        ),
    ]


def _svg_rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), start=base.resolve())).as_posix()


def _text_present(svg: str, value: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    plain = re.sub(r"<[^>]+>", " ", svg)
    plain = html.unescape(" ".join(plain.split()))
    return raw in plain


def _stage_titles(content: dict[str, Any]) -> list[str]:
    out: list[str] = []
    stages = content.get("stages") if isinstance(content.get("stages"), dict) else {}
    for payload in stages.values():
        if isinstance(payload, dict):
            title = str(payload.get("title") or "").strip()
            if title:
                out.append(title)
    return out


def _metrics(full_svg: str, compact_svg: str, content: dict[str, Any]) -> dict[str, Any]:
    header = content.get("header") if isinstance(content.get("header"), dict) else {}
    terminal = content.get("terminal") if isinstance(content.get("terminal"), dict) else {}
    headline = str(header.get("headline") or "").strip()
    subtitle = str(header.get("subtitle") or "").strip()
    terminal_title = str(terminal.get("title") or "").strip()
    titles = _stage_titles(content)
    matched_titles = [title for title in titles if _text_present(compact_svg, title)]
    return {
        "compile_ok_full": full_svg.startswith("<svg") and full_svg.endswith("</svg>"),
        "compile_ok_compact": compact_svg.startswith("<svg") and compact_svg.endswith("</svg>"),
        "full_compact_match": full_svg == compact_svg,
        "headline_present": _text_present(compact_svg, headline),
        "subtitle_present": _text_present(compact_svg, subtitle),
        "terminal_title_present": _text_present(compact_svg, terminal_title),
        "title_matches": len(matched_titles),
        "title_total": len(titles),
        "rects": compact_svg.count("<rect "),
        "paths": compact_svg.count("<path "),
    }


def _build_rows() -> list[dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in _cases():
        full_scene = case.scene_path.read_text(encoding="utf-8").strip()
        compact_scene = case.compact_scene_path.read_text(encoding="utf-8").strip()
        content = json.loads(case.content_path.read_text(encoding="utf-8"))
        full_svg = render_structured_scene_spec15b_svg(full_scene, content=content)
        compact_svg = render_structured_scene_spec15b_svg(compact_scene, content=content)
        full_svg_path = OUT_DIR / f"{case.case_id}.full.svg"
        compact_svg_path = OUT_DIR / f"{case.case_id}.compact.svg"
        full_svg_path.write_text(full_svg, encoding="utf-8")
        compact_svg_path.write_text(compact_svg, encoding="utf-8")
        rows.append(
            {
                "case_id": case.case_id,
                "scene_path": str(case.scene_path),
                "compact_scene_path": str(case.compact_scene_path),
                "content_path": str(case.content_path),
                "full_svg_path": str(full_svg_path),
                "compact_svg_path": str(compact_svg_path),
                "source_asset": case.source_asset,
                "rationale": case.rationale,
                "headline": str(((content.get("header") or {}).get("headline")) or ""),
                "metrics": _metrics(full_svg, compact_svg, content),
            }
        )
    return rows


def _html(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        metrics = row["metrics"]
        compact_svg_rel = _svg_rel(Path(row["compact_svg_path"]), OUT_DIR)
        cards.append(
            f"""
            <section class="card">
              <div class="meta">
                <div>
                  <h2>{html.escape(row['headline'] or row['case_id'])}</h2>
                  <p class="sub">{html.escape(row['rationale'])}</p>
                </div>
                <div class="pill">system_diagram</div>
              </div>
              <div class="checks">
                <span class="ok">full={str(metrics['compile_ok_full']).lower()}</span>
                <span class="ok">compact={str(metrics['compile_ok_compact']).lower()}</span>
                <span>parity={str(metrics['full_compact_match']).lower()}</span>
                <span>headline={str(metrics['headline_present']).lower()}</span>
                <span>subtitle={str(metrics['subtitle_present']).lower()}</span>
                <span>terminal={str(metrics['terminal_title_present']).lower()}</span>
                <span>titles={metrics['title_matches']}/{metrics['title_total']}</span>
                <span>rects={metrics['rects']}</span>
                <span>paths={metrics['paths']}</span>
              </div>
              <div class="paths">
                <a href="{html.escape(_svg_rel(Path(row['scene_path']), OUT_DIR))}">scene DSL</a>
                <a href="{html.escape(_svg_rel(Path(row['compact_scene_path']), OUT_DIR))}">compact DSL</a>
                <a href="{html.escape(_svg_rel(Path(row['content_path']), OUT_DIR))}">content JSON</a>
                <a href="{html.escape(compact_svg_rel)}">compiled SVG</a>
                <span>asset target: {html.escape(row['source_asset'])}</span>
              </div>
              <div class="frame">
                <img src="{html.escape(compact_svg_rel)}" alt="{html.escape(row['case_id'])}"/>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Spec15b Compiler Smoke Report</title>
  <style>
    :root {{
      --bg: #0d1823;
      --ink: #eff5fb;
      --muted: #9fb8cf;
      --card: rgba(16, 27, 39, 0.9);
      --line: rgba(110, 168, 212, 0.14);
      --accent: #7cd0ff;
      --ok: #72e0aa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(86,174,255,0.16), transparent 28%),
        linear-gradient(180deg, #07131f 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{ max-width: 1380px; margin: 0 auto; padding: 48px 24px 64px; }}
    header {{ margin-bottom: 28px; }}
    h1 {{ margin: 0 0 10px; font-size: 42px; line-height: 1.05; }}
    .lede {{ max-width: 920px; color: var(--muted); font-size: 17px; line-height: 1.6; }}
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
      box-shadow: 0 16px 40px rgba(4, 10, 16, 0.28);
    }}
    .summary .box {{ padding: 18px 20px; }}
    .summary .k {{ font: 700 28px/1.1 "IBM Plex Sans", "Segoe UI", sans-serif; }}
    .summary .v {{ color: var(--muted); margin-top: 6px; font: 500 13px/1.4 "IBM Plex Sans", "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; }}
    .card {{ padding: 20px; }}
    .meta {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
    .meta h2 {{ margin: 0 0 6px; font-size: 27px; }}
    .sub {{ margin: 0; color: var(--muted); font-size: 14px; line-height: 1.5; }}
    .pill {{
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(86,174,255,0.12);
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
      background: rgba(255,255,255,0.06);
      color: var(--ink);
      text-decoration: none;
    }}
    .checks .ok {{ background: rgba(114,224,170,0.14); color: var(--ok); }}
    .frame {{
      margin-top: 18px;
      padding: 12px;
      border-radius: 18px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .frame img {{ width: 100%; height: auto; display: block; border-radius: 12px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Spec15b System Diagram Compiler Smoke</h1>
      <p class="lede">Compiler-first validation for the first strict system-diagram gold pack. Each case renders from both the full scene DSL and the compact scene DSL, and the report checks that those outputs stay identical.</p>
    </header>
    <section class="summary">
      <div class="box"><div class="k">{len(rows)}</div><div class="v">Gold Cases</div></div>
      <div class="box"><div class="k">{sum(1 for row in rows if row['metrics']['compile_ok_compact'])}/{len(rows)}</div><div class="v">Compact Compile OK</div></div>
      <div class="box"><div class="k">{sum(1 for row in rows if row['metrics']['full_compact_match'])}/{len(rows)}</div><div class="v">Full/Compact Parity</div></div>
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
    metrics = [row["metrics"] for row in rows]
    report = {
        "schema": "ck.spec15b_compiler_smoke.v1",
        "generated_at": "2026-03-29",
        "family": "system_diagram",
        "totals": {
            "cases": len(rows),
            "compact_compile_ok": sum(1 for m in metrics if m["compile_ok_compact"]),
            "full_compile_ok": sum(1 for m in metrics if m["compile_ok_full"]),
            "full_compact_match": sum(1 for m in metrics if m["full_compact_match"]),
            "headline_present": sum(1 for m in metrics if m["headline_present"]),
            "subtitle_present": sum(1 for m in metrics if m["subtitle_present"]),
            "terminal_present": sum(1 for m in metrics if m["terminal_title_present"]),
        },
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "compiler_smoke_report.json"
    html_path = OUT_DIR / "compiler_smoke_report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(_html(rows), encoding="utf-8")
    print(f"[ok] wrote {json_path}")
    print(f"[ok] wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
