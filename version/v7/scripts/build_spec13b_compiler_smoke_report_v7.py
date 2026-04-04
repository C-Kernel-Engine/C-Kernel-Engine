#!/usr/bin/env python3
"""Build a compiler smoke report for the current spec13b graph-family gold scenes."""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec13b_v7 import render_structured_scene_spec13b_svg


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "version" / "v7" / "reports"
SPEC12_GOLD = REPORTS / "spec12_gold_mappings"
SPEC13B_GOLD = REPORTS / "spec13b_gold_mappings"
OUT_DIR = REPORTS / "spec13b_smoke" / "gold_compiled"


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
            case_id="failure_decision_tree",
            scene_path=SPEC12_GOLD / "failure-decision-tree.scene.compact.dsl",
            content_path=SPEC12_GOLD / "failure-decision-tree.content.json",
            source_asset="ir-v66-failure-decision-tree.svg",
            rationale="Backward-compatible legacy decision tree adapter path.",
        ),
        SmokeCase(
            case_id="pipeline_overview",
            scene_path=SPEC13B_GOLD / "pipeline-overview-flow.scene.compact.dsl",
            content_path=SPEC13B_GOLD / "pipeline-overview-flow.content.json",
            source_asset="pipeline-overview.svg",
            rationale="Minimal end-to-end graph family for prompt -> router -> planner -> compiler -> SVG.",
        ),
        SmokeCase(
            case_id="templates_to_ir",
            scene_path=SPEC13B_GOLD / "templates-to-ir.scene.compact.dsl",
            content_path=SPEC13B_GOLD / "templates-to-ir.content.json",
            source_asset="ir-templates-to-ir.svg",
            rationale="Template-to-IR bridge case to grow graph-family semantic coverage without changing renderer shape.",
        ),
        SmokeCase(
            case_id="ir_lowering_pipeline",
            scene_path=SPEC13B_GOLD / "ir-lowering-pipeline.scene.compact.dsl",
            content_path=SPEC13B_GOLD / "ir-lowering-pipeline.content.json",
            source_asset="ir-lowering-pipeline.svg",
            rationale="Lowering and memory-planning path that is close to pipeline-overview but semantically richer.",
        ),
        SmokeCase(
            case_id="kernel_registry_flow",
            scene_path=SPEC13B_GOLD / "kernel-registry-flow.scene.compact.dsl",
            content_path=SPEC13B_GOLD / "kernel-registry-flow.content.json",
            source_asset="kernel-registry-flow.svg",
            rationale="Compiler/runtime graph with strongly labeled deterministic stages.",
        ),
        SmokeCase(
            case_id="dataflow_stitching",
            scene_path=SPEC13B_GOLD / "ir-dataflow-stitching.scene.compact.dsl",
            content_path=SPEC13B_GOLD / "ir-dataflow-stitching.content.json",
            source_asset="ir-dataflow-stitching.svg",
            rationale="Technical dataflow case to teach explicit op-id and memory-edge stitching in the graph family.",
        ),
        SmokeCase(
            case_id="qwen_layer_dataflow",
            scene_path=SPEC13B_GOLD / "qwen-layer-dataflow.scene.compact.dsl",
            content_path=SPEC13B_GOLD / "qwen-layer-dataflow.content.json",
            source_asset="qwen_layer_dataflow.svg",
            rationale="Architecture/dataflow case to test whether graph-family IR can cover model-path diagrams.",
        ),
        SmokeCase(
            case_id="ir_artifact_lineage",
            scene_path=SPEC13B_GOLD / "ir-artifact-lineage.scene.compact.dsl",
            content_path=SPEC13B_GOLD / "ir-artifact-lineage.content.json",
            source_asset="ir-v66-artifact-lineage.svg",
            rationale="Paper-editorial graph family to test non-dark theme coverage and lineage diagrams.",
        ),
    ]


def _svg_rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), start=base.resolve())).as_posix()


def _text_present(svg: str, value: str) -> bool:
    return bool(value) and value in svg


def _node_titles(content: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for section_name in ("nodes", "outcomes"):
        for payload in (content.get(section_name) or {}).values():
            if isinstance(payload, dict):
                title = str(payload.get("title") or "").strip()
                if title:
                    titles.append(title)
    for section_name in ("svg", "artifact", "runtime", "residual", "finish"):
        payload = content.get(section_name)
        if isinstance(payload, dict):
            title = str(payload.get("title") or "").strip()
            if title:
                titles.append(title)
    return titles


def _metrics(svg: str, content: dict[str, Any]) -> dict[str, Any]:
    headline = str(((content.get("header") or {}).get("headline")) or "").strip()
    subtitle = str(((content.get("header") or {}).get("subtitle")) or "").strip()
    titles = _node_titles(content)
    matched_titles = [title for title in titles if _text_present(svg, title)]
    connectors = svg.count('marker-end="url(#arrowHead)"')
    boxes = len(re.findall(r"<rect ", svg))
    return {
        "headline_present": _text_present(svg, headline),
        "subtitle_present": _text_present(svg, subtitle),
        "title_matches": len(matched_titles),
        "title_total": len(titles),
        "matched_titles": matched_titles,
        "connectors": connectors,
        "rects": boxes,
        "compile_ok": svg.startswith("<svg") and svg.endswith("</svg>"),
    }


def _build_rows() -> list[dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in _cases():
        scene_text = case.scene_path.read_text(encoding="utf-8").strip()
        content = json.loads(case.content_path.read_text(encoding="utf-8"))
        svg = render_structured_scene_spec13b_svg(scene_text, content=content)
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
                "layout": "decision_tree" if "decision_tree" in scene_text else "flow_graph",
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
                <div class="pill">{html.escape(row['layout'])}</div>
              </div>
              <div class="checks">
                <span class="ok">compile_ok={str(metrics['compile_ok']).lower()}</span>
                <span>headline={str(metrics['headline_present']).lower()}</span>
                <span>subtitle={str(metrics['subtitle_present']).lower()}</span>
                <span>titles={metrics['title_matches']}/{metrics['title_total']}</span>
                <span>connectors={metrics['connectors']}</span>
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
  <title>Spec13b Compiler Smoke Report</title>
  <style>
    :root {{
      --bg: #f4f0e6;
      --ink: #16222e;
      --muted: #52616f;
      --card: rgba(255,255,255,0.88);
      --line: rgba(22,34,46,0.12);
      --accent: #0c6a83;
      --ok: #1f6d42;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      background:
        radial-gradient(circle at top left, rgba(12,106,131,0.12), transparent 28%),
        linear-gradient(180deg, #f7f3eb 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 48px 24px 64px; }}
    header {{ margin-bottom: 28px; }}
    h1 {{ margin: 0 0 10px; font-size: 44px; line-height: 1.05; }}
    .lede {{ max-width: 900px; color: var(--muted); font-size: 18px; line-height: 1.6; }}
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
      background: rgba(22,34,46,0.06);
      color: var(--ink);
      text-decoration: none;
    }}
    .checks .ok {{
      background: rgba(31,109,66,0.12);
      color: var(--ok);
    }}
    .frame {{
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      background: #fff;
    }}
    .frame img {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Spec13b Compiler Smoke Report</h1>
      <p class="lede">
        This report answers the boundary question before training: if the current graph-family scene outputs are passed directly into the
        new spec13b compiler, do we get coherent SVG targets that are good enough to train against? These are compiled gold scenes, not
        model generations.
      </p>
    </header>
    <section class="summary">
      <div class="box"><div class="k">{len(rows)}</div><div class="v">Gold Scenes Compiled</div></div>
      <div class="box"><div class="k">{sum(1 for row in rows if row['metrics']['compile_ok'])}</div><div class="v">Compile-OK</div></div>
      <div class="box"><div class="k">{sum(1 for row in rows if row['metrics']['headline_present'])}/{len(rows)}</div><div class="v">Headline Checks</div></div>
      <div class="box"><div class="k">{sum(row['metrics']['title_matches'] for row in rows)}/{sum(row['metrics']['title_total'] for row in rows)}</div><div class="v">Node Title Matches</div></div>
      <div class="box"><div class="k">{sum(row['metrics']['connectors'] for row in rows)}</div><div class="v">Rendered Connectors</div></div>
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
    manifest_path = OUT_DIR / "compiled_manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    html_path = OUT_DIR / "spec13b_compiler_smoke_report.html"
    html_path.write_text(_html(rows), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "html": str(html_path), "count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
