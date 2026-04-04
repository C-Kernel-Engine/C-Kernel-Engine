#!/usr/bin/env python3
"""Build the first spec12 compiler parity report for compact gold mappings."""

from __future__ import annotations

import argparse
import html
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec12_v7 import render_structured_scene_spec12_svg


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "version" / "v7" / "reports" / "spec12_gold_mappings" / "spec12_gold_mappings_compact_20260318.json"
DEFAULT_HTML = ROOT / "version" / "v7" / "reports" / "spec12_alignment_report_20260318.html"
DEFAULT_JSON = ROOT / "version" / "v7" / "reports" / "spec12_alignment_report_20260318.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=False)


def _svg_tags(svg_text: str) -> dict[str, int]:
    root = ET.fromstring(svg_text)
    counts: Counter[str] = Counter()
    for elem in root.iter():
        tag = str(elem.tag).split("}", 1)[-1]
        counts[tag] += 1
    return dict(sorted(counts.items()))


def _preview(svg_text: str) -> str:
    return f'<div class="svg-wrap">{svg_text}</div>'


def _missing_tags(compiled: dict[str, int], real: dict[str, int]) -> list[str]:
    return sorted(tag for tag in real if tag not in compiled and tag not in {"style", "svg", "defs"})


def _extra_tags(compiled: dict[str, int], real: dict[str, int]) -> list[str]:
    return sorted(tag for tag in compiled if tag not in real and tag not in {"svg", "defs"})


def _tag_ratio(compiled: dict[str, int], real: dict[str, int], tag: str) -> float | None:
    real_count = int(real.get(tag) or 0)
    if real_count <= 0:
        return None
    return float(compiled.get(tag, 0)) / float(real_count)


def _gap_read(row: dict[str, Any]) -> dict[str, Any]:
    family = str(row.get("family") or "")
    compiled = row["compiled_tags"]
    real = row["real_tags"]
    missing = _missing_tags(compiled, real)
    extra = _extra_tags(compiled, real)
    rect_ratio = _tag_ratio(compiled, real, "rect")
    text_ratio = _tag_ratio(compiled, real, "text")

    if family == "table_matrix":
        return {
            "status": "directional scaffold",
            "what_is_right": "Header, legend, grouped table blocks, and row-state emphasis are present. The compact DSL is expressive enough for the family.",
            "main_gap": "The asset is much denser than the current compiler output. Group nesting, cell grid detail, divider lines, and repeated table cells are still too shallow.",
            "next_compiler_move": "Add richer table internals: explicit column dividers, denser row rendering, grouped header bands, and secondary note strips.",
            "missing_tags": missing,
            "extra_tags": extra,
            "rect_ratio": rect_ratio,
            "text_ratio": text_ratio,
        }
    if family == "decision_tree":
        return {
            "status": "close on structure",
            "what_is_right": "Topology, branching, marker usage, and outcome-panel structure are already close to the shipped asset. This family is the strongest first-pass parity case.",
            "main_gap": "Edge routing and box geometry are still more generic than the real asset, and the visual hierarchy is heavier than necessary.",
            "next_compiler_move": "Tighten connector routing, reduce card chrome, and make entry/outcome styling more asset-specific.",
            "missing_tags": missing,
            "extra_tags": extra,
            "rect_ratio": rect_ratio,
            "text_ratio": text_ratio,
        }
    if family == "memory_map":
        return {
            "status": "good family fit, missing detail language",
            "what_is_right": "The overall left-tower plus right-info-card composition is correct, and the segment/content split maps cleanly to the family.",
            "main_gap": "The real asset uses more grouping and small callout markers. Circles, grouped rails, and more exact right-column treatments are still missing.",
            "next_compiler_move": "Add circle markers, grouped overlay rails, and stronger code/info-card styling so the memory map reads less like generic panels.",
            "missing_tags": missing,
            "extra_tags": extra,
            "rect_ratio": rect_ratio,
            "text_ratio": text_ratio,
        }
    return {
        "status": "unclassified",
        "what_is_right": "",
        "main_gap": "",
        "next_compiler_move": "",
        "missing_tags": missing,
        "extra_tags": extra,
        "rect_ratio": rect_ratio,
        "text_ratio": text_ratio,
    }


def build_cases(manifest: Path) -> list[dict[str, Any]]:
    doc = _load_json(manifest)
    cases: list[dict[str, Any]] = []
    for row in doc.get("mappings") or []:
        if not isinstance(row, dict):
            continue
        asset_path = ROOT / str(row.get("asset") or "")
        scene_path = ROOT / str(row.get("scene_dsl") or "")
        content_path = ROOT / str(row.get("content_json") or "")
        if not asset_path.exists() or not scene_path.exists() or not content_path.exists():
            raise SystemExit(f"Missing mapping artifact for {row}")
        scene_text = scene_path.read_text(encoding="utf-8")
        content = _load_json(content_path)
        compiled = render_structured_scene_spec12_svg(scene_text, content=content)
        real_svg = asset_path.read_text(encoding="utf-8")
        cases.append(
            {
                "asset": str(row.get("asset") or ""),
                "family": str(row.get("family") or ""),
                "scene_dsl": str(row.get("scene_dsl") or ""),
                "content_json": str(row.get("content_json") or ""),
                "scene_text": scene_text,
                "content": content,
                "compiled_svg": compiled,
                "real_svg": real_svg,
                "compiled_tags": _svg_tags(compiled),
                "real_tags": _svg_tags(real_svg),
                "gap_read": _gap_read(
                    {
                        "family": str(row.get("family") or ""),
                        "compiled_tags": _svg_tags(compiled),
                        "real_tags": _svg_tags(real_svg),
                    }
                ),
            }
        )
    return cases


def _render_html(cases: list[dict[str, Any]]) -> str:
    summary_rows: list[str] = []
    for row in cases:
        gap = row["gap_read"]
        summary_rows.append(
            f"""
            <tr>
              <td>{_escape(Path(row['asset']).name)}</td>
              <td>{_escape(row['family'])}</td>
              <td>{_escape(gap['status'])}</td>
              <td>{_escape(', '.join(gap['missing_tags']) or 'none')}</td>
              <td>{_escape(gap['next_compiler_move'])}</td>
            </tr>
            """
        )
    sections: list[str] = []
    for row in cases:
        gap = row["gap_read"]
        sections.append(
            f"""
            <section class="case">
              <div class="case-top">
                <div>
                  <div class="eyebrow">{_escape(row['family'])}</div>
                  <h2>{_escape(Path(row['asset']).name)}</h2>
                </div>
              </div>
              <div class="preview-grid">
                <div class="panel">
                  <div class="label">Compiled Spec12 Output</div>
                  {_preview(row['compiled_svg'])}
                </div>
                <div class="panel">
                  <div class="label">Real Asset</div>
                  {_preview(row['real_svg'])}
                </div>
              </div>
              <div class="text-grid">
                <div class="panel">
                  <div class="label">Compiler Read</div>
                  <pre>{_escape(json.dumps(gap, indent=2))}</pre>
                </div>
                <div class="panel">
                  <div class="label">Parity Guidance</div>
                  <div class="note-card">
                    <p><strong>Status:</strong> {_escape(gap['status'])}</p>
                    <p><strong>What is right:</strong> {_escape(gap['what_is_right'])}</p>
                    <p><strong>Main gap:</strong> {_escape(gap['main_gap'])}</p>
                    <p><strong>Next compiler move:</strong> {_escape(gap['next_compiler_move'])}</p>
                  </div>
                </div>
              </div>
              <div class="text-grid">
                <div class="panel">
                  <div class="label">Compact scene.dsl</div>
                  <pre>{_escape(row['scene_text'])}</pre>
                </div>
                <div class="panel">
                  <div class="label">content.json</div>
                  <pre>{_escape(json.dumps(row['content'], indent=2))}</pre>
                </div>
              </div>
              <div class="text-grid">
                <div class="panel">
                  <div class="label">Compiled SVG Tags</div>
                  <pre>{_escape(json.dumps(row['compiled_tags'], indent=2))}</pre>
                </div>
                <div class="panel">
                  <div class="label">Real Asset Tags</div>
                  <pre>{_escape(json.dumps(row['real_tags'], indent=2))}</pre>
                </div>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spec12 Alignment Report</title>
  <style>
    :root {{
      --bg: #0b0d12; --panel: rgba(255,255,255,0.05); --border: rgba(255,255,255,0.10);
      --text: #eef2f7; --muted: #98a2b3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; color: var(--text); font-family: "Space Grotesk", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(57,217,138,0.10), transparent 24%),
        linear-gradient(180deg, #11141b 0%, #0b0d12 100%);
    }}
    .page {{ width: min(1480px, calc(100vw - 40px)); margin: 22px auto 40px; }}
    .hero, .panel, .case {{
      border: 1px solid var(--border); border-radius: 22px; background: var(--panel);
      box-shadow: 0 24px 60px rgba(0,0,0,0.28); backdrop-filter: blur(10px);
    }}
    .hero {{ padding: 28px 30px; margin-bottom: 20px; }}
    .case {{ padding: 20px; margin-top: 20px; }}
    .eyebrow {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: rgba(122,162,255,0.16); color: #bfd1ff; text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; font-weight: 700; }}
    h1 {{ margin: 10px 0 8px; font-size: 38px; line-height: 1.05; }}
    h2 {{ margin: 10px 0 8px; font-size: 24px; }}
    p, .meta {{ color: var(--muted); line-height: 1.6; }}
    .preview-grid, .text-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }}
    .panel {{ padding: 14px; }}
    .label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 12px; font-weight: 700; margin-bottom: 10px; }}
    .note-card {{ border-radius: 12px; padding: 12px 14px; background: #0b1220; border: 1px solid rgba(255,255,255,0.08); min-height: 120px; }}
    .note-card p {{ margin: 0 0 10px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
    th, td {{ text-align: left; vertical-align: top; padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
    th {{ color: #bfd1ff; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; }}
    .svg-wrap {{ background: #f8fafc; border-radius: 12px; padding: 12px; min-height: 220px; overflow: auto; }}
    .svg-wrap svg {{ width: 100%; height: auto; display: block; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; border-radius: 12px; padding: 12px; background: #0b1220; color: #d8e4ff; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.55; border: 1px solid rgba(255,255,255,0.08); min-height: 120px; }}
    @media (max-width: 980px) {{
      .preview-grid, .text-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <span class="eyebrow">Spec12 Compiler Parity</span>
      <h1>First Alignment Pass</h1>
      <p>This report compares the first compact spec12 gold mappings against the real shipped assets. The goal is compiler parity before tokenizer and training decisions.</p>
      <div class="meta">Families covered: table_matrix, decision_tree, memory_map</div>
      <table>
        <thead>
          <tr>
            <th>Asset</th>
            <th>Family</th>
            <th>Status</th>
            <th>Missing Tags</th>
            <th>Next Compiler Move</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    cases = build_cases(args.manifest)
    report = {
        "schema": "ck.spec12_alignment_report.v1",
        "manifest": str(args.manifest),
        "count": len(cases),
        "cases": [
            {
                "asset": row["asset"],
                "family": row["family"],
                "scene_dsl": row["scene_dsl"],
                "content_json": row["content_json"],
                "compiled_tags": row["compiled_tags"],
                "real_tags": row["real_tags"],
                "gap_read": row["gap_read"],
            }
            for row in cases
        ],
    }
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.html_out.write_text(_render_html(cases), encoding="utf-8")
    print(args.json_out)
    print(args.html_out)


if __name__ == "__main__":
    main()
