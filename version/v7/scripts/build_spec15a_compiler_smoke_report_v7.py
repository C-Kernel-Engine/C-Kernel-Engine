#!/usr/bin/env python3
"""Build a compiler smoke report for the current spec15a memory-map gold scenes."""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec15a_v7 import render_structured_scene_spec15a_svg


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "version" / "v7" / "reports"
SPEC15A_GOLD = REPORTS / "spec15a_gold_mappings"
OUT_DIR = REPORTS / "spec15a_smoke" / "gold_compiled"


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
            case_id="memory_layout_map",
            scene_path=SPEC15A_GOLD / "memory-layout-map.scene.dsl",
            compact_scene_path=SPEC15A_GOLD / "memory-layout-map.scene.compact.dsl",
            content_path=SPEC15A_GOLD / "memory-layout-map.content.json",
            source_asset="memory-layout-map.svg",
            rationale="Base fixed-offset memory tower with grouped mid-layers and three side cards.",
        ),
        SmokeCase(
            case_id="bump_allocator_quant",
            scene_path=SPEC15A_GOLD / "bump_allocator_quant.scene.dsl",
            compact_scene_path=SPEC15A_GOLD / "bump_allocator_quant.scene.compact.dsl",
            content_path=SPEC15A_GOLD / "bump_allocator_quant.content.json",
            source_asset="bump_allocator_quant.svg",
            rationale="Bounded bump-allocator arena with ordered typed regions, address markers, and operator cards.",
        ),
        SmokeCase(
            case_id="v7_train_memory_canary",
            scene_path=SPEC15A_GOLD / "v7-train-memory-canary.scene.dsl",
            compact_scene_path=SPEC15A_GOLD / "v7-train-memory-canary.scene.compact.dsl",
            content_path=SPEC15A_GOLD / "v7-train-memory-canary.content.json",
            source_asset="v7-train-memory-canary.svg",
            rationale="Operator-facing contiguous training arena with six ordered regions and diagnostic cards.",
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


def _segment_titles(content: dict[str, Any]) -> list[str]:
    out: list[str] = []
    segments = content.get("segments") if isinstance(content.get("segments"), dict) else {}
    for payload in segments.values():
        if isinstance(payload, dict):
            title = str(payload.get("title") or "").strip()
            if title:
                out.append(title)
    return out


def _card_titles(content: dict[str, Any]) -> list[str]:
    out: list[str] = []
    cards = content.get("cards") if isinstance(content.get("cards"), dict) else {}
    for payload in cards.values():
        if isinstance(payload, dict):
            title = str(payload.get("title") or "").strip()
            if title:
                out.append(title)
    return out


def _metrics(full_svg: str, compact_svg: str, content: dict[str, Any]) -> dict[str, Any]:
    headline = str(((content.get("header") or {}).get("headline")) or "").strip()
    subtitle = str(((content.get("header") or {}).get("subtitle")) or "").strip()
    titles = _segment_titles(content) + _card_titles(content)
    matched_titles = [title for title in titles if _text_present(compact_svg, title)]
    return {
        "compile_ok_full": full_svg.startswith("<svg") and full_svg.endswith("</svg>"),
        "compile_ok_compact": compact_svg.startswith("<svg") and compact_svg.endswith("</svg>"),
        "full_compact_match": full_svg == compact_svg,
        "headline_present": _text_present(compact_svg, headline),
        "subtitle_present": _text_present(compact_svg, subtitle),
        "title_matches": len(matched_titles),
        "title_total": len(titles),
        "rects": compact_svg.count("<rect "),
    }


def _build_rows() -> list[dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in _cases():
        full_scene = case.scene_path.read_text(encoding="utf-8").strip()
        compact_scene = case.compact_scene_path.read_text(encoding="utf-8").strip()
        content = json.loads(case.content_path.read_text(encoding="utf-8"))
        full_svg = render_structured_scene_spec15a_svg(full_scene, content=content)
        compact_svg = render_structured_scene_spec15a_svg(compact_scene, content=content)
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
                <div class="pill">memory_map</div>
              </div>
              <div class="checks">
                <span class="ok">full={str(metrics['compile_ok_full']).lower()}</span>
                <span class="ok">compact={str(metrics['compile_ok_compact']).lower()}</span>
                <span>parity={str(metrics['full_compact_match']).lower()}</span>
                <span>headline={str(metrics['headline_present']).lower()}</span>
                <span>subtitle={str(metrics['subtitle_present']).lower()}</span>
                <span>titles={metrics['title_matches']}/{metrics['title_total']}</span>
                <span>rects={metrics['rects']}</span>
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
  <title>Spec15a Compiler Smoke Report</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --ink: #16222e;
      --muted: #5b6873;
      --card: rgba(255,255,255,0.9);
      --line: rgba(22,34,46,0.12);
      --accent: #0c6a83;
      --ok: #1f6d42;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      background:
        radial-gradient(circle at top left, rgba(12,106,131,0.12), transparent 30%),
        linear-gradient(180deg, #f9f6ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 48px 24px 64px; }}
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
    }}
    .summary .box {{ padding: 18px 20px; }}
    .summary .k {{ font: 700 28px/1.1 "IBM Plex Sans", "Segoe UI", sans-serif; }}
    .summary .v {{ color: var(--muted); margin-top: 6px; font: 500 13px/1.4 "IBM Plex Sans", "Segoe UI", sans-serif; text-transform: uppercase; letter-spacing: 0.06em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
    .card {{ padding: 20px; }}
    .meta {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
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
      padding: 12px;
      border-radius: 18px;
      background: rgba(16, 24, 32, 0.06);
      border: 1px solid rgba(16, 24, 32, 0.08);
    }}
    img {{ width: 100%; height: auto; display: block; border-radius: 14px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Spec15a Compiler Smoke Report</h1>
      <p class="lede">Compiler-first smoke pass for the strict domain-agnostic <code>memory_map</code> family. This pass checks that the new spec15a adapter compiles both full and compact scene forms into the same SVG for the currently accepted gold pack.</p>
    </header>
    <section class="summary">
      <div class="box"><div class="k">{len(rows)}</div><div class="v">Smoke Cases</div></div>
      <div class="box"><div class="k">{sum(1 for row in rows if row['metrics']['full_compact_match'])}</div><div class="v">Full/Compact Parity</div></div>
      <div class="box"><div class="k">{sum(1 for row in rows if row['metrics']['compile_ok_full'] and row['metrics']['compile_ok_compact'])}</div><div class="v">Compile Success</div></div>
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
    report_dir = REPORTS / "spec15a_smoke"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "compiler_smoke_report.json"
    html_path = report_dir / "compiler_smoke_report.html"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    html_path.write_text(_html(rows), encoding="utf-8")
    print(f"[OK] wrote: {json_path}")
    print(f"[OK] wrote: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
