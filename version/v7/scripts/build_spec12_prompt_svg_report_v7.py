#!/usr/bin/env python3
"""Build an HTML report for prompt -> generated DSL -> compiled SVG cases."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.1f}%"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _metric_chip(label: str, value: Any, *, good: bool) -> str:
    state = "pass" if good else "fail"
    return f'<span class="chip {state}">{html.escape(label)}: <strong>{html.escape(_yes_no(value))}</strong></span>'


def _svg_markup(svg_text: str | None, fallback: str) -> str:
    payload = str(svg_text or "").strip()
    if not payload:
        return f'<div class="empty">{html.escape(fallback)}</div>'
    return payload


def _code_block(text: str | None) -> str:
    return f"<pre>{html.escape(str(text or '').strip() or '—')}</pre>"


def _section_from_report(report: dict[str, Any], heading: str) -> str:
    totals = dict(report.get("totals") or {})
    results = list(report.get("results") or [])
    split_summary = list(report.get("split_summary") or [])

    summary_cards = [
        f'<article class="metric"><div class="k">Cases</div><div class="v">{int(totals.get("count", 0))}</div></article>',
        f'<article class="metric"><div class="k">Exact</div><div class="v">{html.escape(_pct(totals.get("exact_rate")))}</div></article>',
        f'<article class="metric"><div class="k">Renderable</div><div class="v">{html.escape(_pct(totals.get("renderable_rate")))}</div></article>',
        f'<article class="metric"><div class="k">Materialized</div><div class="v">{html.escape(_pct(totals.get("materialized_exact_rate")))}</div></article>',
    ]

    split_rows: list[str] = []
    for row in split_summary:
        split_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('split') or ''))}</td>"
            f"<td>{int(row.get('count', 0))}</td>"
            f"<td>{html.escape(_pct(row.get('exact_rate')))}</td>"
            f"<td>{html.escape(_pct(row.get('renderable_rate')))}</td>"
            f"<td>{html.escape(_pct(row.get('materialized_exact_rate')))}</td>"
            "</tr>"
        )

    cards: list[str] = []
    current_split = ""
    for row in results:
        split = str(row.get("split") or "").strip()
        if split != current_split:
            current_split = split
            cards.append(f'<h3 class="split-heading">{html.escape(split)}</h3>')

        generated_dsl = str(row.get("parsed_output") or row.get("response_text") or "").strip()
        rendered_svg = str(row.get("rendered_svg") or row.get("materialized_output") or "").strip()
        expected_dsl = str(row.get("expected_output") or "").strip()
        expected_svg = str(row.get("expected_rendered_output") or "").strip()

        cards.append(
            "\n".join(
                [
                    '<section class="case">',
                    f"<h4>{html.escape(str(row.get('label') or row.get('id') or 'case'))}</h4>",
                    '<div class="chips">',
                    _metric_chip("Exact", row.get("exact_match"), good=bool(row.get("exact_match"))),
                    _metric_chip("Renderable", row.get("renderable"), good=bool(row.get("renderable"))),
                    _metric_chip("Materialized", row.get("materialized_exact_match"), good=bool(row.get("materialized_exact_match"))),
                    "</div>",
                    '<div class="grid two">',
                    '<div class="panel">',
                    "<div class=\"panel-title\">Prompt</div>",
                    _code_block(str(row.get("prompt") or "")),
                    "</div>",
                    '<div class="panel">',
                    "<div class=\"panel-title\">Generated Scene DSL</div>",
                    _code_block(generated_dsl),
                    "</div>",
                    "</div>",
                    '<div class="grid two">',
                    '<div class="panel">',
                    "<div class=\"panel-title\">Compiled SVG From Model Output</div>",
                    f'<div class="svg-frame">{_svg_markup(rendered_svg, "No compiled SVG")}</div>',
                    "</div>",
                    '<div class="panel">',
                    "<div class=\"panel-title\">Expected SVG</div>",
                    f'<div class="svg-frame">{_svg_markup(expected_svg, "No expected SVG")}</div>',
                    "</div>",
                    "</div>",
                    "<details>",
                    "<summary>Reference Text</summary>",
                    '<div class="grid two">',
                    '<div class="panel">',
                    "<div class=\"panel-title\">Expected Scene DSL</div>",
                    _code_block(expected_dsl),
                    "</div>",
                    '<div class="panel">',
                    "<div class=\"panel-title\">Raw Response Text</div>",
                    _code_block(str(row.get("response_text") or "")),
                    "</div>",
                    "</div>",
                    "</details>",
                    "</section>",
                ]
            )
        )

    return "\n".join(
        [
            f"<section class=\"report-block\">",
            f"<h2>{html.escape(heading)}</h2>",
            '<div class="summary-grid">',
            *summary_cards,
            "</div>",
            '<div class="table-wrap"><table><thead><tr><th>Split</th><th>Count</th><th>Exact</th><th>Renderable</th><th>Materialized</th></tr></thead>',
            f"<tbody>{''.join(split_rows)}</tbody></table></div>",
            *cards,
            "</section>",
        ]
    )


def build_html(title: str, sections: list[tuple[str, dict[str, Any]]]) -> str:
    body = "\n".join(_section_from_report(report, heading) for heading, report in sections)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8"/>',
            f"  <title>{html.escape(title)}</title>",
            "  <style>",
            "    :root { --bg:#f4efe5; --card:#fffdf8; --ink:#17202a; --muted:#52606d; --line:#d8cfbe; --pass:#166534; --fail:#b42318; }",
            "    body { margin:0; padding:28px; background:linear-gradient(180deg,#f8f4ec 0%,#efe6d6 100%); color:var(--ink); font-family:'IBM Plex Sans','Segoe UI',sans-serif; }",
            "    main { max-width: 1400px; margin: 0 auto; }",
            "    h1,h2,h3,h4 { margin: 0 0 12px; }",
            "    .lede { color: var(--muted); margin: 10px 0 24px; }",
            "    .report-block { margin-bottom: 36px; }",
            "    .summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin: 16px 0 20px; }",
            "    .metric,.case,.panel,.table-wrap { background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:0 8px 20px rgba(23,32,42,0.06); }",
            "    .metric { padding:14px 16px; }",
            "    .metric .k,.panel-title { color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; font-size:12px; font-weight:700; }",
            "    .metric .v { margin-top:6px; font-size:28px; font-weight:700; }",
            "    .table-wrap { padding: 14px; margin-bottom: 18px; overflow-x:auto; }",
            "    table { width:100%; border-collapse:collapse; }",
            "    th,td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); }",
            "    .split-heading { margin: 18px 0 10px; color: var(--muted); text-transform: capitalize; }",
            "    .case { padding:16px; margin-bottom:18px; }",
            "    .chips { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }",
            "    .chip { display:inline-flex; gap:6px; align-items:center; padding:6px 10px; border-radius:999px; font-size:12px; font-weight:700; }",
            "    .chip.pass { background:#dcfce7; color:var(--pass); }",
            "    .chip.fail { background:#fee2e2; color:var(--fail); }",
            "    .grid.two { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-bottom:14px; }",
            "    .panel { padding:12px; }",
            "    pre { margin:10px 0 0; padding:12px; background:#f7f2e8; border:1px solid #e7dcc9; border-radius:12px; white-space:pre-wrap; word-break:break-word; overflow-x:auto; }",
            "    .svg-frame { margin-top:10px; min-height:280px; border:1px solid #e7dcc9; border-radius:12px; background:#fff; padding:8px; display:flex; align-items:center; justify-content:center; overflow:auto; }",
            "    .empty { color:var(--muted); font-size:13px; }",
            "    details { margin-top: 6px; }",
            "    summary { cursor:pointer; color:var(--muted); font-weight:700; }",
            "    @media (max-width: 980px) { .grid.two { grid-template-columns:1fr; } .svg-frame { min-height:220px; } }",
            "  </style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{html.escape(title)}</h1>",
            "<p class=\"lede\">Prompt by prompt: what the model was asked, the scene DSL it generated, and the SVG infographic produced after the compiler.</p>",
            body,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visible-probe", required=True, type=Path, help="Visible probe report JSON")
    ap.add_argument("--hidden-probe", type=Path, default=None, help="Optional hidden probe report JSON")
    ap.add_argument("--output-html", required=True, type=Path, help="Destination HTML")
    ap.add_argument("--title", default="Spec12 Prompt to SVG Report", help="Report title")
    args = ap.parse_args()

    sections: list[tuple[str, dict[str, Any]]] = [("Visible Probe Cases", _load_json(args.visible_probe.expanduser().resolve()))]
    if args.hidden_probe:
        sections.append(("Hidden Prompt-Surface Cases", _load_json(args.hidden_probe.expanduser().resolve())))

    out_path = args.output_html.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(str(args.title), sections), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
