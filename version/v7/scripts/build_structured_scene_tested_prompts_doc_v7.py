#!/usr/bin/env python3
"""Render a prompt-by-prompt report from a structured-scene probe report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.1f}%"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _code_block(text: str) -> str:
    return f"```text\n{text.strip()}\n```"


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    title = str(report.get("title") or "Structured Scene Tested Prompts Report").strip()
    run_name = str(report.get("run_name") or "").strip()
    lines.append(f"# {title}")
    if run_name:
        lines.append("")
        lines.append(f"Run: `{run_name}`")

    totals = dict(report.get("totals") or {})
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Cases: `{totals.get('count', 0)}`")
    lines.append(f"- Exact: `{_pct(totals.get('exact_rate'))}`")
    lines.append(f"- Renderable: `{_pct(totals.get('renderable_rate'))}`")
    lines.append(f"- Materialized exact: `{_pct(totals.get('materialized_exact_rate'))}`")
    lines.append(f"- SVG exact: `{_pct(totals.get('svg_exact_rate'))}`")

    split_summary = list(report.get("split_summary") or [])
    if split_summary:
        lines.append("")
        lines.append("## Split Summary")
        lines.append("")
        lines.append("| Split | Count | Exact | Renderable | SVG Exact |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in split_summary:
            split = str(row.get("split") or "")
            lines.append(
                f"| `{split}` | `{row.get('count', 0)}` | `{_pct(row.get('exact_rate'))}` | "
                f"`{_pct(row.get('renderable_rate'))}` | `{_pct(row.get('svg_exact_rate'))}` |"
            )

    current_split = None
    for result in report.get("results") or []:
        split = str(result.get("split") or "").strip()
        if split != current_split:
            current_split = split
            lines.append("")
            lines.append(f"## {split.title()} Cases")

        label = str(result.get("label") or result.get("id") or "case").strip()
        lines.append("")
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Exact: `{_yes_no(bool(result.get('exact_match')))} `")
        lines.append(f"- Renderable: `{_yes_no(bool(result.get('renderable')))} `")
        lines.append(f"- Valid SVG: `{_yes_no(bool(result.get('valid_svg')))} `")
        lines.append("")
        lines.append("Prompt:")
        lines.append(_code_block(str(result.get("prompt") or "")))
        lines.append("")
        lines.append("Expected:")
        lines.append(_code_block(str(result.get("expected_output") or "")))
        lines.append("")
        lines.append("Response:")
        lines.append(_code_block(str(result.get("response_text") or "")))

    lines.append("")
    return "\n".join(lines)


def _render_html(report: dict[str, Any]) -> str:
    title = str(report.get("title") or "Structured Scene Tested Prompts Report").strip()
    run_name = str(report.get("run_name") or "").strip()
    totals = dict(report.get("totals") or {})
    split_summary = list(report.get("split_summary") or [])
    results = list(report.get("results") or [])

    cards: list[str] = []
    current_split = None
    for result in results:
        split = str(result.get("split") or "").strip()
        if split != current_split:
            current_split = split
            cards.append(f"<h2>{html.escape(split.title())} Cases</h2>")
        exact = bool(result.get("exact_match"))
        status_class = "pass" if exact else "fail"
        cards.append(
            "\n".join(
                [
                    f'<section class="case {status_class}">',
                    f"<h3>{html.escape(str(result.get('label') or result.get('id') or 'case'))}</h3>",
                    '<div class="meta">',
                    f"<span>Exact: <strong>{html.escape(_yes_no(exact))}</strong></span>",
                    f"<span>Renderable: <strong>{html.escape(_yes_no(bool(result.get('renderable'))))}</strong></span>",
                    f"<span>Valid SVG: <strong>{html.escape(_yes_no(bool(result.get('valid_svg'))))}</strong></span>",
                    "</div>",
                    "<h4>Prompt</h4>",
                    f"<pre>{html.escape(str(result.get('prompt') or ''))}</pre>",
                    "<h4>Expected</h4>",
                    f"<pre>{html.escape(str(result.get('expected_output') or ''))}</pre>",
                    "<h4>Response</h4>",
                    f"<pre>{html.escape(str(result.get('response_text') or ''))}</pre>",
                    "</section>",
                ]
            )
        )

    split_rows = []
    for row in split_summary:
        split_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('split') or ''))}</td>"
            f"<td>{int(row.get('count', 0))}</td>"
            f"<td>{html.escape(_pct(row.get('exact_rate')))}</td>"
            f"<td>{html.escape(_pct(row.get('renderable_rate')))}</td>"
            f"<td>{html.escape(_pct(row.get('svg_exact_rate')))}</td>"
            "</tr>"
        )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8"/>',
            f"  <title>{html.escape(title)}</title>",
            "  <style>",
            "    :root { --bg: #f5f1e8; --ink: #1f2933; --muted: #52606d; --card: #fffdf8; --line: #d9d1c2; --pass: #1f7a4d; --fail: #b42318; }",
            "    body { margin: 0; padding: 32px; font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; background: linear-gradient(180deg, #f7f2e8 0%, #efe7d8 100%); color: var(--ink); }",
            "    main { max-width: 1200px; margin: 0 auto; }",
            "    h1, h2, h3 { margin: 0 0 12px; }",
            "    p, li, td, th, span { line-height: 1.4; }",
            "    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 20px 0 28px; }",
            "    .metric, .split-table, .case { background: var(--card); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 6px 18px rgba(31, 41, 51, 0.06); }",
            "    .metric { padding: 16px; }",
            "    .metric .label { display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }",
            "    .metric .value { display: block; margin-top: 6px; font-size: 28px; font-weight: 700; }",
            "    .split-table { padding: 16px; margin: 0 0 28px; overflow-x: auto; }",
            "    table { width: 100%; border-collapse: collapse; }",
            "    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); }",
            "    .case { padding: 18px; margin: 0 0 18px; }",
            "    .case.pass { border-left: 6px solid var(--pass); }",
            "    .case.fail { border-left: 6px solid var(--fail); }",
            "    .meta { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); margin: 0 0 14px; }",
            "    pre { overflow-x: auto; background: #f8f5ee; padding: 12px; border-radius: 12px; border: 1px solid #e6dece; white-space: pre-wrap; word-break: break-word; }",
            "  </style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{html.escape(title)}</h1>",
            f"<p>{html.escape(run_name)}</p>" if run_name else "",
            '<section class="summary">',
            f'<article class="metric"><span class="label">Cases</span><span class="value">{int(totals.get("count", 0))}</span></article>',
            f'<article class="metric"><span class="label">Exact</span><span class="value">{html.escape(_pct(totals.get("exact_rate")))}</span></article>',
            f'<article class="metric"><span class="label">Renderable</span><span class="value">{html.escape(_pct(totals.get("renderable_rate")))}</span></article>',
            f'<article class="metric"><span class="label">Materialized Exact</span><span class="value">{html.escape(_pct(totals.get("materialized_exact_rate")))}</span></article>',
            f'<article class="metric"><span class="label">SVG Exact</span><span class="value">{html.escape(_pct(totals.get("svg_exact_rate")))}</span></article>',
            "</section>",
            '<section class="split-table">',
            "<h2>Split Summary</h2>",
            "<table>",
            "<thead><tr><th>Split</th><th>Count</th><th>Exact</th><th>Renderable</th><th>SVG Exact</th></tr></thead>",
            f"<tbody>{''.join(split_rows)}</tbody>",
            "</table>",
            "</section>",
            *cards,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-report", required=True, help="Path to probe report JSON")
    ap.add_argument("--output-html", required=True, help="Path to HTML output")
    ap.add_argument("--output-md", default="", help="Optional path to Markdown output")
    args = ap.parse_args()

    report_path = Path(args.probe_report).expanduser().resolve()
    report = _load_json(report_path)

    html_out = Path(args.output_html).expanduser().resolve()
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(_render_html(report), encoding="utf-8")

    if args.output_md:
        md_out = Path(args.output_md).expanduser().resolve()
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(_render_markdown(report), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
