#!/usr/bin/env python3
"""Build an HTML progress report for corrected spec10 probe runs."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = Path("~/.cache/ck-engine-v7/models/reports/spec08_spec10_progress_report_20260317.html").expanduser()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_layout(prompt: str) -> str:
    for token in str(prompt or "").split():
        if token.startswith("[layout:") and token.endswith("]"):
            return token[len("[layout:") : -1]
    return "<unknown>"


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _ratio(rate: float, count: int) -> str:
    hits = round(rate * count)
    return f"{hits}/{count}"


def _metric_class(rate: float) -> str:
    if rate >= 0.9:
        return "good"
    if rate >= 0.7:
        return "mid"
    return "bad"


def _build_run_summary(name: str, title: str, run_dir: Path, probe: dict[str, Any]) -> dict[str, Any]:
    totals = dict(probe.get("totals") or {})
    results = list(probe.get("results") or [])
    split_summary = list(probe.get("split_summary") or [])
    count = int(totals.get("count") or len(results) or 0)
    by_layout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_layout[_parse_layout(str(row.get("prompt") or ""))].append(row)

    layout_summary: dict[str, dict[str, Any]] = {}
    for layout, rows in sorted(by_layout.items()):
        denom = len(rows)
        layout_summary[layout] = {
            "count": denom,
            "exact_rate": sum(1 for row in rows if row.get("exact_match")) / denom if denom else 0.0,
            "renderable_rate": sum(1 for row in rows if row.get("renderable")) / denom if denom else 0.0,
            "materialized_exact_rate": sum(1 for row in rows if row.get("materialized_exact_match")) / denom if denom else 0.0,
        }

    misses = [
        row
        for row in results
        if not row.get("exact_match") or not row.get("materialized_exact_match")
    ]
    return {
        "name": name,
        "title": title,
        "run_dir": run_dir,
        "probe": probe,
        "totals": totals,
        "count": count,
        "split_summary": split_summary,
        "layout_summary": layout_summary,
        "misses": misses,
        "results": results,
    }


def _choose_example(run: dict[str, Any], *, want_layout: str | None = None, exact: bool | None = None, materialized: bool | None = None) -> dict[str, Any] | None:
    for row in run["results"]:
        if want_layout and _parse_layout(str(row.get("prompt") or "")) != want_layout:
            continue
        if exact is not None and bool(row.get("exact_match")) != exact:
            continue
        if materialized is not None and bool(row.get("materialized_exact_match")) != materialized:
            continue
        return row
    return None


def _svg_or_placeholder(svg: str | None, label: str) -> str:
    content = str(svg or "").strip()
    if content:
        return content
    return f'<div class="empty">{html.escape(label)}</div>'


def _pre(text: str | None) -> str:
    return f"<pre>{html.escape(str(text or '').strip())}</pre>"


def _link(path: Path, label: str) -> str:
    return f'<a href="{html.escape(path.as_uri())}">{html.escape(label)}</a>'


def _example_section(title: str, note: str, prompt: str, expected_row: dict[str, Any] | None, r3_row: dict[str, Any] | None, r4_row: dict[str, Any] | None) -> str:
    def _cell(run_label: str, row: dict[str, Any] | None, *, expected: bool = False) -> str:
        if row is None:
            return f"<div class='example-meta'><strong>{html.escape(run_label)}</strong><div class='sub'>No case selected</div></div>"
        svg = row.get("expected_rendered_output") if expected else row.get("rendered_svg")
        exact = "expected" if expected else ("exact" if row.get("exact_match") else "not exact")
        materialized = "expected" if expected else ("render match" if row.get("materialized_exact_match") else "render drift")
        render_error = "" if expected else str(row.get("render_error") or "")
        return (
            f"<div class='example-meta'><strong>{html.escape(run_label)}</strong>"
            f"<div class='sub'>{html.escape(exact)} · {html.escape(materialized)}</div>"
            f"{f'<div class=\"sub bad\">{html.escape(render_error)}</div>' if render_error else ''}"
            f"</div>"
            f"<div class='svg-shell'>{_svg_or_placeholder(svg, 'No render')}</div>"
            f"<div class='dsl-label'>DSL</div>{_pre((row.get('expected_output') if expected else row.get('parsed_output')) or '')}"
        )

    return (
        "<section class='section'>"
        f"<h2>{html.escape(title)}</h2>"
        f"<p class='lead-tight'>{html.escape(note)}</p>"
        f"<div class='prompt-box'><div class='dsl-label'>Prompt</div>{_pre(prompt)}</div>"
        "<div class='examples-grid'>"
        f"<div class='example-card'>{_cell('Expected', expected_row, expected=True)}</div>"
        f"<div class='example-card'>{_cell('spec10 r3', r3_row)}</div>"
        f"<div class='example-card'>{_cell('spec10 r4', r4_row)}</div>"
        "</div>"
        "</section>"
    )


def build_report(spec08: dict[str, Any], r3: dict[str, Any], r4: dict[str, Any]) -> str:
    r4_totals = r4["totals"]
    r3_totals = r3["totals"]
    spec08_totals = spec08["totals"]

    delta_exact = (float(r4_totals["exact_rate"]) - float(r3_totals["exact_rate"])) * 100.0
    delta_render = (float(r4_totals["renderable_rate"]) - float(r3_totals["renderable_rate"])) * 100.0
    delta_materialized = (float(r4_totals["materialized_exact_rate"]) - float(r3_totals["materialized_exact_rate"])) * 100.0

    summary_rows = []
    for run in (spec08, r3, r4):
        totals = run["totals"]
        count = run["count"]
        summary_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(run['name'])}</strong><div class='sub'>{html.escape(run['title'])}</div></td>"
            f"<td class='{_metric_class(float(totals['exact_rate']))}'>{_ratio(float(totals['exact_rate']), count)}<div class='sub'>{_pct(float(totals['exact_rate']))}</div></td>"
            f"<td class='{_metric_class(float(totals['renderable_rate']))}'>{_ratio(float(totals['renderable_rate']), count)}<div class='sub'>{_pct(float(totals['renderable_rate']))}</div></td>"
            f"<td class='{_metric_class(float(totals['materialized_exact_rate']))}'>{_ratio(float(totals['materialized_exact_rate']), count)}<div class='sub'>{_pct(float(totals['materialized_exact_rate']))}</div></td>"
            f"<td>{_link(run['run_dir'], 'run dir')}<br>{_link(run['run_dir'] / run['run_dir'].name.replace('spec10_asset_scene_dsl', 'spec10_probe_report').replace('spec08_rich_scene_dsl_l3_d192_h384_ctx512_r1', 'spec08_probe_report.html') if False else run['run_dir'] / ('spec08_probe_report.html' if run['name'].startswith('spec08') else 'spec10_probe_report.html'), 'probe html')}<br>{_link(run['run_dir'] / ('spec08_probe_report.json' if run['name'].startswith('spec08') else 'spec10_probe_report.json'), 'probe json')}</td>"
            "</tr>"
        )

    # The above inline path logic is ugly; use explicit rows instead.
    summary_rows = [
        "<tr>"
        f"<td><strong>{html.escape(spec08['name'])}</strong><div class='sub'>{html.escape(spec08['title'])}</div></td>"
        f"<td class='{_metric_class(float(spec08_totals['exact_rate']))}'>{_ratio(float(spec08_totals['exact_rate']), spec08['count'])}<div class='sub'>{_pct(float(spec08_totals['exact_rate']))}</div></td>"
        f"<td class='{_metric_class(float(spec08_totals['renderable_rate']))}'>{_ratio(float(spec08_totals['renderable_rate']), spec08['count'])}<div class='sub'>{_pct(float(spec08_totals['renderable_rate']))}</div></td>"
        f"<td class='{_metric_class(float(spec08_totals['materialized_exact_rate']))}'>{_ratio(float(spec08_totals['materialized_exact_rate']), spec08['count'])}<div class='sub'>{_pct(float(spec08_totals['materialized_exact_rate']))}</div></td>"
        f"<td>{_link(spec08['run_dir'], 'run dir')}<br>{_link(spec08['run_dir'] / 'spec08_probe_report.html', 'probe html')}<br>{_link(spec08['run_dir'] / 'spec08_probe_report.json', 'probe json')}</td>"
        "</tr>",
        "<tr>"
        f"<td><strong>{html.escape(r3['name'])}</strong><div class='sub'>{html.escape(r3['title'])}</div></td>"
        f"<td class='{_metric_class(float(r3_totals['exact_rate']))}'>{_ratio(float(r3_totals['exact_rate']), r3['count'])}<div class='sub'>{_pct(float(r3_totals['exact_rate']))}</div></td>"
        f"<td class='{_metric_class(float(r3_totals['renderable_rate']))}'>{_ratio(float(r3_totals['renderable_rate']), r3['count'])}<div class='sub'>{_pct(float(r3_totals['renderable_rate']))}</div></td>"
        f"<td class='{_metric_class(float(r3_totals['materialized_exact_rate']))}'>{_ratio(float(r3_totals['materialized_exact_rate']), r3['count'])}<div class='sub'>{_pct(float(r3_totals['materialized_exact_rate']))}</div></td>"
        f"<td>{_link(r3['run_dir'], 'run dir')}<br>{_link(r3['run_dir'] / 'spec10_probe_report.html', 'probe html')}<br>{_link(r3['run_dir'] / 'spec10_probe_report.json', 'probe json')}</td>"
        "</tr>",
        "<tr>"
        f"<td><strong>{html.escape(r4['name'])}</strong><div class='sub'>{html.escape(r4['title'])}</div></td>"
        f"<td class='{_metric_class(float(r4_totals['exact_rate']))}'>{_ratio(float(r4_totals['exact_rate']), r4['count'])}<div class='sub'>{_pct(float(r4_totals['exact_rate']))}</div></td>"
        f"<td class='{_metric_class(float(r4_totals['renderable_rate']))}'>{_ratio(float(r4_totals['renderable_rate']), r4['count'])}<div class='sub'>{_pct(float(r4_totals['renderable_rate']))}</div></td>"
        f"<td class='{_metric_class(float(r4_totals['materialized_exact_rate']))}'>{_ratio(float(r4_totals['materialized_exact_rate']), r4['count'])}<div class='sub'>{_pct(float(r4_totals['materialized_exact_rate']))}</div></td>"
        f"<td>{_link(r4['run_dir'], 'run dir')}<br>{_link(r4['run_dir'] / 'spec10_probe_report.html', 'probe html')}<br>{_link(r4['run_dir'] / 'spec10_probe_report.json', 'probe json')}</td>"
        "</tr>",
    ]

    all_layouts = sorted(set(spec08["layout_summary"]) | set(r3["layout_summary"]) | set(r4["layout_summary"]))
    layout_rows: list[str] = []
    for layout in all_layouts:
        cells = [f"<td><strong>{html.escape(layout)}</strong></td>"]
        for run in (spec08, r3, r4):
            summary = run["layout_summary"].get(layout)
            if not summary:
                cells.extend(["<td class='pending'>n/a</td>"] * 3)
                continue
            cells.append(f"<td class='{_metric_class(summary['exact_rate'])}'>{_ratio(summary['exact_rate'], summary['count'])}</td>")
            cells.append(f"<td class='{_metric_class(summary['renderable_rate'])}'>{_ratio(summary['renderable_rate'], summary['count'])}</td>")
            cells.append(f"<td class='{_metric_class(summary['materialized_exact_rate'])}'>{_ratio(summary['materialized_exact_rate'], summary['count'])}</td>")
        layout_rows.append("<tr>" + "".join(cells) + "</tr>")

    r4_miss_items = []
    for row in r4["misses"]:
        r4_miss_items.append(
            "<li>"
            f"<strong>{html.escape(str(row['id']))}</strong> "
            f"<span class='sub'>{html.escape(_parse_layout(str(row.get('prompt') or '')))}</span>"
            f"<div class='mono'>{html.escape(str(row.get('prompt') or ''))}</div>"
            f"<div class='sub'>{'compiled-correct but non-canonical' if row.get('materialized_exact_match') else 'real render/content drift'}</div>"
            "</li>"
        )

    # Example selection
    expected_success = _choose_example(r4, want_layout="pipeline_lane", exact=True, materialized=True) or _choose_example(r4, exact=True, materialized=True)
    prompt_success = str(expected_success.get("prompt") or "") if expected_success else ""
    r3_success = None
    r4_success = None
    if prompt_success:
        r3_success = next((row for row in r3["results"] if str(row.get("prompt") or "") == prompt_success), None)
        r4_success = next((row for row in r4["results"] if str(row.get("prompt") or "") == prompt_success), None)

    expected_near = next((row for row in r4["misses"] if row.get("materialized_exact_match")), None)
    prompt_near = str(expected_near.get("prompt") or "") if expected_near else ""
    r3_near = next((row for row in r3["results"] if str(row.get("prompt") or "") == prompt_near), None) if prompt_near else None
    r4_near = next((row for row in r4["results"] if str(row.get("prompt") or "") == prompt_near), None) if prompt_near else None

    expected_miss = next((row for row in r4["misses"] if not row.get("materialized_exact_match")), None)
    prompt_miss = str(expected_miss.get("prompt") or "") if expected_miss else ""
    r3_miss = next((row for row in r3["results"] if str(row.get("prompt") or "") == prompt_miss), None) if prompt_miss else None
    r4_miss = next((row for row in r4["results"] if str(row.get("prompt") or "") == prompt_miss), None) if prompt_miss else None

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>spec08-spec10 progress report</title>
<style>
:root {{
  --bg: #0f1720;
  --panel: rgba(15, 23, 32, 0.76);
  --panel-2: rgba(20, 31, 44, 0.92);
  --ink: #edf2f7;
  --muted: #9fb1c3;
  --line: rgba(148, 163, 184, 0.24);
  --good: #34d399;
  --mid: #fbbf24;
  --bad: #fb7185;
  --accent: #38bdf8;
  --accent-2: #f59e0b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  font-family: 'Iowan Old Style', 'Palatino Linotype', 'Book Antiqua', Georgia, serif;
  background:
    radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.22), transparent 28%),
    radial-gradient(circle at 100% 0%, rgba(245, 158, 11, 0.18), transparent 26%),
    linear-gradient(180deg, #091019 0%, #101923 46%, #0b131d 100%);
}}
main {{ max-width: 1420px; margin: 0 auto; padding: 32px 20px 56px; }}
.hero {{ display: grid; gap: 16px; margin-bottom: 24px; }}
.kicker {{ font: 600 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent-2); }}
h1 {{ margin: 0; font-size: clamp(36px, 6vw, 70px); line-height: 0.94; max-width: 10ch; }}
.lead {{ max-width: 90ch; color: var(--muted); font-size: 18px; line-height: 1.5; }}
.lead-tight {{ color: var(--muted); font-size: 15px; line-height: 1.5; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin: 22px 0 28px; }}
.card, .section {{
  background: linear-gradient(180deg, rgba(22, 31, 44, 0.92), rgba(11, 18, 27, 0.96));
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: 0 18px 44px rgba(0,0,0,0.28);
}}
.card {{ padding: 18px; }}
.section {{ padding: 20px; margin-top: 18px; }}
.metric {{ font-size: 42px; line-height: 1; font-weight: 700; margin: 8px 0 6px; }}
.sub {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
.good {{ color: var(--good); font-weight: 700; }}
.mid {{ color: var(--mid); font-weight: 700; }}
.bad {{ color: var(--bad); font-weight: 700; }}
.pending {{ color: var(--muted); font-style: italic; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; vertical-align: top; }}
th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
ul {{ margin: 8px 0 0 18px; padding: 0; }}
li {{ margin: 8px 0; }}
.columns {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; }}
.mono {{ font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); margin-top: 4px; word-break: break-word; }}
.prompt-box {{ margin-top: 12px; }}
.examples-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }}
.example-card {{
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px;
  background: linear-gradient(180deg, rgba(24, 37, 53, 0.92), rgba(10, 18, 28, 0.95));
}}
.example-meta {{ display: grid; gap: 4px; margin-bottom: 10px; }}
.svg-shell {{
  min-height: 220px;
  padding: 10px;
  border-radius: 14px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background:
    linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
    repeating-linear-gradient(45deg, rgba(148,163,184,0.04) 0, rgba(148,163,184,0.04) 10px, transparent 10px, transparent 20px);
  overflow: auto;
}}
.svg-shell svg {{ width: 100%; height: auto; display: block; }}
.empty {{
  min-height: 200px;
  display: grid;
  place-items: center;
  color: var(--muted);
  font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
}}
.dsl-label {{
  font: 600 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-2);
  margin: 12px 0 6px;
}}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  padding: 12px;
  border-radius: 14px;
  background: rgba(5, 10, 17, 0.68);
  border: 1px solid rgba(148, 163, 184, 0.12);
  color: #d7e3ef;
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
}}
@media (max-width: 1100px) {{
  .examples-grid, .columns {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div class="kicker">Corrected Progress Report · 2026-03-17</div>
    <h1>spec10 actually learned the richer scene DSL</h1>
    <div class="lead">
      This report reflects the corrected <strong>spec10</strong> evaluation path. The earlier overnight `0/35` read for `r4` was caused by a probe-contract bug:
      the decoder stopped on <code>[/scene]</code>, and <code>ck_chat.py</code> strips matched stop text from the returned response. After rescoring with the semantic close tag handled in the adapter instead of the decoder, <strong>spec10 r4</strong> is the first asset-grounded scene run that looks genuinely strong.
    </div>
  </section>

  <section class="grid">
    <div class="card"><h2>Corrected spec10 r4</h2><div class="metric good">33/35</div><div class="sub">exact · 35/35 renderable · 34/35 materialized exact</div></div>
    <div class="card"><h2>spec10 r3 → r4</h2><div class="metric good">+{delta_exact:.1f} pts</div><div class="sub">exact · +{delta_render:.1f} pts renderable · +{delta_materialized:.1f} pts materialized</div></div>
    <div class="card"><h2>Against spec08 r1</h2><div class="metric good">Closer</div><div class="sub">spec08 r1 is still the strongest older rich-scene reference at 27/36 exact, but spec10 r4 now exceeds it on percentage terms and is asset-grounded.</div></div>
    <div class="card"><h2>Remaining Gap</h2><div class="metric mid">2 cases</div><div class="sub">1 compiled-correct but non-canonical `comparison_span_chart`, 1 real `poster_stack` paper-editorial drift.</div></div>
  </section>

  <section class="section">
    <h2>Overall Scorecard</h2>
    <table>
      <thead>
        <tr><th>Run</th><th>Exact</th><th>Renderable</th><th>Materialized Exact</th><th>Artifacts</th></tr>
      </thead>
      <tbody>
        {''.join(summary_rows)}
      </tbody>
    </table>
  </section>

  <section class="section">
    <h2>Layout Breakdown</h2>
    <table>
      <thead>
        <tr>
          <th>Layout</th>
          <th colspan="3">spec08 r1</th>
          <th colspan="3">spec10 r3</th>
          <th colspan="3">spec10 r4</th>
        </tr>
        <tr>
          <th></th>
          <th>Exact</th><th>Renderable</th><th>Materialized</th>
          <th>Exact</th><th>Renderable</th><th>Materialized</th>
          <th>Exact</th><th>Renderable</th><th>Materialized</th>
        </tr>
      </thead>
      <tbody>
        {''.join(layout_rows)}
      </tbody>
    </table>
  </section>

  <section class="section columns">
    <div>
      <h2>What Changed</h2>
      <ul>
        <li><strong>Not a fake loss win.</strong> `r4` still has clean training behavior, but the big change here is that evaluation now matches the scene contract correctly.</li>
        <li><strong>The old `r4 0/35` result was wrong.</strong> The model was already emitting well-formed scene documents with `[/scene]`, but the decoder stripped the close tag before scoring.</li>
        <li><strong>`r4` is materially better than `r3` even after correcting both.</strong> `r3` lands at 22/35 exact and 28/35 renderable; `r4` moves to 33/35 exact and 35/35 renderable.</li>
        <li><strong>The remaining misses are narrow.</strong> This is no longer a broad “spec10 doesn’t work” situation.</li>
      </ul>
    </div>
    <div>
      <h2>Remaining `r4` Misses</h2>
      <ul>
        {''.join(r4_miss_items)}
      </ul>
    </div>
  </section>

  {_example_section(
        "Example A · Fully Solved Rich Scene",
        "A representative success case. `r4` matches the expected scene and rendered SVG cleanly; `r3` is visibly weaker on the same prompt.",
        prompt_success,
        expected_success,
        r3_success,
        r4_success,
    )}

  {_example_section(
        "Example B · Compiled Correct But Not Canonical",
        "This `r4` case already compiles to the right infographic, but the emitted scene tokens are not fully canonical yet. That is a small cleanup problem, not a visual failure.",
        prompt_near,
        expected_near,
        r3_near,
        r4_near,
    )}

  {_example_section(
        "Example C · Remaining True Drift",
        "This is one of the two remaining `r4` misses. It still renders, but the scene attributes drift enough to miss the expected compiled output.",
        prompt_miss,
        expected_miss,
        r3_miss,
        r4_miss,
    )}

  <section class="section">
    <h2>Recommendation</h2>
    <ul>
      <li>Promote <strong>spec10 r4</strong> as the working `spec10` baseline.</li>
      <li>Do <strong>not</strong> keep treating the old overnight `0/35` as meaningful; it was a contract bug.</li>
      <li>Do a narrow `r5` only if it targets the two surviving misses. This should be a repair pass, not a blind ladder rung.</li>
      <li>Keep improving the compiler and asset-grounded vocabulary. The training side is now good enough that the visual-language ceiling matters again.</li>
    </ul>
  </section>
</main>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a corrected progress report for spec08/spec10 runs.")
    ap.add_argument("--spec08-run", required=True, type=Path)
    ap.add_argument("--spec10-r3-run", required=True, type=Path)
    ap.add_argument("--spec10-r4-run", required=True, type=Path)
    ap.add_argument("--output", default=str(DEFAULT_REPORT), type=Path)
    args = ap.parse_args()

    spec08_probe = _load_json(args.spec08_run / "spec08_probe_report.json")
    r3_probe = _load_json(args.spec10_r3_run / "spec10_probe_report.json")
    r4_probe = _load_json(args.spec10_r4_run / "spec10_probe_report.json")

    spec08 = _build_run_summary("spec08 r1", "Older rich-scene baseline", args.spec08_run, spec08_probe)
    r3 = _build_run_summary("spec10 r3", "First corrected semantic-contract rung", args.spec10_r3_run, r3_probe)
    r4 = _build_run_summary("spec10 r4", "Corrected asset-grounded baseline", args.spec10_r4_run, r4_probe)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(spec08, r3, r4), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
