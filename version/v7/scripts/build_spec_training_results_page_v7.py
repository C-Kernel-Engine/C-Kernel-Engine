#!/usr/bin/env python3
"""Render the manifest-driven spec training results page for docs/site/_pages."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_FILE = REPO_ROOT / "version" / "v7" / "reports" / "spec_training_manifest.json"
OUT_PAGE = REPO_ROOT / "docs" / "site" / "_pages" / "spec-training-results.html"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _fmt_iso(value: str | None) -> str:
    if not value:
        return "—"
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return escape(value)
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _spec_sort_key(spec_id: str) -> tuple[int, str]:
    head = str(spec_id or "")
    digits = "".join(ch for ch in head if ch.isdigit())
    suffix = "".join(ch for ch in head if ch.isalpha())
    return (int(digits) if digits else 9999, suffix)


def _rung_label(dir_name: str | None) -> str:
    value = str(dir_name or "")
    parts = value.rsplit("_r", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"r{parts[1]}"
    return value or "—"


def _group_runs(manifest: dict) -> list[dict]:
    grouped: dict[str, dict] = {}
    for run in manifest.get("runs", []):
        dir_name = str(run.get("dir_name") or "")
        if not dir_name:
            continue
        group = grouped.setdefault(
            dir_name,
            {
                "spec": run.get("spec"),
                "dir_name": dir_name,
                "run_number": run.get("run_number"),
                "latest_stage": run.get("stage_id"),
                "latest_ended_at": run.get("ended_at"),
                "visible_exact_rate": run.get("exact_rate"),
                "hidden_exact_rate": run.get("hidden_exact_rate"),
                "renderable_rate": run.get("renderable_rate"),
                "overall_exact_rate": run.get("overall_exact_rate"),
                "run_dir_path": run.get("run_dir_path"),
            },
        )
        if run.get("exact_rate") is not None:
            group["visible_exact_rate"] = run.get("exact_rate")
            group["hidden_exact_rate"] = run.get("hidden_exact_rate")
            group["renderable_rate"] = run.get("renderable_rate")
            group["overall_exact_rate"] = run.get("overall_exact_rate")
        previous = str(group.get("latest_ended_at") or "")
        current = str(run.get("ended_at") or "")
        if current >= previous:
            group["latest_stage"] = run.get("stage_id")
            group["latest_ended_at"] = run.get("ended_at")
    return sorted(
        grouped.values(),
        key=lambda item: (
            str(item.get("latest_ended_at") or ""),
            int(item.get("run_number") or -1),
            str(item.get("dir_name") or ""),
        ),
        reverse=True,
    )


def render_page(manifest: dict) -> str:
    specs = sorted(manifest.get("specs", []), key=lambda item: _spec_sort_key(item.get("id", "")))
    grouped_runs = _group_runs(manifest)
    best_specs = [spec for spec in specs if spec.get("best_result")]
    champion = max(
        best_specs,
        key=lambda spec: (
            (spec.get("best_result") or {}).get("exact_rate") or 0.0,
            (spec.get("best_hidden_result") or {}).get("exact_rate") or 0.0,
        ),
        default=None,
    )
    champion_result = champion.get("best_result") if champion else None
    champion_hidden = champion.get("best_hidden_result") if champion else None
    recent_rows = grouped_runs[:16]
    source_roots = manifest.get("source_roots", [])

    ladder_rows = []
    for spec in best_specs:
        best = spec.get("best_result") or {}
        hidden = spec.get("best_hidden_result") or {}
        ladder_rows.append(
            "<tr>"
            f"<td><strong>{escape(spec.get('id', ''))}</strong><br><span class=\"muted\">{escape(spec.get('name', ''))}</span></td>"
            f"<td>{escape(_rung_label(best.get('dir_name')))}</td>"
            f"<td>{_fmt_pct(best.get('exact_rate'))}</td>"
            f"<td>{_fmt_pct(hidden.get('exact_rate'))}</td>"
            f"<td>{_fmt_pct(best.get('renderable_rate'))}</td>"
            f"<td>{escape(str(spec.get('total_runs', '—')))}</td>"
            f"<td><span class=\"status\">{escape(spec.get('status', 'unknown'))}</span><br><span class=\"muted\">{escape(spec.get('lesson', ''))}</span></td>"
            "</tr>"
        )

    recent_run_rows = []
    for run in recent_rows:
        recent_run_rows.append(
            "<tr>"
            f"<td><strong>{escape(str(run.get('spec') or ''))}</strong><br><span class=\"muted\">{escape(_rung_label(run.get('dir_name')))}</span></td>"
            f"<td>{escape(str(run.get('latest_stage') or '—'))}</td>"
            f"<td>{escape(_fmt_iso(run.get('latest_ended_at')))}</td>"
            f"<td>{_fmt_pct(run.get('visible_exact_rate'))}</td>"
            f"<td>{_fmt_pct(run.get('hidden_exact_rate'))}</td>"
            f"<td>{_fmt_pct(run.get('renderable_rate'))}</td>"
            f"<td><code>{escape(str(run.get('dir_name') or ''))}</code></td>"
            "</tr>"
        )

    source_bits = "".join(
        f"<li><code>{escape(str(item.get('path') or ''))}</code> <span class=\"muted\">({escape(str(item.get('kind') or 'source'))})</span></li>"
        for item in source_roots
    )

    return f"""<!-- TITLE: Spec Training Results -->
<!-- NAV: quickstart -->

<style>
.results-hero {{
    position: relative;
    overflow: hidden;
}}

.results-hero::before {{
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 16% 18%, rgba(7,173,248,0.18), transparent 30%),
        radial-gradient(circle at 82% 14%, rgba(255,180,0,0.16), transparent 28%),
        linear-gradient(145deg, rgba(71,180,117,0.08), rgba(255,255,255,0));
    pointer-events: none;
}}

.results-hero-inner {{
    position: relative;
    z-index: 1;
}}

.results-kicker {{
    color: var(--orange);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
}}

.results-subtitle,
.results-note,
.muted {{
    color: var(--text-secondary);
}}

.results-grid,
.split-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
    margin: 1.2rem 0 2rem 0;
}}

.results-card,
.split-card {{
    border: 1px solid var(--grey);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    background: linear-gradient(155deg, rgba(255,255,255,0.02), rgba(7,173,248,0.05));
    box-shadow: var(--shadow-sm);
}}

.metric-label {{
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
}}

.metric-value {{
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0.15rem 0 0.35rem 0;
    color: var(--text-primary);
}}

.results-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0 2rem 0;
    border: 1px solid var(--grey);
    border-radius: 12px;
    overflow: hidden;
}}

.results-table th,
.results-table td {{
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--grey);
    text-align: left;
    vertical-align: top;
}}

.results-table th {{
    color: var(--orange);
    background: var(--dark-card);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}

.results-table tbody tr:nth-child(even) {{
    background: rgba(255,255,255,0.02);
}}

.results-table tbody tr:hover {{
    background: rgba(7,173,248,0.06);
}}

.status {{
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    border: 1px solid rgba(255,180,0,0.25);
    background: rgba(255,180,0,0.10);
    color: var(--text-primary);
    font-size: 0.76rem;
    font-weight: 700;
}}

.policy-callout {{
    margin: 1rem 0 1.8rem 0;
    padding: 1rem 1.1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,180,0,0.25);
    background: linear-gradient(145deg, rgba(255,180,0,0.10), rgba(7,173,248,0.08));
}}

.refresh-block {{
    margin: 1rem 0 0 0;
}}

@media (max-width: 900px) {{
    .results-table {{
        display: block;
        overflow-x: auto;
    }}
}}
</style>

<div class="card results-hero">
    <div class="results-hero-inner">
        <div class="results-kicker">Run History</div>
        <h1>Spec Training Results</h1>
        <p class="results-subtitle">
            Manifest-driven visibility for the training lines. This page is generated from local run ledgers and probe reports,
            while <a href="spec-training-method.html">spec-training-method.html</a> stays focused on the stable method and promotion rules.
        </p>
    </div>
</div>

<div class="policy-callout">
    <strong>Page split:</strong> stable method lives on <a href="spec-training-method.html">Spec Training Method</a>,
    generated run evidence lives here, and bulky raw artifacts are intended to move under <code>training-archive/</code>
    as a future submodule mount instead of growing the main docs tree.
</div>

<div class="results-grid">
    <div class="results-card">
        <div class="metric-label">Specs With Probe Data</div>
        <div class="metric-value">{len(best_specs)}</div>
        <p class="results-note">Generated from {len(source_roots)} source roots and refreshed at {escape(_fmt_iso(manifest.get('generated_at')))}.</p>
    </div>
    <div class="results-card">
        <div class="metric-label">Run Directories</div>
        <div class="metric-value">{escape(str(manifest.get('run_dir_count', 0)))}</div>
        <p class="results-note">Stage records: {escape(str(manifest.get('stage_record_count', 0)))}. Probe reports: {escape(str(manifest.get('probe_count', 0)))}.</p>
    </div>
    <div class="results-card">
        <div class="metric-label">Current Champion</div>
        <div class="metric-value">{escape(str(champion.get('id') if champion else '—'))}</div>
        <p class="results-note">{escape(_rung_label(champion_result.get('dir_name') if champion_result else None))} at {_fmt_pct(champion_result.get('exact_rate') if champion_result else None)} visible exact and {_fmt_pct(champion_hidden.get('exact_rate') if champion_hidden else None)} hidden exact.</p>
    </div>
</div>

<h2>History Split</h2>

<div class="split-grid">
    <div class="split-card">
        <h3>Method Page</h3>
        <p>Keep stable policy here: spec/rung definitions, reset-line rules, promotion gates, and decode-repair versus retrain decisions.</p>
        <p><a href="spec-training-method.html">Open spec-training-method.html</a></p>
    </div>
    <div class="split-card">
        <h3>Generated History</h3>
        <p>This page is generated from <code>version/v7/reports/spec_training_manifest.json</code>, not hand-maintained prose.</p>
        <p class="muted">It should be the first stop for “which rung is best?” and “what regressed?”</p>
    </div>
    <div class="split-card">
        <h3>Archive Mount</h3>
        <p>Raw ledgers, probe reports, tested-prompt reports, and large run folders can move under <code>training-archive/</code> as a submodule without breaking the docs surface.</p>
        <p class="muted">The docs should depend on the compact manifest, not on browsing the archive directly.</p>
    </div>
</div>

<h2>Best Rungs By Spec</h2>

<table class="results-table">
    <thead>
        <tr>
            <th>Spec</th>
            <th>Best Rung</th>
            <th>Visible Exact</th>
            <th>Hidden Exact</th>
            <th>Renderable</th>
            <th>Run Dirs</th>
            <th>Status / Lesson</th>
        </tr>
    </thead>
    <tbody>
        {''.join(ladder_rows)}
    </tbody>
</table>

<h2>Recent Run Ledger</h2>

<table class="results-table">
    <thead>
        <tr>
            <th>Run</th>
            <th>Latest Stage</th>
            <th>Finished</th>
            <th>Visible Exact</th>
            <th>Hidden Exact</th>
            <th>Renderable</th>
            <th>Artifact Dir</th>
        </tr>
    </thead>
    <tbody>
        {''.join(recent_run_rows)}
    </tbody>
</table>

<h2>Archive Policy</h2>

<div class="card">
    <p><strong>Submodule-ready split:</strong> keep the compact manifest and generated docs page in the main repo, then mount a future raw artifact archive at <code>training-archive/</code>.</p>
    <ul>
        <li>Stable docs stay in <code>docs/site/_pages/</code>.</li>
        <li>Run summaries stay in <code>version/v7/reports/spec_training_manifest.json</code>.</li>
        <li>Heavy artifacts move to <code>training-archive/</code> once you are ready to create the separate repo.</li>
    </ul>
    <p class="muted">Current source roots:</p>
    <ul>
        {source_bits}
    </ul>
    <div class="refresh-block">
        <p><strong>Refresh command:</strong></p>
        <pre><code>bash docs/site/build.sh</code></pre>
    </div>
</div>
"""


def main() -> int:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    OUT_PAGE.write_text(render_page(manifest), encoding="utf-8")
    print(f"wrote {OUT_PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
