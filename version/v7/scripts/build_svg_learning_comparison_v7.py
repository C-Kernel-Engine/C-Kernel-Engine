#!/usr/bin/env python3
"""Build a single comparison page for spec02, spec03, and the structured SVG toy."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


MODELS_ROOT = Path.home() / ".cache" / "ck-engine-v7" / "models"
TRAIN_ROOT = MODELS_ROOT / "train"

SPEC02_RUN = TRAIN_ROOT / "svg_l16_d128_h512_v1024_ctx512_spec02"
SPEC03_RUN = TRAIN_ROOT / "svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512"
TOY_RUN = TRAIN_ROOT / "toy_svg_structured_atoms_ctx512_d64_h128"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _html(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def _pct(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{num * 100:.0f}%"


def _style_metric(value: Any, invert: bool = False) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "na"
    good = num <= 0.25 if invert else num >= 0.75
    mid = num <= 0.5 if invert else num >= 0.4
    if good:
        return "good"
    if mid:
        return "mid"
    return "bad"


def _best_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    def score(entry: dict[str, Any]) -> tuple[float, float, float]:
        metrics = entry.get("metrics") or {}
        return (
            float(metrics.get("adherence") or 0.0),
            float(metrics.get("tag_adherence") or 0.0),
            float(metrics.get("valid_svg_rate") or 0.0),
        )

    return max(entries, key=score)


def _metric(entry: dict[str, Any], name: str) -> float | None:
    value = (entry.get("metrics") or {}).get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_output_text(text: Any) -> str:
    return str(text or "").replace("<eos>", "").strip()


def _extract_svg_fragment(text: Any) -> str | None:
    cleaned = _clean_output_text(text)
    start = cleaned.find("<svg")
    if start < 0:
        return None
    end = cleaned.find("</svg>", start)
    if end < 0:
        return None
    return cleaned[start : end + len("</svg>")]


def _collect_spec_probe_outputs(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in entry.get("probe_results") or []:
        samples = probe.get("samples") or []
        sample = samples[0] if samples else {}
        response = _clean_output_text(sample.get("response"))
        scores = sample.get("scores") or {}
        rows.append(
            {
                "label": probe.get("probe_id") or probe.get("prompt") or "probe",
                "prompt": probe.get("prompt") or "",
                "probe_type": probe.get("type") or "probe",
                "svg": _extract_svg_fragment(response),
                "output_text": response,
                "valid_svg": float(scores.get("valid_svg") or 0.0),
                "adherence": float(scores.get("adherence") or 0.0),
                "tag_adherence": float(scores.get("tag_adherence") or 0.0),
            }
        )
    return rows


def _collect_toy_probe_outputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rendered_svg = _clean_output_text(row.get("rendered_svg"))
        out.append(
            {
                "label": row.get("label") or "probe",
                "prompt": row.get("prompt") or "",
                "probe_type": row.get("split") or "probe",
                "svg": rendered_svg or None,
                "output_text": rendered_svg or _clean_output_text(row.get("ir")),
                "exact_match": bool(row.get("exact_match")),
                "renderable": bool(row.get("rendered_svg")),
            }
        )
    return out


def _summarize_spec_run(
    run_dir: Path,
    label: str,
    family: str,
    tokenizer: str,
    dataset_story: str,
    report_name: str,
    probe_phase: str = "best",
) -> dict[str, Any]:
    matrix = _load_json(run_dir / "stage_eval_matrix.json")
    entries = matrix.get("entries") or []
    entries = sorted(entries, key=lambda item: int(item.get("run_order") or 0))
    best = _best_entry(entries)
    latest = entries[-1]
    display = best if probe_phase == "best" else latest
    return {
        "label": label,
        "family": family,
        "run_dir": str(run_dir),
        "report_path": str(run_dir / report_name),
        "latest_phase": latest.get("phase_label"),
        "best_phase": best.get("phase_label"),
        "latest_valid": _metric(latest, "valid_svg_rate"),
        "latest_adh": _metric(latest, "adherence"),
        "latest_tag": _metric(latest, "tag_adherence"),
        "latest_ood": _metric(latest, "ood_robustness"),
        "best_valid": _metric(best, "valid_svg_rate"),
        "best_adh": _metric(best, "adherence"),
        "best_tag": _metric(best, "tag_adherence"),
        "best_ood": _metric(best, "ood_robustness"),
        "entry_count": len(entries),
        "tokenizer_story": tokenizer,
        "dataset_story": dataset_story,
        "gallery_phase": display.get("phase_label"),
        "gallery_phase_kind": probe_phase,
        "gallery_outputs": _collect_spec_probe_outputs(display),
    }


def _summarize_toy_structured(run_dir: Path) -> dict[str, Any]:
    report = _load_json(run_dir / "toy_svg_structured_probe_report.json")
    train = _load_json(run_dir / "train_structured_svg_atoms_stage_a.json")
    rows = report.get("results") or []
    holdouts = [row for row in rows if row.get("split") == "holdout"]
    exact = sum(1 for row in rows if row.get("exact_match"))
    renderable = sum(1 for row in rows if row.get("rendered_svg"))
    holdout_exact = sum(1 for row in holdouts if row.get("exact_match"))
    ck_loss = train.get("ck_loss") or {}
    return {
        "label": "Toy Structured DSL",
        "family": "fixed symbolic SVG IR",
        "run_dir": str(run_dir),
        "report_path": str(run_dir / "svg_training_report_card.html"),
        "probe_count": len(rows),
        "exact": exact,
        "renderable": renderable,
        "holdout_exact": holdout_exact,
        "holdout_count": len(holdouts),
        "final_loss": ck_loss.get("final"),
        "best_loss": ck_loss.get("min"),
        "tokenizer_story": "No-merge fixed vocabulary of prompt atoms and SVG IR atoms.",
        "dataset_story": "Pure DSL -> symbolic SVG IR with deterministic rendering to SVG.",
        "gallery_outputs": _collect_toy_probe_outputs(rows),
    }


def _run_card(run: dict[str, Any]) -> str:
    return f"""
    <article class="run-card">
      <div class="eyebrow">{_html(run['family'])}</div>
      <h2>{_html(run['label'])}</h2>
      <div class="path">{_html(run['run_dir'])}</div>
      <div class="metric-grid">
        <div class="metric">
          <div class="k">Latest Valid</div>
          <div class="v { _style_metric(run.get('latest_valid')) }">{_pct(run.get('latest_valid'))}</div>
        </div>
        <div class="metric">
          <div class="k">Latest Adherence</div>
          <div class="v { _style_metric(run.get('latest_adh')) }">{_pct(run.get('latest_adh'))}</div>
        </div>
        <div class="metric">
          <div class="k">Best Adherence</div>
          <div class="v { _style_metric(run.get('best_adh')) }">{_pct(run.get('best_adh'))}</div>
        </div>
        <div class="metric">
          <div class="k">Latest Tag</div>
          <div class="v { _style_metric(run.get('latest_tag')) }">{_pct(run.get('latest_tag'))}</div>
        </div>
      </div>
      <p><strong>Tokenizer:</strong> {_html(run['tokenizer_story'])}</p>
      <p><strong>Data story:</strong> {_html(run['dataset_story'])}</p>
      <p><strong>Read it as:</strong> best phase <code>{_html(run['best_phase'])}</code>, latest phase <code>{_html(run['latest_phase'])}</code>, {run['entry_count']} evaluated phase(s).</p>
      <p><a href="file://{_html(run['report_path'])}">Open report card</a></p>
    </article>
    """


def _toy_card(run: dict[str, Any]) -> str:
    exact_rate = run["exact"] / max(run["probe_count"], 1)
    render_rate = run["renderable"] / max(run["probe_count"], 1)
    holdout_rate = run["holdout_exact"] / max(run["holdout_count"], 1)
    return f"""
    <article class="run-card">
      <div class="eyebrow">{_html(run['family'])}</div>
      <h2>{_html(run['label'])}</h2>
      <div class="path">{_html(run['run_dir'])}</div>
      <div class="metric-grid">
        <div class="metric">
          <div class="k">Exact</div>
          <div class="v { _style_metric(exact_rate) }">{run['exact']}/{run['probe_count']}</div>
        </div>
        <div class="metric">
          <div class="k">Renderable</div>
          <div class="v { _style_metric(render_rate) }">{run['renderable']}/{run['probe_count']}</div>
        </div>
        <div class="metric">
          <div class="k">Holdout Exact</div>
          <div class="v { _style_metric(holdout_rate) }">{run['holdout_exact']}/{run['holdout_count']}</div>
        </div>
        <div class="metric">
          <div class="k">Best Loss</div>
          <div class="v { _style_metric(run.get('best_loss'), invert=True) }">{float(run['best_loss'] or 0.0):.4f}</div>
        </div>
      </div>
      <p><strong>Tokenizer:</strong> {_html(run['tokenizer_story'])}</p>
      <p><strong>Data story:</strong> {_html(run['dataset_story'])}</p>
      <p><strong>Read it as:</strong> exact symbolic control is now visible, and the renderer turns the IR into final SVG deterministically.</p>
      <p><a href="file://{_html(run['report_path'])}">Open report card</a></p>
    </article>
    """


def _score_chip(value: Any, invert: bool = False) -> str:
    cls = _style_metric(value, invert=invert)
    return cls if cls in {"good", "mid", "bad"} else "na"


def _probe_preview(svg: str | None) -> str:
    if svg:
        return f'<div class="svg-frame">{svg}</div>'
    return '<div class="svg-frame empty">No extractable SVG root in this output.</div>'


def _spec_probe_card(row: dict[str, Any]) -> str:
    return f"""
    <article class="probe-card">
      <div class="probe-head">
        <div>
          <div class="probe-label">{_html(row['label'])}</div>
          <div class="probe-prompt"><code>{_html(row['prompt'])}</code></div>
        </div>
        <div class="probe-chips">
          <span class="chip { _score_chip(row.get('valid_svg')) }">valid {_pct(row.get('valid_svg'))}</span>
          <span class="chip { _score_chip(row.get('adherence')) }">adh {_pct(row.get('adherence'))}</span>
          <span class="chip { _score_chip(row.get('tag_adherence')) }">tag {_pct(row.get('tag_adherence'))}</span>
        </div>
      </div>
      {_probe_preview(row.get('svg'))}
      <pre>{_html(row.get('output_text') or '')}</pre>
    </article>
    """


def _toy_probe_card(row: dict[str, Any]) -> str:
    exact_cls = "good" if row.get("exact_match") else "bad"
    render_cls = "good" if row.get("renderable") else "bad"
    return f"""
    <article class="probe-card">
      <div class="probe-head">
        <div>
          <div class="probe-label">{_html(row['label'])}</div>
          <div class="probe-prompt"><code>{_html(row['prompt'])}</code></div>
        </div>
        <div class="probe-chips">
          <span class="chip stage">{_html(row.get('probe_type') or 'probe')}</span>
          <span class="chip {exact_cls}">{'exact' if row.get('exact_match') else 'drift'}</span>
          <span class="chip {render_cls}">{'renderable' if row.get('renderable') else 'no render'}</span>
        </div>
      </div>
      {_probe_preview(row.get('svg'))}
      <pre>{_html(row.get('output_text') or '')}</pre>
    </article>
    """


def _probe_gallery(run: dict[str, Any], toy: bool = False) -> str:
    outputs = run.get("gallery_outputs") or []
    cards = "".join(_toy_probe_card(row) if toy else _spec_probe_card(row) for row in outputs)
    if toy:
        subtitle = "Rendered SVG from the symbolic IR probes."
    else:
        subtitle = f"Probe outputs from {run['gallery_phase_kind']} phase <code>{_html(run['gallery_phase'])}</code>."
    return f"""
    <div class="gallery-column">
      <div class="eyebrow">{_html(run['family'])}</div>
      <h3>{_html(run['label'])}</h3>
      <p>{subtitle}</p>
      <div class="probe-stack">
        {cards}
      </div>
    </div>
    """


def _build_html(spec02: dict[str, Any], spec03: dict[str, Any], toy: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SVG Learning Comparison</title>
  <style>
    :root {{
      --bg: #0b0d12;
      --panel: rgba(255,255,255,0.05);
      --border: rgba(255,255,255,0.12);
      --text: #edf2f7;
      --muted: #98a2b3;
      --good: #39d98a;
      --mid: #ffb020;
      --bad: #ff7b72;
      --accent: #7aa2ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,0.15), transparent 30%),
        radial-gradient(circle at top right, rgba(57,217,138,0.12), transparent 24%),
        linear-gradient(180deg, #12151c 0%, #090b10 100%);
    }}
    .page {{ width: min(1460px, calc(100vw - 40px)); margin: 24px auto 48px; }}
    .hero, .panel, .run-card {{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 22px;
      box-shadow: 0 24px 60px rgba(0,0,0,0.28);
      backdrop-filter: blur(10px);
    }}
    .hero {{ padding: 28px 30px; margin-bottom: 20px; }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(122,162,255,0.16);
      color: #bfd1ff;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
      font-weight: 800;
    }}
    h1 {{ margin: 12px 0 8px; font-size: 38px; line-height: 1.05; }}
    h2 {{ margin: 10px 0 8px; font-size: 24px; }}
    h3 {{ margin: 0 0 8px; font-size: 20px; }}
    p, li {{ line-height: 1.6; color: var(--muted); }}
    .hero-grid, .cards, .compare-grid {{ display: grid; gap: 18px; }}
    .hero-grid {{ grid-template-columns: 1.5fr 1fr; align-items: start; }}
    .summary-strip {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .summary-card, .metric {{ border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); border-radius: 16px; padding: 14px 16px; }}
    .summary-card .k, .metric .k {{ text-transform: uppercase; letter-spacing: 0.06em; font-size: 12px; font-weight: 700; color: var(--muted); }}
    .summary-card .v, .metric .v {{ margin-top: 8px; font-size: 28px; font-weight: 800; }}
    .panel {{ padding: 24px 26px; margin-top: 20px; }}
    .cards {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .run-card {{ padding: 22px; }}
    .path {{ color: var(--muted); font-size: 13px; word-break: break-word; margin-bottom: 14px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .gallery-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }}
    .gallery-column {{ display: grid; gap: 14px; }}
    .probe-stack {{ display: grid; gap: 14px; }}
    .probe-card {{
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      border-radius: 18px;
      padding: 16px;
    }}
    .probe-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .probe-label {{ font-size: 16px; font-weight: 800; color: var(--text); }}
    .probe-prompt {{ margin-top: 6px; font-size: 13px; color: var(--muted); word-break: break-word; }}
    .probe-chips {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .chip {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 800;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      color: var(--muted);
    }}
    .chip.good {{ color: var(--good); border-color: rgba(57,217,138,0.28); }}
    .chip.mid {{ color: var(--mid); border-color: rgba(255,176,32,0.28); }}
    .chip.bad {{ color: var(--bad); border-color: rgba(255,123,114,0.28); }}
    .chip.stage {{ color: var(--accent); border-color: rgba(122,162,255,0.28); }}
    .svg-frame {{
      margin: 14px 0 12px;
      min-height: 150px;
      border-radius: 14px;
      background: #f8fafc;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      padding: 10px;
    }}
    .svg-frame svg {{ width: 100%; height: auto; max-width: 320px; display: block; }}
    .svg-frame.empty {{
      color: #475569;
      font-size: 13px;
      line-height: 1.5;
      text-align: center;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      border-radius: 14px;
      background: rgba(7, 10, 18, 0.9);
      color: #dce6ff;
      border: 1px solid rgba(255,255,255,0.08);
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
      max-height: 220px;
      overflow: auto;
    }}
    .good {{ color: var(--good); }}
    .mid {{ color: var(--mid); }}
    .bad {{ color: var(--bad); }}
    .na {{ color: var(--muted); }}
    a {{ color: #9fc0ff; }}
    code {{ color: #dce6ff; font-family: ui-monospace, monospace; }}
    ul {{ margin: 10px 0 0 18px; padding: 0; }}
    .compare-grid {{ grid-template-columns: 1fr 1fr; }}
    .compare-card {{ border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); border-radius: 18px; padding: 18px; }}
    @media (max-width: 1100px) {{
      .hero-grid, .cards, .compare-grid, .summary-strip, .gallery-grid {{ grid-template-columns: 1fr; }}
      .page {{ width: min(100vw - 20px, 1460px); }}
      .probe-head {{ flex-direction: column; }}
      .probe-chips {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">SVG Learning Comparison</div>
          <h1>Spec02 vs Spec03 vs Pure DSL</h1>
          <p>This page is the shortest honest story of the three lines. <strong>spec02</strong> shows that staged raw-SVG training can reach usable validity and some adherence. <strong>spec03</strong> shows that cleaner training mechanics alone do not rescue a brittle representation. <strong>the structured toy</strong> shows that a fixed symbolic DSL can teach atomic control much more directly.</p>
          <ul>
            <li><strong>spec02</strong>: strongest practical raw-SVG baseline so far.</li>
            <li><strong>spec03</strong>: pretrain-only line that currently fails before controllable generation emerges.</li>
            <li><strong>toy structured DSL</strong>: best intuition-building proof that symbolic control plus a renderer is the clean path upward.</li>
          </ul>
        </div>
        <div class="summary-strip">
          <div class="summary-card">
            <div class="k">Spec02 Best</div>
            <div class="v good">{_pct(spec02['best_adh'])}</div>
            <p>Best adherence at <code>{_html(spec02['best_phase'])}</code>.</p>
          </div>
          <div class="summary-card">
            <div class="k">Spec03 Latest</div>
            <div class="v bad">{_pct(spec03['latest_adh'])}</div>
            <p>Pretrain-only run with zero valid SVG and zero adherence.</p>
          </div>
          <div class="summary-card">
            <div class="k">Toy Exact</div>
            <div class="v mid">{toy['exact']}/{toy['probe_count']}</div>
            <p>Structured symbolic control with deterministic rendering.</p>
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h3>Run Reports</h3>
      <div class="cards">
        {_run_card(spec02)}
        {_run_card(spec03)}
        {_toy_card(toy)}
      </div>
    </section>

    <section class="panel">
      <h3>Actual Outputs</h3>
      <p>This section shows the concrete outputs behind the comparison. For raw-SVG runs, the page uses the selected evaluation phase output and extracts the generated SVG when it can. For the toy run, it shows the renderer output for each probe.</p>
      <div class="gallery-grid">
        {_probe_gallery(spec02)}
        {_probe_gallery(spec03)}
        {_probe_gallery(toy, toy=True)}
      </div>
    </section>

    <section class="panel">
      <h3>What The Comparison Means</h3>
      <div class="compare-grid">
        <div class="compare-card">
          <h4>Raw SVG Line</h4>
          <p><strong>spec02</strong> is the proof that the engine can train a raw SVG model to emit valid files and partially obey prompt intent. The downside is that tokenization and longer entangled outputs make the control story fragile.</p>
          <p><strong>spec03</strong> confirms the failure mode: if the representation contract is wrong, cleaner training and lower loss do not matter. The model still collapses on validity and adherence.</p>
        </div>
        <div class="compare-card">
          <h4>Symbolic DSL Line</h4>
          <p><strong>the toy structured run</strong> is simpler but much more legible. The model emits an IR like <code>[svg] [circle] [fill:red]</code>, and a renderer turns that into XML. That is why it is the right place to grow charts, text slots, cards, and composition.</p>
        </div>
      </div>
    </section>

    <section class="panel">
      <h3>Next Clean Expansion</h3>
      <ul>
        <li>Keep the structured toy tokenizer fixed and add symbolic stages for <code>[task:chart]</code>, <code>[chart:bar]</code>, <code>[chart:line]</code>, and text slots inside panels.</li>
        <li>Add layout tokens like <code>[layout:row-3]</code>, <code>[panel:a]</code>, and <code>[panel:b]</code> so the model learns composition rather than raw coordinates first.</li>
        <li>Use the same report pattern: prompt probes, rendered SVG, exact IR match, and holdout combinations.</li>
      </ul>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a comparison page for spec02, spec03, and the structured SVG toy")
    ap.add_argument(
        "--output",
        default=str(MODELS_ROOT / "svg_learning_comparison.html"),
        help="Output HTML path",
    )
    args = ap.parse_args()

    spec02 = _summarize_spec_run(
        SPEC02_RUN,
        label="Spec02",
        family="staged raw SVG",
        tokenizer="ASCII BPE over raw SVG rows and prompt tags.",
        dataset_story="Pretrain -> midtrain -> SFT over synthetic SVG rows and instruction-style chart prompts.",
        report_name="svg_training_report_card.html",
        probe_phase="best",
    )
    spec03 = _summarize_spec_run(
        SPEC03_RUN,
        label="Spec03",
        family="raw SVG pretrain only",
        tokenizer="ASCII BPE with protected control tokens, but the run itself only measured pretrain behavior.",
        dataset_story="Structural pretrain over longer SVG rows without the later staged control curriculum.",
        report_name="svg_training_report_card.html",
        probe_phase="latest",
    )
    toy = _summarize_toy_structured(TOY_RUN)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_build_html(spec02, spec03, toy), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
