#!/usr/bin/env python3
"""Build a single-page capability report for the spec04 structured SVG line."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from probe_report_adapters_v7 import apply_output_adapter


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_first(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    return None


def _find_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _parse_prompt_tags(prompt: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for token in str(prompt or "").split():
        if not (token.startswith("[") and token.endswith("]")):
            continue
        body = token[1:-1]
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        tags[key] = value
    return tags


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _read_stage_counts(run_dir: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for stage in ("pretrain", "midtrain", "sft"):
        stage_dir = run_dir / "dataset" / stage
        out[stage] = {
            split: sum(_count_lines(p) for p in sorted((stage_dir / split).glob("*")) if p.is_file())
            for split in ("train", "dev", "test")
        }
    return out


def _scene_samples(render_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = [
        ("single", "single"),
        ("pair", "pair-h"),
        ("pair", "pair-v"),
        ("label-card", "label-card"),
        ("badge", "badge"),
    ]
    out: list[dict[str, Any]] = []
    for kind, layout in wanted:
        for row in render_catalog:
            if row.get("kind") == kind and row.get("layout") == layout and row.get("split") == "train":
                out.append(row)
                break
    return out


def _probe_report_examples(probe_report: dict[str, Any]) -> list[dict[str, Any]]:
    results = probe_report.get("results")
    if not isinstance(results, list):
        return []
    picks: list[dict[str, Any]] = []
    counts = {"train": 0, "dev": 0, "test": 0}
    for row in results:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split") or "").strip()
        if split not in counts or counts[split] >= 2:
            continue
        picks.append(row)
        counts[split] += 1
        if all(v >= 2 for v in counts.values()):
            break
    return picks


def _stage_eval_probe_examples(
    latest_entry: dict[str, Any],
    render_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    probe_results = latest_entry.get("probe_results")
    if not isinstance(probe_results, list) or not probe_results:
        return []
    catalog_by_prompt = {
        str(row.get("prompt") or "").strip(): row
        for row in render_catalog
        if isinstance(row, dict) and str(row.get("prompt") or "").strip()
    }
    adapter_cfg = {
        "name": "text_renderer",
        "renderer": "structured_svg_atoms.v1",
        "stop_markers": ["[/svg]"],
        "preview_mime": "image/svg+xml",
    }
    selected: dict[str, dict[str, Any]] = {}
    order = ["single", "pair", "label-card", "badge", "ood"]
    for row in probe_results:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        tags = _parse_prompt_tags(prompt)
        probe_type = str(row.get("type") or "").strip()
        layout = tags.get("layout", "")
        family = "ood" if probe_type == "ood" else (
            "pair" if layout.startswith("pair") else layout or str(row.get("probe_id") or "")
        )
        if family not in order:
            continue
        samples = row.get("samples")
        if not isinstance(samples, list) or not samples:
            continue
        sample0 = samples[0] if isinstance(samples[0], dict) else {}
        response = str(sample0.get("response") or "").strip()
        if not response:
            continue
        adapted = apply_output_adapter("text_renderer", response, adapter_cfg)
        agg = row.get("agg") if isinstance(row.get("agg"), dict) else {}
        candidate = {
            "source": "stage_eval",
            "split": "eval",
            "label": family if family != "ood" else "ood-validity",
            "kind": str((catalog_by_prompt.get(prompt) or {}).get("kind") or family),
            "layout": layout or family,
            "prompt": prompt,
            "expected_svg": str((catalog_by_prompt.get(prompt) or {}).get("svg_xml") or ""),
            "actual_svg": str(adapted.get("materialized_output") or ""),
            "response_text": str(adapted.get("parsed_output") or response),
            "renderable": bool(adapted.get("renderable")),
            "valid_svg": bool(adapted.get("materialized_output")),
            "metrics": {
                "exact_match": None,
                "svg_exact_match": None,
                "adherence": agg.get("adherence"),
                "tag_adherence": agg.get("tag_adherence"),
                "valid_svg": agg.get("valid_svg"),
            },
            "tail_text": str(adapted.get("tail_text") or ""),
        }
        existing = selected.get(family)
        candidate_score = float(agg.get("adherence") or 0.0)
        existing_score = float(((existing or {}).get("metrics") or {}).get("adherence") or 0.0)
        if existing is None or candidate_score < existing_score:
            selected[family] = candidate
    return [selected[key] for key in order if key in selected]


def _architecture_summary(run_dir: Path) -> tuple[str, str]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return "-", "config.json missing"
    config = _load_json(config_path)
    arch = config.get("architecture") if isinstance(config, dict) and isinstance(config.get("architecture"), dict) else {}
    layers = arch.get("num_layers")
    embed = arch.get("embed_dim")
    hidden = arch.get("hidden_dim")
    heads = arch.get("num_heads")
    ctx = arch.get("context_len")
    vocab = arch.get("vocab_size")
    headline = " · ".join(
        part
        for part in (
            f"{layers}L" if layers is not None else "",
            f"d{embed}" if embed is not None else "",
            f"h{hidden}" if hidden is not None else "",
            f"heads {heads}" if heads is not None else "",
        )
        if part
    )
    detail = " · ".join(
        part
        for part in (
            f"ctx {ctx}" if ctx is not None else "",
            f"vocab {vocab}" if vocab is not None else "",
            str(arch.get("family") or "").strip(),
        )
        if part
    )
    return headline or "-", detail or "architecture unavailable"


def _diagnostic_notice(latest_eval: dict[str, Any], probe_report: dict[str, Any]) -> str:
    valid_svg = float(latest_eval.get("valid_svg_rate") or 0.0)
    adherence = float(latest_eval.get("adherence") or 0.0)
    prefix = float(latest_eval.get("prefix_integrity") or 0.0)
    split_summary = probe_report.get("split_summary") if isinstance(probe_report.get("split_summary"), list) else []
    exact_rates = [float(row.get("exact_rate") or 0.0) for row in split_summary if isinstance(row, dict)]
    if valid_svg >= 0.9 and adherence <= 0.25:
        return (
            "This run is mostly producing parseable SVG shells, but it is not reliably obeying the control tags. "
            "High `valid_svg_rate` and `ood_robustness` here do not mean task correctness."
        )
    if exact_rates and max(exact_rates) == 0.0:
        return "The sampled probe report is at 0% exact match across train/dev/test, so the model is not reproducing the target scenes even on seen prompts."
    if prefix == 0.0:
        return "All latest eval probes have prefix drift: decoding reaches a valid `[svg]` block, but it does not begin cleanly at the intended DSL boundary."
    return ""


def _render_stat_card(label: str, value: str, sub: str = "") -> str:
    return f"""
        <div class="stat-card">
          <div class="stat-label">{html.escape(label)}</div>
          <div class="stat-value">{html.escape(value)}</div>
          <div class="stat-sub">{html.escape(sub)}</div>
        </div>
    """


def _render_scene_card(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    svg_xml = str(row.get("svg_xml") or "")
    kind = str(row.get("kind") or "")
    layout = str(row.get("layout") or "")
    return f"""
        <div class="scene-card">
          <div class="scene-meta">{html.escape(kind)} · {html.escape(layout)}</div>
          <div class="scene-preview">{svg_xml}</div>
          <pre>{html.escape(prompt)}</pre>
        </div>
    """


def _render_probe_card(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt") or "")
    expected_svg = str(row.get("expected_svg") or row.get("expected_rendered_output") or "")
    actual_svg = str(row.get("actual_svg") or row.get("materialized_output") or row.get("rendered_svg") or "")
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    split = str(row.get("split") or "")
    label = str(row.get("label") or "")
    response = str(row.get("response_text") or "")
    renderable = bool(row.get("renderable"))
    valid_svg = bool(row.get("valid_svg"))
    adherence = metrics.get("adherence")
    tag_adherence = metrics.get("tag_adherence")
    tail_text = str(row.get("tail_text") or "")
    return f"""
        <div class="probe-card">
          <div class="probe-head">
            <div>
              <div class="scene-meta">{html.escape(split)} · {html.escape(label)}</div>
              <div class="probe-status">{'renderable' if renderable else 'not renderable'} · {'valid svg' if valid_svg else 'invalid svg'}</div>
            </div>
            <div class="probe-metrics">
              <span>exact {_fmt_pct(metrics.get('exact_match'))}</span>
              <span>svg {_fmt_pct(metrics.get('svg_exact_match'))}</span>
              <span>adh {_fmt_pct(adherence)}</span>
              <span>tag {_fmt_pct(tag_adherence)}</span>
            </div>
          </div>
          <pre>{html.escape(prompt)}</pre>
          <div class="probe-grid">
            <div>
              <div class="probe-label">Expected</div>
              <div class="probe-preview">{expected_svg}</div>
            </div>
            <div>
              <div class="probe-label">Generated</div>
              <div class="probe-preview">{actual_svg}</div>
            </div>
          </div>
          {f'<div class="probe-tail">tail drift after closing tag</div>' if tail_text else ''}
          <details>
            <summary>Generated tokens</summary>
            <pre>{html.escape(response)}</pre>
          </details>
        </div>
    """


def build_report(run_dir: Path) -> str:
    run_dir = run_dir.resolve()
    render_catalog_path = _find_existing(
        [
            run_dir / "dataset" / "manifests" / "generated" / "structured_atoms" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "dataset" / "tokenizer" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "spec04_workspace" / "manifests" / "generated" / "structured_atoms" / "spec04_structured_svg_atoms_render_catalog.json",
            run_dir / "spec04_workspace" / "tokenizer" / "spec04_structured_svg_atoms_render_catalog.json",
        ]
    )
    if render_catalog_path is None:
        raise SystemExit("spec04 render catalog not found under dataset/ or spec04_workspace/")
    render_catalog = _load_json(render_catalog_path)
    if not isinstance(render_catalog, list):
        raise SystemExit(f"expected JSON list render catalog: {render_catalog_path}")

    stage_counts = _read_stage_counts(run_dir)
    kind_counts = Counter(str(row.get("kind") or "-") for row in render_catalog if isinstance(row, dict))
    layout_counts = Counter(str(row.get("layout") or "-") for row in render_catalog if isinstance(row, dict))
    split_counts = Counter(str(row.get("split") or "-") for row in render_catalog if isinstance(row, dict))
    shape_counts = Counter()
    for row in render_catalog:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "")
        for token in ("[shape:circle]", "[shape:rect]", "[shape:triangle]", "[shape2:circle]", "[shape2:rect]", "[shape2:triangle]"):
            if token in prompt:
                shape_counts[token.replace("[", "").replace("]", "")] += 1

    training_path = _find_first(
        run_dir,
        [
            "train_structured_svg_scenes_stage_a.json",
            "train_structured_svg_atoms_stage_a.json",
        ],
    )
    training = _load_json(training_path) if training_path else {}
    ck_loss = training.get("ck_loss") if isinstance(training.get("ck_loss"), dict) else {}

    stage_eval_path = run_dir / "stage_eval_matrix.json"
    stage_eval = _load_json(stage_eval_path) if stage_eval_path.exists() else {}
    entries = stage_eval.get("entries") if isinstance(stage_eval.get("entries"), list) else []
    latest_entry = entries[-1] if entries and isinstance(entries[-1], dict) else {}
    latest_eval = entries[-1]["metrics"] if entries and isinstance(entries[-1], dict) and isinstance(entries[-1].get("metrics"), dict) else {}

    parity_path = run_dir / "training_parity_regimen_latest.json"
    parity = _load_json(parity_path) if parity_path.exists() else {}
    parity_summary = parity.get("summary") if isinstance(parity.get("summary"), dict) else {}

    probe_path = _find_first(run_dir, ["spec04_probe_report.json", "probe_report.json"])
    probe_report = _load_json(probe_path) if probe_path else {}
    probe_summary = probe_report.get("split_summary") if isinstance(probe_report.get("split_summary"), list) else []

    scene_cards = "".join(_render_scene_card(row) for row in _scene_samples(render_catalog))
    observed_rows = _stage_eval_probe_examples(latest_entry, render_catalog) or _probe_report_examples(probe_report)
    probe_cards = "".join(_render_probe_card(row) for row in observed_rows)

    stage_rows = "".join(
        f"<tr><td>{html.escape(stage)}</td><td>{counts['train']}</td><td>{counts['dev']}</td><td>{counts['test']}</td></tr>"
        for stage, counts in stage_counts.items()
    )
    probe_rows = "".join(
        f"<tr><td>{html.escape(str(row.get('split') or ''))}</td><td>{row.get('count', 0)}</td><td>{_fmt_pct(row.get('renderable_rate'))}</td><td>{_fmt_pct(row.get('exact_rate'))}</td><td>{_fmt_pct(row.get('svg_exact_rate'))}</td></tr>"
        for row in probe_summary
        if isinstance(row, dict)
    )

    scene_mix = " · ".join(f"{k}:{v}" for k, v in sorted(kind_counts.items()))
    layout_mix = " · ".join(f"{k}:{v}" for k, v in sorted(layout_counts.items()))
    shape_mix = " · ".join(f"{k}:{v}" for k, v in sorted(shape_counts.items()))
    model_headline, model_detail = _architecture_summary(run_dir)
    diagnosis = _diagnostic_notice(latest_eval, probe_report)

    stats = "".join(
        [
            _render_stat_card("Parity", f"{parity_summary.get('passed_stages', 0)}/{parity_summary.get('total_stages', 0)}", "A-F gates"),
            _render_stat_card("Final CK Loss", _fmt_float(ck_loss.get("final")), f"min {_fmt_float(ck_loss.get('min'))}"),
            _render_stat_card("SVG Valid", _fmt_pct(latest_eval.get("valid_svg_rate")), "stage eval"),
            _render_stat_card("Closure", _fmt_pct(latest_eval.get("closure_success_rate")), "stage eval"),
            _render_stat_card("OOD Validity", _fmt_pct(latest_eval.get("ood_robustness")), "valid SVG under missing tags"),
            _render_stat_card("Task Adherence", _fmt_pct(latest_eval.get("adherence")), "stage eval"),
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spec04 Capability Report</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffdf9;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d6d3d1;
      --accent: #b45309;
      --accent2: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, 'Iowan Old Style', serif; background: linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%); color: var(--ink); }}
    a {{ color: #0f766e; }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 28px 24px 48px; }}
    .hero {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 18px; align-items: start; margin-bottom: 18px; }}
    .hero-main, .hero-side, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }}
    h1, h2, h3 {{ margin: 0 0 10px; font-family: 'Avenir Next Condensed', 'Franklin Gothic Medium', sans-serif; letter-spacing: 0.02em; }}
    h1 {{ font-size: 2.2rem; line-height: 1; }}
    h2 {{ font-size: 1.35rem; margin-top: 10px; }}
    .eyebrow {{ font: 700 0.82rem/1.2 'Avenir Next Condensed', sans-serif; color: var(--accent); text-transform: uppercase; letter-spacing: 0.14em; }}
    .hero-copy {{ color: var(--muted); line-height: 1.55; }}
    .notice {{ margin-top: 16px; background: #fff7ed; border: 1px solid #fdba74; color: #9a3412; border-radius: 14px; padding: 12px 14px; font-size: 0.95rem; line-height: 1.5; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .stat-card {{ background: #fcfbf7; border: 1px solid var(--line); border-radius: 14px; padding: 12px; }}
    .stat-label {{ font: 700 0.76rem/1.2 'Avenir Next Condensed', sans-serif; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }}
    .stat-value {{ font: 700 1.35rem/1.1 'Avenir Next Condensed', sans-serif; color: var(--accent2); margin: 6px 0 4px; }}
    .stat-sub {{ font-size: 0.84rem; color: var(--muted); }}
    .linklist a {{ display: block; margin-bottom: 6px; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ font: 700 0.78rem/1.2 'Avenir Next Condensed', sans-serif; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }}
    .mix-line {{ color: var(--muted); line-height: 1.6; }}
    .scene-grid, .probe-stack {{ display: grid; gap: 14px; }}
    .scene-grid {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .scene-card, .probe-card {{ background: #fcfbf7; border: 1px solid var(--line); border-radius: 14px; padding: 12px; }}
    .scene-meta, .probe-label, .probe-status {{ font: 700 0.78rem/1.2 'Avenir Next Condensed', sans-serif; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
    .scene-preview, .probe-preview {{ background: white; border: 1px solid var(--line); border-radius: 10px; padding: 8px; min-height: 154px; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
    .scene-card pre, .probe-card pre {{ white-space: pre-wrap; word-break: break-word; font: 0.8rem/1.45 'JetBrains Mono', monospace; background: #f6f3ed; border-radius: 10px; padding: 10px; overflow: auto; }}
    .probe-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 8px; }}
    .probe-metrics {{ display: flex; gap: 8px; flex-wrap: wrap; font: 700 0.74rem/1.2 'Avenir Next Condensed', sans-serif; color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em; }}
    .probe-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px; }}
    .probe-tail {{ color: #9a3412; font-size: 0.84rem; margin-bottom: 10px; }}
    details summary {{ cursor: pointer; font: 700 0.78rem/1.2 'Avenir Next Condensed', sans-serif; color: var(--accent2); margin-bottom: 8px; }}
    @media (max-width: 920px) {{
      .hero, .grid2, .probe-grid, .stats {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-main">
        <div class="eyebrow">Spec04 Capability Report</div>
        <h1>{html.escape(run_dir.name)}</h1>
        <p class="hero-copy">
          This page combines the rich structured-scene corpus, the latest stage eval, the parity gate summary,
          and sampled train/dev/test probe outputs into one operator-facing proof page.
        </p>
        <p class="hero-copy"><strong>Actual model:</strong> {html.escape(model_headline)} <span style="color: var(--muted);">({html.escape(model_detail)})</span></p>
        {f'<div class="notice">{html.escape(diagnosis)}</div>' if diagnosis else ''}
        <div class="stats">{stats}</div>
      </div>
      <div class="hero-side">
        <div class="eyebrow">Linked Artifacts</div>
        <div class="linklist">
          <a href="config.json">config.json</a>
          <a href="dataset_viewer.html">dataset_viewer.html</a>
          <a href="ir_report.html">ir_report.html</a>
          <a href="svg_training_report_card.html">svg_training_report_card.html</a>
          <a href="spec04_probe_report.html">spec04_probe_report.html</a>
          <a href="stage_eval_matrix.json">stage_eval_matrix.json</a>
          <a href="training_parity_regimen_latest.json">training_parity_regimen_latest.json</a>
        </div>
      </div>
    </section>

    <section class="grid2">
      <div class="panel">
        <div class="eyebrow">Stage Splits</div>
        <h2>Dataset Rows By Stage</h2>
        <table>
          <thead><tr><th>Stage</th><th>Train</th><th>Dev</th><th>Test</th></tr></thead>
          <tbody>{stage_rows}</tbody>
        </table>
      </div>
      <div class="panel">
        <div class="eyebrow">Corpus Mix</div>
        <h2>Scene Coverage</h2>
        <p class="mix-line"><strong>catalog rows</strong>: {sum(split_counts.values())} · <strong>splits</strong>: {" · ".join(f"{k}:{v}" for k, v in sorted(split_counts.items()))}</p>
        <p class="mix-line"><strong>kinds</strong>: {html.escape(scene_mix)}</p>
        <p class="mix-line"><strong>layouts</strong>: {html.escape(layout_mix)}</p>
        <p class="mix-line"><strong>shape tags</strong>: {html.escape(shape_mix)}</p>
      </div>
    </section>

    <section class="panel" style="margin-top:18px;">
      <div class="eyebrow">Expected Capability</div>
      <h2>Complex Scene Families In The Dataset</h2>
      <div class="scene-grid">{scene_cards}</div>
    </section>

    <section class="grid2">
      <div class="panel">
        <div class="eyebrow">Probe Splits</div>
        <h2>Train / Dev / Test Probe Summary</h2>
        <table>
          <thead><tr><th>Split</th><th>Cases</th><th>Renderable</th><th>Exact</th><th>SVG Exact</th></tr></thead>
          <tbody>{probe_rows}</tbody>
        </table>
      </div>
      <div class="panel">
        <div class="eyebrow">Current Model</div>
        <h2>Latest Stage Eval</h2>
        <table>
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>valid_svg_rate</td><td>{_fmt_pct(latest_eval.get('valid_svg_rate'))}</td></tr>
            <tr><td>closure_success_rate</td><td>{_fmt_pct(latest_eval.get('closure_success_rate'))}</td></tr>
            <tr><td>ood_validity</td><td>{_fmt_pct(latest_eval.get('ood_robustness'))}</td></tr>
            <tr><td>adherence</td><td>{_fmt_pct(latest_eval.get('adherence'))}</td></tr>
            <tr><td>tag_adherence</td><td>{_fmt_pct(latest_eval.get('tag_adherence'))}</td></tr>
            <tr><td>prefix_integrity</td><td>{_fmt_pct(latest_eval.get('prefix_integrity'))}</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel" style="margin-top:18px;">
      <div class="eyebrow">Observed Behavior</div>
      <h2>Balanced Eval Samples</h2>
      <div class="probe-stack">{probe_cards}</div>
    </section>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a consolidated spec04 capability report for a run directory.")
    ap.add_argument("--run", required=True, type=Path, help="Run directory containing dataset + eval artifacts")
    ap.add_argument("--output", required=True, type=Path, help="Output HTML path")
    args = ap.parse_args()

    html_doc = build_report(args.run.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
