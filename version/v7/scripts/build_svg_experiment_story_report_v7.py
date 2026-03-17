#!/usr/bin/env python3
"""Build a one-page presentation report for the SVG training storyline."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/antshiv")
WORKSPACE_ROOT = ROOT / "Workspace" / "C-Kernel-Engine"
MODELS_ROOT = ROOT / ".cache" / "ck-engine-v7" / "models"
TRAIN_ROOT = MODELS_ROOT / "train"

SCREENSHOTS = {
    "spec02_card": WORKSPACE_ROOT / "Screenshots" / "Screenshot 2026-03-10 at 09-48-28 SVG Training Report Card.png",
    "learning_comparison": WORKSPACE_ROOT / "Screenshots" / "Screenshot 2026-03-10 at 09-48-12 SVG Learning Comparison.png",
    "toy_probe": WORKSPACE_ROOT / "Screenshots" / "Screenshot 2026-03-10 at 09-48-03 Toy Structured SVG Probe Report.png",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{payload}"


def _svg_uri(svg_text: str | None) -> str:
    if not svg_text:
        return ""
    payload = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def _fmt_loss(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _metric_tone(value: float | None, good: float = 0.75, warn: float = 0.35) -> str:
    if value is None:
        return "neutral"
    if value >= good:
        return "good"
    if value >= warn:
        return "warn"
    return "bad"


def _metric_chip(label: str, value: str, tone: str = "neutral") -> str:
    return (
        f'<div class="metric-chip metric-{tone}">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<span class="metric-value">{html.escape(value)}</span>'
        "</div>"
    )


def _link_button(label: str, port: int, path: str) -> str:
    return (
        f'<a class="link-button" data-port="{port}" data-path="{html.escape(path)}" href="#">'
        f"{html.escape(label)}</a>"
    )


def _route_card(index: int, title: str, note: str, port: int, path: str) -> str:
    return (
        '<div class="route-card">'
        f'<div class="route-index">{index}</div>'
        f'<div class="route-title">{html.escape(title)}</div>'
        f'<div class="route-note">{html.escape(note)}</div>'
        f'{_link_button("Open", port, path)}'
        "</div>"
    )


def _visual_anchor(title: str, caption: str, image_uri: str, port: int, path: str) -> str:
    return (
        '<article class="visual-anchor">'
        f'<div class="visual-shot"><img src="{image_uri}" alt="{html.escape(title)}" /></div>'
        '<div class="visual-copy">'
        f'<div class="visual-title">{html.escape(title)}</div>'
        f'<p>{html.escape(caption)}</p>'
        f'{_link_button("Open Source Page", port, path)}'
        "</div>"
        "</article>"
    )


def _experiment_card(
    eyebrow: str,
    title: str,
    strap: str,
    summary: str,
    metrics: dict[str, Any],
    worked: list[str],
    failed: list[str],
    takeaway: str,
    links: list[tuple[str, int, str]],
) -> str:
    chips = [
        _metric_chip("Valid SVG", _fmt_pct(metrics.get("valid_svg_rate")), _metric_tone(metrics.get("valid_svg_rate"))),
        _metric_chip("Closure", _fmt_pct(metrics.get("closure_success_rate")), _metric_tone(metrics.get("closure_success_rate"))),
        _metric_chip("Adherence", _fmt_pct(metrics.get("adherence")), _metric_tone(metrics.get("adherence"), good=0.5, warn=0.15)),
        _metric_chip("OOD", _fmt_pct(metrics.get("ood_robustness")), _metric_tone(metrics.get("ood_robustness"), good=0.75, warn=0.3)),
        _metric_chip("Tag Adh", _fmt_pct(metrics.get("tag_adherence")), _metric_tone(metrics.get("tag_adherence"), good=0.5, warn=0.15)),
        _metric_chip("Final Loss", _fmt_loss(metrics.get("final_loss")), "neutral"),
    ]
    worked_html = "".join(f"<li>{html.escape(item)}</li>" for item in worked)
    failed_html = "".join(f"<li>{html.escape(item)}</li>" for item in failed)
    link_html = "".join(_link_button(label, port, path) for label, port, path in links)
    return (
        '<section class="experiment-card">'
        f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<h2>{html.escape(title)}</h2>'
        f'<div class="strap">{html.escape(strap)}</div>'
        f'<p class="summary">{html.escape(summary)}</p>'
        f'<div class="metric-grid">{"".join(chips)}</div>'
        '<div class="work-grid">'
        '<div class="work-block"><h3>What Worked</h3><ul>'
        f"{worked_html}</ul></div>"
        '<div class="work-block"><h3>What Failed</h3><ul>'
        f"{failed_html}</ul></div>"
        "</div>"
        f'<div class="takeaway">{html.escape(takeaway)}</div>'
        f'<div class="link-row">{link_html}</div>'
        "</section>"
    )


def _probe_card(
    title: str,
    prompt: str,
    left_label: str,
    left_svg: str | None,
    right_label: str,
    right_svg: str | None,
    status: str,
    note: str,
) -> str:
    left_uri = _svg_uri(left_svg)
    right_uri = _svg_uri(right_svg)
    return (
        '<article class="probe-card">'
        f'<div class="probe-head"><div class="probe-title">{html.escape(title)}</div><div class="probe-status">{html.escape(status)}</div></div>'
        f'<div class="probe-prompt">{html.escape(prompt)}</div>'
        '<div class="probe-grid">'
        '<div class="probe-pane">'
        f'<div class="probe-pane-title">{html.escape(left_label)}</div>'
        f'<div class="probe-svg-box"><img src="{left_uri}" alt="{html.escape(left_label)}" /></div>'
        "</div>"
        '<div class="probe-pane">'
        f'<div class="probe-pane-title">{html.escape(right_label)}</div>'
        f'<div class="probe-svg-box"><img src="{right_uri}" alt="{html.escape(right_label)}" /></div>'
        "</div>"
        "</div>"
        f'<div class="probe-note">{html.escape(note)}</div>'
        "</article>"
    )


def _stage_card(phase_label: str, metrics: dict[str, Any], final_loss: float, note: str) -> str:
    return (
        '<div class="stage-card">'
        f'<div class="stage-label">{html.escape(phase_label)}</div>'
        f'<div class="stage-loss">loss {_fmt_loss(final_loss)}</div>'
        '<div class="stage-metrics">'
        f'{_metric_chip("Valid", _fmt_pct(metrics.get("valid_svg_rate")), _metric_tone(metrics.get("valid_svg_rate")))}'
        f'{_metric_chip("Adherence", _fmt_pct(metrics.get("adherence")), _metric_tone(metrics.get("adherence"), good=0.5, warn=0.15))}'
        f'{_metric_chip("OOD", _fmt_pct(metrics.get("ood_robustness")), _metric_tone(metrics.get("ood_robustness"), good=0.75, warn=0.3))}'
        "</div>"
        f'<div class="stage-note">{html.escape(note)}</div>'
        "</div>"
    )


def _build_page() -> str:
    spec02 = _load_json(TRAIN_ROOT / "svg_l16_d128_h512_v1024_ctx512_spec02" / "stage_eval_matrix.json")["entries"][-1]
    spec03 = _load_json(TRAIN_ROOT / "svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512" / "stage_eval_matrix.json")["entries"][-1]
    toy_structured = _load_json(TRAIN_ROOT / "toy_svg_structured_atoms_ctx512_d64_h128" / "stage_eval_matrix.json")["entries"][-1]
    toy_probe_results = _load_json(TRAIN_ROOT / "toy_svg_structured_atoms_ctx512_d64_h128" / "toy_svg_structured_probe_report.json")["results"]
    spec04_entries = _load_json(TRAIN_ROOT / "spec04_structured_scenes_ctx512_d64_h128_v224" / "stage_eval_matrix.json")["entries"]
    spec04_latest = spec04_entries[-1]
    spec04_best = next((entry for entry in spec04_entries if entry["phase_label"] == "sft_2"), spec04_latest)
    spec04_probe_results = _load_json(
        TRAIN_ROOT / "spec04_structured_scenes_ctx512_d64_h128_v224" / "spec04_probe_report.json"
    )["results"]
    spec04_manifest = _load_json(
        TRAIN_ROOT
        / "spec04_structured_scenes_ctx512_d64_h128_v224"
        / "dataset"
        / "manifests"
        / "spec04_structured_svg_atoms_workspace_manifest.json"
    )
    toy_parity = _load_json(TRAIN_ROOT / "toy_svg_structured_atoms_ctx512_d64_h128" / "training_parity_regimen_latest.json")[
        "summary"
    ]
    spec04_parity = _load_json(TRAIN_ROOT / "spec04_structured_scenes_ctx512_d64_h128_v224" / "training_parity_regimen_latest.json")[
        "summary"
    ]

    anchors = [
        _visual_anchor(
            "Spec02 Training Report Card",
            "This is the visual style and density people already recognize: stage timeline, metrics, prompt-by-prompt results, and dataset composition.",
            _data_uri(SCREENSHOTS["spec02_card"]),
            7001,
            "svg_training_report_card.html",
        ),
        _visual_anchor(
            "SVG Learning Comparison",
            "This is the cross-line view that lets you explain why we left some paths behind and doubled down on the representation-first path.",
            _data_uri(SCREENSHOTS["learning_comparison"]),
            7002,
            "svg_learning_comparison_latest.html",
        ),
        _visual_anchor(
            "Toy Structured Probe Report",
            "This is the first place the model output looked clean enough to reason about. It made the later Spec04 workflow credible.",
            _data_uri(SCREENSHOTS["toy_probe"]),
            7004,
            "toy_svg_structured_probe_report.html",
        ),
    ]

    route_cards = [
        _route_card(1, "IR Hub", "Open with the run hub to show experiment discipline and cache-local organization.", 7003, "ir_hub.html"),
        _route_card(2, "Spec02", "Recap the large rich SVG line from last week.", 7001, "svg_training_report_card.html"),
        _route_card(3, "Learning Comparison", "Show the whole progression in one visual pass.", 7002, "svg_learning_comparison_latest.html"),
        _route_card(4, "Spec03", "Explain the tokenizer/bootstrap miss clearly and quickly.", 7003, "train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512/svg_training_report_card.html"),
        _route_card(5, "Toy Reset", "Show the structured toy probe report as the turning point.", 7004, "toy_svg_structured_probe_report.html"),
        _route_card(6, "Spec04", "Finish with the current workflow and the current model state.", 7005, "spec04_capability_report.html"),
    ]

    experiments = [
        _experiment_card(
            eyebrow="Spec02",
            title="Scale-First SVG Training",
            strap="Big visual ambition, strong structure, incomplete control.",
            summary=(
                "Spec02 proved the engine could support a large SVG task family and keep valid SVG generation alive across multiple stages. "
                "It looked impressive, but it also showed that a richer corpus does not automatically produce reliable semantic control."
            ),
            metrics={**spec02["metrics"], "final_loss": spec02["final_loss"]},
            worked=[
                "Valid SVG and closure stayed strong.",
                "The line produced presentation-friendly charts, cards, and infographic motifs.",
                "It established the baseline operator view for training reports.",
            ],
            failed=[
                "Adherence was uneven across later stages.",
                "Later fine-tuning could regress despite looking numerically healthier.",
                "The visual domain was richer than the control surface.",
            ],
            takeaway="Spec02 told us the model could draw, but not yet obey.",
            links=[
                ("Training Card", 7001, "svg_training_report_card.html"),
                ("Dataset Viewer", 7001, "dataset_viewer.html"),
                ("IR Report", 7001, "ir_report.html"),
            ],
        ),
        _experiment_card(
            eyebrow="Spec03",
            title="Tokenizer / Bootstrap Correction Attempt",
            strap="Necessary experiment, wrong model path.",
            summary=(
                "Spec03 tried to repair the problem at the tokenizer and bootstrap layer. It was useful as diagnosis, but as a training line "
                "it failed hard enough that it justified resetting the problem into a smaller structured DSL."
            ),
            metrics={**spec03["metrics"], "final_loss": spec03["final_loss"]},
            worked=[
                "It made hidden prompt-contract and tokenizer assumptions visible.",
                "It gave hard failure evidence instead of vague discomfort.",
                "It sharpened the next experiment rather than leaving us guessing.",
            ],
            failed=[
                "No meaningful valid SVG generation.",
                "High repetition and no adherence recovery.",
                "Too much corrective complexity before stable representation.",
            ],
            takeaway="Spec03 was valuable because it failed clearly.",
            links=[
                ("Training Card", 7003, "train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512/svg_training_report_card.html"),
                ("Dataset Viewer", 7003, "train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512/dataset_viewer.html"),
                ("IR Report", 7003, "train/svg_spec03_bootstrap_l16_d128_h512_v2048_ctx512/ir_report.html"),
                ("Comparison", 7002, "svg_learning_comparison_latest.html"),
            ],
        ),
        _experiment_card(
            eyebrow="Toy Reset",
            title="Structured Toy SVG DSL",
            strap="Smaller problem, stronger signal.",
            summary=(
                "The toy structured line is where the project became legible again. The DSL was simple enough to inspect, parity was strict, "
                "and the probe report made successes and failures visually obvious."
            ),
            metrics={**toy_structured["metrics"], "final_loss": toy_structured["final_loss"]},
            worked=[
                f"Full A-F parity passed ({toy_parity['passed_stages']}/{toy_parity['total_stages']}).",
                "Simple shapes and colors became testable at prompt level.",
                "The line created the lesson set that Spec04 now uses.",
            ],
            failed=[
                "The capability ceiling stayed intentionally low.",
                "It was a training/debugging line, not a final product line.",
                "Representation improved faster than breadth.",
            ],
            takeaway="The toy line was the first honest recovery path, not a side experiment.",
            links=[
                ("Probe Report", 7004, "toy_svg_structured_probe_report.html"),
                ("Training Card", 7004, "svg_training_report_card.html"),
                ("Dataset Viewer", 7004, "dataset_viewer.html"),
                ("IR Report", 7004, "ir_report.html"),
            ],
        ),
        _experiment_card(
            eyebrow="Spec04",
            title="Operationalized Structured Training Workflow",
            strap="The strongest workflow line so far, with model quality still catching up.",
            summary=(
                "Spec04 carries the toy DSL into a real cache-backed workflow: split-aware dataset staging, parity, report generation, "
                "probe pages, dataset viewer, and IR visualizer all live under one run folder."
            ),
            metrics={**spec04_latest["metrics"], "final_loss": spec04_latest["final_loss"]},
            worked=[
                f"Parity still passes end to end ({spec04_parity['passed_stages']}/{spec04_parity['total_stages']}).",
                "Latest stages keep valid SVG, closure, and OOD robustness at 100%.",
                "The operator experience is coherent enough to present cleanly.",
            ],
            failed=[
                "Semantic adherence remains low.",
                "Later rebalancing preserved structure more than it improved control.",
                "The workflow is ahead of the model semantics.",
            ],
            takeaway=(
                f"Best semantic hit was {spec04_best['phase_label']} at {_fmt_pct(spec04_best['metrics']['adherence'])} adherence. "
                f"Latest {spec04_latest['phase_label']} is structurally strong but still only {_fmt_pct(spec04_latest['metrics']['adherence'])} adherence."
            ),
            links=[
                ("Capability Report", 7005, "spec04_capability_report.html"),
                ("Probe Report", 7005, "spec04_probe_report.html"),
                ("Training Card", 7005, "svg_training_report_card.html"),
                ("Dataset Viewer", 7005, "dataset_viewer.html"),
                ("IR Report", 7005, "ir_report.html"),
            ],
        ),
    ]

    toy_samples = [
        _probe_card(
            title=row["label"],
            prompt=row["prompt"],
            left_label="Expected",
            left_svg=row.get("expected_svg"),
            right_label="Model",
            right_svg=row.get("rendered_svg"),
            status="Exact match" if row.get("exact_match") else "Renderable",
            note="Toy structured SVG made simple prompt-to-shape control visible and inspectable.",
        )
        for row in toy_probe_results[:4]
    ]
    spec04_samples = [
        _probe_card(
            title=row["label"],
            prompt=row["prompt"],
            left_label="Expected",
            left_svg=row.get("expected_rendered_output"),
            right_label="Current model",
            right_svg=row.get("materialized_output"),
            status="Valid but drifting" if not row.get("exact_match") else "Exact match",
            note="This is the Spec04 story today: the SVG is usually valid, but the intended scene often drifts.",
        )
        for row in spec04_probe_results[:4]
    ]

    stage_notes = {
        "pretrain_2": "The richer structured scenes line becomes trainable, but quality is still mixed.",
        "midtrain_1": "Complex composition training improves structural validity quickly.",
        "sft_1": "Instruction-only SFT caused control-surface forgetting.",
        "sft_2": "Mixed DSL plus instruction SFT recovers structural quality and improves semantics somewhat.",
        "midtrain_2": "Rebalanced midtrain preserves structure and stabilizes the line.",
        "sft_3": "Rebalanced SFT keeps SVG quality high, but semantics remain the main gap.",
    }
    spec04_timeline = [
        _stage_card(entry["phase_label"], entry["metrics"], entry["final_loss"], stage_notes.get(entry["phase_label"], ""))
        for entry in spec04_entries
    ]

    spec04_counts = spec04_manifest["stages"]
    source_counts = spec04_manifest["source_counts"]

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SVG Training Story</title>
  <style>
    :root {{
      --bg: #07131d;
      --panel: #0d1d2a;
      --panel-soft: #132635;
      --panel-raise: #172d3d;
      --line: rgba(159, 193, 216, 0.12);
      --ink: #ecf5fb;
      --muted: #94a9bb;
      --accent: #4ec2c9;
      --accent-2: #f38a5d;
      --green: #47c780;
      --amber: #e4ae47;
      --red: #e26c61;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(78, 194, 201, 0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(243, 138, 93, 0.16), transparent 28%),
        linear-gradient(180deg, #061019 0%, #091723 100%);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    .page {{
      width: min(1380px, calc(100vw - 40px));
      margin: 24px auto 64px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 18px;
    }}
    .hero-card, .aside-card, .experiment-card, .visual-anchor, .gallery-card, .band-card {{
      background: linear-gradient(180deg, rgba(18, 34, 47, 0.95) 0%, rgba(12, 27, 38, 0.98) 100%);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }}
    .hero-card {{
      padding: 30px 32px 28px;
    }}
    .hero-kicker {{
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.24em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 12px;
    }}
    .hero-card h1 {{
      margin: 0;
      font-size: clamp(42px, 5vw, 70px);
      line-height: 0.95;
      letter-spacing: -0.05em;
    }}
    .hero-summary {{
      margin: 16px 0 0;
      font-size: 19px;
      color: #bfd0dc;
      max-width: 920px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .hero-stat {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 20px;
      padding: 16px;
    }}
    .hero-stat strong {{
      display: block;
      font-size: 24px;
      margin-bottom: 4px;
    }}
    .hero-stat span {{
      color: var(--muted);
      font-size: 14px;
    }}
    .aside-card {{
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .aside-card h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .aside-card p {{
      margin: 0;
      color: #bdd1df;
    }}
    .section-title {{
      margin: 30px 0 14px;
      font-size: 13px;
      letter-spacing: 0.20em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 800;
    }}
    .link-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .link-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 0 16px;
      border-radius: 999px;
      color: white;
      text-decoration: none;
      font-weight: 800;
      background: linear-gradient(135deg, var(--accent) 0%, #2c95bb 100%);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 12px 30px rgba(29, 102, 126, 0.28);
    }}
    .route-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
    }}
    .route-card {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 22px;
      padding: 16px;
    }}
    .route-index {{
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .route-title {{
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 6px;
    }}
    .route-note {{
      color: var(--muted);
      font-size: 14px;
      min-height: 64px;
      margin-bottom: 12px;
    }}
    .visual-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .visual-anchor {{
      overflow: hidden;
    }}
    .visual-shot {{
      background: #07131d;
      border-bottom: 1px solid var(--line);
      padding: 14px;
    }}
    .visual-shot img {{
      width: 100%;
      display: block;
      border-radius: 16px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
    }}
    .visual-copy {{
      padding: 18px 20px 20px;
    }}
    .visual-title {{
      font-size: 20px;
      font-weight: 800;
      margin-bottom: 8px;
    }}
    .visual-copy p {{
      margin: 0;
      color: var(--muted);
    }}
    .story-grid {{
      display: grid;
      gap: 16px;
    }}
    .experiment-card {{
      padding: 24px 26px;
    }}
    .eyebrow {{
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 10px;
    }}
    .experiment-card h2 {{
      margin: 0;
      font-size: 32px;
      line-height: 1.02;
      letter-spacing: -0.04em;
    }}
    .strap {{
      margin-top: 6px;
      color: #c7d7e4;
      font-size: 17px;
    }}
    .summary {{
      margin: 16px 0 0;
      color: #c0d0dc;
      max-width: 1040px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .metric-chip {{
      border-radius: 18px;
      padding: 12px 14px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
    }}
    .metric-good {{ background: rgba(71, 199, 128, 0.14); }}
    .metric-warn {{ background: rgba(228, 174, 71, 0.14); }}
    .metric-bad {{ background: rgba(226, 108, 97, 0.14); }}
    .metric-label {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .metric-value {{
      display: block;
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .work-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 18px;
    }}
    .work-block {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 22px;
      padding: 18px 20px;
    }}
    .work-block h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .work-block ul {{
      margin: 0;
      padding-left: 18px;
      color: #c2d2de;
    }}
    .work-block li + li {{
      margin-top: 8px;
    }}
    .takeaway {{
      margin-top: 16px;
      padding: 14px 18px;
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(78, 194, 201, 0.16), rgba(243, 138, 93, 0.12));
      color: #e6f6ff;
      font-weight: 600;
    }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .gallery-card {{
      padding: 20px;
    }}
    .gallery-card h3 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    .gallery-card p {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .probe-grid {{
      display: grid;
      gap: 12px;
    }}
    .probe-card {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 22px;
      padding: 16px;
    }}
    .probe-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .probe-title {{
      font-size: 17px;
      font-weight: 800;
    }}
    .probe-status {{
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }}
    .probe-prompt {{
      color: var(--muted);
      font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      margin-bottom: 12px;
      word-break: break-word;
    }}
    .probe-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .probe-pane {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 18px;
      padding: 12px;
    }}
    .probe-pane-title {{
      color: #d8e7f3;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .probe-svg-box {{
      min-height: 168px;
      border-radius: 14px;
      background: #edf2f7;
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    .probe-svg-box img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .probe-note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .timeline-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .stage-card {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 22px;
      padding: 18px;
    }}
    .stage-label {{
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 4px;
    }}
    .stage-loss {{
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .stage-metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .stage-note {{
      margin-top: 12px;
      color: #c0cfdb;
      font-size: 14px;
    }}
    .spec04-band {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
      margin-top: 18px;
    }}
    .band-card {{
      padding: 22px 24px;
    }}
    .band-card h3 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .band-card p {{
      color: #c2d2de;
    }}
    .count-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .count-box {{
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.05);
    }}
    .count-box strong {{
      display: block;
      font-size: 24px;
    }}
    .count-box span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .footer-note {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 1160px) {{
      .hero,
      .spec04-band,
      .gallery-grid,
      .visual-grid {{
        grid-template-columns: 1fr;
      }}
      .route-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .timeline-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .metric-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 760px) {{
      .page {{
        width: min(100vw - 20px, 1380px);
        margin: 14px auto 48px;
      }}
      .hero-grid,
      .metric-grid,
      .work-grid,
      .probe-grid,
      .timeline-grid,
      .count-grid,
      .stage-metrics,
      .route-grid {{
        grid-template-columns: 1fr;
      }}
      .hero-card,
      .aside-card,
      .experiment-card,
      .visual-anchor,
      .gallery-card,
      .band-card {{
        border-radius: 22px;
      }}
      .hero-card {{
        padding: 24px 20px;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-card">
        <div class="hero-kicker">C-Kernel Engine v7</div>
        <h1>SVG Training Story</h1>
        <p class="hero-summary">
          A presentation-first summary of the training line only: <strong>Spec02</strong>, <strong>Spec03</strong>,
          <strong>Toy Structured SVG</strong>, and <strong>Spec04</strong>. The page focuses on experiments, what improved,
          what failed, and what the visuals actually looked like.
        </p>
        <div class="hero-grid">
          <div class="hero-stat">
            <strong>Spec02</strong>
            <span>Rich SVG training proved the engine could draw, but not reliably obey.</span>
          </div>
          <div class="hero-stat">
            <strong>Spec03</strong>
            <span>The bootstrap/tokenizer correction attempt failed clearly enough to force a reset.</span>
          </div>
          <div class="hero-stat">
            <strong>Spec04</strong>
            <span>The workflow is now strong; semantics are the remaining bottleneck.</span>
          </div>
        </div>
      </div>
      <aside class="aside-card">
        <h2>Executive Readout</h2>
        <p><strong>Best message for today:</strong> workflow maturity is real, parity is disciplined, structural SVG generation is strong, and semantic control is the next hard problem.</p>
        <p><strong>Spec04 current state:</strong> latest stage keeps 100% valid SVG, 100% closure, and 100% OOD robustness, but only {_fmt_pct(spec04_latest["metrics"]["adherence"])} adherence.</p>
        <p><strong>Best Spec04 semantic point:</strong> {spec04_best["phase_label"]} reached {_fmt_pct(spec04_best["metrics"]["adherence"])} adherence and {_fmt_pct(spec04_best["metrics"]["tag_adherence"])} tag adherence.</p>
        <div class="link-row">
          {_link_button("IR Hub", 7003, "ir_hub.html")}
          {_link_button("Learning Comparison", 7002, "svg_learning_comparison_latest.html")}
          {_link_button("Spec04 Capability", 7005, "spec04_capability_report.html")}
        </div>
      </aside>
    </section>

    <div class="section-title">Suggested Presentation Route</div>
    <section class="route-grid">
      {"".join(route_cards)}
    </section>

    <div class="section-title">Visual Anchors</div>
    <section class="visual-grid">
      {"".join(anchors)}
    </section>

    <div class="section-title">Experiment Evolution</div>
    <section class="story-grid">
      {"".join(experiments)}
    </section>

    <div class="section-title">Actual Output Progress</div>
    <section class="gallery-grid">
      <div class="gallery-card">
        <h3>Toy Structured SVG: Simple Control Became Real</h3>
        <p>
          The toy line did not solve the whole problem, but it finally made prompt-to-shape behavior visually legible. This was the recovery point.
        </p>
        <div class="probe-grid">
          {"".join(toy_samples)}
        </div>
      </div>
      <div class="gallery-card">
        <h3>Spec04: Valid SVG, Partial Semantic Drift</h3>
        <p>
          The current line usually produces valid SVG, but the intended scene often drifts toward a nearby template. That is why the workflow looks stronger than the semantics.
        </p>
        <div class="probe-grid">
          {"".join(spec04_samples)}
        </div>
      </div>
    </section>

    <div class="section-title">Spec04 Stage Journey</div>
    <section class="timeline-grid">
      {"".join(spec04_timeline)}
    </section>

    <div class="section-title">Spec04 Scale Context</div>
    <section class="spec04-band">
      <div class="band-card">
        <h3>Why Spec04 Is Not “Only 144 Samples”</h3>
        <p>
          The small 144-row line was the older <code>spec04_structured_atoms_ctx512_d64_h128_v256</code> scaffold. The active line today is
          <code>spec04_structured_scenes_ctx512_d64_h128_v224</code>, and it is already past pretrain.
        </p>
        <div class="count-grid">
          <div class="count-box"><strong>{spec04_counts["pretrain"]["counts"]["train_rows"]:,}</strong><span>pretrain train rows</span></div>
          <div class="count-box"><strong>{spec04_counts["midtrain"]["counts"]["train_rows"]:,}</strong><span>midtrain train rows</span></div>
          <div class="count-box"><strong>{spec04_counts["sft"]["counts"]["train_rows"]:,}</strong><span>SFT train rows</span></div>
          <div class="count-box"><strong>{source_counts["unique_train"]:,}</strong><span>unique train scenes</span></div>
          <div class="count-box"><strong>{source_counts["unique_holdout"]:,}</strong><span>unique holdout scenes</span></div>
          <div class="count-box"><strong>{source_counts["tokenizer_rows"]:,}</strong><span>tokenizer corpus rows</span></div>
        </div>
      </div>
      <div class="band-card">
        <h3>What This Means</h3>
        <p><strong>We did move past pretrain.</strong> The current active line has already gone through richer pretrain, midtrain, and multiple SFT passes.</p>
        <p><strong>Why not jump straight to massive unique data?</strong> Because the current bottleneck is semantic control. If we scale too early, we mostly scale a valid-but-drifting behavior.</p>
        <p><strong>The present win:</strong> one cache-backed run now holds the dataset, tokenizer, parity, probe pages, training reports, and IR visualizer in one operator-visible place.</p>
        <div class="link-row">
          {_link_button("Spec04 Capability", 7005, "spec04_capability_report.html")}
          {_link_button("Spec04 Probe", 7005, "spec04_probe_report.html")}
          {_link_button("Spec04 Training Card", 7005, "svg_training_report_card.html")}
          {_link_button("Spec04 Dataset", 7005, "dataset_viewer.html")}
        </div>
      </div>
    </section>

    <p class="footer-note">
      Generated from cache-backed artifacts and embedded screenshots. This page is intentionally presentation-first: fewer tabs,
      more visual evidence, cleaner claims.
    </p>
  </main>
  <script>
    for (const node of document.querySelectorAll('[data-port][data-path]')) {{
      const port = node.getAttribute('data-port');
      const path = node.getAttribute('data-path').replace(/^\\//, '');
      node.href = `http://${{window.location.hostname}}:${{port}}/${{path}}`;
    }}
  </script>
</body>
</html>
"""
    return html_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_ROOT / "presentation" / "svg_experiment_story_latest.html",
        help="Destination HTML report.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_build_page(), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
