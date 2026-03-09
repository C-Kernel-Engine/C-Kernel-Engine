#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json(path)


def _html(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def _pct(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{num * 100:.0f}%"


def _num(value: Any, digits: int = 3) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{num:.{digits}f}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _extract_svg(response: str) -> tuple[str, str]:
    if not response:
        return "", ""
    start = response.find("<svg")
    end = response.rfind("</svg>")
    if start == -1 or end == -1:
        return response.strip(), ""
    return response[:start].strip(), response[start:end + 6]


def _count_lines(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for _ in handle:
            count += 1
    return count


def _count_control_tags(path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            prefix = line.split("<svg", 1)[0]
            for tag in re.findall(r"\[([^\]]+)\]", prefix):
                counter[tag] += 1
    return counter


def _count_svg_tags_in_dataset(path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            if "<svg" not in line:
                continue
            svg = "<svg" + line.split("<svg", 1)[1]
            for tag in re.findall(r"<\s*([A-Za-z][A-Za-z0-9:_-]*)\b", svg):
                if tag.lower() != "svg":
                    counter[tag] += 1
    return counter


def _count_svg_tags_in_assets(glob_pattern: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for raw in glob.glob(glob_pattern):
        path = Path(raw)
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for tag in re.findall(r"<\s*([A-Za-z][A-Za-z0-9:_-]*)\b", text):
            if tag.lower() != "svg":
                counter[tag] += 1
    return counter


def _resolve_dataset(dataset_name: str, run_dir: Path) -> Path | None:
    if not dataset_name:
        return None
    raw = Path(dataset_name)
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [
        run_dir / dataset_name,
        run_dir / "data" / dataset_name,
        ROOT / "version" / "v7" / "data" / "generated" / dataset_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    basename = raw.name
    for base in (run_dir / "data", ROOT / "version" / "v7" / "data" / "generated"):
        if not base.exists():
            continue
        matches = list(base.rglob(basename))
        if matches:
            return matches[0]
    return None


def _metric_value(entry: dict[str, Any], key: str) -> float | None:
    metrics = entry.get("metrics") or {}
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase_score(entry: dict[str, Any]) -> tuple[float, float, float]:
    adherence = _metric_value(entry, "adherence") or 0.0
    tag_adh = _metric_value(entry, "tag_adherence") or 0.0
    ood = _metric_value(entry, "ood_robustness") or 0.0
    return (adherence, tag_adh, ood)


def _svg_bar(current: float, maximum: float, width: int = 220, height: int = 14, color: str = "#39d98a") -> str:
    maximum = max(maximum, 1.0)
    fill = max(0.0, min(width, width * (current / maximum)))
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-hidden="true">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="7" fill="rgba(255,255,255,0.08)"/>'
        f'<rect x="0" y="0" width="{fill:.1f}" height="{height}" rx="7" fill="{color}"/>'
        "</svg>"
    )


def _sparkline(values: list[float], width: int = 180, height: int = 52, color: str = "#ffb020") -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1e-6)
    points: list[str] = []
    for idx, value in enumerate(values):
        x = 8 + idx * ((width - 16) / max(len(values) - 1, 1))
        y = height - 8 - ((value - lo) / span) * (height - 16)
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" aria-hidden="true">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="rgba(255,255,255,0.04)"/>'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>'
        "</svg>"
    )


def _metric_class(value: float | None) -> str:
    if value is None:
        return "metric-na"
    if value >= 0.75:
        return "metric-good"
    if value >= 0.4:
        return "metric-mid"
    return "metric-bad"


def _loss_drop_class(drop_pct: float | None) -> str:
    if drop_pct is None:
        return "metric-na"
    if drop_pct >= 60.0:
        return "metric-good"
    if drop_pct >= 20.0:
        return "metric-mid"
    return "metric-bad"


def _dataset_profile(path: Path | None, label: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "label": label,
            "path": None,
            "rows": 0,
            "control_tags": Counter(),
            "svg_tags": Counter(),
        }
    return {
        "label": label,
        "path": path,
        "rows": _count_lines(path),
        "control_tags": _count_control_tags(path),
        "svg_tags": _count_svg_tags_in_dataset(path),
    }


def _status_label(ok: bool | None, good: str = "PASS", bad: str = "FAIL", unknown: str = "MISSING") -> str:
    if ok is True:
        return good
    if ok is False:
        return bad
    return unknown


def _status_class(ok: bool | None) -> str:
    if ok is True:
        return "metric-good"
    if ok is False:
        return "metric-bad"
    return "metric-na"


def _dominant_tag(counter: Counter[str], prefix: str) -> str:
    filtered = [(key, value) for key, value in counter.items() if key.startswith(prefix)]
    if not filtered:
        return "—"
    best_key, best_value = max(filtered, key=lambda item: item[1])
    return f"{best_key} ({best_value:,})"


def _missing_target_tags(profile: dict[str, Any], asset_tags: Counter[str], min_count: int = 20) -> list[str]:
    svg_tags = profile.get("svg_tags", Counter())
    return [tag for tag, count in asset_tags.most_common() if count >= min_count and svg_tags.get(tag, 0) == 0]


def _render_top_counts(counter: Counter[str], limit: int = 10, kind: str = "tag") -> str:
    items = counter.most_common(limit)
    if not items:
        return '<div class="empty-note">No counts recorded.</div>'
    maximum = max(v for _, v in items)
    rows = []
    for key, value in items:
        rows.append(
            "<div class=\"bar-row\">"
            f"<div class=\"bar-label\">{_html(key)}</div>"
            f"<div class=\"bar-graphic\">{_svg_bar(float(value), float(maximum), color='#68d391' if kind == 'control' else '#7aa2ff')}</div>"
            f"<div class=\"bar-value\">{value:,}</div>"
            "</div>"
        )
    return "".join(rows)


def _report_cards(entries: list[dict[str, Any]], loss_entries: dict[str, dict[str, Any]]) -> str:
    cards = []
    for entry in entries:
        phase = entry["phase_label"]
        loss_meta = loss_entries.get(phase, {})
        adherence = _metric_value(entry, "adherence")
        tag_adh = _metric_value(entry, "tag_adherence")
        valid = _metric_value(entry, "valid_svg_rate")
        ood = _metric_value(entry, "ood_robustness")
        loss = loss_meta.get("final_loss")
        drop = loss_meta.get("drop_pct")
        cards.append(
            "<div class=\"stage-card\">"
            f"<div class=\"stage-chip\">{_html(phase)}</div>"
            f"<div class=\"stage-loss\">loss {_num(loss, 4)}</div>"
            f"<div class=\"metric-grid\">"
            f"<div><span class=\"label\">valid</span><span class=\"{_metric_class(valid)}\">{_pct(valid)}</span></div>"
            f"<div><span class=\"label\">ood</span><span class=\"{_metric_class(ood)}\">{_pct(ood)}</span></div>"
            f"<div><span class=\"label\">adh</span><span class=\"{_metric_class(adherence)}\">{_pct(adherence)}</span></div>"
            f"<div><span class=\"label\">tag</span><span class=\"{_metric_class(tag_adh)}\">{_pct(tag_adh)}</span></div>"
            f"<div><span class=\"label\">loss drop</span><span class=\"{_loss_drop_class(drop)}\">{drop:.1f}%</span></div>"
            "</div>"
            "</div>"
        )
    return "".join(cards)


def _build_output_matrix(entries: list[dict[str, Any]], probe_order: list[dict[str, Any]]) -> str:
    headers = "".join(
        f"<th><div class=\"probe-head\">{_html(probe.get('probe_id') or probe.get('id') or 'probe')}</div><div class=\"probe-sub\">{_html(probe.get('prompt') or '')}</div></th>"
        for probe in probe_order
    )
    rows = []
    for entry in entries:
        probe_map = {probe["probe_id"]: probe for probe in entry.get("probe_results") or []}
        cells = []
        for probe in probe_order:
            probe_id = probe.get("probe_id") or probe.get("id")
            result = probe_map.get(probe_id) or {}
            samples = result.get("samples") or []
            sample = samples[0] if samples else {}
            response = sample.get("response") or ""
            preamble, svg = _extract_svg(response)
            scores = sample.get("scores") or {}
            metric_line = (
                f"valid {_pct(scores.get('valid_svg'))} · "
                f"adh {_pct(scores.get('adherence'))} · "
                f"tag {_pct(scores.get('tag_adherence'))}"
            )
            if svg:
                preview = (
                    "<div class=\"svg-preview\">"
                    f"{svg}"
                    "</div>"
                )
            else:
                preview = f"<div class=\"text-preview\">{_html((response[:140] + '...') if len(response) > 140 else response or '-')}</div>"
            cells.append(
                "<td class=\"matrix-cell\">"
                f"{preview}"
                f"<div class=\"matrix-metrics\">{_html(metric_line)}</div>"
                f"<div class=\"matrix-preamble\">{_html((preamble[:80] + '...') if len(preamble) > 80 else preamble or '(no preamble)')}</div>"
                "</td>"
            )
        rows.append(
            "<tr>"
            f"<td class=\"sticky-col\"><div class=\"stage-chip\">{_html(entry['phase_label'])}</div></td>"
            + "".join(cells) +
            "</tr>"
        )
    return (
        "<div class=\"matrix-wrap\">"
        "<table class=\"output-matrix\">"
        "<thead><tr><th class=\"sticky-col\">Phase</th>"
        f"{headers}"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows) +
        "</tbody></table></div>"
    )


def _build_stage_table(entries: list[dict[str, Any]], loss_entries: dict[str, dict[str, Any]]) -> str:
    rows = []
    for entry in entries:
        phase = entry["phase_label"]
        loss_meta = loss_entries.get(phase, {})
        rows.append(
            "<tr>"
            f"<td><span class=\"stage-chip\">{_html(phase)}</span></td>"
            f"<td>{_html(Path(str(loss_meta.get('dataset_name') or '-')).name)}</td>"
            f"<td>{_num(loss_meta.get('first_loss'), 4)} -> {_num(loss_meta.get('final_loss'), 4)}</td>"
            f"<td class=\"{_metric_class(_metric_value(entry, 'valid_svg_rate'))}\">{_pct(_metric_value(entry, 'valid_svg_rate'))}</td>"
            f"<td class=\"{_metric_class(_metric_value(entry, 'ood_robustness'))}\">{_pct(_metric_value(entry, 'ood_robustness'))}</td>"
            f"<td class=\"{_metric_class(_metric_value(entry, 'adherence'))}\">{_pct(_metric_value(entry, 'adherence'))}</td>"
            f"<td class=\"{_metric_class(_metric_value(entry, 'tag_adherence'))}\">{_pct(_metric_value(entry, 'tag_adherence'))}</td>"
            f"<td>{int((_metric_value(entry, 'n_samples') or 0))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _diagnosis_points(entries: list[dict[str, Any]], profiles: dict[str, dict[str, Any]], asset_tags: Counter[str]) -> list[str]:
    best = max(entries, key=_phase_score)
    latest = entries[-1]
    sft_entries = [e for e in entries if e.get("stage") == "sft"]
    worst_pool = sft_entries or entries
    worst = min(worst_pool, key=lambda e: _metric_value(e, "adherence") or 0.0)
    has_sft = bool(sft_entries)
    latest_profile = profiles.get("latest_sft") or {}
    latest_svg_tags = latest_profile.get("svg_tags", Counter())
    missing = [tag for tag, count in asset_tags.most_common() if count >= 20 and latest_svg_tags.get(tag, 0) == 0][:6]
    points = [
        (
            f"Best practical checkpoint was {_html(best['phase_label'])}: "
            f"valid SVG {_pct(_metric_value(best, 'valid_svg_rate'))}, "
            f"adherence {_pct(_metric_value(best, 'adherence'))}, "
            f"tag adherence {_pct(_metric_value(best, 'tag_adherence'))}."
        ),
        (
            f"{'Worst control regression' if has_sft else 'Weakest evaluated phase'} was {_html(worst['phase_label'])}: "
            f"valid SVG stayed {_pct(_metric_value(worst, 'valid_svg_rate'))}, "
            f"but adherence fell to {_pct(_metric_value(worst, 'adherence'))}."
        ),
        (
            f"Latest stage {_html(latest['phase_label'])} recovered partway, but still trails the best checkpoint: "
            f"adherence {_pct(_metric_value(latest, 'adherence'))} vs {_pct(_metric_value(best, 'adherence'))}."
        ),
    ]
    if missing:
        points.append(
            f"{'Latest SFT data' if has_sft else 'Latest evaluated data'} still lacks major production-SVG building blocks seen in `docs/site/assets`: "
            + ", ".join(f"`{tag}`" for tag in missing) + "."
        )
    return points


def _build_gap_cards(profiles: dict[str, dict[str, Any]], asset_tags: Counter[str]) -> str:
    rows = []
    for key in ("midtrain", "best_sft", "worst_sft", "latest_sft"):
        profile = profiles.get(key)
        if not profile:
            continue
        label = profile["label"]
        svg_tags = profile["svg_tags"]
        missing = [tag for tag, count in asset_tags.most_common() if count >= 20 and svg_tags.get(tag, 0) == 0][:8]
        rows.append(
            "<div class=\"dataset-card\">"
            f"<h3>{_html(label)}</h3>"
            f"<div class=\"dataset-meta\">{profile['rows']:,} rows</div>"
            f"<div class=\"dataset-path\">{_html(profile['path']) if profile['path'] else 'path unavailable'}</div>"
            "<h4>Top control tags</h4>"
            f"{_render_top_counts(profile['control_tags'], limit=8, kind='control')}"
            "<h4>Top SVG tags</h4>"
            f"{_render_top_counts(profile['svg_tags'], limit=8, kind='svg')}"
            "<h4>Missing vs shipped assets</h4>"
            f"<div class=\"missing-list\">{', '.join(_html(tag) for tag in missing) if missing else 'No high-frequency asset tags missing.'}</div>"
            "</div>"
        )
    return "".join(rows)


def _build_data_flaw_table(profiles: dict[str, dict[str, Any]], asset_tags: Counter[str]) -> str:
    rows = []
    for key in ("midtrain", "best_sft", "worst_sft", "latest_sft"):
        profile = profiles.get(key)
        if not profile:
            continue
        missing = _missing_target_tags(profile, asset_tags)
        control_tags = profile.get("control_tags", Counter())
        svg_tags = profile.get("svg_tags", Counter())
        rows.append(
            "<tr>"
            f"<td><strong>{_html(profile['label'])}</strong></td>"
            f"<td>{profile['rows']:,}</td>"
            f"<td>{len(control_tags):,}</td>"
            f"<td>{_html(_dominant_tag(control_tags, 'style:'))}</td>"
            f"<td>{len(svg_tags):,}</td>"
            f"<td>{len(missing):,}</td>"
            f"<td>{_html(', '.join(missing[:6]) if missing else 'none')}</td>"
            "</tr>"
        )
    return "".join(rows)


def _narrative_section(best_phase: str, latest_phase: str, has_midtrain: bool, has_sft: bool) -> str:
    if not has_midtrain and not has_sft:
        return (
            "<div class=\"narrative-grid\">"
            "<div class=\"narrative-card\">"
            "<h3>Pretrain Result</h3>"
            "<p>This run only measured pretrain. The question here is not instruction following yet; it is whether the model learned valid SVG closure, primitive structure, and enough visual grammar to serve as a useful base.</p>"
            "</div>"
            "<div class=\"narrative-card\">"
            "<h3>Why It Failed</h3>"
            "<p>Pretrain alone can teach syntax and recurring structure, but it does not reliably teach prompt control. If validity is low here, the issue is still representation, tokenizer shape, or raw pretrain corpus fit.</p>"
            "</div>"
            "<div class=\"narrative-card\">"
            "<h3>What To Compare</h3>"
            f"<p>Use this run as the pretrain baseline, then compare it against a stronger staged line. `{_html(best_phase)}` vs `{_html(latest_phase)}` only tells you how far pretrain got before any midtrain or SFT control data existed.</p>"
            "</div>"
            "<div class=\"narrative-card\">"
            "<h3>Next Clean Run</h3>"
            "<p>Keep the tokenizer fixed, improve the representation contract, and then add a staged control curriculum. That is the right comparison against spec02 and the pure DSL toy line.</p>"
            "</div>"
            "</div>"
        )
    return (
        "<div class=\"narrative-grid\">"
        "<div class=\"narrative-card\">"
        "<h3>Pretrain Gap</h3>"
        "<p>Pretrain learned SVG syntax and primitive drawing patterns, but not the full visual vocabulary of the shipped assets. "
        "Your real targets are text-heavy, grouped, gradient-rich, and path-driven. Primitive-only pretrain leaves the model without that grammar.</p>"
        "</div>"
        "<div class=\"narrative-card\">"
        "<h3>Midtrain Gap</h3>"
        "<p>Midtrain introduced charts, cards, tables, and gradients, which is why control improved early. "
        "But it still did not teach explicit numeric data payloads, multi-panel composition, or reconstruction tasks from structured specs.</p>"
        "</div>"
        "<div class=\"narrative-card\">"
        "<h3>SFT Gap</h3>"
        "<p>SFT data taught shallow tag control more than intentful composition. "
        f"`{_html(best_phase)}` was the best balance point. Later runs narrowed or distorted the control contract, so syntax stayed stable while instruction following regressed. "
        f"`{_html(latest_phase)}` recovers some of that, but not enough yet.</p>"
        "</div>"
        "<div class=\"narrative-card\">"
        "<h3>Next Clean Run</h3>"
        "<p>For the next chain, use a stable spec-to-SVG prompt contract, seed from the strongest practical checkpoint, and add asset-derived pretrain data plus render-level canary evals. "
        "That is the shortest path toward production-quality infographics.</p>"
        "</div>"
        "</div>"
    )


def _trust_snapshot(run_dir: Path) -> dict[str, Any]:
    roundtrip = _load_json_if_exists(run_dir / "tokenizer_roundtrip.json")
    train_e2e = _load_json_if_exists(run_dir / "train_e2e_latest.json")
    parity = _load_json_if_exists(run_dir / "training_parity.json")
    regimen = _load_json_if_exists(run_dir / "training_parity_regimen_latest.json")
    dataset_qc = _load_json_if_exists(run_dir / "dataset_qc.json")

    line_eval = roundtrip.get("line_eval") if isinstance(roundtrip.get("line_eval"), dict) else {}
    steps = parity.get("steps") if isinstance(parity.get("steps"), list) else []
    worst_param = max(
        (float(step.get("max_param_diff")) for step in steps if step.get("max_param_diff") is not None),
        default=math.nan,
    )
    last_step = steps[-1].get("step") if steps else None
    regimen_summary = regimen.get("summary") if isinstance(regimen.get("summary"), dict) else {}

    return {
        "dataset_qc_pass": dataset_qc.get("status") == "pass" or dataset_qc.get("passed") is True,
        "roundtrip_pass": roundtrip.get("exact_match") is True and float(line_eval.get("exact_match_rate", 0.0) or 0.0) >= 1.0,
        "roundtrip_rate": line_eval.get("exact_match_rate"),
        "train_e2e_pass": train_e2e.get("pass_parity") if "pass_parity" in train_e2e else train_e2e.get("pass"),
        "worst_param_diff": worst_param,
        "parity_rows": len(steps),
        "parity_last_step": last_step,
        "regimen_pass": regimen_summary.get("passed"),
        "regimen_passed_stages": regimen_summary.get("passed_stages"),
        "regimen_total_stages": regimen_summary.get("total_stages"),
    }


def _trust_verdict(trust: dict[str, Any], latest: dict[str, Any]) -> tuple[str, str]:
    worst = trust.get("worst_param_diff")
    adherence = _metric_value(latest, "adherence") or 0.0
    if trust.get("train_e2e_pass") is not True or trust.get("regimen_pass") is not True:
        return (
            "Numeric trust is broken",
            "Parity gates did not pass cleanly. Do not trust downstream eval until CK vs PyTorch is fixed.",
        )
    if trust.get("roundtrip_pass") is not True:
        return (
            "Tokenizer/data path is broken",
            "Roundtrip fidelity is not clean. Fix the data/tokenizer path before drawing conclusions from training.",
        )
    if math.isfinite(float(worst)) and float(worst) > 1e-5:
        return (
            "Numeric trust is borderline",
            "Parity exists but drift is elevated. You can inspect behavior, but promotion should stay gated.",
        )
    if adherence < 0.7:
        return (
            "Numeric trust is good; data fit is the bottleneck",
            "CK vs PyTorch is healthy. The remaining problem is dataset quality, prompt contract, and target-grammar coverage.",
        )
    return (
        "Run is broadly healthy",
        "Parity, tokenizer fidelity, and task adherence are all in a usable range.",
    )


def _validity_note(valid_svg_rate: float | None) -> str:
    if valid_svg_rate is None:
        return "Validity was not recorded for this stage."
    if valid_svg_rate >= 0.99:
        return "Syntax and closure are solved. The remaining problem is whether the model obeys the requested prompt and visual intent."
    if valid_svg_rate >= 0.5:
        return "Some valid SVGs are emerging, but closure and structure are still unstable enough that control scores are hard to trust."
    return "SVG validity is still broken. The model is failing before higher-level composition or control really matters."


def _adherence_note(adherence: float | None) -> str:
    if adherence is None:
        return "Task adherence was not recorded for this stage."
    if adherence >= 0.75:
        return "Prompt control is already usable. The next work is broadening coverage, not rescuing basic obedience."
    if adherence >= 0.4:
        return "The model sometimes follows the requested visual intent, but control is still inconsistent across probes."
    return "This is the real bottleneck: even when the model emits something structured, it is usually not the requested chart, palette, or composition."


def build_html(run_dir: Path, output_path: Path, assets_glob: str) -> str:
    matrix = _load_json(run_dir / "stage_eval_matrix.json")
    pipeline = _load_json(run_dir / "training_pipeline_latest.json")
    entries = matrix.get("entries") or []
    entries = sorted(entries, key=lambda item: int(item.get("run_order") or 0))
    loss_entries = {
        entry["phase_label"]: entry
        for entry in ((pipeline.get("stage_loss_history") or {}).get("entries") or [])
        if entry.get("phase_label")
    }
    probes = matrix.get("probes") or []
    best = max(entries, key=_phase_score)
    latest = entries[-1]
    sft_entries = [e for e in entries if e.get("stage") == "sft"]
    has_sft = bool(sft_entries)
    has_midtrain = any(e.get("stage") == "midtrain" for e in entries)
    worst_sft = min((sft_entries or entries), key=lambda e: _metric_value(e, "adherence") or 0.0)

    loss_values = [float(loss_entries[e["phase_label"]]["final_loss"]) for e in entries if e["phase_label"] in loss_entries and loss_entries[e["phase_label"]].get("final_loss") is not None]
    adherence_values = [_metric_value(e, "adherence") or 0.0 for e in entries]
    ood_values = [_metric_value(e, "ood_robustness") or 0.0 for e in entries]

    midtrain_entry = next((e for e in entries if e.get("stage") == "midtrain"), None)
    latest_sft_entry = next((e for e in reversed(entries) if e.get("stage") == "sft"), None) or latest
    best_sft_entry = max(sft_entries, key=_phase_score) if sft_entries else best
    worst_sft_entry = worst_sft

    profiles = {
        "midtrain": _dataset_profile(_resolve_dataset(loss_entries.get(midtrain_entry["phase_label"], {}).get("dataset_name", "") if midtrain_entry else "", run_dir), "Midtrain dataset"),
        "best_sft": _dataset_profile(
            _resolve_dataset(loss_entries.get(best_sft_entry["phase_label"], {}).get("dataset_name", ""), run_dir),
            f"{'Best SFT dataset' if has_sft else 'Best evaluated dataset'} ({best_sft_entry['phase_label']})",
        ),
        "worst_sft": _dataset_profile(
            _resolve_dataset(loss_entries.get(worst_sft_entry["phase_label"], {}).get("dataset_name", ""), run_dir),
            f"{'Worst-control SFT dataset' if has_sft else 'Weakest evaluated dataset'} ({worst_sft_entry['phase_label']})",
        ),
        "latest_sft": _dataset_profile(
            _resolve_dataset(loss_entries.get(latest_sft_entry["phase_label"], {}).get("dataset_name", ""), run_dir),
            f"{'Latest SFT dataset' if has_sft else 'Latest evaluated dataset'} ({latest_sft_entry['phase_label']})",
        ),
    }
    latest_profile = profiles.get("latest_sft") or {}
    asset_tags = _count_svg_tags_in_assets(assets_glob)
    diagnosis_points = _diagnosis_points(entries, profiles, asset_tags)
    data_flaw_rows = _build_data_flaw_table(profiles, asset_tags)
    trust = _trust_snapshot(run_dir)
    trust_title, trust_note = _trust_verdict(trust, latest)

    phase_cards = _report_cards(entries, loss_entries)
    stage_table = _build_stage_table(entries, loss_entries)
    output_matrix = _build_output_matrix(entries, probes)
    gap_cards = _build_gap_cards(profiles, asset_tags)
    narrative = _narrative_section(best["phase_label"], latest["phase_label"], has_midtrain, has_sft)
    generated_at = matrix.get("generated_at") or "-"
    headline_title = (
        f"What `{run_dir.name}` learned from pure pretrain"
        if not has_midtrain and not has_sft
        else f"Why `{run_dir.name}` plateaued before it became a beautiful infographic model"
    )
    headline_subhead = (
        "This report synthesizes stage loss, per-stage eval, real prompt outputs, and a direct comparison "
        "between the training corpora and your shipped SVG assets in <code>docs/site/assets/</code>. "
        "It is designed to separate <strong>numeric trust</strong> from <strong>data-fit failure</strong>, "
        "explain what the model learned, show where control regressed, and make the next clean run concrete."
    )
    worst_label = "Worst control regression" if has_sft else "Weakest evaluated phase"
    scoreboard_intro = (
        "All phases kept SVG validity. The movement was in adherence and task control."
        if all((_metric_value(entry, "valid_svg_rate") or 0.0) >= 0.99 for entry in entries)
        else "Stage-by-stage validity, adherence, and grammar coverage for this run."
    )
    latest_valid = _metric_value(latest, "valid_svg_rate")
    latest_adh = _metric_value(latest, "adherence")
    latest_gap_count = len([tag for tag, count in asset_tags.items() if count >= 20 and latest_profile.get("svg_tags", Counter()).get(tag, 0) == 0])
    gap_note = (
        "High-frequency SVG tags from shipped assets that are still absent from the latest SFT dataset."
        if has_sft
        else "High-frequency SVG tags from shipped assets that are still absent from the latest evaluated dataset."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SVG Training Report Card</title>
  <style>
    :root {{
      --bg: #16181d;
      --panel: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
      --panel-solid: #20242c;
      --border: rgba(255,255,255,0.12);
      --text: #eef2f7;
      --muted: #98a2b3;
      --good: #39d98a;
      --mid: #ffb020;
      --bad: #ff6b6b;
      --accent: #7aa2ff;
      --accent-2: #6fe7ff;
      --ink: #0f1116;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(122,162,255,0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(57,217,138,0.12), transparent 28%),
        linear-gradient(180deg, #14171c 0%, #0f1116 100%);
      color: var(--text);
    }}
    .page {{
      width: min(1480px, calc(100vw - 48px));
      margin: 28px auto 48px;
    }}
    .hero, .panel {{
      border: 1px solid var(--border);
      border-radius: 20px;
      background: var(--panel);
      box-shadow: 0 20px 60px rgba(0,0,0,0.28);
      backdrop-filter: blur(6px);
    }}
    .hero {{
      padding: 28px 32px;
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 24px;
      margin-bottom: 22px;
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(122,162,255,0.16);
      color: #bfd1ff;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 14px 0 8px;
      font-size: 40px;
      line-height: 1.05;
    }}
    .subhead {{
      color: var(--muted);
      font-size: 17px;
      line-height: 1.6;
      max-width: 850px;
    }}
    .hero-metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .hero-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 16px;
    }}
    .hero-card .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .hero-card .v {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 800;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 22px;
      margin-bottom: 22px;
    }}
    .panel {{
      padding: 24px 26px;
      overflow: hidden;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .muted {{
      color: var(--muted);
    }}
    ul.findings {{
      margin: 16px 0 0;
      padding-left: 18px;
      line-height: 1.7;
    }}
    .stage-row {{
      display: flex;
      gap: 14px;
      overflow-x: auto;
      padding-bottom: 8px;
      margin-top: 18px;
    }}
    .stage-card {{
      min-width: 220px;
      max-width: 220px;
      padding: 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .stage-chip {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(255,176,32,0.14);
      color: #ffd48f;
      font-weight: 700;
      font-size: 12px;
    }}
    .stage-loss {{
      margin-top: 12px;
      font-size: 20px;
      font-weight: 700;
    }}
    .metric-grid {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 10px;
      font-size: 13px;
    }}
    .metric-grid .label {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric-good {{ color: var(--good); font-weight: 700; }}
    .metric-mid {{ color: var(--mid); font-weight: 700; }}
    .metric-bad {{ color: var(--bad); font-weight: 700; }}
    .metric-na {{ color: var(--muted); }}
    .summary-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .summary-cell {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 16px;
    }}
    .summary-cell h3 {{
      margin: 0 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .summary-cell .main {{
      font-size: 28px;
      font-weight: 800;
    }}
    .summary-cell .small {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }}
    .chart-row {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .chart-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 14px;
    }}
    .chart-card .title {{
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .matrix-wrap {{
      overflow-x: auto;
      padding-bottom: 8px;
    }}
    .output-matrix {{
      min-width: 1280px;
    }}
    .sticky-col {{
      position: sticky;
      left: 0;
      background: #232830;
      z-index: 2;
    }}
    .probe-head {{
      font-weight: 700;
      color: #dde8ff;
      margin-bottom: 6px;
    }}
    .probe-sub {{
      font-size: 11px;
      color: var(--muted);
      line-height: 1.4;
      font-weight: 400;
      text-transform: none;
      letter-spacing: 0;
    }}
    .matrix-cell {{
      min-width: 180px;
      max-width: 180px;
    }}
    .svg-preview, .text-preview {{
      width: 146px;
      height: 104px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: #f8fafc;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--ink);
    }}
    .text-preview {{
      background: rgba(255,255,255,0.04);
      color: var(--text);
      font-size: 12px;
      line-height: 1.5;
      padding: 10px;
      align-items: flex-start;
      justify-content: flex-start;
    }}
    .matrix-metrics {{
      margin-top: 8px;
      font-size: 11px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .matrix-preamble {{
      margin-top: 6px;
      font-size: 11px;
      color: #c3ccd9;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      line-height: 1.4;
      word-break: break-word;
    }}
    .dataset-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .trust-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .trust-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 16px;
    }}
    .trust-card h3 {{
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }}
    .trust-card .value {{
      font-size: 24px;
      font-weight: 800;
    }}
    .trust-card .note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .dataset-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 18px;
    }}
    .dataset-card h3 {{
      margin: 0 0 6px;
      font-size: 18px;
    }}
    .dataset-card h4 {{
      margin: 16px 0 10px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .dataset-meta, .dataset-path, .missing-list {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 110px 1fr 56px;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
      font-size: 13px;
    }}
    .bar-label {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .bar-value {{
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .narrative-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .narrative-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 18px;
    }}
    .narrative-card h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .narrative-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }}
    .prompt-card {{
      margin-top: 18px;
      border: 1px solid rgba(122,162,255,0.24);
      background: rgba(122,162,255,0.08);
      border-radius: 16px;
      padding: 18px;
    }}
    pre {{
      margin: 10px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.6;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #dce6ff;
    }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    .empty-note {{
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 980px) {{
      .hero, .grid-2, .summary-strip, .chart-row, .dataset-grid, .narrative-grid, .trust-grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        width: min(100vw - 20px, 1480px);
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <div class="eyebrow">SVG Training Report Card</div>
        <h1>{headline_title}</h1>
        <div class="subhead">{headline_subhead}</div>
        <ul class="findings">
          {"".join(f"<li>{point}</li>" for point in diagnosis_points)}
        </ul>
      </div>
      <div class="hero-metrics">
        <div class="hero-card">
          <div class="k">Best practical checkpoint</div>
          <div class="v">{_html(best['phase_label'])}</div>
          <div class="muted">adherence {_pct(_metric_value(best, 'adherence'))} · tag {_pct(_metric_value(best, 'tag_adherence'))}</div>
        </div>
        <div class="hero-card">
          <div class="k">Latest checkpoint</div>
          <div class="v">{_html(latest['phase_label'])}</div>
          <div class="muted">adherence {_pct(_metric_value(latest, 'adherence'))} · tag {_pct(_metric_value(latest, 'tag_adherence'))}</div>
        </div>
        <div class="hero-card">
          <div class="k">{worst_label}</div>
          <div class="v">{_html(worst_sft['phase_label'])}</div>
          <div class="muted">syntax held, control dropped to {_pct(_metric_value(worst_sft, 'adherence'))}</div>
        </div>
        <div class="hero-card">
          <div class="k">Target asset language</div>
          <div class="v">{sum(asset_tags.values()):,}</div>
          <div class="muted">SVG nodes counted from {len(glob.glob(assets_glob))} shipped asset files</div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Trust Boundary</h2>
      <div class="muted">{_html(trust_title)}. {_html(trust_note)}</div>
      <div class="trust-grid">
        <div class="trust-card">
          <h3>Dataset QC</h3>
          <div class="value {_status_class(trust['dataset_qc_pass'])}">{_status_label(trust['dataset_qc_pass'])}</div>
          <div class="note">Raw dataset gating before token packing.</div>
        </div>
        <div class="trust-card">
          <h3>Tokenizer Roundtrip</h3>
          <div class="value {_status_class(trust['roundtrip_pass'])}">{_status_label(trust['roundtrip_pass'])}</div>
          <div class="note">Exact line match rate {_pct(trust['roundtrip_rate'])}.</div>
        </div>
        <div class="trust-card">
          <h3>Train E2E Parity</h3>
          <div class="value {_status_class(trust['train_e2e_pass'])}">{_status_label(trust['train_e2e_pass'])}</div>
          <div class="note">End-to-end CK vs PyTorch canary on the train path.</div>
        </div>
        <div class="trust-card">
          <h3>Parity Regimen</h3>
          <div class="value {_status_class(trust['regimen_pass'])}">{_status_label(trust['regimen_pass'])}</div>
          <div class="note">Worst param diff {_num(trust['worst_param_diff'], 6)} across {trust['parity_rows']:,} logged parity rows.</div>
        </div>
      </div>
      <div class="footer-note">
        Regimen stages passed: {trust['regimen_passed_stages'] if trust['regimen_passed_stages'] is not None else '-'} / {trust['regimen_total_stages'] if trust['regimen_total_stages'] is not None else '-'}.
        Latest parity step: {_html(trust['parity_last_step'] if trust['parity_last_step'] is not None else '-')}
      </div>
    </section>

    <section class="panel" style="margin-top:22px;">
      <h2>Stage Scoreboard</h2>
      <div class="muted">{scoreboard_intro}</div>
      <div class="summary-strip">
        <div class="summary-cell">
          <h3>Latest valid SVG</h3>
          <div class="main">{_pct(latest_valid)}</div>
          <div class="small">{_validity_note(latest_valid)}</div>
        </div>
        <div class="summary-cell">
          <h3>Latest adherence</h3>
          <div class="main">{_pct(latest_adh)}</div>
          <div class="small">{_adherence_note(latest_adh)}</div>
        </div>
        <div class="summary-cell">
          <h3>Target grammar gap</h3>
          <div class="main">{latest_gap_count}</div>
          <div class="small">{gap_note}</div>
        </div>
      </div>
      <div class="chart-row">
        <div class="chart-card">
          <div class="title">Final loss by stage</div>
          {_sparkline(loss_values, color="#ffb020")}
        </div>
        <div class="chart-card">
          <div class="title">Adherence by stage</div>
          {_sparkline(adherence_values, color="#39d98a")}
        </div>
        <div class="chart-card">
          <div class="title">OOD robustness by stage</div>
          {_sparkline(ood_values, color="#7aa2ff")}
        </div>
      </div>
      <div class="stage-row">{phase_cards}</div>
      <table style="margin-top:18px;">
        <thead>
          <tr>
            <th>Phase</th>
            <th>Dataset</th>
            <th>Loss</th>
            <th>Valid SVG</th>
            <th>OOD</th>
            <th>Adherence</th>
            <th>Tag adherence</th>
            <th>N</th>
          </tr>
        </thead>
        <tbody>
          {stage_table}
        </tbody>
      </table>
      <div class="footer-note">Generated from <code>stage_eval_matrix.json</code> and <code>training_pipeline_latest.json</code> at {_html(generated_at)}.</div>
    </section>

    <section class="panel" style="margin-top:22px;">
      <h2>Prompt-by-Stage Output Matrix</h2>
      <div class="muted">
        First sample from each eval probe. This is the proof surface: it shows what the model actually rendered, not just the aggregate score.
        A strong stage keeps syntax valid and obeys the requested chart/palette/style without drifting to neutral defaults.
      </div>
      {output_matrix}
    </section>

    <section class="panel" style="margin-top:22px;">
      <h2>Dataset vs Target Asset Gap</h2>
      <div class="muted">
        The shipped assets in <code>docs/site/assets/</code> are the real target distribution. The question is not only whether the model knows SVG,
        but whether the training corpora actually expose the visual vocabulary required for those assets.
      </div>
      <div class="dataset-grid">{gap_cards}</div>
      <table style="margin-top:18px;">
        <thead>
          <tr>
            <th>Dataset slice</th>
            <th>Rows</th>
            <th>Unique control tags</th>
            <th>Dominant style tag</th>
            <th>Unique SVG tags</th>
            <th>Missing target tags</th>
            <th>Main missing examples</th>
          </tr>
        </thead>
        <tbody>
          {data_flaw_rows}
        </tbody>
      </table>
    </section>

    <section class="panel" style="margin-top:22px;">
      <h2>What To Learn Before The Next Run</h2>
      {narrative}
      <div class="prompt-card">
        <div class="eyebrow">AI Handoff Prompt</div>
        <pre>You are reviewing an SVG training run. Explain:
1. which phase is the best practical checkpoint and why,
2. whether the latest checkpoint improved or regressed from that baseline,
3. which control tags or SVG primitives are missing from the training data compared to the shipped assets,
4. whether the next run should add more primitive SVG pretrain, richer asset-derived pretrain, spec-to-SVG SFT, or render-level eval checks.

Base your answer on:
- stage-level metrics (valid_svg, ood, adherence, tag_adherence),
- the prompt-by-stage output matrix,
- and the dataset-vs-target tag gap section.
Avoid hand-wavy claims: tie each conclusion to a visible number or rendered sample.</pre>
      </div>
    </section>
  </div>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a standalone SVG training report card for a run directory.")
    ap.add_argument("--run", required=True, help="Run directory containing stage_eval_matrix.json and training_pipeline_latest.json")
    ap.add_argument("--assets-glob", default="docs/site/assets/*.svg", help="Target asset SVG glob used for grammar comparison")
    ap.add_argument("--output", required=True, help="Output HTML path")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    stage_eval_path = run_dir / "stage_eval_matrix.json"
    if not stage_eval_path.exists():
        hint = (
            "This report builder expects a stage-evaluated SVG run with stage_eval_matrix.json.\n"
            f"Missing: {stage_eval_path}\n"
            "Run stage eval first, or use a report builder registered for that run family.\n"
        )
        raise SystemExit(hint)
    output.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_html(run_dir, output, str((ROOT / args.assets_glob).resolve()) if not Path(args.assets_glob).is_absolute() else args.assets_glob)
    output.write_text(html_text)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
