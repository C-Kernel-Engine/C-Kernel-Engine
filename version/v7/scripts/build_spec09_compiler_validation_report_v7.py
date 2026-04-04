#!/usr/bin/env python3
"""Build a validation report for the spec09 scene compiler."""

from __future__ import annotations

import argparse
import html
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from render_svg_structured_scene_spec09_v7 import render_structured_scene_spec09_svg


VALIDATION_CASES: list[dict[str, Any]] = [
    {
        "id": "poster_dark",
        "title": "Poster Stack Dark",
        "description": "Tall infographic poster with stacked section cards, rail, badge, and footer note.",
        "dsl": """[scene]
[canvas:tall]
[layout:poster_stack]
[theme:infra_dark]
[tone:mixed]
[frame:card]
[density:balanced]
[inset:md]
[gap:md]
[hero:center]
[columns:1]
[emphasis:top]
[rail:accent]
[background:grid]
[connector:line]
[topic:memory_reality]
[header_band:first_principle|llm_memory_reality|the_math_marketing_skips]
[section_card:principle|0_x_inf_eq_0|compute_speed_does_not_matter_if_the_model_does_not_fit]
[section_card:capacity|gpu_80gb_vs_cpu_2tb|memory_capacity_drives_feasible_context]
[section_card:fit_analysis|70b_plus_context|fit_math_favors_cpu_first_deployment]
[section_card:cost_compare|gpu_path_vs_cpu_path|room_for_more_context_on_cpu]
[badge_pill:8x_lower_cost]
[footer_note:c_kernel_engine_cpu_first_llm_inference]
[/scene]""",
    },
    {
        "id": "compare_span",
        "title": "Comparison Span Chart",
        "description": "Gap-centric comparison scene with richer bars, axis, legend, annotation, divider, and conclusion strip.",
        "dsl": """[scene]
[canvas:wide]
[layout:comparison_span_chart]
[theme:infra_dark]
[tone:mixed]
[frame:none]
[density:balanced]
[inset:md]
[gap:md]
[hero:center]
[columns:3]
[emphasis:center]
[rail:none]
[background:none]
[connector:bracket]
[topic:performance_balance]
[header_band:headline|performance_gap|same_floor_very_different_spans]
[compare_bar:gpu_compute|67000_gb_s_eq|5000x_total_span|accent=amber|note=hbm_is_fast_but_capacity_bound]
[compare_bar:cpu_compute|1800_gb_s_eq|144x_total_span|accent=green|note=memory_fit_is_closer_to_the_real_limit]
[axis:bandwidth_axis|same_network_floor]
[legend_row:amber=gpu_cluster|green=cpu_server]
[annotation:bottleneck_shift|memory_and_network_dominate|accent=amber]
[divider:dash]
[span_bracket:gpu_span|5360x]
[span_bracket:cpu_span|144x]
[floor_band:ethernet_same_floor_for_every_cluster]
[thesis_box:gpu_structural_mismatch|cpu_gap_is_closeable|optimize_for_the_constraint_not_the_headline]
[conclusion_strip:match_flops_to_data_movement_speed]
[footer_note:physics_sets_the_budget]
[/scene]""",
    },
    {
        "id": "content_bound_compare",
        "title": "Content-Bound Comparison",
        "description": "Scene DSL plus external JSON payload. The compiler resolves @paths and renders a populated infographic from structure plus data.",
        "dsl": """[scene]
[canvas:wide]
[layout:comparison_span_chart]
[theme:infra_dark]
[tone:mixed]
[frame:card]
[density:balanced]
[inset:md]
[gap:md]
[hero:center]
[columns:3]
[emphasis:center]
[rail:none]
[background:mesh]
[connector:bracket]
[topic:live_balance]
[header_band:@title.kicker|@title.headline|@title.subtitle]
[compare_bar:@bars.gpu.label|@bars.gpu.value|@bars.gpu.caption|accent=@bars.gpu.accent|note=@bars.gpu.note]
[compare_bar:@bars.cpu.label|@bars.cpu.value|@bars.cpu.caption|accent=@bars.cpu.accent|note=@bars.cpu.note]
[axis:@axis.label|@axis.note]
[legend_row:amber=@legend.gpu|green=@legend.cpu]
[annotation:@annotations.0.label|@annotations.0.note|accent=@annotations.0.accent]
[span_bracket:@bars.gpu.bracket_label|@bars.gpu.bracket_value]
[span_bracket:@bars.cpu.bracket_label|@bars.cpu.bracket_value]
[thesis_box:@thesis.title|@thesis.line_1|@thesis.line_2]
[conclusion_strip:@summary.conclusion]
[footer_note:@summary.footer]
[/scene]""",
        "content": {
            "title": {
                "kicker": "live brief",
                "headline": "Compiler-Bound Scene + Data",
                "subtitle": "The model can pick structure while another source supplies final text and values.",
            },
            "bars": {
                "gpu": {
                    "label": "GPU Path",
                    "value": "67,000 GB/s eq",
                    "caption": "5,360x span",
                    "accent": "amber",
                    "note": "HBM is fast but capacity bound.",
                    "bracket_label": "Peak span",
                    "bracket_value": "5360x",
                },
                "cpu": {
                    "label": "CPU Path",
                    "value": "1,800 GB/s eq",
                    "caption": "144x span",
                    "accent": "green",
                    "note": "Fit and cost are closer to deployment reality.",
                    "bracket_label": "Reachable span",
                    "bracket_value": "144x",
                },
            },
            "axis": {
                "label": "Bandwidth scale",
                "note": "Shared network floor underneath both paths",
            },
            "legend": {
                "gpu": "GPU cluster",
                "cpu": "CPU server",
            },
            "annotations": [
                {
                    "label": "Bottleneck shift",
                    "note": "Memory and topology dominate before peak FLOPs.",
                    "accent": "amber",
                }
            ],
            "thesis": {
                "title": "Compiler + content is the right split.",
                "line_1": "Structure can stay canonical while data changes per source.",
                "line_2": "That keeps training focused on composition instead of prose memorization.",
            },
            "summary": {
                "conclusion": "A scene DSL plus JSON payload can render the full infographic, not just the skeleton.",
                "footer": "This is the bridge from layout planning to production-bound content rendering.",
            },
        },
    },
    {
        "id": "pipeline_lane",
        "title": "Pipeline Lane",
        "description": "Wide process diagram with phase divider, repeated stage cards, arrows, a curved connector, and a status pill.",
        "dsl": """[scene]
[canvas:wide]
[layout:pipeline_lane]
[theme:infra_dark]
[tone:amber]
[frame:panel]
[density:balanced]
[inset:md]
[gap:md]
[hero:split]
[columns:3]
[emphasis:top]
[rail:none]
[background:none]
[connector:arrow]
[topic:pipeline_overview]
[header_band:from_config_to_running_model|c_kernel_engine_pipeline|code_generation_to_runtime]
[phase_divider:code_generation|runtime]
[stage_card:config|huggingface_style_input]
[flow_arrow:config->parse]
[stage_card:parse|build_ckir_graph]
[flow_arrow:parse->generate]
[stage_card:generate|emit_clean_c_code]
[flow_arrow:generate->weights]
[stage_card:weights|convert_to_runtime_format]
[flow_arrow:weights->compile]
[stage_card:compile|gcc_avx512_amx]
[curved_connector:compile->run]
[stage_card:run|inference_plus_training]
[badge_pill:phase_1_active]
[footer_note:pytorch_parity_and_runtime_validation]
[/scene]""",
    },
    {
        "id": "dashboard_light",
        "title": "Dashboard Cards Editorial",
        "description": "Light editorial dashboard showing that spec09 can render a paper-like presentation style instead of only dark canvases.",
        "dsl": """[scene]
[canvas:wide]
[layout:dashboard_cards]
[theme:paper_editorial]
[tone:blue]
[frame:panel]
[density:airy]
[inset:lg]
[gap:md]
[hero:center]
[columns:3]
[emphasis:center]
[rail:none]
[background:none]
[connector:line]
[topic:runtime_summary]
[header_band:release_brief|runtime_validation_snapshot|editorial_scorecard]
[section_card:parity|max_diff_lt_1e_5|kernel_checks_match_reference]
[section_card:coverage|97_assets_scanned|scene_vocabulary_is_asset_grounded]
[section_card:renderability|100_percent_valid_svg|compiler_contract_is_stable]
[section_card:next_move|gold_asset_reconstructions|train_only_after_compiler_quality]
[footer_note:paper_mode_for_summary_and_release_assets]
[/scene]""",
    },
    {
        "id": "dual_panel_glow",
        "title": "Dual Panel Compare Glow",
        "description": "Two-panel comparison board with a high-signal dark theme and paired argument structure.",
        "dsl": """[scene]
[canvas:wide]
[layout:dual_panel_compare]
[theme:signal_glow]
[tone:purple]
[frame:panel]
[density:balanced]
[inset:md]
[gap:md]
[hero:split]
[columns:2]
[emphasis:left]
[rail:accent]
[background:mesh]
[connector:line]
[topic:deployment_tradeoffs]
[header_band:decision_frame|gpu_scale_vs_cpu_fit|choose_the_constraint_you_can_close]
[compare_panel:gpu_cluster|67_tb_s_peak|network_and_memory_become_the_real_limit|variant=metric|accent=purple]
[compare_panel:cpu_server|2_tb_ram_fit|fit_and_cost_are_more_controllable|variant=success|accent=green]
[callout_card:constraint_selection_beats_peak_flops|choose_the_part_you_can_close|accent=purple]
[footer_note:paired_argument_layout_for_deployment_memos]
[/scene]""",
    },
    {
        "id": "timeline_flow",
        "title": "Timeline Flow",
        "description": "Sequential narrative flow with numbered stages and explicit directional arrows.",
        "dsl": """[scene]
[canvas:wide]
[layout:timeline_flow]
[theme:signal_glow]
[tone:green]
[frame:card]
[density:balanced]
[inset:md]
[gap:md]
[hero:center]
[columns:4]
[emphasis:top]
[rail:none]
[background:rings]
[connector:arrow]
[topic:iteration_story]
[header_band:iteration_path|representation_first_loop|stabilize_target_then_train]
[stage_card:assets|audit_the_shipped_svgs]
[flow_arrow:assets->dsl]
[stage_card:dsl|define_scene_and_component_contract]
[flow_arrow:dsl->compiler]
[stage_card:compiler|prove_gold_asset_reconstruction]
[flow_arrow:compiler->training]
[stage_card:training|generate_variants_and_train_the_model]
[footer_note:sequence_for_reducing_wasted_compute]
[/scene]""",
    },
    {
        "id": "table_analysis",
        "title": "Table Analysis",
        "description": "Analytical table scene with header row, row-state highlighting, and a compiler-managed callout.",
        "dsl": """[scene]
[canvas:wide]
[layout:table_analysis]
[theme:paper_editorial]
[tone:amber]
[frame:panel]
[density:balanced]
[inset:md]
[gap:sm]
[hero:left]
[columns:1]
[emphasis:top]
[rail:none]
[background:none]
[connector:line]
[topic:validation_matrix]
[header_band:test_matrix|compiler_gate_summary|deterministic_checks_before_training]
[table_header:gate|status|note]
[table_row:svg_parse|pass|all_validation_cases_parse_as_xml|state=success|accent=green]
[table_row:layout_coverage|pass|all_7_scene_families_have_representative_cases|state=success|accent=green]
[table_row:theme_range|pass|dark_signal_and_paper_editorial_modes_render|state=highlight|accent=amber]
[table_row:repeated_components|pass|multiple_section_stage_and_row_blocks_compile|state=normal|accent=blue]
[table_row:remaining_gap|pending|asset_fidelity_still_needs_refinement|state=warning|accent=amber]
[callout_card:compiler_gate_first|only_train_after_output_quality_is_acceptable|accent=amber]
[footer_note:table_layout_for_gate_reports_and_audits]
[/scene]""",
    },
]


def _escape(value: object) -> str:
    return html.escape(str(value), quote=False)


def _tag_counts(svg: str) -> dict[str, int]:
    root = ET.fromstring(svg)
    counts: Counter[str] = Counter()
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1] if "}" in str(elem.tag) else str(elem.tag)
        counts[tag] += 1
    keep = ["svg", "defs", "linearGradient", "filter", "marker", "rect", "line", "path", "circle", "text", "tspan"]
    return {key: int(counts.get(key, 0)) for key in keep if counts.get(key, 0)}


def _layout_meta(case: dict[str, str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw in case["dsl"].split():
        if raw.startswith("[") and raw.endswith("]") and ":" in raw:
            key, value = raw[1:-1].split(":", 1)
            if key in {"canvas", "layout", "theme", "tone", "density", "background", "connector"}:
                meta[key] = value
    return meta


def _row(label: str, value: str) -> str:
    return f"<div class='kv'><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>"


def _case_panel(case: dict[str, Any], svg: str, tags: dict[str, int]) -> str:
    meta = _layout_meta(case)
    pills = "".join(f"<span class='pill'>{_escape(k)}:{_escape(v)}</span>" for k, v in meta.items())
    tag_text = "".join(f"<span class='tag'>{_escape(k)} {v}</span>" for k, v in tags.items())
    content = case.get("content")
    has_content = isinstance(content, dict)
    content_panel = ""
    grid_class = "grid grid-3" if has_content else "grid"
    if has_content:
        content_panel = f"""
        <div class="panel">
          <h3>Content JSON</h3>
          <pre>{_escape(json.dumps(content, indent=2, ensure_ascii=False))}</pre>
        </div>
        """
    return f"""
    <section class="case">
      <div class="head">
        <div>
          <h2>{_escape(case['title'])}</h2>
          <p>{_escape(case['description'])}</p>
        </div>
        <div class="status pass">PASS</div>
      </div>
      <div class="pills">{pills}</div>
      <div class="{grid_class}">
        <div class="panel">
          <h3>Scene DSL</h3>
          <pre>{_escape(case['dsl'])}</pre>
        </div>
        {content_panel}
        <div class="panel">
          <h3>Compiled Output</h3>
          <div class="svg-frame">{svg}</div>
        </div>
      </div>
      <div class="meta-grid">
        <div class="panel compact">
          <h3>Validation</h3>
          {_row("XML parse", "pass")}
          {_row("Rendered bytes", str(len(svg)))}
          {_row("Case id", case["id"])}
        </div>
        <div class="panel compact">
          <h3>Tag Coverage</h3>
          <div class="tags">{tag_text}</div>
        </div>
      </div>
    </section>
    """


def _build_html(cases: list[dict[str, Any]]) -> tuple[str, list[dict[str, object]]]:
    panels: list[str] = []
    manifest_cases: list[dict[str, object]] = []
    for case in cases:
        content = case.get("content")
        svg = render_structured_scene_spec09_svg(case["dsl"], content=content if isinstance(content, dict) else None)
        ET.fromstring(svg)
        tags = _tag_counts(svg)
        panels.append(_case_panel(case, svg, tags))
        manifest_cases.append(
            {
                "id": case["id"],
                "title": case["title"],
                "layout_meta": _layout_meta(case),
                "svg_valid": True,
                "svg_bytes": len(svg),
                "tag_counts": tags,
                "has_content": bool(isinstance(content, dict)),
            }
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Spec09 Compiler Validation</title>
  <style>
    :root {{
      --bg: #09111f;
      --panel: #121b29;
      --panel-2: #172233;
      --ink: #edf4fb;
      --muted: #8fa6bf;
      --line: #2a3850;
      --accent: #ffb400;
      --green: #47b475;
      --shadow: 0 20px 60px rgba(0,0,0,.32);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top, rgba(255,180,0,.08), transparent 24%),
        linear-gradient(180deg, #07101a, var(--bg));
    }}
    .wrap {{
      width: min(1480px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}
    .hero {{
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(180deg, rgba(18,27,41,.96), rgba(10,17,28,.96));
      box-shadow: var(--shadow);
      padding: 28px 30px;
      margin-bottom: 28px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 28px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .hero p + p {{
      margin-top: 8px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-top: 18px;
    }}
    .sum {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(23,34,51,.65);
      padding: 14px 16px;
    }}
    .sum .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .05em;
    }}
    .sum .v {{
      margin-top: 6px;
      font-size: 24px;
      font-weight: 700;
    }}
    .case {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: linear-gradient(180deg, rgba(18,27,41,.96), rgba(10,17,28,.92));
      box-shadow: var(--shadow);
      padding: 22px;
      margin-bottom: 24px;
    }}
    .head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .head h2 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    .head p {{
      margin: 0;
      color: var(--muted);
    }}
    .status {{
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .04em;
      white-space: nowrap;
    }}
    .status.pass {{
      background: rgba(71,180,117,.14);
      color: #a7e6bc;
      border: 1px solid rgba(71,180,117,.35);
    }}
    .pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .pill {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--line);
      background: rgba(86,174,252,.10);
      color: #acd6ff;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 0.95fr 1.25fr;
      gap: 18px;
    }}
    .grid.grid-3 {{
      grid-template-columns: 0.82fr 0.9fr 1.25fr;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(23,34,51,.95), rgba(11,18,30,.9));
      padding: 16px;
    }}
    .panel.compact {{
      min-height: 100%;
    }}
    .panel h3 {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      background: #0b1220;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: auto;
      color: #d9e6f7;
      font: 12.5px/1.48 "IBM Plex Mono", "SFMono-Regular", monospace;
      white-space: pre-wrap;
    }}
    .svg-frame {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #0a101a;
      padding: 10px;
      overflow: auto;
    }}
    .svg-frame svg {{
      display: block;
      width: 100%;
      min-width: 520px;
      height: auto;
    }}
    .meta-grid {{
      margin-top: 16px;
      display: grid;
      grid-template-columns: 0.8fr 1.2fr;
      gap: 18px;
    }}
    .kv {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 7px 0;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
    }}
    .kv strong {{
      color: var(--ink);
      font-weight: 600;
    }}
    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .tag {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: rgba(255,180,0,.10);
      color: #ffd98b;
    }}
    .foot {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 1100px) {{
      .summary, .grid, .meta-grid {{
        grid-template-columns: 1fr;
      }}
      .head {{
        flex-direction: column;
      }}
      .svg-frame svg {{
        min-width: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Spec09 Compiler Validation</h1>
      <p>This report shows what the current <code>spec09</code> compiler can actually emit from representative scene DSL documents.</p>
      <p>Each case is compiled, XML-validated, and rendered inline. This is the current compiler surface, not a promise about future training quality.</p>
      <div class="summary">
        <div class="sum"><div class="k">Cases</div><div class="v">{len(cases)}</div></div>
        <div class="sum"><div class="k">Valid SVG</div><div class="v">{len(cases)}/{len(cases)}</div></div>
        <div class="sum"><div class="k">Scene Families</div><div class="v">7</div></div>
        <div class="sum"><div class="k">Use</div><div class="v">Compiler Surface</div></div>
      </div>
    </section>
    {''.join(panels)}
    <p class="foot">Next step: compare these compiler outputs against the gold asset reconstructions and keep tightening the renderer until they are visually strong enough to seed training data.</p>
  </div>
</body>
</html>
"""
    return html_doc, manifest_cases


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a validation report for the spec09 scene compiler.")
    ap.add_argument("--out", required=True, help="Output HTML path.")
    ap.add_argument("--out-json", default=None, help="Optional JSON manifest path.")
    args = ap.parse_args()

    html_doc, manifest_cases = _build_html(VALIDATION_CASES)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")

    if args.out_json:
        manifest = {
            "schema": "ck.spec09_compiler_validation.v1",
            "cases_total": len(manifest_cases),
            "cases": manifest_cases,
        }
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] wrote HTML: {out}")
    if args.out_json:
        print(f"[OK] wrote JSON: {Path(args.out_json).expanduser().resolve()}")
    print(f"[OK] cases: {len(manifest_cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
