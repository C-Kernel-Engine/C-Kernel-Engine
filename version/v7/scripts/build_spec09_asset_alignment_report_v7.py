#!/usr/bin/env python3
"""Build an HTML alignment report for the draft spec09 scene DSL."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from render_svg_structured_scene_spec09_v7 import render_structured_scene_spec09_svg

GOLD_CASES: list[dict[str, object]] = [
    {
        "asset": "memory-reality-infographic.svg",
        "title": "Memory Reality Infographic",
        "target_family": "poster_stack",
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
[header_band:First_Principle|LLM_Memory_Reality|The_math_marketing_wont_show_you]
[section_card:First_Principle|0_x_inf_eq_0|If_your_model_does_not_fit_in_memory_compute_speed_is_irrelevant|variant=hero|accent=amber]
[section_card:Memory_Capacity|25x_more_memory_capacity|Capacity_sets_the_real_context_boundary|variant=metric|accent=green]
[compare_bar:GPU_VRAM|80_GB|single_device|accent=red]
[compare_bar:CPU_RAM|2000_GB|single_server|accent=green]
[section_card:70B_Model_Plus_Context|KV_cache_plus_total_need|Rows_show_why_CPU_fit_is_the_deployment_boundary|variant=note|accent=blue]
[table_header:Context|GPU_Path|CPU_Path]
[table_row:8K|2x_GPUs|Fits|state=normal|accent=green]
[table_row:32K|2x_GPUs|Fits|state=normal|accent=green]
[table_row:128K|3x_GPUs|Fits|state=highlight|accent=amber]
[table_row:1M|6x_GPUs|Fits|state=warning|accent=red]
[compare_panel:GPU_Path|$240000_plus|6x_80GB_plus_NVLink_plus_network|variant=warning|accent=red]
[compare_panel:CPU_Path|$30000|1x_server_with_2TB_RAM|variant=success|accent=green]
[callout_card:8x_Lower_Cost|CPU_first_fit_is_cheaper_and_simpler|accent=amber]
[footer_note:C_Kernel_Engine_CPU_first_LLM_inference]
[/scene]""",
        "notes": [
            "This asset is a stacked poster, not a single card layout.",
            "The updated mapping uses richer poster internals: compare bars, table rows, compare panels, and a callout card.",
            "The compiler now owns the full poster composition instead of only stacked generic cards.",
        ],
    },
    {
        "asset": "performance-balance.svg",
        "title": "Performance Balance",
        "target_family": "comparison_span_chart",
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
[header_band:headline|The_Performance_Gap|Same_floor_very_different_spans]
[compare_bar:GPU_Compute|67000_GB_s_eq|5000x_total_span|accent=amber|note=HBM_is_fast_but_capacity_bound]
[compare_bar:CPU_Compute|1800_GB_s_eq|144x_total_span|accent=green|note=Fit_and_cost_are_closer_to_deployment_reality]
[axis:Log_scale_height|Same_ethernet_floor]
[legend_row:amber=GPU_cluster|green=CPU_server]
[annotation:Bottleneck_Shift|Memory_and_network_dominate|accent=amber]
[divider:dash]
[span_bracket:GPU_span|5360x]
[span_bracket:CPU_span|144x]
[floor_band:Ethernet_same_floor_for_every_cluster]
[thesis_box:GPU_structural_mismatch|CPU_gap_is_closeable|Optimize_for_the_constraint_not_the_headline]
[conclusion_strip:Match_FLOPs_to_data_movement_speed]
[footer_note:Physics_sets_the_budget]
[/scene]""",
        "notes": [
            "This is a comparison scene where the gap itself is the message.",
            "The richer mapping uses compare bars, axis, legend, annotation, divider, and the original bracket logic.",
            "This family is already the closest to the shipped asset language.",
        ],
    },
    {
        "asset": "pipeline-overview.svg",
        "title": "Pipeline Overview",
        "target_family": "pipeline_lane",
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
[header_band:from_config_to_running_model|C_Kernel_Engine_Pipeline|Code_generation_to_runtime]
[phase_divider:code_generation|runtime]
[stage_card:config|HuggingFace_style_input]
[flow_arrow:config->parse]
[stage_card:parse|Build_CKIR_graph]
[flow_arrow:parse->generate]
[stage_card:generate|Emit_clean_C_code]
[flow_arrow:generate->weights]
[stage_card:weights|Convert_to_runtime_format]
[flow_arrow:weights->compile]
[stage_card:compile|GCC_AVX512_AMX]
[curved_connector:compile->run]
[stage_card:run|Inference_plus_training]
[section_card:Supported_Configs|Llama_Mistral_SmolLM_Qwen|Any_HuggingFace_decoder_only|variant=note|accent=amber]
[section_card:Generated_Output|forward_layer_N_and_backward_layer_N|alloc_buffers_and_runtime_glue|variant=note|accent=amber]
[section_card:PyTorch_Parity|max_diff_lt_1e_5|kernel_checks_match_reference|variant=success|accent=green]
[badge_pill:Phase_1_Active]
[footer_note:PyTorch_parity_and_runtime_validation]
[/scene]""",
        "notes": [
            "This scene needs repeated stage cards plus both straight and curved connectors.",
            "The updated mapping adds the lower info-strip cards from the real asset.",
            "The compiler now uses stage-specific icon treatments and bottom summary cards to get closer to the shipped diagram.",
        ],
    },
    {
        "asset": "cpu-gpu-analysis.svg",
        "title": "CPU vs GPU Cost Analysis",
        "target_family": "dual_panel_compare",
        "dsl": """[scene]
[canvas:wide]
[layout:dual_panel_compare]
[theme:infra_dark]
[tone:mixed]
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
[topic:cpu_gpu_cost]
[header_band:0_x_inf_eq_0|The_Cost_Implication|GPU_memory_limits_force_multi_GPU_purchases]
[compare_panel:CPU_Path|$32000|1_server_done_start_training_today|variant=success|accent=green]
[compare_panel:GPU_Path|$250000_plus|8x_H100s_plus_fabric_and_network|variant=warning|accent=red]
[callout_card:Parallelism_works_but_now_you_must_buy_8_GPUs|Memory_fit_changes_the_purchase_boundary|accent=amber]
[footer_note:Fit_changes_both_topology_and_cost]
[/scene]""",
        "notes": [
            "This asset is a clean dual-path argument with cost as the dominant value layer.",
            "The current dual-panel family can already express the high-level structure without exposing raw geometry.",
        ],
    },
    {
        "asset": "theory-of-constraints.svg",
        "title": "Theory of Constraints",
        "target_family": "dual_panel_compare",
        "dsl": """[scene]
[canvas:wide]
[layout:dual_panel_compare]
[theme:infra_dark]
[tone:mixed]
[frame:panel]
[density:balanced]
[inset:md]
[gap:md]
[hero:split]
[columns:2]
[emphasis:left]
[rail:none]
[background:rings]
[connector:line]
[topic:ethernet_equalizer]
[header_band:Ethernet_Equalizer|At_scale_both_paths_hit_the_same_constraint|NVLink_does_not_scale_to_infinity]
[compare_panel:GPU_Cluster|900_GB_s_inside_node|50_GB_s_between_nodes_over_ethernet|variant=warning|accent=red]
[compare_panel:CPU_Cluster|460_GB_s_inside_node|50_GB_s_between_nodes_over_ethernet|variant=success|accent=green]
[callout_card:Same_network_constraint_15x_lower_cost|Choose_the_architecture_with_the_closable_gap|accent=amber]
[footer_note:Topology_becomes_the_equalizer]
[/scene]""",
        "notes": [
            "This is another strong paired-argument asset with a central thesis and mirrored left/right evidence.",
            "It tests whether the compare family can stay expressive across different subject matter without new raw SVG tokens.",
        ],
    },
    {
        "asset": "training-intuition-map.svg",
        "title": "Training Intuition Map",
        "target_family": "dashboard_cards",
        "dsl": """[scene]
[canvas:wide]
[layout:dashboard_cards]
[theme:infra_dark]
[tone:mixed]
[frame:panel]
[density:airy]
[inset:lg]
[gap:md]
[hero:left]
[columns:3]
[emphasis:top]
[rail:none]
[background:grid]
[connector:line]
[topic:training_intuition]
[header_band:Deep_Training_Intuition_Map|Observe_isolate_adjust_compare|Repeated_failure_and_recovery_builds_intuition]
[section_card:Observe_End_to_End|loss_curve_to_kernel_dispatch|trace_anomalies_across_the_full_stack|variant=hero|accent=amber]
[section_card:Log_Smart_Not_Heavy|tiny_analysis_checkpoints|full_resume_state_only_when_needed|variant=note|accent=blue]
[section_card:Learn_Then_Adjust|one_knob_per_run|compare_against_baseline_at_fixed_steps|variant=success|accent=green]
[section_card:Dense_Early|checkpoint_every_10_steps|capture_volatile_startup_behavior|variant=metric|accent=amber]
[section_card:Sparse_Later|checkpoint_every_200_steps|watch_long_phase_retention_and_drift|variant=metric|accent=blue]
[section_card:Gradient_Triage|observe_isolate_adjust_compare_record|failure_signatures_need_a_runbook|variant=note|accent=green]
[footer_note:Use_this_as_a_runbook_not_just_a_poster]
[/scene]""",
        "notes": [
            "This asset pushes the dashboard family toward higher-density editorial panels.",
            "It is useful for training because it exercises hierarchy, card variants, and multi-section summaries without needing raw SVG emission.",
        ],
    },
]


def _escape(value: object) -> str:
    return html.escape(str(value), quote=False)


def _load_asset_library(path: Path) -> dict[str, dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    entries = doc.get("assets") if isinstance(doc.get("assets"), list) else []
    return {str(row.get("name")): row for row in entries if isinstance(row, dict) and row.get("name")}


def _pill(token: str, kind: str) -> str:
    return f'<span class="pill {kind}">{_escape(token)}</span>'


def _pill_list(values: list[str], kind: str) -> str:
    if not values:
        return '<span class="muted">none</span>'
    return "".join(_pill(v, kind) for v in values)


def _notes(items: list[str]) -> str:
    return "".join(f"<li>{_escape(item)}</li>" for item in items)


def _coverage_table(asset: dict) -> str:
    candidates = asset.get("scene_family_candidates") if isinstance(asset.get("scene_family_candidates"), list) else []
    rows = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_escape(row.get('token'))}</td>"
            f"<td>{_escape(row.get('score'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="2" class="muted">none</td></tr>')
    return (
        "<table>"
        "<thead><tr><th>Scene Candidate</th><th>Score</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _asset_panel(case: dict[str, object], asset: dict[str, object], asset_svg: str) -> str:
    components = [str(v) for v in asset.get("component_tokens", [])] if isinstance(asset.get("component_tokens"), list) else []
    styles = [str(v) for v in asset.get("style_tokens", [])] if isinstance(asset.get("style_tokens"), list) else []
    palette = asset.get("palette") if isinstance(asset.get("palette"), dict) else {}
    accent = palette.get("accent_bucket", "unknown")
    content = case.get("content") if isinstance(case.get("content"), dict) else None
    compiled_svg = render_structured_scene_spec09_svg(str(case["dsl"]), content=content)

    return f"""
    <section class="case">
      <div class="case-head">
        <div>
          <h2>{_escape(case['title'])}</h2>
          <div class="meta">
            <span><strong>Asset</strong>: {_escape(case['asset'])}</span>
            <span><strong>Target Family</strong>: {_escape(case['target_family'])}</span>
            <span><strong>Detected Family</strong>: {_escape(asset.get('scene_family', 'unknown'))}</span>
            <span><strong>Accent</strong>: {_escape(accent)}</span>
          </div>
        </div>
        <div class="status active">Compiler Render Active</div>
      </div>
      <div class="grid">
        <div class="panel">
          <h3>Draft Spec09 DSL</h3>
          <pre>{_escape(case['dsl'])}</pre>
          <h4>Why This Mapping</h4>
          <ul>{_notes(case['notes'])}</ul>
        </div>
        <div class="panel">
          <h3>Compiled Spec09 Output</h3>
          <div class="svg-frame">{compiled_svg}</div>
        </div>
        <div class="panel">
          <h3>Real Asset</h3>
          <div class="svg-frame">{asset_svg}</div>
        </div>
      </div>
      <div class="grid lower">
        <div class="panel">
          <h3>Detected Component Coverage</h3>
          <div class="pill-wrap">{_pill_list(components, 'component')}</div>
        </div>
        <div class="panel">
          <h3>Detected Style Coverage</h3>
          <div class="pill-wrap">{_pill_list(styles, 'style')}</div>
        </div>
        <div class="panel">
          <h3>Scene Family Fit</h3>
          {_coverage_table(asset)}
        </div>
      </div>
    </section>
    """


def _build_html(cases: list[dict[str, object]], assets_by_name: dict[str, dict], assets_root: Path) -> str:
    sections: list[str] = []
    for case in cases:
        asset_name = str(case["asset"])
        asset = assets_by_name.get(asset_name)
        if asset is None:
            raise SystemExit(f"ERROR: asset not found in asset library: {asset_name}")
        asset_svg = (assets_root / asset_name).read_text(encoding="utf-8", errors="ignore")
        sections.append(_asset_panel(case, asset, asset_svg))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Spec09 DSL Asset Alignment</title>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: #111827;
      --panel-2: #172033;
      --ink: #e5edf7;
      --muted: #8aa0b8;
      --line: #2a3850;
      --accent: #fbbf24;
      --green: #47b475;
      --blue: #56aefc;
      --shadow: 0 20px 60px rgba(0,0,0,.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top, rgba(251,191,36,.08), transparent 28%),
        linear-gradient(180deg, #09111f, var(--bg));
      color: var(--ink);
      line-height: 1.45;
    }}
    .wrap {{
      width: min(1440px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}
    h1, h2, h3, h4 {{ margin: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    .hero {{
      background: linear-gradient(135deg, rgba(23,32,51,.92), rgba(13,17,23,.96));
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px 30px;
      margin-bottom: 28px;
    }}
    .hero h1 {{
      font-size: 28px;
      margin-bottom: 10px;
    }}
    .hero p + p {{ margin-top: 8px; }}
    .callout {{
      color: #ffd67a;
    }}
    .case {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(17,24,39,.96), rgba(11,18,30,.96));
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 22px;
      margin-bottom: 28px;
    }}
    .case-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      margin-bottom: 18px;
    }}
    .case-head h2 {{
      font-size: 22px;
      margin-bottom: 8px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    .status {{
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .03em;
      white-space: nowrap;
    }}
    .status.pending {{
      background: rgba(251,191,36,.12);
      color: #ffd67a;
      border: 1px solid rgba(251,191,36,.35);
    }}
    .status.active {{
      background: rgba(71,180,117,.14);
      color: #9fe0b5;
      border: 1px solid rgba(71,180,117,.35);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 0.9fr 1fr 1fr;
      gap: 18px;
    }}
    .grid.lower {{
      grid-template-columns: 1fr 1fr 1fr;
      margin-top: 18px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(23,32,51,.95), rgba(13,17,23,.9));
      padding: 18px;
      overflow: hidden;
    }}
    .panel h3 {{
      font-size: 16px;
      margin-bottom: 12px;
    }}
    .panel h4 {{
      font-size: 13px;
      color: var(--muted);
      margin: 14px 0 8px;
      text-transform: uppercase;
      letter-spacing: .05em;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      background: #0b1220;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: auto;
      color: #d7e4f5;
      font: 13px/1.5 "IBM Plex Mono", "SFMono-Regular", monospace;
      white-space: pre-wrap;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    li + li {{ margin-top: 6px; }}
    .svg-frame {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: #0a101a;
      padding: 10px;
      overflow: auto;
    }}
    .svg-frame svg {{
      width: 100%;
      height: auto;
      display: block;
      min-width: 520px;
    }}
    .pill-wrap {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--line);
    }}
    .pill.component {{
      background: rgba(86,174,252,.12);
      color: #9fd0ff;
    }}
    .pill.style {{
      background: rgba(71,180,117,.12);
      color: #94e0b0;
    }}
    .muted {{
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .foot {{
      margin-top: 22px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 1100px) {{
      .grid, .grid.lower {{
        grid-template-columns: 1fr;
      }}
      .case-head {{
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
      <h1>Spec09 DSL Asset Alignment</h1>
      <p>This report compares the draft <code>spec09_scene_dsl_v2</code> against real shipped SVG assets.</p>
      <p class="callout">Each case now includes an actual <code>spec09</code> compiler render beside the real asset. The question is no longer only “is the grammar expressive enough?” but also “does the compiler already feel directionally correct?”</p>
    </section>
    {''.join(sections)}
    <p class="foot">Next gate: tighten the compiler until these compiled outputs are visually acceptable enough to serve as gold reconstructions, then generate controlled training variants around them.</p>
  </div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a spec09 asset-vs-DSL alignment report.")
    ap.add_argument("--asset-library", required=True, help="Path to spec09 asset library JSON.")
    ap.add_argument("--assets-root", default="docs/site/assets", help="Directory containing real SVG assets.")
    ap.add_argument("--out", required=True, help="Output HTML path.")
    args = ap.parse_args()

    asset_library = Path(args.asset_library).expanduser().resolve()
    assets_root = Path(args.assets_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    assets_by_name = _load_asset_library(asset_library)
    html_doc = _build_html(GOLD_CASES, assets_by_name, assets_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")

    print(f"[OK] wrote: {out}")
    print(f"[OK] asset library: {asset_library}")
    print(f"[OK] assets root: {assets_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
