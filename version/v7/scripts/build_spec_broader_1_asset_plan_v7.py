#!/usr/bin/env python3
"""Build the first broader asset-to-DSL expansion plan from the site asset library."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_JSON = ROOT / "version" / "v7" / "reports" / "asset_dsl_coverage_audit_2026-04-04.json"
DEFAULT_JSON_OUT = ROOT / "version" / "v7" / "reports" / "spec_broader_1_asset_plan_2026-04-04.json"
DEFAULT_MD_OUT = ROOT / "version" / "v7" / "reports" / "spec_broader_1_asset_plan_2026-04-04.md"


FIRST_WAVE_FAMILIES: list[dict[str, Any]] = [
    {
        "family": "comparison_span_chart",
        "priority": "P0",
        "lineage": "existing_spec09_family",
        "why_now": "Strong fit for the largest uncovered comparison/chart bucket and already aligned with prior family work.",
        "dsl_additions": [
            "chart_axis",
            "axis_tick",
            "series_bar",
            "series_line",
            "range_band",
            "threshold_marker",
            "value_tag",
            "legend_block",
            "thesis_box",
        ],
        "compiler_additions": [
            "multi-series chart scaling",
            "range-band rendering",
            "threshold marker layout",
            "paired legend placement",
            "comparison callout anchoring",
        ],
        "assets": [
            {
                "asset": "cpu-gpu-analysis.svg",
                "proposed_form": "dual_bar_analysis",
                "priority": "P0",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Dual gradient comparison board with paired quantitative bars and column narrative.",
            },
            {
                "asset": "performance-balance.svg",
                "proposed_form": "balance_curve_board",
                "priority": "P0",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Hybrid balance board with glow/threshold treatment and asymmetric comparison emphasis.",
            },
            {
                "asset": "theory-of-constraints.svg",
                "proposed_form": "bottleneck_span_board",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Constraint-focused comparison with chained emphasis and bottleneck callout structure.",
            },
            {
                "asset": "scale-economics.svg",
                "proposed_form": "economics_gap_chart",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Span/gap narrative with cost/performance tradeoff treatment.",
            },
            {
                "asset": "v7-cross-entropy-parity-map.svg",
                "proposed_form": "parity_roadmap_board",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Panelized roadmap board with highlighted decision path and grouped text regions.",
            },
        ],
    },
    {
        "family": "table_matrix",
        "priority": "P0",
        "lineage": "new_family_promoted_from_spec12_vocab_draft",
        "why_now": "Dense row/column explainers remain absent from the current bundle contract and block a broad slice of the asset library.",
        "dsl_additions": [
            "table_block",
            "table_header",
            "table_column",
            "table_cell",
            "column_group",
            "row_state",
            "legend_block",
        ],
        "compiler_additions": [
            "true table grid layout",
            "column group headers",
            "row state highlighting",
            "table legend anchoring",
            "cell text wrapping",
        ],
        "assets": [
            {
                "asset": "bf16_format.svg",
                "proposed_form": "format_matrix",
                "priority": "P0",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Compact matrix explaining numeric format structure.",
            },
            {
                "asset": "quantization_grouping.svg",
                "proposed_form": "grouped_quant_table",
                "priority": "P0",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Grouped quantization layout that pressures column-group semantics.",
            },
            {
                "asset": "quantization_overview.svg",
                "proposed_form": "quantization_matrix",
                "priority": "P0",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Rich table/matrix explainer mixing headers, gradients, and directional annotation.",
            },
            {
                "asset": "ir-v66-edge-case-matrix.svg",
                "proposed_form": "edge_case_matrix",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Operational matrix for gate/edge-case coverage rather than a generic comparison board.",
            },
        ],
    },
    {
        "family": "architecture_map",
        "priority": "P0",
        "lineage": "new_family_promoted_from_spec12_vocab_draft",
        "why_now": "System/dataflow/topology assets are a large missing bucket and exceed the current bounded system_diagram family.",
        "dsl_additions": [
            "topology_node",
            "fabric_link",
            "observer_panel",
            "subsystem_cluster",
            "network_band",
            "annotation_callout",
            "step_number",
        ],
        "compiler_additions": [
            "clustered node layout",
            "orthogonal and bus routing",
            "multi-zone topology bands",
            "observer/callout placement",
            "dense connector labeling",
        ],
        "assets": [
            {
                "asset": "qwen_layer_dataflow.svg",
                "proposed_form": "layer_dataflow_stack",
                "priority": "P0",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Tall layered dataflow explainer with legend, arrows, and stage hierarchy.",
            },
            {
                "asset": "ir-dataflow-stitching.svg",
                "proposed_form": "stitched_dataflow_board",
                "priority": "P0",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Compact IR/dataflow board with grouped nodes and data binding relationships.",
            },
            {
                "asset": "tokenizer-architecture.svg",
                "proposed_form": "subsystem_architecture_map",
                "priority": "P0",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Previously identified in spec12 as a strong architecture_map seed.",
            },
            {
                "asset": "architecture-overview.svg",
                "proposed_form": "stacked_architecture_overview",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "High-level architecture board with subsystem grouping pressure.",
            },
            {
                "asset": "rdma-observer-architecture.svg",
                "proposed_form": "observer_topology_map",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Reserved in spec15b; now promoted because the broader branch needs true topology coverage.",
            },
        ],
    },
    {
        "family": "poster_stack",
        "priority": "P1",
        "lineage": "existing_spec09_family",
        "why_now": "Poster-style explainers account for a broad semantic range but can share one structured stacked family.",
        "dsl_additions": [
            "section_card",
            "table_block",
            "chart_axis",
            "series_bar",
            "note_band",
            "annotation_callout",
            "kpi_strip",
        ],
        "compiler_additions": [
            "stacked section composition",
            "mixed table-and-chart sections",
            "poster hero/header treatment",
            "callout strip layout",
            "section-local legends",
        ],
        "assets": [
            {
                "asset": "activation-memory-infographic.svg",
                "proposed_form": "memory_training_poster",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Previously named as a poster_stack candidate in the spec12 vocabulary draft.",
            },
            {
                "asset": "memory-reality-infographic.svg",
                "proposed_form": "resource_reality_poster",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Was held out of strict memory_map; fits better as a poster-style explainer.",
            },
            {
                "asset": "power-delivery-infographic.svg",
                "proposed_form": "power_profile_poster",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Previously named in spec12 as a poster_stack seed with explicit charting pressure.",
            },
            {
                "asset": "c-kernel-engine-overview.svg",
                "proposed_form": "platform_overview_poster",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Hero-overview poster with large focal region and stacked explanation panels.",
            },
        ],
    },
    {
        "family": "dashboard_cards",
        "priority": "P1",
        "lineage": "existing_spec09_family",
        "why_now": "Board-style dashboards let us cover multi-panel training and planning assets without forcing them into comparison or poster families.",
        "dsl_additions": [
            "section_card",
            "metric_bar",
            "kpi_strip",
            "status_pill",
            "note_band",
            "badge_pill",
        ],
        "compiler_additions": [
            "multi-panel dashboard placement",
            "card-level accent styling",
            "metric strip scaling",
            "compact board legend support",
        ],
        "assets": [
            {
                "asset": "training-intuition-map.svg",
                "proposed_form": "training_map_dashboard",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Dense training process board with multiple grouped panels and pathway emphasis.",
            },
            {
                "asset": "v6_plan.svg",
                "proposed_form": "plan_dashboard",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Planning board with multiple cards and summary structure.",
            },
            {
                "asset": "v6_plan_inkscape.svg",
                "proposed_form": "plan_dashboard_variant",
                "priority": "P2",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Variant of the plan board; useful for style variance after the base form exists.",
            },
        ],
    },
    {
        "family": "timeline_flow",
        "priority": "P1",
        "lineage": "spec09_family_adjacent_to_current_timeline_bundle",
        "why_now": "Timeline assets are not the largest gap, but the missing ones are structurally different enough to warrant explicit expansion.",
        "dsl_additions": [
            "phase_divider",
            "stage_card",
            "outcome_panel",
            "branch_label",
            "note_band",
        ],
        "compiler_additions": [
            "vertical ladder layout",
            "phase band placement",
            "timeline-plus-gate hybrid connectors",
            "dense footer/callout handling",
        ],
        "assets": [
            {
                "asset": "ir-v66-gate-ladder.svg",
                "proposed_form": "gate_ladder",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Tall gate ladder with stronger phase-band treatment than the current bounded timeline family.",
            },
            {
                "asset": "ir-v66-runtime-modes.svg",
                "proposed_form": "runtime_mode_timeline",
                "priority": "P1",
                "needs": ["dsl_extension", "compiler_extension", "gold_mapping"],
                "notes": "Timeline-like mode ladder with compact control-state transitions.",
            },
        ],
    },
]


DEFERRED_FAMILIES: list[dict[str, Any]] = [
    {
        "family": "decision_tree",
        "why_deferred": "High-value but currently underrepresented in the shipped asset set; better added after table/architecture coverage lands.",
        "candidate_assets": [
            "ir-v66-failure-decision-tree.svg",
            "ir-v66-test-gates.svg",
        ],
    },
    {
        "family": "spectrum_map",
        "why_deferred": "Needs curved semantic-continuum layout that is probably a separate compiler effort.",
        "candidate_assets": [
            "operator-spectrum-map.svg",
            "operator-spectrum-map-presentation.svg",
        ],
    },
    {
        "family": "technical_diagram",
        "why_deferred": "Kernel math, algorithm walkthroughs, and low-level tensor explainers likely need a denser technical family than the first broader wave.",
        "candidate_assets": [
            "kernel-activations.svg",
            "kernel-attention.svg",
            "kernel-layernorm.svg",
            "kernel-rmsnorm.svg",
            "kernel-rope.svg",
            "kernel-swiglu.svg",
            "mega_fused_attention.svg",
            "per_head_fusion_math.svg",
            "sentencepiece-algorithm.svg",
            "sentencepiece-space-handling.svg",
            "sentencepiece-tricky-cases.svg",
            "tokenizer-hash-vs-trie.svg",
            "v7-residual-gqa-backward.svg",
        ],
    },
]


TRAINING_GATES = [
    "Do not start broader training until every selected first-wave asset has a gold DSL seed and content separation stub.",
    "Do not start broader training until compiler smoke passes for every newly added family.",
    "Do not start broader training until tokenizer additions are frozen for the broader family/form surface.",
    "Do not start broader training until the first-wave asset library can be materialized into a deduped replay-plus-synthesis curriculum.",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_family_plan(raw: dict[str, Any], missing_assets: set[str]) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    for asset_doc in raw.get("assets") or []:
        asset_name = str(asset_doc["asset"]).strip()
        if asset_name not in missing_assets:
            raise ValueError(f"first-wave asset {asset_name!r} is not in the current audit missing set")
        assets.append(
            {
                "asset": asset_name,
                "path": f"docs/site/assets/{asset_name}",
                "proposed_form": asset_doc["proposed_form"],
                "priority": asset_doc["priority"],
                "needs": list(asset_doc["needs"]),
                "notes": asset_doc["notes"],
            }
        )
    return {
        "family": raw["family"],
        "priority": raw["priority"],
        "lineage": raw["lineage"],
        "why_now": raw["why_now"],
        "dsl_additions": list(raw["dsl_additions"]),
        "compiler_additions": list(raw["compiler_additions"]),
        "assets": assets,
        "asset_count": len(assets),
    }


def build_plan(audit: dict[str, Any]) -> dict[str, Any]:
    covered_assets = list(audit.get("covered_assets") or [])
    missing_assets = set(audit.get("missing_assets") or [])
    first_wave_families = [_normalize_family_plan(doc, missing_assets) for doc in FIRST_WAVE_FAMILIES]

    first_wave_assets = [
        asset
        for family in first_wave_families
        for asset in family["assets"]
    ]

    family_counts = {family["family"]: family["asset_count"] for family in first_wave_families}
    all_selected_names = [asset["asset"] for asset in first_wave_assets]
    if len(all_selected_names) != len(set(all_selected_names)):
        raise ValueError("duplicate asset selected in first-wave broader plan")

    return {
        "schema": "ck.spec_broader_1_asset_plan.v1",
        "generated_on": str(date.today()),
        "branch": "spec_broader_1",
        "goal": "Expand the DSL/compiler/tokenizer around a broader site-asset library before launching the next large-scale training line.",
        "baseline": {
            "current_best_run": "spec19_scene_bundle_l3_d192_h384_ctx768_r3d_sft_b_instruction",
            "current_best_exact": 39 / 44,
            "current_best_renderable": 42 / 44,
            "site_asset_count": int(audit.get("asset_count") or 0),
            "current_gold_covered_assets": int(audit.get("covered_count") or 0),
            "current_gold_missing_assets": int(audit.get("missing_count") or 0),
            "current_gold_coverage_rate": float(audit.get("coverage_rate") or 0.0),
        },
        "current_bundle_replay_assets": [
            {
                "asset": asset_name,
                "path": f"docs/site/assets/{asset_name}",
            }
            for asset_name in covered_assets
        ],
        "first_wave": {
            "family_count": len(first_wave_families),
            "new_asset_count": len(first_wave_assets),
            "post_wave_target_gold_asset_count": len(covered_assets) + len(first_wave_assets),
            "family_asset_counts": family_counts,
            "families": first_wave_families,
        },
        "deferred_families": DEFERRED_FAMILIES,
        "training_gates": TRAINING_GATES,
        "next_steps": [
            "Author gold DSL seeds plus content.json packs for the selected first-wave assets.",
            "Extend the compiler family by family and keep smoke reports per family.",
            "Freeze tokenizer additions only after the broader family/form surface is materially covered.",
            "Generate a broader replay-plus-synthesis curriculum only after the first-wave gold pack compiles.",
            "Train the same architecture on the broader curriculum before running a capacity canary.",
        ],
    }


def _render_markdown(plan: dict[str, Any]) -> str:
    baseline = plan["baseline"]
    first_wave = plan["first_wave"]
    lines: list[str] = []
    lines.append("# Spec Broader 1 Asset Plan")
    lines.append("")
    lines.append("## Why This Branch Exists")
    lines.append("")
    lines.append(
        f"- Current best run is `{baseline['current_best_run']}` with exact `{baseline['current_best_exact']:.4f}` and renderable `{baseline['current_best_renderable']:.4f}`."
    )
    lines.append(
        f"- Site assets total: `{baseline['site_asset_count']}`. Current gold DSL/compiler coverage: `{baseline['current_gold_covered_assets']}` assets."
    )
    lines.append(
        f"- This broader branch targets `{first_wave['new_asset_count']}` new assets across `{first_wave['family_count']}` families, bringing the target gold pack to `{first_wave['post_wave_target_gold_asset_count']}` assets."
    )
    lines.append("")
    lines.append("## Current Replay Base")
    lines.append("")
    for asset_doc in plan["current_bundle_replay_assets"]:
        lines.append(f"- `{asset_doc['asset']}`")
    lines.append("")
    lines.append("## First-Wave Families")
    lines.append("")
    for family in first_wave["families"]:
        lines.append(f"### `{family['family']}`")
        lines.append("")
        lines.append(f"- Priority: `{family['priority']}`")
        lines.append(f"- Lineage: `{family['lineage']}`")
        lines.append(f"- Why now: {family['why_now']}")
        lines.append(f"- DSL additions: `{', '.join(family['dsl_additions'])}`")
        lines.append(f"- Compiler additions: `{', '.join(family['compiler_additions'])}`")
        lines.append("- Selected assets:")
        for asset in family["assets"]:
            lines.append(
                f"  - `{asset['asset']}` -> `{asset['proposed_form']}` ({asset['priority']}; needs `{', '.join(asset['needs'])}`)"
            )
    lines.append("")
    lines.append("## Deferred Families")
    lines.append("")
    for family in plan["deferred_families"]:
        lines.append(f"- `{family['family']}`: {family['why_deferred']}")
        for asset in family["candidate_assets"]:
            lines.append(f"  - `{asset}`")
    lines.append("")
    lines.append("## Training Gates")
    lines.append("")
    for gate in plan["training_gates"]:
        lines.append(f"- {gate}")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    for step in plan["next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the first broader asset-to-DSL expansion plan.")
    ap.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON, help="Coverage-audit JSON input path.")
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="Output JSON path.")
    ap.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT, help="Output Markdown path.")
    args = ap.parse_args()

    audit = _load_json(args.audit_json.expanduser().resolve())
    plan = build_plan(audit)

    json_out = args.json_out.expanduser().resolve()
    md_out = args.md_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(_render_markdown(plan), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
