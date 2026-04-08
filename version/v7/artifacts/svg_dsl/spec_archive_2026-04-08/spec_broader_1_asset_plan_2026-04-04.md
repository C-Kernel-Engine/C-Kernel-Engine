# Spec Broader 1 Asset Plan

## Why This Branch Exists

- Current best run is `spec19_scene_bundle_l3_d192_h384_ctx768_r3d_sft_b_instruction` with exact `0.8864` and renderable `0.9545`.
- Site assets total: `97`. Current gold DSL/compiler coverage: `13` assets.
- This broader branch targets `23` new assets across `6` families, bringing the target gold pack to `36` assets.

## Current Replay Base

- `bump_allocator_quant.svg`
- `compute-bandwidth-chasm.svg`
- `ir-pipeline-flow.svg`
- `ir-timeline-why.svg`
- `ir-v66-evolution-timeline.svg`
- `kernel-registry-flow.svg`
- `memory-layout-map.svg`
- `pipeline-overview.svg`
- `quantization-formats.svg`
- `rope-layouts-compared.svg`
- `sentencepiece-vs-bpe-wordpiece.svg`
- `tokenizer-performance-comparison.svg`
- `v7-train-memory-canary.svg`

## First-Wave Families

### `comparison_span_chart`

- Priority: `P0`
- Lineage: `existing_spec09_family`
- Why now: Strong fit for the largest uncovered comparison/chart bucket and already aligned with prior family work.
- DSL additions: `chart_axis, axis_tick, series_bar, series_line, range_band, threshold_marker, value_tag, legend_block, thesis_box`
- Compiler additions: `multi-series chart scaling, range-band rendering, threshold marker layout, paired legend placement, comparison callout anchoring`
- Selected assets:
  - `cpu-gpu-analysis.svg` -> `dual_bar_analysis` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `performance-balance.svg` -> `balance_curve_board` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `theory-of-constraints.svg` -> `bottleneck_span_board` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `scale-economics.svg` -> `economics_gap_chart` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `v7-cross-entropy-parity-map.svg` -> `parity_roadmap_board` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
### `table_matrix`

- Priority: `P0`
- Lineage: `new_family_promoted_from_spec12_vocab_draft`
- Why now: Dense row/column explainers remain absent from the current bundle contract and block a broad slice of the asset library.
- DSL additions: `table_block, table_header, table_column, table_cell, column_group, row_state, legend_block`
- Compiler additions: `true table grid layout, column group headers, row state highlighting, table legend anchoring, cell text wrapping`
- Selected assets:
  - `bf16_format.svg` -> `format_matrix` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `quantization_grouping.svg` -> `grouped_quant_table` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `quantization_overview.svg` -> `quantization_matrix` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `ir-v66-edge-case-matrix.svg` -> `edge_case_matrix` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
### `architecture_map`

- Priority: `P0`
- Lineage: `new_family_promoted_from_spec12_vocab_draft`
- Why now: System/dataflow/topology assets are a large missing bucket and exceed the current bounded system_diagram family.
- DSL additions: `topology_node, fabric_link, observer_panel, subsystem_cluster, network_band, annotation_callout, step_number`
- Compiler additions: `clustered node layout, orthogonal and bus routing, multi-zone topology bands, observer/callout placement, dense connector labeling`
- Selected assets:
  - `qwen_layer_dataflow.svg` -> `layer_dataflow_stack` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `ir-dataflow-stitching.svg` -> `stitched_dataflow_board` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `tokenizer-architecture.svg` -> `subsystem_architecture_map` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `architecture-overview.svg` -> `stacked_architecture_overview` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `rdma-observer-architecture.svg` -> `observer_topology_map` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
### `poster_stack`

- Priority: `P1`
- Lineage: `existing_spec09_family`
- Why now: Poster-style explainers account for a broad semantic range but can share one structured stacked family.
- DSL additions: `section_card, table_block, chart_axis, series_bar, note_band, annotation_callout, kpi_strip`
- Compiler additions: `stacked section composition, mixed table-and-chart sections, poster hero/header treatment, callout strip layout, section-local legends`
- Selected assets:
  - `activation-memory-infographic.svg` -> `memory_training_poster` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `memory-reality-infographic.svg` -> `resource_reality_poster` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `power-delivery-infographic.svg` -> `power_profile_poster` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `c-kernel-engine-overview.svg` -> `platform_overview_poster` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
### `dashboard_cards`

- Priority: `P1`
- Lineage: `existing_spec09_family`
- Why now: Board-style dashboards let us cover multi-panel training and planning assets without forcing them into comparison or poster families.
- DSL additions: `section_card, metric_bar, kpi_strip, status_pill, note_band, badge_pill`
- Compiler additions: `multi-panel dashboard placement, card-level accent styling, metric strip scaling, compact board legend support`
- Selected assets:
  - `training-intuition-map.svg` -> `training_map_dashboard` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `v6_plan.svg` -> `plan_dashboard` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `v6_plan_inkscape.svg` -> `plan_dashboard_variant` (P2; needs `dsl_extension, compiler_extension, gold_mapping`)
### `timeline_flow`

- Priority: `P1`
- Lineage: `spec09_family_adjacent_to_current_timeline_bundle`
- Why now: Timeline assets are not the largest gap, but the missing ones are structurally different enough to warrant explicit expansion.
- DSL additions: `phase_divider, stage_card, outcome_panel, branch_label, note_band`
- Compiler additions: `vertical ladder layout, phase band placement, timeline-plus-gate hybrid connectors, dense footer/callout handling`
- Selected assets:
  - `ir-v66-gate-ladder.svg` -> `gate_ladder` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
  - `ir-v66-runtime-modes.svg` -> `runtime_mode_timeline` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)

## Deferred Families

- `decision_tree`: High-value but currently underrepresented in the shipped asset set; better added after table/architecture coverage lands.
  - `ir-v66-failure-decision-tree.svg`
  - `ir-v66-test-gates.svg`
- `spectrum_map`: Needs curved semantic-continuum layout that is probably a separate compiler effort.
  - `operator-spectrum-map.svg`
  - `operator-spectrum-map-presentation.svg`
- `technical_diagram`: Kernel math, algorithm walkthroughs, and low-level tensor explainers likely need a denser technical family than the first broader wave.
  - `kernel-activations.svg`
  - `kernel-attention.svg`
  - `kernel-layernorm.svg`
  - `kernel-rmsnorm.svg`
  - `kernel-rope.svg`
  - `kernel-swiglu.svg`
  - `mega_fused_attention.svg`
  - `per_head_fusion_math.svg`
  - `sentencepiece-algorithm.svg`
  - `sentencepiece-space-handling.svg`
  - `sentencepiece-tricky-cases.svg`
  - `tokenizer-hash-vs-trie.svg`
  - `v7-residual-gqa-backward.svg`

## Training Gates

- Do not start broader training until every selected first-wave asset has a gold DSL seed and content separation stub.
- Do not start broader training until compiler smoke passes for every newly added family.
- Do not start broader training until tokenizer additions are frozen for the broader family/form surface.
- Do not start broader training until the first-wave asset library can be materialized into a deduped replay-plus-synthesis curriculum.

## Next Steps

- Author gold DSL seeds plus content.json packs for the selected first-wave assets.
- Extend the compiler family by family and keep smoke reports per family.
- Freeze tokenizer additions only after the broader family/form surface is materially covered.
- Generate a broader replay-plus-synthesis curriculum only after the first-wave gold pack compiles.
- Train the same architecture on the broader curriculum before running a capacity canary.
