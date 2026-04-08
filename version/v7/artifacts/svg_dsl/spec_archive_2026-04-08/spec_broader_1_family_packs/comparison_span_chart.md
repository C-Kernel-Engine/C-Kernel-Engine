# Spec Broader 1 Family Pack: comparison_span_chart

- Priority: `P0`
- Lineage: `existing_spec09_family`
- Renderer status: `existing_renderer_family`
- Renderer layout: `comparison_span_chart`
- Gold seed authoring: `can_start_now`
- Why now: Strong fit for the largest uncovered comparison/chart bucket and already aligned with prior family work.

## Replay References

- `compute-bandwidth-chasm.svg`
- `quantization-formats.svg`
- `rope-layouts-compared.svg`
- `sentencepiece-vs-bpe-wordpiece.svg`
- `tokenizer-performance-comparison.svg`

## DSL Additions

- `chart_axis`
- `axis_tick`
- `series_bar`
- `series_line`
- `range_band`
- `threshold_marker`
- `value_tag`
- `legend_block`
- `thesis_box`

## Compiler Additions

- `multi-series chart scaling`
- `range-band rendering`
- `threshold marker layout`
- `paired legend placement`
- `comparison callout anchoring`

## Selected Assets

- `cpu-gpu-analysis.svg` -> `dual_bar_analysis` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
- `performance-balance.svg` -> `balance_curve_board` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
- `theory-of-constraints.svg` -> `bottleneck_span_board` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `scale-economics.svg` -> `economics_gap_chart` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `v7-cross-entropy-parity-map.svg` -> `parity_roadmap_board` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)

## Immediate Next Steps

- Author 2 to 3 gold comparison_span_chart DSL seeds for the highest-priority assets in this pack.
- Compile them through the existing renderer and write a family smoke report.
- Only after smoke passes, widen the family DSL with the listed additions.

