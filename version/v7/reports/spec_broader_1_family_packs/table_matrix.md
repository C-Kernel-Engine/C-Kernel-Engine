# Spec Broader 1 Family Pack: table_matrix

- Priority: `P0`
- Lineage: `new_family_promoted_from_spec12_vocab_draft`
- Renderer status: `precursor_renderer_family`
- Renderer layout: `table_analysis`
- Gold seed authoring: `can_start_with_precursor`
- Why now: Dense row/column explainers remain absent from the current bundle contract and block a broad slice of the asset library.

## Replay References

- `quantization-formats.svg`

## DSL Additions

- `table_block`
- `table_header`
- `table_column`
- `table_cell`
- `column_group`
- `row_state`
- `legend_block`

## Compiler Additions

- `true table grid layout`
- `column group headers`
- `row state highlighting`
- `table legend anchoring`
- `cell text wrapping`

## Selected Assets

- `bf16_format.svg` -> `format_matrix` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
- `quantization_grouping.svg` -> `grouped_quant_table` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
- `quantization_overview.svg` -> `quantization_matrix` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
- `ir-v66-edge-case-matrix.svg` -> `edge_case_matrix` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)

## Immediate Next Steps

- Use the precursor renderer layout `table_analysis` to author exploratory gold seeds.
- Promote `table_matrix` to a first-class family only after the exploratory seeds reveal the missing compiler behavior.
- Write a family-smoke report that clearly separates precursor success from required new-family work.

