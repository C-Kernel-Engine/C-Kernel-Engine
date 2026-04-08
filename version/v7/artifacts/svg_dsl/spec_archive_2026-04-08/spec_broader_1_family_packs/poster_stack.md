# Spec Broader 1 Family Pack: poster_stack

- Priority: `P1`
- Lineage: `existing_spec09_family`
- Renderer status: `existing_renderer_family`
- Renderer layout: `poster_stack`
- Gold seed authoring: `can_start_now`
- Why now: Poster-style explainers account for a broad semantic range but can share one structured stacked family.

## Replay References

- none yet; this family is new to the current covered library

## DSL Additions

- `section_card`
- `table_block`
- `chart_axis`
- `series_bar`
- `note_band`
- `annotation_callout`
- `kpi_strip`

## Compiler Additions

- `stacked section composition`
- `mixed table-and-chart sections`
- `poster hero/header treatment`
- `callout strip layout`
- `section-local legends`

## Selected Assets

- `activation-memory-infographic.svg` -> `memory_training_poster` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `memory-reality-infographic.svg` -> `resource_reality_poster` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `power-delivery-infographic.svg` -> `power_profile_poster` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `c-kernel-engine-overview.svg` -> `platform_overview_poster` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)

## Immediate Next Steps

- Author 2 to 3 gold poster_stack DSL seeds for the highest-priority assets in this pack.
- Compile them through the existing renderer and write a family smoke report.
- Only after smoke passes, widen the family DSL with the listed additions.

