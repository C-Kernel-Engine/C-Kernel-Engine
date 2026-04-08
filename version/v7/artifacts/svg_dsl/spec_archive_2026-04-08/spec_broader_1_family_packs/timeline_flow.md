# Spec Broader 1 Family Pack: timeline_flow

- Priority: `P1`
- Lineage: `spec09_family_adjacent_to_current_timeline_bundle`
- Renderer status: `existing_renderer_family`
- Renderer layout: `timeline_flow`
- Gold seed authoring: `can_start_now`
- Why now: Timeline assets are not the largest gap, but the missing ones are structurally different enough to warrant explicit expansion.

## Replay References

- `ir-v66-evolution-timeline.svg`
- `ir-timeline-why.svg`

## DSL Additions

- `phase_divider`
- `stage_card`
- `outcome_panel`
- `branch_label`
- `note_band`

## Compiler Additions

- `vertical ladder layout`
- `phase band placement`
- `timeline-plus-gate hybrid connectors`
- `dense footer/callout handling`

## Selected Assets

- `ir-v66-gate-ladder.svg` -> `gate_ladder` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `ir-v66-runtime-modes.svg` -> `runtime_mode_timeline` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)

## Immediate Next Steps

- Author 2 to 3 gold timeline_flow DSL seeds for the highest-priority assets in this pack.
- Compile them through the existing renderer and write a family smoke report.
- Only after smoke passes, widen the family DSL with the listed additions.

