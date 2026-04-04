# Spec Broader 1 Family Pack: dashboard_cards

- Priority: `P1`
- Lineage: `existing_spec09_family`
- Renderer status: `existing_renderer_family`
- Renderer layout: `dashboard_cards`
- Gold seed authoring: `can_start_now`
- Why now: Board-style dashboards let us cover multi-panel training and planning assets without forcing them into comparison or poster families.

## Replay References

- none yet; this family is new to the current covered library

## DSL Additions

- `section_card`
- `metric_bar`
- `kpi_strip`
- `status_pill`
- `note_band`
- `badge_pill`

## Compiler Additions

- `multi-panel dashboard placement`
- `card-level accent styling`
- `metric strip scaling`
- `compact board legend support`

## Selected Assets

- `training-intuition-map.svg` -> `training_map_dashboard` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `v6_plan.svg` -> `plan_dashboard` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `v6_plan_inkscape.svg` -> `plan_dashboard_variant` (P2; needs `dsl_extension, compiler_extension, gold_mapping`)

## Immediate Next Steps

- Author 2 to 3 gold dashboard_cards DSL seeds for the highest-priority assets in this pack.
- Compile them through the existing renderer and write a family smoke report.
- Only after smoke passes, widen the family DSL with the listed additions.

