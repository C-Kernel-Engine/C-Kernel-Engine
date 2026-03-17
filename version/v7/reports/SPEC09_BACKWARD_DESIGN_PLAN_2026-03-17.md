# Spec09 Backward Design Plan

The next infographic line should be designed backward from the production assets we want to ship, not forward from the current synthetic training grid.

## Principle

1. Start from real shipped SVG assets in `docs/site/assets/`.
2. Extract the scene/component/style vocabulary those assets actually use.
3. Build a scene DSL that can express that vocabulary.
4. Build or extend the deterministic compiler until hand-mapped gold assets compile back into acceptable SVG.
5. Only then generate training data and train the model to emit the scene DSL.

Training comes after compiler expressivity is proven.

## New Tooling

- Script: [build_spec09_asset_library_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/build_spec09_asset_library_v7.py)
- Grammar: [SPEC09_SCENE_DSL_V2_GRAMMAR_2026-03-17.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC09_SCENE_DSL_V2_GRAMMAR_2026-03-17.md)
- Compiler: [render_svg_structured_scene_spec09_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/render_svg_structured_scene_spec09_v7.py)
- Validation report builder: [build_spec09_compiler_validation_report_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/build_spec09_compiler_validation_report_v7.py)
- Training playbook: [SPEC10_DSL_TRAINING_PLAYBOOK_2026-03-17.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC10_DSL_TRAINING_PLAYBOOK_2026-03-17.md)

The script scans `docs/site/assets/*.svg` and emits:

- scene-family candidates
- reusable component tokens
- compiler-owned style/effect tokens
- a `spec09` seed vocabulary grounded in shipped assets

Suggested usage:

```bash
python3 version/v7/scripts/build_spec09_asset_library_v7.py \
  --out-json /tmp/spec09_asset_library.json \
  --out-md /tmp/spec09_asset_library.md
```

## Current Seed Vocabulary

Scene families:

- `comparison_span_chart`
- `dashboard_cards`
- `pipeline_lane`
- `poster_stack`
- `dual_panel_compare`
- `timeline_flow`
- `table_analysis`

Components:

- `header_band`
- `section_card`
- `side_rail`
- `metric_bar`
- `table_row`
- `stage_card`
- `phase_divider`
- `flow_arrow`
- `curved_connector`
- `floor_band`
- `badge_pill`
- `thesis_box`
- `conclusion_strip`
- `footer_note`

Compiler-owned style tokens:

- `paint:gradient_linear`
- `effect:drop_shadow`
- `effect:glow`
- `connector:arrow_marker`
- `connector:dashed`
- `surface:rounded_card`
- `background:grid_pattern`
- `background:dark_canvas`
- `background:light_canvas`
- `accent:amber`
- `accent:green`
- `accent:blue`
- `accent:mixed`

## Spec09 Workflow

1. Pick 3 to 5 gold assets from different scene families.
2. Hand-map each asset into the new scene DSL.
3. Extend the compiler until those gold assets round-trip cleanly.
4. Add controlled variants around those gold assets:
   topic swap, tone swap, density swap, composition swap.
5. Build training corpora only after step 3 is visually acceptable.
6. Keep deterministic gates:
   parse, compile, render, canary, non-regression by family.

## Why This Is Better

- It aligns the DSL with the visuals we actually want.
- It keeps gradients, markers, filters, and spacing in the compiler.
- It reduces the chance of spending compute on a target representation that cannot produce production-quality output.
- It makes spec10+ a capability scaling problem, not a target-design guessing problem.

For the concrete training-side workflow, dataset shape, and readiness gates, see:

- [SPEC10_DSL_TRAINING_PLAYBOOK_2026-03-17.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC10_DSL_TRAINING_PLAYBOOK_2026-03-17.md)
