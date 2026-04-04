# Spec12 Scene / Component Vocabulary Draft

This draft turns the next 10 concrete site assets into a practical `spec12`
scene-language expansion plan.

The goal is not to train immediately. The goal is to define the next richer
scene/component vocabulary, prove it through compiler parity on real assets, and
only then generate the next training corpus.

## Why Spec12 Exists

`spec11 r2` solved the current keyed-DSL contract:

- structure/content split works
- `scene.dsl + content.json + compiler -> SVG` works
- the current 5-family contract is clean

What it does not solve is production visual range. The shipped assets in
`docs/site/assets/` are still richer than the current training family. `spec12`
should therefore expand the scene language from real assets, not from synthetic
template growth.

## Selected 10-Asset Pack

This pack was chosen to pressure the missing parts of the compiler and scene
language.

| Asset | Proposed Spec12 Family | Why It Matters |
| --- | --- | --- |
| `docs/site/assets/activation-memory-infographic.svg` | `poster_stack` | Dense poster with table-like sections, bars, grouped sections, and header hierarchy. |
| `docs/site/assets/power-delivery-infographic.svg` | `poster_stack` | Poster with power-profile charting, callout strips, and mixed chart + narrative sections. |
| `docs/site/assets/ir-pipeline-flow.svg` | `pipeline_lane` | Tall, staged process flow with numbered steps, info cards, dividers, and guided progression. |
| `docs/site/assets/quantization-formats.svg` | `table_matrix` | High-density matrix/table asset with legend, repeated cells, and strong column structure. |
| `docs/site/assets/operator-spectrum-map-presentation.svg` | `spectrum_map` | Continuum/spectrum composition with poles, path-like progression, and directional emphasis. |
| `docs/site/assets/memory-layout-map.svg` | `memory_map` | Address-space layout with repeated memory regions, offsets, and fixed-map semantics. |
| `docs/site/assets/rdma-observer-architecture.svg` | `architecture_map` | Topology-style explainer with nodes, fabric links, observers, and callout regions. |
| `docs/site/assets/ir-v66-failure-decision-tree.svg` | `decision_tree` | Branching decision flow with labeled edges, entrypoint, outcomes, and triage structure. |
| `docs/site/assets/compute-bandwidth-chasm.svg` | `comparison_span_chart` | Rich comparison asset with bands, arrows, thresholds, thesis, and strong legend treatment. |
| `docs/site/assets/tokenizer-architecture.svg` | `architecture_map` | Multi-stage technical architecture with stage headers, connectors, and subsystem detail. |

## What The 10 Assets Teach

### Already Covered Reasonably Well

- `poster_stack`
- `pipeline_lane`
- `comparison_span_chart`

These should be expanded, not replaced.

### Missing Families That Should Become First-Class

- `table_matrix`
- `decision_tree`
- `architecture_map`
- `memory_map`
- `spectrum_map`

These are distinct enough that forcing them into the current 5-family contract
would make the DSL vague and the compiler brittle.

## Spec12 Scene Families

### Keep From Spec11

- `poster_stack`
- `comparison_span_chart`
- `pipeline_lane`
- `dashboard_cards`
- `dual_panel_compare`

### Add In Spec12

- `table_matrix`
  Use for dense row/column explainer graphics, format charts, and matrix
  comparisons.
- `decision_tree`
  Use for failure routing, gate ladders, diagnosis, and triage explainers.
- `architecture_map`
  Use for system topology, staged subsystems, links, observers, and node/fabric
  diagrams.
- `memory_map`
  Use for contiguous address-space layouts, memory regions, offsets, and fixed
  placement narratives.
- `spectrum_map`
  Use for continua, poles, traversable maps, and directional semantic ranges.

### Defer For A Later Asset Pack

- `timeline_flow`

It remains useful, but it is not the highest-pressure missing family in this
10-asset pack.

## Spec12 Component Additions

These are model-facing semantic components. They should be reusable across
families.

### Table / Matrix Components

- `table_block`
- `table_header`
- `table_column`
- `table_cell`
- `row_state`
- `column_group`
- `legend_block`

Why:
- `quantization-formats.svg` and the table-like poster assets need more than a
  repeated `table_row`.

### Chart / Quantitative Components

- `chart_axis`
- `axis_tick`
- `series_bar`
- `series_line`
- `range_band`
- `value_tag`
- `threshold_marker`

Why:
- `power-delivery-infographic.svg` and `compute-bandwidth-chasm.svg` need
  explicit quantitative scaffolding, not generic metric bars only.

### Decision Components

- `decision_node`
- `decision_edge`
- `branch_label`
- `outcome_panel`
- `entry_badge`

Why:
- `ir-v66-failure-decision-tree.svg` is not a pipeline lane. It needs true
  branching semantics.

### Architecture / Topology Components

- `topology_node`
- `fabric_link`
- `observer_panel`
- `network_band`
- `subsystem_cluster`

Why:
- `rdma-observer-architecture.svg` and `tokenizer-architecture.svg` need
  topology and subsystem grouping, not only linear stage cards.

### Memory Layout Components

- `memory_segment`
- `offset_marker`
- `address_strip`
- `region_bracket`
- `segment_note`

Why:
- `memory-layout-map.svg` has a fixed-map narrative that current cards/bars do
  not express cleanly.

### Spectrum Components

- `spectrum_pole`
- `spectrum_arc`
- `spectrum_node`
- `continuum_band`
- `direction_label`

Why:
- `operator-spectrum-map-presentation.svg` is about semantic distance and
  traversable space, not box-and-arrow stages.

### Shared Richness Components

- `annotation_callout`
- `status_pill`
- `formula_chip`
- `info_card`
- `kpi_strip`
- `note_band`
- `step_number`

Why:
- These recur across the asset pack and should be reusable instead of being
  embedded as one-off poster details.

## Compiler-Owned Expansions

These should not become model-authored geometry tokens.

The compiler should gain:

- real table grid layout
- axis/tick placement
- bar and line series scaling
- threshold and range-band rendering
- orthogonal and branching connector routing
- memory region sizing and offset labeling
- spectrum arc and pole layout
- subsystem clustering
- richer legend layout
- stronger text wrapping with `tspan` blocks
- reusable marker defs, gradients, and shadows

The model should still not emit:

- raw `x/y/width/height`
- raw path control points
- raw gradient stop values
- raw marker ids
- raw filter graph details

## Model-Facing Tokens For Spec12

The model should control:

- `canvas`
- `layout`
- `theme`
- `tone`
- `density`
- `frame`
- `columns`
- `hero`
- `emphasis`
- `rail`
- `background`
- `connector`
- component presence and order
- content references
- high-level layout hints such as:
  - `legend:top|bottom|side`
  - `axis_scale:linear|log|ordinal`
  - `branching:single|binary|fanout`
  - `map_orientation:horizontal|vertical`

The model should not control:

- pixel padding
- exact link routing
- exact tick positions
- exact cell widths
- exact memory segment sizes in pixels

## Content Contract

`spec12` should continue the `spec11` rule:

- structure lives in `scene.dsl`
- visible text and numeric data live in `content.json`
- compiler blends the two

The model should emit keyed references like:

```text
[table_cell:@table_block.0.rows.2.col_1]
[decision_node:@decision.0.start.title]
[memory_segment:@memory_map.0.segment_3.label]
```

It should not emit literal prose or full inline payload rows.

## Token Granularity Rule

Do not regress to full component-row reserved tokens.

Bad:

```text
[table_row:@row.0.a|@row.0.b|@row.0.c|state=highlight|accent=amber]
```

Better:

```text
[table_row]
[cell_ref:row.0.a]
[cell_ref:row.0.b]
[cell_ref:row.0.c]
[row_state:highlight]
[accent:amber]
[/table_row]
```

The exact syntax can still evolve, but `spec12` should reserve structure, not
payload.

## Example Drafts

### `table_matrix`

```text
[scene]
[canvas:wide]
[layout:table_matrix]
[theme:paper_editorial]
[tone:blue]
[frame:panel]
[density:compact]
[inset:md]
[gap:sm]
[hero:left]
[columns:1]
[emphasis:top]
[rail:none]
[background:grid]
[connector:line]
[topic:quantization_formats]
[header_band:@header.0.kicker|@header.0.headline|@header.0.subtitle]
[legend_block:@legend.0.title]
[table_header]
[cell_ref:table.header.0]
[cell_ref:table.header.1]
[cell_ref:table.header.2]
[/table_header]
[table_row]
[cell_ref:table.rows.0.c0]
[cell_ref:table.rows.0.c1]
[cell_ref:table.rows.0.c2]
[row_state:highlight]
[/table_row]
[footer_note:@footer.0.note]
[/scene]
```

### `decision_tree`

```text
[scene]
[canvas:wide]
[layout:decision_tree]
[theme:infra_dark]
[tone:amber]
[frame:panel]
[density:balanced]
[inset:md]
[gap:md]
[hero:center]
[columns:3]
[emphasis:top]
[rail:none]
[background:grid]
[connector:tree]
[topic:failure_triage]
[header_band:@header.0.kicker|@header.0.headline|@header.0.subtitle]
[entry_badge:@decision.0.entry]
[decision_node:@decision.0.start.title]
[decision_edge]
[from_ref:decision.0.start]
[to_ref:decision.0.branch_a]
[branch_label_ref:decision.0.edge_a]
[/decision_edge]
[decision_node:@decision.0.branch_a.title]
[outcome_panel:@decision.0.outcome_a.title|@decision.0.outcome_a.caption]
[footer_note:@footer.0.note]
[/scene]
```

## Training Implications

### Before Training

1. Hand-map the 10 assets into `scene.dsl + content.json`.
2. Compile them back to SVG.
3. Build an alignment report and reject weak parity before training.

### Dataset Rules

1. Mirror real assets with dummy text in the run workspace.
2. Train structure on keyed references, not literal prose.
3. Use repair rows for:
   - wrong layout
   - wrong component order
   - missing required blocks
   - wrong theme/tone
   - missing close tags

### Context-Length Rule

Do not raise context blindly.

For the current solved `spec11` family, `ctx=512` is still sufficient. For
`spec12`, first tokenize the new gold asset pack and measure:

- prompt tokens
- target tokens
- `p95(prompt + target)`
- `max(prompt + target)`

Then apply:

- keep `512` if `p95` stays below roughly `380-400`
- move to `768` if the new scenes land in the `400-600` range
- move to `1024` only if the gold pack proves it is necessary

### Primary Goal Of `spec12 r1`

Do not optimize first for benchmark exactness alone.

The first `spec12` success condition is:

- richer asset parity
- valid keyed structure/content separation
- no monolithic component payload tokens
- stable renderability across the new families

## Recommended Immediate Next Steps

1. Freeze `spec11 r2` as the current working keyed baseline.
2. Use the first hand-written gold mappings in `version/v7/reports/spec12_gold_mappings/` as the initial compiler parity targets.
3. Expand the compiler to support the new families/components.
4. Extend the gold pack from 3 mappings to the full 10-asset set.
5. Build a `spec12` alignment report before any real training run.
