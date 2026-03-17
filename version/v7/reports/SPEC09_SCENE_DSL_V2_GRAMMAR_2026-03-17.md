# Spec09 Scene DSL v2 Grammar

This document turns the asset-library output into a concrete `spec09` target grammar.

The goal is to give the model a richer semantic/compositional interface while keeping raw SVG geometry, gradients, markers, filters, and typography behavior inside the compiler.

## Design Split

The model should control:

- canvas class
- scene family
- component presence and order
- content roles
- theme/tone
- high-level composition hints

The compiler should control:

- exact coordinates
- exact shape primitives
- path routing and control points
- marker defs
- gradient defs
- filter defs
- dash values
- font sizes and wrapping
- spacing scales in pixels

Raw SVG tags like `<rect>`, `<line>`, `<path>`, `<defs>`, `<linearGradient>`, `<marker>`, and `<filter>` are compiler-owned, not model-authored.

## Token Layers

### 1. Scene Tokens

- `canvas:wide|tall|square`
- `layout:comparison_span_chart|dashboard_cards|pipeline_lane|poster_stack|dual_panel_compare|timeline_flow|table_analysis`
- `theme:infra_dark|paper_editorial|signal_glow`
- `tone:amber|green|blue|purple|mixed`
- `density:compact|balanced|airy`
- `frame:none|card|panel`

### 2. Composition Hints

- `columns:1|2|3|4`
- `hero:left|center|split`
- `emphasis:top|left|center`
- `inset:sm|md|lg`
- `gap:sm|md|lg`
- `rail:none|muted|accent`
- `background:none|grid|mesh|rings`
- `connector:line|arrow|bracket|curve`

These are still semantic hints, not raw measurements.

### 3. Component Tokens

- `header_band`
- `section_card`
- `compare_bar`
- `metric_bar`
- `compare_panel`
- `callout_card`
- `annotation`
- `axis`
- `legend_row`
- `divider`
- `side_rail`
- `table_row`
- `table_header`
- `stage_card`
- `phase_divider`
- `flow_arrow`
- `curved_connector`
- `span_bracket`
- `floor_band`
- `badge_pill`
- `thesis_box`
- `conclusion_strip`
- `footer_note`

### 4. Content Roles

Content should resolve through named roles, not raw prose, wherever possible.

Core roles:

- `title`
- `subtitle`
- `kicker`
- `callout`
- `footer`
- `badge`
- `summary`

Comparison roles:

- `left_title`
- `left_value`
- `left_caption`
- `right_title`
- `right_value`
- `right_caption`
- `delta`

Table roles:

- `row_label`
- `row_value`
- `row_note`
- `column_label`

Flow roles:

- `step_title`
- `step_caption`
- `phase_label`
- `stage_label`

## Document Form

The outer contract stays close to `spec07/spec08`:

```text
[scene]
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
[header_band:kicker|title|subtitle]
[phase_divider:phase_left|phase_right]
[stage_card:stage1_title|stage1_caption]
[flow_arrow:stage1->stage2]
[stage_card:stage2_title|stage2_caption]
[flow_arrow:stage2->stage3]
[badge_pill:status]
[footer_note:footer]
[/scene]
```

The payload values can be either:

- inline semantic literals like `gpu_compute` or `fit_math_favors_cpu`
- external content references like `@title.headline` or `@rows.0.note`

That means the compiler can take:

1. a scene DSL document describing structure and roles
2. a JSON payload describing final text and data

and produce a fully populated SVG.

The exact component payload syntax can vary by component, but the grammar should stay canonical:

- one legal start token: `[scene]`
- one legal end token: `[/scene]`
- fixed order for scene-level attributes
- fixed component order within each layout family
- no aliases for the same semantic token

## Canonical Field Order

Every `spec09` scene document should follow this order:

1. `canvas`
2. `layout`
3. `theme`
4. `tone`
5. `frame`
6. `density`
7. `inset`
8. `gap`
9. `hero`
10. `columns`
11. `emphasis`
12. `rail`
13. `background`
14. `connector`
15. `topic`
16. layout-specific components

This is stricter than `spec08` because format control is part of the target.

## Layout Families

### `comparison_span_chart`

Use for contrast graphics where the point is the gap or span between systems, scales, or approaches.

Required components:

- `header_band`
- `metric_bar`
- `span_bracket`
- `thesis_box`
- `conclusion_strip`

Optional components:

- `table_row`
- `floor_band`
- `footer_note`

### `dashboard_cards`

Use for multi-panel summary cards, metric clusters, or wide overview boards.

Required components:

- `header_band`
- `section_card`

Minimum:

- at least 3 `section_card` or `metric_bar`-bearing blocks

Optional components:

- `badge_pill`
- `footer_note`
- `conclusion_strip`

### `pipeline_lane`

Use for staged process diagrams with explicit progression.

Required components:

- `header_band`
- `phase_divider`
- `stage_card`
- `flow_arrow`

Minimum:

- at least 3 `stage_card`
- at least 2 connectors

Optional components:

- `curved_connector`
- `badge_pill`
- `footer_note`

### `poster_stack`

Use for tall infographic posters with stacked sections.

Required components:

- `header_band`
- `section_card`

Minimum:

- at least 3 major `section_card`

Optional components:

- `side_rail`
- `metric_bar`
- `table_row`
- `badge_pill`
- `footer_note`

### `dual_panel_compare`

Use for paired left/right comparison boards.

Required components:

- `header_band`
- `section_card`

Minimum:

- exactly 2 dominant top-level comparison panels

Optional components:

- `metric_bar`
- `badge_pill`
- `footer_note`

### `timeline_flow`

Use for sequential narratives, transitions, or lifecycle diagrams.

Required components:

- `header_band`
- `stage_card`
- `flow_arrow`

Optional components:

- `curved_connector`
- `phase_divider`
- `footer_note`

### `table_analysis`

Use for row/column-heavy analytical explainers.

Required components:

- `header_band`
- `table_row`

Minimum:

- at least 4 `table_row`

Optional components:

- `section_card`
- `floor_band`
- `footer_note`

## Component Payload Forms

The compiler should parse payloads by component type.

### `header_band`

```text
[header_band:kicker|title|subtitle]
```

### `section_card`

```text
[section_card:title|body|callout|variant=hero|accent=amber]
```

### `compare_bar`

```text
[compare_bar:label|value|caption|accent=amber|note=fit_gap]
```

### `metric_bar`

```text
[metric_bar:label|value|caption]
```

### `compare_panel`

```text
[compare_panel:title|value|caption|variant=metric|accent=purple]
```

### `callout_card`

```text
[callout_card:title|note|accent=amber]
```

### `annotation`

```text
[annotation:label|note|accent=amber]
```

### `axis`

```text
[axis:label|note]
```

### `legend_row`

```text
[legend_row:amber=gpu_cluster|green=cpu_server]
```

### `divider`

```text
[divider:dash]
```

### `table_row`

```text
[table_row:row_label|row_value|row_note|state=highlight|accent=amber]
```

### `table_header`

```text
[table_header:column_1|column_2|column_3]
```

### `stage_card`

```text
[stage_card:stage_title|stage_caption]
```

### `phase_divider`

```text
[phase_divider:left_phase|right_phase]
```

### `flow_arrow`

```text
[flow_arrow:source->target]
```

### `curved_connector`

```text
[curved_connector:source->target]
```

### `span_bracket`

```text
[span_bracket:label|value]
```

### `floor_band`

```text
[floor_band:label]
```

### `badge_pill`

```text
[badge_pill:badge]
```

### `thesis_box`

```text
[thesis_box:title|support_1|support_2]
```

### `conclusion_strip`

```text
[conclusion_strip:summary]
```

### `footer_note`

```text
[footer_note:footer]
```

## Asset Mappings

These are the initial gold mappings the compiler should learn to support.

### `memory-reality-infographic.svg`

Target family: `poster_stack`

```text
[scene]
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
[header_band:kicker|title|subtitle]
[section_card:principle_title|principle_formula|principle_caption]
[section_card:capacity_title|gpu_capacity|cpu_capacity]
[section_card:fit_title|ctx_table|fit_summary]
[section_card:gpu_path|gpu_cost|cpu_path]
[badge_pill:cost_delta]
[footer_note:footer]
[/scene]
```

### `performance-balance.svg`

Target family: `comparison_span_chart`

```text
[scene]
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
[header_band:title|subtitle]
[compare_bar:gpu_compute|gpu_bandwidth|gpu_gap|accent=amber]
[compare_bar:cpu_compute|cpu_bandwidth|cpu_gap|accent=green]
[axis:bandwidth_axis|shared_floor]
[legend_row:amber=gpu_cluster|green=cpu_server]
[annotation:bottleneck_shift|memory_and_network_dominate|accent=amber]
[divider:dash]
[span_bracket:gpu_span|gpu_span_value]
[span_bracket:cpu_span|cpu_span_value]
[floor_band:shared_network_floor]
[thesis_box:gpu_claim|cpu_claim|physics_claim]
[conclusion_strip:summary]
[footer_note:footer]
[/scene]
```

## External Content Binding

`spec09` should support external data binding by resolving `@path` references against a JSON payload.

Example scene:

```text
[scene]
[canvas:wide]
[layout:comparison_span_chart]
[theme:infra_dark]
[tone:mixed]
[frame:card]
[density:balanced]
[inset:md]
[gap:md]
[hero:center]
[columns:3]
[emphasis:center]
[rail:none]
[background:mesh]
[connector:bracket]
[topic:live_balance]
[header_band:@title.kicker|@title.headline|@title.subtitle]
[compare_bar:@bars.gpu.label|@bars.gpu.value|@bars.gpu.caption|accent=@bars.gpu.accent|note=@bars.gpu.note]
[compare_bar:@bars.cpu.label|@bars.cpu.value|@bars.cpu.caption|accent=@bars.cpu.accent|note=@bars.cpu.note]
[thesis_box:@thesis.title|@thesis.line_1|@thesis.line_2]
[conclusion_strip:@summary.conclusion]
[footer_note:@summary.footer]
[/scene]
```

Example content payload:

```json
{
  "title": {
    "kicker": "live brief",
    "headline": "Compiler-Bound Scene + Data",
    "subtitle": "Structure and data can arrive from separate systems."
  },
  "bars": {
    "gpu": {
      "label": "GPU Path",
      "value": "67,000 GB/s eq",
      "caption": "5,360x span",
      "accent": "amber",
      "note": "HBM is fast but capacity bound."
    },
    "cpu": {
      "label": "CPU Path",
      "value": "1,800 GB/s eq",
      "caption": "144x span",
      "accent": "green",
      "note": "Fit and cost are closer to deployment reality."
    }
  },
  "thesis": {
    "title": "Compiler + content is the right split.",
    "line_1": "The model chooses structure and emphasis.",
    "line_2": "Another system can provide final data and copy."
  },
  "summary": {
    "conclusion": "The compiler renders the full infographic, not just a skeleton.",
    "footer": "This is the bridge from planning to production content."
  }
}
```

That gives us a clean production split:

- model emits composition and semantic roles
- another source emits content/data JSON
- compiler resolves both into final SVG

### `pipeline-overview.svg`

Target family: `pipeline_lane`

```text
[scene]
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
[header_band:kicker|title|subtitle]
[phase_divider:code_generation|runtime]
[stage_card:config_title|config_caption]
[flow_arrow:config->parse]
[stage_card:parse_title|parse_caption]
[flow_arrow:parse->generate]
[stage_card:generate_title|generate_caption]
[flow_arrow:generate->weights]
[stage_card:weights_title|weights_caption]
[flow_arrow:weights->compile]
[stage_card:compile_title|compile_caption]
[curved_connector:compile->run]
[stage_card:run_title|run_caption]
[badge_pill:status]
[footer_note:footer]
[/scene]
```

## Compiler Responsibilities

The compiler should infer and enforce:

- exact stage-card widths and heights
- connector routing
- arrowhead marker defs
- gradient defs
- shadow defs
- background motifs
- rounded radii
- font hierarchy
- wrapped text using `tspan`
- spacing and alignment rules
- canvas-specific size system

The model should not emit:

- path control points
- marker ids
- gradient stop values
- filter ids
- raw x/y coordinates
- raw font sizes

## Training Implications

This grammar should be validated before `spec10` training by hand-mapping 3 to 5 gold assets and confirming the compiler can round-trip them into acceptable SVG.

Only after that should we:

1. generate controlled synthetic variants
2. build probe contracts around the new families
3. preflight token budgets and effective epochs
4. train the model to emit the grammar

## Immediate Next Step

The next implementation target is not another training run.

It is:

1. choose 3 gold assets
2. write the scene DSL docs for those assets
3. build the `spec09` compiler to render them
4. reject the grammar if the rendered output still looks materially worse than the source assets
