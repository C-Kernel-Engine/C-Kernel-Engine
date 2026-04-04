# Spec12 DSL Compression Analysis

This note is the first compression pass for the `spec12` gold mappings.

The goal is to reduce scene verbosity before:

1. building the `spec12` tokenizer surface
2. choosing the next context length
3. launching any `spec12` training run

## Why This Exists

The first hand-written `spec12` gold mappings were structurally correct but too
verbose. They expressed every field-level binding as a separate scene token.

That makes the compiler contract explicit, but it pushes too much inferable
structure onto the model-facing sequence.

The right next move is:

- keep structure/content separation
- keep topology refs where they matter
- remove scene attributes the compiler can infer
- collapse field-by-field component payloads into object-level refs where the
  component schema is already known

## Compression Rules

### 1. Infer scene defaults from layout profiles

These fields should usually come from layout family defaults unless explicitly
overridden:

- `canvas`
- `frame`
- `inset`
- `gap`
- `hero`
- `columns`
- `emphasis`
- `rail`
- `background`
- `connector`

The model should still choose them when the prompt really asks for a deviation,
but gold parity should not force them into every scene line.

### 2. Replace field-by-field binding with object-level binding

Bad:

```text
[table_row]
[cell_ref:groups.0.rows.0.0]
[cell_ref:groups.0.rows.0.1]
[cell_ref:groups.0.rows.0.2]
[cell_ref:groups.0.rows.0.3]
[row_state:normal]
[/table_row]
```

Better:

```text
[table_block:groups.0]
```

The compiler already knows that a `table_block` contains:

- a title
- a caption
- a header row
- repeated rows
- per-row state

That structure belongs in the compiler schema, not in repeated scene tokens.

### 3. Keep topology refs where they carry real layout meaning

This should stay explicit:

```text
[decision_node:start|nodes.start]
[decision_edge:start->l2|edges.start_l2]
```

Because:

- `start` and `l2` are structural ids
- `nodes.start` and `edges.start_l2` are content refs

That dual-reference pattern is a feature, not noise.

### 4. Do not regress to literal-content payload rows

Compressed does not mean monolithic prose tokens.

Good:

```text
[memory_segment:layer_0|segments.layer_0]
```

Bad:

```text
[memory_segment:Layer_0|known_fixed_region|Q_K_V_O_and_MLP_live_at_fixed_offsets]
```

The first keeps structure and content separate. The second reintroduces the
`spec10` failure mode.

## Structural Compression

The following counts are line/character counts of the scene DSL documents, not
final tokenizer counts. They are still useful because they show how much
structural verbosity was removed before tokenization.

| Asset | Original Lines | Compact Lines | Original Chars | Compact Chars | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `quantization-formats` | `98` | `12` | `2197` | `224` | Table structure moved into `table_block` schema. |
| `failure-decision-tree` | `100` | `21` | `2257` | `647` | Topology/content refs kept, per-node field repetition removed. |
| `memory-layout-map` | `108` | `19` | `2979` | `593` | Segment/object refs replace repeated field bindings. |

## New Compact Gold Files

### Quantization Formats

- [quantization-formats.scene.compact.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/quantization-formats.scene.compact.dsl)
- [quantization-formats.content.compact.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/quantization-formats.content.compact.json)

### Failure Decision Tree

- [failure-decision-tree.scene.compact.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/failure-decision-tree.scene.compact.dsl)
- [failure-decision-tree.content.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/failure-decision-tree.content.json)

### Memory Layout Map

- [memory-layout-map.scene.compact.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/memory-layout-map.scene.compact.dsl)
- [memory-layout-map.content.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/memory-layout-map.content.json)

Compact manifest:

- [spec12_gold_mappings_compact_20260318.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/spec12_gold_mappings_compact_20260318.json)

## What This Means For Context

Do not carry the old `spec11` tokenizer counts forward mechanically. Those
counts were measuring the verbose scene form against the wrong tokenizer
surface.

The next valid sequence is:

1. build compiler parity on the compact gold mappings
2. define the `spec12` tokenizer surface around the compact grammar
3. re-measure token counts
4. choose `ctx=768` or `ctx=1024` from the actual measured distribution

## Immediate Next Step

Do not train yet.

The immediate next step is:

1. implement compiler support for the compact `table_matrix`
2. implement compiler support for the compact `decision_tree`
3. implement compiler support for the compact `memory_map`
4. generate the first `spec12` alignment/parity report from these compact
   mappings
