# Spec12 Compiler Guide

This note documents the first compiler-specific rules for `spec12`.

It exists to keep the new scene families understandable for future contributors
before training begins.

## Core Rule

`spec12` uses two different kinds of references:

1. topology references
2. content references

They are both necessary. They should not be collapsed into one thing.

## 1. Topology References

Topology references describe graph structure, routing, identity, and layout
relationships.

Examples:

```text
[node_id:start]
[panel_id:finish]
[segment_id:layer_0]
[from_ref:start]
[to_ref:l2]
```

These are not visible prose. They are structural handles the compiler uses to:

- place nodes
- route connectors
- attach labels to the correct edge
- preserve identity across repeated components
- keep tree/map layouts deterministic

## 2. Content References

Content references point into `content.json` and provide the visible text or
numeric payload.

Examples:

```text
[title_ref:nodes.start.title]
[body_ref:nodes.l2.body]
[branch_label_ref:edges.start_l2]
[caption_ref:segments.layer_0.caption]
```

These are what the compiler resolves into the final displayed text.

## Why Both Are Needed

Linear poster and panel layouts can sometimes get away with only content refs.
Graph/tree/map layouts cannot.

For example, in a decision tree:

```text
[decision_node]
[node_id:start]
[title_ref:nodes.start.title]
[/decision_node]

[decision_edge]
[from_ref:start]
[to_ref:l2]
[branch_label_ref:edges.start_l2]
[/decision_edge]
```

Here:

- `node_id:start` is structural identity
- `from_ref:start` and `to_ref:l2` are routing instructions
- `title_ref:nodes.start.title` is visible content
- `branch_label_ref:edges.start_l2` is visible edge label content

The compiler must not confuse routing ids with content keys.

## Compiler Responsibilities

For `spec12`, the compiler should own:

- node placement
- branch routing
- edge label placement
- memory-segment sizing and stacking
- table grid layout
- axis and tick placement
- legend placement
- grouped card and section composition

The model should not own:

- exact coordinates
- raw path geometry
- marker defs
- exact table cell widths
- exact edge bend points

## Tokenization Guidance

Topology refs should stay small and stable.

Good:

```text
[node_id:start]
[from_ref:start]
[to_ref:l2]
```

Bad:

```text
[decision_edge:start->l2|label=matrix_smoke]
```

The first form is more compositional and easier to reuse across assets.

## Training Implication

If a new family needs topology refs, add them to the DSL explicitly before
training. Do not hide them inside compiler heuristics and hope the model learns
them indirectly.

That is especially true for:

- `decision_tree`
- `architecture_map`
- `memory_map`
- `spectrum_map`

