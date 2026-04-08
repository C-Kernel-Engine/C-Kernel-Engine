# Spec Broader 1 Family Pack: architecture_map

- Priority: `P0`
- Lineage: `new_family_promoted_from_spec12_vocab_draft`
- Renderer status: `new_renderer_family_required`
- Renderer layout: `new family required`
- Gold seed authoring: `blocked_on_compiler_family`
- Why now: System/dataflow/topology assets are a large missing bucket and exceed the current bounded system_diagram family.

## Replay References

- `pipeline-overview.svg`
- `ir-pipeline-flow.svg`
- `kernel-registry-flow.svg`

## DSL Additions

- `topology_node`
- `fabric_link`
- `observer_panel`
- `subsystem_cluster`
- `network_band`
- `annotation_callout`
- `step_number`

## Compiler Additions

- `clustered node layout`
- `orthogonal and bus routing`
- `multi-zone topology bands`
- `observer/callout placement`
- `dense connector labeling`

## Selected Assets

- `qwen_layer_dataflow.svg` -> `layer_dataflow_stack` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
- `ir-dataflow-stitching.svg` -> `stitched_dataflow_board` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
- `tokenizer-architecture.svg` -> `subsystem_architecture_map` (P0; needs `dsl_extension, compiler_extension, gold_mapping`)
- `architecture-overview.svg` -> `stacked_architecture_overview` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)
- `rdma-observer-architecture.svg` -> `observer_topology_map` (P1; needs `dsl_extension, compiler_extension, gold_mapping`)

## Immediate Next Steps

- Design the `architecture_map` family DSL before authoring training rows.
- Implement the new compiler family and prove smoke parity on at least one representative asset.
- Only after the new family renders correctly should gold seed authoring and tokenizer freezing begin.

