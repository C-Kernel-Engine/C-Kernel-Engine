# Gen1 Full Coverage Plan

- Site assets: `97`
- Already covered: `13`
- Missing today: `84`
- Target gold-covered assets: `97`

## Phases

### `existing_and_precursor_families`

- Purpose: Cover the assets that can be expressed by extending or promoting existing renderer families.
- Asset count: `30`
- Families: `comparison_span_chart, table_matrix, dashboard_cards, poster_stack, memory_training_map, timeline_flow`

### `new_renderer_families`

- Purpose: Add the larger missing families that require new renderer/compiler work before training.
- Asset count: `52`
- Families: `concept_editorial, architecture_map, spectrum_map, decision_tree, technical_diagram`

### `special_cases`

- Purpose: Explicitly decide whether micrographic assets should enter the training surface or remain separate.
- Asset count: `2`
- Families: `utility_micrographic`

## Families

### `comparison_span_chart`

- Readiness: `existing_renderer_family`
- Scope: `extend_existing_family`
- Asset count: `7`
- `comparison-diagram.svg`
- `cpu-gpu-analysis.svg`
- `performance-balance.svg`
- `scale-economics.svg`
- `theory-of-constraints.svg`
- `v7-cross-entropy-parity-map.svg`
- `hybrid-bottleneck.svg`

### `table_matrix`

- Readiness: `precursor_renderer_family`
- Scope: `promote_precursor_to_first_class_family`
- Asset count: `5`
- `bf16_format.svg`
- `ir-v66-edge-case-matrix.svg`
- `quantization_grouping.svg`
- `quantization_overview.svg`
- `quantization_infographic.svg`

### `dashboard_cards`

- Readiness: `existing_renderer_family`
- Scope: `extend_existing_family`
- Asset count: `5`
- `training-intuition-map.svg`
- `v6_plan.svg`
- `v6_plan_inkscape.svg`
- `ir-v66-operator-cheatsheet.svg`
- `ir-output-artifacts.svg`

### `poster_stack`

- Readiness: `existing_renderer_family`
- Scope: `extend_existing_family`
- Asset count: `6`
- `activation-memory-infographic.svg`
- `memory-reality-infographic.svg`
- `power-delivery-infographic.svg`
- `c-kernel-engine-overview.svg`
- `commodity-hardware-pattern.svg`
- `infra-ocean-stability.svg`

### `concept_editorial`

- Readiness: `new_renderer_family`
- Scope: `new_family_required`
- Asset count: `9`
- `concept-flash-attention.svg`
- `concept-gqa.svg`
- `concept-rope-detailed.svg`
- `concept-transformer-overview.svg`
- `engineering-compass.svg`
- `iteration-philosophy.svg`
- `online_softmax_algorithm.svg`
- `two-principles.svg`
- `gpu-workaround-convergence.svg`

### `memory_training_map`

- Readiness: `precursor_renderer_family`
- Scope: `promote_memory_map_to_broader_family`
- Asset count: `5`
- `concept-weight-tying.svg`
- `ir-logic-vs-memory.svg`
- `v7-backprop-ir-stack.svg`
- `v7-grad-accum-window.svg`
- `weight_memory_layout.svg`

### `architecture_map`

- Readiness: `new_renderer_family`
- Scope: `new_family_required`
- Asset count: `11`
- `amx_pipeline.svg`
- `architecture-overview.svg`
- `ethernet-topology.svg`
- `forward-backward-flow.svg`
- `ir-dataflow-stitching.svg`
- `ir-kernel-registry-chain.svg`
- `ir-lowering-pipeline.svg`
- `ir_v2_pipeline.svg`
- `qwen_layer_dataflow.svg`
- `rdma-observer-architecture.svg`
- `tokenizer-architecture.svg`

### `spectrum_map`

- Readiness: `new_renderer_family`
- Scope: `new_family_required`
- Asset count: `2`
- `operator-spectrum-map.svg`
- `operator-spectrum-map-presentation.svg`

### `timeline_flow`

- Readiness: `existing_renderer_family`
- Scope: `extend_existing_family`
- Asset count: `2`
- `ir-v66-gate-ladder.svg`
- `ir-v66-runtime-modes.svg`

### `decision_tree`

- Readiness: `new_renderer_family`
- Scope: `new_family_required`
- Asset count: `2`
- `ir-v66-failure-decision-tree.svg`
- `ir-v66-test-gates.svg`

### `technical_diagram`

- Readiness: `new_renderer_family`
- Scope: `new_family_required`
- Asset count: `28`
- `bf16_rounding.svg`
- `bias_corrected_fusion.svg`
- `hardware_aware_blocking.svg`
- `int8_dot_product.svg`
- `ir-dumb-codegen.svg`
- `ir-fusion-vs-unfused.svg`
- `ir-json-walkthrough.svg`
- `ir-kernel-amp-strategy.svg`
- `ir-producer-consumer-qkv-rope-attn.svg`
- `ir-scratch-vs-persistent.svg`
- `ir-templates-to-ir.svg`
- `ir-v66-artifact-lineage.svg`
- `kernel-activations.svg`
- `kernel-attention.svg`
- `kernel-layernorm.svg`
- `kernel-rmsnorm.svg`
- `kernel-rope.svg`
- `kernel-swiglu.svg`
- `mega_fused_attention.svg`
- `per_head_fusion_math.svg`
- `q4_k_superblock.svg`
- `q4k_block_structure.svg`
- `q5_0_bit_layout.svg`
- `sentencepiece-algorithm.svg`
- `sentencepiece-space-handling.svg`
- `sentencepiece-tricky-cases.svg`
- `tokenizer-hash-vs-trie.svg`
- `v7-residual-gqa-backward.svg`

### `utility_micrographic`

- Readiness: `special_case_family`
- Scope: `small_visual_family_or_explicit_exclusion`
- Asset count: `2`
- `favicon.svg`
- `flamegraph.svg`

## Gates

- Freeze the broader tokenizer, DSL, compiler, and canonicalizer before the first gen1 full-mix run.
- Require gold DSL seeds and compiler smoke for every family included in the full-mix training set.
- Evaluate on held-out recombinations, not only prompt variants.

