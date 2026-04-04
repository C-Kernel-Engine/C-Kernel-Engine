# Asset DSL Coverage Audit

## Summary

- Asset count: `97`
- Covered by current gold DSL generators: `13`
- Missing from current gold DSL generators: `84`
- Coverage rate: `0.1340`

## Covered Families

- `comparison_board_spec14a`: `5` assets
- `timeline_spec14b`: `2` assets
- `memory_map_spec15a`: `3` assets
- `system_diagram_spec15b`: `3` assets

## Missing By Bucket

- `board-chart-or-comparison`: `14`
- `editorial-or-concept`: `10`
- `memory-map-or-training`: `7`
- `system-diagram-or-flow`: `14`
- `timeline`: `2`
- `uncategorized`: `37`

## Covered Assets

- `bump_allocator_quant.svg` via `generate_svg_structured_spec15a_v7.py, generate_svg_structured_spec16_v7.py`
- `compute-bandwidth-chasm.svg` via `generate_svg_structured_spec14a_v7.py`
- `ir-pipeline-flow.svg` via `generate_svg_structured_spec15b_v7.py, generate_svg_structured_spec16_v7.py`
- `ir-timeline-why.svg` via `generate_svg_structured_spec14b_v7.py, generate_svg_structured_spec16_v7.py`
- `ir-v66-evolution-timeline.svg` via `generate_svg_structured_spec14b_v7.py, generate_svg_structured_spec16_v7.py`
- `kernel-registry-flow.svg` via `generate_svg_structured_spec15b_v7.py, generate_svg_structured_spec16_v7.py`
- `memory-layout-map.svg` via `generate_svg_structured_spec15a_v7.py, generate_svg_structured_spec16_v7.py`
- `pipeline-overview.svg` via `generate_svg_structured_spec15b_v7.py, generate_svg_structured_spec16_v7.py`
- `quantization-formats.svg` via `generate_svg_structured_spec14a_v7.py`
- `rope-layouts-compared.svg` via `generate_svg_structured_spec14a_v7.py`
- `sentencepiece-vs-bpe-wordpiece.svg` via `generate_svg_structured_spec14a_v7.py`
- `tokenizer-performance-comparison.svg` via `generate_svg_structured_spec14a_v7.py`
- `v7-train-memory-canary.svg` via `generate_svg_structured_spec15a_v7.py, generate_svg_structured_spec16_v7.py`

## Missing Assets

- `board-chart-or-comparison`
  - `bf16_format.svg`
  - `comparison-diagram.svg`
  - `cpu-gpu-analysis.svg`
  - `ir-v66-edge-case-matrix.svg`
  - `performance-balance.svg`
  - `quantization_grouping.svg`
  - `quantization_infographic.svg`
  - `quantization_overview.svg`
  - `scale-economics.svg`
  - `theory-of-constraints.svg`
  - `training-intuition-map.svg`
  - `v6_plan.svg`
  - `v6_plan_inkscape.svg`
  - `v7-cross-entropy-parity-map.svg`
- `editorial-or-concept`
  - `c-kernel-engine-overview.svg`
  - `concept-flash-attention.svg`
  - `concept-gqa.svg`
  - `concept-rope-detailed.svg`
  - `concept-transformer-overview.svg`
  - `engineering-compass.svg`
  - `iteration-philosophy.svg`
  - `online_softmax_algorithm.svg`
  - `power-delivery-infographic.svg`
  - `two-principles.svg`
- `memory-map-or-training`
  - `activation-memory-infographic.svg`
  - `concept-weight-tying.svg`
  - `ir-logic-vs-memory.svg`
  - `memory-reality-infographic.svg`
  - `v7-backprop-ir-stack.svg`
  - `v7-grad-accum-window.svg`
  - `weight_memory_layout.svg`
- `system-diagram-or-flow`
  - `amx_pipeline.svg`
  - `architecture-overview.svg`
  - `ethernet-topology.svg`
  - `forward-backward-flow.svg`
  - `ir-dataflow-stitching.svg`
  - `ir-kernel-registry-chain.svg`
  - `ir-lowering-pipeline.svg`
  - `ir-v66-operator-cheatsheet.svg`
  - `ir_v2_pipeline.svg`
  - `operator-spectrum-map-presentation.svg`
  - `operator-spectrum-map.svg`
  - `qwen_layer_dataflow.svg`
  - `rdma-observer-architecture.svg`
  - `tokenizer-architecture.svg`
- `timeline`
  - `ir-v66-gate-ladder.svg`
  - `ir-v66-runtime-modes.svg`
- `uncategorized`
  - `bf16_rounding.svg`
  - `bias_corrected_fusion.svg`
  - `commodity-hardware-pattern.svg`
  - `favicon.svg`
  - `flamegraph.svg`
  - `gpu-workaround-convergence.svg`
  - `hardware_aware_blocking.svg`
  - `hybrid-bottleneck.svg`
  - `infra-ocean-stability.svg`
  - `int8_dot_product.svg`
  - `ir-dumb-codegen.svg`
  - `ir-fusion-vs-unfused.svg`
  - `ir-json-walkthrough.svg`
  - `ir-kernel-amp-strategy.svg`
  - `ir-output-artifacts.svg`
  - `ir-producer-consumer-qkv-rope-attn.svg`
  - `ir-scratch-vs-persistent.svg`
  - `ir-templates-to-ir.svg`
  - `ir-v66-artifact-lineage.svg`
  - `ir-v66-failure-decision-tree.svg`
  - `ir-v66-test-gates.svg`
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

## Interpretation

- Current training has been operating on a narrow compiler language relative to the site asset library.
- Broader training should start by expanding DSL/compiler coverage inside the strongest missing buckets, not by adding more local rung patches.
