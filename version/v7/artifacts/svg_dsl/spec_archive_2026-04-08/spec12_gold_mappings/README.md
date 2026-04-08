# Spec12 Gold Mappings

These started as the first three hand-written `spec12` gold mappings and now
include the first semantic-breadth expansion pack under the same compiler
families.

Each mapping contains:

- a `scene.dsl` file describing structure and composition
- a `content.json` file containing visible text and data

These are not training rows yet. They are compiler targets.

The immediate purpose is:

1. prove that the next scene language can express the asset family
2. expose missing compiler features before training
3. keep structure and content separate from the start

## Included Assets

### 1. Quantization Formats

- Source asset: `docs/site/assets/quantization-formats.svg`
- Proposed family: `table_matrix`
- Key compiler pressure:
  - true table layout
  - grouped headers
  - legend block
  - row states
  - note bands

Files:
- [quantization-formats.scene.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/quantization-formats.scene.dsl)
- [quantization-formats.content.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/quantization-formats.content.json)

### 2. Failure Decision Tree

- Source asset: `docs/site/assets/ir-v66-failure-decision-tree.svg`
- Proposed family: `decision_tree`
- Key compiler pressure:
  - branching connectors
  - decision nodes
  - entry badge
  - labeled edges
  - outcome panels

Files:
- [failure-decision-tree.scene.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/failure-decision-tree.scene.dsl)
- [failure-decision-tree.content.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/failure-decision-tree.content.json)

### 3. Memory Layout Map

- Source asset: `docs/site/assets/memory-layout-map.svg`
- Proposed family: `memory_map`
- Key compiler pressure:
  - address strips
  - memory segments
  - offset markers
  - collapsed region brackets
  - side info cards

Files:
- [memory-layout-map.scene.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/memory-layout-map.scene.dsl)
- [memory-layout-map.content.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/memory-layout-map.content.json)

### 4. Edge-Case Matrix

- Source asset: `docs/site/assets/ir-v66-edge-case-matrix.svg`
- Proposed family: `table_matrix`
- Key compiler pressure:
  - longer operator-facing row text
  - grouped operational tables
  - row-state emphasis across multiple owners
  - maintenance-note footer

Files:
- [edge-case-matrix.scene.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/edge-case-matrix.scene.dsl)
- [edge-case-matrix.content.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/edge-case-matrix.content.json)

### 5. Training Memory Canary

- Source asset: `docs/site/assets/v7-train-memory-canary.svg`
- Proposed family: `memory_map`
- Key compiler pressure:
  - contiguous training arena sections
  - diagnostic-phase side cards
  - explicit ownership of forward/backward/optimizer regions
  - canary and audit metadata as external content

Files:
- [training-memory-canary.scene.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/training-memory-canary.scene.dsl)
- [training-memory-canary.content.json](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/training-memory-canary.content.json)

## Immediate Next Step

Use these three mappings to build the first `spec12` compiler parity pass:

1. parse `scene.dsl`
2. bind `content.json`
3. compile to SVG
4. compare against the source asset visually and structurally

That initial parity pass is done. The current next step is broader semantic
coverage inside the existing layout families before adding new renderer
families.

## Compact Scene Variants

The first compression pass is also available now. These compact scene files are
not training data yet. They are the proposed next compiler boundary.

- [quantization-formats.scene.compact.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/quantization-formats.scene.compact.dsl)
- [failure-decision-tree.scene.compact.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/failure-decision-tree.scene.compact.dsl)
- [memory-layout-map.scene.compact.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/memory-layout-map.scene.compact.dsl)
- [edge-case-matrix.scene.compact.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/edge-case-matrix.scene.compact.dsl)
- [training-memory-canary.scene.compact.dsl](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/training-memory-canary.scene.compact.dsl)

Compression analysis:

- [SPEC12_DSL_COMPRESSION_ANALYSIS_2026-03-18.md](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/SPEC12_DSL_COMPRESSION_ANALYSIS_2026-03-18.md)
