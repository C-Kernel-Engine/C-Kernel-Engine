# Spec06 Infographic Line

`spec06` extends the structured SVG line from scene composition into fixed-template infographic composition.

It is intentionally separate from `spec05`.

## Goal

Teach the model to:

- choose the right infographic layout family
- bind the requested topic pack into the correct text slots
- preserve deterministic SVG structure under fixed template geometry

It is not trying to learn free-form design.

## Why Separate From Spec05

- `spec05` remains the stable benchmark for small structured scenes.
- `spec06` introduces a different task family: asset-inspired information layouts.
- Keeping them separate preserves comparability and regression tracking.

## Layout Families

The initial `spec06` layouts are derived from existing site asset patterns:

- `bullet-panel`
  Source style: [memory-reality-infographic.svg](/home/antshiv/Workspace/C-Kernel-Engine/docs/site/assets/memory-reality-infographic.svg)
- `compare-panels`
  Source style: [performance-balance.svg](/home/antshiv/Workspace/C-Kernel-Engine/docs/site/assets/performance-balance.svg)
- `stat-cards`
  Source style: [activation-memory-infographic.svg](/home/antshiv/Workspace/C-Kernel-Engine/docs/site/assets/activation-memory-infographic.svg)
- `spectrum-band`
  Source style: [operator-spectrum-map.svg](/home/antshiv/Workspace/C-Kernel-Engine/docs/site/assets/operator-spectrum-map.svg)
- `flow-steps`
  Source style: [pipeline-overview.svg](/home/antshiv/Workspace/C-Kernel-Engine/docs/site/assets/pipeline-overview.svg)

Each layout keeps geometry deterministic and varies only:

- topic pack
- accent color
- background
- frame
- density

## Training Intent

- `pretrain`
  Teach the new slot-oriented DSL and renderer contract.
- `midtrain`
  Teach composition through layout edits, topic edits, and style edits.
- `sft`
  Optional later layer for natural-language infographic requests.
- `dpo/grpo`
  Not needed until structural exactness is already strong.

## Current Files

- Generator: [generate_svg_structured_spec06_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/generate_svg_structured_spec06_v7.py)
- Materializer: [materialize_spec06_structured_atoms_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/dataset/materialize_spec06_structured_atoms_v7.py)
- Probe contract builder: [build_spec06_probe_contract_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/build_spec06_probe_contract_v7.py)
- Launcher: [spec06_pretrain_midtrain_v7.sh](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/scripts/spec06_pretrain_midtrain_v7.sh)

## Notes

Because literal text is still tokenized through the current ASCII-BPE path, `spec06` keeps slot text short and uses a larger default context window than `spec05`.
