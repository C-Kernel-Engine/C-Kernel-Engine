# Spec12 Gold Token Budget Report

- Generated on: `2026-03-18`
- Manifest: `/home/antshiv/Workspace/C-Kernel-Engine/version/v7/reports/spec12_gold_mappings/spec12_gold_mappings_20260318.json`
- Assumed prompt tokens: `16`
- Max output tokens: `2793`
- Max total tokens: `2809`
- Recommended context length: `1024`
- Tokenizer rework required before training: `true`

| Asset | Family | Output Tokens | Total @512 | Total @768 | Total @1024 |
| --- | --- | ---: | --- | --- | --- |
| `quantization-formats.svg` | `table_matrix` | `1820` | 1836 (overflow, -1324) | 1836 (overflow, -1068) | 1836 (overflow, -812) |
| `ir-v66-failure-decision-tree.svg` | `decision_tree` | `2048` | 2064 (overflow, -1552) | 2064 (overflow, -1296) | 2064 (overflow, -1040) |
| `memory-layout-map.svg` | `memory_map` | `2793` | 2809 (overflow, -2297) | 2809 (overflow, -2041) | 2809 (overflow, -1785) |

## Read

- These counts were measured against the current tokenizer at `/home/antshiv/.cache/ck-engine-v7/models/train/spec11_keyed_scene_dsl_l3_d192_h384_ctx512_r2/tokenizer.json`. They are a readiness diagnostic, not a final spec12 token count.
- Context alone is not the fix. The current tokenizer surface does not yet compress the spec12 scene language enough for training.
- Next step: define the spec12 tokenizer surface after compiler parity on the gold mappings, then re-measure before choosing `ctx`.
- Do not launch training from this alone. Use this together with compiler parity on the same gold mappings.
