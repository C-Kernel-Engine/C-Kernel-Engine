## v7 SVG DSL Artifacts

This directory is the quarantine/archive area for `v7` SVG/spec/gen1 experiment outputs that are not part of the active `v7` text-runtime/backprop surface.

Current archive:

- `gen1_archive_2026-04-05/`
- `spec_archive_2026-04-08/`

Archive policy:

- Generated reports, smoke outputs, gold-pack snapshots, and one-off planning docs from the SVG/spec/gen1 line should land here instead of expanding the active `version/v7/reports/` surface.
- When older scripts or manifests still expect `version/v7/reports/...`, keep a compatibility symlink there and store the real files in the dated archive here.
- Core `v7` text-training/runtime artifacts should stay in their normal run/report locations.
- Script paths are intentionally not rewritten in this first cleanup pass. If an archived SVG/gen1 script is run again, it may recreate fresh outputs under `version/v7/reports/`; those can be re-archived afterward.

Active `v7` focus:

- text inference/runtime
- text backprop/runtime codegen
- A-F PyTorch parity regimen
- family transfer across qwen2/qwen3/qwen35/nanbeige/gemma
