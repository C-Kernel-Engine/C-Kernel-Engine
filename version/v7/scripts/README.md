## v7 Scripts Surface

This directory still contains both:

- the active `v7` text runtime/backprop toolchain
- older SVG/spec/gen1 experiment scripts kept for history and reproducibility

The active `v7` text surface is:

- `ck_run_v7.py`
- `build_ir_train_v7.py`
- `lower_ir2_backward_v7.py`
- `codegen_train_runtime_v7.py`
- `train_parity_epochs_v7.py`
- `run_training_parity_regimen_v7.py`
- `run_backprop_family_matrix_v7.py`
- `check_backprop_stitch_runtime_v7.py`
- `check_replay_determinism_v7.py`
- `check_runtime_replay_accum_v7.py`
- `train_data_pipeline_v7.py`
- `init_tiny_train_model_v7.py`

Family-scoped text parity now means:

- `M0`: live memory-headroom preflight before heavy parity/runtime stages
- `A/B/C`: CK-vs-PyTorch harness parity
- `D/F`: generated-runtime end-to-end checks through `ck_run_v7.py train` with sampled Torch oracle checks
- `E`: determinism sentinel for the parity harness, not a generated-runtime signoff by itself
- `G/H/I` when `--extended-checks` is enabled: longer-horizon and broader generated-runtime parity coverage

Regimen profiles:

- `--stage-profile default`: current broad `A/B/C + D/E/F` surface
- `--stage-profile strict`: controlled signoff lane; keeps `M0`, `A`, and `D/E/F`, skips the heavier `B/C` sweep fanout
- `--stage-profile extended`: default surface plus `G/H/I`

Operator-facing family matrix:

- `run_backprop_family_matrix_v7.py --mode fast`
  - `qwen2`, `qwen3`, `gemma`, `nanbeige`
- `run_backprop_family_matrix_v7.py --mode full`
  - fast set plus `qwen35`

The family matrix keeps the stable make entrypoints on the current `A-F` surface by default.
Use `--extended-checks` on the regimen, or `V7_BACKPROP_MATRIX_EXTENDED=1` on the make targets,
when you want the opt-in `G/H/I` longer-horizon runtime gates.
Use the memory flags on the regimen/matrix wrappers when you need a stricter preflight floor,
or `--no-memory-check` only for intentional local diagnostics.

Operator-facing make entrypoints:

- `make regression-training-fast`
- `make regression-training-full`
- `make training-fast`
- `make training-full`

Those generic training targets are the stable user-facing interface. Today they route to the
`v7` text-backprop lane. If a later version becomes the primary training lane, the
Makefile should update that routing internally without changing the user command.

Compatibility aliases still exist:

- `make regression-backprop-fast`
- `make regression-backprop-full`

Current family transfer target order:

1. `qwen2`
2. `qwen3`
3. `qwen35`
4. `gemma`
5. `nanbeige`

The SVG/spec/gen1 scripts are intentionally still present under their old paths so imports, docs, and historical references do not break. Generated SVG DSL reports and packs were moved to:

- `version/v7/artifacts/svg_dsl/`

If a stricter cleanup pass is done later, move the SVG/spec/gen1 scripts behind compatibility wrappers instead of deleting or blindly renaming them.
