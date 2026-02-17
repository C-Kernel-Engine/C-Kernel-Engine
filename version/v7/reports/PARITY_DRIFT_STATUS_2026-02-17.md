# v7 Train Parity Drift Status (2026-02-17)

## Headline caveat (interpret results with this first)
- Historical limitation: older parity runs used tiny-mapped tensors (`tiny.embedding`, `tiny.rms_gamma`, `tiny.fc1`, `tiny.fc2`) and could produce depth-insensitive outcomes.
- Those pre-fix depth artifacts are kept for traceability but should not be treated as valid depth-vs-drift evidence.
- Pre-fix depth-insensitive examples (mirrored):
  - `version/v7/.cache/reports/parity_drift_2026-02-17/depth_layers_1_train_e2e_latest.json`
  - `version/v7/.cache/reports/parity_drift_2026-02-17/depth_layers_2_train_e2e_latest.json`
  - `version/v7/.cache/reports/parity_drift_2026-02-17/depth_layers_3_train_e2e_latest.json`
- Update in this workspace: parity harness now supports stacked execution in the main path (`--model-kind auto|stacked` + manifest-backed stacked init).

## Post-fix update (added 2026-02-17, later pass)
- We fixed a major train/parity plumbing bug in `version/v7/scripts/ck_run_v7.py`:
  - With `--run <run_dir>`, train/parity paths were still using tiny CLI defaults (`vocab=256,d_model=64,hidden=128,layers=1`) unless explicitly overridden.
  - This made several runs "apples-to-oranges" against run-dir manifests (for example the 24-layer run-dir), creating misleading drift signals.
- Fix implemented:
  - Resolve effective train dims from `run_dir/weights_manifest.json` (`config`/`training.tiny_parity`) and override tiny defaults.
  - Pass resolved dims through both CK runtime path and fallback oracle call.
  - Add `train_dims` block to JSON outputs so requested vs effective dims are explicit per run.
- Code touchpoints:
  - `version/v7/scripts/ck_run_v7.py:393`
  - `version/v7/scripts/ck_run_v7.py:1879`
  - `version/v7/scripts/ck_run_v7.py:3092`
  - `version/v7/scripts/ck_run_v7.py:4198`
  - `version/v7/scripts/ck_run_v7.py:4456`

## Reproducibility stamp
- `git_sha_short`: `4accf5e2`
- `git_sha_full`: `4accf5e28590dd4f4baf917236b8b8c07a6be0c6`
- `python`: `Python 3.12.3` (`.venv`)
- `compiler`: `gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0` (`clang` not present in this env)
- `host`: `Linux antshiv-ThinkPad-T14-Gen-3 6.14.0-37-generic x86_64`

## Why this test exists
- Goal: verify CK training stays numerically close to PyTorch over long horizons, not only that loss decreases.
- Stress (`lr=1e-3`, repetitive tiny text like `Hello!`) is used deliberately to expose weak stability margin early.

## What "drift" means
- Drift = CK and PyTorch training trajectories separate over steps.
- We track `loss_diff`, `max_logit_diff`, `max_grad_diff`, `max_param_diff`.
- Parity fails once observed maxima exceed configured tolerances (`loss_tol=2e-5`, `param_tol=3e-5` in these stress runs).

## Status matrix (exact knobs + outcomes)
| Profile | CK/Torch backend setup | lr | clip (`max_grad_norm`) | text profile | steps | Result | First fail (loss/param) | Artifact |
|---|---|---:|---:|---|---:|---|---|---|
| Stress control | all_torch | 1e-3 | 0.0 | repeated `Hello!` | 1000 | PASS | none | `version/v7/.cache/reports/parity_drift_2026-02-17/v7_repro_hello_131072_1000_alltorch_now.json` |
| Stress target | all_c | 1e-3 | 0.0 | repeated `Hello!` | 1000 | FAIL | 64 / 117 | `version/v7/.cache/reports/parity_drift_2026-02-17/v7_repro_hello_131072_1000_now.json` |
| Stress control-varied data | all_c | 1e-3 | 0.0 | random-token stream | 900 | PASS | none | `version/v7/.cache/reports/parity_drift_2026-02-17/v7_high_lr_900_random.json` |
| Low-LR sensitivity (hello) | all_c | 5e-4 | 1.0 | repeated `Hello!` | 110 | FAIL | 89 / 97 | `version/v7/.cache/reports/parity_drift_2026-02-17/v7_lowlr_hello_localize89.json` |
| Low-LR sensitivity (hello alt run) | all_c | 5e-4 | 0.0 | repeated `Hello!` | 1000 | PASS | none | `version/v7/.cache/reports/parity_drift_2026-02-17/v7_backprop_repro_hello_5e4_1000.json` |
| Production-safe profile | all_c | 5e-4 | 1.0 | realistic mixed text | 192 | PASS | none | `version/v7/.cache/reports/train_parity_realistic_long_horizon_latest.json` |
| Generated runtime parity (realistic) | `ck_run_v7.py train --backend ck --parity-on` (snapshot oracle) | 5e-4 | 1.0 | realistic mixed text | 320 opt steps (`2560` micro) | FAIL (slot-only) | n/a (`loss/param` stayed tiny) | `version/v7/.cache/reports/train_runtime_parity_realistic_latest.json` |
| Generated runtime parity (stress) | `ck_run_v7.py train --backend ck --parity-on` (snapshot oracle) | 1e-3 | 0.0 | repeated `Hello!` | 320 opt steps (`2560` micro) | FAIL (slot-only) | n/a (`loss/param` stayed tiny) | `version/v7/.cache/reports/train_runtime_parity_stress_latest.json` |
| Generated runtime parity (stress, post dim-fix short) | `ck_run_v7.py train --backend ck --parity-on` (snapshot oracle, manifest-resolved dims) | 1e-3 | 0.0 | repeated `Hello!` | 16 opt steps (`128` micro) | PASS | none | `version/v7/.cache/reports/parity_drift_2026-02-17/v7_l24_every8_1024_after_dimfix_venv.json` |
| Generated runtime parity (stress, post dim-fix long) | `ck_run_v7.py train --backend ck --parity-on` (snapshot oracle, manifest-resolved dims) | 1e-3 | 0.0 | repeated `Hello!` | 320 opt steps (`2560` micro) | FAIL (slot-only) | first slot fail at micro `160` (opt `20`) | `version/v7/.cache/reports/parity_drift_2026-02-17/v7_l24_every8_4096_after_dimfix_venv.json` |
| SVG practical trainability check | `ck_run_v7.py train --backend ck --parity-on` | 5e-4 | 1.0 | repeated SVG `<rect>` snippet | 640 opt steps (`640` micro) | PASS | none | `version/v7/.cache/reports/parity_drift_2026-02-17/v7_svg_box_train_ck.json` |

## Key nuance: token/data profile sensitivity at low LR
- At `lr=5e-4`, realistic mixed-text profile can pass while repetitive tiny-text profile can still fail in some runs.
- This is why "low LR passed once" is not enough; text/window profile materially changes drift behavior.

## Generated runtime long-horizon parity (new on 2026-02-17)
- Added make targets to run generated-runtime parity directly:
  - `v7-train-runtime-parity-stress`
  - `v7-train-runtime-parity-realistic`
  - `v7-train-runtime-parity-long-horizon`
- Realistic profile result (`lr=5e-4`, clip `1.0`, 320 optimizer steps):
  - `pass_parity=false`
  - `max_loss_abs_diff=9.537e-07`
  - `final_param_max_abs_diff=0.0`
  - failures are strict activation-slot checks (`failure_modes=['slots']`), first at micro step `1816` (optimizer step `227`)
  - dominant first-bad op: `layer_0:mlp_down` (`act.L0.mlp_down.0.y`)
- Stress profile result (`lr=1e-3`, clip `0.0`, 320 optimizer steps):
  - `pass_parity=false`
  - `max_loss_abs_diff=1.907e-06`
  - `final_param_max_abs_diff=0.0`
  - failures are strict activation-slot checks (`failure_modes=['slots']`), first at micro step `416` (optimizer step `52`)
  - dominant first-bad op: `layer_0:mlp_down` (`act.L0.mlp_down.0.y`)
- Interpretation:
  - This is not a classic "loss blew up" failure.
  - Current blocker is strict runtime-vs-oracle activation-slot parity drift.
  - Remaining ambiguity: true math divergence vs slot mapping/export/ordering discrepancy in runtime oracle plumbing.

### Post dim-fix rerun (same day)
- Long-horizon stress rerun after manifest-dim wiring fix:
  - artifact: `version/v7/.cache/reports/parity_drift_2026-02-17/v7_l24_every8_4096_after_dimfix_venv.json`
  - `steps=2560` micro (`320` optimizer), `pass_parity=false`
  - `max_loss_abs_diff=1.311e-06` (loss remained tight)
  - first fail moved to strict slot-drift at micro step `160` (optimizer step `20`)
  - first bad slot signature:
    - `first_bad_tensor=act.L22.residual_add.1.out`
    - `first_bad_op=layer_22:residual_add`
    - `first_bad_diff=2.44140625e-4` with `activation_threshold=2.0e-4`
- Shorter post-fix stress sanity:
  - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_l24_every8_512_after_dimfix_venv.json`: PASS
  - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_l24_every8_1024_after_dimfix_venv.json`: PASS

## Depth-vs-drift update (post stacked-path wiring)
- Manifest-backed stacked sweep (`lr=1e-3`, `Hello!`, `max_steps=80`):
  - `layers=1`: PASS, `first_param_fail_step=None`, `final_param_max_abs_diff=1.422e-05`
  - `layers=2`: FAIL, `first_param_fail_step=1`, `final_param_max_abs_diff=9.794e-05`
  - `layers=3`: FAIL, `first_param_fail_step=1`, `final_param_max_abs_diff=4.856e-04`
  - artifacts:
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_after_patch_l1_s80.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_after_patch_l2_s80.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_after_patch_l3_s80.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_after_patch_l123_s80_summary.json`
- Synthetic stacked sweep through 24 layers (`lr=1e-3`, `Hello!`, `max_steps=40`):
  - `layers=1`: PASS
  - `layers=2,3,6,12,24`: FAIL (`first_param_fail_step=1` in all tested depths)
  - `final_param_max_abs_diff` grows with depth (e.g. `3.952e-05 @2` -> `1.406e-04 @24`)
  - artifacts:
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_curve_stacked_s40_l1.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_curve_stacked_s40_l2.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_curve_stacked_s40_l3.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_curve_stacked_s40_l6.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_curve_stacked_s40_l12.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_depth_curve_stacked_s40_l24.json`

## Ablations run (C op swaps vs PyTorch)
- Matrix summary (mirrored):
  - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_localize64_matrix_summary.json`
- `Hello!`, `lr=1e-3`, localize around step ~64:
  - `all_torch`: PASS
  - `rms_c_only`: FAIL (loss step 64)
  - `swig_c_only`: FAIL (loss step 64)
  - `loss_c_only`: FAIL (loss step 64)
  - `all_c`: FAIL (loss step 64)
- Localizer indicates forward-stage threshold crossing appears first at trigger point.

## RMSNorm patch made today
- Change: strict RMSNorm opmath/order switched to fp32-style (removed double intermediates) for closer PyTorch parity.
- Code:
  - `src/kernels/rmsnorm_kernels.c`
- `rms=c`, `swig=torch`, `loss=torch` before/after:
  - `first_loss_fail_step`: `64 -> 65`
  - `max_loss_abs_diff`: `2.335e-4 -> 1.388e-4`
  - `max_logit_abs_diff`: `3.590e-3 -> 2.163e-3`
  - `max_grad_abs_diff`: `7.545e-3 -> 4.499e-3`
  - artifacts:
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_localize64_rms_c_only.json`
    - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_localize64_rms_c_only_after_fp32strict.json`
- Net: improvement, not full drift elimination.

## Practical interpretation
- Current state is "trainable with robustness debt under stress," not "completely broken."
- Parity failures are still meaningful warning signals for:
  - seed sensitivity,
  - hyperparameter fragility,
  - late-run surprises as scale/depth/data complexity increase.

## Bugs/oversights we found and fixed
- Fixed: run-dir dim mismatch in train/parity wiring.
  - Root cause: train command defaults (`256/64/128/1`) were used even when `--run` pointed to a manifest with different dims.
  - Effect: misleading parity drift that mixed incompatible model shapes.
  - Fix: manifest-resolved dims are now the effective dims for runtime + oracle paths, and exported in JSON (`train_dims`).
- Fixed (integration-level): make/runtime parity flows now inherit corrected dim behavior automatically through `ck_run_v7.py train`.
  - No Makefile target contract change was required; the underlying train path bug was corrected.

## Mistakes/overlooked diagnostics (kept explicit)
- Running outside `.venv` can silently disable snapshot oracle (`torch is required for snapshot oracle`), causing fallback mode and ambiguous parity interpretation.
- Fallback oracle (`tiny_reference_harness`) is currently non-strict for fail gating (`oracle_strict=false`), so runs can show non-trivial `loss_diff` but still mark `PASS`.
- In fallback mode, checked rows may contain `loss_diff` while `loss_ck/loss_oracle` fields are `None`; this is a reporting-quality issue and can confuse triage.

## Other likely issues still open
- Deep-stack slot drift appears first around residual-add paths (`layer_22:residual_add`) under high-LR repetitive stress, while loss remains tight; likely a cumulative op-order/reduction-order numeric sensitivity rather than a single catastrophic kernel break.
- Slot parity thresholds are fixed absolute values today; depth-aware or scale-aware thresholds may be needed to separate harmless numeric noise from true regressions.
- Activation-slot drift localization still needs one more boundary-dump pass around first-fail checkpoints to confirm whether divergence originates pre-op or at op.

## Artifact durability note
- Canonical copies for this report are mirrored under:
  - `version/v7/.cache/reports/parity_drift_2026-02-17/`
- Original `/tmp/...` paths were preserved during active debugging but are not the stable reference.

## Next work (agreed direction)
- Keep generated-runtime oracle parity (`--backend ck --parity-on`) as primary gate.
- For the current post-fix long-horizon stress fail, localize slot-only mismatch at `layer_22:residual_add` (`act.L22.residual_add.1.out`) via boundary tensor dumps around the first-fail checkpoint (micro `160`).
- Add optimizer-state (`exp_avg`, `exp_avg_sq`) export/compare in runtime-oracle checks.
- Keep high-LR tiny stress as diagnostic signal, not sole production blocker.
- Add an explicit CI guard that fails if `train_dims.source != run_manifest` when `--run` is provided and manifest exists.
