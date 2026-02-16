# v7 Backprop Bring-Up Roadmap (PR1 -> PR6)

## PR1 - Guardrails and Contracts
Goal: enforce strict training prerequisites before runtime work.

- Validate train contracts (L1-L5) with strict training mode.
- Require kernel map + binding coverage for train backward kernels.
- Keep inference protected with smoke gate (Gemma/Qwen2/Qwen3 generate+compile path).

Exit criteria:
- `make v7-validate-contracts` passes in strict mode.
- `make v7-inference-smoke` passes.

---

## PR2 - Execute Generated Training Runtime
Goal: `--backend ck` actually executes generated C runtime.

- Use `init -> IR1 -> IR2 -> codegen_train_runtime_v7.py -> libtrain.so`.
- Add runtime callable surface (`ck_train_step`) and wire ctypes invocation from `ck_run_v7.py`.
- Preserve existing parity harness for `--backend both|pytorch`.

Exit criteria:
- `ck_run_v7.py train --backend ck` runs end-to-end without crash.
- `--backend both` parity path still works unchanged.

---

## Why Training Path Is Separate For Now

- Inference is dominated by quantized weight formats and decode/prefill layout constraints.
- Backprop is predominantly fp32/bf16 with no quantized weight updates in the core path.
- Training adds gradient-specific graph semantics (grad accumulation, loss seed, optimizer state, clip/scale).
- Training memory layout has extra persistent regions (`grad.weight.*`, `optimizer_m`, `optimizer_v`) that inference does not carry.
- Keeping training separate right now reduces risk while contracts stabilize; unification can happen later once train layout and runtime are stable.

---

## PR3 - Make CK Runtime Numerically Real
Goal: remove placeholder behavior and make training math meaningful.

- Hydrate runtime weights from `weights.bump`/manifest into generated weight buffers (`ck_train_init`).
- Ensure CE loss writes to stable scalar capture (`g_loss_scalar`).
- Add optimizer ordering improvements (clip/update hooks in generated runtime).
- Keep train JSON schema compatible with viewer.

Exit criteria:
- CK runtime loss is non-trivial (not constant 0 from uninitialized path).
- Training summary contains real runtime-init metadata and step telemetry.

---

## PR3.5 (PR16 Track) - Generated Memory Diagnostic Kernel
Goal: prove contiguous training layout safety at runtime (no OOB/illegal writes) with generated checks.

- Codegen emits a tensor-slot table (`name`, `offset_floats`, `numel`, `section`, forward/backward writable flags).
- Add diagnostic runtime surface:
  - `ck_train_plant_canaries()`
  - `ck_train_check_canaries(phase_name, first_corrupt_idx)`
  - `ck_train_check_weights_readonly(weight_snapshot, first_corrupt_idx)`
  - `ck_train_memory_diagnostic(oracle_acts, oracle_grads, tolerance)`
- Use generated constants:
  - `CK_CANARY_VALUE=0xDEADBEEF`
  - `CK_CANARY_FLOATS=16` (64B guard per slot boundary)
- In diagnostic mode, reserve canary gaps between tensor slots and verify after forward/backward/optimizer phases.
- Snapshot weights before forward and assert no write during forward phase.
- Optional oracle compare hooks for selected activations/gradients to localize first numeric drift.
- Emit machine-readable report (`memory_diagnostic_latest.json`) in run-dir with first bad tensor slot + phase.

Generated layout contract details:
- Emit one `CKTensorSlot` row per tensor with:
  - `name`
  - `offset_floats`
  - `numel`
  - `section`
  - `writable_fwd`
  - `writable_bwd`
- Keep this table generated from layout/codegen inputs only (no handwritten slot manifests).

Diagnostic execution sequence:
1. Plant canaries.
2. Snapshot weights.
3. Run forward.
4. Check canaries (forward).
5. Check forward weight-readonly contract.
6. Run backward.
7. Check canaries (backward).
8. Compare selected activations to oracle (optional).
9. Compare selected gradients to oracle (optional).
10. Run optimizer.
11. Check canaries (optimizer).
12. Emit structured diagnostic verdict + first-failure slot.

Exit criteria:
- Canary checks pass for forward/backward/optimizer on tiny CK runs.
- Forward readonly-weight assertion passes.
- On induced overflow, report identifies first corrupted slot deterministically.
- `ck_train_memory_diagnostic()` returns non-zero with deterministic `first_corrupt_idx` on seeded corruption tests.

---

## PR3.6 - Current Gaps to Close (Must-Finish Before Full Sign-Off)
Goal: keep runtime status explicit so remaining memory/parity work is unambiguous.

Resolved:
- Generated C training runtime uses contiguous arena allocation (`calloc`) with offsets sourced from `layout_train.json`.
- Static per-tensor fixed-cap buffers are no longer the primary train runtime path.
- Runtime canary diagnostic APIs are implemented and callable from strict CLI mode:
  - `ck_train_plant_canaries()`
  - `ck_train_check_canaries(...)`
  - `ck_train_memory_diagnostic(...)`

Open:
- Full oracle drift localization/dump pipeline is not yet complete end-to-end.
- Optional canary-strict stress sweeps across larger configs are still pending (beyond tiny smoke runs).

Exit criteria:
- Strict CK diagnostic passes without tail or inter-slot canary corruption across tiny and medium configs.
- Canary diagnostic report includes deterministic failing op metadata when induced corruption is injected.
- `--parity-on --dump-on-drift` emits first-divergence artifacts consistently for CK backend.

---

## PR3.7 - Verification Layer (Determinism + Diagnostics)
Goal: prove memory diagnostics are real (not masked) and reproducible.

Implemented checks:
- Toggle-difference check (`CK_RUNTIME_CANARY_CHECKS=0` vs `1`) on identical seed/batches: loss curves must match exactly.
- Intentional `+1` write injection (`CK_RUNTIME_FAULT_INJECT=1`, `CK_FAULT_INJECT_OP_ID=<op>`): must fail with deterministic `backward_trace_canary` and failing `op_id`.
- ASan agreement on same seeded run path via ASan-instrumented runtime variants (`LD_PRELOAD=libasan.so`):
  - clean variant passes,
  - injected variant reports the same negative diagnostic verdict.
- Optional bounds assertions in generated runtime (`CK_RUNTIME_BOUNDS_ASSERT=1`) for pointer-span checks before kernel calls.

Artifacts:
- `memory_diagnostic_latest.json` (strict diagnostic)
- `memory_verification_latest.json` (PR3.7 verification suite)

CLI:
- `--train-runtime-canary-checks`
- `--train-runtime-bounds-assert`
- `--train-runtime-fault-op-id`
- `--train-verify-memory`
- `--train-verify-steps`
- `--train-verify-fault-op-id`

Exit criteria:
- `train --train-strict --train-verify-memory` completes with `memory_verification_latest.json["ok"] == true`.


## PR4 - Oracle Parity for CK Runtime (`--parity-on`)
Goal: CK runtime is primary, PyTorch is periodic oracle.

- Add cadence policy: `debug|balanced|light` + `--parity-every` override.
- Replay oracle windows and compare loss drift at selected steps.
- Write parity artifacts consumable by IR viewer.

Exit criteria:
- `--backend ck --parity-on` emits parity checks and pass/fail verdicts.
- `training_parity.json`/summary fields are populated consistently.

---

## PR4.5 - Throughput Bring-Up (After Strict Parity)
Goal: improve CK training throughput while keeping strict numeric parity gates unchanged.

Scope:
- Add explicit threaded GEMM dispatch in generated v7 training runtime (not serial-only blocked calls).
- Compile training runtime with consistent parallel flags and deterministic runtime thread control.
- Prepack/plan GEMM weight access to reduce cache-miss pressure.
- Reduce backward buffer traffic hot spots (`grad_accumulate`, `memset`, `memmove`) without changing math.
- Keep parity tolerances and gates unchanged during all optimization work.

Incremental status (2026-02-15):
- Step 1 scaffold landed: `train_exec_plan.json` is now generated from IR2 (`generate_train_exec_plan_v7.py`).
- Codegen consumes `--exec-plan` and records per-op dispatch metadata in generated C comments/summary.
- Numeric kernel contracts remain IR-driven (plan metadata is advisory; not allowed to override backward GEMM contract dims).

Exit criteria:
- CK throughput improves on the same train config with parity still passing.
- `--backend ck --parity-on --parity-replay-on-check` remains green.
- Profiling report identifies reduced time in GEMM/backward memory hot spots.

---

## PR5 - Drift Triage and First-Failure Localization
Goal: actionable debugging when parity fails.

- Emit `drift_report.json` with step, threshold, top diffs.
- Add dump controls (`--dump-on-drift`, `--drift-topk`).
- Persist artifacts into run-dir for report linking.

Exit criteria:
- On induced mismatch, report identifies first failing check and top-K diffs.

---

## PR6 - Run-Dir + Viewer Sign-Off
Goal: complete operator workflow from run-dir only.

- Keep all train artifacts under `--run` directory.
- Ensure open_ir_visualizer loads train runtime + parity + profile + drift outputs.
- Final gate sequencing for train path + inference non-regression.

Exit criteria:
- One command sequence: init/train/parity/profile/generate report from same run-dir.
- Viewer tabs show populated training artifacts without manual copying.

---


## PR7 - C-First Operator Runtime (CK Dominant Path)
Goal: make native CK runtime the default control plane for inference + training + profiling, with Python reduced to setup/oracle utilities.

Operator direction:
- Primary runtime path is native C (`ck-cli-v7` + generated runtimes).
- Same run-dir contract for inference and training artifacts.
- Python remains for:
  - model download/conversion/bootstrap,
  - IR generation/codegen orchestration,
  - optional PyTorch oracle parity path.

Scope:
- Add native CLI subcommands for operator workflows:
  - `init` (run-dir creation + IR/codegen trigger),
  - `infer` (native inference),
  - `train` (native CK training runtime),
  - `profile` (perf/vtune/advisor capture),
  - `report-index` (emit artifact manifest for viewer).
- Emit canonical `run_index.json` from native path with stable artifact pointers.
- Keep `open_ir_visualizer.py` as optional pack/render helper; do not require it to run workloads.
- Keep parity and safety gates intact (`--train-strict`, canary/memory diagnostics, replay checks).

C-direct profiling lane (required):
- Add native `ck-train-v7` (or `ck-cli-v7 train`) runtime entry that executes generated `libtrain.so` directly from a run-dir.
- Profile this native entry with `perf`/`vtune`/`advisor` (target process must be C runtime, not Python wrapper).
- Keep Python for prepare/oracle only (download/convert/IR/codegen/oracle replay), not hot-loop execution profiling.

Exit criteria:
- End-to-end inference and CK training can be launched without Python runtime loops.
- Profiling artifacts (perf/flamegraph/vtune/advisor summaries) are generated from native operator commands.
- Viewer can load a run via `run_index.json` with no manual file copying.
- Python is no longer in the critical serving/training execution path (except optional oracle checks).
- VTune/Advisor reference commands point at native CK train/infer binaries directly.

Non-goal for this PR:
- Removing Python from model download/format conversion and IR/codegen utilities.

PR7 Checklist (Operator-Centric):

| Item | Owner | Status | Target Command | Primary Artifact |
|---|---|---|---|---|
| Native `ck-cli-v7 train` subcommand (run-dir + libtrain) | v7 runtime | pending | `./build/ck-cli-v7 train --run /tmp/v7_run ...` | `train_e2e_latest.json` |
| Native `ck-cli-v7 profile` subcommand | v7 perf | pending | `./build/ck-cli-v7 profile --run /tmp/v7_run --tool perf` | `profile_summary.json`, `perf_stat_summary.json` |
| Native `ck-cli-v7 init` subcommand glue (prepare + validate run-dir) | v7 tooling | pending | `./build/ck-cli-v7 init --run /tmp/v7_run ...` | `run_index.json` |
| Canonical `run_index.json` producer from native path | v7 runtime | pending | `./build/ck-cli-v7 report-index --run /tmp/v7_run` | `run_index.json` |
| Viewer first-load from `run_index.json` (primary contract) | v7 viewer | pending | `open_ir_visualizer.py --generate --run /tmp/v7_run --html-only` | `ir_report.html` |
| C-direct VTune/Advisor runbook commands in docs | docs/perf | in_progress | `vtune ... ./build/ck-cli-v7 train ...` | docs page updates |

---

## Critical Commands

```bash
make --no-print-directory v7-help
make --no-print-directory v7-validate-contracts
make --no-print-directory v7-inference-smoke

python3 version/v7/scripts/ck_run_v7.py init --run /tmp/v7_run --generate-ir --generate-runtime --strict
python3 version/v7/scripts/ck_run_v7.py train --run /tmp/v7_run --backend ck --train-epochs 1 --train-seq-len 8 --train-total-tokens 64 --train-grad-accum 2
python3 version/v7/scripts/ck_run_v7.py train --run /tmp/v7_run --backend ck --parity-on --parity-profile balanced --parity-every 50
python3 version/v7/tools/open_ir_visualizer.py --generate --run /tmp/v7_run --html-only
```

---

## Status Clarification (2026-02-15)

Requested caveats were reviewed against current v7 codegen/runtime state:

- `Generated C runtime is still per-tensor static buffers, not yet one contiguous calloc arena.`
  - Status: **resolved**. Runtime now uses a single contiguous `g_memory` allocation via `ck_train_alloc()` + per-tensor offsets (`OFF_*`).
- `Runtime canary diagnostic APIs (...) are in plan, not implemented yet.`
  - Status: **resolved**. Codegen emits `ck_train_plant_canaries`, `ck_train_check_canaries`, and `ck_train_memory_diagnostic`.
- `Full oracle drift localization/dump pipeline is not fully complete.`
  - Status: **still open**. We have parity checks + drift summary, but full first-divergence op localization + complete tensor dump/oracle replay tooling remains PR5 work.

PR4 update in progress:
- Oracle replay is now bounded by parity cadence (`--max-steps` passed to `train_parity_epochs_v7.py`) instead of unbounded full-run replay.

PR4 incremental update:
- Added CK runtime one-step replay verification at parity check points (`--parity-replay-on-check`, `--parity-replay-tol`).
- Generated runtime now exports/imports weight snapshots (`ck_train_get_weight_snapshot_numel`, `ck_train_export_weight_snapshot`, `ck_train_import_weight_snapshot`).
- Replay runs against an isolated runtime instance (`libtrain_replay.so`) to avoid state contamination.
- Drift snapshots can be persisted under `oracle_ck_snapshots/` when `--dump-on-drift` is enabled.

PR4 incremental update (current):

PR4.5 Throughput Track (next):
- Threaded GEMM dispatch in training codegen/runtime.
- Runtime thread policy normalization (`CK_NUM_THREADS`/OpenMP) for deterministic scaling tests.
- GEMM prepack/access planning + backward buffer traffic reduction.
- Perf measurement loop: CK vs torch, then `perf`/VTune/flamegraph on hottest ops.
- Added generated activation snapshot APIs:
  - `ck_train_get_activation_snapshot_numel()`
  - `ck_train_export_activation_snapshot(...)`
- CK runtime train loop now supports snapshot-driven Torch oracle plumbing:
  - preferred source: `torch_snapshot_step` (strict mode when available),
  - fallback source: `tiny_reference_harness` (telemetry-only, non-blocking).
- Parity telemetry now records oracle source/strictness, logits slot metadata, and activation snapshot artifact paths.
- Drift reports now include both weight snapshot and activation snapshot artifact links when dumps are enabled.

### PR3.6 Update (Accumulation Scheduling)

- CK runtime now compiles with explicit `CK_GRAD_ACCUM_STEPS` from CLI `--train-grad-accum`.
- Generated `ck_train_step()` runs forward/backward every micro-step, but optimizer update only on accumulation boundary.
- Runtime wrapper flushes pending accumulation at end-of-run (`ck_train_flush_optimizer`) so final checkpoints/metrics reflect all micro-steps.
- Summary now reports both `micro_steps` and `optimizer_steps` to make schedule mismatches visible.
- `gemm_backward_f32` runtime-call dims are now shape-derived (`[T,D]`, `[O,I]`) instead of numel-derived flattening; this fixes T>1 full-C crashes caused by wrong `aligned_in/aligned_out` contracts.

## PR3.6 Drift Diagnostics Status (2026-02-16)

Implemented now:
- `train_parity_epochs_v7.py` emits optimizer-step diagnostics in JSON:
  - `max_logit_diff`, `max_grad_diff`, `worst_grad_param`
  - `max_exp_avg_diff`, `max_exp_avg_sq_diff`
  - `first_loss_fail_step`, `first_param_fail_step` in `drift_diagnostics`
- Kernel-isolation toggles are available in parity harness:
  - `--ck-rmsnorm-backend {c,torch}`
  - `--ck-swiglu-backend {c,torch}`
  - `--ck-loss-backend {c,torch}`

Measured attribution (AdamW, seq=8, total_tokens=4096, grad_accum=8, max_steps=70):
- all C (`rmsnorm=c, swiglu=c, loss=c`): first loss fail at step ~65
- only `swiglu` in C: largest drift contribution (earliest fail / highest grad+logit drift)
- only `rmsnorm` in C: smaller but still non-zero contribution
- all torch reference: exact parity (0 drift)

Current caveats:
- Long-horizon AdamW parity for full C micro-stack is not yet within strict tolerance at large-step windows.
- Full automatic first-divergence localization to IR op ID in the *PyTorch parity harness* is still partial.
- Snapshot-oracle + generated runtime path has stronger diagnostics, but cross-harness convergence criteria still need unification.

PR3.6 gate wiring update (2026-02-16):
- Added `make v7-train-parity-drift-smoke` (default max_steps=70) for deterministic long-horizon drift detection.
- Added `make v7-train-parity-long-horizon` for full-run parity drift checks.
- `v7-gate-train` now runs drift-smoke by default (`V7_GATE_WITH_LONG_HORIZON_PARITY=1`).
- `make test` now runs drift-smoke by default (`CK_TEST_WITH_V7_LONG_HORIZON=1`).
- Current expected status: drift-smoke FAIL at ~step 65 on full C micro-stack until remaining long-horizon drift is fixed.

---

## PR8 - Long-Horizon Parity Gate Strategy (Stress + Realistic)
Goal: make long-horizon parity actionable by separating pathological stress from realistic blocker gates.

Why this PR:
- `Hello!` repeated-token stress reveals worst-case accumulation behavior quickly (first drift around step ~65 in AdamW `lr=1e-3`).
- Realistic/diverse text tracks production behavior better and already runs significantly longer before drift (loss parity can remain clean to 1000 steps).

Scope:
- Keep two explicit long-horizon tracks:
  - `stress_hello`: repeated tiny-text stress; expected to be harsh and non-blocking at first.
  - `realistic_long`: diverse text corpus; main blocking nightly gate.
- Add stable run configs in Make variables for both tracks (text, step budget, tolerances).
- Record both loss and parameter first-fail steps in artifacts:
  - `first_loss_fail_step`
  - `first_param_fail_step`
- Emit a compact verdict matrix in report JSON:
  - `stress_hello`: pass/warn/fail
  - `realistic_long`: pass/fail (blocking)

Proposed gate policy:
- Nightly BLOCKER:
  - `realistic_long` at >=300 steps must pass.
- Nightly NON-BLOCKING monitor:
  - `stress_hello` (track first-fail movement trend).
- Promotion target:
  - raise blocker from 300 -> 600 -> 1000 as drift is reduced.

Measured baseline (2026-02-16):
- `Hello!`, AdamW, `lr=1e-3`: loss drift starts ~65.
- Diverse text, AdamW, `lr=1e-3`, 300 steps: pass.
- Diverse text, AdamW, `lr=1e-3`, 1000 steps: loss passes, param drift first fails at ~468.

Exit criteria:
- Realistic blocker gate is green at configured horizon (initially 300).
- Stress gate is tracked with trendline and does not block merges.
- CI/nightly artifacts include both first-fail metrics and max diffs.

Status (2026-02-16):
- Implemented in `Makefile`:
  - `v7-train-parity-long-horizon-realistic` (realistic text, `max-steps=320` blocker)
  - `v7-backprop-long-epoch-nightly` now runs:
    1) realistic blocker (blocking)
    2) hello stress monitor (non-blocking)
- Verified pass at stable config (`lr=5e-4`):
  - realistic blocker: PASS
  - hello stress monitor: PASS

---

## PR9 - Full-C Cross-Entropy Tightening (Long-Run Drift)
Goal: eliminate or push out long-horizon parameter drift caused by full-C CE path under strict parity.

Why this PR:
- With all-C kernels on diverse text at `lr=1e-3`, 1000-step run can fail on parameter tolerance around step ~468.
- Replacing only CE path with torch (`--ck-loss-backend torch`) passes full 1000-step loss+param parity.
- This isolates remaining long-run drift primarily to the C CE/autograd integration path.

Scope:
- Tighten `CCrossEntropyFn` and CE gradient ownership/lifetime semantics in parity harness.
- Tighten `softmax_cross_entropy_loss` numerical path to match Torch reduction/order semantics as closely as possible.
- Add CE-focused long-horizon diagnostics:
  - per-step CE loss delta trend,
  - CE gradient max/mean delta trend,
  - worst-token CE contribution diff.
- Add CE stress unit/integration coverage:
  - extreme logit ranges,
  - repeated-token windows,
  - long-horizon optimizer windows.

Implementation notes:
- Preserve existing kernel API; prefer strict-mode branch refinements before introducing new kernels.
- Keep codegen dumb; CE math/order fixes belong in kernel + harness contract, not in generated scheduling logic.

Exit criteria:
- Full-C path (`rmsnorm=c, swiglu=c, loss=c`) passes realistic 1000-step parity at current tolerances.
- `first_param_fail_step` is eliminated (or moved beyond configured long-horizon gate target).
- CE-specific diagnostics are emitted and stable in nightly.

Status (2026-02-16):
- Re-verified isolation matrix at `lr=1e-3` (`Hello!`, 192 optimizer steps):
  - all torch: exact PASS
  - all C: FAIL in stress window (`first_loss_fail_step` typically ~65-114, `first_param_fail_step` ~177)
  - mixed backends can fail earlier than all-C, indicating cumulative-path sensitivity
    rather than a single isolated CE-only defect.
- Current priority remains long-run numerical tightening across strict C path; keep
  realistic blocker green while tracking stress trend movement.
- Added `make v7-train-parity-drift-localize` to emit same-state stage localization
  around target step (`drift-localize-step`, default 65) for reproducible root-cause triage.

---

## Near-Term Command Matrix (PR8/PR9)

```bash
# Stress monitor (non-blocking)
.venv/bin/python version/v7/scripts/train_parity_epochs_v7.py \
  --epochs 6 --seq-len 8 --total-tokens 32768 --grad-accum 8 \
  --optimizer adamw --lr 1e-3 --train-text "Hello!" \
  --max-steps 300 --ck-rmsnorm-backend c --ck-swiglu-backend c --ck-loss-backend c

# Realistic blocker (initially 300)
.venv/bin/python version/v7/scripts/train_parity_epochs_v7.py \
  --epochs 6 --seq-len 8 --total-tokens 32768 --grad-accum 8 \
  --optimizer adamw --lr 1e-3 \
  --train-text "the quick brown fox jumps over the lazy dog. this is a longer corpus line for parity stability checks." \
  --max-steps 300 --ck-rmsnorm-backend c --ck-swiglu-backend c --ck-loss-backend c

# CE isolation proof (target state reference)
.venv/bin/python version/v7/scripts/train_parity_epochs_v7.py \
  --epochs 8 --seq-len 8 --total-tokens 131072 --grad-accum 8 \
  --optimizer adamw --lr 1e-3 \
  --train-text "the quick brown fox jumps over the lazy dog. this is a longer corpus line for parity stability checks." \
  --max-steps 1000 --ck-rmsnorm-backend c --ck-swiglu-backend c --ck-loss-backend torch
```

### PR9 Progress Update (2026-02-16, later run)
- Re-ran stress localization with strict kernels: early step-65 drift no longer reproduces.
- Current stress signature moved right: `first_param_fail_step ~= 800` (`Hello!`, AdamW, `lr=1e-3`, 1000 steps), with `first_loss_fail_step=None`.
- Same run with only `--ck-loss-backend torch` passes to 1000 steps at current tolerances.
- Same run with SGD (all-C kernels) passes to 1000 steps.
- Added per-epoch parity snapshots (`epoch_snapshots`) in `train_parity_epochs_v7.py` to track drift growth trends and top-K parameter deltas across horizon.
