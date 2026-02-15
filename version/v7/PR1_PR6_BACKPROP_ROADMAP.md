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
