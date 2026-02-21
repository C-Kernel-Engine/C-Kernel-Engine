# v7 Regression Ledger

Canonical ledger for v7 training/parity regressions and guardrails.

Purpose:
- keep one durable record of root causes and fixes;
- map each fix to a gate/test and artifact proof;
- prevent new model architectures from repeating old failure patterns.

Machine-readable mirror:
- `version/v7/reports/REGRESSION_LEDGER.json`

## Entries

### V7-2026-02-18-ATTN-SFB
- Title: Attention save-for-backward contract broken by flash forward path
- Status: fixed
- Root cause: Generated runtime used flash forward while backward consumed saved `attn_weights`.
- Fix:
  - `version/v7/scripts/codegen_train_runtime_v7.py`
- Detection gates:
  - `make v7-backprop-stitch-runtime`
  - `make v7-train-runtime-parity-stress`
- Proof artifacts:
  - `version/v7/.cache/reports/attention_save_for_backward_fix_2026-02-18/backprop_grad_slots_step1_before.json`
  - `version/v7/.cache/reports/attention_save_for_backward_fix_2026-02-18/backprop_grad_slots_step1_after.json`
  - `version/v7/.cache/reports/attention_save_for_backward_fix_2026-02-18/train_1tok_after_fix.json`

### V7-2026-02-18-GRAD-ACT-ACCUM
- Title: Grad-activation buffers persisted across micro-steps when `grad_accum > 1`
- Status: fixed
- Root cause: Generated runtime only cleared grads at window start; `grad_activations` must reset every micro-step.
- Fix:
  - `version/v7/scripts/codegen_train_runtime_v7.py`
  - `version/v7/scripts/check_backprop_stitch_runtime_v7.py`
- Detection gates:
  - `make v7-backprop-stitch-runtime-accum`
  - `make v7-train-runtime-parity-stress`
- Proof artifacts:
  - `version/v7/.cache/reports/train_runtime_parity_stress_e1_ga2_after_gradactfix.json`
  - `version/v7/.cache/reports/train_runtime_parity_stress_e1_ga4_after_gradactfix.json`
  - `version/v7/.cache/reports/train_runtime_parity_stress_e1_ga8_after_gradactfix.json`

### V7-2026-02-17-RUN-DIMS
- Title: Train/parity path ignored run-dir manifest dims and used tiny defaults
- Status: fixed
- Root cause: `ck_run_v7.py train` used tiny defaults unless explicit overrides, even with `--run`.
- Fix:
  - `version/v7/scripts/ck_run_v7.py`
- Detection gates:
  - `make v7-train-runtime-parity-stress`
  - `make v7-train-runtime-parity-realistic`
- Proof artifacts:
  - `version/v7/reports/PARITY_DRIFT_STATUS_2026-02-17.md`
  - `version/v7/.cache/reports/parity_drift_2026-02-17/v7_l24_every8_1024_after_dimfix_venv.json`

### V7-2026-02-17-CE-SEMANTICS
- Title: Cross-entropy semantic and numeric mismatch created long-horizon drift
- Status: fixed
- Root cause: CE reduction/ignore-index semantics and numeric path were not fully aligned with PyTorch.
- Fix:
  - `src/kernels/loss_kernels.c`
  - `version/v7/scripts/train_parity_epochs_v7.py`
- Detection gates:
  - `make v7-train-parity-drift-smoke`
  - `make v7-train-parity-long-horizon-realistic`
- Proof artifacts:
  - `docs/site/_pages/v7-cross-entropy-parity.html`
  - `version/v7/.cache/reports/train_parity_drift_smoke_latest.json`
  - `version/v7/.cache/reports/train_parity_realistic_long_horizon_latest.json`

### V7-2026-02-18-OPT-STATE-REPLAY
- Title: Optimizer-state divergence could hide behind weight-only replay checks
- Status: guardrail_added
- Root cause: Earlier replay checks did not explicitly compare optimizer-state snapshots.
- Fix:
  - `version/v7/scripts/ck_run_v7.py`
  - `version/v7/scripts/codegen_train_runtime_v7.py`
- Detection gates:
  - `make v7-replay-accum`
  - `make v7-train-runtime-parity-long-horizon`
- Proof artifacts:
  - `version/v7/.cache/reports/replay_accum_latest.json`
  - `version/v7/.cache/reports/train_runtime_parity_stress_latest.json`

## Update Rule

For every new regression or hardening event:
1. Add an entry in this ledger.
2. Link at least one deterministic gate command.
3. Link at least one proof artifact JSON.
4. Keep status current (`open`, `fixed`, `guardrail_added`, `monitoring`).
