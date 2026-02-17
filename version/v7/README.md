# C-Kernel-Engine v7

v7 is the training foundation track.

Scope for `v7.0`:
- deterministic, fp32-first training contracts
- single-token (`T=1`) PyTorch parity for core backward-capable kernels
- strict gate before adding bf16/perf work

Deferred to `v7.2`:
- bf16 compute with fp32 master weights
- activation recompute/checkpoint policies
- threaded backward fast mode and deeper perf tuning

## Quick Start

```bash
make v7-help
make v7-sync-inference
make v7-infer-run
make v7-validate-contracts
make v7-parity-1tok
make v7-qk-norm-backward-parity
make v7-qk-norm-backward-parity-isa
make v7-qk-norm-backward-parity-isa-strict
make v7-train-ir-smoke
make v7-train-codegen
make v7-train-compile-smoke
make v7-init-tiny
make v7-grad-fd
make v7-replay
make v7-train-parity-3
make v7-train-parity-5
make v7-backprop-production-ready
make v7-gate-train
make v7-gate
```

## v7 Visualizer Dev (No Node)

Training-view extensions for the v7 visualizer live in native ES modules:
- `version/v7/tools/src/*.js`

Bundle them into the single-file report HTML with:

```bash
version/v7/tools/build_ir_visualizer_bundle.sh
```

This keeps runtime artifacts self-contained (no external JS build dependency).

## Deliverables in v7.0

- `contracts/IR_CONTRACT.md`
- `contracts/KERNEL_CONTRACT.md`
- `contracts/RUNTIME_CONTRACT.md`
- `scripts/validate_v7_contracts.py`
- `scripts/run_parity_1token_v7.py`
- `scripts/check_qk_norm_backward_parity_v7.py`
- `scripts/check_fd_gradients_v7.py`
- `scripts/check_replay_determinism_v7.py`
- `scripts/build_ir_train_v7.py`
- `scripts/lower_ir2_backward_v7.py`
- `scripts/validate_ir_train_invariants_v7.py`
- `scripts/train_parity_epochs_v7.py`
- `scripts/codegen_train_runtime_v7.py`
- `scripts/init_tiny_train_model_v7.py`

## Notes

- `v7` is intentionally strict and minimal: correctness first, optimization later.
- `v7-gate` should remain deterministic and reproducible on CPU-only environments.
- `v7-validate-contracts` now verifies that training-required kernels are both registered and bound in
  `kernel_maps/KERNEL_REGISTRY.json` and `kernel_maps/kernel_bindings.json`.
- `v7-train-ir-smoke` defaults to strict unresolved policy (`V7_TRAIN_STRICT_UNRESOLVED=1`) and
  non-partial coverage (`V7_TRAIN_ALLOW_PARTIAL=0`).
- Training IR reads from `weights_manifest.json` (manifest is the source of truth for weight tensors).
- `v7-train-codegen` emits compile-ready C from `ir2_train_backward_latest.json`.
- Generated `ck_train_optimizer_step()` now calls `adamw_update_f32` per `grad.weight.*` tensor.
- `v7-train-compile-smoke` compiles that generated C to an object as an operator gate.
- `v7-init-tiny` provides from-scratch tiny-model initialization (`weights.bump` + `weights_manifest.json`).
- `v7-parity-1tok` now includes `qk_norm_backward` parity in addition to RMSNorm, SwiGLU, and CE.
- Long-horizon backprop targets now pass explicit train safety controls to parity harness:
  `--max-grad-norm`, `--enforce-production-safety`, and AdamW LR threshold checks.
- `make v7-backprop-production-ready` runs `v7-gate-train` + nightly long-horizon bundle with production safety enforcement.
- `profile-v7-vtune` now supports deep capture (`V7_VTUNE_DEEP=1`): hotspots + memory-access + uarch-exploration, exported to `vtune_summary.json` for the IR viewer.
- Inference baseline is synced from `version/v6.6` into `version/v7` so inference and
  backprop can evolve together in one track.
