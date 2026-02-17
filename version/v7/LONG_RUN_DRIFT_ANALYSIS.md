# v7 Long-Run Drift Analysis (CK Runtime)

## Run Config
- backend: ck
- epochs: 3
- seq_len: 8
- total_tokens: 4096
- grad_accum: 8
- vocab/d_model/hidden: 1024/256/1024
- parity: --parity-on --parity-every 50 --parity-profile balanced

## Before Fix
- report: `version/v7/.cache/reports/train_parity_long_horizon_latest.json` (example output path)
- oracle failures: 23
- first failing step: 400 (loss_diff=6.29425048828125e-05, logits_max_abs_diff=5.245208740234375e-06)
- worst failing step: 1500 (loss_diff=15.445653915405273, logits_max_abs_diff=2.288818359375e-05)
- last failing step: 1500 (loss_ck=23.025850296020508, loss_oracle=38.47150421142578)
- final_ck_loss=23.025850296020508 final_torch_loss=38.47150421142578

## Root Cause
- `softmax_cross_entropy_loss` used `-log(prob + 1e-10)` for loss reporting.
- This hard-caps per-token loss near `-log(1e-10) = 23.02585`, causing loss-only drift at long horizon.
- Oracle slot/logit checks stayed tight while scalar loss diverged, confirming loss-semantic mismatch (not forward tensor drift).

## Fix Applied
- file: `src/kernels/loss_kernels.c`
- changed loss term to stable log-sum-exp form:
  - old: `-logf(drow[target] + 1e-10f)`
  - new: `-(target_logit - max_logit - log(sum_exp))`
- gradient path remains unchanged (`drow[target] -= 1`).

## After Fix
- report: `version/v7/.cache/reports/train_parity_long_horizon_latest.json` (same harness after CE fix)
- oracle failures: 0
- pass_parity=True
- max_loss_abs_diff=3.814697265625e-06
- final_ck_loss=39.213863372802734 final_torch_loss=38.47150421142578

## Additional Related Fix During Investigation
- file: `version/v7/scripts/lower_ir2_backward_v7.py`
- `aux.d_scores` now inherits shape/numel from saved attention weights, fixing strict canary failure in backward attention.

## Repro Commands
```bash
make --no-print-directory build/libckernel_engine.so
CK_NUM_THREADS=8 .venv/bin/python version/v7/scripts/ck_run_v7.py train \
  --run version/v7/runs/drift_repro --backend ck \
  --train-epochs 3 --train-seq-len 8 --train-total-tokens 4096 --train-grad-accum 8 \
  --train-vocab 1024 --train-d-model 256 --train-hidden 1024 \
  --profile-train none --parity-on --parity-every 50 --parity-profile balanced \
  --train-json-out version/v7/.cache/reports/train_parity_long_horizon_latest.json
```
