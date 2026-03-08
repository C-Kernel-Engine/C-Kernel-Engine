# v7 SVG Alignment Workflow (DPO / GRPO / PPO)

This workflow runs SVG alignment stages on top of an existing v7 run directory.

Current v7 status:
- `dpo/grpo/ppo` stage metadata is fully tracked in pipeline + visualizer.
- Objective-native artifacts are generated (`*.jsonl`).
- Training execution currently uses CE-surrogate rows (`*_ce_train.txt`) through `train_data_pipeline_v7.py`.

## 1) Prereqs

You should already have a run with:
- tokenizer trained and frozen
- pretrain/midtrain/SFT complete (or at least SFT dataset available)
- run dir under `~/.cache/ck-engine-v7/models/train/<name>`

## 2) Build alignment datasets only

```bash
python3 version/v7/scripts/build_svg_alignment_datasets_v7.py \
  --instruction-data "$RUN/data/${CK_NAME}_instruction_train.txt" \
  --out-dir "$RUN/alignment" \
  --prefix "$CK_NAME" \
  --max-samples 50000 \
  --seed 42
```

Outputs:
- `*_dpo_pairs.jsonl`
- `*_grpo_rollouts.jsonl`
- `*_ppo_preferences.jsonl`
- `*_dpo_ce_train.txt`
- `*_grpo_ce_train.txt`
- `*_ppo_ce_train.txt`
- `*_alignment_manifest.json`

## 3) Plan-only alignment workflow (recommended first pass)

```bash
bash version/v7/scripts/run_svg_alignment_stages_v7.sh \
  --run "$RUN" \
  --instruction-data "$RUN/data/${CK_NAME}_instruction_train.txt" \
  --out-dir "$RUN/alignment" \
  --prefix "$CK_NAME" \
  --max-samples 50000 \
  --epochs 1 --seq-len 512 --total-tokens 1048576 --grad-accum 1 \
  --lr-dpo 8e-5 --lr-grpo 6e-5 --lr-ppo 5e-5 \
  --plan-only \
  --run-dpo --run-grpo --run-ppo
```

This writes objective/native JSONL + CE fallback rows + plan summary, but does not run stage training.

## 4) Execute selected alignment stages

```bash
bash version/v7/scripts/run_svg_alignment_stages_v7.sh \
  --run "$RUN" \
  --instruction-data "$RUN/data/${CK_NAME}_instruction_train.txt" \
  --out-dir "$RUN/alignment" \
  --prefix "$CK_NAME" \
  --max-samples 50000 \
  --epochs 1 --seq-len 512 --total-tokens 1048576 --grad-accum 1 \
  --lr-dpo 8e-5 --lr-grpo 6e-5 --lr-ppo 5e-5 \
  --run-dpo --run-grpo --run-ppo
```

To run only one stage:
- DPO only: `--run-dpo`
- GRPO only: `--run-grpo`
- PPO only: `--run-ppo`

If no stage flags are passed, the script runs all three.

## 5) Visualizer artifacts

The stage runner writes:
- `alignment/alignment_stage_run_latest.json`
- `alignment/train_dpo_report.json`
- `alignment/train_grpo_report.json`
- `alignment/train_ppo_report.json`

And refreshes:
- `training_pipeline_latest.json`
- `training_plan.json`
- `ir_report.html` (unless `--skip-visualizer-refresh`)

In plan-only mode, the stage report files are not produced.

## 6) Notes

- Keep `--reuse-run-tokenizer` behavior for alignment stages (handled by runner).
- Keep token IDs stable across all stages in one run line.
- For strict parity gates, run `run_training_parity_regimen_v7.py` before and after alignment stages.
- Keep the run itself under `~/.cache/ck-engine-v7/models/train/<name>`.
  Do not stage temporary training runs under `version/v7/runs/`; the cache root is the canonical location for training artifacts and IR hub discovery.
- If you want a fresh `ir_report.html`, do not call `scripts/ck_chat.py --generate-visualizer`.
  `ck_chat.py` has no visualizer flag; use `version/v7/tools/open_ir_visualizer.py --generate --run "$RUN" --html-only --strict-run-artifacts`
  or `version/v7/scripts/ck_run_v7.py run ... --run "$RUN" --generate-visualizer`.
