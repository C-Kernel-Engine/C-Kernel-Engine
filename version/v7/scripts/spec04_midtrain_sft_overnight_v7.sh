#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN="${1:-/home/antshiv/.cache/ck-engine-v7/models/train/spec04_structured_scenes_ctx512_d64_h128_v224}"

MIDTRAIN_DATA="$RUN/dataset/midtrain/train/spec04_structured_svg_atoms_midtrain_train.txt"
SFT_DATA="$RUN/dataset/sft/train/spec04_structured_svg_atoms_sft_train.txt"
PROBE_CONTRACT="/tmp/spec04_structured_scenes_probe_contract.json"
MIDTRAIN_TOTAL_TOKENS="${MIDTRAIN_TOTAL_TOKENS:-393216}"
SFT_TOTAL_TOKENS="${SFT_TOTAL_TOKENS:-393216}"

cd "$ROOT"

build_probe_contract() {
  python3 version/v7/scripts/build_spec04_probe_contract_v7.py \
    --run "$RUN" \
    --output "$PROBE_CONTRACT" \
    --per-split 10
}

refresh_reports() {
  python3 version/v7/scripts/build_svg_training_report_card_v7.py \
    --run "$RUN" \
    --output "$RUN/svg_training_report_card.html"
  python3 version/v7/scripts/build_probe_report_v7.py \
    --run "$RUN" \
    --contract "$PROBE_CONTRACT" \
    --output "$RUN/spec04_probe_report.html" \
    --json-out "$RUN/spec04_probe_report.json"
  python3 version/v7/scripts/build_spec04_capability_report_v7.py \
    --run "$RUN" \
    --output "$RUN/spec04_capability_report.html"
  python3 version/v7/tools/open_ir_visualizer.py \
    --generate \
    --run "$RUN" \
    --html-only \
    --strict-run-artifacts \
    --output "$RUN/ir_report.html"
}

build_probe_contract

python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run "$RUN" --stage pretrain

.venv/bin/python version/v7/scripts/train_data_pipeline_v7.py \
  --run "$RUN" \
  --data "$MIDTRAIN_DATA" \
  --tokenizer ascii_bpe \
  --reuse-run-tokenizer \
  --pack-mode sample \
  --no-pack-total-tokens-from-windows \
  --curriculum-stage stage_b \
  --seq-len 128 \
  --epochs 2 \
  --total-tokens "$MIDTRAIN_TOTAL_TOKENS" \
  --grad-accum 1 \
  --optimizer adamw \
  --lr 2.5e-4 \
  --strict-data-gates \
  --no-post-train-eval \
  --json-out "$RUN/train_structured_svg_scenes_stage_b_blended.json"

.venv/bin/python version/v7/scripts/eval_stage_v7.py \
  --run "$RUN" \
  --all-stages \
  --n-samples 5

refresh_reports

python3 version/v7/scripts/promote_latest_checkpoint_v7.py --run "$RUN" --stage midtrain

.venv/bin/python version/v7/scripts/train_data_pipeline_v7.py \
  --run "$RUN" \
  --data "$SFT_DATA" \
  --tokenizer ascii_bpe \
  --reuse-run-tokenizer \
  --pack-mode sample \
  --no-pack-total-tokens-from-windows \
  --curriculum-stage sft \
  --seq-len 256 \
  --epochs 2 \
  --total-tokens "$SFT_TOTAL_TOKENS" \
  --grad-accum 1 \
  --optimizer adamw \
  --lr 6e-5 \
  --strict-data-gates \
  --no-post-train-eval \
  --json-out "$RUN/train_structured_svg_scenes_sft_rebalanced.json"

.venv/bin/python version/v7/scripts/eval_stage_v7.py \
  --run "$RUN" \
  --all-stages \
  --n-samples 5

refresh_reports

printf 'spec04 overnight midtrain+sft finished for %s\n' "$RUN"
