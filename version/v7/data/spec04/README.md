# spec04

`spec04` is a v7 data workspace for the `svg` dataset family.

## Goal

Render structured, valid SVG that follows a closed control contract.

## Layout

- `contracts/`
- `raw_assets/`
- `normalized/`
- `pretrain/train`, `pretrain/dev`, `pretrain/test`
- `midtrain/train`, `midtrain/dev`, `midtrain/test`
- `sft/train`, `sft/dev`, `sft/test`
- `holdout/` (optional canary / OOD / reserved eval bucket)
- `tokenizer/`
- `manifests/`

## Workflow

1. import and inventory source data into `raw_assets/`
2. normalize and placeholderize into `normalized/`
3. derive stage corpora into explicit `train/dev/test` splits
4. build tokenizer corpus in `tokenizer/`
5. keep optional canary / OOD / frozen prompts in `holdout/`
6. store inventory, dedupe, coverage, and fit reports in `manifests/`

## Operator note

This repo workspace is a seed template only.

After staging into a run, keep the working dataset copy, `dataset_viewer.html`,
`ir_report.html`, checkpoints, parity/perf JSON, and other generated training
artifacts together under one cache run directory such as
`~/.cache/ck-engine-v7/models/train/<run-name>`.

## Contract rule

Do not mix incompatible row formats inside one workspace.

Keep:
- one canonical input/output contract
- one tokenizer corpus policy
- one eval contract
- explicit split intent per stage, even if some splits are empty today

Split experimental format changes into a new workspace instead of mutating history in place.
