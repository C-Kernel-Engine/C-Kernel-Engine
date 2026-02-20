# SVG Ablation Plan (2026-02-20)

## Goal
Determine the best next step for SVG training by isolating effects from:
- Tokenizer choice (`ascii`, `byte_bpe`, `hybrid_svg_bpe`)
- Data scale/quality (`current`, `expanded_dedup`, `expanded_dedup_augmented`)
- Model size (`small`, `medium`, `large`)

## Core Decision Rules
- If train loss low but holdout render quality weak: prioritize data/tokenizer, not model size.
- If both train and holdout are weak: increase model size after tokenizer/data baseline is clean.
- Start instruction fine-tuning only when holdout generalization is stable.

## Fixed Evaluation Set
- `in_dist`: held-out random SVG files from seen families.
- `family_holdout`: full held-out template/style families.
- `ood_comp`: compositional stress set (nested groups, transforms, text+path mixes, gradients).

Each split must remain fixed across all runs.

## 3x3x3 Ablation Matrix
Run all cells if budget allows. If budget is tight, run Phase 1 only, then prune.

| Phase | Tokenizer | Data | Model | Purpose |
|---|---|---|---|---|
| 1 | ascii | current | small | Baseline reference |
| 1 | byte_bpe | current | small | Tokenizer effect vs ascii |
| 1 | hybrid_svg_bpe | current | small | Domain-token hybrid check |
| 2 | best_phase1 | expanded_dedup | small | Data scale effect |
| 2 | best_phase1 | expanded_dedup_augmented | small | Data diversity effect |
| 3 | best_phase2 | best_phase2_data | medium | Capacity test |
| 3 | best_phase2 | best_phase2_data | large | Capacity saturation test |
| 4 | best_overall | best_overall_data | best_overall_size | Replication run A |
| 4 | best_overall | best_overall_data | best_overall_size | Replication run B |

## Metrics (per split)
- Token-level: NLL/perplexity.
- Structure validity: XML/SVG parse rate.
- Render quality: raster similarity (SSIM/LPIPS or equivalent).
- Novelty/memorization:
  - Nearest-neighbor token distance vs train corpus.
  - Nearest-neighbor render distance vs train corpus.
  - Memorization flag if generated sample is near-duplicate under both.

## Stop/Go Criteria
- Go to model scaling only if:
  - parse rate >= 99% on `family_holdout`, and
  - render metric improves on `family_holdout` and `ood_comp` with data changes.
- Go to instruction fine-tuning only if:
  - best base model passes above criteria, and
  - prompt-following failures are the main remaining issue.

## Instruction Fine-Tuning Entry Criteria
- Base model selected from Phase 4.
- SFT dataset with prompt->SVG pairs, coverage across style/task types.
- Keep a frozen base checkpoint and compare:
  - controllability gain
  - no regression on structure validity
  - no increase in memorization rate

## Recommended Immediate Runs (minimum useful set)
1. `ascii/current/small` (reconfirm baseline if needed)
2. `byte_bpe/current/small`
3. `hybrid_svg_bpe/current/small`
4. `best_tokenizer/expanded_dedup/small`
5. `best_tokenizer/expanded_dedup_augmented/small`

Only after these 5 runs, decide whether to test `medium`/`large`.
