# v7 SVG Training Ablation Guide

This page is a quick entry point for SVG training decisions in v7.

Canonical plan:
- `version/v7/reports/SVG_ABLATION_PLAN_2026-02-20.md`

## Recommended Order
1. Validate generalization before scaling model size.
2. Run tokenizer sweep first (`ascii`, `byte_bpe`, `hybrid_svg_bpe`) at fixed small model/data.
3. Expand and clean data (dedup + diversity/augmentation) with best tokenizer.
4. Scale model size only if underfitting remains.
5. Start instruction fine-tuning only after base SVG generalization is stable.

## Why This Order
- Better data and tokenization usually reduce memorization risk faster than increasing capacity.
- Bigger embeddings/models can amplify memorization when holdout quality is still weak.
- Instruction tuning improves controllability; it is not the first fix for weak base modeling.

## Minimum Useful Run Set
- `ascii/current/small`
- `byte_bpe/current/small`
- `hybrid_svg_bpe/current/small`
- `best_tokenizer/expanded_dedup/small`
- `best_tokenizer/expanded_dedup_augmented/small`

If these five runs do not show holdout improvement, revisit data construction and split design before testing `medium`/`large`.
