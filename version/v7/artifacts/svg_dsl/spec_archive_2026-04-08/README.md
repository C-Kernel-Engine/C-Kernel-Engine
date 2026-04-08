# v7 SVG DSL Spec Archive

This archive holds the older SVG/spec report surfaces that used to live
directly under `version/v7/reports/`.

Archived here:

- `spec12_gold_mappings/`
- `spec13b_gold_mappings/`
- `spec14a_gold_mappings/`
- `spec14b_gold_mappings/`
- `spec15a_gold_mappings/`
- `spec15b_gold_mappings/`
- `spec_broader_1_family_packs/`
- `spec_broader_1_gold_mappings/`
- `asset_dsl_coverage_audit_2026-04-04.{json,md}`
- `spec_broader_1_asset_plan_2026-04-04.{json,md}`
- `spec_broader_1_bootstrap_queue_2026-04-04.{json,md}`

Compatibility rule:

- `version/v7/reports/...` now keeps symlinks for these paths so older
  scripts, tests, and manifest snapshots continue to resolve.
- New SVG/spec/gen1 generated report outputs should go here rather than
  recreating real directories under `version/v7/reports/`.
