# Eval Contracts (Dataset-Type Aware)

`eval_stage_v7.py` supports contract-driven evaluation via `--probe-config`.

Supported source lookup order:
1. `--probe-config <path>.json`
2. `RUN_DIR/eval_probes.json`
3. `RUN_DIR/eval_contract.json`
4. Built-in SVG defaults

## Run

```bash
python3 version/v7/scripts/eval_stage_v7.py \
  --run RUN_DIR \
  --all-stages \
  --probe-config version/v7/data/eval_contracts/c_codegen.v1.json
```

Regenerate report:

```bash
python3 version/v7/tools/open_ir_visualizer.py --generate --run RUN_DIR --html-only
```

The visualizer reads metric columns dynamically from `stage_eval_matrix.json`:
- stage table columns
- probe detail columns
- headline cards

## Contract schema (minimum)

```json
{
  "schema": "ck.eval_contract.v1",
  "dataset_type": "code",
  "scorer": "text_rules",
  "probes": [
    { "id": "p1", "type": "core", "prompt": "...", "expect_contains_all": ["..."] }
  ],
  "stage_metrics": [
    { "key": "coverage_rate", "source": "contains_all", "probe_type": "core", "format": "pct" }
  ],
  "probe_metrics": [
    { "key": "contains_all", "label": "Contains", "format": "pct" }
  ],
  "headline_metrics": ["coverage_rate"]
}
```

Notes:
- `probes[].type` defaults to `"all"` if omitted.
- `stage_metrics[].probe_type` accepts arbitrary labels (`all`, `core`, `ood`, etc.).
- `text_rules` scorer currently supports:
  - `expect_contains` / `expect_contains_all`
  - `expect_forbid`
  - `expect_prefix`
  - `expect_exact`
  - `expect_regex`

## GPT-3-style rigor applied to this pipeline

Use these as operational requirements, not optional extras:

1. Split discipline:
   - `_train`: gradient updates only
   - `_val`: loss tracking / early-stop decisions
   - `_test`: eval probes only (never used in training decisions)
2. Fixed eval protocol per dataset type:
   - keep one versioned contract file per task
   - avoid changing probes mid-series without bumping contract version
3. Contamination checks:
   - log hash/overlap checks proving test probes are absent from training corpus
4. Report uncertainty:
   - run with `--n-samples >= 3` and surface confidence notes
5. Promotion gates:
   - promote checkpoints only when primary contract metric passes threshold

This keeps stage-to-stage comparisons valid and prevents "loss went down but task quality regressed" decisions.
