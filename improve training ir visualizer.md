# Improve Training IR Visualizer — Full Analysis & v2 Manifest Design

**Date:** 2026-02-27
**Scope:** Complete audit of all guessing/fallback logic in the training pipeline visualizer and Python emitters, plus a clean v2 manifest design that eliminates all of it.

---

## Background

The IR Visualizer training tab (`version/v7/tools/src/training_tabs.js`) renders stage cards, dataset assignments, loss curves, tokenizer coverage, and a stage plan table. To do this it reads `training_pipeline_latest.json` — a file emitted by **two separate Python modules** that write overlapping but inconsistent subsets of fields.

The result: every piece of information the visualizer needs requires a multi-level fallback chain because no single field is reliably present. This document enumerates all 21 fallback chains in the JS and 3 heuristics in Python, explains why they exist, and proposes a v2 manifest that eliminates all of them.

---

## Part 1 — Root Cause: Two Emitters, One File

`training_pipeline_latest.json` is assembled from two independent sources:

### Emitter A — `ck_run_v7.py` / `_build_training_pipeline_payload()`

Called after each training run. Knows only the current run context:

| Field | How it's derived |
|---|---|
| `active_stage` | `train_mode` string |
| `stage_timeline[]` | Inferred: position in hardcoded list `["pretrain","midtrain","sft","dpo","grpo","ppo"]` relative to `active_idx` |
| `data_provenance[]` | Single entry for current run only — no cross-stage history |
| `tokenizer_lineage` | Rebuilt from `config.json`, `weights_manifest.json`, `operator_train_run.json` by trial |
| `execution` | From current training run summary |
| `train_dims` | From run summary |

### Emitter B — `train_data_pipeline_v7.py`

Called at pipeline setup time. Knows the full data pipeline:

| Field | How it's derived |
|---|---|
| `stage_sequence.entries[]` | From training plan stages |
| `dataset_catalog[]` | Filesystem scan + **name heuristics** (see Python Guess 1) |
| `stage_dataset_bindings[]` | Built from `dataset_catalog` (inherits all its heuristics) |
| `data_lab` | QC artifacts, dataset profile, tokenizer roundtrip |
| `tokenizer_lineage` | Actual tokenizer corpora paths |
| `stage_artifacts[]` | Per-stage artifact maps |
| `run_sequence[]` | Execution plan |

These two payloads share field names (`schema`, `active_stage`, `stage_timeline`, `data_provenance`, `tokenizer_lineage`, `execution`, `train_dims`) but populate them independently. They get partially merged via `_apply_training_plan_to_pipeline_payload()` when a training plan is available, but the merge is incomplete and context-dependent.

The same concept — "what dataset does the SFT stage use?" — ends up stored across:
- `data_provenance[stage='sft'].dataset_name`
- `dataset_catalog[stage='sft', kind='active_dataset'].name`
- `stage_dataset_bindings[stage='sft'].datasets[0].name`
- `stage_artifacts.sft.dataset_name`
- `stage_loss_history.entries[stage='sft'][last].dataset_name`

Each field may be missing, null, or stale depending on which emitter ran and when.

---

## Part 2 — All 21 JavaScript Fallback Chains

### Guess 1 — Stage ordering (`training_tabs.js` lines 459–527)

```javascript
// Level 1: stage_sequence.entries[] (from Emitter B)
if (stageSequenceEntries.length > 0) { ... }
// Level 2: stage_timeline[] (from Emitter A)
else if (timeline.length > 0) { ... }
// Level 3: keys from loss history
for (const s of fallbackStageOrder) { if (stageLossByStage[s]) addOrderedStage(s); }
// Level 4: hardcode
if (orderedStages.length === 0) {
  addOrderedStage('pretrain'); addOrderedStage('midtrain'); addOrderedStage('sft'); ...
}
```

Why: stage ordering is in `stage_sequence.entries[]` (Emitter B) or `stage_timeline[]` (Emitter A). They have different schemas: Emitter B uses `seq` (1-indexed), Emitter A uses `order` (0-indexed).

---

### Guess 2 — Stage name duality (line 431–436)

```javascript
function normStageName(value) {
    if (s === 'stage_a') return 'pretrain';  // curriculum alias
    if (s === 'stage_b') return 'midtrain';  // curriculum alias
    return s;
}
```

Why: Python uses both `stage_a`/`stage_b` (curriculum stage names) and `pretrain`/`midtrain` (logical stage names) in different fields of the same document.

---

### Guess 3 — Active stage (line 438)

```javascript
const activeStageKey = normStageName(activeStage) || String(activeStage || 'pretrain');
```

Why: `pipeline.active_stage` may be `"stage_b"`, `"midtrain"`, or missing.

---

### Guess 4 — Dataset name per stage (line 577, 5 levels)

```javascript
const dsName =
    bindDs.name                             // stage_dataset_bindings[stage].datasets[0].name
    || data.dataset_name                    // stage_artifacts[stage].dataset_name
    || cat.name                             // dataset_catalog[kind='active_dataset', stage].name
    || prov.dataset_name                    // data_provenance[stage].dataset_name
    || (latestLossRun && latestLossRun.dataset_name)  // loss history entry
    || null;
```

This is the central bug chain. For SFT: `bindDs.name` = correct instruction dataset. But `cat.name` (level 3) and `prov.dataset_name` (level 4) both pointed to `stage_b.txt` (midtrain data) because `dataset_catalog` was built from filesystem scan and `data_provenance` only tracked the previous (midtrain) run's data.

---

### Guess 5 — Dataset rows (line 584, 3 levels)

```javascript
dsRows: bindDs.rows || cat.rows || null,
```

---

### Guess 6 — Token count (line 585, 4 levels)

```javascript
dsTok: bindDs.tokens || data.token_count || prov.token_count
       || (latestLossRun && latestLossRun.total_tokens) || null,
```

---

### Guess 7 — Byte size (line 586, 2 levels)

```javascript
byteSize: bindDs.bytes || prov.byte_size || null,
```

---

### Guess 8 — Source path (line 587, 5 levels)

```javascript
sourcePath: bindDs.path || data.source_path || prov.source_path
            || cat.path || (latestLossRun && latestLossRun.source_path) || null,
```

---

### Guess 9 — Raw source path (line 588, 5 levels)

```javascript
rawSourcePath: bindDs.path || data.source_path || prov.source_path
               || cat.path || (latestLossRun && latestLossRun.raw_source_path) || null,
```

---

### Guess 10 — Token stream path (line 589)

```javascript
tokenStreamPath: (latestLossRun && latestLossRun.token_stream_path) || null,
```

---

### Guess 11 — Stage status (line 564, 3 levels)

```javascript
const status = String(
    t.status                                      // stage_timeline entry
    || seqMeta.status                             // stage_sequence entry
    || (stage === activeStageKey ? 'active' : 'planned')  // hardcoded inference
);
```

---

### Guess 12 — Active flag (line 565, 3 levels)

```javascript
const isActive = Boolean(
    t.active === true
    || seqMeta.active === true
    || stage === activeStageKey
);
```

---

### Guess 13 — Tokenizer corpus name (lines 383–385)

```javascript
const tokCorpusName =
    (Array.isArray(tokLineage.corpus_datasets) && tokLineage.corpus_datasets.length > 0)
    ? tokLineage.corpus_datasets[0].name
    : (tokCorpusEntry ? tokCorpusEntry.name : null);
```

Why: `tokenizer_lineage.corpus_datasets[]` (Emitter B) vs `dataset_catalog[kind='active_dataset', stage='pretrain']` (Emitter B catalog scan) — same information stored in two places.

---

### Guess 14 — Tokenizer coverage per stage (lines 388–405, 5 levels)

```javascript
function stageTokCoverage(stage) {
    // Level 1: explicit boolean on binding dataset
    if (bindDs && typeof bindDs.in_tokenizer_corpus === 'boolean') return bindDs.in_tokenizer_corpus;

    // Level 2: infer from count fields (not_in_corpus === 0 means covered)
    if (bind.tokenizer_coverage.not_in_corpus > 0 && in_corpus === 0) return false;
    if (in_corpus > 0 && not_in_corpus === 0) return true;

    // Level 3: explicit boolean on catalog entry
    if (cat && typeof cat.tokenizer_coverage === 'boolean') return cat.tokenizer_coverage;

    // Level 4: hardcoded stage exception
    if (stage === 'pretrain') return true;

    // Level 5: FILENAME SUBSTRING HEURISTIC
    return (nm.includes('stage_a') && !nm.includes('stage_b')) ? true : false;
}
```

Level 5 guesses tokenizer coverage from the dataset **filename**.

---

### Guess 15 — `tokManifestCoverage` per catalog row (lines 409–422, 3 levels)

```javascript
function tokManifestCoverage(row) {
    // Level 1: explicit field
    if (typeof row.tokenizer_coverage === 'boolean') return row.tokenizer_coverage ? 'yes' : 'no';

    // Level 2: 20-CHAR PREFIX FUZZY STRING MATCH
    const found = tokLineage.corpus_datasets.some((c) => {
        const cn = String(c.name || c || '').toLowerCase();
        return rn.includes(cn.slice(0, Math.min(cn.length, 20)));
    });
    return found ? 'yes' : 'no';

    // Level 3: hardcoded stage exception
    if (stage === 'pretrain') return 'inferred';
    return 'unknown';
}
```

Level 2 does a **20-character prefix substring match** between dataset names to guess corpus membership.

---

### Guess 16 — Stage sort key (lines 338–346, 4 levels)

```javascript
function stageOrderKey(row, idx) {
    const seq = Number(row.seq);      // Emitter B convention (1-indexed)
    if (Number.isFinite(seq)) return seq;
    const order = Number(row.order);  // Emitter A convention (0-indexed)
    if (Number.isFinite(order)) return order + 1;
    const index = Number(row.index);  // legacy field
    if (Number.isFinite(index)) return index + 1;
    return idx + 1;                   // position fallback
}
```

---

### Guess 17 — Training state label (lines 696–715)

```javascript
let trainState = 'planned';
if (stage === 'unassigned') {
    trainState = 'legacy-unlabeled';  // handles old runs with no stage tag
} else if (meta.isActive || status === 'active') {
    if (latestSteps >= estSteps) trainState = 'completed(last run)';  // infers completion from step count
    else trainState = 'active';
} else if (runs > 0 || status === 'completed') {
    trainState = 'trained';
} else if (meta.dsName) {
    trainState = 'ready';  // has dataset but no runs
}
```

State is **inferred** from a combination of: `isActive` flag, `status` string, number of historical loss runs, whether a dataset name exists, and whether observed steps >= estimated steps.

---

### Guess 18 — Raw path in ledger (lines 757–758)

```javascript
const rawPath = String(
    entry.raw_source_path
    || (entry.token_stream_path ? '-' : srcPath)
    || '-'
);
```

---

### Guess 19 — Seq len for step estimation (lines 684–686)

```javascript
const seqLenRaw = latest && Number.isFinite(Number(latest.seq_len))
    ? Number(latest.seq_len)
    : (meta.isActive && Number.isFinite(execSeqLen) ? execSeqLen : NaN);
```

Why: `execution.seq_len` (Emitter A) vs `stage_loss_history.entries[].seq_len` (loss artifact) — same value stored in different places with different precision.

---

### Guess 20 — Token count for step estimation (lines 687–689)

```javascript
const tokenRaw = Number.isFinite(Number(meta.dsTok))
    ? Number(meta.dsTok)
    : (latest && Number.isFinite(Number(latest.total_tokens)) ? Number(latest.total_tokens) : NaN);
```

---

### Guess 21 — Context length label (line 633)

```javascript
`ctx=${trainDims.context_length ?? '-'}`
```

`trainDims.context_length` (from `train_dims`) is sometimes absent; `execution.seq_len` is a separate field that tracks the same value under a different name. The visualizer uses only one without cross-checking.

---

## Part 3 — Python-Side Heuristics

### Python Guess 1 — Stage assignment by filename (`_infer_dataset_stage`, line 1314)

```python
def _infer_dataset_stage(name: str, active_stage: str) -> str:
    probe = str(name or "").lower()
    if any(tok in probe for tok in ("dpo",)):               return "dpo"
    if any(tok in probe for tok in ("grpo",)):              return "grpo"
    if any(tok in probe for tok in ("ppo", "rl")):          return "ppo"
    if any(tok in probe for tok in ("stage_b", "midtrain")): return "midtrain"
    if any(tok in probe for tok in ("instruction", "sft")):  return "sft"
    if any(tok in probe for tok in ("stage_a", "bridge", "assets", "ascii", "svg")): return "pretrain"
    return active_stage
```

All `dataset_catalog` stage assignments are derived entirely from **filename substring matching**. This is why `svg_pretrain_pack_l16d128_stage_b.txt` (contains `stage_b`) gets assigned to `midtrain`, even when it may be referenced from another stage.

---

### Python Guess 2 — `dataset_catalog` via filesystem scan (lines 1386–1448)

```python
# Scans three separate directories for any *.json manifest files:
manifest_paths.extend(sorted(dataset_path.parent.glob("*manifest.json")))
manifest_paths.extend(sorted(repo_data_dir.glob("svg*_manifest.json")))
manifest_paths.extend(sorted(autopilot_dir.glob("iter_*/*manifest*.json")))
```

Any manifest file found anywhere in those directories gets added to `dataset_catalog` — not just datasets explicitly declared in the training plan. This causes the catalog to include datasets from past experiments and autopilot iterations that were never used in the current run.

---

### Python Guess 3 — Stage completion status inferred from list position (line 2448–2454)

```python
stage_order = ["pretrain", "midtrain", "sft", "dpo", "grpo", "ppo"]
active_idx = stage_order.index(mode)
for idx, stage in enumerate(stage_order):
    if idx < active_idx:  status = "completed"   # INFERRED: earlier in list = completed
    elif idx == active_idx: status = "active"
    else: status = "planned"
```

Stage completion is inferred purely from **list position** — there is no recorded completion timestamp or explicit completion flag per stage. A stage is "completed" simply because it appears before the current `active_idx`.

---

## Part 4 — The v2 Manifest

### Design Principles

1. **One manifest file, two operations only**: *create* at pipeline setup, *patch* after each stage completes
2. **`pipeline[]` is the sole source of truth** — an ordered array of stages; `seq` is authoritative
3. **One dataset per training stage** — no catalog to pick from, no binding array to search
4. **Explicit status** on every stage — set at creation, updated on completion; never inferred from list position
5. **Model and tokenizer are top-level singletons** — written once, never duplicated
6. **No stage name duality** — only logical names: `pretrain`, `midtrain`, `sft`, `dpo`, `grpo`, `ppo`
7. **`metrics` and `checkpoint` are null until completed** — the visualizer renders `null` as `—`, not as a missing field

---

### Full v2 Schema

```json
{
  "schema": "ck-pipeline-manifest-v2",
  "generated_at": "2026-02-27T21:30:00Z",
  "run_id": "svg_l16_d128_h512_v1024_ctx512",
  "run_dir": "/home/antshiv/.cache/ck-engine-v7/models/train/svg_l16_d128_h512_v1024_ctx512",

  "model": {
    "family": "qwen3",
    "arch": "llama",
    "vocab_size": 1024,
    "embed_dim": 128,
    "num_layers": 16,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 16,
    "context_len": 512,
    "template": "qwen3",
    "checkpoint_path": null
  },

  "tokenizer": {
    "type": "ascii_bpe",
    "vocab_size": 1024,
    "path": "/path/to/tokenizer.json",
    "sha256": "51a5e35d3059fab160841a72bdc0bbe710044f03bd3713e1c3d8c684524f9ea7",
    "training_corpora": [
      {
        "name": "svg_pretrain_pack_l16d128_stage_b.txt",
        "path": "/path/to/data/generated/svg_pretrain_pack_l16d128_stage_b.txt",
        "rows": 63013,
        "bytes": 15703061,
        "sha256": "7a42158e11cb1aba713b5597b2efa6e35654992f9ca059481612784b2c6939c9"
      }
    ]
  },

  "pipeline": [
    {
      "seq": 0,
      "name": "data_preparation",
      "type": "data_prep",
      "status": "completed",
      "completed_at": "2026-02-20T10:00:00Z",
      "description": "ASCII-normalize raw SVG sources and pack to token budget",
      "inputs": [
        {
          "name": "svg_autopilot_8_svg_train.txt",
          "path": "/path/to/data/svg_autopilot_8_svg_train.txt",
          "rows": 49500,
          "bytes": null,
          "sha256": null
        }
      ],
      "outputs": [
        {
          "name": "svg_pretrain_pack_l16d128_stage_b.txt",
          "path": "/path/to/data/generated/svg_pretrain_pack_l16d128_stage_b.txt",
          "rows": 63013,
          "bytes": 15703061,
          "sha256": "7a42158e..."
        },
        {
          "name": "svg_pretrain_pack_l16d128_stage_b_syn_instruction_train.txt",
          "path": "/path/to/data/generated/svg_pretrain_pack_l16d128_stage_b_syn_instruction_train.txt",
          "rows": 36000,
          "bytes": 13166535,
          "sha256": null
        }
      ]
    },

    {
      "seq": 1,
      "name": "tokenizer_train",
      "type": "tokenizer_train",
      "status": "completed",
      "completed_at": "2026-02-20T10:30:00Z",
      "description": "Train ASCII BPE tokenizer on stage_b pack corpus",
      "corpus": [
        {
          "name": "svg_pretrain_pack_l16d128_stage_b.txt",
          "rows": 63013,
          "bytes": 15703061
        }
      ],
      "hyperparams": {
        "vocab_size": 1024,
        "min_freq": 2,
        "algorithm": "ascii_bpe"
      },
      "output": {
        "path": "/path/to/tokenizer.json",
        "sha256": "51a5e35d..."
      }
    },

    {
      "seq": 2,
      "name": "pretrain",
      "type": "training",
      "status": "completed",
      "completed_at": "2026-02-21T12:00:00Z",
      "description": "Stage A bridge — shape and form vocabulary foundation",
      "dataset": {
        "name": "svg_pretrain_pack_l16d128_stage_a_bridge.txt",
        "path": "/path/to/data/svg_pretrain_pack_l16d128_stage_a_bridge.txt",
        "rows": null,
        "bytes": null,
        "sha256": null,
        "in_tokenizer_corpus": false
      },
      "hyperparams": {
        "epochs": 10,
        "seq_len": 512,
        "lr": 3e-4,
        "optimizer": "adamw",
        "grad_accum": 1
      },
      "metrics": {
        "loss_start": null,
        "loss_end": null,
        "drop_pct": null,
        "steps": null,
        "tokens_processed": null
      },
      "checkpoint": {
        "path": null,
        "sha256": null,
        "step": null
      }
    },

    {
      "seq": 3,
      "name": "midtrain",
      "type": "training",
      "status": "completed",
      "completed_at": "2026-02-25T18:00:00Z",
      "description": "Stage B full corpus — dense SVG generation coverage",
      "dataset": {
        "name": "svg_pretrain_pack_l16d128_stage_b.txt",
        "path": "/path/to/data/generated/svg_pretrain_pack_l16d128_stage_b.txt",
        "rows": 63013,
        "bytes": 15703061,
        "sha256": "7a42158e...",
        "in_tokenizer_corpus": true
      },
      "hyperparams": {
        "epochs": 10,
        "seq_len": 512,
        "lr": 3e-4,
        "optimizer": "adamw",
        "grad_accum": 1
      },
      "metrics": {
        "loss_start": 3.211,
        "loss_end": 0.847,
        "drop_pct": 73.6,
        "steps": 12300,
        "tokens_processed": 6297600
      },
      "checkpoint": {
        "path": "/path/to/ckpt_midtrain_final.bin",
        "sha256": null,
        "step": 12300
      }
    },

    {
      "seq": 4,
      "name": "sft",
      "type": "training",
      "status": "active",
      "description": "Instruction tuning — <task>...</task><svg .../> pairs",
      "dataset": {
        "name": "svg_pretrain_pack_l16d128_stage_b_syn_instruction_train.txt",
        "path": "/path/to/data/generated/svg_pretrain_pack_l16d128_stage_b_syn_instruction_train.txt",
        "rows": 36000,
        "bytes": 13166535,
        "sha256": null,
        "in_tokenizer_corpus": false
      },
      "hyperparams": {
        "epochs": 10,
        "seq_len": 512,
        "lr": 1e-4,
        "optimizer": "adamw",
        "grad_accum": 1
      },
      "metrics": null,
      "checkpoint": null
    },

    {
      "seq": 5,
      "name": "dpo",
      "type": "training",
      "status": "planned",
      "description": "Preference alignment — ranked SVG quality pairs",
      "dataset": null,
      "hyperparams": null,
      "metrics": null,
      "checkpoint": null
    },

    {
      "seq": 6,
      "name": "grpo",
      "type": "training",
      "status": "planned",
      "dataset": null,
      "hyperparams": null,
      "metrics": null,
      "checkpoint": null
    },

    {
      "seq": 7,
      "name": "ppo",
      "type": "training",
      "status": "planned",
      "dataset": null,
      "hyperparams": null,
      "metrics": null,
      "checkpoint": null
    }
  ]
}
```

---

## Part 5 — How the Visualizer Simplifies

### Current code (21 fallback chains, ~60 lines of resolution logic)

```javascript
// Building stageCardMeta requires 4 lookup maps:
const provByStage = {};          // data_provenance[]
const catalogActiveByStage = {}; // dataset_catalog[kind='active_dataset']
const stageBindingByStage = {};  // stage_dataset_bindings[]
const stageLossByStage = {};     // stage_loss_history.entries[]

// Then per stage, 21 fallback chains:
const dsName = bindDs.name || data.dataset_name || cat.name
               || prov.dataset_name || latestLossRun.dataset_name || null;
const dsRows = bindDs.rows || cat.rows || null;
const covered = stageTokCoverage(stage); // 5-level function
const sourcePath = bindDs.path || data.source_path || prov.source_path
                   || cat.path || latestLossRun.source_path || null;
// ... 17 more chains
```

### v2 code (zero fallback chains, ~10 lines)

```javascript
// v2 manifest — direct field reads
function getStage(manifest, name) {
    return (manifest.pipeline || []).find(s => s.name === name) || null;
}

function buildStageCardMeta(manifest) {
    const meta = {};
    for (const s of manifest.pipeline.filter(p => p.type === 'training')) {
        meta[s.name] = {
            status:      s.status,
            isActive:    s.status === 'active',
            dsName:      s.dataset?.name       ?? null,
            dsRows:      s.dataset?.rows       ?? null,
            dsTok:       null,  // add to schema when tracked
            byteSize:    s.dataset?.bytes      ?? null,
            sourcePath:  s.dataset?.path       ?? null,
            covered:     s.dataset?.in_tokenizer_corpus ?? null,
            lossStart:   s.metrics?.loss_start ?? null,
            lossEnd:     s.metrics?.loss_end   ?? null,
            dropPct:     s.metrics?.drop_pct   ?? null,
            steps:       s.metrics?.steps      ?? null,
            seqLen:      s.hyperparams?.seq_len ?? null,
            checkpoint:  s.checkpoint?.path    ?? null,
            seq:         s.seq,
        };
    }
    return meta;
}
```

All helper functions become unnecessary:
- `provByStage` — eliminated
- `catalogActiveByStage` — eliminated
- `stageBindingByStage` — eliminated
- `primaryBindingDataset()` — eliminated
- `stageTokCoverage()` — eliminated (direct field)
- `tokManifestCoverage()` — eliminated (direct field)
- `normStageName()` — eliminated (v2 uses only logical names)
- `stageOrderKey()` — eliminated (`seq` is authoritative)

---

## Part 6 — Implementation Plan

### Step 1 — Define the v2 training plan config (input to pipeline)

Create `version/v7/config/training_plan_v2.json` with the full pipeline declared upfront:

```json
{
  "schema": "ck-training-plan-v2",
  "model": { "family": "qwen3", "embed_dim": 128, ... },
  "pipeline": [
    { "seq": 0, "name": "data_preparation", "type": "data_prep", ... },
    { "seq": 1, "name": "tokenizer_train", "type": "tokenizer_train", ... },
    { "seq": 2, "name": "pretrain", "type": "training",
      "dataset": { "name": "...", "path": "...", "rows": ..., "in_tokenizer_corpus": false } },
    { "seq": 3, "name": "midtrain", "type": "training", "dataset": { ... } },
    { "seq": 4, "name": "sft", "type": "training",
      "dataset": { "name": "...", "path": "...", "rows": 36000, "in_tokenizer_corpus": false } },
    { "seq": 5, "name": "dpo", "type": "training", "dataset": null },
    { "seq": 6, "name": "grpo", "type": "training", "dataset": null },
    { "seq": 7, "name": "ppo", "type": "training", "dataset": null }
  ]
}
```

No filesystem scanning. No name heuristics. Every dataset is declared by the operator.

---

### Step 2 — `train_data_pipeline_v7.py` — emit v2 manifest at pipeline start

Replace `_build_pipeline_payload()` return value with v2 schema. Build `pipeline[]` by reading from the training plan config — not from scanning the filesystem.

Key changes:
- Remove `_collect_dataset_catalog()` (filesystem scan → heuristic stage assignment)
- Remove `_infer_dataset_stage()` (filename heuristics)
- Remove `_build_stage_dataset_bindings()` (catalog lookup)
- Add `_build_v2_pipeline()` — reads `training_plan_v2.json`, fills in sha256/bytes/rows for datasets that exist, sets `status` from active_stage

---

### Step 3 — `ck_run_v7.py` — patch v2 manifest after each training stage

At end of training (after loss curve is emitted), load the current v2 manifest, find the matching pipeline stage by `name`, and patch:

```python
def _patch_v2_manifest_stage(run_dir: Path, train_mode: str, metrics: dict, checkpoint: dict):
    path = run_dir / "training_pipeline_latest.json"
    manifest = _load_json_dict(path)
    if manifest.get("schema") != "ck-pipeline-manifest-v2":
        return  # don't touch v1 manifests
    for stage in manifest.get("pipeline", []):
        if stage.get("name") == train_mode and stage.get("type") == "training":
            stage["status"] = "completed"
            stage["completed_at"] = _utc_now_iso()
            stage["metrics"] = metrics
            stage["checkpoint"] = checkpoint
            break
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
```

---

### Step 4 — Add v2 reader branch in `training_tabs.js`

```javascript
function renderTrainingPipeline(pipeline) {
    if (pipeline.schema === 'ck-pipeline-manifest-v2') {
        return renderTrainingPipelineV2(pipeline);
    }
    return renderTrainingPipelineV1(pipeline); // existing fallback chain code
}
```

The v2 renderer uses only direct field reads. The v1 renderer stays for backward compat with existing JSON artifacts.

---

### Step 5 — Emit loss curve per stage (not one global curve)

Currently: `training_loss_curve_latest.json` is one flat array of all steps, with no stage tag per entry.

With v2: emit per-stage loss curve files:
- `loss_curve_pretrain.json`
- `loss_curve_midtrain.json`
- `loss_curve_sft.json`

Each file referenced in `pipeline[n].training.loss_curve_path`.

The visualizer loads each file independently — stage boundary markers on the loss chart come from the stage `seq` order, not from embedded `source_stage` tags.

---

## Summary

| | Current v1 | Target v2 |
|---|---|---|
| Python emitters | 2 separate modules, overlapping fields | 1 create + 1 patch operation |
| Stage ordering | 4-level fallback | `seq` field, authoritative |
| Dataset per stage | 5-level fallback chain | `pipeline[n].dataset.name`, direct |
| Tokenizer coverage | 5-level function + filename heuristic | `pipeline[n].dataset.in_tokenizer_corpus`, direct |
| Stage name convention | `stage_a`/`stage_b` + `pretrain`/`midtrain` dual | logical names only |
| Dataset catalog | Filesystem scan + name heuristics | Declared in training plan config |
| JS helper maps | 4 lookup maps + 6 helper functions | 0 |
| JS fallback chains | 21 | 0 |
| Python guesses | 3 (filename heuristics, scan, position-based status) | 0 |
| Loss curve | One global file, no stage tags | One file per stage, referenced by path |
