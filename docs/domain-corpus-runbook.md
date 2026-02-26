# Domain Corpus Runbook (C/C++ + Web + Math/Science)

This runbook defines a practical corpus pipeline for:

- `C`, `C++`, `SQL`, `Cypher`, `bash`, Linux docs
- `HTML`, `CSS`, `JS`, `SVG`
- technical English explanations for instruction tuning

It is designed for v7 workflows and strict ASCII training gates while keeping UTF-8 source-of-truth data.

## 1. Repo Layout

Use this structure under `version/v7/data/`:

```text
version/v7/data/
  corpus/
    raw/
      systems/
      web/
      docs_en/
    normalized/
      systems.jsonl
      web.jsonl
      docs_en.jsonl
    filtered/
      systems_filtered.jsonl
      web_filtered.jsonl
      docs_en_filtered.jsonl
    merged/
      pretrain_mix_utf8.jsonl
      pretrain_mix_ascii.txt
      pretrain_mix_ascii_manifest.json
      utf8_to_ascii_table.tsv
      utf8_to_ascii_table.json
    sft/
      instruction_utf8.jsonl
      instruction_ascii.txt
      instruction_manifest.json
  manifests/
    source_registry.json
    license_policy.json
```

## 2. JSONL Schema

Every normalized row should follow one schema.

```json
{
  "id": "sha1:...",
  "domain": "c|cpp|sql|cypher|bash|linux|html|css|js|svg|math|science|docs_en",
  "task": "pretrain|explain|debug|generate|qa|refactor",
  "text": "single training sample text",
  "source": "repo/path/or/url",
  "project": "nginx|postgresql|mdn|...",
  "license": "MIT|Apache-2.0|BSD-3-Clause|GPL-2.0|...",
  "lang": "en",
  "quality": {
    "syntax_pass": true,
    "lint_pass": false,
    "exec_pass": false,
    "score": 0.0
  },
  "stats": {
    "chars": 0,
    "lines": 0
  }
}
```

Required fields:

- `id`, `domain`, `text`, `source`, `license`, `quality`

## 3. License and Source Policy

Keep two dataset classes:

1. `permissive`: MIT/BSD/Apache/ISC
2. `copyleft`: GPL/LGPL/AGPL (separate shard)

Do not merge classes unless explicitly intended for that run.

Example `source_registry.json` entry:

```json
{
  "project": "postgresql",
  "source_type": "git",
  "path": "/data/src/postgresql",
  "license": "PostgreSQL",
  "class": "permissive",
  "domains": ["c", "sql", "docs_en"]
}
```

## 4. Ingestion Checklist

1. Collect raw source snapshots into `corpus/raw/*`.
2. Normalize to JSONL (`text` only payload per row, one row per sample unit).
3. Attach provenance and license metadata.
4. Run domain syntax gates:
   - `c/cpp`: parser/compile gate on sampled files.
   - `sql/cypher`: parser gate.
   - `bash`: parse + lint gate.
   - `html/css/js/svg`: parser + SVG/XML validity.
5. Deduplicate exact + near-duplicate rows.
6. Remove boilerplate, binary blobs, generated/minified files.
7. Create merged UTF-8 mix (`pretrain_mix_utf8.jsonl`) with target ratios.
8. Project UTF-8 to ASCII training text and emit mapping tables.
9. Run dataset QC and tokenizer roundtrip gates.
10. Freeze manifest hashes before training.

## 5. Domain Quality Gates

Track these metrics per domain:

- row count
- avg/max line length
- syntax pass rate
- dedup drop rate
- non-ASCII conversion count
- unmapped symbol count

Hard fail conditions for strict runs:

- unmapped non-ASCII symbols > 0
- ASCII output contains non-ASCII bytes
- syntax pass below threshold for target domain

## 6. UTF-8 -> ASCII Projection Contract

Use UTF-8 as canonical source, then produce strict ASCII for training.

Artifacts:

- `pretrain_mix_ascii.txt`
- `pretrain_mix_ascii_manifest.json`
- `utf8_to_ascii_table.tsv`
- `utf8_to_ascii_table.json`

Manifest should include:

- `rows_in`, `rows_out`
- `rows_changed`
- `non_ascii_chars_converted`
- `mapped_total_chars`, `unmapped_total_chars`
- `output_non_ascii_chars`

## 7. Pretrain Mixture Targets

Start with this mixture:

- 35% systems code (`c/cpp/sql/bash/cypher`)
- 25% web code (`html/css/js/svg`)
- 25% technical English docs (`docs_en`, math/science prose)
- 15% instruction-style technical pairs (for later SFT bootstrapping)

Adjust only after eval changes are measured.

## 8. Sample Build Commands (v7-Compatible)

Use existing scripts where applicable:

```bash
python3 version/v7/scripts/build_svg_corpus_from_assets_v7.py \
  --assets-dir docs/site/assets \
  --output version/v7/data/corpus/normalized/web_svg_utf8.txt \
  --manifest version/v7/data/corpus/normalized/web_svg_utf8_manifest.json
```

```bash
python3 version/v7/scripts/prepare_ascii_dataset_v7.py \
  --input version/v7/data/corpus/merged/pretrain_mix_utf8.jsonl \
  --input-format jsonl \
  --jsonl-text-key text \
  --output version/v7/data/corpus/merged/pretrain_mix_ascii.txt \
  --ascii-map-common \
  --ascii-mode xml_escape
```

```bash
python3 version/v7/scripts/test_ascii_bpe_roundtrip_v7.py \
  --run "$RUN" \
  --dataset version/v7/data/corpus/merged/pretrain_mix_ascii.txt \
  --require-ascii
```

## 9. Training Integration

For strict pretrain runs, set:

- `--tokenizer ascii_bpe`
- `--require-svg-rows` only for SVG-specific runs
- `--strict-data-gates`
- explicit `--data` path to frozen ASCII shard

Use run dirs under:

- `~/.cache/ck-engine-v7/models/train/<run_name>`

This keeps IR hub tracking consistent.

## 10. SFT/Instruction Data Plan

After base pretraining:

1. Build technical instruction pairs from your corpus:
   - explain code
   - fix bug
   - write query/script
   - generate UI/SVG component
2. Keep answer style concise and technical.
3. Avoid noisy synthetic long chain-of-thought dumps.
4. Validate response format with domain parsers where possible.

## 11. Minimum Reproducibility Bundle

For each run, keep:

- corpus manifest hash
- tokenizer JSON hash
- UTF-8 to ASCII mapping hash
- dataset QC report
- roundtrip report
- training config JSON

This is the minimum needed to explain why a run behaved a certain way.
