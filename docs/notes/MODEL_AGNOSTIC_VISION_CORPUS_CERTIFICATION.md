# Model-Agnostic Vision Corpus Certification

## Objective

Generalize the existing private 40-image Qwen3-VL parity lane into a reusable
vision-model certification harness. The current lane is valuable because it
already provides deterministic corpus ordering, exact token comparison,
three-way native/Python/llama.cpp checks, resumable cases, private artifact
handling, timing, provenance, and fail-closed reporting. Those mechanisms
should be shared rather than copied for each vision family.

This is a harness refactor, not a request to weaken Qwen3-VL certification.
The existing Qwen3-VL command and report must remain compatible until the
generic runner has reproduced the complete 40-image result.

## Current Coupling

`certify_qwen3vl_llamacpp_corpus_v8.py` currently owns both generic corpus
behavior and Qwen3-VL policy. Model-specific assumptions include:

- decoder GGUF plus a separate `mmproj` GGUF;
- the `qwen3vl` chat template and suppressed-thinking mode;
- Qwen3-VL image token budgeting and bridge command;
- llama.cpp as the only oracle type;
- Qwen3-VL names in console output, report schemas, Make variables, and
  nightly suite names;
- one image per sample and one hard-coded OCR prompt.

The corpus manifest itself is close to reusable, but the runner currently
discards sample IDs, prompts, media types, and expected task metadata.

## Required Architecture

Split the implementation into three layers.

### 1. Generic corpus runner

The generic runner owns only model-independent behavior:

- manifest validation and deterministic sample order;
- media path resolution and SHA-256 identities;
- private output permissions and redaction;
- per-case configuration hashes and safe resume;
- process execution, logs, timeouts, cleanup, and continue-on-failure;
- exact sequence comparison and first-divergence reporting;
- per-stage timing and aggregate summary generation;
- runtime, compiler, model, oracle, circuit, and generated-artifact
  provenance;
- fail-closed handling of missing, stale, or incompatible evidence.

Suggested entry point:

```text
version/v8/scripts/certify_multimodal_corpus_v8.py
```

### 2. Model adapter

An adapter translates a generic sample into commands and normalized evidence.
It may describe model behavior, but it must not duplicate corpus lifecycle or
comparison logic.

The adapter contract should expose structured methods equivalent to:

```text
validate_artifacts(config)
capabilities()
build_subject_prefix(sample, case_dir)
build_subject_command(prefix, case_dir)
build_oracle_command(prefix, case_dir)
load_subject_result(case_dir)
load_oracle_result(case_dir)
normalize_result(result) -> token IDs, stop reason, text, timings, provenance
```

The first adapter is Qwen3-VL and should preserve current behavior exactly.
Later adapters can cover Gemma 3/4 vision and other multimodal families.
Prefer declarative model profiles and generated runtime capability metadata
over `if model_family == ...` branches in the generic runner.

An adapter must declare at least:

- adapter ID and schema version;
- supported media kinds and cardinality;
- model artifacts and their roles;
- subject runtime type and required generated ABI capabilities;
- oracle type, version/fingerprint, and decoding mode;
- prompt/chat-template ownership;
- tokenizer and stop-token ownership;
- image preprocessing and visual-token budget contract;
- supported comparison levels;
- context and generation limits.

### 3. Corpus manifest

The manifest describes test cases, not a model implementation. A versioned
shape should support:

```json
{
  "schema": "cke.multimodal_corpus",
  "schema_version": 1,
  "task": "ocr_structured_extraction",
  "samples": [
    {
      "id": "case-001",
      "inputs": [
        {
          "kind": "image",
          "path": "1.jpg",
          "sha256": "optional-pinned-hash"
        }
      ],
      "prompt": "Extract visible form fields as compact JSON.",
      "comparison": {
        "mode": "exact_pre_eos_tokens",
        "max_new_tokens": 128
      }
    }
  ]
}
```

Private manifests and media remain local. Portable nightly tests use public or
synthetic fixtures and must not download confidential data.

## Comparison Semantics

Do not call two models numerically equivalent merely because their decoded OCR
text is similar. The runner should support explicit comparison modes:

- `exact_pre_eos_tokens`: same model, tokenizer, prompt template, greedy
  policy, and compatible oracle; zero token differences;
- `exact_text`: decoded byte-for-byte output where token IDs are not directly
  comparable;
- `task_metric`: cross-model OCR evaluation against a separately supplied
  expected result, with a named metric and threshold.

Qwen3-VL CK versus Qwen3-VL llama.cpp remains
`exact_pre_eos_tokens`. Comparing Qwen3-VL with Gemma vision is a task-quality
comparison, not token parity. Keep these claims and report schemas distinct.

## Runtime Capability Requirements

Before a case runs, the adapter must query the generated runtime and reject a
missing capability. Relevant capabilities include:

- image input accepted;
- supported decoded pixel format or native image provider;
- multimodal prefix/bridge ABI;
- tokenizer and chat-template identity;
- deterministic greedy decoding;
- token trace output;
- context and visual-token limits.

The generic runner must never guess that a decoder accepts an image because an
`mmproj` path happens to be present.

## Migration Sequence

1. Extract pure corpus, redaction, resume, timing, comparison, and summary
   helpers without changing their output.
2. Define and validate the adapter protocol and generic manifest schema.
3. Implement the Qwen3-VL adapter by moving existing command construction and
   report normalization behind that protocol.
4. Keep `certify_qwen3vl_llamacpp_corpus_v8.py` as a thin compatibility wrapper
   and preserve `make test-qwen3vl-private-corpus-parity-auto`.
5. Add a generic target such as `test-v8-private-vision-corpus-auto` with an
   explicit adapter/profile argument. An unconfigured private lane must skip
   truthfully.
6. Reproduce selected 1-image and 5-image Qwen3-VL cases, then the complete
   40-image result with identical token traces and redacted summaries.
7. Add a second adapter only after Qwen3-VL compatibility is proven. Gemma
   vision is a useful next family because it exercises a different image
   frontend and prompt contract.

## Shared Campaign Command

The private campaign uses one manifest for every configured model lane:

```bash
make test-v8-private-vision-corpus-auto \
  V8_PRIVATE_VISION_CORPUS_MANIFEST=/private/ocr/manifest.json \
  V8_PRIVATE_VISION_CORPUS_REQUIRED_IMAGES=40
```

The aggregate target preserves separate evidence classes. Qwen3-VL and
Qwen3.6-VL retain exact same-model oracle parity. Cohere runs the same corpus
through both CKE and the pinned PyTorch reference, then reports task quality
separately from X-Ray tensor parity. A five-image, hash-pinned manifest is the
promotion gate before the complete forty-image corpus. Private images,
ground truth, prompts, and generated text remain outside the repository.

## Performance Debt Evidence

Correctness and performance are separate gates, but the same X-Ray run must
retain enough evidence to prioritize optimization. Each isolated CKE
checkpoint records wall time, child CPU time, average core equivalents,
configured threads, thread-utilization ratio, and idle core-seconds. Reports
rank cumulative checkpoint runs by lost core-seconds under
`performance_debt`. A rank identifies the interval requiring provider-level
measurement; it does not by itself assign all cumulative time to the selected
kernel. This makes a serial bridge, projector, or attention interval visible
without weakening its numerical threshold. Comparisons across commits are
valid only when model, image, geometry, ISA, thread count, and checkpoint
selector are unchanged.

## Required Tests

Add portable tests for:

- invalid schema, media kind, hash, and cardinality;
- unknown adapter and missing runtime capability;
- adapter result normalization into the canonical comparison shape;
- sample-specific prompts and generation limits;
- exact-token, exact-text, and task-metric modes remaining distinct;
- stale model, oracle, tokenizer, template, circuit, or runtime provenance;
- resume rejection after any relevant configuration change;
- private path, sample ID, prompt, and generated-text redaction;
- timeout, subprocess error, partial report, and interrupted-run recovery;
- compatibility wrapper command equivalence;
- synthetic two-adapter execution proving the generic runner contains no
  Qwen3-VL branch;
- nightly registration and truthful skip when private artifacts are absent.

## Completion Gates

The refactor is complete only when:

1. The existing Qwen3-VL 40-image corpus remains exact with unchanged
   tolerances and token limits.
2. Old and new Qwen3-VL commands produce equivalent per-case token traces and
   summaries.
3. The generic runner contains no model-family conditional.
4. A second vision adapter passes portable contract tests and at least one real
   end-to-end sample on supported hardware.
5. Reports identify corpus, adapter, model, tokenizer, template, subject,
   oracle, generated runtime, compiler, ISA, thread policy, and circuit hashes.
6. Private data never enters Git or public nightly artifacts.

## Xeon Agent Handoff

Work from a clean latest-main worktree. First audit the current Qwen3-VL
private-corpus producer, its tests, Make targets, nightly registration, bridge,
native CLI token trace, and llama.cpp parity producer. Implement the migration
above as a focused compatibility-preserving series. Do not duplicate the
40-image loop, relax exactness, invent cross-model token parity, download the
private corpus, or add model checks to the generic runner. Run portable tests
first, then use the configured private corpus for 1, 5, and 40 cases. Record
hashes and commands, clean temporary prefixes, and return an apply-checked
patch plus a report that separates public tests from private evidence.
