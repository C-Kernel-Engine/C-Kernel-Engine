# v7 Inference Compatibility Plan (Training-First Scope)

## Scope

v7 remains training/backprop-first. Inference compatibility work is in scope only when it:

1. prevents obvious regressions in supported families, or
2. adds deterministic bring-up checks for new GGUF models.

Goal: fix inference issues methodically without turning v7 into an inference-only branch.

## Current Nanbeige Findings

### Confirmed facts

1. GGUF chat metadata exists and is loaded.
   - `tokenizer.chat_template` is present in run config.
   - special tokens (`<|im_start|>`, `<|im_end|>`) are present.
2. SentencePiece itself is not the blocker.
   - Nanbeige tokenizer path is `LlamaTokenizer` + SP.
   - Gemma also uses SP in this codebase and can produce coherent output.
3. GGUF-to-BUMP weight category coverage is passing.
   - `missing_category_count = 0` in `gguf_weight_category_coverage.json`.
4. Nanbeige has untied LM head in source config (`tie_word_embeddings=false`) and has `output.weight`.
5. Layerwise QKV contract checks can pass while generation is still poor.
   - This means the main remaining issues are not only in QKV kernels.

### Concrete mismatch observed

For this Nanbeige run:

1. Manifest records `output.weight` (q6_k) and quant summary shows `lm_head=q6_k`.
2. Lowered/codegen logits path in generated artifacts still references `W_TOKEN_EMB` (q4_k) for logits.

This points to a contract propagation/invalidation issue in IR->codegen, not a vocab-size issue and not an SP parser issue.

## Why this fails despite "QKV pass"

QKV checks validate projection math on selected ops/layers.
They do not guarantee:

1. final logits source selection is correct (`lm_head` vs `token_emb`),
2. all generation-time control-token/stop policies are aligned,
3. run artifacts are rebuilt after contract changes.

So model can pass local projection checks and still generate broken chat output.

## Likely Divergence Points

### A) Contract defaults in template vs model-specific metadata

- `version/v7/templates/llama.json` currently defaults logits contract to weight tying.
- Nanbeige needs untied-head semantics.
- This should be runtime/config driven, not family-hardcoded.

### B) Weight alias fallback is too permissive

- In lowering aliases, `lm_head` can fall back to `token_emb`.
- For untied models, this fallback should be disallowed.

### C) Codegen path has hardcoded token embedding usage in logits branches

- Some logits codegen branches directly reference `W_TOKEN_EMB`.
- They must use the lowered-call selected weight source.

### D) Rebuild invalidation gap

- `weights_manifest.json` may update after convert, but lowered/codegen artifacts may remain stale if invalidation is incomplete.

### E) Chat/control token behavior is separate from math parity

- Auto chat-template + special-token handling can still loop on control markers even with correct math.

## Compatibility Architecture (low-risk)

## 1) Single model contract object (source of truth)

Persist and consume these fields end-to-end:

1. `tie_word_embeddings` (required bool after resolve),
2. `lm_head_weight_name` (`output.weight` or `token_emb`),
3. `tokenizer_type`, `bos/eos/stop_ids`, template mode,
4. rope/attention flags already used in lowering.

No implicit family defaults after this stage.

## 2) Strict gates before compile/run

### Gate A: Convert contract gate

Fail if:

1. GGUF category coverage missing,
2. untied model has no `output.weight`,
3. manifest contract and GGUF tie setting conflict.

### Gate B: Lowered-call logits source gate

Fail if untied model and logits op weight arg resolves to token embedding.

### Gate C: Codegen audit gate

Fail if generated logits op uses a macro not matching lowered-call contract.

### Gate D: First-token parity gate

Run deterministic first-token logits compare against reference and fail hard on mismatch.

### Gate E: Chat behavior gate (small)

Validate:

1. auto template does not emit control-token loops for simple greeting prompts,
2. `--chat-template none` remains usable fallback.

## 3) Regression matrix (must stay green)

Keep a small fixed matrix:

1. Qwen2
2. Qwen3
3. Gemma
4. Nanbeige (new family stress)

Per model:

1. template-audit pass,
2. lowered-call error count = 0,
3. first-token parity pass,
4. smoke generation non-empty/non-control-loop response.

## Execution Plan (minimal churn)

### Phase 0: guardrails and reporting (now)

1. keep GGUF category coverage check (already added),
2. add dedicated untied-lm-head lowered/codegen audits,
3. add stale-artifact invalidation on contract hash change.

### Phase 1: contract-driven logits source

1. make logits weight source explicit in lowered ops (`lm_head_weight_name`),
2. remove implicit fallback to `token_emb` for untied models.

### Phase 2: codegen cleanup

1. remove hardcoded `W_TOKEN_EMB` logits assumptions in decode/prefill helpers,
2. only emit pointers selected by lowered-call args.

### Phase 3: chat/template hygiene

1. keep chat controls separate from math parity checks,
2. tighten default stop/control-token filtering only after parity is clean.

## Stop Conditions (to avoid scope creep)

For each new family bring-up, stop once:

1. first-token parity passes,
2. logits source contract is correct,
3. simple greeting prompt produces stable non-loop output,
4. existing family matrix remains green.

Anything beyond that (advanced chat quality tuning) is a separate task.

## Notes for future model bring-up

When a model looks like gibberish:

1. do not start with tokenizer speculation,
2. check convert contract + logits source first,
3. then run first-token parity,
4. then inspect chat-template behavior.

This ordering reduces false leads and keeps debugging deterministic.
