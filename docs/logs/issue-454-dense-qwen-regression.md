# Dense Qwen Regression: Issue 454

## Reproduction

Current artifact: `ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf`.
HF revision: `0669b98607d47046c7c2b3f801011d54a08cfccf`.
File size: 18,973,870,432 bytes.
LFS SHA-256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

Fresh conversion and compilation at 1024 capacity reproduced punctuation
gibberish for `Give me a recipe of clafoutis` on Ryzen Zen 5. The parent of
Flash-Next PR #456 (`e87b90d0f`) also reproduced it with this artifact. Reverting
#456 alone therefore does not resolve the failure. The exact introducing commit
has not been established.

## Diagnosis and Fix

1. The current file has `general.name=Qwen3.8-27B` but no `general.basename`.
   Add a non-overlapping artifact identity rule to select the dense circuit and
   its bounded prefill defaults. Preserve its declared ChatML variant contract.
2. Tokenizer-free comparison found zero CKE logits. Layer-0 X-Ray matched
   normalization, QKV, gate projection and convolution exactly, then found NaNs
   at alpha/beta projections. The FP32 numerical contract was selected from
   shape/QKV facts even though the actual alpha/beta weights were Q8_0.
3. Derive uniform layer-weight storage facts from manifest entries. Require
   FP32 alpha and beta storage in the scalar FP32 contract selectors. Otherwise
   use normal dtype-aware provider selection. No new numerical kernel is needed.
4. Extend the existing lowering guard to reject packed quantized weights passed
   to floating-point providers, before emitting C.
5. Audio's separate nightly failure involved projection port canonicalization.
   Lowering now follows the same canonicalization as graph construction while
   retaining explicitly validated provider interfaces.

## Evidence Scope

The corrected 1024-capacity run generates a coherent clafoutis recipe (64-token
output limit). Tokenizer-free replay of 128 output positions from token ID 1 is
full-vocabulary bit-exact against llama.cpp on Ryzen. This is not 128K numerical
certification, nor certification of every file named Q4_K_M.

The dense-Qwen nightly row covers metadata, chat policy, scalar dtype selection
and rejection of packed-to-float bindings. It is deliberately separate from
Flash-Next coverage. The 131072-capacity plan uses 4096-token prefill chunks and
last-only logits: prefill arena 30,656,220,224 bytes (28.6 GiB), decode arena
27,805,840,384 bytes (25.9 GiB). Prepared weights and process overhead are extra;
this is not a 32 GB host fit guarantee or a full-context execution result.

Real 27B generation and numerical parity require a local
high-memory runner; a lightweight contract pass is not an E2E model pass.

## Contributor Retest

After checking out this fix, use a fresh run directory:

```bash
version/v8/scripts/cks-v8-run run \
  hf://ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --run "$HOME/.cache/cke-issue454-retest" \
  --context-len 16384 \
  --force-convert --force-compile \
  --prompt 'Give me a recipe of clafoutis' \
  --max-tokens 256 --temperature 0
```

This short prompt tests generation with 16K allocated capacity, not 16K consumed
input. Do not treat it as proof that 128K fits a 32 GB machine. Check the memory
plan before allocating a large context. UD artifacts containing unsupported
quantizations remain a separate limitation.

Local component gate:

```bash
make test-v8-qwen38-dense-contracts
```

For real-model numerical acceptance use `compare_multitoken_logits_v8.py` with
the exact GGUF, a freshly generated runtime, a pinned `CK_LLAMA_CPP_ROOT`, and
`--require-bit-exact`. Record prompt token IDs, capacity, actual compared steps,
threads and artifact provenance with the JSON report.
