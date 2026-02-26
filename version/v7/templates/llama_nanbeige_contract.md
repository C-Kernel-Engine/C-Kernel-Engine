# Llama/Nanbeige Contract Notes (Bring-up TODO)

Status: planning guide for strict contract bring-up and parity gating.

## Goal
Make new Llama-family bring-up deterministic: contract first, then kernels/stitching, then chat quality.

## Nanbeige 4.1-3B tokenizer/chat contract

Source signals observed during bring-up:
- tokenizer class: `LlamaTokenizer` (SentencePiece)
- add_bos_token: `true`
- bos_token / bos_token_id: `<|im_start|>` / `166100`
- eos_token / eos_token_id: `<|im_end|>` / `166101`
- known special ids include:
  - `166100` `<|im_start|>`
  - `166101` `<|im_end|>`
  - `166102` `<|endoftext|>`
  - `166103` `<think>`
  - `166104` `</think>`
  - `166105` `<tool_call>`
  - `166106` `</tool_call>`

Contract expectations:
- tokenization mode: sentencepiece
- generation stop ids: include at least `166101` (`<|im_end|>`) and model EOS ids
- chat wrapper semantics must be explicit in contract (no implicit template guessing)
- if no chat template is supplied, runtime must use a declared fallback policy

## Llama-family semantic contract checklist

### tokenizer_contract
- type: sentencepiece | bpe
- add_bos_token / add_eos_token
- bos/eos/pad/unk ids and strings
- additional special ids
- stop id list used by generation loop
- chat wrapper mode (`auto`, explicit template id, `none`)

### attention_contract
- rope variant/type
- rope theta
- rope scaling type/factor
- qk norm enabled/disabled
- kv head layout policy and cache indexing policy

### block_contract
- norm kind: rmsnorm vs layernorm
- residual order
- mlp formula (`gate*up -> down` vs other)
- activation (`silu`, `swiglu`, etc.)
- bias policy (qkv/o/mlp biases on/off)

### logits_contract
- final norm semantics
- lm_head application semantics
- logits scaling/clamping policy

### quant_contract
- per-tensor quant classes
- required kernel families by op

### runtime_invariants
- required call args (e.g. `_kv_copy_bytes`) present and positive
- no lowered-call op may proceed with unresolved args/errors

## Bring-up gate order (must pass before chat)
1. template-audit (weights/template/IR stitch/codegen preflight)
2. lowered-call invariants (strict arg validation)
3. qkv contract check
4. post-attention chain check
5. mlp contract check
6. first-token logits parity (CK vs reference)

Only after all pass: run interactive generation quality checks.

## Notes on naming
Current op id `kv_cache_batch_copy` copies a token block in prefill; not multi-request batching.
Preferred display alias: `kv_cache_token_block_copy`.
