# Qwen3-VL FP16 Split-KV Decode Contract

## Root Cause

The focused Qwen3-VL decode attention comparison ruled out mask, M-RoPE,
shape, output projection, and generic softmax wiring at that boundary.
llama.cpp changes its CPU decode reduction contract when the KV sequence
reaches 512 tokens:

- Q and K are rounded through FP16.
- KV work is split across worker chunks.
- Each chunk produces an FP16 online-softmax/value partial.
- Chunk partials are reduced in FP32, in chunk order.

CK previously used a single FP32-style reduction. Both implementations are
reasonable attention algorithms, but they are not numerically equivalent.
The difference can change top-1 selection after a long mixed visual/text
prefix.

## Oracle Kernel

`attention_forward_decode_head_major_gqa_flash_f16cache_split()` entry point
takes an explicit chunk count so parity tests do not depend on host core count.
It is an oracle/diagnostic entry point, not the production default. Production
routing remains unchanged until stitched model parity accepts the change.

## Regression Matrix

Run:

```bash
make test-attention-f16-split-kv
```

| Case | Purpose |
|---|---|
| KV 511, one chunk | Guard the pre-threshold contract |
| KV 512, four chunks | Guard the exact split threshold |
| KV 1058, H=32, Hkv=8, D=128, 20 chunks | Match the diagnosed Qwen3-VL decode shape |
| KV 513, padded D=80/A=128, GQA | Guard layout, padding, and head mapping |
| llama.cpp at KV 511, 512, and 1058 | Compare directly with `ggml_flash_attn_ext` using F16 K/V |
| FP32 rejection oracle | Prove the test fails if split FP16 is replaced by single FP32 |

The independent Python oracle currently agrees with CK within `1.61e-5`; the
enforced tolerance is `2e-5`. Direct llama.cpp agreement is within `8.32e-6`
under the same `2e-5` limit.
The adversarial FP32 rejection case separates the two contracts by `8.81e-4`.

## Rejected Production Routing

Routing every generated decode layer through the split-KV oracle did not move
the canonical OCR divergence beyond step 31 and worsened its logit metrics:

| Candidate | First mismatch | Cosine | RMSE | Top-k overlap |
|---|---:|---:|---:|---:|
| scalar-dot split | 31 | 0.999045 | 0.258115 | 16/16 |
| llama AVX2 reduction tree | 31 | 0.998776 | 0.282321 | 15/16 |

The split kernel is numerically correct in isolation. The rejected routing
shows that the first unresolved model state is earlier in the mixed-prefill
path; changing persistent decode at the same time only adds another delta.

## Test Integration

- `make llamacpp-parity-full` runs this contract.
- `scripts/nightly_runner.py` exposes it as the named kernel lane
  `FP16 Split-KV Decode Contract`.
- `docs/site/test-report.html` documents the lane so nightly JSON/report output
  can be monitored as individual expandable subtests.

## Remaining Validation

This contract test prevents the reduction semantics from regressing. The full
Qwen3-VL 64-token stitched parity run remains the authority for proving that
the production model path reaches the expected llama.cpp tokens end to end.
