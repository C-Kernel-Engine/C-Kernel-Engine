# Muse-Glimmer text bring-up (2026-09-09)

This note records the first CKE text bring-up for
`meta-models/Muse-Glimmer-30B`. Conversion, declarative lowering, generated-C
compilation, and one-token BF16 parity are complete. Multi-token attention is
still a numerical mismatch, so this is not yet a full text-support claim.

## Frozen reference contract

- Model configuration and weights: Hugging Face branch `main-30B2`
- Reference implementation: Transformers commit
  `4177486a9f199bd7be520eff14431071d5d41ec5`
- Reference class: `MuseGlimmerForConditionalGeneration`
- Reference attention backend: `eager`
- Storage and activation dtype: BF16
- CPU reference setting: `OMP_NUM_THREADS=16`
- `config.json` SHA-256:
  `5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14`
- `model.safetensors.index.json` SHA-256:
  `7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba`
- `chat_template.jinja` SHA-256:
  `cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678`

The Ryzen validation host used Transformers 5.15.1, PyTorch 2.13.0+cpu,
oneDNN 3.12, and the SLEEF symbols supplied by that PyTorch build.

## Implemented contract

- A `muse_glimmer_text` circuit with independent hidden, query, KV, gate,
  and MLP dimensions.
- Three sliding-attention layers followed by one full-attention layer, with
  RoPE enabled only for sliding layers.
- Centered weighted RMSNorm, weighted post-branch RMSNorm, unweighted Q/K
  normalization, query scaling, attention gating, four normalization sites,
  output scaling, and final tanh softcapping.
- Explicit BF16 rounding boundaries for normalization, Q/K scaling, RoPE,
  projections, SwiGLU, residuals, and logits.
- Separate embedding and output-head tensors.
- Strict tensor-family checks and explicit deferral of every vision tensor.
- Generic concat-shape propagation used to pack MLP gate and up projections
  into the layout expected by the existing SwiGLU provider.

The published checkpoint contains 1,436 tensors. Conversion accounts for all
of them: 627 are consumed by the text circuit and 809 vision tensors are
explicitly marked as deferred. There are no unexplained leftovers.

## Ryzen evidence

The complete 59.55 GB checkpoint converted to a 55.5 GiB CKE weight artifact.
Both 52-layer prefill and decode graphs lower, generate C, compile, and load
with context length 128 and prefill chunk length 8.

For input token `[1]`, a real-weight X-ray found exact decoder outputs through
the transformer stack and an exact 6,656-element final normalized hidden
vector. After selecting the oneDNN 3.12 BF16 projection contract for the
output head, all 202,048 logits are bit exact with the eager PyTorch reference.

For input tokens `[1, 2]`, CKE and the reference still select token 24, but the
logit arrays expose the unresolved attention contract:

| Path | Exact logits | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: | ---: |
| Decode | 1,475 / 202,048 | 3.0703125 | 0.6042349 |
| Prefill | 12,089 / 202,048 | 0.34375 | 0.0628345 |

Component oracles are bit exact for all three RMSNorm variants, Q/K
normalization plus query scaling, direct split-half RoPE at nonzero positions,
and the ordered logit scale/softcap chain. Synthetic multi-token X-rays first
diverge when attention combines token values. The selected generic attention
providers accumulate through FP32 paths, while the pinned eager reference uses
BF16 score and value matmuls with a float32 softmax followed by BF16 probability
storage. Decode additionally lacks a Muse-specific BF16 sliding-window cache
provider. Those are the next numerical provider contracts to implement.

## Certification state

| Gate | State |
| --- | --- |
| Configuration and tensor inventory | Pass |
| Synthetic conversion/lowering/generated C | Pass |
| Component numerical contracts | Pass |
| Official one-token text parity | Pass |
| Multi-token prefill | Numerical mismatch at attention |
| Cached decode and window rollover | Numerical mismatch / not certified |
| 2,047 / 2,048 / 2,049-token boundaries | Not tested |
| 4K through 128K context | Not tested |
| Quantization | Not tested |
| Vision and image preprocessing | Deferred |

The next acceptance gate is a causal grouped-query provider that reproduces
the eager BF16 arithmetic for both full and sliding attention, including a
BF16 sliding cache. After that passes at short lengths, cache rollover and the
2,047 / 2,048 / 2,049 boundary sequence can begin.
