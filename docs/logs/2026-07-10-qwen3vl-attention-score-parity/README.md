# Qwen3-VL AVX2 attention score parity

## Scope

Canonical Qwen3-VL OCR replay at generated step 31:

- image: `ocr/1 81.jpg`
- image SHA256: `eb3becd507063f184d417a6baccfdb22d86277308f21a679aac4e029ed8f25ff`
- visual prefix: `1008 x 16384`
- grid: `36 x 28`
- context: `4096`
- threads: `20`
- prompt: `Extract visible form fields as compact JSON.`

CK persistent decode and CK full replay both choose token `788` (`":`), while
llama.cpp chooses token `1269` (`_name`). The full-replay failure rules out a
KV-cache-only bug.

## Boundary attribution

The post-RoPE comparator previously mixed two llama callback occurrences and
treated CK's token-major RoPE dump as raw head-major scratch storage. After
normalizing both sides:

| Boundary | Max abs | RMSE | Cosine |
| --- | ---: | ---: | ---: |
| post-RoPE Q | `3.8147e-6` | `3.8907e-7` | effectively `1.0` |
| post-RoPE K, final row | `4.5776e-5` | `2.3371e-6` | effectively `1.0` |

This rules out a material M-RoPE circuit/layout error at the failing row.

Focused layer-0, query-1057 captures then compared the active 1058-token
attention window. llama.cpp's unfused graph pads the score dimension to 1280;
only its first 1058 entries are active.

| Boundary | Worst max abs | Worst RMSE | Interpretation |
| --- | ---: | ---: | --- |
| raw Q dot K score | `4.5780e-2` | `1.3684e-2` per head | first measurable semantic delta |
| softmax probability | `3.8594e-4` | `1.5196e-5` per head | mask and normalization remain tight |
| value-weighted context | `6.4460e-5` | `1.2867e-5` per head | consistent with the model-level attention drift |

Score recomputation identifies the contract difference:

- llama score vs FP32 Q/K recomputation: RMSE `8.6550e-3`
- llama score vs FP16-rounded Q/K recomputation: RMSE `3.5507e-6`
- CK score vs CK FP32 Q/K recomputation: RMSE `3.7754e-6`

llama's score path therefore follows an FP16-rounded Q/K contract, while CK's
strict mixed-prefill path currently computes the score from FP32 Q/K. This is
not a causal-mask, softmax-shape, output-projection, or M-RoPE wiring failure.

## Rejected change

The existing `attention_flash_query_causal_exact_f16kv` implementation is not
a valid replacement for this path. Its online FP16 output accumulation made
the selected context comparison worse. A production routing change must match
llama's complete score, softmax, and value-reduction contract, not merely round
inputs to FP16.

## Tooling changes

- Label the second llama `Qcur`/`Kcur` callback occurrence as post-RoPE.
- Extract CK post-RoPE rows from their actual `[token, head, dim]` dump layout.
- Disable llama flash attention only when `kq`, `kq_soft_max`, or `kqv`
  diagnostics are explicitly requested.
- Emit bounded CK score, probability, and context vectors for a selected
  layer/query/head without changing the computed output.

## Next step

llama.cpp's CPU flash path adds one more required semantic for this case. For a
single query with at least 512 KV rows, it partitions KV into one chunk per
worker. Each worker computes an FP16 Q/K/V online-softmax partial, then the
partials are combined in chunk order with an FP32 accumulator. CK currently
uses a single FP32-style reduction for this mixed-prefix replay.

Implement and benchmark a bounded split-KV reference with that exact contract.
Then vectorize the proven reference and share the split-KV dispatcher with
persistent decode. Accept it only if it reduces model-level `kqv_out` error,
keeps one-token parity, and does not move the 64-token divergence earlier.
