# Qwen3-VL AVX2 Attention Parity

## Scope

Canonical GGUF replay on AVX2 using:

- decoder: Qwen3VL-8B-Instruct-Q4_K_M.gguf
- image: local canonical `ocr/1 81.jpg` converted through the shared PPM path
- visual prefix: 1008 x 16384, grid 36 x 28
- context: 4096
- threads: 20
- prompt: `Extract visible form fields as compact JSON.`
- llama.cpp: repository-pinned commit

Private OCR images and model weights are not committed.

## Root Cause

The strict causal prefill path used FP32 Q/K dot products and an FP32 value reduction. llama.cpp's unfused GGUF graph uses a different numerical contract:

1. Q and K are rounded to FP16 before the score dot product.
2. Scaled softmax is evaluated in FP32.
3. Softmax probabilities are rounded to FP16.
4. V is rounded to FP16 before the value reduction.

At layer 0, head 25, generated step 31:

| Comparison | Max abs | RMSE |
|---|---:|---:|
| FP32 CK-style scores vs llama scores | 0.02710 | 0.00750 |
| FP16 Q/K scores vs llama scores | 0.0000153 | 0.00000273 |
| New strict attention, identical llama inputs | 0.0000000149 | 0.00000000114 |
| New strict attention, CK upstream inputs | 0.00000451 | 0.000000186 |

The Q4_K output projection was ruled out. Feeding llama's `kqv_out` through CK's layer-0 Q4_K x Q8_K projection, bias, and weight view reproduced llama `kqv_wo` within 2.38e-7 max. The projection amplified an upstream attention delta; it did not create it.

## End-to-End Result

| Run | First mismatch | Result |
|---|---:|---|
| Previous strict baseline | 31 | CK `":` vs llama `_name` |
| New FP16-unfused strict attention, 32 tokens | none | 32/32 persistent tokens matched |
| New FP16-unfused strict attention, 64 tokens | 44 | CK `form` vs llama `section` |
| Full replay at step 44 | 44 | Still mismatched; not KV-state-only |

The generated output remained coherent JSON through the shared prefix. This is a material parity improvement, not full 64-token closure.

## Step-44 Attribution

The bounded layer sweep shows accumulation rather than a new circuit discontinuity:

| Boundary | Max abs |
|---|---:|
| layer 0 output | 0.00279 |
| layer 4 output | 0.09347 |
| layer 16 output | 0.22189 |
| layer 28 output | 0.85219 |
| layer 35 output | 9.60400 |

Layer-0 granular replay remains tight through projections, then diverges first at attention:

| Boundary | Max abs |
|---|---:|
| attention norm | 4.5e-8 |
| Q projection | 2.68e-7 |
| K projection | 1.79e-7 |
| V projection | 3.7e-8 |
| Q norm | 3.34e-6 |
| K norm | 4.58e-5 |
| attention output | 1.64e-4 |
| post-attention residual | 4.60e-4 |
| layer output | 2.79e-3 |

## Guards Added

- `unfused_f16_causal(T=17,H=4,D=32)` in `unittest/test_attention_f16_split_kv.py`
- absolute source-token provenance after mixed-prefix rebasing
- canonical llama `kqv_wo` to CK output-projection alignment
- `make test-v8-dump-alignment`
- `make qwen3vl-parity-guards`
- nightly/report row for stitched dump alignment

The aggregate guard includes the existing BF16 nightly lane. The GGUF FP16 cache contract is not applied to BF16 arithmetic.

## Next Target

At step 44, determine the exact llama AUTO flash-attention reduction contract. The new implementation is exact against llama's unfused graph, but the canonical AUTO model path still differs at `kqv_out`. Keep performance work disabled until the 64-token Q4_K_M replay is parity-clean.
