# Qwen3-VL AVX2 encoder circuit attribution

## Scope

Canonical GGUF parity run using:

- image: `ocr/1 81.jpg`, converted to the canonical PPM
- source JPEG SHA-256: `eb3becd507063f184d417a6baccfdb22d86277308f21a679aac4e029ed8f25ff`
- image token cap: 1024
- visual grid: `36 x 28`
- visual prefix: `1008 x 16384`
- prompt: `Extract visible form fields as compact JSON.`
- context: 4096
- threads: 20

## Fixed circuit defects

### Vision M-RoPE width

The production vision kernel treated `n_dims=36` as the total rotary width. In
GGML's Qwen3-VL vision contract it is the half-width offset: a 72-wide head is
rotated as `(i, i + 36)`, with 18 height pairs and 18 width pairs. The kernel
also used the wrong frequency denominator as a consequence.

The corrected production kernel now matches the Qwen3-VL vision reference for
the complete 72-wide head within one FP32 ULP:

| Tensor | Maximum absolute difference |
|---|---:|
| layer 0 Q after M-RoPE | `4.768e-7` |
| layer 0 K after M-RoPE | `4.768e-7` |

The focused vision unit test uses the real Qwen3-VL dimensions so this semantic
error cannot regress behind a smaller synthetic shape.

### Q8 projection routing

Qwen3-VL Q8 projector and DeepStack projections now use the existing Q8
activation-contract adapter through architecture metadata. The selection is
data-driven (`q8_0_contract_ops`) rather than embedded in generated C.

Strict attention diagnostics can cache the GGML attention result for the next
Q8 projection. A token-count/product collision made that cache unsafe when it
was consumed solely by element count, so cached substitution now requires the
explicit diagnostic variable `CK_STRICT_GEMM_USE_CACHED_A=1`. Production does
not use this substitution.

### Diagnostic build selection

The Make build-flags stamp existed but was not reevaluated after its first
creation. Changing `CK_LLAMA_PARITY_ENGINE`, OpenMP, or ISA flags could
therefore leave an incompatible `libckernel_engine.so` marked up to date. The
stamp is now checked on every Make invocation and only changes timestamp when
the effective compiler/linker flags differ. The strict-oracle symbol gate now
tests the library variant it requested rather than whichever variant happened
to be built last.

## Fresh circuit attribution

After the build-stamp correction, a fresh strict-oracle run passes every
comparable layer-0 boundary exactly:

| Boundary | Maximum absolute difference |
|---|---:|
| Q after M-RoPE | `0` |
| K after M-RoPE | `0` |
| raw attention context (`kqv_out`) | `0` |
| Q8 attention output projection | `0` |
| residual after attention | `0` |
| MLP output | `0` |
| layer output | `0` |

The full 27-layer strict encoder also matches llama.cpp exactly:

| Metric | Value |
|---|---:|
| compared FP32 values | `16,515,072` |
| cosine | `1.0` |
| RMSE | `0` |
| mean absolute difference | `0` |
| maximum absolute difference | `0` |

This proves the generated circuit, weight views, tensor order, DeepStack
branches, projector, and final `1008 x 16384` bridge layout. Production AVX2
still accumulates numerical drift: the current full prefix is approximately
cosine `0.99950`, RMSE `0.00514`, and maximum absolute difference `0.467`.
That is an optimized-kernel parity problem, not a remaining stitching problem.

## End-to-end result

The production encoder and persistent decoder were tested against llama.cpp
for 64 greedy tokens, stopping at the first top-1 mismatch.

| Prefix source | First mismatch | CK | llama.cpp | Logit cosine | Top-16 overlap |
|---|---:|---|---|---:|---:|
| CK production encoder | 20 | `5` | `submit` | `0.999366` | `16/16` |
| exact llama.cpp encoder prefix | 31 | `":` | `_name` | `0.999365` | `15/16` |

Before the M-RoPE correction, this AVX2 lane failed at step 4. The corrected
encoder now reproduces the Xeon first-mismatch signature at step 20.

Full replay at step 20 still chooses CK token `5`, so this is not an
incremental KV-cache-only failure. Persistent-vs-full-replay layer-0 tensors
remain close, while full CK replay still differs from llama.cpp logits.

## Remaining work

This change does not claim 64-token exact parity. The next targets are:

1. Reduce production AVX2 attention/Q8 projection accumulation drift while
   preserving the now-proven circuit contract.
2. Attribute the later decoder-only mismatch exposed at step 31 with an exact
   llama.cpp visual prefix.
3. Run the OCR image sweep only after the canonical image passes the required
   long-token criterion.

No tolerance was relaxed.
