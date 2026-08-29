fix(v8/vision): align qwen3-vl mrope circuit

Why: Qwen3-VL AVX2 diverged from llama.cpp early because the production vision
kernel interpreted GGML's M-RoPE n_dims as total width instead of the half-width
pair offset. Strict Q8 diagnostics could also consume a stale attention cache
when unrelated tensor shapes had the same element count.

What: implement the 72-wide Qwen3-VL vision rotary contract, guard strict cached
Q8 input substitution behind an explicit diagnostic switch, route Q8 projector
and DeepStack operations through architecture metadata, and bound stitched CK
dumps with the same op filter used for llama.cpp. Revalidate the Make build-flags
stamp on every invocation so strict-oracle, OpenMP, and ISA variants cannot reuse
an incompatible engine library.

Validation: vision kernels and Qwen3-VL template tests pass. The strict 27-layer
encoder matches all 16,515,072 final-prefix floats exactly against llama.cpp.
Layer-0 post-RoPE Q and K are within 4.768e-7 in production attribution.
Canonical OCR persistent parity moves from step 4 to step 20 and now matches the
Xeon signature. An exact llama visual prefix moves the remaining decoder mismatch
to step 31. Full replay at step 20 confirms the residual mismatch is not
KV-cache-only. No tolerance was relaxed.
