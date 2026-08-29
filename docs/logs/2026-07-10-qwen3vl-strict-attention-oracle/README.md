# Qwen3-VL strict attention oracle guard

## Why this change exists

The v8 stitched Qwen3-VL harness enabled `--strict-parity`, but the bridge and
numeric-parity subprocesses rebuilt the ordinary CK engine. The llama/GGML
attention oracle was therefore absent and strict attention silently fell back
to the CK implementation. That made the first reported `attn_output` and MLP-up
divergences unsuitable for locating the remaining model stitching bug.

This change makes the llama-backed engine an explicit, opt-in diagnostic build
and propagates that build mode through the complete stitched parity subprocess
tree. Normal production builds remain independent of a llama.cpp checkout.

## Changes

- `CK_LLAMA_PARITY_ENGINE=1` adds the llama/GGML attention oracle to
  `build/libckernel_engine.so`.
- Strict non-causal attention uses the multi-head GGML graph oracle when that
  diagnostic engine is loaded.
- The oracle executes through its owned GGML context, avoiding the backend
  scheduler assertion caused by tensors without backend buffers.
- `make test-qwen3vl-strict-attn-oracle-build` verifies that the diagnostic
  library exports the multi-head oracle symbol.
- `make llamacpp-parity-stitched` propagates the diagnostic build mode to every
  child process and internal `make` invocation.

## Validation

```sh
make test-qwen3vl-strict-attn-oracle-build
git diff --check
```

Canonical Qwen3-VL OCR diagnostic configuration:

- image: `ocr/1 81.jpg`, canonical PPM equivalent
- image token cap: 1024
- visual output shape: `1008 x 16384`
- prompt: `Extract visible form fields as compact JSON.`
- context: 4096
- threads: 20

With the diagnostic oracle enabled, comparable tensors at encoder layers 0, 1,
5, and 26 passed with maximum absolute difference 0.0. The final prefix is
improved but not parity-clean:

| Metric | Value |
|---|---:|
| cosine | 0.9999405675 |
| RMSE | 0.0017708609 |
| mean absolute difference | 0.0012538217 |
| maximum absolute difference | 0.0231761634 |

## Remaining divergence

The next attribution target is after the final encoder block: final post-LN,
spatial merge, DeepStack branches, projector FC1/GELU/FC2, and branch/prefix
concatenation. The attention oracle is diagnostic infrastructure, not a claim
that production AVX2 attention or the full Qwen3-VL prefix is parity-complete.
