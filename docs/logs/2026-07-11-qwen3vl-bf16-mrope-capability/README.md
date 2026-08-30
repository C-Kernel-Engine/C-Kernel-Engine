# Qwen3-VL BF16 Vision M-RoPE Capability

## Why

The vision M-RoPE kernel now interprets `n_dims` as the full rotary width, but
the BF16 safetensors conversion path still emitted half width and the circuit
did not request an output-storage contract. That allowed the corrected leaf
kernel and the generated model route to disagree.

## Resolution

- Added explicit FP32, BF16-output, and FP16-output vision M-RoPE capabilities.
- Bound Qwen3-VL BF16 vision layers to
  `mrope_qk_vision_bf16_storage` through the numerical contract resolver.
- Emit full rotary width (`72`) and spatial sections `[18,18,0,0]` for the
  production geometry.
- Propagate the exact kernel and contract through GraphIR, LoweredIR, call IR,
  and generated C.
- Require resolved `operator_family=vision_mrope` and
  `position_transform.pairing=multi_section` metadata. Lowering and diagnostic
  routing consume these semantics rather than inferring behavior from a kernel
  or function-name prefix.
- Added exact ABI bindings for the two storage wrappers. No prefix fallback or
  model-name dispatch was introduced.

## Validation

- Numerical execution resolver: 15/15 pass.
- X-ray architecture: 8/8 pass.
- Qwen3-VL circuit/codegen: 21/21 pass.
- Renamed-kernel semantic routing guard: pass; exact kernel identity remains a
  resolver output asserted independently in call IR.
- Template/dataflow audit: 10/10 pass.
- Vision kernel suite: pass.
- Storage matrix at head dimensions 8 and 72:
  - FP32 max error: `2.38e-7`.
  - BF16-rounded max error: `0`.
  - FP16-rounded max error: `0`.
- Safetensors conversion, call-IR binding, and generated-C smoke: pass.
- FP16 split-KV llama.cpp oracle: 14/14 pass.
- v8 regression-fast: pass for Gemma3, Qwen2, Qwen3, Qwen3.5, and Nanbeige.

The full safetensors test file has two unrelated failures that reproduce on the
pre-patch worktree: Nemotron Mamba input-slot expectations and GLM4 source-map
normalization. They are not changed or represented as passing here.

## Remaining Native Gate

This AVX2 node proves resolver, circuit, conversion, generated-code, and scalar
storage semantics. It cannot prove native AVX-512 BF16 end-to-end behavior.
Rerun `make xray-vision-parity` on Xeon with the real checkpoint/runtime/call
IR. Acceptance requires the first divergent checkpoint to move beyond vision
M-RoPE before mixed-prefill, teacher-forced decode, or OCR promotion.
