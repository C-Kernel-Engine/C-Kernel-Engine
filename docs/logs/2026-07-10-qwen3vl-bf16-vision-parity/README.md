# Qwen3-VL BF16 Vision Parity

## Scope

This change hardens the Qwen3-VL safetensors/BF16 vision path around two semantic contracts:

1. Learned 2D position embeddings use PyTorch bilinear `align_corners=True` coordinates.
2. Vision M-RoPE uses frequency-pair width `head_dim / 2` with valid repeated axis sections.

The real model measurements below were produced on the Xeon BF16 system using the canonical private OCR image. The
image and model weights are not committed.

## Xeon Result

| Boundary | Before | After |
|---|---:|---:|
| Position RMSE | 0.00794 | 0.00116 |
| Post-RoPE Q cosine | 0.6167 | 0.999970 |
| Layer-0 output projection cosine | - | 0.999985 |
| Final prefix cosine | 0.308 | 0.996712 |
| Deepstack slice cosine | - | 0.9991-0.9999 |

The remaining real-model drift is concentrated in the main final projector path. Full BF16 end-to-end parity is not
claimed.

## Regression Guards

- numerical C-vs-PyTorch align-corners interpolation, maximum observed error `8.34e-7`
- synthetic safetensors conversion and all-28-weight consumption
- interpolation-policy preservation through IR and kernel selection
- vision M-RoPE width and section arguments preserved through call lowering
- invalid M-RoPE sections fail lowering instead of silently disabling rotation
- bounded semantic exports for Q/K/V, RoPE, attention, normalization, and projection boundaries
- named nightly row for the 16-test Qwen3-VL template/codegen contract suite
- self-building `make nightly-bf16`, including the vision library required by the M-RoPE guard

## Validation

- `make test-v8-vision-kernels`
- `make test-v8-qwen3vl` (`16/16`)
- `make nightly-bf16` (`12/12`)
- Python syntax checks for changed scripts
- `git diff --check`

## Next Xeon Target

Run the canonical BF16 image through bounded captures at `post_ln`, `projector_fc1`, projector GELU, `projector_fc2`,
each deepstack slice, and final concatenation. Compare CK against PyTorch at each boundary and report cosine, RMSE,
maximum absolute error, and worst row. This will distinguish accumulated BF16 rounding from projector layout, bias,
activation, or concatenation semantics.
