# Kernel Maps

This directory describes the decoder execution plan and buffer layout used by
`scripts/gen_kernel_specs.py` to generate `ckernel_kernel_specs.*`.

## Files

- `global_buffers.json`: Global buffers (weights, activations, gradients).
- `decoder_layer_plan.json`: Forward plan (per layer).
- `decoder_layer_plan_backward.json`: Backward plan (per layer, reverse order).
- `kernels/*.json`: Kernel definitions with forward/backward bindings.

## Kernel Schema (kernels/*.json)

Example:

```json
{
  "name": "rmsnorm",
  "impl": {
    "forward": "rmsnorm_forward",
    "backward": "rmsnorm_backward",
    "sources": ["src/kernels/rmsnorm_kernels.c"]
  },
  "buffers": [
    {"id": "input", "role": "input", "shape": [{"dim": "tokens"}, {"dim": "aligned_embed"}]},
    {"id": "gamma", "role": "weight", "shape": [{"dim": "aligned_embed"}]},
    {"id": "out", "role": "output", "shape": [{"dim": "tokens"}, {"dim": "aligned_embed"}]},
    {"id": "rstd", "role": "activation", "shape": [{"dim": "tokens"}]}
  ],
  "buffers_backward": [
    {"id": "d_out", "role": "grad", "shape": [{"dim": "tokens"}, {"dim": "aligned_embed"}]},
    {"id": "input", "role": "input", "shape": [{"dim": "tokens"}, {"dim": "aligned_embed"}]},
    {"id": "gamma", "role": "weight", "shape": [{"dim": "aligned_embed"}]},
    {"id": "rstd", "role": "activation", "shape": [{"dim": "tokens"}]},
    {"id": "d_input", "role": "grad", "shape": [{"dim": "tokens"}, {"dim": "aligned_embed"}]},
    {"id": "d_gamma", "role": "grad", "shape": [{"dim": "aligned_embed"}]}
  ]
}
```

Notes:
- `buffers` are used by the forward plan; `buffers_backward` by the backward plan.
- If `buffers_backward` is omitted, backward steps reuse `buffers`.
- `role` controls layout behavior (weights vs activations vs grads).
- `sources` is the list of `.c` files needed to compile the kernel.

## Plan Schema

Each plan step binds kernel buffer IDs to named layout buffers:

```json
{
  "kernel": "rmsnorm",
  "bind": {
    "input": "input",
    "gamma": "ln1_gamma",
    "out": "ln1_out",
    "rstd": "ln1_rstd"
  }
}
```

Backward steps should bind the `buffers_backward` IDs.
