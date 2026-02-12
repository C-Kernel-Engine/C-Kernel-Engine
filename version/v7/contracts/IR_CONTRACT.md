# v7 IR Contract (Training)

## Goals

- Keep codegen simple by making IR call-ready.
- Make forward/backward memory ownership explicit.
- Support deterministic and fast reduction modes without changing graph semantics.

## Required Fields Per Op

- `op`: semantic operation id (`rmsnorm`, `attention`, `swiglu`, etc.)
- `function`: concrete kernel symbol selected from registry
- `inputs`: tensor references with dtype/layout metadata
- `outputs`: tensor references with dtype/layout metadata
- `scratch`: optional temp buffers with explicit lifetime class
- `attrs`: op parameters (eps, head_dim, num_heads, etc.)
- `grad_rule`: backward wiring id (for v7 lowering)

## Producer/Consumer Rules

- Every output tensor has exactly one producer.
- Every input tensor must map to an existing producer or an input binding.
- Gradient accumulation edges must be explicit in IR, not inferred in codegen.

## Training Lowering Preconditions

- IR2 backward synthesis must fail fast if required training kernels are missing from kernel maps.
- A required training kernel is valid only when both are present: `KERNEL_REGISTRY.json` contains the
  selected kernel function/id, and `kernel_bindings.json` contains argument bindings for that kernel id.
- No implicit fallback to hardcoded function names is allowed in codegen.

## Determinism Rules

- Reduction mode must be present as an explicit runtime setting.
- IR topology must be invariant across deterministic and fast modes.
- Only kernel implementation path may differ by mode.
