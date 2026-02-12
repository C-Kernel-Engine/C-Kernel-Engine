# v7 Runtime Contract (Training)

## Training Step Semantics

- Microbatch forward/backward may run multiple times per optimizer step.
- Loss is scaled by `1 / grad_accum_steps` before backward.
- Weight update runs exactly once per optimizer step.

## Memory Ownership

- Weights and optimizer state are persistent.
- Activations and gradients are step-scoped.
- Scratch buffers are per-op and reusable when lifetime does not overlap.

## Reproducibility

- Deterministic mode must support fixed-seed reproducibility checks.
- Runtime reports include mode, dtype policy, and thread config.

## v7.0 Policy

- fp32-only correctness path.
- single-token parity (`T=1`) gate for core backward kernels.
- threaded and bf16 paths are not part of v7.0 pass criteria.
