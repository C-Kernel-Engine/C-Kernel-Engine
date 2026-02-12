# v7 Kernel Contract (Training)

## Required v7.0 Backward-Capable Families

- `rmsnorm_forward` + `rmsnorm_backward`
- `swiglu_forward` + `swiglu_backward`
- `softmax_cross_entropy_loss` (loss + dlogits)
- `attention_forward_causal_head_major_gqa_exact` + `attention_backward_causal_head_major_gqa`

## ABI Expectations

- C symbols remain stable and discoverable from shared libs.
- Input/output pointers and dimensions must match declared signatures.
- fp32 path is canonical for correctness in v7.0.

## Numerical Expectations

- Max absolute diff thresholds are enforced by parity harness.
- Backward kernels must be tested against PyTorch reference.
- No NaN/Inf tolerated in nominal test ranges.

## ISA Modes

- Deterministic mode: predictable reduction order, stable reproducibility.
- Fast mode: allows optimized reductions and threading once validated.

v7.0 gate focuses on deterministic fp32 behavior only.
