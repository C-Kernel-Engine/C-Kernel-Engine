ADR-0005: Kernel Selection Happens in Lowering
==============================================

Status
- Accepted

Context
- Multiple kernel variants exist for each operation:
  - GEMM: serial, parallel, 1x1 (decode)
  - Attention: batch, single-query (decode), GQA vs MHA
  - RMSNorm: serial, parallel
- Runtime kernel selection adds branching overhead.
- Mode and shape determine the optimal kernel.

Decision
- Graph IR uses abstract ops (GEMM, Attention, RMSNorm).
- Lowering resolves to concrete kernel function names based on:
  1. Execution mode (prefill/decode/backward)
  2. Tensor shapes (M large vs M=1)
  3. Architecture features (GQA vs MHA)
- Generated code has no runtime kernel dispatch.

Example:
```
Graph IR:  {"op": "GEMM", "inputs": ["normed", "wq"], "output": "q"}

Prefill Lowered: gemm_blocked_parallel_bf16(...)
Decode Lowered:  gemm_1x1_bf16(...)
```

Kernel Registry:
- Maintain mapping: (op, mode, constraints) → kernel function.
- Validates kernel signatures match src/kernels/ at codegen time.

Consequences
- No runtime branching for kernel selection.
- Mode-specific code paths are explicit and testable.
- Registry must stay in sync with src/kernels/.

Alternatives
- Runtime dispatch (rejected: branching overhead, unclear which kernel used).
- Single kernel per op (rejected: loses optimization opportunities).
