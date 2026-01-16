ADR-0001: IR v4 Pipeline Split
==============================

Status
- Accepted

Context
- IR v2 proved useful for lowering and memory planning, but v3 is a deterministic
  layout generator that bypasses IR and hard-codes the graph.
- We need a pipeline that preserves semantic intent while producing deterministic
  layout and codegen outputs.

Decision
- Adopt a split pipeline for IR v4:
  1) Graph IR (macro): ops, buffers, section structure (header/body/footer).
  2) Lowered IR: per-mode (prefill/decode/backward), kernel variants, resolved dims.
  3) Layout IR: offsets, sizes, alignment, canaries.
  4) Codegen: deterministic C from lowered IR + layout.

Consequences
- Graph IR remains human-debuggable and close to architecture intent.
- Lowering owns kernel selection and mode-specific buffer materialization.
- Layout is deterministic and fully resolved for runtime simplicity.
- Requires a graph-to-lowered and lowered-to-layout step.

Alternatives
- Single IR that mixes graph + offsets (rejected: loses semantic intent).
- Continue v3-only layout generation (rejected: no generalization).

