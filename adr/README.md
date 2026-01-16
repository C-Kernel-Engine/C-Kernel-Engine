Architecture Decision Records (ADRs)
====================================

Purpose
- Capture key decisions for IR, layout, and codegen so we can revisit tradeoffs.

Format
- Use "000X-short-title.md" numbering.
- Keep sections: Status, Context, Decision, Consequences, Alternatives.

ADR Index
---------

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-ir-v4-pipeline.md) | IR v4 Pipeline Split | Accepted |
| [0002](0002-templates-as-canonical.md) | Templates as Canonical Architecture Source | Accepted |
| [0003](0003-mode-specific-lowering.md) | Mode-Specific Lowering (Prefill/Decode/Backward) | Accepted |
| [0004](0004-deterministic-layout-canaries.md) | Deterministic Layout + Canaries for Debug | Accepted |
| [0005](0005-kernel-selection-in-lowering.md) | Kernel Selection Happens in Lowering | Accepted |
| [0006](0006-weight-mapping-via-template.md) | Weights Map from HuggingFace Names via Template | Accepted |

See also: [v4-design.md](../v4-design.md) for the overall pipeline design.

