# IR v4 Design (Bridge v2 + v3)

This document defines IR v4 as a bridge between:
- v2: explicit lowering and memory planning
- v3: deterministic layout and generated C with canaries

The goal is to keep the architecture intent clear while producing a fully
deterministic layout and codegen output.

References
- ADR-0001: IR v4 pipeline split
- ADR-0002: Templates as canonical architecture source
- ADR-0003: Mode-specific lowering

---

## 1. Pipeline Overview

Inputs:
- config.json (model dimensions and flags)
- safetensors header (weight names + shapes)
- template.yaml (architecture and weight mapping)

Outputs:
- graph.json (macro graph IR)
- lowered.json (mode-specific IR)
- layout.json / layout.map (resolved offsets + canaries)
- generated_*.h/.c (deterministic C output)

Flow:

1) Template selection + bindings
2) Graph IR (macro)
3) Lowering (prefill/decode/backward)
4) Layout (bump offsets + canaries)
5) Codegen

---

## 2. Graph IR (Macro)

Graph IR captures:
- header/body/footer section structure
- ops and their wiring (explicit residuals)
- buffer definitions with symbolic shapes
- weight mapping (HF -> internal names)

Key fields:
- symbols: E, H, KV, D, I, T, V
- sections[]: header/body/footer ops + buffers
- when[]: optional mode guards for ops/buffers

---

## 3. Lowered IR

Lowering resolves:
- per-mode program lists (prefill/decode/backward)
- kernel variants (e.g., q4_k_q8_k vs bf16)
- resolved shapes and strides
- buffer materialization rules

Lowered IR is the direct input to the layout planner.

---

## 4. Layout IR

Layout IR provides:
- final resolved shapes
- byte offsets + sizes (alignment enforced)
- canary markers
- totals (weights/activations/cache/grad)

This is deterministic and feeds directly into generated code.

---

## 5. Codegen Expectations

The generated runtime should:
- use offsets only (no pointer tables in weights)
- provide per-mode entrypoints (prefill/decode/backward)
- support deterministic layouts with optional canary checks

