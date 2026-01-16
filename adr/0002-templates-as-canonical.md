ADR-0002: Templates as Canonical Architecture Source
=====================================================

Status
- Accepted

Context
- We need to support multiple model families without hard-coding per-model graphs.
- We want a deterministic, inspectable pipeline and minimal runtime dependencies.

Decision
- Use templates (YAML) as the canonical architecture specification.
- Build Graph IR by combining:
  - config.json (dimensions, flags)
  - safetensors header (weight names/shapes)
  - template.yaml (ops, buffers, section structure, weight mapping)

Consequences
- Architecture intent is explicit and version-controlled.
- New models can be added by writing templates and weight maps.
- Graph IR is generated deterministically from data, not code.

Alternatives
- Python-only templates (rejected: code execution for edits).
- Infer graph directly from weights (rejected: loses semantics and structure).

