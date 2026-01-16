ADR-0003: Mode-Specific Lowering (Prefill/Decode/Backward)
===========================================================

Status
- Accepted

Context
- Prefill and decode have different memory footprints and kernel requirements.
- Training needs activations and gradients that inference does not.

Decision
- Lower Graph IR into per-mode programs:
  - prefill: full sequence buffers, optional attention scratch.
  - decode: token=1 buffers, kv-cache update, minimal scratch.
  - backward: gradients + saved activations.
- Buffer materialization is controlled by "when" guards in the template/graph.

Consequences
- Enables smaller decode memory and faster paths.
- Layout and codegen are mode-specific and deterministic.
- Requires explicit "when" metadata in templates/graph.

Alternatives
- Single unified layout for all modes (rejected: too large and slower).

