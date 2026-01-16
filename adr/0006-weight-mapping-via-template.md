ADR-0006: Weights Map from HuggingFace Names via Template
=========================================================

Status
- Accepted

Context
- HuggingFace models use inconsistent naming across architectures:
  - Llama: model.layers.0.self_attn.q_proj.weight
  - GPT-2: transformer.h.0.attn.c_attn.weight (combined QKV)
  - Qwen2: model.layers.0.self_attn.q_proj.weight
- Weight names must map correctly when loading safetensors.

Decision
- Weight name mapping defined in architecture templates (YAML).
- Bidirectional: HuggingFace name pattern ↔ internal name pattern.
- Layer index placeholder {i} expanded for per-layer weights.

Template Format:
```yaml
weight_mapping:
  "model.embed_tokens.weight": "embed.weight"
  "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attn.wq"
  "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.attn.wk"
  "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate"
  "model.norm.weight": "final_norm.gamma"
  "lm_head.weight": "lm_head.weight"
```

Special Cases:
- Weight tying: multiple HF names → same internal name
- Combined weights (GPT-2): split tensor along axis

Validation:
- At codegen, verify all mapped weights exist in safetensors header.
- Warn about unmapped weights in safetensors.

Consequences
- New architectures added by writing templates, not code.
- Explicit, version-controlled weight mappings.
- Generated code includes weight offset table with both names.

Alternatives
- Hardcoded per-model mappings in Python (rejected: poor extensibility).
- Auto-infer from weight names (rejected: loses explicit control).
