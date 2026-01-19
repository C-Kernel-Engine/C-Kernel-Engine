# Graph Templates (v6.6)

This directory holds model graph templates. A template defines the per-layer
operation sequence (the logical graph). It is the missing piece between the
weights manifest (what tensors exist) and the kernel registry (what kernels
can execute ops).

Templates are architecture-level, not model-specific. They are parameterized by
model config and the weights manifest.

Do not confuse these templates with the buffer templates used inside
`version/v6.6/scripts/build_ir_v6_6.py`. Those are memory layout templates.
These files describe compute graphs.

## Why this exists

1) We need an explicit op order for each model family (Qwen2, LLaMA, Mistral).
2) Kernel selection uses the weights manifest to pick quant variants, but
   it needs a canonical op sequence to know which kernels to try.
3) Fusion is a graph rewrite. We need a graph first.
4) Prefill vs decode are execution plans derived from the same graph
   (e.g., KV cache vs full attention), not separate templates.

## What a template does

- Defines the op sequence for each layer.
- Allows per-layer overrides (for MoE or hybrid blocks).
- Encodes architecture flags (bias usage, rope type, activation type).
- Provides a stable op vocabulary for IR1.

## What a template does not do

- It does not store weights or offsets (that is `weights_manifest.json`).
- It does not choose kernels (that is done by kernel registry + selection).
- It does not define memory layout (that is IR2 + memory planner).

## File naming

- One JSON per model family. Examples: `qwen2.json`, `llama.json`, `mistral.json`.
- Keep names lowercase and stable; they are referenced by IR tooling.

## Minimal schema

```json
{
  "version": 1,
  "name": "qwen2",
  "family": "llama",
  "flags": {
    "use_qkv_bias": "from_weights",
    "activation": "swiglu",
    "rope": "rope"
  },
  "block_types": {
    "dense": {
      "ops": [
        "rmsnorm",
        "qkv_proj",
        "rope_qk",
        "attn",
        "out_proj",
        "residual_add",
        "rmsnorm",
        "mlp_gate_up",
        "silu_mul",
        "mlp_down",
        "residual_add"
      ]
    }
  },
  "layer_map": {
    "default": "dense",
    "overrides": {
      "2": "moe",
      "6": "moe"
    }
  }
}
```

Notes:
- `block_types` defines named block sequences. Each block has a single `ops`
  list. Prefill/decode are derived later by the planner.
- `layer_map` chooses which block a layer uses. Overrides are string keys
  of layer indices.
- Ops are stable IDs; IR1 maps them to op builders and then to kernel variants.

## Optional fields

- `op_defs`: map of op id -> metadata (inputs/outputs, params). If present,
  the validator checks that every op in sequences exists in this map.
- `constraints`: architecture limits (e.g., `num_kv_heads <= num_heads`).
- `defaults`: default params used by op builders.

## Validation

Run the validator before wiring into IR:

```bash
python3 version/v6.6/scripts/validate_templates.py
```

The validator checks structure and basic consistency. It does not verify
kernel availability; that happens during IR build.

## How IR uses templates

1) Load template by name (e.g., `qwen2.json`).
2) Fill dims/flags from model config and weights manifest.
3) Build IR1 graph from op sequence.
4) Derive execution plan (prefill/decode) from graph and cache policy.
5) Run fusion passes using kernel registry metadata.
6) Emit IR2 + memory plan + codegen.
