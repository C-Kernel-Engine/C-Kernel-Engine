# Graph Templates (v8)

This directory holds model graph templates. A template defines the per-layer
operation sequence (the logical graph). It is the missing piece between the
weights manifest (what tensors exist) and the kernel registry (what kernels
can execute ops).

Templates are architecture-level, not model-specific. They are parameterized by
model config and the weights manifest.

Do not confuse these templates with the buffer templates used inside
`version/v8/scripts/build_ir_v8.py`. Those are memory layout templates.
These files describe compute graphs.

## Why this exists

1) We need an explicit op order for each model family (Qwen2, LLaMA, Mistral).
2) Kernel selection uses the weights manifest to pick quant variants, but
   it needs a canonical op sequence to know which kernels to try.
3) Fusion is a graph rewrite. We need a graph first.
4) Prefill vs decode are execution plans derived from the same graph
   (e.g., KV cache vs full attention), not separate templates.

## What a template does

- Defines the forward pass op sequence for each block.
- Allows per-layer overrides (for MoE or hybrid blocks).
- Encodes architecture flags (bias usage, rope type, activation type).
- Provides a stable op vocabulary for IR1.

## What a template does not do

- It does not define backward pass (derived automatically from forward sequence).
- It does not store weights or offsets (that is `weights_manifest.json`).
- It does not choose kernels (that is done by kernel registry + selection).
- It does not define memory layout (that is IR2 + memory planner).

## File naming

- One JSON per model family. Examples: `qwen2.json`, `llama.json`, `mistral.json`.
- Keep names lowercase and stable; they are referenced by IR tooling.

## Bring-up Notes

- Use `llama_nanbeige_contract.md` as the working checklist for Llama-family
  model onboarding when output quality is gibberish despite successful compile.
- It captures tokenizer/chat-stop semantics and strict gate ordering needed
  before interactive generation is considered valid.

## Schema Evolution

### Version 1 (Legacy - Text-only Models)

Original schema for decoder-only text models with per-layer type overrides.

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

**Limitations of v1:**
- Cannot represent multi-modal models (vision, audio encoders)
- `layer_map` is decoder-centric (assumes single repeating block)
- No way to specify execution order across different model components
- JSON objects are unordered; sequence must be inferred

### Version 2 (Current - Multi-Modal Support)

Extended schema for multi-modal models with explicit execution sequences.

```json
{
  "version": 2,
  "name": "qwen2_vlm",
  "family": "llama",
  "flags": {
    "use_qkv_bias": "from_weights",
    "activation": "swiglu",
    "rope": "rope"
  },
  "sequence": ["vision_encoder", "decoder"],
  "block_types": {
    "vision_encoder": {
      "sequence": ["header", "body", "footer"],
      "header": [
        "image_tokenizer",
        "vision_embedding"
      ],
      "body": {
        "type": "dense",
        "ops": [
          "layernorm",
          "qkv_proj",
          "attn",
          "out_proj",
          "residual_add",
          "layernorm",
          "mlp_up",
          "gelu",
          "mlp_down",
          "residual_add"
        ]
      },
      "footer": [
        "vision_projection"
      ]
    },
    "decoder": {
      "sequence": ["header", "body", "footer"],
      "header": [
        "tokenizer",
        "dense_embedding_lookup"
      ],
      "body": {
        "type": "dense",
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
      },
      "footer": [
        "weight_tying",
        "logits"
      ]
    }
  }
}
```

**Key changes in v2:**
- **Top-level `"sequence"`**: Array explicitly defining block execution order
  (critical since JSON objects are unordered). For text-only models: `["decoder"]`.
- **Block-level `"sequence"`**: Each block defines its phase execution order
  (typically `["header", "body", "footer"]`). This makes execution flow explicit
  rather than relying on convention.
- `block_types`: Each block now has three phases:
  - `header`: Operations run once at block start (tokenization, embeddings)
  - `body`: Repeating layer operations with explicit `type` field
    - `type`: Layer variant (`"dense"`, `"moe"`, `"sparse"`, etc.)
    - `ops`: Per-layer operation sequence
  - `footer`: Operations run once at block end (projections, output heads)
- `layer_map`: **Removed**. Body type is specified inline via `body.type`.

**Migration from v1 to v2:**
- Set `"version": 2`
- Add top-level `"sequence": ["decoder"]` for text-only models
- Add block-level `"sequence": ["header", "body", "footer"]` to each block
- Wrap ops in block structure:
  - Move one-time ops (tokenizer, embeddings) to `header`
  - Move repeating layer ops to `body.ops`
  - Move output ops (logits) to `footer`
  - Set `body.type` based on layer type (usually `"dense"`)
- Remove `layer_map` field

**Notes:**
- Top-level `sequence` defines block execution order (required in v2)
- Block-level `sequence` defines phase execution order within each block (required in v2)
- `body.type` indicates layer variant (dense, moe, etc.)
- Ops are stable IDs; IR1 maps them to op builders and kernel variants
- For per-layer type mixing (e.g., layers 2,6 are MoE), define separate blocks
  in `block_types` and list them explicitly in `sequence`
- Both sequence arrays are critical: JSON objects have no guaranteed order

## Optional fields

- `op_defs`: map of op id -> metadata (inputs/outputs, params). If present,
  the validator checks that every op in sequences exists in this map.
- `constraints`: architecture limits (e.g., `num_kv_heads <= num_heads`).
- `defaults`: default params used by op builders.

## Validation

Run the validator before wiring into IR:

```bash
python3 version/v8/scripts/build_ir_v8.py --help
```

The validator checks structure and basic consistency. It does not verify
kernel availability; that happens during IR build.

## How IR uses templates

1) Load template by name (e.g., `qwen2.json`).
2) Fill dims/flags from model config and weights manifest.
3) Build IR1 graph from op sequence.
4) Derive execution plans from graph:
   - **Prefill/decode**: Different KV cache strategies from same forward graph
   - **Backpropagation**: Reverse traversal of forward sequence (automatic)
5) Run fusion passes using kernel registry metadata.
6) Emit IR2 + memory plan + codegen.

**Note on backprop**: The template only defines the forward pass. Backward pass is
derived by reversing the sequence and using op-specific gradient rules. No separate
backward template is needed.
