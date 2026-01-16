# IR v2 Format Specification

This document describes the IR v2 (Intermediate Representation version 2) format
used by C-Kernel-Engine to represent neural network models in a portable,
machine-readable JSON format.

## Overview

IR v2 is a self-documenting JSON format that captures:
- Model configuration (dimensions, hyperparameters)
- Buffer definitions (weights, activations, gradients)
- Computation graph (kernel invocations and their bindings)

The format is designed to be:
1. **Portable** - Same IR works for different batch sizes
2. **Self-documenting** - Contains notes explaining the format
3. **Extensible** - Supports encoder, decoder, and encoder-decoder architectures

## Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  HuggingFace    │     │   kernel_maps/*.json │     │  build_ir_v2.py │
│  config.json    │────▶│   (buffer & layer    │────▶│  (the compiler) │
│                 │     │    definitions)      │     │                 │
└─────────────────┘     └──────────────────────┘     └────────┬────────┘
                                                              │
                                                              ▼
                        ┌──────────────────────┐     ┌─────────────────┐
                        │  generated_v2.c      │◀────│   ir_v2.json    │
                        │  (C runtime)         │     │  (portable IR)  │
                        └──────────────────────┘     └─────────────────┘
```

## File Structure

```json
{
  "version": 2,
  "notes": ["... format guide ..."],
  "config": { ... model dimensions ... },
  "dimensions": [ ... symbolic dimension table ... ],
  "buffers": [ ... weight/activation definitions ... ],
  "nodes": [ ... kernel invocations ... ],
  "meta": { ... optional metadata ... }
}
```

## Dimensions

The `dimensions` array maps numeric IDs to named values from the model config.
This allows shapes to be symbolic, making the IR portable across batch sizes.

```json
"dimensions": [
  {"id": 0, "name": "tokens", "value": 131072},
  {"id": 1, "name": "embed", "value": 896},
  {"id": 2, "name": "aligned_embed", "value": 896},
  {"id": 3, "name": "head_dim", "value": 64},
  {"id": 5, "name": "num_heads", "value": 14},
  {"id": 6, "name": "num_kv_heads", "value": 2},
  {"id": 8, "name": "intermediate", "value": 4864},
  {"id": 10, "name": "vocab", "value": 151936}
]
```

### Dimension Mapping

| ID | Name | Source in config.json |
|----|------|----------------------|
| 0 | tokens | batch_size × seq_len (runtime) |
| 1 | embed | hidden_size |
| 2 | aligned_embed | hidden_size (64-byte aligned) |
| 3 | head_dim | hidden_size / num_attention_heads |
| 5 | num_heads | num_attention_heads |
| 6 | num_kv_heads | num_key_value_heads |
| 8 | intermediate | intermediate_size |
| 10 | vocab | vocab_size |

## Buffers

Each buffer represents a tensor in the model:

```json
{
  "name": "token_emb",
  "scope": "global",
  "role": "weight",
  "dtype": "bf16",
  "shape": [{"dim": 10, "mult": 1, "div": 1}, {"dim": 2, "mult": 1, "div": 1}],
  "resolved_shape": [272269312],
  "alias_of": null,
  "condition": null
}
```

### Buffer Fields

| Field | Description |
|-------|-------------|
| `name` | Unique identifier |
| `scope` | `global` (shared) or `layer` (per-layer) |
| `role` | `weight`, `activation`, or `grad` |
| `dtype` | Data type: `fp32`, `bf16`, `fp16`, `q4_k`, `q6_k`, etc. |
| `shape` | Symbolic dimensions: `[{dim, mult, div}, ...]` |
| `resolved_shape` | Concrete byte sizes after alignment |
| `alias_of` | Buffer this aliases (e.g., tied embeddings) |
| `condition` | When this buffer is used (e.g., `rope_theta>0`) |

### Shape Resolution

To compute the size of a shape dimension:
```
size = dimensions[dim].value × mult / div
```

Example: `{"dim": 3, "mult": 1, "div": 2}` with head_dim=64 → 32

## Nodes

Each node represents one kernel invocation:

```json
{
  "layer": 0,
  "op": "rmsnorm",
  "kernel": "rmsnorm_forward",
  "flags": 0,
  "condition": null,
  "bindings": [
    {"arg": "input", "buffer": "input"},
    {"arg": "gamma", "buffer": "ln1_gamma"},
    {"arg": "out", "buffer": "ln1_out"},
    {"arg": "rstd", "buffer": "ln1_rstd"}
  ]
}
```

### Node Fields

| Field | Description |
|-------|-------------|
| `layer` | Transformer layer index (0-based), or -1 for global ops |
| `op` | Operation type (rmsnorm, attention, mlp, etc.) |
| `kernel` | C function name to call |
| `flags` | Bitflags for kernel variants |
| `condition` | When to execute (e.g., `rope_theta>0`) |
| `bindings` | Maps kernel arguments to buffer names |

## Input Files

### 1. HuggingFace config.json

Downloaded from the model repository, provides dimensions:

```json
{
  "hidden_size": 896,
  "vocab_size": 151936,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "intermediate_size": 4864,
  "num_hidden_layers": 24
}
```

### 2. kernel_maps/global_buffers.json

Defines global buffers (embeddings, final norm, logits):

```json
{
  "buffers": [
    {"name": "token_emb", "scope": "global", "role": "weight",
     "shape": [{"dim": "vocab"}, {"dim": "aligned_embed"}]},
    {"name": "logits", "scope": "global", "role": "activation",
     "shape": [{"dim": "tokens"}, {"dim": "vocab"}]}
  ]
}
```

### 3. kernel_maps/decoder_layer_plan.json

Defines the operations in each transformer layer:

```json
{
  "steps": [
    {"kernel": "rmsnorm", "bind": {"input": "input", "gamma": "ln1_gamma", "out": "ln1_out"}},
    {"kernel": "qkv_project", "bind": {"input": "ln1_out", "wq": "wq", ...}},
    {"kernel": "rope", "when": "rope_theta>0", "bind": {"q": "q", "k": "k", ...}},
    {"kernel": "attention", "bind": {"q": "q", "k": "k", "v": "v", ...}},
    {"kernel": "mlp_up", "bind": {...}},
    {"kernel": "swiglu", "bind": {...}},
    {"kernel": "mlp_down", "bind": {...}},
    {"kernel": "residual_add", "bind": {...}}
  ]
}
```

## Usage

### Generate IR v2

```bash
# From HuggingFace model (uses safetensors)
make ir-v2 IR_V2_HF=Qwen/Qwen2-0.5B

# From GGUF quantized model
make ir-v2 IR_V2_HF=Qwen/Qwen2-0.5B-Instruct-GGUF IR_V2_WEIGHTS=qwen2-0_5b-instruct-q4_k_m.gguf
```

### Generate C Runtime

```bash
./build/ck_ir_v2_demo --ir build/ir_v2.json --emit build/generated_v2.c
```

## Extending for Encoder-Decoder

The format supports encoder-decoder models by:

1. Adding `encoder_layer_plan.json` for encoder blocks
2. Adding cross-attention buffers in `global_buffers.json`
3. Setting `"scope": "encoder"` or `"scope": "decoder"` on buffers

## See Also

- [03-ir-and-codegen-design.md](03-ir-and-codegen-design.md) - Original IR design
- [qwen_layer_dataflow.svg](qwen_layer_dataflow.svg) - Layer data flow diagram
- [ir_v2_pipeline.svg](ir_v2_pipeline.svg) - IR v2 pipeline diagram
