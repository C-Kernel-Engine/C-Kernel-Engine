# IR Pipeline: Template → GraphIR → LoweredIR

This document explains how templates flow through the IR generation pipeline.

## Overview: The Three-Stage Pipeline

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Template   │  →   │  GraphIR    │  →   │  LoweredIR  │
│   (v2 JSON) │      │ (Symbolic)  │      │ (Concrete)  │
└─────────────┘      └─────────────┘      └─────────────┘
    Stage 1              Stage 2              Stage 3
   Parser           Op Builders          Kernel Selection
```

## Stage 1: Template Parsing

**Input:** Template v2 JSON (e.g., `qwen2.json`)
**Tool:** `parse_template_v2.py`
**Output:** List of `OpNode` objects

### What Happens:

```
Template JSON                    →    OpNode Objects
─────────────                          ───────────────
{                                      OpNode(
  "sequence": ["decoder"],               op_id="rmsnorm",
  "block_types": {                       block="decoder",
    "decoder": {                         phase="body",
      "body": {                          layer_index=0
        "ops": [                       )
          "rmsnorm",
          "qkv_proj",                  OpNode(
          ...                            op_id="qkv_proj",
        ]                                block="decoder",
      }                                  phase="body",
    }                                    layer_index=0
  }                                    )
}                                      ...
```

**Purpose:** Parse template structure and generate flat operation sequence.

## Stage 2: GraphIR Generation (Op Builders)

**Input:** `OpNode` objects + config + manifest
**Tool:** `op_builders_v7.py` (what we're building)
**Output:** `GraphIR` with symbolic `Op` objects

### What Happens:

```
OpNode                          →    GraphIR Op (Symbolic)
──────                               ──────────────────────
OpNode(                              Op(
  op_id="rmsnorm",                     op="rmsnorm",          ← Op type for kernel family
  block="decoder",                     name="layer_0_ln1",    ← Unique identifier
  phase="body",                        inputs=["layer.0.input"],  ← SYMBOLIC tensor name
  layer_index=0                        outputs=["layer.0.ln1_out"], ← SYMBOLIC tensor name
)                                      weights=["layer.0.ln1_gamma"], ← Weight reference
                                       kernel=None,           ← NOT selected yet!
                                       params={"eps": 1e-5}   ← From config
                                     )
```

### Why Op Builders Are Needed:

The template says **WHAT** operations to do, but not **HOW** they wire together:

| Template Provides | Op Builder Adds |
|-------------------|-----------------|
| Op name: `"rmsnorm"` | Op type: `"rmsnorm"` (for kernel registry) |
| Layer index: `0` | Unique name: `"layer_0_ln1"` |
| - | Input tensor: `"layer.0.input"` (from prev op) |
| - | Output tensor: `"layer.0.ln1_out"` (for next op) |
| - | Weight name: `"layer.0.ln1_gamma"` (from manifest) |
| - | Parameters: `{"eps": 1e-5}` (from config) |

**Key Point:** GraphIR uses **symbolic names** (strings), not memory addresses!

### Example: Full Layer Wiring

```
Template Ops:          GraphIR Ops:                              Tensor Flow:
─────────────          ────────────                              ────────────

"rmsnorm"         →    Op(op="rmsnorm",                          input
                         inputs=["layer.0.input"],                  ↓
                         outputs=["layer.0.ln1_out"])          ln1_out
                                                                    ↓
"qkv_proj"        →    Op(op="qkv_projection",                 q, k, v
                         inputs=["layer.0.ln1_out"],               ↓
                         outputs=["layer.0.q",                  (after rope)
                                  "layer.0.k",                      ↓
                                  "layer.0.v"])                attn_out
                                                                    ↓
"rope_qk"         →    Op(op="rope",                           attn_out
                         inputs=["layer.0.q",                       ↓
                                 "layer.0.k"],                attn_proj_out
                         outputs=["layer.0.q",
                                  "layer.0.k"])

"attn"            →    Op(op="attention",
                         inputs=["layer.0.q",
                                 "layer.0.k",
                                 "layer.0.v"],
                         outputs=["layer.0.attn_out"])

"out_proj"        →    Op(op="linear",
                         inputs=["layer.0.attn_out"],
                         outputs=["layer.0.attn_proj_out"])
```

**Op builders connect the dots** by:
1. Tracking previous op's output → use as next op's input
2. Generating consistent tensor names
3. Mapping to weight names from manifest

## Stage 3: Lowering (Kernel Selection)

**Input:** `GraphIR` with `Op` objects (kernel=None)
**Tool:** `v7_ir_lowering.py` (already exists)
**Output:** `LoweredIR` with concrete kernels

### What Happens:

```
GraphIR Op                      →    LoweredIR Op (Concrete)
──────────                           ───────────────────────
Op(                                  Op(
  op="rmsnorm",                        op="rmsnorm",
  inputs=["layer.0.input"],            inputs=["layer.0.input"],
  outputs=["layer.0.ln1_out"],         outputs=["layer.0.ln1_out"],
  weights=["layer.0.ln1_gamma"],       weights=["layer.0.ln1_gamma"],
  kernel=None  ← NO KERNEL             kernel="rmsnorm_fp32"  ← KERNEL SELECTED!
)                                    )
                                        ↑
                                        │
                                    Kernel Registry
                                    ───────────────
                                    Looks at:
                                    - op type: "rmsnorm"
                                    - weight dtype: "fp32"
                                    - mode: "decode"

                                    Finds: "rmsnorm_fp32_decode_kernel"
```

**How kernel is selected:**

```python
# Lowering stage (simplified)
for op in graph_ir.ops:
    if op.kernel is None:  # Not selected yet
        # Look up in kernel registry
        weight_dtype = manifest["weights"][op.weights[0]]["dtype"]
        kernel_id = registry.find_kernel(
            op_type=op.op,
            weight_dtype=weight_dtype,
            mode="decode"
        )
        op.kernel = kernel_id  # Fill in the kernel!
```

## Why This Three-Stage Design?

### Separation of Concerns:

| Stage | Responsibility | Knows About |
|-------|---------------|-------------|
| **Template** | What ops, what order | Model architecture |
| **GraphIR** | How ops wire together | Tensor flow, weight names |
| **LoweredIR** | Which kernels to use | Hardware, quantization |

### Benefits:

1. **Template is clean:** Just lists operations, no kernel details
2. **GraphIR is portable:** Same IR works for different hardware
3. **Lowering is flexible:** Can swap kernels without changing IR

### Example: Same GraphIR, Different Kernels

```
GraphIR (same for all):
Op(op="rmsnorm", kernel=None, ...)

LoweredIR (different based on quant):
- FP32 weights  → kernel="rmsnorm_fp32_kernel"
- Q8_0 weights  → kernel="rmsnorm_q8_0_kernel"
- Q4_K weights  → kernel="rmsnorm_q4_k_kernel"
```

## Complete Example: rmsnorm → attention

```
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 1: Template Parsing                                            │
├──────────────────────────────────────────────────────────────────────┤
│ Input:  ["rmsnorm", "qkv_proj", "rope_qk", "attn"]                  │
│ Output: [OpNode("rmsnorm"), OpNode("qkv_proj"), ...]                │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 2: GraphIR Generation (Op Builders)                            │
├──────────────────────────────────────────────────────────────────────┤
│ OpNode("rmsnorm")                                                    │
│   → Op(op="rmsnorm", inputs=["layer.0.input"],                      │
│        outputs=["layer.0.ln1_out"], kernel=None)                    │
│                                                                       │
│ OpNode("qkv_proj")                                                   │
│   → Op(op="qkv_projection", inputs=["layer.0.ln1_out"],            │
│        outputs=["layer.0.q", "layer.0.k", "layer.0.v"],            │
│        kernel=None)                                                  │
│                                                                       │
│ OpNode("rope_qk")                                                    │
│   → Op(op="rope", inputs=["layer.0.q", "layer.0.k"],               │
│        outputs=["layer.0.q", "layer.0.k"], kernel=None)            │
│                                                                       │
│ OpNode("attn")                                                       │
│   → Op(op="attention", inputs=["layer.0.q", "layer.0.k", ...],     │
│        outputs=["layer.0.attn_out"], kernel=None)                  │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Stage 3: Lowering (Kernel Selection)                                 │
├──────────────────────────────────────────────────────────────────────┤
│ Check manifest: weights are Q8_0                                     │
│ Check registry: find kernels for Q8_0 + decode mode                  │
│                                                                       │
│ Op(op="rmsnorm", ...)                                                │
│   kernel=None → kernel="rmsnorm_q8_0_decode"                        │
│                                                                       │
│ Op(op="qkv_projection", ...)                                         │
│   kernel=None → kernel="qkv_proj_q8_0_decode"                       │
│                                                                       │
│ Op(op="rope", ...)                                                   │
│   kernel=None → kernel="rope_inplace"                               │
│                                                                       │
│ Op(op="attention", ...)                                              │
│   kernel=None → kernel="attention_decode_q8_0"                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Summary: Why Op Builders?

**Problem:** Template gives us operation names, but `ir_types_v7.py` needs:
- Tensor names (strings)
- Op wiring (which output → which input)
- Weight references
- Parameters from config

**Solution:** Op builders bridge the gap:
```
Template OpNode → [Op Builder] → GraphIR Op
```

Op builders know:
- How to name tensors: `"layer.{L}.ln{N}_out"`
- How to wire ops: previous output → next input
- What weights are needed: `"layer.{L}.ln{N}_gamma"`
- What params come from config: `{"eps": config["rms_eps"]}`

**Without op builders**, we'd have to hardcode all this logic inline, which would be:
- Hard to maintain
- Not reusable
- Difficult to test
- Coupled to specific architectures

**With op builders**, we have clean separation:
- Template: architecture specification
- Op builders: wiring logic (reusable, testable)
- IR types: data structures (generic)
