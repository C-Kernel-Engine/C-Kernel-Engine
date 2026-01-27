# Kernel Mapping: Template → Kernel Op → Kernel Function

## The Three Levels

### Level 1: Template Ops (Architecture)
**File**: `qwen2.json`
**Purpose**: Describe model architecture with user-friendly names

```json
{
  "body": {
    "ops": [
      "qkv_proj",     ← Template op name
      "out_proj",     ← Template op name
      "mlp_down",     ← Template op name
      "logits"        ← Template op name
    ]
  }
}
```

### Level 2: Kernel Ops (Logical Operations)
**File**: `KERNEL_REGISTRY.json`
**Purpose**: Group operations by kernel family (what computation)

```json
{
  "kernels": [
    {
      "op": "qkv_projection",  ← Kernel op type
      "id": "qkv_proj_...",
      ...
    },
    {
      "op": "gemv",  ← Kernel op type (shared by multiple template ops!)
      "id": "gemv_q8_0_q8_0",
      ...
    }
  ]
}
```

### Level 3: Kernel Functions (Implementations)
**File**: `kernel_maps/*.json` → C functions
**Purpose**: Concrete implementation (how to compute with specific quantization)

```c
// Kernel function for gemv with q8_0 weights
void gemv_q8_0_q8_0(
    const void* weight,     // q8_0 quantized
    const void* activation, // q8_0 quantized
    void* output           // fp32
);

// Different kernel function for gemv with q4_k weights
void gemv_q4_k_q8_k(
    const void* weight,     // q4_k quantized
    const void* activation, // q8_k quantized
    void* output           // fp32
);
```

## Complete Flow Example

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Template (qwen2.json)                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ "out_proj" ───┐                                                     │
│ "mlp_down" ───┼─── Three different template ops                    │
│ "logits"   ───┘                                                     │
│                                                                      │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 │ op_builders_hybrid_v6_6.py maps all three to:
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Kernel Op Type (GraphIR)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Op(op="gemv", name="out_proj", ...)   ───┐                         │
│ Op(op="gemv", name="mlp_down", ...)   ───┼─── Same kernel op type  │
│ Op(op="gemv", name="logits", ...)     ───┘                         │
│                                                                      │
│ All use logical "gemv" operation                                    │
│ (Generic Matrix-Vector multiplication)                              │
│                                                                      │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 │ Lowering stage selects specific kernel based on:
                 │  - Weight quantization (q8_0, q4_k, fp32, etc.)
                 │  - Activation quantization
                 │  - Mode (decode vs prefill)
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Concrete Kernel Function (LoweredIR)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ If weights are q8_0:                                                │
│   Op(op="gemv", kernel="gemv_q8_0_q8_0", ...)                      │
│                                                                      │
│ If weights are q4_k:                                                │
│   Op(op="gemv", kernel="gemv_q4_k_q8_k", ...)                      │
│                                                                      │
│ If weights are fp32:                                                │
│   Op(op="gemv", kernel="gemv_fp32_fp32", ...)                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Mapping Tables

### Template Op → Kernel Op (Level 1 → 2)

Defined in `op_builders_hybrid_v6_6.py`:

| Template Op (qwen2.json) | Kernel Op (Registry) | Notes |
|--------------------------|---------------------|-------|
| `tokenizer` | `tokenizer` | Metadata only |
| `dense_embedding_lookup` | `embedding` | |
| `rmsnorm` | `rmsnorm` | |
| `qkv_proj` | `qkv_projection` | Specialized kernel |
| `rope_qk` | `rope` | In-place operation |
| `attn` | `attention` | Flash attention |
| `out_proj` | **`gemv`** | Linear projection |
| `residual_add` | `residual_add` | Element-wise add |
| `mlp_gate_up` | `fused_mlp_block` | Fused gate+up |
| `silu_mul` | `swiglu` | SiLU activation |
| `mlp_down` | **`gemv`** | Linear projection |
| `weight_tying` | `weight_tying` | Metadata only |
| `logits` | **`gemv`** | Linear projection |

**Key insight**: Multiple template ops map to the same kernel op (`gemv`)!

### Kernel Op → Kernel Functions (Level 2 → 3)

Defined in `KERNEL_REGISTRY.json`:

| Kernel Op | Available Implementations | Selected By |
|-----------|-------------------------|-------------|
| `gemv` | `gemv_q8_0_q8_0`<br>`gemv_q4_k_q8_k`<br>`gemv_q5_0_q8_0`<br>`gemv_q6_k_q8_k` | Weight dtype |
| `qkv_projection` | `qkv_proj_q8_0_...`<br>`qkv_proj_fp32_...` | Weight dtype |
| `attention` | `attention_forward_causal_...` | Mode (decode/prefill) |
| `rmsnorm` | `rmsnorm_fp32_...` | Always fp32 |
| `rope` | `rope_inplace_...` | Head dim |

## Why Three Levels?

### Separation of Concerns

1. **Template level**:
   - **Audience**: ML engineers, model architects
   - **Question**: "What operations does this model need?"
   - **Example**: "out_proj", "mlp_down" (descriptive names)

2. **Kernel op level**:
   - **Audience**: IR generator, compiler
   - **Question**: "What type of computation is this?"
   - **Example**: "gemv" (all linear projections are matrix-vector products)

3. **Kernel function level**:
   - **Audience**: Runtime, code generator
   - **Question**: "Which C function should I call?"
   - **Example**: "gemv_q8_0_q8_0" (specific implementation for q8_0 weights)

### Flexibility

This design allows:
- **One template op → One kernel op**: `qkv_proj` → `qkv_projection`
- **Many template ops → One kernel op**: `out_proj`, `mlp_down`, `logits` → `gemv`
- **One kernel op → Many kernel functions**: `gemv` → `gemv_q8_0_q8_0`, `gemv_q4_k_q8_k`, ...

### Quantization Independence

Template doesn't specify quantization:
```json
{
  "ops": ["mlp_down"]  // No quantization info
}
```

Kernel registry has all quantization variants:
```json
{
  "op": "gemv",
  "variants": [
    {"quant": {"weight": "q8_0"}},
    {"quant": {"weight": "q4_k"}},
    {"quant": {"weight": "fp32"}}
  ]
}
```

Lowering selects based on actual weights:
```python
if weight.dtype == "q8_0":
    kernel = "gemv_q8_0_q8_0"
elif weight.dtype == "q4_k":
    kernel = "gemv_q4_k_q8_k"
```

## Op Builder Role

**Op builders bridge Level 1 → Level 2**:

```python
# Input: Template op node
op_node = OpNode(op_id="mlp_down", layer_index=0)

# Op builder maps template op → kernel op
TEMPLATE_TO_KERNEL_MAP = {
    "mlp_down": "gemv"  # Level 1 → Level 2
}

# Build GraphIR Op with kernel op type
op = Op(
    op="gemv",  # ← Kernel op type (Level 2)
    name="layer.0.mlp_down",
    inputs=["layer.0.mlp_act_out"],
    outputs=["layer.0.mlp_out"],
    weights=["layer.0.w2"],
    kernel=None  # ← Will be filled in lowering (Level 3)
)
```

## Kernel Selection (Lowering)

**Lowering stage fills Level 3**:

```python
# Input: GraphIR Op with kernel op type
op = Op(op="gemv", kernel=None, weights=["layer.0.w2"])

# Look up weight quantization
weight_dtype = manifest["weights"]["layer.0.w2"]["dtype"]  # "q8_0"

# Find matching kernel
for kernel in registry["kernels"]:
    if kernel["op"] == "gemv" and kernel["quant"]["weight"] == "q8_0":
        op.kernel = kernel["id"]  # "gemv_q8_0_q8_0"
        break

# Output: LoweredIR Op with concrete kernel
op = Op(op="gemv", kernel="gemv_q8_0_q8_0", ...)
```

## Summary

```
Template Op        Kernel Op         Kernel Function
(What)             (Type)            (How)
─────────────     ──────────────    ────────────────────
"out_proj"    →   "gemv"       →   "gemv_q8_0_q8_0"
"mlp_down"    →   "gemv"       →   "gemv_q8_0_q8_0"
"logits"      →   "gemv"       →   "gemv_q8_0_q8_0"

"qkv_proj"    →   "qkv_projection" → "qkv_proj_q8_0_..."

"attn"        →   "attention"  →   "attention_forward_causal_..."
```

**Three levels, three purposes**:
1. **Template**: Human-readable architecture
2. **Kernel Op**: Logical operation type (what computation)
3. **Kernel Function**: Concrete implementation (how to compute)

**Two mappings**:
1. **Op Builders** (compile-time): Template → Kernel Op
2. **Lowering** (compile-time): Kernel Op → Kernel Function (based on quantization)

This design allows the same template to work with different quantizations without any changes!
