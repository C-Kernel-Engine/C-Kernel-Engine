# Op Builders: Auto-Generation System

## Quick Start

```bash
# 1. Generate op builders from kernel registry (automatic sync!)
python gen_op_builders_from_registry.py

# 2. Check coverage
python op_builders_hybrid_v6_6.py

# 3. Use in IR generation
python build_ir_v6_6.py --test
```

## What Problem Does This Solve?

**Before**: Manual op builders that could get out of sync with kernels
**After**: Auto-generated op builders guaranteed to match kernel registry

### The Problem

You have a template (qwen2.json) that says:
```json
{
  "ops": ["qkv_proj", "out_proj", "mlp_down"]
}
```

You need to:
1. Map these template ops to kernel ops
2. Create GraphIR Op objects with correct inputs/outputs
3. Ensure every op has a corresponding kernel

**Challenge**: Keeping these three things in sync manually is error-prone!

### The Solution

**Auto-generate** op builders from the kernel registry:

```
KERNEL_REGISTRY.json  →  gen_op_builders_from_registry.py  →  op_builders_auto.py
(source of truth)         (auto-generator)                     (generated code)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KERNEL REGISTRY                          │
│  Single source of truth for all kernels                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Auto-generate
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              OP BUILDERS (AUTO)                              │
│  14 op builders generated from kernel metadata              │
│  • build_attention_op()                                      │
│  • build_embedding_op()                                      │
│  • build_gemv_op()                                           │
│  • ...                                                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ + Manual special ops
                 ↓
┌─────────────────────────────────────────────────────────────┐
│              OP BUILDERS (HYBRID)                            │
│  Auto builders + Manual overrides + Template mapping        │
│  • Auto: Standard ops (gemv, attention, ...)                │
│  • Manual: Special ops (tokenizer, weight_tying)            │
│  • Wrappers: Context-aware (rmsnorm counter, residual)      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ Used by IR generator
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                     GRAPHIR                                  │
│  Ops with guaranteed kernel availability                     │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Type | Description |
|------|------|-------------|
| `KERNEL_REGISTRY.json` | Data | Source of truth (14 op types, 22 kernel variants) |
| `gen_op_builders_from_registry.py` | Tool | Auto-generator (extracts metadata, generates code) |
| `op_builders_auto.py` | Generated | Auto-generated builders (DO NOT EDIT) |
| `op_builders_hybrid_v6_6.py` | Code | Hybrid (auto + manual + mapping) |
| `build_ir_v6_6.py` | User | Uses hybrid builders |

## The Three Levels of Ops

### Level 1: Template Ops (Human-Readable)

```json
// qwen2.json
{
  "ops": [
    "qkv_proj",    ← Template op (descriptive name)
    "out_proj",    ← Template op
    "mlp_down"     ← Template op
  ]
}
```

### Level 2: Kernel Ops (Logical Types)

```python
# op_builders_hybrid_v6_6.py
TEMPLATE_TO_KERNEL_MAP = {
    "qkv_proj": "qkv_projection",  ← Kernel op type
    "out_proj": "gemv",            ← Kernel op type (shared!)
    "mlp_down": "gemv",            ← Kernel op type (shared!)
}
```

### Level 3: Kernel Functions (Concrete Implementations)

```json
// KERNEL_REGISTRY.json
{
  "op": "gemv",
  "id": "gemv_q8_0_q8_0",  ← Kernel function for q8_0 weights
}
{
  "op": "gemv",
  "id": "gemv_q4_k_q8_k",  ← Kernel function for q4_k weights
}
```

## How It Works

### 1. Auto-Generation

```bash
$ python gen_op_builders_from_registry.py
Loading kernel registry: KERNEL_REGISTRY.json
  Found 22 kernels

Analyzing kernel metadata...
  Extracted metadata for 14 op types

Generating op builders...
✓ Generated 14 op builders → op_builders_auto.py
```

**What it extracts from each kernel**:
- **Inputs**: `["q", "k", "v"]` → generates input handling
- **Outputs**: `["out"]` → generates output handling
- **Scratch**: `["scores"]` → generates scratch buffer list
- **Dims**: `["H", "D"]` → infers params `{num_heads, head_dim}`

**What it generates**:
```python
def build_attention_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build attention op (auto-generated from kernel registry)."""
    # Input handling
    input_name = ctx.prev_output or make_tensor_name(op_node, 'input')
    k_name = make_tensor_name(op_node, 'k')
    v_name = make_tensor_name(op_node, 'v')

    # Output handling
    output_name = make_tensor_name(op_node, 'out')

    # Create op
    op = Op(
        op="attention",  # Kernel op type
        name=make_tensor_name(op_node, "attention"),
        inputs=[input_name, k_name, v_name],
        outputs=[output_name],
        scratch=['scores'],
        params={
            'num_heads': ctx.config.get('num_heads', 0),
            'head_dim': ctx.config.get('head_dim', 0),
        }
    )
    ctx.set_output(output_name)
    return op
```

### 2. Hybrid Approach

Why not use auto-generated builders directly?
- **Special ops**: `tokenizer`, `weight_tying` are metadata-only (no kernels)
- **Context tracking**: `rmsnorm` needs counter (ln1, ln2), `residual_add` needs saved input
- **Template mapping**: Need to map user-friendly template names to kernel ops

Solution: **Hybrid** = Auto + Manual

```python
# op_builders_hybrid_v6_6.py

# Start with auto-generated
OP_BUILDERS = AUTO_BUILDERS.copy()

# Add manual builders for special ops
OP_BUILDERS.update({
    "tokenizer": build_tokenizer_op,         # Metadata only
    "weight_tying": build_weight_tying_op,   # Metadata only
})

# Add wrappers for context-aware ops
OP_BUILDERS.update({
    "rmsnorm": build_rmsnorm_op_wrapper,     # Tracks counter
    "residual_add": build_residual_add_op_wrapper,  # Saves residual input
})
```

### 3. Template Mapping

```python
TEMPLATE_TO_KERNEL_MAP = {
    # Header
    "tokenizer": "tokenizer",
    "dense_embedding_lookup": "embedding",

    # Attention
    "qkv_proj": "qkv_projection",
    "rope_qk": "rope",
    "attn": "attention",
    "out_proj": "gemv",  # Linear projection

    # Residual
    "residual_add": "residual_add",

    # MLP
    "mlp_gate_up": "fused_mlp_block",
    "silu_mul": "swiglu",
    "mlp_down": "gemv",  # Linear projection

    # Footer
    "weight_tying": "weight_tying",
    "logits": "gemv",  # Linear projection
}
```

**Key insight**: Multiple template ops can map to same kernel op!
- `out_proj`, `mlp_down`, `logits` all → `gemv`

### 4. Usage in IR Generation

```python
# build_ir_v6_6.py
from op_builders_hybrid_v6_6 import build_op_from_template, OpContext

ctx = OpContext(config)

for op_node in template_ops:
    # op_node.op_id = "mlp_down" (template op)

    # Build op (automatically maps to kernel op "gemv")
    ir_op = build_op_from_template(op_node, ctx, manifest)

    # ir_op.op = "gemv" (kernel op type)
    # ir_op.kernel = None (filled in lowering)
```

## Benefits

### 1. **Guaranteed Kernel Availability**

✅ **Auto-generated** → Every op builder corresponds to a real kernel
❌ **Manual** → Op builder might reference non-existent kernel

### 2. **Zero Maintenance**

When kernels change:
```bash
# Regenerate registry
python gen_kernel_registry_from_maps.py

# Regenerate op builders
python gen_op_builders_from_registry.py

# Done! Op builders automatically updated.
```

### 3. **Early Error Detection**

```bash
$ python build_ir_v6_6.py --test

# Missing kernel detected immediately:
❌ ERROR: Missing kernels for required operations!
  - new_fancy_op (needed 2x)

Action required:
  1. Implement missing kernels in kernel registry
  2. Or remove these operations from the template
```

### 4. **Coverage Reports**

```bash
$ python op_builders_hybrid_v6_6.py

Template Op → Kernel Op Mapping:
  ✓ (auto) attn → attention
  ✓ (auto) qkv_proj → qkv_projection
  ✓ (manual) tokenizer → tokenizer

Total template ops: 13
Auto-generated: 14
Manual: 2
```

## Workflows

### Adding a New Kernel

1. Add kernel to kernel maps
2. Regenerate registry:
   ```bash
   python gen_kernel_registry_from_maps.py
   ```
3. Regenerate op builders:
   ```bash
   python gen_op_builders_from_registry.py
   ```
4. Update template mapping (if needed):
   ```python
   TEMPLATE_TO_KERNEL_MAP["new_template_op"] = "new_kernel_op"
   ```

### Adding a New Template Op

1. Add op to template (qwen2.json)
2. Map to existing kernel op:
   ```python
   TEMPLATE_TO_KERNEL_MAP["new_op"] = "gemv"  # Reuse existing!
   ```
3. Done! (no op builder needed, already generated)

### Regenerating After Kernel Changes

```bash
# One command regenerates everything:
python gen_op_builders_from_registry.py && python build_ir_v6_6.py --test
```

## Implementation Details

### Metadata Extraction

From this kernel:
```json
{
  "op": "attention",
  "inputs": [
    {"name": "q", "shape": ["H", "T", "D"]},
    {"name": "k", "shape": ["KV", "S", "D"]},
    {"name": "v", "shape": ["KV", "S", "D"]}
  ],
  "outputs": [
    {"name": "out", "shape": ["H", "T", "D"]}
  ],
  "dims": ["H", "KV", "T", "S", "D"]
}
```

We extract:
```python
{
  "inputs": ["q", "k", "v"],
  "outputs": ["out"],
  "params": ["num_heads", "num_kv_heads", "head_dim"]  # H→num_heads, D→head_dim
}
```

### Dimension Inference

```python
dim_to_param = {
    'H': 'num_heads',
    'KV': 'num_kv_heads',
    'D': 'head_dim',
    'E': 'embed_dim',
    'V': 'vocab_size',
    'I': 'intermediate_size',
}
```

### Weight Inference

Identify weight tensors by:
1. Name patterns: `wq`, `wk`, `wv`, `weight`, `w1`, `w2`
2. Shape: 2D or 3D tensors (not activation tensors)
3. Description: "weight" or "matrix" in description

## Testing

```bash
# 1. Generate op builders
python gen_op_builders_from_registry.py

# 2. Check coverage
python op_builders_hybrid_v6_6.py

# 3. Test with manifest
python build_ir_v6_6.py --test

# 4. Full pipeline test
python test_pipeline_v2.py
```

## Summary

**Before**: Manual op builders, prone to errors, hard to maintain

**After**: Auto-generated from kernel registry, guaranteed correctness

**Key Innovation**: Three-level mapping (Template → Kernel Op → Kernel Function) with auto-generation of the middle layer

**Result**: Same template works with any quantization, zero maintenance!
