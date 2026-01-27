# Op Builder Design: Auto-Generation from Kernel Registry

## Problem Statement

Previously, op builders were manually written and had to be kept in sync with:
1. Template operations (qwen2.json, etc.)
2. Kernel registry (KERNEL_REGISTRY.json)
3. IR types (ir_types_v6_6.py)

This led to:
- **Maintenance burden**: Every new kernel required manual op builder
- **Version skew**: Template ops might not match kernel ops
- **Hard faults**: Missing kernels only discovered at runtime

## Solution: Auto-Generated Op Builders

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   KERNEL REGISTRY                           │
│  (KERNEL_REGISTRY.json - source of truth)                   │
│                                                              │
│  {                                                           │
│    "kernels": [                                              │
│      {                                                       │
│        "op": "attention",                                    │
│        "inputs": [...],  ← Extract metadata                 │
│        "outputs": [...], ← Extract metadata                 │
│        "scratch": [...], ← Extract metadata                 │
│        "dims": [...]     ← Infer params                     │
│      }                                                       │
│    ]                                                         │
│  }                                                           │
└───────────────┬─────────────────────────────────────────────┘
                │
                │ gen_op_builders_from_registry.py
                ↓
┌─────────────────────────────────────────────────────────────┐
│              AUTO-GENERATED OP BUILDERS                      │
│  (op_builders_auto.py)                                       │
│                                                              │
│  def build_attention_op(...):                                │
│      # Auto-generated from kernel metadata                  │
│      op = Op(                                                │
│          op="attention",                                     │
│          inputs=[q, k, v],   ← From kernel.inputs           │
│          outputs=[out],      ← From kernel.outputs          │
│          scratch=[scores],   ← From kernel.scratch          │
│          params={...}        ← From kernel.dims             │
│      )                                                       │
└───────────────┬─────────────────────────────────────────────┘
                │
                │ + manual special ops
                ↓
┌─────────────────────────────────────────────────────────────┐
│              HYBRID OP BUILDERS                              │
│  (op_builders_hybrid_v6_6.py)                                │
│                                                              │
│  - Auto builders for kernel ops                              │
│  - Manual builders for special ops (tokenizer, weight_tying)│
│  - Template → Kernel mapping                                │
│  - Context-aware wrappers (rmsnorm counter, residual)       │
└───────────────┬─────────────────────────────────────────────┘
                │
                │ Used by build_ir_v6_6.py
                ↓
┌─────────────────────────────────────────────────────────────┐
│                      GRAPHIR                                 │
│  Ops with guaranteed kernel availability                     │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. gen_op_builders_from_registry.py

**Purpose**: Analyze kernel registry and auto-generate op builder functions

**Input**: `KERNEL_REGISTRY.json`
**Output**: `op_builders_auto.py`

**What it extracts**:
```python
{
  "op_type": {
    "inputs": ["q", "k", "v"],           # From kernel.inputs[]
    "outputs": ["out"],                   # From kernel.outputs[]
    "scratch": ["scores"],                # From kernel.scratch[]
    "weights": ["wq", "wk", "wv"],       # Inferred from inputs (weight tensors)
    "params": ["num_heads", "head_dim"]  # Inferred from dims (H→num_heads, D→head_dim)
  }
}
```

**Generation logic**:
- Extract inputs/outputs/scratch from kernel spec
- Infer weights from input tensors (2D/3D shapes, weight naming)
- Infer params from dimension names (H→num_heads, D→head_dim, E→embed_dim)
- Generate Python function with proper tensor naming
- Create registry mapping op_type → builder function

### 2. op_builders_auto.py

**Auto-generated file** - DO NOT EDIT MANUALLY!

Contains:
- OpContext class (state tracking)
- Tensor naming helpers (make_tensor_name, make_weight_name)
- Auto-generated builder functions for all kernel ops
- OP_BUILDERS registry

**Example generated function**:
```python
def build_attention_op(op_node: OpNode, ctx: OpContext, manifest: Dict) -> Op:
    """Build attention op (auto-generated from kernel registry)."""
    input_name = ctx.prev_output or make_tensor_name(op_node, 'input')
    k_name = make_tensor_name(op_node, 'k')
    v_name = make_tensor_name(op_node, 'v')
    output_name = make_tensor_name(op_node, 'out')

    op = Op(
        op="attention",
        name=make_tensor_name(op_node, "attention"),
        inputs=[input_name, k_name, v_name],
        outputs=[output_name],
        weights=[],
        scratch=['scores'],
        params={
            'num_heads': ctx.config.get('num_heads', 0),
            'head_dim': ctx.config.get('head_dim', 0),
        }
    )
    ctx.set_output(output_name)
    return op
```

### 3. op_builders_hybrid_v6_6.py

**Hybrid approach**: Auto + Manual builders

**Contains**:
1. **Template → Kernel mapping**:
   ```python
   TEMPLATE_TO_KERNEL_MAP = {
       "qkv_proj": "qkv_projection",  # Template op → Kernel op
       "out_proj": "gemv",            # Multiple template ops can map to same kernel
       "mlp_down": "gemv",
       ...
   }
   ```

2. **Manual builders** for special ops:
   - `tokenizer`: Metadata only, no kernel
   - `weight_tying`: Metadata only, no kernel

3. **Context-aware wrappers**:
   - `build_rmsnorm_op_wrapper`: Tracks rmsnorm counter (ln1, ln2)
   - `build_residual_add_op_wrapper`: Tracks residual counter, saves residual input

4. **Unified registry**:
   ```python
   OP_BUILDERS = AUTO_BUILDERS.copy()  # Start with auto-generated
   OP_BUILDERS.update({                # Override with manual/wrappers
       "tokenizer": build_tokenizer_op,
       "rmsnorm": build_rmsnorm_op_wrapper,
       ...
   })
   ```

## Benefits

### 1. **Guaranteed Kernel Availability**

Auto-generation ensures every op builder corresponds to a real kernel:
- ❌ **Before**: Op builder might reference non-existent kernel → runtime crash
- ✅ **After**: Op builder auto-generated from kernel registry → guaranteed match

### 2. **Zero Maintenance**

When kernels change:
- ❌ **Before**: Manually update op_builders_v6_6.py
- ✅ **After**: Re-run `gen_op_builders_from_registry.py` → done!

### 3. **Early Error Detection**

Missing kernels caught at generation time:
```bash
$ python gen_op_builders_from_registry.py
✓ Generated 14 op builders

# If template uses unsupported op:
$ python build_ir_v6_6.py --test
❌ ERROR: Missing kernels for required operations!
  - new_fancy_op (needed 2x)
```

### 4. **Documentation**

Coverage report shows mapping:
```bash
$ python op_builders_hybrid_v6_6.py
Template Op → Kernel Op Mapping:
  ✓ (auto) attn → attention
  ✓ (auto) qkv_proj → qkv_projection
  ✓ (manual) tokenizer → tokenizer
  ...
```

## Workflow

### Adding a New Kernel

1. Add kernel to kernel maps → regenerate `KERNEL_REGISTRY.json`
2. Re-run auto-generation:
   ```bash
   python gen_op_builders_from_registry.py
   ```
3. Update template mapping in `op_builders_hybrid_v6_6.py`:
   ```python
   TEMPLATE_TO_KERNEL_MAP = {
       ...
       "new_template_op": "new_kernel_op",
   }
   ```
4. Done! Op builder automatically generated.

### Adding a New Template Op

1. Add op to template (e.g., `qwen2.json`)
2. Map to kernel op in `TEMPLATE_TO_KERNEL_MAP`:
   ```python
   "new_template_op": "existing_kernel_op"  # Reuse existing kernel
   ```
3. If special behavior needed, add wrapper:
   ```python
   def build_new_op_wrapper(...):
       # Custom logic (counters, state tracking, etc.)
       ...

   OP_BUILDERS["new_kernel_op"] = build_new_op_wrapper
   ```

### Updating Kernels

1. Modify kernel in kernel maps
2. Regenerate registry:
   ```bash
   python gen_kernel_registry_from_maps.py
   ```
3. Regenerate op builders:
   ```bash
   python gen_op_builders_from_registry.py
   ```
4. Op builders automatically updated!

## Files

| File | Type | Purpose |
|------|------|---------|
| `KERNEL_REGISTRY.json` | Data | Source of truth for kernels |
| `gen_op_builders_from_registry.py` | Tool | Auto-generator |
| `op_builders_auto.py` | Generated | Auto-generated builders (DO NOT EDIT) |
| `op_builders_hybrid_v6_6.py` | Code | Hybrid (auto + manual) |
| `build_ir_v6_6.py` | User | Uses hybrid builders |

## Edge Cases

### Metadata-Only Ops

Ops like `tokenizer` and `weight_tying` don't have kernels:
- Not in kernel registry
- Manually implemented in hybrid file
- Marked as metadata-only in mapping

### Multi-Use Kernels

Single kernel used by multiple template ops:
```python
TEMPLATE_TO_KERNEL_MAP = {
    "out_proj": "gemv",   # All use same gemv kernel
    "mlp_down": "gemv",
    "logits": "gemv",
}
```

### Context-Aware Ops

Ops that need state tracking (counters, saved tensors):
- Auto-generated builder → basic functionality
- Wrapper function → adds state tracking
- Wrapper overrides auto-generated in OP_BUILDERS

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

## Future Improvements

### 1. Template Validation Against Registry

Validate templates at load time:
```python
def validate_template(template, registry):
    for op in template.ops:
        if op not in TEMPLATE_TO_KERNEL_MAP:
            raise ValueError(f"Unknown template op: {op}")
        kernel_op = TEMPLATE_TO_KERNEL_MAP[op]
        if kernel_op not in registry:
            raise ValueError(f"No kernel for {op} → {kernel_op}")
```

### 2. Weight Inference from Manifest

Auto-detect weight names from manifest instead of hardcoding:
```python
def infer_weights_from_manifest(op_node, manifest):
    layer_weights = manifest["weights"][op_node.layer_index]
    return [w for w in layer_weights if matches_op(w, op_node.op_id)]
```

### 3. Dim Inference from Config

Auto-infer dimensions from config instead of hardcoding:
```python
# Instead of:
params={"num_heads": 8}

# Infer:
params=infer_params_from_config(kernel.dims, config)
```

### 4. Kernel Selection at IR Build Time

Choose kernel variant based on quantization:
```python
# Instead of: op="gemv" (generic)
# Choose:     op="gemv_q8_0" (quantization-specific)

kernel_variant = select_kernel_variant(
    op_type="gemv",
    weight_dtype=manifest.weights[weight_name].dtype
)
```

## Summary

Auto-generation solves the fundamental problem of keeping op builders in sync with kernels:

- **Before**: Manual sync, error-prone, runtime failures
- **After**: Auto-generated, guaranteed match, early error detection

The hybrid approach provides flexibility for special cases while maintaining automation for the common case.
