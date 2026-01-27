# IR1 Architecture: Direct Template → Kernel Mapping

## Current State (2026-01-23)

**ACTIVE**: `build_ir_v6_6.py` (424 lines)
**REDUNDANT**: `ir_types_v6_6.py`, `op_builders_*.py` (old approach with intermediate abstractions)

## What is IR1?

IR1 is a simple list of C function names (kernel IDs) in execution order.

**Example**:
```json
{
  "format": "ir1-kernel-sequence",
  "version": 1,
  "mode": "decode",
  "kernels": [
    "rmsnorm_forward",
    "gemv_q5_0_q8_0",
    "gemv_q5_0_q8_0",
    "gemv_q8_0_q8_0",
    "rope_forward_qk",
    "attention_forward_causal_head_major_gqa_flash_strided",
    ...
  ]
}
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   WEIGHTS MANIFEST                          │
│  Contains:                                                   │
│    - template (v1 or v2 - auto-converted)                   │
│    - quant_summary (per-layer weight quantization)          │
│    - config (model parameters)                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ build_ir_v6_6.py
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                   VALIDATION PHASE                           │
│                                                              │
│  [1/2] Template ops → Kernel ops mapping check              │
│        Source: TEMPLATE_TO_KERNEL_OP dict                   │
│        HARD FAULT if unmapped ops found                     │
│                                                              │
│  [2/2] Kernel availability check                            │
│        Source: KERNEL_REGISTRY.json                         │
│        HARD FAULT if required kernels missing               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ All checks passed
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                  IR1 GENERATION PHASE                        │
│                                                              │
│  For each layer (0 to num_layers-1):                        │
│    For each template op in sequence:                        │
│      1. Map template op → kernel op (TEMPLATE_TO_KERNEL_OP) │
│      2. Read weight quantization from quant_summary          │
│      3. Find matching kernel from KERNEL_REGISTRY.json      │
│      4. HARD FAULT if kernel not found                      │
│      5. Append kernel ID to list                            │
│                                                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                        IR1 OUTPUT                            │
│  Simple list of C function names (kernel IDs)               │
└─────────────────────────────────────────────────────────────┘
```

## Two-Level Mapping System

### Level 1: Template Op → Kernel Op

**Source**: `TEMPLATE_TO_KERNEL_OP` dict in `build_ir_v6_6.py`

```python
TEMPLATE_TO_KERNEL_OP = {
    # Attention block
    "rmsnorm": "rmsnorm",
    "qkv_proj": "qkv_projection",  # Or fallback to 3x gemv
    "rope_qk": "rope",
    "attn": "attention",
    "out_proj": "gemv",
    "residual_add": "residual_add",

    # MLP block
    "mlp_gate_up": "fused_mlp_block",  # Or fallback to 2x gemv
    "silu_mul": "swiglu",
    "mlp_down": "gemv",
}
```

**Key insight**: Multiple template ops can map to the same kernel op!
- `out_proj`, `mlp_down` → both use `gemv`

### Level 2: Kernel Op + Quantization → Kernel Function

**Source**: `KERNEL_REGISTRY.json` (auto-generated from kernel maps)

```json
{
  "kernels": [
    {
      "op": "gemv",
      "id": "gemv_q5_0_q8_0",    ← C function name
      "quant": {"weight": "q5_0"}
    },
    {
      "op": "gemv",
      "id": "gemv_q8_0_q8_0",    ← Different implementation
      "quant": {"weight": "q8_0"}
    }
  ]
}
```

**Selection logic**:
1. Template op `out_proj` → kernel op `gemv`
2. Read `quant_summary["layer.0"]["wo"]` → `"q5_0"`
3. Find kernel: `op="gemv"` AND `quant.weight="q5_0"` → `"gemv_q5_0_q8_0"`

## HARD FAULT Philosophy

**No silent failures!** If anything is missing, we stop immediately with clear error messages.

### HARD FAULT 1: Unmapped Template Op

```
❌ HARD FAULT: Template ops have no kernel mapping!
  - new_fancy_op

Action required:
  Add mappings to TEMPLATE_TO_KERNEL_OP in build_ir_v6_6.py
```

### HARD FAULT 2: Missing Kernel in Registry

```
❌ HARD FAULT: Required kernels not found in registry!
  - qkv_projection
    (needed by: qkv_proj)

Action required:
  1. Implement missing kernels
  2. Add to kernel maps and regenerate KERNEL_REGISTRY.json
```

### HARD FAULT 3: Kernel Not Found During Generation

```
RuntimeError: HARD FAULT: No kernel found for out_proj (op=gemv, quant=q3_k)
```

**Why this matters**: Downstream code generation, memory planning, and execution will all fail if kernels are missing. Better to fail early with clear messages.

## Template Format

### Template V2 (Preferred)

```json
{
  "sequence": ["decoder"],
  "block_types": {
    "decoder": {
      "body": {
        "ops": ["rmsnorm", "qkv_proj", "rope_qk", "attn", ...]
      }
    }
  }
}
```

### Template V1 (Auto-converted)

```json
{
  "block_types": {
    "dense": {
      "ops": ["rmsnorm", "qkv_proj", "rope_qk", "attn", ...]
    }
  }
}
```

**Auto-conversion**: If v2 not found, v1 is converted on-the-fly:
```python
template["sequence"] = ["decoder"]
template["block_types"]["decoder"] = {
    "body": {"ops": v1_block["ops"]}
}
```

## Quantization Handling

### Ops with FP32 Weights (Learnable but Not Quantized)

```python
FP32_WEIGHT_OPS = {"rmsnorm"}
```

RMSNorm has learnable scale parameters but they're always fp32.

### Ops with Quantized Weights

```python
OP_WEIGHT_MAP = {
    "qkv_proj": ["wq", "wk", "wv"],  # 3 weights
    "out_proj": ["wo"],
    "mlp_gate_up": ["w1"],
    "mlp_down": ["w2"],
}
```

Read quantization from `quant_summary`:
```python
layer_quant = quant_summary["layer.0"]
wo_quant = layer_quant["wo"]  # e.g., "q5_0"
```

### Ops Without Model Weights

```python
# rope, attention, swiglu, residual_add
quant = {"weight": "none"}
```

These are compute-only ops (no weight tensors).

## Fused Kernels with Fallback

Some template ops prefer fused kernels but can fall back to multiple calls:

### QKV Projection

**Preferred**: Fused `qkv_projection` kernel
**Fallback**: 3 separate `gemv` calls (one for wq, wk, wv)

```python
kernel_id = find_kernel(registry, op="qkv_projection", quant={"weight": wq_quant})

if kernel_id:
    # Fused kernel found
    arranged_kernels.append(kernel_id)
else:
    # Fallback: 3 gemv calls
    kernel_q = find_kernel(registry, op="gemv", quant={"weight": wq_quant})
    kernel_k = find_kernel(registry, op="gemv", quant={"weight": wk_quant})
    kernel_v = find_kernel(registry, op="gemv", quant={"weight": wv_quant})
    arranged_kernels.extend([kernel_q, kernel_k, kernel_v])
```

### MLP Gate+Up

**Preferred**: Fused `fused_mlp_block` kernel
**Fallback**: 2 separate `gemv` calls (gate + up)

## Data-Driven Kernel Selection

**No hardcoded if/elif chains!**

The kernel selection logic is completely data-driven using:
1. `TEMPLATE_TO_KERNEL_OP` - template op → kernel op mapping
2. `OP_WEIGHT_MAP` - template op → weight names mapping
3. `FP32_WEIGHT_OPS` - ops with fp32 weights
4. `quant_summary` - actual weight quantization per layer

**Example flow**:
```python
template_op = "mlp_down"

# Step 1: Map to kernel op
kernel_op = TEMPLATE_TO_KERNEL_OP[template_op]  # "gemv"

# Step 2: Determine quantization
if template_op in OP_WEIGHT_MAP:
    weight_name = OP_WEIGHT_MAP[template_op][0]  # "w2"
    weight_quant = layer_quant[weight_name]       # "q6_k"

# Step 3: Find kernel
kernel_id = find_kernel(
    registry,
    op=kernel_op,
    quant={"weight": weight_quant},
    mode="decode"
)
# Result: "gemv_q6_k_q8_k"
```

## Example Output

For Qwen2-0.5B (24 layers):
- **312 kernel calls** total
- **10 unique kernels**:
  - `rmsnorm_forward`
  - `gemv_q5_0_q8_0`, `gemv_q8_0_q8_0`, `gemv_q4_k_q8_k`, `gemv_q6_k_q8_k`
  - `rope_forward_qk`
  - `attention_forward_causal_head_major_gqa_flash_strided`
  - `ck_residual_add_token_major`
  - `mega_fused_outproj_mlp_prefill`
  - `swiglu_forward`

## Key Benefits

1. **Simple**: Just a list of function names - no complex IR types
2. **Direct**: Template + Quant → Kernels (no intermediate abstractions)
3. **Safe**: HARD FAULT on any missing kernel
4. **Compact**: 424 lines vs 1000+ with old approach
5. **Maintainable**: Single source of truth for mappings
6. **Data-driven**: No hardcoded if/elif chains

## What's Next?

IR1 is the first stage. Next stages:

1. **IR2**: Add tensor metadata (shapes, memory layout)
2. **Memory Planning**: Allocate buffers, plan reuse
3. **Code Generation**: Generate C code that calls these kernels
4. **Execution**: Run the generated code

But IR1 is complete and working!
