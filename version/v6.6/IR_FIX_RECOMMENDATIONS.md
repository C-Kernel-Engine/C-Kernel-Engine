# V6.6 IR Pipeline Fix Recommendations

## Executive Summary

The divergence between v6.5 and v6.6 is caused by **3 specific issues** in the IR pipeline:

1. **Bug #1 (CRITICAL)**: Hardcoded `q8_0` for logits/lm_head kernel selection
2. **Bug #2 (HIGH)**: No input quantization specification - always assumes FP32 activations
3. **Bug #3 (MEDIUM)**: No per-head layout option - always uses full-matrix projection

---

## Bug #1: Hardcoded Logits Quantization

### Location
**File:** `version/v6.6/scripts/build_ir_v6_6.py`
**Lines:** 625-628

### Current Code
```python
if isinstance(weight_info, list) and not weight_info:
    # These use fixed quant (typically q8_0)
    kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": "q8_0"}, mode=mode)
    return [kernel_id] if kernel_id else []
```

### Problem
- Hardcodes `"q8_0"` for all header/footer ops (embedding, logits)
- Ignores actual weight dtype from manifest/quant_summary
- For weight-tied models, token_emb IS q8_0, so this works accidentally
- For non-weight-tied models, lm_head could be q4_k or fp32

### Fix
```python
if isinstance(weight_info, list) and not weight_info:
    # Look up actual weight dtype from manifest/quant_summary
    if op == "logits":
        if manifest.get("weight_tying", True):
            weight_dtype = quant_summary.get("token_emb_dtype", "q8_0")
        else:
            weight_dtype = quant_summary.get("lm_head_dtype", "q8_0")
    elif op == "dense_embedding_lookup":
        weight_dtype = quant_summary.get("token_emb_dtype", "q8_0")
    else:
        weight_dtype = "q8_0"  # fallback
    kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": weight_dtype}, mode=mode)
    return [kernel_id] if kernel_id else []
```

### Required Changes Elsewhere

**File:** `version/v6.6/scripts/convert_hf_to_bump_v6_6.py`

Add to quant_summary output (around line 537):
```python
quant_summary["token_emb_dtype"] = "q8_0"  # or actual dtype
quant_summary["lm_head_dtype"] = lm_head_dtype  # if not weight_tying
quant_summary["weight_tying"] = tie
```

---

## Bug #2: No Input Quantization Specification

### Location
**File:** `version/v6.6/scripts/build_ir_v6_6.py`
**Function:** `find_kernel()` (lines 285-357)
**Function:** `map_op_to_kernel()` (lines 561-637)

### Current Behavior
```python
def activation_priority(k):
    act = k.get("quant", {}).get("activation", "fp32")
    if act == "fp32":
        return 0  # Prefer fp32 activation
    return 1  # quantized last
```

This **always prefers FP32 activation kernels**, which means:
- `gemv_q5_0` (FP32 input) is selected over `gemv_q5_0_q8_0` (Q8_0 input)
- No quantization of input before GEMV
- Different compute path than v6.5

### V6.5 Behavior (Reference)
V6.5 **explicitly quantizes** rmsnorm output to Q8_0 before projection:
```c
// V6.5 code
quantize_row_q8_0(ln1_out, ln1_q8, aligned_embed_dim);
gemm_nt_q5_0_q8_0(ln1_q8, wq_h, bq_h, q_h, ...);
```

### Fix Option A: Add Input Quantization to IR1

**Step 1:** Extend quant_summary structure in converter:
```python
quant_summary = {
    "layer.0": {
        "wq": "q5_0",
        "wk": "q5_0",
        "wv": "q8_0",
        # NEW: input quantization for each projection
        "input_q": "q8_0",   # Q projection input
        "input_k": "q8_0",   # K projection input
        "input_v": "q8_0",   # V projection input
        "input_o": "q8_0",   # O projection input
        "input_mlp": "q8_0", # MLP input
    },
    ...
}
```

**Step 2:** Modify find_kernel() to accept input dtype:
```python
def find_kernel(
    registry: Dict,
    op: str,
    quant: Dict[str, str],  # {"weight": "q5_0", "activation": "q8_0"}
    mode: str = "decode"
) -> Optional[str]:
    # Filter by BOTH weight AND activation dtype
    ...
```

**Step 3:** Modify map_op_to_kernel() to pass input dtype:
```python
# In map_op_to_kernel()
weight_dtype = layer_quant[weight_key]
input_dtype = layer_quant.get(f"input_{weight_key[1:]}", "fp32")  # e.g., input_q for wq
kernel_id = find_kernel(
    registry,
    op=kernel_op,
    quant={"weight": weight_dtype, "activation": input_dtype},
    mode=mode
)
```

### Fix Option B: Add Explicit Quantize Ops to IR

Instead of changing kernel selection, add explicit `quantize_row_q8_0` ops to the IR:

**In template (`qwen2.json`):**
```json
{
  "sequence": [
    "rmsnorm",
    "quantize_input",    // NEW: explicit quantization op
    "q_proj",
    "k_proj",
    "v_proj",
    ...
  ]
}
```

**In build_ir_v6_6.py:**
```python
TEMPLATE_TO_KERNEL_OP = {
    ...
    "quantize_input": "quantize_q8_0",  # Maps to quantize_row_q8_0 kernel
    ...
}
```

---

## Bug #3: No Per-Head Layout Option

### Location
**File:** `version/v6.6/scripts/build_ir_v6_6.py`
**Lines:** 163-192 (TEMPLATE_TO_KERNEL_OP)

### Current Code
```python
TEMPLATE_TO_KERNEL_OP = {
    "qkv_proj": "qkv_projection",    # Has special handling for prefill
    "out_proj": "matmul",            # Always full-matrix
    "mlp_gate_up": "matmul",         # Always full-matrix
    ...
}
```

### V6.5 Behavior (Reference)
V6.5 uses **per-head projection** with head-major layout:
```c
// V6.5 code - processes each head separately
for (int h = 0; h < H; h++) {
    gemm_nt_q5_0_q8_0(ln1_q8, wq_h, bq_h, q_h, ...);  // per-head
}
```

### Fix Option A: Template-Driven Layout Selection

**Step 1:** Add layout flag to template:
```json
{
  "model": "qwen2",
  "config": {
    "use_head_major_qkv": true,
    "use_head_major_outproj": true
  },
  ...
}
```

**Step 2:** Modify map_op_to_kernel() to check layout:
```python
if op == "q_proj":
    if template_config.get("use_head_major_qkv", False):
        kernel_op = "qkv_projection_head_major"
    else:
        kernel_op = "matmul"
```

**Step 3:** Add head-major kernels to registry:
```json
{
  "id": "gemm_nt_q5_0_q8_0_head_major",
  "op": "qkv_projection_head_major",
  "quant": {"weight": "q5_0", "activation": "q8_0"},
  ...
}
```

### Fix Option B: Fused QKV Kernel with Head-Major Output

Already partially implemented for prefill mode (lines 582-589):
```python
if op == "qkv_proj" and mode == "prefill":
    kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": "mixed"}, mode=mode)
```

Extend this to decode mode for head-major layout.

---

## Data Flow Diagram: Current vs Fixed

### Current (Wrong)
```
rmsnorm_output (FP32)
       │
       └──> gemv_q5_0 (FP32 input)  ← WRONG KERNEL
              │
              └──> Q projection (FP32)
```

### Fixed (Correct)
```
rmsnorm_output (FP32)
       │
       └──> quantize_row_q8_0
              │
              └──> ln1_q8 (Q8_0)
                      │
                      └──> gemm_nt_q5_0_q8_0 (Q8_0 input)  ← CORRECT KERNEL
                              │
                              └──> Q projection (FP32)
```

---

## Implementation Order

### Phase 1: Fix Logits Kernel Selection (Bug #1)
1. Modify `convert_hf_to_bump_v6_6.py` to add token_emb/lm_head dtype to quant_summary
2. Modify `build_ir_v6_6.py:625-628` to use actual dtype instead of hardcoded q8_0
3. Test with weight-tied and non-weight-tied models

### Phase 2: Add Input Quantization (Bug #2)
1. Add input quantization to quant_summary structure
2. Modify `find_kernel()` to match both weight AND activation dtype
3. Add quantize_row_q8_0 ops to IR if needed
4. Test numerics against v6.5 and PyTorch

### Phase 3: Add Head-Major Layout Option (Bug #3)
1. Add layout flags to template
2. Add head-major kernels to registry
3. Modify kernel selection to respect layout flags
4. Test performance and correctness

---

## Files to Modify

| File | Changes |
|------|---------|
| `convert_hf_to_bump_v6_6.py` | Add token_emb_dtype, lm_head_dtype, weight_tying to quant_summary |
| `build_ir_v6_6.py` | Fix lines 625-628, modify find_kernel(), map_op_to_kernel() |
| `ir_types_v6_6.py` | Extend QuickQuant, LayerQuant types if needed |
| `templates/qwen2.json` | Add layout configuration flags |
| `kernel_maps/KERNEL_REGISTRY.json` | Add head-major kernel variants |
| `codegen_v6_6.py` | May need updates for quantize ops |

---

## Testing Strategy

1. **Unit Test**: Kernel selection returns correct kernel for each weight+input dtype combo
2. **Integration Test**: Generated code matches v6.5 kernel calls
3. **Numerical Test**: Logits match between v6.5 and v6.6 within tolerance
4. **Reference Test**: Both match PyTorch/HuggingFace reference
