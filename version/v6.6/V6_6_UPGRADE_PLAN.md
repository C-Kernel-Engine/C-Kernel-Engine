# V6.6 Upgrade Plan: Achieving Numerical Parity with V6.5

## Executive Summary

After detailed analysis, the divergence between V6.5 and V6.6 is caused by **different kernel paths** for the same mathematical operations. The weights and biases are IDENTICAL between versions.

### Key Findings

| Aspect | V6.5 | V6.6 | Impact |
|--------|------|------|--------|
| QKV kernel | `gemv_q5_0_q8_0` (Q8_0 input) | `gemv_q5_0` (FP32 input) | Different |
| Input quantization | `quantize_row_q8_0` before GEMV | None (uses FP32 directly) | Different |
| Weight files | 514 MB | 397 MB | Same data, different format |
| QKV weights | Identical bytes | Identical bytes | No diff |
| QKV biases | Identical values | Identical values | No diff |
| Extra biases (bo,b1,b2) | Not present | Present but all zeros | No impact |

### Root Cause

V6.5 decode uses this path:
```c
// 1. Quantize activation to Q8_0
quantize_row_q8_0(ln1_out, ln1_q8, aligned_embed_dim);

// 2. Use Q8_0 input kernel
gemv_q5_0_q8_0(q_token, WQ, ln1_q8, H * head_dim, aligned_embed_dim);
```

V6.6 decode uses this path:
```c
// 1. Use FP32 input kernel directly
gemv_q5_0(output, WQ, input, 896, 896);
```

Despite FP32 being higher precision, the different computation paths (different dequantization order, different accumulation) produce different results.

---

## Clarifications Requested by User

### GEMM vs GEMV

- **GEMV (GEneral Matrix-Vector)**: M=1 (single vector), used for decode (one token at a time)
- **GEMM (GEneral Matrix-Matrix)**: M>1 (batch of vectors), used for prefill (multiple tokens)

Both V6.5 and V6.6 correctly use:
- GEMV for decode (M=1)
- GEMM for prefill (M>1)

The naming is slightly confusing because even "gemv" kernels can handle M>1 in some implementations, but the principle is:
- **Decode** = single token = vector operations = lower arithmetic intensity
- **Prefill** = multiple tokens = matrix operations = higher arithmetic intensity

### Why FP32 Activation Isn't the Bug

You correctly noted that FP32 (higher precision) cannot produce worse results than Q8_0 (lower precision) in a mathematical sense. The divergence is not about "worse" results - it's about **different** results due to:

1. **Different dequantization order**: Q5_0 × Q8_0 dequantizes both before multiplication, Q5_0 × FP32 only dequantizes weights
2. **Different accumulation precision**: The intermediate sums happen at different points
3. **Accumulated differences over 24 layers**: Small numerical differences compound

Neither is "wrong" - they're just different. The question is which matches PyTorch/HuggingFace reference better.

---

## Confirmed Bug #1: Hardcoded q8_0 for Logits

**Location**: `version/v6.6/scripts/build_ir_v6_6.py:625-628`

```python
if isinstance(weight_info, list) and not weight_info:
    # These use fixed quant (typically q8_0)
    kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": "q8_0"}, mode=mode)
    return [kernel_id] if kernel_id else []
```

**Problem**: Hardcodes `q8_0` for header/footer ops regardless of actual weight dtype.

**Fix**: Look up actual dtype from manifest:
```python
if isinstance(weight_info, list) and not weight_info:
    if op == "logits":
        if manifest.get("weight_tying", True):
            weight_dtype = quant_summary.get("token_emb_dtype", "q8_0")
        else:
            weight_dtype = quant_summary.get("lm_head_dtype", "q8_0")
    elif op == "dense_embedding_lookup":
        weight_dtype = quant_summary.get("token_emb_dtype", "q8_0")
    else:
        weight_dtype = "q8_0"
    kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": weight_dtype}, mode=mode)
```

---

## Upgrade Strategy Options

### Option A: Match V6.5 Kernel Selection (Recommended)

Make V6.6 use the same kernels as V6.5 to achieve byte-for-byte parity:

#### Changes Required

1. **Add input quantization ops to IR**
   - Modify template or IR builder to insert `quantize_q8_0` ops before projections
   - This matches V6.5's `quantize_row_q8_0()` call

2. **Update kernel selection to prefer Q8_0 input kernels**
   - Modify `find_kernel()` in `build_ir_v6_6.py` to select Q8_0 activation kernels when input is quantized
   - Change the priority from FP32-first to Q8_0-matching

3. **Update codegen to emit quantization calls**
   - Modify `codegen_v6_6.py` to generate `quantize_row_q8_0()` calls before GEMV

#### Implementation Steps

```
Phase 1: Fix Bug #1 (Logits kernel)
├── Modify convert_hf_to_bump_v6_6.py: Add token_emb_dtype to quant_summary
├── Modify build_ir_v6_6.py:625-628: Use actual dtype
└── Test: Verify logits kernel matches weight dtype

Phase 2: Add Input Quantization
├── Option 2A: Implicit (kernel selection)
│   └── Modify find_kernel() to prefer Q8_0 input kernels
│       └── Add flag: use_q8_activation = True
│       └── Priority: Q8_0 input > FP32 input when flag is set
│
└── Option 2B: Explicit (IR ops)
    └── Modify template/IR to add quantize_input ops
        └── rmsnorm → quantize_q8_0 → q_proj
        └── Emit quantize_row_q8_0() in codegen

Phase 3: Validation
├── Run V6.5 vs V6.6 comparison
├── Verify argmax matches
└── Verify logits within tolerance
```

### Option B: Validate Against PyTorch Reference

Before making changes, determine which version is more accurate:

```bash
cd /home/antshiv/Workspace/C-Kernel-Engine
python version/v6.6/test/validate_against_pytorch.py
```

If V6.6 (FP32 path) is closer to PyTorch reference:
- Keep V6.6's kernel selection
- Accept that V6.5 has accumulated quantization error
- Document the expected difference

If V6.5 (Q8_0 path) is closer to PyTorch reference:
- Implement Option A
- The Q8_0 quantization acts as regularization similar to training

### Option C: Make It Configurable

Add a flag to control kernel selection:

```json
// In template config
{
  "decode_activation_quant": "q8_0",  // or "fp32"
  "prefill_activation_quant": "q8_0"  // or "fp32"
}
```

This allows users to choose their preferred compute path.

---

## Detailed Implementation Plan (Option A)

### Step 1: Fix Logits Kernel Selection

**File**: `version/v6.6/scripts/convert_hf_to_bump_v6_6.py`

Add around line 537:
```python
quant_summary["token_emb_dtype"] = "q8_0"  # From GGUF metadata
quant_summary["weight_tying"] = tie
if not tie:
    quant_summary["lm_head_dtype"] = lm_head_quant  # Actual quant type
```

**File**: `version/v6.6/scripts/build_ir_v6_6.py`

Replace lines 625-628:
```python
if isinstance(weight_info, list) and not weight_info:
    # Look up actual dtype from quant_summary
    weight_dtype = _get_header_footer_dtype(op, quant_summary, manifest)
    kernel_id = find_kernel(registry, op=kernel_op, quant={"weight": weight_dtype}, mode=mode)
    return [kernel_id] if kernel_id else []

def _get_header_footer_dtype(op: str, quant_summary: dict, manifest: dict) -> str:
    if op == "logits":
        if manifest.get("weight_tying", True):
            return quant_summary.get("token_emb_dtype", "q8_0")
        return quant_summary.get("lm_head_dtype", "q8_0")
    if op == "dense_embedding_lookup":
        return quant_summary.get("token_emb_dtype", "q8_0")
    return "q8_0"
```

### Step 2: Add Input Quantization Support

**File**: `version/v6.6/scripts/build_ir_v6_6.py`

Add to quant_summary structure:
```python
# In process_layer or equivalent
layer_quant["activation_quant"] = "q8_0"  # Default for V6.5 compatibility
```

Modify `find_kernel()`:
```python
def find_kernel(
    registry: Dict,
    op: str,
    quant: Dict[str, str],  # {"weight": "q5_0", "activation": "q8_0"}
    mode: str = "decode",
    prefer_activation_quant: bool = True  # New parameter
) -> Optional[str]:
    candidates = [k for k in registry if k.get("op") == op and k.get("mode", "both") in (mode, "both")]

    # Filter by weight dtype
    weight_dtype = quant.get("weight")
    if weight_dtype:
        candidates = [k for k in candidates if k.get("quant", {}).get("weight") == weight_dtype]

    # Prioritize by activation dtype
    activation_dtype = quant.get("activation", "fp32")

    def priority(k):
        k_act = k.get("quant", {}).get("activation", "fp32")
        if prefer_activation_quant and k_act == activation_dtype:
            return 0  # Exact match
        if k_act == "fp32":
            return 1  # FP32 fallback
        return 2  # Other

    candidates.sort(key=priority)
    return candidates[0]["id"] if candidates else None
```

### Step 3: Update Codegen for Quantization

**File**: `version/v6.6/scripts/codegen_v6_6.py`

Add quantization call generation:
```python
def emit_quantize_op(ctx, input_buf, output_buf, size):
    return f"quantize_row_q8_0({input_buf}, {output_buf}, {size});"
```

Insert before projection ops when activation_quant == "q8_0":
```python
if layer_config.get("activation_quant") == "q8_0":
    add_line(emit_quantize_op(ctx, "layer_input", "ln1_q8", embed_dim))
    # Use ln1_q8 as input to projections
```

### Step 4: Validation

Create test script:
```python
#!/usr/bin/env python3
"""Validate V6.6 matches V6.5 after upgrade."""

def test_parity():
    # Run both versions
    v65_logits = run_v65(token=100)
    v66_logits = run_v66(token=100)

    # Check argmax
    assert np.argmax(v65_logits) == np.argmax(v66_logits), "Argmax mismatch"

    # Check tolerance
    max_diff = np.max(np.abs(v65_logits - v66_logits))
    assert max_diff < 0.01, f"Max diff too large: {max_diff}"

    print("PASS: V6.6 matches V6.5")
```

---

## Files to Modify

| File | Priority | Changes |
|------|----------|---------|
| `build_ir_v6_6.py` | HIGH | Fix logits kernel selection, add activation quant handling |
| `convert_hf_to_bump_v6_6.py` | HIGH | Add token_emb_dtype, weight_tying to quant_summary |
| `codegen_v6_6.py` | MEDIUM | Emit quantize_row_q8_0 calls if needed |
| `ir_types_v6_6.py` | LOW | Add activation_quant field to LayerQuant |
| `templates/qwen2.json` | LOW | Add activation_quant config option |
| `kernel_maps/KERNEL_REGISTRY.json` | LOW | Already has both FP32 and Q8_0 variants |

---

## Testing Strategy

1. **Unit Test**: Kernel selection returns correct kernel for weight+activation dtype combo
2. **Integration Test**: Generated code matches V6.5 kernel calls
3. **Numerical Test**: Logits match between V6.5 and V6.6 within tolerance (<0.01 max diff)
4. **Reference Test**: Compare both against PyTorch to determine ground truth
5. **Regression Test**: Ensure no performance degradation

---

## Timeline Estimate

- **Phase 1 (Bug #1 fix)**: Straightforward code change
- **Phase 2 (Input quantization)**: Requires more extensive changes to IR and codegen
- **Phase 3 (Validation)**: Depends on test infrastructure

---

## Recommendation

Start with **Option B (PyTorch validation)** to establish ground truth, then implement **Option A** if V6.5's Q8_0 path is closer to reference.

If V6.6's FP32 path is closer to reference, consider documenting the expected divergence and marking V6.5's behavior as "legacy quantized path" rather than changing V6.6.
