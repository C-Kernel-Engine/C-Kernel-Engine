# Fusion Status - v6.6 Pipeline

## Current Status

✅ **Fusion infrastructure implemented and working**
❌ **Prefill fusion patterns need updating in kernel registry**

## How Fusion Works

The v6.6 pipeline implements **registry-driven fusion**:

1. Scan `KERNEL_REGISTRY.json` for kernels with a `"fuses"` field
2. Match consecutive kernel sequences in IR1
3. Replace matching sequences with fused kernels
4. Track statistics (fusions applied, kernels removed)

Example from registry:
```json
{
  "id": "mega_fused_attention_prefill",
  "op": "fused_attention_block",
  "fuses": [
    "rmsnorm_forward",
    "gemv_q5_0_q8_0",    // ← Problem: uses GEMV (decode-specific)
    "gemv_q5_0_q8_0",
    "gemv_q8_0_q8_0",
    "rope_forward_qk",
    "attention_forward_causal_head_major_gqa_flash_strided",
    "gemv_q5_0_q8_0",
    "ck_residual_add_token_major"
  ]
}
```

## Current Results

### Decode Mode ✅

**Fusion Working!**

```
IR1:    312 kernels
Fusion: 12 fusions applied, 84 kernels removed
Final:  228 kernels (27% reduction)
```

Example fusion:
```
Replacing: rmsnorm_forward + gemv_q5_0_q8_0 + gemv_q5_0_q8_0 + gemv_q8_0_q8_0 +
           rope_forward_qk + attention_forward_causal_head_major_gqa_flash_strided +
           gemv_q5_0_q8_0 + ck_residual_add_token_major
With:      mega_fused_attention_prefill
```

### Prefill Mode ❌

**Fusion Not Working - Pattern Mismatch**

```
IR1:    312 kernels
Fusion: 0 fusions applied
Final:  312 kernels (no reduction)
```

**Problem**: Prefill uses GEMM kernels, but fusion patterns expect GEMV kernels.

Prefill IR1 has:
```
rmsnorm_forward
gemm_nt_q5_0_q8_0    // ← Matrix-matrix (prefill)
gemm_nt_q5_0_q8_0
gemm_nt_q8_0_q8_0
rope_forward_qk
attention_forward_causal_head_major_gqa_flash_strided
gemm_nt_q5_0_q8_0
ck_residual_add_token_major
```

Fusion pattern expects:
```
rmsnorm_forward
gemv_q5_0_q8_0       // ← Matrix-vector (decode)
gemv_q5_0_q8_0
gemv_q8_0_q8_0
rope_forward_qk
attention_forward_causal_head_major_gqa_flash_strided
gemv_q5_0_q8_0
ck_residual_add_token_major
```

## Root Cause

The kernel registry has **decode-specific fusion patterns** even though the fused kernel is named `mega_fused_attention_prefill`.

This is likely because:
1. The fused kernel was originally designed for decode
2. The naming is misleading (should be `_decode` not `_prefill`)
3. OR the fusion sequence needs updating to match prefill kernels

## Solution Options

### Option 1: Add Prefill Fusion Patterns to Registry

Create new fusion patterns that match GEMM sequences:

```json
{
  "id": "mega_fused_attention_gemm_prefill",
  "op": "fused_attention_block",
  "fuses": [
    "rmsnorm_forward",
    "gemm_nt_q5_0_q8_0",    // ← GEMM for prefill
    "gemm_nt_q5_0_q8_0",
    "gemm_nt_q8_0_q8_0",
    "rope_forward_qk",
    "attention_forward_causal_head_major_gqa_flash_strided",
    "gemm_nt_q5_0_q8_0",
    "ck_residual_add_token_major"
  ]
}
```

### Option 2: Make Fusion Patterns Quantization-Agnostic

Use wildcards or patterns that match both gemv and gemm variants:

```json
{
  "fuses_pattern": [
    "rmsnorm_forward",
    "matmul_*_q5_0_q8_0",    // ← Matches both gemv and gemm
    "matmul_*_q5_0_q8_0",
    "matmul_*_q8_0_q8_0",
    ...
  ]
}
```

### Option 3: Fix Kernel Naming

If `mega_fused_attention_prefill` actually only works for decode:
- Rename to `mega_fused_attention_decode`
- Create separate `mega_fused_attention_prefill` that handles GEMM

## Files to Update

1. **Kernel Registry**: `version/v6.6/kernel_maps/KERNEL_REGISTRY.json`
   - Add prefill-specific fusion patterns
   - OR fix existing pattern sequences to use GEMM

2. **Kernel Maps**: `version/v6.6/kernel_maps/*.json`
   - Ensure fused kernels have correct "fuses" field
   - Regenerate registry after changes

3. **Build Script**: No changes needed in `build_ir_v6_6.py`
   - Fusion logic is already registry-driven
   - Will automatically use new patterns when registry updates

## Testing

After fixing the registry, test with:

```bash
# Should show fusions applied for both modes
python build_ir_v6_6.py --manifest=... --mode=prefill --layout-output=layout_prefill.json
python build_ir_v6_6.py --manifest=... --mode=decode --layout-output=layout_decode.json
```

Expected output:
```
Prefill: X fusions applied, Y kernels removed
Decode:  12 fusions applied, 84 kernels removed
```

## Summary

**Fusion infrastructure**: ✅ Fully implemented and working
**Decode fusion**: ✅ Working (27% kernel reduction)
**Prefill fusion**: ❌ Blocked by registry patterns using wrong kernel types

**Action needed**: Update kernel registry fusion patterns to use GEMM kernels for prefill mode.
