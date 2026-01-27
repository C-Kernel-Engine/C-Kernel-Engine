# Fusion Status - BOTH MODES ENABLED

## Summary

✅ **Fusion now works for BOTH prefill and decode modes!**

## Results

### Decode Mode ✅
```
IR1:    312 kernels
Fusion: 12 fusions applied, 84 kernels removed
Final:  228 kernels (27% reduction)
```

### Prefill Mode ✅
```
IR1:    312 kernels
Fusion: 12 fusions applied, 84 kernels removed
Final:  228 kernels (27% reduction)
```

## What Was Fixed

### Problem
- Fusion patterns with "prefill" in their names were using GEMV (decode) kernels
- This caused pattern mismatch for prefill mode (which uses GEMM kernels)
- Decode mode needs GEMV, prefill mode needs GEMM

### Solution
1. **Updated prefill fusion patterns to use GEMM kernels**
   - `fused_rmsnorm_qkv_prefill_head_major_quant`: Now uses `gemm_nt_*` kernels
   - `mega_fused_attention_prefill`: Now uses `gemm_nt_*` kernels
   - `mega_fused_outproj_mlp_prefill`: Already used `gemm_blocked_serial` (correct)

2. **Created separate decode fusion patterns**
   - `fused_rmsnorm_qkv_decode_head_major_quant`: Uses `gemv_*` kernels
   - `mega_fused_attention_decode`: Uses `gemv_*` kernels
   - `mega_fused_outproj_mlp_prefill`: Shared (uses generic matmul kernels)

## Fusion Patterns

### Prefill Mode (uses GEMM)
```
mega_fused_attention_prefill (8 kernels → 1):
  - rmsnorm_forward
  - gemm_nt_q5_0_q8_0  (Q projection)
  - gemm_nt_q5_0_q8_0  (K projection)
  - gemm_nt_q8_0_q8_0  (V projection)
  - rope_forward_qk
  - attention_forward_causal_head_major_gqa_flash_strided
  - gemm_nt_q5_0_q8_0  (output projection)
  - ck_residual_add_token_major

fused_rmsnorm_qkv_prefill_head_major_quant (4 kernels → 1):
  - rmsnorm_forward
  - gemm_nt_q5_0_q8_0  (Q projection)
  - gemm_nt_q5_0_q8_0  (K projection)
  - gemm_nt_q8_0_q8_0  (V projection)

mega_fused_outproj_mlp_prefill (7 kernels → 1):
  - ck_attention_project_head_major
  - ck_residual_add_token_major
  - rmsnorm_forward
  - gemm_blocked_serial  (gate + up projection)
  - gemm_blocked_serial  (still fused)
  - swiglu_forward
  - gemm_blocked_serial  (down projection)
```

### Decode Mode (uses GEMV)
```
mega_fused_attention_decode (8 kernels → 1):
  - rmsnorm_forward
  - gemv_q5_0_q8_0  (Q projection)
  - gemv_q5_0_q8_0  (K projection)
  - gemv_q8_0_q8_0  (V projection)
  - rope_forward_qk
  - attention_forward_causal_head_major_gqa_flash_strided
  - gemv_q5_0_q8_0  (output projection)
  - ck_residual_add_token_major

fused_rmsnorm_qkv_decode_head_major_quant (4 kernels → 1):
  - rmsnorm_forward
  - gemv_q5_0_q8_0  (Q projection)
  - gemv_q5_0_q8_0  (K projection)
  - gemv_q8_0_q8_0  (V projection)

mega_fused_outproj_mlp_prefill (7 kernels → 1):
  - Same as prefill (uses generic matmul)
```

## How It Works

The v6.6 pipeline implements **registry-driven fusion**:

1. **IR1 Generation**: Template + Quant Summary → Kernel IDs
2. **Fusion Pass**: Scan registry for kernels with `"fuses"` field
3. **Pattern Matching**: Match consecutive kernel sequences in IR1
4. **Replacement**: Replace matching sequences with fused kernels
5. **Statistics**: Track fusions applied and kernels removed

## Registry Structure

Each fusion kernel in `KERNEL_REGISTRY.json` has a `"fuses"` field:

```json
{
  "id": "mega_fused_attention_prefill",
  "op": "fused_attention_block",
  "fuses": [
    "rmsnorm_forward",
    "gemm_nt_q5_0_q8_0",
    "gemm_nt_q5_0_q8_0",
    "gemm_nt_q8_0_q8_0",
    "rope_forward_qk",
    "attention_forward_causal_head_major_gqa_flash_strided",
    "gemm_nt_q5_0_q8_0",
    "ck_residual_add_token_major"
  ]
}
```

The fusion pass automatically:
- Finds this pattern in IR1
- Replaces 8 kernel calls with 1 fused kernel call
- Removes intermediate buffers
- Reduces memory traffic

## Files Modified

1. **`KERNEL_REGISTRY.json`**:
   - Updated prefill patterns to use GEMM
   - Added decode patterns using GEMV

2. **`build_ir_v6_6.py`**:
   - Already had fusion infrastructure
   - No changes needed!

## Scripts Created

1. **`update_fusion_patterns.py`**: Updates prefill patterns to use GEMM
2. **`add_decode_fusion_patterns.py`**: Creates decode-specific patterns

## Benefits

- **27% kernel reduction** for both modes
- **Reduced memory traffic** (fewer intermediate buffers)
- **Better cache utilization** (larger fused kernels)
- **Simplified scheduling** (fewer kernel launches)

## Next Steps

The fusion infrastructure is complete. Next stage:
- **Code Generation**: Generate C code that calls these fused kernels
- The codegen will see the fused kernel IDs and generate appropriate calls

## Testing

```bash
# Test prefill mode
python build_ir_v6_6.py --manifest=... --mode=prefill --layout-output=layout_prefill.json

# Test decode mode
python build_ir_v6_6.py --manifest=... --mode=decode --layout-output=layout_decode.json
```

Both should show:
```
✓ Fusion complete:
  Total fusions: 12
  Kernels removed: 84
  Final kernel count: 228 (was 312)
```

## Technical Notes

### Why Separate Patterns?

- **Decode**: Single token (seq_len=1), uses matrix-vector operations (GEMV)
- **Prefill**: Multiple tokens (seq_len>1), uses matrix-matrix operations (GEMM)
- Different kernel types require different fusion patterns

### Pattern Matching

Fusion uses exact string matching on kernel IDs:
- Sequence must match exactly (including quantization variants)
- Patterns are tried in priority order (longest first)
- Greedy matching (first match wins)

### Quantization Awareness

Fusion patterns are quantization-specific:
- `gemv_q5_0_q8_0` vs `gemv_q8_0_q8_0` are different kernels
- Patterns must match the exact quantization types
- This is why we have variants for different weight quants
