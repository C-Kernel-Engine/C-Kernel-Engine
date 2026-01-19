# v6 Debugging Summary

## Bugs Fixed ✅

### 1. SwiGLU Down Projection Dimensions
**File**: `scripts/v6/codegen_v6.py`
**Change**: Fixed dimension order for W2 projection
**Before**: `gemv(..., aligned_intermediate_dim, aligned_embed_dim)`
**After**: `gemv(..., aligned_embed_dim, aligned_intermediate_dim)`

### 2. KV Cache Stride Calculation
**File**: `scripts/v6/codegen_v6.py`
**Change**: Fixed `kv_head_stride` calculation
**Before**: `kv_head_stride = aligned_context_window * aligned_head_dim` (WRONG - 32768x too large!)
**After**: `kv_head_stride = H_kv * aligned_head_dim` (CORRECT)

### 3. GEMV Kernel Dispatch
**File**: `src/kernels/gemm_kernels_q5_0.c`
**Change**: Re-enabled AVX implementation
**Result**: Better performance

### 4. Token Index Tracking
**File**: `src/v6/ck_cli_v6.c`
**Change**: model.c wrapper handles token_index internally
**Result**: Proper KV cache writes

## Current Status

Despite these fixes, the model still produces gibberish output:
- Input: "Hello" → Output: "from（ [ = is"
- Input: "Paris is the capital of" → Output: "踪 CITY DA"

## Remaining Issues

The output is improving (more English-like, less repetitive) but still not coherent. This suggests:

### Possible Root Causes

1. **KV Cache Read/Write Mismatch**
   - Write: `k_cache[token_index * H_kv + h]`
   - Read: Need to verify attention reads the same layout

2. **Quantization Bugs**
   - New AVX kernels may have bugs
   - Dequantization errors

3. **Attention Implementation**
   - Flash attention may have bugs
   - QK^T calculation may be wrong

4. **RoPE Implementation**
   - RoPE may be applying to wrong positions

## Next Steps

1. **Verify KV Cache Layout**
   ```c
   // Write: k_cache[token_index * H_kv + h][dim]
   float *k_head = k_cache + ((size_t)token_index * H_kv + h) * aligned_head_dim;

   // Read: Should match the same layout
   ```

2. **Test with Reference Implementation**
   - Use scalar GEMV kernels instead of AVX
   - Disable optimizations

3. **Add Debug Output**
   - Print intermediate values
   - Check for NaN/Inf

4. **Compare with PyTorch**
   - Use parity mode
   - Compare layer outputs

## Files Modified

1. `scripts/v6/codegen_v6.py` - Fixed dimensions and strides
2. `src/kernels/gemm_kernels_q5_0.c` - Re-enabled AVX
3. `src/v6/ck_cli_v6.c` - Token index handling

## Key Lesson

The stride calculation bug was critical: `aligned_context_window * head_dim` instead of `H_kv * head_dim` caused writes to completely wrong memory locations!
