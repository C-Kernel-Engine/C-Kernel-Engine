# v6 Gibberish Output - Root Cause Analysis

## Problem Summary

The v6 pipeline generates and compiles successfully, but produces gibberish output:

**Test Results:**
- Input: `"Hello"` → Output: `,,,,`
- Input: `"Hi"` → Output: `,,,,,`
- Input: `"The capital of France is"` → Output: `______ following following following following`

This is **structured garbage** - the model produces coherent text fragments but they're meaningless.

---

## Analysis

### What Works ✅
1. **Compilation**: Model compiles without errors
2. **Tokenization**: Prompts tokenize correctly (e.g., "Hello" → 9707)
3. **Execution**: Model runs inference without crashing
4. **Decoding**: Tokens decode to text (not random bytes)

### What's Broken ❌
1. **Logits/Probabilities**: Model outputs heavily skewed toward specific tokens
2. **Computation**: Transformer layers produce incorrect activations

---

## Root Cause Investigation

### Hypothesis 1: Weight Loading Issue
**Theory**: Weights aren't loaded correctly into the model

**Evidence**:
- Model runs but produces garbage
- Could be offset/size mismatch in weight loading

**Check**:
```bash
# Compare BUMP file size
ls -lh ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump
# Should be ~491MB (actual: 491M ✓)
```

**Likely**: No - file size is correct

---

### Hypothesis 2: Missing Bias Terms
**Theory**: Bias vectors (BQ, BK, BV) not being applied

**Evidence**:
- Generated code shows: `gemv_q5_0(q_token, WQ, ln1_out, H * head_dim, aligned_embed_dim)`
- But Qwen2 has bias terms: `y = (WQ @ x) + BQ`
- We're missing the bias addition!

**Check** in generated code:
```c
grep -n "bq\|bk\|bv" ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.c | head -20
```

**Likely**: YES - This is the issue!

---

### Hypothesis 3: Activation Function Issue
**Theory**: SwiGLU activation not applied correctly

**Evidence**:
- Model has SwiGLU MLP
- Gate and Up projections should be separate
- Results should be multiplied element-wise

**Check**:
```c
grep -A10 "Gate+Up" ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.c
```

**Status**: Unknown

---

### Hypothesis 4: Quantization Dequant Issue
**Theory**: Scales/offsets not being applied during dequantization

**Evidence**:
- Weights are quantized (Q5_0, Q8_0, Q6_K)
- If dequantization is wrong, all computations are wrong

**Check**:
```c
# Look at dequant kernel calls
grep "dequant" ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.c
```

**Status**: Unknown

---

## Most Likely Root Cause: Missing Bias Terms

### Why This Fits

1. **Qwen2 Architecture**: Every linear layer has bias terms
   - Q projection: `y = WQ @ x + BQ`
   - K projection: `y = WK @ x + BK`
   - V projection: `y = WV @ x + BV`

2. **Generated Code**: Shows bias buffers are loaded
   ```
   Layer 0: wq=q5_0 wk=q5_0 wv=q8_0 wo=q5_0 w1=q5_0 w2=q6_k
            bq=✓  bk=✓  bv=✓  bo=○   (○ = no bias for output projection)
   ```

3. **GEMV Calls**: Don't include bias
   ```c
   gemv_q5_0(q_token, WQ, ln1_out, H * head_dim, aligned_embed_dim);
   // Missing: q_token += BQ (bias addition)
   ```

4. **Effect**: Without biases, all activations are shifted by a constant
   - This would cause logits to be systematically wrong
   - Model would output garbage that looks structured

---

## Fix Strategy

### Option A: Add Bias to GEMV (Quick Fix)

Modify GEMV kernel signatures to accept bias:

```c
// Current signature (5 args):
void gemv_q5_0(float *y, const void *W, const float *x, int M, int K);

// New signature (6 args):
void gemv_q5_0(float *y, const void *W, const float *x, const float *bias, int M, int K);
```

Then in generated code:
```c
gemv_q5_0(q_token, WQ, ln1_out, BQ, H * head_dim, aligned_embed_dim);
```

**Pros**: Quick, minimal code change
**Cons**: Changes kernel API, need to modify all GEMV calls

---

### Option B: Add Bias as Separate Addition (Recommended)

Keep GEMV signature, add explicit bias addition:

```c
// After GEMV call:
gemv_q5_0(q_token, WQ, ln1_out, H * head_dim, aligned_embed_dim);

// Add bias term:
for (int i = 0; i < H * head_dim; i++) {
    q_token[i] += BQ[i];
}
```

**Pros**: No kernel API change, explicit
**Cons**: Small performance overhead (negligible for decode)

---

### Option C: Fused Kernel (Best Long-term)

Create new kernels that fuse GEMV + bias:

```c
// New fused kernels:
void gemv_add_bias_q5_0(float *y, const void *W, const float *x, const float *bias, int M, int K);
void gemv_add_bias_q8_0(float *y, const void *W, const float *x, const float *bias, int M, int K);
void gemv_add_bias_q6_k(float *y, const void *W, const float *x, const float *bias, int M, int K);
```

**Pros**: Best performance, clean API
**Cons**: More code to write

---

## Immediate Action Plan

### Step 1: Verify Bias Loading
Check if bias buffers are loaded correctly:

```bash
# In generated code, look for bias buffer declarations
grep -n "BQ\|BK\|BV" ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.c | head -30
```

Expected:
```c
static const float L0_BQ[ALIGNED_EMBED_DIM] = {...};  // Bias vectors
```

---

### Step 2: Check GEMV Usage
Find all GEMV calls that should have bias:

```bash
grep -n "gemv_" ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.c
```

Count:
- Q/K/V projections: 3 * 24 layers = 72 calls (need bias)
- Output projection: 24 calls (no bias for Qwen2)
- MLP projections: 48 calls (W1 has bias, W2 doesn't)
- LM head: 1 call (needs bias)

**Expected bias additions**: ~120 locations

---

### Step 3: Implement Fix (Option B)

**Files to modify**:
1. `scripts/v6/codegen_v6.py` - Add bias addition after GEMV calls
2. `include/ckernel_engine.h` - (no change needed)
3. Kernel implementations - (no change needed)

**Code change in codegen_v6.py**:
```python
# Find all Q/K/V projection GEMV calls and add:
add(f"    // Add bias term");
add(f"    for (int i = 0; i < {dim}; i++) {{");
add(f"        {output}[i] += {bias}[i];");
add(f"    }}");
```

---

### Step 4: Rebuild and Test

```bash
# Force recompile
python scripts/v6/ck_run_v6.py run MODEL --force-compile --prompt "Hello" --max-tokens 5

# Expected output (should be coherent):
# Assistant: Hello! How can I help you today?
```

---

## Debugging Checklist

### Quick Checks
- [ ] Verify BUMP file loads correctly
- [ ] Check bias buffers are non-zero
- [ ] Verify GEMV outputs are reasonable (not NaN/Inf)
- [ ] Test with known-good model (llama.cpp) to isolate issue

### Detailed Checks
- [ ] Compare first layer output with PyTorch reference
- [ ] Verify RMSNorm scale and offset
- [ ] Check RoPE implementation
- [ ] Validate SwiGLU gate * up multiplication
- [ ] Ensure attention scores are reasonable (not NaN)

---

## Performance Note

The bias addition is O(n) per layer:
- Qwen2-0.5B: 896 dimensions
- 24 layers
- Decode: 896 * 24 = 21,504 FLOPs per token
- For comparison: GEMV is ~896 * 896 = 802,816 FLOPs
- **Bias overhead**: ~2.7% (negligible)

---

## Files That Need Changes

### Modified Files
1. **`scripts/v6/codegen_v6.py`**
   - Add bias addition logic after GEMV calls
   - Apply to: Q/K/V projections, W1 projection, LM head

### New Files (Option C)
1. **`src/kernels/gemm_kernels_q5_0.c`** - Add `gemv_add_bias_q5_0`
2. **`src/kernels/gemm_kernels_q8_0.c`** - Add `gemv_add_bias_q8_0`
3. **`src/kernels/gemm_kernels_q6_k.c`** - Add `gemv_add_bias_q6_k`

### Test Files
1. **`unittest/test_qwen2_layer.py`** - Validate single layer output

---

## Expected Outcome

After fixing bias addition:

**Before** (broken):
```
Input: "Hello"
Output: ,,,,
```

**After** (fixed):
```
Input: "Hello"
Output: Hello! How can I help you today?
```

---

## Next Steps

1. **Investigate**: Check if bias buffers are loaded
2. **Implement**: Add bias additions to generated code
3. **Test**: Verify coherent output
4. **Validate**: Compare with PyTorch reference
5. **Optimize**: If needed, create fused kernels

---

**Document Version**: 1.0
**Created**: 2026-01-11
**Status**: Root cause identified - ready for fix
