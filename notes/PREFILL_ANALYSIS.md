# Prefill Kernel Analysis: The Missing INT8 Batch Kernels

## 🔍 Current State

### ✅ Prefill IS Using Batch GEMM
From `~/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-prefill.c`:

```c
// Line 314: Q projection - WRONG KERNEL!
gemm_nt_q5_0(ln1_out, wq_h, bq_h, q_h, num_tokens, ...);

// Line 324: K projection - WRONG KERNEL!
gemm_nt_q5_0(ln1_out, wk_h, bk_h, k_h, num_tokens, ...);

// Line 334: V projection - RIGHT KERNEL!
gemm_nt_q8_0(ln1_out, wv_h, bv_h, v_h, num_tokens, ...);
```

**Status**: ✅ Using batch GEMM (processes all tokens at once)  
**Problem**: ❌ NOT using INT8 optimized kernels!

---

## 🚨 The Critical Issue

### Prefill Kernels vs Decode Kernels

| Operation | Prefill Uses | Decode Uses | INT8? |
|-----------|--------------|-------------|-------|
| **Q projection** | `gemm_nt_q5_0` | `gemv_q5_0_q8_0_avx` | ❌ vs ✅ |
| **K projection** | `gemm_nt_q5_0` | `gemv_q5_0_q8_0_avx` | ❌ vs ✅ |
| **V projection** | `gemm_nt_q8_0` | `gemv_q8_0_q8_0_avx` | ❌ vs ✅ |
| **Output proj** | `gemm_nt_q5_0` | `gemv_q5_0_q8_0_avx` | ❌ vs ✅ |
| **MLP gate/up** | `gemm_nt_q5_0` | `gemv_q5_0_q8_0_avx` | ❌ vs ✅ |
| **MLP down** | `gemm_nt_q6_k` | `gemv_q6_k` | ❌ vs ❌ |

**KEY INSIGHT**: 
- **Decode**: Uses INT8 kernels everywhere (`*_q8_0`)
- **Prefill**: Uses FP32 kernels (`gemm_nt_q5_0`, `gemm_nt_q8_0`)

**This explains why prefill is 2.5x SLOWER than decode!**

---

## 📊 Performance Impact

### Why This Matters

```
Current prefill (FP32):
  Q proj: gemm_nt_q5_0(float, q5_0, float) → SLOW
  K proj: gemm_nt_q5_0(float, q5_0, float) → SLOW
  V proj: gemm_nt_q8_0(float, q8_0, float) → SLOW

Target prefill (INT8):
  Q proj: gemm_nt_q5_0_q8_0(q8_0, q5_0, q8_0) → FAST
  K proj: gemm_nt_q5_0_q8_0(q8_0, q5_0, q8_0) → FAST
  V proj: gemm_nt_q8_0_q8_0(q8_0, q8_0, q8_0) → FAST
```

**Expected speedup**: 4-8x just from INT8 batch kernels!

---

## 🛠️ What We Have vs What We Need

### ✅ Available (for decode - single token)
```c
// src/kernels/gemm_kernels_q5_0.c
void gemv_q5_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
void vec_dot_q5_0_q8_0_avx(int n, float *s, const void *vx, const void *vy);
```

### ❌ Missing (for prefill - batch)
```c
// NEED TO CREATE THESE:
void gemm_nt_q5_0_q8_0(const void *A_q8, const void *B_q5_0, 
                        void *C, int M, int N, int K);
void gemm_nt_q8_0_q8_0(const void *A_q8, const void *B_q8_0,
                        void *C, int M, int N, int K);
```

---

## 🎯 The Fix

### Step 1: Create Batch INT8 Kernels

**File**: `src/kernels/gemm_kernels_q5_0.c`

```c
// ADD THIS FUNCTION:
void gemm_nt_q5_0_q8_0(const float *A_q8,        // Input: INT8 activations
                        const void *B_q5_0,       // Weights: Q5_0
                        float *C,                  // Output: FP32
                        int M, int N, int K) {
    // Batch version of gemv_q5_0_q8_0
    // Process M x N matrix, K dimension
    // Use vec_dot_q5_0_q8_0_avx for each element
}
```

### Step 2: Update Codegen

**File**: `scripts/v6.5/codegen_v6_5.py`

```python
# Change from:
gemm_nt_q5_0(ln1_out, wq_h, bq_h, q_h, num_tokens, ...)

# To:
gemm_nt_q5_0_q8_0(ln1_q8, wq_h, bq_h, q_h, num_tokens, ...)
```

### Step 3: Add Quantization

**File**: `ck-kernel-prefill.c`

```c
// Add quantization before GEMM:
quantize_row_q8_0(ln1_out, ln1_q8, aligned_embed_dim * num_tokens);

// Then use INT8 GEMM:
gemm_nt_q5_0_q8_0(ln1_q8, wq_h, bq_h, q_h, num_tokens, ...);
```

---

## 📈 Expected Results

### Performance Projection

| Change | Prefill Speed | Decode Speed | Overall |
|--------|---------------|-------------|---------|
| **Current** | 1.18 tok/s | 2.91 tok/s | 1.50 tok/s |
| **+ INT8 Batch** | 6-8 tok/s | 2.91 tok/s | 4.0 tok/s |
| **+ OpenMP** | 12-16 tok/s | 2.91 tok/s | 6.0 tok/s |
| **+ Final opt** | 15-20 tok/s | 2.91 tok/s | 8.0 tok/s |

**Expected prefill improvement**: **8-12x**

---

## 🚀 Implementation Plan

### Day 1: Create Batch INT8 Kernels

```bash
# Create gemm_nt_q5_0_q8_0()
# Based on: gemv_q5_0_q8_0() + gemm_nt_q5_0()
# Location: src/kernels/gemm_kernels_q5_0.c

# Test: Process M x N matrix using vec_dot_q5_0_q8_0_avx
```

**Target**: 1.18 → 4 tok/s

### Day 2: Create Q8_0 x Q8_0 Batch Kernel

```bash
# Create gemm_nt_q8_0_q8_0()
# Based on: gemv_q8_0_q8_0() + gemm_nt_q8_0()
# Location: src/kernels/gemm_kernels_q8_0.c
```

**Target**: 4 → 6 tok/s

### Day 3: Update Codegen

```bash
# scripts/v6.5/codegen_v6_5.py
# Replace gemm_nt_q5_0 → gemm_nt_q5_0_q8_0
# Replace gemm_nt_q8_0 → gemm_nt_q8_0_q8_0
# Add quantization layer
```

**Target**: 6 → 8 tok/s

### Day 4: OpenMP Parallelization

```bash
# Add #pragma omp parallel to prefill
# Parallelize across layers
```

**Target**: 8 → 12 tok/s

---

## ✅ Verification

### Check Prefill Uses INT8

```bash
# After fix, verify in generated code:
grep "gemm_nt.*q8_0" ~/.cache/ck-engine-v6.5/models/*/ck-kernel-prefill.c

# Should see:
# gemm_nt_q5_0_q8_0(...)
# gemm_nt_q8_0_q8_0(...)
```

### Benchmark

```bash
# Before:
python3 scripts/v6.5/profile_inference.py | grep Prefill
# Expected: 1.18 tok/s

# After:
python3 scripts/v6.5/profile_inference.py | grep Prefill
# Expected: 6-8 tok/s
```

---

## 💡 Key Insight

**The problem is NOT that prefill is using gemv instead of gemm**  
(prefill IS using gemm, which is correct!)

**The problem IS that prefill is using FP32 kernels instead of INT8 kernels**

**Decode**: `gemv_q5_0_q8_0_avx` (INT8) → 2.91 tok/s  
**Prefill**: `gemm_nt_q5_0` (FP32) → 1.18 tok/s

**Solution**: Create INT8 batch kernels for prefill!

---

## 🎯 Bottom Line

**Why prefill is slow**:
- Uses FP32 kernels instead of INT8 kernels
- Missing: `gemm_nt_q5_0_q8_0`, `gemm_nt_q8_0_q8_0`

**Fix**:
1. Create batch INT8 kernels (1 day)
2. Update codegen (1 day)
3. Add OpenMP (1 day)

**Result**: 1.18 → 12 tok/s = **10x improvement**

**Start with creating `gemm_nt_q5_0_q8_0`! 🚀**
