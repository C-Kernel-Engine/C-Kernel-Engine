# 🚀 IMMEDIATE ACTION: Fix Prefill INT8 Batch Kernels

## Status Summary

✅ **Prefill IS using batch GEMM** (not gemv)  
❌ **Prefill NOT using INT8 kernels** (uses FP32 instead)  
❌ **Missing: `gemm_nt_q5_0_q8_0`** (batch INT8 kernel)  
❌ **Missing: `gemm_nt_q8_0_q8_0`** (batch INT8 kernel)

---

## 🎯 The Plan

### Step 1: Create Batch INT8 Kernels (Day 1)

**File**: `src/kernels/gemm_kernels_q5_0.c`

**Add these functions**:
```c
// Batch version of gemv_q5_0_q8_0
void gemm_nt_q5_0_q8_0(const float *A_q8, const void *B_q5_0, 
                        float *C, int M, int N, int K);

// Batch version of gemv_q8_0_q8_0  
void gemm_nt_q8_0_q8_0(const float *A_q8, const void *B_q8_0,
                        float *C, int M, int N, int K);
```

**Implementation**: Based on existing `gemv_q5_0_q8_0()` + `gemm_nt_q5_0()`

**Test**: Unit test to verify correctness vs llama.cpp

### Step 2: Update Codegen (Day 2)

**File**: `scripts/v6.5/codegen_v6_5.py`

**Change**:
```python
# From:
gemm_nt_q5_0(ln1_out, wq_h, bq_h, q_h, num_tokens, ...)
gemm_nt_q8_0(ln1_out, wv_h, bv_h, v_h, num_tokens, ...)

# To:
gemm_nt_q5_0_q8_0(ln1_q8, wq_h, bq_h, q_h, num_tokens, ...)
gemm_nt_q8_0_q8_0(ln1_q8, wv_h, bv_h, v_h, num_tokens, ...)
```

**Add**: Quantization layer before GEMM

### Step 3: Rebuild Model (Day 3)

```bash
# Regenerate with INT8 batch kernels
python scripts/v6.5/ck_run_v6_5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --force-compile --int8-activation

# Verify prefill uses INT8:
grep "gemm_nt.*q8_0" ~/.cache/ck-engine-v6.5/models/*/ck-kernel-prefill.c
```

### Step 4: Add OpenMP (Day 4)

**File**: `ck-kernel-prefill.c`

```c
#include <omp.h>

#pragma omp parallel for collapse(2)
for (int layer = 0; layer < num_layers; layer++) {
    for (int head = 0; head < num_heads; head++) {
        process_layer_head(layer, head);
    }
}
```

---

## 📊 Expected Results

| Stage | Prefill | Decode | Overall | Gain |
|-------|---------|--------|---------|------|
| **Current** | 1.18 tok/s | 2.91 tok/s | 1.50 tok/s | 1.0x |
| **+ INT8 Batch** | 6-8 tok/s | 2.91 tok/s | 4.0 tok/s | **4.0x** |
| **+ OpenMP** | 12-16 tok/s | 2.91 tok/s | 6.0 tok/s | **6.0x** |
| **+ Final** | 15-20 tok/s | 2.91 tok/s | 8.0 tok/s | **8.0x** |

**Expected prefill speedup**: **8-12x** 🚀

---

## ✅ Verification Checklist

### After Day 1:
- [ ] `gemm_nt_q5_0_q8_0()` implemented
- [ ] `gemm_nt_q8_0_q8_0()` implemented
- [ ] Unit tests pass

### After Day 2:
- [ ] Codegen updated
- [ ] Quantization layer added

### After Day 3:
- [ ] Model regenerated
- [ ] Prefill uses INT8 kernels
- [ ] Performance test: `python3 scripts/v6.5/profile_inference.py | grep Prefill`

### After Day 4:
- [ ] OpenMP parallelization added
- [ ] All cores utilized
- [ ] Performance test confirms speedup

---

## 🎯 Success Metrics

### Prefill Speed
```bash
# Target after Day 3:
grep Prefill scripts/v6.5/profile_inference.py
# Should show: 6-8 tok/s (was 1.18)

# Target after Day 4:
# Should show: 12-16 tok/s
```

### Generated Code
```bash
# Verify INT8 kernels in prefill:
grep -E "gemm_nt_q5_0_q8_0|gemm_nt_q8_0_q8_0" ~/.cache/ck-engine-v6.5/models/*/ck-kernel-prefill.c

# Should find 6+ occurrences
```

### Overall Performance
```bash
# Test end-to-end:
python3 scripts/v6.5/profile_inference.py
# Target: 6+ tok/s overall (was 1.5)
```

---

## 🔧 Tools

### Profile
```bash
./run_flamegraph_v6.sh
# Compare before/after
```

### Benchmark
```bash
python3 benchmark_v65.py
# Track improvement
```

### Debug
```bash
# Check kernel usage:
objdump -d ck-kernel-inference.so | grep "gemm_nt.*q8_0"

# Check assembly:
objdump -d ck-kernel-inference.so | grep -A20 gemm_nt_q5_0_q8_0
```

---

## 💡 Why This Will Work

### Root Cause
- **Decode**: `gemv_q5_0_q8_0_avx` (INT8) → 2.91 tok/s ✅
- **Prefill**: `gemm_nt_q5_0` (FP32) → 1.18 tok/s ❌

### Solution
- Create batch INT8 kernels for prefill
- Match decode's INT8 optimization
- **Expected**: 4-8x speedup

### Proof
- llama.cpp uses INT8 batch kernels for prefill
- We have INT8 kernels (decode proves it)
- Just need batch versions

---

## 🚀 Start Now

### Option 1: Implement from scratch
```bash
# Create from existing kernels
vim src/kernels/gemm_kernels_q5_0.c
# Add gemm_nt_q5_0_q8_0() based on gemv_q5_0_q8_0()
```

### Option 2: Use starter code
```bash
# I've created: src/kernels/gemm_kernels_q5_0_q8_0_batch.c
# Copy functions to gemm_kernels_q5_0.c
# Add to header
# Compile
```

**Start with Day 1 - create the kernels!**

**Expected: 1.18 → 6 tok/s = 5x improvement just from INT8! 🚀**
