# Quick Summary: INT8 Batch Kernels for Prefill

## ✅ Confirmed

1. **Prefill IS using batch GEMM** (not gemv) - GOOD!
2. **Prefill is NOT using INT8 kernels** - THIS IS THE PROBLEM
3. **Decode IS using INT8 kernels** - 2.91 tok/s
4. **Missing kernels exist in codebase** - we have the pieces!

## ❌ The Gap

| Component | Decode | Prefill | Status |
|-----------|--------|---------|--------|
| Q/K/V projections | `gemv_q5_0_q8_0_avx` (INT8) | `gemm_nt_q5_0` (FP32) | ❌ |
| Output projection | `gemv_q5_0_q8_0_avx` (INT8) | `gemm_nt_q5_0` (FP32) | ❌ |
| MLP | `gemv_q5_0_q8_0_avx` (INT8) | `gemm_nt_q5_0` (FP32) | ❌ |

**Result**: Prefill 2.5x slower than decode!

---

## 🎯 The Fix

### Create 2 Batch INT8 Kernels

```c
// 1. Add to src/kernels/gemm_kernels_q5_0.c:
void gemm_nt_q5_0_q8_0(const float *A_q8, const void *B_q5_0, 
                           float *C, int M, int N, int K);

// 2. Add to src/kernels/gemm_kernels_q8_0.c:
void gemm_nt_q8_0_q8_0(const float *A_q8, const void *B_q8_0,
                           float *C, int M, int N, int K);
```

**Implementation**: Based on existing `gemv_q5_0_q8_0()` + `gemm_nt_q5_0()`

**Use**: `vec_dot_q5_0_q8_0_avx()` - already optimized!

---

## 📊 Expected Impact

| Stage | Prefill | Decode | Overall | Gain |
|-------|---------|--------|---------|------|
| **Current** | 1.18 tok/s | 2.91 tok/s | 1.50 tok/s | 1.0x |
| **+ INT8 Batch** | 6-8 tok/s | 2.91 tok/s | 4.0 tok/s | **2.7x** |

**Expected prefill improvement**: **5-7x** 🚀

---

## 🚀 Next Steps

### Day 1: Implement Kernels
```bash
# Create gemm_nt_q5_0_q8_0() in src/kernels/gemm_kernels_q5_0.c
# Create gemm_nt_q8_0_q8_0() in src/kernels/gemm_kernels_q8_0.c
# Test with unit tests
```

### Day 2: Update Codegen
```bash
# Edit scripts/v6.5/codegen_v6_5.py
# Replace gemm_nt_q5_0 → gemm_nt_q5_0_q8_0
# Replace gemm_nt_q8_0 → gemm_nt_q8_0_q8_0
# Add quantization layer
```

### Day 3: Rebuild & Test
```bash
# Regenerate model
python scripts/v6.5/ck_run_v6_5.py run <model> --force-compile

# Verify INT8 kernels
grep "gemm_nt.*q8_0" ~/.cache/ck-engine-v6.5/models/*/ck-kernel-prefill.c

# Test performance
python3 scripts/v6.5/profile_inference.py | grep Prefill
# Expected: 6-8 tok/s (was 1.18)
```

---

## 💡 Why This Will Work

- **We have the building blocks** - `gemv_q5_0_q8_0()` and `vec_dot_q5_0_q8_0_avx()`
- **llama.cpp proves INT8 works** - they use INT8 for prefill
- **Decode already uses INT8** - 2.91 tok/s shows it works
- **Just need batch versions** - straightforward implementation

---

## 🎯 Bottom Line

**Problem**: Prefill uses FP32, decode uses INT8  
**Solution**: Create INT8 batch kernels  
**Expected**: 1.18 → 6-8 tok/s (5-7x)  
**Timeline**: 2-3 days  
**Impact**: 2.7x overall improvement

**Start with `gemm_nt_q5_0_q8_0()` - the starter code is ready! 🚀**
