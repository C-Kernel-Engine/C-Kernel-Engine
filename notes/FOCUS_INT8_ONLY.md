# 🎯 FOCUS: INT8 Batch Kernels Only (Skip OpenMP)

## Priority Order

### 1. Create Batch INT8 Kernels (Day 1-2)
**Goal**: Match decode's INT8 optimization for prefill

```c
// Add to src/kernels/gemm_kernels_q5_0.c:

// Batch version of gemv_q5_0_q8_0
void gemm_nt_q5_0_q8_0(const float *A_q8, const void *B_q5_0, 
                        float *C, int M, int N, int K);

// Add to src/kernels/gemm_kernels_q8_0.c:

// Batch version of gemv_q8_0_q8_0  
void gemm_nt_q8_0_q8_0(const float *A_q8, const void *B_q8_0,
                        float *C, int M, int N, int K);
```

**Expected**: 1.18 → 4-6 tok/s (4-5x improvement)

---

### 2. Update Codegen (Day 2-3)
**Goal**: Make prefill use INT8 kernels

**File**: `scripts/v6.5/codegen_v6_5.py`

**Changes**:
```python
# In prefill kernel generation:
# FROM:
gemm_nt_q5_0(ln1_out, wq_h, bq_h, q_h, num_tokens, ...)
# TO:
gemm_nt_q5_0_q8_0(ln1_q8, wq_h, bq_h, q_h, num_tokens, ...)

# FROM:
gemm_nt_q8_0(ln1_out, wv_h, bv_h, v_h, num_tokens, ...)
# TO:
gemm_nt_q8_0_q8_0(ln1_q8, wv_h, bv_h, v_h, num_tokens, ...)
```

**Add**: Quantization layer before GEMM

**Expected**: 4-6 → 6-8 tok/s (1.5x improvement)

---

### 3. Test & Verify (Day 3)
```bash
# Rebuild model
python scripts/v6.5/ck_run_v6_5.py run <model> --force-compile --int8-activation

# Verify INT8 kernels in prefill
grep "gemm_nt.*q8_0" ~/.cache/ck-engine-v6.5/models/*/ck-kernel-prefill.c

# Test performance
python3 scripts/v6.5/profile_inference.py | grep Prefill
# Expected: 6-8 tok/s (was 1.18)
```

---

## Expected Results (No OpenMP)

| Stage | Prefill | Decode | Overall | Notes |
|-------|---------|--------|---------|-------|
| **Current** | 1.18 tok/s | 2.91 tok/s | 1.50 tok/s | FP32 kernels |
| **+ INT8 Batch** | 6-8 tok/s | 2.91 tok/s | 4.0 tok/s | INT8 kernels |
| **Expected gain** | **5-7x** | 1.0x | **2.7x** | Just from INT8! |

**This alone gets us to 4 tok/s overall - already 2.7x improvement!**

---

## Implementation Details

### Kernel Design

**Based on**:
- `gemv_q5_0_q8_0()` - single token INT8 kernel
- `gemm_nt_q5_0()` - batch FP32 kernel

**Pattern**:
```c
void gemm_nt_q5_0_q8_0(const float *A_q8, const void *B_q5_0, 
                        float *C, int M, int N, int K) {
    // M = num_tokens (batch size)
    // N = output_dim  
    // K = input_dim
    
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < M; row++) {
            // Process K dimension
            for (int block = 0; block < K/QK5_0; block++) {
                // Use vec_dot_q5_0_q8_0_avx for each row
                vec_dot_q5_0_q8_0_avx(K, &C[row*N + col], ...);
            }
        }
    }
}
```

**Use**: `vec_dot_q5_0_q8_0_avx()` - we already have this optimized kernel!

---

## Why Skip OpenMP?

1. **INT8 is 80% of the problem** - biggest win
2. **OpenMP is easy** - just add `#pragma omp parallel for`
3. **OpenMP needs correct kernel first** - no point parallelizing slow kernels
4. **Debugging** - easier to debug single-threaded first

**Add OpenMP later** - it's 1 day of work, not critical path.

---

## Success Criteria

### Day 1:
- [ ] `gemm_nt_q5_0_q8_0()` implemented and tested
- [ ] `gemm_nt_q8_0_q8_0()` implemented and tested
- [ ] Unit tests pass

### Day 2:
- [ ] Codegen updated to use INT8 kernels
- [ ] Model regenerated
- [ ] Prefill uses INT8 kernels

### Day 3:
- [ ] Performance test: 6-8 tok/s prefill
- [ ] Generated code verified
- [ ] Overall performance: 4+ tok/s

---

## Tools

### Test the kernel:
```bash
# Unit test
python3 scripts/v6.5/test_kernel_direct.py

# Profile
./run_flamegraph_v6.sh

# Verify kernels
objdump -d ck-kernel-inference.so | grep "gemm_nt.*q8_0"
```

### Performance check:
```bash
# Before fix:
python3 scripts/v6.5/profile_inference.py | grep Prefill
# Expected: 1.18 tok/s

# After fix:
python3 scripts/v6.5/profile_inference.py | grep Prefill  
# Expected: 6-8 tok/s
```

---

## Bottom Line

**Focus**: Create batch INT8 kernels  
**Skip**: OpenMP (for now)  
**Expected**: 1.18 → 6-8 tok/s (5-7x)  
**Timeline**: 2-3 days  
**Impact**: Gets us to 4 tok/s overall (2.7x improvement)

**Start with `gemm_nt_q5_0_q8_0()` - use the starter code I provided! 🚀**
