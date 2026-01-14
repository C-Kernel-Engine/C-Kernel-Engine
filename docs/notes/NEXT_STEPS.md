# Next Steps: Fix Prefill INT8 Batch Kernels

## Current Situation

✅ `--int8-activation` flag exists  
✅ Decode uses INT8 kernels (fast)  
❌ Prefill uses FP32 kernels (slow)  
❌ Batch INT8 kernels missing  
❌ Codegen not set up for prefill INT8  

---

## Step 1: Create Batch INT8 Kernels (30 mins)

### Add to `src/kernels/gemm_kernels_q5_0.c`:

```c
// After existing gemv_q5_0_q8_0 function:
void gemm_nt_q5_0_q8_0(
    const float *A_q8,    // Input: Q8_0 activations (M*K)
    const void *B_q5,     // Weights: Q5_0 (N*K blocks)
    float *C,              // Output: FP32 (M*N)
    int M,                 // Batch size (tokens)
    int N,                 // Output dim
    int K)                 // Input dim
{
    const block_q5_0 *weights = (const block_q5_0 *)B_q5;
    const int blocks_per_col = K / QK5_0;
    
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < M; row++) {
            vec_dot_q5_0_q8_0_avx(
                K,
                &C[row * N + col],
                &weights[col * blocks_per_col],
                &A_q8[row * blocks_per_col * QK5_0]
            );
        }
    }
}
```

**Implementation**: Uses existing `vec_dot_q5_0_q8_0_avx()` - already optimized!

---

## Step 2: Update Codegen (15 mins)

**File**: `scripts/v6.5/codegen_v6_5.py`

**Find**: Lines with `gemm_nt_q5_0` and `gemm_nt_q8_0`

**Replace**:
```python
# FROM:
gemm_nt_q5_0(ln1_out, wq_h, bq_h, q_h, num_tokens, ...)

# TO:
gemm_nt_q5_0_q8_0(ln1_q8, wq_h, bq_h, q_h, num_tokens, ...)
```

---

## Step 3: Rebuild (5 mins)

```bash
python scripts/v6.5/ck_run_v6_5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --force-compile
```

---

## Step 4: Verify (2 mins)

```bash
# Check prefill uses INT8:
grep "gemm_nt.*q8_0" ~/.cache/ck-engine-v6.5/models/*/ck-kernel-prefill.c

# Test performance:
python3 scripts/v6.5/profile_inference.py | grep Prefill
# Expected: 6-8 tok/s (was 1.18)
```

---

## Expected Result

| Stage | Prefill | Overall | Gain |
|-------|---------|---------|------|
| **Current** | 1.18 tok/s | 1.50 tok/s | 1.0x |
| **+ Batch INT8** | 6-8 tok/s | 4.0 tok/s | **2.7x** |

---

## Why This Will Work

- **Uses existing optimized kernel**: `vec_dot_q5_0_q8_0_avx()`
- **llama.cpp proves it works**: INT8 for prefill
- **Decode already fast**: Shows INT8 works
- **Simple implementation**: Just batch version of existing kernel

---

## Total Time: ~1 hour

1. Add kernel (30 min)
2. Update codegen (15 min)
3. Rebuild (5 min)
4. Verify (2 min)
5. Test (5 min)

**Result**: 2.7x overall improvement! 🚀
