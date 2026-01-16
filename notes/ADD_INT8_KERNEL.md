# Add INT8 Batch Kernel for Prefill

## Status
✅ Build fixed  
✅ Module import fixed  
✅ Performance: Decode 3.10 tok/s (good!)  
❌ Prefill still slow: 1.25 tok/s  

## The Fix
Add batch INT8 kernel to `src/kernels/gemm_kernels_q5_0.c`

### Step 1: Add Function (5 mins)
```bash
vim src/kernels/gemm_kernels_q5_0.c
# Add at end of file:
```

```c
/*
 * Batch version of gemv_q5_0_q8_0 for prefill
 */
void gemm_nt_q5_0_q8_0(
    const float *A_q8,
    const void *B_q5,
    float *C,
    int M,
    int N,
    int K)
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

### Step 2: Update Codegen (2 mins)
```bash
vim scripts/v6.5/codegen_v6_5.py
# Find: gemm_nt_q5_0(
# Replace: gemm_nt_q5_0_q8_0(
```

### Step 3: Rebuild & Test
```bash
python scripts/v6.5/ck_run_v6_5.py run <model> --force-compile
python3 scripts/v6.5/profile_inference.py | grep Prefill
```

### Expected Result
- **Prefill**: 1.25 → 6-8 tok/s (5-6x improvement)
- **Overall**: 1.6 → 4-5 tok/s

## Ready to Implement?
The kernel is simple - just copy it to the file!
