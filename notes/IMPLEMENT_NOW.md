# Implement INT8 Batch Kernel - Step by Step

## Current Status
- ✅ Decode: 3.90 tok/s (FAST - using INT8)
- ❌ Prefill: 1.55 tok/s (SLOW - using FP32)

## The Fix

### Step 1: Add Kernel Function (3 mins)
```bash
# Open the file
vim src/kernels/gemm_kernels_q5_0.c

# Add this at the END of the file (before the last #endif):
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
# Find and replace
vim scripts/v6.5/codegen_v6_5.py

# Search for: gemm_nt_q5_0(
# Replace with: gemm_nt_q5_0_q8_0(

# Do this for:
# - Q projection
# - K projection  
# - Output projection
# - MLP gate/up
```

### Step 3: Rebuild (5 mins)
```bash
python scripts/v6.5/ck_run_v6_5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --force-compile
```

### Step 4: Test
```bash
python3 scripts/v6.5/profile_inference.py | grep Prefill
```

## Expected Result
- **Prefill**: 1.55 → 6-8 tok/s (5x improvement!)
- **Overall**: ~1.6 → 4-5 tok/s

## Need Help?
The kernel is simple - just add the function and update the codegen!
