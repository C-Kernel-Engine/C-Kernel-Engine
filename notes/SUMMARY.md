# Summary: Fix Prefill INT8 Batch Kernel

## Current Status
- ✅ Decode: 3.90 tok/s (FAST - INT8 kernels working)
- ❌ Prefill: 1.55 tok/s (SLOW - FP32 kernels)

## The Fix
**Add INT8 batch kernel for prefill**

### Option 1: Manual (10 mins)
1. Add function to `src/kernels/gemm_kernels_q5_0.c`
2. Update `scripts/v6.5/codegen_v6_5.py`
3. Rebuild

### Option 2: Automated (1 min)
```bash
./add_int8_kernel.sh
python scripts/v6.5/ck_run_v6_5.py run hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf --force-compile
```

## Expected Result
- **Prefill**: 1.55 → 6-8 tok/s (5x)
- **Overall**: ~1.6 → 4-5 tok/s (3x)

## Why This Works
- Decode uses INT8 (gemv_q5_0_q8_0) → 3.90 tok/s ✅
- Prefill uses FP32 (gemm_nt_q5_0) → 1.55 tok/s ❌
- Solution: INT8 batch kernel (gemm_nt_q5_0_q8_0) → 6-8 tok/s ✅

## Files Modified
1. `src/kernels/gemm_kernels_q5_0.c` - Add kernel
2. `scripts/v6.5/codegen_v6_5.py` - Use INT8 kernel

Ready to implement!
