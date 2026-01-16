# C-Kernel-Engine Development Log

## 2024-12-28: CLI Multi-threading and Performance Metrics

### Summary
Major improvements to the `ck` CLI tool for better performance on multi-core systems and added llama.cpp-style performance metrics.

### Changes Made

#### 1. Fixed Intel oneAPI Runtime Library Issue (`libimf.so`)
- **Problem**: `libmodel.so` compiled in cache couldn't find Intel runtime libraries
- **Solution**: Added dynamic detection of `ONEAPI_ROOT` environment variable
- **Files**: `tools/ck.c`
- Added Intel compiler and MKL rpath detection when compiling `libmodel.so`
- Use `--force-convert` to recompile cached models with correct rpaths

#### 2. Added Multi-threading Support for `libmodel.so`
- **Problem**: Generated `libmodel.so` was running single-threaded (no OpenMP/MKL linked)
- **Solution**: Updated compile command in `tools/ck.c`
- Added `-fopenmp` for OpenMP multi-threading
- Added `-DUSE_MKL -lmkl_rt` for Intel MKL GEMM acceleration
- Added dynamic SIMD detection via `/proc/cpuinfo`:
  - AVX-512 flags on Xeon: `-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma`
  - AVX2 fallback: `-mavx2 -mfma`
  - AVX fallback: `-mavx`
  - SSE4.2 fallback: `-msse4.2`

#### 3. Added Performance Metrics (llama.cpp style)
- **File**: `scripts/ck_chat.py`
- Added timing for prefill, decode, and sampling phases
- Output format:
  ```
  prompt eval:   403.10 ms /    9 tokens (  44.79 ms/tok,   22.33 tok/s)
        decode: 26837.66 ms /  512 runs   (  52.42 ms/tok,   19.08 tok/s)
        sample:   107.76 ms /  512 runs   (   0.21 ms/tok)
         total: 27356.37 ms /  521 tokens
  ```
- Added `/stats` command in chat to toggle display
- Added `--no-stats` CLI flag

#### 4. Added `--generate-only` Option
- **Purpose**: Generate C code without compiling, for inspection
- **Files**: `tools/ck.c`, `Makefile`
- Usage: `./build/ck run HuggingFaceTB/SmolLM-135M --generate-only`
- Shows generated file paths and manual compile command
- Added `make generate-model MODEL=<name>` target

#### 5. Verbose Intel Detection Output
- Added verbose logging for Intel rpath and MKL detection
- Shows SIMD flags being used when `--verbose` is passed

### Architecture Discussion: Token vs Tensor Parallelism

Analyzed the C-Transformer parallelism pattern:
- **Current (wrong for prefill)**: OpenMP inside each kernel (tensor parallelism)
- **Better for prefill**: Token parallelism at model level, serial kernels
- **Decode (M=1)**: Needs parallel GEMV (tensor parallelism on K dimension)

**C-Transformer pattern** (to implement later):
```c
#pragma omp parallel num_threads(num_cores)
{
    int tid = omp_get_thread_num();
    int start = tid * tokens_per_core;
    int count = min(tokens_per_core, total - start);
    gemm_serial(input + start*H, W, bias, out + start*H, count, N, K);
}
```

### Verified: Flash Attention in Use
- Confirmed flash attention is used for inference (`scores_offset = 0`)
- Training uses non-flash (needs scores for backward pass)
- Decode uses `attention_forward_decode_head_major_gqa_flash()`

### Model Precision
- HuggingFace SmolLM-135M is stored as **BF16** (`torch_dtype: bfloat16`)
- Conversion scripts use `.float()` to convert to **FP32**
- `weights.bump` file is 519MB ≈ 135M params × 4 bytes (FP32)
- All inference runs in FP32 (generated code uses `sizeof(float)`)

### AVX Kernel Support
Hand-optimized SIMD kernels exist for GEMM:
- **AVX-512**: 6×32 microkernel in `src/kernels/gemm_microkernel.c`
  - Uses `_mm512_fmadd_ps` for fused multiply-add
  - Processes 32 floats (2×16) per iteration
- **AVX2**: 6×16 microkernel fallback
  - Uses `_mm256_fmadd_ps`
  - Processes 16 floats (2×8) per iteration
- Other kernels (LayerNorm, ReLU, etc.) rely on compiler auto-vectorization

### Performance Notes
- SmolLM-135M on single core: ~19-30 tok/s (reasonable for 135M params)
- With multi-threading on Xeon (24 cores + AVX512 + MKL): expect 5-10x improvement

### Files Modified
- `tools/ck.c` - Multi-threading, SIMD detection, generate-only option
- `scripts/ck_chat.py` - Performance metrics
- `Makefile` - Added `generate-model` target, updated help

### Testing
```bash
# Rebuild with fixes
make clean && make ck-cli

# Force recompile model with new settings
./build/ck run HuggingFaceTB/SmolLM-135M --force-convert --verbose

# Generate C file only (inspect before compile)
./build/ck run HuggingFaceTB/SmolLM-135M --generate-only
```

---

## Previous Sessions

(Add older development notes here as needed)
