# Performance Optimization Plan: C-Kernel-Engine

**Goal:** Close the 60x performance gap with `llama.cpp` for Qwen2-0.5B (and other models).

## 1. Diagnostics & Profiling (The "Why")

Before writing code, we must empirically confirm the bottlenecks.

### 1.1 CPU Profiling with `perf`
The hypothesis is that `dot_q4_k_q8_k_ref` is the bottleneck.

**Action:**
Run the model under `perf record`:
```bash
# Install linux-tools if needed: sudo apt install linux-tools-generic
perf record -g -- python3 scripts/ck_run_v5.py run Qwen/Qwen2-0.5B --weight-dtype=q4_k --max-tokens 20
perf report
```
**Expectation:** >90% of time spent in `dot_q4_k_q8_k_ref` or `gemv_q4_k_q8_k_ref`.

### 1.2 Python <-> C Overhead
Verify if `ctypes` overhead or data copying is significant.

**Action:**
Modify `scripts/ck_chat.py` to measure time inside `self.lib.ck_model_decode` vs. time in `np.ctypeslib.as_array`.
- If `as_array` + copy takes >1ms per token, it's a problem for small models (0.5B runs very fast, so 1ms is huge overhead).

### 1.3 Memory Bandwidth
For 0.5B models, we might be memory bound if kernels are fast enough.

**Action:**
Calculate theoretical peak:
- Model size: ~0.4GB (Q4_K).
- RAM Bandwidth: ~40GB/s (DDR4 dual channel).
- Max tokens/sec = 40 / 0.4 = 100 t/s.
- If we are getting < 2 t/s, we are Compute Bound (scalar emulation), not Memory Bound.

---

## 2. Kernel Optimization (The "How")

**Current State:** `src/kernels/gemm_kernels_q4k_q8k_avx2.c` contains a TODO and falls back to scalar `ref` implementation.

### 2.1 Implement AVX2 Q4_K Kernel
We must port or implement the SIMD logic for Q4_K dot products.

**Key Technical Challenges:**
- **Nibble Extraction:** Q4_K stores weights in low/high nibbles across 32-byte blocks. AVX2 byte shuffling (`_mm256_shuffle_epi8`) is required to unpack them efficiently.
- **Scale Management:** Q4_K uses block scales (`scales`) and offsets (`dmin`). These must be broadcast and applied to accumulators.
- **Register Blocking:** We need to process multiple rows (M) or multiple K-blocks per loop to hide latency.

**Implementation Steps:**
1.  **Unpack:** Load 32 bytes of Q4 data. Use `vpshufb` to separate low/high nibbles into bytes.
2.  **MADD:** Use `_mm256_madd_epi16` (VPMADDWD) to multiply 16x 8-bit weights with 16x 8-bit activations (Q8_K).
3.  **Accumulate:** Horizontal sum the results.

### 2.2 Add AVX-512 / VNNI Support
If the hardware supports it (Intel Icelake+, AMD Zen 4), AVX-512 `_mm512_dpbusd_epi32` (VPDPBUSD) gives a massive speedup by doing 4x byte-multiplies per cycle.

**Action:**
- Check `src/cpu_features.c` to ensure `__AVX512VNNI__` is detected.
- Implement `gemm_kernels_q4k_q8k_vnni.c`.

---

## 3. Runtime & Fusion Improvements

### 3.1 Python Wrapper Optimization
The current `ck_chat.py` copies the *entire* logits table (vocab size ~150k for Qwen) every token.

**Optimization:**
- **Logit Pointer:** Expose a C function `ck_model_get_logits_ptr(token_idx)` that returns a raw pointer.
- **Argmax in C:** Implement `ck_model_sample_argmax()` in C to avoid passing floats to Python entirely for greedy decoding.
- **Top-K in C:** Implement `ck_model_sample_top_k(k, temp)` in C.

### 3.2 Threading
Check `omp_set_num_threads`. Small models (0.5B) often run *slower* with too many threads due to synchronization overhead.
- **Experiment:** Try `OMP_NUM_THREADS=4` vs `8` vs `1`.

### 3.3 Fusion
The generated code calls `gemm` then `activations` then `norm`.
- **Fusion:** Merge `RMSNorm` + `Q8_K Quantization` into a single kernel to keep data in L1 cache.

---

## 4. Execution Plan (Immediate Next Steps)

1.  **Benchmark:** Run `perf` to confirm the scalar bottleneck.
2.  **Kernel Dev:** Create a standalone test `tests/bench_q4k.c` that runs just the `dot_q4_k_q8_k` function in a loop to iterate fast.
3.  **Implement:** Fill in `gemv_q4_k_q8_k_avx2.c`.
4.  **Integrate:** Recompile and measure.

