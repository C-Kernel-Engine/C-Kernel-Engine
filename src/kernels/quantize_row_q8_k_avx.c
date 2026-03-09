/**
 * @file quantize_row_q8_k_avx.c
 * @brief AVX entrypoint for exact Q8_K row quantization
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 */

#include "ckernel_quant.h"

void quantize_row_q8_k_sse(const float *x, void *vy, int k);

void quantize_row_q8_k_avx(const float *x, void *vy, int k) {
    /* Reuse the parity-clean SIMD implementation until a wider AVX variant is
     * worth maintaining separately. */
    quantize_row_q8_k_sse(x, vy, k);
}
