/**
 * @file gemm_kernels_q4k_q8k_avx2.c
 * @brief AVX2 Q4_K x Q8_K matvec kernel (inference only)
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * Requires AVX2 for 256-bit integer operations.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

#if defined(__AVX2__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static inline void load_q8_even_odd_16(const int8_t *q8,
                                       __m256i *even16,
                                       __m256i *odd16) {
    const __m128i q8_lo = _mm_loadu_si128((const __m128i *)q8);
    const __m128i q8_hi = _mm_loadu_si128((const __m128i *)(q8 + 16));
    const __m128i even_mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    const __m128i odd_mask = _mm_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80,
        (char)0x80, (char)0x80, (char)0x80, (char)0x80);

    const __m128i q8_lo_even = _mm_shuffle_epi8(q8_lo, even_mask);
    const __m128i q8_hi_even = _mm_shuffle_epi8(q8_hi, even_mask);
    const __m128i q8_even = _mm_unpacklo_epi64(q8_lo_even, q8_hi_even);

    const __m128i q8_lo_odd = _mm_shuffle_epi8(q8_lo, odd_mask);
    const __m128i q8_hi_odd = _mm_shuffle_epi8(q8_hi, odd_mask);
    const __m128i q8_odd = _mm_unpacklo_epi64(q8_lo_odd, q8_hi_odd);

    *even16 = _mm256_cvtepi8_epi16(q8_even);
    *odd16 = _mm256_cvtepi8_epi16(q8_odd);
}

static inline int32_t dot_q4_q8_32_avx2(const uint8_t *q4,
                                        const int8_t *q8) {
    const __m128i q4_packed = _mm_loadu_si128((const __m128i *)q4);
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m128i q4_lo = _mm_and_si128(q4_packed, mask4);
    const __m128i q4_hi = _mm_and_si128(_mm_srli_epi16(q4_packed, 4), mask4);

    const __m256i q4_lo16 = _mm256_cvtepu8_epi16(q4_lo);
    const __m256i q4_hi16 = _mm256_cvtepu8_epi16(q4_hi);

    __m256i q8_even16;
    __m256i q8_odd16;
    load_q8_even_odd_16(q8, &q8_even16, &q8_odd16);

    const __m256i prod_lo = _mm256_madd_epi16(q4_lo16, q8_even16);
    const __m256i prod_hi = _mm256_madd_epi16(q4_hi16, q8_odd16);

    return hsum256_epi32(prod_lo) + hsum256_epi32(prod_hi);
}
#endif

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
    /* TODO: Implement AVX2 version with correct Q4_K memory layout.
     * For now, fall back to reference implementation which has been
     * fixed to use the correct layout.
     */
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
}
