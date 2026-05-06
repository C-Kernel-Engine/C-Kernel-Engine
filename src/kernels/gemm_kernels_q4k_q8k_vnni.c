/**
 * @file gemm_kernels_q4k_q8k_vnni.c
 * @brief VNNI Q4_K x Q8_K matvec kernel (inference only)
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
 * Requires AVX512-VNNI for vpdpbusd instruction.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ckernel_quant.h"

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#include <immintrin.h>
#endif

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K);

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static inline int32_t dot_q4_k_q8_k_32_vnni(const uint8_t *q4_packed_32,
                                            const int8_t *q8_32,
                                            int high_nibble) {
    const __m256i packed = _mm256_loadu_si256((const __m256i *)q4_packed_32);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i q4_bytes = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
    const __m256i q8_bytes = _mm256_loadu_si256((const __m256i *)q8_32);
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4_bytes, q8_bytes);
    return hsum256_epi32(acc);
}

static inline float dot_q4_k_q8_k_vnni_block(const block_q4_K *w,
                                             const block_q8_K *x) {
    uint8_t sc[8], m_val[8];
    unpack_q4_k_scales(w->scales, sc, m_val);

    const float d = CK_FP16_TO_FP32(w->d) * x->d;
    const float dmin = CK_FP16_TO_FP32(w->dmin) * x->d;
    float sumf = 0.0f;

    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const uint8_t *qs = &w->qs[q_offset];
        const int8_t *q8_lo = &x->qs[j];
        const int8_t *q8_hi = &x->qs[j + 32];

        const int32_t sum_lo = dot_q4_k_q8_k_32_vnni(qs, q8_lo, 0);
        const int32_t sum_hi = dot_q4_k_q8_k_32_vnni(qs, q8_hi, 1);
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

        sumf += d * (float)sc[is] * (float)sum_lo;
        sumf -= dmin * (float)m_val[is] * (float)bsum_lo;
        sumf += d * (float)sc[is + 1] * (float)sum_hi;
        sumf -= dmin * (float)m_val[is + 1] * (float)bsum_hi;
    }

    return sumf;
}
#endif

void gemv_q4_k_q8_k_vnni(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    if (!y || !W || !x_q8 || M <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        float sumf = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            sumf += dot_q4_k_q8_k_vnni_block(&w_row[b], &x[b]);
        }
        y[row] = sumf;
    }
#else
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}
