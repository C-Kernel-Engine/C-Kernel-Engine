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

#include "ckernel_quant.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

#if defined(__AVX2__)
static inline float hsum256_ps_q4k(__m256 v) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static float dot_q4_k_q8_k_avx2(const block_q4_K *w,
                                const block_q8_K *x,
                                int k)
{
    const int nb = k / QK_K;
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_min = _mm_setzero_ps();
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);

    for (int i = 0; i < nb; ++i) {
        uint8_t sc[8], m_val[8];
        unpack_q4_k_scales(w[i].scales, sc, m_val);

        const float d = CK_FP16_TO_FP32(w[i].d) * x[i].d;
        const float dmin = -CK_FP16_TO_FP32(w[i].dmin) * x[i].d;

        const __m128i mins = _mm_loadl_epi64((const __m128i *)m_val);
        const __m128i q8_sums = _mm_hadd_epi16(
                _mm_loadu_si128((const __m128i *)&x[i].bsums[0]),
                _mm_loadu_si128((const __m128i *)&x[i].bsums[8]));
        const __m128i min_products = _mm_madd_epi16(_mm_cvtepu8_epi16(mins), q8_sums);
        acc_min = _mm_fmadd_ps(
                _mm_set1_ps(dmin), _mm_cvtepi32_ps(min_products), acc_min);

        __m256i block_sum = _mm256_setzero_si256();
        for (int group = 0; group < QK_K / 64; ++group) {
            const __m256i packed = _mm256_loadu_si256(
                    (const __m256i *)&w[i].qs[group * 32]);
            const __m256i q4_lo = _mm256_and_si256(packed, nibble_mask);
            const __m256i q4_hi = _mm256_and_si256(
                    _mm256_srli_epi16(packed, 4), nibble_mask);
            const __m256i q8_lo = _mm256_loadu_si256(
                    (const __m256i *)&x[i].qs[group * 64]);
            const __m256i q8_hi = _mm256_loadu_si256(
                    (const __m256i *)&x[i].qs[group * 64 + 32]);
            __m256i lo = _mm256_maddubs_epi16(q4_lo, q8_lo);
            __m256i hi = _mm256_maddubs_epi16(q4_hi, q8_hi);
            lo = _mm256_madd_epi16(_mm256_set1_epi16(sc[2 * group]), lo);
            hi = _mm256_madd_epi16(_mm256_set1_epi16(sc[2 * group + 1]), hi);
            block_sum = _mm256_add_epi32(block_sum, _mm256_add_epi32(lo, hi));
        }
        acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d), _mm256_cvtepi32_ps(block_sum), acc);
    }

    acc_min = _mm_add_ps(acc_min, _mm_movehl_ps(acc_min, acc_min));
    acc_min = _mm_add_ss(acc_min, _mm_movehdup_ps(acc_min));
    return hsum256_ps_q4k(acc) + _mm_cvtss_f32(acc_min);
}
#endif

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
#if defined(__AVX2__)
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        y[row] = dot_q4_k_q8_k_avx2(w_row, x, K);
    }
#else
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}
