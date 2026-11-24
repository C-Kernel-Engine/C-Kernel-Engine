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

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CK_Q4K_RUNTIME_AVX2 1
#define CK_Q4K_AVX2_TARGET __attribute__((target("avx2,fma,f16c")))
#endif

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

#if defined(CK_Q4K_RUNTIME_AVX2)
CK_Q4K_AVX2_TARGET
static inline float hsum256_ps_q4k(__m256 v) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

CK_Q4K_AVX2_TARGET
static inline __m256i q4k_scale_pair_shuffle(int pair)
{
    const int lo = 2 * pair;
    return _mm256_set1_epi16((short)(((lo + 1) << 8) | lo));
}

CK_Q4K_AVX2_TARGET
static inline __attribute__((always_inline)) void dot_q4_k_q8_k_rows_avx2(
    const block_q4_K *w, const block_q8_K *x0, const block_q8_K *x1,
    int k, int rows, float *out0, float *out1)
{
    const block_q8_K *inputs[2] = {x0, x1};
    float *outputs[2] = {out0, out1};
    const int nb = k / QK_K;
    __m256 acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    __m128 acc_min[2] = {_mm_setzero_ps(), _mm_setzero_ps()};
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    const uint32_t mask_6bit = UINT32_C(0x3f3f3f3f);
    const uint32_t mask_4bit = UINT32_C(0x0f0f0f0f);
    const uint32_t mask_2bit = UINT32_C(0x03030303);

    for (int i = 0; i < nb; ++i) {
        uint32_t packed[4];
        memcpy(packed, w[i].scales, 12);
        packed[3] = ((packed[2] >> 4) & mask_4bit) |
                    (((packed[1] >> 6) & mask_2bit) << 4);
        const uint32_t upper_scales = packed[1] & mask_6bit;
        packed[1] = (packed[2] & mask_4bit) |
                    (((packed[0] >> 6) & mask_2bit) << 4);
        packed[2] = upper_scales;
        packed[0] &= mask_6bit;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)packed[3], (int)packed[2], (int)packed[1], (int)packed[0]));
        const __m128i scale_bytes = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(scale_bytes, scale_bytes);
        const __m128 scales_fp32 = _mm_cvtph_ps(_mm_setr_epi16(w[i].d, w[i].dmin, 0, 0, 0, 0, 0, 0));
        const float wd = _mm_cvtss_f32(scales_fp32);
        const float wm = -_mm_cvtss_f32(_mm_shuffle_ps(scales_fp32, scales_fp32, _MM_SHUFFLE(1, 1, 1, 1)));
        __m256i block_sum[2] = {_mm256_setzero_si256(), _mm256_setzero_si256()};

        for (int group = 0; group < QK_K / 64; ++group) {
            const __m256i codes = _mm256_loadu_si256((const __m256i *)&w[i].qs[group * 32]);
            const __m256i q4_lo = _mm256_and_si256(codes, nibble_mask);
            const __m256i q4_hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), nibble_mask);
            const __m256i scale_lo = _mm256_shuffle_epi8(scales, q4k_scale_pair_shuffle(2 * group));
            const __m256i scale_hi = _mm256_shuffle_epi8(scales, q4k_scale_pair_shuffle(2 * group + 1));
            for (int r = 0; r < rows; ++r) {
                const block_q8_K *x = &inputs[r][i];
                const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)&x->qs[group * 64]);
                const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)&x->qs[group * 64 + 32]);
                const __m256i lo = _mm256_madd_epi16(scale_lo, _mm256_maddubs_epi16(q4_lo, q8_lo));
                const __m256i hi = _mm256_madd_epi16(scale_hi, _mm256_maddubs_epi16(q4_hi, q8_hi));
                block_sum[r] = _mm256_add_epi32(block_sum[r], _mm256_add_epi32(lo, hi));
            }
        }
        for (int r = 0; r < rows; ++r) {
            const block_q8_K *x = &inputs[r][i];
            const __m128i sums = _mm_hadd_epi16(
                _mm_loadu_si128((const __m128i *)&x->bsums[0]),
                _mm_loadu_si128((const __m128i *)&x->bsums[8]));
            const __m128i min_products = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), sums);
            acc_min[r] = _mm_fmadd_ps(_mm_set1_ps(wm * x->d), _mm_cvtepi32_ps(min_products), acc_min[r]);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(wd * x->d), _mm256_cvtepi32_ps(block_sum[r]), acc[r]);
        }
    }
    for (int r = 0; r < rows; ++r) {
        __m128 minimum = _mm_add_ps(acc_min[r], _mm_movehl_ps(acc_min[r], acc_min[r]));
        minimum = _mm_add_ss(minimum, _mm_movehdup_ps(minimum));
        *outputs[r] = hsum256_ps_q4k(acc[r]) + _mm_cvtss_f32(minimum);
    }
}

CK_Q4K_AVX2_TARGET
static float dot_q4_k_q8_k_avx2(const block_q4_K *w, const block_q8_K *x, int k)
{
    float out;
    dot_q4_k_q8_k_rows_avx2(w, x, NULL, k, 1, &out, NULL);
    return out;
}

CK_Q4K_AVX2_TARGET
static void dot_q4_k_q8_k_pair_avx2(const block_q4_K *w, const block_q8_K *x0,
                                  const block_q8_K *x1, int k, float *out0, float *out1)
{
    dot_q4_k_q8_k_rows_avx2(w, x0, x1, k, 2, out0, out1);
}
#endif

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
#if defined(CK_Q4K_RUNTIME_AVX2)
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }

    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") ||
        !__builtin_cpu_supports("f16c")) {
        gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
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


void gemv_q4_k_q8_k_pair_checked_cpu(float *out0, float *out1, const void *weights,
                                    const void *x0, const void *x1, int m, int k)
{
#if defined(CK_Q4K_RUNTIME_AVX2)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c")) {
        const block_q4_K *blocks = weights;
        const size_t per_row = (size_t)k / QK_K;
        for (int row = 0; row < m; ++row)
            dot_q4_k_q8_k_pair_avx2(blocks + (size_t)row * per_row, x0, x1, k, out0 + row, out1 + row);
        return;
    }
#endif
    gemv_q4_k_q8_k_ref(out0, weights, x0, m, k);
    gemv_q4_k_q8_k_ref(out1, weights, x1, m, k);
}
