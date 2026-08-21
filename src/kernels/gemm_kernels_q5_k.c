/**
 * @file gemm_kernels_q5_k.c
 * @brief GEMM/GEMV kernels with Q5_K quantized weights
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
 * Implements matrix multiplication where:
 *   - Activations (input): FP32 (quantized internally to Q8_K for dot path)
 *   - Weights: Q5_K (5-bit super-block quant)
 *   - Output: FP32
 *
 * Q5_K Format (256 weights per super-block):
 *   - d: FP16 super-block scale
 *   - dmin: FP16 super-block minimum
 *   - scales[12]: 8 sub-block scales + 8 sub-block mins (6 bits each, packed)
 *   - qh[32]: high bits for 256 weights (1 bit each)
 *   - qs[128]: low 4 bits for 256 weights (4 bits each)
 *
 * Total: 2 + 2 + 12 + 32 + 128 = 176 bytes per 256 weights = 5.5 bits/weight
 *
 * Dequantization formula (matches llama.cpp):
 *   w = d * scale * q - dmin * mins
 *   where q = qs_val | (qh_bit << 4) = 5-bit value [0, 31]
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "ckernel_quant.h"

/* Include SIMD headers based on available extensions */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__)
#include <immintrin.h>
#endif

/* Q5_K constants */
#define QK_K 256
#define CK_Q5K_STACK_Q8_BLOCKS 128

static int ck_q5k_debug_fp32_fallback(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("CK_DEBUG_Q5K_FP32_FALLBACK");
        cached = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return cached;
}

static int ck_q5k_debug_generic_dot(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("CK_DEBUG_Q5K_GENERIC_DOT");
        cached = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return cached;
}

/* Q5_K block definition is required by this kernel file.
 * Keep a local ggml-compatible layout to decouple from shared headers. */
typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} block_q5_K;

/* Load-time representation used by the optional prepared prefill provider.
 * It expands only integer metadata: FP16 super-block scales are retained
 * verbatim, while sub-block scales/mins and 5-bit codes become byte-addressable.
 * The dot-product and FP32 reduction order remain unchanged. */
typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t scales[8];
    uint8_t mins[8];
    uint8_t qs[QK_K];
} block_q5_K_prepared;

_Static_assert(sizeof(block_q5_K_prepared) == 276,
               "Q5_K prepared-size contract changed");

/* Unpack 8 per-subblock scales and mins from packed Q5_K scale bytes.
 * This mirrors the packing contract used by llama.cpp. */
static inline void unpack_q5_k_scales(const uint8_t *scales,
                                      uint8_t *sc,
                                      uint8_t *m) {
    sc[0] = scales[0] & 0x3F;
    sc[1] = scales[1] & 0x3F;
    sc[2] = scales[2] & 0x3F;
    sc[3] = scales[3] & 0x3F;

    m[0] = scales[4] & 0x3F;
    m[1] = scales[5] & 0x3F;
    m[2] = scales[6] & 0x3F;
    m[3] = scales[7] & 0x3F;

    sc[4] = (scales[8]  & 0x0F) | ((scales[0] >> 6) << 4);
    sc[5] = (scales[9]  & 0x0F) | ((scales[1] >> 6) << 4);
    sc[6] = (scales[10] & 0x0F) | ((scales[2] >> 6) << 4);
    sc[7] = (scales[11] & 0x0F) | ((scales[3] >> 6) << 4);

    m[4] = (scales[8]  >> 4) | ((scales[4] >> 6) << 4);
    m[5] = (scales[9]  >> 4) | ((scales[5] >> 6) << 4);
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4);
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4);
}

static inline uint8_t q5_k_quant_value(const block_q5_K *block, int subblock, int i) {
    const uint8_t *ql = block->qs + (subblock / 2) * 32;
    const uint8_t low = (subblock & 1) ? (uint8_t)(ql[i] >> 4) : (uint8_t)(ql[i] & 0x0F);
    const uint8_t high = (block->qh[i] & (uint8_t)(1u << subblock)) ? 16u : 0u;
    return (uint8_t)(low | high);
}

size_t ck_q5_k_prepared_block_size(void)
{
    return sizeof(block_q5_K_prepared);
}

void ck_q5_k_prepare_weight(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) return;
    const block_q5_K *input = (const block_q5_K *)src;
    block_q5_K_prepared *output = (block_q5_K_prepared *)dst;
    const size_t blocks = (size_t)N * (size_t)(K / QK_K);
    for (size_t b = 0; b < blocks; ++b) {
        output[b].d = input[b].d;
        output[b].dmin = input[b].dmin;
        unpack_q5_k_scales(input[b].scales, output[b].scales, output[b].mins);
        for (int sb = 0; sb < 8; ++sb) {
            for (int i = 0; i < 32; ++i) {
                output[b].qs[sb * 32 + i] = q5_k_quant_value(&input[b], sb, i);
            }
        }
    }
}

/* quantize_row_q8_k() is implemented in gemm_kernels_q4k_q8k.c */
void quantize_row_q8_k(const float *x, void *vy, int k);

#if defined(__AVX2__)
static inline __m256i ck_q5k_scale_shuffle_avx2(int i)
{
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *)(const void *)(k_shuffle + 32 * i));
}

static inline __m256i ck_mm256_set_m128i(__m128i hi, __m128i lo)
{
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline float ck_q5k_hsum256_ps(__m256 v)
{
    __m128 sum = _mm256_extractf128_ps(v, 1);
    sum = _mm_add_ps(sum, _mm256_castps256_ps128(v));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

static float dot_q5_k_q8_k_row_avx2(const block_q5_K *w, const block_q8_K *x, int nb) {
    static const uint32_t kmask1 = 0x3f3f3f3fU;
    static const uint32_t kmask2 = 0x0f0f0f0fU;
    static const uint32_t kmask3 = 0x03030303U;

    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);

    uint32_t utmp[4] = {0, 0, 0, 0};
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const block_q5_K *wb = &w[b];
        const block_q8_K *xb = &x[b];
        const uint8_t *q5 = wb->qs;
        const int8_t *q8 = xb->qs;

        const float d = CK_FP16_TO_FP32(wb->d) * xb->d;
        const float dmin = -CK_FP16_TO_FP32(wb->dmin) * xb->d;

        memcpy(utmp, wb->scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)(const void *)xb->bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * (float)_mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = ck_mm256_set_m128i(sc128, sc128);
        const __m256i hbits = _mm256_loadu_si256((const __m256i *)(const void *)wb->qh);
        __m256i hmask = mone;
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, ck_q5k_scale_shuffle_avx2(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, ck_q5k_scale_shuffle_avx2(2 * j + 1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i *)(const void *)q5);
            q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)(const void *)q8);
            q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)(const void *)q8);
            q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return ck_q5k_hsum256_ps(acc) + summs;
}

static void dot_q5_k_q8_k_rows4_avx2(
        const block_q5_K *w,
        const block_q8_K *const x[4],
        int rows,
        int nb,
        float out[4])
{
    static const uint32_t kmask1 = 0x3f3f3f3fU;
    static const uint32_t kmask2 = 0x0f0f0f0fU;
    static const uint32_t kmask3 = 0x03030303U;
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    float summs[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int b = 0; b < nb; ++b) {
        const block_q5_K *wb = &w[b];
        uint32_t utmp[4] = {0, 0, 0, 0};
        memcpy(utmp, wb->scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) |
                  (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) |
                  (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32((int)utmp[3], (int)utmp[2],
                          (int)utmp[1], (int)utmp[0]));
        const __m128i sc128 =
            _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = ck_mm256_set_m128i(sc128, sc128);
        const __m256i hbits = _mm256_loadu_si256(
            (const __m256i *)(const void *)wb->qh);
        __m256i sumi[4] = {
            _mm256_setzero_si256(), _mm256_setzero_si256(),
            _mm256_setzero_si256(), _mm256_setzero_si256(),
        };

        for (int row = 0; row < rows; ++row) {
            const block_q8_K *xb = &x[row][b];
            const __m256i q8sums = _mm256_loadu_si256(
                (const __m256i *)(const void *)xb->bsums);
            const __m128i q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(
                _mm256_extracti128_si256(mins_and_scales, 1), q8s);
            const __m128i hsum = _mm_hadd_epi32(
                _mm_hadd_epi32(prod, mzero), mzero);
            const float dmin = -CK_FP16_TO_FP32(wb->dmin) * xb->d;
            summs[row] += dmin * (float)_mm_extract_epi32(hsum, 0);
        }

        const uint8_t *q5 = wb->qs;
        __m256i hmask = mone;
        int bit = 0;
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(
                scales, ck_q5k_scale_shuffle_avx2(2 * j));
            const __m256i scale_1 = _mm256_shuffle_epi8(
                scales, ck_q5k_scale_shuffle_avx2(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256(
                (const __m256i *)(const void *)q5);
            q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(
                _mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            for (int row = 0; row < rows; ++row) {
                const block_q8_K *xb = &x[row][b];
                const __m256i q8_0 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)&xb->qs[j * 64]);
                const __m256i q8_1 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)&xb->qs[j * 64 + 32]);
                __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
                p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                p16_1 = _mm256_madd_epi16(scale_1, p16_1);
                sumi[row] = _mm256_add_epi32(
                    sumi[row], _mm256_add_epi32(p16_0, p16_1));
            }
        }

        for (int row = 0; row < rows; ++row) {
            const float d = CK_FP16_TO_FP32(wb->d) * x[row][b].d;
            acc[row] = _mm256_fmadd_ps(
                _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[row]), acc[row]);
        }
    }

    for (int row = 0; row < rows; ++row) {
        out[row] = ck_q5k_hsum256_ps(acc[row]) + summs[row];
    }
}

static float dot_q5_k_prepared_q8_k_row_avx2(
        const block_q5_K_prepared *w, const block_q8_K *x, int nb)
{
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    for (int b = 0; b < nb; ++b) {
        const block_q5_K_prepared *wb = &w[b];
        const block_q8_K *xb = &x[b];
        const float d = CK_FP16_TO_FP32(wb->d) * xb->d;
        const float dmin = -CK_FP16_TO_FP32(wb->dmin) * xb->d;

        const __m128i mins8 = _mm_loadl_epi64((const __m128i *)(const void *)wb->mins);
        const __m256i mins16 = _mm256_cvtepu8_epi16(mins8);
        const __m256i q8sums = _mm256_loadu_si256((const __m256i *)(const void *)xb->bsums);
        const __m128i q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(
                _mm256_castsi256_si128(mins16), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * (float)_mm_extract_epi32(hsum, 0);

        __m256i sumi = _mm256_setzero_si256();
        for (int sb = 0; sb < 8; ++sb) {
            const __m256i q5 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(wb->qs + sb * 32));
            const __m256i q8 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(xb->qs + sb * 32));
            __m256i p16 = _mm256_maddubs_epi16(q5, q8);
            const __m256i scale = _mm256_set1_epi16((int16_t)wb->scales[sb]);
            p16 = _mm256_madd_epi16(scale, p16);
            sumi = _mm256_add_epi32(sumi, p16);
        }
        acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return ck_q5k_hsum256_ps(acc) + summs;
}

static void dot_q5_k_prepared_q8_k_m4_avx2(
        const block_q5_K_prepared *w,
        const block_q8_K *const x[4],
        int rows, int nb, float out[4])
{
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    float summs[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int b = 0; b < nb; ++b) {
        const block_q5_K_prepared *wb = &w[b];
        const __m128i mins8 = _mm_loadl_epi64((const __m128i *)(const void *)wb->mins);
        const __m256i mins16 = _mm256_cvtepu8_epi16(mins8);
        __m256i sumi[4] = {
            _mm256_setzero_si256(), _mm256_setzero_si256(),
            _mm256_setzero_si256(), _mm256_setzero_si256(),
        };

        for (int r = 0; r < rows; ++r) {
            const block_q8_K *xb = &x[r][b];
            const __m256i q8sums = _mm256_loadu_si256(
                    (const __m256i *)(const void *)xb->bsums);
            const __m128i q8s = _mm_hadd_epi16(
                    _mm256_extracti128_si256(q8sums, 0),
                    _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(
                    _mm256_castsi256_si128(mins16), q8s);
            const __m128i hsum = _mm_hadd_epi32(
                    _mm_hadd_epi32(prod, mzero), mzero);
            const float dmin = -CK_FP16_TO_FP32(wb->dmin) * xb->d;
            summs[r] += dmin * (float)_mm_extract_epi32(hsum, 0);
        }

        for (int sb = 0; sb < 8; ++sb) {
            const __m256i q5 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(wb->qs + sb * 32));
            const __m256i scale = _mm256_set1_epi16((int16_t)wb->scales[sb]);
            for (int r = 0; r < rows; ++r) {
                const __m256i q8 = _mm256_loadu_si256(
                        (const __m256i *)(const void *)(x[r][b].qs + sb * 32));
                __m256i p16 = _mm256_maddubs_epi16(q5, q8);
                p16 = _mm256_madd_epi16(scale, p16);
                sumi[r] = _mm256_add_epi32(sumi[r], p16);
            }
        }
        for (int r = 0; r < rows; ++r) {
            const float d = CK_FP16_TO_FP32(wb->d) * x[r][b].d;
            acc[r] = _mm256_fmadd_ps(
                    _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r] = ck_q5k_hsum256_ps(acc[r]) + summs[r];
    }
}
#endif

/* Llama-compatible dot path: Q5_K weights x Q8_K activations for a full row.
 * Keep the eight lane sums live across all blocks, matching ggml's generic
 * Q5_K/Q8_K reduction order. Reducing each block to a scalar first is close,
 * but can move borderline logits in long decode parity tests. */
static float dot_q5_k_q8_k_row(const block_q5_K *w, const block_q8_K *x, int nb) {
#if defined(__AVX2__)
    if (!ck_q5k_debug_generic_dot()) {
        return dot_q5_k_q8_k_row_avx2(w, x, nb);
    }
#endif

    static const uint32_t kmask1 = 0x3f3f3f3fU;
    static const uint32_t kmask2 = 0x0f0f0f0fU;
    static const uint32_t kmask3 = 0x03030303U;

    uint32_t utmp[4] = {0, 0, 0, 0};
    const uint8_t *scales = (const uint8_t *)&utmp[0];
    const uint8_t *mins = (const uint8_t *)&utmp[2];

    int8_t aux8[QK_K];
    int16_t aux16[8];
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int32_t aux32[8];

    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q5_K *wb = &w[b];
        const block_q8_K *xb = &x[b];
        const uint8_t *q4 = wb->qs;
        const uint8_t *hm = wb->qh;
        int8_t *a = aux8;
        uint8_t m = 1;
        memset(aux32, 0, sizeof(aux32));

        for (int j = 0; j < QK_K / 64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32;
            m <<= 1;

            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32;
            m <<= 1;

            q4 += 32;
        }

        memcpy(utmp, wb->scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K / 16; ++j) {
            sumi += (int)xb->bsums[j] * (int)mins[j / 2];
        }

        a = aux8;
        const int8_t *q8 = xb->qs;
        int is = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            const int32_t scale = (int32_t)scales[is++];

            for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;

            for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;

            for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;

            for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }

        const float d = CK_FP16_TO_FP32(wb->d) * xb->d;
        for (int l = 0; l < 8; ++l) {
            sums[l] += d * (float)aux32[l];
        }
        const float dmin = CK_FP16_TO_FP32(wb->dmin) * xb->d;
        sumf -= dmin * (float)sumi;
    }

    for (int l = 0; l < 8; ++l) {
        sumf += sums[l];
    }
    return sumf;
}

/* FP32 fallback for oversized K (very rare for current models). */
static void gemv_q5_k_ref_fp32(float *y, const void *W, const float *x, int M, int K)
{
    const block_q5_K *blocks = (const block_q5_K *)W;
    const int blocks_per_row = K / QK_K;

    for (int m = 0; m < M; m++) {
        const float *x_row = x;
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const block_q5_K *block = &blocks[m * blocks_per_row + b];
            const float d = CK_FP16_TO_FP32(block->d);
            const float dmin = CK_FP16_TO_FP32(block->dmin);
            uint8_t sc_arr[8], m_arr[8];
            unpack_q5_k_scales(block->scales, sc_arr, m_arr);

            for (int sb = 0; sb < 8; sb++) {
                const float d_sub = d * (float)sc_arr[sb];
                const float m_sub = dmin * (float)m_arr[sb];

                for (int i = 0; i < 32; i++) {
                    const uint8_t q = q5_k_quant_value(block, sb, i);
                    sum += (d_sub * (float)q - m_sub) * x_row[b * QK_K + sb * 32 + i];
                }
            }
        }

        y[m] = sum;
    }
}

static void gemm_nt_q5_k_ref_fp32(const float *A,
                                  const void *B,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    const block_q5_K *blocks = (const block_q5_K *)B;
    const int blocks_per_col = K / QK_K;

    for (int m = 0; m < M; m++) {
        const float *a_row = &A[m * K];

        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            const block_q5_K *w_row = &blocks[n * blocks_per_col];
            for (int b = 0; b < blocks_per_col; b++) {
                const block_q5_K *block = &w_row[b];
                const float d = CK_FP16_TO_FP32(block->d);
                const float dmin = CK_FP16_TO_FP32(block->dmin);
                uint8_t sc_arr[8], m_arr[8];
                unpack_q5_k_scales(block->scales, sc_arr, m_arr);

                for (int sb = 0; sb < 8; sb++) {
                    const float d_sub = d * (float)sc_arr[sb];
                    const float m_sub = dmin * (float)m_arr[sb];

                    for (int i = 0; i < 32; i++) {
                        const uint8_t q = q5_k_quant_value(block, sb, i);
                        sum += (d_sub * (float)q - m_sub) * a_row[b * QK_K + sb * 32 + i];
                    }
                }
            }

            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

/* ============================================================================
 * Q5_K x Q8_K Kernels (explicit contract)
 *
 * WHY THESE EXIST:
 *   llama.cpp's Q5_K matmul contract is "Q5_K weights x Q8_K activations".
 *   The activation quantization is part of the numerical contract, not just
 *   an optimization. If we accidentally do FP32 activation dot here, we can
 *   get large parity drift at attn_proj/mlp_down while tests still pass if
 *   they compare against FP32-dequant references.
 *
 *   These entry points make the contract explicit in code:
 *     - gemv_q5_k_q8_k(): decode-style matrix-vector (single token)
 *     - gemm_nt_q5_k_q8_k(): prefill-style matrix-matrix
 * ============================================================================ */

void gemv_q5_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K)
{
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }
    if (K % QK_K != 0) {
        return;
    }

    const block_q5_K *blocks = (const block_q5_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int m = 0; m < M; ++m) {
        const block_q5_K *w_row = &blocks[m * blocks_per_row];
        y[m] = dot_q5_k_q8_k_row(w_row, x, blocks_per_row);
    }
}

void gemm_nt_q5_k_q8_k_ref(const void *A_q8,
                           const void *B,
                           const float *bias,
                           float *C,
                           int M, int N, int K)
{
    if (!A_q8 || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }
    if (K % QK_K != 0) {
        return;
    }

    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q5_K *W = (const block_q5_K *)B;
    const int blocks_per_row = K / QK_K;

    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = &A[m * blocks_per_row];
        for (int n = 0; n < N; ++n) {
            const block_q5_K *w_row = &W[n * blocks_per_row];
            const float sum = dot_q5_k_q8_k_row(w_row, a_row, blocks_per_row);
            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
}

void gemm_nt_q5_k_prepared(const float *A,
                           const void *B_prepared,
                           const float *bias,
                           float *C,
                           int M, int N, int K)
{
#if !defined(__AVX2__)
    (void)A; (void)B_prepared; (void)bias; (void)C;
    (void)M; (void)N; (void)K;
#else
    if (!A || !B_prepared || !C || M <= 0 || N <= 0 || K <= 0 ||
            (K % QK_K) != 0) return;
    const int blocks_per_row = K / QK_K;
    if (blocks_per_row > CK_Q5K_STACK_Q8_BLOCKS) return;
    const block_q5_K_prepared *W = (const block_q5_K_prepared *)B_prepared;
    for (int m = 0; m < M; ++m) {
        block_q8_K a_q8[blocks_per_row];
        quantize_row_q8_k(A + (size_t)m * K, a_q8, K);
        for (int n = 0; n < N; ++n) {
            const float sum = dot_q5_k_prepared_q8_k_row_avx2(
                    W + (size_t)n * blocks_per_row, a_q8, blocks_per_row);
            C[(size_t)m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    }
#endif
}

void gemm_nt_q5_k_prepared_m4(const float *A,
                              const void *B_prepared,
                              const float *bias,
                              float *C,
                              int M, int N, int K)
{
#if !defined(__AVX2__)
    (void)A; (void)B_prepared; (void)bias; (void)C;
    (void)M; (void)N; (void)K;
#else
    if (!A || !B_prepared || !C || M <= 0 || N <= 0 || K <= 0 ||
            (K % QK_K) != 0) return;
    const int blocks_per_row = K / QK_K;
    if (blocks_per_row > CK_Q5K_STACK_Q8_BLOCKS) return;
    const block_q5_K_prepared *W = (const block_q5_K_prepared *)B_prepared;
    for (int m = 0; m < M; m += 4) {
        const int rows = M - m < 4 ? M - m : 4;
        block_q8_K a_q8[4][blocks_per_row];
        const block_q8_K *row_ptrs[4] = {
            a_q8[0], a_q8[1], a_q8[2], a_q8[3],
        };
        for (int r = 0; r < rows; ++r) {
            quantize_row_q8_k(A + (size_t)(m + r) * K, a_q8[r], K);
        }
        for (int n = 0; n < N; ++n) {
            float sums[4];
            dot_q5_k_prepared_q8_k_m4_avx2(
                    W + (size_t)n * blocks_per_row,
                    row_ptrs, rows, blocks_per_row, sums);
            for (int r = 0; r < rows; ++r) {
                C[(size_t)(m + r) * N + n] = sums[r] + (bias ? bias[n] : 0.0f);
            }
        }
    }
#endif
}

void gemm_nt_q5_k_prepared_q8_m4_nrange(const void *A_q8,
                                         const void *B_prepared,
                                         const float *bias,
                                         float *C,
                                         int M, int N, int K,
                                         int n_begin, int n_end)
{
#if !defined(__AVX2__)
    (void)A_q8; (void)B_prepared; (void)bias; (void)C;
    (void)M; (void)N; (void)K; (void)n_begin; (void)n_end;
#else
    if (!A_q8 || !B_prepared || !C || M <= 0 || N <= 0 || K <= 0 ||
            (K % QK_K) != 0 || n_begin < 0 || n_end > N ||
            n_begin >= n_end) return;
    const int blocks_per_row = K / QK_K;
    if (blocks_per_row > CK_Q5K_STACK_Q8_BLOCKS) return;
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q5_K_prepared *W =
        (const block_q5_K_prepared *)B_prepared;

    for (int m = 0; m < M; m += 4) {
        const int rows = M - m < 4 ? M - m : 4;
        const block_q8_K *row_ptrs[4] = {
            A + (size_t)(m + 0) * blocks_per_row,
            A + (size_t)(m + (rows > 1 ? 1 : 0)) * blocks_per_row,
            A + (size_t)(m + (rows > 2 ? 2 : 0)) * blocks_per_row,
            A + (size_t)(m + (rows > 3 ? 3 : 0)) * blocks_per_row,
        };
        for (int n = n_begin; n < n_end; ++n) {
            float sums[4];
            dot_q5_k_prepared_q8_k_m4_avx2(
                W + (size_t)n * blocks_per_row,
                row_ptrs, rows, blocks_per_row, sums);
            for (int r = 0; r < rows; ++r) {
                C[(size_t)(m + r) * N + n] =
                    sums[r] + (bias ? bias[n] : 0.0f);
            }
        }
    }
#endif
}

/* ============================================================================
 * FP32 adapter path (keeps existing call sites stable)
 *
 * Existing generated code and orchestration call gemv_q5_k/gemm_nt_q5_k with
 * FP32 activations. These adapter functions quantize activations to Q8_K and
 * then call the explicit Q5_K x Q8_K kernels above.
 * ============================================================================ */

void gemv_q5_k_ref(float *y, const void *W, const float *x, int M, int K)
{
    if (!y || !W || !x || M <= 0 || K <= 0) {
        return;
    }
    if (ck_q5k_debug_fp32_fallback()) {
        gemv_q5_k_ref_fp32(y, W, x, M, K);
        return;
    }
    if (K % QK_K != 0) {
        gemv_q5_k_ref_fp32(y, W, x, M, K);
        return;
    }

    const block_q5_K *blocks = (const block_q5_K *)W;
    const int blocks_per_row = K / QK_K;
    if (blocks_per_row > CK_Q5K_STACK_Q8_BLOCKS) {
        gemv_q5_k_ref_fp32(y, W, x, M, K);
        return;
    }

    block_q8_K x_q8[CK_Q5K_STACK_Q8_BLOCKS];
    /* Q8_K bytes are part of the numerical ABI. Use the shared provider,
     * whose FP-contraction policy is validated against llama.cpp. */
    quantize_row_q8_k(x, x_q8, K);
    gemv_q5_k_q8_k_ref(y, blocks, x_q8, M, K);
}

/* ============================================================================
 * GEMM NT Reference: C = A @ B^T + bias
 *   - A: FP32 activation matrix [M, K] (quantized internally to Q8_K per row)
 *   - B: Q5_K weight matrix [N, K] (stored transposed, accessed as [N, K])
 *   - bias: Optional FP32 bias [N]
 *   - C: FP32 output matrix [M, N]
 * ============================================================================ */

void gemm_nt_q5_k_ref(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }
    if (ck_q5k_debug_fp32_fallback()) {
        gemm_nt_q5_k_ref_fp32(A, B, bias, C, M, N, K);
        return;
    }
    if (K % QK_K != 0) {
        gemm_nt_q5_k_ref_fp32(A, B, bias, C, M, N, K);
        return;
    }

    const block_q5_K *blocks = (const block_q5_K *)B;
    const int blocks_per_col = K / QK_K;
    if (blocks_per_col > CK_Q5K_STACK_Q8_BLOCKS) {
        gemm_nt_q5_k_ref_fp32(A, B, bias, C, M, N, K);
        return;
    }

    for (int m = 0; m < M; ++m) {
        const float *a_row = &A[m * K];
        block_q8_K a_q8[CK_Q5K_STACK_Q8_BLOCKS];
        quantize_row_q8_k(a_row, a_q8, K);
        gemm_nt_q5_k_q8_k_ref(a_q8, blocks, bias, &C[m * N], 1, N, K);
    }
}

/* ============================================================================
 * Dispatch wrappers - select best available implementation
 * ============================================================================ */

void gemv_q5_k_q8_k(float *y,
                    const void *W,
                    const void *x_q8,
                    int M, int K)
{
#if defined(__AVX512F__)
    /* TODO: AVX-512 implementation */
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#elif defined(__AVX2__)
    /* TODO: AVX-2 implementation */
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#elif defined(__AVX__)
    /* TODO: AVX implementation */
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#elif defined(__SSE4_1__)
    /* TODO: SSE4.1 implementation */
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#else
    gemv_q5_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}

void gemm_nt_q5_k_q8_k(const void *A_q8,
                       const void *B,
                       const float *bias,
                       float *C,
                       int M, int N, int K)
{
#if defined(__AVX512F__)
    /* TODO: AVX-512 implementation */
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#elif defined(__AVX2__)
    /* TODO: AVX-2 implementation */
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#elif defined(__AVX__)
    /* TODO: AVX implementation */
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#elif defined(__SSE4_1__)
    /* TODO: SSE4.1 implementation */
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#else
    gemm_nt_q5_k_q8_k_ref(A_q8, B, bias, C, M, N, K);
#endif
}

void gemm_q5_k_q8_k_compact_rows4(float *output,
                                   int output_stride,
                                   const void *weights,
                                   const void *const input_rows[4],
                                   int rows,
                                   int output_dim,
                                   int input_dim)
{
    if (!output || !weights || !input_rows || rows <= 0 || rows > 4 ||
        output_stride < output_dim || output_dim <= 0 || input_dim <= 0 ||
        (input_dim % QK_K) != 0) {
        return;
    }
    for (int row = 0; row < rows; ++row) {
        if (!input_rows[row]) return;
    }

#if defined(__AVX2__)
    const block_q5_K *blocks = (const block_q5_K *)weights;
    const int blocks_per_row = input_dim / QK_K;
    const block_q8_K *inputs[4] = {
        (const block_q8_K *)input_rows[0],
        (const block_q8_K *)input_rows[rows > 1 ? 1 : 0],
        (const block_q8_K *)input_rows[rows > 2 ? 2 : 0],
        (const block_q8_K *)input_rows[rows > 3 ? 3 : 0],
    };
    for (int n = 0; n < output_dim; ++n) {
        float values[4];
        dot_q5_k_q8_k_rows4_avx2(
            blocks + (size_t)n * (size_t)blocks_per_row,
            inputs, rows, blocks_per_row, values);
        for (int row = 0; row < rows; ++row) {
            output[(size_t)row * (size_t)output_stride + (size_t)n] =
                values[row];
        }
    }
#else
    for (int row = 0; row < rows; ++row) {
        gemv_q5_k_q8_k(
            output + (size_t)row * (size_t)output_stride,
            weights, input_rows[row], output_dim, input_dim);
    }
#endif
}

void gemv_q5_k(float *y, const void *W, const float *x, int M, int K)
{
#if defined(__AVX512F__)
    /* TODO: AVX-512 implementation */
    gemv_q5_k_ref(y, W, x, M, K);
#elif defined(__AVX2__)
    /* TODO: AVX-2 implementation */
    gemv_q5_k_ref(y, W, x, M, K);
#elif defined(__AVX__)
    /* TODO: AVX implementation */
    gemv_q5_k_ref(y, W, x, M, K);
#elif defined(__SSE4_1__)
    /* TODO: SSE4.1 implementation */
    gemv_q5_k_ref(y, W, x, M, K);
#else
    gemv_q5_k_ref(y, W, x, M, K);
#endif
}

void gemm_nt_q5_k(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
#if defined(__AVX512F__)
    /* TODO: AVX-512 implementation */
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#elif defined(__AVX2__)
    /* TODO: AVX-2 implementation */
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#elif defined(__AVX__)
    /* TODO: AVX implementation */
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#elif defined(__SSE4_1__)
    /* TODO: SSE4.1 implementation */
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#else
    gemm_nt_q5_k_ref(A, B, bias, C, M, N, K);
#endif
}
