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
 * The canonical providers require AVX2. The output-interleaved provider uses
 * 256-bit VNNI from AVX-VNNI or AVX-512 VNNI+VL.
 *
 * Packed-meta status:
 * Packed layouts are internal providers selected by the declared production
 * dispatcher. Weight-identity caches own their lifetime until runtime shutdown;
 * the kernel map records layout and ISA requirements. Canonical GGUF-layout
 * Q4_K remains the parity fallback for unsupported ISAs and uncovered shapes.
 */

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ckernel_engine.h"
#include "ck_threadpool.h"
#include "ckernel_quant.h"
#include "ck_speed_profiles.h"

static int ck_q4k_x16_chunk4_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("CK_Q4K_X16_CHUNK4");
        cached = env ? ck_env_value_truthy(env) : 1;
    }
    return cached;
}

#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define CK_HAS_AVX_VNNI_256 1
static inline __m256i ck_dpbusd_i32x8(__m256i acc, __m256i lhs, __m256i rhs)
{
    return _mm256_dpbusd_epi32(acc, lhs, rhs);
}
#elif defined(__AVXVNNI__)
#define CK_HAS_AVX_VNNI_256 1
static inline __m256i ck_dpbusd_i32x8(__m256i acc, __m256i lhs, __m256i rhs)
{
    return _mm256_dpbusd_avx_epi32(acc, lhs, rhs);
}
#endif

void gemv_q4_k_q8_k_ref(float *y,
                        const void *W,
                        const void *x_q8,
                        int M, int K);

void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K);

#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVX2__)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
    return _mm_cvtsi128_si32(sum);
}
#endif

#if defined(CK_HAS_AVX_VNNI_256)
static inline int32_t dot_q4_k_q8_k_32_vnni(const uint8_t *q4_packed_32,
                                            const int8_t *q8_32,
                                            int high_nibble) {
    const __m256i q8_bytes = _mm256_loadu_si256((const __m256i *)q8_32);
    const __m256i packed = _mm256_loadu_si256((const __m256i *)q4_packed_32);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i q4_bytes = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
    __m256i acc = _mm256_setzero_si256();
    acc = ck_dpbusd_i32x8(acc, q4_bytes, q8_bytes);
    return hsum256_epi32(acc);
}

static inline int32_t dot_q4_k_q8_k_32_vnni_q8v(const uint8_t *q4_packed_32,
                                                 __m256i q8_bytes,
                                                 int high_nibble) {
    const __m256i packed = _mm256_loadu_si256((const __m256i *)q4_packed_32);
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    const __m256i q4_bytes = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
    __m256i acc = _mm256_setzero_si256();
    acc = ck_dpbusd_i32x8(acc, q4_bytes, q8_bytes);
    return hsum256_epi32(acc);
}

static inline __m256i q4_k_unpack_32_vnni_bytes(__m256i packed, int high_nibble) {
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    return high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
}

static inline int32_t dot_q4_k_q8_k_32_vnni_q4v_q8v(__m256i q4_bytes, __m256i q8_bytes) {
    __m256i acc = _mm256_setzero_si256();
    acc = ck_dpbusd_i32x8(acc, q4_bytes, q8_bytes);
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
        if (j + 128 < QK_K) {
            __builtin_prefetch(&w->qs[q_offset + 64], 0, 1);
            __builtin_prefetch(&x->qs[j + 128], 0, 1);
        }

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

#if defined(__AVX2__)
static inline __m256i q4_k_unpack_32_avx2_bytes(__m256i packed, int high_nibble) {
    const __m256i mask4 = _mm256_set1_epi8(0x0F);
    return high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4)
        : _mm256_and_si256(packed, mask4);
}

static inline __m256i dot_q4_k_q8_k_32_avx2_i32x8(__m256i q4_bytes,
                                                   __m256i q8_bytes) {
    const __m256i pair_sums_i16 = _mm256_maddubs_epi16(q4_bytes, q8_bytes);
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(pair_sums_i16, ones);
}

static inline int32_t dot_q4_k_q8_k_32_avx2_q4v_q8v(__m256i q4_bytes,
                                                     __m256i q8_bytes) {
    const __m256i sums_i32 = dot_q4_k_q8_k_32_avx2_i32x8(q4_bytes, q8_bytes);
    return hsum256_epi32(sums_i32);
}

static inline __m256i dot_q4_k_q8_k_32_avx2_q4v_q8v_scaled_i32x8(__m256i q4_bytes,
                                                                   __m256i q8_bytes,
                                                                   uint8_t scale) {
    const __m256i pair_sums_i16 = _mm256_maddubs_epi16(q4_bytes, q8_bytes);
    const __m256i scale_i16 = _mm256_set1_epi16((int16_t)scale);
    return _mm256_madd_epi16(pair_sums_i16, scale_i16);
}

#endif


typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t sc[8];
    uint8_t m[8];
    uint8_t qs[QK_K];
} block_q4_K_packed_u8;

typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t sc[8];
    uint8_t m[8];
    uint8_t qs[QK_K / 2];
} block_q4_K_packed_meta;

typedef struct {
    ck_half d[8];
    ck_half dmin[8];
    uint8_t sc[8][8];
    uint8_t m[8][8];
    uint8_t qs[8][QK_K / 2];
    uint8_t active;
    uint8_t reserved[7];
} block_q4_K_packed_meta_x8;

typedef struct {
    ck_half d[16];
    ck_half dmin[16];
    uint8_t sc[16][8];
    uint8_t m[16][8];
    uint8_t qs[16][QK_K / 2];
    uint8_t active;
    uint8_t reserved[15];
} block_q4_K_packed_meta_x16;

typedef struct {
    ck_half d[16];
    ck_half dmin[16];
    uint8_t sc[16][8];
    uint8_t m[16][8];
    uint8_t qs[16][QK_K];
    uint8_t active;
    uint8_t reserved[15];
} block_q4_K_packed_u8_x16;

typedef struct {
    ck_half d[8];
    ck_half dmin[8];
    uint8_t sc[8][8];
    uint8_t m[8][8];
    uint8_t qs[(QK_K / 2) * 8];
    uint8_t active;
    uint8_t reserved[7];
} block_q4_K_packed_vnni_x8;

size_t q4_k_packed_vnni_x8_block_size(void)
{
    return sizeof(block_q4_K_packed_vnni_x8);
}

int ck_q4k_packed_vnni_x8_available(void)
{
#if defined(CK_HAS_AVX_VNNI_256)
    return 1;
#else
    return 0;
#endif
}

void pack_q4_k_to_packed_vnni_x8(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_vnni_x8 *out = (block_q4_K_packed_vnni_x8 *)dst;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 7) / 8;
    memset(out, 0,
           (size_t)groups * (size_t)blocks_per_row * sizeof(*out));

    for (int group = 0; group < groups; ++group) {
        const int n0 = group * 8;
        const int active = (n0 + 8 <= N) ? 8 : (N - n0);
        for (int block = 0; block < blocks_per_row; ++block) {
            block_q4_K_packed_vnni_x8 *packed =
                    out + (size_t)group * (size_t)blocks_per_row +
                    (size_t)block;
            packed->active = (uint8_t)active;
            for (int lane = 0; lane < active; ++lane) {
                const block_q4_K *source =
                        in + (size_t)(n0 + lane) * (size_t)blocks_per_row +
                        (size_t)block;
                uint8_t scales[8];
                uint8_t mins[8];
                packed->d[lane] = source->d;
                packed->dmin[lane] = source->dmin;
                unpack_q4_k_scales(source->scales, scales, mins);
                for (int subblock = 0; subblock < 8; ++subblock) {
                    packed->sc[subblock][lane] = scales[subblock];
                    packed->m[subblock][lane] = mins[subblock];
                }
                for (int pair = 0; pair < QK_K / 64; ++pair) {
                    for (int segment = 0; segment < 8; ++segment) {
                        memcpy(packed->qs + (size_t)pair * 256u +
                                               (size_t)segment * 32u +
                                               (size_t)lane * 4u,
                               source->qs + (size_t)pair * 32u +
                                               (size_t)segment * 4u,
                               4u);
                    }
                }
            }
        }
    }
}

#if defined(__AVX2__)
static inline float hsum256_ps_q4k_packed(__m256 v)
{
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

/* Preserve llama.cpp's Q4_K x Q8_K AVX2 arithmetic while reading the x16
 * packed weight layout. Packing changes addresses only; it must not collapse
 * the eight FP32 accumulation lanes or interleave minimum terms into them. */
static inline float dot_q4_k_packed_meta_x16_q8_k_llama_avx2(
        const block_q4_K_packed_meta_x16 *weights,
        int blocks_per_row,
        int lane,
        const block_q8_K *activation)
{
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_min = _mm_setzero_ps();
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);

    for (int block = 0; block < blocks_per_row; ++block) {
        const block_q4_K_packed_meta_x16 *w = &weights[block];
        const block_q8_K *x = &activation[block];
        const float d = CK_FP16_TO_FP32(w->d[lane]) * x->d;
        const float dmin = -CK_FP16_TO_FP32(w->dmin[lane]) * x->d;

        const __m128i mins = _mm_loadl_epi64((const __m128i *)w->m[lane]);
        const __m128i q8_sums = _mm_hadd_epi16(
                _mm_loadu_si128((const __m128i *)&x->bsums[0]),
                _mm_loadu_si128((const __m128i *)&x->bsums[8]));
        const __m128i min_products = _mm_madd_epi16(_mm_cvtepu8_epi16(mins), q8_sums);
        acc_min = _mm_fmadd_ps(
                _mm_set1_ps(dmin), _mm_cvtepi32_ps(min_products), acc_min);

        __m256i block_sum = _mm256_setzero_si256();
        for (int group = 0; group < QK_K / 64; ++group) {
            const __m256i packed = _mm256_loadu_si256(
                    (const __m256i *)&w->qs[lane][group * 32]);
            const __m256i q4_lo = _mm256_and_si256(packed, nibble_mask);
            const __m256i q4_hi = _mm256_and_si256(
                    _mm256_srli_epi16(packed, 4), nibble_mask);
            const __m256i q8_lo = _mm256_loadu_si256(
                    (const __m256i *)&x->qs[group * 64]);
            const __m256i q8_hi = _mm256_loadu_si256(
                    (const __m256i *)&x->qs[group * 64 + 32]);
            __m256i lo = _mm256_maddubs_epi16(q4_lo, q8_lo);
            __m256i hi = _mm256_maddubs_epi16(q4_hi, q8_hi);
            lo = _mm256_madd_epi16(_mm256_set1_epi16(w->sc[lane][2 * group]), lo);
            hi = _mm256_madd_epi16(_mm256_set1_epi16(w->sc[lane][2 * group + 1]), hi);
            block_sum = _mm256_add_epi32(block_sum, _mm256_add_epi32(lo, hi));
        }
        acc = _mm256_fmadd_ps(
                _mm256_set1_ps(d), _mm256_cvtepi32_ps(block_sum), acc);
    }

    acc_min = _mm_add_ps(acc_min, _mm_movehl_ps(acc_min, acc_min));
    acc_min = _mm_add_ss(acc_min, _mm_movehdup_ps(acc_min));
    return hsum256_ps_q4k_packed(acc) + _mm_cvtss_f32(acc_min);
}
#endif

size_t q4_k_packed_u8_block_size(void)
{
    return sizeof(block_q4_K_packed_u8);
}

size_t q4_k_packed_meta_block_size(void)
{
    return sizeof(block_q4_K_packed_meta);
}

size_t q4_k_packed_meta_x8_block_size(void)
{
    return sizeof(block_q4_K_packed_meta_x8);
}

size_t q4_k_packed_meta_x16_block_size(void)
{
    return sizeof(block_q4_K_packed_meta_x16);
}

size_t q4_k_packed_u8_x16_block_size(void)
{
    return sizeof(block_q4_K_packed_u8_x16);
}

void pack_q4_k_to_packed_u8(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_u8 *out = (block_q4_K_packed_u8 *)dst;
    const int blocks_per_row = K / QK_K;
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_K *sb = in + (size_t)n * (size_t)blocks_per_row + (size_t)b;
            block_q4_K_packed_u8 *pb = out + (size_t)n * (size_t)blocks_per_row + (size_t)b;
            pb->d = sb->d;
            pb->dmin = sb->dmin;
            unpack_q4_k_scales(sb->scales, pb->sc, pb->m);
            for (int j = 0, q_offset = 0; j < QK_K; j += 64, q_offset += 32) {
                const uint8_t *qs = &sb->qs[q_offset];
                for (int l = 0; l < 32; ++l) {
                    pb->qs[j + l] = qs[l] & 0x0F;
                    pb->qs[j + 32 + l] = qs[l] >> 4;
                }
            }
        }
    }
}

void pack_q4_k_to_packed_meta(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_meta *out = (block_q4_K_packed_meta *)dst;
    const int blocks_per_row = K / QK_K;
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_K *sb = in + (size_t)n * (size_t)blocks_per_row + (size_t)b;
            block_q4_K_packed_meta *pb = out + (size_t)n * (size_t)blocks_per_row + (size_t)b;
            pb->d = sb->d;
            pb->dmin = sb->dmin;
            unpack_q4_k_scales(sb->scales, pb->sc, pb->m);
            memcpy(pb->qs, sb->qs, sizeof(pb->qs));
        }
    }
}

void pack_q4_k_to_packed_meta_x8(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_meta_x8 *out = (block_q4_K_packed_meta_x8 *)dst;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 7) / 8;
    memset(out, 0, (size_t)groups * (size_t)blocks_per_row * sizeof(*out));

    for (int g = 0; g < groups; ++g) {
        const int n0 = g * 8;
        const int active = (n0 + 8 <= N) ? 8 : (N - n0);
        for (int b = 0; b < blocks_per_row; ++b) {
            block_q4_K_packed_meta_x8 *pb = out + (size_t)g * (size_t)blocks_per_row + (size_t)b;
            pb->active = (uint8_t)active;
            for (int lane = 0; lane < active; ++lane) {
                const block_q4_K *sb = in + (size_t)(n0 + lane) * (size_t)blocks_per_row + (size_t)b;
                pb->d[lane] = sb->d;
                pb->dmin[lane] = sb->dmin;
                unpack_q4_k_scales(sb->scales, pb->sc[lane], pb->m[lane]);
                memcpy(pb->qs[lane], sb->qs, sizeof(pb->qs[lane]));
            }
        }
    }
}

void pack_q4_k_to_packed_meta_x16(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_meta_x16 *out = (block_q4_K_packed_meta_x16 *)dst;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 15) / 16;
    memset(out, 0, (size_t)groups * (size_t)blocks_per_row * sizeof(*out));

    for (int g = 0; g < groups; ++g) {
        const int n0 = g * 16;
        const int active = (n0 + 16 <= N) ? 16 : (N - n0);
        for (int b = 0; b < blocks_per_row; ++b) {
            block_q4_K_packed_meta_x16 *pb = out + (size_t)g * (size_t)blocks_per_row + (size_t)b;
            pb->active = (uint8_t)active;
            for (int lane = 0; lane < active; ++lane) {
                const block_q4_K *sb = in + (size_t)(n0 + lane) * (size_t)blocks_per_row + (size_t)b;
                pb->d[lane] = sb->d;
                pb->dmin[lane] = sb->dmin;
                unpack_q4_k_scales(sb->scales, pb->sc[lane], pb->m[lane]);
                memcpy(pb->qs[lane], sb->qs, sizeof(pb->qs[lane]));
            }
        }
    }
}

void pack_q4_k_to_packed_u8_x16(const void *src, void *dst, int N, int K)
{
    if (!src || !dst || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q4_K *in = (const block_q4_K *)src;
    block_q4_K_packed_u8_x16 *out = (block_q4_K_packed_u8_x16 *)dst;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 15) / 16;
    memset(out, 0, (size_t)groups * (size_t)blocks_per_row * sizeof(*out));

    for (int g = 0; g < groups; ++g) {
        const int n0 = g * 16;
        const int active = (n0 + 16 <= N) ? 16 : (N - n0);
        for (int b = 0; b < blocks_per_row; ++b) {
            block_q4_K_packed_u8_x16 *pb = out + (size_t)g * (size_t)blocks_per_row + (size_t)b;
            pb->active = (uint8_t)active;
            for (int lane = 0; lane < active; ++lane) {
                const block_q4_K *sb = in + (size_t)(n0 + lane) * (size_t)blocks_per_row + (size_t)b;
                pb->d[lane] = sb->d;
                pb->dmin[lane] = sb->dmin;
                unpack_q4_k_scales(sb->scales, pb->sc[lane], pb->m[lane]);
                for (int j = 0, q_offset = 0; j < QK_K; j += 64, q_offset += 32) {
                    const uint8_t *qs = &sb->qs[q_offset];
                    for (int l = 0; l < 32; ++l) {
                        pb->qs[lane][j + l] = qs[l] & 0x0F;
                        pb->qs[lane][j + 32 + l] = qs[l] >> 4;
                    }
                }
            }
        }
    }
}

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
static inline int32_t dot_q4_packed_u8_q8_32_vnni(const uint8_t *q4_32,
                                                   const int8_t *q8_32)
{
    const __m256i q4 = _mm256_loadu_si256((const __m256i *)q4_32);
    const __m256i q8 = _mm256_loadu_si256((const __m256i *)q8_32);
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4, q8);
    return hsum256_epi32(acc);
}

static inline int32_t dot_q4_packed_u8_q8_32_vnni_q8v(const uint8_t *q4_32,
                                                       __m256i q8)
{
    const __m256i q4 = _mm256_loadu_si256((const __m256i *)q4_32);
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, q4, q8);
    return hsum256_epi32(acc);
}
#endif

#if !(defined(__AVX512VNNI__) && defined(__AVX512VL__))
static inline int32_t dot_q4_packed_u8_q8_32_ref(const uint8_t *q4_32,
                                                  const int8_t *q8_32)
{
    int32_t acc = 0;
    for (int i = 0; i < 32; ++i) {
        acc += (int32_t)q4_32[i] * (int32_t)q8_32[i];
    }
    return acc;
}
#endif

static inline float dot_q4_k_packed_u8_q8_k_block(const block_q4_K_packed_u8 *w,
                                                   const block_q8_K *x)
{
    const float d = CK_FP16_TO_FP32(w->d) * x->d;
    const float dmin = CK_FP16_TO_FP32(w->dmin) * x->d;
    float sumf = 0.0f;
    for (int j = 0; j < QK_K; j += 32) {
        const int is = j / 32;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const int32_t sumi = dot_q4_packed_u8_q8_32_vnni(&w->qs[j], &x->qs[j]);
#else
        const int32_t sumi = dot_q4_packed_u8_q8_32_ref(&w->qs[j], &x->qs[j]);
#endif
        const int32_t bsum = (int32_t)x->bsums[j / 16] + (int32_t)x->bsums[j / 16 + 1];
        sumf += d * (float)w->sc[is] * (float)sumi;
        sumf -= dmin * (float)w->m[is] * (float)bsum;
    }
    return sumf;
}

static inline float dot_q4_k_packed_meta_q8_k_block(const block_q4_K_packed_meta *w,
                                                     const block_q8_K *x)
{
    const float d = CK_FP16_TO_FP32(w->d) * x->d;
    const float dmin = CK_FP16_TO_FP32(w->dmin) * x->d;
    float sumf = 0.0f;
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const uint8_t *qs = &w->qs[q_offset];
        const int8_t *q8_lo = &x->qs[j];
        const int8_t *q8_hi = &x->qs[j + 32];

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const int32_t sum_lo = dot_q4_k_q8_k_32_vnni(qs, q8_lo, 0);
        const int32_t sum_hi = dot_q4_k_q8_k_32_vnni(qs, q8_hi, 1);
#else
        int32_t sum_lo = 0;
        int32_t sum_hi = 0;
        for (int l = 0; l < 32; ++l) {
            sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo[l];
            sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi[l];
        }
#endif
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

        sumf += d * (float)w->sc[is] * (float)sum_lo;
        sumf -= dmin * (float)w->m[is] * (float)bsum_lo;
        sumf += d * (float)w->sc[is + 1] * (float)sum_hi;
        sumf -= dmin * (float)w->m[is + 1] * (float)bsum_hi;
    }
    return sumf;
}

static inline void accum_q4_k_packed_meta_x8_q8_k_block(float acc[8],
                                                         const block_q4_K_packed_meta_x8 *w,
                                                         int active,
                                                         const block_q8_K *x)
{
    const float xd = x->d;
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const int8_t *q8_lo_ptr = &x->qs[j];
        const int8_t *q8_hi_ptr = &x->qs[j + 32];
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#elif defined(__AVX2__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#endif

        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_lo, 0);
            const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_hi, 1);
#elif defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i q4_lo = q4_k_unpack_32_avx2_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_avx2_bytes(packed, 1);
            const int32_t sum_lo = dot_q4_k_q8_k_32_avx2_q4v_q8v(q4_lo, q8_lo);
            const int32_t sum_hi = dot_q4_k_q8_k_32_avx2_q4v_q8v(q4_hi, q8_hi);
#else
            int32_t sum_lo = 0;
            int32_t sum_hi = 0;
            for (int l = 0; l < 32; ++l) {
                sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo_ptr[l];
                sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi_ptr[l];
            }
#endif
            const float d = CK_FP16_TO_FP32(w->d[lane]) * xd;
            const float dmin = CK_FP16_TO_FP32(w->dmin[lane]) * xd;
            acc[lane] += d * (float)w->sc[lane][is] * (float)sum_lo;
            acc[lane] -= dmin * (float)w->m[lane][is] * (float)bsum_lo;
            acc[lane] += d * (float)w->sc[lane][is + 1] * (float)sum_hi;
            acc[lane] -= dmin * (float)w->m[lane][is + 1] * (float)bsum_hi;
        }
    }
}

/* Match the loaded-model Q4_Kx8 contract: each pair of 32-element subblocks
 * is reduced in integer arithmetic, followed by one FP32 value FMA and one
 * minimum FMA. The two FP32 accumulators remain separate until the end. */
static inline void accum_q4_k_packed_meta_x8_q8_k_superblock(
        float acc[8], float acc_min[8],
        const block_q4_K_packed_meta_x8 *w, int active,
        const block_q8_K *x)
{
    const float xd = x->d;
#if defined(__AVX2__)
    float d[8] = {0};
    float dmin[8] = {0};
    for (int lane = 0; lane < active; ++lane) {
        d[lane] = CK_FP16_TO_FP32(w->d[lane]);
        dmin[lane] = CK_FP16_TO_FP32(w->dmin[lane]);
    }
    const __m256 scale = _mm256_mul_ps(_mm256_loadu_ps(d), _mm256_set1_ps(xd));
    const __m256 min_scale = _mm256_mul_ps(_mm256_loadu_ps(dmin), _mm256_set1_ps(xd));
#endif

    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        int32_t iacc[8] = {0};
        int32_t iacc_min[8] = {0};
        const int8_t *q8_lo_ptr = &x->qs[j];
        const int8_t *q8_hi_ptr = &x->qs[j + 32];
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#elif defined(__AVX2__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#endif
        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_lo, 0);
            const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_hi, 1);
#elif defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i q4_lo = q4_k_unpack_32_avx2_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_avx2_bytes(packed, 1);
            const int32_t sum_lo = dot_q4_k_q8_k_32_avx2_q4v_q8v(q4_lo, q8_lo);
            const int32_t sum_hi = dot_q4_k_q8_k_32_avx2_q4v_q8v(q4_hi, q8_hi);
#else
            int32_t sum_lo = 0;
            int32_t sum_hi = 0;
            for (int l = 0; l < 32; ++l) {
                sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo_ptr[l];
                sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi_ptr[l];
            }
#endif
            iacc[lane] = (int32_t)w->sc[lane][is] * sum_lo +
                         (int32_t)w->sc[lane][is + 1] * sum_hi;
            iacc_min[lane] = (int32_t)w->m[lane][is] * bsum_lo +
                             (int32_t)w->m[lane][is + 1] * bsum_hi;
        }

#if defined(__AVX2__)
    const __m256 acc_vec = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)iacc)),
            scale,
            _mm256_loadu_ps(acc));
    const __m256 acc_min_vec = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)iacc_min)),
            min_scale,
            _mm256_loadu_ps(acc_min));
    _mm256_storeu_ps(acc, acc_vec);
    _mm256_storeu_ps(acc_min, acc_min_vec);
#else
    for (int lane = 0; lane < active; ++lane) {
        const float d = CK_FP16_TO_FP32(w->d[lane]) * xd;
        const float dmin = CK_FP16_TO_FP32(w->dmin[lane]) * xd;
        acc[lane] = fmaf((float)iacc[lane], d, acc[lane]);
        acc_min[lane] = fmaf((float)iacc_min[lane], dmin, acc_min[lane]);
    }
#endif
    }
}

/* Multi-row form of the exact repacked-matmul contract. Q4 unpacking is shared
 * across rows, while every output retains the same ascending subblock order,
 * separate value/minimum accumulators, and final subtraction as the scalar-row
 * provider. Callers currently use four or eight rows to measure the reuse versus
 * register-pressure tradeoff without changing the numerical contract. */
static inline void accum_q4_k_packed_meta_x8_q8_k_superblock_rows(
        float acc[8][8], float acc_min[8][8],
        const block_q4_K_packed_meta_x8 *w, int active,
        const block_q8_K *x[8], int rows)
{
#if defined(__AVX2__)
    float wd[8] = {0};
    float wdmin[8] = {0};
    for (int lane = 0; lane < active; ++lane) {
        wd[lane] = CK_FP16_TO_FP32(w->d[lane]);
        wdmin[lane] = CK_FP16_TO_FP32(w->dmin[lane]);
    }
    const __m256 weight_scale = _mm256_loadu_ps(wd);
    const __m256 weight_min_scale = _mm256_loadu_ps(wdmin);

    for (int j = 0, is = 0, q_offset = 0;
         j < QK_K; j += 64, is += 2, q_offset += 32) {
        int32_t iacc[8][8] = {{0}};
        int32_t iacc_min[8][8] = {{0}};
        __m256i q8_lo[8];
        __m256i q8_hi[8];
        int32_t bsum_lo[8];
        int32_t bsum_hi[8];

        for (int row = 0; row < rows; ++row) {
            q8_lo[row] = _mm256_loadu_si256((const __m256i *)&x[row]->qs[j]);
            q8_hi[row] = _mm256_loadu_si256((const __m256i *)&x[row]->qs[j + 32]);
            bsum_lo[row] = (int32_t)x[row]->bsums[j / 16] +
                            (int32_t)x[row]->bsums[j / 16 + 1];
            bsum_hi[row] = (int32_t)x[row]->bsums[(j + 32) / 16] +
                            (int32_t)x[row]->bsums[(j + 32) / 16 + 1];
        }

        for (int lane = 0; lane < active; ++lane) {
            const __m256i packed = _mm256_loadu_si256(
                    (const __m256i *)&w->qs[lane][q_offset]);
#if defined(CK_HAS_AVX_VNNI_256)
            const __m256i q4_lo = q4_k_unpack_32_vnni_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_vnni_bytes(packed, 1);
#else
            const __m256i q4_lo = q4_k_unpack_32_avx2_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_avx2_bytes(packed, 1);
#endif
            const int32_t scale_lo = (int32_t)w->sc[lane][is];
            const int32_t scale_hi = (int32_t)w->sc[lane][is + 1];
            const int32_t min_lo = (int32_t)w->m[lane][is];
            const int32_t min_hi = (int32_t)w->m[lane][is + 1];

            for (int row = 0; row < rows; ++row) {
#if defined(CK_HAS_AVX_VNNI_256)
                const int32_t sum_lo =
                        dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_lo, q8_lo[row]);
                const int32_t sum_hi =
                        dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_hi, q8_hi[row]);
#else
                const int32_t sum_lo =
                        dot_q4_k_q8_k_32_avx2_q4v_q8v(q4_lo, q8_lo[row]);
                const int32_t sum_hi =
                        dot_q4_k_q8_k_32_avx2_q4v_q8v(q4_hi, q8_hi[row]);
#endif
                iacc[row][lane] = scale_lo * sum_lo + scale_hi * sum_hi;
                iacc_min[row][lane] =
                        min_lo * bsum_lo[row] + min_hi * bsum_hi[row];
            }
        }

        for (int row = 0; row < rows; ++row) {
            const __m256 row_scale = _mm256_set1_ps(x[row]->d);
            const __m256 value = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(
                            _mm256_loadu_si256((const __m256i *)iacc[row])),
                    _mm256_mul_ps(weight_scale, row_scale),
                    _mm256_loadu_ps(acc[row]));
            const __m256 minimum = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(
                            _mm256_loadu_si256((const __m256i *)iacc_min[row])),
                    _mm256_mul_ps(weight_min_scale, row_scale),
                    _mm256_loadu_ps(acc_min[row]));
            _mm256_storeu_ps(acc[row], value);
            _mm256_storeu_ps(acc_min[row], minimum);
        }
    }
#else
    for (int row = 0; row < rows; ++row) {
        accum_q4_k_packed_meta_x8_q8_k_superblock(
                acc[row], acc_min[row], w, active, x[row]);
    }
#endif
}

/* VNNI-native 4M x 8N microkernel. Q4 bytes are interleaved by output lane,
 * allowing each vpdpbusd lane to accumulate one output column. The standard
 * Q8_K row remains unchanged; four activation bytes are broadcast to all
 * output lanes. Float updates preserve the accepted pairwise split-min order. */
static inline void accum_q4_k_packed_vnni_x8_q8_k_4m_superblock(
        float acc[4][8], float acc_min[4][8],
        const block_q4_K_packed_vnni_x8 *w,
        const block_q8_K *x[4], int rows)
{
#if defined(CK_HAS_AVX_VNNI_256)
    float wd[8];
    float wdmin[8];
    for (int lane = 0; lane < 8; ++lane) {
        wd[lane] = CK_FP16_TO_FP32(w->d[lane]);
        wdmin[lane] = CK_FP16_TO_FP32(w->dmin[lane]);
    }
    const __m256 weight_scale = _mm256_loadu_ps(wd);
    const __m256 weight_min_scale = _mm256_loadu_ps(wdmin);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);

    for (int pair = 0; pair < QK_K / 64; ++pair) {
        const int j = pair * 64;
        const int is = pair * 2;
        __m256i sum_lo[4];
        __m256i sum_hi[4];
        for (int row = 0; row < rows; ++row) {
            sum_lo[row] = _mm256_setzero_si256();
            sum_hi[row] = _mm256_setzero_si256();
        }

        for (int segment = 0; segment < 8; ++segment) {
            const __m256i packed = _mm256_loadu_si256(
                    (const __m256i *)(w->qs + (size_t)pair * 256u +
                                      (size_t)segment * 32u));
            const __m256i q4_lo = _mm256_and_si256(packed, nibble_mask);
            const __m256i q4_hi = _mm256_and_si256(
                    _mm256_srli_epi16(packed, 4), nibble_mask);
            for (int row = 0; row < rows; ++row) {
                int32_t q8_lo_word;
                int32_t q8_hi_word;
                memcpy(&q8_lo_word, x[row]->qs + j + segment * 4,
                       sizeof(q8_lo_word));
                memcpy(&q8_hi_word, x[row]->qs + j + 32 + segment * 4,
                       sizeof(q8_hi_word));
                sum_lo[row] = ck_dpbusd_i32x8(
                        sum_lo[row], q4_lo, _mm256_set1_epi32(q8_lo_word));
                sum_hi[row] = ck_dpbusd_i32x8(
                        sum_hi[row], q4_hi, _mm256_set1_epi32(q8_hi_word));
            }
        }

        const __m256i scale_lo = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)w->sc[is]));
        const __m256i scale_hi = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)w->sc[is + 1]));
        const __m256i min_lo = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)w->m[is]));
        const __m256i min_hi = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64((const __m128i *)w->m[is + 1]));

        for (int row = 0; row < rows; ++row) {
            const __m256i weighted = _mm256_add_epi32(
                    _mm256_mullo_epi32(sum_lo[row], scale_lo),
                    _mm256_mullo_epi32(sum_hi[row], scale_hi));
            const int32_t bsum_lo =
                    (int32_t)x[row]->bsums[j / 16] +
                    (int32_t)x[row]->bsums[j / 16 + 1];
            const int32_t bsum_hi =
                    (int32_t)x[row]->bsums[(j + 32) / 16] +
                    (int32_t)x[row]->bsums[(j + 32) / 16 + 1];
            const __m256i weighted_min = _mm256_add_epi32(
                    _mm256_mullo_epi32(min_lo, _mm256_set1_epi32(bsum_lo)),
                    _mm256_mullo_epi32(min_hi, _mm256_set1_epi32(bsum_hi)));
            const __m256 row_scale = _mm256_set1_ps(x[row]->d);
            const __m256 value = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(weighted),
                    _mm256_mul_ps(weight_scale, row_scale),
                    _mm256_loadu_ps(acc[row]));
            const __m256 minimum = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(weighted_min),
                    _mm256_mul_ps(weight_min_scale, row_scale),
                    _mm256_loadu_ps(acc_min[row]));
            _mm256_storeu_ps(acc[row], value);
            _mm256_storeu_ps(acc_min[row], minimum);
        }
    }
#else
    (void)acc;
    (void)acc_min;
    (void)w;
    (void)x;
    (void)rows;
#endif
}

/* The repacked GEMV provider has a distinct reduction boundary from GEMM:
 * all four 64-element pairs in one Q4_K block are combined in int32 before
 * one value FMA and one minimum FMA update the FP32 accumulators. */
static inline void accum_q4_k_packed_meta_x8_q8_k_gemv_block(
        float acc[8], float acc_min[8],
        const block_q4_K_packed_meta_x8 *w, int active,
        const block_q8_K *x)
{
    int32_t iacc[8] = {0};
    int32_t iacc_min[8] = {0};
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const int8_t *q8_lo_ptr = &x->qs[j];
        const int8_t *q8_hi_ptr = &x->qs[j + 32];
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#elif defined(__AVX2__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#endif
        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_lo, 0);
            const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_hi, 1);
#elif defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i q4_lo = q4_k_unpack_32_avx2_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_avx2_bytes(packed, 1);
            const int32_t sum_lo = dot_q4_k_q8_k_32_avx2_q4v_q8v(q4_lo, q8_lo);
            const int32_t sum_hi = dot_q4_k_q8_k_32_avx2_q4v_q8v(q4_hi, q8_hi);
#else
            int32_t sum_lo = 0;
            int32_t sum_hi = 0;
            for (int l = 0; l < 32; ++l) {
                sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo_ptr[l];
                sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi_ptr[l];
            }
#endif
            iacc[lane] += (int32_t)w->sc[lane][is] * sum_lo +
                           (int32_t)w->sc[lane][is + 1] * sum_hi;
            iacc_min[lane] += (int32_t)w->m[lane][is] * bsum_lo +
                               (int32_t)w->m[lane][is + 1] * bsum_hi;
        }
    }

#if defined(__AVX2__)
    float d[8] = {0};
    float dmin[8] = {0};
    for (int lane = 0; lane < active; ++lane) {
        d[lane] = CK_FP16_TO_FP32(w->d[lane]);
        dmin[lane] = CK_FP16_TO_FP32(w->dmin[lane]);
    }
    const __m256 xd = _mm256_set1_ps(x->d);
    const __m256 acc_vec = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)iacc)),
            _mm256_mul_ps(_mm256_loadu_ps(d), xd), _mm256_loadu_ps(acc));
    const __m256 min_vec = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)iacc_min)),
            _mm256_mul_ps(_mm256_loadu_ps(dmin), xd), _mm256_loadu_ps(acc_min));
    _mm256_storeu_ps(acc, acc_vec);
    _mm256_storeu_ps(acc_min, min_vec);
#else
    for (int lane = 0; lane < active; ++lane) {
        const float d = CK_FP16_TO_FP32(w->d[lane]) * x->d;
        const float dmin = CK_FP16_TO_FP32(w->dmin[lane]) * x->d;
        acc[lane] = fmaf((float)iacc[lane], d, acc[lane]);
        acc_min[lane] = fmaf((float)iacc_min[lane], dmin, acc_min[lane]);
    }
#endif
}




static inline void accum_q4_k_packed_meta_x8_q8_k_block_mreuse(float acc[8][8],
                                                                const block_q4_K_packed_meta_x8 *w,
                                                                int active,
                                                                const block_q8_K *A,
                                                                int blocks_per_vec,
                                                                int block_index,
                                                                int m0,
                                                                int m_count)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        for (int lane0 = 0; lane0 < active; lane0 += 4) {
            const int lanes = (lane0 + 4 <= active) ? 4 : (active - lane0);
            __m256i q4_lo[4];
            __m256i q4_hi[4];
            float wd[4], wdmin[4], sc_lo[4], sc_hi[4], min_lo[4], min_hi[4];

            for (int l = 0; l < lanes; ++l) {
                const int lane = lane0 + l;
                const uint8_t *qs = &w->qs[lane][q_offset];
                const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
                q4_lo[l] = q4_k_unpack_32_vnni_bytes(packed, 0);
                q4_hi[l] = q4_k_unpack_32_vnni_bytes(packed, 1);
                wd[l] = CK_FP16_TO_FP32(w->d[lane]);
                wdmin[l] = CK_FP16_TO_FP32(w->dmin[lane]);
                sc_lo[l] = (float)w->sc[lane][is];
                sc_hi[l] = (float)w->sc[lane][is + 1];
                min_lo[l] = (float)w->m[lane][is];
                min_hi[l] = (float)w->m[lane][is + 1];
            }

            for (int mt = 0; mt < m_count; ++mt) {
                const block_q8_K *x = A + (size_t)(m0 + mt) * (size_t)blocks_per_vec + (size_t)block_index;
                const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                        (int32_t)x->bsums[j / 16 + 1];
                const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                        (int32_t)x->bsums[(j + 32) / 16 + 1];
                const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)&x->qs[j]);
                const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)&x->qs[j + 32]);
                const float xd = x->d;
                for (int l = 0; l < lanes; ++l) {
                    const int lane = lane0 + l;
                    const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_lo[l], q8_lo);
                    const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_hi[l], q8_hi);
                    const float d = wd[l] * xd;
                    const float dmin = wdmin[l] * xd;
                    acc[mt][lane] += d * sc_lo[l] * (float)sum_lo;
                    acc[mt][lane] -= dmin * min_lo[l] * (float)bsum_lo;
                    acc[mt][lane] += d * sc_hi[l] * (float)sum_hi;
                    acc[mt][lane] -= dmin * min_hi[l] * (float)bsum_hi;
                }
            }
        }
    }
#else
    for (int mt = 0; mt < m_count; ++mt) {
        const block_q8_K *x = A + (size_t)(m0 + mt) * (size_t)blocks_per_vec + (size_t)block_index;
        accum_q4_k_packed_meta_x8_q8_k_block(acc[mt], w, active, x);
    }
#endif
}


static inline void accum_q4_k_packed_u8_x16_q8_k_block(float acc[16],
                                                        const block_q4_K_packed_u8_x16 *w,
                                                        int active,
                                                        const block_q8_K *x)
{
    const float xd = x->d;
    for (int j = 0; j < QK_K; j += 32) {
        const int is = j / 32;
        const int8_t *q8_ptr = &x->qs[j];
        const int32_t bsum = (int32_t)x->bsums[j / 16] + (int32_t)x->bsums[j / 16 + 1];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const __m256i q8 = _mm256_loadu_si256((const __m256i *)q8_ptr);
#endif
        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *q4 = &w->qs[lane][j];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const int32_t sumi = dot_q4_packed_u8_q8_32_vnni_q8v(q4, q8);
#else
            const int32_t sumi = dot_q4_packed_u8_q8_32_ref(q4, q8_ptr);
#endif
            const float d = CK_FP16_TO_FP32(w->d[lane]) * xd;
            const float dmin = CK_FP16_TO_FP32(w->dmin[lane]) * xd;
            acc[lane] += d * (float)w->sc[lane][is] * (float)sumi;
            acc[lane] -= dmin * (float)w->m[lane][is] * (float)bsum;
        }
    }
}


static inline void accum_q4_k_packed_meta_x16_q8_k_block_mreuse(float acc[8][16],
                                                                 const block_q4_K_packed_meta_x16 *w,
                                                                 int active,
                                                                 const block_q8_K *A,
                                                                 int blocks_per_vec,
                                                                 int block_index,
                                                                 int m0,
                                                                 int m_count)
{
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i q4_lo = q4_k_unpack_32_vnni_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_vnni_bytes(packed, 1);
#elif defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i q4_lo = q4_k_unpack_32_avx2_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_avx2_bytes(packed, 1);
#endif
            const float wd = CK_FP16_TO_FP32(w->d[lane]);
            const float wdmin = CK_FP16_TO_FP32(w->dmin[lane]);
#if !defined(__AVX2__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
            const float sc_lo = (float)w->sc[lane][is];
            const float sc_hi = (float)w->sc[lane][is + 1];
#endif
            const float min_lo = (float)w->m[lane][is];
            const float min_hi = (float)w->m[lane][is + 1];

            for (int mt = 0; mt < m_count; ++mt) {
                const block_q8_K *x = A + (size_t)(m0 + mt) * (size_t)blocks_per_vec + (size_t)block_index;
                const int8_t *q8_lo_ptr = &x->qs[j];
                const int8_t *q8_hi_ptr = &x->qs[j + 32];
                const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                        (int32_t)x->bsums[j / 16 + 1];
                const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                        (int32_t)x->bsums[(j + 32) / 16 + 1];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
                const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
                const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_lo, q8_lo);
                const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_hi, q8_hi);
#elif defined(__AVX2__)
                const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
                const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
                const __m256i sum_lo_v = dot_q4_k_q8_k_32_avx2_q4v_q8v_scaled_i32x8(q4_lo, q8_lo, w->sc[lane][is]);
                const __m256i sum_hi_v = dot_q4_k_q8_k_32_avx2_q4v_q8v_scaled_i32x8(q4_hi, q8_hi, w->sc[lane][is + 1]);
                const int32_t sum_scaled = hsum256_epi32(_mm256_add_epi32(sum_lo_v, sum_hi_v));
#else
                int32_t sum_lo = 0;
                int32_t sum_hi = 0;
                for (int l = 0; l < 32; ++l) {
                    sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo_ptr[l];
                    sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi_ptr[l];
                }
#endif
                const float xd = x->d;
                const float d = wd * xd;
                const float dmin = wdmin * xd;
#if defined(__AVX2__) && !(defined(__AVX512VNNI__) && defined(__AVX512VL__))
                acc[mt][lane] += d * (float)sum_scaled;
                acc[mt][lane] -= dmin * min_lo * (float)bsum_lo;
                acc[mt][lane] -= dmin * min_hi * (float)bsum_hi;
#else
                acc[mt][lane] += d * sc_lo * (float)sum_lo;
                acc[mt][lane] -= dmin * min_lo * (float)bsum_lo;
                acc[mt][lane] += d * sc_hi * (float)sum_hi;
                acc[mt][lane] -= dmin * min_hi * (float)bsum_hi;
#endif
            }
        }
    }
}

static inline void accum_q4_k_packed_meta_x16_q8_k_block_mreuse_chunk4(float acc[8][16],
                                                                        const block_q4_K_packed_meta_x16 *w,
                                                                        int active,
                                                                        const block_q8_K *A,
                                                                        int blocks_per_vec,
                                                                        int block_index,
                                                                        int m0,
                                                                        int m_count)
{
#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVX2__)
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        for (int lane0 = 0; lane0 < active; lane0 += 4) {
            const int lanes = (lane0 + 4 <= active) ? 4 : (active - lane0);
            __m256i q4_lo[4];
            __m256i q4_hi[4];
            float wd[4], wdmin[4], sc_lo[4], sc_hi[4], min_lo[4], min_hi[4];

            for (int l = 0; l < lanes; ++l) {
                const int lane = lane0 + l;
                const uint8_t *qs = &w->qs[lane][q_offset];
                const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                q4_lo[l] = q4_k_unpack_32_vnni_bytes(packed, 0);
                q4_hi[l] = q4_k_unpack_32_vnni_bytes(packed, 1);
#else
                q4_lo[l] = q4_k_unpack_32_avx2_bytes(packed, 0);
                q4_hi[l] = q4_k_unpack_32_avx2_bytes(packed, 1);
#endif
                wd[l] = CK_FP16_TO_FP32(w->d[lane]);
                wdmin[l] = CK_FP16_TO_FP32(w->dmin[lane]);
                sc_lo[l] = (float)w->sc[lane][is];
                sc_hi[l] = (float)w->sc[lane][is + 1];
                min_lo[l] = (float)w->m[lane][is];
                min_hi[l] = (float)w->m[lane][is + 1];
            }

            for (int mt = 0; mt < m_count; ++mt) {
                const block_q8_K *x = A + (size_t)(m0 + mt) * (size_t)blocks_per_vec + (size_t)block_index;
                const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                        (int32_t)x->bsums[j / 16 + 1];
                const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                        (int32_t)x->bsums[(j + 32) / 16 + 1];
                const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)&x->qs[j]);
                const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)&x->qs[j + 32]);
                const float xd = x->d;
                for (int l = 0; l < lanes; ++l) {
                    const int lane = lane0 + l;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                    const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_lo[l], q8_lo);
                    const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q4v_q8v(q4_hi[l], q8_hi);
#else
                    const __m256i sum_lo_v = dot_q4_k_q8_k_32_avx2_q4v_q8v_scaled_i32x8(q4_lo[l], q8_lo, w->sc[lane][is]);
                    const __m256i sum_hi_v = dot_q4_k_q8_k_32_avx2_q4v_q8v_scaled_i32x8(q4_hi[l], q8_hi, w->sc[lane][is + 1]);
                    const int32_t sum_scaled = hsum256_epi32(_mm256_add_epi32(sum_lo_v, sum_hi_v));
#endif
                    const float d = wd[l] * xd;
                    const float dmin = wdmin[l] * xd;
#if defined(__AVX2__) && !(defined(__AVX512VNNI__) && defined(__AVX512VL__))
                    acc[mt][lane] += d * (float)sum_scaled;
                    acc[mt][lane] -= dmin * min_lo[l] * (float)bsum_lo;
                    acc[mt][lane] -= dmin * min_hi[l] * (float)bsum_hi;
#else
                    acc[mt][lane] += d * sc_lo[l] * (float)sum_lo;
                    acc[mt][lane] -= dmin * min_lo[l] * (float)bsum_lo;
                    acc[mt][lane] += d * sc_hi[l] * (float)sum_hi;
                    acc[mt][lane] -= dmin * min_hi[l] * (float)bsum_hi;
#endif
                }
            }
        }
    }
#else
    accum_q4_k_packed_meta_x16_q8_k_block_mreuse(acc, w, active, A, blocks_per_vec, block_index, m0, m_count);
#endif
}

static inline void accum_q4_k_packed_meta_x16_q8_k_block(float acc[16],
                                                          const block_q4_K_packed_meta_x16 *w,
                                                          int active,
                                                          const block_q8_K *x)
{
    const float xd = x->d;
    for (int j = 0, is = 0, q_offset = 0; j < QK_K; j += 64, is += 2, q_offset += 32) {
        const int8_t *q8_lo_ptr = &x->qs[j];
        const int8_t *q8_hi_ptr = &x->qs[j + 32];
        const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                (int32_t)x->bsums[j / 16 + 1];
        const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                (int32_t)x->bsums[(j + 32) / 16 + 1];

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#elif defined(__AVX2__)
        const __m256i q8_lo = _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
        const __m256i q8_hi = _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#endif

        for (int lane = 0; lane < active; ++lane) {
            const uint8_t *qs = &w->qs[lane][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            const int32_t sum_lo = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_lo, 0);
            const int32_t sum_hi = dot_q4_k_q8_k_32_vnni_q8v(qs, q8_hi, 1);
#elif defined(__AVX2__)
            const __m256i packed = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i q4_lo = q4_k_unpack_32_avx2_bytes(packed, 0);
            const __m256i q4_hi = q4_k_unpack_32_avx2_bytes(packed, 1);
            const __m256i sum_lo_v = dot_q4_k_q8_k_32_avx2_q4v_q8v_scaled_i32x8(q4_lo, q8_lo, w->sc[lane][is]);
            const __m256i sum_hi_v = dot_q4_k_q8_k_32_avx2_q4v_q8v_scaled_i32x8(q4_hi, q8_hi, w->sc[lane][is + 1]);
            const int32_t sum_scaled = hsum256_epi32(_mm256_add_epi32(sum_lo_v, sum_hi_v));
#else
            int32_t sum_lo = 0;
            int32_t sum_hi = 0;
            for (int l = 0; l < 32; ++l) {
                sum_lo += (int32_t)(qs[l] & 0x0F) * (int32_t)q8_lo_ptr[l];
                sum_hi += (int32_t)(qs[l] >> 4) * (int32_t)q8_hi_ptr[l];
            }
#endif
            const float d = CK_FP16_TO_FP32(w->d[lane]) * xd;
            const float dmin = CK_FP16_TO_FP32(w->dmin[lane]) * xd;
#if defined(__AVX2__) && !(defined(__AVX512VNNI__) && defined(__AVX512VL__))
            acc[lane] += d * (float)sum_scaled;
            acc[lane] -= dmin * (float)w->m[lane][is] * (float)bsum_lo;
            acc[lane] -= dmin * (float)w->m[lane][is + 1] * (float)bsum_hi;
#else
            acc[lane] += d * (float)w->sc[lane][is] * (float)sum_lo;
            acc[lane] -= dmin * (float)w->m[lane][is] * (float)bsum_lo;
            acc[lane] += d * (float)w->sc[lane][is + 1] * (float)sum_hi;
            acc[lane] -= dmin * (float)w->m[lane][is + 1] * (float)bsum_hi;
#endif
        }
    }
}

void gemm_nt_q4_k_packed_u8_q8_k(const void *A_q8,
                                  const void *B_packed,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_u8 *W = (const block_q4_K_packed_u8 *)B_packed;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;
    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; ++n) {
            const block_q4_K_packed_u8 *w_row = W + (size_t)n * (size_t)blocks_per_row;
            float sum = bias ? bias[n] : 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q4_k_packed_u8_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}


void gemm_nt_q4_k_packed_meta_q8_k(const void *A_q8,
                                    const void *B_packed,
                                    const float *bias,
                                    float *C,
                                    int M, int N, int K)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta *W = (const block_q4_K_packed_meta *)B_packed;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;
    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; ++n) {
            const block_q4_K_packed_meta *w_row = W + (size_t)n * (size_t)blocks_per_row;
            float sum = bias ? bias[n] : 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q4_k_packed_meta_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}

void gemm_nt_q4_k_packed_meta_q8_k_tile(const void *A_q8,
                                         const void *B_packed,
                                         const float *bias,
                                         float *C,
                                         int M, int N, int K,
                                         int m0, int m1, int n0, int n1)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    if (m0 < 0) m0 = 0;
    if (n0 < 0) n0 = 0;
    if (m1 > M) m1 = M;
    if (n1 > N) n1 = N;
    if (m0 >= m1 || n0 >= n1) {
        return;
    }

    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta *W = (const block_q4_K_packed_meta *)B_packed;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;

    for (int m = m0; m < m1; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int n = n0; n < n1; ++n) {
            const block_q4_K_packed_meta *w_row = W + (size_t)n * (size_t)blocks_per_row;
            float sum = bias ? bias[n] : 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q4_k_packed_meta_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}

void gemm_nt_q4_k_packed_meta_x8_q8_k(const void *A_q8,
                                       const void *B_packed_x8,
                                       const float *bias,
                                       float *C,
                                       int M, int N, int K)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta_x8 *W = (const block_q4_K_packed_meta_x8 *)B_packed_x8;
    const int blocks_per_vec = K / QK_K;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 7) / 8;

    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_vec;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int g = 0; g < groups; ++g) {
            const int n0 = g * 8;
            const int active = (n0 + 8 <= N) ? 8 : (N - n0);
            float acc[8];
            for (int lane = 0; lane < active; ++lane) {
                acc[lane] = bias ? bias[n0 + lane] : 0.0f;
            }
            for (int b = 0; b < blocks_per_row; ++b) {
                const block_q4_K_packed_meta_x8 *w_group =
                    W + (size_t)g * (size_t)blocks_per_row + (size_t)b;
                accum_q4_k_packed_meta_x8_q8_k_block(acc, w_group, active, &a_row[b]);
            }
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[lane];
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_superblock_order(
        const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
        int M, int N, int K)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta_x8 *W = (const block_q4_K_packed_meta_x8 *)B_packed_x8;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 7) / 8;

    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_row;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int g = 0; g < groups; ++g) {
            const int n0 = g * 8;
            const int active = (n0 + 8 <= N) ? 8 : (N - n0);
            float acc[8] = {0};
            float acc_min[8] = {0};
            for (int b = 0; b < blocks_per_row; ++b) {
                const block_q4_K_packed_meta_x8 *w_group =
                        W + (size_t)g * (size_t)blocks_per_row + (size_t)b;
                accum_q4_k_packed_meta_x8_q8_k_superblock(
                        acc, acc_min, w_group, active, &a_row[b]);
            }
            float values[8];
#if defined(__AVX2__)
            _mm256_storeu_ps(values, _mm256_sub_ps(_mm256_loadu_ps(acc), _mm256_loadu_ps(acc_min)));
#else
            for (int lane = 0; lane < active; ++lane) {
                values[lane] = acc[lane] - acc_min[lane];
            }
#endif
            for (int lane = 0; lane < active; ++lane) {
                float value = values[lane];
                if (bias) {
                    value += bias[n0 + lane];
                }
                c_row[n0 + lane] = value;
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
        const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
        int M, int N, int K)
{
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 ||
        (K % QK_K) != 0 || (N % 16) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta_x8 *W =
            (const block_q4_K_packed_meta_x8 *)B_packed_x8;
    const int blocks_per_row = K / QK_K;

    for (int row = 0; row < M; ++row) {
        const block_q8_K *a_row = A + (size_t)row * (size_t)blocks_per_row;
        float *c_row = C + (size_t)row * (size_t)N;
        for (int n0 = 0; n0 < N; n0 += 16) {
            const int group0 = n0 / 8;
            __m512 acc = _mm512_setzero_ps();
            __m512 acc_min = _mm512_setzero_ps();

            for (int b = 0; b < blocks_per_row; ++b) {
                const block_q4_K_packed_meta_x8 *w0 =
                        W + (size_t)group0 * (size_t)blocks_per_row + (size_t)b;
                const block_q4_K_packed_meta_x8 *w1 =
                        W + (size_t)(group0 + 1) * (size_t)blocks_per_row + (size_t)b;
                const block_q8_K *x = &a_row[b];
                float d[16];
                float dmin[16];
                for (int lane = 0; lane < 16; ++lane) {
                    const block_q4_K_packed_meta_x8 *w = lane < 8 ? w0 : w1;
                    const int wl = lane & 7;
                    d[lane] = CK_FP16_TO_FP32(w->d[wl]);
                    dmin[lane] = CK_FP16_TO_FP32(w->dmin[wl]);
                }
                const __m512 scale =
                        _mm512_mul_ps(_mm512_loadu_ps(d), _mm512_set1_ps(x->d));
                const __m512 min_scale =
                        _mm512_mul_ps(_mm512_loadu_ps(dmin), _mm512_set1_ps(x->d));

                for (int j = 0, is = 0, q_offset = 0;
                     j < QK_K;
                     j += 64, is += 2, q_offset += 32) {
                    int32_t iacc[16];
                    int32_t iacc_min[16];
                    const int8_t *q8_lo_ptr = &x->qs[j];
                    const int8_t *q8_hi_ptr = &x->qs[j + 32];
                    const int32_t bsum_lo = (int32_t)x->bsums[j / 16] +
                                            (int32_t)x->bsums[j / 16 + 1];
                    const int32_t bsum_hi = (int32_t)x->bsums[(j + 32) / 16] +
                                            (int32_t)x->bsums[(j + 32) / 16 + 1];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                    const __m256i q8_lo =
                            _mm256_loadu_si256((const __m256i *)q8_lo_ptr);
                    const __m256i q8_hi =
                            _mm256_loadu_si256((const __m256i *)q8_hi_ptr);
#endif
                    for (int lane = 0; lane < 16; ++lane) {
                        const block_q4_K_packed_meta_x8 *w = lane < 8 ? w0 : w1;
                        const int wl = lane & 7;
                        const uint8_t *qs = &w->qs[wl][q_offset];
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                        const int32_t sum_lo =
                                dot_q4_k_q8_k_32_vnni_q8v(qs, q8_lo, 0);
                        const int32_t sum_hi =
                                dot_q4_k_q8_k_32_vnni_q8v(qs, q8_hi, 1);
#else
                        int32_t sum_lo = 0;
                        int32_t sum_hi = 0;
                        for (int i = 0; i < 32; ++i) {
                            sum_lo += (int32_t)(qs[i] & 0x0F) *
                                      (int32_t)q8_lo_ptr[i];
                            sum_hi += (int32_t)(qs[i] >> 4) *
                                      (int32_t)q8_hi_ptr[i];
                        }
#endif
                        /* llama.cpp's AVX-512 q4_K_8x8 provider keeps each
                         * 32-value dot in an int16 lane through PMADDUBSW and
                         * wrapping VPADDW operations, then widens while
                         * applying the two sub-block scales. */
                        const int16_t packed_sum_lo = (int16_t)sum_lo;
                        const int16_t packed_sum_hi = (int16_t)sum_hi;
                        iacc[lane] = (int32_t)w->sc[wl][is] * (int32_t)packed_sum_lo +
                                     (int32_t)w->sc[wl][is + 1] * (int32_t)packed_sum_hi;
                        iacc_min[lane] = (int32_t)w->m[wl][is] * bsum_lo +
                                         (int32_t)w->m[wl][is + 1] * bsum_hi;
                    }
                    acc = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(_mm512_loadu_si512(iacc)),
                            scale,
                            acc);
                    acc_min = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(_mm512_loadu_si512(iacc_min)),
                            min_scale,
                            acc_min);
                }
            }
            __m512 value = _mm512_sub_ps(acc, acc_min);
            if (bias) {
                value = _mm512_add_ps(value, _mm512_loadu_ps(bias + n0));
            }
            _mm512_storeu_ps(c_row + n0, value);
        }
    }
#else
    gemm_nt_q4_k_packed_meta_x8_q8_k_superblock_order(
            A_q8, B_packed_x8, bias, C, M, N, K);
#endif
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order(
        const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
        int M, int N, int K)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    const block_q8_K *A = (const block_q8_K *)A_q8;
    const block_q4_K_packed_meta_x8 *W = (const block_q4_K_packed_meta_x8 *)B_packed_x8;
    const int blocks_per_row = K / QK_K;
    const int groups = (N + 7) / 8;
    for (int m = 0; m < M; ++m) {
        const block_q8_K *a_row = A + (size_t)m * (size_t)blocks_per_row;
        float *c_row = C + (size_t)m * (size_t)N;
        for (int g = 0; g < groups; ++g) {
            const int n0 = g * 8;
            const int active = (n0 + 8 <= N) ? 8 : (N - n0);
            float acc[8] = {0};
            float acc_min[8] = {0};
            for (int b = 0; b < blocks_per_row; ++b) {
                const block_q4_K_packed_meta_x8 *w_group =
                        W + (size_t)g * (size_t)blocks_per_row + (size_t)b;
                accum_q4_k_packed_meta_x8_q8_k_gemv_block(
                        acc, acc_min, w_group, active, &a_row[b]);
            }
            float values[8];
#if defined(__AVX2__)
            _mm256_storeu_ps(values, _mm256_sub_ps(
                    _mm256_loadu_ps(acc), _mm256_loadu_ps(acc_min)));
#else
            for (int lane = 0; lane < active; ++lane) values[lane] = acc[lane] - acc_min[lane];
#endif
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = values[lane] + (bias ? bias[n0 + lane] : 0.0f);
            }
        }
    }
}


typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_meta *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
} gemm_q4_packed_meta_thread_work_t;

typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_meta_x8 *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
    int groups;
    int tile_m;
    int jobs;
} gemm_q4_packed_meta_x8_thread_work_t;

typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_vnni_x8 *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int blocks_per_row;
    int groups;
} gemm_q4_packed_vnni_x8_thread_work_t;

typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_meta_x16 *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
    int groups;
    int tile_m;
    int jobs;
} gemm_q4_packed_meta_x16_thread_work_t;

typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_u8_x16 *W;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
    int groups;
    int tile_m;
    int jobs;
} gemm_q4_packed_u8_x16_thread_work_t;

static void gemm_q4_packed_meta_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_thread_work_t *a = (gemm_q4_packed_meta_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    const int dm = (a->M + nth - 1) / nth;
    const int m0 = dm * ith;
    const int m1 = (m0 + dm < a->M) ? (m0 + dm) : a->M;
    if (m0 >= a->M) {
        return;
    }
    for (int m = m0; m < m1; ++m) {
        const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        for (int n = 0; n < a->N; ++n) {
            const block_q4_K_packed_meta *w_row = a->W + (size_t)n * (size_t)a->blocks_per_row;
            float sum = a->bias ? a->bias[n] : 0.0f;
            for (int b = 0; b < a->blocks_per_row; ++b) {
                sum += dot_q4_k_packed_meta_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}

void gemm_nt_q4_k_packed_meta_q8_k_threaded(const void *A_q8,
                                             const void *B_packed,
                                             const float *bias,
                                             float *C,
                                             int M, int N, int K,
                                             int active_threads)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > M) {
        active_threads = M;
    }
    if (active_threads <= 1) {
        gemm_nt_q4_k_packed_meta_q8_k(A_q8, B_packed, bias, C, M, N, K);
        return;
    }
    gemm_q4_packed_meta_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta *)B_packed,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
    };
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_thread_fn, &work);
}


static void gemm_q4_packed_meta_nsplit_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_thread_work_t *a = (gemm_q4_packed_meta_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    const int dn = (a->N + nth - 1) / nth;
    const int n0 = dn * ith;
    const int n1 = (n0 + dn < a->N) ? (n0 + dn) : a->N;
    if (n0 >= a->N) {
        return;
    }
    for (int m = 0; m < a->M; ++m) {
        const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        for (int n = n0; n < n1; ++n) {
            const block_q4_K_packed_meta *w_row = a->W + (size_t)n * (size_t)a->blocks_per_row;
            float sum = a->bias ? a->bias[n] : 0.0f;
            for (int b = 0; b < a->blocks_per_row; ++b) {
                sum += dot_q4_k_packed_meta_q8_k_block(&w_row[b], &a_row[b]);
            }
            c_row[n] = sum;
        }
    }
}

void gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(const void *A_q8,
                                                    const void *B_packed,
                                                    const float *bias,
                                                    float *C,
                                                    int M, int N, int K,
                                                    int active_threads)
{
    if (!A_q8 || !B_packed || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > N) {
        active_threads = N;
    }
    if (active_threads <= 1) {
        gemm_nt_q4_k_packed_meta_q8_k(A_q8, B_packed, bias, C, M, N, K);
        return;
    }
    gemm_q4_packed_meta_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta *)B_packed,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
    };
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_nsplit_thread_fn, &work);
}

static void gemm_q4_packed_meta_x8_nsplit_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x8_thread_work_t *a = (gemm_q4_packed_meta_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    const int dg = (a->groups + nth - 1) / nth;
    const int g0 = dg * ith;
    const int g1 = (g0 + dg < a->groups) ? (g0 + dg) : a->groups;
    if (g0 >= a->groups) {
        return;
    }

    for (int m = 0; m < a->M; ++m) {
        const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        for (int g = g0; g < g1; ++g) {
            const int n0 = g * 8;
            const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
            float acc[8];
            for (int lane = 0; lane < active; ++lane) {
                acc[lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
            for (int b = 0; b < a->blocks_per_row; ++b) {
                const block_q4_K_packed_meta_x8 *w_group =
                    a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
                accum_q4_k_packed_meta_x8_q8_k_block(acc, w_group, active, &a_row[b]);
            }
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[lane];
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_nsplit(const void *A_q8,
                                                       const void *B_packed_x8,
                                                       const float *bias,
                                                       float *C,
                                                       int M, int N, int K,
                                                       int active_threads)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 7) / 8;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > groups) {
        active_threads = groups;
    }
    if (active_threads <= 1) {
        gemm_nt_q4_k_packed_meta_x8_q8_k(A_q8, B_packed_x8, bias, C, M, N, K);
        return;
    }
    gemm_q4_packed_meta_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x8 *)B_packed_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
    };
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x8_nsplit_thread_fn, &work);
}

static void gemm_q4_packed_meta_x8_mtile_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x8_thread_work_t *a = (gemm_q4_packed_meta_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int n0 = g * 8;
        const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
        if (m0 >= a->M || g >= a->groups) {
            continue;
        }

        float acc[8][8];
        for (int mt_lane = 0; mt_lane < m1 - m0; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x8 *w_group =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            for (int m = m0; m < m1; ++m) {
                const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
                accum_q4_k_packed_meta_x8_q8_k_block(acc[m - m0], w_group, active, &a_row[b]);
            }
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[m - m0][lane];
            }
        }
    }
}


static void gemm_q4_packed_meta_x8_mreuse_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x8_thread_work_t *a = (gemm_q4_packed_meta_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int m_count = m1 - m0;
        const int n0 = g * 8;
        const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
        if (m0 >= a->M || g >= a->groups || m_count <= 0) {
            continue;
        }

        float acc[8][8];
        for (int mt_lane = 0; mt_lane < m_count; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x8 *w_group =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            accum_q4_k_packed_meta_x8_q8_k_block_mreuse(acc, w_group, active, a->A,
                                                         a->blocks_per_vec, b, m0, m_count);
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[m - m0][lane];
            }
        }
    }
}

/* Reorder independent output work only. Each output keeps ascending K-block
 * traversal, separate value/minimum accumulators, and one final subtraction,
 * which is the llama.cpp repacked-matmul numerical contract. */
static void gemm_q4_packed_meta_x8_split_min_mreuse_thread_fn(
        int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x8_thread_work_t *a =
            (gemm_q4_packed_meta_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int m_count = m1 - m0;
        const int n0 = g * 8;
        const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
        if (m_count <= 0 || g >= a->groups) {
            continue;
        }

        float acc[8][8] = {{0}};
        float acc_min[8][8] = {{0}};
        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x8 *w_group =
                    a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            for (int m = m0; m < m1; ++m) {
                const block_q8_K *a_row =
                        a->A + (size_t)m * (size_t)a->blocks_per_vec;
                accum_q4_k_packed_meta_x8_q8_k_superblock(
                        acc[m - m0], acc_min[m - m0], w_group, active, &a_row[b]);
            }
        }

        for (int m = m0; m < m1; ++m) {
            float values[8];
#if defined(__AVX2__)
            _mm256_storeu_ps(values,
                    _mm256_sub_ps(_mm256_loadu_ps(acc[m - m0]),
                                  _mm256_loadu_ps(acc_min[m - m0])));
#else
            for (int lane = 0; lane < active; ++lane) {
                values[lane] = acc[m - m0][lane] - acc_min[m - m0][lane];
            }
#endif
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                float value = values[lane];
                if (a->bias) value += a->bias[n0 + lane];
                c_row[n0 + lane] = value;
            }
        }
    }
}

static void gemm_q4_packed_meta_x8_split_min_4m_thread_fn(
        int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x8_thread_work_t *a =
            (gemm_q4_packed_meta_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    const int row_tiles = (a->M + 3) / 4;
    const int total = row_tiles * a->groups;
    for (int job = ith; job < total; job += nth) {
        const int g = job / row_tiles;
        const int row_tile = job - g * row_tiles;
        const int m0 = row_tile * 4;
        const int rows = (m0 + 4 <= a->M) ? 4 : (a->M - m0);
        const int n0 = g * 8;
        const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
        if (rows <= 0 || active <= 0 || g >= a->groups) {
            continue;
        }

        float acc[8][8] = {{0}};
        float acc_min[8][8] = {{0}};
        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x8 *w_group =
                    a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            const block_q8_K *x[8] = {NULL};
            for (int row = 0; row < rows; ++row) {
                x[row] = a->A + (size_t)(m0 + row) *
                                  (size_t)a->blocks_per_vec + (size_t)b;
            }
            accum_q4_k_packed_meta_x8_q8_k_superblock_rows(
                    acc, acc_min, w_group, active, x, rows);
        }

        for (int row = 0; row < rows; ++row) {
            float values[8];
#if defined(__AVX2__)
            _mm256_storeu_ps(values,
                    _mm256_sub_ps(_mm256_loadu_ps(acc[row]),
                                  _mm256_loadu_ps(acc_min[row])));
#else
            for (int lane = 0; lane < active; ++lane) {
                values[lane] = acc[row][lane] - acc_min[row][lane];
            }
#endif
            float *c_row = a->C + (size_t)(m0 + row) * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                float value = values[lane];
                if (a->bias) value += a->bias[n0 + lane];
                c_row[n0 + lane] = value;
            }
        }
    }
}

static void gemm_q4_packed_meta_x8_split_min_8m_thread_fn(
        int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x8_thread_work_t *a =
            (gemm_q4_packed_meta_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    const int row_tiles = (a->M + 7) / 8;
    const int total = row_tiles * a->groups;
    for (int job = ith; job < total; job += nth) {
        const int g = job / row_tiles;
        const int row_tile = job - g * row_tiles;
        const int m0 = row_tile * 8;
        const int rows = (m0 + 8 <= a->M) ? 8 : (a->M - m0);
        const int n0 = g * 8;
        const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
        if (rows <= 0 || active <= 0 || g >= a->groups) {
            continue;
        }

        float acc[8][8] = {{0}};
        float acc_min[8][8] = {{0}};
        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x8 *w_group =
                    a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            const block_q8_K *x[8] = {NULL};
            for (int row = 0; row < rows; ++row) {
                x[row] = a->A + (size_t)(m0 + row) *
                                  (size_t)a->blocks_per_vec + (size_t)b;
            }
            accum_q4_k_packed_meta_x8_q8_k_superblock_rows(
                    acc, acc_min, w_group, active, x, rows);
        }

        for (int row = 0; row < rows; ++row) {
            float values[8];
#if defined(__AVX2__)
            _mm256_storeu_ps(values,
                    _mm256_sub_ps(_mm256_loadu_ps(acc[row]),
                                  _mm256_loadu_ps(acc_min[row])));
#else
            for (int lane = 0; lane < active; ++lane) {
                values[lane] = acc[row][lane] - acc_min[row][lane];
            }
#endif
            float *c_row = a->C + (size_t)(m0 + row) * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                float value = values[lane];
                if (a->bias) value += a->bias[n0 + lane];
                c_row[n0 + lane] = value;
            }
        }
    }
}

static void gemm_q4_packed_vnni_x8_q8k_4m_thread_fn(
        int ith, int nth, void *args)
{
    gemm_q4_packed_vnni_x8_thread_work_t *a =
            (gemm_q4_packed_vnni_x8_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    const int row_tiles = (a->M + 3) / 4;
    const int total = row_tiles * a->groups;
    for (int job = ith; job < total; job += nth) {
        const int group = job / row_tiles;
        const int row_tile = job - group * row_tiles;
        const int m0 = row_tile * 4;
        const int rows = (m0 + 4 <= a->M) ? 4 : (a->M - m0);
        const int n0 = group * 8;
        const int active = (n0 + 8 <= a->N) ? 8 : (a->N - n0);
        if (rows <= 0 || active <= 0 || group >= a->groups) {
            continue;
        }

        float acc[4][8] = {{0}};
        float acc_min[4][8] = {{0}};
        for (int block = 0; block < a->blocks_per_row; ++block) {
            const block_q4_K_packed_vnni_x8 *weights =
                    a->W + (size_t)group * (size_t)a->blocks_per_row +
                           (size_t)block;
            const block_q8_K *x[4] = {NULL};
            for (int row = 0; row < rows; ++row) {
                x[row] = a->A + (size_t)(m0 + row) *
                                  (size_t)a->blocks_per_row + (size_t)block;
            }
            accum_q4_k_packed_vnni_x8_q8_k_4m_superblock(
                    acc, acc_min, weights, x, rows);
        }

        for (int row = 0; row < rows; ++row) {
            float values[8];
            _mm256_storeu_ps(values, _mm256_sub_ps(
                    _mm256_loadu_ps(acc[row]),
                    _mm256_loadu_ps(acc_min[row])));
            float *output = a->C + (size_t)(m0 + row) * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                output[n0 + lane] = values[lane] +
                        (a->bias ? a->bias[n0 + lane] : 0.0f);
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_mtile(const void *A_q8,
                                                      const void *B_packed_x8,
                                                      const float *bias,
                                                      float *C,
                                                      int M, int N, int K,
                                                      int tile_m,
                                                      int active_threads)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 7) / 8;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    if (active_threads <= 1) {
        gemm_nt_q4_k_packed_meta_x8_q8_k(A_q8, B_packed_x8, bias, C, M, N, K);
        return;
    }
    gemm_q4_packed_meta_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x8 *)B_packed_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x8_mtile_thread_fn, &work);
}


void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_mreuse(const void *A_q8,
                                                       const void *B_packed_x8,
                                                       const float *bias,
                                                       float *C,
                                                       int M, int N, int K,
                                                       int tile_m,
                                                       int active_threads)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 7) / 8;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    gemm_q4_packed_meta_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x8 *)B_packed_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    if (active_threads <= 1) {
        gemm_q4_packed_meta_x8_mreuse_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x8_mreuse_thread_fn, &work);
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_mreuse(
        const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
        int M, int N, int K, int tile_m, int active_threads)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 ||
        (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 7) / 8;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int jobs = ((M + tm - 1) / tm) * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) active_threads = jobs;

    gemm_q4_packed_meta_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x8 *)B_packed_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    if (active_threads <= 1 || !pool) {
        gemm_q4_packed_meta_x8_split_min_mreuse_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(
            pool, active_threads,
            gemm_q4_packed_meta_x8_split_min_mreuse_thread_fn, &work);
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_4m(
        const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
        int M, int N, int K, int active_threads)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 ||
        (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 7) / 8;
    const int jobs = ((M + 3) / 4) * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) active_threads = jobs;

    gemm_q4_packed_meta_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x8 *)B_packed_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = 4,
        .jobs = jobs,
    };
    if (active_threads <= 1 || !pool) {
        gemm_q4_packed_meta_x8_split_min_4m_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(
            pool, active_threads,
            gemm_q4_packed_meta_x8_split_min_4m_thread_fn, &work);
}

void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_8m(
        const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
        int M, int N, int K, int active_threads)
{
    if (!A_q8 || !B_packed_x8 || !C || M <= 0 || N <= 0 || K <= 0 ||
        (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 7) / 8;
    const int jobs = ((M + 7) / 8) * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) active_threads = jobs;

    gemm_q4_packed_meta_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x8 *)B_packed_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = 8,
        .jobs = jobs,
    };
    if (active_threads <= 1 || !pool) {
        gemm_q4_packed_meta_x8_split_min_8m_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(
            pool, active_threads,
            gemm_q4_packed_meta_x8_split_min_8m_thread_fn, &work);
}

void gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m(
        const void *A_q8, const void *B_packed_vnni_x8, const float *bias,
        float *C, int M, int N, int K, int active_threads)
{
    if (!A_q8 || !B_packed_vnni_x8 || !C || M <= 0 || N <= 0 || K <= 0 ||
        (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_capacity(pool) : 1;
    const int groups = (N + 7) / 8;
    const int jobs = ((M + 3) / 4) * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) active_threads = jobs;

    gemm_q4_packed_vnni_x8_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_vnni_x8 *)B_packed_vnni_x8,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .blocks_per_row = K / QK_K,
        .groups = groups,
    };
    if (active_threads <= 1 || !pool) {
        gemm_q4_packed_vnni_x8_q8k_4m_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(
            pool, active_threads,
            gemm_q4_packed_vnni_x8_q8k_4m_thread_fn, &work);
}


static void gemm_q4_packed_meta_x16_mtile_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x16_thread_work_t *a = (gemm_q4_packed_meta_x16_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int n0 = g * 16;
        const int active = (n0 + 16 <= a->N) ? 16 : (a->N - n0);
        if (m0 >= a->M || g >= a->groups) {
            continue;
        }

        float acc[8][16];
        for (int mt_lane = 0; mt_lane < m1 - m0; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x16 *w_group =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            for (int m = m0; m < m1; ++m) {
                const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
                accum_q4_k_packed_meta_x16_q8_k_block(acc[m - m0], w_group, active, &a_row[b]);
            }
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[m - m0][lane];
            }
        }
    }
}


static inline void gemm_q4_packed_meta_x16_mreuse_process_job(
    const gemm_q4_packed_meta_x16_thread_work_t *a,
    int job,
    int mt,
    int tile_m)
{
    const int g = job / mt;
    const int tm = job - g * mt;
    const int m0 = tm * tile_m;
    const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
    const int m_count = m1 - m0;
    const int n0 = g * 16;
    const int active = (n0 + 16 <= a->N) ? 16 : (a->N - n0);
    if (m0 >= a->M || g >= a->groups || m_count <= 0) {
        return;
    }

#if defined(__AVX2__)
    const block_q4_K_packed_meta_x16 *w_group =
        a->W + (size_t)g * (size_t)a->blocks_per_row;
    for (int m = m0; m < m1; ++m) {
        const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        for (int lane = 0; lane < active; ++lane) {
            float value = dot_q4_k_packed_meta_x16_q8_k_llama_avx2(
                    w_group, a->blocks_per_row, lane, a_row);
            c_row[n0 + lane] = value + (a->bias ? a->bias[n0 + lane] : 0.0f);
        }
    }
#else
    float acc[8][16];
    for (int mt_lane = 0; mt_lane < m_count; ++mt_lane) {
        for (int lane = 0; lane < active; ++lane) {
            acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
        }
    }
    for (int b = 0; b < a->blocks_per_row; ++b) {
        const block_q4_K_packed_meta_x16 *w_block =
            a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
        if (ck_q4k_x16_chunk4_enabled()) {
            accum_q4_k_packed_meta_x16_q8_k_block_mreuse_chunk4(acc, w_block, active, a->A,
                                                                 a->blocks_per_vec, b, m0, m_count);
        } else {
            accum_q4_k_packed_meta_x16_q8_k_block_mreuse(acc, w_block, active, a->A,
                                                          a->blocks_per_vec, b, m0, m_count);
        }
    }
    for (int m = m0; m < m1; ++m) {
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        for (int lane = 0; lane < active; ++lane) {
            c_row[n0 + lane] = acc[m - m0][lane];
        }
    }
#endif
}

static void gemm_q4_packed_meta_x16_mreuse_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_meta_x16_thread_work_t *a = (gemm_q4_packed_meta_x16_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        gemm_q4_packed_meta_x16_mreuse_process_job(a, job, mt, tile_m);
    }
}

void gemm_nt_q4_k_packed_meta_x16_q8_k_threaded_mreuse(const void *A_q8,
                                                        const void *B_packed_x16,
                                                        const float *bias,
                                                        float *C,
                                                        int M, int N, int K,
                                                        int tile_m,
                                                        int active_threads)
{
    if (!A_q8 || !B_packed_x16 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 15) / 16;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    gemm_q4_packed_meta_x16_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x16 *)B_packed_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    if (active_threads <= 1) {
        gemm_q4_packed_meta_x16_mreuse_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x16_mreuse_thread_fn, &work);
}

void gemm_nt_q4_k_packed_meta_x16_q8_k_threaded_mtile(const void *A_q8,
                                                       const void *B_packed_x16,
                                                       const float *bias,
                                                       float *C,
                                                       int M, int N, int K,
                                                       int tile_m,
                                                       int active_threads)
{
    if (!A_q8 || !B_packed_x16 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 15) / 16;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    gemm_q4_packed_meta_x16_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x16 *)B_packed_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    if (active_threads <= 1) {
        gemm_q4_packed_meta_x16_mtile_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_meta_x16_mtile_thread_fn, &work);
}


typedef struct {
    const block_q8_K *A;
    const block_q4_K_packed_meta_x16 *W;
    const float *bias;
    float *C;
    int M;
    int D;
    int K;
    int tile_m;
    int blocks_per_vec;
    int blocks_per_row;
    int groups_d;
    int jobs;
} gemm_q4_gateup_swiglu_x16_work_t;


static void gemm_q4_gateup_swiglu_x16_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_gateup_swiglu_x16_work_t *a = (gemm_q4_gateup_swiglu_x16_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;

    for (int job = ith; job < a->jobs; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int m_count = m1 - m0;
        const int d0 = g * 16;
        const int active = (d0 + 16 <= a->D) ? 16 : (a->D - d0);
        if (m_count <= 0 || active <= 0 || g >= a->groups_d) {
            continue;
        }

        float acc_gate[8][16];
        float acc_up[8][16];
        for (int mt_lane = 0; mt_lane < m_count; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc_gate[mt_lane][lane] = a->bias ? a->bias[d0 + lane] : 0.0f;
                acc_up[mt_lane][lane] = a->bias ? a->bias[a->D + d0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_meta_x16 *w_gate =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            const block_q4_K_packed_meta_x16 *w_up =
                a->W + (size_t)(a->groups_d + g) * (size_t)a->blocks_per_row + (size_t)b;
            if (ck_q4k_x16_chunk4_enabled()) {
                accum_q4_k_packed_meta_x16_q8_k_block_mreuse_chunk4(acc_gate, w_gate, active, a->A,
                                                                     a->blocks_per_vec, b, m0, m_count);
                accum_q4_k_packed_meta_x16_q8_k_block_mreuse_chunk4(acc_up, w_up, active, a->A,
                                                                     a->blocks_per_vec, b, m0, m_count);
            } else {
                accum_q4_k_packed_meta_x16_q8_k_block_mreuse(acc_gate, w_gate, active, a->A,
                                                              a->blocks_per_vec, b, m0, m_count);
                accum_q4_k_packed_meta_x16_q8_k_block_mreuse(acc_up, w_up, active, a->A,
                                                              a->blocks_per_vec, b, m0, m_count);
            }
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->D;
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (int lane = 0; lane < active; ++lane) {
                const float gate = acc_gate[m - m0][lane];
                const float up = acc_up[m - m0][lane];
                c_row[d0 + lane] = (gate / (1.0f + expf(-gate))) * up;
            }
        }
    }
}

void gemm_nt_q4_k_packed_meta_x16_gateup_swiglu_fused_vnni(const void *A_q8,
                                                            const void *B_packed_x16,
                                                            const float *bias,
                                                            float *C,
                                                            int M, int D, int K,
                                                            int tile_m,
                                                            int active_threads)
{
    if (!A_q8 || !B_packed_x16 || !C || M <= 0 || D <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int groups_d = (D + 15) / 16;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups_d;
    if (active_threads <= 0 || active_threads > pool_threads) {
        active_threads = pool_threads;
    }
    if (active_threads > jobs) {
        active_threads = jobs;
    }
    if (active_threads < 1) {
        active_threads = 1;
    }

    gemm_q4_gateup_swiglu_x16_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_meta_x16 *)B_packed_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .D = D,
        .K = K,
        .tile_m = tm,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups_d = groups_d,
        .jobs = jobs,
    };
    if (!pool || active_threads <= 1) {
        gemm_q4_gateup_swiglu_x16_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_gateup_swiglu_x16_thread_fn, &work);
}


static void gemm_q4_packed_u8_x16_mtile_thread_fn(int ith, int nth, void *args)
{
    gemm_q4_packed_u8_x16_thread_work_t *a = (gemm_q4_packed_u8_x16_thread_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) return;
    int tile_m = a->tile_m > 0 ? a->tile_m : 4;
    if (tile_m > 8) tile_m = 8;
    const int mt = (a->M + tile_m - 1) / tile_m;
    const int total = mt * a->groups;

    for (int job = ith; job < total; job += nth) {
        const int g = job / mt;
        const int tm = job - g * mt;
        const int m0 = tm * tile_m;
        const int m1 = (m0 + tile_m < a->M) ? (m0 + tile_m) : a->M;
        const int n0 = g * 16;
        const int active = (n0 + 16 <= a->N) ? 16 : (a->N - n0);
        if (m0 >= a->M || g >= a->groups) continue;

        float acc[8][16];
        for (int mt_lane = 0; mt_lane < m1 - m0; ++mt_lane) {
            for (int lane = 0; lane < active; ++lane) {
                acc[mt_lane][lane] = a->bias ? a->bias[n0 + lane] : 0.0f;
            }
        }

        for (int b = 0; b < a->blocks_per_row; ++b) {
            const block_q4_K_packed_u8_x16 *w_group =
                a->W + (size_t)g * (size_t)a->blocks_per_row + (size_t)b;
            for (int m = m0; m < m1; ++m) {
                const block_q8_K *a_row = a->A + (size_t)m * (size_t)a->blocks_per_vec;
                accum_q4_k_packed_u8_x16_q8_k_block(acc[m - m0], w_group, active, &a_row[b]);
            }
        }

        for (int m = m0; m < m1; ++m) {
            float *c_row = a->C + (size_t)m * (size_t)a->N;
            for (int lane = 0; lane < active; ++lane) {
                c_row[n0 + lane] = acc[m - m0][lane];
            }
        }
    }
}

void gemm_nt_q4_k_packed_u8_x16_q8_k_threaded_mtile(const void *A_q8,
                                                     const void *B_packed_u8_x16,
                                                     const float *bias,
                                                     float *C,
                                                     int M, int N, int K,
                                                     int tile_m,
                                                     int active_threads)
{
    if (!A_q8 || !B_packed_u8_x16 || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) return;
    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int groups = (N + 15) / 16;
    int tm = tile_m > 0 ? tile_m : 4;
    if (tm > 8) tm = 8;
    const int mt = (M + tm - 1) / tm;
    const int jobs = mt * groups;
    if (active_threads <= 0 || active_threads > pool_threads) active_threads = pool_threads;
    if (active_threads > jobs) active_threads = jobs;
    gemm_q4_packed_u8_x16_thread_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K_packed_u8_x16 *)B_packed_u8_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
        .groups = groups,
        .tile_m = tm,
        .jobs = jobs,
    };
    if (active_threads <= 1) {
        gemm_q4_packed_u8_x16_mtile_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active_threads, gemm_q4_packed_u8_x16_mtile_thread_fn, &work);
}


void gemv_q4_k_q8_k_vnni(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const char *fast_env = getenv("CK_ENABLE_Q4K_Q8K_VNNI_FAST");
    const int fast_disabled = fast_env && fast_env[0] && fast_env[0] == '0';
    if (!fast_disabled && !ck_strict_parity_enabled()) {
        if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
            return;
        }

        const block_q4_K *blocks = (const block_q4_K *)W;
        const block_q8_K *x = (const block_q8_K *)x_q8;
        const int blocks_per_row = K / QK_K;

        for (int row = 0; row < M; ++row) {
            const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b) {
                sum += dot_q4_k_q8_k_vnni_block(&w_row[b], &x[b]);
            }
            y[row] = sum;
        }
        return;
    }
#endif

    /* Strict/debug parity keeps the llama-style scalar accumulation path.
     * Production AVX-512 hosts use VNNI by default; set
     * CK_ENABLE_Q4K_Q8K_VNNI_FAST=0 or CK_DEBUG_Q4K_Q8_REF=1 when attributing
     * borderline logit movement against scalar/reference behavior.
     */
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
}


static inline float ck_q4k_silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

typedef struct {
    const block_q8_K *A;
    const block_q4_K *W;
    const float *bias;
    float *C;
    int M;
    int D;
    int K;
    int blocks_per_vec;
    int blocks_per_row;
} gemm_q4_gateup_swiglu_work_t;

static void gemm_q4_gateup_swiglu_thread_fn(int ith, int nth, void *args)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    gemm_q4_gateup_swiglu_work_t *a = (gemm_q4_gateup_swiglu_work_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) return;

    const int dd = (a->D + nth - 1) / nth;
    const int d0 = dd * ith;
    const int d1 = (d0 + dd < a->D) ? (d0 + dd) : a->D;
    if (d0 >= a->D) return;

    for (int d = d0; d < d1; ++d) {
        const block_q4_K *w_gate = a->W + (size_t)d * (size_t)a->blocks_per_row;
        const block_q4_K *w_up = a->W + (size_t)(a->D + d) * (size_t)a->blocks_per_row;
        const float b_gate = a->bias ? a->bias[d] : 0.0f;
        const float b_up = a->bias ? a->bias[a->D + d] : 0.0f;
        for (int m = 0; m < a->M; ++m) {
            const block_q8_K *x = a->A + (size_t)m * (size_t)a->blocks_per_vec;
            float gate = b_gate;
            float up = b_up;
            for (int b = 0; b < a->blocks_per_row; ++b) {
                gate += dot_q4_k_q8_k_vnni_block(&w_gate[b], &x[b]);
                up += dot_q4_k_q8_k_vnni_block(&w_up[b], &x[b]);
            }
            a->C[(size_t)m * (size_t)a->D + (size_t)d] = ck_q4k_silu_f32(gate) * up;
        }
    }
#else
    (void)ith;
    (void)nth;
    (void)args;
#endif
}

void gemm_nt_q4_k_q8_k_gateup_swiglu_fused_vnni(const void *A_q8,
                                                 const void *B_gate_up,
                                                 const float *bias,
                                                 float *C,
                                                 int M,
                                                 int D,
                                                 int K,
                                                 int threads)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    if (!A_q8 || !B_gate_up || !C || M <= 0 || D <= 0 || K <= 0 || (K % QK_K) != 0) {
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int active = threads > 0 ? threads : (pool ? ck_threadpool_n_threads(pool) : 1);
    if (active < 1) active = 1;
    if (active > D) active = D;

    gemm_q4_gateup_swiglu_work_t work = {
        .A = (const block_q8_K *)A_q8,
        .W = (const block_q4_K *)B_gate_up,
        .bias = bias,
        .C = C,
        .M = M,
        .D = D,
        .K = K,
        .blocks_per_vec = K / QK_K,
        .blocks_per_row = K / QK_K,
    };

    if (active <= 1 || !pool) {
        gemm_q4_gateup_swiglu_thread_fn(0, 1, &work);
        return;
    }
    ck_threadpool_dispatch_n(pool, active, gemm_q4_gateup_swiglu_thread_fn, &work);
#else
    (void)A_q8;
    (void)B_gate_up;
    (void)bias;
    (void)C;
    (void)M;
    (void)D;
    (void)K;
    (void)threads;
#endif
}


void gemv_q4_k_q8_k_parallel_vnni(float *y,
                                  const void *W,
                                  const void *x_q8,
                                  int M, int K,
                                  int ith, int nth)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    if (!y || !W || !x_q8 || M <= 0 || K <= 0) {
        return;
    }
    if (ith < 0 || nth <= 0 || ith >= nth) {
        return;
    }

    const int dr = (M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < M) ? (r0 + dr) : M;
    if (r0 >= M) {
        return;
    }

    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *x = (const block_q8_K *)x_q8;
    const int blocks_per_row = K / QK_K;

    for (int row = r0; row < r1; ++row) {
        const block_q4_K *w_row = blocks + (size_t)row * (size_t)blocks_per_row;
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            sum += dot_q4_k_q8_k_vnni_block(&w_row[b], &x[b]);
        }
        y[row] = sum;
    }
#else
    (void)y;
    (void)W;
    (void)x_q8;
    (void)M;
    (void)K;
    (void)ith;
    (void)nth;
#endif
}
