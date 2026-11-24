#ifndef CK_PACKED_K_DOT_ORDERED_H
#define CK_PACKED_K_DOT_ORDERED_H

#include "ckernel_engine.h"
#include "ckernel_quant.h"

#include <string.h>

/* Decode one packed block at a time. FP32 operands are accumulated in channel
 * order in FP64, without requiring a dense weight buffer. */
#define CK_DEFINE_PACKED_ORDERED_DOT(name, type, block_width, decode) \
static inline float name(const float *input, const unsigned char *weight, size_t width) \
{ \
    double sum = 0.0; \
    for (size_t first = 0; first < width; first += block_width, weight += sizeof(type)) { \
        type block; \
        float values[block_width]; \
        memcpy(&block, weight, sizeof(block)); \
        decode(&block, values, block_width); \
        for (size_t channel = 0; channel < block_width; ++channel) \
            sum += (double)input[first + channel] * (double)values[channel]; \
    } \
    return (float)sum; \
}

CK_DEFINE_PACKED_ORDERED_DOT(ck_packed_q4_k_dot_ordered_scalar, block_q4_K, QK_K, dequant_q4_k_row)
CK_DEFINE_PACKED_ORDERED_DOT(ck_packed_q6_k_dot_ordered, block_q6_K, QK_K, dequant_q6_k_row)
CK_DEFINE_PACKED_ORDERED_DOT(ck_packed_q5_0_dot_ordered, block_q5_0, QK5_0, dequant_q5_0_row)

#undef CK_DEFINE_PACKED_ORDERED_DOT

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

/* Vectorize unpacking and products while preserving the scalar sum order.
 * Separate FP32 multiply/subtract preserves the packed storage conversion. */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
__attribute__((target("avx2,f16c")))
static inline float ck_packed_q4_k_dot_ordered_avx2(
    const float *input, const unsigned char *weight, size_t width)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    double sum = 0.0;
    const __m256i mask = _mm256_set1_epi32(15);
    for (size_t first = 0; first < width; first += QK_K, weight += sizeof(block_q4_K)) {
        block_q4_K block;
        memcpy(&block, weight, sizeof(block));
        const float scale = _cvtsh_ss(block.d), minimum = _cvtsh_ss(block.dmin);
        uint8_t scales[8], mins[8];
        unpack_q4_k_scales(block.scales, scales, mins);
        for (int group = 0; group < 8; ++group) {
            const __m256 d = _mm256_set1_ps(scale * (float)scales[group]);
            const __m256 m = _mm256_set1_ps(minimum * (float)mins[group]);
            const uint8_t *qs = block.qs + (group / 2) * 32;
            const float *row = input + first + group * 32;
            for (int j = 0; j < 32; j += 8) {
                const __m128i bytes = _mm_loadl_epi64((const __m128i *)(qs + j));
                __m256i codes = _mm256_cvtepu8_epi32(bytes);
                if (group & 1) codes = _mm256_srli_epi32(codes, 4);
                codes = _mm256_and_si256(codes, mask);
                const __m256 expanded = _mm256_sub_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(codes)), m);
                const __m256 x = _mm256_loadu_ps(row + j);
                double products[8];
                _mm256_storeu_pd(products, _mm256_mul_pd(
                    _mm256_cvtps_pd(_mm256_castps256_ps128(x)),
                    _mm256_cvtps_pd(_mm256_castps256_ps128(expanded))));
                _mm256_storeu_pd(products + 4, _mm256_mul_pd(
                    _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)),
                    _mm256_cvtps_pd(_mm256_extractf128_ps(expanded, 1))));
                for (int k = 0; k < 8; ++k) sum += products[k];
            }
        }
    }
    return (float)sum;
}
#endif

static inline float ck_packed_q4_k_dot_ordered(
    const float *input, const unsigned char *weight, size_t width)
{
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c"))
        return ck_packed_q4_k_dot_ordered_avx2(input, weight, width);
#endif
    return ck_packed_q4_k_dot_ordered_scalar(input, weight, width);
}

#endif
