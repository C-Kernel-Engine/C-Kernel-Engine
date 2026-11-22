#ifndef CK_PACKED_Q8_ACTIVATION_CHECKED_H
#define CK_PACKED_Q8_ACTIVATION_CHECKED_H

#include "ckernel_quant.h"
#include <math.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CK_Q8_X86_AVX2 1
#endif

/* Input rows are finite. Emit little-endian Q8_0 blocks with nearest-even codes. */
static inline int ck_pack_q8_activation_rne(const float *input, unsigned char *packed, size_t width)
{
    for (size_t i = 0; i < width; i += QK8_0, packed += sizeof(block_q8_0)) {
        float peak = 0.0f;
        for (size_t j = 0; j < QK8_0; ++j)
            peak = fmaxf(peak, fabsf(input[i + j]));
        const uint16_t scale = ck_fp32_to_fp16_soft(peak / 127.0f);
        if ((scale & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) return -1;
        packed[0] = (unsigned char)scale;
        packed[1] = (unsigned char)(scale >> 8);
        const float inverse = scale ? 127.0f / peak : 0.0f;
        for (size_t j = 0; j < QK8_0; ++j) {
            const float value = input[i + j] * inverse;
            const float magnitude = fminf(127.0f, fabsf(value));
            int rounded = (int)magnitude;
            const float fraction = magnitude - (float)rounded;
            if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) ++rounded;
            packed[2 + j] = (unsigned char)(value < 0.0f ? -rounded : rounded);
        }
    }
    return 0;
}

static inline float ck_q8_activation_dot_scalar(const unsigned char *weight, const unsigned char *input,
                                         size_t width)
{
    /* The x86 GGML reduction groups four signed products into each of
     * eight FP32 FMA lanes, then combines lanes in a fixed tree. */
    float lanes[8] = {0};
    for (size_t i = 0; i < width; i += QK8_0, weight += sizeof(block_q8_0), input += sizeof(block_q8_0)) {
        const float dw = ck_fp16_to_fp32_soft((uint16_t)(weight[0] | ((uint16_t)weight[1] << 8)));
        const float dx = ck_fp16_to_fp32_soft((uint16_t)(input[0] | ((uint16_t)input[1] << 8)));
        const float scale = dw * dx;
        for (size_t lane = 0; lane < 8; ++lane) {
            int dot = 0;
            for (size_t j = 2 + 4 * lane; j < 6 + 4 * lane; ++j) {
                const int w = weight[j], x = input[j];
                dot += (w < 128 ? w : w - 256) * (x < 128 ? x : x - 256);
            }
            lanes[lane] = fmaf(scale, (float)dot, lanes[lane]);
        }
    }
    return ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
         + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
}

#ifdef CK_Q8_X86_AVX2
static __attribute__((target("avx2,fma"))) float
ck_q8_activation_dot_avx2(const unsigned char *weight, const unsigned char *input, size_t width)
{
    __m256 sum = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t i = 0; i < width; i += QK8_0, weight += sizeof(block_q8_0), input += sizeof(block_q8_0)) {
        const __m256i w = _mm256_loadu_si256((const __m256i *)(weight + 2));
        const __m256i x = _mm256_loadu_si256((const __m256i *)(input + 2));
        /* Packed activations are in [-127, 127], so the signed transfer and
         * each pair sum are exact even for a weight code of -128. */
        const __m256i pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
        const __m256 dots = _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, ones));
        const float dw = ck_fp16_to_fp32_soft((uint16_t)(weight[0] | ((uint16_t)weight[1] << 8)));
        const float dx = ck_fp16_to_fp32_soft((uint16_t)(input[0] | ((uint16_t)input[1] << 8)));
        sum = _mm256_fmadd_ps(_mm256_set1_ps(dw * dx), dots, sum);
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, sum);
    return ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
         + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
}
#endif

static inline float ck_q8_activation_dot(const unsigned char *weight, const unsigned char *input,
                                         size_t width)
{
#ifdef CK_Q8_X86_AVX2
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return ck_q8_activation_dot_avx2(weight, input, width);
#endif
    return ck_q8_activation_dot_scalar(weight, input, width);
}

#endif
