#ifndef CK_PACKED_Q8_ACTIVATION_CHECKED_H
#define CK_PACKED_Q8_ACTIVATION_CHECKED_H

#include "ckernel_quant.h"
#include <math.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CK_Q8_X86_SIMD 1
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

#ifdef CK_Q8_X86_SIMD
static inline __attribute__((target("avx2"))) __m256
ck_q8_activation_products(const unsigned char *weight, __m256i activation)
{
    const __m256i w = _mm256_loadu_si256((const __m256i *)(weight + 2));
    /* Packed activations are in [-127, 127], so the signed transfer and
     * each pair sum are exact even for a weight code of -128. */
    const __m256i pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(activation, w));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

static inline __attribute__((target("avx2"))) float
ck_q8_activation_reduce(__m256 sum)
{
    const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_movehdup_ps(pairs)));
}

static inline uint16_t ck_q8_x86_scale_bits(const unsigned char *block)
{
    uint16_t scale;
    memcpy(&scale, block, sizeof(scale));
    return scale;
}

static __attribute__((target("avx2,fma,f16c"))) float
ck_q8_activation_dot_avx2_f16c(const unsigned char *weight, const unsigned char *input, size_t width)
{
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < width; i += QK8_0, weight += sizeof(block_q8_0), input += sizeof(block_q8_0)) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(input + 2));
        const uint32_t scales = ck_q8_x86_scale_bits(weight) | ((uint32_t)ck_q8_x86_scale_bits(input) << 16);
        const __m128 decoded = _mm_cvtph_ps(_mm_cvtsi32_si128((int)scales));
        const __m128 scale = _mm_mul_ss(decoded, _mm_shuffle_ps(decoded, decoded, _MM_SHUFFLE(1, 1, 1, 1)));
        sum = _mm256_fmadd_ps(_mm256_broadcastss_ps(scale), ck_q8_activation_products(weight, x), sum);
    }
    return ck_q8_activation_reduce(sum);
}

static __attribute__((target("avx2,fma,f16c"))) void
ck_q8_activation_dot_pair_avx2_f16c(const unsigned char *gate, const unsigned char *up,
                                    const unsigned char *input, size_t width, float *g, float *u)
{
    __m256 gate_sum = _mm256_setzero_ps(), up_sum = _mm256_setzero_ps();
    for (size_t i = 0; i < width; i += QK8_0) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(input + 2));
        const uint32_t weights = ck_q8_x86_scale_bits(gate) | ((uint32_t)ck_q8_x86_scale_bits(up) << 16);
        const __m128i halves = _mm_set_epi32(0, 0, ck_q8_x86_scale_bits(input), (int)weights);
        const __m128 decoded = _mm_cvtph_ps(halves);
        const __m128 scales = _mm_mul_ps(decoded, _mm_shuffle_ps(decoded, decoded, _MM_SHUFFLE(2, 2, 2, 2)));
        const __m128 up_scale = _mm_shuffle_ps(scales, scales, _MM_SHUFFLE(1, 1, 1, 1));
        gate_sum = _mm256_fmadd_ps(_mm256_broadcastss_ps(scales), ck_q8_activation_products(gate, x), gate_sum);
        up_sum = _mm256_fmadd_ps(_mm256_broadcastss_ps(up_scale), ck_q8_activation_products(up, x), up_sum);
        gate += sizeof(block_q8_0);
        up += sizeof(block_q8_0);
        input += sizeof(block_q8_0);
    }
    *g = ck_q8_activation_reduce(gate_sum);
    *u = ck_q8_activation_reduce(up_sum);
}

static inline int ck_q8_activation_has_f16c(void)
{
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}
#endif

static inline float ck_q8_activation_dot(const unsigned char *weight, const unsigned char *input,
                                         size_t width)
{
#ifdef CK_Q8_X86_SIMD
    if (ck_q8_activation_has_f16c())
        return ck_q8_activation_dot_avx2_f16c(weight, input, width);
#endif
    return ck_q8_activation_dot_scalar(weight, input, width);
}

static inline void ck_q8_activation_dot_pair(const unsigned char *gate, const unsigned char *up,
                                             const unsigned char *input, size_t width, float *g, float *u)
{
#ifdef CK_Q8_X86_SIMD
    if (ck_q8_activation_has_f16c()) {
        ck_q8_activation_dot_pair_avx2_f16c(gate, up, input, width, g, u);
        return;
    }
#endif
    *g = ck_q8_activation_dot_scalar(gate, input, width);
    *u = ck_q8_activation_dot_scalar(up, input, width);
}

#endif
