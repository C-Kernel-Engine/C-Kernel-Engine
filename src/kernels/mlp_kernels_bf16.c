/**
 * @file mlp_kernels_bf16.c
 * @brief Optimized BF16 MLP Kernels
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
 * Uses direct BF16 GEMM instead of converting to FP32.
 * Layout: input[T,D] -> fc1[T,4D] -> GELU -> fc2[T,D]
 *
 * All functions use caller-provided scratch buffers (no internal malloc).
 */

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bf16_utils.h"
#include "ckernel_engine.h"

// Suppress false positive warnings about uninitialized variables
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

/* Forward declaration of optimized BF16 GEMM */
extern void gemm_bf16_fp32out(const uint16_t *A, const uint16_t *B,
                               const float *bias, float *C,
                               int M, int N, int K);

/* GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static inline float gelu_scalar(float x)
{
    const float c = 0.7978845608f;  /* sqrt(2/pi) */
    const float k = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x3)));
}

#if defined(__AVX512F__)
/* Vectorized GELU using polynomial approximation of tanh */
static inline __m512 gelu_avx512(__m512 x)
{
    const __m512 c = _mm512_set1_ps(0.7978845608f);
    const __m512 k = _mm512_set1_ps(0.044715f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);

    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 x3 = _mm512_mul_ps(x2, x);

    __m512 inner = _mm512_fmadd_ps(k, x3, x);
    inner = _mm512_mul_ps(c, inner);

    __m512 inner2 = _mm512_mul_ps(inner, inner);
    __m512 num = _mm512_add_ps(_mm512_set1_ps(27.0f), inner2);
    __m512 den = _mm512_fmadd_ps(_mm512_set1_ps(9.0f), inner2, _mm512_set1_ps(27.0f));
    __m512 tanh_approx = _mm512_mul_ps(inner, _mm512_div_ps(num, den));

    tanh_approx = _mm512_min_ps(tanh_approx, one);
    tanh_approx = _mm512_max_ps(tanh_approx, _mm512_set1_ps(-1.0f));

    __m512 result = _mm512_add_ps(one, tanh_approx);
    result = _mm512_mul_ps(half, _mm512_mul_ps(x, result));

    return result;
}
#endif

/**
 * Optimized MLP Forward (BF16 weights, FP32 activations)
 *
 * Caller-provided scratch buffers:
 *   scratch_bias1_f: [4*D] floats
 *   scratch_bias2_f: [D] floats
 *   scratch_fc1_bf16: [T * 4*D] uint16_t (BF16)
 */
void mlp_token_parallel_bf16(const uint16_t *input,
                             const uint16_t *W_fc1,
                             const uint16_t *b_fc1,
                             const uint16_t *W_fc2,
                             const uint16_t *b_fc2,
                             float *fc1_output,
                             float *output,
                             int T,
                             int aligned_dim,
                             int num_threads,
                             float *scratch_bias1_f,
                             float *scratch_bias2_f,
                             uint16_t *scratch_fc1_bf16)
{
    if (!input || !W_fc1 || !b_fc1 || !W_fc2 || !b_fc2 || !fc1_output || !output) return;
    if (!scratch_bias1_f || !scratch_bias2_f || !scratch_fc1_bf16) return;

    (void)num_threads;
    const int D = aligned_dim;
    const int fourD = 4 * D;

    /* Convert biases to FP32 */
    for (int i = 0; i < fourD; ++i) {
        scratch_bias1_f[i] = bf16_to_float(b_fc1[i]);
    }
    for (int i = 0; i < D; ++i) {
        scratch_bias2_f[i] = bf16_to_float(b_fc2[i]);
    }

    /* FC1: [T, D] x [4D, D].T -> [T, 4D] */
    gemm_bf16_fp32out(input, W_fc1, scratch_bias1_f, fc1_output, T, fourD, D);

    /* GELU activation */
#if defined(__AVX512F__)
    #pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        float *row = fc1_output + (size_t)t * fourD;
        int j = 0;
        for (; j <= fourD - 16; j += 16) {
            __m512 x = _mm512_loadu_ps(row + j);
            _mm512_storeu_ps(row + j, gelu_avx512(x));
        }
        for (; j < fourD; ++j) {
            row[j] = gelu_scalar(row[j]);
        }
    }
#else
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < fourD; ++j) {
            fc1_output[t * fourD + j] = gelu_scalar(fc1_output[t * fourD + j]);
        }
    }
#endif

    /* Convert FP32 activations to BF16 */
#if defined(__AVX512F__)
    #pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        float *src = fc1_output + (size_t)t * fourD;
        uint16_t *dst = scratch_fc1_bf16 + (size_t)t * fourD;
        int j = 0;
        for (; j <= fourD - 16; j += 16) {
            __m512 fp32 = _mm512_loadu_ps(src + j);
            __m512i as_int = _mm512_castps_si512(fp32);
            __m512i lsb = _mm512_srli_epi32(as_int, 16);
            lsb = _mm512_and_si512(lsb, _mm512_set1_epi32(1));
            __m512i rounding = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
            __m512i rounded = _mm512_add_epi32(as_int, rounding);
            __m512i shifted = _mm512_srli_epi32(rounded, 16);
            __m256i bf16 = _mm512_cvtepi32_epi16(shifted);
            _mm256_storeu_si256((__m256i *)(dst + j), bf16);
        }
        for (; j < fourD; ++j) {
            dst[j] = float_to_bf16(src[j]);
        }
    }
#else
    for (size_t i = 0; i < (size_t)T * fourD; ++i) {
        scratch_fc1_bf16[i] = float_to_bf16(fc1_output[i]);
    }
#endif

    /* FC2: BF16 GEMM with FP32 output */
    gemm_bf16_fp32out(scratch_fc1_bf16, W_fc2, scratch_bias2_f, output, T, D, fourD);
}

/**
 * Alternative: Fully FP32 activations throughout
 *
 * Caller-provided scratch buffers:
 *   scratch_input_f: [T * D] floats
 *   scratch_bias1_f: [4*D] floats
 *   scratch_bias2_f: [D] floats
 *   scratch_fc1_bf16: [T * 4*D] uint16_t (BF16)
 */
void mlp_token_parallel_bf16_fp32act(const uint16_t *input,
                                      const uint16_t *W_fc1,
                                      const uint16_t *b_fc1,
                                      const uint16_t *W_fc2,
                                      const uint16_t *b_fc2,
                                      float *fc1_output,
                                      float *output,
                                      int T,
                                      int aligned_dim,
                                      int num_threads,
                                      float *scratch_input_f,
                                      float *scratch_bias1_f,
                                      float *scratch_bias2_f,
                                      uint16_t *scratch_fc1_bf16)
{
    if (!input || !W_fc1 || !b_fc1 || !W_fc2 || !b_fc2 || !fc1_output || !output) return;
    if (!scratch_input_f || !scratch_bias1_f || !scratch_bias2_f || !scratch_fc1_bf16) return;

    (void)num_threads;
    const int D = aligned_dim;
    const int fourD = 4 * D;

    /* Convert input and biases to FP32 */
    bf16_tensor_to_float(input, scratch_input_f, (size_t)T * D);
    bf16_tensor_to_float(b_fc1, scratch_bias1_f, fourD);
    bf16_tensor_to_float(b_fc2, scratch_bias2_f, D);

    /* FC1 */
    gemm_bf16_fp32out(input, W_fc1, scratch_bias1_f, fc1_output, T, fourD, D);

    /* GELU */
#if defined(__AVX512F__)
    #pragma omp parallel for
    for (int t = 0; t < T; ++t) {
        float *row = fc1_output + (size_t)t * fourD;
        int j = 0;
        for (; j <= fourD - 16; j += 16) {
            __m512 x = _mm512_loadu_ps(row + j);
            _mm512_storeu_ps(row + j, gelu_avx512(x));
        }
        for (; j < fourD; ++j) {
            row[j] = gelu_scalar(row[j]);
        }
    }
#else
    for (size_t i = 0; i < (size_t)T * fourD; ++i) {
        fc1_output[i] = gelu_scalar(fc1_output[i]);
    }
#endif

    /* Convert fc1_output to BF16 for FC2 */
    float_tensor_to_bf16(fc1_output, scratch_fc1_bf16, (size_t)T * fourD);
    gemm_bf16_fp32out(scratch_fc1_bf16, W_fc2, scratch_bias2_f, output, T, D, fourD);
}

#pragma GCC diagnostic pop
