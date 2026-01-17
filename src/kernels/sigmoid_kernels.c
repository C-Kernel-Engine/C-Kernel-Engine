/**
 * @file sigmoid_kernels.c
 * @brief Sigmoid activation kernels with SIMD (AVX512)
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
 * Sigmoid: y = 1 / (1 + exp(-x))
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Core sigmoid scalar kernel */
float sigmoid_scalar(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

#if defined(__AVX512F__)
// Fast exp approximation using polynomial (avoids SVML dependency)
// Based on Schraudolph's algorithm with refinement
static inline __m512 exp_approx_avx512(__m512 x)
{
    // Clamp x to avoid overflow/underflow
    const __m512 max_val = _mm512_set1_ps(88.0f);
    const __m512 min_val = _mm512_set1_ps(-88.0f);
    x = _mm512_max_ps(_mm512_min_ps(x, max_val), min_val);

    // exp(x) = 2^(x * log2(e)) = 2^(x * 1.4426950408889634)
    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    __m512 z = _mm512_mul_ps(x, log2e);

    // Split into integer and fractional parts
    __m512 zf = _mm512_roundscale_ps(z, _MM_FROUND_TO_NEAREST_INT);
    __m512 f = _mm512_sub_ps(z, zf);  // fractional part in [-0.5, 0.5]

    // Polynomial approximation for 2^f where f in [-0.5, 0.5]
    // 2^f ≈ 1 + f*ln(2) + f^2*ln(2)^2/2 + f^3*ln(2)^3/6 + ...
    // Using optimized coefficients
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(0.6931471805599453f);   // ln(2)
    const __m512 c2 = _mm512_set1_ps(0.2402265069591007f);   // ln(2)^2/2
    const __m512 c3 = _mm512_set1_ps(0.05550410866482158f);  // ln(2)^3/6
    const __m512 c4 = _mm512_set1_ps(0.009618129107628478f); // ln(2)^4/24

    // Horner's method: c0 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    __m512 poly = _mm512_fmadd_ps(f, c4, c3);
    poly = _mm512_fmadd_ps(f, poly, c2);
    poly = _mm512_fmadd_ps(f, poly, c1);
    poly = _mm512_fmadd_ps(f, poly, c0);

    // Scale by 2^n using integer manipulation
    __m512i zi = _mm512_cvtps_epi32(zf);
    zi = _mm512_add_epi32(zi, _mm512_set1_epi32(127));  // Add IEEE754 exponent bias
    zi = _mm512_slli_epi32(zi, 23);  // Shift to exponent position
    __m512 scale = _mm512_castsi512_ps(zi);

    return _mm512_mul_ps(poly, scale);
}

static inline __m512 sigmoid_avx512_vec(__m512 x)
{
    __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), x);
    __m512 exp_neg = exp_approx_avx512(neg);
    __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg);
    return _mm512_div_ps(_mm512_set1_ps(1.0f), denom);
}

static void sigmoid_forward_avx512(const float *input,
                                   float *output,
                                   size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 in_vec = _mm512_loadu_ps(input + i);
        __m512 sig = sigmoid_avx512_vec(in_vec);
        _mm512_storeu_ps(output + i, sig);
    }

    for (; i < n; ++i) {
        output[i] = sigmoid_scalar(input[i]);
    }
}

static void sigmoid_backward_avx512(const float *input,
                                    const float *d_output,
                                    float *d_input,
                                    size_t n)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 in_vec = _mm512_loadu_ps(input + i);
        __m512 s = sigmoid_avx512_vec(in_vec);
        __m512 dout = _mm512_loadu_ps(d_output + i);
        __m512 grad = _mm512_mul_ps(_mm512_mul_ps(s, _mm512_sub_ps(one, s)), dout);
        _mm512_storeu_ps(d_input + i, grad);
    }

    for (; i < n; ++i) {
        float x = input[i];
        float s = sigmoid_scalar(x);
        float s_prime = s * (1.0f - s);
        d_input[i] = d_output[i] * s_prime;
    }
}
#endif

// Vectorized (loop) sigmoid forward over a contiguous buffer.
void sigmoid_forward(const float *input,
                     float *output,
                     size_t n)
{
#if defined(__AVX512F__)
    sigmoid_forward_avx512(input, output, n);
#else
    for (size_t i = 0; i < n; ++i) {
        output[i] = sigmoid_scalar(input[i]);
    }
#endif
}

// Sigmoid backward over a contiguous buffer:
// Given dY and X, compute dX = dY * s * (1 - s),
// where s = sigmoid(X).
void sigmoid_backward(const float *input,
                      const float *d_output,
                      float *d_input,
                      size_t n)
{
#if defined(__AVX512F__)
    sigmoid_backward_avx512(input, d_output, d_input, n);
#else
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float s = sigmoid_scalar(x);
        float s_prime = s * (1.0f - s);
        d_input[i] = d_output[i] * s_prime;
    }
#endif
}
