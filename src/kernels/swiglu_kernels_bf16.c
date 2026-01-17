/**
 * @file swiglu_kernels_bf16.c
 * @brief SwiGLU activation kernels for BF16 tensors
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
 * SwiGLU: y = silu(gate) * up = (gate * sigmoid(gate)) * up
 */

#include <stdint.h>
#include <math.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

#if defined(__AVX512F__)
#include <immintrin.h>

/* Fast exp approximation for AVX-512 */
static inline __m512 exp512_fast_bf16(__m512 x) {
    // Clamp to avoid overflow/underflow
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));

    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    __m512 z = _mm512_mul_ps(x, log2e);
    __m512 zf = _mm512_roundscale_ps(z, _MM_FROUND_TO_NEAREST_INT);
    __m512 f = _mm512_sub_ps(z, zf);

    // Polynomial for 2^f
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 c2 = _mm512_set1_ps(0.2402265069591007f);
    const __m512 c3 = _mm512_set1_ps(0.05550410866482158f);
    const __m512 c4 = _mm512_set1_ps(0.009618129107628478f);

    __m512 poly = _mm512_fmadd_ps(f, c4, c3);
    poly = _mm512_fmadd_ps(f, poly, c2);
    poly = _mm512_fmadd_ps(f, poly, c1);
    poly = _mm512_fmadd_ps(f, poly, c0);

    __m512i zi = _mm512_cvtps_epi32(zf);
    zi = _mm512_add_epi32(zi, _mm512_set1_epi32(127));
    zi = _mm512_slli_epi32(zi, 23);
    __m512 scale = _mm512_castsi512_ps(zi);

    return _mm512_mul_ps(poly, scale);
}

// AVX-512 sigmoid: 1 / (1 + exp(-x))
static inline __m512 sigmoid512_fast_bf16(__m512 x) {
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    __m512 exp_neg = exp512_fast_bf16(neg_x);
    __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(one, _mm512_add_ps(one, exp_neg));
}
#endif

void swiglu_forward_bf16(const uint16_t *input,
                         uint16_t *output,
                         int tokens,
                         int dim)
{
    if (!input || !output || tokens <= 0 || dim <= 0) {
        return;
    }

    const int T = tokens;
    const int D = dim;

    for (int t = 0; t < T; ++t) {
        const uint16_t *row = input + (size_t)t * (size_t)(2 * D);
        uint16_t *out_row = output + (size_t)t * (size_t)D;
        int d = 0;

#if defined(__AVX512F__)
        // AVX-512: Process 16 floats at a time
        for (; d + 16 <= D; d += 16) {
            __m512 a = bf16_loadu_cvt_fp32(&row[d]);         // gate
            __m512 b = bf16_loadu_cvt_fp32(&row[D + d]);     // value

            __m512 s = sigmoid512_fast_bf16(a);             // sigmoid(a)
            __m512 silu = _mm512_mul_ps(a, s);              // silu(a) = a * sigmoid(a)
            __m512 y = _mm512_mul_ps(silu, b);              // y = silu(a) * b

            fp32_cvt_storeu_bf16(&out_row[d], y);
        }
#endif

        // Scalar fallback for remaining elements
        for (; d < D; ++d) {
            float a = bf16_to_float(row[d]);
            float b = bf16_to_float(row[D + d]);
            float s = sigmoid_scalar(a);
            float silu = a * s;
            out_row[d] = float_to_bf16(silu * b);
        }
    }
}

void swiglu_backward_bf16(const uint16_t *input,
                          const uint16_t *d_output,
                          uint16_t *d_input,
                          int tokens,
                          int dim)
{
    if (!input || !d_output || !d_input || tokens <= 0 || dim <= 0) {
        return;
    }

    const int T = tokens;
    const int D = dim;

    for (int t = 0; t < T; ++t) {
        const uint16_t *row = input + (size_t)t * (size_t)(2 * D);
        const uint16_t *dy_row = d_output + (size_t)t * (size_t)D;
        uint16_t *dx_row = d_input + (size_t)t * (size_t)(2 * D);
        int d = 0;

#if defined(__AVX512F__)
        // AVX-512: Process 16 floats at a time
        __m512 one = _mm512_set1_ps(1.0f);
        for (; d + 16 <= D; d += 16) {
            __m512 a = bf16_loadu_cvt_fp32(&row[d]);         // gate
            __m512 b = bf16_loadu_cvt_fp32(&row[D + d]);     // value
            __m512 dy = bf16_loadu_cvt_fp32(&dy_row[d]);

            __m512 s = sigmoid512_fast_bf16(a);             // sigmoid(a)
            __m512 silu = _mm512_mul_ps(a, s);              // silu(a) = a * s
            __m512 s_prime = _mm512_mul_ps(s, _mm512_sub_ps(one, s)); // s * (1 - s)
            __m512 silu_prime = _mm512_fmadd_ps(a, s_prime, s);       // s + a * s_prime

            // dA = dy * b * silu_prime
            __m512 dA = _mm512_mul_ps(dy, _mm512_mul_ps(b, silu_prime));
            // dB = dy * silu
            __m512 dB = _mm512_mul_ps(dy, silu);

            fp32_cvt_storeu_bf16(&dx_row[d], dA);
            fp32_cvt_storeu_bf16(&dx_row[D + d], dB);
        }
#endif

        // Scalar fallback for remaining elements
        for (; d < D; ++d) {
            float a = bf16_to_float(row[d]);
            float b = bf16_to_float(row[D + d]);
            float dy = bf16_to_float(dy_row[d]);

            float s = sigmoid_scalar(a);
            float silu = a * s;
            float s_prime = s * (1.0f - s);
            float silu_prime = s + a * s_prime;

            float dA = dy * b * silu_prime;
            float dB = dy * silu;

            dx_row[d] = float_to_bf16(dA);
            dx_row[D + d] = float_to_bf16(dB);
        }
    }
}
