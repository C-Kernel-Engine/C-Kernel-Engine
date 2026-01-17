/**
 * @file relu_kernels.c
 * @brief ReLU activation kernels with SIMD (SSE/AVX/AVX512)
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
 * ReLU: y = max(0, x)
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

/* ReLU forward: y = max(0, x) */
void relu_forward(const float *input, float *output, size_t n)
{
    size_t i = 0;

#if defined(__AVX512F__)
    __m512 vzero = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        __m512 vy = _mm512_max_ps(vx, vzero);
        _mm512_storeu_ps(output + i, vy);
    }
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(input + i);
        __m256 vy = _mm256_max_ps(vx, vzero);
        _mm256_storeu_ps(output + i, vy);
    }
#endif

    // Scalar fallback
    for (; i < n; ++i) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : 0.0f;
    }
}

// ReLU forward in-place: x = max(0, x)
void relu_forward_inplace(float *data, size_t n)
{
    size_t i = 0;

#if defined(__AVX512F__)
    __m512 vzero = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 vx = _mm512_loadu_ps(data + i);
        __m512 vy = _mm512_max_ps(vx, vzero);
        _mm512_storeu_ps(data + i, vy);
    }
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(data + i);
        __m256 vy = _mm256_max_ps(vx, vzero);
        _mm256_storeu_ps(data + i, vy);
    }
#endif

    // Scalar fallback
    for (; i < n; ++i) {
        float x = data[i];
        if (x < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// ReLU backward: dx = (x > 0) ? dy : 0
void relu_backward(const float *input,
                   const float *d_output,
                   float *d_input,
                   size_t n)
{
    size_t i = 0;

#if defined(__AVX512F__)
    __m512 vzero = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 vx = _mm512_loadu_ps(input + i);
        __m512 vdy = _mm512_loadu_ps(d_output + i);
        __mmask16 mask = _mm512_cmp_ps_mask(vx, vzero, _CMP_GT_OQ);
        __m512 vdx = _mm512_maskz_mov_ps(mask, vdy);
        _mm512_storeu_ps(d_input + i, vdx);
    }
#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(input + i);
        __m256 vdy = _mm256_loadu_ps(d_output + i);
        // Result is all 1s (0xFFFFFFFF) if true, 0 if false.
        __m256 mask = _mm256_cmp_ps(vx, vzero, _CMP_GT_OQ);
        __m256 vdx = _mm256_and_ps(mask, vdy);
        _mm256_storeu_ps(d_input + i, vdx);
    }
#endif

    // Scalar fallback
    for (; i < n; ++i) {
        d_input[i] = (input[i] > 0.0f) ? d_output[i] : 0.0f;
    }
}
