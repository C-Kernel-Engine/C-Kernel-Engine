/**
 * @file add_kernels_bf16.c
 * @brief Element-wise addition kernels for BF16 tensors
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
 * Used for residual connections in transformer models:
 *   residual = x + sublayer_output
 *
 * Supports:
 *   - Forward: y = a + b
 *   - Forward with scale: y = a + alpha * b
 *   - Backward: d_a = d_y, d_b = d_y (gradient flows through unchanged)
 *   - In-place: a += b
 */

#include "bf16_utils.h"
#include "ckernel_engine.h"

#include <stdint.h>
#include <stddef.h>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

/* =============================================================================
 * Forward: y = a + b
 * ============================================================================= */

void add_forward_bf16(const uint16_t *a,
                      const uint16_t *b,
                      uint16_t *y,
                      size_t n)
{
    if (!a || !b || !y || n == 0) {
        return;
    }

    size_t i = 0;

#if defined(__AVX512F__)
    /* AVX-512: Process 16 bf16 elements at a time */
    for (; i + 16 <= n; i += 16) {
        __m512 av = bf16_loadu_cvt_fp32(&a[i]);
        __m512 bv = bf16_loadu_cvt_fp32(&b[i]);
        __m512 yv = _mm512_add_ps(av, bv);
        fp32_cvt_storeu_bf16(&y[i], yv);
    }
#endif

    /* Scalar fallback */
    for (; i < n; ++i) {
        float af = bf16_to_float(a[i]);
        float bf = bf16_to_float(b[i]);
        y[i] = float_to_bf16(af + bf);
    }
}

/* =============================================================================
 * Forward with scale: y = a + alpha * b
 * Useful for gradient accumulation or weighted residuals
 * ============================================================================= */

void add_scaled_forward_bf16(const uint16_t *a,
                             const uint16_t *b,
                             uint16_t *y,
                             float alpha,
                             size_t n)
{
    if (!a || !b || !y || n == 0) {
        return;
    }

    size_t i = 0;

#if defined(__AVX512F__)
    __m512 alpha_v = _mm512_set1_ps(alpha);
    for (; i + 16 <= n; i += 16) {
        __m512 av = bf16_loadu_cvt_fp32(&a[i]);
        __m512 bv = bf16_loadu_cvt_fp32(&b[i]);
        __m512 yv = _mm512_fmadd_ps(bv, alpha_v, av);  /* a + alpha * b */
        fp32_cvt_storeu_bf16(&y[i], yv);
    }
#endif

    for (; i < n; ++i) {
        float af = bf16_to_float(a[i]);
        float bf = bf16_to_float(b[i]);
        y[i] = float_to_bf16(af + alpha * bf);
    }
}

/* =============================================================================
 * In-place: a += b
 * ============================================================================= */

void add_inplace_bf16(uint16_t *a,
                      const uint16_t *b,
                      size_t n)
{
    if (!a || !b || n == 0) {
        return;
    }

    size_t i = 0;

#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        __m512 av = bf16_loadu_cvt_fp32(&a[i]);
        __m512 bv = bf16_loadu_cvt_fp32(&b[i]);
        __m512 yv = _mm512_add_ps(av, bv);
        fp32_cvt_storeu_bf16(&a[i], yv);
    }
#endif

    for (; i < n; ++i) {
        float af = bf16_to_float(a[i]);
        float bf = bf16_to_float(b[i]);
        a[i] = float_to_bf16(af + bf);
    }
}

/* =============================================================================
 * In-place scaled: a += alpha * b
 * ============================================================================= */

void add_scaled_inplace_bf16(uint16_t *a,
                             const uint16_t *b,
                             float alpha,
                             size_t n)
{
    if (!a || !b || n == 0) {
        return;
    }

    size_t i = 0;

#if defined(__AVX512F__)
    __m512 alpha_v = _mm512_set1_ps(alpha);
    for (; i + 16 <= n; i += 16) {
        __m512 av = bf16_loadu_cvt_fp32(&a[i]);
        __m512 bv = bf16_loadu_cvt_fp32(&b[i]);
        __m512 yv = _mm512_fmadd_ps(bv, alpha_v, av);
        fp32_cvt_storeu_bf16(&a[i], yv);
    }
#endif

    for (; i < n; ++i) {
        float af = bf16_to_float(a[i]);
        float bf = bf16_to_float(b[i]);
        a[i] = float_to_bf16(af + alpha * bf);
    }
}

/* =============================================================================
 * Backward: d_a = d_y, d_b = d_y
 *
 * For y = a + b, gradients pass through unchanged:
 *   dy/da = 1, dy/db = 1
 *
 * This is a simple copy operation, but we provide it for API consistency.
 * If d_a == d_y or d_b == d_y (in-place), no copy needed.
 * ============================================================================= */

void add_backward_bf16(const uint16_t *d_y,
                       uint16_t *d_a,
                       uint16_t *d_b,
                       size_t n)
{
    if (!d_y || n == 0) {
        return;
    }

    size_t i = 0;

    /* Copy to d_a if not in-place */
    if (d_a && d_a != d_y) {
#if defined(__AVX512F__)
        for (; i + 32 <= n; i += 32) {
            __m512i v0 = _mm512_loadu_si512((const __m512i*)&d_y[i]);
            __m512i v1 = _mm512_loadu_si512((const __m512i*)&d_y[i + 32]);
            _mm512_storeu_si512((__m512i*)&d_a[i], v0);
            _mm512_storeu_si512((__m512i*)&d_a[i + 32], v1);
        }
#endif
        for (; i < n; ++i) {
            d_a[i] = d_y[i];
        }
    }

    /* Copy to d_b if not in-place */
    i = 0;
    if (d_b && d_b != d_y) {
#if defined(__AVX512F__)
        for (; i + 32 <= n; i += 32) {
            __m512i v0 = _mm512_loadu_si512((const __m512i*)&d_y[i]);
            __m512i v1 = _mm512_loadu_si512((const __m512i*)&d_y[i + 32]);
            _mm512_storeu_si512((__m512i*)&d_b[i], v0);
            _mm512_storeu_si512((__m512i*)&d_b[i + 32], v1);
        }
#endif
        for (; i < n; ++i) {
            d_b[i] = d_y[i];
        }
    }
}

/* =============================================================================
 * 2D tensor version: add_forward_2d_bf16
 * For [T, D] shaped tensors (common in transformers)
 * ============================================================================= */

void add_forward_2d_bf16(const uint16_t *a,
                         const uint16_t *b,
                         uint16_t *y,
                         int tokens,
                         int dim,
                         int aligned_dim)
{
    if (!a || !b || !y || tokens <= 0 || dim <= 0) {
        return;
    }

    for (int t = 0; t < tokens; ++t) {
        const uint16_t *a_row = a + (size_t)t * aligned_dim;
        const uint16_t *b_row = b + (size_t)t * aligned_dim;
        uint16_t *y_row = y + (size_t)t * aligned_dim;

        int d = 0;

#if defined(__AVX512F__)
        for (; d + 16 <= dim; d += 16) {
            __m512 av = bf16_loadu_cvt_fp32(&a_row[d]);
            __m512 bv = bf16_loadu_cvt_fp32(&b_row[d]);
            __m512 yv = _mm512_add_ps(av, bv);
            fp32_cvt_storeu_bf16(&y_row[d], yv);
        }
#endif

        for (; d < dim; ++d) {
            float af = bf16_to_float(a_row[d]);
            float bf = bf16_to_float(b_row[d]);
            y_row[d] = float_to_bf16(af + bf);
        }
    }
}

/* =============================================================================
 * FP32 versions (for reference/testing)
 * ============================================================================= */

/**
 * Element-wise add: y = a + b
 * @test test_add.py::TestAddForward::test_add_forward_f32
 * @test test_add.py::TestAddForward::test_add_inplace_f32
 * @test test_multi_layer_parity.py::TestMultiLayerParity::test_residual_add
 *
 * Element-wise addition of two vectors.
 *
 * After changes: make test
 */
void add_forward_f32(const float *a,
                     const float *b,
                     float *y,
                     size_t n)
{
    if (!a || !b || !y || n == 0) {
        return;
    }

    size_t i = 0;

#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        __m512 av = _mm512_loadu_ps(&a[i]);
        __m512 bv = _mm512_loadu_ps(&b[i]);
        __m512 yv = _mm512_add_ps(av, bv);
        _mm512_storeu_ps(&y[i], yv);
    }
#endif

#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        __m256 yv = _mm256_add_ps(av, bv);
        _mm256_storeu_ps(&y[i], yv);
    }
#endif

    for (; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}

void add_inplace_f32(float *a,
                     const float *b,
                     size_t n)
{
    if (!a || !b || n == 0) {
        return;
    }

    size_t i = 0;

#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        __m512 av = _mm512_loadu_ps(&a[i]);
        __m512 bv = _mm512_loadu_ps(&b[i]);
        __m512 yv = _mm512_add_ps(av, bv);
        _mm512_storeu_ps(&a[i], yv);
    }
#endif

#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        __m256 yv = _mm256_add_ps(av, bv);
        _mm256_storeu_ps(&a[i], yv);
    }
#endif

    for (; i < n; ++i) {
        a[i] = a[i] + b[i];
    }
}
