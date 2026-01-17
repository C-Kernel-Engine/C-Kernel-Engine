/**
 * @file axpy_kernels.c
 * @brief AXPY kernels for FP32: y = y + alpha * x
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
 * Classic BLAS Level-1 operation used in MoE expert output accumulation.
 * When gathering expert outputs: output += weight[i] * expert_output[i]
 *
 * Operations:
 *   - axpy_f32: y += alpha * x (in-place)
 *   - axpy_strided_f32: strided version for non-contiguous memory
 *   - weighted_sum_f32: sum multiple vectors with weights
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* =============================================================================
 * AXPY: y = y + alpha * x
 *
 * Core operation for MoE expert gathering:
 *   output = sum_i(weight_i * expert_output_i)
 *
 * Implemented as: output += weight * expert_output (called for each expert)
 * ============================================================================= */

/**
 * @brief In-place AXPY: y += alpha * x
 * @test test_axpy.py::TestAXPY::test_axpy_f32
 * @test test_axpy.py::TestAXPY::test_axpy_vs_naive
 *
 * In-place scaled vector addition: y += alpha * x
 * BLAS-like axpy operation.
 *
 * After changes: make test
 */
void axpy_f32(float *y,
              const float *x,
              float alpha,
              int n)
{
    if (!y || !x || n <= 0) {
        return;
    }

    int i = 0;

#ifdef __AVX512F__
    __m512 valpha = _mm512_set1_ps(alpha);
    for (; i + 16 <= n; i += 16) {
        __m512 vy = _mm512_loadu_ps(&y[i]);
        __m512 vx = _mm512_loadu_ps(&x[i]);
        vy = _mm512_fmadd_ps(vx, valpha, vy);  /* y = y + alpha * x */
        _mm512_storeu_ps(&y[i], vy);
    }
#endif

#ifdef __AVX2__
    __m256 valpha256 = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8) {
        __m256 vy = _mm256_loadu_ps(&y[i]);
        __m256 vx = _mm256_loadu_ps(&x[i]);
        vy = _mm256_fmadd_ps(vx, valpha256, vy);
        _mm256_storeu_ps(&y[i], vy);
    }
#endif

    /* Scalar remainder */
    for (; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

/* =============================================================================
 * Scaled copy: y = alpha * x
 *
 * First step when accumulating: initialize output with first expert's result.
 * ============================================================================= */

/**
 * @brief Scaled copy: y = alpha * x
 *
 * @param y Output vector [n]
 * @param x Input vector [n]
 * @param alpha Scalar multiplier
 * @param n Vector length
 */
void scal_copy_f32(float *y,
                   const float *x,
                   float alpha,
                   int n)
{
    if (!y || !x || n <= 0) {
        return;
    }

    int i = 0;

#ifdef __AVX512F__
    __m512 valpha = _mm512_set1_ps(alpha);
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vy = _mm512_mul_ps(vx, valpha);
        _mm512_storeu_ps(&y[i], vy);
    }
#endif

#ifdef __AVX2__
    __m256 valpha256 = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_mul_ps(vx, valpha256);
        _mm256_storeu_ps(&y[i], vy);
    }
#endif

    for (; i < n; i++) {
        y[i] = alpha * x[i];
    }
}

/* =============================================================================
 * Weighted sum: y = sum_i(weights[i] * x[i])
 *
 * Combine multiple expert outputs with their routing weights in one pass.
 * More efficient than multiple axpy calls when all inputs are available.
 * ============================================================================= */

/**
 * @brief Weighted sum of k vectors: y = sum_i(weights[i] * vectors[i])
 *
 * @param y Output vector [n]
 * @param vectors Array of k input vector pointers, each [n]
 * @param weights Array of k scalar weights
 * @param k Number of vectors to combine
 * @param n Vector length
 */
void weighted_sum_f32(float *y,
                      const float **vectors,
                      const float *weights,
                      int k,
                      int n)
{
    if (!y || !vectors || !weights || k <= 0 || n <= 0) {
        return;
    }

    /* Initialize with first vector */
    scal_copy_f32(y, vectors[0], weights[0], n);

    /* Accumulate rest */
    for (int i = 1; i < k; i++) {
        axpy_f32(y, vectors[i], weights[i], n);
    }
}

/* =============================================================================
 * Zero-initialized AXPY accumulation
 *
 * Zero output first, then accumulate. Useful when output may contain garbage.
 * ============================================================================= */

/**
 * @brief Zero output then accumulate: y = 0; y += alpha * x
 *
 * @param y Output vector [n], zeroed then accumulated
 * @param x Input vector [n]
 * @param alpha Scalar multiplier
 * @param n Vector length
 */
void axpy_zero_f32(float *y,
                   const float *x,
                   float alpha,
                   int n)
{
    if (!y || n <= 0) {
        return;
    }

    memset(y, 0, n * sizeof(float));

    if (x) {
        axpy_f32(y, x, alpha, n);
    }
}

/* =============================================================================
 * 2D batched AXPY for [tokens, hidden] shaped tensors
 *
 * Process multiple tokens at once, common in transformer inference.
 * ============================================================================= */

/**
 * @brief Batched AXPY for 2D tensors: Y[t,:] += alpha * X[t,:]
 *
 * @param Y Output tensor [num_tokens, dim]
 * @param X Input tensor [num_tokens, dim]
 * @param alpha Scalar multiplier
 * @param num_tokens Number of tokens
 * @param dim Hidden dimension
 * @param y_stride Stride between Y rows (for alignment)
 * @param x_stride Stride between X rows
 */
void axpy_2d_f32(float *Y,
                 const float *X,
                 float alpha,
                 int num_tokens,
                 int dim,
                 int y_stride,
                 int x_stride)
{
    if (!Y || !X || num_tokens <= 0 || dim <= 0) {
        return;
    }

    /* Default strides if not specified */
    if (y_stride <= 0) y_stride = dim;
    if (x_stride <= 0) x_stride = dim;

    for (int t = 0; t < num_tokens; t++) {
        axpy_f32(Y + t * y_stride, X + t * x_stride, alpha, dim);
    }
}

/* =============================================================================
 * MoE-specific: Accumulate expert output with routing weight
 *
 * Convenience wrapper with clear semantics for MoE usage.
 * ============================================================================= */

/**
 * @brief Accumulate expert output: output += routing_weight * expert_output
 *
 * @param output Token output buffer [hidden_dim], accumulated in place
 * @param expert_output Expert's output for this token [hidden_dim]
 * @param routing_weight Softmax routing weight for this expert
 * @param hidden_dim Hidden dimension
 */
void moe_accumulate_expert_f32(float *output,
                               const float *expert_output,
                               float routing_weight,
                               int hidden_dim)
{
    axpy_f32(output, expert_output, routing_weight, hidden_dim);
}
