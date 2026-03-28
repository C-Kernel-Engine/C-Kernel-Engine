/**
 * @file gemm_kernels_f16.c
 * @brief GEMM kernels with FP16 (half-precision) weights
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
 * Implements matrix multiplication where:
 *   - Weights: FP16 (IEEE half-precision, used by vision encoders)
 *   - Activations: FP32
 *   - Output: FP32
 *
 * Used for multimodal projection layers (mmproj-*.gguf files).
 */

#include <stdint.h>
#include <stddef.h>
#include "ckernel_quant.h"  /* For ck_fp16_to_fp32 */

#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* ============================================================================
 * FP16 Conversion Utilities (if not using F16C)
 * ============================================================================ */

#ifndef __F16C__
/* Software FP16 to FP32 conversion (already in ggml_quants.h) */
#define fp16_to_fp32(x) ggml_fp16_to_fp32(x)
#define fp32_to_fp16(x) ggml_fp32_to_fp16(x)
#else
/* Hardware F16C support */
#include <immintrin.h>
static inline float fp16_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
}
static inline uint16_t fp32_to_fp16(float f) {
    return _cvtss_sh(f, 0);
}
#endif

/* ============================================================================
 * GEMV: y = W @ x  (W is FP16, x and y are FP32)
 * ============================================================================ */

/**
 * @brief Matrix-vector multiply with FP16 weights (scalar reference)
 *
 * @param y Output vector [M]
 * @param W Weight matrix in FP16 [M x K]
 * @param x Input vector [K]
 * @param M Number of output rows
 * @param K Number of columns
 */
void gemv_f16_ref(float *y,
                  const uint16_t *W,
                  const float *x,
                  int M, int K)
{
    for (int row = 0; row < M; row++) {
        float sum = 0.0f;
        const uint16_t *w_row = &W[row * K];

        for (int k = 0; k < K; k++) {
            float w = fp16_to_fp32(w_row[k]);
            sum += w * x[k];
        }

        y[row] = sum;
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-vector multiply with FP16 weights (AVX-512)
 *
 * Converts FP16 to FP32 in registers using VCVTPH2PS.
 */
void gemv_f16_avx512(float *y,
                     const uint16_t *W,
                     const float *x,
                     int M, int K)
{
    const int K16 = K / 16 * 16;

    for (int row = 0; row < M; row++) {
        __m512 acc = _mm512_setzero_ps();
        const uint16_t *w_row = &W[row * K];

        /* Process 16 elements at a time */
        for (int k = 0; k < K16; k += 16) {
            /* Load 16 x FP16 weights */
            __m256i w_f16 = _mm256_loadu_si256((const __m256i *)&w_row[k]);

            /* Convert FP16 to FP32 */
            __m512 w_f32 = _mm512_cvtph_ps(w_f16);

            /* Load 16 x FP32 inputs */
            __m512 x_vec = _mm512_loadu_ps(&x[k]);

            /* FMA */
            acc = _mm512_fmadd_ps(w_f32, x_vec, acc);
        }

        /* Horizontal sum */
        float sum = _mm512_reduce_add_ps(acc);

        /* Handle remainder */
        for (int k = K16; k < K; k++) {
            sum += fp16_to_fp32(w_row[k]) * x[k];
        }

        y[row] = sum;
    }
}
#endif /* __AVX512F__ */

/**
 * @brief Auto-dispatch GEMV based on available SIMD
 */
void gemv_f16(float *y,
              const uint16_t *W,
              const float *x,
              int M, int K)
{
#ifdef __AVX512F__
    gemv_f16_avx512(y, W, x, M, K);
#else
    gemv_f16_ref(y, W, x, M, K);
#endif
}

/* ============================================================================
 * GEMM: Y = W @ X  (W is FP16, X and Y are FP32)
 * ============================================================================ */

/**
 * @brief Matrix-matrix multiply with FP16 weights (scalar reference)
 *
 * @param Y Output matrix [M x N]
 * @param W Weight matrix in FP16 [M x K]
 * @param X Input matrix [K x N]
 * @param M Number of output rows
 * @param N Batch size
 * @param K Hidden dimension
 */
void gemm_f16_ref(float *Y,
                  const uint16_t *W,
                  const float *X,
                  int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_f16_ref(&Y[n * M], W, &X[n * K], M, K);
    }
}

#ifdef __AVX512F__
/**
 * @brief Matrix-matrix multiply with FP16 weights (AVX-512)
 */
void gemm_f16_avx512(float *Y,
                     const uint16_t *W,
                     const float *X,
                     int M, int N, int K)
{
    const int K16 = K / 16 * 16;

    for (int row = 0; row < M; row++) {
        const uint16_t *w_row = &W[row * K];

        /* Pre-convert weight row to FP32 in cache-sized chunks */
        /* For now, convert on-the-fly per batch element */

        for (int n = 0; n < N; n++) {
            __m512 acc = _mm512_setzero_ps();
            const float *x_col = &X[n * K];

            for (int k = 0; k < K16; k += 16) {
                __m256i w_f16 = _mm256_loadu_si256((const __m256i *)&w_row[k]);
                __m512 w_f32 = _mm512_cvtph_ps(w_f16);
                __m512 x_vec = _mm512_loadu_ps(&x_col[k]);
                acc = _mm512_fmadd_ps(w_f32, x_vec, acc);
            }

            float sum = _mm512_reduce_add_ps(acc);

            for (int k = K16; k < K; k++) {
                sum += fp16_to_fp32(w_row[k]) * x_col[k];
            }

            Y[n * M + row] = sum;
        }
    }
}
#endif /* __AVX512F__ */

/**
 * @brief Auto-dispatch GEMM based on available SIMD
 */
void gemm_f16(float *Y,
              const uint16_t *W,
              const float *X,
              int M, int N, int K)
{
#ifdef __AVX512F__
    gemm_f16_avx512(Y, W, X, M, N, K);
#else
    gemm_f16_ref(Y, W, X, M, N, K);
#endif
}

/**
 * @brief NT GEMM wrapper for FP16 weights with the engine's standard ABI.
 *
 * Contract:
 *   A: [M, K] fp32 activation matrix
 *   B: [N, K] fp16 weight matrix stored row-major (transposed layout)
 *   C: [M, N] fp32 output matrix
 *
 * Internally gemm_f16 expects:
 *   W: [output_rows, K]
 *   X: batch-major [batch, K]
 *   Y: batch-major [batch, output_rows]
 * so we swap the output/batch dimensions when dispatching.
 */
void gemm_nt_f16(const float *A,
                 const void *B,
                 const float *bias,
                 float *C,
                 int M, int N, int K)
{
    gemm_f16(C, (const uint16_t *)B, A, N, M, K);

    if (!bias) {
        return;
    }

    for (int i = 0; i < M; ++i) {
        float *c_row = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            c_row[j] += bias[j];
        }
    }
}

/* ============================================================================
 * FP16 Tensor Conversion Utilities
 * ============================================================================ */

/**
 * @brief Convert FP16 tensor to FP32
 */
void convert_f16_to_f32(float *dst, const uint16_t *src, size_t count)
{
#ifdef __AVX512F__
    const size_t count16 = count / 16 * 16;

    for (size_t i = 0; i < count16; i += 16) {
        __m256i f16 = _mm256_loadu_si256((const __m256i *)&src[i]);
        __m512 f32 = _mm512_cvtph_ps(f16);
        _mm512_storeu_ps(&dst[i], f32);
    }

    for (size_t i = count16; i < count; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
#else
    for (size_t i = 0; i < count; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
#endif
}

/**
 * @brief Convert FP32 tensor to FP16
 */
void convert_f32_to_f16(uint16_t *dst, const float *src, size_t count)
{
#ifdef __AVX512F__
    const size_t count16 = count / 16 * 16;

    for (size_t i = 0; i < count16; i += 16) {
        __m512 f32 = _mm512_loadu_ps(&src[i]);
        __m256i f16 = _mm512_cvtps_ph(f32, 0);
        _mm256_storeu_si256((__m256i *)&dst[i], f16);
    }

    for (size_t i = count16; i < count; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
#else
    for (size_t i = 0; i < count; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
#endif
}

/* ============================================================================
 * Backward Pass: Gradient w.r.t. Input
 *
 * Given: dL/dY (gradient of loss w.r.t. output)
 * Compute: dL/dX = W^T @ dL/dY
 *
 * For F16 weights, we convert to FP32 on-the-fly during backprop.
 * ============================================================================ */

/**
 * @brief Backward pass: compute input gradient (scalar reference)
 *
 * @param dX Output gradient w.r.t. input [K]
 * @param W Weight matrix in FP16 format [M x K]
 * @param dY Gradient w.r.t. output [M]
 * @param M Number of output rows
 * @param K Number of columns (input dimension)
 */
void gemv_f16_backward_ref(float *dX,
                           const uint16_t *W,
                           const float *dY,
                           int M, int K)
{
    /* Zero output gradient */
    for (int k = 0; k < K; k++) {
        dX[k] = 0.0f;
    }

    /* Accumulate: dX += W^T @ dY */
    for (int row = 0; row < M; row++) {
        const float dy = dY[row];
        const uint16_t *w_row = &W[row * K];

        for (int k = 0; k < K; k++) {
            float w = fp16_to_fp32(w_row[k]);
            dX[k] += w * dy;
        }
    }
}

#ifdef __AVX512F__
/**
 * @brief Backward pass with AVX-512
 */
void gemv_f16_backward_avx512(float *dX,
                              const uint16_t *W,
                              const float *dY,
                              int M, int K)
{
    const int K16 = K / 16 * 16;

    /* Zero output */
    for (int k = 0; k < K16; k += 16) {
        _mm512_storeu_ps(&dX[k], _mm512_setzero_ps());
    }
    for (int k = K16; k < K; k++) {
        dX[k] = 0.0f;
    }

    for (int row = 0; row < M; row++) {
        const __m512 vdy = _mm512_set1_ps(dY[row]);
        const uint16_t *w_row = &W[row * K];

        for (int k = 0; k < K16; k += 16) {
            /* Load and convert F16 weights */
            __m256i w_f16 = _mm256_loadu_si256((const __m256i *)&w_row[k]);
            __m512 w_f32 = _mm512_cvtph_ps(w_f16);

            /* Compute gradient */
            __m512 grad = _mm512_mul_ps(w_f32, vdy);

            /* Accumulate */
            __m512 dx_cur = _mm512_loadu_ps(&dX[k]);
            _mm512_storeu_ps(&dX[k], _mm512_add_ps(dx_cur, grad));
        }

        /* Remainder */
        for (int k = K16; k < K; k++) {
            dX[k] += fp16_to_fp32(w_row[k]) * dY[row];
        }
    }
}
#endif

/**
 * @brief Auto-dispatch backward
 */
void gemv_f16_backward(float *dX,
                       const uint16_t *W,
                       const float *dY,
                       int M, int K)
{
#ifdef __AVX512F__
    gemv_f16_backward_avx512(dX, W, dY, M, K);
#else
    gemv_f16_backward_ref(dX, W, dY, M, K);
#endif
}

/**
 * @brief Batched backward pass
 */
void gemm_f16_backward(float *dX,
                       const uint16_t *W,
                       const float *dY,
                       int M, int N, int K)
{
    for (int n = 0; n < N; n++) {
        gemv_f16_backward(&dX[n * K], W, &dY[n * M], M, K);
    }
}

/* ============================================================================
 * Dot Product Utility
 * ============================================================================ */

float dot_f16(const uint16_t *w_f16, const float *x, int K)
{
    float result;
    gemv_f16(&result, w_f16, x, 1, K);
    return result;
}
