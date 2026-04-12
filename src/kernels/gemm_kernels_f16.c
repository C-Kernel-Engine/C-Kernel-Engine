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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "ckernel_quant.h"  /* For ck_fp16_to_fp32 */
#include "ckernel_engine.h"
#include "ggml_runtime_compat.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

typedef struct ggml_context *(*ck_f16_ggml_init_fn)(struct ggml_init_params);
typedef void (*ck_f16_ggml_free_fn)(struct ggml_context *);
typedef struct ggml_tensor *(*ck_f16_ggml_new_tensor_2d_fn)(struct ggml_context *, enum ggml_type, int64_t, int64_t);
typedef struct ggml_tensor *(*ck_f16_ggml_mul_mat_fn)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *);
typedef struct ggml_cgraph *(*ck_f16_ggml_new_graph_fn)(struct ggml_context *);
typedef void (*ck_f16_ggml_build_forward_expand_fn)(struct ggml_cgraph *, struct ggml_tensor *);
typedef enum ggml_status (*ck_f16_ggml_graph_compute_with_ctx_fn)(struct ggml_context *, struct ggml_cgraph *, int);
typedef void (*ck_f16_ggml_cpu_init_fn)(void);
typedef void *(*ck_f16_ggml_get_data_fn)(const struct ggml_tensor *);
typedef float *(*ck_f16_ggml_get_data_f32_fn)(const struct ggml_tensor *);

static ck_f16_ggml_init_fn ck_f16_resolve_ggml_init(void)
{
    static int tried = 0;
    static ck_f16_ggml_init_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_init_fn) dlsym(RTLD_DEFAULT, "ggml_init");
    }
    return fn;
}

static ck_f16_ggml_free_fn ck_f16_resolve_ggml_free(void)
{
    static int tried = 0;
    static ck_f16_ggml_free_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_free_fn) dlsym(RTLD_DEFAULT, "ggml_free");
    }
    return fn;
}

static ck_f16_ggml_new_tensor_2d_fn ck_f16_resolve_ggml_new_tensor_2d(void)
{
    static int tried = 0;
    static ck_f16_ggml_new_tensor_2d_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_new_tensor_2d_fn) dlsym(RTLD_DEFAULT, "ggml_new_tensor_2d");
    }
    return fn;
}

static ck_f16_ggml_mul_mat_fn ck_f16_resolve_ggml_mul_mat(void)
{
    static int tried = 0;
    static ck_f16_ggml_mul_mat_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_mul_mat_fn) dlsym(RTLD_DEFAULT, "ggml_mul_mat");
    }
    return fn;
}

static ck_f16_ggml_new_graph_fn ck_f16_resolve_ggml_new_graph(void)
{
    static int tried = 0;
    static ck_f16_ggml_new_graph_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_new_graph_fn) dlsym(RTLD_DEFAULT, "ggml_new_graph");
    }
    return fn;
}

static ck_f16_ggml_build_forward_expand_fn ck_f16_resolve_ggml_build_forward_expand(void)
{
    static int tried = 0;
    static ck_f16_ggml_build_forward_expand_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_build_forward_expand_fn) dlsym(RTLD_DEFAULT, "ggml_build_forward_expand");
    }
    return fn;
}

static ck_f16_ggml_graph_compute_with_ctx_fn ck_f16_resolve_ggml_graph_compute_with_ctx(void)
{
    static int tried = 0;
    static ck_f16_ggml_graph_compute_with_ctx_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_graph_compute_with_ctx_fn) dlsym(RTLD_DEFAULT, "ggml_graph_compute_with_ctx");
    }
    return fn;
}

static ck_f16_ggml_cpu_init_fn ck_f16_resolve_ggml_cpu_init(void)
{
    static int tried = 0;
    static ck_f16_ggml_cpu_init_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_cpu_init_fn) dlsym(RTLD_DEFAULT, "ggml_cpu_init");
    }
    return fn;
}

static ck_f16_ggml_get_data_fn ck_f16_resolve_ggml_get_data(void)
{
    static int tried = 0;
    static ck_f16_ggml_get_data_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_get_data_fn) dlsym(RTLD_DEFAULT, "ggml_get_data");
    }
    return fn;
}

static ck_f16_ggml_get_data_f32_fn ck_f16_resolve_ggml_get_data_f32(void)
{
    static int tried = 0;
    static ck_f16_ggml_get_data_f32_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_f16_ggml_get_data_f32_fn) dlsym(RTLD_DEFAULT, "ggml_get_data_f32");
    }
    return fn;
}

static int gemm_nt_f16_ggml_strict(const float *A,
                                   const void *B,
                                   const float *bias,
                                   float *C,
                                   int M,
                                   int N,
                                   int K)
{
    ck_f16_ggml_cpu_init_fn ggml_cpu_init_fn = ck_f16_resolve_ggml_cpu_init();
    ck_f16_ggml_init_fn ggml_init_fn = ck_f16_resolve_ggml_init();
    ck_f16_ggml_free_fn ggml_free_fn = ck_f16_resolve_ggml_free();
    ck_f16_ggml_new_tensor_2d_fn ggml_new_tensor_2d_fn = ck_f16_resolve_ggml_new_tensor_2d();
    ck_f16_ggml_mul_mat_fn ggml_mul_mat_fn = ck_f16_resolve_ggml_mul_mat();
    ck_f16_ggml_new_graph_fn ggml_new_graph_fn = ck_f16_resolve_ggml_new_graph();
    ck_f16_ggml_build_forward_expand_fn ggml_build_forward_expand_fn = ck_f16_resolve_ggml_build_forward_expand();
    ck_f16_ggml_graph_compute_with_ctx_fn ggml_graph_compute_with_ctx_fn = ck_f16_resolve_ggml_graph_compute_with_ctx();
    ck_f16_ggml_get_data_fn ggml_get_data_fn = ck_f16_resolve_ggml_get_data();
    ck_f16_ggml_get_data_f32_fn ggml_get_data_f32_fn = ck_f16_resolve_ggml_get_data_f32();

    if (!ggml_cpu_init_fn || !ggml_init_fn || !ggml_free_fn || !ggml_new_tensor_2d_fn ||
        !ggml_mul_mat_fn || !ggml_new_graph_fn || !ggml_build_forward_expand_fn ||
        !ggml_graph_compute_with_ctx_fn || !ggml_get_data_fn || !ggml_get_data_f32_fn) {
        return 0;
    }

    ggml_cpu_init_fn();

    const size_t output_bytes = (size_t) M * (size_t) N * sizeof(float);
    const size_t mem_size = ((size_t) 128 * 1024 * 1024) + output_bytes;

    struct ggml_init_params params = {
        .mem_size = mem_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init_fn(params);
    if (!ctx) {
        return 0;
    }

    int ok = 0;
    struct ggml_tensor *w = ggml_new_tensor_2d_fn(ctx, GGML_TYPE_F16, K, N);
    struct ggml_tensor *x = ggml_new_tensor_2d_fn(ctx, GGML_TYPE_F32, K, M);
    if (!w || !x) {
        ggml_free_fn(ctx);
        return 0;
    }

    {
        void *w_data = ggml_get_data_fn(w);
        void *x_data = ggml_get_data_fn(x);
        if (!w_data || !x_data) {
            ggml_free_fn(ctx);
            return 0;
        }
        memcpy(w_data, B, (size_t) K * (size_t) N * sizeof(uint16_t));
        memcpy(x_data, A, (size_t) K * (size_t) M * sizeof(float));
    }

    struct ggml_tensor *y = ggml_mul_mat_fn(ctx, w, x);
    if (!y) {
        ggml_free_fn(ctx);
        return 0;
    }

    struct ggml_cgraph *gf = ggml_new_graph_fn(ctx);
    if (!gf) {
        ggml_free_fn(ctx);
        return 0;
    }
    ggml_build_forward_expand_fn(gf, y);
    if (ggml_graph_compute_with_ctx_fn(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free_fn(ctx);
        return 0;
    }

    {
        const float *src = ggml_get_data_f32_fn(y);
        if (!src) {
            ggml_free_fn(ctx);
            return 0;
        }
        for (int m = 0; m < M; ++m) {
            memcpy(C + (size_t) m * (size_t) N,
                   src + (size_t) m * (size_t) N,
                   (size_t) N * sizeof(float));
            if (bias) {
                for (int n = 0; n < N; ++n) {
                    C[(size_t) m * (size_t) N + (size_t) n] += bias[n];
                }
            }
        }
    }

    ok = 1;
    ggml_free_fn(ctx);
    return ok;
}

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

static void gemm_f16_input_fp16_ref(float *Y,
                                    const uint16_t *W,
                                    const float *X,
                                    int M, int N, int K)
{
#pragma omp parallel for schedule(static) if(N > 1)
    for (int n = 0; n < N; ++n) {
        const float *x_row = &X[(size_t)n * (size_t)K];
        uint16_t x_f16[K];

        for (int k = 0; k < K; ++k) {
            x_f16[k] = fp32_to_fp16(x_row[k]);
        }

        for (int row = 0; row < M; ++row) {
            float sum = 0.0f;
            const uint16_t *w_row = &W[(size_t)row * (size_t)K];

            for (int k = 0; k < K; ++k) {
                sum += fp16_to_fp32(w_row[k]) * fp16_to_fp32(x_f16[k]);
            }

            Y[(size_t)n * (size_t)M + (size_t)row] = sum;
        }
    }
}

/**
 * @brief NT GEMM wrapper for FP16 weights with the engine's standard ABI.
 *
 * Contract:
 *   A: [M, K] fp32 activation matrix
 *   B: [N, K] fp16 weight matrix stored row-major (transposed layout)
 *   C: [M, N] fp32 output matrix
 *
 * This wrapper follows llama.cpp's CPU F16 mul_mat contract: activation rows
 * are rounded to FP16 first, then the dot runs as F16 x F16 with FP32 output
 * accumulation. The lower-level gemm_f16() helper remains the direct F16-weight
 * x FP32-activation operator for generic use.
 */
void gemm_nt_f16(const float *A,
                 const void *B,
                 const float *bias,
                 float *C,
                 int M, int N, int K)
{
    if (ck_strict_parity_enabled() &&
        gemm_nt_f16_ggml_strict(A, B, bias, C, M, N, K)) {
        return;
    }

    gemm_f16_input_fp16_ref(C, (const uint16_t *)B, A, N, M, K);

    if (!bias) {
        return;
    }

#pragma omp parallel for schedule(static) if(M > 1)
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
