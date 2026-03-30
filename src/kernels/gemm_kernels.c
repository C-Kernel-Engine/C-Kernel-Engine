/**
 * @file gemm_kernels.c
 * @brief General matrix multiply (GEMM) kernels with SIMD (SSE/AVX/AVX512)
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
 * LEGACY EXCEPTION: This file contains OpenMP for backward compatibility.
 * New kernels should NOT use OpenMP internally.
 *
 * GEMM: C = alpha * A @ B + beta * C (with optional bias)
 */

#include "ckernel_engine.h"
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static inline int ck_min(int a, int b) { return a < b ? a : b; }

static inline void ck_gemm_add_bias(float *C, const float *bias, int M, int N)
{
    if (!bias) {
        return;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        float *c_row = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            c_row[j] += bias[j];
        }
    }
}

// AVX1 horizontal sum helper (no _mm256_reduce_add_ps in AVX1)
#if defined(__AVX__) && !defined(__AVX512F__)
static inline float hsum256_ps(__m256 v) {
    // Sum upper and lower 128-bit lanes
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    // Horizontal add within 128-bit
    __m128 shuf = _mm_movehdup_ps(sum128);  // [1,1,3,3]
    __m128 sums = _mm_add_ps(sum128, shuf); // [0+1,1+1,2+3,3+3]
    shuf = _mm_movehl_ps(shuf, sums);       // [2+3,3+3,...]
    sums = _mm_add_ss(sums, shuf);          // [0+1+2+3,...]
    return _mm_cvtss_f32(sums);
}
#endif

// Fast path for M=1: parallelize across output channels (j).
// This is the common decode-time shape (matrix-vector) and is otherwise single-threaded
// in the blocked GEMM code because M=1 provides no parallelism on the row dimension.
static void gemm_nt_matvec_parallel(const float *A,          // [K]
                                   const float *B,          // [N x K] (row-major, transposed layout)
                                   const float *bias,       // [N] or NULL
                                   float *C,                // [N]
                                   int N,
                                   int K)
{
#pragma omp parallel for schedule(static)
    for (int j = 0; j < N; ++j) {
        const float *b_row = B + (size_t)j * (size_t)K;
        float sum = bias ? bias[j] : 0.0f;

#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k <= K - 16; k += 16) {
            __m512 a_vec = _mm512_loadu_ps(A + k);
            __m512 b_vec = _mm512_loadu_ps(b_row + k);
            acc = _mm512_fmadd_ps(a_vec, b_vec, acc);
        }
        sum += _mm512_reduce_add_ps(acc);
        for (; k < K; ++k) {
            sum += A[k] * b_row[k];
        }
#elif defined(__AVX__)
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k <= K - 8; k += 8) {
            __m256 a_vec = _mm256_loadu_ps(A + k);
            __m256 b_vec = _mm256_loadu_ps(b_row + k);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(a_vec, b_vec));
        }
        sum += hsum256_ps(acc);
        for (; k < K; ++k) {
            sum += A[k] * b_row[k];
        }
#else
        for (int k = 0; k < K; ++k) {
            sum += A[k] * b_row[k];
        }
#endif

        C[j] = sum;
    }
}

static void gemm_naive_serial_double(const float *A,
                                     const float *B,
                                     const float *bias,
                                     float *C,
                                     int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = bias ? (double)bias[j] : 0.0;
            for (int k = 0; k < K; k++) {
                sum += (double)A[i * K + k] * (double)B[j * K + k];
            }
            C[i * N + j] = (float)sum;
        }
    }
}

static void gemm_naive_serial_float(const float *A,
                                    const float *B,
                                    const float *bias,
                                    float *C,
                                    int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// Naive parallel GEMM (reference baseline) – copied from C-Transformer.
void gemm_naive_parallel(const float *A,
                         const float *B,
                         const float *bias,
                         float *C,
                         int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_naive_serial_float(A, B, bias, C, M, N, K);
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            float bias_val = bias ? bias[j] : 0.0f;
            C[i * N + j] = sum + bias_val;
        }
    }
}

// AVX-512 optimized GEMM with AVX1 fallback
void gemm_avx512_parallel(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_naive_serial_float(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m512 sum_vec = _mm512_setzero_ps();
            int k;
            for (k = 0; k <= K - 16; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            float sum = _mm512_reduce_add_ps(sum_vec);
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            float bias_val = bias ? bias[j] : 0.0f;
            C[i * N + j] = sum + bias_val;
        }
    }
#elif defined(__AVX__)
    // AVX1 path: 256-bit vectors, no FMA (use mul + add)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k;
            for (k = 0; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }
            float sum = hsum256_ps(sum_vec);
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            float bias_val = bias ? bias[j] : 0.0f;
            C[i * N + j] = sum + bias_val;
        }
    }
#else
    gemm_naive_parallel(A, B, bias, C, M, N, K);
#endif
}

// Cache-blocked GEMM with fine-grained parallelism and AVX1 fallback
void gemm_fine_grained_parallel(const float *A,
                                const float *B,
                                const float *bias,
                                float *C,
                                int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_naive_serial_float(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    const int block_size = 64;
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
#pragma omp parallel for collapse(3)
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int j_end = ck_min(jj + block_size, N);
                int k_end = ck_min(kk + block_size, K);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#pragma omp atomic
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
#elif defined(__AVX__)
    // AVX1 cache-blocked version
    const int block_size = 32;  // Smaller block for L1 cache
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
#pragma omp parallel for collapse(3)
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int j_end = ck_min(jj + block_size, N);
                int k_end = ck_min(kk + block_size, K);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        __m256 sum_vec = _mm256_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 8; k += 8) {
                            __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                            __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                            sum_vec = _mm256_add_ps(sum_vec, prod);
                        }
                        float partial_sum = hsum256_ps(sum_vec);
                        for (; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#pragma omp atomic
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
#else
    gemm_naive_parallel(A, B, bias, C, M, N, K);
#endif
}

// =============================================================================
// GEMM_NN: C[M,N] = A[M,K] @ B[K,N] + bias[N]
// B is stored row-major as [K,N] (no transpose)
// Used for backward d_input computation: d_input = d_output @ W
// =============================================================================

static void gemm_nn_serial_double(const float *A,
                                  const float *B,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = bias ? (double)bias[j] : 0.0;
            for (int k = 0; k < K; k++) {
                sum += (double)A[i * K + k] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)sum;
        }
    }
}

void gemm_nn_parallel(const float *A,
                      const float *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_nn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void gemm_nn_avx512(const float *A,
                    const float *B,
                    const float *bias,
                    float *C,
                    int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_nn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    // For gemm_nn, we can't vectorize over K easily since B[k,j] has stride N.
    // Instead, vectorize over N (output columns) when N >= 16.
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        int j = 0;
        // Process 16 output columns at a time
        for (; j <= N - 16; j += 16) {
            __m512 sum_vec = bias ? _mm512_loadu_ps(&bias[j]) : _mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m512 a_broadcast = _mm512_set1_ps(A[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                sum_vec = _mm512_fmadd_ps(a_broadcast, b_vec, sum_vec);
            }
            _mm512_storeu_ps(&C[i * N + j], sum_vec);
        }
        // Handle remaining columns
        for (; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#elif defined(__AVX__)
    // AVX1: vectorize over N (8 columns at a time)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        int j = 0;
        for (; j <= N - 8; j += 8) {
            __m256 sum_vec = bias ? _mm256_loadu_ps(&bias[j]) : _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 prod = _mm256_mul_ps(a_broadcast, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }
            _mm256_storeu_ps(&C[i * N + j], sum_vec);
        }
        for (; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#else
    gemm_nn_parallel(A, B, bias, C, M, N, K);
#endif
}

/*
 * Profiling duplicate of gemm_nn_avx512.
 * Keep numerics and shape contract identical; this exists only so VTune/Advisor
 * can attribute runtime to a distinct symbol during A/B experiments.
 */
#if defined(__GNUC__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((noinline, noipa))
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
void gemm_nn_avx512_probe(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_nn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    // For gemm_nn, we can't vectorize over K easily since B[k,j] has stride N.
    // Instead, vectorize over N (output columns) when N >= 16.
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        int j = 0;
        // Process 16 output columns at a time
        for (; j <= N - 16; j += 16) {
            __m512 sum_vec = bias ? _mm512_loadu_ps(&bias[j]) : _mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m512 a_broadcast = _mm512_set1_ps(A[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                sum_vec = _mm512_fmadd_ps(a_broadcast, b_vec, sum_vec);
            }
            _mm512_storeu_ps(&C[i * N + j], sum_vec);
        }
        // Handle remaining columns
        for (; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#elif defined(__AVX__)
    // AVX1: vectorize over N (8 columns at a time)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        int j = 0;
        for (; j <= N - 8; j += 8) {
            __m256 sum_vec = bias ? _mm256_loadu_ps(&bias[j]) : _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 prod = _mm256_mul_ps(a_broadcast, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }
            _mm256_storeu_ps(&C[i * N + j], sum_vec);
        }
        for (; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#else
    gemm_nn_parallel(A, B, bias, C, M, N, K);
#endif
}

static int ck_gemm_nn_impl_probe_enabled(void)
{
    static int cached = -1;
    if (cached != -1) {
        return cached;
    }
    cached = 0;
    const char *v = getenv("CK_GEMM_NN_IMPL");
    if (!v || !v[0]) {
        return cached;
    }
    if (strcmp(v, "probe") == 0 || strcmp(v, "dup") == 0 || strcmp(v, "1") == 0) {
        cached = 1;
    }
    return cached;
}

/* Keep legacy symbol name for ABI stability.
 * Actual ISA path is selected at compile time in gemm_nn_avx512().
 * This wrapper avoids ISA-specific naming at call sites and in new code.
 */
void gemm_nn_simd(const float *A,
                  const float *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
    if (ck_gemm_nn_impl_probe_enabled()) {
        gemm_nn_avx512_probe(A, B, bias, C, M, N, K);
        return;
    }
    gemm_nn_avx512(A, B, bias, C, M, N, K);
}

void gemm_nn_blocked(const float *A,
                     const float *B,
                     const float *bias,
                     float *C,
                     int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_nn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    const int block_size = 64;
#elif defined(__AVX__)
    const int block_size = 32;
#else
    const int block_size = 32;
#endif
    // Initialize C with bias (parallelized)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
    // Blocked multiply-accumulate (parallelized over M blocks)
#pragma omp parallel for
    for (int ii = 0; ii < M; ii += block_size) {
        for (int kk = 0; kk < K; kk += block_size) {
            for (int jj = 0; jj < N; jj += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int k_end = ck_min(kk + block_size, K);
                int j_end = ck_min(jj + block_size, N);

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[i * K + k];
#if defined(__AVX512F__)
                        __m512 a_broadcast = _mm512_set1_ps(a_val);
                        int j;
                        for (j = jj; j <= j_end - 16; j += 16) {
                            __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                            __m512 c_vec = _mm512_loadu_ps(&C[i * N + j]);
                            c_vec = _mm512_fmadd_ps(a_broadcast, b_vec, c_vec);
                            _mm512_storeu_ps(&C[i * N + j], c_vec);
                        }
                        for (; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#elif defined(__AVX__)
                        __m256 a_broadcast = _mm256_set1_ps(a_val);
                        int j;
                        for (j = jj; j <= j_end - 8; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                            __m256 prod = _mm256_mul_ps(a_broadcast, b_vec);
                            c_vec = _mm256_add_ps(c_vec, prod);
                            _mm256_storeu_ps(&C[i * N + j], c_vec);
                        }
                        for (; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#else
                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#endif
                    }
                }
            }
        }
    }
}

// =============================================================================
// GEMM_TN: C[M,N] = A[K,M].T @ B[K,N] + bias[N]
// A is stored row-major as [K,M], B is stored row-major as [K,N]
// Used for backward d_W computation: d_W = d_output.T @ input
// =============================================================================

static void gemm_tn_serial_double(const float *A,
                                  const float *B,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = bias ? (double)bias[j] : 0.0;
            for (int k = 0; k < K; k++) {
                // A.T[i,k] = A[k,i] = A[k*M + i]
                sum += (double)A[k * M + i] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)sum;
        }
    }
}

void gemm_tn_parallel(const float *A,
                      const float *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_tn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void gemm_tn_avx512(const float *A,
                    const float *B,
                    const float *bias,
                    float *C,
                    int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_tn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    // Vectorize over N (output columns)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        int j = 0;
        for (; j <= N - 16; j += 16) {
            __m512 sum_vec = bias ? _mm512_loadu_ps(&bias[j]) : _mm512_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m512 a_broadcast = _mm512_set1_ps(A[k * M + i]);
                __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                sum_vec = _mm512_fmadd_ps(a_broadcast, b_vec, sum_vec);
            }
            _mm512_storeu_ps(&C[i * N + j], sum_vec);
        }
        for (; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#elif defined(__AVX__)
    // AVX1: vectorize over N (8 columns at a time)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        int j = 0;
        for (; j <= N - 8; j += 8) {
            __m256 sum_vec = bias ? _mm256_loadu_ps(&bias[j]) : _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 a_broadcast = _mm256_set1_ps(A[k * M + i]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 prod = _mm256_mul_ps(a_broadcast, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }
            _mm256_storeu_ps(&C[i * N + j], sum_vec);
        }
        for (; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
#else
    gemm_tn_parallel(A, B, bias, C, M, N, K);
#endif
}

void gemm_tn_blocked(const float *A,
                     const float *B,
                     const float *bias,
                     float *C,
                     int M, int N, int K)
{
    if (ck_strict_parity_enabled()) {
        gemm_tn_serial_double(A, B, bias, C, M, N, K);
        return;
    }
#if defined(__AVX512F__)
    const int block_size = 64;
#elif defined(__AVX__)
    const int block_size = 32;
#else
    const int block_size = 32;
#endif
    // Initialize C with bias (parallelized)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
    // Blocked multiply-accumulate (parallelized over M blocks)
#pragma omp parallel for
    for (int ii = 0; ii < M; ii += block_size) {
        for (int kk = 0; kk < K; kk += block_size) {
            for (int jj = 0; jj < N; jj += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int k_end = ck_min(kk + block_size, K);
                int j_end = ck_min(jj + block_size, N);

                for (int k = kk; k < k_end; k++) {
                    for (int i = ii; i < i_end; i++) {
                        float a_val = A[k * M + i];
#if defined(__AVX512F__)
                        __m512 a_broadcast = _mm512_set1_ps(a_val);
                        int j;
                        for (j = jj; j <= j_end - 16; j += 16) {
                            __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                            __m512 c_vec = _mm512_loadu_ps(&C[i * N + j]);
                            c_vec = _mm512_fmadd_ps(a_broadcast, b_vec, c_vec);
                            _mm512_storeu_ps(&C[i * N + j], c_vec);
                        }
                        for (; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#elif defined(__AVX__)
                        __m256 a_broadcast = _mm256_set1_ps(a_val);
                        int j;
                        for (j = jj; j <= j_end - 8; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                            __m256 prod = _mm256_mul_ps(a_broadcast, b_vec);
                            c_vec = _mm256_add_ps(c_vec, prod);
                            _mm256_storeu_ps(&C[i * N + j], c_vec);
                        }
                        for (; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#else
                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
#endif
                    }
                }
            }
        }
    }
}

// =============================================================================
// Original GEMM_NT: C[M,N] = A[M,K] @ B[N,K].T + bias[N]
// B is stored row-major as [N,K] (transposed in the multiply)
// =============================================================================

// Serial cache-blocked GEMM with SIMD (AVX/AVX512).
// Note: B is stored as [N x K] (transposed layout).
void gemm_blocked_serial(const float *A,
                         const float *B,
                         const float *bias,
                         float *C,
                         int M, int N, int K)
{
    // Ensure threads are initialized (auto-detects on first call)
    (void)ck_get_num_threads();

    if (ck_strict_parity_enabled()) {
        gemm_naive_serial_float(A, B, bias, C, M, N, K);
        return;
    }

    // Decode-time matvec (M=1) is extremely common and benefits from parallelism over N.
    // Lower threshold to parallelize more ops; OpenMP overhead is ~1-2μs per barrier.
    // For N*K >= 64K elements, parallel is worthwhile.
    if (M == 1 && (size_t)N * (size_t)K >= 65536) {
        gemm_nt_matvec_parallel(A, B, bias, C, N, K);
        return;
    }

    /*
     * Use gemm_microkernel for large matrices - it uses MKL/oneDNN when available,
     * which is substantially faster than our hand-written SIMD kernels.
     * B is stored as [N x K] (transposed), so we pass B_transposed=1.
     * Note: Use threshold of 32 to avoid numerical precision issues with small matrices.
     */
    if (M >= 32 && N >= 32 && K >= 32) {
        gemm_microkernel(A, B, C, M, N, K, 1);  // B_transposed=1
        ck_gemm_add_bias(C, bias, M, N);
        return;
    }
#if defined(__AVX512F__)
    const int block_size = 64;
#elif defined(__AVX__)
    const int block_size = 32;
#else
    const int block_size = 32;
#endif
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias ? bias[j] : 0.0f;
        }
    }
    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {
                int i_end = ck_min(ii + block_size, M);
                int j_end = ck_min(jj + block_size, N);
                int k_end = ck_min(kk + block_size, K);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
#if defined(__AVX512F__)
                        __m512 sum_vec = _mm512_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 16; k += 16) {
                            __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        float partial_sum = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#elif defined(__AVX__)
                        __m256 sum_vec = _mm256_setzero_ps();
                        int k;
                        for (k = kk; k <= k_end - 8; k += 8) {
                            __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                            __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                            sum_vec = _mm256_add_ps(sum_vec, prod);
                        }
                        float partial_sum = hsum256_ps(sum_vec);
                        for (; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#else
                        float partial_sum = 0.0f;
                        for (int k = kk; k < k_end; k++) {
                            partial_sum += A[i * K + k] * B[j * K + k];
                        }
#endif
                        C[i * N + j] += partial_sum;
                    }
                }
            }
        }
    }
}
