/**
 * @file gemm_kernels_bf16.c
 * @brief Optimized BF16 GEMM Kernels for AVX-512
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
 * Layout:
 *   A: [M x K] row-major (BF16)
 *   B: [N x K] row-major, stored as [out x in] (BF16)
 *   C: [M x N] row-major (BF16 or FP32)
 *
 * Key optimizations:
 *   1. AVX-512 BF16 instructions (VDPBF16PS) when available
 *   2. Cache blocking for L1/L2 efficiency
 *   3. Vectorized BF16<->FP32 conversion
 *   4. OpenMP parallelization
 */

#include <stdint.h>
#include <string.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bf16_utils.h"
#include "ckernel_engine.h"

/* Block sizes tuned for typical L1/L2 cache */
#define BLK_M 64
#define BLK_N 64
#define BLK_K 256

static inline int ck_min_i(int a, int b) { return a < b ? a : b; }

/* ==========================================================================
 * Reference Implementation (scalar, for correctness testing)
 * Kept for debugging/validation but not called in normal operation.
 * ========================================================================== */
__attribute__((unused))
static void gemm_bf16_scalar(const uint16_t *A,
                             const uint16_t *B,
                             const uint16_t *bias,
                             uint16_t *C,
                             int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bf16_to_float(bias[j]) : 0.0f;
            const size_t a_row = (size_t)i * (size_t)K;
            const size_t b_row = (size_t)j * (size_t)K;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_float(A[a_row + k]) * bf16_to_float(B[b_row + k]);
            }
            C[(size_t)i * (size_t)N + j] = float_to_bf16(sum);
        }
    }
}

#if defined(__AVX512F__)

/* ==========================================================================
 * AVX-512F: Vectorized BF16 conversion + FMA
 * Works on all AVX-512 CPUs (no BF16 instruction required)
 *
 * BF16 conversion functions (bf16x16_to_fp32, fp32x16_to_bf16) are now
 * provided by bf16_utils.h for consistency across all kernels.
 * ========================================================================== */

/* BF16 dot product: 16 pairs, accumulate to FP32 */
static inline __m512 bf16_dot16(__m256i a_bf16, __m256i b_bf16, __m512 acc)
{
    __m512 a_fp32 = bf16x16_to_fp32(a_bf16);
    __m512 b_fp32 = bf16x16_to_fp32(b_bf16);
    return _mm512_fmadd_ps(a_fp32, b_fp32, acc);
}

/* ==========================================================================
 * AVX-512 Vectorized GEMM (using AVX-512F, works everywhere)
 * C[M,N] = A[M,K] @ B[N,K].T
 * ========================================================================== */
static void gemm_bf16_avx512(const uint16_t *A,
                             const uint16_t *B,
                             const uint16_t *bias,
                             uint16_t *C,
                             int M, int N, int K)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        const uint16_t *a_row = A + (size_t)i * K;

        for (int j = 0; j < N; ++j) {
            const uint16_t *b_row = B + (size_t)j * K;

            /* Initialize accumulator */
            __m512 sum_vec = _mm512_setzero_ps();

            /* Vectorized inner loop: process 16 elements at a time */
            int k = 0;
            for (; k <= K - 16; k += 16) {
                __m256i a_bf16 = _mm256_loadu_si256((const __m256i *)(a_row + k));
                __m256i b_bf16 = _mm256_loadu_si256((const __m256i *)(b_row + k));
                sum_vec = bf16_dot16(a_bf16, b_bf16, sum_vec);
            }

            /* Horizontal sum */
            float sum = _mm512_reduce_add_ps(sum_vec);

            /* Scalar tail */
            for (; k < K; ++k) {
                sum += bf16_to_float(a_row[k]) * bf16_to_float(b_row[k]);
            }

            /* Add bias */
            if (bias) {
                sum += bf16_to_float(bias[j]);
            }

            C[(size_t)i * N + j] = float_to_bf16(sum);
        }
    }
}

/* ==========================================================================
 * Cache-Blocked AVX-512 GEMM
 * Better memory access pattern for large matrices
 * ========================================================================== */
static void gemm_bf16_blocked_avx512(const uint16_t *A,
                                      const uint16_t *B,
                                      const uint16_t *bias,
                                      uint16_t *C,
                                      int M, int N, int K)
{
    /* Initialize C with bias */
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float b = bias ? bf16_to_float(bias[j]) : 0.0f;
            C[(size_t)i * N + j] = float_to_bf16(b);
        }
    }

    /* Blocked GEMM */
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < M; ii += BLK_M) {
        for (int jj = 0; jj < N; jj += BLK_N) {
            int i_end = ck_min_i(ii + BLK_M, M);
            int j_end = ck_min_i(jj + BLK_N, N);

            /* Local FP32 accumulator for this block */
            float acc[BLK_M][BLK_N];
            for (int i = 0; i < BLK_M; ++i) {
                for (int j = 0; j < BLK_N; ++j) {
                    acc[i][j] = 0.0f;
                }
            }

            /* K-dimension blocking */
            for (int kk = 0; kk < K; kk += BLK_K) {
                int k_end = ck_min_i(kk + BLK_K, K);

                for (int i = ii; i < i_end; ++i) {
                    const uint16_t *a_row = A + (size_t)i * K;
                    int local_i = i - ii;

                    for (int j = jj; j < j_end; ++j) {
                        const uint16_t *b_row = B + (size_t)j * K;
                        int local_j = j - jj;

                        __m512 sum_vec = _mm512_setzero_ps();

                        int k = kk;
                        for (; k <= k_end - 16; k += 16) {
                            __m256i a_bf16 = _mm256_loadu_si256((const __m256i *)(a_row + k));
                            __m256i b_bf16 = _mm256_loadu_si256((const __m256i *)(b_row + k));
                            sum_vec = bf16_dot16(a_bf16, b_bf16, sum_vec);
                        }

                        float partial = _mm512_reduce_add_ps(sum_vec);
                        for (; k < k_end; ++k) {
                            partial += bf16_to_float(a_row[k]) * bf16_to_float(b_row[k]);
                        }

                        acc[local_i][local_j] += partial;
                    }
                }
            }

            /* Write accumulated results back */
            for (int i = ii; i < i_end; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    float old_val = bf16_to_float(C[(size_t)i * N + j]);
                    float new_val = old_val + acc[i - ii][j - jj];
                    C[(size_t)i * N + j] = float_to_bf16(new_val);
                }
            }
        }
    }
}

/*
 * Native AVX-512 BF16 support (VDPBF16PS instruction)
 * Only compiles on Ice Lake / Sapphire Rapids or newer
 * Compile with: -mavx512bf16 (gcc/clang) or /arch:AVX512 (MSVC with recent SDK)
 */
#if defined(__AVX512BF16__) && defined(__AVX512VL__)

/* Load 32 BF16 values into __m512bh */
static inline __m512bh load_bf16x32(const uint16_t *ptr)
{
    return (__m512bh)_mm512_loadu_si512((const __m512i *)ptr);
}

static void gemm_bf16_native(const uint16_t *A,
                              const uint16_t *B,
                              const uint16_t *bias,
                              uint16_t *C,
                              int M, int N, int K)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            /* Initialize accumulator */
            __m512 sum_vec = _mm512_setzero_ps();

            /* Native BF16 dot product: 32 pairs per instruction! */
            int k = 0;
            for (; k <= K - 32; k += 32) {
                __m512bh a_vec = load_bf16x32(A + (size_t)i * K + k);
                __m512bh b_vec = load_bf16x32(B + (size_t)j * K + k);
                sum_vec = _mm512_dpbf16_ps(sum_vec, a_vec, b_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            /* Scalar tail */
            for (; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)i * K + k]) *
                       bf16_to_float(B[(size_t)j * K + k]);
            }

            if (bias) {
                sum += bf16_to_float(bias[j]);
            }

            C[(size_t)i * N + j] = float_to_bf16(sum);
        }
    }
}

#define HAVE_NATIVE_BF16 1
#else
#define HAVE_NATIVE_BF16 0
#endif /* __AVX512BF16__ && __AVX512VL__ */

#endif /* __AVX512F__ */

/* ==========================================================================
 * Public API: Auto-dispatch to best available implementation
 * ========================================================================== */
void gemm_blocked_serial_bf16(const uint16_t *A,
                              const uint16_t *B,
                              const uint16_t *bias,
                              uint16_t *C,
                              int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

#if HAVE_NATIVE_BF16
    /* Native BF16 instructions available (Ice Lake / Sapphire Rapids+) */
    gemm_bf16_native(A, B, bias, C, M, N, K);
#elif defined(__AVX512F__)
    /* Use AVX-512F with software BF16 conversion */
    if (M * N > 4096) {
        gemm_bf16_blocked_avx512(A, B, bias, C, M, N, K);
    } else {
        gemm_bf16_avx512(A, B, bias, C, M, N, K);
    }
#else
    /* Scalar fallback */
    gemm_bf16_scalar(A, B, bias, C, M, N, K);
#endif
}

/* ==========================================================================
 * GEMM with FP32 output (useful for intermediate computations)
 * ========================================================================== */
void gemm_bf16_fp32out(const uint16_t *A,
                       const uint16_t *B,
                       const float *bias,
                       float *C,
                       int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

#if defined(__AVX512F__)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        const uint16_t *a_row = A + (size_t)i * K;

        for (int j = 0; j < N; ++j) {
            const uint16_t *b_row = B + (size_t)j * K;

            __m512 sum_vec = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - 16; k += 16) {
                __m256i a_bf16 = _mm256_loadu_si256((const __m256i *)(a_row + k));
                __m256i b_bf16 = _mm256_loadu_si256((const __m256i *)(b_row + k));
                sum_vec = bf16_dot16(a_bf16, b_bf16, sum_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            for (; k < K; ++k) {
                sum += bf16_to_float(a_row[k]) * bf16_to_float(b_row[k]);
            }

            if (bias) {
                sum += bias[j];
            }

            C[(size_t)i * N + j] = sum;
        }
    }
#else
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)i * K + k]) *
                       bf16_to_float(B[(size_t)j * K + k]);
            }
            C[(size_t)i * N + j] = sum;
        }
    }
#endif
}

/* ==========================================================================
 * Backward kernels for training
 * ========================================================================== */

/* gemm_nn_bf16: C = A @ B (no transpose), for dL/dX computation */
void gemm_nn_bf16(const uint16_t *A,
                  const uint16_t *B,
                  const uint16_t *bias,
                  uint16_t *C,
                  int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

#if defined(__AVX512F__)
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        /* Initialize row with bias */
        int j = 0;
        for (; j <= N - 16; j += 16) {
            __m512 b_vec = bias ? bf16x16_to_fp32(_mm256_loadu_si256((const __m256i *)(bias + j)))
                                : _mm512_setzero_ps();
            __m256i out = fp32x16_to_bf16(b_vec);
            _mm256_storeu_si256((__m256i *)(C + (size_t)i * N + j), out);
        }
        for (; j < N; ++j) {
            float b = bias ? bf16_to_float(bias[j]) : 0.0f;
            C[(size_t)i * N + j] = float_to_bf16(b);
        }

        /* Accumulate: C[i,:] += A[i,k] * B[k,:] */
        for (int k = 0; k < K; ++k) {
            float a_val = bf16_to_float(A[(size_t)i * K + k]);
            __m512 a_broadcast = _mm512_set1_ps(a_val);

            j = 0;
            for (; j <= N - 16; j += 16) {
                __m256i b_bf16 = _mm256_loadu_si256((const __m256i *)(B + (size_t)k * N + j));
                __m512 b_fp32 = bf16x16_to_fp32(b_bf16);

                __m256i c_bf16 = _mm256_loadu_si256((const __m256i *)(C + (size_t)i * N + j));
                __m512 c_fp32 = bf16x16_to_fp32(c_bf16);

                c_fp32 = _mm512_fmadd_ps(a_broadcast, b_fp32, c_fp32);

                __m256i c_out = fp32x16_to_bf16(c_fp32);
                _mm256_storeu_si256((__m256i *)(C + (size_t)i * N + j), c_out);
            }
            for (; j < N; ++j) {
                float c_val = bf16_to_float(C[(size_t)i * N + j]);
                c_val += a_val * bf16_to_float(B[(size_t)k * N + j]);
                C[(size_t)i * N + j] = float_to_bf16(c_val);
            }
        }
    }
#else
    /* Scalar fallback */
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bf16_to_float(bias[j]) : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)i * K + k]) *
                       bf16_to_float(B[(size_t)k * N + j]);
            }
            C[(size_t)i * N + j] = float_to_bf16(sum);
        }
    }
#endif
}

/* gemm_tn_bf16: C = A.T @ B, for dL/dW computation */
void gemm_tn_bf16(const uint16_t *A,
                  const uint16_t *B,
                  const uint16_t *bias,
                  uint16_t *C,
                  int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    /* A is [K x M], we want A.T which is [M x K] */
    /* B is [K x N] */
    /* C is [M x N] */

#if defined(__AVX512F__)
    /* Initialize C with bias */
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float b = bias ? bf16_to_float(bias[j]) : 0.0f;
            C[(size_t)i * N + j] = float_to_bf16(b);
        }
    }

    /* Accumulate: C[i,j] += sum_k A[k,i] * B[k,j] */
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m512 sum_vec = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - 16; k += 16) {
                /* Gather A[k:k+16, i] - strided access */
                __m512 a_fp32 = _mm512_setzero_ps();
                for (int kk = 0; kk < 16; ++kk) {
                    float val = bf16_to_float(A[(size_t)(k + kk) * M + i]);
                    a_fp32 = _mm512_mask_mov_ps(a_fp32, 1 << kk, _mm512_set1_ps(val));
                }

                /* Note: B has stride N, so we need to gather element by element */
                __m512 b_fp32 = _mm512_setzero_ps();
                for (int kk = 0; kk < 16; ++kk) {
                    float val = bf16_to_float(B[(size_t)(k + kk) * N + j]);
                    b_fp32 = _mm512_mask_mov_ps(b_fp32, 1 << kk, _mm512_set1_ps(val));
                }

                sum_vec = _mm512_fmadd_ps(a_fp32, b_fp32, sum_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            for (; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)k * M + i]) *
                       bf16_to_float(B[(size_t)k * N + j]);
            }

            float old_val = bf16_to_float(C[(size_t)i * N + j]);
            C[(size_t)i * N + j] = float_to_bf16(old_val + sum);
        }
    }
#else
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bf16_to_float(bias[j]) : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += bf16_to_float(A[(size_t)k * M + i]) *
                       bf16_to_float(B[(size_t)k * N + j]);
            }
            C[(size_t)i * N + j] = float_to_bf16(sum);
        }
    }
#endif
}
