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
 *   4. CKE thread-pool row parallelization
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_ONEDNN
#include <dnnl.h>
#include <pthread.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__linux__) && defined(__AMX_TILE__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bf16_utils.h"
#include "ckernel_engine.h"
#include "ck_threadpool.h"

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

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILE_DATA
#define XFEATURE_XTILE_DATA 18
#endif

typedef struct ck_amx_tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} ck_amx_tile_config;

_Static_assert(sizeof(ck_amx_tile_config) == 64,
               "AMX tile configuration must occupy exactly 64 bytes");

static int ck_amx_request_xtile_data(void)
{
#if defined(__linux__)
    static int state = 0;
    if (state == 1) {
        return 1;
    }
    if (state == -1) {
        return 0;
    }
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILE_DATA);
    state = (rc == 0) ? 1 : -1;
    return state == 1;
#else
    return 0;
#endif
}

static void ck_amx_config_bf16_16x16x32(void)
{
    ck_amx_tile_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;

    cfg.rows[0] = 16;       /* A: 16 rows x 32 BF16 */
    cfg.colsb[0] = 64;
    cfg.rows[1] = 16;       /* B: 16 K-pair rows x 16 BF16-pair columns */
    cfg.colsb[1] = 64;
    cfg.rows[2] = 16;       /* C: 16 rows x 16 FP32 */
    cfg.colsb[2] = 64;
    for (int tile = 3; tile <= 5; ++tile) {
        cfg.rows[tile] = 16;
        cfg.colsb[tile] = 64;
    }

    _tile_loadconfig(&cfg);
}

static void ck_amx_config_bf16_16x16_kblock(int k_block)
{
    ck_amx_tile_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;
    cfg.rows[0] = 16;
    cfg.colsb[0] = (uint16_t)(k_block * (int)sizeof(uint16_t));
    cfg.rows[1] = (uint8_t)(k_block / 2);
    cfg.colsb[1] = 64;
    cfg.rows[2] = 16;
    cfg.colsb[2] = 64;
    /* Keep the palette structurally identical to the proven 16x16x32
     * configuration. */
    for (int tile = 3; tile <= 5; ++tile) {
        cfg.rows[tile] = 16;
        cfg.colsb[tile] = 64;
    }
    /* GCC 11 models ldtilecfg as reading less than its architectural 64-byte
     * operand and can otherwise delete dynamic row/column descriptor stores. */
    __asm__ volatile("" : : "m"(cfg) : "memory");
    _tile_loadconfig(&cfg);
}

static void ck_pack_bf16_ktile_pairs_16x16(uint16_t *dst,
                                             const uint16_t *B,
                                             int K,
                                             int j,
                                             int k)
{
    for (int kp = 0; kp < 16; ++kp) {
        const int k0 = k + kp * 2;
        for (int nn = 0; nn < 16; ++nn) {
            dst[(size_t)kp * 32u + (size_t)nn * 2u + 0u] =
                B[(size_t)(j + nn) * (size_t)K + (size_t)k0];
            dst[(size_t)kp * 32u + (size_t)nn * 2u + 1u] =
                B[(size_t)(j + nn) * (size_t)K + (size_t)(k0 + 1)];
        }
    }
}

static void gemm_bf16_fp32out_amx(const uint16_t *A,
                                  const uint16_t *B,
                                  const float *bias,
                                  float *C,
                                  int M, int N, int K)
{
    ck_amx_config_bf16_16x16x32();

    uint16_t b_tile[16 * 32];

    for (int i = 0; i < M; i += 16) {
        for (int j = 0; j < N; j += 16) {
            _tile_zero(2);

            for (int k = 0; k < K; k += 32) {
                ck_pack_bf16_ktile_pairs_16x16(b_tile, B, K, j, k);
                _tile_loadd(0, A + (size_t)i * (size_t)K + (size_t)k, K * (int)sizeof(uint16_t));
                _tile_loadd(1, b_tile, 32 * (int)sizeof(uint16_t));
                _tile_dpbf16ps(2, 0, 1);
            }

            _tile_stored(2, C + (size_t)i * (size_t)N + (size_t)j, N * (int)sizeof(float));

            if (bias) {
                for (int ii = 0; ii < 16; ++ii) {
                    float *c_row = C + (size_t)(i + ii) * (size_t)N + (size_t)j;
                    for (int jj = 0; jj < 16; ++jj) {
                        c_row[jj] += bias[j + jj];
                    }
                }
            }
        }
    }

    _tile_release();
}

#define HAVE_AMX_BF16 1
#else
#define HAVE_AMX_BF16 0
#endif /* __AMX_TILE__ && __AMX_BF16__ */

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

#if HAVE_NATIVE_BF16
#if HAVE_AMX_BF16
    const char *amx_env = getenv("CK_BF16_AMX");
    if (amx_env && amx_env[0] == '1' &&
        (M % 16) == 0 && (N % 16) == 0 && (K % 32) == 0 &&
        M >= 16 && N >= 16 && K >= 32 && ck_amx_request_xtile_data()) {
        gemm_bf16_fp32out_amx(A, B, bias, C, M, N, K);
        return;
    }
#endif

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        const uint16_t *a_row = A + (size_t)i * K;
        int j = 0;

        for (; j + 4 <= N; j += 4) {
            const uint16_t *b0 = B + (size_t)(j + 0) * K;
            const uint16_t *b1 = B + (size_t)(j + 1) * K;
            const uint16_t *b2 = B + (size_t)(j + 2) * K;
            const uint16_t *b3 = B + (size_t)(j + 3) * K;
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - 32; k += 32) {
                const __m512bh a_vec = load_bf16x32(a_row + k);
                acc0 = _mm512_dpbf16_ps(acc0, a_vec, load_bf16x32(b0 + k));
                acc1 = _mm512_dpbf16_ps(acc1, a_vec, load_bf16x32(b1 + k));
                acc2 = _mm512_dpbf16_ps(acc2, a_vec, load_bf16x32(b2 + k));
                acc3 = _mm512_dpbf16_ps(acc3, a_vec, load_bf16x32(b3 + k));
            }

            float s0 = _mm512_reduce_add_ps(acc0);
            float s1 = _mm512_reduce_add_ps(acc1);
            float s2 = _mm512_reduce_add_ps(acc2);
            float s3 = _mm512_reduce_add_ps(acc3);
            for (; k < K; ++k) {
                const float a = bf16_to_float(a_row[k]);
                s0 += a * bf16_to_float(b0[k]);
                s1 += a * bf16_to_float(b1[k]);
                s2 += a * bf16_to_float(b2[k]);
                s3 += a * bf16_to_float(b3[k]);
            }
            if (bias) {
                s0 += bias[j + 0];
                s1 += bias[j + 1];
                s2 += bias[j + 2];
                s3 += bias[j + 3];
            }
            C[(size_t)i * N + (j + 0)] = s0;
            C[(size_t)i * N + (j + 1)] = s1;
            C[(size_t)i * N + (j + 2)] = s2;
            C[(size_t)i * N + (j + 3)] = s3;
        }

        for (; j < N; ++j) {
            const uint16_t *b_row = B + (size_t)j * K;
            __m512 sum_vec = _mm512_setzero_ps();

            int k = 0;
            for (; k <= K - 32; k += 32) {
                const __m512bh a_vec = load_bf16x32(a_row + k);
                const __m512bh b_vec = load_bf16x32(b_row + k);
                sum_vec = _mm512_dpbf16_ps(sum_vec, a_vec, b_vec);
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
#elif defined(__AVX512F__)
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


/* ============================================================================
 * Inference kernels for exact safetensors/BUMP BF16 weights.
 *
 * The v8 inference graph currently keeps activation streams as FP32.  These
 * wrappers preserve the established quantized/FP16 inference ABI while consuming
 * BF16 row-major weights from safetensors BUMP artifacts:
 *   GEMV: y[M]      = W[M,K] @ bf16_round(x[K])
 *   GEMM: C[M,N]    = bf16_round(A[M,K]) @ W[N,K].T + bias[N]
 *
 * Rounding the FP32 activation to BF16 before multiply gives a closer contract
 * to a BF16 PyTorch model than multiplying full FP32 activations by BF16
 * weights, while still avoiding a separate activation-conversion buffer.
 * ========================================================================== */
static void gemv_bf16_row_range(float *y,
                                const uint16_t *w,
                                const float *x,
                                int M, int K,
                                int row_begin, int row_end)
{
    if (!y || !w || !x || M <= 0 || K <= 0 ||
        row_begin < 0 || row_begin >= row_end || row_end > M) {
        return;
    }

    for (int i = row_begin; i < row_end; ++i) {
        const uint16_t *w_row = w + (size_t)i * (size_t)K;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float xb = bf16_to_float(float_to_bf16(x[k]));
            sum += xb * bf16_to_float(w_row[k]);
        }
        y[i] = sum;
    }
}

void gemv_bf16(float *y,
               const void *W,
               const float *x,
               int M, int K)
{
    gemv_bf16_row_range(
        y, (const uint16_t *)W, x, M, K, 0, M);
}

typedef struct {
    float *y;
    const uint16_t *w;
    const float *x;
    int M;
    int K;
} ck_gemv_bf16_args_t;

static void ck_gemv_bf16_rows(int begin, int end, void *opaque)
{
    const ck_gemv_bf16_args_t *args = (const ck_gemv_bf16_args_t *)opaque;
    gemv_bf16_row_range(
        args->y, args->w, args->x, args->M, args->K, begin, end);
}

void gemv_bf16_parallel_dispatch(float *y,
                                 const void *W,
                                 const float *x,
                                 int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M < 2 ||
        (size_t)M * (size_t)K <= 65536) {
        gemv_bf16(y, W, x, M, K);
        return;
    }

    ck_gemv_bf16_args_t args = {
        .y = y, .w = (const uint16_t *)W, .x = x, .M = M, .K = K,
    };
    int active = ck_threadpool_n_threads(pool);
    if (active > M) active = M;
    int grain = M / (active * 4);
    if (grain < 1) grain = 1;
    ck_threadpool_parallel_for_n(
        pool, active, 0, M, grain, ck_gemv_bf16_rows, &args);
}

static void gemv_bf16_bf16_storage_row_range(float *y,
                                              const uint16_t *w,
                                              const float *x,
                                              int M, int K,
                                              int row_begin, int row_end)
{
    gemv_bf16_row_range(y, w, x, M, K, row_begin, row_end);
    for (int row = row_begin; row < row_end; ++row) {
        y[row] = bf16_to_float(float_to_bf16(y[row]));
    }
}

static void ck_gemv_bf16_storage_rows(int begin, int end, void *opaque)
{
    const ck_gemv_bf16_args_t *args = (const ck_gemv_bf16_args_t *)opaque;
    gemv_bf16_bf16_storage_row_range(
        args->y, args->w, args->x, args->M, args->K, begin, end);
}

void gemv_bf16_bf16_storage_parallel_dispatch(float *y,
                                               const void *W,
                                               const float *x,
                                               int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M < 2 ||
        (size_t)M * (size_t)K <= 65536) {
        gemv_bf16_bf16_storage_row_range(
            y, (const uint16_t *)W, x, M, K, 0, M);
        return;
    }

    ck_gemv_bf16_args_t args = {
        .y = y, .w = (const uint16_t *)W, .x = x, .M = M, .K = K,
    };
    int active = ck_threadpool_n_threads(pool);
    if (active > M) active = M;
    int grain = M / (active * 4);
    if (grain < 1) grain = 1;
    ck_threadpool_parallel_for_n(
        pool, active, 0, M, grain, ck_gemv_bf16_storage_rows, &args);
}

void gemv_bf16_bf16_storage(float *y,
                            const void *W,
                            const float *x,
                            int M, int K)
{
    gemv_bf16_bf16_storage_parallel_dispatch(y, W, x, M, K);
}

void gemm_nt_bf16_row_range(const float *A,
                            const void *B,
                            const float *bias,
                            float *C,
                            int M, int N, int K,
                            int row_begin, int row_end)
{
    const uint16_t *w = (const uint16_t *)B;
    if (!A || !w || !C || M <= 0 || N <= 0 || K <= 0 ||
        row_begin < 0 || row_begin >= row_end || row_end > M) {
        return;
    }

    for (int i = row_begin; i < row_end; ++i) {
        const float *a_row = A + (size_t)i * (size_t)K;
        float *c_row = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            const uint16_t *w_row = w + (size_t)j * (size_t)K;
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                const float ab = bf16_to_float(float_to_bf16(a_row[k]));
                sum += ab * bf16_to_float(w_row[k]);
            }
            c_row[j] = sum;
        }
    }
}

void gemm_nt_bf16(const float *A,
                  const void *B,
                  const float *bias,
                  float *C,
                  int M, int N, int K)
{
    if (M <= 0) return;
    gemm_nt_bf16_row_range(A, B, bias, C, M, N, K, 0, M);
}

typedef struct {
    const float *A;
    const void *B;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
} ck_gemm_nt_bf16_exact_args_t;

static void ck_gemm_nt_bf16_exact_rows(int begin, int end, void *opaque)
{
    const ck_gemm_nt_bf16_exact_args_t *args =
        (const ck_gemm_nt_bf16_exact_args_t *)opaque;
    gemm_nt_bf16_row_range(
        args->A, args->B, args->bias, args->C,
        args->M, args->N, args->K, begin, end);
}

void gemm_nt_bf16_parallel_dispatch(const float *A,
                                    const void *B,
                                    const float *bias,
                                    float *C,
                                    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    const char *disabled = getenv("CK_DISABLE_BF16_GEMM_PARALLEL_PREFILL");
    if ((disabled && disabled[0] && strcmp(disabled, "0") != 0) ||
        !pool || ck_threadpool_n_threads(pool) <= 1 || M < 2 ||
        (size_t)M * (size_t)N <= 4096) {
        gemm_nt_bf16(A, B, bias, C, M, N, K);
        return;
    }

    ck_gemm_nt_bf16_exact_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C, .M = M, .N = N, .K = K,
    };
    int active = ck_threadpool_n_threads(pool);
    if (active > M) active = M;
    int grain = M / (active * 4);
    if (grain < 1) grain = 1;
    ck_threadpool_parallel_for_n(
        pool, active, 0, M, grain, ck_gemm_nt_bf16_exact_rows, &args);
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

/*
 * Mixed-precision BF16 linear backward for training.
 *
 * Forward contract:
 *   Y[t, o] = dot(input[t, :], weight[o, :]) + bias[o]
 *
 * Inputs are BF16 storage, math and gradients are FP32. This mirrors the
 * standard mixed-precision training contract where activations/weights may be
 * BF16 but gradient accumulation remains FP32.
 */
typedef struct {
    const float *A;
    const uint16_t *B;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
} ck_gemm_bf16_native_args_t;

static void ck_gemm_bf16_native_work(int ith, int nth, void *opaque)
{
    ck_gemm_bf16_native_args_t *args = (ck_gemm_bf16_native_args_t *)opaque;
    const int N = args->N;
    const int K = args->K;
    enum { ROW_TILE = 4 };
    uint16_t *a_bf16 = (uint16_t *)alloca(
        (size_t)ROW_TILE * (size_t)K * sizeof(uint16_t));

    for (int row0 = ith * ROW_TILE; row0 < args->M; row0 += nth * ROW_TILE) {
        const int rows = args->M - row0 < ROW_TILE ? args->M - row0 : ROW_TILE;
        for (int r = 0; r < rows; ++r) {
            const float *src = args->A + (size_t)(row0 + r) * (size_t)K;
            uint16_t *ar = a_bf16 + (size_t)r * (size_t)K;
            for (int k = 0; k < K; ++k) ar[k] = float_to_bf16(src[k]);
        }

#if HAVE_NATIVE_BF16
        int j = 0;
        for (; j + 4 <= N; j += 4) {
            const uint16_t *b0 = args->B + (size_t)(j + 0) * K;
            const uint16_t *b1 = args->B + (size_t)(j + 1) * K;
            const uint16_t *b2 = args->B + (size_t)(j + 2) * K;
            const uint16_t *b3 = args->B + (size_t)(j + 3) * K;
            __m512 acc[ROW_TILE][4];
            for (int r = 0; r < rows; ++r) {
                for (int lane = 0; lane < 4; ++lane) acc[r][lane] = _mm512_setzero_ps();
            }
            int k = 0;
            for (; k <= K - 32; k += 32) {
                const __m512bh bv[4] = {
                    load_bf16x32(b0 + k), load_bf16x32(b1 + k),
                    load_bf16x32(b2 + k), load_bf16x32(b3 + k)
                };
                for (int r = 0; r < rows; ++r) {
                    const __m512bh av =
                        load_bf16x32(a_bf16 + (size_t)r * (size_t)K + k);
                    for (int lane = 0; lane < 4; ++lane) {
                        acc[r][lane] = _mm512_dpbf16_ps(acc[r][lane], av, bv[lane]);
                    }
                }
            }
            for (int r = 0; r < rows; ++r) {
                float sums[4];
                for (int lane = 0; lane < 4; ++lane) {
                    sums[lane] = _mm512_reduce_add_ps(acc[r][lane]);
                }
                const uint16_t *ar = a_bf16 + (size_t)r * (size_t)K;
                for (int tail = k; tail < K; ++tail) {
                    const float av = bf16_to_float(ar[tail]);
                    sums[0] += av * bf16_to_float(b0[tail]);
                    sums[1] += av * bf16_to_float(b1[tail]);
                    sums[2] += av * bf16_to_float(b2[tail]);
                    sums[3] += av * bf16_to_float(b3[tail]);
                }
                float *dst = args->C + (size_t)(row0 + r) * (size_t)N;
                for (int lane = 0; lane < 4; ++lane) {
                    if (args->bias) sums[lane] += args->bias[j + lane];
                    dst[j + lane] = bf16_to_float(float_to_bf16(sums[lane]));
                }
            }
        }
        for (; j < N; ++j) {
            const uint16_t *b = args->B + (size_t)j * K;
            __m512 acc[ROW_TILE];
            for (int r = 0; r < rows; ++r) acc[r] = _mm512_setzero_ps();
            int k = 0;
            for (; k <= K - 32; k += 32) {
                const __m512bh bv = load_bf16x32(b + k);
                for (int r = 0; r < rows; ++r) {
                    acc[r] = _mm512_dpbf16_ps(
                        acc[r], load_bf16x32(a_bf16 + (size_t)r * (size_t)K + k), bv);
                }
            }
            for (int r = 0; r < rows; ++r) {
                const uint16_t *ar = a_bf16 + (size_t)r * (size_t)K;
                float sum = _mm512_reduce_add_ps(acc[r]);
                for (int tail = k; tail < K; ++tail) {
                    sum += bf16_to_float(ar[tail]) * bf16_to_float(b[tail]);
                }
                if (args->bias) sum += args->bias[j];
                args->C[(size_t)(row0 + r) * (size_t)N + j] =
                    bf16_to_float(float_to_bf16(sum));
            }
        }
#else
        for (int r = 0; r < rows; ++r) {
            const uint16_t *ar = a_bf16 + (size_t)r * (size_t)K;
            float *dst = args->C + (size_t)(row0 + r) * (size_t)N;
            for (int j = 0; j < N; ++j) {
                const uint16_t *b = args->B + (size_t)j * K;
                float sum = args->bias ? args->bias[j] : 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += bf16_to_float(ar[k]) * bf16_to_float(b[k]);
                }
                dst[j] = bf16_to_float(float_to_bf16(sum));
            }
        }
#endif
    }
}

void gemm_nt_bf16_native_bf16_storage(const float *A,
                                       const void *B,
                                       const float *bias,
                                       float *C,
                                       int M, int N, int K)
{
    const uint16_t *weights = (const uint16_t *)B;
    if (!A || !weights || !C || M <= 0 || N <= 0 || K <= 0) return;

    ck_gemm_bf16_native_args_t args = {
        .A = A, .B = weights, .bias = bias, .C = C, .M = M, .N = N, .K = K
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > M) active = M;
    if (active > 24) active = 24;
    if (!pool || active <= 1 || (size_t)M * (size_t)N <= 4096) {
        ck_gemm_bf16_native_work(0, 1, &args);
        return;
    }
    ck_threadpool_dispatch_n(pool, active, ck_gemm_bf16_native_work, &args);
}

typedef struct {
    const float *src;
    uint16_t *dst;
    size_t count;
} ck_bf16_convert_args_t;

static void ck_bf16_convert_work(int ith, int nth, void *opaque)
{
    ck_bf16_convert_args_t *args = (ck_bf16_convert_args_t *)opaque;
    const size_t begin = args->count * (size_t)ith / (size_t)nth;
    const size_t end = args->count * (size_t)(ith + 1) / (size_t)nth;
    for (size_t i = begin; i < end; ++i) args->dst[i] = float_to_bf16(args->src[i]);
}

typedef struct {
    const uint16_t *A;
    const uint16_t *B;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int failed;
} ck_gemm_bf16_amx_args_t;

static void ck_gemm_bf16_amx_work(int ith, int nth, void *opaque)
{
#if HAVE_AMX_BF16
    ck_gemm_bf16_amx_args_t *args = (ck_gemm_bf16_amx_args_t *)opaque;
    if (!ck_amx_request_xtile_data()) {
        __atomic_store_n(&args->failed, 1, __ATOMIC_RELAXED);
        return;
    }
    ck_amx_config_bf16_16x16x32();
    uint16_t b_tile[16 * 32];
    const int mb = args->M / 16;
    const int nb = args->N / 16;
    const int m_groups = (mb + 3) / 4;
    const int jobs = m_groups * nb;

    for (int job = ith; job < jobs; job += nth) {
        const int m_group = job / nb;
        const int j = (job % nb) * 16;
        const int group_blocks = (mb - m_group * 4 < 4) ? mb - m_group * 4 : 4;
        _tile_zero(2);
        if (group_blocks > 1) _tile_zero(3);
        if (group_blocks > 2) _tile_zero(4);
        if (group_blocks > 3) _tile_zero(5);
        for (int k = 0; k < args->K; k += 32) {
            ck_pack_bf16_ktile_pairs_16x16(b_tile, args->B, args->K, j, k);
            _tile_loadd(1, b_tile, 32 * (int)sizeof(uint16_t));
            for (int g = 0; g < group_blocks; ++g) {
                const int i = (m_group * 4 + g) * 16;
                _tile_loadd(0, args->A + (size_t)i * args->K + k,
                            args->K * (int)sizeof(uint16_t));
                switch (g) {
                    case 0: _tile_dpbf16ps(2, 0, 1); break;
                    case 1: _tile_dpbf16ps(3, 0, 1); break;
                    case 2: _tile_dpbf16ps(4, 0, 1); break;
                    default: _tile_dpbf16ps(5, 0, 1); break;
                }
            }
        }
        for (int g = 0; g < group_blocks; ++g) {
            const int i = (m_group * 4 + g) * 16;
            float *tile_dst = args->C + (size_t)i * args->N + j;
            const int tile_stride = args->N * (int)sizeof(float);
            switch (g) {
                case 0: _tile_stored(2, tile_dst, tile_stride); break;
                case 1: _tile_stored(3, tile_dst, tile_stride); break;
                case 2: _tile_stored(4, tile_dst, tile_stride); break;
                default: _tile_stored(5, tile_dst, tile_stride); break;
            }
            if (args->bias) {
                for (int ii = 0; ii < 16; ++ii) {
                    float *row = args->C + (size_t)(i + ii) * args->N + j;
                    for (int jj = 0; jj < 16; ++jj) row[jj] += args->bias[j + jj];
                }
            }
        }
    }
    _tile_release();
#else
    (void)ith; (void)nth; (void)opaque;
#endif
}

typedef struct {
    float *values;
    size_t count;
} ck_bf16_round_args_t;

static void ck_bf16_round_work(int ith, int nth, void *opaque)
{
    ck_bf16_round_args_t *args = (ck_bf16_round_args_t *)opaque;
    const size_t begin = args->count * (size_t)ith / (size_t)nth;
    const size_t end = args->count * (size_t)(ith + 1) / (size_t)nth;
    for (size_t i = begin; i < end; ++i) {
        args->values[i] = bf16_to_float(float_to_bf16(args->values[i]));
    }
}

int ck_gemm_bf16_amx_available(void)
{
#if HAVE_AMX_BF16
    return ck_amx_request_xtile_data();
#else
    return 0;
#endif
}

int ck_gemm_bf16_fp32out_amx_raw(const uint16_t *A,
                                 const uint16_t *B,
                                 float *C,
                                 int M, int N, int K,
                                 int accumulate)
{
#if HAVE_AMX_BF16
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 ||
        (M % 16) != 0 || (N % 16) != 0 || (K % 2) != 0 ||
        !ck_amx_request_xtile_data()) {
        return 0;
    }
    /* Match oneDNN BRGEMM: largest even divisor of K no greater than 32. */
    int k_block = K < 32 ? K : 32;
    while (k_block > 2 && K % k_block != 0) k_block -= 2;
    ck_amx_config_bf16_16x16_kblock(k_block);
    uint16_t b_tile[16 * 32];
    for (int i = 0; i < M; i += 16) {
        for (int j = 0; j < N; j += 16) {
            if (accumulate) {
                _tile_loadd(2, C + (size_t)i * (size_t)N + (size_t)j,
                            N * (int)sizeof(float));
            } else {
                _tile_zero(2);
            }
            for (int k = 0; k < K; k += k_block) {
                memset(b_tile, 0, sizeof(b_tile));
                for (int kp = 0; kp < k_block / 2; ++kp) {
                    const int k0 = k + kp * 2;
                    for (int nn = 0; nn < 16; ++nn) {
                        b_tile[(size_t)kp * 32u + (size_t)nn * 2u] =
                            B[(size_t)(j + nn) * (size_t)K + (size_t)k0];
                        b_tile[(size_t)kp * 32u + (size_t)nn * 2u + 1u] =
                            B[(size_t)(j + nn) * (size_t)K + (size_t)k0 + 1u];
                    }
                }
                _tile_loadd(0, A + (size_t)i * (size_t)K + (size_t)k,
                            K * (int)sizeof(uint16_t));
                _tile_loadd(1, b_tile, 32 * (int)sizeof(uint16_t));
                _tile_dpbf16ps(2, 0, 1);
            }
            _tile_stored(2, C + (size_t)i * (size_t)N + (size_t)j,
                         N * (int)sizeof(float));
        }
    }
    _tile_release();
    return 1;
#else
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K; (void)accumulate;
    return 0;
#endif
}

void gemm_nt_bf16_amx_bf16_storage_workspace(const float *A,
                                              const void *B,
                                              const float *bias,
                                              float *C,
                                              int M, int N, int K,
                                              uint16_t *a_bf16,
                                              size_t a_bf16_bytes)
{
#if HAVE_AMX_BF16
    if (!A || !B || !C || M < 16 || N < 16 || K < 32 ||
        (M % 16) != 0 || (N % 16) != 0 || (K % 32) != 0) {
        fprintf(stderr,
            "HARD KERNEL CONTRACT FAULT: AMX BF16 GEMM requires non-null buffers "
            "and M%%16=N%%16=K%%32=0 (M=%d N=%d K=%d)\n",
            M, N, K);
        abort();
    }
    const size_t input_count = (size_t)M * K;
    if (!a_bf16 || input_count > SIZE_MAX / sizeof(uint16_t) ||
        a_bf16_bytes < input_count * sizeof(uint16_t)) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: AMX BF16 activation workspace is too small\n");
        abort();
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > 24) active = 24;
    ck_bf16_convert_args_t convert = {.src=A, .dst=a_bf16, .count=input_count};
    if (pool && active > 1) ck_threadpool_dispatch_n(pool, active, ck_bf16_convert_work, &convert);
    else ck_bf16_convert_work(0, 1, &convert);
    ck_gemm_bf16_amx_args_t gemm = {
        .A=a_bf16, .B=(const uint16_t *)B, .bias=bias, .C=C,
        .M=M, .N=N, .K=K, .failed=0
    };
    if (pool && active > 1) ck_threadpool_dispatch_n(pool, active, ck_gemm_bf16_amx_work, &gemm);
    else ck_gemm_bf16_amx_work(0, 1, &gemm);
    if (__atomic_load_n(&gemm.failed, __ATOMIC_RELAXED)) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: AMX tile permission request failed\n");
        abort();
    }
    ck_bf16_round_args_t round = {.values=C, .count=(size_t)M * N};
    if (pool && active > 1) ck_threadpool_dispatch_n(pool, active, ck_bf16_round_work, &round);
    else ck_bf16_round_work(0, 1, &round);
    return;
#else
    (void)A; (void)B; (void)bias; (void)C; (void)M; (void)N; (void)K;
    (void)a_bf16; (void)a_bf16_bytes;
    fprintf(stderr,
            "HARD KERNEL CONTRACT FAULT: gemm_nt_bf16_amx_bf16_storage was selected "
            "without AMX BF16 support\n");
    abort();
#endif
}

void gemm_nt_bf16_amx_bf16_storage(const float *A,
                                    const void *B,
                                    const float *bias,
                                    float *C,
                                    int M, int N, int K)
{
    size_t input_count = 0;
    if (M > 0 && K > 0 && (size_t)M <= SIZE_MAX / (size_t)K) {
        input_count = (size_t)M * (size_t)K;
    }
    uint16_t *workspace = input_count > 0 && input_count <= SIZE_MAX / sizeof(uint16_t)
        ? (uint16_t *)malloc(input_count * sizeof(uint16_t))
        : NULL;
    if (!workspace) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: AMX BF16 compatibility workspace allocation failed\n");
        abort();
    }
    gemm_nt_bf16_amx_bf16_storage_workspace(
        A, B, bias, C, M, N, K, workspace, input_count * sizeof(uint16_t));
    free(workspace);
}

void gemm_nt_bf16_prefill_shape_safe_bf16_storage_workspace(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K, uint16_t *a_bf16, size_t a_bf16_bytes)
{
    const int amx_shape = M >= 16 && N >= 16 && K >= 32 &&
                          (M % 16) == 0 && (N % 16) == 0 && (K % 32) == 0;
    if (amx_shape && ck_gemm_bf16_amx_available()) {
        gemm_nt_bf16_amx_bf16_storage_workspace(
            A, B, bias, C, M, N, K, a_bf16, a_bf16_bytes);
        return;
    }
    gemm_nt_bf16_native_bf16_storage(A, B, bias, C, M, N, K);
}

void gemm_nt_bf16_prefill_shape_safe_bf16_storage(const float *A,
                                                   const void *B,
                                                   const float *bias,
                                                   float *C,
                                                   int M, int N, int K)
{
    const int amx_shape = M >= 16 && N >= 16 && K >= 32 &&
                          (M % 16) == 0 && (N % 16) == 0 && (K % 32) == 0;
    if (amx_shape && ck_gemm_bf16_amx_available()) {
        gemm_nt_bf16_amx_bf16_storage(A, B, bias, C, M, N, K);
        return;
    }
    gemm_nt_bf16_native_bf16_storage(A, B, bias, C, M, N, K);
}

#ifdef USE_ONEDNN
static dnnl_engine_t ck_pytorch_brgemm_engine;
static dnnl_stream_t ck_pytorch_brgemm_stream;
static int ck_pytorch_brgemm_init_status = -1;
static pthread_once_t ck_pytorch_brgemm_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t ck_pytorch_brgemm_lock = PTHREAD_MUTEX_INITIALIZER;
static const dnnl_version_t *ck_pytorch_brgemm_version;

static void ck_pytorch_brgemm_init(void)
{
    const dnnl_version_t *version = dnnl_version();
    if (!version || version->cpu_runtime != DNNL_RUNTIME_OMP || !version->hash) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: PyTorch oneDNN BF16 BRGEMM requires "
                "an identity-bearing OpenMP oneDNN runtime "
                "(found %d.%d.%d runtime=%u hash=%s)\n",
                version ? version->major : -1,
                version ? version->minor : -1,
                version ? version->patch : -1,
                version ? version->cpu_runtime : 0,
                version && version->hash ? version->hash : "<missing>");
        return;
    }
    ck_pytorch_brgemm_version = version;
    if (dnnl_engine_create(&ck_pytorch_brgemm_engine, dnnl_cpu, 0) != dnnl_success) return;
    if (dnnl_stream_create(&ck_pytorch_brgemm_stream, ck_pytorch_brgemm_engine,
                           dnnl_stream_default_flags) != dnnl_success) {
        dnnl_engine_destroy(ck_pytorch_brgemm_engine);
        ck_pytorch_brgemm_engine = NULL;
        return;
    }
    ck_pytorch_brgemm_init_status = 0;
}

static void ck_pytorch_brgemm_fault(const char *message, int M, int N, int K)
{
    fprintf(stderr, "HARD KERNEL CONTRACT FAULT: PyTorch oneDNN BF16 BRGEMM %s "
                    "(M=%d N=%d K=%d)\n", message, M, N, K);
    abort();
}

static void ck_pytorch_brgemm_require_version(int major, int minor, int patch,
                                               const char *source_hash,
                                               const char *provider,
                                               int M, int N, int K)
{
    pthread_once(&ck_pytorch_brgemm_once, ck_pytorch_brgemm_init);
    if (ck_pytorch_brgemm_init_status != 0 || !ck_pytorch_brgemm_version) {
        ck_pytorch_brgemm_fault("could not initialize oneDNN", M, N, K);
    }
    if (ck_pytorch_brgemm_version->major != major ||
        ck_pytorch_brgemm_version->minor != minor ||
        ck_pytorch_brgemm_version->patch != patch ||
        strcmp(ck_pytorch_brgemm_version->hash, source_hash) != 0) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: %s requires oneDNN %d.%d.%d "
                "OpenMP at %s (found %d.%d.%d runtime=%u hash=%s)\n",
                provider, major, minor, patch, source_hash,
                ck_pytorch_brgemm_version->major,
                ck_pytorch_brgemm_version->minor,
                ck_pytorch_brgemm_version->patch,
                ck_pytorch_brgemm_version->cpu_runtime,
                ck_pytorch_brgemm_version->hash);
        abort();
    }
}
#endif

static void gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage_impl(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
#ifdef USE_ONEDNN
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        ck_pytorch_brgemm_fault("received an invalid tensor contract", M, N, K);
    }
    if (ck_pytorch_brgemm_init_status != 0) {
        ck_pytorch_brgemm_fault("could not initialize oneDNN", M, N, K);
    }

    const size_t input_count = (size_t)M * (size_t)K;
    const size_t output_count = (size_t)M * (size_t)N;
    uint16_t *input_bf16 = (uint16_t *)malloc(input_count * sizeof(*input_bf16));
    uint16_t *output_bf16 = (uint16_t *)malloc(output_count * sizeof(*output_bf16));
    uint16_t *bias_bf16 = bias ? (uint16_t *)malloc((size_t)N * sizeof(*bias_bf16)) : NULL;
    if (!input_bf16 || !output_bf16 || (bias && !bias_bf16)) {
        free(bias_bf16);
        free(output_bf16);
        free(input_bf16);
        ck_pytorch_brgemm_fault("workspace allocation failed", M, N, K);
    }
    for (size_t i = 0; i < input_count; ++i) input_bf16[i] = float_to_bf16(A[i]);
    if (bias) {
        for (int j = 0; j < N; ++j) bias_bf16[j] = float_to_bf16(bias[j]);
        for (int i = 0; i < M; ++i) {
            memcpy(output_bf16 + (size_t)i * (size_t)N,
                   bias_bf16, (size_t)N * sizeof(*bias_bf16));
        }
    }

    dnnl_memory_desc_t src_md = NULL, weights_md = NULL, dst_md = NULL;
    dnnl_primitive_attr_t attr = NULL;
    dnnl_post_ops_t post_ops = NULL;
    dnnl_primitive_desc_t primitive_desc = NULL;
    dnnl_primitive_t primitive = NULL;
    dnnl_memory_t src_mem = NULL, weights_mem = NULL, dst_mem = NULL;
    dnnl_dims_t src_dims = {M, K};
    dnnl_dims_t weights_dims = {K, N};
    dnnl_dims_t dst_dims = {M, N};
    dnnl_dims_t src_strides = {K, 1};
    dnnl_dims_t weights_strides = {1, K};
    dnnl_dims_t dst_strides = {N, 1};
    dnnl_status_t status = dnnl_success;

#define CK_DNNL(call) do { status = (call); if (status != dnnl_success) goto cleanup; } while (0)
    pthread_mutex_lock(&ck_pytorch_brgemm_lock);
    CK_DNNL(dnnl_memory_desc_create_with_strides(&src_md, 2, src_dims, dnnl_bf16, src_strides));
    CK_DNNL(dnnl_memory_desc_create_with_strides(
        &weights_md, 2, weights_dims, dnnl_bf16, weights_strides));
    CK_DNNL(dnnl_memory_desc_create_with_strides(&dst_md, 2, dst_dims, dnnl_bf16, dst_strides));
    if (bias) {
        CK_DNNL(dnnl_primitive_attr_create(&attr));
        CK_DNNL(dnnl_post_ops_create(&post_ops));
        CK_DNNL(dnnl_post_ops_append_sum(post_ops, 1.0f, 0, dnnl_bf16));
        CK_DNNL(dnnl_primitive_attr_set_post_ops(attr, post_ops));
    }
    CK_DNNL(dnnl_matmul_primitive_desc_create(
        &primitive_desc, ck_pytorch_brgemm_engine, src_md, weights_md, NULL, dst_md, attr));
    CK_DNNL(dnnl_primitive_create(&primitive, primitive_desc));
    CK_DNNL(dnnl_memory_create(&src_mem, src_md, ck_pytorch_brgemm_engine, input_bf16));
    CK_DNNL(dnnl_memory_create(
        &weights_mem, weights_md, ck_pytorch_brgemm_engine, (void *)B));
    CK_DNNL(dnnl_memory_create(&dst_mem, dst_md, ck_pytorch_brgemm_engine, output_bf16));
    dnnl_exec_arg_t args[] = {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem},
    };
    CK_DNNL(dnnl_primitive_execute(
        primitive, ck_pytorch_brgemm_stream, (int)(sizeof(args) / sizeof(args[0])), args));
    CK_DNNL(dnnl_stream_wait(ck_pytorch_brgemm_stream));

cleanup:
    if (dst_mem) dnnl_memory_destroy(dst_mem);
    if (weights_mem) dnnl_memory_destroy(weights_mem);
    if (src_mem) dnnl_memory_destroy(src_mem);
    if (primitive) dnnl_primitive_destroy(primitive);
    if (primitive_desc) dnnl_primitive_desc_destroy(primitive_desc);
    if (dst_md) dnnl_memory_desc_destroy(dst_md);
    if (weights_md) dnnl_memory_desc_destroy(weights_md);
    if (src_md) dnnl_memory_desc_destroy(src_md);
    if (post_ops) dnnl_post_ops_destroy(post_ops);
    if (attr) dnnl_primitive_attr_destroy(attr);
    pthread_mutex_unlock(&ck_pytorch_brgemm_lock);
#undef CK_DNNL

    if (status != dnnl_success) {
        free(bias_bf16);
        free(output_bf16);
        free(input_bf16);
        ck_pytorch_brgemm_fault("execution failed", M, N, K);
    }
    for (size_t i = 0; i < output_count; ++i) C[i] = bf16_to_float(output_bf16[i]);
    free(bias_bf16);
    free(output_bf16);
    free(input_bf16);
#else
    (void)A; (void)B; (void)bias; (void)C; (void)M; (void)N; (void)K;
    fprintf(stderr, "HARD KERNEL CONTRACT FAULT: PyTorch oneDNN BF16 BRGEMM was "
                    "selected without USE_ONEDNN=1\n");
    abort();
#endif
}

void gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage(const float *A,
                                                       const void *B,
                                                       const float *bias,
                                                       float *C,
                                                       int M, int N, int K)
{
#ifdef USE_ONEDNN
    ck_pytorch_brgemm_require_version(
        3, 7, 1, "8d263e693366ef8db40acc569cc7d8edf644556d",
        "gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage", M, N, K);
#endif
    gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage_impl(
        A, B, bias, C, M, N, K);
}

void gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
#ifdef USE_ONEDNN
    ck_pytorch_brgemm_require_version(
        3, 12, 0, "80afa71049cd69a3df32adcccb623b12cd7baa22",
        "gemm_nt_bf16_pytorch_onednn_3_12_brgemm_bf16_storage", M, N, K);
#endif
    gemm_nt_bf16_pytorch_onednn_brgemm_bf16_storage_impl(
        A, B, bias, C, M, N, K);
}

void patch_projection_bf16_pytorch_onednn_conv3d_storage(
    const float *input, const void *weights, const float *bias, float *output,
    int batch, int out_channels, int in_channels, int temporal,
    int patch_h, int patch_w)
{
#ifdef USE_ONEDNN
    if (!input || !weights || !bias || !output || batch <= 0 ||
        out_channels <= 0 || in_channels <= 0 || temporal <= 0 ||
        patch_h <= 0 || patch_w <= 0) {
        ck_pytorch_brgemm_fault("invalid Conv3D patch contract", batch,
                                out_channels, in_channels * temporal * patch_h * patch_w);
    }
    pthread_once(&ck_pytorch_brgemm_once, ck_pytorch_brgemm_init);
    if (ck_pytorch_brgemm_init_status != 0) {
        ck_pytorch_brgemm_fault("could not initialize oneDNN Conv3D", batch,
                                out_channels, in_channels * temporal * patch_h * patch_w);
    }
    ck_pytorch_brgemm_require_version(
        3, 7, 1, "8d263e693366ef8db40acc569cc7d8edf644556d",
        "patch_projection_bf16_pytorch_onednn_conv3d_storage",
        batch, out_channels, in_channels * temporal * patch_h * patch_w);

    const size_t input_count = (size_t)batch * (size_t)in_channels *
        (size_t)temporal * (size_t)patch_h * (size_t)patch_w;
    const size_t output_count = (size_t)batch * (size_t)out_channels;
    uint16_t *input_bf16 = (uint16_t *)malloc(input_count * sizeof(*input_bf16));
    uint16_t *bias_bf16 = (uint16_t *)malloc((size_t)out_channels * sizeof(*bias_bf16));
    uint16_t *output_bf16 = (uint16_t *)malloc(output_count * sizeof(*output_bf16));
    if (!input_bf16 || !bias_bf16 || !output_bf16) {
        free(output_bf16);
        free(bias_bf16);
        free(input_bf16);
        ck_pytorch_brgemm_fault("Conv3D workspace allocation failed", batch,
                                out_channels, in_channels * temporal * patch_h * patch_w);
    }
    for (size_t i = 0; i < input_count; ++i) input_bf16[i] = float_to_bf16(input[i]);
    for (int i = 0; i < out_channels; ++i) bias_bf16[i] = float_to_bf16(bias[i]);

    dnnl_dims_t src_dims = {batch, in_channels, temporal, patch_h, patch_w};
    dnnl_dims_t weight_dims = {
        out_channels, in_channels, temporal, patch_h, patch_w};
    dnnl_dims_t bias_dims = {out_channels};
    dnnl_dims_t dst_dims = {batch, out_channels, 1, 1, 1};
    dnnl_dims_t strides = {temporal, patch_h, patch_w};
    dnnl_dims_t dilates = {0, 0, 0};
    dnnl_dims_t padding = {0, 0, 0};

    dnnl_memory_desc_t user_src_md = NULL, user_weight_md = NULL;
    dnnl_memory_desc_t bias_md = NULL, user_dst_md = NULL;
    dnnl_memory_desc_t any_src_md = NULL, any_weight_md = NULL, any_dst_md = NULL;
    dnnl_primitive_desc_t conv_pd = NULL;
    dnnl_primitive_t conv = NULL;
    dnnl_primitive_desc_t reorder_pd = NULL;
    dnnl_primitive_t reorder = NULL;
    dnnl_exec_arg_t reorder_args[2];
    dnnl_memory_t user_src = NULL, user_weight = NULL, bias_mem = NULL, user_dst = NULL;
    dnnl_memory_t conv_src = NULL, conv_weight = NULL, conv_dst = NULL;
    dnnl_status_t status = dnnl_success;

#define CK_DNNL_CONV(call) do { status = (call); if (status != dnnl_success) goto cleanup_conv; } while (0)
    pthread_mutex_lock(&ck_pytorch_brgemm_lock);
    CK_DNNL_CONV(dnnl_memory_desc_create_with_tag(
        &user_src_md, 5, src_dims, dnnl_bf16, dnnl_ncdhw));
    CK_DNNL_CONV(dnnl_memory_desc_create_with_tag(
        &user_weight_md, 5, weight_dims, dnnl_bf16, dnnl_oidhw));
    CK_DNNL_CONV(dnnl_memory_desc_create_with_tag(
        &bias_md, 1, bias_dims, dnnl_bf16, dnnl_x));
    CK_DNNL_CONV(dnnl_memory_desc_create_with_tag(
        &user_dst_md, 5, dst_dims, dnnl_bf16, dnnl_ncdhw));
    CK_DNNL_CONV(dnnl_memory_desc_create_with_tag(
        &any_src_md, 5, src_dims, dnnl_bf16, dnnl_format_tag_any));
    CK_DNNL_CONV(dnnl_memory_desc_create_with_tag(
        &any_weight_md, 5, weight_dims, dnnl_bf16, dnnl_format_tag_any));
    CK_DNNL_CONV(dnnl_memory_desc_create_with_tag(
        &any_dst_md, 5, dst_dims, dnnl_bf16, dnnl_format_tag_any));
    CK_DNNL_CONV(dnnl_convolution_forward_primitive_desc_create(
        &conv_pd, ck_pytorch_brgemm_engine, dnnl_forward_training,
        dnnl_convolution_direct, any_src_md, any_weight_md, bias_md, any_dst_md,
        strides, dilates, padding, padding, NULL));

    const_dnnl_memory_desc_t conv_src_md =
        dnnl_primitive_desc_query_md(conv_pd, dnnl_query_src_md, 0);
    const_dnnl_memory_desc_t conv_weight_md =
        dnnl_primitive_desc_query_md(conv_pd, dnnl_query_weights_md, 0);
    const_dnnl_memory_desc_t conv_dst_md =
        dnnl_primitive_desc_query_md(conv_pd, dnnl_query_dst_md, 0);
    CK_DNNL_CONV(dnnl_memory_create(
        &user_src, user_src_md, ck_pytorch_brgemm_engine, input_bf16));
    CK_DNNL_CONV(dnnl_memory_create(
        &user_weight, user_weight_md, ck_pytorch_brgemm_engine, (void *)weights));
    CK_DNNL_CONV(dnnl_memory_create(
        &bias_mem, bias_md, ck_pytorch_brgemm_engine, bias_bf16));
    CK_DNNL_CONV(dnnl_memory_create(
        &user_dst, user_dst_md, ck_pytorch_brgemm_engine, output_bf16));
    CK_DNNL_CONV(dnnl_memory_create(
        &conv_src, conv_src_md, ck_pytorch_brgemm_engine, DNNL_MEMORY_ALLOCATE));
    CK_DNNL_CONV(dnnl_memory_create(
        &conv_weight, conv_weight_md, ck_pytorch_brgemm_engine, DNNL_MEMORY_ALLOCATE));
    CK_DNNL_CONV(dnnl_memory_create(
        &conv_dst, conv_dst_md, ck_pytorch_brgemm_engine, DNNL_MEMORY_ALLOCATE));

    CK_DNNL_CONV(dnnl_reorder_primitive_desc_create(
        &reorder_pd, user_src_md, ck_pytorch_brgemm_engine,
        conv_src_md, ck_pytorch_brgemm_engine, NULL));
    CK_DNNL_CONV(dnnl_primitive_create(&reorder, reorder_pd));
    reorder_args[0] = (dnnl_exec_arg_t){DNNL_ARG_FROM, user_src};
    reorder_args[1] = (dnnl_exec_arg_t){DNNL_ARG_TO, conv_src};
    CK_DNNL_CONV(dnnl_primitive_execute(
        reorder, ck_pytorch_brgemm_stream, 2, reorder_args));
    dnnl_primitive_destroy(reorder); reorder = NULL;
    dnnl_primitive_desc_destroy(reorder_pd); reorder_pd = NULL;

    CK_DNNL_CONV(dnnl_reorder_primitive_desc_create(
        &reorder_pd, user_weight_md, ck_pytorch_brgemm_engine,
        conv_weight_md, ck_pytorch_brgemm_engine, NULL));
    CK_DNNL_CONV(dnnl_primitive_create(&reorder, reorder_pd));
    reorder_args[0] = (dnnl_exec_arg_t){DNNL_ARG_FROM, user_weight};
    reorder_args[1] = (dnnl_exec_arg_t){DNNL_ARG_TO, conv_weight};
    CK_DNNL_CONV(dnnl_primitive_execute(
        reorder, ck_pytorch_brgemm_stream, 2, reorder_args));
    dnnl_primitive_destroy(reorder); reorder = NULL;
    dnnl_primitive_desc_destroy(reorder_pd); reorder_pd = NULL;

    CK_DNNL_CONV(dnnl_primitive_create(&conv, conv_pd));
    dnnl_exec_arg_t conv_args[] = {
        {DNNL_ARG_SRC, conv_src},
        {DNNL_ARG_WEIGHTS, conv_weight},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, conv_dst},
    };
    CK_DNNL_CONV(dnnl_primitive_execute(
        conv, ck_pytorch_brgemm_stream,
        (int)(sizeof(conv_args) / sizeof(conv_args[0])), conv_args));

    CK_DNNL_CONV(dnnl_reorder_primitive_desc_create(
        &reorder_pd, conv_dst_md, ck_pytorch_brgemm_engine,
        user_dst_md, ck_pytorch_brgemm_engine, NULL));
    CK_DNNL_CONV(dnnl_primitive_create(&reorder, reorder_pd));
    reorder_args[0] = (dnnl_exec_arg_t){DNNL_ARG_FROM, conv_dst};
    reorder_args[1] = (dnnl_exec_arg_t){DNNL_ARG_TO, user_dst};
    CK_DNNL_CONV(dnnl_primitive_execute(
        reorder, ck_pytorch_brgemm_stream, 2, reorder_args));
    CK_DNNL_CONV(dnnl_stream_wait(ck_pytorch_brgemm_stream));

cleanup_conv:
    if (reorder) dnnl_primitive_destroy(reorder);
    if (reorder_pd) dnnl_primitive_desc_destroy(reorder_pd);
    if (conv) dnnl_primitive_destroy(conv);
    if (conv_dst) dnnl_memory_destroy(conv_dst);
    if (conv_weight) dnnl_memory_destroy(conv_weight);
    if (conv_src) dnnl_memory_destroy(conv_src);
    if (user_dst) dnnl_memory_destroy(user_dst);
    if (bias_mem) dnnl_memory_destroy(bias_mem);
    if (user_weight) dnnl_memory_destroy(user_weight);
    if (user_src) dnnl_memory_destroy(user_src);
    if (conv_pd) dnnl_primitive_desc_destroy(conv_pd);
    if (any_dst_md) dnnl_memory_desc_destroy(any_dst_md);
    if (any_weight_md) dnnl_memory_desc_destroy(any_weight_md);
    if (any_src_md) dnnl_memory_desc_destroy(any_src_md);
    if (user_dst_md) dnnl_memory_desc_destroy(user_dst_md);
    if (bias_md) dnnl_memory_desc_destroy(bias_md);
    if (user_weight_md) dnnl_memory_desc_destroy(user_weight_md);
    if (user_src_md) dnnl_memory_desc_destroy(user_src_md);
    pthread_mutex_unlock(&ck_pytorch_brgemm_lock);
#undef CK_DNNL_CONV

    if (status != dnnl_success) {
        free(output_bf16);
        free(bias_bf16);
        free(input_bf16);
        ck_pytorch_brgemm_fault("oneDNN Conv3D execution failed", batch,
                                out_channels, in_channels * temporal * patch_h * patch_w);
    }
    for (size_t i = 0; i < output_count; ++i) output[i] = bf16_to_float(output_bf16[i]);
    free(output_bf16);
    free(bias_bf16);
    free(input_bf16);
#else
    (void)input; (void)weights; (void)bias; (void)output; (void)batch;
    (void)out_channels; (void)in_channels; (void)temporal; (void)patch_h; (void)patch_w;
    fprintf(stderr, "HARD KERNEL CONTRACT FAULT: PyTorch oneDNN BF16 Conv3D "
                    "was selected without USE_ONEDNN=1\n");
    abort();
#endif
}

void patch_projection_image_bf16_pytorch_onednn_conv3d_storage(
    const float *image, const void *weights_t0, const void *weights_t1,
    const float *bias, float *output, int channels, int image_h, int image_w,
    int patch_size, int out_channels, int merge_size)
{
#ifdef USE_ONEDNN
    if (!image || !weights_t0 || !weights_t1 || !bias || !output ||
        channels <= 0 || image_h <= 0 || image_w <= 0 || patch_size <= 0 ||
        out_channels <= 0 || merge_size <= 0 || image_h % patch_size != 0 ||
        image_w % patch_size != 0) {
        ck_pytorch_brgemm_fault("invalid image patch projection contract",
                                image_h, image_w, patch_size);
    }
    const int grid_h = image_h / patch_size;
    const int grid_w = image_w / patch_size;
    if (grid_h % merge_size != 0 || grid_w % merge_size != 0) {
        ck_pytorch_brgemm_fault("patch grid is not merge-tile aligned",
                                grid_h, grid_w, merge_size);
    }
    const int batch = grid_h * grid_w;
    const int temporal = 2;
    const int half_k = channels * patch_size * patch_size;
    const int full_k = temporal * half_k;
    float *patches = (float *)malloc((size_t)batch * (size_t)full_k * sizeof(*patches));
    uint16_t *weights = (uint16_t *)malloc(
        (size_t)out_channels * (size_t)full_k * sizeof(*weights));
    if (!patches || !weights) {
        free(weights);
        free(patches);
        ck_pytorch_brgemm_fault("image patch projection workspace allocation failed",
                                batch, out_channels, full_k);
    }

    for (int tok = 0; tok < batch; ++tok) {
        const int tiles_per_row = grid_w / merge_size;
        const int tile_area = merge_size * merge_size;
        const int tile = tok / tile_area;
        const int within = tok % tile_area;
        const int patch_y = (tile / tiles_per_row) * merge_size + within / merge_size;
        const int patch_x = (tile % tiles_per_row) * merge_size + within % merge_size;
        float *dst = patches + (size_t)tok * (size_t)full_k;
        for (int c = 0; c < channels; ++c) {
            for (int t = 0; t < temporal; ++t) {
                for (int py = 0; py < patch_size; ++py) {
                    const float *src = image +
                        ((size_t)c * (size_t)image_h +
                         (size_t)(patch_y * patch_size + py)) * (size_t)image_w +
                        (size_t)(patch_x * patch_size);
                    memcpy(dst, src, (size_t)patch_size * sizeof(*dst));
                    dst += patch_size;
                }
            }
        }
    }

    const uint16_t *w0 = (const uint16_t *)weights_t0;
    const uint16_t *w1 = (const uint16_t *)weights_t1;
    for (int n = 0; n < out_channels; ++n) {
        uint16_t *dst = weights + (size_t)n * (size_t)full_k;
        for (int c = 0; c < channels; ++c) {
            const size_t channel_offset =
                (size_t)n * (size_t)half_k +
                (size_t)c * (size_t)patch_size * (size_t)patch_size;
            const size_t plane_bytes =
                (size_t)patch_size * (size_t)patch_size * sizeof(*dst);
            memcpy(dst, w0 + channel_offset, plane_bytes);
            dst += patch_size * patch_size;
            memcpy(dst, w1 + channel_offset, plane_bytes);
            dst += patch_size * patch_size;
        }
    }

    patch_projection_bf16_pytorch_onednn_conv3d_storage(
        patches, weights, bias, output, batch, out_channels, channels,
        temporal, patch_size, patch_size);
    free(weights);
    free(patches);
#else
    (void)image; (void)weights_t0; (void)weights_t1; (void)bias; (void)output;
    (void)channels; (void)image_h; (void)image_w; (void)patch_size;
    (void)out_channels; (void)merge_size;
    fprintf(stderr, "HARD KERNEL CONTRACT FAULT: PyTorch oneDNN BF16 image "
                    "patch projection was selected without USE_ONEDNN=1\n");
    abort();
#endif
}

typedef struct {
    const float *image;
    const uint16_t *weights_t0;
    const uint16_t *weights_t1;
    const float *bias;
    float *output;
    int channels;
    int image_h;
    int image_w;
    int patch_size;
    int out_channels;
    int merge_size;
    int grid_w;
    int batch;
} ck_patch_projection_bf16_native_args_t;

static void ck_patch_projection_bf16_native_work(
    int ith, int nth, void *opaque)
{
    ck_patch_projection_bf16_native_args_t *args =
        (ck_patch_projection_bf16_native_args_t *)opaque;
    const int begin = args->batch * ith / nth;
    const int end = args->batch * (ith + 1) / nth;
    const int patch_area = args->patch_size * args->patch_size;
    const int half_k = args->channels * patch_area;
    const int tiles_per_row = args->grid_w / args->merge_size;
    const int tile_area = args->merge_size * args->merge_size;

    for (int tok = begin; tok < end; ++tok) {
        const int tile = tok / tile_area;
        const int within = tok % tile_area;
        const int patch_y =
            (tile / tiles_per_row) * args->merge_size + within / args->merge_size;
        const int patch_x =
            (tile % tiles_per_row) * args->merge_size + within % args->merge_size;
        for (int n = 0; n < args->out_channels; ++n) {
            float sum = args->bias
                ? bf16_to_float(float_to_bf16(args->bias[n]))
                : 0.0f;
#if defined(__AVX512BF16__) && defined(__AVX512VL__)
            if (args->patch_size == 16) {
                __m256 acc = _mm256_setzero_ps();
                for (int c = 0; c < args->channels; ++c) {
                    for (int t = 0; t < 2; ++t) {
                        const uint16_t *weights = t == 0
                            ? args->weights_t0 : args->weights_t1;
                        const uint16_t *weight_plane = weights +
                            (size_t)n * (size_t)half_k +
                            (size_t)c * (size_t)patch_area;
                        for (int py = 0; py < 16; ++py) {
                            const float *src = args->image +
                                ((size_t)c * (size_t)args->image_h +
                                 (size_t)(patch_y * 16 + py)) *
                                    (size_t)args->image_w +
                                (size_t)(patch_x * 16);
                            const __m256bh image_bf16 = _mm256_cvtne2ps_pbh(
                                _mm256_loadu_ps(src + 8), _mm256_loadu_ps(src));
                            const __m256bh weight_bf16 = (__m256bh)_mm256_loadu_si256(
                                (const __m256i *)(weight_plane + (size_t)py * 16u));
                            acc = _mm256_dpbf16_ps(acc, image_bf16, weight_bf16);
                        }
                    }
                }
                float lanes[8];
                _mm256_storeu_ps(lanes, acc);
                const float sum01 = lanes[0] + lanes[1];
                const float sum23 = lanes[2] + lanes[3];
                const float sum45 = lanes[4] + lanes[5];
                const float sum67 = lanes[6] + lanes[7];
                sum += (sum01 + sum23) + (sum45 + sum67);
            } else
#endif
            {
                for (int c = 0; c < args->channels; ++c) {
                    for (int t = 0; t < 2; ++t) {
                        const uint16_t *weights = t == 0
                            ? args->weights_t0 : args->weights_t1;
                        const uint16_t *weight_plane = weights +
                            (size_t)n * (size_t)half_k +
                            (size_t)c * (size_t)patch_area;
                        for (int py = 0; py < args->patch_size; ++py) {
                            const float *src = args->image +
                                ((size_t)c * (size_t)args->image_h +
                                 (size_t)(patch_y * args->patch_size + py)) *
                                    (size_t)args->image_w +
                                (size_t)(patch_x * args->patch_size);
                            for (int px = 0; px < args->patch_size; ++px) {
                                const float value = bf16_to_float(float_to_bf16(src[px]));
                                const float weight = bf16_to_float(
                                    weight_plane[(size_t)py *
                                        (size_t)args->patch_size + (size_t)px]);
                                sum += value * weight;
                            }
                        }
                    }
                }
            }
            args->output[(size_t)tok * (size_t)args->out_channels + (size_t)n] =
                bf16_to_float(float_to_bf16(sum));
        }
    }
}

void patch_projection_image_bf16_native_storage(
    const float *image, const void *weights_t0, const void *weights_t1,
    const float *bias, float *output, int channels, int image_h, int image_w,
    int patch_size, int out_channels, int merge_size)
{
    if (!image || !weights_t0 || !weights_t1 || !output || channels <= 0 ||
        image_h <= 0 || image_w <= 0 || patch_size <= 0 || out_channels <= 0 ||
        merge_size <= 0 || image_h % patch_size != 0 ||
        image_w % patch_size != 0) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: invalid native BF16 image patch projection\n");
        abort();
    }
    const int grid_h = image_h / patch_size;
    const int grid_w = image_w / patch_size;
    if (grid_h % merge_size != 0 || grid_w % merge_size != 0) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: native BF16 patch grid is not merge aligned\n");
        abort();
    }
    ck_patch_projection_bf16_native_args_t args = {
        .image = image,
        .weights_t0 = (const uint16_t *)weights_t0,
        .weights_t1 = (const uint16_t *)weights_t1,
        .bias = bias,
        .output = output,
        .channels = channels,
        .image_h = image_h,
        .image_w = image_w,
        .patch_size = patch_size,
        .out_channels = out_channels,
        .merge_size = merge_size,
        .grid_w = grid_w,
        .batch = grid_h * grid_w,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > args.batch) active = args.batch;
    if (active > 24) active = 24;
    if (pool && active > 1) {
        ck_threadpool_dispatch_n(
            pool, active, ck_patch_projection_bf16_native_work, &args);
    } else {
        ck_patch_projection_bf16_native_work(0, 1, &args);
    }
}

void gemm_nt_bf16_bf16_storage_row_range(const float *A,
                                          const void *B,
                                          const float *bias,
                                          float *C,
                                          int M, int N, int K,
                                          int row_begin, int row_end)
{
    gemm_nt_bf16_row_range(A, B, bias, C, M, N, K, row_begin, row_end);
    for (int row = row_begin; row < row_end; ++row) {
        float *dst = C + (size_t)row * (size_t)N;
        for (int col = 0; col < N; ++col) {
            dst[col] = bf16_to_float(float_to_bf16(dst[col]));
        }
    }
}

static void ck_gemm_nt_bf16_storage_exact_rows(int begin, int end, void *opaque)
{
    const ck_gemm_nt_bf16_exact_args_t *args =
        (const ck_gemm_nt_bf16_exact_args_t *)opaque;
    gemm_nt_bf16_bf16_storage_row_range(
        args->A, args->B, args->bias, args->C,
        args->M, args->N, args->K, begin, end);
}

void gemm_nt_bf16_bf16_storage_parallel_dispatch(const float *A,
                                                  const void *B,
                                                  const float *bias,
                                                  float *C,
                                                  int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M < 2 ||
        (size_t)M * (size_t)N <= 4096) {
        gemm_nt_bf16_bf16_storage_row_range(A, B, bias, C, M, N, K, 0, M);
        return;
    }

    ck_gemm_nt_bf16_exact_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C, .M = M, .N = N, .K = K,
    };
    int active = ck_threadpool_n_threads(pool);
    if (active > M) active = M;
    int grain = M / (active * 4);
    if (grain < 1) grain = 1;
    ck_threadpool_parallel_for_n(
        pool, active, 0, M, grain, ck_gemm_nt_bf16_storage_exact_rows, &args);
}

void gemm_nt_bf16_bf16_storage(const float *A,
                                const void *B,
                                const float *bias,
                                float *C,
                                int M, int N, int K)
{
    gemm_nt_bf16_bf16_storage_parallel_dispatch(A, B, bias, C, M, N, K);
}

void gemm_backward_bf16_mixed(const uint16_t *d_output,
                              const uint16_t *input,
                              const uint16_t *weight,
                              float *d_input,
                              float *d_weight,
                              float *d_bias,
                              int tokens,
                              int in_dim,
                              int out_dim)
{
    if (!d_output || !input || !weight || tokens <= 0 || in_dim <= 0 || out_dim <= 0) {
        return;
    }

    if (d_input) {
        for (int t = 0; t < tokens; ++t) {
            for (int i = 0; i < in_dim; ++i) {
                float sum = 0.0f;
                for (int o = 0; o < out_dim; ++o) {
                    const float dy = bf16_to_float(d_output[(size_t)t * (size_t)out_dim + (size_t)o]);
                    const float w = bf16_to_float(weight[(size_t)o * (size_t)in_dim + (size_t)i]);
                    sum += dy * w;
                }
                d_input[(size_t)t * (size_t)in_dim + (size_t)i] = sum;
            }
        }
    }

    if (d_weight) {
        for (int o = 0; o < out_dim; ++o) {
            for (int i = 0; i < in_dim; ++i) {
                float sum = 0.0f;
                for (int t = 0; t < tokens; ++t) {
                    const float dy = bf16_to_float(d_output[(size_t)t * (size_t)out_dim + (size_t)o]);
                    const float x = bf16_to_float(input[(size_t)t * (size_t)in_dim + (size_t)i]);
                    sum += dy * x;
                }
                d_weight[(size_t)o * (size_t)in_dim + (size_t)i] = sum;
            }
        }
    }

    if (d_bias) {
        for (int o = 0; o < out_dim; ++o) {
            float sum = 0.0f;
            for (int t = 0; t < tokens; ++t) {
                sum += bf16_to_float(d_output[(size_t)t * (size_t)out_dim + (size_t)o]);
            }
            d_bias[o] = sum;
        }
    }
}
