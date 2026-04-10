/**
 * @file prefill_fused_gemm.c
 * @brief Fused kernels for prefill phase with proper 2D tiling
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
 * KEY INSIGHT:
 * ------------
 * Naive M-dimension tiling (token tiles) causes weight reloading:
 *   - 32 token tiles × 4MB weights = 128MB DRAM reads!
 *
 * Correct approach: Tile along N (output/weight) dimension OUTER,
 * M (token) dimension INNER. This way:
 *   - Load weight tile once
 *   - Process ALL tokens against that weight tile
 *   - Weight tile stays in cache while streaming through tokens
 *
 * TILING STRATEGY:
 * ----------------
 * For C[M,N] = RMSNorm(A[M,K]) × B[N,K]^T:
 *
 *   for n_tile in [0, N, TILE_N]:           # Outer: weight tiles
 *     load B[n_tile:n_tile+TILE_N, :] into L3
 *     for m_tile in [0, M, TILE_M]:         # Inner: token tiles
 *       x_norm = rmsnorm(A[m_tile])         # x_norm in L2
 *       C[m_tile, n_tile] = x_norm × B_tile # Consumes B from L3
 *
 * Cache behavior:
 *   - Weight tile (TILE_N × K × 4 bytes) fits in L3
 *   - x_norm tile (TILE_M × K × 4 bytes) fits in L2
 *   - Weights loaded once per tile, reused across all token tiles
 */

#include "ckernel_engine.h"
#include "ckernel_quant.h"
#include <math.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Tile sizes chosen for your cache hierarchy:
 * - L2 = 256KB: x_norm tile = TILE_M × hidden × 4
 * - L3 = 6MB: weight tile = TILE_N × hidden × 4
 *
 * For hidden=896:
 *   TILE_M = 64 → 64×896×4 = 224KB (fits L2)
 *   TILE_N = 256 → 256×896×4 = 896KB (fits L3 with room for x_norm)
 */
#define PREFILL_TILE_M 64
#define PREFILL_TILE_N 256

static size_t align_up_size(size_t value, size_t align) {
    return (value + align - 1) & ~(align - 1);
}

/* Helper: horizontal sum for AVX */
#if defined(__AVX__) && !defined(__AVX512F__)
static inline float hsum256_prefill(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif

/**
 * @brief Compute RMSNorm for a tile of tokens
 */
static void rmsnorm_tile(const float *input,
                         const float *gamma,
                         float *output,
                         int tile_m,
                         int embed_dim,
                         int aligned_embed_dim,
                         float eps)
{
    for (int t = 0; t < tile_m; ++t) {
        const float *x = input + (size_t)t * (size_t)aligned_embed_dim;
        float *y = output + (size_t)t * (size_t)aligned_embed_dim;

#if defined(__AVX512F__)
        __m512 sum_sq_vec = _mm512_setzero_ps();
        int d = 0;
        for (; d + 16 <= embed_dim; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            sum_sq_vec = _mm512_fmadd_ps(xv, xv, sum_sq_vec);
        }
        float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);
        for (; d < embed_dim; ++d) {
            sum_sq += x[d] * x[d];
        }

        float rstd = 1.0f / sqrtf(sum_sq / (float)embed_dim + eps);
        __m512 rstd_vec = _mm512_set1_ps(rstd);

        d = 0;
        for (; d + 16 <= embed_dim; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            __m512 gv = gamma ? _mm512_loadu_ps(&gamma[d]) : _mm512_set1_ps(1.0f);
            __m512 yv = _mm512_mul_ps(_mm512_mul_ps(xv, rstd_vec), gv);
            _mm512_storeu_ps(&y[d], yv);
        }
        for (; d < embed_dim; ++d) {
            float g = gamma ? gamma[d] : 1.0f;
            y[d] = x[d] * rstd * g;
        }

#elif defined(__AVX__)
        __m256 sum_sq_vec = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= embed_dim; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, _mm256_mul_ps(xv, xv));
        }
        float sum_sq = hsum256_prefill(sum_sq_vec);
        for (; d < embed_dim; ++d) {
            sum_sq += x[d] * x[d];
        }

        float rstd = 1.0f / sqrtf(sum_sq / (float)embed_dim + eps);
        __m256 rstd_vec = _mm256_set1_ps(rstd);

        d = 0;
        for (; d + 8 <= embed_dim; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 gv = gamma ? _mm256_loadu_ps(&gamma[d]) : _mm256_set1_ps(1.0f);
            __m256 yv = _mm256_mul_ps(_mm256_mul_ps(xv, rstd_vec), gv);
            _mm256_storeu_ps(&y[d], yv);
        }
        for (; d < embed_dim; ++d) {
            float g = gamma ? gamma[d] : 1.0f;
            y[d] = x[d] * rstd * g;
        }
#else
        float sum_sq = 0.0f;
        for (int d = 0; d < embed_dim; ++d) {
            sum_sq += x[d] * x[d];
        }
        float rstd = 1.0f / sqrtf(sum_sq / (float)embed_dim + eps);
        for (int d = 0; d < embed_dim; ++d) {
            float g = gamma ? gamma[d] : 1.0f;
            y[d] = x[d] * rstd * g;
        }
#endif

        for (int d = embed_dim; d < aligned_embed_dim; ++d) {
            y[d] = 0.0f;
        }
    }
}

static int qkv_q8_0_dtype_supported(CKDataType dt) {
    return (dt == CK_DT_Q5_0 || dt == CK_DT_Q8_0);
}

static int qkv_q8_k_dtype_supported(CKDataType dt) {
    return (dt == CK_DT_Q4_K || dt == CK_DT_Q6_K);
}

static void gemm_nt_q8_0_dispatch(const void *A_q8,
                                 const void *B,
                                 const float *bias,
                                 float *C,
                                 int M,
                                 int N,
                                 int K,
                                 CKDataType dt)
{
    switch (dt) {
    case CK_DT_Q5_0:
        gemm_nt_q5_0_q8_0(A_q8, B, bias, C, M, N, K);
        break;
    case CK_DT_Q8_0:
        gemm_nt_q8_0_q8_0(A_q8, B, bias, C, M, N, K);
        break;
    default:
        break;
    }
}

static void gemm_nt_q8_k_qkv_dispatch(const void *A_q8k,
                                      const void *B,
                                      const float *bias,
                                      float *C,
                                      int M,
                                      int N,
                                      int K,
                                      CKDataType dt)
{
    switch (dt) {
    case CK_DT_Q4_K:
        gemm_nt_q4_k_q8_k(A_q8k, B, bias, C, M, N, K);
        break;
    case CK_DT_Q6_K:
        gemm_nt_q6_k_q8_k(A_q8k, B, bias, C, M, N, K);
        break;
    default:
        break;
    }
}

/**
 * @brief GEMM tile with N-dimension tiling (weight reuse)
 *
 * Computes: C[tile_m × tile_n] = A[tile_m × K] × B[tile_n × K]^T
 * where B_tile is a slice of rows from the weight matrix.
 *
 * Uses MKL if available for optimal performance.
 *
 * @param A       Input tile [tile_m × K]
 * @param B_tile  Weight tile [tile_n × K] (transposed layout)
 * @param C       Output tile [tile_m × tile_n] (column slice of full output)
 * @param C_stride Stride between rows of C (= full N dimension)
 */
#ifdef USE_MKL
#include <mkl.h>
#endif

static void gemm_tile_nt_strided(const float *A,
                                  const float *B_tile,
                                  float *C,
                                  int tile_m,
                                  int tile_n,
                                  int K,
                                  int C_stride)
{
#ifdef USE_MKL
    /* Use MKL SGEMM: C = A × B^T
     * But MKL expects contiguous output, so we need to handle strided output.
     * For now, if C_stride == tile_n (contiguous), use MKL directly.
     * Otherwise, fall back to naive. */
    if (C_stride == tile_n) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    tile_m, tile_n, K,
                    1.0f, A, K, B_tile, K,
                    0.0f, C, tile_n);
        return;
    }
    /* Strided output - use MKL per row */
    for (int i = 0; i < tile_m; ++i) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    tile_n, K,
                    1.0f, B_tile, K, A + (size_t)i * K, 1,
                    0.0f, C + (size_t)i * C_stride, 1);
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < tile_m; ++i) {
        const float *a_row = A + (size_t)i * K;
        float *c_row = C + (size_t)i * C_stride;

        for (int j = 0; j < tile_n; ++j) {
            const float *b_row = B_tile + (size_t)j * K;
            float sum = 0.0f;

#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                __m512 av = _mm512_loadu_ps(a_row + k);
                __m512 bv = _mm512_loadu_ps(b_row + k);
                acc = _mm512_fmadd_ps(av, bv, acc);
            }
            sum = _mm512_reduce_add_ps(acc);
            for (; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
#elif defined(__AVX__)
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                __m256 av = _mm256_loadu_ps(a_row + k);
                __m256 bv = _mm256_loadu_ps(b_row + k);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
            }
            sum = hsum256_prefill(acc);
            for (; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
#else
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
#endif
            c_row[j] = sum;
        }
    }
#endif
}

static void add_bias_tile(float *out,
                          const float *bias,
                          int tile_m,
                          int out_dim)
{
    if (!out || !bias) {
        return;
    }
    for (int i = 0; i < tile_m; ++i) {
        float *row = out + (size_t)i * (size_t)out_dim;
        for (int j = 0; j < out_dim; ++j) {
            row[j] += bias[j];
        }
    }
}

/**
 * @brief Fused RMSNorm + single GEMM with 2D tiling (weight reuse)
 *
 * Tiles along N (weights) OUTER, M (tokens) INNER.
 * Weight tiles are reused across all token tiles.
 */
static void fused_rmsnorm_gemm_2d_tiled(
    const float *x,           /* [seq_len × hidden] input */
    const float *gamma,       /* [hidden] RMSNorm weights */
    const float *W,           /* [out_dim × hidden] weight matrix (transposed) */
    float *output,            /* [seq_len × out_dim] output */
    int seq_len,
    int hidden,
    int out_dim,
    float eps,
    float *x_norm_scratch)    /* [TILE_M × hidden] scratch for normalized tile */
{
    /* Outer loop: tile along output dimension (N) - weight tiles */
    for (int n_start = 0; n_start < out_dim; n_start += PREFILL_TILE_N) {
        int tile_n = (n_start + PREFILL_TILE_N <= out_dim)
                         ? PREFILL_TILE_N
                         : (out_dim - n_start);

        /* Weight tile pointer - this tile stays in L3 cache */
        const float *W_tile = W + (size_t)n_start * hidden;

        /* Inner loop: tile along token dimension (M) */
        for (int m_start = 0; m_start < seq_len; m_start += PREFILL_TILE_M) {
            int tile_m = (m_start + PREFILL_TILE_M <= seq_len)
                             ? PREFILL_TILE_M
                             : (seq_len - m_start);

            const float *x_tile = x + (size_t)m_start * hidden;
            float *out_tile = output + (size_t)m_start * out_dim + n_start;

            /* Compute RMSNorm for this token tile (only on first weight tile) */
            if (n_start == 0) {
                rmsnorm_tile(x_tile, gamma, x_norm_scratch, tile_m, hidden, hidden, eps);
            } else {
                /* Recompute x_norm for this tile (we can't cache all of it) */
                /* TODO: For very large N, consider caching x_norm chunks */
                rmsnorm_tile(x_tile, gamma, x_norm_scratch, tile_m, hidden, hidden, eps);
            }

            /* GEMM: x_norm_tile × W_tile^T → output tile */
            gemm_tile_nt_strided(x_norm_scratch, W_tile, out_tile,
                                  tile_m, tile_n, hidden, out_dim);
        }
    }
}

/**
 * @brief Fused RMSNorm + QKV projection for prefill (v3 optimized)
 *
 * KEY INSIGHT: For Qwen2-0.5B, all QKV weights fit in L3:
 *   Wq (896×896) + Wk (128×896) + Wv (128×896) = 4.1MB < 6MB L3
 *
 * So we use M-tiling (tokens) only:
 * 1. For each token tile:
 *    a. Compute RMSNorm ONCE into scratch (x_norm stays in L2)
 *    b. Do all three GEMMs (Q, K, V) against cached x_norm
 *    c. Weights stay hot in L3 across all token tiles
 *
 * This avoids both:
 *   - Large x_norm intermediate buffer (only TILE_M × hidden in L2)
 *   - RMSNorm recomputation (done once per token tile, used 3×)
 */
void fused_rmsnorm_qkv_prefill(
    const float *x,
    const float *gamma,
    const float *Wq,
    const float *Wk,
    const float *Wv,
    float *Q,
    float *K,
    float *V,
    int seq_len,
    int hidden,
    int q_dim,
    int kv_dim,
    float eps,
    float *scratch)
{
    /* scratch is x_norm tile: [TILE_M × hidden] fits in L2 */

    /* Process token tiles - weights stay in L3 across all tiles */
    for (int m_start = 0; m_start < seq_len; m_start += PREFILL_TILE_M) {
        int tile_m = (m_start + PREFILL_TILE_M <= seq_len)
                         ? PREFILL_TILE_M : (seq_len - m_start);

        const float *x_tile = x + (size_t)m_start * hidden;

        /* Step 1: RMSNorm for this token tile (computed ONCE, used 3×) */
        rmsnorm_tile(x_tile, gamma, scratch, tile_m, hidden, hidden, eps);

        /* Step 2: Q projection - x_norm is hot in L2, Wq hot in L3 */
        float *Q_tile = Q + (size_t)m_start * q_dim;
        gemm_tile_nt_strided(scratch, Wq, Q_tile, tile_m, q_dim, hidden, q_dim);

        /* Step 3: K projection - x_norm still hot, Wk displaces some Wq */
        float *K_tile = K + (size_t)m_start * kv_dim;
        gemm_tile_nt_strided(scratch, Wk, K_tile, tile_m, kv_dim, hidden, kv_dim);

        /* Step 4: V projection - x_norm still hot, Wv displaces Wk */
        float *V_tile = V + (size_t)m_start * kv_dim;
        gemm_tile_nt_strided(scratch, Wv, V_tile, tile_m, kv_dim, hidden, kv_dim);
    }
}

/**
 * @brief Fused RMSNorm + QKV projection for prefill (head-major outputs)
 *
 * Q is written as [num_heads, seq_len, aligned_head_dim].
 * K/V are written with kv_stride_tokens for KV-cache compatibility.
 */
void fused_rmsnorm_qkv_prefill_head_major(
    const float *x,
    const float *gamma,
    const float *Wq, const float *Bq,
    const float *Wk, const float *Bk,
    const float *Wv, const float *Bv,
    float *Q,
    float *K,
    float *V,
    int seq_len,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    int kv_stride_tokens,
    float eps,
    float *scratch)
{
    if (!x || !gamma || !Wq || !Wk || !Wv || !Q || !K || !V || !scratch) {
        return;
    }
    if (seq_len <= 0 || embed_dim <= 0 || aligned_embed_dim <= 0 ||
        head_dim <= 0 || aligned_head_dim <= 0 ||
        num_heads <= 0 || num_kv_heads <= 0) {
        return;
    }
    if (kv_stride_tokens < seq_len) {
        return;
    }

    const size_t q_head_stride = (size_t)seq_len * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;
    const size_t head_w_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;

    for (int m_start = 0; m_start < seq_len; m_start += PREFILL_TILE_M) {
        int tile_m = (m_start + PREFILL_TILE_M <= seq_len)
                         ? PREFILL_TILE_M : (seq_len - m_start);

        const float *x_tile = x + (size_t)m_start * (size_t)aligned_embed_dim;
        rmsnorm_tile(x_tile, gamma, scratch, tile_m, embed_dim, aligned_embed_dim, eps);

        for (int h = 0; h < num_heads; ++h) {
            const float *wq_h = Wq + (size_t)h * head_w_stride;
            const float *bq_h = Bq ? (Bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
            float *q_h = Q + (size_t)h * q_head_stride + (size_t)m_start * (size_t)aligned_head_dim;

            gemm_tile_nt_strided(scratch, wq_h, q_h,
                                 tile_m, aligned_head_dim, aligned_embed_dim, aligned_head_dim);
            add_bias_tile(q_h, bq_h, tile_m, aligned_head_dim);
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            const float *wk_h = Wk + (size_t)h * head_w_stride;
            const float *wv_h = Wv + (size_t)h * head_w_stride;
            const float *bk_h = Bk ? (Bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
            const float *bv_h = Bv ? (Bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
            float *k_h = K + (size_t)h * kv_head_stride + (size_t)m_start * (size_t)aligned_head_dim;
            float *v_h = V + (size_t)h * kv_head_stride + (size_t)m_start * (size_t)aligned_head_dim;

            gemm_tile_nt_strided(scratch, wk_h, k_h,
                                 tile_m, aligned_head_dim, aligned_embed_dim, aligned_head_dim);
            add_bias_tile(k_h, bk_h, tile_m, aligned_head_dim);

            gemm_tile_nt_strided(scratch, wv_h, v_h,
                                 tile_m, aligned_head_dim, aligned_embed_dim, aligned_head_dim);
            add_bias_tile(v_h, bv_h, tile_m, aligned_head_dim);
        }
    }
}

/**
 * @brief Fused RMSNorm + QKV projection for prefill (head-major, Q8 activations)
 *
 * Supports Q5_0 or Q8_0 weights with Q8_0 activations.
 * Writes K/V directly into KV cache layout (kv_stride_tokens).
 */
void fused_rmsnorm_qkv_prefill_head_major_quant(
    const float *x,
    const float *gamma,
    const void *Wq, const float *Bq, CKDataType wq_dt,
    const void *Wk, const float *Bk, CKDataType wk_dt,
    const void *Wv, const float *Bv, CKDataType wv_dt,
    float *Q,
    float *K,
    float *V,
    int seq_len,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    int kv_stride_tokens,
    float eps,
    void *scratch)
{
    if (!x || !gamma || !Wq || !Wk || !Wv || !Q || !K || !V || !scratch) {
        return;
    }
    if (seq_len <= 0 || embed_dim <= 0 || aligned_embed_dim <= 0 ||
        head_dim <= 0 || aligned_head_dim <= 0 ||
        num_heads <= 0 || num_kv_heads <= 0) {
        return;
    }
    if (aligned_embed_dim % 32 != 0) {
        return;
    }
    if (kv_stride_tokens < seq_len) {
        return;
    }
    /* Determine quantization path: Q8_0 activations for Q5_0/Q8_0 weights,
     * Q8_K activations for Q4_K/Q6_K weights. All QKV weights must use
     * the same quantization family. */
    int use_q8_k_path = qkv_q8_k_dtype_supported(wq_dt);
    int use_q8_0_path = qkv_q8_0_dtype_supported(wq_dt);

    if (!use_q8_k_path && !use_q8_0_path) {
        /* Unsupported dtype for wq */
        return;
    }

    /* Verify all dtypes are from the same family */
    if (use_q8_k_path) {
        if (!qkv_q8_k_dtype_supported(wk_dt) || !qkv_q8_k_dtype_supported(wv_dt)) {
            return;  /* Mixed Q8_K and Q8_0 paths not supported */
        }
    } else {
        if (!qkv_q8_0_dtype_supported(wk_dt) || !qkv_q8_0_dtype_supported(wv_dt)) {
            return;
        }
    }

    const size_t float_bytes = (size_t)PREFILL_TILE_M * (size_t)aligned_embed_dim * sizeof(float);
    /* Q8_K has larger blocks (256) than Q8_0 (32), so use appropriate size */
    const CKDataType act_quant_type = use_q8_k_path ? CK_DT_Q8_K : CK_DT_Q8_0;
    const size_t q8_row_bytes = ck_dtype_row_bytes(act_quant_type, (size_t)aligned_embed_dim);
    const size_t q8_bytes = (size_t)PREFILL_TILE_M * q8_row_bytes;
    const size_t q8_offset = align_up_size(float_bytes, 64);

    float *normed = (float *)scratch;
    uint8_t *q8_tile = (uint8_t *)scratch + q8_offset;
    (void)q8_bytes;

    const size_t q_head_stride = (size_t)seq_len * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;
    const size_t head_w_elems = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;
    const size_t wq_head_bytes = ck_dtype_row_bytes(wq_dt, head_w_elems);
    const size_t wk_head_bytes = ck_dtype_row_bytes(wk_dt, head_w_elems);
    const size_t wv_head_bytes = ck_dtype_row_bytes(wv_dt, head_w_elems);

    for (int m_start = 0; m_start < seq_len; m_start += PREFILL_TILE_M) {
        int tile_m = (m_start + PREFILL_TILE_M <= seq_len)
                         ? PREFILL_TILE_M : (seq_len - m_start);

        const float *x_tile = x + (size_t)m_start * (size_t)aligned_embed_dim;
        rmsnorm_tile(x_tile, gamma, normed, tile_m, embed_dim, aligned_embed_dim, eps);

        /* Quantize activations to appropriate format */
        for (int t = 0; t < tile_m; ++t) {
            const float *row = normed + (size_t)t * (size_t)aligned_embed_dim;
            if (use_q8_k_path) {
                quantize_row_q8_k(row,
                                  q8_tile + (size_t)t * q8_row_bytes,
                                  aligned_embed_dim);
            } else {
                quantize_row_q8_0(row,
                                  q8_tile + (size_t)t * q8_row_bytes,
                                  aligned_embed_dim);
            }
        }

        for (int h = 0; h < num_heads; ++h) {
            const uint8_t *wq_h = (const uint8_t *)Wq + (size_t)h * wq_head_bytes;
            const float *bq_h = Bq ? (Bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
            float *q_h = Q + (size_t)h * q_head_stride + (size_t)m_start * (size_t)aligned_head_dim;

            if (use_q8_k_path) {
                gemm_nt_q8_k_qkv_dispatch(q8_tile, wq_h, bq_h, q_h,
                                          tile_m, aligned_head_dim, aligned_embed_dim, wq_dt);
            } else {
                gemm_nt_q8_0_dispatch(q8_tile, wq_h, bq_h, q_h,
                                      tile_m, aligned_head_dim, aligned_embed_dim, wq_dt);
            }
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            const uint8_t *wk_h = (const uint8_t *)Wk + (size_t)h * wk_head_bytes;
            const uint8_t *wv_h = (const uint8_t *)Wv + (size_t)h * wv_head_bytes;
            const float *bk_h = Bk ? (Bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
            const float *bv_h = Bv ? (Bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
            float *k_h = K + (size_t)h * kv_head_stride + (size_t)m_start * (size_t)aligned_head_dim;
            float *v_h = V + (size_t)h * kv_head_stride + (size_t)m_start * (size_t)aligned_head_dim;

            if (use_q8_k_path) {
                gemm_nt_q8_k_qkv_dispatch(q8_tile, wk_h, bk_h, k_h,
                                          tile_m, aligned_head_dim, aligned_embed_dim, wk_dt);
                gemm_nt_q8_k_qkv_dispatch(q8_tile, wv_h, bv_h, v_h,
                                          tile_m, aligned_head_dim, aligned_embed_dim, wv_dt);
            } else {
                gemm_nt_q8_0_dispatch(q8_tile, wk_h, bk_h, k_h,
                                      tile_m, aligned_head_dim, aligned_embed_dim, wk_dt);
                gemm_nt_q8_0_dispatch(q8_tile, wv_h, bv_h, v_h,
                                      tile_m, aligned_head_dim, aligned_embed_dim, wv_dt);
            }
        }
    }
}

size_t fused_rmsnorm_qkv_prefill_head_major_quant_scratch_size(int aligned_embed_dim) {
    if (aligned_embed_dim <= 0) {
        return 0;
    }
    const size_t float_bytes = (size_t)PREFILL_TILE_M * (size_t)aligned_embed_dim * sizeof(float);
    /* Use max of Q8_0 and Q8_K sizes to support both paths */
    const size_t q8_0_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)aligned_embed_dim);
    const size_t q8_k_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)aligned_embed_dim);
    const size_t q8_row_bytes = (q8_k_row_bytes > q8_0_row_bytes) ? q8_k_row_bytes : q8_0_row_bytes;
    const size_t q8_bytes = (size_t)PREFILL_TILE_M * q8_row_bytes;
    return align_up_size(float_bytes, 64) + q8_bytes;
}

/**
 * @brief Unfused version for comparison
 */
void unfused_rmsnorm_qkv_prefill(
    const float *x,
    const float *gamma,
    const float *Wq,
    const float *Wk,
    const float *Wv,
    float *x_norm,
    float *Q,
    float *K,
    float *V,
    int seq_len,
    int hidden,
    int q_dim,
    int kv_dim,
    float eps)
{
    /* Step 1: Full RMSNorm → writes x_norm to memory */
    rmsnorm_tile(x, gamma, x_norm, seq_len, hidden, hidden, eps);

    /* Step 2: Separate GEMMs with N-outer tiling for weight reuse */
    /* Q projection */
    for (int n_start = 0; n_start < q_dim; n_start += PREFILL_TILE_N) {
        int tile_n = (n_start + PREFILL_TILE_N <= q_dim)
                         ? PREFILL_TILE_N : (q_dim - n_start);
        const float *W_tile = Wq + (size_t)n_start * hidden;

        for (int m_start = 0; m_start < seq_len; m_start += PREFILL_TILE_M) {
            int tile_m = (m_start + PREFILL_TILE_M <= seq_len)
                             ? PREFILL_TILE_M : (seq_len - m_start);
            const float *x_tile = x_norm + (size_t)m_start * hidden;
            float *out_tile = Q + (size_t)m_start * q_dim + n_start;
            gemm_tile_nt_strided(x_tile, W_tile, out_tile,
                                  tile_m, tile_n, hidden, q_dim);
        }
    }

    /* K projection */
    for (int n_start = 0; n_start < kv_dim; n_start += PREFILL_TILE_N) {
        int tile_n = (n_start + PREFILL_TILE_N <= kv_dim)
                         ? PREFILL_TILE_N : (kv_dim - n_start);
        const float *W_tile = Wk + (size_t)n_start * hidden;

        for (int m_start = 0; m_start < seq_len; m_start += PREFILL_TILE_M) {
            int tile_m = (m_start + PREFILL_TILE_M <= seq_len)
                             ? PREFILL_TILE_M : (seq_len - m_start);
            const float *x_tile = x_norm + (size_t)m_start * hidden;
            float *out_tile = K + (size_t)m_start * kv_dim + n_start;
            gemm_tile_nt_strided(x_tile, W_tile, out_tile,
                                  tile_m, tile_n, hidden, kv_dim);
        }
    }

    /* V projection */
    for (int n_start = 0; n_start < kv_dim; n_start += PREFILL_TILE_N) {
        int tile_n = (n_start + PREFILL_TILE_N <= kv_dim)
                         ? PREFILL_TILE_N : (kv_dim - n_start);
        const float *W_tile = Wv + (size_t)n_start * hidden;

        for (int m_start = 0; m_start < seq_len; m_start += PREFILL_TILE_M) {
            int tile_m = (m_start + PREFILL_TILE_M <= seq_len)
                             ? PREFILL_TILE_M : (seq_len - m_start);
            const float *x_tile = x_norm + (size_t)m_start * hidden;
            float *out_tile = V + (size_t)m_start * kv_dim + n_start;
            gemm_tile_nt_strided(x_tile, W_tile, out_tile,
                                  tile_m, tile_n, hidden, kv_dim);
        }
    }
}

/**
 * @brief Get scratch size for fused prefill
 */
size_t fused_rmsnorm_qkv_scratch_size(int hidden) {
    return (size_t)PREFILL_TILE_M * hidden * sizeof(float);
}

/**
 * @brief Fused MLP for prefill with proper tiling
 */
void fused_mlp_swiglu_prefill_bias(
    const float *x,
    const float *W_gate,
    const float *W_up,
    const float *W_down,
    const float *B_gate,
    const float *B_up,
    const float *B_down,
    float *output,
    int seq_len,
    int hidden,
    int intermediate,
    float *scratch)
{
    /* MLP is more complex because we have:
     * gate = x @ W_gate
     * up = x @ W_up
     * hidden = silu(gate) * up
     * out = hidden @ W_down
     *
     * The intermediate (gate, up, hidden) is large: seq_len × intermediate
     * For Qwen2-0.5B: 1024 × 4864 × 4 = 19.4MB (way bigger than L3!)
     *
     * Strategy: Tile along intermediate dimension for gate/up,
     * then fuse SwiGLU, then tile down projection.
     */

    /* scratch layout:
     * [gate_tile: TILE_M × TILE_N_INTER]
     * [up_tile: TILE_M × TILE_N_INTER]
     */
    const int TILE_N_INTER = 512;  /* Intermediate tile size */
    float *gate_tile = scratch;
    float *up_tile = scratch + (size_t)PREFILL_TILE_M * TILE_N_INTER;
    float *hidden_tile = gate_tile;  /* Reuse gate_tile for hidden after SwiGLU */

    /* For each chunk of intermediate dimension */
    for (int inter_start = 0; inter_start < intermediate; inter_start += TILE_N_INTER) {
        int tile_inter = (inter_start + TILE_N_INTER <= intermediate)
                             ? TILE_N_INTER : (intermediate - inter_start);

        const float *W_gate_tile = W_gate + (size_t)inter_start * hidden;
        const float *W_up_tile = W_up + (size_t)inter_start * hidden;

        /* For each chunk of tokens */
        for (int m_start = 0; m_start < seq_len; m_start += PREFILL_TILE_M) {
            int tile_m = (m_start + PREFILL_TILE_M <= seq_len)
                             ? PREFILL_TILE_M : (seq_len - m_start);

            const float *x_tile = x + (size_t)m_start * hidden;

            /* Compute gate and up projections for this tile */
            gemm_tile_nt_strided(x_tile, W_gate_tile, gate_tile,
                                  tile_m, tile_inter, hidden, tile_inter);
            gemm_tile_nt_strided(x_tile, W_up_tile, up_tile,
                                  tile_m, tile_inter, hidden, tile_inter);
            if (B_gate) {
                add_bias_tile(gate_tile, B_gate + inter_start, tile_m, tile_inter);
            }
            if (B_up) {
                add_bias_tile(up_tile, B_up + inter_start, tile_m, tile_inter);
            }

            /* Fused SwiGLU: hidden = silu(gate) * up */
            for (int i = 0; i < tile_m; ++i) {
                float *g = gate_tile + (size_t)i * tile_inter;
                float *u = up_tile + (size_t)i * tile_inter;
                for (int j = 0; j < tile_inter; ++j) {
                    float gv = g[j];
                    float silu = gv / (1.0f + expf(-gv));
                    g[j] = silu * u[j];  /* hidden_tile = gate_tile */
                }
            }

            /* Down projection: accumulate into output
             * out[m_start:, :] += hidden_tile @ W_down[inter_start:, :]^T
             */
            float *out_tile = output + (size_t)m_start * hidden;

            /* This is trickier - W_down is [hidden × intermediate]
             * We have hidden_tile[tile_m × tile_inter]
             * We want out[tile_m × hidden] += hidden_tile × W_down[:, inter_start:inter_start+tile_inter]^T
             *
             * For proper accumulation, need to handle this carefully.
             * For now, use a simpler approach: accumulate partial results.
             */
            for (int i = 0; i < tile_m; ++i) {
                float *h = hidden_tile + (size_t)i * tile_inter;
                float *o = out_tile + (size_t)i * hidden;

                for (int d = 0; d < hidden; ++d) {
                    const float *w_row = W_down + (size_t)d * intermediate + inter_start;
                    float sum = (inter_start == 0)
                        ? (B_down ? B_down[d] : 0.0f)
                        : o[d];

#if defined(__AVX512F__)
                    __m512 acc = _mm512_setzero_ps();
                    int j = 0;
                    for (; j + 16 <= tile_inter; j += 16) {
                        __m512 hv = _mm512_loadu_ps(h + j);
                        __m512 wv = _mm512_loadu_ps(w_row + j);
                        acc = _mm512_fmadd_ps(hv, wv, acc);
                    }
                    sum += _mm512_reduce_add_ps(acc);
                    for (; j < tile_inter; ++j) {
                        sum += h[j] * w_row[j];
                    }
#elif defined(__AVX__)
                    __m256 acc = _mm256_setzero_ps();
                    int j = 0;
                    for (; j + 8 <= tile_inter; j += 8) {
                        __m256 hv = _mm256_loadu_ps(h + j);
                        __m256 wv = _mm256_loadu_ps(w_row + j);
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(hv, wv));
                    }
                    sum += hsum256_prefill(acc);
                    for (; j < tile_inter; ++j) {
                        sum += h[j] * w_row[j];
                    }
#else
                    for (int j = 0; j < tile_inter; ++j) {
                        sum += h[j] * w_row[j];
                    }
#endif
                    o[d] = sum;
                }
            }
        }
    }
}

void fused_mlp_swiglu_prefill(
    const float *x,
    const float *W_gate,
    const float *W_up,
    const float *W_down,
    float *output,
    int seq_len,
    int hidden,
    int intermediate,
    float *scratch)
{
    fused_mlp_swiglu_prefill_bias(x, W_gate, W_up, W_down,
                                  NULL, NULL, NULL,
                                  output, seq_len, hidden, intermediate,
                                  scratch);
}

/**
 * @brief Get scratch size for fused MLP
 */
size_t fused_mlp_swiglu_scratch_size(int intermediate) {
    const int TILE_N_INTER = 512;
    /* gate_tile + up_tile */
    return 2 * (size_t)PREFILL_TILE_M * TILE_N_INTER * sizeof(float);
}

static inline float silu_prefill(float x) {
    return x / (1.0f + expf(-x));
}

static int mlp_q8_0_dtype_supported(CKDataType dt) {
    return (dt == CK_DT_Q5_0 || dt == CK_DT_Q8_0);
}

static int mlp_q8_k_dtype_supported(CKDataType dt) {
    return (dt == CK_DT_Q4_K || dt == CK_DT_Q6_K);
}

static void gemm_nt_q8_0_mlp_dispatch(const void *A_q8,
                                     const void *B,
                                     const float *bias,
                                     float *C,
                                     int M,
                                     int N,
                                     int K,
                                     CKDataType dt)
{
    switch (dt) {
    case CK_DT_Q5_0:
        gemm_nt_q5_0_q8_0(A_q8, B, bias, C, M, N, K);
        break;
    case CK_DT_Q8_0:
        gemm_nt_q8_0_q8_0(A_q8, B, bias, C, M, N, K);
        break;
    default:
        break;
    }
}

static void gemm_nt_q8_k_mlp_dispatch(const void *A_q8,
                                     const void *B,
                                     const float *bias,
                                     float *C,
                                     int M,
                                     int N,
                                     int K,
                                     CKDataType dt)
{
    switch (dt) {
    case CK_DT_Q4_K:
        gemm_nt_q4_k_q8_k(A_q8, B, bias, C, M, N, K);
        break;
    case CK_DT_Q6_K:
        gemm_nt_q6_k_q8_k(A_q8, B, bias, C, M, N, K);
        break;
    default:
        break;
    }
}

/**
 * @brief Quantized fused MLP for prefill (W1=gate+up, W2=down)
 *
 * Uses Q8_0 activations for W1 (Q5_0/Q8_0 weights) and Q8_K activations
 * for W2 (Q4_K/Q6_K weights).
 */
void fused_mlp_swiglu_prefill_w1w2_quant(
    const float *x,
    const void *W1,
    const float *B1,
    CKDataType w1_dt,
    const void *W2,
    const float *B2,
    CKDataType w2_dt,
    float *output,
    int seq_len,
    int embed_dim,
    int aligned_embed_dim,
    int intermediate_dim,
    int aligned_intermediate_dim,
    void *scratch)
{
    if (!x || !W1 || !W2 || !output || !scratch) {
        return;
    }
    if (seq_len <= 0 || embed_dim <= 0 || aligned_embed_dim <= 0 ||
        intermediate_dim <= 0 || aligned_intermediate_dim <= 0) {
        return;
    }
    if (aligned_embed_dim < embed_dim || aligned_intermediate_dim < intermediate_dim) {
        return;
    }
    if ((aligned_embed_dim % 32) != 0 || (aligned_intermediate_dim % 256) != 0) {
        return;
    }
    if (!mlp_q8_0_dtype_supported(w1_dt) || !mlp_q8_k_dtype_supported(w2_dt)) {
        return;
    }

    const int tile_m_max = PREFILL_TILE_M;
    const int inter = aligned_intermediate_dim;
    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)aligned_embed_dim);
    const size_t q8k_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)aligned_intermediate_dim);
    const size_t w1_row_bytes = ck_dtype_row_bytes(w1_dt, (size_t)aligned_embed_dim);

    uint8_t *scratch_bytes = (uint8_t *)scratch;
    size_t q8_bytes = (size_t)tile_m_max * q8_row_bytes;
    size_t gate_bytes = (size_t)tile_m_max * (size_t)inter * sizeof(float);
    size_t up_bytes = gate_bytes;
    size_t gate_offset = align_up_size(q8_bytes, 64);
    size_t up_offset = gate_offset + align_up_size(gate_bytes, 64);
    size_t q8k_offset = up_offset + align_up_size(up_bytes, 64);

    uint8_t *q8_tile = scratch_bytes;
    float *gate_tile = (float *)(scratch_bytes + gate_offset);
    float *up_tile = (float *)(scratch_bytes + up_offset);
    uint8_t *q8k_tile = scratch_bytes + q8k_offset;

    const uint8_t *w1_base = (const uint8_t *)W1;
    const uint8_t *w_gate = w1_base;
    const uint8_t *w_up = w1_base + (size_t)inter * w1_row_bytes;

    const float *b_gate = B1;
    const float *b_up = B1 ? (B1 + (size_t)inter) : NULL;

    for (int m_start = 0; m_start < seq_len; m_start += tile_m_max) {
        int tile_m = (m_start + tile_m_max <= seq_len)
                         ? tile_m_max : (seq_len - m_start);

        const float *x_tile = x + (size_t)m_start * (size_t)aligned_embed_dim;
        float *out_tile = output + (size_t)m_start * (size_t)aligned_embed_dim;

        for (int t = 0; t < tile_m; ++t) {
            const float *row = x_tile + (size_t)t * (size_t)aligned_embed_dim;
            quantize_row_q8_0(row,
                              q8_tile + (size_t)t * q8_row_bytes,
                              aligned_embed_dim);
        }

        gemm_nt_q8_0_mlp_dispatch(q8_tile, w_gate, b_gate, gate_tile,
                                 tile_m, inter, aligned_embed_dim, w1_dt);
        gemm_nt_q8_0_mlp_dispatch(q8_tile, w_up, b_up, up_tile,
                                 tile_m, inter, aligned_embed_dim, w1_dt);

        for (int i = 0; i < tile_m; ++i) {
            float *g = gate_tile + (size_t)i * (size_t)inter;
            float *u = up_tile + (size_t)i * (size_t)inter;
            for (int j = 0; j < inter; ++j) {
                g[j] = silu_prefill(g[j]) * u[j];
            }
        }

        for (int i = 0; i < tile_m; ++i) {
            const float *row = gate_tile + (size_t)i * (size_t)inter;
            quantize_row_q8_k(row,
                              q8k_tile + (size_t)i * q8k_row_bytes,
                              aligned_intermediate_dim);
        }

        gemm_nt_q8_k_mlp_dispatch(q8k_tile, W2, B2, out_tile,
                                  tile_m, aligned_embed_dim, aligned_intermediate_dim, w2_dt);
    }
}

size_t fused_mlp_swiglu_prefill_w1w2_quant_scratch_size(int aligned_embed_dim,
                                                        int aligned_intermediate_dim)
{
    if (aligned_embed_dim <= 0 || aligned_intermediate_dim <= 0) {
        return 0;
    }
    const size_t q8_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)aligned_embed_dim);
    const size_t q8k_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)aligned_intermediate_dim);
    const size_t q8_bytes = (size_t)PREFILL_TILE_M * q8_row_bytes;
    const size_t gate_bytes = (size_t)PREFILL_TILE_M * (size_t)aligned_intermediate_dim * sizeof(float);
    const size_t up_bytes = gate_bytes;
    const size_t q8k_bytes = (size_t)PREFILL_TILE_M * q8k_row_bytes;

    return align_up_size(q8_bytes, 64) +
           align_up_size(gate_bytes, 64) +
           align_up_size(up_bytes, 64) +
           align_up_size(q8k_bytes, 64);
}
