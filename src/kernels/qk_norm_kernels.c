/**
 * @file qk_norm_kernels.c
 * @brief Per-head RMSNorm on Q and K (Qwen3-style QK norm)
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes:
 *   make v7-qk-norm-backward-parity-isa
 *   python unittest/test_qk_norm.py
 *
 * QK Norm normalizes each head's query/key vectors independently before RoPE.
 * This stabilizes Q*K^T dot products before softmax, preventing attention
 * collapse from large magnitude vectors.
 *
 * Why only Q and K, not V?
 *   V does not participate in the attention score computation (Q*K^T).
 *   The softmax saturation problem comes from large Q*K^T values, so only
 *   Q and K magnitudes matter. V is linearly combined after softmax weights
 *   are computed -- normalizing it would change output scale but not fix
 *   attention stability.
 *
 * Data layout after QKV projection (head-major):
 *   Q: [num_heads, num_tokens, head_dim]       contiguous
 *   K: [num_kv_heads, num_tokens, head_dim]     contiguous
 *
 * We treat Q as [num_heads * num_tokens] rows of [head_dim] elements.
 * rmsnorm_forward normalizes each row independently. The gamma weight [head_dim]
 * is shared across all heads (Qwen3 design: one gamma per Q, one per K).
 */

#include <math.h>
#include <stddef.h>  /* NULL */
#include <stdlib.h>  /* getenv */
#include <string.h>  /* strcmp */

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVXVNNI__)
#include <immintrin.h>
#endif

/* rmsnorm_forward is declared in ckernel_engine.h */
void rmsnorm_forward(const float *input,
                     const float *gamma,
                     float *output,
                     float *rstd_cache,
                     int tokens,
                     int d_model,
                     int aligned_embed_dim,
                     float eps);

void rmsnorm_backward(const float *d_output,
                      const float *input,
                      const float *gamma,
                      const float *rstd_cache,
                      float *d_input,
                      float *d_gamma,
                      int tokens,
                      int d_model,
                      int aligned_embed_dim);

typedef enum {
    QK_NORM_ISA_SCALAR = 0,
    QK_NORM_ISA_AVX = 1,
    QK_NORM_ISA_AVX2 = 2,
    QK_NORM_ISA_AVX_VNNI = 3,
    QK_NORM_ISA_AUTO = -1
} QKNormISA;

static int g_qk_norm_last_isa = QK_NORM_ISA_SCALAR;

int qk_norm_backward_last_isa(void)
{
    return g_qk_norm_last_isa;
}

static QKNormISA qk_norm_parse_forced_isa(void)
{
    const char *forced = getenv("CK_QK_NORM_BACKWARD_ISA");
    if (!forced || forced[0] == '\0' || strcmp(forced, "auto") == 0) {
        return QK_NORM_ISA_AUTO;
    }
    if (strcmp(forced, "scalar") == 0) {
        return QK_NORM_ISA_SCALAR;
    }
    if (strcmp(forced, "avx") == 0) {
        return QK_NORM_ISA_AVX;
    }
    if (strcmp(forced, "avx2") == 0) {
        return QK_NORM_ISA_AVX2;
    }
    if (strcmp(forced, "avx_vnni") == 0) {
        return QK_NORM_ISA_AVX_VNNI;
    }
    /* Unknown value -> keep behavior deterministic by falling back. */
    return QK_NORM_ISA_SCALAR;
}

static int qk_norm_isa_compiled(QKNormISA isa)
{
    switch (isa) {
    case QK_NORM_ISA_SCALAR:
        return 1;
#if defined(__AVX__)
    case QK_NORM_ISA_AVX:
        return 1;
#endif
#if defined(__AVX2__)
    case QK_NORM_ISA_AVX2:
        return 1;
#endif
#if defined(__AVXVNNI__)
    case QK_NORM_ISA_AVX_VNNI:
        return 1;
#endif
    default:
        return 0;
    }
}

static QKNormISA qk_norm_select_isa(void)
{
    QKNormISA forced = qk_norm_parse_forced_isa();
    if (forced != QK_NORM_ISA_AUTO) {
        return qk_norm_isa_compiled(forced) ? forced : QK_NORM_ISA_SCALAR;
    }

#if defined(__AVXVNNI__)
    return QK_NORM_ISA_AVX_VNNI;
#elif defined(__AVX2__)
    return QK_NORM_ISA_AVX2;
#elif defined(__AVX__)
    return QK_NORM_ISA_AVX;
#else
    return QK_NORM_ISA_SCALAR;
#endif
}

static void qk_norm_compute_rstd_scalar(const float *input,
                                        float *rstd_cache,
                                        int rows,
                                        int head_dim,
                                        float eps)
{
    for (int r = 0; r < rows; ++r) {
        const float *x = input + (size_t)r * (size_t)head_dim;
        double sum_sq = 0.0;
        for (int d = 0; d < head_dim; ++d) {
            double v = (double)x[d];
            sum_sq += v * v;
        }
        float mean_sq = (float)(sum_sq / (double)head_dim);
        rstd_cache[r] = 1.0f / sqrtf(mean_sq + eps);
    }
}

#if defined(__AVX__)
static inline float qk_norm_hsum256(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static void qk_norm_compute_rstd_avx(const float *input,
                                     float *rstd_cache,
                                     int rows,
                                     int head_dim,
                                     float eps)
{
    for (int r = 0; r < rows; ++r) {
        const float *x = input + (size_t)r * (size_t)head_dim;
        __m256 sum_sq_v = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 xv2 = _mm256_mul_ps(xv, xv);
            sum_sq_v = _mm256_add_ps(sum_sq_v, xv2);
        }
        float sum_sq = qk_norm_hsum256(sum_sq_v);
        for (; d < head_dim; ++d) {
            sum_sq += x[d] * x[d];
        }
        float mean_sq = sum_sq / (float)head_dim;
        rstd_cache[r] = 1.0f / sqrtf(mean_sq + eps);
    }
}
#endif

#if defined(__AVX2__)
static void qk_norm_compute_rstd_avx2(const float *input,
                                      float *rstd_cache,
                                      int rows,
                                      int head_dim,
                                      float eps)
{
    for (int r = 0; r < rows; ++r) {
        const float *x = input + (size_t)r * (size_t)head_dim;
        __m256 sum_sq_v = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
#if defined(__FMA__)
            sum_sq_v = _mm256_fmadd_ps(xv, xv, sum_sq_v);
#else
            __m256 xv2 = _mm256_mul_ps(xv, xv);
            sum_sq_v = _mm256_add_ps(sum_sq_v, xv2);
#endif
        }
        float sum_sq = qk_norm_hsum256(sum_sq_v);
        for (; d < head_dim; ++d) {
            sum_sq += x[d] * x[d];
        }
        float mean_sq = sum_sq / (float)head_dim;
        rstd_cache[r] = 1.0f / sqrtf(mean_sq + eps);
    }
}
#endif

#if defined(__AVXVNNI__)
static void qk_norm_compute_rstd_avx_vnni(const float *input,
                                          float *rstd_cache,
                                          int rows,
                                          int head_dim,
                                          float eps)
{
#if defined(__AVX2__)
    qk_norm_compute_rstd_avx2(input, rstd_cache, rows, head_dim, eps);
#elif defined(__AVX__)
    qk_norm_compute_rstd_avx(input, rstd_cache, rows, head_dim, eps);
#else
    qk_norm_compute_rstd_scalar(input, rstd_cache, rows, head_dim, eps);
#endif
}
#endif

static void qk_norm_compute_rstd(const float *input,
                                 float *rstd_cache,
                                 int rows,
                                 int head_dim,
                                 float eps)
{
    QKNormISA selected = qk_norm_select_isa();
    g_qk_norm_last_isa = (int)selected;
    switch (selected) {
#if defined(__AVXVNNI__)
    case QK_NORM_ISA_AVX_VNNI:
        qk_norm_compute_rstd_avx_vnni(input, rstd_cache, rows, head_dim, eps);
        return;
#endif
#if defined(__AVX2__)
    case QK_NORM_ISA_AVX2:
        qk_norm_compute_rstd_avx2(input, rstd_cache, rows, head_dim, eps);
        return;
#endif
#if defined(__AVX__)
    case QK_NORM_ISA_AVX:
        qk_norm_compute_rstd_avx(input, rstd_cache, rows, head_dim, eps);
        return;
#endif
    case QK_NORM_ISA_SCALAR:
    case QK_NORM_ISA_AUTO:
    default:
        qk_norm_compute_rstd_scalar(input, rstd_cache, rows, head_dim, eps);
        return;
    }
}

/**
 * Per-head RMSNorm on Q and K.
 *
 * @param q             Q scratch buffer [num_heads * num_tokens * head_dim], in-place
 * @param k             K scratch buffer [num_kv_heads * num_tokens * head_dim], in-place
 * @param q_gamma       Q norm gamma weights [head_dim]
 * @param k_gamma       K norm gamma weights [head_dim]
 * @param num_heads     Number of query heads (e.g. 32 for Qwen3-8B)
 * @param num_kv_heads  Number of KV heads (e.g. 8 for Qwen3-8B with GQA)
 * @param num_tokens    Number of tokens (1 for decode, T for prefill)
 * @param head_dim      Dimension per head (e.g. 128)
 * @param eps           RMSNorm epsilon (e.g. 1e-6)
 *
 * @test unittest/test_qk_norm.py
 */
void qk_norm_forward(float *q, float *k,
                     const float *q_gamma, const float *k_gamma,
                     int num_heads, int num_kv_heads,
                     int num_tokens, int head_dim, float eps)
{
    /* Q norm: [num_heads * num_tokens] rows of [head_dim]
     * Each row is one head's vector for one token. */
    rmsnorm_forward(q, q_gamma, q, NULL,
                    num_heads * num_tokens, head_dim, head_dim, eps);

    /* K norm: [num_kv_heads * num_tokens] rows of [head_dim]
     * Same logic, fewer rows when using GQA. */
    rmsnorm_forward(k, k_gamma, k, NULL,
                    num_kv_heads * num_tokens, head_dim, head_dim, eps);
}

/**
 * Backward pass for per-head QK RMSNorm.
 *
 * This computes:
 * - d_q / d_k for the Q and K activations
 * - d_q_gamma / d_k_gamma for shared per-head gamma vectors
 *
 * Implementation is reference-first and deterministic:
 * 1) recompute row rstd values from saved q/k inputs
 * 2) call rmsnorm_backward on flattened [rows, head_dim] views
 */
void qk_norm_backward(const float *d_q_out, const float *d_k_out,
                      const float *q_in, const float *k_in,
                      const float *q_gamma, const float *k_gamma,
                      float *d_q_in, float *d_k_in,
                      float *d_q_gamma, float *d_k_gamma,
                      int num_heads, int num_kv_heads,
                      int num_tokens, int head_dim, float eps)
{
    int q_rows = num_heads * num_tokens;
    int k_rows = num_kv_heads * num_tokens;

    if (q_rows > 0) {
        float q_rstd_cache[q_rows];
        qk_norm_compute_rstd(q_in, q_rstd_cache, q_rows, head_dim, eps);
        rmsnorm_backward(d_q_out, q_in, q_gamma, q_rstd_cache,
                         d_q_in, d_q_gamma, q_rows, head_dim, head_dim);
    }

    if (k_rows > 0) {
        float k_rstd_cache[k_rows];
        qk_norm_compute_rstd(k_in, k_rstd_cache, k_rows, head_dim, eps);
        rmsnorm_backward(d_k_out, k_in, k_gamma, k_rstd_cache,
                         d_k_in, d_k_gamma, k_rows, head_dim, head_dim);
    }
}
