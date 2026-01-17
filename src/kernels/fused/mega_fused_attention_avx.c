/**
 * @file mega_fused_attention_avx.c
 * @brief Mega-Fused Attention for AVX (256-bit) and AVX-512 (512-bit)
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. NO memcpy for layout - use strided access, not copies
 * 4. API must define: inputs, outputs, workspace, and memory layouts
 * 5. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * VIOLATION: Uses malloc for intermediate buffers and memcpy for layout.
 * TODO: Refactor to use bump allocator workspace and strided access.
 *
 * Holy grail fusion: RMSNorm → QKV → RoPE → Flash Attention → OutProj + Residual
 *
 * AVX approach: Keep intermediates in L1 cache (not registers)
 * AVX-512 approach: Keep intermediates in registers
 *
 * Both achieve the same goal: Eliminate DRAM traffic for intermediates.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ckernel_engine.h"
#include "ckernel_quant.h"
#include "ck_features.h"

#if defined(__AVX__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Local helpers (keep this file self-contained). */
#if defined(__AVX__) && !defined(__AVX512F__)
static inline float ck_hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

static inline float ck_dot_f32(const float *a, const float *b, int len)
{
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i <= len - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#elif defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= len - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float sum = ck_hsum256_ps(acc);
    for (; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

/*============================================================================
 * Configuration - AVX vs AVX-512
 *============================================================================*/

#if defined(__AVX512F__)

/* AVX-512: Can keep more in registers */
#define MEGA_VLEN 16  /* 512 / 32 */
#define MEGA_REGS 32  /* 32 ZMM registers */
#define MEGA_Q_TILE 64
#define MEGA_KV_TILE 64
#define MEGA_STACK_MAX 8192

/* Register allocation for AVX-512 */
#define REG_Q_ACCUM    "ZMM0-ZMM11"   /* 12 regs for Q tile */
#define REG_K_TILE     "ZMM12-ZMM17"  /* 6 regs for K tile */
#define REG_V_TILE     "ZMM18-ZMM23"  /* 6 regs for V tile */
#define REG_O_ACCUM    "ZMM24-ZMM27"  /* 4 regs for O tile */
#define REG_SOFTMAX    "ZMM28-ZMM29"  /* 2 regs for m, l */
#define REG_TEMP       "ZMM30-ZMM31"  /* 2 regs for temps */

#else

/* AVX: Smaller tiles to fit in L1 cache */
#define MEGA_VLEN 8   /* 256 / 32 */
#define MEGA_REGS 16  /* 16 YMM registers */
#define MEGA_Q_TILE 32
#define MEGA_KV_TILE 32
#define MEGA_STACK_MAX 8192

/* Register allocation for AVX - use L1 cache for larger working set */
#define REG_Q_ACCUM    "YMM0-YMM7"    /* 8 regs for Q tile */
#define REG_K_TILE     "YMM8-YMM11"   /* 4 regs for K tile */
#define REG_V_TILE     "YMM12-YMM15"  /* 4 regs for V tile */
#define REG_O_ACCUM    "Stack+L1"     /* O in L1 cache */
#define REG_SOFTMAX    "YMM0-YMM1"    /* 2 regs for m, l */
#define REG_TEMP       "YMM2-YMM3"    /* 2 regs for temps */

#endif

/*============================================================================
 * Phase 1: RMSNorm + QKV Fusion (AVX version)
 *
 * Keep ln1_row in stack buffer, not DRAM.
 * Q/K/V go directly to next operation.
 *============================================================================*/

/**
 * @brief Fused RMSNorm + QKV for decode (single token)
 *
 * Intermediates stay in L1/L2. Output buffers are head-major.
 */
void mega_fuse_rmsnorm_qkv_avx(
    float *q_out,
    float *k_out,
    float *v_out,
    const float *input,
    const float *gamma,
    const float *wq,
    const float *bq,
    const float *wk,
    const float *bk,
    const float *wv,
    const float *bv,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    float eps)
{
    if (!q_out || !k_out || !v_out || !input || !wq || !wk || !wv) {
        return;
    }
    if (embed_dim <= 0 || aligned_embed_dim <= 0 || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    float ln1_row[aligned_embed_dim];
    float sum_sq = 0.0f;

#if defined(__AVX512F__)
    __m512 sum_vec = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= embed_dim; i += 16) {
        __m512 xv = _mm512_loadu_ps(input + i);
        sum_vec = _mm512_fmadd_ps(xv, xv, sum_vec);
    }
    sum_sq = _mm512_reduce_add_ps(sum_vec);
    for (; i < embed_dim; ++i) {
        sum_sq += input[i] * input[i];
    }
#elif defined(__AVX__)
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= embed_dim; i += 8) {
        __m256 xv = _mm256_loadu_ps(input + i);
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(xv, xv));
    }
    sum_sq = ck_hsum256_ps(sum_vec);
    for (; i < embed_dim; ++i) {
        sum_sq += input[i] * input[i];
    }
#else
    for (int i = 0; i < embed_dim; ++i) {
        sum_sq += input[i] * input[i];
    }
#endif

    float rstd = 1.0f / sqrtf(sum_sq / (float)embed_dim + eps);

#if defined(__AVX512F__)
    if (gamma) {
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        int j = 0;
        for (; j + 16 <= embed_dim; j += 16) {
            __m512 xv = _mm512_loadu_ps(input + j);
            __m512 gv = _mm512_loadu_ps(gamma + j);
            __m512 yv = _mm512_mul_ps(_mm512_mul_ps(xv, rstd_vec), gv);
            _mm512_storeu_ps(ln1_row + j, yv);
        }
        for (; j < embed_dim; ++j) {
            ln1_row[j] = input[j] * rstd * gamma[j];
        }
    } else {
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        int j = 0;
        for (; j + 16 <= embed_dim; j += 16) {
            __m512 xv = _mm512_loadu_ps(input + j);
            __m512 yv = _mm512_mul_ps(xv, rstd_vec);
            _mm512_storeu_ps(ln1_row + j, yv);
        }
        for (; j < embed_dim; ++j) {
            ln1_row[j] = input[j] * rstd;
        }
    }
#elif defined(__AVX__)
    if (gamma) {
        __m256 rstd_vec = _mm256_set1_ps(rstd);
        int j = 0;
        for (; j + 8 <= embed_dim; j += 8) {
            __m256 xv = _mm256_loadu_ps(input + j);
            __m256 gv = _mm256_loadu_ps(gamma + j);
            __m256 yv = _mm256_mul_ps(_mm256_mul_ps(xv, rstd_vec), gv);
            _mm256_storeu_ps(ln1_row + j, yv);
        }
        for (; j < embed_dim; ++j) {
            ln1_row[j] = input[j] * rstd * gamma[j];
        }
    } else {
        __m256 rstd_vec = _mm256_set1_ps(rstd);
        int j = 0;
        for (; j + 8 <= embed_dim; j += 8) {
            __m256 xv = _mm256_loadu_ps(input + j);
            __m256 yv = _mm256_mul_ps(xv, rstd_vec);
            _mm256_storeu_ps(ln1_row + j, yv);
        }
        for (; j < embed_dim; ++j) {
            ln1_row[j] = input[j] * rstd;
        }
    }
#else
    for (int j = 0; j < embed_dim; ++j) {
        ln1_row[j] = input[j] * rstd * (gamma ? gamma[j] : 1.0f);
    }
#endif

    for (int j = embed_dim; j < aligned_embed_dim; ++j) {
        ln1_row[j] = 0.0f;
    }

    const size_t head_w_stride = (size_t)aligned_head_dim * (size_t)aligned_embed_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *wq_h = wq + (size_t)h * head_w_stride;
        const float *bq_h = bq ? (bq + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *q_h = q_out + (size_t)h * (size_t)aligned_head_dim;
        for (int d = 0; d < head_dim; ++d) {
            const float *row = wq_h + (size_t)d * (size_t)aligned_embed_dim;
            float sum = ck_dot_f32(ln1_row, row, aligned_embed_dim);
            q_h[d] = sum + (bq_h ? bq_h[d] : 0.0f);
        }
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            q_h[d] = 0.0f;
        }
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *wk_h = wk + (size_t)h * head_w_stride;
        const float *wv_h = wv + (size_t)h * head_w_stride;
        const float *bk_h = bk ? (bk + (size_t)h * (size_t)aligned_head_dim) : NULL;
        const float *bv_h = bv ? (bv + (size_t)h * (size_t)aligned_head_dim) : NULL;
        float *k_h = k_out + (size_t)h * (size_t)aligned_head_dim;
        float *v_h = v_out + (size_t)h * (size_t)aligned_head_dim;
        for (int d = 0; d < head_dim; ++d) {
            const float *wk_row = wk_h + (size_t)d * (size_t)aligned_embed_dim;
            const float *wv_row = wv_h + (size_t)d * (size_t)aligned_embed_dim;
            float k_sum = ck_dot_f32(ln1_row, wk_row, aligned_embed_dim);
            float v_sum = ck_dot_f32(ln1_row, wv_row, aligned_embed_dim);
            k_h[d] = k_sum + (bk_h ? bk_h[d] : 0.0f);
            v_h[d] = v_sum + (bv_h ? bv_h[d] : 0.0f);
        }
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_h[d] = 0.0f;
            v_h[d] = 0.0f;
        }
    }
}

/*============================================================================
 * Phase 2: RoPE In-Place (Q/K still hot in L1/L2)
 *============================================================================*/

/**
 * @brief Apply RoPE to Q and K (in-place, from L1)
 *
 * Q and K are already in L1 from QKV projection.
 * Just apply rotation in-place.
 */
void mega_fuse_rope_inplace_avx(
    float *q,          /* [num_heads * aligned_head_dim] - in L1 */
    float *k,          /* [num_kv_heads * aligned_head_dim] - in L1 */
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim)
{
    if (!q || !k || !rope_cos || !rope_sin || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }
    if ((head_dim & 1) != 0) {
        return;
    }

    int half = head_dim / 2;
    const float *cos_ptr = rope_cos + (size_t)pos * (size_t)half;
    const float *sin_ptr = rope_sin + (size_t)pos * (size_t)half;

    for (int h = 0; h < num_heads; ++h) {
        float *q_h = q + (size_t)h * (size_t)aligned_head_dim;
        int i = 0;
#if defined(__AVX512F__)
        for (; i + 16 <= half; i += 16) {
            __m512 q0 = _mm512_loadu_ps(q_h + i);
            __m512 q1 = _mm512_loadu_ps(q_h + i + half);
            __m512 cos = _mm512_loadu_ps(cos_ptr + i);
            __m512 sin = _mm512_loadu_ps(sin_ptr + i);

            __m512 q_rot0 = _mm512_sub_ps(_mm512_mul_ps(q0, cos), _mm512_mul_ps(q1, sin));
            __m512 q_rot1 = _mm512_add_ps(_mm512_mul_ps(q0, sin), _mm512_mul_ps(q1, cos));

            _mm512_storeu_ps(q_h + i, q_rot0);
            _mm512_storeu_ps(q_h + i + half, q_rot1);
        }
#elif defined(__AVX__)
        for (; i + 8 <= half; i += 8) {
            __m256 q0 = _mm256_loadu_ps(q_h + i);
            __m256 q1 = _mm256_loadu_ps(q_h + i + half);
            __m256 cos = _mm256_loadu_ps(cos_ptr + i);
            __m256 sin = _mm256_loadu_ps(sin_ptr + i);

            __m256 q_rot0 = _mm256_sub_ps(_mm256_mul_ps(q0, cos), _mm256_mul_ps(q1, sin));
            __m256 q_rot1 = _mm256_add_ps(_mm256_mul_ps(q0, sin), _mm256_mul_ps(q1, cos));

            _mm256_storeu_ps(q_h + i, q_rot0);
            _mm256_storeu_ps(q_h + i + half, q_rot1);
        }
#endif
        for (; i < half; ++i) {
            float q0 = q_h[i];
            float q1 = q_h[i + half];
            float c = cos_ptr[i];
            float s = sin_ptr[i];
            q_h[i] = q0 * c - q1 * s;
            q_h[i + half] = q0 * s + q1 * c;
        }
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            q_h[d] = 0.0f;
        }
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        float *k_h = k + (size_t)h * (size_t)aligned_head_dim;
        int i = 0;
#if defined(__AVX512F__)
        for (; i + 16 <= half; i += 16) {
            __m512 k0 = _mm512_loadu_ps(k_h + i);
            __m512 k1 = _mm512_loadu_ps(k_h + i + half);
            __m512 cos = _mm512_loadu_ps(cos_ptr + i);
            __m512 sin = _mm512_loadu_ps(sin_ptr + i);

            __m512 k_rot0 = _mm512_sub_ps(_mm512_mul_ps(k0, cos), _mm512_mul_ps(k1, sin));
            __m512 k_rot1 = _mm512_add_ps(_mm512_mul_ps(k0, sin), _mm512_mul_ps(k1, cos));

            _mm512_storeu_ps(k_h + i, k_rot0);
            _mm512_storeu_ps(k_h + i + half, k_rot1);
        }
#elif defined(__AVX__)
        for (; i + 8 <= half; i += 8) {
            __m256 k0 = _mm256_loadu_ps(k_h + i);
            __m256 k1 = _mm256_loadu_ps(k_h + i + half);
            __m256 cos = _mm256_loadu_ps(cos_ptr + i);
            __m256 sin = _mm256_loadu_ps(sin_ptr + i);

            __m256 k_rot0 = _mm256_sub_ps(_mm256_mul_ps(k0, cos), _mm256_mul_ps(k1, sin));
            __m256 k_rot1 = _mm256_add_ps(_mm256_mul_ps(k0, sin), _mm256_mul_ps(k1, cos));

            _mm256_storeu_ps(k_h + i, k_rot0);
            _mm256_storeu_ps(k_h + i + half, k_rot1);
        }
#endif
        for (; i < half; ++i) {
            float k0 = k_h[i];
            float k1 = k_h[i + half];
            float c = cos_ptr[i];
            float s = sin_ptr[i];
            k_h[i] = k0 * c - k1 * s;
            k_h[i + half] = k0 * s + k1 * c;
        }
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_h[d] = 0.0f;
        }
    }
}

/*============================================================================
 * Phase 3: Flash Attention with Online Softmax
 *
 * O, m, l stay in registers across all KV tiles.
 * K/V stream from KV cache (in L2).
 *============================================================================*/

/**
 * @brief Flash attention with online softmax (AVX version)
 *
 * Key insight: O, m, l stay in registers throughout!
 * K/V tiles stream from L2 cache.
 *
 * @param o_out         Output [num_heads * aligned_head_dim] - in registers/L1
 * @param q             Q tensor [num_heads * aligned_head_dim] - from L1
 * @param kv_cache_k    KV cache K [num_kv_heads * cache_capacity * aligned_head_dim]
 * @param kv_cache_v    KV cache V [num_kv_heads * cache_capacity * aligned_head_dim]
 * @param num_heads     Number of heads
 * @param num_kv_heads  Number of KV heads
 * @param seq_len       Current sequence length
 * @param cache_capacity KV cache capacity (head stride)
 * @param head_dim      Head dimension
 * @param aligned_head_dim Aligned head dimension
 */
void mega_fuse_flash_attention_avx(
    float *o_out,
    const float *q,
    const float *kv_cache_k,
    const float *kv_cache_v,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int cache_capacity,
    int head_dim,
    int aligned_head_dim)
{
    const int hd = head_dim;
    const float scale = 1.0f / sqrtf((float)hd);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; h++) {
        const float *q_h = q + (size_t)h * (size_t)aligned_head_dim;
        const int kv_idx = h % num_kv_heads;
        const float *k_cache = kv_cache_k + (size_t)kv_idx * head_stride;
        const float *v_cache = kv_cache_v + (size_t)kv_idx * head_stride;

        /* O, m, l in registers for this head */
        float o_h[aligned_head_dim]; /* in L1 */
        float m = -INFINITY; /* running max */
        float l = 0.0f;      /* running sum */

        /* Initialize O to zeros */
        memset(o_h, 0, (size_t)aligned_head_dim * sizeof(float));

        /* Iterate over KV cache tiles */
        for (int t = 0; t < seq_len; t += MEGA_KV_TILE) {
            int tile_end = t + MEGA_KV_TILE;
            if (tile_end > seq_len) tile_end = seq_len;
            int tile_size = tile_end - t;

            /* Load K tile from L2 cache */
            float k_tile[MEGA_KV_TILE * hd];
            for (int i = 0; i < tile_size; i++) {
                memcpy(k_tile + (size_t)i * (size_t)hd,
                       k_cache + (size_t)(t + i) * (size_t)aligned_head_dim,
                       (size_t)hd * sizeof(float));
            }

            /* S_ij = Q @ K_tile.T / sqrt(d) - in registers */
            float s_row[MEGA_KV_TILE];
            for (int j = 0; j < tile_size; j++) {
                s_row[j] = 0.0f;
                for (int i = 0; i < hd; i++) {
                    s_row[j] += q_h[i] * k_tile[j * hd + i];
                }
                s_row[j] *= scale;
            }

            /* Online softmax update */
            float m_new = m;
            for (int j = 0; j < tile_size; j++) {
                if (s_row[j] > m_new) m_new = s_row[j];
            }

            float l_new = 0.0f;
            for (int j = 0; j < tile_size; j++) {
                float p = expf(s_row[j] - m_new);
                s_row[j] = p;
                l_new += p;
            }

            /* Scale O by exp(m - m_new) and add P @ V */
            float exp_m_diff = expf(m - m_new);
            for (int i = 0; i < hd; i++) {
                o_h[i] *= exp_m_diff;
            }

            /* Load V tile and accumulate */
            for (int j = 0; j < tile_size; j++) {
                float p = s_row[j];
                for (int i = 0; i < hd; i++) {
                    o_h[i] += p * v_cache[(size_t)(t + j) * (size_t)aligned_head_dim + (size_t)i];
                }
            }

            l = l * exp_m_diff + l_new;
            m = m_new;
        }

        /* Normalize by l */
        for (int i = 0; i < hd; i++) {
            o_h[i] /= l;
        }
        for (int i = hd; i < aligned_head_dim; ++i) {
            o_h[i] = 0.0f;
        }

        /* Store O - still in L1, goes to output projection */
        memcpy(o_out + (size_t)h * (size_t)aligned_head_dim,
               o_h,
               (size_t)aligned_head_dim * sizeof(float));
    }
}

/*============================================================================
 * Phase 4: Full Mega-Fused Attention Decode
 *
 * RMSNorm → QKV → RoPE → Flash Attn → OutProj + Residual
 * All intermediates in L1/L2, single DRAM round-trip.
 *============================================================================*/

static void mega_fuse_output_proj_residual(
    const float *attn_token,
    const float *wo,
    const float *bo,
    const float *residual,
    float *output,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int head_dim,
    int aligned_head_dim)
{
    if (!attn_token || !wo || !output) {
        return;
    }

    const size_t head_weight_stride = (size_t)aligned_embed_dim * (size_t)aligned_head_dim;

    for (int j = 0; j < embed_dim; ++j) {
        float sum = bo ? bo[j] : 0.0f;
        for (int h = 0; h < num_heads; ++h) {
            const float *o_h = attn_token + (size_t)h * (size_t)aligned_head_dim;
            const float *wo_row = wo + (size_t)h * head_weight_stride + (size_t)j * (size_t)aligned_head_dim;
            sum += ck_dot_f32(o_h, wo_row, head_dim);
        }
        output[j] = sum + (residual ? residual[j] : 0.0f);
    }

    for (int j = embed_dim; j < aligned_embed_dim; ++j) {
        output[j] = 0.0f;
    }
}

/**
 * @brief Full mega-fused attention for decode
 *
 * RMSNorm → QKV → RoPE → Flash Attn → OutProj + Residual
 */
void mega_fused_attention_decode(
    float *output,         /* [aligned_embed_dim] */
    const float *input,    /* [aligned_embed_dim] */
    const float *residual, /* [aligned_embed_dim] */
    const float *ln1_gamma,
    const float *wq, const float *bq,
    const float *wk, const float *bk,
    const float *wv, const float *bv,
    const float *wo, const float *bo,
    float *kv_cache_k,
    float *kv_cache_v,
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int embed_dim,
    int aligned_embed_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int aligned_head_dim,
    int cache_capacity,
    float eps)
{
    if (!output || !input || !ln1_gamma || !wq || !wk || !wv || !wo ||
        !kv_cache_k || !kv_cache_v) {
        return;
    }
    if (embed_dim <= 0 || aligned_embed_dim <= 0 || head_dim <= 0 || aligned_head_dim <= 0 ||
        num_heads <= 0 || num_kv_heads <= 0 || cache_capacity <= 0) {
        return;
    }
    if (pos < 0 || pos >= cache_capacity) {
        return;
    }
    if (aligned_embed_dim < embed_dim || aligned_head_dim < head_dim) {
        return;
    }

    const size_t q_elems = (size_t)num_heads * (size_t)aligned_head_dim;
    const size_t kv_elems = (size_t)num_kv_heads * (size_t)aligned_head_dim;

    float q_stack[MEGA_STACK_MAX];
    float k_stack[MEGA_STACK_MAX];
    float v_stack[MEGA_STACK_MAX];
    float o_stack[MEGA_STACK_MAX];

    float *q = q_stack;
    float *k = k_stack;
    float *v = v_stack;
    float *o = o_stack;

    int free_q = 0;
    int free_k = 0;
    int free_v = 0;
    int free_o = 0;

    if (q_elems > MEGA_STACK_MAX) {
        q = (float *)malloc(q_elems * sizeof(float));
        if (!q) {
            return;
        }
        free_q = 1;
    }
    if (kv_elems > MEGA_STACK_MAX) {
        k = (float *)malloc(kv_elems * sizeof(float));
        if (!k) {
            if (free_q) free(q);
            return;
        }
        v = (float *)malloc(kv_elems * sizeof(float));
        if (!v) {
            if (free_q) free(q);
            free(k);
            return;
        }
        free_k = 1;
        free_v = 1;
    }
    if (q_elems > MEGA_STACK_MAX) {
        o = (float *)malloc(q_elems * sizeof(float));
        if (!o) {
            if (free_q) free(q);
            if (free_k) free(k);
            if (free_v) free(v);
            return;
        }
        free_o = 1;
    }

    mega_fuse_rmsnorm_qkv_avx(q, k, v, input, ln1_gamma,
                              wq, bq, wk, bk, wv, bv,
                              embed_dim, aligned_embed_dim,
                              num_heads, num_kv_heads,
                              head_dim, aligned_head_dim, eps);

    if (rope_cos && rope_sin) {
        mega_fuse_rope_inplace_avx(q, k, rope_cos, rope_sin, pos,
                                   num_heads, num_kv_heads,
                                   head_dim, aligned_head_dim);
    }

    kv_cache_write_head_major(k, v,
                              kv_cache_k, kv_cache_v,
                              num_kv_heads, pos,
                              cache_capacity,
                              head_dim, aligned_head_dim);

    mega_fuse_flash_attention_avx(o, q, kv_cache_k, kv_cache_v,
                                  num_heads, num_kv_heads,
                                  pos + 1, cache_capacity,
                                  head_dim, aligned_head_dim);

    mega_fuse_output_proj_residual(o, wo, bo, residual, output,
                                   embed_dim, aligned_embed_dim,
                                   num_heads, head_dim, aligned_head_dim);

    if (free_q) free(q);
    if (free_k) free(k);
    if (free_v) free(v);
    if (free_o) free(o);
}
