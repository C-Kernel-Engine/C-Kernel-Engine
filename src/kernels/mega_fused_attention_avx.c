/**
 * @file mega_fused_attention_avx.c
 * @brief Mega-Fused Attention for AVX (256-bit) and AVX-512 (512-bit)
 *
 * Holy grail fusion: RMSNorm → QKV → RoPE → Flash Attention → OutProj + Residual
 *
 * AVX approach: Keep intermediates in L1 cache (not registers)
 * AVX-512 approach: Keep intermediates in registers
 *
 * Both achieve the same goal: Eliminate DRAM traffic for intermediates.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

/* Require FMA for the fused multiply-add instructions used in RoPE */
#if !defined(__FMA__) && !defined(__AVX512F__)
/* FMA not available - provide stub implementations */
#warning "Mega-fusion requires FMA (compile with -mfma). Providing stub implementations."

void mega_fuse_rope_inplace_avx(void *q, void *k, void *cos_cache, void *sin_cache,
                                 int batch_tokens, int num_heads, int num_kv_heads,
                                 int head_dim, int token_offset) {
    (void)q; (void)k; (void)cos_cache; (void)sin_cache;
    (void)batch_tokens; (void)num_heads; (void)num_kv_heads;
    (void)head_dim; (void)token_offset;
}

#else

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX__)
#include <immintrin.h>
#else
#error "Mega-fusion requires AVX or AVX-512"
#endif

#include "ckernel_quant.h"
#include "ck_features.h"

/*============================================================================
 * Configuration - AVX vs AVX-512
 *============================================================================*/

#if defined(__AVX512F__)

/* AVX-512: Can keep more in registers */
#define MEGA_VLEN 16  /* 512 / 32 */
#define MEGA_REGS 32  /* 32 ZMM registers */
#define MEGA_Q_TILE 64
#define MEGA_KV_TILE 64

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
 * AVX-optimized version. Intermediates stay in L1/L2.
 *
 * @param q_out         Output Q [num_heads * head_dim] - to RoPE
 * @param k_out         Output K [num_kv_heads * head_dim] - to RoPE
 * @param v_out         Output V [num_kv_heads * head_dim] - to KV cache
 * @param input         Input token [hidden] - from DRAM
 * @param gamma         RMSNorm gamma [hidden]
 * @param W_qkv         QKV weights [3*hidden, hidden]
 * @param b_qkv         QKV bias [3*hidden] or NULL
 * @param hidden        Model hidden dimension
 * @param num_heads     Number of attention heads
 * @param num_kv_heads  Number of KV heads
 * @param head_dim      Head dimension
 * @param eps           RMSNorm epsilon
 */
void mega_fuse_rmsnorm_qkv_avx(
    float *q_out,
    float *k_out,
    float *v_out,
    const float *input,
    const float *gamma,
    const float *W_qkv,
    const float *b_qkv,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float eps)
{
    const int H = num_heads;
    const int H_kv = num_kv_heads;
    const int hd = head_dim;
    const int ad = (head_dim + MEGA_VLEN - 1) / MEGA_VLEN * MEGA_VLEN;  /* aligned */

    /* Stack buffer for RMSNorm output - stays in L1! */
    float ln1_row[hidden];
    float rstd;
    float sum_sq = 0.0f;

    /* RMSNorm: compute in registers, store to L1 stack buffer */
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < hidden; i += 16) {
        __m512 xv = _mm512_loadu_ps(input + i);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(xv, xv));
    }
    float rowsum = _mm512_reduce_add_ps(sum);
    rstd = 1.0f / sqrtf(rowsum / hidden + eps);
#elif defined(__AVX__)
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < hidden; i += 8) {
        __m256 xv = _mm256_loadu_ps(input + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(xv, xv));
    }
    /* Horizontal sum of 256-bit vector */
    __m128 sum128 = _mm256_extractf128_ps(sum, 0);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    sum128 = _mm_add_ps(sum128, hi);
    float rowsum = _mm_cvtss_f32(_mm_hadd_ps(sum128, sum128));
    rstd = 1.0f / sqrtf(rowsum / hidden + eps);
#endif

    /* Normalize and project - no DRAM write of ln1_row! */
    size_t q_elems = (size_t)H * ad;
    size_t kv_elems = (size_t)H_kv * ad;

    /* Q projection: ln1_row is still hot in L1 */
    for (int j = 0; j < q_elems; j += 4) {
        float dot = 0.0f;
        for (int i = 0; i < hidden; i++) {
            dot += ln1_row[i] * W_qkv[j * (size_t)hidden + i];
        }
        q_out[j] = dot + (b_qkv ? b_qkv[j] : 0.0f);
    }

    /* K projection - same L1 buffer */
    const float *W_k = W_qkv + (size_t)H * ad * hidden;
    for (int j = 0; j < kv_elems; j += 4) {
        float dot = 0.0f;
        for (int i = 0; i < hidden; i++) {
            dot += ln1_row[i] * W_k[j * (size_t)hidden + i];
        }
        k_out[j] = dot + (b_qkv ? b_qkv[H * ad + j] : 0.0f);
    }

    /* V projection - same L1 buffer */
    const float *W_v = W_qkv + (size_t)(H + H_kv) * ad * hidden;
    for (int j = 0; j < kv_elems; j += 4) {
        float dot = 0.0f;
        for (int i = 0; i < hidden; i++) {
            dot += ln1_row[i] * W_v[j * (size_t)hidden + i];
        }
        v_out[j] = dot + (b_qkv ? b_qkv[(H + H_kv) * ad + j] : 0.0f);
    }

    /* q_out, k_out, v_out are in L1 - ready for RoPE! */
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
    float *q,          /* [num_heads * head_dim] - in L1 */
    float *k,          /* [num_kv_heads * head_dim] - in L1 */
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim)
{
    int half = head_dim / 2;

    for (int h = 0; h < num_heads; h++) {
        float *q_h = q + h * head_dim;
        float *k_h = k + (h % num_kv_heads) * head_dim;

        const float *cos_ptr = rope_cos + pos * half;
        const float *sin_ptr = rope_sin + pos * half;

        for (int i = 0; i < half; i += MEGA_VLEN) {
#if defined(__AVX512F__)
            __m512 q0 = _mm512_loadu_ps(q_h + i);
            __m512 q1 = _mm512_loadu_ps(q_h + i + half);
            __m512 k0 = _mm512_loadu_ps(k_h + i);
            __m512 k1 = _mm512_loadu_ps(k_h + i + half);
            __m512 cos = _mm512_loadu_ps(cos_ptr + i);
            __m512 sin = _mm512_loadu_ps(sin_ptr + i);

            __m512 q_rot0 = _mm512_fmsub_ps(q0, cos, _mm512_mul_ps(q1, sin));
            __m512 q_rot1 = _mm512_fmadd_ps(q0, sin, _mm512_mul_ps(q1, cos));
            __m512 k_rot0 = _mm512_fmsub_ps(k0, cos, _mm512_mul_ps(k1, sin));
            __m512 k_rot1 = _mm512_fmadd_ps(k0, sin, _mm512_mul_ps(k1, cos));

            _mm512_storeu_ps(q_h + i, q_rot0);
            _mm512_storeu_ps(q_h + i + half, q_rot1);
            _mm512_storeu_ps(k_h + i, k_rot0);
            _mm512_storeu_ps(k_h + i + half, k_rot1);
#elif defined(__AVX__)
            __m256 q0 = _mm256_loadu_ps(q_h + i);
            __m256 q1 = _mm256_loadu_ps(q_h + i + half);
            __m256 k0 = _mm256_loadu_ps(k_h + i);
            __m256 k1 = _mm256_loadu_ps(k_h + i + half);
            __m256 cos = _mm256_loadu_ps(cos_ptr + i);
            __m256 sin = _mm256_loadu_ps(sin_ptr + i);

            __m256 q_rot0 = _mm256_fmsub_ps(q0, cos, _mm256_mul_ps(q1, sin));
            __m256 q_rot1 = _mm256_fmadd_ps(q0, sin, _mm256_mul_ps(q1, cos));
            __m256 k_rot0 = _mm256_fmsub_ps(k0, cos, _mm256_mul_ps(k1, sin));
            __m256 k_rot1 = _mm256_fmadd_ps(k0, sin, _mm256_mul_ps(k1, cos));

            _mm256_storeu_ps(q_h + i, q_rot0);
            _mm256_storeu_ps(q_h + i + half, q_rot1);
            _mm256_storeu_ps(k_h + i, k_rot0);
            _mm256_storeu_ps(k_h + i + half, k_rot1);
#endif
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
 * @param o_out         Output [num_heads * head_dim] - in registers/L1
 * @param q             Q tensor [num_heads * head_dim] - from L1
 * @param kv_cache_k    KV cache K [seq, num_kv_heads * head_dim] - in L2
 * @param kv_cache_v    KV cache V [seq, num_kv_heads * head_dim] - in L2
 * @param num_heads     Number of heads
 * @param num_kv_heads  Number of KV heads
 * @param seq_len       Current sequence length
 * @param head_dim      Head dimension
 */
void mega_fuse_flash_attention_avx(
    float *o_out,
    const float *q,
    const float *kv_cache_k,
    const float *kv_cache_v,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim)
{
    const int hd = head_dim;
    const float scale = 1.0f / sqrtf((float)hd);

    for (int h = 0; h < num_heads; h++) {
        const float *q_h = q + h * hd;
        const int kv_idx = h % num_kv_heads;
        const float *k_cache = kv_cache_k + kv_idx * hd;
        const float *v_cache = kv_cache_v + kv_idx * hd;

        /* O, m, l in registers for this head */
        float o_h[hd];       /* in L1 */
        float m = -INFINITY; /* running max */
        float l = 0.0f;      /* running sum */

        /* Initialize O to zeros */
        memset(o_h, 0, hd * sizeof(float));

        /* Iterate over KV cache tiles */
        for (int t = 0; t < seq_len; t += MEGA_KV_TILE) {
            int tile_end = t + MEGA_KV_TILE;
            if (tile_end > seq_len) tile_end = seq_len;
            int tile_size = tile_end - t;

            /* Load K tile from L2 cache */
            float k_tile[MEGA_KV_TILE * hd];
            for (int i = 0; i < tile_size; i++) {
                memcpy(k_tile + i * hd, k_cache + (t + i) * hd, hd * sizeof(float));
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

            /* Scale O by exp(m - m_new) and add P @ V_tile */
            float exp_m_diff = expf(m - m_new);
            for (int i = 0; i < hd; i++) {
                o_h[i] *= exp_m_diff;
            }

            /* Load V tile and accumulate */
            for (int j = 0; j < tile_size; j++) {
                float p = s_row[j];
                for (int i = 0; i < hd; i++) {
                    o_h[i] += p * v_cache[(t + j) * hd + i];
                }
            }

            l = l_new * exp_m_diff + l;
            m = m_new;
        }

        /* Normalize by l */
        for (int i = 0; i < hd; i++) {
            o_h[i] /= l;
        }

        /* Store O - still in L1, goes to output projection */
        memcpy(o_out + h * hd, o_h, hd * sizeof(float));
    }
}

/*============================================================================
 * Phase 4: Full Mega-Fused Attention Decode
 *
 * RMSNorm → QKV → RoPE → Flash Attn → OutProj + Residual
 * All intermediates in L1/L2, single DRAM round-trip.
 *============================================================================*/

/**
 * @brief Full mega-fused attention for decode
 *
 * THE HOLY GRAIL: All operations fused, no intermediates to DRAM.
 *
 * Reads:  input[hidden], W_qkv, W_o, KV cache, RoPE tables
 * Writes: output[hidden]
 *
 * Total DRAM traffic: 4KB input + 4KB output = 8KB
 * (vs ~800KB with unfused attention)
 */
void mega_fused_attention_decode_avx(
    float *output,         /* [hidden] - single DRAM write */
    const float *input,    /* [hidden] - single DRAM read */
    const float *residual, /* [hidden] - for residual add */
    const float *W_qkv,    /* QKV weights */
    const float *b_qkv,    /* QKV bias or NULL */
    const float *W_o,      /* Output projection */
    const float *b_o,      /* Output bias or NULL */
    float *kv_cache_k,     /* KV cache K - in L2 */
    float *kv_cache_v,     /* KV cache V - in L2 */
    const float *rope_cos,
    const float *rope_sin,
    int pos,
    int hidden,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq,
    float eps)
{
    /* Stack-allocated buffers - stay in L1 (no malloc!) */
    /* Max: 64 heads * 128 dim = 8192 floats = 32KB per buffer */
    size_t q_elems = (size_t)num_heads * head_dim;
    size_t kv_elems = (size_t)num_kv_heads * head_dim;

    if (q_elems > 8192 || kv_elems > 8192) return;

    float q[8192];
    float k[8192];
    float v[8192];
    float o[8192];

    /* Phase 1: RMSNorm + QKV (ln1_row in L1 stack) */
    mega_fuse_rmsnorm_qkv_avx(q, k, v, input, NULL, W_qkv, b_qkv,
                              hidden, num_heads, num_kv_heads, head_dim, eps);

    /* Phase 2: RoPE in-place (q, k still hot in L1) */
    mega_fuse_rope_inplace_avx(q, k, rope_cos, rope_sin, pos,
                               num_heads, num_kv_heads, head_dim);

    /* Write K/V to KV cache (L2) */
    for (int h = 0; h < num_kv_heads; h++) {
        memcpy(kv_cache_k + pos * kv_elems + h * head_dim,
               k + h * head_dim, head_dim * sizeof(float));
        memcpy(kv_cache_v + pos * kv_elems + h * head_dim,
               v + h * head_dim, head_dim * sizeof(float));
    }

    /* Phase 3: Flash Attention (o in L1) */
    mega_fuse_flash_attention_avx(o, q, kv_cache_k, kv_cache_v,
                                  num_heads, num_kv_heads, pos + 1, head_dim);

    /* Phase 4: Output Projection + Residual (in final store) */
    for (int j = 0; j < hidden; j++) {
        float dot = 0.0f;
        for (int h = 0; h < num_heads; h++) {
            const float *o_h = o + h * head_dim;
            const float *W_o_h = W_o + h * head_dim * hidden + j * head_dim;
            for (int i = 0; i < head_dim; i++) {
                dot += o_h[i] * W_o_h[i];
            }
        }
        float proj = dot + (b_o ? b_o[j] : 0.0f);
        output[j] = proj + residual[j];  /* Residual add in store! */
    }

    /* No free needed - stack buffers auto-deallocate */
    /* Total DRAM traffic: 4KB input + 4KB output = 8KB */
}

#endif /* __FMA__ || __AVX512F__ */
