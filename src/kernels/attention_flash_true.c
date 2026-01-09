/**
 * @file attention_flash_true.c
 * @brief Flash-style attention (online softmax, causal, streaming).
 *
 * Layout:
 *   Q/K/V/Out: [T, H, D_h] contiguous
 *
 * Causal alignment:
 *   Queries are assumed to correspond to the last T_q positions in the KV cache.
 *   This makes T_q == T_k behave like standard causal prefill, and T_q == 1
 *   behave like decode over a full KV cache.
 *
 * Notes:
 * - This is O(T_k) per query head; it avoids materializing the score matrix.
 * - SIMD paths are provided for AVX-512 and AVX.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX__) && !defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifndef CK_FLASH_ATTN_TILE_K
#define CK_FLASH_ATTN_TILE_K 32
#endif

#ifndef CK_FLASH_ATTN_FAST_EXP
#define CK_FLASH_ATTN_FAST_EXP 0
#endif

static inline float ck_fast_expf(float x) {
    const float max_val = 88.0f;
    const float min_val = -88.0f;
    if (x > max_val) {
        x = max_val;
    } else if (x < min_val) {
        x = min_val;
    }

    const float log2e = 1.4426950408889634f;
    float z = x * log2e;
    float zf = nearbyintf(z);
    float f = z - zf;

    const float c0 = 1.0f;
    const float c1 = 0.6931471805599453f;
    const float c2 = 0.2402265069591007f;
    const float c3 = 0.05550410866482158f;
    const float c4 = 0.009618129107628478f;

    float poly = ((c4 * f + c3) * f + c2) * f + c1;
    poly = poly * f + c0;

    int32_t zi = (int32_t)zf + 127;
    uint32_t bits = (uint32_t)zi << 23;
    union {
        uint32_t i;
        float f;
    } u;
    u.i = bits;
    return poly * u.f;
}

static inline float ck_expf(float x) {
#if CK_FLASH_ATTN_FAST_EXP
    return ck_fast_expf(x);
#else
    return expf(x);
#endif
}

static inline int ck_flash_attn_tile_k(int D_h) {
    int tile = CK_FLASH_ATTN_TILE_K;
    if (D_h > 128) {
        tile = CK_FLASH_ATTN_TILE_K / 4;
    } else if (D_h > 64) {
        tile = CK_FLASH_ATTN_TILE_K / 2;
    }

    if (CK_FLASH_ATTN_TILE_K >= 8 && tile < 8) {
        tile = 8;
    }
    if (tile > CK_FLASH_ATTN_TILE_K) {
        tile = CK_FLASH_ATTN_TILE_K;
    }
    if (tile < 1) {
        tile = 1;
    }
    return tile;
}

int ck_flash_attn_choose_tile_k(int D_h) {
    return ck_flash_attn_tile_k(D_h);
}

int ck_flash_attn_fast_exp_kind(void) {
#if CK_FLASH_ATTN_FAST_EXP
#if defined(__AVX512F__)
    return 512;
#elif defined(__AVX__)
    return 256;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

static inline int max_k_for_query(int t_q, int T_q, int T_k) {
    int q_pos_offset = (T_k > T_q) ? (T_k - T_q) : 0;
    int max_k = q_pos_offset + t_q;
    if (max_k >= T_k) {
        max_k = T_k - 1;
    }
    return max_k;
}

/* ============================================================================
 * SCALAR REFERENCE IMPLEMENTATION
 * ============================================================================ */

/**
 * @brief Scalar flash-style attention (online softmax)
 */
static void attention_flash_decode_scalar(
    float *out,
    const float *q,
    const float *k,
    const float *v,
    int T_q,
    int T_k,
    int H,
    int D_h,
    float scale)
{
    const int total = T_q * H;
    const size_t stride = (size_t)H * (size_t)D_h;
    const int tile_k = ck_flash_attn_tile_k(D_h);

    for (int idx = 0; idx < total; ++idx) {
        const int t_q = idx / H;
        const int h = idx - t_q * H;
        const int max_k = max_k_for_query(t_q, T_q, T_k);

        const float *q_head = q + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
        float *out_head = out + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
        const float *k_base = k + (size_t)h * (size_t)D_h;
        const float *v_base = v + (size_t)h * (size_t)D_h;

        for (int d = 0; d < D_h; ++d) {
            out_head[d] = 0.0f;
        }

        float m = -INFINITY;
        float s = 0.0f;

        float scores[CK_FLASH_ATTN_TILE_K];

        for (int t_k0 = 0; t_k0 <= max_k; t_k0 += tile_k) {
            int blk_len = max_k - t_k0 + 1;
            if (blk_len > tile_k) {
                blk_len = tile_k;
            }

            float m_block = -INFINITY;
            for (int bi = 0; bi < blk_len; ++bi) {
                const int t_k = t_k0 + bi;
                const float *k_head = k_base + (size_t)t_k * stride;

                float dot = 0.0f;
                for (int d = 0; d < D_h; ++d) {
                    dot += q_head[d] * k_head[d];
                }

                float score = dot * scale;
                scores[bi] = score;
                if (score > m_block) {
                    m_block = score;
                }
            }

            if (m_block > m) {
                float scale_old = (m == -INFINITY) ? 0.0f : ck_expf(m - m_block);
                s *= scale_old;
                for (int d = 0; d < D_h; ++d) {
                    out_head[d] *= scale_old;
                }
                m = m_block;
            }

            for (int bi = 0; bi < blk_len; ++bi) {
                const int t_k = t_k0 + bi;
                const float *v_head = v_base + (size_t)t_k * stride;
                float w = ck_expf(scores[bi] - m);
                s += w;
                for (int d = 0; d < D_h; ++d) {
                    out_head[d] += w * v_head[d];
                }
            }
        }

        if (s > 0.0f) {
            float inv_s = 1.0f / s;
            for (int d = 0; d < D_h; ++d) {
                out_head[d] *= inv_s;
            }
        } else {
            for (int d = 0; d < D_h; ++d) {
                out_head[d] = 0.0f;
            }
        }
    }
}

#if defined(__AVX512F__)

/* ============================================================================
 * AVX-512 IMPLEMENTATION (16 floats per vector)
 * ============================================================================ */

#if CK_FLASH_ATTN_FAST_EXP
static inline __m512 ck_fast_exp512_ps(__m512 x) {
    const __m512 max_val = _mm512_set1_ps(88.0f);
    const __m512 min_val = _mm512_set1_ps(-88.0f);
    x = _mm512_min_ps(x, max_val);
    x = _mm512_max_ps(x, min_val);

    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    __m512 z = _mm512_mul_ps(x, log2e);
    __m512 zf = _mm512_roundscale_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(z, zf);

    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 c2 = _mm512_set1_ps(0.2402265069591007f);
    const __m512 c3 = _mm512_set1_ps(0.05550410866482158f);
    const __m512 c4 = _mm512_set1_ps(0.009618129107628478f);

#if defined(__FMA__)
    __m512 poly = _mm512_fmadd_ps(c4, f, c3);
    poly = _mm512_fmadd_ps(poly, f, c2);
    poly = _mm512_fmadd_ps(poly, f, c1);
    poly = _mm512_fmadd_ps(poly, f, c0);
#else
    __m512 poly = _mm512_add_ps(_mm512_mul_ps(c4, f), c3);
    poly = _mm512_add_ps(_mm512_mul_ps(poly, f), c2);
    poly = _mm512_add_ps(_mm512_mul_ps(poly, f), c1);
    poly = _mm512_add_ps(_mm512_mul_ps(poly, f), c0);
#endif

    __m512i zi = _mm512_cvtps_epi32(zf);
    zi = _mm512_add_epi32(zi, _mm512_set1_epi32(127));
    zi = _mm512_slli_epi32(zi, 23);
    __m512 pow2 = _mm512_castsi512_ps(zi);
    return _mm512_mul_ps(poly, pow2);
}
#endif

static inline float ck_dot_f32_avx512(const float *q, const float *k, int D_h) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();

    int d = 0;
    for (; d + 32 <= D_h; d += 32) {
        __m512 q0 = _mm512_loadu_ps(q + d);
        __m512 k0 = _mm512_loadu_ps(k + d);
        __m512 q1 = _mm512_loadu_ps(q + d + 16);
        __m512 k1 = _mm512_loadu_ps(k + d + 16);
        sum0 = _mm512_fmadd_ps(q0, k0, sum0);
        sum1 = _mm512_fmadd_ps(q1, k1, sum1);
    }
    for (; d + 16 <= D_h; d += 16) {
        __m512 q0 = _mm512_loadu_ps(q + d);
        __m512 k0 = _mm512_loadu_ps(k + d);
        sum0 = _mm512_fmadd_ps(q0, k0, sum0);
    }

    sum0 = _mm512_add_ps(sum0, sum1);
    float dot = _mm512_reduce_add_ps(sum0);
    for (; d < D_h; ++d) {
        dot += q[d] * k[d];
    }
    return dot;
}

/**
 * @brief AVX-512 optimized flash attention decode
 */
static void attention_flash_decode_avx512(
    float *out,
    const float *q,
    const float *k,
    const float *v,
    int T_q,
    int T_k,
    int H,
    int D_h,
    float scale)
{
    const int total = T_q * H;
    const size_t stride = (size_t)H * (size_t)D_h;
    const int tile_k = ck_flash_attn_tile_k(D_h);

    for (int idx = 0; idx < total; ++idx) {
        const int t_q = idx / H;
        const int h = idx - t_q * H;
        const int max_k = max_k_for_query(t_q, T_q, T_k);

        const float *q_head = q + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
        float *out_head = out + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
        const float *k_base = k + (size_t)h * (size_t)D_h;
        const float *v_base = v + (size_t)h * (size_t)D_h;

        int d = 0;
        for (; d + 16 <= D_h; d += 16) {
            _mm512_storeu_ps(out_head + d, _mm512_setzero_ps());
        }
        for (; d < D_h; ++d) {
            out_head[d] = 0.0f;
        }

        float m = -INFINITY;
        float s = 0.0f;

        float scores[CK_FLASH_ATTN_TILE_K];

        for (int t_k0 = 0; t_k0 <= max_k; t_k0 += tile_k) {
            int blk_len = max_k - t_k0 + 1;
            if (blk_len > tile_k) {
                blk_len = tile_k;
            }

            float m_block = -INFINITY;
            for (int bi = 0; bi < blk_len; ++bi) {
                const int t_k = t_k0 + bi;
                const float *k_head = k_base + (size_t)t_k * stride;

                float dot = ck_dot_f32_avx512(q_head, k_head, D_h);

                float score = dot * scale;
                scores[bi] = score;
                if (score > m_block) {
                    m_block = score;
                }
            }

            if (m_block > m) {
                float scale_old = (m == -INFINITY) ? 0.0f : ck_expf(m - m_block);
                s *= scale_old;
                __m512 scale_old_vec = _mm512_set1_ps(scale_old);
                d = 0;
                for (; d + 16 <= D_h; d += 16) {
                    __m512 out_v = _mm512_loadu_ps(out_head + d);
                    _mm512_storeu_ps(out_head + d, _mm512_mul_ps(out_v, scale_old_vec));
                }
                for (; d < D_h; ++d) {
                    out_head[d] *= scale_old;
                }
                m = m_block;
            }

#if CK_FLASH_ATTN_FAST_EXP
            int bi_vec = 0;
            __m512 m_vec = _mm512_set1_ps(m);
            for (; bi_vec + 16 <= blk_len; bi_vec += 16) {
                __m512 s_vec = _mm512_loadu_ps(scores + bi_vec);
                s_vec = _mm512_sub_ps(s_vec, m_vec);
                __m512 w_vec = ck_fast_exp512_ps(s_vec);
                _mm512_storeu_ps(scores + bi_vec, w_vec);
            }
            for (; bi_vec < blk_len; ++bi_vec) {
                scores[bi_vec] = ck_fast_expf(scores[bi_vec] - m);
            }
#endif

            for (int bi = 0; bi < blk_len; ++bi) {
                const int t_k = t_k0 + bi;
                const float *v_head = v_base + (size_t)t_k * stride;
#if CK_FLASH_ATTN_FAST_EXP
                float w = scores[bi];
#else
                float w = ck_expf(scores[bi] - m);
#endif
                s += w;

                __m512 w_vec = _mm512_set1_ps(w);
                d = 0;
                for (; d + 16 <= D_h; d += 16) {
                    __m512 out_v = _mm512_loadu_ps(out_head + d);
                    __m512 v_v = _mm512_loadu_ps(v_head + d);
                    out_v = _mm512_fmadd_ps(w_vec, v_v, out_v);
                    _mm512_storeu_ps(out_head + d, out_v);
                }
                for (; d < D_h; ++d) {
                    out_head[d] += w * v_head[d];
                }
            }
        }

        if (s > 0.0f) {
            float inv_s = 1.0f / s;
            __m512 inv_s_vec = _mm512_set1_ps(inv_s);
            d = 0;
            for (; d + 16 <= D_h; d += 16) {
                __m512 out_v = _mm512_loadu_ps(out_head + d);
                _mm512_storeu_ps(out_head + d, _mm512_mul_ps(out_v, inv_s_vec));
            }
            for (; d < D_h; ++d) {
                out_head[d] *= inv_s;
            }
        } else {
            for (int d0 = 0; d0 < D_h; ++d0) {
                out_head[d0] = 0.0f;
            }
        }
    }
}

#endif  // __AVX512F__

#if defined(__AVX__) && !defined(__AVX512F__)

/* ============================================================================
 * AVX IMPLEMENTATION (8 floats per vector)
 * ============================================================================ */

#if CK_FLASH_ATTN_FAST_EXP
static inline __m256 ck_pow2_256_ps(__m256 zf) {
    __m128 z0 = _mm256_castps256_ps128(zf);
    __m128 z1 = _mm256_extractf128_ps(zf, 1);

    __m128i i0 = _mm_cvtps_epi32(z0);
    __m128i i1 = _mm_cvtps_epi32(z1);
    i0 = _mm_add_epi32(i0, _mm_set1_epi32(127));
    i1 = _mm_add_epi32(i1, _mm_set1_epi32(127));
    i0 = _mm_slli_epi32(i0, 23);
    i1 = _mm_slli_epi32(i1, 23);

    __m128 f0 = _mm_castsi128_ps(i0);
    __m128 f1 = _mm_castsi128_ps(i1);
    __m256 out = _mm256_castps128_ps256(f0);
    return _mm256_insertf128_ps(out, f1, 1);
}

static inline __m256 ck_fast_exp256_ps(__m256 x) {
    const __m256 max_val = _mm256_set1_ps(88.0f);
    const __m256 min_val = _mm256_set1_ps(-88.0f);
    x = _mm256_min_ps(x, max_val);
    x = _mm256_max_ps(x, min_val);

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 z = _mm256_mul_ps(x, log2e);
    __m256 zf = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(z, zf);

    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628478f);

#if defined(__FMA__)
    __m256 poly = _mm256_fmadd_ps(c4, f, c3);
    poly = _mm256_fmadd_ps(poly, f, c2);
    poly = _mm256_fmadd_ps(poly, f, c1);
    poly = _mm256_fmadd_ps(poly, f, c0);
#else
    __m256 poly = _mm256_add_ps(_mm256_mul_ps(c4, f), c3);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c2);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c1);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c0);
#endif

    __m256 pow2 = ck_pow2_256_ps(zf);
    return _mm256_mul_ps(poly, pow2);
}
#endif

static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(shuf, sums);
    return _mm_cvtss_f32(sums);
}

static inline float ck_dot_f32_avx(const float *q, const float *k, int D_h) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int d = 0;
    for (; d + 16 <= D_h; d += 16) {
        __m256 q0 = _mm256_loadu_ps(q + d);
        __m256 k0 = _mm256_loadu_ps(k + d);
        __m256 q1 = _mm256_loadu_ps(q + d + 8);
        __m256 k1 = _mm256_loadu_ps(k + d + 8);
    #if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(q0, k0, sum0);
        sum1 = _mm256_fmadd_ps(q1, k1, sum1);
    #else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q0, k0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(q1, k1));
    #endif
    }
    for (; d + 8 <= D_h; d += 8) {
        __m256 q0 = _mm256_loadu_ps(q + d);
        __m256 k0 = _mm256_loadu_ps(k + d);
    #if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(q0, k0, sum0);
    #else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q0, k0));
    #endif
    }

    __m256 sum = _mm256_add_ps(sum0, sum1);
    float dot = hsum256_ps(sum);
    for (; d < D_h; ++d) {
        dot += q[d] * k[d];
    }
    return dot;
}

static void attention_flash_decode_avx(
    float *out,
    const float *q,
    const float *k,
    const float *v,
    int T_q,
    int T_k,
    int H,
    int D_h,
    float scale)
{
    const int total = T_q * H;
    const size_t stride = (size_t)H * (size_t)D_h;
    const int tile_k = ck_flash_attn_tile_k(D_h);

    for (int idx = 0; idx < total; ++idx) {
        const int t_q = idx / H;
        const int h = idx - t_q * H;
        const int max_k = max_k_for_query(t_q, T_q, T_k);

        const float *q_head = q + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
        float *out_head = out + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
        const float *k_base = k + (size_t)h * (size_t)D_h;
        const float *v_base = v + (size_t)h * (size_t)D_h;

        int d = 0;
        for (; d + 8 <= D_h; d += 8) {
            _mm256_storeu_ps(out_head + d, _mm256_setzero_ps());
        }
        for (; d < D_h; ++d) {
            out_head[d] = 0.0f;
        }

        float m = -INFINITY;
        float s = 0.0f;

        float scores[CK_FLASH_ATTN_TILE_K];

        for (int t_k0 = 0; t_k0 <= max_k; t_k0 += tile_k) {
            int blk_len = max_k - t_k0 + 1;
            if (blk_len > tile_k) {
                blk_len = tile_k;
            }

            float m_block = -INFINITY;
            for (int bi = 0; bi < blk_len; ++bi) {
                const int t_k = t_k0 + bi;
                const float *k_head = k_base + (size_t)t_k * stride;

                float dot = ck_dot_f32_avx(q_head, k_head, D_h);

                float score = dot * scale;
                scores[bi] = score;
                if (score > m_block) {
                    m_block = score;
                }
            }

            if (m_block > m) {
                float scale_old = (m == -INFINITY) ? 0.0f : ck_expf(m - m_block);
                s *= scale_old;
                __m256 scale_old_vec = _mm256_set1_ps(scale_old);
                d = 0;
                for (; d + 8 <= D_h; d += 8) {
                    __m256 out_v = _mm256_loadu_ps(out_head + d);
                    _mm256_storeu_ps(out_head + d, _mm256_mul_ps(out_v, scale_old_vec));
                }
                for (; d < D_h; ++d) {
                    out_head[d] *= scale_old;
                }
                m = m_block;
            }

#if CK_FLASH_ATTN_FAST_EXP
            int bi_vec = 0;
            __m256 m_vec = _mm256_set1_ps(m);
            for (; bi_vec + 8 <= blk_len; bi_vec += 8) {
                __m256 s_vec = _mm256_loadu_ps(scores + bi_vec);
                s_vec = _mm256_sub_ps(s_vec, m_vec);
                __m256 w_vec = ck_fast_exp256_ps(s_vec);
                _mm256_storeu_ps(scores + bi_vec, w_vec);
            }
            for (; bi_vec < blk_len; ++bi_vec) {
                scores[bi_vec] = ck_fast_expf(scores[bi_vec] - m);
            }
#endif

            for (int bi = 0; bi < blk_len; ++bi) {
                const int t_k = t_k0 + bi;
                const float *v_head = v_base + (size_t)t_k * stride;
#if CK_FLASH_ATTN_FAST_EXP
                float w = scores[bi];
#else
                float w = ck_expf(scores[bi] - m);
#endif
                s += w;

                __m256 w_vec = _mm256_set1_ps(w);
                d = 0;
                for (; d + 8 <= D_h; d += 8) {
                    __m256 out_v = _mm256_loadu_ps(out_head + d);
                    __m256 v_v = _mm256_loadu_ps(v_head + d);
                #if defined(__FMA__)
                    out_v = _mm256_fmadd_ps(w_vec, v_v, out_v);
                #else
                    out_v = _mm256_add_ps(out_v, _mm256_mul_ps(w_vec, v_v));
                #endif
                    _mm256_storeu_ps(out_head + d, out_v);
                }
                for (; d < D_h; ++d) {
                    out_head[d] += w * v_head[d];
                }
            }
        }

        if (s > 0.0f) {
            float inv_s = 1.0f / s;
            __m256 inv_s_vec = _mm256_set1_ps(inv_s);
            d = 0;
            for (; d + 8 <= D_h; d += 8) {
                __m256 out_v = _mm256_loadu_ps(out_head + d);
                _mm256_storeu_ps(out_head + d, _mm256_mul_ps(out_v, inv_s_vec));
            }
            for (; d < D_h; ++d) {
                out_head[d] *= inv_s;
            }
        } else {
            for (int d0 = 0; d0 < D_h; ++d0) {
                out_head[d0] = 0.0f;
            }
        }
    }
}

#endif  // __AVX__

/* ============================================================================
 * DISPATCHER FUNCTION
 * ============================================================================ */

/**
 * @brief Main flash attention function with SIMD dispatch
 *
 * @param out Output [T_q, H, D_h]
 * @param q Query [T_q, H, D_h]
 * @param k Key [T_k, H, D_h]
 * @param v Value [T_k, H, D_h]
 * @param T_q Number of query tokens (1 for decode)
 * @param T_k Number of key/value tokens (context length)
 * @param H Number of heads
 * @param D_h Head dimension
 * @param scale 1/sqrt(D_h)
 */
void attention_flash_decode(
    float *out,
    const float *q,
    const float *k,
    const float *v,
    int T_q,
    int T_k,
    int H,
    int D_h,
    float scale)
{
    if (!out || !q || !k || !v) {
        return;
    }
    if (T_q <= 0 || T_k <= 0 || H <= 0 || D_h <= 0) {
        return;
    }

    // Dispatch based on CPU features
#if defined(__AVX512F__)
    attention_flash_decode_avx512(out, q, k, v, T_q, T_k, H, D_h, scale);
#elif defined(__AVX__) && !defined(__AVX512F__)
    attention_flash_decode_avx(out, q, k, v, T_q, T_k, H, D_h, scale);
#else
    attention_flash_decode_scalar(out, q, k, v, T_q, T_k, H, D_h, scale);
#endif
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * @brief Initialize flash attention buffers
 */
void attention_flash_init(int max_context, int max_heads, int max_head_dim) {
    // For future optimization: pre-allocate scratch buffers
    // Currently using stack/heap allocation
}

/**
 * @brief Clean up flash attention resources
 */
void attention_flash_cleanup(void) {
    // For future optimization: free pre-allocated buffers
}
