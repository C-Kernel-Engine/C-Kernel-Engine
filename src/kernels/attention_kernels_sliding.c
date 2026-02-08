/**
 * @file attention_kernels_sliding.c
 * @brief Sliding-window flash attention kernels split from attention_kernels.c
 */

#include "ckernel_engine.h"
#include <math.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Local head-major index helper: [head][token][dim] with aligned dim stride. */
static inline size_t qkv_index(int h,
                               int t,
                               int d,
                               int num_tokens,
                               int aligned_head_dim)
{
    return ((size_t)h * (size_t)num_tokens + (size_t)t) * (size_t)aligned_head_dim
         + (size_t)d;
}

#if defined(__AVX2__)
static inline float hsum256_ps_flash(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif

#if defined(__AVX__) && !defined(__AVX2__)
static inline float hsum256_ps_flash_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif

// ============================================================================
// SLIDING-WINDOW ATTENTION - Flash-style with sliding window mask
// ============================================================================
//
// Sliding-window attention: each token attends only to the last W tokens.
// For token at position i, the valid key range is [max(0, i - W + 1) .. i].
// This is equivalent to causal attention with a window size limit.
//
// Key difference from regular causal attention:
// - Regular causal: token i attends to [0 .. i] (all previous tokens)
// - Sliding window: token i attends to [max(0, i - W + 1) .. i] (last W tokens only)

// ============================================================================
// AVX-512 Sliding-Window Flash Attention
// ============================================================================
#if defined(__AVX512F__)
static void attention_flash_query_sliding_avx512(const float *q_vec,
                                                  const float *k_head,
                                                  const float *v_head,
                                                  int query_pos,        // Position of query token (0-indexed)
                                                  int kv_tokens,        // Total KV tokens available
                                                  int head_dim,
                                                  int aligned_head_dim,
                                                  float scale,
                                                  float *out_vec,
                                                  int sliding_window)   // Window size (0 = no limit)
{
    float m = -INFINITY;
    float s = 0.0f;

    // Compute sliding window bounds
    int window_start = 0;
    if (sliding_window > 0) {
        window_start = query_pos - sliding_window + 1;
        if (window_start < 0) window_start = 0;
    }

    // Zero output using SIMD
    int d = 0;
    for (; d + 16 <= aligned_head_dim; d += 16) {
        _mm512_storeu_ps(&out_vec[d], _mm512_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    // Process only tokens in the sliding window [window_start .. min(query_pos, kv_tokens-1)]
    int effective_kv_end = query_pos < kv_tokens ? query_pos : kv_tokens - 1;
    for (int j = window_start; j <= effective_kv_end; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        // Vectorized dot product Q·K
        __m512 dot_acc = _mm512_setzero_ps();
        d = 0;
        for (; d + 16 <= head_dim; d += 16) {
            __m512 q_v = _mm512_loadu_ps(&q_vec[d]);
            __m512 k_v = _mm512_loadu_ps(&k_vec[d]);
            dot_acc = _mm512_fmadd_ps(q_v, k_v, dot_acc);
        }
        float dot = _mm512_reduce_add_ps(dot_acc);
        for (; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;

            __m512 exp_m_vec = _mm512_set1_ps(exp_m);
            d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 out_v = _mm512_loadu_ps(&out_vec[d]);
                __m512 v_v = _mm512_loadu_ps(&v_vec[d]);
                out_v = _mm512_fmadd_ps(out_v, exp_m_vec, v_v);
                _mm512_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_m + v_vec[d];
            }

            s += 1.0f;
            m = score;
        } else {
            float e = expf(score - m);
            s += e;

            __m512 e_vec = _mm512_set1_ps(e);
            d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 out_v = _mm512_loadu_ps(&out_vec[d]);
                __m512 v_v = _mm512_loadu_ps(&v_vec[d]);
                out_v = _mm512_fmadd_ps(e_vec, v_v, out_v);
                _mm512_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    // Normalize: out /= s
    float inv_s = 1.0f / s;
    __m512 inv_s_vec = _mm512_set1_ps(inv_s);
    d = 0;
    for (; d + 16 <= head_dim; d += 16) {
        __m512 out_v = _mm512_loadu_ps(&out_vec[d]);
        _mm512_storeu_ps(&out_vec[d], _mm512_mul_ps(out_v, inv_s_vec));
    }
    for (; d < head_dim; ++d) {
        out_vec[d] *= inv_s;
    }

    // Zero padding
    for (d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}
#endif // __AVX512F__

// ============================================================================
// AVX2 Sliding-Window Flash Attention
// ============================================================================
#if defined(__AVX2__)
static void attention_flash_query_sliding_avx2(const float *q_vec,
                                                const float *k_head,
                                                const float *v_head,
                                                int query_pos,
                                                int kv_tokens,
                                                int head_dim,
                                                int aligned_head_dim,
                                                float scale,
                                                float *out_vec,
                                                int sliding_window)
{
    float m = -INFINITY;
    float s = 0.0f;

    int window_start = 0;
    if (sliding_window > 0) {
        window_start = query_pos - sliding_window + 1;
        if (window_start < 0) window_start = 0;
    }

    int d = 0;
    for (; d + 8 <= aligned_head_dim; d += 8) {
        _mm256_storeu_ps(&out_vec[d], _mm256_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    int effective_kv_end = query_pos < kv_tokens ? query_pos : kv_tokens - 1;
    for (int j = window_start; j <= effective_kv_end; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        __m256 dot_acc = _mm256_setzero_ps();
        d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 q_v = _mm256_loadu_ps(&q_vec[d]);
            __m256 k_v = _mm256_loadu_ps(&k_vec[d]);
            dot_acc = _mm256_fmadd_ps(q_v, k_v, dot_acc);
        }
        float dot = hsum256_ps_flash(dot_acc);
        for (; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;

            __m256 exp_m_vec = _mm256_set1_ps(exp_m);
            d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
                __m256 v_v = _mm256_loadu_ps(&v_vec[d]);
                out_v = _mm256_fmadd_ps(out_v, exp_m_vec, v_v);
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_m + v_vec[d];
            }

            s += 1.0f;
            m = score;
        } else {
            float e = expf(score - m);
            s += e;

            __m256 e_vec = _mm256_set1_ps(e);
            d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
                __m256 v_v = _mm256_loadu_ps(&v_vec[d]);
                out_v = _mm256_fmadd_ps(e_vec, v_v, out_v);
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    float inv_s = 1.0f / s;
    __m256 inv_s_vec = _mm256_set1_ps(inv_s);
    d = 0;
    for (; d + 8 <= head_dim; d += 8) {
        __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
        _mm256_storeu_ps(&out_vec[d], _mm256_mul_ps(out_v, inv_s_vec));
    }
    for (; d < head_dim; ++d) {
        out_vec[d] *= inv_s;
    }

    for (d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}
#endif // __AVX2__

// ============================================================================
// AVX Sliding-Window Flash Attention (no FMA)
// ============================================================================
#if defined(__AVX__) && !defined(__AVX2__)
static void attention_flash_query_sliding_avx(const float *q_vec,
                                               const float *k_head,
                                               const float *v_head,
                                               int query_pos,
                                               int kv_tokens,
                                               int head_dim,
                                               int aligned_head_dim,
                                               float scale,
                                               float *out_vec,
                                               int sliding_window)
{
    float m = -INFINITY;
    float s = 0.0f;

    int window_start = 0;
    if (sliding_window > 0) {
        window_start = query_pos - sliding_window + 1;
        if (window_start < 0) window_start = 0;
    }

    int d = 0;
    for (; d + 8 <= aligned_head_dim; d += 8) {
        _mm256_storeu_ps(&out_vec[d], _mm256_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    int effective_kv_end = query_pos < kv_tokens ? query_pos : kv_tokens - 1;
    for (int j = window_start; j <= effective_kv_end; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        __m256 dot_acc = _mm256_setzero_ps();
        d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 q_v = _mm256_loadu_ps(&q_vec[d]);
            __m256 k_v = _mm256_loadu_ps(&k_vec[d]);
            dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(q_v, k_v));
        }
        float dot = hsum256_ps_flash_avx(dot_acc);
        for (; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;

            __m256 exp_m_vec = _mm256_set1_ps(exp_m);
            d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
                __m256 v_v = _mm256_loadu_ps(&v_vec[d]);
                out_v = _mm256_add_ps(_mm256_mul_ps(out_v, exp_m_vec), v_v);
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_m + v_vec[d];
            }

            s += 1.0f;
            m = score;
        } else {
            float e = expf(score - m);
            s += e;

            __m256 e_vec = _mm256_set1_ps(e);
            d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
                __m256 v_v = _mm256_loadu_ps(&v_vec[d]);
                out_v = _mm256_add_ps(out_v, _mm256_mul_ps(e_vec, v_v));
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    float inv_s = 1.0f / s;
    __m256 inv_s_vec = _mm256_set1_ps(inv_s);
    d = 0;
    for (; d + 8 <= head_dim; d += 8) {
        __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
        _mm256_storeu_ps(&out_vec[d], _mm256_mul_ps(out_v, inv_s_vec));
    }
    for (; d < head_dim; ++d) {
        out_vec[d] *= inv_s;
    }

    for (d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}
#endif // __AVX__ && !__AVX2__

// ============================================================================
// Scalar Sliding-Window Flash Attention Fallback
// ============================================================================
static void attention_flash_query_sliding(const float *q_vec,
                                           const float *k_head,
                                           const float *v_head,
                                           int query_pos,
                                           int kv_tokens,
                                           int head_dim,
                                           int aligned_head_dim,
                                           float scale,
                                           float *out_vec,
                                           int sliding_window)
{
    float m = -INFINITY;
    float s = 0.0f;

    int window_start = 0;
    if (sliding_window > 0) {
        window_start = query_pos - sliding_window + 1;
        if (window_start < 0) window_start = 0;
    }

    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    int effective_kv_end = query_pos < kv_tokens ? query_pos : kv_tokens - 1;
    for (int j = window_start; j <= effective_kv_end; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] *= exp_m;
            }
            s += 1.0f;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] += v_vec[d];
            }
            m = score;
        } else {
            float e = expf(score - m);
            s += e;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    float inv_s = 1.0f / s;
    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] *= inv_s;
    }
    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}

/**
 * Flash attention forward with sliding window (prefill)
 * @test test_attention.py::TestAttentionForward::test_sliding_window_prefill
 *
 * Sliding-window attention for prefill: each token attends to the last W tokens.
 * When sliding_window <= 0, behaves like regular causal attention.
 *
 * After changes: make test
 */
void attention_forward_causal_head_major_gqa_flash_strided_sliding(
    const float *q,
    const float *k,
    const float *v,
    float *output,
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int kv_stride_tokens,
    int sliding_window)
{
    if (!q || !k || !v || !output) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }
    if (kv_stride_tokens < num_tokens) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const int T = num_tokens;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;

#if defined(__AVX512F__)
    #define SLIDING_FLASH_IMPL attention_flash_query_sliding_avx512
#elif defined(__AVX2__)
    #define SLIDING_FLASH_IMPL attention_flash_query_sliding_avx2
#elif defined(__AVX__)
    #define SLIDING_FLASH_IMPL attention_flash_query_sliding_avx
#else
    #define SLIDING_FLASH_IMPL attention_flash_query_sliding
#endif

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *k_head = k + (size_t)kv_head * kv_head_stride;
        const float *v_head = v + (size_t)kv_head * kv_head_stride;

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            SLIDING_FLASH_IMPL(q_vec, k_head, v_head,
                               /*query_pos=*/i,
                               /*kv_tokens=*/T,
                               head_dim, aligned_head_dim,
                               scale, out_vec,
                               sliding_window);
        }
    }

#undef SLIDING_FLASH_IMPL
}

/**
 * Flash attention decode with sliding window
 * @test test_attention.py::TestAttentionForward::test_sliding_window_decode
 *
 * Single query token attends to the last W tokens in the KV cache.
 * For decode: effective_kv_tokens = min(kv_tokens, sliding_window)
 *
 * After changes: make test
 */
void attention_forward_decode_head_major_gqa_flash_sliding(
    const float *q_token,
    const float *k_cache,
    const float *v_cache,
    float *out_token,
    int num_heads,
    int num_kv_heads,
    int kv_tokens,
    int cache_capacity,
    int head_dim,
    int aligned_head_dim,
    int sliding_window)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens <= 0 || kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

    // Compute effective KV tokens based on sliding window
    int effective_kv_tokens = kv_tokens;
    if (sliding_window > 0 && sliding_window < kv_tokens) {
        effective_kv_tokens = sliding_window;
    }

    // Guard against empty window (shouldn't happen with kv_tokens >= 1)
    if (effective_kv_tokens <= 0) {
        return;
    }

    // Offset to start reading from the last effective_kv_tokens entries
    int kv_start_offset = kv_tokens - effective_kv_tokens;

#if defined(__AVX512F__)
    #define SLIDING_DECODE_IMPL attention_flash_query_sliding_avx512
#elif defined(__AVX2__)
    #define SLIDING_DECODE_IMPL attention_flash_query_sliding_avx2
#elif defined(__AVX__)
    #define SLIDING_DECODE_IMPL attention_flash_query_sliding_avx
#else
    #define SLIDING_DECODE_IMPL attention_flash_query_sliding
#endif

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
        // Offset K/V pointer to start from the first token in the sliding window
        const float *k_head = k_cache + (size_t)kv_head * head_stride
                            + (size_t)kv_start_offset * (size_t)aligned_head_dim;
        const float *v_head = v_cache + (size_t)kv_head * head_stride
                            + (size_t)kv_start_offset * (size_t)aligned_head_dim;
        float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;

        // Use query_pos relative to the windowed KV (last token = effective_kv_tokens - 1)
        // sliding_window = 0 since we've already windowed via K/V pointer offset
        SLIDING_DECODE_IMPL(q_head, k_head, v_head,
                            /*query_pos=*/effective_kv_tokens - 1,
                            /*kv_tokens=*/effective_kv_tokens,
                            head_dim, aligned_head_dim,
                            scale, out_head,
                            /*sliding_window=*/0);
    }

#undef SLIDING_DECODE_IMPL
}
