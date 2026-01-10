#include "bf16_utils.h"
#include "ckernel_engine.h"
#include <math.h>
#include <stdlib.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

static float *convert_bf16_tensor(const uint16_t *src, size_t count)
{
    float *dst = (float *)malloc(count * sizeof(float));
    if (!dst) {
        return NULL;
    }
    // Use vectorized conversion when available (AVX-512)
    bf16_tensor_to_float(src, dst, count);
    return dst;
}

// Helpers for head-major layouts used in attention.
// Q/K/V layout: [head][token][head_dim] with stride aligned_head_dim.
static inline size_t qkv_index(int h,
                               int t,
                               int d,
                               int num_tokens,
                               int aligned_head_dim)
{
    return ((size_t)h * (size_t)num_tokens + (size_t)t) * (size_t)aligned_head_dim
         + (size_t)d;
}

// Scores layout matches causal_softmax_head_major:
// [head][query_token][key_token] with stride aligned_context_window.
static inline size_t score_index(int h,
                                 int i,
                                 int j,
                                 int aligned_context_window)
{
    return ((size_t)h * (size_t)aligned_context_window * (size_t)aligned_context_window)
         + (size_t)i * (size_t)aligned_context_window
         + (size_t)j;
}

// Naive, reference-quality scaled dot-product attention with causal mask.
//
// Q, K, V are head-major:
//   q[h, t, d] at q[h * T * aligned_head_dim + t * aligned_head_dim + d]
//   same for k and v.
//
// scores buffer must be at least:
//   num_heads * aligned_context_window * aligned_context_window floats.
//
// output has same layout as Q/V:
//   out[h, t, d] at out[h * T * aligned_head_dim + t * aligned_head_dim + d]
//
// Only the first head_dim elements of each vector participate in the math;
// aligned_head_dim allows callers to pad to cache-friendly widths.
void attention_forward_causal_head_major(const float *q,
                                         const float *k,
                                         const float *v,
                                         float *scores,
                                         float *output,
                                         int num_heads,
                                         int num_tokens,
                                         int head_dim,
                                         int aligned_head_dim,
                                         int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Phase 1: compute scaled dot-product scores Q·K^T / sqrt(d_k),
    // lower triangle only (j <= i).
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            // Ensure upper triangle is zeroed so there are no stale values
            // before the softmax kernel runs.
            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    // Phase 2: apply causal row-wise softmax in-place over j <= i.
    causal_softmax_head_major(scores,
                              num_heads,
                              num_tokens,
                              aligned_context_window);

    // Phase 3: attention weights · V.
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);

            // Zero the full aligned head slice so padded dims stay clean.
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            // Weighted sum over causal positions.
            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

// Exact version using standard library expf for softmax.
// Slower but provides maximum accuracy - used for accuracy testing.
void attention_forward_causal_head_major_exact(const float *q,
                                                const float *k,
                                                const float *v,
                                                float *scores,
                                                float *output,
                                                int num_heads,
                                                int num_tokens,
                                                int head_dim,
                                                int aligned_head_dim,
                                                int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Phase 1: compute scaled dot-product scores Q·K^T / sqrt(d_k),
    // lower triangle only (j <= i).
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            // Ensure upper triangle is zeroed so there are no stale values
            // before the softmax kernel runs.
            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    // Phase 2: apply causal row-wise softmax using exact expf.
    causal_softmax_head_major_exact(scores,
                                     num_heads,
                                     num_tokens,
                                     aligned_context_window);

    // Phase 3: attention weights · V.
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);

            // Zero the full aligned head slice so padded dims stay clean.
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            // Weighted sum over causal positions.
            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

// GQA-aware scaled dot-product attention with causal mask.
// Q has num_heads; K/V have num_kv_heads. Each query head maps to a KV head.
void attention_forward_causal_head_major_gqa(const float *q,
                                             const float *k,
                                             const float *v,
                                             float *scores,
                                             float *output,
                                             int num_heads,
                                             int num_kv_heads,
                                             int num_tokens,
                                             int head_dim,
                                             int aligned_head_dim,
                                             int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    causal_softmax_head_major(scores,
                              num_heads,
                              num_tokens,
                              aligned_context_window);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

// GQA attention forward using exact softmax (standard library expf).
// Slower but provides maximum accuracy. Used by BF16 wrapper to avoid
// approximation error accumulating with BF16 precision loss.
void attention_forward_causal_head_major_gqa_exact(const float *q,
                                                    const float *k,
                                                    const float *v,
                                                    float *scores,
                                                    float *output,
                                                    int num_heads,
                                                    int num_kv_heads,
                                                    int num_tokens,
                                                    int head_dim,
                                                    int aligned_head_dim,
                                                    int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    // Use exact softmax with standard library expf
    causal_softmax_head_major_exact(scores,
                                     num_heads,
                                     num_tokens,
                                     aligned_context_window);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

void attention_forward_causal_head_major_gqa_bf16(const uint16_t *q,
                                                  const uint16_t *k,
                                                  const uint16_t *v,
                                                  float *scores,
                                                  float *output,
                                                  int num_heads,
                                                  int num_kv_heads,
                                                  int num_tokens,
                                                  int head_dim,
                                                  int aligned_head_dim,
                                                  int aligned_context_window)
{
    const size_t q_elems = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_elems = (size_t)num_kv_heads * (size_t)num_tokens * (size_t)aligned_head_dim;

    float *q_float = convert_bf16_tensor(q, q_elems);
    if (!q_float) return;
    float *k_float = convert_bf16_tensor(k, kv_elems);
    if (!k_float) {
        free(q_float);
        return;
    }
    float *v_float = convert_bf16_tensor(v, kv_elems);
    if (!v_float) {
        free(q_float);
        free(k_float);
        return;
    }

    // Use exact version to avoid fast exp approximation error accumulating
    // with BF16 precision loss.
    attention_forward_causal_head_major_gqa_exact(q_float, k_float, v_float,
                                                   scores, output,
                                                   num_heads, num_kv_heads,
                                                   num_tokens, head_dim,
                                                   aligned_head_dim, aligned_context_window);

    free(q_float);
    free(k_float);
    free(v_float);
}

// ============================================================================
// ATTENTION FORWARD - Flash-style (no scores materialization)
// ============================================================================
//
// Computes the same causal attention output as `attention_forward_causal_head_major_gqa`,
// but does not materialize the [H, T, T] score/weight matrices. This is useful for:
//   - Prefill: avoids large scratch buffers and improves cache locality
//   - Decode: supports KV-cache attention for a single token
//
// SIMD-optimized implementations for AVX-512, AVX2, and AVX follow.

// ============================================================================
// AVX-512 SIMD Flash Attention (16 floats per vector)
// ============================================================================
#if defined(__AVX512F__)
static void attention_flash_query_causal_avx512(const float *q_vec,
                                                 const float *k_head,
                                                 const float *v_head,
                                                 int kv_tokens,
                                                 int head_dim,
                                                 int aligned_head_dim,
                                                 float scale,
                                                 float *out_vec)
{
    // Online softmax: m = running max, s = running sum(exp(score - m))
    float m = -INFINITY;
    float s = 0.0f;

    // Zero output using SIMD
    int d = 0;
    for (; d + 16 <= aligned_head_dim; d += 16) {
        _mm512_storeu_ps(&out_vec[d], _mm512_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    for (int j = 0; j < kv_tokens; ++j) {
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
        // Scalar tail
        for (; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;

            // Vectorized: out *= exp_m, then out += v
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

            // Vectorized: out += e * v
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
// AVX2 SIMD Flash Attention (8 floats per vector)
// ============================================================================
#if defined(__AVX2__)
static inline float hsum256_ps_flash(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static void attention_flash_query_causal_avx2(const float *q_vec,
                                               const float *k_head,
                                               const float *v_head,
                                               int kv_tokens,
                                               int head_dim,
                                               int aligned_head_dim,
                                               float scale,
                                               float *out_vec)
{
    float m = -INFINITY;
    float s = 0.0f;

    // Zero output using SIMD
    int d = 0;
    for (; d + 8 <= aligned_head_dim; d += 8) {
        _mm256_storeu_ps(&out_vec[d], _mm256_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        // Vectorized dot product Q·K
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

    // Normalize
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
// AVX SIMD Flash Attention (8 floats per vector, no FMA)
// ============================================================================
#if defined(__AVX__) && !defined(__AVX2__)
static inline float hsum256_ps_flash_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static void attention_flash_query_causal_avx(const float *q_vec,
                                              const float *k_head,
                                              const float *v_head,
                                              int kv_tokens,
                                              int head_dim,
                                              int aligned_head_dim,
                                              float scale,
                                              float *out_vec)
{
    float m = -INFINITY;
    float s = 0.0f;

    // Zero output using SIMD
    int d = 0;
    for (; d + 8 <= aligned_head_dim; d += 8) {
        _mm256_storeu_ps(&out_vec[d], _mm256_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        // Vectorized dot product Q·K (no FMA, use mul + add)
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
                // out = out * exp_m + v (no FMA)
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
                // out = out + e * v (no FMA)
                out_v = _mm256_add_ps(out_v, _mm256_mul_ps(e_vec, v_v));
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    // Normalize
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
// Scalar fallback (original implementation)
// ============================================================================
static void attention_flash_query_causal(const float *q_vec,
                                        const float *k_head,
                                        const float *v_head,
                                        int kv_tokens,
                                        int head_dim,
                                        int aligned_head_dim,
                                        float scale,
                                        float *out_vec)
{
    // Online softmax:
    //   m = running max, s = running sum(exp(score - m))
    //   out = sum(exp(score - m) * v)
    float m = -INFINITY;
    float s = 0.0f;

    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    for (int j = 0; j < kv_tokens; ++j) {
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

void attention_forward_causal_head_major_gqa_flash(const float *q,
                                                   const float *k,
                                                   const float *v,
                                                   float *output,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int num_tokens,
                                                   int head_dim,
                                                   int aligned_head_dim)
{
    if (!q || !k || !v || !output) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const int T = num_tokens;

    // Select SIMD implementation based on compile-time CPU features
#if defined(__AVX512F__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx512
#elif defined(__AVX2__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx2
#elif defined(__AVX__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx
#else
    #define FLASH_QUERY_IMPL attention_flash_query_causal
#endif

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *k_head = k + (size_t)kv_head * (size_t)T * (size_t)aligned_head_dim;
        const float *v_head = v + (size_t)kv_head * (size_t)T * (size_t)aligned_head_dim;

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            FLASH_QUERY_IMPL(q_vec, k_head, v_head,
                             /*kv_tokens=*/i + 1,
                             head_dim, aligned_head_dim,
                             scale, out_vec);
        }
    }

#undef FLASH_QUERY_IMPL
}

// Compatibility wrapper for legacy decode symbol; uses true flash decode.
void attention_forward_decode_head_major_gqa_flash(const float *q_token,
                                                   const float *k_cache,
                                                   const float *v_cache,
                                                   float *out_token,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int kv_tokens,
                                                   int cache_capacity,
                                                   int head_dim,
                                                   int aligned_head_dim)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
        const float *k_head = k_cache + (size_t)kv_head * head_stride;
        const float *v_head = v_cache + (size_t)kv_head * head_stride;
        float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;

        attention_flash_decode(out_head,
                               q_head,
                               k_head,
                               v_head,
                               1,
                               kv_tokens,
                               1,
                               aligned_head_dim,
                               scale);
    }
}

/**
 * @brief WARNING: This is NOT true flash attention!
 *
 * This function is named "flash" but implements regular attention with O(n) complexity.
 * It's kept for reference and as a fallback.
 *
 * TRUE flash attention is implemented in attention_flash_true.c
 */
void attention_forward_decode_head_major_gqa_regular(const float *q_token,
                                                   const float *k_cache,
                                                   const float *v_cache,
                                                   float *out_token,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int kv_tokens,
                                                   int cache_capacity,
                                                   int head_dim,
                                                   int aligned_head_dim)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

    // Select SIMD implementation based on compile-time CPU features
#if defined(__AVX512F__)
    #define FLASH_QUERY_IMPL_DECODE attention_flash_query_causal_avx512
#elif defined(__AVX2__)
    #define FLASH_QUERY_IMPL_DECODE attention_flash_query_causal_avx2
#elif defined(__AVX__)
    #define FLASH_QUERY_IMPL_DECODE attention_flash_query_causal_avx
#else
    #define FLASH_QUERY_IMPL_DECODE attention_flash_query_causal
#endif

#pragma omp parallel for schedule(static) if(num_heads > 1)
    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_vec = q_token + (size_t)h * (size_t)aligned_head_dim;
        const float *k_head = k_cache + (size_t)kv_head * head_stride;
        const float *v_head = v_cache + (size_t)kv_head * head_stride;
        float *out_vec = out_token + (size_t)h * (size_t)aligned_head_dim;

        FLASH_QUERY_IMPL_DECODE(q_vec, k_head, v_head,
                                 kv_tokens, head_dim, aligned_head_dim,
                                 scale, out_vec);
    }

#undef FLASH_QUERY_IMPL_DECODE
}

// ============================================================================
// ATTENTION BACKWARD - Causal, Head-Major, GQA-aware
// ============================================================================
//
// Backward pass for scaled dot-product attention with causal mask.
//
// Given:
//   d_output: gradient from the layer above [num_heads, T, head_dim]
//   q, k, v: saved activations from forward pass
//   attn_weights: saved softmax output from forward [num_heads, T, T]
//
// Computes:
//   d_q: gradient w.r.t. queries  [num_heads, T, head_dim]
//   d_k: gradient w.r.t. keys     [num_kv_heads, T, head_dim]
//   d_v: gradient w.r.t. values   [num_kv_heads, T, head_dim]
//
// Math derivation:
//   Forward: scores = Q @ K^T / sqrt(d)
//            weights = causal_softmax(scores)
//            output = weights @ V
//
//   Backward through V multiply:
//     d_weights = d_output @ V^T           [H, T, T]
//     d_v = weights^T @ d_output           [H_kv, T, d]
//
//   Backward through softmax:
//     d_scores = softmax_backward(d_weights, weights)
//
//   Backward through Q @ K^T:
//     d_q = d_scores @ K / sqrt(d)         [H, T, d]
//     d_k = d_scores^T @ Q / sqrt(d)       [H_kv, T, d]
//
// For GQA: multiple query heads share the same KV head, so we accumulate
// gradients from all query heads that map to each KV head.
//
void attention_backward_causal_head_major_gqa_bf16(
    const uint16_t *d_output,      // [num_heads, T, aligned_head_dim]
    float *d_x,                    // [num_heads, T, aligned_head_dim]
    const uint16_t *q,             // [num_heads, T, aligned_head_dim]
    const uint16_t *k,             // [num_kv_heads, T, aligned_head_dim]
    const uint16_t *v,             // [num_kv_heads, T, aligned_head_dim]
    const float *attn_weights,     // [num_heads, T, aligned_context_window]
    float *d_q,                    // [num_heads, T, aligned_head_dim] output
    float *d_k,                    // [num_kv_heads, T, aligned_head_dim] output
    float *d_v,                    // [num_kv_heads, T, aligned_head_dim] output
    float *d_scores,               // [num_heads, T, aligned_context_window] scratch
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window)
{
    (void)d_x;
    const size_t head_elems = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_elems = (size_t)num_kv_heads * (size_t)num_tokens * (size_t)aligned_head_dim;

    float *d_output_f = convert_bf16_tensor(d_output, head_elems);
    if (!d_output_f) {
        return;
    }
    float *q_f = convert_bf16_tensor(q, head_elems);
    if (!q_f) {
        free(d_output_f);
        return;
    }
    float *k_f = convert_bf16_tensor(k, kv_elems);
    if (!k_f) {
        free(d_output_f);
        free(q_f);
        return;
    }
    float *v_f = convert_bf16_tensor(v, kv_elems);
    if (!v_f) {
        free(d_output_f);
        free(q_f);
        free(k_f);
        return;
    }

    attention_backward_causal_head_major_gqa(d_output_f, q_f, k_f, v_f,
                                             attn_weights,
                                             d_q, d_k, d_v, d_scores,
                                             num_heads, num_kv_heads,
                                             num_tokens, head_dim,
                                             aligned_head_dim, aligned_context_window);

    free(d_output_f);
    free(q_f);
    free(k_f);
    free(v_f);
}

void attention_backward_causal_head_major_gqa(
    const float *d_output,      // [num_heads, T, aligned_head_dim]
    const float *q,             // [num_heads, T, aligned_head_dim]
    const float *k,             // [num_kv_heads, T, aligned_head_dim]
    const float *v,             // [num_kv_heads, T, aligned_head_dim]
    const float *attn_weights,  // [num_heads, T, aligned_context_window]
    float *d_q,                 // [num_heads, T, aligned_head_dim] output
    float *d_k,                 // [num_kv_heads, T, aligned_head_dim] output
    float *d_v,                 // [num_kv_heads, T, aligned_head_dim] output
    float *d_scores,            // [num_heads, T, aligned_context_window] scratch
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);
    int T = num_tokens;
    int H = num_heads;
    int H_kv = num_kv_heads;
    int hd = head_dim;
    int ad = aligned_head_dim;
    int aw = aligned_context_window;

    const size_t d_q_elems = (size_t)H * (size_t)T * (size_t)ad;
    const size_t kv_elems = (size_t)H_kv * (size_t)T * (size_t)ad;
    /* Zero the aligned outputs so padded lanes never leak garbage to downstream GEMMs. */
    for (size_t idx = 0; idx < d_q_elems; ++idx) {
        d_q[idx] = 0.0f;
    }
    for (size_t idx = 0; idx < kv_elems; ++idx) {
        d_k[idx] = 0.0f;
        d_v[idx] = 0.0f;
    }

    // Process each query head
    for (int h = 0; h < H; ++h) {
        // Which KV head does this query head use?
        int kv_h = (int)((long long)h * (long long)H_kv / (long long)H);

        // ----------------------------------------------------------------
        // Step 1: d_weights = d_output @ V^T  and  d_v += weights^T @ d_output
        // ----------------------------------------------------------------
        // For each query position i, compute d_weights[i, j] for j <= i
        // and accumulate d_v[j] contributions

        for (int i = 0; i < T; ++i) {
            size_t d_out_base = qkv_index(h, i, 0, T, ad);

            for (int j = 0; j <= i; ++j) {
                size_t v_base = qkv_index(kv_h, j, 0, T, ad);
                size_t w_idx = score_index(h, i, j, aw);
                float w = attn_weights[w_idx];

                // d_weights[h, i, j] = d_output[h, i, :] @ v[kv_h, j, :]^T
                float dot = 0.0f;
                for (int dd = 0; dd < hd; ++dd) {
                    dot += d_output[d_out_base + dd] * v[v_base + dd];
                }
                d_scores[w_idx] = dot;

                // d_v[kv_h, j, :] += weights[h, i, j] * d_output[h, i, :]
                for (int dd = 0; dd < hd; ++dd) {
                    d_v[v_base + dd] += w * d_output[d_out_base + dd];
                }
            }

            // Zero out upper triangle of d_scores
            for (int j = i + 1; j < T; ++j) {
                d_scores[score_index(h, i, j, aw)] = 0.0f;
            }
            /* Scores scratch uses aligned_context_window, zero the padded columns. */
            for (int j = T; j < aw; ++j) {
                d_scores[score_index(h, i, j, aw)] = 0.0f;
            }
        }

        // ----------------------------------------------------------------
        // Step 2: Backward through softmax (in-place on d_scores for this head)
        // ----------------------------------------------------------------
        // d_scores = softmax_backward(d_scores, attn_weights)
        // Formula: d_score[i,j] = w[i,j] * (d_w[i,j] - sum_k(w[i,k] * d_w[i,k]))

        for (int i = 0; i < T; ++i) {
            int base = h * aw * aw + i * aw;

            // Compute dot product: sum_j w[i,j] * d_w[i,j]
            float dot_product = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float wt = attn_weights[base + j];
                float dw = d_scores[base + j];
                dot_product += wt * dw;
            }

            // Apply softmax backward formula
            for (int j = 0; j <= i; ++j) {
                float wt = attn_weights[base + j];
                float dw = d_scores[base + j];
                d_scores[base + j] = wt * (dw - dot_product);
            }
        }

        // ----------------------------------------------------------------
        // Step 3: d_q = d_scores @ K * scale
        //         d_k += d_scores^T @ Q * scale
        // ----------------------------------------------------------------

        for (int i = 0; i < T; ++i) {
            size_t d_q_base = qkv_index(h, i, 0, T, ad);
            size_t q_base = qkv_index(h, i, 0, T, ad);

            // d_q[h, i, :] = sum_j d_scores[h, i, j] * k[kv_h, j, :] * scale
            // d_k[kv_h, j, :] += d_scores[h, i, j] * q[h, i, :] * scale
            for (int j = 0; j <= i; ++j) {
                size_t k_base = qkv_index(kv_h, j, 0, T, ad);
                size_t d_k_base = qkv_index(kv_h, j, 0, T, ad);
                float ds = d_scores[score_index(h, i, j, aw)] * scale;

                for (int dd = 0; dd < hd; ++dd) {
                    d_q[d_q_base + dd] += ds * k[k_base + dd];
                    d_k[d_k_base + dd] += ds * q[q_base + dd];
                }
            }
        }
    }
}

// Non-GQA version (num_heads == num_kv_heads)
void attention_backward_causal_head_major(
    const float *d_output,
    const float *q,
    const float *k,
    const float *v,
    const float *attn_weights,
    float *d_q,
    float *d_k,
    float *d_v,
    float *d_scores,
    int num_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window)
{
    attention_backward_causal_head_major_gqa(
        d_output, q, k, v, attn_weights,
        d_q, d_k, d_v, d_scores,
        num_heads, num_heads,  // num_kv_heads == num_heads
        num_tokens, head_dim, aligned_head_dim, aligned_context_window);
}
