/**
 * @file attention_kernels.c
 * @brief Attention score/softmax/output kernels with SIMD (SSE/AVX/AVX512)
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
 * Attention: softmax(Q @ K^T / sqrt(d)) @ V
 * Supports GQA (grouped-query attention) with head broadcasting.
 */

#include "bf16_utils.h"
#include "ckernel_engine.h"
#include <math.h>
#include <stdlib.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Convert BF16 tensor to FP32 using caller-provided buffer (no malloc!) */
static void convert_bf16_tensor_to_buf(const uint16_t *src, float *dst, size_t count)
{
    if (!dst || !src) return;
    bf16_tensor_to_float(src, dst, count);
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

// Match llama.cpp flash-attention input handling where F32 K/V are rounded through F16.
static inline float ck_round_fp16_scalar(float x) {
    return CK_FP16_TO_FP32(CK_FP32_TO_FP16(x));
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

/**
 * Causal attention forward (score-matrix version)
 * @test test_attention.py::TestAttentionForward::test_causal_forward
 * @test test_attention.py::TestAttentionForward::test_gqa_broadcast
 * @test test_attention.py::TestAttentionForward::test_exact_vs_fast
 * @test test_parity.py::test_attention_parity
 *
 * Computes softmax(Q @ K^T / sqrt(d)) @ V with causal masking.
 * Uses O(N^2) memory for scores matrix.
 *
 * After changes: make test && make llamacpp-parity-full
 */
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

/**
 * Causal attention forward (exact version using stdlib expf)
 * @test test_attention.py::TestAttentionForward::test_exact_single
 * @test test_attention.py::TestAttentionForward::test_exact_vs_fast
 *
 * Uses standard library expf for numerical accuracy reference.
 * Slower but provides maximum accuracy.
 *
 * After changes: make test
 */
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

/**
 * GQA causal attention forward (score-matrix version)
 * @test test_attention.py::TestAttentionForward::test_gqa_forward
 * @test test_attention.py::TestAttentionForward::test_gqa_broadcast
 * @test test_attention_backward.py::TestAttentionBackwardGQA::test_gqa_backward
 * @test test_parity.py::test_attention_gqa_parity
 *
 * Grouped-query attention: Q has num_heads, K/V have num_kv_heads.
 * Each query head maps to a KV head via ratio.
 *
 * After changes: make test && make llamacpp-parity-full
 */
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

/**
 * GQA causal attention forward (exact version using stdlib expf)
 * @test test_attention.py::TestAttentionForward::test_gqa_exact
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_gqa
 *
 * Uses standard library expf for numerical accuracy reference.
 * Used by BF16 wrapper to avoid approximation error accumulation.
 *
 * After changes: make test
 */
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

/**
 * BF16 GQA causal attention forward
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_forward
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_gqa
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_flash
 *
 * Accepts BF16 inputs, converts to FP32, uses exact softmax.
 * Caller provides scratch buffers (no per-call malloc).
 *
 * After changes: make test
 */
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
                                                  int aligned_context_window,
                                                  float *scratch_q,
                                                  float *scratch_k,
                                                  float *scratch_v)
{
    const size_t q_elems = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_elems = (size_t)num_kv_heads * (size_t)num_tokens * (size_t)aligned_head_dim;

    if (!scratch_q || !scratch_k || !scratch_v) return;

    convert_bf16_tensor_to_buf(q, scratch_q, q_elems);
    convert_bf16_tensor_to_buf(k, scratch_k, kv_elems);
    convert_bf16_tensor_to_buf(v, scratch_v, kv_elems);

    // Use exact version to avoid fast exp approximation error accumulating
    // with BF16 precision loss.
    attention_forward_causal_head_major_gqa_exact(scratch_q, scratch_k, scratch_v,
                                                   scores, output,
                                                   num_heads, num_kv_heads,
                                                   num_tokens, head_dim,
                                                   aligned_head_dim, aligned_context_window);
    /* No free - caller owns scratch buffers */
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

// Strict parity reference for flash-style query path.
// Uses a two-pass exact softmax formulation per query:
// 1) max(score), 2) exp(score-max) accumulation for sum and weighted V.
// This avoids online-softmax re-normalization drift in long reductions.
static void attention_flash_query_causal_exact(const float *q_vec,
                                               const float *k_head,
                                               const float *v_head,
                                               int kv_tokens,
                                               int head_dim,
                                               int aligned_head_dim,
                                               float scale,
                                               float *out_vec)
{
    if (kv_tokens <= 0) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int d = 0; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    float max_score = -INFINITY;
    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;
        float w = expf(score - max_score);
        sum += w;
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] += w * v_vec[d];
        }
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] *= inv_sum;
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }
    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}

// Llama-parity attention reference: K/V are rounded through F16 before use.
static void attention_flash_query_causal_exact_f16kv(const float *q_vec,
                                                     const float *k_head,
                                                     const float *v_head,
                                                     int kv_tokens,
                                                     int head_dim,
                                                     int aligned_head_dim,
                                                     float scale,
                                                     float *out_vec)
{
    if (kv_tokens <= 0) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int d = 0; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    // Mirror llama.cpp GGML flash-attention more closely:
    // - Q is converted through FP16 before the KQ dot
    // - V accumulation is rounded through FP16 at each update
    // - the softmax accumulator uses the online max/sum form
    float sum = 0.0f;
    float max_score = -INFINITY;
    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += ck_round_fp16_scalar(q_vec[d]) * ck_round_fp16_scalar(k_vec[d]);
        }
        float score = dot * scale;

        const float prev_max = max_score;
        float max_scale = 1.0f;
        float value_scale = 1.0f;

        if (score > max_score) {
            max_score = score;
            max_scale = isfinite(prev_max) ? expf(prev_max - max_score) : 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] = ck_round_fp16_scalar(out_vec[d] * max_scale);
            }
        } else {
            value_scale = expf(score - max_score);
        }

        for (int d = 0; d < head_dim; ++d) {
            const float v_rounded = ck_round_fp16_scalar(v_vec[d]);
            const float updated = out_vec[d] + value_scale * v_rounded;
            out_vec[d] = ck_round_fp16_scalar(updated);
        }

        sum = sum * max_scale + value_scale;
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] *= inv_sum;
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }
    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}

/**
 * Flash attention forward for GQA (prefill, no score materialization)
 * @test test_flash_attention.py::TestFlashAttention::test_flash_forward
 * @test test_flash_attention.py::TestFlashAttention::test_flash_vs_score_matrix
 * @test test_flash_attention.py::TestFlashAttention::test_flash_gqa
 * @test test_attention.py::TestAttentionForward::test_flash_forward
 *
 * Online softmax with streaming KV. O(N) memory instead of O(N^2).
 * For prefill: all tokens attend to previous tokens.
 *
 * After changes: make test && make llamacpp-parity-full
 */
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

    if (ck_strict_parity_enabled()) {
        for (int h = 0; h < num_heads; ++h) {
            int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *k_head = k + (size_t)kv_head * (size_t)T * (size_t)aligned_head_dim;
            const float *v_head = v + (size_t)kv_head * (size_t)T * (size_t)aligned_head_dim;

            for (int i = 0; i < T; ++i) {
                const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
                float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
                attention_flash_query_causal_exact(q_vec, k_head, v_head,
                                                   /*kv_tokens=*/i + 1,
                                                   head_dim, aligned_head_dim,
                                                   scale, out_vec);
            }
        }
        return;
    }

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

/**
 * Flash attention forward with custom KV stride (for KV cache)
 * @test test_flash_attention.py::TestFlashAttention::test_flash_strided
 * @test test_kv_cache_attention.py::TestKVCacheAttention::test_flash_attention
 *
 * Variant with configurable kv_stride_tokens for KV cache layouts
 * where K/V may not be contiguous in memory.
 *
 * After changes: make test
 */
void attention_forward_causal_head_major_gqa_flash_strided(const float *q,
                                                           const float *k,
                                                           const float *v,
                                                           float *output,
                                                           int num_heads,
                                                           int num_kv_heads,
                                                           int num_tokens,
                                                           int head_dim,
                                                           int aligned_head_dim,
                                                           int kv_stride_tokens)
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

    if (ck_strict_parity_enabled()) {
        for (int h = 0; h < num_heads; ++h) {
            int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *k_head = k + (size_t)kv_head * kv_head_stride;
            const float *v_head = v + (size_t)kv_head * kv_head_stride;

            for (int i = 0; i < T; ++i) {
                const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
                float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
                attention_flash_query_causal_exact(q_vec, k_head, v_head,
                                                   /*kv_tokens=*/i + 1,
                                                   head_dim, aligned_head_dim,
                                                   scale, out_vec);
            }
        }
        return;
    }

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
        const float *k_head = k + (size_t)kv_head * kv_head_stride;
        const float *v_head = v + (size_t)kv_head * kv_head_stride;

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

void attention_forward_causal_head_major_gqa_flash_strided_f16kv(const float *q,
                                                                 const float *k,
                                                                 const float *v,
                                                                 float *output,
                                                                 int num_heads,
                                                                 int num_kv_heads,
                                                                 int num_tokens,
                                                                 int head_dim,
                                                                 int aligned_head_dim,
                                                                 int kv_stride_tokens)
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

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *k_head = k + (size_t)kv_head * kv_head_stride;
        const float *v_head = v + (size_t)kv_head * kv_head_stride;

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            attention_flash_query_causal_exact_f16kv(q_vec, k_head, v_head,
                                                     /*kv_tokens=*/i + 1,
                                                     head_dim, aligned_head_dim,
                                                     scale, out_vec);
        }
    }
}

/**
 * Flash attention decode (single token attends to KV cache)
 * @test test_flash_attention.py::TestFlashAttention::test_flash_decode
 * @test test_kv_cache_attention.py::TestKVCacheAttention::test_flash_decode
 * @test test_fused_attention_decode.py::TestFusedAttentionDecode::test_flash_decode
 * @test test_attention.py::TestAttentionForward::test_flash_decode
 *
 * Single query token attends to kv_tokens in KV cache.
 * Uses true flash attention from attention_flash_true.c.
 *
 * After changes: make test && make llamacpp-parity-full
 */
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

void attention_forward_decode_head_major_gqa_flash_f16kv(const float *q_token,
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

        attention_flash_query_causal_exact_f16kv(q_head,
                                                 k_head,
                                                 v_head,
                                                 kv_tokens,
                                                 head_dim,
                                                 aligned_head_dim,
                                                 scale,
                                                 out_head);
    }
}

/**
 * @brief WARNING: This is NOT true flash attention!
 *
 * This function is named "flash" but implements regular attention with O(n) complexity.
 * It's kept for reference and as a fallback.
 *
 * TRUE flash attention is implemented in attention_flash_true.c
 * @test test_kv_cache_attention.py::TestKVCacheAttention::test_regular_decode
 * @test test_attention.py::TestAttentionForward::test_regular_decode
 *
 * Regular attention decode (score-matrix version) for fallback.
 *
 * After changes: make test
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
/**
 * BF16 attention backward with caller-provided scratch buffers
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_backward
 *
 * Accepts BF16 inputs, converts to FP32, runs FP32 backward.
 * Caller provides scratch buffers (no per-call malloc).
 *
 * After changes: make test
 */
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
    int aligned_context_window,
    float *scratch_d_output,
    float *scratch_q,
    float *scratch_k,
    float *scratch_v)
{
    (void)d_x;
    const size_t head_elems = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_elems = (size_t)num_kv_heads * (size_t)num_tokens * (size_t)aligned_head_dim;

    if (!scratch_d_output || !scratch_q || !scratch_k || !scratch_v) return;

    convert_bf16_tensor_to_buf(d_output, scratch_d_output, head_elems);
    convert_bf16_tensor_to_buf(q, scratch_q, head_elems);
    convert_bf16_tensor_to_buf(k, scratch_k, kv_elems);
    convert_bf16_tensor_to_buf(v, scratch_v, kv_elems);

    attention_backward_causal_head_major_gqa(scratch_d_output, scratch_q, scratch_k, scratch_v,
                                             attn_weights,
                                             d_q, d_k, d_v, d_scores,
                                             num_heads, num_kv_heads,
                                             num_tokens, head_dim,
                                             aligned_head_dim, aligned_context_window);
    /* No free - caller owns scratch buffers */
}

/**
 * GQA causal attention backward (score-matrix version)
 * @test test_attention_backward.py::TestAttentionBackwardGQA::test_gqa_backward
 * @test test_attention_backward.py::TestAttentionBackwardGQA::test_gqa_vs_separate
 * @test test_parity.py::test_attention_backward_parity
 *
 * Computes dQ, dK, dV given dOutput and attention weights.
 * Supports grouped-query attention with head broadcasting.
 *
 * After changes: make test && make llamacpp-parity-full
 */
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

/**
 * Causal attention backward (non-GQA version)
 * @test test_attention_backward.py::TestAttentionBackward::test_backward
 * @test test_attention_backward.py::TestAttentionBackward::test_backward_vs_separate
 * @test test_parity.py::test_attention_backward_parity
 *
 * Non-GQA version where num_heads == num_kv_heads.
 * Simpler than GQA, no head broadcasting needed.
 *
 * After changes: make test && make llamacpp-parity-full
 */
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
