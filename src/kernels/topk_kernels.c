/**
 * @file topk_kernels.c
 * @brief Top-K selection kernels for MoE router dispatch
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
 * Provides efficient top-K selection from a score vector.
 * Used in Mixture-of-Experts models to select which experts process each token.
 *
 * Operations:
 *   - topk_f32: Find top-K indices and values from N scores
 *   - topk_softmax_f32: Top-K with softmax normalization of selected scores
 */

#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <math.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* =============================================================================
 * Top-K Selection (scalar reference)
 *
 * Finds the K largest values in an array and returns their indices and values.
 * Uses a simple min-heap approach: maintain K best, replace minimum when better found.
 *
 * For small K (typical MoE: K=2-8), this is efficient. O(N*K) complexity.
 * ============================================================================= */

/**
 * @brief Find top-K indices and values from a score vector
 *
 * @param scores Input scores [n]
 * @param n Number of scores (e.g., number of experts)
 * @param k Number of top scores to select
 * @param indices Output: indices of top-K scores [k], sorted descending by value
 * @param values Output: top-K score values [k], sorted descending (can be NULL)
 */
void topk_f32(const float *scores,
              int n,
              int k,
              int *indices,
              float *values)
{
    if (!scores || !indices || n <= 0 || k <= 0) {
        return;
    }

    /* Clamp k to n */
    if (k > n) {
        k = n;
    }

    /* Initialize with first k elements */
    float local_values[k];
    for (int i = 0; i < k; i++) {
        indices[i] = i;
        local_values[i] = scores[i];
    }

    /* Find the minimum in our current top-k */
    int min_idx = 0;
    for (int i = 1; i < k; i++) {
        if (local_values[i] < local_values[min_idx]) {
            min_idx = i;
        }
    }

    /* Scan remaining elements */
    for (int i = k; i < n; i++) {
        if (scores[i] > local_values[min_idx]) {
            /* Replace the minimum */
            indices[min_idx] = i;
            local_values[min_idx] = scores[i];

            /* Find new minimum */
            min_idx = 0;
            for (int j = 1; j < k; j++) {
                if (local_values[j] < local_values[min_idx]) {
                    min_idx = j;
                }
            }
        }
    }

    /* Sort results in descending order (simple insertion sort for small k) */
    for (int i = 1; i < k; i++) {
        float val = local_values[i];
        int idx = indices[i];
        int j = i - 1;
        while (j >= 0 && local_values[j] < val) {
            local_values[j + 1] = local_values[j];
            indices[j + 1] = indices[j];
            j--;
        }
        local_values[j + 1] = val;
        indices[j + 1] = idx;
    }

    /* Copy values if output requested */
    if (values) {
        for (int i = 0; i < k; i++) {
            values[i] = local_values[i];
        }
    }
}

/* =============================================================================
 * Top-K with Softmax Normalization
 *
 * Finds top-K and normalizes the selected scores using softmax.
 * This is the standard MoE gating: select experts, then compute routing weights.
 * ============================================================================= */

/**
 * @brief Find top-K indices with softmax-normalized weights
 *
 * @param scores Input scores [n] (router logits)
 * @param n Number of scores
 * @param k Number of top scores to select
 * @param indices Output: indices of top-K scores [k]
 * @param weights Output: softmax-normalized weights for selected [k], sum to 1.0
 */
void topk_softmax_f32(const float *scores,
                      int n,
                      int k,
                      int *indices,
                      float *weights)
{
    if (!scores || !indices || !weights || n <= 0 || k <= 0) {
        return;
    }

    if (k > n) {
        k = n;
    }

    /* First get top-K indices and values */
    float values[k];
    topk_f32(scores, n, k, indices, values);

    /* Compute softmax over the selected values */
    /* Find max for numerical stability */
    float max_val = values[0];
    for (int i = 1; i < k; i++) {
        if (values[i] > max_val) {
            max_val = values[i];
        }
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        weights[i] = expf(values[i] - max_val);
        sum += weights[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < k; i++) {
        weights[i] *= inv_sum;
    }
}

size_t moe_softmax_topk_router_workspace_bytes(int n_experts)
{
    if (n_experts <= 0) {
        return 0;
    }
    return ((size_t)n_experts * sizeof(float) + 63u) & ~(size_t)63u;
}

#if defined(__AVX2__) && defined(__FMA__)
static inline __m256 ck_moe_ggml_expf256(__m256 x)
{
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(
        _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
        _mm256_set1_ps(126),
        _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(0x1.0e4020p-7f),
                b,
                _mm256_set1_ps(0x1.573e2ep-5f)),
            u,
            _mm256_fmadd_ps(
                _mm256_set1_ps(0x1.555e66p-3f),
                b,
                _mm256_set1_ps(0x1.fffdb6p-2f))),
        u,
        _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
        return _mm256_fmadd_ps(j, k, k);
    }
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(
            n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000u));
    const __m256 s1 = _mm256_castsi256_ps(
        _mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
        _mm256_set1_ps(192),
        _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_fmadd_ps(k, j, k)))));
}
#endif

static double ck_moe_llama_softmax_row(float *probabilities,
                                        const float *logits,
                                        int n_experts,
                                        float max_value)
{
    double sum = 0.0;
    int expert = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; expert + 7 < n_experts; expert += 8) {
        const __m256 value = ck_moe_ggml_expf256(_mm256_sub_ps(
            _mm256_loadu_ps(logits + expert), _mm256_set1_ps(max_value)));
        _mm256_storeu_ps(probabilities + expert, value);
        __m128 half = _mm_add_ps(
            _mm256_extractf128_ps(value, 1), _mm256_castps256_ps128(value));
        half = _mm_add_ps(half, _mm_movehl_ps(half, half));
        half = _mm_add_ss(half, _mm_movehdup_ps(half));
        sum += (double)_mm_cvtss_f32(half);
    }
#endif
    for (; expert < n_experts; ++expert) {
        const float value = expf(logits[expert] - max_value);
        probabilities[expert] = value;
        sum += (double)value;
    }
    return sum;
}

int moe_softmax_topk_router_llama_f32_workspace(
    const float *logits,
    int *indices,
    float *weights,
    int rows,
    int n_experts,
    int top_k,
    float routed_scaling_factor,
    void *workspace,
    size_t workspace_bytes)
{
    const size_t required = moe_softmax_topk_router_workspace_bytes(n_experts);
    if (!logits || !indices || !weights || !workspace || rows <= 0 ||
        n_experts <= 0 || top_k <= 0 || top_k > n_experts ||
        !isfinite(routed_scaling_factor) || required == 0 ||
        workspace_bytes < required) {
        return -1;
    }

    float *probabilities = (float *)workspace;
    for (int row = 0; row < rows; ++row) {
        const float *row_logits = logits + (size_t)row * (size_t)n_experts;
        int *row_indices = indices + (size_t)row * (size_t)top_k;
        float *row_weights = weights + (size_t)row * (size_t)top_k;
        float max_value = -INFINITY;
        for (int expert = 0; expert < n_experts; ++expert) {
            if (!isfinite(row_logits[expert])) {
                return -2;
            }
            if (row_logits[expert] > max_value) {
                max_value = row_logits[expert];
            }
        }

        const double softmax_sum = ck_moe_llama_softmax_row(
            probabilities, row_logits, n_experts, max_value);
        const float inverse_softmax_sum = (float)(1.0 / softmax_sum);
        for (int expert = 0; expert < n_experts; ++expert) {
            probabilities[expert] *= inverse_softmax_sum;
        }

        topk_f32(probabilities, n_experts, top_k, row_indices, NULL);
        double selected_sum_f64 = 0.0;
        for (int slot = 0; slot < top_k; ++slot) {
            row_weights[slot] = probabilities[row_indices[slot]];
            selected_sum_f64 += (double)row_weights[slot];
        }
        float selected_sum = (float)selected_sum_f64;
        if (selected_sum < 6.103515625e-5f) {
            selected_sum = 6.103515625e-5f;
        }
        for (int slot = 0; slot < top_k; ++slot) {
            row_weights[slot] =
                (row_weights[slot] / selected_sum) * routed_scaling_factor;
        }
    }
    return 0;
}

/**
 * @brief Backward for hard top-k followed by softmax over selected values.
 *
 * Matches PyTorch behavior for:
 *   values, indices = torch.topk(scores, k, dim=-1)
 *   weights = torch.softmax(values, dim=-1)
 *
 * The hard selected indices are treated as fixed for this backward pass.
 * Gradients are scattered only to selected scores; unselected scores are zero.
 */
void topk_softmax_backward_f32(const int *indices,
                               const float *weights,
                               const float *d_weights,
                               float *d_scores,
                               int num_tokens,
                               int n_experts_or_keys,
                               int k)
{
    if (!indices || !weights || !d_weights || !d_scores ||
        num_tokens <= 0 || n_experts_or_keys <= 0 || k <= 0) {
        return;
    }

    const size_t total = (size_t)num_tokens * (size_t)n_experts_or_keys;
    for (size_t i = 0; i < total; ++i) {
        d_scores[i] = 0.0f;
    }

    for (int t = 0; t < num_tokens; ++t) {
        const int *row_indices = indices + (size_t)t * (size_t)k;
        const float *row_weights = weights + (size_t)t * (size_t)k;
        const float *row_d_weights = d_weights + (size_t)t * (size_t)k;
        float *row_d_scores = d_scores + (size_t)t * (size_t)n_experts_or_keys;

        float dot = 0.0f;
        for (int i = 0; i < k; ++i) {
            const int idx = row_indices[i];
            if (idx >= 0 && idx < n_experts_or_keys) {
                dot += row_weights[i] * row_d_weights[i];
            }
        }

        for (int i = 0; i < k; ++i) {
            const int idx = row_indices[i];
            if (idx >= 0 && idx < n_experts_or_keys) {
                row_d_scores[idx] += row_weights[i] * (row_d_weights[i] - dot);
            }
        }
    }
}

/* =============================================================================
 * Batched Top-K (for multiple tokens)
 *
 * Process multiple tokens at once, each with its own routing scores.
 * ============================================================================= */

/**
 * @brief Batched top-K selection for multiple tokens
 *
 * @param scores Input scores [num_tokens, n_experts]
 * @param num_tokens Number of tokens
 * @param n_experts Number of experts
 * @param k Number of experts to select per token
 * @param indices Output: selected expert indices [num_tokens, k]
 * @param weights Output: routing weights [num_tokens, k] (can be NULL for no softmax)
 */
void topk_batched_f32(const float *scores,
                      int num_tokens,
                      int n_experts,
                      int k,
                      int *indices,
                      float *weights)
{
    if (!scores || !indices || num_tokens <= 0 || n_experts <= 0 || k <= 0) {
        return;
    }

    for (int t = 0; t < num_tokens; t++) {
        const float *token_scores = scores + t * n_experts;
        int *token_indices = indices + t * k;

        if (weights) {
            float *token_weights = weights + t * k;
            topk_softmax_f32(token_scores, n_experts, k, token_indices, token_weights);
        } else {
            topk_f32(token_scores, n_experts, k, token_indices, NULL);
        }
    }
}

/* =============================================================================
 * Argmax (special case of top-1)
 * ============================================================================= */

/**
 * @brief Find index of maximum value
 *
 * @param scores Input scores [n]
 * @param n Number of scores
 * @return Index of maximum value
 */
int argmax_f32(const float *scores, int n)
{
    if (!scores || n <= 0) {
        return -1;
    }

    int max_idx = 0;
    float max_val = scores[0];

#ifdef __AVX512F__
    /* AVX-512 vectorized argmax for large arrays */
    if (n >= 16) {
        __m512 vmax = _mm512_set1_ps(-FLT_MAX);
        __m512i vidx = _mm512_setzero_si512();
        __m512i vcur_max_idx = _mm512_setzero_si512();

        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&scores[i]);
            __m512i cur_idx = _mm512_add_epi32(
                _mm512_set1_epi32(i),
                _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            );

            __mmask16 gt_mask = _mm512_cmp_ps_mask(v, vmax, _CMP_GT_OQ);
            vmax = _mm512_mask_blend_ps(gt_mask, vmax, v);
            vcur_max_idx = _mm512_mask_blend_epi32(gt_mask, vcur_max_idx, cur_idx);
        }

        /* Horizontal reduction */
        float vals[16];
        int idxs[16];
        _mm512_storeu_ps(vals, vmax);
        _mm512_storeu_si512(idxs, vcur_max_idx);

        max_val = vals[0];
        max_idx = idxs[0];
        for (int j = 1; j < 16; j++) {
            if (vals[j] > max_val) {
                max_val = vals[j];
                max_idx = idxs[j];
            }
        }

        /* Handle remainder */
        for (; i < n; i++) {
            if (scores[i] > max_val) {
                max_val = scores[i];
                max_idx = i;
            }
        }

        return max_idx;
    }
#endif

    /* Scalar fallback */
    for (int i = 1; i < n; i++) {
        if (scores[i] > max_val) {
            max_val = scores[i];
            max_idx = i;
        }
    }

    return max_idx;
}

/* =============================================================================
 * Speculative decode verifier
 * ============================================================================= */

/**
 * @brief Greedy one-token speculative verification.
 *
 * The draft model proposes draft_token. The target model is authoritative:
 * if draft_token equals argmax(target_logits), the candidate is accepted and
 * emitted. Otherwise the target argmax is emitted and the draft path must be
 * reset or rewound by the runtime loop.
 *
 * @param target_logits Target/backbone logits [vocab_size]
 * @param vocab_size Number of logits
 * @param draft_token Candidate token from draft/assistant model
 * @param accepted Output scalar: 1 if accepted, 0 otherwise
 * @param verified_token Output scalar: accepted draft token or target argmax
 */
void speculative_verify_greedy_f32(const float *target_logits,
                                   int vocab_size,
                                   int draft_token,
                                   int *accepted,
                                   int *verified_token)
{
    const int target_token = argmax_f32(target_logits, vocab_size);
    const int ok = (target_token >= 0 && draft_token == target_token) ? 1 : 0;

    if (accepted) {
        *accepted = ok;
    }
    if (verified_token) {
        *verified_token = ok ? draft_token : target_token;
    }
}

/**
 * @brief Commit one verified speculative token and update decode counters.
 *
 * This is the minimal state transition for the first Gemma4 assistant bridge:
 * greedy, one draft token, target remains authoritative. For this milestone the
 * draft cache is kept synchronized with the target position after each token.
 * Multi-token speculative decoding can later replace this with prefix accept
 * and partial draft-cache rollback.
 */
void speculative_commit_one_i32(int accepted,
                                int verified_token,
                                int *token_buffer,
                                int *token_count,
                                int max_tokens,
                                int *target_position,
                                int *draft_position,
                                int *accepted_count,
                                int *rejected_count)
{
    int next_count = token_count ? *token_count : 0;
    if (token_buffer && token_count && next_count >= 0 && next_count < max_tokens) {
        token_buffer[next_count] = verified_token;
        next_count += 1;
        *token_count = next_count;
    }

    if (target_position) {
        *target_position += 1;
        if (draft_position) {
            *draft_position = *target_position;
        }
    } else if (draft_position) {
        *draft_position += 1;
    }

    if (accepted) {
        if (accepted_count) {
            *accepted_count += 1;
        }
    } else {
        if (rejected_count) {
            *rejected_count += 1;
        }
    }
}


/* =============================================================================
 * Group-limited MoE router for Nemotron-H/DeepSeek-style routed experts.
 *
 * Contract:
 *   scores          [rows, n_experts] router probabilities after sigmoid
 *   correction_bias [n_experts] optional score correction used only for choice
 *   indices         [rows, top_k]
 *   weights         [rows, top_k]
 *
 * Selection matches the HF/Nemotron policy:
 *   choice_scores = scores + correction_bias
 *   group_scores = sum(top2(choice_scores within group))
 *   selected_groups = topk(group_scores, topk_group)
 *   selected_experts = topk(choice_scores masked to selected groups, top_k)
 *   weights = gather(scores, selected_experts)
 *   if norm_topk_prob: weights /= sum(weights) + 1e-20
 *   weights *= routed_scaling_factor
 * ============================================================================= */

static void ck_topk_insert_desc(int idx, float val, int *indices, float *values, int k)
{
    for (int pos = 0; pos < k; ++pos) {
        if (indices[pos] < 0 || val > values[pos] || (val == values[pos] && idx < indices[pos])) {
            for (int j = k - 1; j > pos; --j) {
                indices[j] = indices[j - 1];
                values[j] = values[j - 1];
            }
            indices[pos] = idx;
            values[pos] = val;
            return;
        }
    }
}

static void group_limited_topk_router_f32_impl(const float *scores,
                                               const float *correction_bias,
                                               int *indices,
                                               float *weights,
                                               int rows,
                                               int n_experts,
                                               int top_k,
                                               int n_group,
                                               int topk_group,
                                               int norm_topk_prob,
                                               float routed_scaling_factor,
                                               int apply_sigmoid)
{
    if (!scores || !indices || !weights || rows <= 0 || n_experts <= 0 ||
        top_k <= 0 || n_group <= 0 || topk_group <= 0) {
        return;
    }
    if (top_k > n_experts) top_k = n_experts;
    if (n_group > n_experts) n_group = n_experts;
    if (topk_group > n_group) topk_group = n_group;
    const int experts_per_group = n_experts / n_group;
    if (experts_per_group <= 0 || experts_per_group * n_group != n_experts) {
        return;
    }

    for (int r = 0; r < rows; ++r) {
        const float *row_probs = scores + (size_t)r * (size_t)n_experts;
        float row_scores[n_experts];
        for (int e = 0; e < n_experts; ++e) {
            row_scores[e] = apply_sigmoid
                ? (1.0f / (1.0f + expf(-row_probs[e])))
                : row_probs[e];
        }
        int *row_indices = indices + (size_t)r * (size_t)top_k;
        float *row_weights = weights + (size_t)r * (size_t)top_k;

        int selected_groups[topk_group];
        float selected_group_scores[topk_group];
        for (int i = 0; i < topk_group; ++i) {
            selected_groups[i] = -1;
            selected_group_scores[i] = -FLT_MAX;
        }

        for (int g = 0; g < n_group; ++g) {
            float best0 = -FLT_MAX;
            float best1 = -FLT_MAX;
            const int start = g * experts_per_group;
            for (int j = 0; j < experts_per_group; ++j) {
                const int e = start + j;
                const float v = row_scores[e] + (correction_bias ? correction_bias[e] : 0.0f);
                if (v > best0) {
                    best1 = best0;
                    best0 = v;
                } else if (v > best1) {
                    best1 = v;
                }
            }
            const float group_score = best0 + ((experts_per_group >= 2) ? best1 : 0.0f);
            ck_topk_insert_desc(g, group_score, selected_groups, selected_group_scores, topk_group);
        }

        int out_idx[top_k];
        float out_choice[top_k];
        for (int i = 0; i < top_k; ++i) {
            out_idx[i] = -1;
            out_choice[i] = -FLT_MAX;
        }

        for (int sg = 0; sg < topk_group; ++sg) {
            const int g = selected_groups[sg];
            if (g < 0) continue;
            const int start = g * experts_per_group;
            for (int j = 0; j < experts_per_group; ++j) {
                const int e = start + j;
                const float v = row_scores[e] + (correction_bias ? correction_bias[e] : 0.0f);
                ck_topk_insert_desc(e, v, out_idx, out_choice, top_k);
            }
        }

        float denom = 1.0e-20f;
        for (int i = 0; i < top_k; ++i) {
            const int e = out_idx[i];
            const float w = (e >= 0 && e < n_experts) ? row_scores[e] : 0.0f;
            row_indices[i] = e;
            row_weights[i] = w;
            denom += w;
        }
        for (int i = 0; i < top_k; ++i) {
            float w = row_weights[i];
            if (norm_topk_prob) {
                w /= denom;
            }
            row_weights[i] = w * routed_scaling_factor;
        }
    }
}

void nemotron_group_limited_topk_router_f32(const float *scores,
                                            const float *correction_bias,
                                            int *indices,
                                            float *weights,
                                            int rows,
                                            int n_experts,
                                            int top_k,
                                            int n_group,
                                            int topk_group,
                                            int norm_topk_prob,
                                            float routed_scaling_factor)
{
    group_limited_topk_router_f32_impl(
        scores, correction_bias, indices, weights, rows, n_experts, top_k,
        n_group, topk_group, norm_topk_prob, routed_scaling_factor, 0
    );
}

void group_limited_topk_router_sigmoid_f32(const float *logits,
                                           const float *correction_bias,
                                           int *indices,
                                           float *weights,
                                           int rows,
                                           int n_experts,
                                           int top_k,
                                           int n_group,
                                           int topk_group,
                                           int norm_topk_prob,
                                           float routed_scaling_factor)
{
    group_limited_topk_router_f32_impl(
        logits, correction_bias, indices, weights, rows, n_experts, top_k,
        n_group, topk_group, norm_topk_prob, routed_scaling_factor, 1
    );
}
