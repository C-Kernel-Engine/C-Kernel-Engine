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

#ifdef __AVX512F__
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
