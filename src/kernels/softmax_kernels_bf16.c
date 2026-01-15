#include <stddef.h>
#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

// Suppress false positive warnings about uninitialized variables
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

/*
 * BF16 causal softmax with caller-provided scratch buffer.
 * scratch: [num_heads * aligned_context_window * aligned_context_window] floats
 */
void causal_softmax_head_major_bf16(uint16_t *scores,
                                   int num_heads,
                                   int num_tokens,
                                   int aligned_context_window,
                                   float *scratch)
{
    if (!scores || num_heads <= 0 || num_tokens <= 0 || aligned_context_window <= 0) return;
    if (!scratch) return;

    const size_t total = (size_t)num_heads *
                         (size_t)aligned_context_window *
                         (size_t)aligned_context_window;

    bf16_tensor_to_float(scores, scratch, total);
    causal_softmax_head_major(scratch, num_heads, num_tokens, aligned_context_window);
    float_tensor_to_bf16(scratch, scores, total);
}

/*
 * BF16 backward causal softmax with caller-provided scratch buffers.
 * scratch_d_scores, scratch_weights: each [num_heads * aligned_context_window * aligned_context_window] floats
 */
void backward_causal_softmax_head_major_bf16(uint16_t *d_scores,
                                            const uint16_t *weights,
                                            int num_heads,
                                            int num_tokens,
                                            int aligned_context_window,
                                            float *scratch_d_scores,
                                            float *scratch_weights)
{
    if (!d_scores || !weights || num_heads <= 0 || num_tokens <= 0 || aligned_context_window <= 0) return;
    if (!scratch_d_scores || !scratch_weights) return;

    const size_t total = (size_t)num_heads *
                         (size_t)aligned_context_window *
                         (size_t)aligned_context_window;

    bf16_tensor_to_float(d_scores, scratch_d_scores, total);
    bf16_tensor_to_float(weights, scratch_weights, total);
    backward_causal_softmax_head_major(scratch_d_scores, scratch_weights, num_heads, num_tokens, aligned_context_window);
    float_tensor_to_bf16(scratch_d_scores, d_scores, total);
}

#pragma GCC diagnostic pop
