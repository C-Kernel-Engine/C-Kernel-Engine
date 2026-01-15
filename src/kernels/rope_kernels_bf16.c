#include <stdint.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

// Suppress false positive warnings about uninitialized variables
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

/*
 * BF16 RoPE forward with caller-provided scratch buffer.
 * scratch: [num_heads * num_tokens * aligned_head_dim] floats
 */
void rope_forward_bf16(uint16_t *x,
                       const float *cos_cache,
                       const float *sin_cache,
                       int num_heads,
                       int num_tokens,
                       int head_dim,
                       int aligned_head_dim,
                       int pos_offset,
                       float *scratch)
{
    if (!scratch) return;

    size_t total = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;

    bf16_tensor_to_float(x, scratch, total);
    rope_forward(scratch, cos_cache, sin_cache,
                 num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
    float_tensor_to_bf16(scratch, x, total);
}

/*
 * BF16 RoPE backward with caller-provided scratch buffers.
 * scratch_d_out, scratch_d_x: each [num_heads * num_tokens * aligned_head_dim] floats
 */
void rope_backward_bf16(const uint16_t *d_out,
                        uint16_t *d_x,
                        const float *cos_cache,
                        const float *sin_cache,
                        int num_heads,
                        int num_tokens,
                        int head_dim,
                        int aligned_head_dim,
                        int pos_offset,
                        float *scratch_d_out,
                        float *scratch_d_x)
{
    if (!scratch_d_out || !scratch_d_x) return;

    size_t total = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;

    bf16_tensor_to_float(d_out, scratch_d_out, total);
    rope_backward(scratch_d_out, scratch_d_x, cos_cache, sin_cache,
                  num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
    float_tensor_to_bf16(scratch_d_x, d_x, total);
}

/*
 * BF16 RoPE forward for Q and K with caller-provided scratch buffers.
 * scratch_q: [num_heads * num_tokens * aligned_head_dim] floats
 * scratch_k: [num_kv_heads * num_tokens * aligned_head_dim] floats
 */
void rope_forward_qk_bf16(uint16_t *q,
                          uint16_t *k,
                          const float *cos_cache,
                          const float *sin_cache,
                          int num_heads,
                          int num_kv_heads,
                          int num_tokens,
                          int head_dim,
                          int aligned_head_dim,
                          int pos_offset,
                          float *scratch_q,
                          float *scratch_k)
{
    if (!q || !k) return;

    rope_forward_bf16(q, cos_cache, sin_cache,
                      num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset, scratch_q);
    rope_forward_bf16(k, cos_cache, sin_cache,
                      num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset, scratch_k);
}

/*
 * BF16 RoPE backward for Q and K with caller-provided scratch buffers.
 */
void rope_backward_qk_bf16(const uint16_t *d_q_out,
                           const uint16_t *d_k_out,
                           uint16_t *d_q,
                           uint16_t *d_k,
                           const float *cos_cache,
                           const float *sin_cache,
                           int num_heads,
                           int num_kv_heads,
                           int num_tokens,
                           int head_dim,
                           int aligned_head_dim,
                           int pos_offset,
                           float *scratch_dq_out,
                           float *scratch_dq,
                           float *scratch_dk_out,
                           float *scratch_dk)
{
    if (!d_q_out || !d_k_out || !d_q || !d_k) return;

    rope_backward_bf16(d_q_out, d_q, cos_cache, sin_cache,
                       num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset,
                       scratch_dq_out, scratch_dq);
    rope_backward_bf16(d_k_out, d_k, cos_cache, sin_cache,
                       num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset,
                       scratch_dk_out, scratch_dk);
}

#pragma GCC diagnostic pop
