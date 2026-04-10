/**
 * @file kv_cache_kernels.c
 * @brief KV-cache helper kernels (head-major layout)
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
 * Small, explicit helpers used by the runtime/orchestrator to maintain
 * per-layer KV caches during autoregressive decoding.
 *
 * Layout:
 *   k_cache[kv_head, token, aligned_head_dim]
 *   v_cache[kv_head, token, aligned_head_dim]
 * with contiguous row-major storage and stride aligned_head_dim.
 */

#include "ckernel_engine.h"

#include <stddef.h>
#include <string.h>

static inline void ck_local_fp32_to_fp16_row(const float *src, uint16_t *dst, int n)
{
    if (!src || !dst || n <= 0) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        dst[i] = CK_FP32_TO_FP16(src[i]);
    }
}

void kv_cache_repack_head_major_inplace(float *buf,
                                        int num_heads,
                                        int tokens,
                                        int cache_capacity,
                                        int aligned_head_dim)
{
    if (!buf) {
        return;
    }
    if (num_heads <= 0 || tokens <= 0 || cache_capacity <= 0 || aligned_head_dim <= 0) {
        return;
    }
    if (tokens > cache_capacity) {
        tokens = cache_capacity;
    }
    if (tokens == cache_capacity) {
        return;
    }

    const size_t old_head_stride = (size_t)tokens * (size_t)aligned_head_dim;
    const size_t new_head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;
    const size_t bytes = (size_t)tokens * (size_t)aligned_head_dim * sizeof(float);

    // Move head blocks from high to low to avoid overwriting source data
    // for heads that have not yet been moved.
    for (int h = num_heads - 1; h >= 0; --h) {
        float *src = buf + (size_t)h * old_head_stride;
        float *dst = buf + (size_t)h * new_head_stride;
        memmove(dst, src, bytes);
    }
}

void kv_cache_write_head_major(const float *__restrict k_token,
                               const float *__restrict v_token,
                               float *__restrict k_cache,
                               float *__restrict v_cache,
                               int num_kv_heads,
                               int token_index,
                               int cache_capacity,
                               int head_dim,
                               int aligned_head_dim)
{
    if (!k_token || !v_token || !k_cache || !v_cache) {
        return;
    }
    if (num_kv_heads <= 0 || token_index < 0 || cache_capacity <= 0) {
        return;
    }
    if (token_index >= cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;
    const size_t token_stride = (size_t)aligned_head_dim;

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *k_src = k_token + (size_t)h * token_stride;
        const float *v_src = v_token + (size_t)h * token_stride;

        float *k_dst = k_cache + (size_t)h * head_stride + (size_t)token_index * token_stride;
        float *v_dst = v_cache + (size_t)h * head_stride + (size_t)token_index * token_stride;

        for (int d = 0; d < head_dim; ++d) {
            k_dst[d] = k_src[d];
            v_dst[d] = v_src[d];
        }
        for (int d = head_dim; d < aligned_head_dim; ++d) {
            k_dst[d] = 0.0f;
            v_dst[d] = 0.0f;
        }
    }
}

void kv_cache_store(float *__restrict kv_cache_k,
                    float *__restrict kv_cache_v,
                    const float *__restrict k,
                    const float *__restrict v,
                    int layer,
                    int pos,
                    int num_kv_heads,
                    int head_dim,
                    int max_seq_len)
{
    (void)layer;
    kv_cache_write_head_major(k, v,
                              kv_cache_k, kv_cache_v,
                              num_kv_heads,
                              pos,
                              max_seq_len,
                              head_dim,
                              head_dim);
}

void kv_cache_store_f16(uint16_t *__restrict kv_cache_k,
                        uint16_t *__restrict kv_cache_v,
                        const float *__restrict k,
                        const float *__restrict v,
                        int layer,
                        int pos,
                        int num_kv_heads,
                        int head_dim,
                        int max_seq_len)
{
    (void)layer;
    if (!kv_cache_k || !kv_cache_v || !k || !v) {
        return;
    }
    if (num_kv_heads <= 0 || pos < 0 || head_dim <= 0 || max_seq_len <= 0) {
        return;
    }
    if (pos >= max_seq_len) {
        return;
    }

    const size_t head_stride = (size_t)max_seq_len * (size_t)head_dim;
    const size_t token_stride = (size_t)head_dim;

    for (int h = 0; h < num_kv_heads; ++h) {
        const float *k_src = k + (size_t)h * token_stride;
        const float *v_src = v + (size_t)h * token_stride;
        uint16_t *k_dst = kv_cache_k + (size_t)h * head_stride + (size_t)pos * token_stride;
        uint16_t *v_dst = kv_cache_v + (size_t)h * head_stride + (size_t)pos * token_stride;
        ck_local_fp32_to_fp16_row(k_src, k_dst, head_dim);
        ck_local_fp32_to_fp16_row(v_src, v_dst, head_dim);
    }
}

/**
 * @brief Copy logits to position-indexed location in output buffer.
 *
 * Used in decode mode to copy single-token logits from position 0 to
 * the correct sequence position. This moves buffer management logic
 * from codegen to the IR layer, making codegen "dumb" - just emit
 * kernel calls, no runtime if-statements.
 *
 * @param src       Source logits buffer (single token) [vocab_size]
 * @param dst       Destination logits buffer [max_seq_len, vocab_size]
 * @param position  Token position index (0-based)
 * @param vocab_size Number of logits per token
 */
void logits_copy_to_position(const float *__restrict src,
                              float *__restrict dst,
                              int position,
                              int vocab_size)
{
    if (!src || !dst || position < 0 || vocab_size <= 0) {
        return;
    }

    // Copy logits to dst[position * vocab_size : (position+1) * vocab_size]
    // Use memmove for safety in case src and dst overlap (e.g., src == dst)
    float *dst_pos = dst + (size_t)position * (size_t)vocab_size;
    memmove(dst_pos, src, (size_t)vocab_size * sizeof(float));
}
