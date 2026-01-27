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
