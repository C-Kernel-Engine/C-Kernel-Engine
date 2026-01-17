/**
 * @file embedding_kernels_bf16.c
 * @brief Token/position embedding lookup kernels for BF16
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 */

#include <stdint.h>
#include <string.h>

#include "bf16_utils.h"
#include "ckernel_engine.h"

void embedding_forward_bf16(const int32_t *token_ids,
                            int token_count,
                            int vocab_size,
                            const uint16_t *token_embeddings,
                            const uint16_t *pos_embeddings,
                            uint16_t *output,
                            int embed_dim,
                            int aligned_embed_dim,
                            int context_window,
                            int add_pos)
{
    if (!token_ids || !token_embeddings || !output) {
        return;
    }

    int tokens = token_count;
    if (tokens < 0) tokens = 0;
    if (tokens > context_window) tokens = context_window;

    for (int t = 0; t < tokens; ++t) {
        int id = token_ids[t];
        if (id < 0 || id >= vocab_size) {
            id = 0;
        }

        const uint16_t *tok = token_embeddings + (size_t)id * (size_t)aligned_embed_dim;
        const uint16_t *pos = pos_embeddings ? (pos_embeddings + (size_t)t * (size_t)aligned_embed_dim) : NULL;
        uint16_t *out = output + (size_t)t * (size_t)aligned_embed_dim;

        if (add_pos && pos) {
            for (int d = 0; d < embed_dim; ++d) {
                float v = bf16_to_float(tok[d]) + bf16_to_float(pos[d]);
                out[d] = float_to_bf16(v);
            }
        } else {
            for (int d = 0; d < embed_dim; ++d) {
                out[d] = tok[d];
            }
        }

        for (int d = embed_dim; d < aligned_embed_dim; ++d) {
            out[d] = 0;
        }
    }

    for (int t = tokens; t < context_window; ++t) {
        uint16_t *out = output + (size_t)t * (size_t)aligned_embed_dim;
        memset(out, 0, (size_t)aligned_embed_dim * sizeof(uint16_t));
    }
}

void embedding_backward_bf16(const int32_t *token_ids,
                             int token_count,
                             const uint16_t *d_output,
                             uint16_t *d_token_embeddings,
                             uint16_t *d_pos_embeddings,
                             int vocab_size,
                             int embed_dim,
                             int aligned_embed_dim,
                             int context_window,
                             int add_pos)
{
    if (!token_ids || !d_output || !d_token_embeddings) {
        return;
    }

    int tokens = token_count;
    if (tokens < 0) tokens = 0;
    if (tokens > context_window) tokens = context_window;

    for (int t = 0; t < tokens; ++t) {
        int id = token_ids[t];
        if (id < 0 || id >= vocab_size) {
            id = 0;
        }

        const uint16_t *d_out = d_output + (size_t)t * (size_t)aligned_embed_dim;
        uint16_t *d_tok = d_token_embeddings + (size_t)id * (size_t)aligned_embed_dim;
        uint16_t *d_pos = d_pos_embeddings ? (d_pos_embeddings + (size_t)t * (size_t)aligned_embed_dim) : NULL;

        for (int d = 0; d < embed_dim; ++d) {
            float grad = bf16_to_float(d_out[d]);

            float cur_tok = bf16_to_float(d_tok[d]);
            d_tok[d] = float_to_bf16(cur_tok + grad);

            if (add_pos && d_pos) {
                float cur_pos = bf16_to_float(d_pos[d]);
                d_pos[d] = float_to_bf16(cur_pos + grad);
            }
        }
    }
}

