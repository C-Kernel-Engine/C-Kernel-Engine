/**
 * @file embedding_kernels.c
 * @brief Token/position embedding lookup kernels
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
 * Embedding: out[t] = token_embed[token_id[t]] + pos_embed[t]
 */

#include "ckernel_engine.h"
#include "ckernel_dtype.h"

#include <string.h>

void embedding_forward(const int32_t *token_ids,
                       int token_count,
                       int vocab_size,
                       const float *token_embeddings,
                       const float *pos_embeddings,
                       float *output,
                       int embed_dim,
                       int aligned_embed_dim,
                       int context_window,
                       int add_pos)
{
    if (!token_ids || !token_embeddings || !output) {
        return;
    }

    int tokens = token_count;
    if (tokens < 0) {
        tokens = 0;
    }
    if (tokens > context_window) {
        tokens = context_window;
    }

    for (int t = 0; t < tokens; ++t) {
        int id = token_ids[t];
        if (id < 0 || id >= vocab_size) {
            id = 0;
        }

        const float *tok = token_embeddings + (size_t)id * (size_t)aligned_embed_dim;
        const float *pos = pos_embeddings ? (pos_embeddings + (size_t)t * (size_t)aligned_embed_dim) : NULL;
        float *out = output + (size_t)t * (size_t)aligned_embed_dim;

        if (add_pos && pos) {
            for (int d = 0; d < embed_dim; ++d) {
                out[d] = tok[d] + pos[d];
            }
        } else {
            for (int d = 0; d < embed_dim; ++d) {
                out[d] = tok[d];
            }
        }

        for (int d = embed_dim; d < aligned_embed_dim; ++d) {
            out[d] = 0.0f;
        }
    }

    for (int t = tokens; t < context_window; ++t) {
        float *out = output + (size_t)t * (size_t)aligned_embed_dim;
        memset(out, 0, (size_t)aligned_embed_dim * sizeof(float));
    }
}

void embedding_forward_q4_k(const int32_t *token_ids,
                            int token_count,
                            int vocab_size,
                            const void *token_embeddings,
                            const float *pos_embeddings,
                            float *output,
                            int embed_dim,
                            int aligned_embed_dim,
                            int context_window,
                            int add_pos)
{
    if (!token_ids || !token_embeddings || !output) {
        return;
    }

    int tokens = token_count;
    if (tokens < 0) {
        tokens = 0;
    }
    if (tokens > context_window) {
        tokens = context_window;
    }

    const size_t row_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, (size_t)aligned_embed_dim);
    const uint8_t *base = (const uint8_t *)token_embeddings;

    for (int t = 0; t < tokens; ++t) {
        int id = token_ids[t];
        if (id < 0 || id >= vocab_size) {
            id = 0;
        }

        const void *tok = base + (size_t)id * row_bytes;
        const float *pos = pos_embeddings ? (pos_embeddings + (size_t)t * (size_t)aligned_embed_dim) : NULL;
        float *out = output + (size_t)t * (size_t)aligned_embed_dim;

        dequant_q4_k_row(tok, out, (size_t)aligned_embed_dim);

        if (add_pos && pos) {
            for (int d = 0; d < embed_dim; ++d) {
                out[d] += pos[d];
            }
        }

        for (int d = embed_dim; d < aligned_embed_dim; ++d) {
            out[d] = 0.0f;
        }
    }

    for (int t = tokens; t < context_window; ++t) {
        float *out = output + (size_t)t * (size_t)aligned_embed_dim;
        memset(out, 0, (size_t)aligned_embed_dim * sizeof(float));
    }
}

void embedding_forward_q8_0(const int32_t *token_ids,
                            int token_count,
                            int vocab_size,
                            const void *token_embeddings,
                            const float *pos_embeddings,
                            float *output,
                            int embed_dim,
                            int aligned_embed_dim,
                            int context_window,
                            int add_pos)
{
    if (!token_ids || !token_embeddings || !output) {
        return;
    }

    int tokens = token_count;
    if (tokens < 0) {
        tokens = 0;
    }
    if (tokens > context_window) {
        tokens = context_window;
    }

    const size_t row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)aligned_embed_dim);
    const uint8_t *base = (const uint8_t *)token_embeddings;

    for (int t = 0; t < tokens; ++t) {
        int id = token_ids[t];
        if (id < 0 || id >= vocab_size) {
            id = 0;
        }

        const void *tok = base + (size_t)id * row_bytes;
        const float *pos = pos_embeddings ? (pos_embeddings + (size_t)t * (size_t)aligned_embed_dim) : NULL;
        float *out = output + (size_t)t * (size_t)aligned_embed_dim;

        dequant_q8_0_row(tok, out, (size_t)aligned_embed_dim);

        if (add_pos && pos) {
            for (int d = 0; d < embed_dim; ++d) {
                out[d] += pos[d];
            }
        }

        for (int d = embed_dim; d < aligned_embed_dim; ++d) {
            out[d] = 0.0f;
        }
    }

    for (int t = tokens; t < context_window; ++t) {
        float *out = output + (size_t)t * (size_t)aligned_embed_dim;
        memset(out, 0, (size_t)aligned_embed_dim * sizeof(float));
    }
}

void embedding_forward_q6_k(const int32_t *token_ids,
                            int token_count,
                            int vocab_size,
                            const void *token_embeddings,
                            const float *pos_embeddings,
                            float *output,
                            int embed_dim,
                            int aligned_embed_dim,
                            int context_window,
                            int add_pos)
{
    if (!token_ids || !token_embeddings || !output) {
        return;
    }

    int tokens = token_count;
    if (tokens < 0) {
        tokens = 0;
    }
    if (tokens > context_window) {
        tokens = context_window;
    }

    const size_t row_bytes = ck_dtype_row_bytes(CK_DT_Q6_K, (size_t)aligned_embed_dim);
    const uint8_t *base = (const uint8_t *)token_embeddings;

    for (int t = 0; t < tokens; ++t) {
        int id = token_ids[t];
        if (id < 0 || id >= vocab_size) {
            id = 0;
        }

        const void *tok = base + (size_t)id * row_bytes;
        const float *pos = pos_embeddings ? (pos_embeddings + (size_t)t * (size_t)aligned_embed_dim) : NULL;
        float *out = output + (size_t)t * (size_t)aligned_embed_dim;

        dequant_q6_k_row(tok, out, (size_t)aligned_embed_dim);

        if (add_pos && pos) {
            for (int d = 0; d < embed_dim; ++d) {
                out[d] += pos[d];
            }
        }

        for (int d = embed_dim; d < aligned_embed_dim; ++d) {
            out[d] = 0.0f;
        }
    }

    for (int t = tokens; t < context_window; ++t) {
        float *out = output + (size_t)t * (size_t)aligned_embed_dim;
        memset(out, 0, (size_t)aligned_embed_dim * sizeof(float));
    }
}

void embedding_backward(const int32_t *token_ids,
                        int token_count,
                        const float *d_output,
                        float *d_token_embeddings,
                        float *d_pos_embeddings,
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
    if (tokens < 0) {
        tokens = 0;
    }
    if (tokens > context_window) {
        tokens = context_window;
    }

    for (int t = 0; t < tokens; ++t) {
        int id = token_ids[t];
        if (id < 0 || id >= vocab_size) {
            id = 0;
        }

        const float *d_out = d_output + (size_t)t * (size_t)aligned_embed_dim;
        float *d_tok = d_token_embeddings + (size_t)id * (size_t)aligned_embed_dim;
        float *d_pos = d_pos_embeddings ? (d_pos_embeddings + (size_t)t * (size_t)aligned_embed_dim) : NULL;

        for (int d = 0; d < embed_dim; ++d) {
            float grad = d_out[d];
            d_tok[d] += grad;
            if (add_pos && d_pos) {
                d_pos[d] += grad;
            }
        }
    }
}
