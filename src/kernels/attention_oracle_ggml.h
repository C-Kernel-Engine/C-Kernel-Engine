#ifndef CK_ATTENTION_ORACLE_GGML_H
#define CK_ATTENTION_ORACLE_GGML_H

#ifndef CK_ENABLE_LLAMA_CPP_PARITY
#define CK_ENABLE_LLAMA_CPP_PARITY 0
#endif

// Strict parity oracles for encoder-style full attention.
// These are ggml-backed composite helpers, not production CK kernels.

#if CK_ENABLE_LLAMA_CPP_PARITY
int ck_attention_head_full_ggml_graph_oracle_regular(const float *q_head,
                                                     const float *k_head,
                                                     const float *v_head,
                                                     float *out_head,
                                                     int num_tokens,
                                                     int head_dim,
                                                     int aligned_head_dim,
                                                     float scale);

int ck_attention_full_ggml_graph_oracle_multihead(const float *q,
                                                  const float *k,
                                                  const float *v,
                                                  float *output,
                                                  int num_heads,
                                                  int num_kv_heads,
                                                  int num_tokens,
                                                  int head_dim,
                                                  int aligned_head_dim,
                                                  int kv_stride_tokens,
                                                  float scale);
#else
static inline int ck_attention_head_full_ggml_graph_oracle_regular(const float *q_head,
                                                                   const float *k_head,
                                                                   const float *v_head,
                                                                   float *out_head,
                                                                   int num_tokens,
                                                                   int head_dim,
                                                                   int aligned_head_dim,
                                                                   float scale)
{
    (void) q_head;
    (void) k_head;
    (void) v_head;
    (void) out_head;
    (void) num_tokens;
    (void) head_dim;
    (void) aligned_head_dim;
    (void) scale;
    return 0;
}

static inline int ck_attention_full_ggml_graph_oracle_multihead(const float *q,
                                                                const float *k,
                                                                const float *v,
                                                                float *output,
                                                                int num_heads,
                                                                int num_kv_heads,
                                                                int num_tokens,
                                                                int head_dim,
                                                                int aligned_head_dim,
                                                                int kv_stride_tokens,
                                                                float scale)
{
    (void) q;
    (void) k;
    (void) v;
    (void) output;
    (void) num_heads;
    (void) num_kv_heads;
    (void) num_tokens;
    (void) head_dim;
    (void) aligned_head_dim;
    (void) kv_stride_tokens;
    (void) scale;
    return 0;
}
#endif

#endif
