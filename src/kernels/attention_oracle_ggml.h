#ifndef CK_ATTENTION_ORACLE_GGML_H
#define CK_ATTENTION_ORACLE_GGML_H

// Strict parity oracles for encoder-style full attention.
// These are ggml-backed composite helpers, not production CK kernels.

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

#endif
