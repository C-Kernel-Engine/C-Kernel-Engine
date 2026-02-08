/**
 * @file ckernel_orchestration.h
 *
 * ===========================================================================
 * LEGACY HEADER - NOT USED IN v6.6
 * ===========================================================================
 *
 * This header declares v6.5 orchestration functions that are NO LONGER USED.
 * v6.6 uses IR Lower 3 + codegen instead of hardcoded orchestration.
 *
 * v6.6 Architecture (REPLACEMENT):
 *   - Kernel dispatch: version/v6.6/scripts/build_ir_v6_6.py + ckernel_codegen.c
 *   - Memory planning: version/v6.6/scripts/memory_planner_v6_6.py
 *   - Registry: version/v6.6/kernel_maps/KERNEL_REGISTRY.json
 *   - Kernel bindings: version/v6.6/kernel_maps/kernel_bindings.json
 *
 * Deprecated functions (NOT used in v6.6):
 *   - ck_layer_forward_rmsnorm_swiglu*      -> IR Lower 3 + mega_fused_* kernels
 *   - ck_qkv_project_head_major*            -> q_proj/k_proj/v_proj ops in IR
 *   - ck_attention_project_head_major*      -> out_proj op in IR
 *   - ck_mlp_swiglu_forward*                -> mlp_gate_up/mlp_down ops in IR
 *   - ck_gemm_nt_quant                      -> KERNEL_REGISTRY.json dispatch
 *   - ck_residual_add_token_major           -> residual_add op in IR
 *
 * To remove completely:
 *   1. Delete this header
 *   2. Delete ckernel_orchestration.c
 *   3. Remove from Makefile SRCS list
 *
 * Last used: v6.5
 * Deprecated: v6.6 (2026-02)
 * ===========================================================================
 */

#ifndef CKERNEL_ORCHESTRATION_H
#define CKERNEL_ORCHESTRATION_H

#include <stddef.h>
#include "ckernel_dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int tokens;
    int embed_dim;
    int aligned_embed_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int aligned_head_dim;
    int aligned_context_window;
    int intermediate_dim;
    int aligned_intermediate_dim;
    float eps;
    int rope_pos_offset;

    const float *input;     /* [T x aligned_embed_dim] */
    const float *ln1_gamma; /* [aligned_embed_dim] */
    const float *ln2_gamma; /* [aligned_embed_dim] */

    const float *rope_cos; /* [max_seq_len x head_dim/2] */
    const float *rope_sin; /* [max_seq_len x head_dim/2] */

    const float *wq; /* [num_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bq; /* [num_heads x aligned_head_dim] */
    const float *wk; /* [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bk; /* [num_kv_heads x aligned_head_dim] */
    const float *wv; /* [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bv; /* [num_kv_heads x aligned_head_dim] */

    const float *wo; /* [H x aligned_embed_dim x aligned_head_dim] */
    const float *bo; /* [aligned_embed_dim] */

    const float *w1; /* [2*aligned_intermediate_dim x aligned_embed_dim] */
    const float *b1; /* [2*aligned_intermediate_dim] */
    const float *w2; /* [aligned_embed_dim x aligned_intermediate_dim] */
    const float *b2; /* [aligned_embed_dim] */

    float *ln1_out;   /* [T x aligned_embed_dim] */
    float *ln1_rstd;  /* [T] (optional) */
    float *q;         /* [num_heads x T x aligned_head_dim] */
    float *k;         /* [num_kv_heads x T x aligned_head_dim] */
    float *v;         /* [num_kv_heads x T x aligned_head_dim] */
    float *scores;    /* [num_heads x aligned_context_window x aligned_context_window] */
    float *attn_out;  /* [num_heads x T x aligned_head_dim] */
    float *proj_tmp;  /* [T x aligned_embed_dim] */
    float *proj_scratch; /* [T x aligned_embed_dim], required if num_heads > 1 */
    float *residual1; /* [T x aligned_embed_dim] */
    float *ln2_out;   /* [T x aligned_embed_dim] */
    float *ln2_rstd;  /* [T] (optional) */
    float *fc1_out;   /* [T x 2*aligned_intermediate_dim] */
    float *swiglu_out;/* [T x aligned_intermediate_dim] */
    float *mlp_out;   /* [T x aligned_embed_dim] */
    float *output;    /* [T x aligned_embed_dim] */
} CKLayerForwardParams;

typedef struct {
    int tokens;
    int embed_dim;
    int aligned_embed_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int aligned_head_dim;
    int aligned_context_window;
    int intermediate_dim;
    int aligned_intermediate_dim;
    float eps;
    int rope_pos_offset;

    const float *input;     /* [T x aligned_embed_dim] */
    const float *ln1_gamma; /* [aligned_embed_dim] */
    const float *ln2_gamma; /* [aligned_embed_dim] */
    const float *ln1_out;   /* [T x aligned_embed_dim] */
    const float *ln1_rstd;  /* [T] */
    const float *ln2_out;   /* [T x aligned_embed_dim] */
    const float *ln2_rstd;  /* [T] */

    const float *rope_cos; /* [max_seq_len x head_dim/2] */
    const float *rope_sin; /* [max_seq_len x head_dim/2] */

    const float *wq; /* [num_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bq; /* [num_heads x aligned_head_dim] */
    const float *wk; /* [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bk; /* [num_kv_heads x aligned_head_dim] */
    const float *wv; /* [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bv; /* [num_kv_heads x aligned_head_dim] */

    const float *wo; /* [H x aligned_embed_dim x aligned_head_dim] */
    const float *bo; /* [aligned_embed_dim] */

    const float *w1; /* [2*aligned_intermediate_dim x aligned_embed_dim] */
    const float *b1; /* [2*aligned_intermediate_dim] */
    const float *w2; /* [aligned_embed_dim x aligned_intermediate_dim] */
    const float *b2; /* [aligned_embed_dim] */

    const float *q;         /* [num_heads x T x aligned_head_dim] */
    const float *k;         /* [num_kv_heads x T x aligned_head_dim] */
    const float *v;         /* [num_kv_heads x T x aligned_head_dim] */
    const float *scores;    /* [num_heads x aligned_context_window x aligned_context_window] */
    const float *attn_out;  /* [num_heads x T x aligned_head_dim] */
    const float *residual1; /* [T x aligned_embed_dim] */
    const float *fc1_out;   /* [T x 2*aligned_intermediate_dim] */
    const float *swiglu_out;/* [T x aligned_intermediate_dim] */

    float *d_output;    /* [T x aligned_embed_dim] */
    float *d_input;     /* [T x aligned_embed_dim] */
    float *d_ln1_gamma; /* [aligned_embed_dim] */
    float *d_ln2_gamma; /* [aligned_embed_dim] */
    float *d_wq;        /* [num_heads x aligned_head_dim x aligned_embed_dim] */
    float *d_bq;        /* [num_heads x aligned_head_dim] */
    float *d_wk;        /* [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    float *d_bk;        /* [num_kv_heads x aligned_head_dim] */
    float *d_wv;        /* [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    float *d_bv;        /* [num_kv_heads x aligned_head_dim] */
    float *d_wo;        /* [H x aligned_embed_dim x aligned_head_dim] */
    float *d_bo;        /* [aligned_embed_dim] */
    float *d_w1;        /* [2*aligned_intermediate_dim x aligned_embed_dim] */
    float *d_b1;        /* [2*aligned_intermediate_dim] */
    float *d_w2;        /* [aligned_embed_dim x aligned_intermediate_dim] */
    float *d_b2;        /* [aligned_embed_dim] */

    float *d_ln1_out;    /* [T x aligned_embed_dim] */
    float *d_q;          /* [num_heads x T x aligned_head_dim] */
    float *d_k;          /* [num_kv_heads x T x aligned_head_dim] */
    float *d_v;          /* [num_kv_heads x T x aligned_head_dim] */
    float *d_scores;     /* [num_heads x aligned_context_window x aligned_context_window] */
    float *d_attn_out;   /* [num_heads x T x aligned_head_dim] */
    float *d_proj_tmp;   /* [T x aligned_embed_dim] */
    float *d_residual1;  /* [T x aligned_embed_dim] */
    float *d_ln2_out;    /* [T x aligned_embed_dim] */
    float *d_fc1_out;    /* [T x 2*aligned_intermediate_dim] */
    float *d_swiglu_out; /* [T x aligned_intermediate_dim] */
    float *d_mlp_out;    /* [T x aligned_embed_dim] */
} CKLayerBackwardParams;

void ck_residual_add_token_major(const float *a,
                                 const float *b,
                                 float *out,
                                 int tokens,
                                 int aligned_embed_dim);

/* Generic quantized GEMM dispatcher (NT layout) */
void ck_gemm_nt_quant(const float *A,
                      const void *B,
                      const float *bias,
                      float *C,
                      int M, int N, int K,
                      CKDataType dtype);

void ck_qkv_project_head_major(const float *input,
                               const float *wq, const float *bq,
                               const float *wk, const float *bk,
                               const float *wv, const float *bv,
                               float *q, float *k, float *v,
                               int tokens,
                               int kv_stride_tokens,
                               int aligned_embed_dim,
                               int num_heads,
                               int num_kv_heads,
                               int aligned_head_dim);

void ck_qkv_project_head_major_token(const float *input_row,
                                     const float *wq, const float *bq,
                                     const float *wk, const float *bk,
                                     const float *wv, const float *bv,
                                     float *q_token,
                                     float *k_token,
                                     float *v_token,
                                     int aligned_embed_dim,
                                     int num_heads,
                                     int num_kv_heads,
                                     int aligned_head_dim);

void ck_attention_project_head_major(const float *attn_out,
                                     const float *wo,
                                     const float *bo,
                                     float *out,
                                     float *scratch,
                                     int tokens,
                                     int aligned_embed_dim,
                                     int num_heads,
                                     int aligned_head_dim);

void ck_attention_project_head_major_decode_token(const float *attn_token,
                                                  const float *wo,
                                                  const float *bo,
                                                  float *out_token,
                                                  int embed_dim,
                                                  int aligned_embed_dim,
                                                  int num_heads,
                                                  int aligned_head_dim);

void ck_mlp_swiglu_forward(const float *input,
                           const float *w1,
                           const float *b1,
                           const float *w2,
                           const float *b2,
                           float *fc1_out,
                           float *swiglu_out,
                           float *output,
                           int tokens,
                           int aligned_embed_dim,
                           int aligned_intermediate_dim);

void ck_mlp_swiglu_forward_fused_token(const float *input_row,
                                       const float *w1,
                                       const float *b1,
                                       const float *w2,
                                       const float *b2,
                                       float *swiglu_row,
                                       float *output_row,
                                       int aligned_embed_dim,
                                       int aligned_intermediate_dim);

// Fully fused MLP for decode (single token).
// All three projections (gate, up, down) fused into one kernel.
// Eliminates DRAM round-trip for intermediate swiglu values.
// Best for AVX-512 systems with many cores (24+).
void ck_mlp_swiglu_forward_fully_fused_token(const float *input_row,
                                              const float *w1,
                                              const float *b1,
                                              const float *w2,
                                              const float *b2,
                                              float *output_row,
                                              int aligned_embed_dim,
                                              int aligned_intermediate_dim);

void ck_layer_forward_rmsnorm_swiglu(const CKLayerForwardParams *p);
void ck_layer_forward_rmsnorm_swiglu_ref(const CKLayerForwardParams *p);

// Decode-style layer forward for autoregressive generation.
//
// Computes only a single token at `token_index`, while attending over the
// KV-cache stored in `p->k`/`p->v` in head-major cache layout:
//   k/v: [num_kv_heads, cache_capacity, aligned_head_dim]
//
// The caller is responsible for:
//   - ensuring `p->k`/`p->v` already contain tokens [0..token_index-1]
//   - setting `p->rope_pos_offset` to the absolute position for this token
//   - passing a matching `cache_capacity` (usually model context_window)
void ck_layer_forward_rmsnorm_swiglu_decode(const CKLayerForwardParams *p,
                                           int token_index,
                                           int cache_capacity);

// Decode-style layer forward using fused SwiGLU (gate+up) matvec.
// Inference-only fast path: produces the same outputs as the unfused decode path.
void ck_layer_forward_rmsnorm_swiglu_decode_fused(const CKLayerForwardParams *p,
                                                  int token_index,
                                                  int cache_capacity);

// Decode-style layer forward using fused attention (QKV+RoPE+KV+attention+Wo).
// Optionally pairs with fused SwiGLU via ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_mlp.
void ck_layer_forward_rmsnorm_swiglu_decode_fused_attn(const CKLayerForwardParams *p,
                                                       int token_index,
                                                       int cache_capacity);

void ck_layer_forward_rmsnorm_swiglu_decode_fused_attn_mlp(const CKLayerForwardParams *p,
                                                           int token_index,
                                                           int cache_capacity);

/* ============================================================================
 * Quantized (Q4_K / Q4_K_M) inference orchestration
 *
 * These entry points mirror the fp32 paths but accept weight matrices stored
 * in GGML-compatible Q4_K blocks. Activations remain fp32 by default; the
 * decode path can switch to Q8_K activations via CK_Q8K_ACTIVATIONS=1 (or
 * auto-enable when strict parity is off).
 *
 * Design note:
 *  - If you enable Q4_K weights, ensure the relevant K dimensions are a
 *    multiple of 256 (QK_K). The engine keeps the quantized weights in their
 *    compact block form and dequantizes on-the-fly inside GEMM/GEMV kernels.
 * ============================================================================ */

typedef struct {
    int tokens;
    int embed_dim;
    int aligned_embed_dim;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int aligned_head_dim;
    int aligned_context_window;
    int intermediate_dim;
    int aligned_intermediate_dim;
    float eps;
    int rope_pos_offset;

    const float *input;     /* [T x aligned_embed_dim] */
    const float *ln1_gamma; /* [aligned_embed_dim] */
    const float *ln2_gamma; /* [aligned_embed_dim] */

    const float *rope_cos; /* [max_seq_len x head_dim/2] */
    const float *rope_sin; /* [max_seq_len x head_dim/2] */

    const void *wq;  /* Q4_K: [num_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bq; /* [num_heads x aligned_head_dim] */
    const void *wk;  /* Q4_K: [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bk; /* [num_kv_heads x aligned_head_dim] */
    const void *wv;  /* Q4_K: [num_kv_heads x aligned_head_dim x aligned_embed_dim] */
    const float *bv; /* [num_kv_heads x aligned_head_dim] */

    const void *wo;  /* Q4_K: [aligned_embed_dim x (num_heads*aligned_head_dim)] */
    const float *bo; /* [aligned_embed_dim] */

    const void *w1;  /* Q4_K: [2*aligned_intermediate_dim x aligned_embed_dim] */
    const float *b1; /* [2*aligned_intermediate_dim] */
    const void *w2;  /* Q4_K: [aligned_embed_dim x aligned_intermediate_dim] */
    const float *b2; /* [aligned_embed_dim] */

    float *ln1_out;   /* [T x aligned_embed_dim] */
    float *ln1_rstd;  /* [T] (optional) */
    float *q;         /* [num_heads x T x aligned_head_dim] */
    float *k;         /* [num_kv_heads x T x aligned_head_dim] */
    float *v;         /* [num_kv_heads x T x aligned_head_dim] */
    float *scores;    /* [num_heads x aligned_context_window x aligned_context_window] */
    float *attn_out;  /* [num_heads x T x aligned_head_dim] */
    float *proj_tmp;  /* [T x aligned_embed_dim] */
    float *proj_scratch; /* [T x aligned_embed_dim], required (transpose buffer) */
    float *residual1; /* [T x aligned_embed_dim] */
    float *ln2_out;   /* [T x aligned_embed_dim] */
    float *ln2_rstd;  /* [T] (optional) */
    float *fc1_out;   /* [T x 2*aligned_intermediate_dim] */
    float *swiglu_out;/* [T x aligned_intermediate_dim] */
    float *mlp_out;   /* [T x aligned_embed_dim] */
    float *output;    /* [T x aligned_embed_dim] */

    CKDataType wq_dtype;
    CKDataType wk_dtype;
    CKDataType wv_dtype;
    CKDataType wo_dtype;
    CKDataType w1_dtype;
    CKDataType w2_dtype;
} CKLayerForwardParamsQ4K;

void ck_layer_forward_rmsnorm_swiglu_q4_k(const CKLayerForwardParamsQ4K *p);
void ck_layer_forward_rmsnorm_swiglu_decode_q4_k(const CKLayerForwardParamsQ4K *p,
                                                 int token_index,
                                                 int cache_capacity);
void ck_layer_forward_rmsnorm_swiglu_quant(const CKLayerForwardParamsQ4K *p);
void ck_layer_forward_rmsnorm_swiglu_decode_quant(const CKLayerForwardParamsQ4K *p,
                                                  int token_index,
                                                  int cache_capacity);

void ck_residual_add_backward(const float *d_out,
                              float *d_a,
                              float *d_b,
                              int tokens,
                              int aligned_embed_dim);

void ck_attention_project_head_major_backward(const float *d_out,
                                              const float *attn_out,
                                              const float *wo,
                                              float *d_attn_out,
                                              float *d_wo,
                                              float *d_bo,
                                              int tokens,
                                              int aligned_embed_dim,
                                              int num_heads,
                                              int aligned_head_dim);

void ck_qkv_project_head_major_backward(const float *d_q,
                                        const float *d_k,
                                        const float *d_v,
                                        const float *input,
                                        const float *wq,
                                        const float *bq,
                                        const float *wk,
                                        const float *bk,
                                        const float *wv,
                                        const float *bv,
                                        float *d_input,
                                        float *d_wq,
                                        float *d_bq,
                                        float *d_wk,
                                        float *d_bk,
                                        float *d_wv,
                                        float *d_bv,
                                        float *scratch,
                                        int tokens,
                                        int aligned_embed_dim,
                                        int num_heads,
                                        int num_kv_heads,
                                        int aligned_head_dim,
                                        int num_threads);

void ck_layer_backward_rmsnorm_swiglu(const CKLayerBackwardParams *p);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* CKERNEL_ORCHESTRATION_H */
