/**
 * @file ck_parallel_prefill_v8.h
 * @brief Thread-pool-parallel GEMM dispatch for v8 prefill
 *
 * Provides parallel wrappers for GEMM kernels used in generated prefill code.
 * These functions use ck_threadpool to split GEMM row work (M dimension)
 * across threads. Each thread computes rows [r0, r1) of the output matrix C.
 *
 * Integration:
 *   1. #include "ck_parallel_prefill_v8.h" in ck-kernel-inference.c
 *   2. Call ck_parallel_prefill_init() in ck_model_init()
 *   3. Call ck_parallel_prefill_shutdown() in ck_model_free()
 *   4. GEMM calls are automatically parallelized via macro override
 *
 * All GEMM kernels share the same 7-arg signature:
 *   gemm_nt_<type>(A, B, bias, C, M, N, K)
 *
 * Kernel types:
 *   - gemm_nt_q5_0_q8_0: Q8_0 activations x Q5_0 weights (Q/K proj, out proj, MLP gate+up)
 *   - gemm_nt_q8_0_q8_0: Q8_0 activations x Q8_0 weights (V proj)
 *   - gemm_nt_q4_k_q8_k: Q8_K activations x Q4_K weights (Q/K/out/MLP proj)
 *   - gemm_nt_q6_k_q8_k: Q8_K activations x Q6_K weights (MLP down proj)
 *   - gemm_nt_q5_1_q8_1: FP32 activations x Q5_1 weights
 *   - gemm_nt_q5_k:      FP32 activations x Q5_K weights
 */

#ifndef CK_PARALLEL_PREFILL_V8_H
#define CK_PARALLEL_PREFILL_V8_H

#include "ck_threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Lifecycle */
void ck_parallel_prefill_init(void);
void ck_parallel_prefill_shutdown(void);

/** Release provider caches owned by a completed prefill phase before decode. */
void ck_parallel_prefill_release_transient_caches(void);

/* Exact row-parallel providers for prefill operations whose canonical
 * arithmetic is serial within each independent row. */
void quantize_batch_q8_k_4row_nearest_even_parallel_dispatch(
    const float *x, void *y, int num_rows, int k);
void rmsnorm_forward_llama_production_parallel_dispatch(
    const float *input, const float *gamma, float *output, float *rstd_cache,
    int tokens, int d_model, int aligned_embed_dim, float eps);
void recurrent_norm_gate_llama_avx2_parallel_dispatch(
    const float *x, const float *gate, const float *weight, float *out,
    int rows, int num_heads, int head_dim, float eps);

/** Release lazily repacked Q4_K weights at model/test teardown. */
void ck_q4k_packed_weight_cache_clear(void);

/**
 * Prepare one Q4_K weight for the AVX-512 VNNI x16 prefill provider.
 *
 * Returns 1 when the weight is eligible and resident in the cache, otherwise
 * 0. Generated initialization uses this to keep repacking out of the first
 * measured prompt without changing the on-disk weight format.
 */
int ck_q4k_prepare_vnni_x16_weight(const void *B, int N, int K);
int ck_q4k_prepare_vnni_x8_weight(const void *B, int N, int K);
int ck_moe_prepare_q4k_gate_up_vnni_x8(
    const void *gate,
    const void *up,
    int intermediate_dim,
    int hidden_dim,
    int n_experts);
int ck_q5_0_prepare_q8_0_weight(const void *B, int N, int K);
int ck_q5_k_prepare_expanded_weight(const void *B, int N, int K);
int ck_q6_k_prepare_expanded_weight(const void *B, int N, int K);

int moe_swiglu_expert_forward_q4k_q5k_auto_prepared_workspace(
    const float *hidden,
    const int *indices,
    const float *routing_weights,
    const void *expert_gate,
    const void *expert_up,
    const void *expert_down,
    float *output,
    int rows,
    int hidden_dim,
    int intermediate_dim,
    int n_experts,
    int top_k,
    void *workspace,
    size_t workspace_bytes);

void gated_deltanet_llama_prefill_parallel_dispatch(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int rows, int num_heads, int group_count, int state_dim, float norm_eps);

/**
 * Execute llama.cpp's 64-token DeltaNet prefill order while assigning
 * independent recurrent heads to the shared thread pool. Unlike the legacy
 * experimental dispatcher, selecting this provider is sufficient; it does
 * not depend on an environment switch.
 */
void gated_deltanet_llama_chunk64_prefill_parallel_dispatch(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int rows, int num_heads, int group_count, int state_dim, float norm_eps);

/* Parallel GEMM Wrappers - same signatures as serial GEMM functions */
void gemm_nt_q5_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q8_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q8_0_q8_0_contract_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q4_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

/* Preserve independent row-group reduction boundaries across a logical
 * segmented prefix.  Segment lengths are runtime data (for example text,
 * vision, text), and must sum to M.  Invalid plans fail closed to the
 * ordinary unified provider. */
void gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K, const int *segment_lengths, int num_segments);

void gemm_nt_q6_k_q8_k_segmented_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K, const int *segment_lengths, int num_segments);

void gemv_q4_k_q8_k_repacked_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int N, int K);

void gemm_nt_q4_k_q8_k_gateup_swiglu_x16_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int D, int K);

void gemm_nt_q6_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q5_1_q8_1_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void geglu_forward_exact_parallel_dispatch(
    const float *input, float *output, int tokens, int dim);

void gemm_nt_q5_k_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_f32_llama_production_parallel_dispatch(
    const float *A, const float *B, const float *bias, float *C,
    int M, int N, int K);

void gemm_nt_q5_k_parallel_dispatch_with_scratch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K, void *scratch, size_t scratch_bytes);

/* Macro overrides - when CK_PARALLEL_PREFILL is defined,
 * preprocessor redirects serial gemm_nt_*() calls to thread pool dispatch */
#ifdef CK_PARALLEL_PREFILL

#define gemm_nt_q5_0_q8_0(A, B, bias, C, M, N, K) \
    gemm_nt_q5_0_q8_0_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q8_0_q8_0(A, B, bias, C, M, N, K) \
    gemm_nt_q8_0_q8_0_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q8_0_q8_0_contract(A, B, bias, C, M, N, K) \
    gemm_nt_q8_0_q8_0_contract_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q4_k_q8_k(A, B, bias, C, M, N, K) \
    gemm_nt_q4_k_q8_k_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q6_k_q8_k(A, B, bias, C, M, N, K) \
    gemm_nt_q6_k_q8_k_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q5_1_q8_1(A, B, bias, C, M, N, K) \
    gemm_nt_q5_1_q8_1_parallel_dispatch(A, B, bias, C, M, N, K)

#define gemm_nt_q5_k(A, B, bias, C, M, N, K) \
    gemm_nt_q5_k_parallel_dispatch(A, B, bias, C, M, N, K)

#define gated_deltanet_llama_avx2_prefill_forward( \
        q, k, v, g, beta, state_in, state_out, out, \
        rows, num_heads, group_count, state_dim, norm_eps) \
    gated_deltanet_llama_prefill_parallel_dispatch( \
        q, k, v, g, beta, state_in, state_out, out, \
        rows, num_heads, group_count, state_dim, norm_eps)

#endif /* CK_PARALLEL_PREFILL */

#ifdef __cplusplus
}
#endif

#endif /* CK_PARALLEL_PREFILL_V8_H */
