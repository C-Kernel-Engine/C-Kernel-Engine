/**
 * @file ck_parallel_prefill.c
 * @brief Thread-pool-parallel GEMM dispatch for v7 prefill
 *
 * Wraps each GEMM kernel call in a threadpool dispatch, splitting work
 * across the M (tokens) dimension. Each thread processes rows [r0, r1):
 *
 *   int dr = (M + nth - 1) / nth;
 *   int r0 = dr * ith;
 *   int r1 = min(r0 + dr, M);
 *   serial_gemm(A + r0 * A_row_bytes, B, bias, C + r0 * N, r1-r0, N, K);
 *
 * B (weights) and bias are shared read-only across all threads.
 *
 * Fast path: if M <= 1 or pool has <= 1 thread, calls serial directly.
 *
 * Q6_K x Q8_K prefill scheduling note:
 * ------------------------------------
 * The 2D scheduler is a load-balancing tool, not a universally faster
 * Q6 kernel. Row splitting reuses each Q8_K activation row across the full N
 * output dimension. Splitting N into tiles creates more independent jobs, but
 * rereads the same activation tile once per N tile and adds scheduler work.
 *
 * Local i7 roofline-style sweeps on 2026-06-09 showed:
 *   - Qwen2-like MLP-down (N=896, K=4864): 2D was slower through M=256 and
 *     only +1.4% at M=512.
 *   - Nanbeige/large-Q6-like MLP-down (N=2560, K=10240): 2D was faster from
 *     M=16 onward, +5% to +23% depending on M.
 *
 * Production therefore follows the kernel-map shape contract: wide Q6 shapes
 * use output tiles, while narrow shapes retain independent-row scheduling.
 * CK_FORCE_Q6K_Q8K_2D_PREFILL and CK_DISABLE_Q6K_Q8K_2D_PREFILL are benchmark
 * controls only; generated model code does not need either flag.
 *
 * Reuses the same global thread pool as decode (ck_threadpool_global()).
 */

#include "ck_parallel_prefill_v8.h"
#include "ck_threadpool.h"
#include "ckernel_quant.h"
#include "ck_speed_profiles.h"

#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

enum {
    CK_GEMM_ROUTE_OUTPUT_TILES = 1 << 0,
    CK_GEMM_ROUTE_COMPACT_M4 = 1 << 1,
    CK_GEMM_ROUTE_BATCHED_TAIL = 1 << 2,
};

typedef struct {
    int min_m;
    int max_m;
    int min_n;
    int max_n;
    int min_k;
    int max_k;
    int tile_m;
    int tile_n;
    int max_threads;
    unsigned int flags;
} ck_gemm_route_v8;

#include "ck_kernel_dispatch_policy_v8.inc"

static const ck_gemm_route_v8 *ck_find_gemm_route_v8(
        const ck_gemm_route_v8 *routes, size_t route_count,
        int M, int N, int K)
{
    if (!routes || M <= 0 || N <= 0 || K <= 0) return NULL;
    for (size_t index = 0; index < route_count; ++index) {
        const ck_gemm_route_v8 *route = &routes[index];
        if (M >= route->min_m && M <= route->max_m &&
            N >= route->min_n && N <= route->max_n &&
            K >= route->min_k && K <= route->max_k) {
            return route;
        }
    }
    return NULL;
}

/* Serial GEMM kernels (defined in src/kernels/) */
extern void gemm_nt_q5_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q5_0_q8_0_m2n4(const void *A, const void *B,
                                    const float *bias, float *C,
                                    int M, int N, int K);
extern void gemm_nt_q5_0_q8_0_m2n4_tile(const void *A, const void *B,
                                         const float *bias, float *C,
                                         int M, int N, int K, int ldc);
extern void gemm_nt_q5_0_q8_0_m4n2(const void *A, const void *B,
                                    const float *bias, float *C,
                                    int M, int N, int K);
extern void gemm_nt_q5_0_q8_0_m4n2_tile(const void *A, const void *B,
                                         const float *bias, float *C,
                                         int M, int N, int K, int ldc);
extern void gemm_nt_q8_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q8_0_q8_0_m2n4(const void *A, const void *B,
                                     const float *bias, float *C,
                                     int M, int N, int K);
extern void gemm_nt_q8_0_q8_0_m2n4_tile(const void *A, const void *B,
                                         const float *bias, float *C,
                                         int M, int N, int K, int ldc);
extern void gemm_nt_q4_k_q8_k(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemv_q4_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern size_t q4_k_packed_meta_block_size(void);
extern size_t q4_k_packed_meta_x8_block_size(void);
extern size_t q4_k_packed_meta_x16_block_size(void);
extern void pack_q4_k_to_packed_meta(const void *src, void *dst, int N, int K);
extern void pack_q4_k_to_packed_meta_x8(const void *src, void *dst, int N, int K);
extern void pack_q4_k_to_packed_meta_x16(const void *src, void *dst, int N, int K);
extern void gemm_nt_q4_k_packed_meta_q8_k_threaded(const void *A_q8,
                                                    const void *B_packed,
                                                    const float *bias,
                                                    float *C,
                                                    int M, int N, int K,
                                                    int threads);
extern void gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(const void *A_q8,
                                                          const void *B_packed,
                                                          const float *bias,
                                                          float *C,
                                                          int M, int N, int K,
                                                          int threads);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_nsplit(const void *A_q8,
                                                             const void *B_packed_x8,
                                                             const float *bias,
                                                             float *C,
                                                             int M, int N, int K,
                                                             int threads);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_mtile(const void *A_q8,
                                                            const void *B_packed_x8,
                                                            const float *bias,
                                                            float *C,
                                                            int M, int N, int K,
                                                            int tile_m,
                                                            int threads);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_mreuse(const void *A_q8,
                                                             const void *B_packed_x8,
                                                             const float *bias,
                                                             float *C,
                                                             int M, int N, int K,
                                                             int tile_m,
                                                             int threads);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_mreuse(
    const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
    int M, int N, int K, int tile_m, int threads);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_4m(
    const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
    int M, int N, int K, int threads);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_8m(
    const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
    int M, int N, int K, int threads);
extern size_t q4_k_packed_vnni_x8_block_size(void);
extern int ck_q4k_packed_vnni_x8_available(void);
extern int ck_q4k_packed_vnni_x8_compact_order_available(void);
extern void pack_q4_k_to_packed_vnni_x8(
    const void *src, void *dst, int N, int K);
extern void gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m(
    const void *A_q8, const void *B_packed_vnni_x8, const float *bias,
    float *C, int M, int N, int K, int threads);
extern size_t q4_k_packed_vnni_x16_block_size(void);
extern int ck_q4k_packed_vnni_x16_available(void);
extern void pack_q4_k_to_packed_vnni_x16(
    const void *src, void *dst, int N, int K);
extern void gemm_nt_q4_k_packed_vnni_x16_q8_k_split_min_threaded_16m(
    const void *A_q8, const void *B_packed_vnni_x16, const float *bias,
    float *C, int M, int N, int K, int threads);
extern void gemm_nt_q4_k_packed_vnni_x16_q8_k_gemv_order(
    const void *A_q8, const void *B_packed_vnni_x16,
    const float *bias, float *C, int M, int N, int K);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_superblock_order(
    const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
    int M, int N, int K);
extern void gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
    const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
    int M, int N, int K);
extern void gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order(
    const void *A_q8, const void *B_packed_x8, const float *bias, float *C,
    int M, int N, int K);
extern void gemm_nt_q4_k_packed_meta_x16_q8_k_threaded_mreuse(const void *A_q8,
                                                              const void *B_packed_x16,
                                                              const float *bias,
                                                              float *C,
                                                              int M, int N, int K,
                                                              int tile_m,
                                                              int active_threads);
extern void gemm_nt_q4_k_packed_meta_x16_q8_k_threaded_mtile(const void *A_q8,
                                                             const void *B_packed_x16,
                                                             const float *bias,
                                                             float *C,
                                                             int M, int N, int K,
                                                             int tile_m,
                                                             int active_threads);
extern void gemm_nt_q4_k_packed_meta_x16_gateup_swiglu_fused_vnni(const void *A_q8,
                                                                   const void *B_packed_x16,
                                                                   const float *bias,
                                                                   float *C,
                                                                   int M, int D, int K,
                                                                   int tile_m,
                                                                   int active_threads);
extern int moe_swiglu_expert_forward_q4k_q5k_parallel_workspace(
    const float *hidden, const int *indices, const float *routing_weights,
    const void *expert_gate, const void *expert_up, const void *expert_down,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int n_experts, int top_k, void *workspace, size_t workspace_bytes);
extern int moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace(
    const float *hidden, const int *indices, const float *routing_weights,
    const void *expert_gate, const void *expert_up, const void *expert_down,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int n_experts, int top_k, void *workspace, size_t workspace_bytes);
extern int moe_swiglu_expert_forward_q4k_q5k_bucketed_prepared_workspace(
    const float *hidden, const int *indices, const float *routing_weights,
    const void *expert_gate, const void *expert_up, const void *expert_down,
    const void *expert_gate_packed, const void *expert_up_packed,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int n_experts, int top_k, void *workspace, size_t workspace_bytes);
extern void gemm_nt_q4_k_packed_meta_q8_k_tile(const void *A_q8,
                                               const void *B_packed,
                                               const float *bias,
                                               float *C,
                                               int M, int N, int K,
                                               int m0, int m1, int n0, int n1);
extern void gemm_nt_q6_k_q8_k(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q8_0_q8_0_contract(const float *A, const void *B,
                                        const float *bias, float *C,
                                        int M, int N, int K);
extern void gemv_q8_0_q8_0_contract(float *y, const void *W,
                                     const float *x, int M, int K);
extern int ck_strict_parity_enabled(void);
extern void swiglu_forward_exact(const float *input, float *output, int tokens, int dim);
extern void geglu_forward_exact(const float *input, float *output, int tokens, int dim);
extern void gemm_nt_q6_k_q8_k_tile(const void *A, const void *B, const float *bias,
                                    float *C, int M, int N, int K,
                                    int m0, int m1, int n0, int n1);
extern void gemm_nt_q6_k_q8_k_m4_tile(const void *A, const void *B, const float *bias,
                                      float *C, int M, int N, int K,
                                      int m0, int m1, int n0, int n1);
extern void gemm_nt_q6_k_q8_k_tiled(const void *A, const void *B, const float *bias,
                                      float *C, int M, int N, int K);
extern size_t ck_q6_k_prepared_block_size(void);
extern void ck_q6_k_prepare_weight(const void *src, void *dst, int N, int K);
extern void gemm_nt_q6_k_q8_k_prepared(const void *A, const void *B_prepared,
                                        const float *bias, float *C,
                                        int M, int N, int K);
extern void gemm_nt_q6_k_q8_k_prepared_tile(
    const void *A, const void *B_prepared, const float *bias, float *C,
    int M, int N, int K, int m0, int m1, int n0, int n1);
extern void gemm_nt_q5_1_q8_1(const float *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q5_1_q8_1_m4(const float *A, const void *B, const float *bias,
                                   float *C, int M, int N, int K);
extern void gemm_nt_q5_1_q8_1_m8(const float *A, const void *B, const float *bias,
                                   float *C, int M, int N, int K);
extern void gemm_nt_q5_k(const float *A, const void *B, const float *bias,
                          float *C, int M, int N, int K);
extern size_t ck_q5_k_prepared_block_size(void);
extern void ck_q5_k_prepare_weight(const void *src, void *dst, int N, int K);
extern void gemm_nt_q5_k_prepared(const float *A, const void *B_prepared,
                                  const float *bias, float *C,
                                  int M, int N, int K);
extern void gemm_nt_q5_k_prepared_m4(const float *A, const void *B_prepared,
                                     const float *bias, float *C,
                                     int M, int N, int K);
extern void quantize_row_q8_k(const float *x, void *y, int k);
extern void quantize_batch_q8_k_4row_nearest_even(
    const float *x, void *y, int num_rows, int k);
extern void rmsnorm_forward_llama_production(
    const float *input, const float *gamma, float *output, float *rstd_cache,
    int tokens, int d_model, int aligned_embed_dim, float eps);
extern void recurrent_norm_gate_llama_avx2_forward(
    const float *x, const float *gate, const float *weight, float *out,
    int rows, int num_heads, int head_dim, float eps);
extern void gemm_nt_q5_k_prepared_q8_m4_nrange(
    const void *A_q8, const void *B_prepared, const float *bias, float *C,
    int M, int N, int K, int n_begin, int n_end);
extern void gated_deltanet_llama_avx2_prefill_forward(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int rows, int num_heads, int group_count, int state_dim, float norm_eps);
extern void gated_deltanet_llama_avx2_forward_head_range(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int num_heads, int group_count, int state_dim, float norm_eps,
    int head_begin, int head_end);
extern void gated_deltanet_llama_chunk64_head_forward(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int rows, int num_heads, int group_count, int head, int state_dim);
extern void gated_deltanet_llama_chunk64_prefill_forward(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int rows, int num_heads, int group_count, int state_dim, float norm_eps);

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

void ck_parallel_prefill_init(void)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (pool) {
        fprintf(stderr, "[CK parallel prefill] Initialized with %d threads\n",
                ck_threadpool_n_threads(pool));
    }
}

/* ============================================================================
 * Argument Packing Struct
 * ============================================================================ */

typedef struct {
    const void  *A;           /* Input activations (quantized) */
    const void  *B;           /* Weight matrix (quantized, read-only) */
    const float *bias;        /* Optional bias vector (read-only) */
    float       *C;           /* Output matrix [M x N] */
    int          M;           /* Number of tokens (rows to split) */
    int          N;           /* Output dimension */
    int          K;           /* Input dimension */
    size_t       A_row_bytes; /* Bytes per row of A (for pointer arithmetic) */
    int          tile_m;      /* 2D scheduler token tile height */
    int          tile_n;      /* 2D scheduler output tile width */
    int          use_q6_m4;   /* Reuse Q6 unpack across four token rows */
    int          use_q6_prepared; /* B is expanded Q6 integer metadata */
} gemm_args_t;

typedef struct {
    const void  *A;
    const void  *B_packed_x8;
    const float *bias;
    float       *C;
    int          M;
    int          N;
    int          K;
    size_t       A_row_bytes;
} q4k_repacked_gemv_args_t;

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    const float *g;
    const float *beta;
    const float *state_in;
    float *state_out;
    float *out;
    int rows;
    int num_heads;
    int group_count;
    int state_dim;
    float norm_eps;
} deltanet_prefill_args_t;

typedef struct {
    const float *input;
    float *output;
    int tokens;
    int dim;
} geglu_args_t;

typedef struct {
    const float *input;
    void *output;
    int rows;
    int k;
} q8_k_quantize_args_t;

typedef struct {
    const float *input;
    const float *gamma;
    float *output;
    float *rstd_cache;
    int tokens;
    int d_model;
    int aligned_embed_dim;
    float eps;
} rmsnorm_exact_args_t;

typedef struct {
    const float *x;
    const float *gate;
    const float *weight;
    float *out;
    int rows;
    int num_heads;
    int head_dim;
    float eps;
} recurrent_norm_gate_args_t;

static int ck_min_int(int a, int b) { return a < b ? a : b; }
static int ck_env_enabled(const char *name);

static int ck_independent_row_active_threads(
        ck_threadpool_t *pool, int rows, int grain_size)
{
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    const int jobs = (rows + grain_size - 1) / grain_size;
    if (active > jobs) active = jobs;
    return active > 0 ? active : 1;
}

static void work_quantize_q8_k_rows(int begin, int end, void *userdata)
{
    q8_k_quantize_args_t *args = (q8_k_quantize_args_t *)userdata;
    const size_t input_row_elems = (size_t)args->k;
    const size_t output_row_bytes =
        (size_t)(args->k / QK_K) * sizeof(block_q8_K);
    quantize_batch_q8_k_4row_nearest_even(
        args->input + (size_t)begin * input_row_elems,
        (unsigned char *)args->output + (size_t)begin * output_row_bytes,
        end - begin, args->k);
}

static void work_rmsnorm_exact_rows(int begin, int end, void *userdata)
{
    rmsnorm_exact_args_t *args = (rmsnorm_exact_args_t *)userdata;
    const size_t offset = (size_t)begin * (size_t)args->aligned_embed_dim;
    rmsnorm_forward_llama_production(
        args->input + offset, args->gamma, args->output + offset,
        args->rstd_cache ? args->rstd_cache + begin : NULL,
        end - begin, args->d_model, args->aligned_embed_dim, args->eps);
}

static void work_recurrent_norm_gate_rows(int begin, int end, void *userdata)
{
    recurrent_norm_gate_args_t *args =
        (recurrent_norm_gate_args_t *)userdata;
    const size_t row_elems =
        (size_t)args->num_heads * (size_t)args->head_dim;
    const size_t offset = (size_t)begin * row_elems;
    recurrent_norm_gate_llama_avx2_forward(
        args->x + offset, args->gate + offset, args->weight,
        args->out + offset, end - begin, args->num_heads,
        args->head_dim, args->eps);
}

void quantize_batch_q8_k_4row_nearest_even_parallel_dispatch(
    const float *x, void *y, int num_rows, int k)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    const int grain = 16;
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || num_rows < grain * 2 ||
        k <= 0 || (k % QK_K) != 0) {
        quantize_batch_q8_k_4row_nearest_even(x, y, num_rows, k);
        return;
    }

    q8_k_quantize_args_t args = {
        .input = x,
        .output = y,
        .rows = num_rows,
        .k = k,
    };
    ck_threadpool_parallel_for_n(
        pool, ck_independent_row_active_threads(pool, num_rows, grain),
        0, num_rows, grain, work_quantize_q8_k_rows, &args);
}

void rmsnorm_forward_llama_production_parallel_dispatch(
    const float *input, const float *gamma, float *output, float *rstd_cache,
    int tokens, int d_model, int aligned_embed_dim, float eps)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    const int grain = 8;
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || tokens < grain * 2 ||
        d_model <= 0 || aligned_embed_dim < d_model) {
        rmsnorm_forward_llama_production(
            input, gamma, output, rstd_cache,
            tokens, d_model, aligned_embed_dim, eps);
        return;
    }

    rmsnorm_exact_args_t args = {
        .input = input,
        .gamma = gamma,
        .output = output,
        .rstd_cache = rstd_cache,
        .tokens = tokens,
        .d_model = d_model,
        .aligned_embed_dim = aligned_embed_dim,
        .eps = eps,
    };
    ck_threadpool_parallel_for_n(
        pool, ck_independent_row_active_threads(pool, tokens, grain),
        0, tokens, grain, work_rmsnorm_exact_rows, &args);
}

void recurrent_norm_gate_llama_avx2_parallel_dispatch(
    const float *x, const float *gate, const float *weight, float *out,
    int rows, int num_heads, int head_dim, float eps)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    const int grain = 4;
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || rows < grain * 2 ||
        num_heads <= 0 || head_dim <= 0) {
        recurrent_norm_gate_llama_avx2_forward(
            x, gate, weight, out, rows, num_heads, head_dim, eps);
        return;
    }

    recurrent_norm_gate_args_t args = {
        .x = x,
        .gate = gate,
        .weight = weight,
        .out = out,
        .rows = rows,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .eps = eps,
    };
    ck_threadpool_parallel_for_n(
        pool, ck_independent_row_active_threads(pool, rows, grain),
        0, rows, grain, work_recurrent_norm_gate_rows, &args);
}

static int ck_deltanet_chunk64_available(void)
{
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

static void work_deltanet_chunk64_heads(int ith, int nth, void *userdata)
{
    deltanet_prefill_args_t *args = (deltanet_prefill_args_t *)userdata;
    for (int head = ith; head < args->num_heads; head += nth) {
        gated_deltanet_llama_chunk64_head_forward(
            args->q, args->k, args->v, args->g, args->beta,
            args->state_in, args->state_out, args->out,
            args->rows, args->num_heads, args->group_count,
            head, args->state_dim);
    }
}

static void work_deltanet_chunk64_head_range(int begin, int end, void *userdata)
{
    deltanet_prefill_args_t *args = (deltanet_prefill_args_t *)userdata;
    if (!args || begin < 0 || begin >= end || end > args->num_heads) return;

    for (int head = begin; head < end; ++head) {
        gated_deltanet_llama_chunk64_head_forward(
            args->q, args->k, args->v, args->g, args->beta,
            args->state_in, args->state_out, args->out,
            args->rows, args->num_heads, args->group_count,
            head, args->state_dim);
    }
}

/*
 * Preserve the llama.cpp fused recurrent arithmetic while amortizing one
 * thread-pool dispatch across the whole prompt.  Every worker owns disjoint
 * heads and advances those heads through all rows in order, so there are no
 * cross-worker state dependencies and each per-head reduction tree is
 * identical to gated_deltanet_llama_avx2_prefill_forward().
 */
static void work_deltanet_exact_prefill_heads(int ith, int nth, void *userdata)
{
    deltanet_prefill_args_t *args = (deltanet_prefill_args_t *)userdata;
    const int head_begin = (args->num_heads * ith) / nth;
    const int head_end = (args->num_heads * (ith + 1)) / nth;
    const size_t qk_stride =
        (size_t)args->group_count * (size_t)args->state_dim;
    const size_t value_stride =
        (size_t)args->num_heads * (size_t)args->state_dim;
    const size_t gate_stride = (size_t)args->num_heads;

    for (int row = 0; row < args->rows; ++row) {
        gated_deltanet_llama_avx2_forward_head_range(
            args->q + (size_t)row * qk_stride,
            args->k + (size_t)row * qk_stride,
            args->v + (size_t)row * value_stride,
            args->g + (size_t)row * gate_stride,
            args->beta + (size_t)row * gate_stride,
            row == 0 ? args->state_in : args->state_out,
            args->state_out,
            args->out + (size_t)row * value_stride,
            args->num_heads, args->group_count, args->state_dim,
            args->norm_eps, head_begin, head_end);
    }
}

void gated_deltanet_llama_chunk64_prefill_parallel_dispatch(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int rows, int num_heads, int group_count, int state_dim, float norm_eps)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 ||
        rows <= 1 || num_heads <= 1 || group_count <= 0 ||
        num_heads % group_count != 0 || state_dim <= 0 || state_dim > 256 ||
        !ck_deltanet_chunk64_available()) {
        gated_deltanet_llama_chunk64_prefill_forward(
            q, k, v, g, beta, state_in, state_out, out,
            rows, num_heads, group_count, state_dim, norm_eps);
        return;
    }

    deltanet_prefill_args_t args = {
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .state_in = state_in,
        .state_out = state_out,
        .out = out,
        .rows = rows,
        .num_heads = num_heads,
        .group_count = group_count,
        .state_dim = state_dim,
        .norm_eps = norm_eps,
    };
    int active = ck_threadpool_n_threads(pool);
    if (rows <= 128) {
        const int paired_head_workers = (num_heads + 1) / 2;
        if (active > paired_head_workers) active = paired_head_workers;
    } else if (active > num_heads) {
        active = num_heads;
    }
    ck_threadpool_parallel_for_n(
        pool, active, 0, num_heads, 1,
        work_deltanet_chunk64_head_range, &args);
}

void gated_deltanet_llama_prefill_parallel_dispatch(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int rows, int num_heads, int group_count, int state_dim, float norm_eps)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 ||
        rows <= 1 || num_heads <= 1 || group_count <= 0 ||
        num_heads % group_count != 0 || state_dim <= 0 || state_dim > 256) {
        gated_deltanet_llama_avx2_prefill_forward(
            q, k, v, g, beta, state_in, state_out, out,
            rows, num_heads, group_count, state_dim, norm_eps);
        return;
    }

    deltanet_prefill_args_t args = {
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .state_in = state_in,
        .state_out = state_out,
        .out = out,
        .rows = rows,
        .num_heads = num_heads,
        .group_count = group_count,
        .state_dim = state_dim,
        .norm_eps = norm_eps,
    };
    int active = ck_threadpool_n_threads(pool);
    if (active > num_heads) active = num_heads;
    if (ck_env_enabled("CK_ENABLE_DELTANET_CHUNK64_PREFILL") &&
        ck_deltanet_chunk64_available()) {
        ck_threadpool_dispatch_n(
            pool, active, work_deltanet_chunk64_heads, &args);
    } else {
        ck_threadpool_dispatch_n(
            pool, active, work_deltanet_exact_prefill_heads, &args);
    }
}

static int ck_env_enabled(const char *name)
{
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static void ck_q4k_prefill_debug_dispatch(const char *path, int M, int N, int K, int active)
{
    if (!ck_env_enabled("CK_DEBUG_Q4K_PREFILL_DISPATCH")) return;
    fprintf(stderr, "[CK q4k prefill] path=%s M=%d N=%d K=%d active=%d\n",
            path, M, N, K, active);
}

static int ck_env_int_or2(const char *primary, const char *secondary, int fallback)
{
    const char *v = getenv(primary);
    if ((!v || !v[0]) && secondary) v = getenv(secondary);
    if (!v || !v[0]) return fallback;
    int parsed = atoi(v);
    return parsed > 0 ? parsed : fallback;
}

static int ck_ceil_div_int(int a, int b)
{
    return (a + b - 1) / b;
}

static int ck_select_gemm_active_threads(const ck_threadpool_t *pool, int M, int N, int K);

static int ck_q6k_q8k_2d_prefill_forced(void)
{
    return ck_env_enabled("CK_FORCE_Q6K_Q8K_2D_PREFILL");
}

typedef struct ck_q4k_packed_meta_cache_entry {
    const void *src;
    int N;
    int K;
    void *packed;
    struct ck_q4k_packed_meta_cache_entry *next;
} ck_q4k_packed_meta_cache_entry_t;

static pthread_mutex_t ck_q4k_packed_meta_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q4k_packed_meta_cache_entry_t *ck_q4k_packed_meta_cache_head = NULL;

typedef struct ck_q4k_packed_meta_x8_cache_entry {
    const void *src;
    int N;
    int K;
    void *packed;
    struct ck_q4k_packed_meta_x8_cache_entry *next;
} ck_q4k_packed_meta_x8_cache_entry_t;

static pthread_mutex_t ck_q4k_packed_meta_x8_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q4k_packed_meta_x8_cache_entry_t *ck_q4k_packed_meta_x8_cache_head = NULL;

typedef struct ck_q4k_packed_vnni_x8_cache_entry {
    const void *src;
    int N;
    int K;
    void *packed;
    struct ck_q4k_packed_vnni_x8_cache_entry *next;
} ck_q4k_packed_vnni_x8_cache_entry_t;

static pthread_mutex_t ck_q4k_packed_vnni_x8_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q4k_packed_vnni_x8_cache_entry_t *ck_q4k_packed_vnni_x8_cache_head = NULL;

typedef struct ck_q4k_packed_vnni_x16_cache_entry {
    const void *src;
    int N;
    int K;
    void *packed;
    struct ck_q4k_packed_vnni_x16_cache_entry *next;
} ck_q4k_packed_vnni_x16_cache_entry_t;

static pthread_mutex_t ck_q4k_packed_vnni_x16_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q4k_packed_vnni_x16_cache_entry_t *ck_q4k_packed_vnni_x16_cache_head = NULL;


typedef struct ck_q4k_packed_meta_x16_cache_entry {
    const void *src;
    int N;
    int K;
    void *packed;
    struct ck_q4k_packed_meta_x16_cache_entry *next;
} ck_q4k_packed_meta_x16_cache_entry_t;

static pthread_mutex_t ck_q4k_packed_meta_x16_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q4k_packed_meta_x16_cache_entry_t *ck_q4k_packed_meta_x16_cache_head = NULL;

typedef struct ck_q5_0_q8_0_cache_entry {
    const void *src;
    int N;
    int K;
    block_q8_0 *prepared;
    struct ck_q5_0_q8_0_cache_entry *next;
} ck_q5_0_q8_0_cache_entry_t;

static pthread_mutex_t ck_q5_0_q8_0_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q5_0_q8_0_cache_entry_t *ck_q5_0_q8_0_cache_head = NULL;

typedef struct ck_q5_k_prepared_cache_entry {
    const void *src;
    int N;
    int K;
    void *prepared;
    struct ck_q5_k_prepared_cache_entry *next;
} ck_q5_k_prepared_cache_entry_t;

static pthread_mutex_t ck_q5_k_prepared_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q5_k_prepared_cache_entry_t *ck_q5_k_prepared_cache_head = NULL;

typedef struct ck_q6_k_prepared_cache_entry {
    const void *src;
    int N;
    int K;
    void *prepared;
    struct ck_q6_k_prepared_cache_entry *next;
} ck_q6_k_prepared_cache_entry_t;

static pthread_mutex_t ck_q6_k_prepared_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q6_k_prepared_cache_entry_t *ck_q6_k_prepared_cache_head = NULL;

static void ck_q5_0_expand_exact_q8_0(
        const block_q5_0 *src, block_q8_0 *dst, size_t blocks)
{
    for (size_t block = 0; block < blocks; ++block) {
        uint32_t high_bits;
        memcpy(&high_bits, src[block].qh, sizeof(high_bits));
        dst[block].d = src[block].d;
        for (int lane = 0; lane < QK5_0 / 2; ++lane) {
            const uint8_t packed = src[block].qs[lane];
            dst[block].qs[lane] = (int8_t)(
                ((packed & 0x0f) | (((high_bits >> lane) & 1u) << 4)) - 16);
            dst[block].qs[lane + QK5_0 / 2] = (int8_t)(
                ((packed >> 4) | (((high_bits >> (lane + 16)) & 1u) << 4)) - 16);
        }
    }
}

static block_q8_0 *ck_find_prepared_q5_0_q8_0(
        const void *B, int N, int K)
{
    block_q8_0 *prepared = NULL;
    pthread_mutex_lock(&ck_q5_0_q8_0_cache_mu);
    for (ck_q5_0_q8_0_cache_entry_t *entry = ck_q5_0_q8_0_cache_head;
         entry; entry = entry->next) {
        if (entry->src == B && entry->N == N && entry->K == K) {
            prepared = entry->prepared;
            break;
        }
    }
    pthread_mutex_unlock(&ck_q5_0_q8_0_cache_mu);
    return prepared;
}

int ck_q5_0_prepare_q8_0_weight(const void *B, int N, int K)
{
#if !defined(__AVX2__)
    (void)B;
    (void)N;
    (void)K;
    return 0;
#else
    if (!B || N <= 0 || K <= 0 || (K % QK5_0) != 0) return 0;
    if (ck_find_prepared_q5_0_q8_0(B, N, K)) return 1;

    const size_t blocks_per_row = (size_t)K / QK5_0;
    if ((size_t)N > SIZE_MAX / blocks_per_row) return 0;
    const size_t blocks = (size_t)N * blocks_per_row;
    if (blocks > SIZE_MAX / sizeof(block_q8_0)) return 0;
    block_q8_0 *prepared = (block_q8_0 *)malloc(blocks * sizeof(*prepared));
    ck_q5_0_q8_0_cache_entry_t *entry =
        (ck_q5_0_q8_0_cache_entry_t *)malloc(sizeof(*entry));
    if (!prepared || !entry) {
        free(prepared);
        free(entry);
        return 0;
    }
    ck_q5_0_expand_exact_q8_0((const block_q5_0 *)B, prepared, blocks);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->prepared = prepared;

    pthread_mutex_lock(&ck_q5_0_q8_0_cache_mu);
    for (ck_q5_0_q8_0_cache_entry_t *existing = ck_q5_0_q8_0_cache_head;
         existing; existing = existing->next) {
        if (existing->src == B && existing->N == N && existing->K == K) {
            pthread_mutex_unlock(&ck_q5_0_q8_0_cache_mu);
            free(prepared);
            free(entry);
            return 1;
        }
    }
    entry->next = ck_q5_0_q8_0_cache_head;
    ck_q5_0_q8_0_cache_head = entry;
    pthread_mutex_unlock(&ck_q5_0_q8_0_cache_mu);
    return 1;
#endif
}

static void ck_q5_0_q8_0_cache_clear(void)
{
    pthread_mutex_lock(&ck_q5_0_q8_0_cache_mu);
    while (ck_q5_0_q8_0_cache_head) {
        ck_q5_0_q8_0_cache_entry_t *entry = ck_q5_0_q8_0_cache_head;
        ck_q5_0_q8_0_cache_head = entry->next;
        free(entry->prepared);
        free(entry);
    }
    pthread_mutex_unlock(&ck_q5_0_q8_0_cache_mu);
}

static void *ck_find_prepared_q5_k(const void *B, int N, int K)
{
    void *prepared = NULL;
    pthread_mutex_lock(&ck_q5_k_prepared_cache_mu);
    for (ck_q5_k_prepared_cache_entry_t *entry = ck_q5_k_prepared_cache_head;
         entry; entry = entry->next) {
        if (entry->src == B && entry->N == N && entry->K == K) {
            prepared = entry->prepared;
            break;
        }
    }
    pthread_mutex_unlock(&ck_q5_k_prepared_cache_mu);
    return prepared;
}

int ck_q5_k_prepare_expanded_weight(const void *B, int N, int K)
{
#if !defined(__AVX2__)
    (void)B; (void)N; (void)K;
    return 0;
#else
    if (!B || N <= 0 || K <= 0 || (K % 256) != 0) return 0;
    if (ck_find_prepared_q5_k(B, N, K)) return 1;
    const size_t block_size = ck_q5_k_prepared_block_size();
    const size_t blocks_per_row = (size_t)K / 256u;
    if ((size_t)N > SIZE_MAX / blocks_per_row) return 0;
    const size_t blocks = (size_t)N * blocks_per_row;
    if (block_size == 0 || blocks > SIZE_MAX / block_size) return 0;

    void *prepared = malloc(blocks * block_size);
    ck_q5_k_prepared_cache_entry_t *entry =
        (ck_q5_k_prepared_cache_entry_t *)malloc(sizeof(*entry));
    if (!prepared || !entry) {
        free(prepared);
        free(entry);
        return 0;
    }
    ck_q5_k_prepare_weight(B, prepared, N, K);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->prepared = prepared;

    pthread_mutex_lock(&ck_q5_k_prepared_cache_mu);
    for (ck_q5_k_prepared_cache_entry_t *existing = ck_q5_k_prepared_cache_head;
         existing; existing = existing->next) {
        if (existing->src == B && existing->N == N && existing->K == K) {
            pthread_mutex_unlock(&ck_q5_k_prepared_cache_mu);
            free(prepared);
            free(entry);
            return 1;
        }
    }
    entry->next = ck_q5_k_prepared_cache_head;
    ck_q5_k_prepared_cache_head = entry;
    pthread_mutex_unlock(&ck_q5_k_prepared_cache_mu);
    return 1;
#endif
}

static void ck_q5_k_prepared_cache_clear(void)
{
    pthread_mutex_lock(&ck_q5_k_prepared_cache_mu);
    while (ck_q5_k_prepared_cache_head) {
        ck_q5_k_prepared_cache_entry_t *entry = ck_q5_k_prepared_cache_head;
        ck_q5_k_prepared_cache_head = entry->next;
        free(entry->prepared);
        free(entry);
    }
    pthread_mutex_unlock(&ck_q5_k_prepared_cache_mu);
}

static void *ck_find_prepared_q6_k(const void *B, int N, int K)
{
    void *prepared = NULL;
    pthread_mutex_lock(&ck_q6_k_prepared_cache_mu);
    for (ck_q6_k_prepared_cache_entry_t *entry = ck_q6_k_prepared_cache_head;
         entry; entry = entry->next) {
        if (entry->src == B && entry->N == N && entry->K == K) {
            prepared = entry->prepared;
            break;
        }
    }
    pthread_mutex_unlock(&ck_q6_k_prepared_cache_mu);
    return prepared;
}

int ck_q6_k_prepare_expanded_weight(const void *B, int N, int K)
{
#if !defined(__AVX2__)
    (void)B; (void)N; (void)K;
    return 0;
#else
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0) return 0;
    if (ck_find_prepared_q6_k(B, N, K)) return 1;
    const size_t block_size = ck_q6_k_prepared_block_size();
    const size_t blocks_per_row = (size_t)K / QK_K;
    if ((size_t)N > SIZE_MAX / blocks_per_row) return 0;
    const size_t blocks = (size_t)N * blocks_per_row;
    if (block_size == 0 || blocks > SIZE_MAX / block_size) return 0;

    void *prepared = malloc(blocks * block_size);
    ck_q6_k_prepared_cache_entry_t *entry =
        (ck_q6_k_prepared_cache_entry_t *)malloc(sizeof(*entry));
    if (!prepared || !entry) {
        free(prepared);
        free(entry);
        return 0;
    }
    ck_q6_k_prepare_weight(B, prepared, N, K);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->prepared = prepared;

    pthread_mutex_lock(&ck_q6_k_prepared_cache_mu);
    for (ck_q6_k_prepared_cache_entry_t *existing = ck_q6_k_prepared_cache_head;
         existing; existing = existing->next) {
        if (existing->src == B && existing->N == N && existing->K == K) {
            pthread_mutex_unlock(&ck_q6_k_prepared_cache_mu);
            free(prepared);
            free(entry);
            return 1;
        }
    }
    entry->next = ck_q6_k_prepared_cache_head;
    ck_q6_k_prepared_cache_head = entry;
    pthread_mutex_unlock(&ck_q6_k_prepared_cache_mu);
    return 1;
#endif
}

static void ck_q6_k_prepared_cache_clear(void)
{
    pthread_mutex_lock(&ck_q6_k_prepared_cache_mu);
    while (ck_q6_k_prepared_cache_head) {
        ck_q6_k_prepared_cache_entry_t *entry = ck_q6_k_prepared_cache_head;
        ck_q6_k_prepared_cache_head = entry->next;
        free(entry->prepared);
        free(entry);
    }
    pthread_mutex_unlock(&ck_q6_k_prepared_cache_mu);
}

void ck_q4k_packed_weight_cache_clear(void)
{
    pthread_mutex_lock(&ck_q4k_packed_meta_cache_mu);
    while (ck_q4k_packed_meta_cache_head) {
        ck_q4k_packed_meta_cache_entry_t *entry = ck_q4k_packed_meta_cache_head;
        ck_q4k_packed_meta_cache_head = entry->next;
        free(entry->packed);
        free(entry);
    }
    pthread_mutex_unlock(&ck_q4k_packed_meta_cache_mu);

    pthread_mutex_lock(&ck_q4k_packed_meta_x8_cache_mu);
    while (ck_q4k_packed_meta_x8_cache_head) {
        ck_q4k_packed_meta_x8_cache_entry_t *entry = ck_q4k_packed_meta_x8_cache_head;
        ck_q4k_packed_meta_x8_cache_head = entry->next;
        free(entry->packed);
        free(entry);
    }
    pthread_mutex_unlock(&ck_q4k_packed_meta_x8_cache_mu);

    pthread_mutex_lock(&ck_q4k_packed_vnni_x8_cache_mu);
    while (ck_q4k_packed_vnni_x8_cache_head) {
        ck_q4k_packed_vnni_x8_cache_entry_t *entry =
                ck_q4k_packed_vnni_x8_cache_head;
        ck_q4k_packed_vnni_x8_cache_head = entry->next;
        free(entry->packed);
        free(entry);
    }
    pthread_mutex_unlock(&ck_q4k_packed_vnni_x8_cache_mu);

    pthread_mutex_lock(&ck_q4k_packed_vnni_x16_cache_mu);
    while (ck_q4k_packed_vnni_x16_cache_head) {
        ck_q4k_packed_vnni_x16_cache_entry_t *entry =
                ck_q4k_packed_vnni_x16_cache_head;
        ck_q4k_packed_vnni_x16_cache_head = entry->next;
        free(entry->packed);
        free(entry);
    }
    pthread_mutex_unlock(&ck_q4k_packed_vnni_x16_cache_mu);

    pthread_mutex_lock(&ck_q4k_packed_meta_x16_cache_mu);
    while (ck_q4k_packed_meta_x16_cache_head) {
        ck_q4k_packed_meta_x16_cache_entry_t *entry = ck_q4k_packed_meta_x16_cache_head;
        ck_q4k_packed_meta_x16_cache_head = entry->next;
        free(entry->packed);
        free(entry);
    }
    pthread_mutex_unlock(&ck_q4k_packed_meta_x16_cache_mu);
}

void ck_parallel_prefill_shutdown(void)
{
    /* Pool ownership remains with decode; prefill owns packed-weight caches. */
    ck_q4k_packed_weight_cache_clear();
    ck_q5_0_q8_0_cache_clear();
    ck_q5_k_prepared_cache_clear();
    ck_q6_k_prepared_cache_clear();
}

void ck_parallel_prefill_release_transient_caches(void)
{
    /* Packed pointers are looked up per provider call and are never retained
     * in model state. A combined runtime can therefore release prefill's
     * representations before decode establishes its own provider cache. */
    ck_q4k_packed_weight_cache_clear();
}

/* Q4_K packed-meta prefill experiment
 * -----------------------------------
 * This is intentionally kept in the v8 prefill dispatcher for now instead of
 * being promoted to a default src/kernels entry point. The pure kernel pieces
 * live in gemm_kernels_q4k_q8k_vnni.c, but this code owns runtime policy:
 * shape gating, thread selection, and temporary packed-weight lifetime.
 *
 * What we tested locally on the i7 laptop (2026-06-11):
 *   - Standalone Qwen3.5-like Q4_K x Q8_K shapes (N=3584,K=1024) improved
 *     about 1.2x-1.4x across M=128..1024, depending on thread count.
 *   - Nanbeige-like large Q4_K shapes (N=10496,K=2560) usually improved, but
 *     N-split ownership regressed at some early 12-thread M=512 experiments,
 *     but the later dispatch matrix showed it was the best measured candidate
 *     for the Qwen3.5-like shapes used by the v8 prompt path.
 *   - Full-model Qwen3.5 prefill improved at prompt 128 and 256 tokens
 *     (roughly 1.10x-1.13x), while prompt 512 was only a small full-model win
 *     because other operators became the bottleneck.
 *   - Kernel-level parity stayed tight in the standalone benchmark
 *     (max abs around 6e-05 to 1.2e-04), and v8/threadpool quick gates passed.
 *
 * What is still missing before promotion:
 *   - Packed weights should be produced at model-load/conversion time or stored
 *     in the model runtime layout. This lazy cache is acceptable for profiling,
 *     but it is not the final ownership model and currently leaks until process
 *     exit.
 *   - The dispatch rule must be hardware-swept on Xeon/AVX-512 and lower-core
 *     AVX2 machines. The current i7 data supports Qwen3.5-like M-split, not a
 *     universal Q4_K policy.
 *   - N-split should remain a profiling option until it is consistently faster
 *     on a target platform. On this laptop it was mixed.
 *   - Model-level profiling must show that the full prompt path benefits after
 *     other bottlenecks such as Q5_K recurrent projection, SSM conv, attention,
 *     and Q6/Q4 down projection are accounted for.
 *
 * Promotion criteria:
 *   1. Add a real packed-weight layout to the v8 model/load path, with explicit
 *      memory accounting and cleanup.
 *   2. Keep canonical GGUF-layout Q4_K kernels as the parity fallback.
 *   3. Select packed-meta only through shape/hardware dispatch after benchmark
 *      sweeps pass for the target CPU class.
 *   4. Add nightly sweep coverage so CK can learn/verify which Q4_K prefill
 *      policy is best per hardware/model shape.
 */
static void *ck_get_q4k_packed_meta_cached(const void *B, int N, int K)
{
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0) return NULL;

    pthread_mutex_lock(&ck_q4k_packed_meta_cache_mu);
    for (ck_q4k_packed_meta_cache_entry_t *e = ck_q4k_packed_meta_cache_head; e; e = e->next) {
        if (e->src == B && e->N == N && e->K == K) {
            void *packed = e->packed;
            pthread_mutex_unlock(&ck_q4k_packed_meta_cache_mu);
            return packed;
        }
    }

    const size_t blocks = (size_t)N * (size_t)(K / QK_K);
    const size_t bytes = blocks * q4_k_packed_meta_block_size();
    void *packed = malloc(bytes);
    ck_q4k_packed_meta_cache_entry_t *entry =
        (ck_q4k_packed_meta_cache_entry_t *)malloc(sizeof(*entry));
    if (!packed || !entry) {
        free(packed);
        free(entry);
        pthread_mutex_unlock(&ck_q4k_packed_meta_cache_mu);
        return NULL;
    }

    pack_q4_k_to_packed_meta(B, packed, N, K);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->packed = packed;
    entry->next = ck_q4k_packed_meta_cache_head;
    ck_q4k_packed_meta_cache_head = entry;
    pthread_mutex_unlock(&ck_q4k_packed_meta_cache_mu);
    return packed;
}

static void *ck_get_q4k_packed_meta_x8_cached(const void *B, int N, int K)
{
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0) return NULL;

    pthread_mutex_lock(&ck_q4k_packed_meta_x8_cache_mu);
    for (ck_q4k_packed_meta_x8_cache_entry_t *e = ck_q4k_packed_meta_x8_cache_head; e; e = e->next) {
        if (e->src == B && e->N == N && e->K == K) {
            void *packed = e->packed;
            pthread_mutex_unlock(&ck_q4k_packed_meta_x8_cache_mu);
            return packed;
        }
    }

    const size_t groups = (size_t)((N + 7) / 8);
    const size_t blocks = groups * (size_t)(K / QK_K);
    const size_t bytes = blocks * q4_k_packed_meta_x8_block_size();
    void *packed = malloc(bytes);
    ck_q4k_packed_meta_x8_cache_entry_t *entry =
        (ck_q4k_packed_meta_x8_cache_entry_t *)malloc(sizeof(*entry));
    if (!packed || !entry) {
        free(packed);
        free(entry);
        pthread_mutex_unlock(&ck_q4k_packed_meta_x8_cache_mu);
        return NULL;
    }

    pack_q4_k_to_packed_meta_x8(B, packed, N, K);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->packed = packed;
    entry->next = ck_q4k_packed_meta_x8_cache_head;
    ck_q4k_packed_meta_x8_cache_head = entry;
    pthread_mutex_unlock(&ck_q4k_packed_meta_x8_cache_mu);
    return packed;
}

static void *ck_get_q4k_packed_vnni_x8_cached(const void *B, int N, int K)
{
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0) return NULL;

    pthread_mutex_lock(&ck_q4k_packed_vnni_x8_cache_mu);
    for (ck_q4k_packed_vnni_x8_cache_entry_t *e =
                 ck_q4k_packed_vnni_x8_cache_head;
         e; e = e->next) {
        if (e->src == B && e->N == N && e->K == K) {
            void *packed = e->packed;
            pthread_mutex_unlock(&ck_q4k_packed_vnni_x8_cache_mu);
            return packed;
        }
    }

    const size_t groups = (size_t)((N + 7) / 8);
    const size_t blocks = groups * (size_t)(K / QK_K);
    const size_t bytes = blocks * q4_k_packed_vnni_x8_block_size();
    void *packed = malloc(bytes);
    ck_q4k_packed_vnni_x8_cache_entry_t *entry =
            (ck_q4k_packed_vnni_x8_cache_entry_t *)malloc(sizeof(*entry));
    if (!packed || !entry) {
        free(packed);
        free(entry);
        pthread_mutex_unlock(&ck_q4k_packed_vnni_x8_cache_mu);
        return NULL;
    }

    pack_q4_k_to_packed_vnni_x8(B, packed, N, K);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->packed = packed;
    entry->next = ck_q4k_packed_vnni_x8_cache_head;
    ck_q4k_packed_vnni_x8_cache_head = entry;
    pthread_mutex_unlock(&ck_q4k_packed_vnni_x8_cache_mu);
    return packed;
}

int ck_q4k_prepare_vnni_x8_weight(const void *B, int N, int K)
{
    if (!ck_q4k_packed_vnni_x8_available()) return 0;
    return ck_get_q4k_packed_vnni_x8_cached(B, N, K) != NULL;
}

static const void *ck_find_prepared_q4k_packed_vnni_x8(
        const void *B, int N, int K)
{
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0) return NULL;

    pthread_mutex_lock(&ck_q4k_packed_vnni_x8_cache_mu);
    for (ck_q4k_packed_vnni_x8_cache_entry_t *entry =
             ck_q4k_packed_vnni_x8_cache_head;
         entry; entry = entry->next) {
        if (entry->src == B && entry->N == N && entry->K == K) {
            const void *packed = entry->packed;
            pthread_mutex_unlock(&ck_q4k_packed_vnni_x8_cache_mu);
            return packed;
        }
    }
    pthread_mutex_unlock(&ck_q4k_packed_vnni_x8_cache_mu);
    return NULL;
}

int ck_moe_prepare_q4k_gate_up_vnni_x8(
    const void *gate,
    const void *up,
    int intermediate_dim,
    int hidden_dim,
    int n_experts)
{
    if (!gate || !up || intermediate_dim <= 0 || hidden_dim <= 0 ||
        n_experts <= 0 || intermediate_dim > INT_MAX / n_experts ||
        !ck_q4k_packed_vnni_x8_compact_order_available()) {
        return 0;
    }
    const int output_rows = intermediate_dim * n_experts;
    return ck_q4k_prepare_vnni_x8_weight(gate, output_rows, hidden_dim) +
           ck_q4k_prepare_vnni_x8_weight(up, output_rows, hidden_dim);
}

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
    size_t workspace_bytes)
{
    if (rows < 512 || intermediate_dim <= 0 || n_experts <= 0 ||
        intermediate_dim > INT_MAX / n_experts) {
        return moe_swiglu_expert_forward_q4k_q5k_parallel_workspace(
            hidden, indices, routing_weights,
            expert_gate, expert_up, expert_down, output,
            rows, hidden_dim, intermediate_dim, n_experts, top_k,
            workspace, workspace_bytes);
    }

    const int output_rows = intermediate_dim * n_experts;
    const void *gate_packed = ck_find_prepared_q4k_packed_vnni_x8(
        expert_gate, output_rows, hidden_dim);
    const void *up_packed = ck_find_prepared_q4k_packed_vnni_x8(
        expert_up, output_rows, hidden_dim);
    if (!gate_packed || !up_packed) {
        return moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace(
            hidden, indices, routing_weights,
            expert_gate, expert_up, expert_down, output,
            rows, hidden_dim, intermediate_dim, n_experts, top_k,
            workspace, workspace_bytes);
    }
    return moe_swiglu_expert_forward_q4k_q5k_bucketed_prepared_workspace(
        hidden, indices, routing_weights,
        expert_gate, expert_up, expert_down, gate_packed, up_packed, output,
        rows, hidden_dim, intermediate_dim, n_experts, top_k,
        workspace, workspace_bytes);
}

static void *ck_get_q4k_packed_vnni_x16_cached(const void *B, int N, int K)
{
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0) return NULL;

    pthread_mutex_lock(&ck_q4k_packed_vnni_x16_cache_mu);
    for (ck_q4k_packed_vnni_x16_cache_entry_t *e =
                 ck_q4k_packed_vnni_x16_cache_head;
         e; e = e->next) {
        if (e->src == B && e->N == N && e->K == K) {
            void *packed = e->packed;
            pthread_mutex_unlock(&ck_q4k_packed_vnni_x16_cache_mu);
            return packed;
        }
    }

    const size_t groups = (size_t)((N + 15) / 16);
    const size_t blocks = groups * (size_t)(K / QK_K);
    const size_t bytes = blocks * q4_k_packed_vnni_x16_block_size();
    void *packed = malloc(bytes);
    ck_q4k_packed_vnni_x16_cache_entry_t *entry =
            (ck_q4k_packed_vnni_x16_cache_entry_t *)malloc(sizeof(*entry));
    if (!packed || !entry) {
        free(packed);
        free(entry);
        pthread_mutex_unlock(&ck_q4k_packed_vnni_x16_cache_mu);
        return NULL;
    }

    pack_q4_k_to_packed_vnni_x16(B, packed, N, K);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->packed = packed;
    entry->next = ck_q4k_packed_vnni_x16_cache_head;
    ck_q4k_packed_vnni_x16_cache_head = entry;
    pthread_mutex_unlock(&ck_q4k_packed_vnni_x16_cache_mu);
    return packed;
}

/* Return a view into an x16 weight that model initialization already packed.
 * A row-aligned projection subrange can reuse a combined x16 allocation
 * without repacking or changing arithmetic. Do not create a cache entry here:
 * the prepared entry is the capability signal for this dispatch.
 */
static void *ck_find_prepared_q4k_packed_vnni_x16(
        const void *B, int N, int K)
{
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0 || (N % 16) != 0) {
        return NULL;
    }

    const size_t raw_row_bytes =
            (size_t)(K / QK_K) * sizeof(block_q4_K);
    const size_t query_bytes = (size_t)N * raw_row_bytes;
    const uintptr_t query_begin = (uintptr_t)B;
    const uintptr_t query_end = query_begin + query_bytes;
    if (query_end < query_begin) return NULL;

    pthread_mutex_lock(&ck_q4k_packed_vnni_x16_cache_mu);
    for (ck_q4k_packed_vnni_x16_cache_entry_t *e =
                 ck_q4k_packed_vnni_x16_cache_head;
         e; e = e->next) {
        if (e->K != K) continue;

        const uintptr_t entry_begin = (uintptr_t)e->src;
        const size_t entry_bytes = (size_t)e->N * raw_row_bytes;
        const uintptr_t entry_end = entry_begin + entry_bytes;
        if (entry_end < entry_begin || query_begin < entry_begin ||
            query_end > entry_end) {
            continue;
        }

        const size_t byte_offset = (size_t)(query_begin - entry_begin);
        if ((byte_offset % raw_row_bytes) != 0) continue;
        const size_t row_offset = byte_offset / raw_row_bytes;
        if ((row_offset % 16u) != 0) continue;

        const size_t packed_group_bytes =
                (size_t)(K / QK_K) *
                q4_k_packed_vnni_x16_block_size();
        void *packed = (unsigned char *)e->packed +
                (row_offset / 16u) * packed_group_bytes;
        pthread_mutex_unlock(&ck_q4k_packed_vnni_x16_cache_mu);
        return packed;
    }
    pthread_mutex_unlock(&ck_q4k_packed_vnni_x16_cache_mu);
    return NULL;
}


static void *ck_get_q4k_packed_meta_x16_cached(const void *B, int N, int K)
{
    if (!B || N <= 0 || K <= 0 || (K % QK_K) != 0) return NULL;

    pthread_mutex_lock(&ck_q4k_packed_meta_x16_cache_mu);
    for (ck_q4k_packed_meta_x16_cache_entry_t *e = ck_q4k_packed_meta_x16_cache_head; e; e = e->next) {
        if (e->src == B && e->N == N && e->K == K) {
            void *packed = e->packed;
            pthread_mutex_unlock(&ck_q4k_packed_meta_x16_cache_mu);
            return packed;
        }
    }

    const size_t groups = (size_t)((N + 15) / 16);
    const size_t blocks = groups * (size_t)(K / QK_K);
    const size_t bytes = blocks * q4_k_packed_meta_x16_block_size();
    void *packed = malloc(bytes);
    ck_q4k_packed_meta_x16_cache_entry_t *entry =
        (ck_q4k_packed_meta_x16_cache_entry_t *)malloc(sizeof(*entry));
    if (!packed || !entry) {
        free(packed);
        free(entry);
        pthread_mutex_unlock(&ck_q4k_packed_meta_x16_cache_mu);
        return NULL;
    }

    pack_q4_k_to_packed_meta_x16(B, packed, N, K);
    entry->src = B;
    entry->N = N;
    entry->K = K;
    entry->packed = packed;
    entry->next = ck_q4k_packed_meta_x16_cache_head;
    ck_q4k_packed_meta_x16_cache_head = entry;
    pthread_mutex_unlock(&ck_q4k_packed_meta_x16_cache_mu);
    return packed;
}

static int ck_should_use_q4k_packed_meta_prefill(int M, int N, int K)
{
    if (ck_env_enabled("CK_DISABLE_Q4K_PACKED_META_PREFILL")) return 0;
    if (!ck_env_enabled("CK_ENABLE_Q4K_PACKED_META_PREFILL") &&
        !ck_env_enabled("CK_FORCE_Q4K_PACKED_META_PREFILL") &&
        !ck_speed_profile_qwen3vl_ocr_fast()) return 0;
    if (M <= 1 || N <= 0 || K <= 0 || (K % QK_K) != 0) return 0;

    const int min_m = ck_env_int_or2("CK_Q4K_PACKED_META_MIN_M", NULL, 32);
    if (M < min_m) return 0;

    if (getenv("CK_FORCE_Q4K_PACKED_META_PREFILL")) return 1;

    /* On the 24-physical-core Xeon Qwen3-VL OCR path, short-prefix wide
     * projections such as M=79,N=24576,K=4096 and N=4096,K=4096 are faster
     * through the canonical output-row Q4_K schedule than the lazy packed-meta
     * N-split cache path. Keep packed-meta available through FORCE/env, but do
     * not select it by default for this wide short-prefix family. */
    if (M < 128 && N >= 4096 && K >= 4096) return 0;

    /* The dispatch matrix benchmark tracks canonical serial, canonical pool,
     * packed M-split, packed N-split, and a llama.cpp shim. Local AVX2 data
     * shows packed N-split is the best measured Q4_K prefill candidate for
     * Qwen-like shapes:
     *   - M=32,N=1024,K=1024:  ~1.36x over canonical
     *   - M=32,N=896,K=4864:   ~1.25x over canonical
     *   - M=128,N=896,K=4864:  ~1.33x over canonical
     *
     * Keep decode on GEMV. This gate is prefill-only because M > 1 and is
     * still shape-gated; use CK_DISABLE_Q4K_PACKED_META_PREFILL=1 to force
     * canonical Q4_K while validating a new CPU. */
    /* Local i7 sweeps show that narrow output projections do not reliably
     * recover the packed-meta scheduling cost:
     *   - N=512,K=1024 regresses/slightly loses at M=128.
     *   - N=896,K=4864 was slower than canonical pool in the dispatch matrix.
     * Keep packed-meta on the wider Qwen3.5 gate/up style shapes that win, and
     * let force/env tuning override this when collecting new hardware data. */
    if (N >= 512 && K >= 1024) return 1;
    if (K <= 2048 && N >= 1024 && N <= 8192) return 1;
    return 0;
}

static int ck_should_use_q4k_packed_meta_x16_prefill(int M, int N, int K)
{
    if (ck_env_enabled("CK_DISABLE_Q4K_PACKED_META_X16_PREFILL")) return 0;
    if (M <= 1 || N <= 0 || K <= 0 || (K % QK_K) != 0) return 0;

    const int min_m = ck_env_int_or2("CK_Q4K_PACKED_META_X16_MIN_M", NULL, 16);
    if (M < min_m) return 0;
    if (ck_env_enabled("CK_FORCE_Q4K_PACKED_META_X16_PREFILL")) return 1;

    /* Default prefill path for wide Q4_K x Q8_K projections.
     *
     * The canonical v8 fallback is intentionally safe, but it implements GEMM
     * as row-split GEMV calls. That is appropriate for decode (M == 1), but it
     * destroys prefill reuse and shows up in Advisor/VTune as a low-AI
     * gemv_q4_k_q8_k_avx2 hotspot. The packed x16 path keeps a small token tile
     * hot across 16 output rows and is the measured win for Nanbeige/Qwen-style
     * prefill shapes on AVX2. Keep CK_DISABLE_Q4K_PACKED_META_X16_PREFILL as
     * the production escape hatch for new CPUs or model-specific regressions. */
    if (N >= 512 && K >= 1024) return 1;
    return 0;
}

static int ck_should_use_q4k_packed_meta_x8_mreuse_prefill(int M, int N, int K)
{
    if (ck_env_enabled("CK_DISABLE_Q4K_PACKED_META_X8_MREUSE_PREFILL")) return 0;
    if (!ck_env_enabled("CK_ENABLE_Q4K_PACKED_META_X8_MREUSE_PREFILL") &&
        !ck_env_enabled("CK_FORCE_Q4K_PACKED_META_X8_MREUSE_PREFILL") &&
        !ck_speed_profile_qwen3vl_ocr_fast()) return 0;
    if (M <= 1 || N <= 0 || K <= 0 || (K % QK_K) != 0) return 0;

    const int min_m = ck_env_int_or2("CK_Q4K_PACKED_META_X8_MREUSE_MIN_M", NULL, 128);
    if (M < min_m) return 0;
    if (ck_env_enabled("CK_FORCE_Q4K_PACKED_META_X8_MREUSE_PREFILL")) return 1;

    /* Measured on Qwen3-VL OCR mixed-prefill shapes:
     *   proj/down M ~= 1028, N=4096, K=4096/11008.
     * The M-reuse path keeps one x8 packed output group hot across a small
     * token tile and avoids the large-shape reread cost of plain N-split. */
    if (N >= 768 && K >= 1024) return 1;
    return 0;
}

static int ck_should_use_q4k_packed_meta_x8_prefill(int M, int N, int K)
{
    if (ck_env_enabled("CK_DISABLE_Q4K_PACKED_META_X8_PREFILL")) return 0;
    if (!ck_env_enabled("CK_ENABLE_Q4K_PACKED_META_X8_PREFILL") &&
        !ck_env_enabled("CK_FORCE_Q4K_PACKED_META_X8_PREFILL") &&
        !ck_speed_profile_qwen3vl_ocr_fast()) return 0;
    if (M <= 1 || N <= 0 || K <= 0 || (K % QK_K) != 0) return 0;

    const int min_m = ck_env_int_or2("CK_Q4K_PACKED_META_X8_MIN_M", NULL, 16);
    if (M < min_m) return 0;
    if (getenv("CK_FORCE_Q4K_PACKED_META_X8_PREFILL")) return 1;
    const int x8_max_m_default = ck_speed_profile_qwen3vl_ocr_fast() ? 2048 : 64;
    const int max_m = ck_env_int_or2("CK_Q4K_PACKED_META_X8_MAX_M", NULL, x8_max_m_default);
    if (max_m > 0 && M > max_m) return 0;

    /* Experimental x8 prefill gate, derived from the dispatch matrix:
     *   small:        M=8,  N=256,  K=256   -> useful microbench, not model-critical
     *   qwen35_qkv:   M=32, N=1024, K=1024  -> x8 wins over packed-N locally
     *   qwen35_down:  M=32, N=896,  K=4864  -> x8 wins over packed-N locally
     *   wide:         M=64, N=1024, K=2560  -> x8 wins over packed-N locally
     *   prefill128:   M=128,N=896,  K=4864  -> x8 loses locally; use canonical pool
     *
     * Keep the gate on measured short/medium Qwen/Nemotron-family prefill shapes and retain
     * CK_DISABLE_Q4K_PACKED_META_X8_PREFILL as the production escape hatch. */
    if (N >= 768 && K >= 1024) return 1;
    return 0;
}

static int ck_should_use_q4k_packed_meta_x8mt_prefill(int M, int N, int K)
{
    if (ck_env_enabled("CK_DISABLE_Q4K_PACKED_META_X8MT_PREFILL")) return 0;
    if (!ck_env_enabled("CK_ENABLE_Q4K_PACKED_META_X8MT_PREFILL") &&
        !ck_env_enabled("CK_FORCE_Q4K_PACKED_META_X8MT_PREFILL")) return 0;
    if (M <= 1 || N <= 0 || K <= 0 || (K % QK_K) != 0) return 0;

    const int min_m = ck_env_int_or2("CK_Q4K_PACKED_META_X8MT_MIN_M", NULL, 16);
    if (M < min_m) return 0;
    if (ck_env_enabled("CK_FORCE_Q4K_PACKED_META_X8MT_PREFILL")) return 1;

    /* Token-tile x output-tile path is still experimental. It is measured via
     * the dispatch matrix and model perf-stat lane before shape promotion. */
    if (N >= 768 && K >= 1024) return 1;
    return 0;
}

static int ck_should_use_q4k_packed_meta_2d_prefill(const ck_threadpool_t *pool,
                                                     int M, int N, int K,
                                                     int tile_m, int tile_n)
{
    if (!ck_env_enabled("CK_ENABLE_Q4K_PACKED_META_2D_PREFILL")) return 0;
    if (!pool || ck_threadpool_n_threads(pool) <= 1) return 0;
    if (M <= 1 || N <= 0 || K <= 0 || (K % QK_K) != 0) return 0;

    const int tm = tile_m > 0 ? tile_m : 16;
    const int tn = tile_n > 0 ? tile_n : 256;
    const int mt = ck_ceil_div_int(M, tm);
    const int nt = ck_ceil_div_int(N, tn);
    const int jobs = mt * nt;
    const int active = ck_select_gemm_active_threads(pool, M, N, K);

    if (jobs < active * 2) return 0;
    if (getenv("CK_FORCE_Q4K_PACKED_META_2D_PREFILL")) return 1;

    /* Experimental path: measure before promotion. The current packed-meta dot
     * loop still computes one output at a time, so 2D scheduling improves job
     * balance but can reread activation tiles. */
    return 0;
}

static int ck_should_use_q6k_q8k_2d_prefill(const ck_threadpool_t *pool,
                                             int M, int N, int K,
                                             int tile_m, int tile_n)
{
    if (ck_env_enabled("CK_DISABLE_Q6K_Q8K_2D_PREFILL")) return 0;
    if (ck_q6k_q8k_2d_prefill_forced()) return 1;
    if (!pool || ck_threadpool_n_threads(pool) <= 1) return 0;
    if (M <= 1 || N <= 0 || K <= 0) return 0;
    if (K % QK_K != 0) return 0;

    const int tm = tile_m > 0 ? tile_m : 16;
    const int tn = tile_n > 0 ? tile_n : 256;
    const int mt = ck_ceil_div_int(M, tm);
    const int nt = ck_ceil_div_int(N, tn);
    const int jobs = mt * nt;
    const int active = ck_select_gemm_active_threads(pool, M, N, K);

    if (jobs < active * 2) return 0;

    const ck_gemm_route_v8 *route = ck_find_gemm_route_v8(
        ck_policy_gemm_nt_q6_k_q8_k_prefill_schedule,
        CK_POLICY_GEMM_NT_Q6_K_Q8_K_PREFILL_SCHEDULE_COUNT,
        M, N, K);
    return route && (route->flags & CK_GEMM_ROUTE_OUTPUT_TILES) != 0;
}

static int ck_should_use_q6k_q8k_m4_prefill(int M, int N, int K)
{
    if (ck_env_enabled("CK_DISABLE_Q6K_Q8K_M4_PREFILL")) return 0;
    if (ck_env_enabled("CK_ENABLE_Q6K_Q8K_M4_PREFILL")) return 1;

    const ck_gemm_route_v8 *route = ck_find_gemm_route_v8(
        ck_policy_gemm_nt_q6_k_q8_k_prefill_schedule,
        CK_POLICY_GEMM_NT_Q6_K_Q8_K_PREFILL_SCHEDULE_COUNT,
        M, N, K);
    return route && (route->flags & CK_GEMM_ROUTE_COMPACT_M4) != 0;
}

static int ck_shape_aware_enabled(const ck_threadpool_t *pool)
{
    const int pool_threads = ck_threadpool_n_threads(pool);
    if (ck_env_enabled("CK_DISABLE_SHAPE_AWARE_THREADPOOL")) return 0;
    if (getenv("CK_GEMM_THREAD_CAP") || getenv("CK_GEMV_THREAD_CAP")) return 1;
    return pool_threads > 16;
}

static int ck_select_gemm_active_threads(const ck_threadpool_t *pool, int M, int N, int K)
{
    const int pool_threads = ck_threadpool_n_threads(pool);
    if (pool_threads <= 1 || M <= 1 || N <= 0 || K <= 0) return 1;
    if (!ck_shape_aware_enabled(pool)) return pool_threads;

    if (getenv("CK_GEMM_THREAD_CAP") || getenv("CK_GEMV_THREAD_CAP")) {
        return ck_min_int(pool_threads,
                          ck_env_int_or2("CK_GEMM_THREAD_CAP", "CK_GEMV_THREAD_CAP", pool_threads));
    }

    if (N >= 4096 || K >= 4096) return ck_min_int(pool_threads, ck_env_int_or2("CK_GEMM_THREAD_CAP", "CK_GEMV_THREAD_CAP", 24));
    if (M >= 512) return ck_min_int(pool_threads, 24);
    return pool_threads;
}

static int ck_select_q4k_vnni_active_threads(
        const ck_threadpool_t *pool, int M, int N, int K)
{
    const int base = ck_select_gemm_active_threads(pool, M, N, K);
    const int capacity = ck_threadpool_capacity(pool);
    if (capacity <= base || M < 512 || N < 4096 || K < 4096) {
        return base;
    }

    /* The pool capacity already represents the bounded SMT extension selected
     * at initialization. Ordinary kernels continue to see the physical-core
     * default through ck_threadpool_n_threads(). */
    return capacity;
}

static const ck_gemm_route_v8 *ck_q4k_avx512_x16_prefill_route(
        int M, int N, int K)
{
    /*
     * Keep the AVX-512 provider sweep-only until a real generated graph shows
     * a stable end-to-end win. The isolated and live batched GEMMs win, but a
     * complete hybrid graph can have a different numerical trajectory.
     */
    if (!ck_env_enabled("CK_ENABLE_Q4K_AVX512_X16_EXPERIMENTAL") &&
        !ck_env_enabled("CK_V8_FORCE_BATCHED_PREFILL")) {
        return NULL;
    }
    if (ck_env_enabled("CK_DISABLE_Q4K_AVX512_X16_PREFILL")) return NULL;
    if (!ck_q4k_packed_vnni_x16_available()) return NULL;
    return ck_find_gemm_route_v8(
        ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_prefill,
        CK_POLICY_GEMM_NT_Q4_K_Q8_K_AVX512_VNNI_X16_PREFILL_COUNT,
        M, N, K);
}

int ck_q4k_prepare_vnni_x16_weight(const void *B, int N, int K)
{
    /*
     * Reuse production eligibility so initialization cannot create a second,
     * subtly different provider-selection policy.
     */
    if (!ck_q4k_avx512_x16_prefill_route(16, N, K)) return 0;
    return ck_get_q4k_packed_vnni_x16_cached(B, N, K) != NULL;
}

static int ck_select_q4k_avx512_x16_threads(
        const ck_threadpool_t *pool, int M, int N, int K,
        const ck_gemm_route_v8 *route)
{
    int active = ck_select_gemm_active_threads(pool, M, N, K);
    if (route && route->max_threads > 0 && active > route->max_threads) {
        active = route->max_threads;
    }
    return active;
}

static int ck_should_run_gemm_serial(const ck_threadpool_t *pool, int M, int N, int K)
{
    if (!ck_shape_aware_enabled(pool)) return 0;
    const int threshold = ck_env_int_or2("CK_GEMM_SMALL_SERIAL_THRESHOLD", NULL, 16);
    return M < threshold && N < 512 && K < 512;
}

/* ============================================================================
 * Work Functions (called on each thread)
 *
 * Each computes rows [r0, r1) of the output by calling the serial GEMM
 * on a sub-range of A and C.
 * ============================================================================ */

static void work_gemm_nt_q5_0_q8_0(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q5_0_q8_0(
        (const char *)a->A + (size_t)r0 * a->A_row_bytes,
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q5_0_q8_0_m4n2_range(int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end || end > a->M) return;

    gemm_nt_q5_0_q8_0_m4n2(
        (const char *)a->A + (size_t)begin * a->A_row_bytes,
        a->B, a->bias, a->C + (size_t)begin * a->N,
        end - begin, a->N, a->K);
}

static void work_gemm_nt_q5_0_q8_0_m4n2_output_tiles(
        int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end || a->tile_n <= 0) return;

    const size_t weight_row_bytes =
        (size_t)(a->K / QK5_0) * sizeof(block_q5_0);
    for (int job = begin; job < end; ++job) {
        const int n0 = job * a->tile_n;
        if (n0 >= a->N) break;
        const int n1 = ck_min_int(n0 + a->tile_n, a->N);
        gemm_nt_q5_0_q8_0_m4n2_tile(
            a->A,
            (const char *)a->B + (size_t)n0 * weight_row_bytes,
            a->bias ? a->bias + n0 : NULL,
            a->C + n0,
            a->M, n1 - n0, a->K, a->N);
    }
}

static void work_gemm_nt_q8_0_q8_0(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q8_0_q8_0(
        (const char *)a->A + (size_t)r0 * a->A_row_bytes,
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q8_0_q8_0_m2n4(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q8_0_q8_0_m2n4(
        (const char *)a->A + (size_t)r0 * a->A_row_bytes,
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q8_0_q8_0_range(int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end || end > a->M) return;

    gemm_nt_q8_0_q8_0(
        (const char *)a->A + (size_t)begin * a->A_row_bytes,
        a->B,
        a->bias,
        a->C + (size_t)begin * a->N,
        end - begin, a->N, a->K
    );
}

static void work_gemm_nt_q8_0_q8_0_m2n4_range(int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end || end > a->M) return;

    gemm_nt_q8_0_q8_0_m2n4(
        (const char *)a->A + (size_t)begin * a->A_row_bytes,
        a->B,
        a->bias,
        a->C + (size_t)begin * a->N,
        end - begin, a->N, a->K
    );
}

static void work_gemm_nt_q8_0_q8_0_m2n4_output_tiles(
        int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end || a->tile_n <= 0) return;

    const size_t weight_row_bytes =
        (size_t)(a->K / QK8_0) * sizeof(block_q8_0);
    for (int job = begin; job < end; ++job) {
        const int n0 = job * a->tile_n;
        if (n0 >= a->N) break;
        const int n1 = ck_min_int(n0 + a->tile_n, a->N);
        gemm_nt_q8_0_q8_0_m2n4_tile(
            a->A,
            (const char *)a->B + (size_t)n0 * weight_row_bytes,
            a->bias ? a->bias + n0 : NULL,
            a->C + n0,
            a->M, n1 - n0, a->K, a->N);
    }
}

static inline void work_gemm_nt_q8_0_q8_0_contract_rows(
        const gemm_args_t *a, int begin, int end)
{
    for (int m = begin; m < end; ++m) {
        float *output = a->C + (size_t)m * (size_t)a->N;
        gemv_q8_0_q8_0_contract(
            output, a->B,
            (const float *)a->A + (size_t)m * (size_t)a->K,
            a->N, a->K);
        if (a->bias) {
            for (int n = 0; n < a->N; ++n) output[n] += a->bias[n];
        }
    }
}

static void work_gemm_nt_q8_0_q8_0_contract(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    const int dr = (a->M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = ck_min_int(r0 + dr, a->M);
    if (r0 >= a->M) return;

    work_gemm_nt_q8_0_q8_0_contract_rows(a, r0, r1);
}

static void work_gemm_nt_q8_0_q8_0_contract_range(
        int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end || end > a->M) return;
    work_gemm_nt_q8_0_q8_0_contract_rows(a, begin, end);
}

static void work_gemm_nt_q4_k_q8_k(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    /* Do not call gemm_nt_q4_k_q8_k() here: that raw implementation can
     * start its own internal output-row threadpool for large Q4_K shapes.
     * This worker already runs inside the v8 prefill pool, so nesting the
     * same pool can corrupt scheduling and parity. Use the one-token GEMV
     * primitive directly for each assigned token row. */
    for (int m = r0; m < r1; ++m) {
        const void *x_row = (const char *)a->A + (size_t)m * a->A_row_bytes;
        float *c_row = a->C + (size_t)m * (size_t)a->N;
        gemv_q4_k_q8_k(c_row, a->B, x_row, a->N, a->K);
        if (a->bias) {
            for (int n = 0; n < a->N; ++n) {
                c_row[n] += a->bias[n];
            }
        }
    }
}

static void work_gemm_nt_q4_k_q8_k_pairwise_split_min(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    const int dr = (a->M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    if ((a->N % 16) == 0) {
        gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
            (const char *)a->A + (size_t)r0 * a->A_row_bytes,
            a->B,
            a->bias,
            a->C + (size_t)r0 * (size_t)a->N,
            r1 - r0, a->N, a->K
        );
    } else {
        gemm_nt_q4_k_packed_meta_x8_q8_k_superblock_order(
            (const char *)a->A + (size_t)r0 * a->A_row_bytes,
            a->B,
            a->bias,
            a->C + (size_t)r0 * (size_t)a->N,
            r1 - r0, a->N, a->K
        );
    }
}

static void work_gemv_q4_k_q8_k_repacked(int ith, int nth, void *args)
{
    const q4k_repacked_gemv_args_t *a = (const q4k_repacked_gemv_args_t *)args;
    const int groups = (a->N + 7) / 8;
    const int dg = (groups + nth - 1) / nth;
    const int g0 = dg * ith;
    const int g1 = ck_min_int(g0 + dg, groups);
    if (g0 >= groups) return;

    const int n0 = g0 * 8;
    const int n1 = ck_min_int(g1 * 8, a->N);
    const size_t packed_group_bytes =
            (size_t)(a->K / QK_K) * q4_k_packed_meta_x8_block_size();
    gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order(
        a->A,
        (const char *)a->B_packed_x8 + (size_t)g0 * packed_group_bytes,
        a->bias ? a->bias + n0 : NULL,
        a->C + n0,
        1, n1 - n0, a->K
    );
}

static void work_gemv_q4_k_q8_k_repacked_x16(int ith, int nth, void *args)
{
    const q4k_repacked_gemv_args_t *a =
            (const q4k_repacked_gemv_args_t *)args;
    const int groups = (a->N + 15) / 16;
    const int dg = (groups + nth - 1) / nth;
    const int g0 = dg * ith;
    const int g1 = ck_min_int(g0 + dg, groups);
    if (g0 >= groups) return;

    const int n0 = g0 * 16;
    const int n1 = ck_min_int(g1 * 16, a->N);
    const size_t packed_group_bytes =
            (size_t)(a->K / QK_K) *
            q4_k_packed_vnni_x16_block_size();
    for (int row = 0; row < a->M; ++row) {
        gemm_nt_q4_k_packed_vnni_x16_q8_k_gemv_order(
                (const char *)a->A + (size_t)row * a->A_row_bytes,
                (const char *)a->B_packed_x8 +
                        (size_t)g0 * packed_group_bytes,
                a->bias ? a->bias + n0 : NULL,
                a->C + (size_t)row * (size_t)a->N + n0,
                1,
                n1 - n0,
                a->K);
    }
}

static void run_gemv_q4_k_q8_k_repacked_parallel(
        ck_threadpool_t *pool,
        const void *A, const void *B_packed_x8, const float *bias, float *C,
        int N, int K, int thread_cap)
{
    q4k_repacked_gemv_args_t args = {
        .A = A,
        .B_packed_x8 = B_packed_x8,
        .bias = bias,
        .C = C,
        .M = 1,
        .N = N,
        .K = K,
        .A_row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K),
    };
    const int groups = (N + 7) / 8;
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (thread_cap > 0 && active > thread_cap) active = thread_cap;
    if (active > groups) active = groups;
    if (active <= 1 || !pool) {
        work_gemv_q4_k_q8_k_repacked(0, 1, &args);
        return;
    }
    ck_threadpool_dispatch_n(
            pool, active, work_gemv_q4_k_q8_k_repacked, &args);
}

static void run_gemv_q4_k_q8_k_repacked_x16_parallel(
        ck_threadpool_t *pool,
        const void *A, const void *B_packed_x16,
        const float *bias, float *C,
        int M, int N, int K, int thread_cap)
{
    q4k_repacked_gemv_args_t args = {
        .A = A,
        .B_packed_x8 = B_packed_x16,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .A_row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K),
    };
    const int groups = (N + 15) / 16;
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (thread_cap > 0 && active > thread_cap) active = thread_cap;
    if (active > groups) active = groups;
    if (active <= 1 || !pool) {
        work_gemv_q4_k_q8_k_repacked_x16(0, 1, &args);
        return;
    }
    ck_threadpool_dispatch_n(
            pool, active, work_gemv_q4_k_q8_k_repacked_x16, &args);
}

static void work_gemm_nt_q6_k_q8_k(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    if (a->use_q6_prepared) {
        gemm_nt_q6_k_q8_k_prepared(
            (const char *)a->A + (size_t)r0 * a->A_row_bytes,
            a->B,
            a->bias,
            a->C + (size_t)r0 * a->N,
            r1 - r0, a->N, a->K
        );
    } else if (ck_env_enabled("CK_ENABLE_Q6K_Q8K_TILED_PREFILL")) {
        gemm_nt_q6_k_q8_k_tiled(
            (const char *)a->A + (size_t)r0 * a->A_row_bytes,
            a->B,
            a->bias,
            a->C + (size_t)r0 * a->N,
            r1 - r0, a->N, a->K
        );
    } else {
        gemm_nt_q6_k_q8_k(
            (const char *)a->A + (size_t)r0 * a->A_row_bytes,
            a->B,
            a->bias,
            a->C + (size_t)r0 * a->N,
            r1 - r0, a->N, a->K
        );
    }
}

static inline void work_gemm_nt_q6_k_q8_k_2d_job(
        const gemm_args_t *a, int job, int mt, int tile_m, int tile_n)
{
    const int jm = job % mt;
    const int jn = job / mt;
    const int m0 = jm * tile_m;
    const int m1 = ck_min_int(m0 + tile_m, a->M);
    const int n0 = jn * tile_n;
    const int n1 = ck_min_int(n0 + tile_n, a->N);
    if (a->use_q6_prepared) {
        gemm_nt_q6_k_q8_k_prepared_tile(
            a->A, a->B, a->bias, a->C,
            a->M, a->N, a->K, m0, m1, n0, n1);
    } else if (a->use_q6_m4) {
        gemm_nt_q6_k_q8_k_m4_tile(a->A, a->B, a->bias, a->C,
                                  a->M, a->N, a->K, m0, m1, n0, n1);
    } else {
        gemm_nt_q6_k_q8_k_tile(a->A, a->B, a->bias, a->C,
                               a->M, a->N, a->K, m0, m1, n0, n1);
    }
}

static void work_gemm_nt_q6_k_q8_k_2d(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) return;

    const int tile_m = a->tile_m > 0 ? a->tile_m : 16;
    const int tile_n = a->tile_n > 0 ? a->tile_n : 256;
    const int mt = ck_ceil_div_int(a->M, tile_m);
    const int total = mt * ck_ceil_div_int(a->N, tile_n);
    for (int job = ith; job < total; job += nth) {
        work_gemm_nt_q6_k_q8_k_2d_job(a, job, mt, tile_m, tile_n);
    }
}

static void work_gemm_nt_q6_k_q8_k_2d_range(int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end) return;
    const int tile_m = a->tile_m > 0 ? a->tile_m : 16;
    const int tile_n = a->tile_n > 0 ? a->tile_n : 256;
    const int mt = ck_ceil_div_int(a->M, tile_m);
    for (int job = begin; job < end; ++job) {
        work_gemm_nt_q6_k_q8_k_2d_job(a, job, mt, tile_m, tile_n);
    }
}

static void work_gemm_nt_q4_k_packed_meta_q8_k_2d(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || ith < 0 || nth <= 0 || ith >= nth) return;

    const int tile_m = a->tile_m > 0 ? a->tile_m : 16;
    const int tile_n = a->tile_n > 0 ? a->tile_n : 256;
    const int mt = ck_ceil_div_int(a->M, tile_m);
    const int nt = ck_ceil_div_int(a->N, tile_n);
    const int total = mt * nt;

    for (int job = ith; job < total; job += nth) {
        const int jm = job % mt;
        const int jn = job / mt;
        const int m0 = jm * tile_m;
        const int m1 = ck_min_int(m0 + tile_m, a->M);
        const int n0 = jn * tile_n;
        const int n1 = ck_min_int(n0 + tile_n, a->N);
        gemm_nt_q4_k_packed_meta_q8_k_tile(a->A, a->B, a->bias, a->C,
                                           a->M, a->N, a->K, m0, m1, n0, n1);
    }
}

static void work_gemm_nt_q5_1_q8_1(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q5_1_q8_1_m4(
        (const float *)((const char *)a->A + (size_t)r0 * a->A_row_bytes),
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q5_1_q8_1_reuse_range(int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end) return;
    const int group_rows = a->tile_m == 8 ? 8 : 4;

    for (int job = begin; job < end; ++job) {
        const int r0 = job * group_rows;
        if (r0 >= a->M) break;
        const int rows = ck_min_int(group_rows, a->M - r0);
        const float *input = (const float *)((const char *)a->A +
                                              (size_t)r0 * a->A_row_bytes);
        float *output = a->C + (size_t)r0 * a->N;
        if (group_rows == 8) {
            gemm_nt_q5_1_q8_1_m8(
                input, a->B, a->bias, output, rows, a->N, a->K);
        } else {
            gemm_nt_q5_1_q8_1_m4(
                input, a->B, a->bias, output, rows, a->N, a->K);
        }
    }
}

static void work_geglu_exact_rows(int ith, int nth, void *args)
{
    const geglu_args_t *a = (const geglu_args_t *)args;
    const int rows = ck_ceil_div_int(a->tokens, nth);
    const int begin = rows * ith;
    const int end = ck_min_int(begin + rows, a->tokens);
    if (begin >= end) return;

    geglu_forward_exact(
        a->input + (size_t)begin * (size_t)(2 * a->dim),
        a->output + (size_t)begin * (size_t)a->dim,
        end - begin, a->dim);
}

static void work_geglu_exact_range(int begin, int end, void *args)
{
    const geglu_args_t *a = (const geglu_args_t *)args;
    if (!a || begin < 0 || begin >= end || end > a->tokens) return;

    geglu_forward_exact(
        a->input + (size_t)begin * (size_t)(2 * a->dim),
        a->output + (size_t)begin * (size_t)a->dim,
        end - begin, a->dim);
}

static void work_gemm_nt_q5_k(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemm_nt_q5_k(
        (const float *)((const char *)a->A + (size_t)r0 * a->A_row_bytes),
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
}

static void work_gemm_nt_q5_k_prepared(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    const int dr = (a->M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = ck_min_int(r0 + dr, a->M);
    if (r0 >= r1) return;
    gemm_nt_q5_k_prepared(
        (const float *)((const char *)a->A + (size_t)r0 * a->A_row_bytes),
        a->B, a->bias, a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K);
}

static void work_gemm_nt_q5_k_prepared_m4(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    const int dr = (a->M + nth - 1) / nth;
    const int r0 = dr * ith;
    const int r1 = ck_min_int(r0 + dr, a->M);
    if (r0 >= r1) return;
    gemm_nt_q5_k_prepared_m4(
        (const float *)((const char *)a->A + (size_t)r0 * a->A_row_bytes),
        a->B, a->bias, a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K);
}

static void work_gemm_nt_q5_k_prepared_nrange(
    int begin, int end, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    if (!a || begin < 0 || begin >= end) return;
    const int tile_n = a->tile_n > 0 ? a->tile_n : 64;
    for (int job = begin; job < end; ++job) {
        const int n0 = job * tile_n;
        const int n1 = ck_min_int(n0 + tile_n, a->N);
        if (n0 >= n1) break;
        gemm_nt_q5_k_prepared_q8_m4_nrange(
            a->A, a->B, a->bias, a->C,
            a->M, a->N, a->K, n0, n1);
    }
}

/* ============================================================================
 * Parallel Dispatch Wrappers
 *
 * Same signature as serial GEMM functions. Pack args, dispatch to pool.
 * Fast path: M <= 1 or single thread -> call serial directly.
 * ============================================================================ */

void gemm_nt_q5_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    const block_q8_0 *prepared = ck_find_prepared_q5_0_q8_0(B, N, K);
    if (prepared) {
        gemm_nt_q8_0_q8_0_parallel_dispatch(A, prepared, bias, C, M, N, K);
        return;
    }
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q5_0_q8_0(A, B, bias, C, M, N, K);
        return;
    }

    /* A is Q8_0: row_bytes = (K / QK8_0) * sizeof(block_q8_0) */
    size_t A_row_bytes = (size_t)(K / QK8_0) * sizeof(block_q8_0);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes,
        .tile_n = 32,
    };
    const int active = ck_select_gemm_active_threads(pool, M, N, K);
#if defined(__AVX2__)
    if (ck_gemm_dynamic_schedule_enabled()) {
        if (M < 256) {
            const int jobs = ck_ceil_div_int(N, args.tile_n);
            ck_threadpool_parallel_for_n(
                pool, active, 0, jobs, 1,
                work_gemm_nt_q5_0_q8_0_m4n2_output_tiles, &args);
        } else {
            ck_threadpool_parallel_for_n(
                pool, active, 0, M, 4,
                work_gemm_nt_q5_0_q8_0_m4n2_range, &args);
        }
        return;
    }
#endif
    ck_threadpool_dispatch_n(pool, active, work_gemm_nt_q5_0_q8_0, &args);
}

void gemm_nt_q8_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q8_0_q8_0(A, B, bias, C, M, N, K);
        return;
    }

    /* A is Q8_0: row_bytes = (K / QK8_0) * sizeof(block_q8_0) */
    size_t A_row_bytes = (size_t)(K / QK8_0) * sizeof(block_q8_0);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes,
        .tile_n = 32,
    };
    const int active = ck_select_gemm_active_threads(pool, M, N, K);
#if defined(__AVX2__)
    const int use_m2n4 = M >= 2;
#else
    const int use_m2n4 = 0;
#endif
    if (ck_gemm_dynamic_schedule_enabled()) {
        if (use_m2n4 && M < 256) {
            const int jobs = ck_ceil_div_int(N, args.tile_n);
            ck_threadpool_parallel_for_n(
                pool, active, 0, jobs, 1,
                work_gemm_nt_q8_0_q8_0_m2n4_output_tiles, &args);
        } else if (use_m2n4 && M >= 256) {
            ck_threadpool_parallel_for_n(
                pool, active, 0, M, 2,
                work_gemm_nt_q8_0_q8_0_m2n4_range, &args);
        } else {
            ck_threadpool_parallel_for_n(
                pool, active, 0, M, 1,
                work_gemm_nt_q8_0_q8_0_range, &args);
        }
    } else {
        ck_threadpool_dispatch_n(
            pool, active,
            (use_m2n4 && M >= 64) ? work_gemm_nt_q8_0_q8_0_m2n4
                                  : work_gemm_nt_q8_0_q8_0,
            &args);
    }
}

void gemm_nt_q8_0_q8_0_contract_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (ck_strict_parity_enabled() ||
        ck_env_enabled("CK_DISABLE_Q80_CONTRACT_PARALLEL_PREFILL") ||
        !pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 ||
        ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q8_0_q8_0_contract(A, B, bias, C, M, N, K);
        return;
    }

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = (size_t)K * sizeof(float),
    };
    int active = ck_select_gemm_active_threads(pool, M, N, K);
    if (active > M) active = M;
    if (ck_gemm_dynamic_schedule_enabled()) {
        const int grain = 4;
        ck_threadpool_parallel_for_n(
            pool, active, 0, M, grain,
            work_gemm_nt_q8_0_q8_0_contract_range, &args);
    } else {
        ck_threadpool_dispatch_n(
            pool, active, work_gemm_nt_q8_0_q8_0_contract, &args);
    }
}


void gemm_nt_q4_k_q8_k_gateup_swiglu_x16_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int D, int K)
{
    if (!A || !B || !C || M <= 0 || D <= 0 || K <= 0 || (K % QK_K) != 0) return;

    const int N = D * 2;
    ck_threadpool_t *pool = ck_threadpool_global();
    void *packed = ck_get_q4k_packed_meta_x16_cached(B, N, K);
    if (packed) {
        const int tile_m = ck_env_int_or2("CK_Q4K_GATEUP_SWIGLU_X16_TILE_M", "CK_PREFILL_TILE_M", 8);
        int active = pool ? ck_threadpool_n_threads(pool) : 1;
        const int cap = ck_env_int_or2("CK_Q4K_GATEUP_SWIGLU_X16_THREAD_CAP", "CK_GEMM_THREAD_CAP", 20);
        if (cap > 0 && active > cap) active = cap;
        gemm_nt_q4_k_packed_meta_x16_gateup_swiglu_fused_vnni(
            A, packed, bias, C, M, D, K, tile_m, active);
        return;
    }

    /* Correctness fallback for allocation/packing failure. This path is not
     * performance-critical because the fused call is env-gated by codegen. */
    float *tmp = (float *)malloc((size_t)M * (size_t)N * sizeof(float));
    if (!tmp) return;
    gemm_nt_q4_k_q8_k_parallel_dispatch(A, B, bias, tmp, M, N, K);
    swiglu_forward_exact(tmp, C, M, D);
    free(tmp);
}

void gemm_nt_q4_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (pool && ck_should_use_q4k_packed_meta_x16_prefill(M, N, K)) {
        void *packed_x16 = ck_get_q4k_packed_meta_x16_cached(B, N, K);
        if (packed_x16) {
            int active = ck_select_gemm_active_threads(pool, M, N, K);
            const int cap = ck_env_int_or2("CK_Q4K_PACKED_META_X16_THREAD_CAP", "CK_GEMM_THREAD_CAP", 20);
            if (cap > 0 && active > cap) active = cap;
            const int tile_m = ck_env_int_or2("CK_Q4K_PACKED_META_X16_TILE_M", "CK_PREFILL_TILE_M", 8);
            if (ck_env_enabled("CK_Q4K_PACKED_META_X16_MTILE")) {
                ck_q4k_prefill_debug_dispatch("x16_mtile", M, N, K, active);
                gemm_nt_q4_k_packed_meta_x16_q8_k_threaded_mtile(
                    A, packed_x16, bias, C, M, N, K,
                    tile_m,
                    active
                );
            } else {
                ck_q4k_prefill_debug_dispatch("x16_mreuse", M, N, K, active);
                gemm_nt_q4_k_packed_meta_x16_q8_k_threaded_mreuse(
                    A, packed_x16, bias, C, M, N, K,
                    tile_m,
                    active
                );
            }
            return;
        }
    }

    if (pool && ck_should_use_q4k_packed_meta_x8_mreuse_prefill(M, N, K)) {
        void *packed_x8 = ck_get_q4k_packed_meta_x8_cached(B, N, K);
        if (packed_x8) {
            int active = ck_select_gemm_active_threads(pool, M, N, K);
            const int cap = ck_env_int_or2("CK_Q4K_PACKED_META_X8_MREUSE_THREAD_CAP", "CK_GEMM_THREAD_CAP", 20);
            if (cap > 0 && active > cap) active = cap;
            const int tile_m = ck_env_int_or2("CK_Q4K_PACKED_META_X8_MREUSE_TILE_M", "CK_PREFILL_TILE_M", 4);
            ck_q4k_prefill_debug_dispatch("x8_mreuse", M, N, K, active);
            gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_mreuse(
                A, packed_x8, bias, C, M, N, K,
                tile_m,
                active
            );
            return;
        }
    }

    if (pool && ck_should_use_q4k_packed_meta_x8mt_prefill(M, N, K)) {
        void *packed_x8 = ck_get_q4k_packed_meta_x8_cached(B, N, K);
        if (packed_x8) {
            const int active = ck_select_gemm_active_threads(pool, M, N, K);
            const int tile_m = ck_env_int_or2("CK_Q4K_PACKED_META_X8MT_TILE_M", "CK_PREFILL_TILE_M", 2);
            ck_q4k_prefill_debug_dispatch("x8_mtile", M, N, K, active);
            gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_mtile(
                A, packed_x8, bias, C, M, N, K,
                tile_m,
                active
            );
            return;
        }
    }

    if (pool && ck_should_use_q4k_packed_meta_x8_prefill(M, N, K)) {
        void *packed_x8 = ck_get_q4k_packed_meta_x8_cached(B, N, K);
        if (packed_x8) {
            const int active = ck_select_gemm_active_threads(pool, M, N, K);
            ck_q4k_prefill_debug_dispatch("x8_nsplit", M, N, K, active);
            gemm_nt_q4_k_packed_meta_x8_q8_k_threaded_nsplit(
                A, packed_x8, bias, C, M, N, K,
                active
            );
            return;
        }
    }

    if (pool && ck_should_use_q4k_packed_meta_prefill(M, N, K)) {
        void *packed = ck_get_q4k_packed_meta_cached(B, N, K);
        if (packed) {
            const int active = ck_select_gemm_active_threads(pool, M, N, K);
            gemm_args_t args = {
                .A = A, .B = packed, .bias = bias, .C = C,
                .M = M, .N = N, .K = K,
                .A_row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K),
                .tile_m = ck_env_int_or2("CK_Q4K_PACKED_META_TILE_M", "CK_PREFILL_TILE_M", 16),
                .tile_n = ck_env_int_or2("CK_Q4K_PACKED_META_TILE_N", "CK_PREFILL_TILE_N", 256)
            };
            if (ck_should_use_q4k_packed_meta_2d_prefill(pool, M, N, K, args.tile_m, args.tile_n)) {
                ck_q4k_prefill_debug_dispatch("packed_meta_2d", M, N, K, active);
                ck_threadpool_dispatch_n(pool, active, work_gemm_nt_q4_k_packed_meta_q8_k_2d, &args);
                return;
            }
            ck_q4k_prefill_debug_dispatch("packed_meta_nsplit", M, N, K, active);
            gemm_nt_q4_k_packed_meta_q8_k_threaded_nsplit(
                A, packed, bias, C, M, N, K,
                active
            );
            return;
        }
    }

    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        ck_q4k_prefill_debug_dispatch("serial", M, N, K, 1);
        gemm_nt_q4_k_q8_k(A, B, bias, C, M, N, K);
        return;
    }

    /* A is Q8_K: row_bytes = (K / QK_K) * sizeof(block_q8_K) */
    size_t A_row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };

    /* Canonical fallback safety path:
     * CK_DISABLE_Q4K_PACKED_META_PREFILL must be a reliable escape hatch for
     * debugging a new packed layout. Do not fall back to the raw Q4_K GEMM for
     * prefill M>1 here; that implementation can start its own internal
     * scheduling for large Q4_K shapes. This v8 dispatcher already owns the
     * active threadpool, so the safe canonical path is row splitting with the
     * one-token Q4_K GEMV primitive in work_gemm_nt_q4_k_q8_k().
     */
    const int active = ck_select_gemm_active_threads(pool, M, N, K);
    ck_q4k_prefill_debug_dispatch("fallback_row_gemv", M, N, K, active);
    ck_threadpool_dispatch_n(pool, active, work_gemm_nt_q4_k_q8_k, &args);
}

void gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 || (K % QK_K) != 0) return;

    ck_threadpool_t *pool = ck_threadpool_global();
    const size_t row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K);
    const int packed_rows = M - (M % 4);
    const int serial = packed_rows > 0 &&
            (!pool || ck_threadpool_n_threads(pool) <= 1 ||
             ck_should_run_gemm_serial(pool, packed_rows, N, K));
    const ck_gemm_route_v8 *x16_route = !serial
        ? ck_q4k_avx512_x16_prefill_route(packed_rows, N, K)
        : NULL;
    void *packed_vnni_x16 = NULL;
    if (x16_route) {
        packed_vnni_x16 =
                ck_get_q4k_packed_vnni_x16_cached(B, N, K);
    }
    void *packed_vnni_x8 = NULL;
    if (!packed_vnni_x16 &&
        !serial && packed_rows >= 16 && N >= 512 && K >= 1024 &&
        ck_q4k_packed_vnni_x8_available() &&
        !ck_env_enabled("CK_DISABLE_Q4K_VNNI_X8_PREFILL")) {
        packed_vnni_x8 = ck_get_q4k_packed_vnni_x8_cached(B, N, K);
    }
    void *packed_x8 = NULL;
    if ((!packed_vnni_x16 && !packed_vnni_x8) ||
        (packed_rows < M && !packed_vnni_x16)) {
        packed_x8 = ck_get_q4k_packed_meta_x8_cached(B, N, K);
        if (!packed_x8) return;
    }

    if (packed_rows > 0 &&
        serial) {
        ck_q4k_prefill_debug_dispatch(
                (N % 16) == 0 ? "pairwise_serial_x16" : "pairwise_serial_x8",
                packed_rows, N, K, 1);
        if ((N % 16) == 0) {
            gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
                A, packed_x8, bias, C, packed_rows, N, K);
        } else {
            gemm_nt_q4_k_packed_meta_x8_q8_k_superblock_order(
                A, packed_x8, bias, C, packed_rows, N, K);
        }
    } else if (packed_rows > 0) {
        int active = ck_select_gemm_active_threads(pool, packed_rows, N, K);
        /* The map-selected x16 route preserves the same pairwise split-min
         * arithmetic as x8 while doubling output lanes per dot product. */
        if (packed_vnni_x16) {
            active = ck_select_q4k_avx512_x16_threads(
                    pool, packed_rows, N, K, x16_route);
            ck_q4k_prefill_debug_dispatch(
                    "avx512_vnni_x16_16m", M, N, K, active);
            gemm_nt_q4_k_packed_vnni_x16_q8_k_split_min_threaded_16m(
                    A, packed_vnni_x16, bias, C,
                    packed_rows, N, K, active);
        } else
        if (packed_vnni_x8) {
            active = ck_select_q4k_vnni_active_threads(
                    pool, packed_rows, N, K);
            ck_q4k_prefill_debug_dispatch("vnni_x8_4m", M, N, K, active);
            gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m(
                    A, packed_vnni_x8, bias, C,
                    packed_rows, N, K, active);
        } else
        if (packed_rows >= 16 && N >= 512 && (N % 16) == 0) {
            ck_q4k_prefill_debug_dispatch(
                    "pairwise_x8_8m", packed_rows, N, K, active);
            gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_8m(
                    A, packed_x8, bias, C, packed_rows, N, K, active);
        } else {
            ck_q4k_prefill_debug_dispatch(
                    "pairwise_row_split", packed_rows, N, K, active);
            gemm_args_t args = {
                .A = A, .B = packed_x8, .bias = bias, .C = C,
                .M = packed_rows, .N = N, .K = K, .A_row_bytes = row_bytes
            };
            ck_threadpool_dispatch_n(
                pool, active, work_gemm_nt_q4_k_q8_k_pairwise_split_min, &args);
        }
    }

    /* The loaded CPU provider executes complete four-row groups through its
     * repacked matrix kernel and routes residual rows through its repacked
     * GEMV order. The two reduction boundaries are numerically distinct. */
    if (packed_rows < M) {
        const int tail_rows = M - packed_rows;
        const int tail_thread_cap = x16_route ? x16_route->max_threads : 0;
        ck_q4k_prefill_debug_dispatch(
                "pairwise_residual_gemv_parallel",
                tail_rows, N, K,
                pool ? ck_threadpool_n_threads(pool) : 1);
        /* The route metadata enables batched residual rows only where the
         * measured cache schedule wins; all paths retain exact GEMV order. */
        if (packed_vnni_x16 && x16_route &&
            (x16_route->flags & CK_GEMM_ROUTE_BATCHED_TAIL) != 0 &&
            !ck_env_enabled("CK_DISABLE_Q4K_X16_BATCHED_TAIL")) {
            run_gemv_q4_k_q8_k_repacked_x16_parallel(
                    pool,
                    (const char *)A + (size_t)packed_rows * row_bytes,
                    packed_vnni_x16,
                    bias,
                    C + (size_t)packed_rows * (size_t)N,
                    tail_rows, N, K, tail_thread_cap);
        } else {
            for (int row = 0; row < tail_rows; ++row) {
                const void *a_row =
                        (const char *)A +
                        (size_t)(packed_rows + row) * row_bytes;
                float *c_row =
                        C + (size_t)(packed_rows + row) * (size_t)N;
                if (packed_vnni_x16) {
                    run_gemv_q4_k_q8_k_repacked_x16_parallel(
                            pool, a_row, packed_vnni_x16, bias, c_row,
                            1, N, K, tail_thread_cap);
                } else {
                    run_gemv_q4_k_q8_k_repacked_parallel(
                            pool, a_row, packed_x8, bias, c_row,
                            N, K, tail_thread_cap);
                }
            }
        }
    }
}

void gemm_nt_q4_k_q8_k_segmented_pairwise_split_min_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K, const int *segment_lengths, int num_segments)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 ||
        !segment_lengths || num_segments <= 0) {
        gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
            A, B, bias, C, M, N, K);
        return;
    }

    int total_rows = 0;
    for (int segment = 0; segment < num_segments; ++segment) {
        const int rows = segment_lengths[segment];
        if (rows < 0 || rows > M - total_rows) {
            gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
                A, B, bias, C, M, N, K);
            return;
        }
        total_rows += rows;
    }
    if (total_rows != M) {
        gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
            A, B, bias, C, M, N, K);
        return;
    }

    const size_t a_row_bytes =
        (size_t)(K / QK_K) * sizeof(block_q8_K);
    int row_offset = 0;
    for (int segment = 0; segment < num_segments; ++segment) {
        const int rows = segment_lengths[segment];
        if (rows > 0) {
            gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
                (const uint8_t *)A + (size_t)row_offset * a_row_bytes,
                B,
                bias,
                C + (size_t)row_offset * (size_t)N,
                rows,
                N,
                K);
        }
        row_offset += rows;
    }
}

void gemv_q4_k_q8_k_repacked_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int N, int K)
{
    if (!y || !W || !x_q8 || N <= 0 || K <= 0 || (K % QK_K) != 0) return;

    /* Reuse x16 weights only for a map-owned decode route and when model
     * initialization prepared the exact view. */
    const ck_gemm_route_v8 *decode_route = ck_find_gemm_route_v8(
        ck_policy_gemm_nt_q4_k_q8_k_avx512_vnni_x16_decode_prepared,
        CK_POLICY_GEMM_NT_Q4_K_Q8_K_AVX512_VNNI_X16_DECODE_PREPARED_COUNT,
        1, N, K);
    if (decode_route &&
        ck_q4k_packed_vnni_x16_available()) {
        void *packed_x16 =
                ck_find_prepared_q4k_packed_vnni_x16(W, N, K);
        if (packed_x16) {
            run_gemv_q4_k_q8_k_repacked_x16_parallel(
                    ck_threadpool_global(), x_q8, packed_x16, NULL, y,
                    1, N, K, 0);
            return;
        }
    }

    void *packed_x8 = ck_get_q4k_packed_meta_x8_cached(W, N, K);
    if (!packed_x8) {
        fprintf(stderr, "Q4_K repacked decode contract: weight packing failed\n");
        abort();
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    run_gemv_q4_k_q8_k_repacked_parallel(
            pool, x_q8, packed_x8, NULL, y, N, K, 0);
}

void gemm_nt_q6_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    void *prepared = ck_strict_parity_enabled()
        ? NULL : ck_find_prepared_q6_k(B, N, K);
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        if (prepared) {
            gemm_nt_q6_k_q8_k_prepared(A, prepared, bias, C, M, N, K);
        } else {
            gemm_nt_q6_k_q8_k(A, B, bias, C, M, N, K);
        }
        return;
    }

    /* A is Q8_K: row_bytes = (K / QK_K) * sizeof(block_q8_K) */
    size_t A_row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K);

    const ck_gemm_route_v8 *schedule = ck_find_gemm_route_v8(
        ck_policy_gemm_nt_q6_k_q8_k_prefill_schedule,
        CK_POLICY_GEMM_NT_Q6_K_Q8_K_PREFILL_SCHEDULE_COUNT,
        M, N, K);
    const int default_tile_m = schedule && schedule->tile_m > 0
        ? schedule->tile_m : 16;
    const int default_tile_n = schedule && schedule->tile_n > 0
        ? schedule->tile_n : 256;
    gemm_args_t args = {
        .A = A, .B = prepared ? prepared : B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes,
        .tile_m = ck_env_int_or2("CK_PREFILL_TILE_M", NULL, default_tile_m),
        .tile_n = ck_env_int_or2("CK_PREFILL_TILE_N", NULL, default_tile_n),
        .use_q6_m4 = !prepared && ck_should_use_q6k_q8k_m4_prefill(M, N, K),
        .use_q6_prepared = prepared != NULL
    };
    int active = ck_select_gemm_active_threads(pool, M, N, K);
    if (!getenv("CK_GEMM_THREAD_CAP") && !getenv("CK_GEMV_THREAD_CAP")) {
        const int q6_profile_cap = (ck_speed_profile_qwen3vl_ocr_fast() && M >= 512 && N == 4096 && K == 12288) ? 16 : active;
        const int q6_cap = ck_env_int_or2("CK_Q6K_Q8K_THREAD_CAP", NULL, q6_profile_cap);
        active = ck_min_int(active, q6_cap);
    }
    if (ck_should_use_q6k_q8k_2d_prefill(pool, M, N, K, args.tile_m, args.tile_n)) {
        if (ck_gemm_dynamic_schedule_enabled()) {
            const int mt = ck_ceil_div_int(M, args.tile_m);
            const int nt = ck_ceil_div_int(N, args.tile_n);
            const int grain = 1;
            ck_threadpool_parallel_for_n(
                pool, active, 0, mt * nt, grain,
                work_gemm_nt_q6_k_q8_k_2d_range, &args);
        } else {
            ck_threadpool_dispatch_n(
                pool, active, work_gemm_nt_q6_k_q8_k_2d, &args);
        }
    } else {
        ck_threadpool_dispatch_n(pool, active, work_gemm_nt_q6_k_q8_k, &args);
    }
}

void gemm_nt_q6_k_q8_k_segmented_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K, const int *segment_lengths, int num_segments)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 ||
        !segment_lengths || num_segments <= 0) {
        gemm_nt_q6_k_q8_k_parallel_dispatch(A, B, bias, C, M, N, K);
        return;
    }

    int total_rows = 0;
    for (int segment = 0; segment < num_segments; ++segment) {
        const int rows = segment_lengths[segment];
        if (rows < 0 || rows > M - total_rows) {
            gemm_nt_q6_k_q8_k_parallel_dispatch(A, B, bias, C, M, N, K);
            return;
        }
        total_rows += rows;
    }
    if (total_rows != M) {
        gemm_nt_q6_k_q8_k_parallel_dispatch(A, B, bias, C, M, N, K);
        return;
    }

    const size_t a_row_bytes =
        (size_t)(K / QK_K) * sizeof(block_q8_K);
    int row_offset = 0;
    for (int segment = 0; segment < num_segments; ++segment) {
        const int rows = segment_lengths[segment];
        if (rows > 0) {
            gemm_nt_q6_k_q8_k_parallel_dispatch(
                (const uint8_t *)A + (size_t)row_offset * a_row_bytes,
                B,
                bias,
                C + (size_t)row_offset * (size_t)N,
                rows,
                N,
                K);
        }
        row_offset += rows;
    }
}

void gemm_nt_q5_1_q8_1_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q5_1_q8_1_m4(A, B, bias, C, M, N, K);
        return;
    }

    /* A is FP32 token-major [M, K] */
    size_t A_row_bytes = (size_t)K * sizeof(float);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes,
        .tile_m = M >= 64 ? 8 : 4
    };
    int active = ck_select_gemm_active_threads(pool, M, N, K);
    if (ck_gemm_dynamic_schedule_enabled()) {
        const int jobs = ck_ceil_div_int(M, args.tile_m);
        active = ck_min_int(active, jobs);
        ck_threadpool_parallel_for_n(
            pool, active, 0, jobs, 1,
            work_gemm_nt_q5_1_q8_1_reuse_range, &args);
    } else {
        ck_threadpool_dispatch_n(
            pool, active, work_gemm_nt_q5_1_q8_1, &args);
    }
}

void geglu_forward_exact_parallel_dispatch(
    const float *input, float *output, int tokens, int dim)
{
    if (!input || !output || tokens <= 0 || dim <= 0) return;

    const size_t input_bytes = (size_t)tokens * 2u * (size_t)dim * sizeof(*input);
    const size_t output_bytes = (size_t)tokens * (size_t)dim * sizeof(*output);
    const uintptr_t input_begin = (uintptr_t)input;
    const uintptr_t input_end = input_begin + input_bytes;
    const uintptr_t output_begin = (uintptr_t)output;
    const uintptr_t output_end = output_begin + output_bytes;
    const int buffers_overlap = output_begin < input_end && input_begin < output_end;

    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || tokens <= 1 || buffers_overlap) {
        geglu_forward_exact(input, output, tokens, dim);
        return;
    }

    geglu_args_t args = {
        .input = input,
        .output = output,
        .tokens = tokens,
        .dim = dim,
    };
    const int active = ck_min_int(ck_threadpool_n_threads(pool), tokens);
    if (ck_gemm_dynamic_schedule_enabled()) {
        ck_threadpool_parallel_for_n(
            pool, active, 0, tokens, 1,
            work_geglu_exact_range, &args);
    } else {
        ck_threadpool_dispatch_n(
            pool, active, work_geglu_exact_rows, &args);
    }
}

static void gemm_nt_q5_k_parallel_dispatch_impl(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K, void *scratch, size_t scratch_bytes)
{
    void *prepared = ck_find_prepared_q5_k(B, N, K);
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        if (prepared) {
            if (M <= 256) gemm_nt_q5_k_prepared_m4(A, prepared, bias, C, M, N, K);
            else gemm_nt_q5_k_prepared(A, prepared, bias, C, M, N, K);
        }
        else gemm_nt_q5_k(A, B, bias, C, M, N, K);
        return;
    }

    /* A is FP32 token-major [M, K]. Prepared medium/long prefill quantizes
     * it once, then partitions output columns so each worker repeatedly uses
     * a cache-sized weight slice across all rows. */
    size_t A_row_bytes = (size_t)K * sizeof(float);

    if (prepared && M >= 64 && M <= 256 && (K % QK_K) == 0) {
        const int blocks_per_row = K / QK_K;
        const size_t q8_bytes =
            (size_t)M * (size_t)blocks_per_row * sizeof(block_q8_K);
        if (scratch && scratch_bytes >= q8_bytes) {
            block_q8_K *A_q8 = (block_q8_K *)scratch;
            for (int m = 0; m < M; ++m) {
                quantize_row_q8_k(
                    A + (size_t)m * K,
                    A_q8 + (size_t)m * blocks_per_row, K);
            }
            gemm_args_t nsplit_args = {
                .A = A_q8, .B = prepared, .bias = bias, .C = C,
                .M = M, .N = N, .K = K,
                .tile_n = 64,
            };
            const int jobs = ck_ceil_div_int(N, nsplit_args.tile_n);
            const int active = ck_min_int(
                ck_select_gemm_active_threads(pool, M, N, K), jobs);
            ck_threadpool_parallel_for_n(
                pool, active, 0, jobs, 1,
                work_gemm_nt_q5_k_prepared_nrange, &nsplit_args);
            return;
        }
    }

    gemm_args_t args = {
        .A = A, .B = prepared ? prepared : B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(
        pool, ck_select_gemm_active_threads(pool, M, N, K),
        prepared ? (M <= 256 ? work_gemm_nt_q5_k_prepared_m4
                             : work_gemm_nt_q5_k_prepared)
                 : work_gemm_nt_q5_k,
        &args);
}

void gemm_nt_q5_k_parallel_dispatch_with_scratch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K, void *scratch, size_t scratch_bytes)
{
    gemm_nt_q5_k_parallel_dispatch_impl(
        A, B, bias, C, M, N, K, scratch, scratch_bytes);
}

void gemm_nt_q5_k_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    /* Compatibility callers remain allocation-free. Generated runtimes pass
     * planner-owned scratch through the map-owned ABI above. */
    gemm_nt_q5_k_parallel_dispatch_impl(A, B, bias, C, M, N, K, NULL, 0);
}
