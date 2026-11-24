/**
 * @file ck_parallel_decode.c
 * @brief Thread-pool-parallel GEMV dispatch for v7 decode
 *
 * Wraps each GEMV kernel call in a threadpool dispatch.
 * Each wrapper:
 *   1. Packs arguments into a stack-local struct
 *   2. Dispatches to threadpool (all threads execute _parallel_simd variant)
 *   3. Returns when all threads complete
 *
 * For single-thread builds (n_threads=1), dispatch() fast-paths to
 * a direct call with no overhead.
 */

#include "ck_parallel_decode_v8.h"
#include "ck_threadpool.h"
#include "ckernel_engine.h"
#include "ckernel_quant.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* External parallel GEMV kernels (defined in src/kernels/) */
extern void gemv_q5_0_parallel_simd(float *y, const void *W, const float *x,
                                     int M, int K, int ith, int nth);
extern void gemv_q5_0_q8_0_parallel_simd(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);
extern void gemv_q6_k_q8_k_parallel_simd(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);
extern void gemv_q8_0_q8_0_parallel_simd(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);

/* Q4_K parallel SIMD — check if it exists, otherwise fall back */
extern void gemv_q4_k_q8_k_parallel_simd(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);
extern void gemv_q4_k_q8_k_parallel_vnni(float *y, const void *W, const void *x_q8,
                                          int M, int K, int ith, int nth);
extern void gemv_q4_k_q8_k_parallel(float *y, const void *W, const void *x_q8,
                                     int M, int K, int ith, int nth);

/* Serial fallbacks (for correctness reference) */
extern void gemv_q5_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q6_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q4_k_q8_k(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q8_0_q8_0(float *y, const void *W, const void *x_q8, int M, int K);
extern void gemv_q5_1_q8_1(float *y, const void *W, const float *x, int M, int K);
extern void gemv_q5_k(float *y, const void *W, const float *x, int M, int K);
extern void gemv_q8_0_q8_0_contract(float *y, const void *W, const float *x, int M, int K);

extern void gated_deltanet_llama_avx2_forward(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int num_heads, int group_count, int state_dim, float norm_eps);
extern void gated_deltanet_llama_avx2_forward_head_range(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int num_heads, int group_count, int state_dim, float norm_eps,
    int head_begin, int head_end);

/* Serial fused fallback */
extern void gemv_fused_q5_0_bias_dispatch(float *y, const void *W, const float *x,
                                           const float *bias, int M, int K);

/* Quantization (from ckernel_quant.h / quantize.c) */
extern void quantize_row_q8_0(const float *x, void *vy, int k);

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

void ck_parallel_decode_init(void)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (pool) {
        fprintf(stderr, "[CK parallel decode] Initialized with %d threads\n",
                ck_threadpool_n_threads(pool));
    }
}

void ck_parallel_decode_shutdown(void)
{
    ck_threadpool_global_destroy();
}

/* ============================================================================
 * Argument Packing Structs
 *
 * Each GEMV dispatch needs to pass (y, W, x, M, K) to all threads.
 * We pack into a stack-local struct and pass its address.
 * ============================================================================ */

typedef struct {
    float      *y;
    const void *W;
    const void *x_q8;
    int         M;
    int         K;
} gemv_args_t;

typedef struct {
    float       *y;
    const void  *W;
    const float *x;
    int          M;
    int          K;
    size_t       W_row_bytes;
} gemv_fp32_args_t;

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    const float *g;
    const float *beta;
    const float *state_in;
    float *state_out;
    float *out;
    int num_heads;
    int group_count;
    int state_dim;
    float norm_eps;
} deltanet_decode_args_t;

static int ck_min_int(int a, int b) { return a < b ? a : b; }

static int ck_env_enabled(const char *name)
{
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static int ck_env_int_or(const char *name, int fallback)
{
    const char *v = getenv(name);
    if (!v || !v[0]) return fallback;
    int parsed = atoi(v);
    return parsed > 0 ? parsed : fallback;
}

static int ck_shape_aware_enabled(const ck_threadpool_t *pool)
{
    const int pool_threads = ck_threadpool_n_threads(pool);
    if (ck_env_enabled("CK_DISABLE_SHAPE_AWARE_THREADPOOL")) return 0;
    if (getenv("CK_GEMV_THREAD_CAP") || getenv("CK_GEMV_SMALL_SERIAL_THRESHOLD")) return 1;
    return pool_threads > 16;
}

static int ck_select_gemv_active_threads(const ck_threadpool_t *pool, int M, int K)
{
    const int pool_threads = ck_threadpool_n_threads(pool);
    if (pool_threads <= 1 || M <= 1 || K <= 0) return 1;
    if (!ck_shape_aware_enabled(pool)) return pool_threads;
    if (M >= 4096) return ck_min_int(pool_threads, ck_env_int_or("CK_GEMV_THREAD_CAP", 32));
    if (M >= 512) return ck_min_int(pool_threads, ck_env_int_or("CK_GEMV_THREAD_CAP", 12));
    return 1;
}

static int ck_select_fused_q5_active_threads(const ck_threadpool_t *pool, int M, int K)
{
    const int pool_threads = ck_threadpool_n_threads(pool);
    if (pool_threads <= 1 || M <= 1 || K <= 0) return 1;
    if (!ck_shape_aware_enabled(pool)) return pool_threads;
    if (M < 512) return 1;
    if (M < 1024 && K < 2048) return 1;
    return ck_min_int(pool_threads, ck_env_int_or("CK_GEMV_THREAD_CAP", 24));
}

static int ck_should_run_gemv_serial(const ck_threadpool_t *pool, int M, int K)
{
    const int pool_threads = ck_threadpool_n_threads(pool);
    if (!ck_shape_aware_enabled(pool)) return 0;
    if (M < ck_env_int_or("CK_GEMV_SMALL_SERIAL_THRESHOLD", 512)) return 1;
    return pool_threads > 16 && M < 1024 && K < 2048;
}

/* Q5_K weights are row-major packed super-blocks.
 * Keep a local layout copy to compute byte offsets without pulling in
 * kernel-private headers. Must stay ggml-compatible. */
typedef struct {
    ck_half d;
    ck_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} ck_block_q5_K_layout_t;

/* ============================================================================
 * Work Functions (called on each thread)
 * ============================================================================ */

static void work_gemv_q5_0_q8_0(int ith, int nth, void *args)
{
    const gemv_args_t *a = (const gemv_args_t *)args;
    gemv_q5_0_q8_0_parallel_simd(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
}

static void work_gated_deltanet_llama_avx2(
    int ith, int nth, void *opaque)
{
    const deltanet_decode_args_t *a =
        (const deltanet_decode_args_t *)opaque;
    const int head_begin = (a->num_heads * ith) / nth;
    const int head_end = (a->num_heads * (ith + 1)) / nth;
    gated_deltanet_llama_avx2_forward_head_range(
        a->q, a->k, a->v, a->g, a->beta,
        a->state_in, a->state_out, a->out,
        a->num_heads, a->group_count, a->state_dim, a->norm_eps,
        head_begin, head_end);
}

void gated_deltanet_llama_avx2_parallel_forward(
    const float *q, const float *k, const float *v,
    const float *g, const float *beta,
    const float *state_in, float *state_out, float *out,
    int num_heads, int group_count, int state_dim, float norm_eps)
{
#if defined(__AVX2__)
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || num_heads <= 1) {
        gated_deltanet_llama_avx2_forward(
            q, k, v, g, beta, state_in, state_out, out,
            num_heads, group_count, state_dim, norm_eps);
        return;
    }
    int active = ck_threadpool_n_threads(pool);
    if (active > num_heads) active = num_heads;
    deltanet_decode_args_t args = {
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .state_in = state_in,
        .state_out = state_out,
        .out = out,
        .num_heads = num_heads,
        .group_count = group_count,
        .state_dim = state_dim,
        .norm_eps = norm_eps,
    };
    ck_threadpool_dispatch_n(
        pool, active, work_gated_deltanet_llama_avx2, &args);
#else
    gated_deltanet_llama_avx2_forward(
        q, k, v, g, beta, state_in, state_out, out,
        num_heads, group_count, state_dim, norm_eps);
#endif
}

/* ============================================================================
 * RWKV-7 channelmix decode (T=1): row-parallel dispatch
 *
 * The serial provider runs two row-major GEMVs:
 *   hidden[j] = relu(Wk[j,:] @ xk)^2     (Wk [K,C])
 *   out[i]    = Wv[i,:] @ hidden         (Wv [C,K])
 * Both passes are row-independent, and every output element is reduced over
 * the same operands in the same order as the serial kernel. Splitting rows
 * across workers therefore produces bitwise-identical results.
 *
 * Any binding the serial kernel would reject (or that this dispatcher cannot
 * prove safe) is delegated to the serial provider, so its error contract is
 * preserved rather than re-implemented.
 * ========================================================================== */

static int ck_parallel_ranges_overlap(const void *a, size_t a_bytes,
                                      const void *b, size_t b_bytes)
{
    const uintptr_t a0 = (uintptr_t)a;
    const uintptr_t b0 = (uintptr_t)b;
    if (a_bytes > UINTPTR_MAX - a0 || b_bytes > UINTPTR_MAX - b0) return 1;
    return a0 < b0 + b_bytes && b0 < a0 + a_bytes;
}

/* Mirror the serial provider's alias rules exactly. The in-place carries it
 * allows (shift_out == shift_in, shift_out == x, out == x) are also safe with
 * row splitting: every read of those inputs happens in the serial prologue or
 * before the first write to the aliased output. */
int ck_rwkv7_channelmix_parallel_binding_ok(
    const float *x, const float *shift_in, const float *mu,
    const float *Wk, const float *Wv,
    float *shift_out, float *out,
    float *scratch, int dim, int hidden_dim)
{
    if (dim <= 0 || hidden_dim <= 0 || (size_t)dim > SIZE_MAX / sizeof(float)) return 0;
    const size_t vec_bytes = (size_t)dim * sizeof(float);
    if ((size_t)hidden_dim > SIZE_MAX / vec_bytes) return 0;
    const size_t weight_bytes = vec_bytes * (size_t)hidden_dim;
    const size_t scratch_bytes =
        ((size_t)dim + (size_t)hidden_dim) * sizeof(float);
    const float *const inputs[5] = {x, shift_in, mu, Wk, Wv};
    const size_t sizes[5] = {
        vec_bytes, vec_bytes, vec_bytes, weight_bytes, weight_bytes,
    };
    for (int j = 0; j < 5; ++j) {
        if (ck_parallel_ranges_overlap(
                scratch, scratch_bytes, inputs[j], sizes[j])) {
            return 0;
        }
        const int shift_out_carry_ok =
            (j == 0 && shift_out == x) || (j == 1 && shift_out == shift_in);
        if (!shift_out_carry_ok &&
            ck_parallel_ranges_overlap(
                shift_out, vec_bytes, inputs[j], sizes[j])) {
            return 0;
        }
        const int out_carry_ok = (j == 0 && out == x);
        if (!out_carry_ok &&
            ck_parallel_ranges_overlap(out, vec_bytes, inputs[j], sizes[j])) {
            return 0;
        }
    }
    return !ck_parallel_ranges_overlap(scratch, scratch_bytes, shift_out, vec_bytes) &&
           !ck_parallel_ranges_overlap(scratch, scratch_bytes, out, vec_bytes) &&
           !ck_parallel_ranges_overlap(shift_out, vec_bytes, out, vec_bytes);
}

typedef struct {
    const float *xk;
    const float *Wk;
    float *hidden;
    int dim;
    int hidden_dim;
} ck_rwkv7_channelmix_k_args_t;

static void ck_rwkv7_channelmix_k_rows(int begin, int end, void *opaque)
{
    const ck_rwkv7_channelmix_k_args_t *a =
        (const ck_rwkv7_channelmix_k_args_t *)opaque;
    /* Same primitive the serial provider runs: bitwise-identical by
     * construction, not by convention. */
    ck_rwkv7_channelmix_hidden_rows(
        a->xk, a->Wk, a->hidden, a->dim, a->hidden_dim, begin, end);
}

typedef struct {
    const float *hidden;
    const float *Wv;
    float *out;
    int dim;
    int hidden_dim;
} ck_rwkv7_channelmix_v_args_t;

static void ck_rwkv7_channelmix_v_rows(int begin, int end, void *opaque)
{
    const ck_rwkv7_channelmix_v_args_t *a =
        (const ck_rwkv7_channelmix_v_args_t *)opaque;
    ck_rwkv7_channelmix_out_rows(
        a->hidden, a->Wv, a->out, a->dim, a->hidden_dim, begin, end);
}

static int ck_rwkv7_channelmix_rows_per_worker(int rows, int active)
{
    int grain = rows / (active * 4);
    if (grain < 8) grain = 8;
    if (grain > rows) grain = rows;
    return grain;
}

int ck_rwkv7_channelmix_decode_parallel(const float *x,
                                        const float *shift_in,
                                        const float *mu,
                                        const float *Wk,
                                        const float *Wv,
                                        float *shift_out,
                                        float *out,
                                        int dim,
                                        int hidden_dim,
                                        float *scratch,
                                        size_t scratch_elems)
{
    if (!x || !shift_in || !mu || !Wk || !Wv || !shift_out || !out ||
        dim <= 0 || hidden_dim <= 0 || hidden_dim > 8192 || !scratch ||
        scratch_elems < (size_t)dim + (size_t)hidden_dim) {
        return ck_rwkv7_channelmix_decode(
            x, shift_in, mu, Wk, Wv, shift_out, out, dim, hidden_dim,
            scratch, scratch_elems);
    }

    /* Require a binding the serial provider accepts. In-place carries it
     * allows (shift_out == shift_in, out == x) stay on the parallel path; the
     * dispatcher only splits rows, and those reads complete first. */
    if (!ck_rwkv7_channelmix_parallel_binding_ok(
            x, shift_in, mu, Wk, Wv, shift_out, out,
            scratch, dim, hidden_dim)) {
        return ck_rwkv7_channelmix_decode(
            x, shift_in, mu, Wk, Wv, shift_out, out, dim, hidden_dim,
            scratch, scratch_elems);
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    const int pool_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const size_t macs = (size_t)dim * (size_t)hidden_dim;
    if (!pool || pool_threads <= 1 || ck_threadpool_in_dispatch(pool) || dim < 64 ||
        macs < (size_t)(1u << 20)) {
        return ck_rwkv7_channelmix_decode(
            x, shift_in, mu, Wk, Wv, shift_out, out, dim, hidden_dim,
            scratch, scratch_elems);
    }

    float *xk = scratch;
    float *hidden = scratch + dim;
    for (int i = 0; i < dim; ++i) {
        xk[i] = x[i] + mu[i] * (shift_in[i] - x[i]);
        shift_out[i] = x[i];
    }

    int active = pool_threads;
    if (active > hidden_dim) active = hidden_dim;
    ck_rwkv7_channelmix_k_args_t k_args = {
        .xk = xk, .Wk = Wk, .hidden = hidden,
        .dim = dim, .hidden_dim = hidden_dim,
    };
    ck_threadpool_parallel_for_n(
        pool, active, 0, hidden_dim,
        ck_rwkv7_channelmix_rows_per_worker(hidden_dim, active),
        ck_rwkv7_channelmix_k_rows, &k_args);

    active = pool_threads;
    if (active > dim) active = dim;
    ck_rwkv7_channelmix_v_args_t v_args = {
        .hidden = hidden, .Wv = Wv, .out = out,
        .dim = dim, .hidden_dim = hidden_dim,
    };
    ck_threadpool_parallel_for_n(
        pool, active, 0, dim,
        ck_rwkv7_channelmix_rows_per_worker(dim, active),
        ck_rwkv7_channelmix_v_rows, &v_args);
    return 0;
}

static void work_gemv_q6_k_q8_k(int ith, int nth, void *args)
{
    const gemv_args_t *a = (const gemv_args_t *)args;
    gemv_q6_k_q8_k_parallel_simd(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
}

static void work_gemv_q4_k_q8_k(int ith, int nth, void *args)
{
    const gemv_args_t *a = (const gemv_args_t *)args;
#if defined(CK_TARGET_ARM)
    gemv_q4_k_q8_k_parallel(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
#else
    const int rows_per_worker = (a->M + nth - 1) / nth;
    const int row_begin = rows_per_worker * ith;
    const int row_end = (row_begin + rows_per_worker < a->M)
        ? row_begin + rows_per_worker
        : a->M;
    if (row_begin >= a->M) {
        return;
    }

    const size_t row_bytes = (size_t)(a->K / QK_K) * sizeof(block_q4_K);
    gemv_q4_k_q8_k(
        a->y + row_begin,
        (const char *)a->W + (size_t)row_begin * row_bytes,
        a->x_q8,
        row_end - row_begin,
        a->K);
#endif
}

static void work_gemv_q8_0_q8_0(int ith, int nth, void *args)
{
    const gemv_args_t *a = (const gemv_args_t *)args;
    gemv_q8_0_q8_0_parallel_simd(a->y, a->W, a->x_q8, a->M, a->K, ith, nth);
}

static void work_gemv_q5_1_q8_1(int ith, int nth, void *args)
{
    const gemv_fp32_args_t *a = (const gemv_fp32_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemv_q5_1_q8_1(
        a->y + r0,
        (const char *)a->W + (size_t)r0 * a->W_row_bytes,
        a->x,
        r1 - r0,
        a->K
    );
}

static void work_gemv_q5_k(int ith, int nth, void *args)
{
    const gemv_fp32_args_t *a = (const gemv_fp32_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    gemv_q5_k(
        a->y + r0,
        (const char *)a->W + (size_t)r0 * a->W_row_bytes,
        a->x,
        r1 - r0,
        a->K
    );
}

/* ============================================================================
 * Parallel Dispatch Wrappers
 *
 * These have the SAME signature as the serial GEMV functions.
 * They pack args and dispatch to the threadpool.
 * ============================================================================ */

void gemv_q5_0_q8_0_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || ck_should_run_gemv_serial(pool, M, K)) {
        /* Fall back to serial — avoids overhead for single-thread */
        gemv_q5_0_q8_0(y, W, x_q8, M, K);
        return;
    }

    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8, .M = M, .K = K };
    ck_threadpool_dispatch_n(pool, ck_select_gemv_active_threads(pool, M, K), work_gemv_q5_0_q8_0, &args);
}

void gemv_q6_k_q8_k_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || ck_should_run_gemv_serial(pool, M, K)) {
        gemv_q6_k_q8_k(y, W, x_q8, M, K);
        return;
    }

    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8, .M = M, .K = K };
    ck_threadpool_dispatch_n(pool, ck_select_gemv_active_threads(pool, M, K), work_gemv_q6_k_q8_k, &args);
}

void gemv_q4_k_q8_k_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || ck_should_run_gemv_serial(pool, M, K)) {
        gemv_q4_k_q8_k(y, W, x_q8, M, K);
        return;
    }

    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8, .M = M, .K = K };
    ck_threadpool_dispatch_n(pool, ck_select_gemv_active_threads(pool, M, K), work_gemv_q4_k_q8_k, &args);
}

void gemv_q8_0_q8_0_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || ck_should_run_gemv_serial(pool, M, K)) {
        gemv_q8_0_q8_0(y, W, x_q8, M, K);
        return;
    }

    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8, .M = M, .K = K };
    ck_threadpool_dispatch_n(pool, ck_select_gemv_active_threads(pool, M, K), work_gemv_q8_0_q8_0, &args);
}

void gemv_q5_1_q8_1_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || ck_should_run_gemv_serial(pool, M, K) || M <= 1 || (K % QK5_1) != 0) {
        gemv_q5_1_q8_1(y, W, x, M, K);
        return;
    }

    const size_t W_row_bytes = (size_t)(K / QK5_1) * sizeof(block_q5_1);
    gemv_fp32_args_t args = {
        .y = y, .W = W, .x = x, .M = M, .K = K, .W_row_bytes = W_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemv_active_threads(pool, M, K), work_gemv_q5_1_q8_1, &args);
}

void gemv_q5_k_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || ck_should_run_gemv_serial(pool, M, K) || M <= 1 || (K % QK_K) != 0) {
        gemv_q5_k(y, W, x, M, K);
        return;
    }

    const size_t W_row_bytes = (size_t)(K / QK_K) * sizeof(ck_block_q5_K_layout_t);
    gemv_fp32_args_t args = {
        .y = y, .W = W, .x = x, .M = M, .K = K, .W_row_bytes = W_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemv_active_threads(pool, M, K), work_gemv_q5_k, &args);
}

/* Keep stack footprint aligned with the contract kernel implementation. */
#define CK_Q80_STACK_Q8_BLOCKS 1024

void gemv_q8_0_q8_0_contract_parallel_dispatch(
    float *y, const void *W, const float *x, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || ck_should_run_gemv_serial(pool, M, K)) {
        gemv_q8_0_q8_0_contract(y, W, x, M, K);
        return;
    }

    if ((K % QK8_0) != 0) {
        gemv_q8_0_q8_0_contract(y, W, x, M, K);
        return;
    }

    const int n_blocks = K / QK8_0;
    if (n_blocks <= 0 || n_blocks > CK_Q80_STACK_Q8_BLOCKS) {
        gemv_q8_0_q8_0_contract(y, W, x, M, K);
        return;
    }

    block_q8_0 x_q8_buf[n_blocks];
    quantize_row_q8_0(x, x_q8_buf, K);
    gemv_q8_0_q8_0_parallel_dispatch(y, W, x_q8_buf, M, K);
}

/* ============================================================================
 * Fused GEMV Parallel Dispatch
 *
 * gemv_fused_q5_0_bias = quantize(x) + gemv(W, x_q8) + bias
 *
 * Split into:
 *   1. Quantize FP32 x → Q8_0 (serial, fast: O(K))
 *   2. Parallel GEMV across threads (the expensive part: O(M*K))
 *   3. Add bias (serial, fast: O(M))
 * ============================================================================ */

/* Max K we support on stack: 16384 elements = ~17KB Q8_0 buffer.
 * Qwen2-0.5B max K is 4864, so this is plenty. */
#define FUSED_Q8_STACK_MAX 16384

void gemv_fused_q5_0_bias_parallel_dispatch(
    float *y, const void *W, const float *x, const float *bias, int M, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || ck_should_run_gemv_serial(pool, M, K)) {
        /* Fall back to original fused kernel (serial) */
        gemv_fused_q5_0_bias_dispatch(y, W, x, bias, M, K);
        return;
    }

    /* Step 1: Quantize FP32 x to Q8_0 (serial, fast) */
    int n_blocks = K / QK8_0;
    block_q8_0 x_q8_buf[n_blocks]; /* VLA on stack — safe for K <= ~16K */

    quantize_row_q8_0(x, x_q8_buf, K);

    /* Step 2: Parallel GEMV — reuse existing parallel dispatch */
    gemv_args_t args = { .y = y, .W = W, .x_q8 = x_q8_buf, .M = M, .K = K };
    ck_threadpool_dispatch_n(pool, ck_select_fused_q5_active_threads(pool, M, K), work_gemv_q5_0_q8_0, &args);

    /* Step 3: Add bias (serial, fast) */
    if (bias) {
        for (int i = 0; i < M; i++) {
            y[i] += bias[i];
        }
    }
}
