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
 * The optional 2D scheduler is a load-balancing tool, not a universally faster
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
 * Therefore CK_ENABLE_Q6K_Q8K_2D_PREFILL is still opt-in, and the dispatcher
 * additionally gates it to wide Q6 shapes by default. Use
 * CK_FORCE_Q6K_Q8K_2D_PREFILL=1 only for benchmarking raw 2D behavior.
 *
 * Reuses the same global thread pool as decode (ck_threadpool_global()).
 */

#include "ck_parallel_prefill_v8.h"
#include "ck_threadpool.h"
#include "ckernel_quant.h"
#include "ck_speed_profiles.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

/* Serial GEMM kernels (defined in src/kernels/) */
extern void gemm_nt_q5_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q8_0_q8_0(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
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
extern void pack_q4_k_to_packed_vnni_x8(
    const void *src, void *dst, int N, int K);
extern void gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m(
    const void *A_q8, const void *B_packed_vnni_x8, const float *bias,
    float *C, int M, int N, int K, int threads);
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
extern void gemm_nt_q4_k_packed_meta_q8_k_tile(const void *A_q8,
                                               const void *B_packed,
                                               const float *bias,
                                               float *C,
                                               int M, int N, int K,
                                               int m0, int m1, int n0, int n1);
extern void gemm_nt_q6_k_q8_k(const void *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void swiglu_forward_exact(const float *input, float *output, int tokens, int dim);
extern void gemm_nt_q6_k_q8_k_tile(const void *A, const void *B, const float *bias,
                                    float *C, int M, int N, int K,
                                    int m0, int m1, int n0, int n1);
extern void gemm_nt_q6_k_q8_k_tiled(const void *A, const void *B, const float *bias,
                                      float *C, int M, int N, int K);
extern void gemm_nt_q5_1_q8_1(const float *A, const void *B, const float *bias,
                                float *C, int M, int N, int K);
extern void gemm_nt_q5_k(const float *A, const void *B, const float *bias,
                          float *C, int M, int N, int K);

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
} gemm_args_t;

typedef struct {
    const void  *A;
    const void  *B_packed_x8;
    const float *bias;
    float       *C;
    int          N;
    int          K;
} q4k_repacked_gemv_args_t;

static int ck_min_int(int a, int b) { return a < b ? a : b; }

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


typedef struct ck_q4k_packed_meta_x16_cache_entry {
    const void *src;
    int N;
    int K;
    void *packed;
    struct ck_q4k_packed_meta_x16_cache_entry *next;
} ck_q4k_packed_meta_x16_cache_entry_t;

static pthread_mutex_t ck_q4k_packed_meta_x16_cache_mu = PTHREAD_MUTEX_INITIALIZER;
static ck_q4k_packed_meta_x16_cache_entry_t *ck_q4k_packed_meta_x16_cache_head = NULL;

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
    if (!ck_env_enabled("CK_ENABLE_Q6K_Q8K_2D_PREFILL")) return 0;
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

    /* 2D tiling pays off only when the output dimension is wide enough that
     * N-side job balance offsets the extra activation rereads. Narrow MLP-down
     * shapes such as Qwen2/Gemma/Qwen3.5 remain faster with row splitting. */
    if (N < 2048 || K < 8192) return 0;

    return 1;
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

static void work_gemm_nt_q6_k_q8_k(int ith, int nth, void *args)
{
    const gemm_args_t *a = (const gemm_args_t *)args;
    int dr = (a->M + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < a->M) ? (r0 + dr) : a->M;
    if (r0 >= a->M) return;

    if (ck_env_enabled("CK_ENABLE_Q6K_Q8K_TILED_PREFILL")) {
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

static void work_gemm_nt_q6_k_q8_k_2d(int ith, int nth, void *args)
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
        gemm_nt_q6_k_q8_k_tile(a->A, a->B, a->bias, a->C,
                               a->M, a->N, a->K, m0, m1, n0, n1);
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

    gemm_nt_q5_1_q8_1(
        (const float *)((const char *)a->A + (size_t)r0 * a->A_row_bytes),
        a->B,
        a->bias,
        a->C + (size_t)r0 * a->N,
        r1 - r0, a->N, a->K
    );
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
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q5_0_q8_0, &args);
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
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q8_0_q8_0, &args);
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
    void *packed_vnni = NULL;
#if defined(__AVXVNNI__) || \
    (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    if (!serial && packed_rows >= 16 && N >= 512 && K >= 1024 &&
        !ck_env_enabled("CK_DISABLE_Q4K_VNNI_X8_PREFILL")) {
        packed_vnni = ck_get_q4k_packed_vnni_x8_cached(B, N, K);
    }
#endif
    void *packed_x8 = NULL;
    if (!packed_vnni || packed_rows < M) {
        packed_x8 = ck_get_q4k_packed_meta_x8_cached(B, N, K);
        if (!packed_x8) return;
    }

    if (packed_rows > 0 &&
        serial) {
        if ((N % 16) == 0) {
            gemm_nt_q4_k_packed_meta_x16_q8_k_llama_order(
                A, packed_x8, bias, C, packed_rows, N, K);
        } else {
            gemm_nt_q4_k_packed_meta_x8_q8_k_superblock_order(
                A, packed_x8, bias, C, packed_rows, N, K);
        }
    } else if (packed_rows > 0) {
        int active = ck_select_gemm_active_threads(pool, packed_rows, N, K);
        /* The VNNI layout interleaves Q4 bytes across eight output columns,
         * so each dot-product lane advances one output without horizontal
         * reduction. Packing is cached by weight identity and does not occur
         * in the steady-state call. Exact 8M remains the allocation/ISA
         * fallback and retains the same pairwise split-min contract. */
#if defined(__AVXVNNI__) || \
    (defined(__AVX512VNNI__) && defined(__AVX512VL__))
        if (packed_vnni) {
            active = ck_select_q4k_vnni_active_threads(
                    pool, packed_rows, N, K);
            ck_q4k_prefill_debug_dispatch("vnni_x8_4m", M, N, K, active);
            gemm_nt_q4_k_packed_vnni_x8_q8_k_split_min_threaded_4m(
                    A, packed_vnni, bias, C, packed_rows, N, K, active);
        } else
#endif
        if (packed_rows >= 16 && N >= 512 && (N % 16) == 0) {
            gemm_nt_q4_k_packed_meta_x8_q8_k_split_min_threaded_8m(
                    A, packed_x8, bias, C, packed_rows, N, K, active);
        } else {
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
        gemm_nt_q4_k_packed_meta_x8_q8_k_gemv_order(
            (const char *)A + (size_t)packed_rows * row_bytes,
            packed_x8, bias, C + (size_t)packed_rows * (size_t)N,
            M - packed_rows, N, K);
    }
}

void gemv_q4_k_q8_k_repacked_parallel_dispatch(
    float *y, const void *W, const void *x_q8, int N, int K)
{
    if (!y || !W || !x_q8 || N <= 0 || K <= 0 || (K % QK_K) != 0) return;

    void *packed_x8 = ck_get_q4k_packed_meta_x8_cached(W, N, K);
    if (!packed_x8) {
        fprintf(stderr, "Q4_K repacked decode contract: weight packing failed\n");
        abort();
    }

    q4k_repacked_gemv_args_t args = {
        .A = x_q8, .B_packed_x8 = packed_x8, .bias = NULL,
        .C = y, .N = N, .K = K
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    const int groups = (N + 7) / 8;
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > groups) active = groups;
    if (active <= 1) {
        work_gemv_q4_k_q8_k_repacked(0, 1, &args);
        return;
    }
    ck_threadpool_dispatch_n(pool, active, work_gemv_q4_k_q8_k_repacked, &args);
}

void gemm_nt_q6_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q6_k_q8_k(A, B, bias, C, M, N, K);
        return;
    }

    /* A is Q8_K: row_bytes = (K / QK_K) * sizeof(block_q8_K) */
    size_t A_row_bytes = (size_t)(K / QK_K) * sizeof(block_q8_K);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes,
        .tile_m = ck_env_int_or2("CK_PREFILL_TILE_M", NULL, 16),
        .tile_n = ck_env_int_or2("CK_PREFILL_TILE_N", NULL, 256)
    };
    int active = ck_select_gemm_active_threads(pool, M, N, K);
    if (!getenv("CK_GEMM_THREAD_CAP") && !getenv("CK_GEMV_THREAD_CAP")) {
        const int q6_profile_cap = (ck_speed_profile_qwen3vl_ocr_fast() && M >= 512 && N == 4096 && K == 12288) ? 16 : active;
        const int q6_cap = ck_env_int_or2("CK_Q6K_Q8K_THREAD_CAP", NULL, q6_profile_cap);
        active = ck_min_int(active, q6_cap);
    }
    if (ck_should_use_q6k_q8k_2d_prefill(pool, M, N, K, args.tile_m, args.tile_n)) {
        ck_threadpool_dispatch_n(pool, active, work_gemm_nt_q6_k_q8_k_2d, &args);
    } else {
        ck_threadpool_dispatch_n(pool, active, work_gemm_nt_q6_k_q8_k, &args);
    }
}

void gemm_nt_q5_1_q8_1_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q5_1_q8_1(A, B, bias, C, M, N, K);
        return;
    }

    /* A is FP32 token-major [M, K] */
    size_t A_row_bytes = (size_t)K * sizeof(float);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q5_1_q8_1, &args);
}

void gemm_nt_q5_k_parallel_dispatch(
    const float *A, const void *B, const float *bias, float *C,
    int M, int N, int K)
{
    ck_threadpool_t *pool = ck_threadpool_global();
    if (!pool || ck_threadpool_n_threads(pool) <= 1 || M <= 1 || ck_should_run_gemm_serial(pool, M, N, K)) {
        gemm_nt_q5_k(A, B, bias, C, M, N, K);
        return;
    }

    /* A is FP32 token-major [M, K] */
    size_t A_row_bytes = (size_t)K * sizeof(float);

    gemm_args_t args = {
        .A = A, .B = B, .bias = bias, .C = C,
        .M = M, .N = N, .K = K,
        .A_row_bytes = A_row_bytes
    };
    ck_threadpool_dispatch_n(pool, ck_select_gemm_active_threads(pool, M, N, K), work_gemm_nt_q5_k, &args);
}
