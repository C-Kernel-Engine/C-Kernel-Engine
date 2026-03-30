/**
 * @file gemm_kernels_q8_0_q8_0_contract.c
 * @brief FP32 API adapters that enforce Q8_0 x Q8_0 activation contract
 */

#include "ckernel_engine.h"
#include "ckernel_quant.h"
#include "../../llama.cpp/ggml/include/ggml.h"

#include <dlfcn.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK_Q80_STACK_Q8_BLOCKS 1024

typedef struct {
    char magic[8];
    uint32_t version;
    int32_t layer_id;
    char op_name[32];
    uint32_t dtype;
    uint32_t rank;
    int64_t shape[4];
    uint32_t elem_count;
    int32_t token_id;
    uint8_t reserved[32];
} __attribute__((packed)) ck_q80_contract_dump_header_t;

static const char ck_q80_contract_magic[8] = {'C', 'K', 'D', 'M', 'P', '\0', '\0', '\0'};
static const uint32_t ck_q80_contract_version = 1u;
static int ck_q80_contract_cached_gemm_seq = 0;

static int ck_q80_contract_dump_enabled(void)
{
    const char *v = getenv("CK_STRICT_GEMM_DUMP");
    return v && v[0] && strcmp(v, "0") != 0;
}

static void ck_q80_contract_dump_tensor(const char *name,
                                        int layer_id,
                                        const float *data,
                                        size_t elem_count)
{
    const char *dir = getenv("CK_PARITY_DIR");
    if (!dir || !dir[0] || !data || elem_count == 0 || !name || !name[0]) {
        return;
    }

    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, "strict_internal.bin");
    FILE *f = fopen(path, "ab");
    if (!f) {
        return;
    }

    ck_q80_contract_dump_header_t h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, ck_q80_contract_magic, sizeof(ck_q80_contract_magic));
    h.version = ck_q80_contract_version;
    h.layer_id = layer_id;
    strncpy(h.op_name, name, sizeof(h.op_name) - 1);
    h.dtype = 0u;
    h.rank = 1u;
    h.shape[0] = (int64_t) elem_count;
    h.elem_count = (uint32_t) elem_count;
    h.token_id = 0;

    fwrite(&h, sizeof(h), 1, f);
    fwrite(data, sizeof(float), elem_count, f);
    fclose(f);
}

typedef struct ggml_context *(*ck_q80_ggml_init_fn)(struct ggml_init_params);
typedef void (*ck_q80_ggml_free_fn)(struct ggml_context *);
typedef struct ggml_tensor *(*ck_q80_ggml_new_tensor_2d_fn)(struct ggml_context *, enum ggml_type, int64_t, int64_t);
typedef struct ggml_tensor *(*ck_q80_ggml_mul_mat_fn)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *);
typedef struct ggml_cgraph *(*ck_q80_ggml_new_graph_fn)(struct ggml_context *);
typedef void (*ck_q80_ggml_build_forward_expand_fn)(struct ggml_cgraph *, struct ggml_tensor *);
typedef enum ggml_status (*ck_q80_ggml_graph_compute_with_ctx_fn)(struct ggml_context *, struct ggml_cgraph *, int);
typedef void (*ck_q80_ggml_cpu_init_fn)(void);

static ck_q80_ggml_init_fn ck_q80_resolve_ggml_init(void)
{
    static int tried = 0;
    static ck_q80_ggml_init_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_q80_ggml_init_fn) dlsym(RTLD_DEFAULT, "ggml_init");
    }
    return fn;
}

static ck_q80_ggml_free_fn ck_q80_resolve_ggml_free(void)
{
    static int tried = 0;
    static ck_q80_ggml_free_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_q80_ggml_free_fn) dlsym(RTLD_DEFAULT, "ggml_free");
    }
    return fn;
}

static ck_q80_ggml_new_tensor_2d_fn ck_q80_resolve_ggml_new_tensor_2d(void)
{
    static int tried = 0;
    static ck_q80_ggml_new_tensor_2d_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_q80_ggml_new_tensor_2d_fn) dlsym(RTLD_DEFAULT, "ggml_new_tensor_2d");
    }
    return fn;
}

static ck_q80_ggml_mul_mat_fn ck_q80_resolve_ggml_mul_mat(void)
{
    static int tried = 0;
    static ck_q80_ggml_mul_mat_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_q80_ggml_mul_mat_fn) dlsym(RTLD_DEFAULT, "ggml_mul_mat");
    }
    return fn;
}

static ck_q80_ggml_new_graph_fn ck_q80_resolve_ggml_new_graph(void)
{
    static int tried = 0;
    static ck_q80_ggml_new_graph_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_q80_ggml_new_graph_fn) dlsym(RTLD_DEFAULT, "ggml_new_graph");
    }
    return fn;
}

static ck_q80_ggml_build_forward_expand_fn ck_q80_resolve_ggml_build_forward_expand(void)
{
    static int tried = 0;
    static ck_q80_ggml_build_forward_expand_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_q80_ggml_build_forward_expand_fn) dlsym(RTLD_DEFAULT, "ggml_build_forward_expand");
    }
    return fn;
}

static ck_q80_ggml_graph_compute_with_ctx_fn ck_q80_resolve_ggml_graph_compute_with_ctx(void)
{
    static int tried = 0;
    static ck_q80_ggml_graph_compute_with_ctx_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_q80_ggml_graph_compute_with_ctx_fn) dlsym(RTLD_DEFAULT, "ggml_graph_compute_with_ctx");
    }
    return fn;
}

static ck_q80_ggml_cpu_init_fn ck_q80_resolve_ggml_cpu_init(void)
{
    static int tried = 0;
    static ck_q80_ggml_cpu_init_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_q80_ggml_cpu_init_fn) dlsym(RTLD_DEFAULT, "ggml_cpu_init");
    }
    return fn;
}

static int gemm_nt_q8_0_q8_0_ggml_strict(const float *A,
                                         const void *B,
                                         const float *bias,
                                         float *C,
                                         int M,
                                         int N,
                                         int K)
{
    ck_q80_ggml_cpu_init_fn ggml_cpu_init_fn = ck_q80_resolve_ggml_cpu_init();
    ck_q80_ggml_init_fn ggml_init_fn = ck_q80_resolve_ggml_init();
    ck_q80_ggml_free_fn ggml_free_fn = ck_q80_resolve_ggml_free();
    ck_q80_ggml_new_tensor_2d_fn ggml_new_tensor_2d_fn = ck_q80_resolve_ggml_new_tensor_2d();
    ck_q80_ggml_mul_mat_fn ggml_mul_mat_fn = ck_q80_resolve_ggml_mul_mat();
    ck_q80_ggml_new_graph_fn ggml_new_graph_fn = ck_q80_resolve_ggml_new_graph();
    ck_q80_ggml_build_forward_expand_fn ggml_build_forward_expand_fn = ck_q80_resolve_ggml_build_forward_expand();
    ck_q80_ggml_graph_compute_with_ctx_fn ggml_graph_compute_with_ctx_fn = ck_q80_resolve_ggml_graph_compute_with_ctx();

    if (!ggml_cpu_init_fn || !ggml_init_fn || !ggml_free_fn || !ggml_new_tensor_2d_fn ||
        !ggml_mul_mat_fn || !ggml_new_graph_fn || !ggml_build_forward_expand_fn ||
        !ggml_graph_compute_with_ctx_fn) {
        return 0;
    }

    ggml_cpu_init_fn();

    const size_t output_bytes = (size_t) M * (size_t) N * sizeof(float);
    const size_t mem_size = ((size_t) 128 * 1024 * 1024) + output_bytes;

    struct ggml_init_params params = {
        .mem_size = mem_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init_fn(params);
    if (!ctx) {
        return 0;
    }

    int ok = 0;
    struct ggml_tensor *w = ggml_new_tensor_2d_fn(ctx, GGML_TYPE_Q8_0, K, N);
    struct ggml_tensor *x = ggml_new_tensor_2d_fn(ctx, GGML_TYPE_F32, K, M);
    if (!w || !x) {
        ggml_free_fn(ctx);
        return 0;
    }

    w->data = (void *) B;
    x->data = (void *) A;

    struct ggml_tensor *y = ggml_mul_mat_fn(ctx, w, x);
    if (!y) {
        ggml_free_fn(ctx);
        return 0;
    }

    struct ggml_cgraph *gf = ggml_new_graph_fn(ctx);
    if (!gf) {
        ggml_free_fn(ctx);
        return 0;
    }
    ggml_build_forward_expand_fn(gf, y);
    if (ggml_graph_compute_with_ctx_fn(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free_fn(ctx);
        return 0;
    }

    {
        const float *src = (const float *) y->data;
        for (int m = 0; m < M; ++m) {
            memcpy(C + (size_t) m * (size_t) N,
                   src + (size_t) m * (size_t) N,
                   (size_t) N * sizeof(float));
            if (bias) {
                for (int n = 0; n < N; ++n) {
                    C[(size_t) m * (size_t) N + (size_t) n] += bias[n];
                }
            }
        }
    }

    ok = 1;
    ggml_free_fn(ctx);
    return ok;
}

void vec_dot_q8_0_q8_0_ref(int n, float *s, const void *vx, const void *vy);

static inline int ck_nearest_int_q8_0_ref(float fval)
{
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static void quantize_row_q8_0_ref_local(const float *x,
                                        block_q8_0 *y,
                                        int k)
{
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            const float v = x[i * QK8_0 + j];
            const float av = fabsf(v);
            if (av > amax) {
                amax = av;
            }
        }

        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = CK_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i * QK8_0 + j] * id;
            int q = ck_nearest_int_q8_0_ref(x0);
            if (q > 127) {
                q = 127;
            }
            if (q < -127) {
                q = -127;
            }
            y[i].qs[j] = (int8_t) q;
        }
    }
}

static void gemv_q8_0_q8_0_ref_rows(float *y,
                                    const void *W,
                                    const void *x_q8,
                                    int M,
                                    int K)
{
    const block_q8_0 *w_blocks = (const block_q8_0 *)W;
    const int blocks_per_row = K / QK8_0;

    for (int row = 0; row < M; ++row) {
        vec_dot_q8_0_q8_0_ref(
            K,
            &y[row],
            &w_blocks[row * blocks_per_row],
            x_q8
        );
    }
}

void gemv_q8_0_q8_0_contract(float *y,
                             const void *W,
                             const float *x,
                             int M,
                             int K)
{
    if (!y || !W || !x || M <= 0 || K <= 0) {
        return;
    }

    if ((K % QK8_0) != 0) {
        gemv_q8_0(y, W, x, M, K);
        return;
    }

    const int blocks_per_row = K / QK8_0;
    if (blocks_per_row > CK_Q80_STACK_Q8_BLOCKS) {
        gemv_q8_0(y, W, x, M, K);
        return;
    }

    block_q8_0 x_q8[CK_Q80_STACK_Q8_BLOCKS];
    if (ck_strict_parity_enabled()) {
        quantize_row_q8_0_ref_local(x, x_q8, K);
        gemv_q8_0_q8_0_ref_rows(y, W, x_q8, M, K);
        return;
    }
    quantize_row_q8_0(x, x_q8, K);
    gemv_q8_0_q8_0(y, W, x_q8, M, K);
}

void gemm_nt_q8_0_q8_0_contract(const float *A,
                                const void *B,
                                const float *bias,
                                float *C,
                                int M,
                                int N,
                                int K)
{
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    const float *A_use = A;
    const int strict = ck_strict_parity_enabled();
    int strict_cached_layer = -1;
    if (ck_strict_parity_enabled()) {
        const float *cached = ck_strict_consume_next_gemm_a((size_t) M * (size_t) K);
        if (cached) {
            A_use = cached;
            strict_cached_layer = ck_q80_contract_cached_gemm_seq++;
        }
    }

    if (strict && strict_cached_layer >= 0 && ck_q80_contract_dump_enabled()) {
        ck_q80_contract_dump_tensor("strict_out_proj_input",
                                    strict_cached_layer,
                                    A_use,
                                    (size_t) M * (size_t) K);
    }

    if (strict &&
        gemm_nt_q8_0_q8_0_ggml_strict(A_use, B, bias, C, M, N, K)) {
        if (strict && strict_cached_layer >= 0 && ck_q80_contract_dump_enabled()) {
            ck_q80_contract_dump_tensor("strict_out_proj_output",
                                        strict_cached_layer,
                                        C,
                                        (size_t) M * (size_t) N);
        }
        return;
    }

    for (int m = 0; m < M; ++m) {
        gemv_q8_0_q8_0_contract(&C[m * N], B, &A_use[m * K], N, K);
        if (bias) {
            for (int n = 0; n < N; ++n) {
                C[m * N + n] += bias[n];
            }
        }
    }

    if (strict && strict_cached_layer >= 0 && ck_q80_contract_dump_enabled()) {
        ck_q80_contract_dump_tensor("strict_out_proj_output",
                                    strict_cached_layer,
                                    C,
                                    (size_t) M * (size_t) N);
    }
}
