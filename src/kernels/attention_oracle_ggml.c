/**
 * @file attention_oracle_ggml.c
 * @brief Strict ggml-backed full-attention oracles for parity debugging.
 *
 * These helpers intentionally mirror ggml graph composition for encoder-style
 * full attention. They are composite parity scaffolding, not production CK
 * kernels, and are only used from strict parity paths.
 */

#define CK_ENABLE_LLAMA_CPP_PARITY 1
#include "attention_oracle_ggml.h"
#include "ckernel_engine.h"
#include "../../llama.cpp/ggml/include/ggml.h"
#include "../../llama.cpp/ggml/include/ggml-backend.h"
#include "../../llama.cpp/ggml/include/ggml-alloc.h"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ggml_context *(*ck_ggml_init_fn)(struct ggml_init_params);
typedef void (*ck_ggml_free_fn)(struct ggml_context *);
typedef struct ggml_tensor *(*ck_ggml_new_tensor_1d_fn)(struct ggml_context *, enum ggml_type, int64_t);
typedef struct ggml_tensor *(*ck_ggml_new_tensor_2d_fn)(struct ggml_context *, enum ggml_type, int64_t, int64_t);
typedef struct ggml_tensor *(*ck_ggml_view_3d_fn)(struct ggml_context *, struct ggml_tensor *, int64_t, int64_t, int64_t, size_t, size_t, size_t);
typedef struct ggml_tensor *(*ck_ggml_permute_fn)(struct ggml_context *, struct ggml_tensor *, int, int, int, int);
typedef struct ggml_tensor *(*ck_ggml_cont_fn)(struct ggml_context *, struct ggml_tensor *);
typedef struct ggml_tensor *(*ck_ggml_cont_2d_fn)(struct ggml_context *, struct ggml_tensor *, int64_t, int64_t);
typedef struct ggml_tensor *(*ck_ggml_mul_mat_graph_fn)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *);
typedef struct ggml_tensor *(*ck_ggml_soft_max_ext_fn)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, float, float);
typedef struct ggml_cgraph *(*ck_ggml_new_graph_fn)(struct ggml_context *);
typedef void (*ck_ggml_build_forward_expand_fn)(struct ggml_cgraph *, struct ggml_tensor *);
typedef enum ggml_status (*ck_ggml_graph_compute_with_ctx_fn)(struct ggml_context *, struct ggml_cgraph *, int);
typedef void (*ck_ggml_cpu_init_fn)(void);
typedef void (*ck_ggml_set_input_fn)(struct ggml_tensor *);
typedef ggml_backend_t (*ck_ggml_backend_init_by_type_fn)(enum ggml_backend_dev_type, const char *);
typedef void (*ck_ggml_backend_free_fn)(ggml_backend_t);
typedef void (*ck_ggml_backend_cpu_set_n_threads_fn)(ggml_backend_t, int);
typedef ggml_backend_buffer_type_t (*ck_ggml_backend_get_default_buffer_type_fn)(ggml_backend_t);
typedef void (*ck_ggml_backend_tensor_set_fn)(struct ggml_tensor *, const void *, size_t, size_t);
typedef void (*ck_ggml_backend_tensor_get_fn)(const struct ggml_tensor *, void *, size_t, size_t);
typedef ggml_backend_sched_t (*ck_ggml_backend_sched_new_fn)(ggml_backend_t *, ggml_backend_buffer_type_t *, int, size_t, bool, bool);
typedef void (*ck_ggml_backend_sched_free_fn)(ggml_backend_sched_t);
typedef void (*ck_ggml_backend_sched_reset_fn)(ggml_backend_sched_t);
typedef bool (*ck_ggml_backend_sched_alloc_graph_fn)(ggml_backend_sched_t, struct ggml_cgraph *);
typedef enum ggml_status (*ck_ggml_backend_sched_graph_compute_fn)(ggml_backend_sched_t, struct ggml_cgraph *);

static ck_ggml_backend_tensor_get_fn ck_resolve_ggml_backend_tensor_get(void);

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
} __attribute__((packed)) ck_attention_oracle_dump_header_t;

static const char ck_attention_oracle_magic[8] = {'C', 'K', 'D', 'M', 'P', '\0', '\0', '\0'};
static const uint32_t ck_attention_oracle_version = 1u;
static int ck_attention_oracle_layer_seq = 0;

static int ck_attention_oracle_dump_enabled(void)
{
    const char *v = getenv("CK_STRICT_ATTN_DUMP");
    return v && v[0] && strcmp(v, "0") != 0;
}

static int ck_attention_oracle_meta_dump_enabled(void)
{
    const char *v = getenv("CK_STRICT_ATTN_DUMP_META");
    return v && v[0] && strcmp(v, "0") != 0;
}

static int ck_attention_oracle_exact_dump_layer(int *layer_id_out)
{
    const char *v = getenv("CK_STRICT_ATTN_DUMP_LAYER");
    if (!v || !v[0]) {
        return 0;
    }
    char *end = NULL;
    long parsed = strtol(v, &end, 10);
    if (end == v || (end && *end != '\0') || parsed < 0 || parsed > INT32_MAX) {
        return 0;
    }
    if (layer_id_out) {
        *layer_id_out = (int) parsed;
    }
    return 1;
}

static int ck_attention_oracle_should_dump_layer(int layer_id)
{
    int exact_layer = -1;
    if (!ck_attention_oracle_exact_dump_layer(&exact_layer)) {
        return 1;
    }
    return layer_id == exact_layer;
}

static inline size_t ck_attention_oracle_qkv_index(int h,
                                                   int t,
                                                   int d,
                                                   int num_tokens,
                                                   int aligned_head_dim)
{
    return ((size_t) h * (size_t) num_tokens + (size_t) t) * (size_t) aligned_head_dim +
           (size_t) d;
}

static int ck_attention_oracle_dump_layer_id(void)
{
    const int layer_id = ck_attention_oracle_layer_seq;
    ck_attention_oracle_layer_seq += 1;
    return layer_id;
}

static float ck_attention_oracle_tensor_f32_at(const struct ggml_tensor *t,
                                               size_t i0,
                                               size_t i1,
                                               size_t i2,
                                               size_t i3)
{
    const uint8_t *base = (const uint8_t *) t->data;
    const size_t off =
        i0 * (size_t) t->nb[0] +
        i1 * (size_t) t->nb[1] +
        i2 * (size_t) t->nb[2] +
        i3 * (size_t) t->nb[3];
    return *(const float *) (base + off);
}

static void ck_attention_oracle_dump_tensor(const char *name,
                                            int layer_id,
                                            const struct ggml_tensor *t)
{
    const char *dir = getenv("CK_PARITY_DIR");
    if (!ck_attention_oracle_dump_enabled() ||
        !dir || !dir[0] || !name || !name[0] || !t || !t->data ||
        !ck_attention_oracle_should_dump_layer(layer_id)) {
        return;
    }
    if (t->type != GGML_TYPE_F32) {
        return;
    }

    const int64_t ne0 = t->ne[0] > 0 ? t->ne[0] : 1;
    const int64_t ne1 = t->ne[1] > 0 ? t->ne[1] : 1;
    const int64_t ne2 = t->ne[2] > 0 ? t->ne[2] : 1;
    const int64_t ne3 = t->ne[3] > 0 ? t->ne[3] : 1;
    const size_t elem_count =
        (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;
    if (elem_count == 0) {
        return;
    }
    const size_t nbytes = elem_count * sizeof(float);
    float *host = NULL;
    const uint8_t *base = NULL;

    if (t->buffer != NULL) {
        ck_ggml_backend_tensor_get_fn ggml_backend_tensor_get_fn = ck_resolve_ggml_backend_tensor_get();
        if (!ggml_backend_tensor_get_fn) {
            return;
        }
        host = (float *) malloc(nbytes);
        if (!host) {
            return;
        }
        ggml_backend_tensor_get_fn(t, host, 0, nbytes);
        base = (const uint8_t *) host;
    } else if (t->data != NULL) {
        base = (const uint8_t *) t->data;
    }
    if (!base) {
        free(host);
        return;
    }

    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, "strict_internal.bin");
    FILE *f = fopen(path, "ab");
    if (!f) {
        free(host);
        return;
    }

    ck_attention_oracle_dump_header_t h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, ck_attention_oracle_magic, sizeof(ck_attention_oracle_magic));
    h.version = ck_attention_oracle_version;
    h.layer_id = layer_id;
    strncpy(h.op_name, name, sizeof(h.op_name) - 1);
    h.dtype = 0u;
    h.rank = 1u;
    h.shape[0] = (int64_t) elem_count;
    h.elem_count = (uint32_t) elem_count;
    h.token_id = 0;

    fwrite(&h, sizeof(h), 1, f);
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const size_t off =
                        (size_t) i0 * (size_t) t->nb[0] +
                        (size_t) i1 * (size_t) t->nb[1] +
                        (size_t) i2 * (size_t) t->nb[2] +
                        (size_t) i3 * (size_t) t->nb[3];
                    const float v = *(const float *) (base + off);
                    fwrite(&v, sizeof(v), 1, f);
                }
            }
        }
    }
    fclose(f);
    free(host);
}

static void ck_attention_oracle_dump_meta(const char *name,
                                          int layer_id,
                                          const struct ggml_tensor *t)
{
    const char *dir = getenv("CK_PARITY_DIR");
    if (!ck_attention_oracle_meta_dump_enabled() ||
        !dir || !dir[0] || !name || !name[0] || !t ||
        !ck_attention_oracle_should_dump_layer(layer_id)) {
        return;
    }

    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, "meta.jsonl");
    FILE *f = fopen(path, "a");
    if (!f) {
        return;
    }

    fprintf(
        f,
        "{\"name\":\"%s\",\"layer_id\":%d,\"type\":%d,"
        "\"ne\":[%lld,%lld,%lld,%lld],"
        "\"nb\":[%zu,%zu,%zu,%zu]}\n",
        name,
        layer_id,
        (int) t->type,
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
        (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3]);
    fclose(f);
}

static ck_ggml_init_fn ck_resolve_ggml_init(void)
{
    static int tried = 0;
    static ck_ggml_init_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_init_fn) dlsym(RTLD_DEFAULT, "ggml_init");
    }
    return fn;
}

static ck_ggml_free_fn ck_resolve_ggml_free(void)
{
    static int tried = 0;
    static ck_ggml_free_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_free_fn) dlsym(RTLD_DEFAULT, "ggml_free");
    }
    return fn;
}

static ck_ggml_new_tensor_1d_fn ck_resolve_ggml_new_tensor_1d(void)
{
    static int tried = 0;
    static ck_ggml_new_tensor_1d_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_new_tensor_1d_fn) dlsym(RTLD_DEFAULT, "ggml_new_tensor_1d");
    }
    return fn;
}

static ck_ggml_new_tensor_2d_fn ck_resolve_ggml_new_tensor_2d(void)
{
    static int tried = 0;
    static ck_ggml_new_tensor_2d_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_new_tensor_2d_fn) dlsym(RTLD_DEFAULT, "ggml_new_tensor_2d");
    }
    return fn;
}

static ck_ggml_view_3d_fn ck_resolve_ggml_view_3d(void)
{
    static int tried = 0;
    static ck_ggml_view_3d_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_view_3d_fn) dlsym(RTLD_DEFAULT, "ggml_view_3d");
    }
    return fn;
}

static ck_ggml_permute_fn ck_resolve_ggml_permute(void)
{
    static int tried = 0;
    static ck_ggml_permute_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_permute_fn) dlsym(RTLD_DEFAULT, "ggml_permute");
    }
    return fn;
}

static ck_ggml_cont_fn ck_resolve_ggml_cont(void)
{
    static int tried = 0;
    static ck_ggml_cont_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_cont_fn) dlsym(RTLD_DEFAULT, "ggml_cont");
    }
    return fn;
}

static ck_ggml_cont_2d_fn ck_resolve_ggml_cont_2d(void)
{
    static int tried = 0;
    static ck_ggml_cont_2d_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_cont_2d_fn) dlsym(RTLD_DEFAULT, "ggml_cont_2d");
    }
    return fn;
}

static ck_ggml_mul_mat_graph_fn ck_resolve_ggml_mul_mat_graph(void)
{
    static int tried = 0;
    static ck_ggml_mul_mat_graph_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_mul_mat_graph_fn) dlsym(RTLD_DEFAULT, "ggml_mul_mat");
    }
    return fn;
}

static ck_ggml_soft_max_ext_fn ck_resolve_ggml_soft_max_ext(void)
{
    static int tried = 0;
    static ck_ggml_soft_max_ext_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_soft_max_ext_fn) dlsym(RTLD_DEFAULT, "ggml_soft_max_ext");
    }
    return fn;
}

static ck_ggml_new_graph_fn ck_resolve_ggml_new_graph(void)
{
    static int tried = 0;
    static ck_ggml_new_graph_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_new_graph_fn) dlsym(RTLD_DEFAULT, "ggml_new_graph");
    }
    return fn;
}

static ck_ggml_build_forward_expand_fn ck_resolve_ggml_build_forward_expand(void)
{
    static int tried = 0;
    static ck_ggml_build_forward_expand_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_build_forward_expand_fn) dlsym(RTLD_DEFAULT, "ggml_build_forward_expand");
    }
    return fn;
}

static ck_ggml_graph_compute_with_ctx_fn ck_resolve_ggml_graph_compute_with_ctx(void)
{
    static int tried = 0;
    static ck_ggml_graph_compute_with_ctx_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_graph_compute_with_ctx_fn) dlsym(RTLD_DEFAULT, "ggml_graph_compute_with_ctx");
    }
    return fn;
}

static ck_ggml_cpu_init_fn ck_resolve_ggml_cpu_init(void)
{
    static int tried = 0;
    static ck_ggml_cpu_init_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_cpu_init_fn) dlsym(RTLD_DEFAULT, "ggml_cpu_init");
    }
    return fn;
}

static ck_ggml_set_input_fn ck_resolve_ggml_set_input(void)
{
    static int tried = 0;
    static ck_ggml_set_input_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_set_input_fn) dlsym(RTLD_DEFAULT, "ggml_set_input");
    }
    return fn;
}

static ck_ggml_backend_init_by_type_fn ck_resolve_ggml_backend_init_by_type(void)
{
    static int tried = 0;
    static ck_ggml_backend_init_by_type_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_init_by_type_fn) dlsym(RTLD_DEFAULT, "ggml_backend_init_by_type");
    }
    return fn;
}

static ck_ggml_backend_free_fn ck_resolve_ggml_backend_free(void)
{
    static int tried = 0;
    static ck_ggml_backend_free_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_free_fn) dlsym(RTLD_DEFAULT, "ggml_backend_free");
    }
    return fn;
}

static ck_ggml_backend_get_default_buffer_type_fn ck_resolve_ggml_backend_get_default_buffer_type(void)
{
    static int tried = 0;
    static ck_ggml_backend_get_default_buffer_type_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_get_default_buffer_type_fn) dlsym(RTLD_DEFAULT, "ggml_backend_get_default_buffer_type");
    }
    return fn;
}

static ck_ggml_backend_cpu_set_n_threads_fn ck_resolve_ggml_backend_cpu_set_n_threads(void)
{
    static int tried = 0;
    static ck_ggml_backend_cpu_set_n_threads_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_cpu_set_n_threads_fn) dlsym(RTLD_DEFAULT, "ggml_backend_cpu_set_n_threads");
    }
    return fn;
}

static ck_ggml_backend_sched_new_fn ck_resolve_ggml_backend_sched_new(void)
{
    static int tried = 0;
    static ck_ggml_backend_sched_new_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_sched_new_fn) dlsym(RTLD_DEFAULT, "ggml_backend_sched_new");
    }
    return fn;
}

static ck_ggml_backend_sched_free_fn ck_resolve_ggml_backend_sched_free(void)
{
    static int tried = 0;
    static ck_ggml_backend_sched_free_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_sched_free_fn) dlsym(RTLD_DEFAULT, "ggml_backend_sched_free");
    }
    return fn;
}

static ck_ggml_backend_sched_reset_fn ck_resolve_ggml_backend_sched_reset(void)
{
    static int tried = 0;
    static ck_ggml_backend_sched_reset_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_sched_reset_fn) dlsym(RTLD_DEFAULT, "ggml_backend_sched_reset");
    }
    return fn;
}

static ck_ggml_backend_sched_alloc_graph_fn ck_resolve_ggml_backend_sched_alloc_graph(void)
{
    static int tried = 0;
    static ck_ggml_backend_sched_alloc_graph_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_sched_alloc_graph_fn) dlsym(RTLD_DEFAULT, "ggml_backend_sched_alloc_graph");
    }
    return fn;
}

static ck_ggml_backend_tensor_set_fn ck_resolve_ggml_backend_tensor_set(void)
{
    static int tried = 0;
    static ck_ggml_backend_tensor_set_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_tensor_set_fn) dlsym(RTLD_DEFAULT, "ggml_backend_tensor_set");
    }
    return fn;
}

static ck_ggml_backend_tensor_get_fn ck_resolve_ggml_backend_tensor_get(void)
{
    static int tried = 0;
    static ck_ggml_backend_tensor_get_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_tensor_get_fn) dlsym(RTLD_DEFAULT, "ggml_backend_tensor_get");
    }
    return fn;
}

static ck_ggml_backend_sched_graph_compute_fn ck_resolve_ggml_backend_sched_graph_compute(void)
{
    static int tried = 0;
    static ck_ggml_backend_sched_graph_compute_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_backend_sched_graph_compute_fn) dlsym(RTLD_DEFAULT, "ggml_backend_sched_graph_compute");
    }
    return fn;
}

int ck_attention_head_full_ggml_graph_oracle_regular(const float *q_head,
                                                     const float *k_head,
                                                     const float *v_head,
                                                     float *out_head,
                                                     int num_tokens,
                                                     int head_dim,
                                                     int aligned_head_dim,
                                                     float scale)
{
    const char *disable_env = getenv("CK_STRICT_DISABLE_REGULAR_ATTN_ORACLE");
    if (disable_env && disable_env[0] && strcmp(disable_env, "0") != 0) {
        return 0;
    }
    ck_ggml_cpu_init_fn ggml_cpu_init_fn = ck_resolve_ggml_cpu_init();
    ck_ggml_init_fn ggml_init_fn = ck_resolve_ggml_init();
    ck_ggml_free_fn ggml_free_fn = ck_resolve_ggml_free();
    ck_ggml_new_tensor_1d_fn ggml_new_tensor_1d_fn = ck_resolve_ggml_new_tensor_1d();
    ck_ggml_new_tensor_2d_fn ggml_new_tensor_2d_fn = ck_resolve_ggml_new_tensor_2d();
    ck_ggml_view_3d_fn ggml_view_3d_fn = ck_resolve_ggml_view_3d();
    ck_ggml_permute_fn ggml_permute_fn = ck_resolve_ggml_permute();
    ck_ggml_cont_fn ggml_cont_fn = ck_resolve_ggml_cont();
    ck_ggml_cont_2d_fn ggml_cont_2d_fn = ck_resolve_ggml_cont_2d();
    ck_ggml_mul_mat_graph_fn ggml_mul_mat_fn = ck_resolve_ggml_mul_mat_graph();
    ck_ggml_soft_max_ext_fn ggml_soft_max_ext_fn = ck_resolve_ggml_soft_max_ext();
    ck_ggml_new_graph_fn ggml_new_graph_fn = ck_resolve_ggml_new_graph();
    ck_ggml_build_forward_expand_fn ggml_build_forward_expand_fn = ck_resolve_ggml_build_forward_expand();
    ck_ggml_graph_compute_with_ctx_fn ggml_graph_compute_with_ctx_fn = ck_resolve_ggml_graph_compute_with_ctx();
    ck_ggml_set_input_fn ggml_set_input_fn = ck_resolve_ggml_set_input();

    if (!ggml_cpu_init_fn || !ggml_init_fn || !ggml_free_fn ||
        !ggml_new_tensor_1d_fn || !ggml_new_tensor_2d_fn ||
        !ggml_view_3d_fn || !ggml_permute_fn || !ggml_cont_fn || !ggml_cont_2d_fn ||
        !ggml_mul_mat_fn || !ggml_soft_max_ext_fn || !ggml_new_graph_fn ||
        !ggml_build_forward_expand_fn || !ggml_graph_compute_with_ctx_fn ||
        !ggml_set_input_fn) {
        return 0;
    }

    ggml_cpu_init_fn();

    const size_t row_bytes = (size_t) aligned_head_dim * sizeof(float);
    const size_t tensor_bytes = (size_t) num_tokens * row_bytes;
    const size_t kq_bytes = (size_t) num_tokens * (size_t) num_tokens * sizeof(float);
    const size_t mem_size = (size_t) 128 * 1024 * 1024 + tensor_bytes * 3 + kq_bytes * 2;

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
    struct ggml_tensor *q_base =
        ggml_new_tensor_1d_fn(ctx, GGML_TYPE_F32, (int64_t) num_tokens * (int64_t) aligned_head_dim);
    struct ggml_tensor *k_base =
        ggml_new_tensor_1d_fn(ctx, GGML_TYPE_F32, (int64_t) num_tokens * (int64_t) aligned_head_dim);
    struct ggml_tensor *v_base =
        ggml_new_tensor_1d_fn(ctx, GGML_TYPE_F32, (int64_t) num_tokens * (int64_t) aligned_head_dim);
    if (!q_base || !k_base || !v_base) {
        ggml_free_fn(ctx);
        return 0;
    }

    q_base->data = (void *) q_head;
    k_base->data = (void *) k_head;
    v_base->data = (void *) v_head;
    ggml_set_input_fn(q_base);
    ggml_set_input_fn(k_base);
    ggml_set_input_fn(v_base);

    struct ggml_tensor *q_cur = ggml_view_3d_fn(ctx, q_base, head_dim, 1, num_tokens, row_bytes, row_bytes, 0);
    struct ggml_tensor *k_cur = ggml_view_3d_fn(ctx, k_base, head_dim, 1, num_tokens, row_bytes, row_bytes, 0);
    struct ggml_tensor *v_cur = ggml_view_3d_fn(ctx, v_base, head_dim, 1, num_tokens, row_bytes, row_bytes, 0);
    if (!q_cur || !k_cur || !v_cur) {
        ggml_free_fn(ctx);
        return 0;
    }

    struct ggml_tensor *q = ggml_permute_fn(ctx, q_cur, 0, 2, 1, 3);
    struct ggml_tensor *k = ggml_permute_fn(ctx, k_cur, 0, 2, 1, 3);
    struct ggml_tensor *v = ggml_permute_fn(ctx, v_cur, 1, 2, 0, 3);
    if (!q || !k || !v) {
        ggml_free_fn(ctx);
        return 0;
    }

    v = ggml_cont_fn(ctx, v);
    struct ggml_tensor *kq_scores = ggml_mul_mat_fn(ctx, k, q);
    struct ggml_tensor *kq_scores_dump = kq_scores ? ggml_cont_fn(ctx, kq_scores) : NULL;
    struct ggml_tensor *kq_softmax_src = kq_scores_dump ? kq_scores_dump : kq_scores;
    struct ggml_tensor *kq = kq_softmax_src ?
        ggml_soft_max_ext_fn(ctx, kq_softmax_src, NULL, scale, 0.0f) : NULL;
    struct ggml_tensor *kqv = kq ? ggml_mul_mat_fn(ctx, v, kq) : NULL;
    struct ggml_tensor *cur = kqv ? ggml_permute_fn(ctx, kqv, 0, 2, 1, 3) : NULL;
    if (cur) {
        cur = ggml_cont_2d_fn(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }
    if (!cur) {
        ggml_free_fn(ctx);
        return 0;
    }

    struct ggml_cgraph *gf = ggml_new_graph_fn(ctx);
    if (!gf) {
        ggml_free_fn(ctx);
        return 0;
    }
    if (kq_scores_dump) {
        ggml_build_forward_expand_fn(gf, kq_scores_dump);
    }
    ggml_build_forward_expand_fn(gf, cur);
    if (ggml_graph_compute_with_ctx_fn(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free_fn(ctx);
        return 0;
    }

    {
        const float *src = (const float *) cur->data;
        for (int t = 0; t < num_tokens; ++t) {
            memcpy(out_head + (size_t) t * (size_t) aligned_head_dim,
                   src + (size_t) t * (size_t) head_dim,
                   (size_t) head_dim * sizeof(float));
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                out_head[(size_t) t * (size_t) aligned_head_dim + (size_t) d] = 0.0f;
            }
        }
    }

    ok = 1;
    ggml_free_fn(ctx);
    return ok;
}

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
                                                  float scale)
{
    const char *disable_env = getenv("CK_STRICT_DISABLE_MULTIHEAD_ATTN_ORACLE");
    if (disable_env && disable_env[0] && strcmp(disable_env, "0") != 0) {
        return 0;
    }
    if (!q || !k || !v || !output) {
        return 0;
    }
    if (num_heads <= 0 || num_tokens <= 0 || head_dim <= 0) {
        return 0;
    }
    if (num_heads != num_kv_heads) {
        return 0;
    }
    if (aligned_head_dim < head_dim) {
        return 0;
    }
    if (kv_stride_tokens != num_tokens) {
        return 0;
    }
    const int layer_id = ck_attention_oracle_dump_layer_id();

    ck_ggml_cpu_init_fn ggml_cpu_init_fn = ck_resolve_ggml_cpu_init();
    ck_ggml_init_fn ggml_init_fn = ck_resolve_ggml_init();
    ck_ggml_free_fn ggml_free_fn = ck_resolve_ggml_free();
    ck_ggml_new_tensor_1d_fn ggml_new_tensor_1d_fn = ck_resolve_ggml_new_tensor_1d();
    ck_ggml_new_tensor_2d_fn ggml_new_tensor_2d_fn = ck_resolve_ggml_new_tensor_2d();
    ck_ggml_view_3d_fn ggml_view_3d_fn = ck_resolve_ggml_view_3d();
    ck_ggml_permute_fn ggml_permute_fn = ck_resolve_ggml_permute();
    ck_ggml_cont_fn ggml_cont_fn = ck_resolve_ggml_cont();
    ck_ggml_cont_2d_fn ggml_cont_2d_fn = ck_resolve_ggml_cont_2d();
    ck_ggml_mul_mat_graph_fn ggml_mul_mat_fn = ck_resolve_ggml_mul_mat_graph();
    ck_ggml_soft_max_ext_fn ggml_soft_max_ext_fn = ck_resolve_ggml_soft_max_ext();
    ck_ggml_new_graph_fn ggml_new_graph_fn = ck_resolve_ggml_new_graph();
    ck_ggml_build_forward_expand_fn ggml_build_forward_expand_fn = ck_resolve_ggml_build_forward_expand();
    ck_ggml_set_input_fn ggml_set_input_fn = ck_resolve_ggml_set_input();
    ck_ggml_backend_init_by_type_fn ggml_backend_init_by_type_fn = ck_resolve_ggml_backend_init_by_type();
    ck_ggml_backend_free_fn ggml_backend_free_fn = ck_resolve_ggml_backend_free();
    ck_ggml_backend_cpu_set_n_threads_fn ggml_backend_cpu_set_n_threads_fn = ck_resolve_ggml_backend_cpu_set_n_threads();
    ck_ggml_backend_get_default_buffer_type_fn ggml_backend_get_default_buffer_type_fn = ck_resolve_ggml_backend_get_default_buffer_type();
    ck_ggml_backend_sched_new_fn ggml_backend_sched_new_fn = ck_resolve_ggml_backend_sched_new();
    ck_ggml_backend_sched_free_fn ggml_backend_sched_free_fn = ck_resolve_ggml_backend_sched_free();
    ck_ggml_backend_sched_reset_fn ggml_backend_sched_reset_fn = ck_resolve_ggml_backend_sched_reset();
    ck_ggml_backend_sched_alloc_graph_fn ggml_backend_sched_alloc_graph_fn = ck_resolve_ggml_backend_sched_alloc_graph();
    ck_ggml_backend_tensor_set_fn ggml_backend_tensor_set_fn = ck_resolve_ggml_backend_tensor_set();
    ck_ggml_backend_tensor_get_fn ggml_backend_tensor_get_fn = ck_resolve_ggml_backend_tensor_get();
    ck_ggml_backend_sched_graph_compute_fn ggml_backend_sched_graph_compute_fn = ck_resolve_ggml_backend_sched_graph_compute();

    if (!ggml_cpu_init_fn || !ggml_init_fn || !ggml_free_fn ||
        !ggml_new_tensor_1d_fn || !ggml_new_tensor_2d_fn ||
        !ggml_view_3d_fn || !ggml_permute_fn || !ggml_cont_fn || !ggml_cont_2d_fn ||
        !ggml_mul_mat_fn || !ggml_soft_max_ext_fn || !ggml_new_graph_fn ||
        !ggml_build_forward_expand_fn || !ggml_backend_init_by_type_fn ||
        !ggml_set_input_fn ||
        !ggml_backend_free_fn || !ggml_backend_cpu_set_n_threads_fn ||
        !ggml_backend_get_default_buffer_type_fn || !ggml_backend_sched_new_fn ||
        !ggml_backend_sched_free_fn || !ggml_backend_sched_reset_fn ||
        !ggml_backend_sched_alloc_graph_fn || !ggml_backend_tensor_set_fn ||
        !ggml_backend_tensor_get_fn || !ggml_backend_sched_graph_compute_fn) {
        return 0;
    }

    ggml_cpu_init_fn();

    const size_t row_bytes = (size_t) aligned_head_dim * sizeof(float);
    const size_t head_bytes = (size_t) num_tokens * row_bytes;
    const size_t tensor_bytes = (size_t) num_heads * head_bytes;
    const size_t packed_row_bytes = (size_t) num_heads * (size_t) head_dim * sizeof(float);
    const size_t fused_qkv_row_bytes = (size_t) 3 * packed_row_bytes;
    const size_t score_bytes =
        (size_t) num_heads * (size_t) num_tokens * (size_t) num_tokens * sizeof(float);
    const size_t mem_size =
        (size_t) 1024 * 1024 * 1024 + tensor_bytes * 4 + score_bytes * 2;

    struct ggml_init_params params = {
        .mem_size = mem_size,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context *ctx = ggml_init_fn(params);
    if (!ctx) {
        return 0;
    }

    int ok = 0;
    struct ggml_tensor *qkv_base = ggml_new_tensor_2d_fn(
        ctx, GGML_TYPE_F32, 3 * (int64_t) num_heads * (int64_t) head_dim, num_tokens);
    if (!qkv_base) {
        ggml_free_fn(ctx);
        return 0;
    }
    ggml_set_input_fn(qkv_base);

    const size_t embd_elems = (size_t) num_heads * (size_t) head_dim;
    const size_t qkv_pack_elems = (size_t) num_tokens * 3 * embd_elems;
    float *qkv_pack = (float *) malloc(qkv_pack_elems * sizeof(float));
    if (!qkv_pack) {
        ggml_free_fn(ctx);
        return 0;
    }
    for (int t = 0; t < num_tokens; ++t) {
        float *q_tok = qkv_pack + (size_t) t * 3 * embd_elems;
        float *k_tok = q_tok + embd_elems;
        float *v_tok = k_tok + embd_elems;
        for (int h = 0; h < num_heads; ++h) {
            memcpy(q_tok + (size_t) h * (size_t) head_dim,
                   q + ck_attention_oracle_qkv_index(h, t, 0, num_tokens, aligned_head_dim),
                   (size_t) head_dim * sizeof(float));
            memcpy(k_tok + (size_t) h * (size_t) head_dim,
                   k + ck_attention_oracle_qkv_index(h, t, 0, num_tokens, aligned_head_dim),
                   (size_t) head_dim * sizeof(float));
            memcpy(v_tok + (size_t) h * (size_t) head_dim,
                   v + ck_attention_oracle_qkv_index(h, t, 0, num_tokens, aligned_head_dim),
                   (size_t) head_dim * sizeof(float));
        }
    }
    qkv_base->data = qkv_pack;

    struct ggml_tensor *q_cur = ggml_view_3d_fn(ctx,
                                                qkv_base,
                                                head_dim,
                                                num_heads,
                                                num_tokens,
                                                (size_t) head_dim * sizeof(float),
                                                fused_qkv_row_bytes,
                                                0);
    struct ggml_tensor *k_cur = ggml_view_3d_fn(ctx,
                                                qkv_base,
                                                head_dim,
                                                num_heads,
                                                num_tokens,
                                                (size_t) head_dim * sizeof(float),
                                                fused_qkv_row_bytes,
                                                packed_row_bytes);
    struct ggml_tensor *v_cur = ggml_view_3d_fn(ctx,
                                                qkv_base,
                                                head_dim,
                                                num_heads,
                                                num_tokens,
                                                (size_t) head_dim * sizeof(float),
                                                fused_qkv_row_bytes,
                                                2 * packed_row_bytes);
    if (!q_cur || !k_cur || !v_cur) {
        free(qkv_pack);
        ggml_free_fn(ctx);
        return 0;
    }

    struct ggml_cgraph *gf = ggml_new_graph_fn(ctx);
    if (!gf) {
        free(qkv_pack);
        ggml_free_fn(ctx);
        return 0;
    }

    ggml_build_forward_expand_fn(gf, q_cur);
    ggml_build_forward_expand_fn(gf, k_cur);
    ggml_build_forward_expand_fn(gf, v_cur);

    struct ggml_tensor *q_perm = ggml_permute_fn(ctx, q_cur, 0, 2, 1, 3);
    struct ggml_tensor *k_perm = ggml_permute_fn(ctx, k_cur, 0, 2, 1, 3);
    struct ggml_tensor *v_perm = ggml_permute_fn(ctx, v_cur, 1, 2, 0, 3);
    if (!q_perm || !k_perm || !v_perm) {
        free(qkv_pack);
        ggml_free_fn(ctx);
        return 0;
    }

    v_perm = ggml_cont_fn(ctx, v_perm);
    struct ggml_tensor *kq_scores = v_perm ? ggml_mul_mat_fn(ctx, k_perm, q_perm) : NULL;
    struct ggml_tensor *kq_scores_dump = kq_scores ? ggml_cont_fn(ctx, kq_scores) : NULL;
    struct ggml_tensor *kq_softmax_src = kq_scores_dump ? kq_scores_dump : kq_scores;
    struct ggml_tensor *kq_softmax = kq_softmax_src ?
        ggml_soft_max_ext_fn(ctx, kq_softmax_src, NULL, scale, 0.0f) : NULL;
    struct ggml_tensor *kqv = kq_softmax ? ggml_mul_mat_fn(ctx, v_perm, kq_softmax) : NULL;
    struct ggml_tensor *cur = kqv ? ggml_permute_fn(ctx, kqv, 0, 2, 1, 3) : NULL;
    if (cur) {
        cur = ggml_cont_2d_fn(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }
    if (!cur) {
        free(qkv_pack);
        ggml_free_fn(ctx);
        return 0;
    }

    if (kq_scores_dump) {
        ggml_build_forward_expand_fn(gf, kq_scores_dump);
    }
    ggml_build_forward_expand_fn(gf, cur);
    ggml_backend_t backend = ggml_backend_init_by_type_fn(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        free(qkv_pack);
        ggml_free_fn(ctx);
        return 0;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type_fn(backend);
    ggml_backend_t backends[1] = { backend };
    ggml_backend_buffer_type_t bufts[1] = { buft };
    ggml_backend_sched_t sched = ggml_backend_sched_new_fn(backends, bufts, 1, 8192, false, true);
    if (!sched) {
        free(qkv_pack);
        ggml_backend_free_fn(backend);
        ggml_free_fn(ctx);
        return 0;
    }
    ggml_backend_cpu_set_n_threads_fn(backend, 1);
    ggml_backend_sched_reset_fn(sched);
    if (!ggml_backend_sched_alloc_graph_fn(sched, gf)) {
        free(qkv_pack);
        ggml_backend_sched_free_fn(sched);
        ggml_backend_free_fn(backend);
        ggml_free_fn(ctx);
        return 0;
    }
    ggml_backend_tensor_set_fn(qkv_base, qkv_pack, 0, qkv_pack_elems * sizeof(float));
    free(qkv_pack);
    if (ggml_backend_sched_graph_compute_fn(sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free_fn(sched);
        ggml_backend_free_fn(backend);
        ggml_free_fn(ctx);
        return 0;
    }

    ck_attention_oracle_dump_tensor("kq_scores", layer_id, kq_scores_dump ? kq_scores_dump : kq_scores);
    ck_attention_oracle_dump_tensor("kq_softmax", layer_id, kq_softmax);
    ck_attention_oracle_dump_tensor("kqv_raw", layer_id, kqv);
    ck_attention_oracle_dump_meta("q_cur", layer_id, q_cur);
    ck_attention_oracle_dump_meta("k_cur", layer_id, k_cur);
    ck_attention_oracle_dump_meta("v_cur", layer_id, v_cur);
    ck_attention_oracle_dump_meta("q_perm", layer_id, q_perm);
    ck_attention_oracle_dump_meta("k_perm", layer_id, k_perm);
    ck_attention_oracle_dump_meta("v_perm", layer_id, v_perm);
    ck_attention_oracle_dump_meta("kq_scores", layer_id, kq_scores_dump ? kq_scores_dump : kq_scores);
    ck_attention_oracle_dump_meta("kq_softmax", layer_id, kq_softmax);
    ck_attention_oracle_dump_meta("kqv_raw", layer_id, kqv);
    ck_attention_oracle_dump_meta("kqv_out", layer_id, cur);

    {
        const size_t cur_elems = (size_t) cur->ne[0] * (size_t) cur->ne[1] * (size_t) cur->ne[2] * (size_t) cur->ne[3];
        float *cur_host = (float *) malloc(cur_elems * sizeof(float));
        if (!cur_host) {
            ggml_backend_sched_free_fn(sched);
            ggml_backend_free_fn(backend);
            ggml_free_fn(ctx);
            return 0;
        }
        ggml_backend_tensor_get_fn(cur, cur_host, 0, cur_elems * sizeof(float));
        const float *src = cur_host;
        ck_strict_store_next_gemm_a(src, (size_t) num_tokens * (size_t) num_heads * (size_t) head_dim);
        const size_t token_width = (size_t) num_heads * (size_t) head_dim;
        for (int t = 0; t < num_tokens; ++t) {
            const float *token_src = src + (size_t) t * token_width;
            for (int h = 0; h < num_heads; ++h) {
                float *dst = output + ck_attention_oracle_qkv_index(h, t, 0, num_tokens, aligned_head_dim);
                memcpy(dst, token_src + (size_t) h * (size_t) head_dim, (size_t) head_dim * sizeof(float));
                for (int d = head_dim; d < aligned_head_dim; ++d) {
                    dst[d] = 0.0f;
                }
            }
        }
        free(cur_host);
    }

    ok = 1;
    ggml_backend_sched_free_fn(sched);
    ggml_backend_free_fn(backend);
    ggml_free_fn(ctx);
    return ok;
}
