/**
 * @file attention_kernels.c
 * @brief Attention score/softmax/output kernels with SIMD (SSE/AVX/AVX512)
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
 * Attention: softmax(Q @ K^T / sqrt(d)) @ V
 * Supports GQA (grouped-query attention) with head broadcasting.
 */

#ifndef CK_ENABLE_LLAMA_CPP_PARITY
#define CK_ENABLE_LLAMA_CPP_PARITY 0
#endif

#include "bf16_utils.h"
#include "attention_oracle_ggml.h"
#include "ckernel_engine.h"
#include "ck_threadpool.h"
#if CK_ENABLE_LLAMA_CPP_PARITY
#include <ggml.h>
#endif
#include <dlfcn.h>
#ifndef RTLD_DEFAULT
#define RTLD_DEFAULT ((void *)0)
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ck_speed_profiles.h"
#include <string.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSE2__)
#include <immintrin.h>
#endif

/* Convert BF16 tensor to FP32 using caller-provided buffer (no malloc!) */
static void convert_bf16_tensor_to_buf(const uint16_t *src, float *dst, size_t count)
{
    if (!dst || !src) return;
    bf16_tensor_to_float(src, dst, count);
}

// Helpers for head-major layouts used in attention.
// Q/K/V layout: [head][token][head_dim] with stride aligned_head_dim.
static inline size_t qkv_index(int h,
                               int t,
                               int d,
                               int num_tokens,
                               int aligned_head_dim)
{
    return ((size_t)h * (size_t)num_tokens + (size_t)t) * (size_t)aligned_head_dim
         + (size_t)d;
}

// Match llama.cpp flash-attention input handling where F32 K/V are rounded through F16.
static inline float ck_round_fp16_scalar(float x) {
    return CK_FP16_TO_FP32(CK_FP32_TO_FP16(x));
}

static void ck_round_fp16_buffer(const float *src, float *dst, size_t count)
{
    size_t i = 0;
#if defined(__AVX2__) && defined(__F16C__)
    for (; i + 8 <= count; i += 8) {
        const __m256 value = _mm256_loadu_ps(src + i);
        const __m128i half = _mm256_cvtps_ph(
            value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(half));
    }
#endif
    for (; i < count; ++i) {
        dst[i] = ck_round_fp16_scalar(src[i]);
    }
}

static inline void ck_local_fp16_to_fp32_row(const uint16_t *src, float *dst, int n)
{
    if (!src || !dst || n <= 0) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        dst[i] = CK_FP16_TO_FP32(src[i]);
    }
}

static inline void ck_local_fp16_to_fp32_2d(const uint16_t *src,
                                             float *dst,
                                             int rows,
                                             int cols,
                                             int src_stride,
                                             int dst_stride)
{
    if (!src || !dst || rows <= 0 || cols <= 0) {
        return;
    }
    for (int r = 0; r < rows; ++r) {
        ck_local_fp16_to_fp32_row(src + (size_t)r * (size_t)src_stride,
                                  dst + (size_t)r * (size_t)dst_stride,
                                  cols);
    }
}

#if defined(__GNUC__) || defined(__clang__)
#define CK_NOINLINE __attribute__((noinline))
#else
#define CK_NOINLINE
#endif

#if defined(__clang__)
#define CK_OPTNONE __attribute__((optnone))
#elif defined(__GNUC__)
#define CK_OPTNONE __attribute__((optimize("O0")))
#else
#define CK_OPTNONE
#endif

static CK_NOINLINE CK_OPTNONE float ck_vec_dot_f32_strict(const float *x,
                                                          const float *y,
                                                          int n)
{
    float sumf = 0.0f;
    for (int i = 0; i < n; ++i) {
        volatile float prod = x[i] * y[i];
        volatile float next = sumf + prod;
        sumf = next;
    }
    return sumf;
}

static CK_NOINLINE CK_OPTNONE float ck_vec_dot_f32x_f32_to_f32_via_f64(const float *x,
                                                                       const float *y,
                                                                       int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        volatile double prod = (double) x[i] * (double) y[i];
        volatile double next = sum + prod;
        sum = next;
    }
    return (float) sum;
}

static CK_NOINLINE CK_OPTNONE float ck_vec_dot_f32_reverse_strict(const float *x,
                                                                  const float *y,
                                                                  int n)
{
    float sumf = 0.0f;
    for (int i = n - 1; i >= 0; --i) {
        volatile float prod = x[i] * y[i];
        volatile float next = sumf + prod;
        sumf = next;
    }
    return sumf;
}

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
} __attribute__((packed)) ck_attention_vec_dump_header_t;

static const char ck_attention_vec_dump_magic[8] = {'C', 'K', 'D', 'M', 'P', '\0', '\0', '\0'};
static const uint32_t ck_attention_vec_dump_version = 1u;
static int ck_attention_vec_dump_layer_seq = 0;

static void ck_attention_trace_query(const char *tag,
                                     int layer_id,
                                     int head_id,
                                     int query_id,
                                     int value);

static int ck_attention_vec_dump_enabled(void)
{
    const char *v = getenv("CK_STRICT_ATTN_VEC_DUMP");
    return v && v[0] && strcmp(v, "0") != 0;
}

static int ck_attention_vec_dump_vcols_enabled(void)
{
    const char *v = getenv("CK_STRICT_ATTN_DUMP_VCOLS");
    return v && v[0] && strcmp(v, "0") != 0;
}

static int ck_attention_reverse_out_dot_enabled(void)
{
    const char *v = getenv("CK_STRICT_ATTN_REVERSE_OUT_DOT");
    return v && v[0] && strcmp(v, "0") != 0;
}

#if CK_ENABLE_LLAMA_CPP_PARITY
static int ck_attention_ggml_out_graph_enabled(void)
{
    const char *v = getenv("CK_STRICT_ATTN_GGML_OUT_GRAPH");
    return v && v[0] && strcmp(v, "0") != 0;
}
#else
static int ck_attention_ggml_out_graph_enabled(void)
{
    return 0;
}
#endif

static int ck_attention_vec_dump_parse_env_int(const char *name, int *out)
{
    const char *v = getenv(name);
    if (!v || !v[0]) {
        return 0;
    }
    char *end = NULL;
    long parsed = strtol(v, &end, 10);
    if (end == v || (end && *end != '\0') || parsed < 0 || parsed > INT32_MAX) {
        return 0;
    }
    if (out) {
        *out = (int) parsed;
    }
    return 1;
}

static int ck_attention_vec_dump_should_emit(int layer_id, int head_id, int query_id)
{
    if (!ck_attention_vec_dump_enabled()) {
        return 0;
    }
    int want_layer = -1;
    int want_head = -1;
    int want_query = -1;
    if (!ck_attention_vec_dump_parse_env_int("CK_STRICT_ATTN_DUMP_LAYER", &want_layer)) {
        return 0;
    }
    const int have_head = ck_attention_vec_dump_parse_env_int("CK_STRICT_ATTN_DUMP_HEAD", &want_head);
    const int have_query = ck_attention_vec_dump_parse_env_int("CK_STRICT_ATTN_DUMP_QUERY", &want_query);
    const int trace_target = layer_id == want_layer &&
                             (!have_head || head_id == want_head) &&
                             (!have_query || query_id == want_query);
    if (layer_id != want_layer) {
        return 0;
    }
    if (have_head && head_id != want_head) {
        return 0;
    }
    if (have_query && query_id != want_query) {
        return 0;
    }
    if (trace_target) {
        ck_attention_trace_query("vec_dump_should_emit", layer_id, head_id, query_id, 1);
    }
    return 1;
}

static int ck_attention_vec_dump_next_layer_id(void)
{
    const int layer_id = ck_attention_vec_dump_layer_seq;
    ck_attention_vec_dump_layer_seq += 1;
    return layer_id;
}

static void ck_attention_vec_dump_tensor(const char *name,
                                         int layer_id,
                                         int query_id,
                                         const float *data,
                                         size_t elem_count)
{
    const char *dir = getenv("CK_PARITY_DIR");
    if (!dir || !dir[0] || !name || !name[0] || !data || elem_count == 0) {
        return;
    }

    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, "strict_internal.bin");
    FILE *f = fopen(path, "ab");
    if (!f) {
        return;
    }

    ck_attention_vec_dump_header_t h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, ck_attention_vec_dump_magic, sizeof(ck_attention_vec_dump_magic));
    h.version = ck_attention_vec_dump_version;
    h.layer_id = layer_id;
    strncpy(h.op_name, name, sizeof(h.op_name) - 1);
    h.dtype = 0u;
    h.rank = 1u;
    h.shape[0] = (int64_t) elem_count;
    h.elem_count = (uint32_t) elem_count;
    h.token_id = query_id;

    fwrite(&h, sizeof(h), 1, f);
    fwrite(data, sizeof(float), elem_count, f);
    fclose(f);
}

static void ck_attention_trace(const char *branch, int layer_id, int head_id)
{
    const char *enabled = getenv("CK_STRICT_ATTN_TRACE");
    const char *dir = getenv("CK_PARITY_DIR");
    if (!enabled || !enabled[0] || strcmp(enabled, "0") == 0 || !dir || !dir[0] || !branch || !branch[0]) {
        return;
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, "strict_trace.txt");
    FILE *f = fopen(path, "a");
    if (!f) {
        return;
    }
    fprintf(f, "layer=%d head=%d branch=%s\n", layer_id, head_id, branch);
    fclose(f);
}

static void ck_attention_trace_query(const char *tag,
                                     int layer_id,
                                     int head_id,
                                     int query_id,
                                     int value)
{
    const char *enabled = getenv("CK_STRICT_ATTN_TRACE");
    const char *dir = getenv("CK_PARITY_DIR");
    if (!enabled || !enabled[0] || strcmp(enabled, "0") == 0 || !dir || !dir[0] || !tag || !tag[0]) {
        return;
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, "strict_trace.txt");
    FILE *f = fopen(path, "a");
    if (!f) {
        return;
    }
    fprintf(f, "layer=%d head=%d query=%d tag=%s value=%d\n", layer_id, head_id, query_id, tag, value);
    fclose(f);
}

static void ck_attention_trace_float(const char *tag,
                                     int layer_id,
                                     int head_id,
                                     float value)
{
    const char *enabled = getenv("CK_STRICT_ATTN_TRACE");
    const char *dir = getenv("CK_PARITY_DIR");
    if (!enabled || !enabled[0] || strcmp(enabled, "0") == 0 || !dir || !dir[0] || !tag || !tag[0]) {
        return;
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, "strict_trace.txt");
    FILE *f = fopen(path, "a");
    if (!f) {
        return;
    }
    fprintf(f, "layer=%d head=%d tag=%s float=%.17g\n", layer_id, head_id, tag, (double) value);
    fclose(f);
}

static void ck_attention_vec_dump_selected_query(const float *raw_scores,
                                                 const float *probs,
                                                 const float *out_vec,
                                                 const float *v_cols,
                                                 int kv_tokens,
                                                 int head_dim,
                                                 int layer_id,
                                                 int head_id,
                                                 int query_id)
{
    if (!ck_attention_vec_dump_should_emit(layer_id, head_id, query_id)) {
        return;
    }
    ck_attention_trace_query("vec_dump_selected_query", layer_id, head_id, query_id, 1);
    char name[32];
    snprintf(name, sizeof(name), "kq_scores_h%d_q%d", head_id, query_id);
    ck_attention_vec_dump_tensor(name, layer_id, query_id, raw_scores, (size_t) kv_tokens);
    snprintf(name, sizeof(name), "kq_soft_h%d_q%d", head_id, query_id);
    ck_attention_vec_dump_tensor(name, layer_id, query_id, probs, (size_t) kv_tokens);
    snprintf(name, sizeof(name), "kqv_out_h%d_q%d", head_id, query_id);
    ck_attention_vec_dump_tensor(name, layer_id, query_id, out_vec, (size_t) head_dim);
    if (v_cols && ck_attention_vec_dump_vcols_enabled()) {
        snprintf(name, sizeof(name), "vcols_h%d_q%d", head_id, query_id);
        ck_attention_vec_dump_tensor(name, layer_id, query_id, v_cols, (size_t) kv_tokens * (size_t) head_dim);
    }
}

static inline void ck_vec_scale_f32_inplace(float *x, int n, float scale);
static inline float ck_vec_max_f32_contig(const float *x, int n);

#if CK_ENABLE_LLAMA_CPP_PARITY
struct ggml_compute_params;
typedef void (*ck_ggml_vec_dot_f32_fn)(int, float *, size_t, const float *, size_t, const float *, size_t, int);
typedef double (*ck_ggml_vec_soft_max_f32_fn)(int, float *, const float *, float);
typedef void (*ck_ggml_compute_forward_mul_mat_fn)(const struct ggml_compute_params *, struct ggml_tensor *);
typedef void (*ck_ggml_compute_forward_soft_max_fn)(const struct ggml_compute_params *, struct ggml_tensor *);
typedef void (*ck_ggml_cpu_init_fn)(void);
typedef struct ggml_context *(*ck_ggml_init_fn)(struct ggml_init_params);
typedef void (*ck_ggml_free_fn)(struct ggml_context *);
typedef struct ggml_tensor *(*ck_ggml_new_tensor_2d_fn)(struct ggml_context *, enum ggml_type, int64_t, int64_t);
typedef struct ggml_tensor *(*ck_ggml_mul_mat_graph_fn)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *);
typedef struct ggml_cgraph *(*ck_ggml_new_graph_fn)(struct ggml_context *);
typedef void (*ck_ggml_build_forward_expand_fn)(struct ggml_cgraph *, struct ggml_tensor *);
typedef enum ggml_status (*ck_ggml_graph_compute_with_ctx_fn)(struct ggml_context *, struct ggml_cgraph *, int);
typedef void (*ck_ggml_set_input_fn)(struct ggml_tensor *);

struct ggml_threadpool;
struct ggml_compute_params {
    int ith, nth;
    size_t wsize;
    void * wdata;
    struct ggml_threadpool * threadpool;
    bool use_ref;
};

static void *ck_resolve_ggml_cpu_so_handle(void)
{
    static int tried = 0;
    static void *handle = NULL;
    if (!tried) {
        tried = 1;
        const char *env_path = getenv("CK_GGML_CPU_SO");
        const char *env_dir = getenv("CK_GGML_LIB_DIR");
        const char *dirs[] = {
            "/opt/app-root/src/Software/llama.cpp/build/bin",
            "./llama.cpp/build/bin",
            "llama.cpp/build/bin",
            NULL,
        };
        char path_buf[512];
        if (env_dir && env_dir[0]) {
            snprintf(path_buf, sizeof(path_buf), "%s/libggml-base.so", env_dir);
            dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
            snprintf(path_buf, sizeof(path_buf), "%s/libggml.so", env_dir);
            dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
            snprintf(path_buf, sizeof(path_buf), "%s/libggml-cpu.so", env_dir);
            handle = dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
        }
        for (int i = 0; !handle && dirs[i] != NULL; ++i) {
            snprintf(path_buf, sizeof(path_buf), "%s/libggml-base.so", dirs[i]);
            dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
            snprintf(path_buf, sizeof(path_buf), "%s/libggml.so", dirs[i]);
            dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
            snprintf(path_buf, sizeof(path_buf), "%s/libggml-cpu.so", dirs[i]);
            handle = dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
            if (handle) {
                break;
            }
        }
        if (!handle && env_path && env_path[0]) {
            handle = dlopen(env_path, RTLD_NOW | RTLD_GLOBAL);
        }
        const char *candidates[] = {
            "libggml-cpu.so",
            "libggml-cpu.so.0",
            NULL,
        };
        for (int i = 0; !handle && candidates[i] != NULL; ++i) {
            handle = dlopen(candidates[i], RTLD_NOW | RTLD_GLOBAL);
        }
    }
    return handle;
}

static void *ck_resolve_ggml_symbol(const char *name)
{
    void *sym = dlsym(RTLD_DEFAULT, name);
    if (sym) {
        return sym;
    }
    void *handle = ck_resolve_ggml_cpu_so_handle();
    if (!handle) {
        return NULL;
    }
    return dlsym(handle, name);
}

static ck_ggml_vec_dot_f32_fn ck_resolve_ggml_vec_dot_f32(void)
{
    static int tried = 0;
    static ck_ggml_vec_dot_f32_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_vec_dot_f32_fn) ck_resolve_ggml_symbol("ggml_vec_dot_f32");
    }
    return fn;
}

static ck_ggml_vec_soft_max_f32_fn ck_resolve_ggml_vec_soft_max_f32(void)
{
    static int tried = 0;
    static ck_ggml_vec_soft_max_f32_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_vec_soft_max_f32_fn) ck_resolve_ggml_symbol("ggml_vec_soft_max_f32");
    }
    return fn;
}

static ck_ggml_compute_forward_mul_mat_fn ck_resolve_ggml_compute_forward_mul_mat(void)
{
    static int tried = 0;
    static ck_ggml_compute_forward_mul_mat_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_compute_forward_mul_mat_fn) ck_resolve_ggml_symbol("ggml_compute_forward_mul_mat");
    }
    return fn;
}

static ck_ggml_compute_forward_soft_max_fn ck_resolve_ggml_compute_forward_soft_max(void)
{
    static int tried = 0;
    static ck_ggml_compute_forward_soft_max_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_compute_forward_soft_max_fn) ck_resolve_ggml_symbol("ggml_compute_forward_soft_max");
    }
    return fn;
}

static ck_ggml_cpu_init_fn ck_resolve_ggml_cpu_init(void)
{
    static int tried = 0;
    static ck_ggml_cpu_init_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_cpu_init_fn) ck_resolve_ggml_symbol("ggml_cpu_init");
    }
    return fn;
}

static ck_ggml_init_fn ck_resolve_ggml_init(void)
{
    static int tried = 0;
    static ck_ggml_init_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_init_fn) ck_resolve_ggml_symbol("ggml_init");
    }
    return fn;
}

static ck_ggml_free_fn ck_resolve_ggml_free(void)
{
    static int tried = 0;
    static ck_ggml_free_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_free_fn) ck_resolve_ggml_symbol("ggml_free");
    }
    return fn;
}

static ck_ggml_new_tensor_2d_fn ck_resolve_ggml_new_tensor_2d(void)
{
    static int tried = 0;
    static ck_ggml_new_tensor_2d_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_new_tensor_2d_fn) ck_resolve_ggml_symbol("ggml_new_tensor_2d");
    }
    return fn;
}

static ck_ggml_mul_mat_graph_fn ck_resolve_ggml_mul_mat_graph(void)
{
    static int tried = 0;
    static ck_ggml_mul_mat_graph_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_mul_mat_graph_fn) ck_resolve_ggml_symbol("ggml_mul_mat");
    }
    return fn;
}

static ck_ggml_new_graph_fn ck_resolve_ggml_new_graph(void)
{
    static int tried = 0;
    static ck_ggml_new_graph_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_new_graph_fn) ck_resolve_ggml_symbol("ggml_new_graph");
    }
    return fn;
}

static ck_ggml_build_forward_expand_fn ck_resolve_ggml_build_forward_expand(void)
{
    static int tried = 0;
    static ck_ggml_build_forward_expand_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_build_forward_expand_fn) ck_resolve_ggml_symbol("ggml_build_forward_expand");
    }
    return fn;
}

static ck_ggml_graph_compute_with_ctx_fn ck_resolve_ggml_graph_compute_with_ctx(void)
{
    static int tried = 0;
    static ck_ggml_graph_compute_with_ctx_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_graph_compute_with_ctx_fn) ck_resolve_ggml_symbol("ggml_graph_compute_with_ctx");
    }
    return fn;
}

static ck_ggml_set_input_fn ck_resolve_ggml_set_input(void)
{
    static int tried = 0;
    static ck_ggml_set_input_fn fn = NULL;
    if (!tried) {
        tried = 1;
        fn = (ck_ggml_set_input_fn) ck_resolve_ggml_symbol("ggml_set_input");
    }
    return fn;
}

static inline void ck_ggml_init_tensor_f32(struct ggml_tensor *t,
                                           int64_t ne0,
                                           int64_t ne1,
                                           int64_t ne2,
                                           int64_t ne3,
                                           size_t nb0,
                                           size_t nb1,
                                           size_t nb2,
                                           size_t nb3,
                                           void *data)
{
    memset(t, 0, sizeof(*t));
    t->type = GGML_TYPE_F32;
    t->buffer = NULL;
    t->ne[0] = ne0;
    t->ne[1] = ne1;
    t->ne[2] = ne2;
    t->ne[3] = ne3;
    t->nb[0] = nb0;
    t->nb[1] = nb1;
    t->nb[2] = nb2;
    t->nb[3] = nb3;
    t->op = GGML_OP_NONE;
    t->data = data;
}
#endif

#if defined(__SSE2__)
#if defined(__FMA__)
#define CK_MADD128(x, y, z) _mm_fmadd_ps(x, y, z)
#define CK_NMADD128(x, y, z) _mm_fnmadd_ps(x, y, z)
#else
#define CK_MADD128(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define CK_NMADD128(x, y, z) _mm_sub_ps(z, _mm_mul_ps(x, y))
#endif

static inline float ck_hsum128_ps(__m128 v) {
    v = _mm_add_ps(v, _mm_movehl_ps(v, v));
    v = _mm_add_ss(v, _mm_movehdup_ps(v));
    return _mm_cvtss_f32(v);
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
static inline float ck_hsum256_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    return ck_hsum128_ps(_mm_add_ps(lo, hi));
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
static inline __m256 ck_ggml_v_expf256(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(
        _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                      _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)),
            u,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
        u,
        _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
        return _mm256_fmadd_ps(j, k, k);
    }
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000u));
    const __m256 s1 =
        _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                      _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(_mm256_castsi256_ps(c),
                              _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}
#endif

#if defined(__SSE2__)
static inline __m128 ck_ggml_v_expf128(__m128 x) {
    const __m128 r = _mm_set1_ps(0x1.8p23f);
    const __m128 z = CK_MADD128(x, _mm_set1_ps(0x1.715476p+0f), r);
    const __m128 n = _mm_sub_ps(z, r);
    const __m128 b = CK_NMADD128(n, _mm_set1_ps(0x1.7f7d1cp-20f),
                                 CK_NMADD128(n, _mm_set1_ps(0x1.62e4p-1f), x));
    const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    const __m128 k = _mm_castsi128_ps(
        _mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    const __m128i c = _mm_castps_si128(
        _mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    const __m128 u = _mm_mul_ps(b, b);
    const __m128 j = CK_MADD128(
        CK_MADD128(
            CK_MADD128(_mm_set1_ps(0x1.0e4020p-7f), b, _mm_set1_ps(0x1.573e2ep-5f)),
            u,
            CK_MADD128(_mm_set1_ps(0x1.555e66p-3f), b, _mm_set1_ps(0x1.fffdb6p-2f))),
        u,
        _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_ps(_mm_castsi128_ps(c))) {
        return CK_MADD128(j, k, k);
    }
    const __m128i g = _mm_and_si128(
        _mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
        _mm_set1_epi32(0x82000000u));
    const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    const __m128i d = _mm_castps_si128(
        _mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(
            _mm_castsi128_ps(d),
            _mm_or_ps(
                _mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(CK_MADD128(s2, j, s2), s1)),
                _mm_andnot_ps(_mm_castsi128_ps(c), CK_MADD128(k, j, k)))));
}
#endif

static CK_NOINLINE CK_OPTNONE float ck_ggml_vec_dot_f32_contig(const float *x,
                                                               const float *y,
                                                               int n)
{
    // Keep this a literal port of ggml_vec_dot_f32 so strict parity can match
    // llama.cpp's CPU attention path instead of merely approximating it.
#if defined(__AVX__)
    float sumf = 0.0f;
    const int np = (n & ~31);
    __m256 sum[4] = {
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    };

    for (int i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; ++j) {
            const __m256 ax = _mm256_loadu_ps(x + i + j * 8);
            const __m256 ay = _mm256_loadu_ps(y + i + j * 8);
#if defined(__FMA__)
            sum[j] = _mm256_fmadd_ps(ax, ay, sum[j]);
#else
            sum[j] = _mm256_add_ps(_mm256_mul_ps(ax, ay), sum[j]);
#endif
        }
    }

    sum[0] = _mm256_add_ps(sum[0], sum[2]);
    sum[1] = _mm256_add_ps(sum[1], sum[3]);
    sum[0] = _mm256_add_ps(sum[0], sum[1]);
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sum[0]),
                                 _mm256_extractf128_ps(sum[0], 1));
    const __m128 t1 = _mm_hadd_ps(t0, t0);
    sumf = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }
    return sumf;
#elif defined(__SSE2__)
    float sumf = 0.0f;
    const int np = (n & ~15);
    __m128 sum[4] = {
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
    };

    for (int i = 0; i < np; i += 16) {
        for (int j = 0; j < 4; ++j) {
            const __m128 ax = _mm_loadu_ps(x + i + j * 4);
            const __m128 ay = _mm_loadu_ps(y + i + j * 4);
#if defined(__FMA__)
            sum[j] = _mm_fmadd_ps(ax, ay, sum[j]);
#else
            sum[j] = _mm_add_ps(_mm_mul_ps(ax, ay), sum[j]);
#endif
        }
    }

    sum[0] = _mm_add_ps(sum[0], sum[2]);
    sum[1] = _mm_add_ps(sum[1], sum[3]);
    sum[0] = _mm_add_ps(sum[0], sum[1]);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    sum[0] = _mm_add_ps(sum[0], _mm_movehl_ps(sum[0], sum[0]));
    sum[0] = _mm_add_ss(sum[0], _mm_movehdup_ps(sum[0]));
#else
    __m128 tmp = _mm_shuffle_ps(sum[0], sum[0], _MM_SHUFFLE(2, 3, 0, 1));
    sum[0] = _mm_add_ps(sum[0], tmp);
    tmp = _mm_movehl_ps(tmp, sum[0]);
    sum[0] = _mm_add_ss(sum[0], tmp);
#endif
    sumf = _mm_cvtss_f32(sum[0]);

    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }
    return sumf;
#else
    double sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (double) (x[i] * y[i]);
    }
    return (float) sumf;
#endif
}

static inline float ck_attention_dot_f16_unfused_llama(const uint16_t *x,
                                                        const uint16_t *y,
                                                        int n)
{
    int i = 0;
#if defined(__AVX2__) && defined(__F16C__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        const __m256 x0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (x + i)));
        const __m256 y0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (y + i)));
        const __m256 x1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (x + i + 8)));
        const __m256 y1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (y + i + 8)));
        const __m256 x2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (x + i + 16)));
        const __m256 y2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (y + i + 16)));
        const __m256 x3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (x + i + 24)));
        const __m256 y3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (y + i + 24)));
#if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(x0, y0, sum0);
        sum1 = _mm256_fmadd_ps(x1, y1, sum1);
        sum2 = _mm256_fmadd_ps(x2, y2, sum2);
        sum3 = _mm256_fmadd_ps(x3, y3, sum3);
#else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x0, y0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(x1, y1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(x2, y2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(x3, y3));
#endif
    }
    sum0 = _mm256_add_ps(sum0, sum2);
    sum1 = _mm256_add_ps(sum1, sum3);
    sum0 = _mm256_add_ps(sum0, sum1);
    const __m128 pair = _mm_add_ps(
        _mm256_castps256_ps128(sum0),
        _mm256_extractf128_ps(sum0, 1));
    const __m128 half = _mm_hadd_ps(pair, pair);
    float result = _mm_cvtss_f32(_mm_hadd_ps(half, half));
#else
    float result = 0.0f;
#endif
    for (; i < n; ++i) {
        result += CK_FP16_TO_FP32(x[i]) * CK_FP16_TO_FP32(y[i]);
    }
    return result;
}

static CK_NOINLINE CK_OPTNONE float ck_attention_strict_scale_f32(int head_dim)
{
    // Keep strict parity on the precise libm sqrtf path. icx -O3 on AVX2 was
    // lowering 1/sqrtf(d) to a slightly smaller effective scale, which is
    // enough to move layer-0 vision softmax by ~2e-7 and snowball later.
    volatile float hd = (float) head_dim;
    float (*sqrtf_fn)(float) = sqrtf;
    volatile float root = sqrtf_fn(hd);
    volatile float one = 1.0f;
    volatile float scale = one / root;
    return scale;
}

typedef float (*ck_attention_math_f32_fn)(float);

static float ck_attention_reference_expf(float value)
{
#if defined(__linux__)
    static void *libm_handle = NULL;
    static ck_attention_math_f32_fn fn = NULL;
    static int resolved = 0;
    if (!resolved) {
        libm_handle = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
        if (libm_handle) {
            fn = (ck_attention_math_f32_fn) dlsym(libm_handle, "expf");
        }
        resolved = 1;
    }
    if (fn) {
        return fn(value);
    }
#endif
    return expf(value);
}

static CK_NOINLINE CK_OPTNONE double ck_ggml_vec_soft_max_row(int n,
                                                              float *y,
                                                              const float *x,
                                                              float max)
{
    int i = 0;
    double sum = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        const __m256 val = ck_ggml_v_expf256(
            _mm256_sub_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (double) _mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        const __m128 val = ck_ggml_v_expf128(
            _mm_sub_ps(_mm_loadu_ps(x + i), _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        __m128 acc = _mm_add_ps(val, _mm_movehl_ps(val, val));
        acc = _mm_add_ss(acc, _mm_movehdup_ps(acc));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 acc = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, acc);
        acc = _mm_add_ss(acc, tmp);
#endif
        sum += (double) _mm_cvtss_f32(acc);
    }
#endif

    for (; i < n; ++i) {
        const float val = expf(x[i] - max);
        y[i] = val;
        sum += (double) val;
    }

    return sum;
}

// Scores layout matches causal_softmax_head_major:
// [head][query_token][key_token] with stride aligned_context_window.
static inline size_t score_index(int h,
                                 int i,
                                 int j,
                                 int aligned_context_window)
{
    return ((size_t)h * (size_t)aligned_context_window * (size_t)aligned_context_window)
         + (size_t)i * (size_t)aligned_context_window
         + (size_t)j;
}

/**
 * Causal attention forward (score-matrix version)
 * @test test_attention.py::TestAttentionForward::test_causal_forward
 * @test test_attention.py::TestAttentionForward::test_gqa_broadcast
 * @test test_attention.py::TestAttentionForward::test_exact_vs_fast
 * @test test_parity.py::test_attention_parity
 *
 * Computes softmax(Q @ K^T / sqrt(d)) @ V with causal masking.
 * Uses O(N^2) memory for scores matrix.
 *
 * After changes: make test && make llamacpp-parity-full
 */
void attention_forward_causal_head_major(const float *q,
                                         const float *k,
                                         const float *v,
                                         float *scores,
                                         float *output,
                                         int num_heads,
                                         int num_tokens,
                                         int head_dim,
                                         int aligned_head_dim,
                                         int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Phase 1: compute scaled dot-product scores Q·K^T / sqrt(d_k),
    // lower triangle only (j <= i).
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            // Ensure upper triangle is zeroed so there are no stale values
            // before the softmax kernel runs.
            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    // Phase 2: apply causal row-wise softmax in-place over j <= i.
    causal_softmax_head_major(scores,
                              num_heads,
                              num_tokens,
                              aligned_context_window);

    // Phase 3: attention weights · V.
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);

            // Zero the full aligned head slice so padded dims stay clean.
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            // Weighted sum over causal positions.
            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

/**
 * Causal attention forward (exact version using stdlib expf)
 * @test test_attention.py::TestAttentionForward::test_exact_single
 * @test test_attention.py::TestAttentionForward::test_exact_vs_fast
 *
 * Uses standard library expf for numerical accuracy reference.
 * Slower but provides maximum accuracy.
 *
 * After changes: make test
 */
void attention_forward_causal_head_major_exact(const float *q,
                                                const float *k,
                                                const float *v,
                                                float *scores,
                                                float *output,
                                                int num_heads,
                                                int num_tokens,
                                                int head_dim,
                                                int aligned_head_dim,
                                                int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Phase 1: compute scaled dot-product scores Q·K^T / sqrt(d_k),
    // lower triangle only (j <= i).
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            // Ensure upper triangle is zeroed so there are no stale values
            // before the softmax kernel runs.
            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    // Phase 2: apply causal row-wise softmax using exact expf.
    causal_softmax_head_major_exact(scores,
                                     num_heads,
                                     num_tokens,
                                     aligned_context_window);

    // Phase 3: attention weights · V.
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);

            // Zero the full aligned head slice so padded dims stay clean.
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            // Weighted sum over causal positions.
            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(h, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

/**
 * GQA causal attention forward (score-matrix version)
 * @test test_attention.py::TestAttentionForward::test_gqa_forward
 * @test test_attention.py::TestAttentionForward::test_gqa_broadcast
 * @test test_attention_backward.py::TestAttentionBackwardGQA::test_gqa_backward
 * @test test_parity.py::test_attention_gqa_parity
 *
 * Grouped-query attention: Q has num_heads, K/V have num_kv_heads.
 * Each query head maps to a KV head via ratio.
 *
 * After changes: make test && make llamacpp-parity-full
 */
void attention_forward_causal_head_major_gqa(const float *q,
                                             const float *k,
                                             const float *v,
                                             float *scores,
                                             float *output,
                                             int num_heads,
                                             int num_kv_heads,
                                             int num_tokens,
                                             int head_dim,
                                             int aligned_head_dim,
                                             int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    causal_softmax_head_major(scores,
                              num_heads,
                              num_tokens,
                              aligned_context_window);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

/**
 * GQA causal attention forward (exact version using stdlib expf)
 * @test test_attention.py::TestAttentionForward::test_gqa_exact
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_gqa
 *
 * Uses standard library expf for numerical accuracy reference.
 * Used by BF16 wrapper to avoid approximation error accumulation.
 *
 * After changes: make test
 */
void attention_forward_causal_head_major_gqa_exact(const float *q,
                                                    const float *k,
                                                    const float *v,
                                                    float *scores,
                                                    float *output,
                                                    int num_heads,
                                                    int num_kv_heads,
                                                    int num_tokens,
                                                    int head_dim,
                                                    int aligned_head_dim,
                                                    int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                size_t base_q = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
                size_t base_k = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    dot += q[base_q + d] * k[base_k + d];
                }

                scores[score_index(h, i, j, aligned_context_window)] = dot * scale;
            }

            for (int j = i + 1; j < num_tokens; ++j) {
                scores[score_index(h, i, j, aligned_context_window)] = 0.0f;
            }
        }
    }

    // Use exact softmax with standard library expf
    causal_softmax_head_major_exact(scores,
                                     num_heads,
                                     num_tokens,
                                     aligned_context_window);

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        for (int i = 0; i < num_tokens; ++i) {
            size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
            for (int d = 0; d < aligned_head_dim; ++d) {
                output[out_base + d] = 0.0f;
            }

            for (int j = 0; j <= i; ++j) {
                float w = scores[score_index(h, i, j, aligned_context_window)];
                size_t v_base = qkv_index(kv_head, j, 0, num_tokens, aligned_head_dim);

                for (int d = 0; d < head_dim; ++d) {
                    output[out_base + d] += w * v[v_base + d];
                }
            }
        }
    }
}

/**
 * BF16 GQA causal attention forward
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_forward
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_gqa
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_flash
 *
 * Accepts BF16 inputs, converts to FP32, uses exact softmax.
 * Caller provides scratch buffers (no per-call malloc).
 *
 * After changes: make test
 */
void attention_forward_causal_head_major_gqa_bf16(const uint16_t *q,
                                                  const uint16_t *k,
                                                  const uint16_t *v,
                                                  float *scores,
                                                  float *output,
                                                  int num_heads,
                                                  int num_kv_heads,
                                                  int num_tokens,
                                                  int head_dim,
                                                  int aligned_head_dim,
                                                  int aligned_context_window,
                                                  float *scratch_q,
                                                  float *scratch_k,
                                                  float *scratch_v)
{
    const size_t q_elems = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_elems = (size_t)num_kv_heads * (size_t)num_tokens * (size_t)aligned_head_dim;

    if (!scratch_q || !scratch_k || !scratch_v) return;

    convert_bf16_tensor_to_buf(q, scratch_q, q_elems);
    convert_bf16_tensor_to_buf(k, scratch_k, kv_elems);
    convert_bf16_tensor_to_buf(v, scratch_v, kv_elems);

    // Use exact version to avoid fast exp approximation error accumulating
    // with BF16 precision loss.
    attention_forward_causal_head_major_gqa_exact(scratch_q, scratch_k, scratch_v,
                                                   scores, output,
                                                   num_heads, num_kv_heads,
                                                   num_tokens, head_dim,
                                                   aligned_head_dim, aligned_context_window);
    /* No free - caller owns scratch buffers */
}

// ============================================================================
// ATTENTION FORWARD - Flash-style (no scores materialization)
// ============================================================================
//
// Computes the same causal attention output as `attention_forward_causal_head_major_gqa`,
// but does not materialize the [H, T, T] score/weight matrices. This is useful for:
//   - Prefill: avoids large scratch buffers and improves cache locality
//   - Decode: supports KV-cache attention for a single token
//
// SIMD-optimized implementations for AVX-512, AVX2, and AVX follow.

// ============================================================================
// AVX-512 SIMD Flash Attention (16 floats per vector)
// ============================================================================
#if defined(__AVX512F__)
static void attention_flash_query_causal_avx512(const float *q_vec,
                                                 const float *k_head,
                                                 const float *v_head,
                                                 int kv_tokens,
                                                 int head_dim,
                                                 int aligned_head_dim,
                                                 float scale,
                                                 float *out_vec)
{
    // Online softmax: m = running max, s = running sum(exp(score - m))
    float m = -INFINITY;
    float s = 0.0f;

    // Zero output using SIMD
    int d = 0;
    for (; d + 16 <= aligned_head_dim; d += 16) {
        _mm512_storeu_ps(&out_vec[d], _mm512_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        // Vectorized dot product Q·K
        __m512 dot_acc = _mm512_setzero_ps();
        d = 0;
        for (; d + 16 <= head_dim; d += 16) {
            __m512 q_v = _mm512_loadu_ps(&q_vec[d]);
            __m512 k_v = _mm512_loadu_ps(&k_vec[d]);
            dot_acc = _mm512_fmadd_ps(q_v, k_v, dot_acc);
        }
        float dot = _mm512_reduce_add_ps(dot_acc);
        // Scalar tail
        for (; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;

            // Vectorized: out *= exp_m, then out += v
            __m512 exp_m_vec = _mm512_set1_ps(exp_m);
            d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 out_v = _mm512_loadu_ps(&out_vec[d]);
                __m512 v_v = _mm512_loadu_ps(&v_vec[d]);
                out_v = _mm512_fmadd_ps(out_v, exp_m_vec, v_v);
                _mm512_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_m + v_vec[d];
            }

            s += 1.0f;
            m = score;
        } else {
            float e = expf(score - m);
            s += e;

            // Vectorized: out += e * v
            __m512 e_vec = _mm512_set1_ps(e);
            d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 out_v = _mm512_loadu_ps(&out_vec[d]);
                __m512 v_v = _mm512_loadu_ps(&v_vec[d]);
                out_v = _mm512_fmadd_ps(e_vec, v_v, out_v);
                _mm512_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    // Normalize: out /= s
    float inv_s = 1.0f / s;
    __m512 inv_s_vec = _mm512_set1_ps(inv_s);
    d = 0;
    for (; d + 16 <= head_dim; d += 16) {
        __m512 out_v = _mm512_loadu_ps(&out_vec[d]);
        _mm512_storeu_ps(&out_vec[d], _mm512_mul_ps(out_v, inv_s_vec));
    }
    for (; d < head_dim; ++d) {
        out_vec[d] *= inv_s;
    }

    // Zero padding
    for (d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}
#endif // __AVX512F__

// ============================================================================
// AVX2 SIMD Flash Attention (8 floats per vector)
// ============================================================================
#if defined(__AVX2__)
static inline float hsum256_ps_flash(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static void attention_flash_query_causal_avx2(const float *q_vec,
                                               const float *k_head,
                                               const float *v_head,
                                               int kv_tokens,
                                               int head_dim,
                                               int aligned_head_dim,
                                               float scale,
                                               float *out_vec)
{
    float m = -INFINITY;
    float s = 0.0f;

    // Zero output using SIMD
    int d = 0;
    for (; d + 8 <= aligned_head_dim; d += 8) {
        _mm256_storeu_ps(&out_vec[d], _mm256_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        // Vectorized dot product Q·K
        __m256 dot_acc = _mm256_setzero_ps();
        d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 q_v = _mm256_loadu_ps(&q_vec[d]);
            __m256 k_v = _mm256_loadu_ps(&k_vec[d]);
            dot_acc = _mm256_fmadd_ps(q_v, k_v, dot_acc);
        }
        float dot = hsum256_ps_flash(dot_acc);
        for (; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;

            __m256 exp_m_vec = _mm256_set1_ps(exp_m);
            d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
                __m256 v_v = _mm256_loadu_ps(&v_vec[d]);
                out_v = _mm256_fmadd_ps(out_v, exp_m_vec, v_v);
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_m + v_vec[d];
            }

            s += 1.0f;
            m = score;
        } else {
            float e = expf(score - m);
            s += e;

            __m256 e_vec = _mm256_set1_ps(e);
            d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
                __m256 v_v = _mm256_loadu_ps(&v_vec[d]);
                out_v = _mm256_fmadd_ps(e_vec, v_v, out_v);
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    // Normalize
    float inv_s = 1.0f / s;
    __m256 inv_s_vec = _mm256_set1_ps(inv_s);
    d = 0;
    for (; d + 8 <= head_dim; d += 8) {
        __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
        _mm256_storeu_ps(&out_vec[d], _mm256_mul_ps(out_v, inv_s_vec));
    }
    for (; d < head_dim; ++d) {
        out_vec[d] *= inv_s;
    }

    for (d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}
#endif // __AVX2__

// ============================================================================
// AVX SIMD Flash Attention (8 floats per vector, no FMA)
// ============================================================================
#if defined(__AVX__) && !defined(__AVX2__)
static inline float hsum256_ps_flash_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static void attention_flash_query_causal_avx(const float *q_vec,
                                              const float *k_head,
                                              const float *v_head,
                                              int kv_tokens,
                                              int head_dim,
                                              int aligned_head_dim,
                                              float scale,
                                              float *out_vec)
{
    float m = -INFINITY;
    float s = 0.0f;

    // Zero output using SIMD
    int d = 0;
    for (; d + 8 <= aligned_head_dim; d += 8) {
        _mm256_storeu_ps(&out_vec[d], _mm256_setzero_ps());
    }
    for (; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        // Vectorized dot product Q·K (no FMA, use mul + add)
        __m256 dot_acc = _mm256_setzero_ps();
        d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 q_v = _mm256_loadu_ps(&q_vec[d]);
            __m256 k_v = _mm256_loadu_ps(&k_vec[d]);
            dot_acc = _mm256_add_ps(dot_acc, _mm256_mul_ps(q_v, k_v));
        }
        float dot = hsum256_ps_flash_avx(dot_acc);
        for (; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;

            __m256 exp_m_vec = _mm256_set1_ps(exp_m);
            d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
                __m256 v_v = _mm256_loadu_ps(&v_vec[d]);
                // out = out * exp_m + v (no FMA)
                out_v = _mm256_add_ps(_mm256_mul_ps(out_v, exp_m_vec), v_v);
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_m + v_vec[d];
            }

            s += 1.0f;
            m = score;
        } else {
            float e = expf(score - m);
            s += e;

            __m256 e_vec = _mm256_set1_ps(e);
            d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
                __m256 v_v = _mm256_loadu_ps(&v_vec[d]);
                // out = out + e * v (no FMA)
                out_v = _mm256_add_ps(out_v, _mm256_mul_ps(e_vec, v_v));
                _mm256_storeu_ps(&out_vec[d], out_v);
            }
            for (; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    // Normalize
    float inv_s = 1.0f / s;
    __m256 inv_s_vec = _mm256_set1_ps(inv_s);
    d = 0;
    for (; d + 8 <= head_dim; d += 8) {
        __m256 out_v = _mm256_loadu_ps(&out_vec[d]);
        _mm256_storeu_ps(&out_vec[d], _mm256_mul_ps(out_v, inv_s_vec));
    }
    for (; d < head_dim; ++d) {
        out_vec[d] *= inv_s;
    }

    for (d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}
#endif // __AVX__ && !__AVX2__

// ============================================================================
// Scalar fallback (original implementation)
// ============================================================================
static void attention_flash_query_causal(const float *q_vec,
                                        const float *k_head,
                                        const float *v_head,
                                        int kv_tokens,
                                        int head_dim,
                                        int aligned_head_dim,
                                        float scale,
                                        float *out_vec)
{
    // Online softmax:
    //   m = running max, s = running sum(exp(score - m))
    //   out = sum(exp(score - m) * v)
    float m = -INFINITY;
    float s = 0.0f;

    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;

        if (score > m) {
            float exp_m = (m == -INFINITY) ? 0.0f : expf(m - score);
            s *= exp_m;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] *= exp_m;
            }
            s += 1.0f;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] += v_vec[d];
            }
            m = score;
        } else {
            float e = expf(score - m);
            s += e;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] += e * v_vec[d];
            }
        }
    }

    float inv_s = 1.0f / s;
    for (int d = 0; d < head_dim; ++d) {
        out_vec[d] *= inv_s;
    }
    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}

// Strict parity reference for flash-style query path.
// Uses a two-pass exact softmax formulation per query:
// 1) max(score), 2) exp(score-max) accumulation for sum and weighted V.
// This avoids online-softmax re-normalization drift in long reductions.
static void attention_flash_query_causal_exact(const float *q_vec,
                                               const float *k_head,
                                               const float *v_head,
                                               int kv_tokens,
                                               int head_dim,
                                               int aligned_head_dim,
                                               float scale,
                                               float *out_vec)
{
    if (kv_tokens <= 0) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int d = 0; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    float max_score = -INFINITY;
    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;
        if (score > max_score) {
            max_score = score;
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        float score = dot * scale;
        float w = expf(score - max_score);
        sum += w;
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] += w * v_vec[d];
        }
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] *= inv_sum;
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }
    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}

static void ck_attention_vec_dump_exact_query(const float *q_vec,
                                              const float *k_head,
                                              const float *out_vec,
                                              int kv_tokens,
                                              int head_dim,
                                              int aligned_head_dim,
                                              float scale,
                                              int layer_id,
                                              int head_id,
                                              int query_id)
{
    if (!ck_attention_vec_dump_should_emit(layer_id, head_id, query_id)) {
        return;
    }

    float *raw_scores = (float *) alloca((size_t) kv_tokens * sizeof(float));
    float *probs = (float *) alloca((size_t) kv_tokens * sizeof(float));
    float max_score = -INFINITY;
    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t) j * (size_t) aligned_head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_vec[d] * k_vec[d];
        }
        raw_scores[j] = dot;
        const float scaled = dot * scale;
        probs[j] = scaled;
        if (scaled > max_score) {
            max_score = scaled;
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < kv_tokens; ++j) {
        probs[j] = expf(probs[j] - max_score);
        sum += probs[j];
    }
    if (sum > 0.0f) {
        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < kv_tokens; ++j) {
            probs[j] *= inv_sum;
        }
    } else {
        memset(probs, 0, (size_t) kv_tokens * sizeof(float));
    }

    ck_attention_vec_dump_selected_query(raw_scores, probs, out_vec, NULL,
                                         kv_tokens, head_dim,
                                         layer_id, head_id, query_id);
}

// Llama-parity attention reference: K/V are rounded through F16 before use.
static void attention_flash_query_causal_exact_f16kv(const float *q_vec,
                                                     const float *k_head,
                                                     const float *v_head,
                                                     int kv_tokens,
                                                     int head_dim,
                                                     int aligned_head_dim,
                                                     float scale,
                                                     float *out_vec)
{
    if (kv_tokens <= 0) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int d = 0; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }

    // Mirror llama.cpp GGML flash-attention more closely:
    // - Q is converted through FP16 before the KQ dot
    // - V accumulation is rounded through FP16 at each update
    // - the softmax accumulator uses the online max/sum form
    float sum = 0.0f;
    float max_score = -INFINITY;
    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += ck_round_fp16_scalar(q_vec[d]) * ck_round_fp16_scalar(k_vec[d]);
        }
        float score = dot * scale;

        const float prev_max = max_score;
        float max_scale = 1.0f;
        float value_scale = 1.0f;

        if (score > max_score) {
            max_score = score;
            max_scale = isfinite(prev_max) ? expf(prev_max - max_score) : 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] = ck_round_fp16_scalar(out_vec[d] * max_scale);
            }
        } else {
            value_scale = expf(score - max_score);
        }

        for (int d = 0; d < head_dim; ++d) {
            const float v_rounded = ck_round_fp16_scalar(v_vec[d]);
            const float updated = out_vec[d] + value_scale * v_rounded;
            out_vec[d] = ck_round_fp16_scalar(updated);
        }

        sum = sum * max_scale + value_scale;
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] *= inv_sum;
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }
    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}

static CK_NOINLINE CK_OPTNONE void attention_query_full_exact_regular(const float *q_vec,
                                                                      const float *k_head,
                                                                      const float *v_cols,
                                                                      int kv_tokens,
                                                                      int head_dim,
                                                                      int aligned_head_dim,
                                                                      float scale,
                                                                      float *score_row,
                                                                      float *out_vec,
                                                                      int layer_id,
                                                                      int head_id,
                                                                      int query_id)
{
    if (kv_tokens <= 0) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t) j * (size_t) aligned_head_dim;
        score_row[j] = ck_vec_dot_f32_strict(q_vec, k_vec, head_dim);
    }
    float *raw_dump = NULL;
    if (ck_attention_vec_dump_should_emit(layer_id, head_id, query_id)) {
        raw_dump = (float *) alloca((size_t) kv_tokens * sizeof(float));
        memcpy(raw_dump, score_row, (size_t) kv_tokens * sizeof(float));
    }

    float max_score = -INFINITY;
    for (int j = 0; j < kv_tokens; ++j) {
        const float score = score_row[j] * scale;
        score_row[j] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    double sum = 0.0;
    for (int j = 0; j < kv_tokens; ++j) {
        const float w = expf(score_row[j] - max_score);
        score_row[j] = w;
        volatile double next = sum + (double) w;
        sum = next;
    }

    if (sum > 0.0) {
        const float inv_sum = (float) (1.0 / sum);
        for (int j = 0; j < kv_tokens; ++j) {
            score_row[j] *= inv_sum;
        }
        for (int d = 0; d < head_dim; ++d) {
            const float *v_col = v_cols + (size_t) d * (size_t) kv_tokens;
            const float dot = ck_vec_dot_f32x_f32_to_f32_via_f64(score_row, v_col, kv_tokens);
            out_vec[d] = dot;
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }

    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
    if (raw_dump) {
        ck_attention_vec_dump_selected_query(raw_dump, score_row, out_vec, v_cols, kv_tokens, head_dim,
                                             layer_id, head_id, query_id);
    }
}

static CK_NOINLINE CK_OPTNONE void attention_query_full_ggml_regular(const float *q_vec,
                                                                     const float *k_head,
                                                                     const float *v_cols,
                                                                     int kv_tokens,
                                                                     int head_dim,
                                                                     int aligned_head_dim,
                                                                     float scale,
                                                                     float *score_row,
                                                                     float *out_vec,
                                                                     int layer_id,
                                                                     int head_id,
                                                                     int query_id)
{
    if (kv_tokens <= 0) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t) j * (size_t) aligned_head_dim;
        score_row[j] = ck_ggml_vec_dot_f32_contig(q_vec, k_vec, head_dim);
    }
    float *raw_dump = NULL;
    if (ck_attention_vec_dump_should_emit(layer_id, head_id, query_id)) {
        raw_dump = (float *) alloca((size_t) kv_tokens * sizeof(float));
        memcpy(raw_dump, score_row, (size_t) kv_tokens * sizeof(float));
    }

    float *logit_row = (float *) alloca((size_t) kv_tokens * sizeof(float));
    memcpy(logit_row, score_row, (size_t) kv_tokens * sizeof(float));
    ck_vec_scale_f32_inplace(logit_row, kv_tokens, scale);
    const float max_score = ck_vec_max_f32_contig(logit_row, kv_tokens);
    const double sum = ck_ggml_vec_soft_max_row(kv_tokens, score_row, logit_row, max_score);
    if (sum > 0.0) {
        const float inv_sum = (float) (1.0 / sum);
        ck_vec_scale_f32_inplace(score_row, kv_tokens, inv_sum);
        for (int d = 0; d < head_dim; ++d) {
            const float *v_col = v_cols + (size_t) d * (size_t) kv_tokens;
            out_vec[d] = ck_ggml_vec_dot_f32_contig(score_row, v_col, kv_tokens);
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }

    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
    if (raw_dump) {
        ck_attention_vec_dump_selected_query(raw_dump, score_row, out_vec, v_cols, kv_tokens, head_dim,
                                             layer_id, head_id, query_id);
    }
}

static CK_NOINLINE CK_OPTNONE void attention_query_full_ggml_regular_direct_v(const float *q_vec,
                                                                              const float *k_head,
                                                                              const float *v_head,
                                                                              int kv_tokens,
                                                                              int head_dim,
                                                                              int aligned_head_dim,
                                                                              float scale,
                                                                              float *score_row,
                                                                              float *out_vec)
{
    if (kv_tokens <= 0) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t) j * (size_t) aligned_head_dim;
        score_row[j] = ck_ggml_vec_dot_f32_contig(q_vec, k_vec, head_dim);
    }

    float *logit_row = (float *) alloca((size_t) kv_tokens * sizeof(float));
    memcpy(logit_row, score_row, (size_t) kv_tokens * sizeof(float));
    ck_vec_scale_f32_inplace(logit_row, kv_tokens, scale);
    const float max_score = ck_vec_max_f32_contig(logit_row, kv_tokens);
    const double sum = ck_ggml_vec_soft_max_row(kv_tokens, score_row, logit_row, max_score);
    if (sum > 0.0) {
        const float inv_sum = (float) (1.0 / sum);
        ck_vec_scale_f32_inplace(score_row, kv_tokens, inv_sum);
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        for (int j = 0; j < kv_tokens; ++j) {
            const float w = score_row[j];
            const float *v_vec = v_head + (size_t) j * (size_t) aligned_head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] += w * v_vec[d];
            }
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }

    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}

#if CK_ENABLE_LLAMA_CPP_PARITY
static CK_NOINLINE CK_OPTNONE void attention_query_full_dyn_ggml_regular(const float *q_vec,
                                                                         const float *k_head,
                                                                         const float *v_cols,
                                                                         int kv_tokens,
                                                                         int head_dim,
                                                                         int aligned_head_dim,
                                                                         float scale,
                                                                         float *score_row,
                                                                         float *prob_row,
                                                                         float *out_vec,
                                                                         ck_ggml_vec_dot_f32_fn dot_fn,
                                                                         ck_ggml_vec_soft_max_f32_fn softmax_fn,
                                                                         int layer_id,
                                                                         int head_id,
                                                                         int query_id)
{
    if (kv_tokens <= 0 || !dot_fn || !softmax_fn || !score_row || !prob_row) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t) j * (size_t) aligned_head_dim;
        float dot = 0.0f;
        dot_fn(head_dim, &dot, 0, q_vec, 0, k_vec, 0, 1);
        score_row[j] = dot;
    }
    float *raw_dump = NULL;
    if (ck_attention_vec_dump_should_emit(layer_id, head_id, query_id)) {
        raw_dump = (float *) alloca((size_t) kv_tokens * sizeof(float));
        memcpy(raw_dump, score_row, (size_t) kv_tokens * sizeof(float));
    }

    // Mirror ggml_compute_forward_soft_max_f32:
    // copy raw scores to scratch, scale the scratch buffer, compute the max from
    // that scaled buffer, then emit exp/logits into a separate output buffer.
    memcpy(prob_row, score_row, (size_t) kv_tokens * sizeof(float));
    ck_vec_scale_f32_inplace(prob_row, kv_tokens, scale);
    const float max_score = ck_vec_max_f32_contig(prob_row, kv_tokens);
    const double sum = softmax_fn(kv_tokens, score_row, prob_row, max_score);
    if (sum > 0.0) {
        const float inv_sum = (float) (1.0 / sum);
        ck_vec_scale_f32_inplace(score_row, kv_tokens, inv_sum);
        const int reverse_out_dot = ck_attention_reverse_out_dot_enabled();
        for (int d = 0; d < head_dim; ++d) {
            const float *v_col = v_cols + (size_t) d * (size_t) kv_tokens;
            if (reverse_out_dot) {
                out_vec[d] = ck_vec_dot_f32_reverse_strict(score_row, v_col, kv_tokens);
            } else {
                float dot = 0.0f;
                dot_fn(kv_tokens, &dot, 0, score_row, 0, v_col, 0, 1);
                out_vec[d] = dot;
            }
        }
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }

    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
    if (raw_dump) {
        ck_attention_vec_dump_selected_query(raw_dump, score_row, out_vec, v_cols, kv_tokens, head_dim,
                                             layer_id, head_id, query_id);
    }
}

static int attention_out_mul_mat_graph_block(const float *v_cols,
                                             const float *prob_block,
                                             int kv_tokens,
                                             int head_dim,
                                             int query_block,
                                             float *out_block)
{
    ck_ggml_cpu_init_fn ggml_cpu_init_fn = ck_resolve_ggml_cpu_init();
    ck_ggml_init_fn ggml_init_fn = ck_resolve_ggml_init();
    ck_ggml_free_fn ggml_free_fn = ck_resolve_ggml_free();
    ck_ggml_new_tensor_2d_fn ggml_new_tensor_2d_fn = ck_resolve_ggml_new_tensor_2d();
    ck_ggml_mul_mat_graph_fn ggml_mul_mat_fn = ck_resolve_ggml_mul_mat_graph();
    ck_ggml_new_graph_fn ggml_new_graph_fn = ck_resolve_ggml_new_graph();
    ck_ggml_build_forward_expand_fn ggml_build_forward_expand_fn = ck_resolve_ggml_build_forward_expand();
    ck_ggml_graph_compute_with_ctx_fn ggml_graph_compute_with_ctx_fn = ck_resolve_ggml_graph_compute_with_ctx();
    ck_ggml_set_input_fn ggml_set_input_fn = ck_resolve_ggml_set_input();

    if (!ggml_cpu_init_fn || !ggml_init_fn || !ggml_free_fn ||
        !ggml_new_tensor_2d_fn || !ggml_mul_mat_fn || !ggml_new_graph_fn ||
        !ggml_build_forward_expand_fn || !ggml_graph_compute_with_ctx_fn ||
        !ggml_set_input_fn || !v_cols || !prob_block || !out_block ||
        kv_tokens <= 0 || head_dim <= 0 || query_block <= 0) {
        return 0;
    }

    ggml_cpu_init_fn();

    const size_t out_bytes = (size_t) head_dim * (size_t) query_block * sizeof(float);
    const size_t mem_size = (size_t) 8 * 1024 * 1024 + out_bytes + (size_t) 512 * 1024;
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
    struct ggml_tensor *v_tensor = ggml_new_tensor_2d_fn(ctx, GGML_TYPE_F32, kv_tokens, head_dim);
    struct ggml_tensor *prob_tensor = ggml_new_tensor_2d_fn(ctx, GGML_TYPE_F32, kv_tokens, query_block);
    if (!v_tensor || !prob_tensor) {
        ggml_free_fn(ctx);
        return 0;
    }

    v_tensor->data = (void *) v_cols;
    prob_tensor->data = (void *) prob_block;
    ggml_set_input_fn(v_tensor);
    ggml_set_input_fn(prob_tensor);

    struct ggml_tensor *out_tensor = ggml_mul_mat_fn(ctx, v_tensor, prob_tensor);
    struct ggml_cgraph *gf = out_tensor ? ggml_new_graph_fn(ctx) : NULL;
    if (!out_tensor || !gf) {
        ggml_free_fn(ctx);
        return 0;
    }

    ggml_build_forward_expand_fn(gf, out_tensor);
    if (ggml_graph_compute_with_ctx_fn(ctx, gf, 1) == GGML_STATUS_SUCCESS) {
        memcpy(out_block, out_tensor->data, out_bytes);
        ok = 1;
    }

    ggml_free_fn(ctx);
    return ok;
}

static CK_NOINLINE CK_OPTNONE int attention_head_full_dyn_ggml_regular_graph_out(const float *q_head,
                                                                                  const float *k_head,
                                                                                  const float *v_cols,
                                                                                  int kv_tokens,
                                                                                  int head_dim,
                                                                                  int aligned_head_dim,
                                                                                  float scale,
                                                                                  float *score_row,
                                                                                  float *prob_row,
                                                                                  float *out_head,
                                                                                  ck_ggml_vec_dot_f32_fn dot_fn,
                                                                                  ck_ggml_vec_soft_max_f32_fn softmax_fn,
                                                                                  int layer_id,
                                                                                  int head_id)
{
    if (!q_head || !k_head || !v_cols || !out_head || !dot_fn || !softmax_fn ||
        kv_tokens <= 0 || head_dim <= 0 || aligned_head_dim < head_dim) {
        return 0;
    }

    enum { CK_STRICT_OUT_BLOCK = 64 };
    float *prob_block = (float *) alloca((size_t) CK_STRICT_OUT_BLOCK * (size_t) kv_tokens * sizeof(float));
    float *out_block = (float *) alloca((size_t) CK_STRICT_OUT_BLOCK * (size_t) head_dim * sizeof(float));
    float *raw_dump = (float *) alloca((size_t) kv_tokens * sizeof(float));
    for (int q0 = 0; q0 < kv_tokens; q0 += CK_STRICT_OUT_BLOCK) {
        const int qn = (q0 + CK_STRICT_OUT_BLOCK <= kv_tokens) ? CK_STRICT_OUT_BLOCK : (kv_tokens - q0);
        float *block_probs = prob_block;
        float *block_out = out_block;
        int have_raw_dump = 0;
        int dump_query = -1;
        int dump_qi = -1;

        for (int qi = 0; qi < qn; ++qi) {
            const int query_id = q0 + qi;
            const float *q_vec = q_head + (size_t) query_id * (size_t) aligned_head_dim;
            float *prob_col = block_probs + (size_t) qi * (size_t) kv_tokens;

            for (int j = 0; j < kv_tokens; ++j) {
                const float *k_vec = k_head + (size_t) j * (size_t) aligned_head_dim;
                float dot = 0.0f;
                dot_fn(head_dim, &dot, 0, q_vec, 0, k_vec, 0, 1);
                score_row[j] = dot;
            }

            if (ck_attention_vec_dump_should_emit(layer_id, head_id, query_id)) {
                memcpy(raw_dump, score_row, (size_t) kv_tokens * sizeof(float));
                have_raw_dump = 1;
                dump_query = query_id;
                dump_qi = qi;
            }

            memcpy(prob_row, score_row, (size_t) kv_tokens * sizeof(float));
            ck_vec_scale_f32_inplace(prob_row, kv_tokens, scale);
            const float max_score = ck_vec_max_f32_contig(prob_row, kv_tokens);
            const double sum = softmax_fn(kv_tokens, prob_col, prob_row, max_score);
            if (sum > 0.0) {
                const float inv_sum = (float) (1.0 / sum);
                ck_vec_scale_f32_inplace(prob_col, kv_tokens, inv_sum);
            } else {
                memset(prob_col, 0, (size_t) kv_tokens * sizeof(float));
            }
        }

        if (!attention_out_mul_mat_graph_block(v_cols, block_probs, kv_tokens, head_dim, qn, block_out)) {
            return 0;
        }

        for (int qi = 0; qi < qn; ++qi) {
            float *dst = out_head + (size_t) (q0 + qi) * (size_t) aligned_head_dim;
            const float *src = block_out + (size_t) qi * (size_t) head_dim;
            memcpy(dst, src, (size_t) head_dim * sizeof(float));
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                dst[d] = 0.0f;
            }
        }

        if (have_raw_dump && dump_qi >= 0) {
            ck_attention_vec_dump_selected_query(raw_dump,
                                                 block_probs + (size_t) dump_qi * (size_t) kv_tokens,
                                                 block_out + (size_t) dump_qi * (size_t) head_dim,
                                                 v_cols,
                                                 kv_tokens,
                                                 head_dim,
                                                 layer_id,
                                                 head_id,
                                                 dump_query);
        }
    }

    return 1;
}

static CK_NOINLINE CK_OPTNONE void attention_query_full_dyn_ggml_regular_matmul_out(const float *q_vec,
                                                                                     const float *k_head,
                                                                                     const float *v_cols,
                                                                                     int kv_tokens,
                                                                                     int head_dim,
                                                                                     int aligned_head_dim,
                                                                                     float scale,
                                                                                     float *score_row,
                                                                                     float *prob_row,
                                                                                     float *out_vec,
                                                                                     ck_ggml_vec_dot_f32_fn dot_fn,
                                                                                     ck_ggml_vec_soft_max_f32_fn softmax_fn,
                                                                                     ck_ggml_compute_forward_mul_mat_fn mul_mat_fn,
                                                                                     int layer_id,
                                                                                     int head_id,
                                                                                     int query_id)
{
    if (kv_tokens <= 0 || !dot_fn || !softmax_fn || !mul_mat_fn || !score_row || !prob_row) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t) j * (size_t) aligned_head_dim;
        float dot = 0.0f;
        dot_fn(head_dim, &dot, 0, q_vec, 0, k_vec, 0, 1);
        score_row[j] = dot;
    }
    float *raw_dump = NULL;
    if (ck_attention_vec_dump_should_emit(layer_id, head_id, query_id)) {
        raw_dump = (float *) alloca((size_t) kv_tokens * sizeof(float));
        memcpy(raw_dump, score_row, (size_t) kv_tokens * sizeof(float));
    }

    memcpy(prob_row, score_row, (size_t) kv_tokens * sizeof(float));
    ck_vec_scale_f32_inplace(prob_row, kv_tokens, scale);
    const float max_score = ck_vec_max_f32_contig(prob_row, kv_tokens);
    const double sum = softmax_fn(kv_tokens, score_row, prob_row, max_score);
    if (sum > 0.0) {
        const float inv_sum = (float) (1.0 / sum);
        ck_vec_scale_f32_inplace(score_row, kv_tokens, inv_sum);

        const size_t prob_row_bytes = (size_t) kv_tokens * sizeof(float);
        const size_t out_row_bytes = (size_t) head_dim * sizeof(float);

        struct ggml_tensor v_tensor;
        struct ggml_tensor prob_tensor;
        struct ggml_tensor out_tensor;

        ck_ggml_init_tensor_f32(&v_tensor,
                                kv_tokens, head_dim, 1, 1,
                                sizeof(float),
                                prob_row_bytes,
                                prob_row_bytes * (size_t) head_dim,
                                prob_row_bytes * (size_t) head_dim,
                                (void *) v_cols);
        ck_ggml_init_tensor_f32(&prob_tensor,
                                kv_tokens, 1, 1, 1,
                                sizeof(float),
                                prob_row_bytes,
                                prob_row_bytes,
                                prob_row_bytes,
                                score_row);
        ck_ggml_init_tensor_f32(&out_tensor,
                                head_dim, 1, 1, 1,
                                sizeof(float),
                                out_row_bytes,
                                out_row_bytes,
                                out_row_bytes,
                                out_vec);
        out_tensor.src[0] = &v_tensor;
        out_tensor.src[1] = &prob_tensor;

        memset(out_vec, 0, (size_t) head_dim * sizeof(float));
        struct ggml_compute_params mul_params = {
            .ith = 0,
            .nth = 1,
            .wsize = 0,
            .wdata = NULL,
            .threadpool = NULL,
            .use_ref = false,
        };
        mul_mat_fn(&mul_params, &out_tensor);
    } else {
        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
    }

    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
    if (raw_dump) {
        ck_attention_vec_dump_selected_query(raw_dump, score_row, out_vec, v_cols, kv_tokens, head_dim,
                                             layer_id, head_id, query_id);
    }
}

static CK_NOINLINE CK_OPTNONE void attention_query_full_ggml_compute_regular(const float *q_vec,
                                                                             const float *k_head,
                                                                             const float *v_cols,
                                                                             int kv_tokens,
                                                                             int head_dim,
                                                                             int aligned_head_dim,
                                                                             float scale,
                                                                             float *score_row,
                                                                             float *prob_row,
                                                                             float *out_vec,
                                                                             ck_ggml_vec_dot_f32_fn dot_fn,
                                                                             ck_ggml_compute_forward_mul_mat_fn mul_mat_fn,
                                                                             ck_ggml_compute_forward_soft_max_fn softmax_compute_fn)
{
    if (kv_tokens <= 0 || !dot_fn || !mul_mat_fn || !softmax_compute_fn) {
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_vec[d] = 0.0f;
        }
        return;
    }

    const size_t k_row_bytes = (size_t) aligned_head_dim * sizeof(float);
    const size_t q_row_bytes = (size_t) head_dim * sizeof(float);
    const size_t score_row_bytes = (size_t) kv_tokens * sizeof(float);
    const size_t softmax_work_elems = (size_t) kv_tokens + 16u;
    float *softmax_work = (float *) alloca(softmax_work_elems * sizeof(float));

    struct ggml_tensor k_tensor;
    struct ggml_tensor q_tensor;
    struct ggml_tensor score_tensor;
    struct ggml_tensor soft_tensor;

    ck_ggml_init_tensor_f32(&k_tensor,
                            head_dim, kv_tokens, 1, 1,
                            sizeof(float),
                            k_row_bytes,
                            k_row_bytes * (size_t) kv_tokens,
                            k_row_bytes * (size_t) kv_tokens,
                            (void *) k_head);
    ck_ggml_init_tensor_f32(&q_tensor,
                            head_dim, 1, 1, 1,
                            sizeof(float),
                            q_row_bytes,
                            q_row_bytes,
                            q_row_bytes,
                            (void *) q_vec);
    ck_ggml_init_tensor_f32(&score_tensor,
                            kv_tokens, 1, 1, 1,
                            sizeof(float),
                            score_row_bytes,
                            score_row_bytes,
                            score_row_bytes,
                            score_row);
    score_tensor.src[0] = &k_tensor;
    score_tensor.src[1] = &q_tensor;

    struct ggml_compute_params mul_params = {
        .ith = 0,
        .nth = 1,
        .wsize = 0,
        .wdata = NULL,
        .threadpool = NULL,
        .use_ref = false,
    };
    mul_mat_fn(&mul_params, &score_tensor);

    ck_ggml_init_tensor_f32(&soft_tensor,
                            kv_tokens, 1, 1, 1,
                            sizeof(float),
                            score_row_bytes,
                            score_row_bytes,
                            score_row_bytes,
                            prob_row);
    soft_tensor.src[0] = &score_tensor;
    {
        const float max_bias = 0.0f;
        memcpy((char *) soft_tensor.op_params + 0, &scale, sizeof(float));
        memcpy((char *) soft_tensor.op_params + sizeof(float), &max_bias, sizeof(float));
    }

    struct ggml_compute_params soft_params = {
        .ith = 0,
        .nth = 1,
        .wsize = softmax_work_elems * sizeof(float),
        .wdata = softmax_work,
        .threadpool = NULL,
        .use_ref = false,
    };
    softmax_compute_fn(&soft_params, &soft_tensor);

    for (int d = 0; d < head_dim; ++d) {
        const float *v_col = v_cols + (size_t) d * (size_t) kv_tokens;
        float dot = 0.0f;
        dot_fn(kv_tokens, &dot, 0, prob_row, 0, v_col, 0, 1);
        out_vec[d] = dot;
    }

    for (int d = head_dim; d < aligned_head_dim; ++d) {
        out_vec[d] = 0.0f;
    }
}
#endif

/* Strict ggml-backed full-attention oracles live in attention_oracle_ggml.c. */

#define CK_GGML_FA_TILE_Q 64
#define CK_GGML_FA_TILE_KV 64

static inline void ck_vec_scale_f32_inplace(float *x, int n, float scale)
{
    for (int i = 0; i < n; ++i) {
        x[i] *= scale;
    }
}

static inline float ck_vec_max_f32_contig(const float *x, int n)
{
    float max_val = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    return max_val;
}

#if defined(__AVX__) || defined(__AVX2__)
static inline void ck_attention_simd_gemm_ukernel_6x2(float *c,
                                                       const float *a,
                                                       const float *b,
                                                       int k,
                                                       int n)
{
    __m256 acc[6][2];
    for (int i = 0; i < 6; ++i) {
        acc[i][0] = _mm256_loadu_ps(c + (size_t) i * (size_t) n + 0);
        acc[i][1] = _mm256_loadu_ps(c + (size_t) i * (size_t) n + 8);
    }

    for (int kk = 0; kk < k; ++kk) {
        const __m256 bv0 = _mm256_loadu_ps(b + (size_t) kk * (size_t) n + 0);
        const __m256 bv1 = _mm256_loadu_ps(b + (size_t) kk * (size_t) n + 8);
        for (int i = 0; i < 6; ++i) {
            const __m256 p = _mm256_set1_ps(a[(size_t) i * (size_t) k + (size_t) kk]);
#if defined(__FMA__)
            acc[i][0] = _mm256_fmadd_ps(bv0, p, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(bv1, p, acc[i][1]);
#else
            acc[i][0] = _mm256_add_ps(_mm256_mul_ps(bv0, p), acc[i][0]);
            acc[i][1] = _mm256_add_ps(_mm256_mul_ps(bv1, p), acc[i][1]);
#endif
        }
    }

    for (int i = 0; i < 6; ++i) {
        _mm256_storeu_ps(c + (size_t) i * (size_t) n + 0, acc[i][0]);
        _mm256_storeu_ps(c + (size_t) i * (size_t) n + 8, acc[i][1]);
    }
}

static inline void ck_attention_simd_gemm_ukernel_6x1(float *c,
                                                       const float *a,
                                                       const float *b,
                                                       int k,
                                                       int n)
{
    __m256 acc[6];
    for (int i = 0; i < 6; ++i) {
        acc[i] = _mm256_loadu_ps(c + (size_t) i * (size_t) n);
    }

    for (int kk = 0; kk < k; ++kk) {
        const __m256 bv = _mm256_loadu_ps(b + (size_t) kk * (size_t) n);
        for (int i = 0; i < 6; ++i) {
            const __m256 p = _mm256_set1_ps(a[(size_t) i * (size_t) k + (size_t) kk]);
#if defined(__FMA__)
            acc[i] = _mm256_fmadd_ps(bv, p, acc[i]);
#else
            acc[i] = _mm256_add_ps(_mm256_mul_ps(bv, p), acc[i]);
#endif
        }
    }

    for (int i = 0; i < 6; ++i) {
        _mm256_storeu_ps(c + (size_t) i * (size_t) n, acc[i]);
    }
}

static inline void ck_attention_simd_gemm_ukernel_1x2(float *c,
                                                       const float *a,
                                                       const float *b,
                                                       int k,
                                                       int n)
{
    __m256 acc0 = _mm256_loadu_ps(c + 0);
    __m256 acc1 = _mm256_loadu_ps(c + 8);
    for (int kk = 0; kk < k; ++kk) {
        const __m256 bv0 = _mm256_loadu_ps(b + (size_t) kk * (size_t) n + 0);
        const __m256 bv1 = _mm256_loadu_ps(b + (size_t) kk * (size_t) n + 8);
        const __m256 p = _mm256_set1_ps(a[kk]);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(bv0, p, acc0);
        acc1 = _mm256_fmadd_ps(bv1, p, acc1);
#else
        acc0 = _mm256_add_ps(_mm256_mul_ps(bv0, p), acc0);
        acc1 = _mm256_add_ps(_mm256_mul_ps(bv1, p), acc1);
#endif
    }
    _mm256_storeu_ps(c + 0, acc0);
    _mm256_storeu_ps(c + 8, acc1);
}

static inline void ck_attention_simd_gemm_ukernel_1x1(float *c,
                                                       const float *a,
                                                       const float *b,
                                                       int k,
                                                       int n)
{
    __m256 acc = _mm256_loadu_ps(c);
    for (int kk = 0; kk < k; ++kk) {
        const __m256 bv = _mm256_loadu_ps(b + (size_t) kk * (size_t) n);
        const __m256 p = _mm256_set1_ps(a[kk]);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(bv, p, acc);
#else
        acc = _mm256_add_ps(_mm256_mul_ps(bv, p), acc);
#endif
    }
    _mm256_storeu_ps(c, acc);
}

static CK_NOINLINE CK_OPTNONE void ck_attention_matmul_f32_accum(float *c,
                                                                 const float *a,
                                                                 const float *b,
                                                                 int m,
                                                                 int k,
                                                                 int n)
{
    int ii = 0;
    for (; ii + 6 <= m; ii += 6) {
        int jj = 0;
        for (; jj + 16 <= n; jj += 16) {
            ck_attention_simd_gemm_ukernel_6x2(c + jj, a, b + jj, k, n);
        }
        for (; jj + 8 <= n; jj += 8) {
            ck_attention_simd_gemm_ukernel_6x1(c + jj, a, b + jj, k, n);
        }
        for (; jj < n; ++jj) {
            for (int i = 0; i < 6; ++i) {
                float sum = c[(size_t) i * (size_t) n + (size_t) jj];
                for (int kk = 0; kk < k; ++kk) {
                    sum += a[(size_t) i * (size_t) k + (size_t) kk] * b[(size_t) kk * (size_t) n + (size_t) jj];
                }
                c[(size_t) i * (size_t) n + (size_t) jj] = sum;
            }
        }
        a += (size_t) 6 * (size_t) k;
        c += (size_t) 6 * (size_t) n;
    }

    for (; ii < m; ++ii) {
        int jj = 0;
        for (; jj + 16 <= n; jj += 16) {
            ck_attention_simd_gemm_ukernel_1x2(c + jj, a, b + jj, k, n);
        }
        for (; jj + 8 <= n; jj += 8) {
            ck_attention_simd_gemm_ukernel_1x1(c + jj, a, b + jj, k, n);
        }
        for (; jj < n; ++jj) {
            float sum = c[jj];
            for (int kk = 0; kk < k; ++kk) {
                sum += a[kk] * b[(size_t) kk * (size_t) n + (size_t) jj];
            }
            c[jj] = sum;
        }
        a += k;
        c += n;
    }
}
#elif defined(__SSE2__)
static inline void ck_attention_simd_gemm_ukernel_2x2(float *c,
                                                       const float *a,
                                                       const float *b,
                                                       int k,
                                                       int n)
{
    __m128 acc[2][2];
    for (int i = 0; i < 2; ++i) {
        acc[i][0] = _mm_loadu_ps(c + (size_t) i * (size_t) n + 0);
        acc[i][1] = _mm_loadu_ps(c + (size_t) i * (size_t) n + 4);
    }

    for (int kk = 0; kk < k; ++kk) {
        const __m128 bv0 = _mm_loadu_ps(b + (size_t) kk * (size_t) n + 0);
        const __m128 bv1 = _mm_loadu_ps(b + (size_t) kk * (size_t) n + 4);
        for (int i = 0; i < 2; ++i) {
            const __m128 p = _mm_set1_ps(a[(size_t) i * (size_t) k + (size_t) kk]);
#if defined(__FMA__)
            acc[i][0] = _mm_fmadd_ps(bv0, p, acc[i][0]);
            acc[i][1] = _mm_fmadd_ps(bv1, p, acc[i][1]);
#else
            acc[i][0] = _mm_add_ps(_mm_mul_ps(bv0, p), acc[i][0]);
            acc[i][1] = _mm_add_ps(_mm_mul_ps(bv1, p), acc[i][1]);
#endif
        }
    }

    for (int i = 0; i < 2; ++i) {
        _mm_storeu_ps(c + (size_t) i * (size_t) n + 0, acc[i][0]);
        _mm_storeu_ps(c + (size_t) i * (size_t) n + 4, acc[i][1]);
    }
}

static inline void ck_attention_simd_gemm_ukernel_2x1(float *c,
                                                       const float *a,
                                                       const float *b,
                                                       int k,
                                                       int n)
{
    __m128 acc[2];
    for (int i = 0; i < 2; ++i) {
        acc[i] = _mm_loadu_ps(c + (size_t) i * (size_t) n);
    }

    for (int kk = 0; kk < k; ++kk) {
        const __m128 bv = _mm_loadu_ps(b + (size_t) kk * (size_t) n);
        for (int i = 0; i < 2; ++i) {
            const __m128 p = _mm_set1_ps(a[(size_t) i * (size_t) k + (size_t) kk]);
#if defined(__FMA__)
            acc[i] = _mm_fmadd_ps(bv, p, acc[i]);
#else
            acc[i] = _mm_add_ps(_mm_mul_ps(bv, p), acc[i]);
#endif
        }
    }

    for (int i = 0; i < 2; ++i) {
        _mm_storeu_ps(c + (size_t) i * (size_t) n, acc[i]);
    }
}

static inline void ck_attention_simd_gemm_ukernel_1x2(float *c,
                                                       const float *a,
                                                       const float *b,
                                                       int k,
                                                       int n)
{
    __m128 acc0 = _mm_loadu_ps(c + 0);
    __m128 acc1 = _mm_loadu_ps(c + 4);
    for (int kk = 0; kk < k; ++kk) {
        const __m128 bv0 = _mm_loadu_ps(b + (size_t) kk * (size_t) n + 0);
        const __m128 bv1 = _mm_loadu_ps(b + (size_t) kk * (size_t) n + 4);
        const __m128 p = _mm_set1_ps(a[kk]);
#if defined(__FMA__)
        acc0 = _mm_fmadd_ps(bv0, p, acc0);
        acc1 = _mm_fmadd_ps(bv1, p, acc1);
#else
        acc0 = _mm_add_ps(_mm_mul_ps(bv0, p), acc0);
        acc1 = _mm_add_ps(_mm_mul_ps(bv1, p), acc1);
#endif
    }
    _mm_storeu_ps(c + 0, acc0);
    _mm_storeu_ps(c + 4, acc1);
}

static inline void ck_attention_simd_gemm_ukernel_1x1(float *c,
                                                       const float *a,
                                                       const float *b,
                                                       int k,
                                                       int n)
{
    __m128 acc = _mm_loadu_ps(c);
    for (int kk = 0; kk < k; ++kk) {
        const __m128 bv = _mm_loadu_ps(b + (size_t) kk * (size_t) n);
        const __m128 p = _mm_set1_ps(a[kk]);
#if defined(__FMA__)
        acc = _mm_fmadd_ps(bv, p, acc);
#else
        acc = _mm_add_ps(_mm_mul_ps(bv, p), acc);
#endif
    }
    _mm_storeu_ps(c, acc);
}

static CK_NOINLINE CK_OPTNONE void ck_attention_matmul_f32_accum(float *c,
                                                                 const float *a,
                                                                 const float *b,
                                                                 int m,
                                                                 int k,
                                                                 int n)
{
    int ii = 0;
    for (; ii + 2 <= m; ii += 2) {
        int jj = 0;
        for (; jj + 8 <= n; jj += 8) {
            ck_attention_simd_gemm_ukernel_2x2(c + jj, a, b + jj, k, n);
        }
        for (; jj + 4 <= n; jj += 4) {
            ck_attention_simd_gemm_ukernel_2x1(c + jj, a, b + jj, k, n);
        }
        for (; jj < n; ++jj) {
            for (int i = 0; i < 2; ++i) {
                float sum = c[(size_t) i * (size_t) n + (size_t) jj];
                for (int kk = 0; kk < k; ++kk) {
                    sum += a[(size_t) i * (size_t) k + (size_t) kk] * b[(size_t) kk * (size_t) n + (size_t) jj];
                }
                c[(size_t) i * (size_t) n + (size_t) jj] = sum;
            }
        }
        a += (size_t) 2 * (size_t) k;
        c += (size_t) 2 * (size_t) n;
    }

    for (; ii < m; ++ii) {
        int jj = 0;
        for (; jj + 8 <= n; jj += 8) {
            ck_attention_simd_gemm_ukernel_1x2(c + jj, a, b + jj, k, n);
        }
        for (; jj + 4 <= n; jj += 4) {
            ck_attention_simd_gemm_ukernel_1x1(c + jj, a, b + jj, k, n);
        }
        for (; jj < n; ++jj) {
            float sum = c[jj];
            for (int kk = 0; kk < k; ++kk) {
                sum += a[kk] * b[(size_t) kk * (size_t) n + (size_t) jj];
            }
            c[jj] = sum;
        }
        a += k;
        c += n;
    }
}
#else
static CK_NOINLINE CK_OPTNONE void ck_attention_matmul_f32_accum(float *c,
                                                                 const float *a,
                                                                 const float *b,
                                                                 int m,
                                                                 int k,
                                                                 int n)
{
    for (int i = 0; i < m; ++i) {
        float *c_row = c + (size_t) i * (size_t) n;
        const float *a_row = a + (size_t) i * (size_t) k;
        for (int kk = 0; kk < k; ++kk) {
            const float a_ik = a_row[kk];
            const float *b_row = b + (size_t) kk * (size_t) n;
            for (int j = 0; j < n; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
#endif

static CK_NOINLINE CK_OPTNONE void ck_attention_full_tiled_f16kv_fp32_range(
    const float *q,
    const float *k,
    const float *v,
    float *output,
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int kv_stride_tokens,
    int ith,
    int nth)
{
    if (!q || !k || !v || !output || num_heads <= 0 || num_kv_heads <= 0 ||
        num_tokens <= 0 || head_dim <= 0 || aligned_head_dim < head_dim ||
        kv_stride_tokens < num_tokens || num_heads % num_kv_heads != 0) {
        return;
    }

    const float scale = ck_attention_strict_scale_f32(head_dim);
    const int T = num_tokens;
    const size_t kv_head_stride = (size_t) kv_stride_tokens * (size_t) aligned_head_dim;

    float *q_tile = (float *) alloca((size_t) CK_GGML_FA_TILE_Q * (size_t) head_dim * sizeof(float));
    float *k_tile = (float *) alloca((size_t) head_dim * (size_t) CK_GGML_FA_TILE_KV * sizeof(float));
    float *v_tile = (float *) alloca((size_t) CK_GGML_FA_TILE_KV * (size_t) head_dim * sizeof(float));
    float *kq = (float *) alloca((size_t) CK_GGML_FA_TILE_Q * (size_t) CK_GGML_FA_TILE_KV * sizeof(float));
    float *vkq = (float *) alloca((size_t) CK_GGML_FA_TILE_Q * (size_t) head_dim * sizeof(float));

    const int total_rows = num_heads * T;
    const int rows_per_worker = (total_rows + nth - 1) / nth;
    int ir = rows_per_worker * ith;
    const int ir1 = (ir + rows_per_worker) < total_rows
        ? (ir + rows_per_worker)
        : total_rows;

    while (ir < ir1) {
        const int h = ir / T;
        const int iq = ir - h * T;
        int tile_rows = ir1 - ir;
        if (tile_rows > CK_GGML_FA_TILE_Q) tile_rows = CK_GGML_FA_TILE_Q;
        if (tile_rows > T - iq) tile_rows = T - iq;
        const int kv_head = (int) ((long long) h * (long long) num_kv_heads / (long long) num_heads);
        const float *k_head = k + (size_t) kv_head * kv_head_stride;
        const float *v_head = v + (size_t) kv_head * kv_head_stride;

        float sum_row[CK_GGML_FA_TILE_Q];
        float max_row[CK_GGML_FA_TILE_Q];

        for (int tq = 0; tq < CK_GGML_FA_TILE_Q; ++tq) {
            sum_row[tq] = 0.0f;
            max_row[tq] = -INFINITY;
        }

        memset(vkq, 0, (size_t) CK_GGML_FA_TILE_Q * (size_t) head_dim * sizeof(float));
        memset(q_tile, 0, (size_t) CK_GGML_FA_TILE_Q * (size_t) head_dim * sizeof(float));
        memset(k_tile, 0, (size_t) head_dim * (size_t) CK_GGML_FA_TILE_KV * sizeof(float));
        memset(v_tile, 0, (size_t) CK_GGML_FA_TILE_KV * (size_t) head_dim * sizeof(float));

        for (int tq = 0; tq < tile_rows; ++tq) {
            const float *q_vec = q + qkv_index(h, iq + tq, 0, T, aligned_head_dim);
            memcpy(q_tile + (size_t) tq * (size_t) head_dim, q_vec, (size_t) head_dim * sizeof(float));
        }

        for (int ik = 0; ik < T; ik += CK_GGML_FA_TILE_KV) {
                const int kv_tile = (T - ik) < CK_GGML_FA_TILE_KV ? (T - ik) : CK_GGML_FA_TILE_KV;
                memset(kq, 0, (size_t) CK_GGML_FA_TILE_Q * (size_t) CK_GGML_FA_TILE_KV * sizeof(float));

                for (int tk = 0; tk < kv_tile; ++tk) {
                    const float *k_vec = k_head + (size_t) (ik + tk) * (size_t) aligned_head_dim;
                    const float *v_vec = v_head + (size_t) (ik + tk) * (size_t) aligned_head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        k_tile[(size_t) d * (size_t) CK_GGML_FA_TILE_KV + (size_t) tk] =
                            ck_round_fp16_scalar(k_vec[d]);
                        v_tile[(size_t) tk * (size_t) head_dim + (size_t) d] =
                            ck_round_fp16_scalar(v_vec[d]);
                    }
                }

                ck_attention_matmul_f32_accum(kq,
                                              q_tile,
                                              k_tile,
                                              CK_GGML_FA_TILE_Q,
                                              head_dim,
                                              CK_GGML_FA_TILE_KV);
                ck_vec_scale_f32_inplace(kq,
                                         CK_GGML_FA_TILE_Q * CK_GGML_FA_TILE_KV,
                                         scale);

                if (kv_tile < CK_GGML_FA_TILE_KV) {
                    for (int tq = 0; tq < CK_GGML_FA_TILE_Q; ++tq) {
                        float *kq_row = kq + (size_t) tq * (size_t) CK_GGML_FA_TILE_KV;
                        for (int tk = kv_tile; tk < CK_GGML_FA_TILE_KV; ++tk) {
                            kq_row[tk] = -INFINITY;
                        }
                    }
                }

                for (int tq = 0; tq < tile_rows; ++tq) {
                    float *kq_row = kq + (size_t) tq * (size_t) CK_GGML_FA_TILE_KV;
                    const float tile_max = ck_vec_max_f32_contig(kq_row, CK_GGML_FA_TILE_KV);
                    if (tile_max == -INFINITY) {
                        memset(kq_row, 0, (size_t) CK_GGML_FA_TILE_KV * sizeof(float));
                        continue;
                    }

                    const float old_max = max_row[tq];
                    const float new_max = old_max > tile_max ? old_max : tile_max;
                    if (new_max > old_max) {
                        const float ms = ck_attention_reference_expf(old_max - new_max);
                        ck_vec_scale_f32_inplace(vkq + (size_t) tq * (size_t) head_dim,
                                                 head_dim,
                                                 ms);
                        sum_row[tq] *= ms;
                    }
                    max_row[tq] = new_max;
                    sum_row[tq] = (float) (
                        (double) sum_row[tq] +
                        ck_ggml_vec_soft_max_row(
                            CK_GGML_FA_TILE_KV, kq_row, kq_row, new_max));
                }

                ck_attention_matmul_f32_accum(vkq,
                                              kq,
                                              v_tile,
                                              CK_GGML_FA_TILE_Q,
                                              CK_GGML_FA_TILE_KV,
                                              head_dim);
            }

        for (int tq = 0; tq < tile_rows; ++tq) {
            float *out_vec = output + qkv_index(h, iq + tq, 0, T, aligned_head_dim);
            const float inv_sum = sum_row[tq] == 0.0f ? 0.0f : (1.0f / sum_row[tq]);
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] = vkq[(size_t) tq * (size_t) head_dim + (size_t) d] * inv_sum;
            }
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                out_vec[d] = 0.0f;
            }
        }
        ir += tile_rows;
    }
}

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    float *output;
    int num_heads;
    int num_kv_heads;
    int num_tokens;
    int head_dim;
    int aligned_head_dim;
    int kv_stride_tokens;
} ck_attention_full_tiled_f16kv_fp32_args_t;

static void ck_attention_full_tiled_f16kv_fp32_work(int ith, int nth, void *opaque)
{
    ck_attention_full_tiled_f16kv_fp32_args_t *args =
        (ck_attention_full_tiled_f16kv_fp32_args_t *) opaque;
    ck_attention_full_tiled_f16kv_fp32_range(
        args->q, args->k, args->v, args->output,
        args->num_heads, args->num_kv_heads, args->num_tokens,
        args->head_dim, args->aligned_head_dim, args->kv_stride_tokens,
        ith, nth);
}

void attention_forward_full_head_major_gqa_tiled_f16kv_fp32_strided(
    const float *q,
    const float *k,
    const float *v,
    float *output,
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int kv_stride_tokens)
{
    if (!q || !k || !v || !output || num_heads <= 0 || num_kv_heads <= 0 ||
        num_tokens <= 0 || head_dim <= 0 || aligned_head_dim < head_dim ||
        kv_stride_tokens < num_tokens || num_heads % num_kv_heads != 0) {
        return;
    }

    ck_attention_full_tiled_f16kv_fp32_args_t args = {
        .q = q,
        .k = k,
        .v = v,
        .output = output,
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .num_tokens = num_tokens,
        .head_dim = head_dim,
        .aligned_head_dim = aligned_head_dim,
        .kv_stride_tokens = kv_stride_tokens,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    const int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (pool && active > 1) {
        ck_threadpool_dispatch_n(pool, active, ck_attention_full_tiled_f16kv_fp32_work, &args);
    } else {
        ck_attention_full_tiled_f16kv_fp32_work(0, 1, &args);
    }
}

/**
 * Flash attention forward for GQA (prefill, no score materialization)
 * @test test_flash_attention.py::TestFlashAttention::test_flash_forward
 * @test test_flash_attention.py::TestFlashAttention::test_flash_vs_score_matrix
 * @test test_flash_attention.py::TestFlashAttention::test_flash_gqa
 * @test test_attention.py::TestAttentionForward::test_flash_forward
 *
 * Online softmax with streaming KV. O(N) memory instead of O(N^2).
 * For prefill: all tokens attend to previous tokens.
 *
 * After changes: make test && make llamacpp-parity-full
 */

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    float *output;
    int num_heads;
    int num_kv_heads;
    int num_tokens;
    int head_dim;
    int aligned_head_dim;
    int kv_stride_tokens;
    int causal;
    float scale;
} ck_attention_parallel_args_t;

static inline void ck_attention_flash_query_auto(const float *q_vec,
                                                 const float *k_head,
                                                 const float *v_head,
                                                 int kv_tokens,
                                                 int head_dim,
                                                 int aligned_head_dim,
                                                 float scale,
                                                 float *out_vec)
{
#if defined(__AVX512F__)
    attention_flash_query_causal_avx512(q_vec, k_head, v_head,
                                        kv_tokens, head_dim, aligned_head_dim,
                                        scale, out_vec);
#elif defined(__AVX2__)
    attention_flash_query_causal_avx2(q_vec, k_head, v_head,
                                      kv_tokens, head_dim, aligned_head_dim,
                                      scale, out_vec);
#elif defined(__AVX__)
    attention_flash_query_causal_avx(q_vec, k_head, v_head,
                                     kv_tokens, head_dim, aligned_head_dim,
                                     scale, out_vec);
#else
    attention_flash_query_causal(q_vec, k_head, v_head,
                                 kv_tokens, head_dim, aligned_head_dim,
                                 scale, out_vec);
#endif
}


#if defined(__AVX512F__)
static int ck_attention_qblock4_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        cached = ck_env_truthy_or_qwen3vl_ocr_profile("CK_ATTENTION_QBLOCK4");
    }
    return cached;
}

static int ck_attention_qblock8_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        cached = ck_env_truthy_or_qwen3vl_ocr_profile("CK_ATTENTION_QBLOCK8");
    }
    return cached;
}

static int ck_attention_qblock_fast_exp_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        cached = ck_env_truthy_or_qwen3vl_ocr_profile("CK_ATTENTION_QBLOCK_FAST_EXP");
    }
    return cached;
}

static inline float ck_attention_qblock_fast_expf(float x)
{
    if (x > 88.0f) x = 88.0f;
    else if (x < -88.0f) x = -88.0f;

    const float log2e = 1.4426950408889634f;
    const float z = x * log2e;
    const float zf = nearbyintf(z);
    const float f = z - zf;

    const float c0 = 1.0f;
    const float c1 = 0.6931471805599453f;
    const float c2 = 0.2402265069591007f;
    const float c3 = 0.05550410866482158f;
    const float c4 = 0.009618129107628478f;

    float poly = ((c4 * f + c3) * f + c2) * f + c1;
    poly = poly * f + c0;

    union { uint32_t i; float f; } u;
    u.i = (uint32_t)((int32_t)zf + 127) << 23;
    return poly * u.f;
}

static inline float ck_attention_qblock_expf(float x)
{
    return ck_attention_qblock_fast_exp_enabled() ? ck_attention_qblock_fast_expf(x) : expf(x);
}

static inline float ck_attention_dot72_avx512(const float *q_vec, const float *k_vec)
{
    __m512 acc = _mm512_setzero_ps();
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(q_vec + 0),  _mm512_loadu_ps(k_vec + 0),  acc);
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(q_vec + 16), _mm512_loadu_ps(k_vec + 16), acc);
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(q_vec + 32), _mm512_loadu_ps(k_vec + 32), acc);
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(q_vec + 48), _mm512_loadu_ps(k_vec + 48), acc);
    float dot = _mm512_reduce_add_ps(acc);
    for (int d = 64; d < 72; ++d) {
        dot += q_vec[d] * k_vec[d];
    }
    return dot;
}

static void attention_flash_query4_full_avx512(const float *q_head,
                                                const float *k_head,
                                                const float *v_head,
                                                int q0,
                                                int q_count,
                                                int kv_tokens,
                                                int aligned_head_dim,
                                                float scale,
                                                float *out_head)
{
    float m[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float ssum[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float out[4][72];
    for (int r = 0; r < q_count; ++r) {
        for (int d = 0; d < 72; ++d) out[r][d] = 0.0f;
    }

    const float *qv[4] = { NULL, NULL, NULL, NULL };
    for (int r = 0; r < q_count; ++r) {
        qv[r] = q_head + (size_t)(q0 + r) * (size_t)aligned_head_dim;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;
        float score[4];
        float scale_old[4];
        float scale_v[4];

        for (int r = 0; r < q_count; ++r) {
            score[r] = ck_attention_dot72_avx512(qv[r], k_vec) * scale;
            if (score[r] > m[r]) {
                scale_old[r] = (m[r] == -INFINITY) ? 0.0f : ck_attention_qblock_expf(m[r] - score[r]);
                scale_v[r] = 1.0f;
                ssum[r] *= scale_old[r];
                ssum[r] += 1.0f;
                m[r] = score[r];
            } else {
                scale_old[r] = 1.0f;
                scale_v[r] = ck_attention_qblock_expf(score[r] - m[r]);
                ssum[r] += scale_v[r];
            }
        }

        for (int d = 0; d < 72; d += 16) {
            const int width = (d + 16 <= 72) ? 16 : (72 - d);
            if (width == 16) {
                const __m512 vv = _mm512_loadu_ps(v_vec + d);
                for (int r = 0; r < q_count; ++r) {
                    __m512 ov = _mm512_loadu_ps(out[r] + d);
                    ov = _mm512_fmadd_ps(ov, _mm512_set1_ps(scale_old[r]), _mm512_mul_ps(_mm512_set1_ps(scale_v[r]), vv));
                    _mm512_storeu_ps(out[r] + d, ov);
                }
            } else {
                for (int r = 0; r < q_count; ++r) {
                    for (int t = 0; t < width; ++t) {
                        out[r][d + t] = out[r][d + t] * scale_old[r] + scale_v[r] * v_vec[d + t];
                    }
                }
            }
        }
    }

    for (int r = 0; r < q_count; ++r) {
        float *dst = out_head + (size_t)(q0 + r) * (size_t)aligned_head_dim;
        const float inv = ssum[r] == 0.0f ? 0.0f : (1.0f / ssum[r]);
        const __m512 invv = _mm512_set1_ps(inv);
        for (int d = 0; d + 16 <= 72; d += 16) {
            _mm512_storeu_ps(dst + d, _mm512_mul_ps(_mm512_loadu_ps(out[r] + d), invv));
        }
        for (int d = 64; d < 72; ++d) dst[d] = out[r][d] * inv;
        for (int d = 72; d < aligned_head_dim; ++d) dst[d] = 0.0f;
    }
}
static void attention_flash_query8_full_avx512(const float *q_head,
                                                const float *k_head,
                                                const float *v_head,
                                                int q0,
                                                int q_count,
                                                int kv_tokens,
                                                int aligned_head_dim,
                                                float scale,
                                                float *out_head)
{
    float m[8] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float ssum[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float out[8][72];
    for (int r = 0; r < q_count; ++r) {
        for (int d = 0; d < 72; ++d) out[r][d] = 0.0f;
    }

    const float *qv[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    for (int r = 0; r < q_count; ++r) {
        qv[r] = q_head + (size_t)(q0 + r) * (size_t)aligned_head_dim;
    }

    for (int j = 0; j < kv_tokens; ++j) {
        const float *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
        const float *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;
        float score[8];
        float scale_old[8];
        float scale_v[8];

        for (int r = 0; r < q_count; ++r) {
            score[r] = ck_attention_dot72_avx512(qv[r], k_vec) * scale;
            if (score[r] > m[r]) {
                scale_old[r] = (m[r] == -INFINITY) ? 0.0f : ck_attention_qblock_expf(m[r] - score[r]);
                scale_v[r] = 1.0f;
                ssum[r] *= scale_old[r];
                ssum[r] += 1.0f;
                m[r] = score[r];
            } else {
                scale_old[r] = 1.0f;
                scale_v[r] = ck_attention_qblock_expf(score[r] - m[r]);
                ssum[r] += scale_v[r];
            }
        }

        for (int d = 0; d < 72; d += 16) {
            const int width = (d + 16 <= 72) ? 16 : (72 - d);
            if (width == 16) {
                const __m512 vv = _mm512_loadu_ps(v_vec + d);
                for (int r = 0; r < q_count; ++r) {
                    __m512 ov = _mm512_loadu_ps(out[r] + d);
                    ov = _mm512_fmadd_ps(ov, _mm512_set1_ps(scale_old[r]), _mm512_mul_ps(_mm512_set1_ps(scale_v[r]), vv));
                    _mm512_storeu_ps(out[r] + d, ov);
                }
            } else {
                for (int r = 0; r < q_count; ++r) {
                    for (int t = 0; t < width; ++t) {
                        out[r][d + t] = out[r][d + t] * scale_old[r] + scale_v[r] * v_vec[d + t];
                    }
                }
            }
        }
    }

    for (int r = 0; r < q_count; ++r) {
        float *dst = out_head + (size_t)(q0 + r) * (size_t)aligned_head_dim;
        const float inv = ssum[r] == 0.0f ? 0.0f : (1.0f / ssum[r]);
        const __m512 invv = _mm512_set1_ps(inv);
        for (int d = 0; d + 16 <= 72; d += 16) {
            _mm512_storeu_ps(dst + d, _mm512_mul_ps(_mm512_loadu_ps(out[r] + d), invv));
        }
        for (int d = 64; d < 72; ++d) dst[d] = out[r][d] * inv;
        for (int d = 72; d < aligned_head_dim; ++d) dst[d] = 0.0f;
    }
}

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    float *output;
    int num_heads;
    int num_kv_heads;
    int num_tokens;
    int aligned_head_dim;
    int kv_stride_tokens;
    float scale;
} ck_attention_qblock4_args_t;

static void ck_attention_full_qblock4_work(int ith, int nth, void *opaque)
{
    ck_attention_qblock4_args_t *args = (ck_attention_qblock4_args_t *) opaque;
    const int T = args->num_tokens;
    const int q_blocks = (T + 3) / 4;
    const int total = args->num_heads * q_blocks;
    const size_t head_stride = (size_t)T * (size_t)args->aligned_head_dim;
    const size_t kv_head_stride = (size_t)args->kv_stride_tokens * (size_t)args->aligned_head_dim;

    for (int idx = ith; idx < total; idx += nth) {
        const int h = idx / q_blocks;
        const int qb = idx - h * q_blocks;
        const int q0 = qb * 4;
        const int q_count = (q0 + 4 <= T) ? 4 : (T - q0);
        const int kv_head = (int)((long long)h * (long long)args->num_kv_heads / (long long)args->num_heads);
        const float *q_head = args->q + (size_t)h * head_stride;
        const float *k_head = args->k + (size_t)kv_head * kv_head_stride;
        const float *v_head = args->v + (size_t)kv_head * kv_head_stride;
        float *out_head = args->output + (size_t)h * head_stride;
        attention_flash_query4_full_avx512(q_head, k_head, v_head,
                                           q0, q_count, T,
                                           args->aligned_head_dim,
                                           args->scale,
                                           out_head);
    }
}

static void ck_attention_full_qblock8_work(int ith, int nth, void *opaque)
{
    ck_attention_qblock4_args_t *args = (ck_attention_qblock4_args_t *) opaque;
    const int T = args->num_tokens;
    const int q_blocks = (T + 7) / 8;
    const int total = args->num_heads * q_blocks;
    const size_t head_stride = (size_t)T * (size_t)args->aligned_head_dim;
    const size_t kv_head_stride = (size_t)args->kv_stride_tokens * (size_t)args->aligned_head_dim;

    for (int idx = ith; idx < total; idx += nth) {
        const int h = idx / q_blocks;
        const int qb = idx - h * q_blocks;
        const int q0 = qb * 8;
        const int q_count = (q0 + 8 <= T) ? 8 : (T - q0);
        const int kv_head = (int)((long long)h * (long long)args->num_kv_heads / (long long)args->num_heads);
        const float *q_head = args->q + (size_t)h * head_stride;
        const float *k_head = args->k + (size_t)kv_head * kv_head_stride;
        const float *v_head = args->v + (size_t)kv_head * kv_head_stride;
        float *out_head = args->output + (size_t)h * head_stride;
        attention_flash_query8_full_avx512(q_head, k_head, v_head,
                                           q0, q_count, T,
                                           args->aligned_head_dim,
                                           args->scale,
                                           out_head);
    }
}

#endif

static void ck_attention_full_grid_work(int ith, int nth, void *opaque)
{
    ck_attention_parallel_args_t *args = (ck_attention_parallel_args_t *) opaque;
    const int T = args->num_tokens;
    const int total = args->num_heads * T;
    const size_t kv_head_stride = (size_t) args->kv_stride_tokens * (size_t) args->aligned_head_dim;

    for (int idx = ith; idx < total; idx += nth) {
        const int h = idx / T;
        const int i = idx - h * T;
        const int kv_head = (int) ((long long) h * (long long) args->num_kv_heads / (long long) args->num_heads);
        const float *k_head = args->k + (size_t) kv_head * kv_head_stride;
        const float *v_head = args->v + (size_t) kv_head * kv_head_stride;
        const float *q_vec = args->q + qkv_index(h, i, 0, T, args->aligned_head_dim);
        float *out_vec = args->output + qkv_index(h, i, 0, T, args->aligned_head_dim);
        const int kv_tokens = args->causal ? (i + 1) : T;
        ck_attention_flash_query_auto(q_vec, k_head, v_head,
                                      kv_tokens,
                                      args->head_dim,
                                      args->aligned_head_dim,
                                      args->scale,
                                      out_vec);
    }
}

static int ck_attention_parallel_enabled(int total_queries, int num_tokens, int head_dim)
{
    const char *disable = getenv("CK_DISABLE_ATTENTION_THREADPOOL");
    if (disable && disable[0] && strcmp(disable, "0") != 0) return 0;
    if (total_queries < 2048 || num_tokens < 128 || head_dim <= 0) return 0;
    return 1;
}

static int ck_attention_pick_active_threads(const ck_threadpool_t *pool, int total_queries, int num_tokens)
{
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;
    if (nth <= 1) return 1;
    const char *cap_env = getenv("CK_ATTENTION_THREAD_CAP");
    int cap = cap_env && cap_env[0] ? atoi(cap_env) : 24;
    if (cap < 1) cap = 1;
    if (cap > nth) cap = nth;
    int active = (total_queries + 127) / 128;
    if (num_tokens >= 1024 && active < 8) active = 8;
    if (active > cap) active = cap;
    if (active > nth) active = nth;
    return active < 1 ? 1 : active;
}

static int ck_attention_strict_unfused_f16_enabled(void)
{
    const char *value = getenv("CK_STRICT_ATTN_F16_UNFUSED");
    return !value || !value[0] || strcmp(value, "0") != 0;
}

static int attention_forward_head_major_gqa_unfused_f16_strict(
    const float *q,
    const float *k,
    const float *v,
    float *output,
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int kv_stride_tokens,
    int causal,
    float scale,
    int debug_layer_id)
{
    const int T = num_tokens;
    const size_t kv_head_stride = (size_t) kv_stride_tokens * (size_t) aligned_head_dim;
    const size_t kv_half_count = (size_t) T * (size_t) aligned_head_dim;
    uint16_t *k_half = (uint16_t *) malloc(kv_half_count * sizeof(uint16_t));
    uint16_t *v_half = (uint16_t *) malloc(kv_half_count * sizeof(uint16_t));
    if (!k_half || !v_half) {
        free(k_half);
        free(v_half);
        return 0;
    }
    uint16_t *q_half = (uint16_t *) alloca((size_t) aligned_head_dim * sizeof(uint16_t));
    uint16_t *prob_half = (uint16_t *) alloca((size_t) T * sizeof(uint16_t));
    uint16_t *v_col_half = (uint16_t *) alloca((size_t) T * sizeof(uint16_t));
    float *raw_scores = (float *) alloca((size_t) T * sizeof(float));
    float *logits = (float *) alloca((size_t) T * sizeof(float));
    int cached_kv_head = -1;

    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = (int) ((long long) h * (long long) num_kv_heads /
                                   (long long) num_heads);
        const float *k_head = k + (size_t) kv_head * kv_head_stride;
        const float *v_head = v + (size_t) kv_head * kv_head_stride;
        if (kv_head != cached_kv_head) {
            for (size_t idx = 0; idx < kv_half_count; ++idx) {
                k_half[idx] = CK_FP32_TO_FP16(k_head[idx]);
                v_half[idx] = CK_FP32_TO_FP16(v_head[idx]);
            }
            cached_kv_head = kv_head;
        }

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            const int kv_tokens = causal ? i + 1 : T;
            for (int d = 0; d < aligned_head_dim; ++d) {
                q_half[d] = CK_FP32_TO_FP16(q_vec[d]);
            }
            for (int j = 0; j < kv_tokens; ++j) {
                raw_scores[j] = ck_attention_dot_f16_unfused_llama(
                    q_half,
                    k_half + (size_t) j * (size_t) aligned_head_dim,
                    head_dim);
                logits[j] = raw_scores[j] * scale;
            }

            const float max_score = ck_vec_max_f32_contig(logits, kv_tokens);
            const double sum = ck_ggml_vec_soft_max_row(kv_tokens, logits, logits, max_score);
            const float inv_sum = sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
            for (int j = 0; j < kv_tokens; ++j) {
                logits[j] *= inv_sum;
                prob_half[j] = CK_FP32_TO_FP16(logits[j]);
            }
            for (int d = 0; d < head_dim; ++d) {
                for (int j = 0; j < kv_tokens; ++j) {
                    v_col_half[j] = v_half[(size_t) j * (size_t) aligned_head_dim + (size_t) d];
                }
                out_vec[d] = ck_attention_dot_f16_unfused_llama(prob_half, v_col_half, kv_tokens);
            }
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                out_vec[d] = 0.0f;
            }
            ck_attention_vec_dump_selected_query(raw_scores, logits, out_vec, NULL,
                                                 kv_tokens, head_dim,
                                                 debug_layer_id, h, i);
        }
    }
    free(k_half);
    free(v_half);
    return 1;
}

static void attention_forward_head_major_gqa_flash_impl(const float *q,
                                                        const float *k,
                                                        const float *v,
                                                        float *output,
                                                        int num_heads,
                                                        int num_kv_heads,
                                                        int num_tokens,
                                                        int head_dim,
                                                        int aligned_head_dim,
                                                        int kv_stride_tokens,
                                                        int causal,
                                                        int round_full_kv_fp16,
                                                        float scale)
{
    if (!q || !k || !v || !output) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0 ||
        head_dim <= 0 || aligned_head_dim < head_dim) {
        return;
    }
    if (kv_stride_tokens < num_tokens) {
        return;
    }

    const int T = num_tokens;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;

    if (ck_strict_parity_enabled()) {
        const float strict_scale = ck_attention_strict_scale_f32(head_dim);
        const int debug_layer_id = ck_attention_vec_dump_enabled()
            ? ck_attention_vec_dump_next_layer_id()
            : -1;
        if (causal && ck_attention_strict_unfused_f16_enabled()) {
            if (attention_forward_head_major_gqa_unfused_f16_strict(
                    q, k, v, output,
                    num_heads, num_kv_heads, num_tokens,
                    head_dim, aligned_head_dim, kv_stride_tokens,
                    causal, strict_scale, debug_layer_id)) {
                return;
            }
        }
#if CK_ENABLE_LLAMA_CPP_PARITY
        if (!causal &&
            ck_attention_full_ggml_graph_oracle_multihead(q,
                                                          k,
                                                          v,
                                                          output,
                                                          num_heads,
                                                          num_kv_heads,
                                                          num_tokens,
                                                          head_dim,
                                                          aligned_head_dim,
                                                          kv_stride_tokens,
                                                          strict_scale)) {
            return;
        }
#endif
        float *score_row = (float *) alloca((size_t) T * sizeof(float));
        float *logit_row = (float *) alloca((size_t) T * sizeof(float));
        float *v_cols = (float *) alloca((size_t) head_dim * (size_t) T * sizeof(float));
        for (int h = 0; h < num_heads; ++h) {
            int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *k_head = k + (size_t)kv_head * kv_head_stride;
            const float *v_head = v + (size_t)kv_head * kv_head_stride;

            for (int d = 0; d < head_dim; ++d) {
                float *dst_col = v_cols + (size_t) d * (size_t) T;
                for (int j = 0; j < T; ++j) {
                    dst_col[j] = v_head[(size_t) j * (size_t) aligned_head_dim + (size_t) d];
                }
            }

            for (int i = 0; i < T; ++i) {
                const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
                float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
                const int kv_tokens = causal ? (i + 1) : T;
                if (kv_tokens <= 0) {
                    for (int d = 0; d < aligned_head_dim; ++d) {
                        out_vec[d] = 0.0f;
                    }
                    continue;
                }
                for (int j = 0; j < kv_tokens; ++j) {
                    const float *k_vec = k_head + (size_t) j * (size_t) aligned_head_dim;
                    score_row[j] = ck_ggml_vec_dot_f32_contig(q_vec, k_vec, head_dim);
                }
                float *raw_dump = NULL;
                if (ck_attention_vec_dump_should_emit(debug_layer_id, h, i)) {
                    raw_dump = (float *) alloca((size_t) kv_tokens * sizeof(float));
                    memcpy(raw_dump, score_row, (size_t) kv_tokens * sizeof(float));
                }
                memcpy(logit_row, score_row, (size_t) kv_tokens * sizeof(float));
                ck_vec_scale_f32_inplace(logit_row, kv_tokens, strict_scale);
                const float max_score = ck_vec_max_f32_contig(logit_row, kv_tokens);
                const double sum = ck_ggml_vec_soft_max_row(kv_tokens, score_row, logit_row, max_score);
                if (sum > 0.0) {
                    const float inv_sum = (float) (1.0 / sum);
                    ck_vec_scale_f32_inplace(score_row, kv_tokens, inv_sum);
                    for (int d = 0; d < head_dim; ++d) {
                        const float *v_col = v_cols + (size_t) d * (size_t) T;
                        out_vec[d] = ck_ggml_vec_dot_f32_contig(score_row, v_col, kv_tokens);
                    }
                } else {
                    for (int d = 0; d < head_dim; ++d) {
                        out_vec[d] = 0.0f;
                    }
                }
                for (int d = head_dim; d < aligned_head_dim; ++d) {
                    out_vec[d] = 0.0f;
                }
                if (raw_dump) {
                    ck_attention_vec_dump_selected_query(raw_dump, score_row, out_vec, v_cols,
                                                         kv_tokens, head_dim,
                                                         debug_layer_id, h, i);
                }
            }
        }
        return;
    }

    /*
     * Full GGML-style attention consumes FP32 Q with K/V rounded through
     * FP16.  Materialize that semantic input once so serial, threaded, and
     * ISA-specific implementations all see the same values.  Causal callers
     * retain their existing FP32 K/V contract.
     */
    float *rounded_kv = NULL;
    const float *compute_k = k;
    const float *compute_v = v;
    if (!causal && round_full_kv_fp16) {
        if ((size_t) kv_stride_tokens > SIZE_MAX / (size_t) num_kv_heads) {
            return;
        }
        const size_t rows = (size_t) num_kv_heads * (size_t) kv_stride_tokens;
        if ((size_t) aligned_head_dim > SIZE_MAX / rows) {
            return;
        }
        const size_t elements = rows * (size_t) aligned_head_dim;
        if (elements > SIZE_MAX / (2 * sizeof(float))) {
            return;
        }
        rounded_kv = (float *) malloc(2 * elements * sizeof(float));
        if (!rounded_kv) {
            return;
        }
        compute_k = rounded_kv;
        compute_v = rounded_kv + elements;
        ck_round_fp16_buffer(k, rounded_kv, elements);
        ck_round_fp16_buffer(v, rounded_kv + elements, elements);
    }

    // Select SIMD implementation based on compile-time CPU features
#if defined(__AVX512F__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx512
#elif defined(__AVX2__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx2
#elif defined(__AVX__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx
#else
    #define FLASH_QUERY_IMPL attention_flash_query_causal
#endif

    const int total_queries = num_heads * T;
#if defined(__AVX512F__)
    if (!causal && head_dim == 72 && aligned_head_dim >= 72 && ck_attention_qblock8_enabled()) {
        ck_threadpool_t *pool = ck_threadpool_global();
        const int q_blocks = (T + 7) / 8;
        const int total_blocks = num_heads * q_blocks;
        int active = ck_attention_pick_active_threads(pool, total_blocks, T);
        if (pool && active > 1) {
            ck_attention_qblock4_args_t args = {
                .q = q,
                .k = compute_k,
                .v = compute_v,
                .output = output,
                .num_heads = num_heads,
                .num_kv_heads = num_kv_heads,
                .num_tokens = num_tokens,
                .aligned_head_dim = aligned_head_dim,
                .kv_stride_tokens = kv_stride_tokens,
                .scale = scale,
            };
            ck_threadpool_dispatch_n(pool, active, ck_attention_full_qblock8_work, &args);
            free(rounded_kv);
            return;
        }
    }

    if (!causal && head_dim == 72 && aligned_head_dim >= 72 && ck_attention_qblock4_enabled()) {
        ck_threadpool_t *pool = ck_threadpool_global();
        const int q_blocks = (T + 3) / 4;
        const int total_blocks = num_heads * q_blocks;
        int active = ck_attention_pick_active_threads(pool, total_blocks, T);
        if (pool && active > 1) {
            ck_attention_qblock4_args_t args = {
                .q = q,
                .k = compute_k,
                .v = compute_v,
                .output = output,
                .num_heads = num_heads,
                .num_kv_heads = num_kv_heads,
                .num_tokens = num_tokens,
                .aligned_head_dim = aligned_head_dim,
                .kv_stride_tokens = kv_stride_tokens,
                .scale = scale,
            };
            ck_threadpool_dispatch_n(pool, active, ck_attention_full_qblock4_work, &args);
            free(rounded_kv);
            return;
        }
    }
#endif
    if (ck_attention_parallel_enabled(total_queries, T, head_dim)) {
        ck_threadpool_t *pool = ck_threadpool_global();
        const int active = ck_attention_pick_active_threads(pool, total_queries, T);
        if (pool && active > 1) {
            ck_attention_parallel_args_t args = {
                .q = q,
                .k = compute_k,
                .v = compute_v,
                .output = output,
                .num_heads = num_heads,
                .num_kv_heads = num_kv_heads,
                .num_tokens = num_tokens,
                .head_dim = head_dim,
                .aligned_head_dim = aligned_head_dim,
                .kv_stride_tokens = kv_stride_tokens,
                .causal = causal,
                .scale = scale,
            };
            ck_threadpool_dispatch_n(pool, active, ck_attention_full_grid_work, &args);
            free(rounded_kv);
            return;
        }
    }

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *k_head = compute_k + (size_t)kv_head * kv_head_stride;
        const float *v_head = compute_v + (size_t)kv_head * kv_head_stride;

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            const int kv_tokens = causal ? (i + 1) : T;
            FLASH_QUERY_IMPL(q_vec, k_head, v_head,
                             kv_tokens,
                             head_dim, aligned_head_dim,
                             scale, out_vec);
        }
    }

    free(rounded_kv);

#undef FLASH_QUERY_IMPL
}

void attention_forward_causal_head_major_gqa_flash(const float *q,
                                                   const float *k,
                                                   const float *v,
                                                   float *output,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int num_tokens,
                                                   int head_dim,
                                                   int aligned_head_dim)
{
    attention_forward_head_major_gqa_flash_impl(q, k, v, output,
                                                num_heads, num_kv_heads,
                                                num_tokens, head_dim,
                                                aligned_head_dim,
                                                /*kv_stride_tokens=*/num_tokens,
                                                /*causal=*/1,
                                                /*round_full_kv_fp16=*/0,
                                                1.0f / sqrtf((float)head_dim));
}

void attention_forward_full_head_major_gqa_flash(const float *q,
                                                 const float *k,
                                                 const float *v,
                                                 float *output,
                                                 int num_heads,
                                                 int num_kv_heads,
                                                 int num_tokens,
                                                 int head_dim,
                                                 int aligned_head_dim)
{
    attention_forward_head_major_gqa_flash_impl(q, k, v, output,
                                                num_heads, num_kv_heads,
                                                num_tokens, head_dim,
                                                aligned_head_dim,
                                                /*kv_stride_tokens=*/num_tokens,
                                                /*causal=*/0,
                                                /*round_full_kv_fp16=*/1,
                                                1.0f / sqrtf((float)head_dim));
}

/**
 * Flash attention forward with custom KV stride (for KV cache)
 * @test test_flash_attention.py::TestFlashAttention::test_flash_strided
 * @test test_kv_cache_attention.py::TestKVCacheAttention::test_flash_attention
 *
 * Variant with configurable kv_stride_tokens for KV cache layouts
 * where K/V may not be contiguous in memory.
 *
 * After changes: make test
 */
void attention_forward_causal_head_major_gqa_flash_strided(const float *q,
                                                           const float *k,
                                                           const float *v,
                                                           float *output,
                                                           int num_heads,
                                                           int num_kv_heads,
                                                           int num_tokens,
                                                           int head_dim,
                                                           int aligned_head_dim,
                                                           int kv_stride_tokens)
{
    attention_forward_head_major_gqa_flash_impl(q, k, v, output,
                                                num_heads, num_kv_heads,
                                                num_tokens, head_dim,
                                                aligned_head_dim,
                                                kv_stride_tokens,
                                                /*causal=*/1,
                                                /*round_full_kv_fp16=*/0,
                                                1.0f / sqrtf((float)head_dim));
}

void attention_forward_full_head_major_gqa_flash_strided(const float *q,
                                                         const float *k,
                                                         const float *v,
                                                         float *output,
                                                         int num_heads,
                                                         int num_kv_heads,
                                                         int num_tokens,
                                                         int head_dim,
                                                         int aligned_head_dim,
                                                         int kv_stride_tokens)
{
    attention_forward_head_major_gqa_flash_impl(q, k, v, output,
                                                num_heads, num_kv_heads,
                                                num_tokens, head_dim,
                                                aligned_head_dim,
                                                kv_stride_tokens,
                                                /*causal=*/0,
                                                /*round_full_kv_fp16=*/1,
                                                1.0f / sqrtf((float)head_dim));
}


static float ck_bf16_dot_contract(const float *a, const float *b, int count)
{
#if defined(__AVX512BF16__)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= count; i += 32) {
        const __m512bh av = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(a + i));
        const __m512bh bv = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(b + i + 16), _mm512_loadu_ps(b + i));
        acc = _mm512_dpbf16_ps(acc, av, bv);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < count; ++i) {
        const float av = bf16_to_float(float_to_bf16(a[i]));
        const float bv = bf16_to_float(float_to_bf16(b[i]));
        sum += av * bv;
    }
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float av = bf16_to_float(float_to_bf16(a[i]));
        const float bv = bf16_to_float(float_to_bf16(b[i]));
        sum += av * bv;
    }
    return sum;
#endif
}

static int ck_attention_full_bf16_sdpa_tiled_range(
    const float *q, const float *k, const float *v, float *output,
    int num_heads, int num_kv_heads, int num_tokens,
    int head_dim, int aligned_head_dim, int kv_stride_tokens,
    int head_begin, int head_step)
{
    if (!q || !k || !v || !output || num_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || num_tokens <= 0 || aligned_head_dim < head_dim ||
        kv_stride_tokens < num_tokens || num_heads % num_kv_heads != 0) return 0;
    if ((size_t)num_tokens > SIZE_MAX / (size_t)head_dim) return 0;
    const size_t v_col_count = (size_t)head_dim * (size_t)num_tokens;
    if (v_col_count > SIZE_MAX / sizeof(float)) return 0;
    if ((size_t)num_tokens > SIZE_MAX / (size_t)aligned_head_dim) return 0;
    if ((size_t)kv_stride_tokens > SIZE_MAX / (size_t)aligned_head_dim) return 0;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;
    if ((size_t)num_heads > SIZE_MAX / q_head_stride) return 0;
    if ((size_t)num_kv_heads > SIZE_MAX / kv_head_stride) return 0;
    float scores[512];
    float probs[512];
    float *v_cols = (float *)malloc(v_col_count * sizeof(float));
    if (!v_cols) return 0;
    for (int h = head_begin; h < num_heads; h += head_step) {
        const int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *qh = q + (size_t)h * q_head_stride;
        const float *kh = k + (size_t)kv_head * kv_head_stride;
        const float *vh = v + (size_t)kv_head * kv_head_stride;
        float *oh = output + (size_t)h * q_head_stride;
        for (int d = 0; d < head_dim; ++d) {
            float *col = v_cols + (size_t)d * (size_t)num_tokens;
            for (int t = 0; t < num_tokens; ++t) {
                col[t] = vh[(size_t)t * (size_t)aligned_head_dim + (size_t)d];
            }
        }
        for (int row = 0; row < num_tokens; ++row) {
            const float *qrow = qh + (size_t)row * (size_t)aligned_head_dim;
            float *dst = oh + (size_t)row * (size_t)aligned_head_dim;
            for (int d = 0; d < head_dim; ++d) dst[d] = 0.0f;
            float running_max = -INFINITY;
            float running_sum = 0.0f;
            for (int n = 0; n < num_tokens; n += 512) {
                const int block = num_tokens - n < 512 ? num_tokens - n : 512;
                float block_max = -INFINITY;
                for (int j = 0; j < block; ++j) {
                    scores[j] = ck_bf16_dot_contract(
                        qrow,
                        kh + (size_t)(n + j) * (size_t)aligned_head_dim,
                        head_dim) * scale;
                    if (scores[j] > block_max) block_max = scores[j];
                }
                const float merged_max = running_max > block_max ? running_max : block_max;
                const float old_scale = isfinite(running_max) ? expf(running_max - merged_max) : 0.0f;
                float block_sum = 0.0f;
                for (int j = 0; j < block; ++j) {
                    const float p = expf(scores[j] - merged_max);
                    block_sum += p;
                    probs[j] = bf16_to_float(float_to_bf16(p));
                }
                for (int d = 0; d < head_dim; ++d) {
                    const float partial = ck_bf16_dot_contract(
                        probs,
                        v_cols + (size_t)d * (size_t)num_tokens + (size_t)n,
                        block);
                    dst[d] = dst[d] * old_scale + partial;
                }
                running_sum = running_sum * old_scale + block_sum;
                running_max = merged_max;
            }
            const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
            for (int d = 0; d < head_dim; ++d) dst[d] *= inv;
            for (int d = head_dim; d < aligned_head_dim; ++d) dst[d] = 0.0f;
        }
    }
    free(v_cols);
    return 1;
}

typedef struct {
    const float *q;
    const float *k;
    const float *v;
    float *output;
    int num_heads;
    int num_kv_heads;
    int num_tokens;
    int head_dim;
    int aligned_head_dim;
    int kv_stride_tokens;
    int failed;
} ck_attention_bf16_sdpa_args_t;

static void ck_attention_bf16_sdpa_work(int ith, int nth, void *opaque)
{
    ck_attention_bf16_sdpa_args_t *args = (ck_attention_bf16_sdpa_args_t *)opaque;
    if (!ck_attention_full_bf16_sdpa_tiled_range(
            args->q, args->k, args->v, args->output,
            args->num_heads, args->num_kv_heads, args->num_tokens,
            args->head_dim, args->aligned_head_dim, args->kv_stride_tokens,
            ith, nth)) {
        __atomic_store_n(&args->failed, 1, __ATOMIC_RELAXED);
    }
}

static int ck_attention_full_bf16_sdpa_tiled(
    const float *q, const float *k, const float *v, float *output,
    int num_heads, int num_kv_heads, int num_tokens,
    int head_dim, int aligned_head_dim, int kv_stride_tokens)
{
    ck_attention_bf16_sdpa_args_t args = {
        .q=q, .k=k, .v=v, .output=output,
        .num_heads=num_heads, .num_kv_heads=num_kv_heads,
        .num_tokens=num_tokens, .head_dim=head_dim,
        .aligned_head_dim=aligned_head_dim, .kv_stride_tokens=kv_stride_tokens,
        .failed=0
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > num_heads) active = num_heads;
    if (pool && active > 1) {
        ck_threadpool_dispatch_n(pool, active, ck_attention_bf16_sdpa_work, &args);
    } else {
        ck_attention_bf16_sdpa_work(0, 1, &args);
    }
    return __atomic_load_n(&args.failed, __ATOMIC_RELAXED) == 0;
}

void attention_forward_full_head_major_gqa_flash_strided_bf16_storage(
    const float *q,
    const float *k,
    const float *v,
    float *output,
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int kv_stride_tokens)
{
    attention_forward_head_major_gqa_flash_impl(
        q, k, v, output,
        num_heads, num_kv_heads, num_tokens, head_dim,
        aligned_head_dim, kv_stride_tokens,
        /*causal=*/0,
        /*round_full_kv_fp16=*/0,
        1.0f / sqrtf((float)head_dim)
    );
    const size_t count = (size_t)num_heads * (size_t)num_tokens
                       * (size_t)aligned_head_dim;
    for (size_t i = 0; i < count; ++i) {
        output[i] = bf16_to_float(float_to_bf16(output[i]));
    }
}

void attention_forward_full_head_major_gqa_sdpa_bf16_storage(
    const float *q,
    const float *k,
    const float *v,
    float *output,
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int kv_stride_tokens)
{
    if (ck_attention_full_bf16_sdpa_tiled(
            q, k, v, output, num_heads, num_kv_heads, num_tokens,
            head_dim, aligned_head_dim, kv_stride_tokens)) {
        const size_t count = (size_t)num_heads * (size_t)num_tokens
                           * (size_t)aligned_head_dim;
        for (size_t i = 0; i < count; ++i) {
            output[i] = bf16_to_float(float_to_bf16(output[i]));
        }
        return;
    }
    fprintf(stderr, "CK numerical contract failure: BF16 tiled SDPA received invalid dimensions or could not allocate scratch\n");
}


void attention_forward_causal_head_major_gqa_flash_strided_gemma4(const float *q,
                                                                  const float *k,
                                                                  const float *v,
                                                                  float *output,
                                                                  int num_heads,
                                                                  int num_kv_heads,
                                                                  int num_tokens,
                                                                  int head_dim,
                                                                  int aligned_head_dim,
                                                                  int kv_stride_tokens)
{
    (void)head_dim;
    attention_forward_head_major_gqa_flash_impl(q, k, v, output,
                                                num_heads, num_kv_heads,
                                                num_tokens, head_dim,
                                                aligned_head_dim,
                                                kv_stride_tokens,
                                                /*causal=*/1,
                                                /*round_full_kv_fp16=*/0,
                                                1.0f);
}

void attention_forward_causal_head_major_shared_kv_gemma4(const float *q,
                                                          float *output,
                                                          int num_heads,
                                                          int num_tokens,
                                                          int head_dim,
                                                          int aligned_head_dim,
                                                          int kv_stride_tokens)
{
    attention_forward_causal_head_major_gqa_flash_strided_gemma4(
        q, q, q, output, num_heads, num_heads, num_tokens,
        head_dim, aligned_head_dim, kv_stride_tokens
    );
}

void attention_forward_full_head_major_gqa_flash_strided_gemma4(const float *q,
                                                                const float *k,
                                                                const float *v,
                                                                float *output,
                                                                int num_heads,
                                                                int num_kv_heads,
                                                                int num_tokens,
                                                                int head_dim,
                                                                int aligned_head_dim,
                                                                int kv_stride_tokens)
{
    (void)head_dim;
    attention_forward_head_major_gqa_flash_impl(q, k, v, output,
                                                num_heads, num_kv_heads,
                                                num_tokens, head_dim,
                                                aligned_head_dim,
                                                kv_stride_tokens,
                                                /*causal=*/0,
                                                /*round_full_kv_fp16=*/1,
                                                1.0f);
}


void attention_forward_mixed_visual_chunk_head_major_gqa_flash_strided_gemma4(const float *q,
                                                                              const float *k,
                                                                              const float *v,
                                                                              float *output,
                                                                              int num_heads,
                                                                              int num_kv_heads,
                                                                              int num_tokens,
                                                                              int head_dim,
                                                                              int aligned_head_dim,
                                                                              int kv_stride_tokens,
                                                                              int visual_start,
                                                                              int visual_tokens)
{
    if (!q || !k || !v || !output) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }
    if (kv_stride_tokens < num_tokens || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }
    if (visual_start < 0 || visual_tokens <= 0 || visual_start >= num_tokens) {
        attention_forward_causal_head_major_gqa_flash_strided_gemma4(q, k, v, output,
                                                                     num_heads, num_kv_heads,
                                                                     num_tokens, head_dim,
                                                                     aligned_head_dim,
                                                                     kv_stride_tokens);
        return;
    }

    int visual_end = visual_start + visual_tokens;
    if (visual_end > num_tokens) {
        visual_end = num_tokens;
    }
    if (visual_end <= visual_start) {
        attention_forward_causal_head_major_gqa_flash_strided_gemma4(q, k, v, output,
                                                                     num_heads, num_kv_heads,
                                                                     num_tokens, head_dim,
                                                                     aligned_head_dim,
                                                                     kv_stride_tokens);
        return;
    }

    const int T = num_tokens;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;
    const float scale = 1.0f;

    if (ck_strict_parity_enabled()) {
        const int debug_layer_id = ck_attention_vec_dump_enabled()
            ? ck_attention_vec_dump_next_layer_id()
            : -1;
        for (int h = 0; h < num_heads; ++h) {
            int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *k_head = k + (size_t)kv_head * kv_head_stride;
            const float *v_head = v + (size_t)kv_head * kv_head_stride;

            for (int i = 0; i < T; ++i) {
                const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
                float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
                const int in_visual = (i >= visual_start && i < visual_end);
                const int kv_tokens = in_visual ? visual_end : (i + 1);
                attention_flash_query_causal_exact(q_vec, k_head, v_head,
                                                   kv_tokens,
                                                   head_dim, aligned_head_dim,
                                                   scale, out_vec);
                ck_attention_vec_dump_exact_query(q_vec, k_head, out_vec,
                                                  kv_tokens,
                                                  head_dim, aligned_head_dim,
                                                  scale,
                                                  debug_layer_id, h, i);
            }
        }
        return;
    }

#if defined(__AVX512F__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx512
#elif defined(__AVX2__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx2
#elif defined(__AVX__)
    #define FLASH_QUERY_IMPL attention_flash_query_causal_avx
#else
    #define FLASH_QUERY_IMPL attention_flash_query_causal
#endif

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *k_head = k + (size_t)kv_head * kv_head_stride;
        const float *v_head = v + (size_t)kv_head * kv_head_stride;

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            const int in_visual = (i >= visual_start && i < visual_end);
            const int kv_tokens = in_visual ? visual_end : (i + 1);
            FLASH_QUERY_IMPL(q_vec, k_head, v_head,
                             kv_tokens,
                             head_dim, aligned_head_dim,
                             scale, out_vec);
        }
    }

#undef FLASH_QUERY_IMPL
}

void attention_forward_causal_head_major_gqa_flash_strided_f16kv(const float *q,
                                                                 const float *k,
                                                                 const float *v,
                                                                 float *output,
                                                                 int num_heads,
                                                                 int num_kv_heads,
                                                                 int num_tokens,
                                                                 int head_dim,
                                                                 int aligned_head_dim,
                                                                 int kv_stride_tokens)
{
    if (!q || !k || !v || !output) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }
    if (kv_stride_tokens < num_tokens) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const int T = num_tokens;
    const size_t kv_head_stride = (size_t)kv_stride_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *k_head = k + (size_t)kv_head * kv_head_stride;
        const float *v_head = v + (size_t)kv_head * kv_head_stride;

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            attention_flash_query_causal_exact_f16kv(q_vec, k_head, v_head,
                                                     /*kv_tokens=*/i + 1,
                                                     head_dim, aligned_head_dim,
                                                     scale, out_vec);
        }
    }
}

void attention_forward_full_head_major_gqa_exact_strided(const float *q,
                                                         const float *k,
                                                         const float *v,
                                                         float *output,
                                                         int num_heads,
                                                         int num_kv_heads,
                                                         int num_tokens,
                                                         int head_dim,
                                                         int aligned_head_dim,
                                                         int kv_stride_tokens)
{
    if (!q || !k || !v || !output) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }
    if (kv_stride_tokens < num_tokens) {
        return;
    }

    const float scale = ck_strict_parity_enabled()
        ? ck_attention_strict_scale_f32(head_dim)
        : 1.0f / sqrtf((float) head_dim);
    const int T = num_tokens;
    const size_t kv_head_stride = (size_t) kv_stride_tokens * (size_t) aligned_head_dim;
    const int debug_layer_id = ck_strict_parity_enabled() ? ck_attention_vec_dump_next_layer_id() : -1;
    float *score_row = (float *) alloca((size_t) T * sizeof(float));
    float *v_cols = (float *) alloca((size_t) head_dim * (size_t) T * sizeof(float));
    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = (int) ((long long) h * (long long) num_kv_heads / (long long) num_heads);
        const float *k_head = k + (size_t) kv_head * kv_head_stride;
        const float *v_head = v + (size_t) kv_head * kv_head_stride;

#if CK_ENABLE_LLAMA_CPP_PARITY
        float *out_head = output + (size_t) h * (size_t) T * (size_t) aligned_head_dim;
        if (ck_strict_parity_enabled() &&
            ck_attention_head_full_ggml_graph_oracle_regular(
                q + (size_t) h * (size_t) T * (size_t) aligned_head_dim,
                k_head,
                v_head,
                out_head,
                T,
                head_dim,
                aligned_head_dim,
                scale)) {
            ck_attention_trace("regular_graph_oracle", debug_layer_id, h);
            continue;
        }
#endif

        for (int d = 0; d < head_dim; ++d) {
            float *dst_col = v_cols + (size_t) d * (size_t) T;
            for (int j = 0; j < T; ++j) {
                dst_col[j] = v_head[(size_t) j * (size_t) aligned_head_dim + (size_t) d];
            }
        }

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            attention_query_full_exact_regular(q_vec,
                                               k_head,
                                               v_cols,
                                               T,
                                               head_dim,
                                               aligned_head_dim,
                                               scale,
                                               score_row,
                                               out_vec,
                                               debug_layer_id,
                                               h,
                                               i);
        }
    }
}

void attention_forward_full_head_major_gqa_ggml_strided(const float *q,
                                                        const float *k,
                                                        const float *v,
                                                        float *output,
                                                        int num_heads,
                                                        int num_kv_heads,
                                                        int num_tokens,
                                                        int head_dim,
                                                        int aligned_head_dim,
                                                        int kv_stride_tokens)
{
    if (!q || !k || !v || !output) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }
    if (kv_stride_tokens < num_tokens) {
        return;
    }

    const int strict = ck_strict_parity_enabled();
    const float scale = strict
        ? ck_attention_strict_scale_f32(head_dim)
        : 1.0f / sqrtf((float) head_dim);
    const int T = num_tokens;
    const size_t kv_head_stride = (size_t) kv_stride_tokens * (size_t) aligned_head_dim;
    const int debug_layer_id = strict ? ck_attention_vec_dump_next_layer_id() : -1;
#if CK_ENABLE_LLAMA_CPP_PARITY
    ck_ggml_vec_dot_f32_fn dot_fn = NULL;
    ck_ggml_vec_soft_max_f32_fn softmax_fn = NULL;
    ck_ggml_compute_forward_mul_mat_fn mul_mat_fn = NULL;
    ck_ggml_compute_forward_soft_max_fn softmax_compute_fn = NULL;
    if (strict) {
        if (ck_attention_full_ggml_graph_oracle_multihead(q,
                                                          k,
                                                          v,
                                                          output,
                                                          num_heads,
                                                          num_kv_heads,
                                                          num_tokens,
                                                          head_dim,
                                                          aligned_head_dim,
                                                          kv_stride_tokens,
                                                          scale)) {
            return;
        }
        dot_fn = ck_resolve_ggml_vec_dot_f32();
        softmax_fn = ck_resolve_ggml_vec_soft_max_f32();
        mul_mat_fn = ck_resolve_ggml_compute_forward_mul_mat();
        softmax_compute_fn = ck_resolve_ggml_compute_forward_soft_max();
    }
#endif
    if (strict) {
        float *score_row = (float *) alloca((size_t) T * sizeof(float));
        float *v_cols = (float *) alloca((size_t) head_dim * (size_t) T * sizeof(float));
#if CK_ENABLE_LLAMA_CPP_PARITY
        float *prob_row = (float *) alloca((size_t) T * sizeof(float));
#endif

        for (int h = 0; h < num_heads; ++h) {
            const int kv_head = (int) ((long long) h * (long long) num_kv_heads / (long long) num_heads);
            const float *k_head = k + (size_t) kv_head * kv_head_stride;
            const float *v_head = v + (size_t) kv_head * kv_head_stride;

#if CK_ENABLE_LLAMA_CPP_PARITY
            float *out_head = output + (size_t) h * (size_t) T * (size_t) aligned_head_dim;
            if (ck_attention_head_full_ggml_graph_oracle_regular(
                    q + (size_t) h * (size_t) T * (size_t) aligned_head_dim,
                    k_head,
                    v_head,
                    out_head,
                    T,
                    head_dim,
                    aligned_head_dim,
                    scale)) {
                continue;
            }
#endif

            for (int d = 0; d < head_dim; ++d) {
                float *dst_col = v_cols + (size_t) d * (size_t) T;
                for (int j = 0; j < T; ++j) {
                    dst_col[j] = v_head[(size_t) j * (size_t) aligned_head_dim + (size_t) d];
                }
            }

#if CK_ENABLE_LLAMA_CPP_PARITY
            if (dot_fn && softmax_fn && ck_attention_ggml_out_graph_enabled()) {
                if (attention_head_full_dyn_ggml_regular_graph_out(
                        q + (size_t) h * (size_t) T * (size_t) aligned_head_dim,
                        k_head,
                        v_cols,
                        T,
                        head_dim,
                        aligned_head_dim,
                        scale,
                        score_row,
                        prob_row,
                        out_head,
                        dot_fn,
                        softmax_fn,
                        debug_layer_id,
                        h)) {
                    if (T > 0) {
                        ck_attention_trace("dyn_ggml_regular_graph_out", debug_layer_id, h);
                        ck_attention_trace_float("scale", debug_layer_id, h, scale);
                    }
                    continue;
                }
            }
#endif

            for (int i = 0; i < T; ++i) {
                const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
                float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
#if CK_ENABLE_LLAMA_CPP_PARITY
                if (dot_fn && softmax_fn) {
                    if (i == 0) {
                        ck_attention_trace("dyn_ggml_regular", debug_layer_id, h);
                        ck_attention_trace_float("scale", debug_layer_id, h, scale);
                    }
                    attention_query_full_dyn_ggml_regular(q_vec,
                                                          k_head,
                                                          v_cols,
                                                          T,
                                                          head_dim,
                                                          aligned_head_dim,
                                                          scale,
                                                          score_row,
                                                          prob_row,
                                                          out_vec,
                                                          dot_fn,
                                                          softmax_fn,
                                                          debug_layer_id,
                                                          h,
                                                          i);
                } else if (dot_fn && mul_mat_fn && softmax_compute_fn) {
                    if (i == 0) {
                        ck_attention_trace("ggml_compute_regular", debug_layer_id, h);
                    }
                    attention_query_full_ggml_compute_regular(q_vec,
                                                              k_head,
                                                              v_cols,
                                                              T,
                                                              head_dim,
                                                              aligned_head_dim,
                                                              scale,
                                                              score_row,
                                                              prob_row,
                                                              out_vec,
                                                              dot_fn,
                                                              mul_mat_fn,
                                                              softmax_compute_fn);
                } else
#endif
                {
                    if (i == 0) {
                        ck_attention_trace("ggml_regular", debug_layer_id, h);
                    }
                    attention_query_full_ggml_regular(q_vec,
                                                      k_head,
                                                      v_cols,
                                                      T,
                                                      head_dim,
                                                      aligned_head_dim,
                                                      scale,
                                                      score_row,
                                                      out_vec,
                                                      debug_layer_id,
                                                      h,
                                                      i);
                }
            }
        }
        return;
    }

#pragma omp parallel for schedule(static) if(num_heads > 1)
    for (int h = 0; h < num_heads; ++h) {
        float *score_row_heap = (float *) malloc((size_t) T * sizeof(float));
        float *v_cols = (float *) malloc((size_t) head_dim * (size_t) T * sizeof(float));
        float *score_row = score_row_heap ? score_row_heap : (float *) alloca((size_t) T * sizeof(float));
        const int kv_head = (int) ((long long) h * (long long) num_kv_heads / (long long) num_heads);
        const float *k_head = k + (size_t) kv_head * kv_head_stride;
        const float *v_head = v + (size_t) kv_head * kv_head_stride;

        if (v_cols) {
            for (int d = 0; d < head_dim; ++d) {
                float *dst_col = v_cols + (size_t) d * (size_t) T;
                for (int j = 0; j < T; ++j) {
                    dst_col[j] = v_head[(size_t) j * (size_t) aligned_head_dim + (size_t) d];
                }
            }
        }

        for (int i = 0; i < T; ++i) {
            const float *q_vec = q + qkv_index(h, i, 0, T, aligned_head_dim);
            float *out_vec = output + qkv_index(h, i, 0, T, aligned_head_dim);
            if (score_row && v_cols) {
                attention_query_full_ggml_regular(q_vec,
                                                  k_head,
                                                  v_cols,
                                                  T,
                                                  head_dim,
                                                  aligned_head_dim,
                                                  scale,
                                                  score_row,
                                                  out_vec,
                                                  -1,
                                                  h,
                                                  i);
            } else if (score_row) {
                attention_query_full_ggml_regular_direct_v(q_vec,
                                                           k_head,
                                                           v_head,
                                                           T,
                                                           head_dim,
                                                           aligned_head_dim,
                                                           scale,
                                                           score_row,
                                                           out_vec);
            }
        }
        free(v_cols);
        free(score_row_heap);
    }
}

/**
 * Flash attention decode (single token attends to KV cache)
 * @test test_flash_attention.py::TestFlashAttention::test_flash_decode
 * @test test_kv_cache_attention.py::TestKVCacheAttention::test_flash_decode
 * @test test_fused_attention_decode.py::TestFusedAttentionDecode::test_flash_decode
 * @test test_attention.py::TestAttentionForward::test_flash_decode
 *
 * Single query token attends to kv_tokens in KV cache.
 * Uses true flash attention from attention_flash_true.c.
 *
 * After changes: make test && make llamacpp-parity-full
 */
void attention_forward_decode_head_major_gqa_flash(const float *q_token,
                                                   const float *k_cache,
                                                   const float *v_cache,
                                                   float *out_token,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int kv_tokens,
                                                   int cache_capacity,
                                                   int head_dim,
                                                   int aligned_head_dim)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
        const float *k_head = k_cache + (size_t)kv_head * head_stride;
        const float *v_head = v_cache + (size_t)kv_head * head_stride;
        float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;

        attention_flash_decode(out_head,
                               q_head,
                               k_head,
                               v_head,
                               1,
                               kv_tokens,
                               1,
                               aligned_head_dim,
                               scale);
    }
}


void attention_forward_decode_head_major_gqa_flash_gemma4(const float *q_token,
                                                          const float *k_cache,
                                                          const float *v_cache,
                                                          float *out_token,
                                                          int num_heads,
                                                          int num_kv_heads,
                                                          int kv_tokens,
                                                          int cache_capacity,
                                                          int head_dim,
                                                          int aligned_head_dim)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const float scale = 1.0f;
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
        const float *k_head = k_cache + (size_t)kv_head * head_stride;
        const float *v_head = v_cache + (size_t)kv_head * head_stride;
        float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;

        attention_flash_decode(out_head,
                               q_head,
                               k_head,
                               v_head,
                               1,
                               kv_tokens,
                               1,
                               aligned_head_dim,
                               scale);
    }
}

void attention_forward_chunk_head_major_gqa_flash_gemma4(const float *q_chunk,
                                                         const float *k_cache,
                                                         const float *v_cache,
                                                         float *out_chunk,
                                                         int num_heads,
                                                         int num_kv_heads,
                                                         int q_tokens,
                                                         int kv_tokens,
                                                         int cache_capacity,
                                                         int head_dim,
                                                         int aligned_head_dim)
{
    if (!q_chunk || !k_cache || !v_cache || !out_chunk) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || q_tokens <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const float scale = 1.0f;
    const size_t cache_head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;
    const size_t q_head_stride = (size_t)q_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *k_head = k_cache + (size_t)kv_head * cache_head_stride;
        const float *v_head = v_cache + (size_t)kv_head * cache_head_stride;
        for (int t = 0; t < q_tokens; ++t) {
            const float *q_head = q_chunk + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
            float *out_head = out_chunk + (size_t)h * q_head_stride + (size_t)t * (size_t)aligned_head_dim;
            attention_flash_decode(out_head,
                                   q_head,
                                   k_head,
                                   v_head,
                                   1,
                                   kv_tokens,
                                   1,
                                   aligned_head_dim,
                                   scale);
        }
    }
}

void attention_forward_decode_head_major_shared_kv_gemma4(const float *q_token,
                                                          const float *k_cache,
                                                          const float *v_cache,
                                                          float *out_token,
                                                          int num_heads,
                                                          int kv_tokens,
                                                          int cache_capacity,
                                                          int head_dim,
                                                          int aligned_head_dim)
{
    attention_forward_decode_head_major_gqa_flash_gemma4(
        q_token, k_cache, v_cache, out_token, num_heads, num_heads,
        kv_tokens, cache_capacity, head_dim, aligned_head_dim
    );
}

void attention_forward_decode_head_major_gqa_flash_f16kv(const float *q_token,
                                                         const float *k_cache,
                                                         const float *v_cache,
                                                         float *out_token,
                                                         int num_heads,
                                                         int num_kv_heads,
                                                         int kv_tokens,
                                                         int cache_capacity,
                                                         int head_dim,
                                                         int aligned_head_dim)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
        const float *k_head = k_cache + (size_t)kv_head * head_stride;
        const float *v_head = v_cache + (size_t)kv_head * head_stride;
        float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;

        attention_flash_query_causal_exact_f16kv(q_head,
                                                 k_head,
                                                 v_head,
                                                 kv_tokens,
                                                 head_dim,
                                                 aligned_head_dim,
                                                 scale,
                                                 out_head);
    }
}

typedef struct {
    const float *q_token;
    const uint16_t *k_cache;
    const uint16_t *v_cache;
    float *partials;
    int num_heads;
    int num_kv_heads;
    int kv_tokens;
    int cache_capacity;
    int head_dim;
    int aligned_head_dim;
    int split_chunks;
} ck_attention_f16_split_args_t;

static inline float ck_attention_dot_f16_llama(const uint16_t *x,
                                                const uint16_t *y,
                                                int n)
{
    int i = 0;
#if defined(__AVX512FP16__)
    // Match ggml's native AVX-512 FP16 path: four 32-lane FP16
    // accumulators per 128 values, then its fixed pairwise tree.
    __m512h sum0 = _mm512_setzero_ph();
    __m512h sum1 = _mm512_setzero_ph();
    __m512h sum2 = _mm512_setzero_ph();
    __m512h sum3 = _mm512_setzero_ph();
    const int n128 = n & ~127;
    for (; i < n128; i += 128) {
        const __m512h x0 = _mm512_loadu_ph(x + i);
        const __m512h y0 = _mm512_loadu_ph(y + i);
        const __m512h x1 = _mm512_loadu_ph(x + i + 32);
        const __m512h y1 = _mm512_loadu_ph(y + i + 32);
        const __m512h x2 = _mm512_loadu_ph(x + i + 64);
        const __m512h y2 = _mm512_loadu_ph(y + i + 64);
        const __m512h x3 = _mm512_loadu_ph(x + i + 96);
        const __m512h y3 = _mm512_loadu_ph(y + i + 96);
        sum0 = _mm512_fmadd_ph(x0, y0, sum0);
        sum1 = _mm512_fmadd_ph(x1, y1, sum1);
        sum2 = _mm512_fmadd_ph(x2, y2, sum2);
        sum3 = _mm512_fmadd_ph(x3, y3, sum3);
    }
    sum0 = _mm512_add_ph(sum0, sum2);
    sum1 = _mm512_add_ph(sum1, sum3);
    sum0 = _mm512_add_ph(sum0, sum1);
    float result = (float) _mm512_reduce_add_ph(sum0);
#elif defined(__AVX512F__)
    // Match ggml's AVX-512 FP16-to-FP32 path: four 16-lane
    // accumulators per 64 values, then its fixed pairwise tree.
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    const int n64 = n & ~63;
    for (; i < n64; i += 64) {
        const __m512 x0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (x + i)));
        const __m512 y0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (y + i)));
        const __m512 x1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (x + i + 16)));
        const __m512 y1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (y + i + 16)));
        const __m512 x2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (x + i + 32)));
        const __m512 y2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (y + i + 32)));
        const __m512 x3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (x + i + 48)));
        const __m512 y3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (y + i + 48)));
        sum0 = _mm512_fmadd_ps(x0, y0, sum0);
        sum1 = _mm512_fmadd_ps(x1, y1, sum1);
        sum2 = _mm512_fmadd_ps(x2, y2, sum2);
        sum3 = _mm512_fmadd_ps(x3, y3, sum3);
    }
    sum0 = _mm512_add_ps(sum0, sum2);
    sum1 = _mm512_add_ps(sum1, sum3);
    sum0 = _mm512_add_ps(sum0, sum1);
    float result = _mm512_reduce_add_ps(sum0);
#elif defined(__AVX2__) && defined(__F16C__)
    // Match ggml_vec_dot_f16's AVX reduction contract: four independent
    // accumulators per 32 values, followed by its fixed pairwise tree.
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        const __m256 x0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (x + i)));
        const __m256 y0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (y + i)));
        const __m256 x1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (x + i + 8)));
        const __m256 y1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (y + i + 8)));
        const __m256 x2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (x + i + 16)));
        const __m256 y2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (y + i + 16)));
        const __m256 x3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (x + i + 24)));
        const __m256 y3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (y + i + 24)));
#if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(x0, y0, sum0);
        sum1 = _mm256_fmadd_ps(x1, y1, sum1);
        sum2 = _mm256_fmadd_ps(x2, y2, sum2);
        sum3 = _mm256_fmadd_ps(x3, y3, sum3);
#else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x0, y0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(x1, y1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(x2, y2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(x3, y3));
#endif
    }
    sum0 = _mm256_add_ps(sum0, sum2);
    sum1 = _mm256_add_ps(sum1, sum3);
    sum0 = _mm256_add_ps(sum0, sum1);
    const __m128 pair = _mm_add_ps(
        _mm256_castps256_ps128(sum0),
        _mm256_extractf128_ps(sum0, 1));
    const __m128 half = _mm_hadd_ps(pair, pair);
    float result = _mm_cvtss_f32(_mm_hadd_ps(half, half));
#else
    float result = 0.0f;
#endif
    for (; i < n; ++i) {
        result += CK_FP16_TO_FP32(x[i]) * CK_FP16_TO_FP32(y[i]);
    }
    return result;
}

static void ck_attention_f16_split_work(int ith, int nth, void *opaque)
{
    ck_attention_f16_split_args_t *args = (ck_attention_f16_split_args_t *) opaque;
    const int partial_stride = args->aligned_head_dim + 2;
    const int total_jobs = args->num_heads * args->split_chunks;
    const int chunk_size = (args->kv_tokens + args->split_chunks - 1) / args->split_chunks;
    const size_t head_stride = (size_t) args->cache_capacity * (size_t) args->aligned_head_dim;
    const float scale = ck_attention_strict_scale_f32(args->head_dim);
    uint16_t *q_half = (uint16_t *) alloca((size_t) args->aligned_head_dim * sizeof(uint16_t));
    uint16_t *acc_half = (uint16_t *) alloca((size_t) args->aligned_head_dim * sizeof(uint16_t));

    for (int job = ith; job < total_jobs; job += nth) {
        const int h = job / args->split_chunks;
        const int chunk = job % args->split_chunks;
        const int kv_head = (int) ((long long) h * (long long) args->num_kv_heads /
                                   (long long) args->num_heads);
        const int begin = chunk * chunk_size;
        const int end = begin + chunk_size < args->kv_tokens
            ? begin + chunk_size
            : args->kv_tokens;
        const float *q_head = args->q_token + (size_t) h * (size_t) args->aligned_head_dim;
        const uint16_t *k_head = args->k_cache + (size_t) kv_head * head_stride;
        const uint16_t *v_head = args->v_cache + (size_t) kv_head * head_stride;
        float *partial = args->partials + (size_t) job * (size_t) partial_stride;

        for (int d = 0; d < args->aligned_head_dim; ++d) {
            q_half[d] = CK_FP32_TO_FP16(q_head[d]);
            acc_half[d] = CK_FP32_TO_FP16(0.0f);
        }

        float sum = 0.0f;
        float max_score = -INFINITY;
        for (int j = begin; j < end; ++j) {
            const uint16_t *k_vec = k_head + (size_t) j * (size_t) args->aligned_head_dim;
            const uint16_t *v_vec = v_head + (size_t) j * (size_t) args->aligned_head_dim;
            const float dot = ck_attention_dot_f16_llama(q_half, k_vec, args->head_dim);
            const float score = dot * scale;
            const float old_max = max_score;
            float max_scale = 1.0f;
            float value_scale = 1.0f;

            if (score > max_score) {
                max_score = score;
                max_scale = isfinite(old_max) ? expf(old_max - max_score) : 0.0f;
                for (int d = 0; d < args->head_dim; ++d) {
                    const float scaled = CK_FP16_TO_FP32(acc_half[d]) * max_scale;
                    acc_half[d] = CK_FP32_TO_FP16(scaled);
                }
            } else {
                value_scale = expf(score - max_score);
            }

            for (int d = 0; d < args->head_dim; ++d) {
                const float updated = CK_FP16_TO_FP32(acc_half[d]) +
                                      value_scale * CK_FP16_TO_FP32(v_vec[d]);
                acc_half[d] = CK_FP32_TO_FP16(updated);
            }
            sum = sum * max_scale + value_scale;
        }

        partial[0] = max_score;
        partial[1] = sum;
        for (int d = 0; d < args->head_dim; ++d) {
            partial[2 + d] = CK_FP16_TO_FP32(acc_half[d]);
        }
        for (int d = args->head_dim; d < args->aligned_head_dim; ++d) {
            partial[2 + d] = 0.0f;
        }
    }
}

void attention_forward_decode_head_major_gqa_flash_f16cache_split(const float *q_token,
                                                                  const uint16_t *k_cache,
                                                                  const uint16_t *v_cache,
                                                                  float *out_token,
                                                                  int num_heads,
                                                                  int num_kv_heads,
                                                                  int kv_tokens,
                                                                  int cache_capacity,
                                                                  int head_dim,
                                                                  int aligned_head_dim,
                                                                  int split_chunks)
{
    if (!q_token || !k_cache || !v_cache || !out_token ||
        num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 ||
        cache_capacity <= 0 || kv_tokens > cache_capacity ||
        head_dim <= 0 || aligned_head_dim < head_dim) {
        return;
    }

    if (split_chunks < 1) {
        split_chunks = 1;
    }
    if (split_chunks > kv_tokens) {
        split_chunks = kv_tokens;
    }

    const int partial_stride = aligned_head_dim + 2;
    const size_t max_partial_bytes = 1024u * 1024u;
    while (split_chunks > 1 &&
           (size_t) num_heads * (size_t) split_chunks * (size_t) partial_stride * sizeof(float) >
               max_partial_bytes) {
        split_chunks = (split_chunks + 1) / 2;
    }

    const size_t resolved_count = (size_t) num_heads * (size_t) split_chunks * (size_t) partial_stride;
    float *partials = (float *) alloca(resolved_count * sizeof(float));
    ck_attention_f16_split_args_t args = {
        .q_token = q_token,
        .k_cache = k_cache,
        .v_cache = v_cache,
        .partials = partials,
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .kv_tokens = kv_tokens,
        .cache_capacity = cache_capacity,
        .head_dim = head_dim,
        .aligned_head_dim = aligned_head_dim,
        .split_chunks = split_chunks,
    };

    ck_threadpool_t *pool = ck_threadpool_global();
    int active_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    const int total_jobs = num_heads * split_chunks;
    if (active_threads > total_jobs) {
        active_threads = total_jobs;
    }
    if (pool && active_threads > 1) {
        ck_threadpool_dispatch_n(pool, active_threads, ck_attention_f16_split_work, &args);
    } else {
        ck_attention_f16_split_work(0, 1, &args);
    }

    for (int h = 0; h < num_heads; ++h) {
        float *out_head = out_token + (size_t) h * (size_t) aligned_head_dim;
        float final_max = -INFINITY;
        float final_sum = 0.0f;
        for (int d = 0; d < aligned_head_dim; ++d) {
            out_head[d] = 0.0f;
        }

        for (int chunk = 0; chunk < split_chunks; ++chunk) {
            const float *partial = partials +
                ((size_t) h * (size_t) split_chunks + (size_t) chunk) * (size_t) partial_stride;
            const float chunk_max = partial[0];
            const float chunk_sum = partial[1];
            if (chunk_sum == 0.0f) {
                continue;
            }
            const float new_max = fmaxf(final_max, chunk_max);
            const float old_scale = isfinite(final_max) ? expf(final_max - new_max) : 0.0f;
            const float chunk_scale = expf(chunk_max - new_max);
            for (int d = 0; d < head_dim; ++d) {
                out_head[d] = out_head[d] * old_scale + partial[2 + d] * chunk_scale;
            }
            final_sum = final_sum * old_scale + chunk_sum * chunk_scale;
            final_max = new_max;
        }

        if (final_sum > 0.0f) {
            const float inv_sum = 1.0f / final_sum;
            for (int d = 0; d < head_dim; ++d) {
                out_head[d] *= inv_sum;
            }
        }
    }
}

ck_attention_status_t attention_forward_decode_head_major_gqa_flash_f16cache_contract(
    const float *q_token,
    const uint16_t *k_cache,
    const uint16_t *v_cache,
    float *out_token,
    int num_heads,
    int num_kv_heads,
    int kv_tokens,
    int cache_capacity,
    int head_dim,
    int aligned_head_dim,
    ck_attention_reduction_t reduction)
{
    if (!q_token || !k_cache || !v_cache || !out_token ||
        num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 ||
        cache_capacity <= 0 || kv_tokens > cache_capacity ||
        head_dim <= 0 || aligned_head_dim < head_dim) {
        return CK_ATTENTION_STATUS_INVALID_ARGUMENT;
    }

    switch (reduction) {
    case CK_ATTN_REDUCTION_FP32_ONLINE:
        attention_forward_decode_head_major_gqa_flash_f16cache(
            q_token, k_cache, v_cache, out_token,
            num_heads, num_kv_heads, kv_tokens, cache_capacity,
            head_dim, aligned_head_dim);
        return CK_ATTENTION_STATUS_OK;

    case CK_ATTN_REDUCTION_F16_ONLINE_FP32_MERGE: {
        const int split_chunks = kv_tokens >= 512 ? ck_get_num_threads() : 1;
        attention_forward_decode_head_major_gqa_flash_f16cache_split(
            q_token, k_cache, v_cache, out_token,
            num_heads, num_kv_heads, kv_tokens, cache_capacity,
            head_dim, aligned_head_dim, split_chunks);
        return CK_ATTENTION_STATUS_OK;
    }

    default:
        return CK_ATTENTION_STATUS_UNSUPPORTED_CONTRACT;
    }
}

ck_attention_status_t attention_forward_causal_head_major_gqa_prefill_append_f16cache_contract(
    const float *q,
    const uint16_t *k_cache,
    const uint16_t *v_cache,
    float *output,
    int num_heads,
    int num_kv_heads,
    int q_tokens,
    int past_tokens,
    int cache_capacity,
    int head_dim,
    int aligned_head_dim,
    ck_attention_reduction_t reduction)
{
    if (!q || !k_cache || !v_cache || !output ||
        num_heads <= 0 || num_kv_heads <= 0 || q_tokens <= 0 ||
        past_tokens < 0 || past_tokens + q_tokens > cache_capacity ||
        head_dim <= 0 || aligned_head_dim < head_dim) {
        return CK_ATTENTION_STATUS_INVALID_ARGUMENT;
    }

    const size_t token_elems = (size_t) num_heads * (size_t) aligned_head_dim;
    float *q_token = (float *) malloc(token_elems * sizeof(float));
    float *out_token = (float *) malloc(token_elems * sizeof(float));
    if (!q_token || !out_token) {
        free(q_token);
        free(out_token);
        return CK_ATTENTION_STATUS_INVALID_ARGUMENT;
    }

    ck_attention_status_t status = CK_ATTENTION_STATUS_OK;
    for (int t = 0; t < q_tokens; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            const float *src = q +
                ((size_t) h * (size_t) q_tokens + (size_t) t) * (size_t) aligned_head_dim;
            memcpy(q_token + (size_t) h * (size_t) aligned_head_dim,
                   src,
                   (size_t) aligned_head_dim * sizeof(float));
        }

        status = attention_forward_decode_head_major_gqa_flash_f16cache_contract(
            q_token, k_cache, v_cache, out_token,
            num_heads, num_kv_heads, past_tokens + t + 1, cache_capacity,
            head_dim, aligned_head_dim, reduction);
        if (status != CK_ATTENTION_STATUS_OK) {
            break;
        }

        for (int h = 0; h < num_heads; ++h) {
            float *dst = output +
                ((size_t) h * (size_t) q_tokens + (size_t) t) * (size_t) aligned_head_dim;
            memcpy(dst,
                   out_token + (size_t) h * (size_t) aligned_head_dim,
                   (size_t) aligned_head_dim * sizeof(float));
        }
    }

    free(q_token);
    free(out_token);
    return status;
}

void attention_forward_decode_head_major_gqa_flash_f16cache(const float *q_token,
                                                            const uint16_t *k_cache,
                                                            const uint16_t *v_cache,
                                                            float *out_token,
                                                            int num_heads,
                                                            int num_kv_heads,
                                                            int kv_tokens,
                                                            int cache_capacity,
                                                            int head_dim,
                                                            int aligned_head_dim)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;
    const size_t scratch_elems = (size_t)kv_tokens * (size_t)aligned_head_dim;
    const size_t scratch_bytes = scratch_elems * sizeof(float);
    const size_t max_stack_bytes = 1024u * 1024u;

    if (scratch_bytes * 2u > max_stack_bytes) {
        for (int h = 0; h < num_heads; ++h) {
            int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            const uint16_t *k_head = k_cache + (size_t)kv_head * head_stride;
            const uint16_t *v_head = v_cache + (size_t)kv_head * head_stride;
            float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;

            for (int d = 0; d < aligned_head_dim; ++d) {
                out_head[d] = 0.0f;
            }

            float max_score = -INFINITY;
            for (int j = 0; j < kv_tokens; ++j) {
                const uint16_t *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_head[d] * CK_FP16_TO_FP32(k_vec[d]);
                }
                const float score = dot * scale;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float sum = 0.0f;
            for (int j = 0; j < kv_tokens; ++j) {
                const uint16_t *k_vec = k_head + (size_t)j * (size_t)aligned_head_dim;
                const uint16_t *v_vec = v_head + (size_t)j * (size_t)aligned_head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_head[d] * CK_FP16_TO_FP32(k_vec[d]);
                }
                const float w = expf(dot * scale - max_score);
                sum += w;
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] += w * CK_FP16_TO_FP32(v_vec[d]);
                }
            }

            if (sum > 0.0f) {
                const float inv_sum = 1.0f / sum;
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] *= inv_sum;
                }
            }
            for (int d = head_dim; d < aligned_head_dim; ++d) {
                out_head[d] = 0.0f;
            }
        }
        return;
    }

    float *k_head_fp32 = (float *)alloca(scratch_bytes);
    float *v_head_fp32 = (float *)alloca(scratch_bytes);

    for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        const uint16_t *k_head = k_cache + (size_t)kv_head * head_stride;
        const uint16_t *v_head = v_cache + (size_t)kv_head * head_stride;
        const int q_begin = (int)((long long)kv_head * (long long)num_heads / (long long)num_kv_heads);
        const int q_end = (int)((long long)(kv_head + 1) * (long long)num_heads / (long long)num_kv_heads);

        ck_local_fp16_to_fp32_2d(k_head, k_head_fp32, kv_tokens, aligned_head_dim, aligned_head_dim, aligned_head_dim);
        ck_local_fp16_to_fp32_2d(v_head, v_head_fp32, kv_tokens, aligned_head_dim, aligned_head_dim, aligned_head_dim);

        for (int h = q_begin; h < q_end; ++h) {
            const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;

            attention_flash_decode(out_head,
                                   q_head,
                                   k_head_fp32,
                                   v_head_fp32,
                                   1,
                                   kv_tokens,
                                   1,
                                   aligned_head_dim,
                                   scale);
        }
    }
}

/**
 * @brief WARNING: This is NOT true flash attention!
 *
 * This function is named "flash" but implements regular attention with O(n) complexity.
 * It's kept for reference and as a fallback.
 *
 * TRUE flash attention is implemented in attention_flash_true.c
 * @test test_kv_cache_attention.py::TestKVCacheAttention::test_regular_decode
 * @test test_attention.py::TestAttentionForward::test_regular_decode
 *
 * Regular attention decode (score-matrix version) for fallback.
 *
 * After changes: make test
 */
void attention_forward_decode_head_major_gqa_regular(const float *q_token,
                                                   const float *k_cache,
                                                   const float *v_cache,
                                                   float *out_token,
                                                   int num_heads,
                                                   int num_kv_heads,
                                                   int kv_tokens,
                                                   int cache_capacity,
                                                   int head_dim,
                                                   int aligned_head_dim)
{
    if (!q_token || !k_cache || !v_cache || !out_token) {
        return;
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
        return;
    }
    if (kv_tokens > cache_capacity) {
        return;
    }

    const int strict = ck_strict_parity_enabled();
    const float scale = strict
        ? ck_attention_strict_scale_f32(head_dim)
        : 1.0f / sqrtf((float) head_dim);
    const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;

    // Select SIMD implementation based on compile-time CPU features
#if defined(__AVX512F__)
    #define FLASH_QUERY_IMPL_DECODE attention_flash_query_causal_avx512
#elif defined(__AVX2__)
    #define FLASH_QUERY_IMPL_DECODE attention_flash_query_causal_avx2
#elif defined(__AVX__)
    #define FLASH_QUERY_IMPL_DECODE attention_flash_query_causal_avx
#else
    #define FLASH_QUERY_IMPL_DECODE attention_flash_query_causal
#endif

#pragma omp parallel for schedule(static) if(num_heads > 1)
    for (int h = 0; h < num_heads; ++h) {
        int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
        const float *q_vec = q_token + (size_t)h * (size_t)aligned_head_dim;
        const float *k_head = k_cache + (size_t)kv_head * head_stride;
        const float *v_head = v_cache + (size_t)kv_head * head_stride;
        float *out_vec = out_token + (size_t)h * (size_t)aligned_head_dim;

        if (strict) {
            attention_flash_query_causal_exact_f16kv(q_vec,
                                                     k_head,
                                                     v_head,
                                                     kv_tokens,
                                                     head_dim,
                                                     aligned_head_dim,
                                                     scale,
                                                     out_vec);
            continue;
        }

        FLASH_QUERY_IMPL_DECODE(q_vec, k_head, v_head,
                                 kv_tokens, head_dim, aligned_head_dim,
                                 scale, out_vec);
    }

#undef FLASH_QUERY_IMPL_DECODE
}

// ============================================================================
// ATTENTION BACKWARD - Causal, Head-Major, GQA-aware
// ============================================================================
//
// Backward pass for scaled dot-product attention with causal mask.
//
// Given:
//   d_output: gradient from the layer above [num_heads, T, head_dim]
//   q, k, v: saved activations from forward pass
//   attn_weights: saved softmax output from forward [num_heads, T, T]
//
// Computes:
//   d_q: gradient w.r.t. queries  [num_heads, T, head_dim]
//   d_k: gradient w.r.t. keys     [num_kv_heads, T, head_dim]
//   d_v: gradient w.r.t. values   [num_kv_heads, T, head_dim]
//
// Math derivation:
//   Forward: scores = Q @ K^T / sqrt(d)
//            weights = causal_softmax(scores)
//            output = weights @ V
//
//   Backward through V multiply:
//     d_weights = d_output @ V^T           [H, T, T]
//     d_v = weights^T @ d_output           [H_kv, T, d]
//
//   Backward through softmax:
//     d_scores = softmax_backward(d_weights, weights)
//
//   Backward through Q @ K^T:
//     d_q = d_scores @ K / sqrt(d)         [H, T, d]
//     d_k = d_scores^T @ Q / sqrt(d)       [H_kv, T, d]
//
// For GQA: multiple query heads share the same KV head, so we accumulate
// gradients from all query heads that map to each KV head.
//
/**
 * BF16 attention backward with caller-provided scratch buffers
 * @test bf16/test_attention_bf16.py::TestAttentionBF16::test_bf16_backward
 *
 * Accepts BF16 inputs, converts to FP32, runs FP32 backward.
 * Caller provides scratch buffers (no per-call malloc).
 *
 * After changes: make test
 */
void attention_backward_causal_head_major_gqa_bf16(
    const uint16_t *d_output,      // [num_heads, T, aligned_head_dim]
    float *d_x,                    // [num_heads, T, aligned_head_dim]
    const uint16_t *q,             // [num_heads, T, aligned_head_dim]
    const uint16_t *k,             // [num_kv_heads, T, aligned_head_dim]
    const uint16_t *v,             // [num_kv_heads, T, aligned_head_dim]
    const float *attn_weights,     // [num_heads, T, aligned_context_window]
    float *d_q,                    // [num_heads, T, aligned_head_dim] output
    float *d_k,                    // [num_kv_heads, T, aligned_head_dim] output
    float *d_v,                    // [num_kv_heads, T, aligned_head_dim] output
    float *d_scores,               // [num_heads, T, aligned_context_window] scratch
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window,
    float *scratch_d_output,
    float *scratch_q,
    float *scratch_k,
    float *scratch_v)
{
    (void)d_x;
    const size_t head_elems = (size_t)num_heads * (size_t)num_tokens * (size_t)aligned_head_dim;
    const size_t kv_elems = (size_t)num_kv_heads * (size_t)num_tokens * (size_t)aligned_head_dim;

    if (!scratch_d_output || !scratch_q || !scratch_k || !scratch_v) return;

    convert_bf16_tensor_to_buf(d_output, scratch_d_output, head_elems);
    convert_bf16_tensor_to_buf(q, scratch_q, head_elems);
    convert_bf16_tensor_to_buf(k, scratch_k, kv_elems);
    convert_bf16_tensor_to_buf(v, scratch_v, kv_elems);

    attention_backward_causal_head_major_gqa(scratch_d_output, scratch_q, scratch_k, scratch_v,
                                             attn_weights,
                                             d_q, d_k, d_v, d_scores,
                                             num_heads, num_kv_heads,
                                             num_tokens, head_dim,
                                             aligned_head_dim, aligned_context_window);
    /* No free - caller owns scratch buffers */
}

/**
 * GQA causal attention backward (score-matrix version)
 * @test test_attention_backward.py::TestAttentionBackwardGQA::test_gqa_backward
 * @test test_attention_backward.py::TestAttentionBackwardGQA::test_gqa_vs_separate
 * @test test_parity.py::test_attention_backward_parity
 *
 * Computes dQ, dK, dV given dOutput and attention weights.
 * Supports grouped-query attention with head broadcasting.
 *
 * After changes: make test && make llamacpp-parity-full
 */
void attention_backward_causal_head_major_gqa(
    const float *d_output,      // [num_heads, T, aligned_head_dim]
    const float *q,             // [num_heads, T, aligned_head_dim]
    const float *k,             // [num_kv_heads, T, aligned_head_dim]
    const float *v,             // [num_kv_heads, T, aligned_head_dim]
    const float *attn_weights,  // [num_heads, T, aligned_context_window]
    float *d_q,                 // [num_heads, T, aligned_head_dim] output
    float *d_k,                 // [num_kv_heads, T, aligned_head_dim] output
    float *d_v,                 // [num_kv_heads, T, aligned_head_dim] output
    float *d_scores,            // [num_heads, T, aligned_context_window] scratch
    int num_heads,
    int num_kv_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window)
{
    const float scale = 1.0f / sqrtf((float)head_dim);
    int T = num_tokens;
    int H = num_heads;
    int H_kv = num_kv_heads;
    int hd = head_dim;
    int ad = aligned_head_dim;
    int aw = aligned_context_window;

    const size_t d_q_elems = (size_t)H * (size_t)T * (size_t)ad;
    const size_t kv_elems = (size_t)H_kv * (size_t)T * (size_t)ad;
    /* Zero the aligned outputs so padded lanes never leak garbage to downstream GEMMs. */
    for (size_t idx = 0; idx < d_q_elems; ++idx) {
        d_q[idx] = 0.0f;
    }
    for (size_t idx = 0; idx < kv_elems; ++idx) {
        d_k[idx] = 0.0f;
        d_v[idx] = 0.0f;
    }

    // Process each query head
    for (int h = 0; h < H; ++h) {
        // Which KV head does this query head use?
        int kv_h = (int)((long long)h * (long long)H_kv / (long long)H);

        // ----------------------------------------------------------------
        // Step 1: d_weights = d_output @ V^T  and  d_v += weights^T @ d_output
        // ----------------------------------------------------------------
        // For each query position i, compute d_weights[i, j] for j <= i
        // and accumulate d_v[j] contributions

        for (int i = 0; i < T; ++i) {
            size_t d_out_base = qkv_index(h, i, 0, T, ad);

            for (int j = 0; j <= i; ++j) {
                size_t v_base = qkv_index(kv_h, j, 0, T, ad);
                size_t w_idx = score_index(h, i, j, aw);
                float w = attn_weights[w_idx];

                // d_weights[h, i, j] = d_output[h, i, :] @ v[kv_h, j, :]^T
                float dot = 0.0f;
                for (int dd = 0; dd < hd; ++dd) {
                    dot += d_output[d_out_base + dd] * v[v_base + dd];
                }
                d_scores[w_idx] = dot;

                // d_v[kv_h, j, :] += weights[h, i, j] * d_output[h, i, :]
                for (int dd = 0; dd < hd; ++dd) {
                    d_v[v_base + dd] += w * d_output[d_out_base + dd];
                }
            }

            // Zero out upper triangle of d_scores
            for (int j = i + 1; j < T; ++j) {
                d_scores[score_index(h, i, j, aw)] = 0.0f;
            }
            /* Scores scratch uses aligned_context_window, zero the padded columns. */
            for (int j = T; j < aw; ++j) {
                d_scores[score_index(h, i, j, aw)] = 0.0f;
            }
        }

        // ----------------------------------------------------------------
        // Step 2: Backward through softmax (in-place on d_scores for this head)
        // ----------------------------------------------------------------
        // d_scores = softmax_backward(d_scores, attn_weights)
        // Formula: d_score[i,j] = w[i,j] * (d_w[i,j] - sum_k(w[i,k] * d_w[i,k]))

        for (int i = 0; i < T; ++i) {
            int base = h * aw * aw + i * aw;

            // Compute dot product: sum_j w[i,j] * d_w[i,j]
            float dot_product = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float wt = attn_weights[base + j];
                float dw = d_scores[base + j];
                dot_product += wt * dw;
            }

            // Apply softmax backward formula
            for (int j = 0; j <= i; ++j) {
                float wt = attn_weights[base + j];
                float dw = d_scores[base + j];
                d_scores[base + j] = wt * (dw - dot_product);
            }
        }

        // ----------------------------------------------------------------
        // Step 3: d_q = d_scores @ K * scale
        //         d_k += d_scores^T @ Q * scale
        // ----------------------------------------------------------------

        for (int i = 0; i < T; ++i) {
            size_t d_q_base = qkv_index(h, i, 0, T, ad);
            size_t q_base = qkv_index(h, i, 0, T, ad);

            // d_q[h, i, :] = sum_j d_scores[h, i, j] * k[kv_h, j, :] * scale
            // d_k[kv_h, j, :] += d_scores[h, i, j] * q[h, i, :] * scale
            for (int j = 0; j <= i; ++j) {
                size_t k_base = qkv_index(kv_h, j, 0, T, ad);
                size_t d_k_base = qkv_index(kv_h, j, 0, T, ad);
                float ds = d_scores[score_index(h, i, j, aw)] * scale;

                for (int dd = 0; dd < hd; ++dd) {
                    d_q[d_q_base + dd] += ds * k[k_base + dd];
                    d_k[d_k_base + dd] += ds * q[q_base + dd];
                }
            }
        }
    }
}

/**
 * Causal attention backward (non-GQA version)
 * @test test_attention_backward.py::TestAttentionBackward::test_backward
 * @test test_attention_backward.py::TestAttentionBackward::test_backward_vs_separate
 * @test test_parity.py::test_attention_backward_parity
 *
 * Non-GQA version where num_heads == num_kv_heads.
 * Simpler than GQA, no head broadcasting needed.
 *
 * After changes: make test && make llamacpp-parity-full
 */
void attention_backward_causal_head_major(
    const float *d_output,
    const float *q,
    const float *k,
    const float *v,
    const float *attn_weights,
    float *d_q,
    float *d_k,
    float *d_v,
    float *d_scores,
    int num_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int aligned_context_window)
{
    attention_backward_causal_head_major_gqa(
        d_output, q, k, v, attn_weights,
        d_q, d_k, d_v, d_scores,
        num_heads, num_heads,  // num_kv_heads == num_heads
        num_tokens, head_dim, aligned_head_dim, aligned_context_window);
}
