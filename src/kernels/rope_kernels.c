/**
 * @file rope_kernels.c
 * @brief RoPE (Rotary Position Embedding) kernels with SIMD
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
 * Applies rotary position embeddings to query and key vectors.
 * Used by Llama, SmolLM, and most modern transformer architectures.
 *
 * Math (Llama-style rotate-half):
 *   Split rotary_dim into two halves (0..half-1, half..rotary_dim-1).
 *   For each position m and index i in [0, half):
 *     x0 = x[i], x1 = x[i + half]
 *     x'[i]       = x0 * cos(m * theta_i) - x1 * sin(m * theta_i)
 *     x'[i+half]  = x0 * sin(m * theta_i) + x1 * cos(m * theta_i)
 *
 *   Where theta_i = 1 / (base^(2i/d)), typically base=10000.
 *
 * Layout:
 *   x: [num_heads, num_tokens, head_dim] head-major
 *   cos_cache, sin_cache: [max_seq_len, rotary_dim/2] precomputed
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ckernel_engine.h"
#include "bf16_utils.h"
#include "ckernel_quant.h"
#include "ggml_runtime_compat.h"

#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef void (*ck_ggml_cpu_init_fn)(void);
typedef struct ggml_context *(*ck_ggml_init_fn)(struct ggml_init_params);
typedef void (*ck_ggml_free_fn)(struct ggml_context *);
typedef struct ggml_tensor *(*ck_ggml_new_tensor_1d_fn)(struct ggml_context *, enum ggml_type, int64_t);
typedef struct ggml_tensor *(*ck_ggml_view_3d_fn)(struct ggml_context *, struct ggml_tensor *, int64_t, int64_t, int64_t, size_t, size_t, size_t);
typedef struct ggml_tensor *(*ck_ggml_rope_multi_inplace_fn)(struct ggml_context *,
                                                             struct ggml_tensor *,
                                                             struct ggml_tensor *,
                                                             struct ggml_tensor *,
                                                             int,
                                                             int[GGML_MROPE_SECTIONS],
                                                             int,
                                                             int,
                                                             float,
                                                             float,
                                                             float,
                                                             float,
                                                             float,
                                                             float);
typedef struct ggml_cgraph *(*ck_ggml_new_graph_fn)(struct ggml_context *);
typedef void (*ck_ggml_build_forward_expand_fn)(struct ggml_cgraph *, struct ggml_tensor *);
typedef enum ggml_status (*ck_ggml_graph_compute_with_ctx_fn)(struct ggml_context *, struct ggml_cgraph *, int);
typedef void *(*ck_ggml_get_data_fn)(const struct ggml_tensor *);
typedef float (*ck_rope_math_f32_fn)(float);
typedef float (*ck_rope_math_f32_binary_fn)(float, float);

static ck_rope_math_f32_fn ck_rope_resolve_system_math_f32(const char *name)
{
#if defined(__linux__)
    static void *libm_handle = NULL;
    if (!libm_handle) {
        libm_handle = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
    }
    if (libm_handle) {
        ck_rope_math_f32_fn fn = (ck_rope_math_f32_fn)dlsym(libm_handle, name);
        if (fn) {
            return fn;
        }
    }
#else
    (void)name;
#endif
    return NULL;
}

/* ICX links libimf ahead of the system math library. Its sinf/cosf results can
 * differ from a GCC-built ggml oracle by one ULP, which is enough to alter Q8
 * blocks downstream. Resolve the platform libm implementation explicitly for
 * the ggml-compatible M-RoPE arithmetic contract. */
static float ck_rope_reference_cosf(float value)
{
    static ck_rope_math_f32_fn fn = NULL;
    if (!fn) {
        fn = ck_rope_resolve_system_math_f32("cosf");
    }
    return fn ? fn(value) : cosf(value);
}

static float ck_rope_reference_sinf(float value)
{
    static ck_rope_math_f32_fn fn = NULL;
    if (!fn) {
        fn = ck_rope_resolve_system_math_f32("sinf");
    }
    return fn ? fn(value) : sinf(value);
}

static float ck_rope_reference_powf(float base, float exponent)
{
    static ck_rope_math_f32_binary_fn fn = NULL;
    if (!fn) {
#if defined(__linux__)
        static void *libm_handle = NULL;
        if (!libm_handle) {
            libm_handle = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
        }
        if (libm_handle) {
            fn = (ck_rope_math_f32_binary_fn)dlsym(libm_handle, "powf");
        }
#endif
    }
    return fn ? fn(base, exponent) : powf(base, exponent);
}

static void ck_rope_ensure_ggml_loaded(void)
{
    static int tried = 0;
    if (tried) {
        return;
    }
    tried = 1;

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
        void *cpu = dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
        if (cpu) {
            return;
        }
    }
    for (int i = 0; dirs[i] != NULL; ++i) {
        snprintf(path_buf, sizeof(path_buf), "%s/libggml-base.so", dirs[i]);
        dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
        snprintf(path_buf, sizeof(path_buf), "%s/libggml.so", dirs[i]);
        dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
        snprintf(path_buf, sizeof(path_buf), "%s/libggml-cpu.so", dirs[i]);
        void *cpu = dlopen(path_buf, RTLD_NOW | RTLD_GLOBAL);
        if (cpu) {
            return;
        }
    }
}

static void *ck_rope_resolve_ggml_symbol(const char *name)
{
    void *sym = dlsym(RTLD_DEFAULT, name);
    if (sym) {
        return sym;
    }
    ck_rope_ensure_ggml_loaded();
    sym = dlsym(RTLD_DEFAULT, name);
    return sym;
}

static ck_ggml_cpu_init_fn ck_resolve_ggml_cpu_init(void) {
    static ck_ggml_cpu_init_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_cpu_init_fn) ck_rope_resolve_ggml_symbol("ggml_cpu_init");
    }
    return fn;
}

static ck_ggml_init_fn ck_resolve_ggml_init(void) {
    static ck_ggml_init_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_init_fn) ck_rope_resolve_ggml_symbol("ggml_init");
    }
    return fn;
}

static ck_ggml_free_fn ck_resolve_ggml_free(void) {
    static ck_ggml_free_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_free_fn) ck_rope_resolve_ggml_symbol("ggml_free");
    }
    return fn;
}

static ck_ggml_new_tensor_1d_fn ck_resolve_ggml_new_tensor_1d(void) {
    static ck_ggml_new_tensor_1d_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_new_tensor_1d_fn) ck_rope_resolve_ggml_symbol("ggml_new_tensor_1d");
    }
    return fn;
}

static ck_ggml_view_3d_fn ck_resolve_ggml_view_3d(void) {
    static ck_ggml_view_3d_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_view_3d_fn) ck_rope_resolve_ggml_symbol("ggml_view_3d");
    }
    return fn;
}

static ck_ggml_rope_multi_inplace_fn ck_resolve_ggml_rope_multi_inplace(void) {
    static ck_ggml_rope_multi_inplace_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_rope_multi_inplace_fn) ck_rope_resolve_ggml_symbol("ggml_rope_multi_inplace");
    }
    return fn;
}

static ck_ggml_new_graph_fn ck_resolve_ggml_new_graph(void) {
    static ck_ggml_new_graph_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_new_graph_fn) ck_rope_resolve_ggml_symbol("ggml_new_graph");
    }
    return fn;
}

static ck_ggml_build_forward_expand_fn ck_resolve_ggml_build_forward_expand(void) {
    static ck_ggml_build_forward_expand_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_build_forward_expand_fn) ck_rope_resolve_ggml_symbol("ggml_build_forward_expand");
    }
    return fn;
}

static ck_ggml_graph_compute_with_ctx_fn ck_resolve_ggml_graph_compute_with_ctx(void) {
    static ck_ggml_graph_compute_with_ctx_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_graph_compute_with_ctx_fn) ck_rope_resolve_ggml_symbol("ggml_graph_compute_with_ctx");
    }
    return fn;
}

static ck_ggml_get_data_fn ck_resolve_ggml_get_data(void) {
    static ck_ggml_get_data_fn fn = NULL;
    if (!fn) {
        fn = (ck_ggml_get_data_fn) ck_rope_resolve_ggml_symbol("ggml_get_data");
    }
    return fn;
}

/* Forward declarations for extended RoPE entry points used by wrappers. */
void rope_forward_with_rotary_dim(float *x,
                                  const float *cos_cache,
                                  const float *sin_cache,
                                  int num_heads,
                                  int num_tokens,
                                  int head_dim,
                                  int aligned_head_dim,
                                  int pos_offset,
                                  int rotary_dim);
void rope_forward_strided_with_rotary_dim(float *x,
                                          const float *cos_cache,
                                          const float *sin_cache,
                                          int num_heads,
                                          int num_tokens,
                                          int head_dim,
                                          int aligned_head_dim,
                                          int pos_offset,
                                          int head_stride_tokens,
                                          int rotary_dim);
void rope_forward_qk_with_rotary_dim(float *q,
                                     float *k,
                                     const float *cos_cache,
                                     const float *sin_cache,
                                     int num_heads,
                                     int num_kv_heads,
                                     int num_tokens,
                                     int head_dim,
                                     int aligned_head_dim,
                                     int pos_offset,
                                     int rotary_dim);
void rope_forward_qk_strided_with_rotary_dim(float *q,
                                             float *k,
                                             const float *cos_cache,
                                             const float *sin_cache,
                                             int num_heads,
                                             int num_kv_heads,
                                             int num_tokens,
                                             int head_dim,
                                             int aligned_head_dim,
                                             int pos_offset,
                                             int q_stride_tokens,
                                             int k_stride_tokens,
                                             int rotary_dim);
void rope_forward_qk_pairwise_with_rotary_dim(float *q,
                                              float *k,
                                              const float *cos_cache,
                                              const float *sin_cache,
                                              int num_heads,
                                              int num_kv_heads,
                                              int num_tokens,
                                              int head_dim,
                                              int aligned_head_dim,
                                              int pos_offset,
                                              int rotary_dim);

/**
 * Precompute RoPE cos/sin cache (split layout: head_dim/2)
 * Legacy layout used before rotary_dim/scaling support.
 *
 * @param cos_cache Output: [max_seq_len, head_dim/2] cos values
 * @param sin_cache Output: [max_seq_len, head_dim/2] sin values
 * @param max_seq_len Maximum sequence length for cache
 * @param head_dim Full head dimension
 * @param base RoPE base frequency (theta)
 */
void rope_precompute_cache_split(float *cos_cache,
                                 float *sin_cache,
                                 int max_seq_len,
                                 int head_dim,
                                 float base)
{
    int half_dim = head_dim / 2;
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            const float exponent = ((float)(2 * i)) / (float)head_dim;
            const float freq_f = 1.0f / powf(base, exponent);
            const float angle_f = (float)pos * freq_f;
            cos_cache[pos * half_dim + i] = cosf(angle_f);
            sin_cache[pos * half_dim + i] = sinf(angle_f);
        }
    }
}

/**
 * Precompute RoPE cos/sin cache with rotary_dim and scaling support
 * @test test_rope.py::TestRoPECache::test_cache_computation
 * @test test_rope.py::TestRoPECache::test_cache_values
 *
 * Precomputes cos(m * theta_i) and sin(m * theta_i) for positions 0..max_seq_len-1.
 * Only computes for first rotary_dim channels; remaining head_dim - rotary_dim
 * channels are NOT rotated (pass through unchanged).
 *
 * Scaling types:
 *   - "none": Standard RoPE
 *   - "linear": Scale positions by 1/scaling_factor
 *   - "dynamic": NTK-aware dynamic scaling
 *   - "yarn": YaRN scaling (beta-based)
 *
 * @param cos_cache Output: [max_seq_len, rotary_dim/2] cos values
 * @param sin_cache Output: [max_seq_len, rotary_dim/2] sin values
 * @param max_seq_len Maximum sequence length for cache
 * @param head_dim Full head dimension (for frequency computation)
 * @param base RoPE base frequency (theta)
 * @param rotary_dim Number of dimensions to rotate (0 = use head_dim)
 * @param scaling_type Scaling type string: "none", "linear", "dynamic", "yarn"
 * @param scaling_factor Scaling factor (1.0 = no scaling)
 *
 * After changes: make test
 */
void rope_precompute_cache(float *cos_cache,
                           float *sin_cache,
                           int max_seq_len,
                           int head_dim,
                           float base,
                           int rotary_dim,
                           const char *scaling_type,
                           float scaling_factor)
{
    // Use rotary_dim = head_dim if not specified
    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        rotary_dim = head_dim;
    }

    // Use no scaling if not specified
    int is_linear_scaling = 0;
    if (scaling_type != NULL && strcmp(scaling_type, "linear") == 0 && scaling_factor > 0.0f && scaling_factor != 1.0f) {
        is_linear_scaling = 1;
    }

    int rotary_half = rotary_dim / 2;

    for (int pos = 0; pos < max_seq_len; ++pos) {
        // Apply linear scaling to position if needed
        float effective_pos = (float)pos;
        if (is_linear_scaling) {
            effective_pos = (float)pos / scaling_factor;
        }

        for (int i = 0; i < rotary_half; ++i) {
            // Match the FP32 reference contract directly. Computing this via
            // long-double log/exp makes the final float depend on the host
            // libm and long-double ABI.
            const float exponent = ((float)(2 * i)) / (float)rotary_dim;
            const float freq_f = 1.0f / powf(base, exponent);
            float angle_f = effective_pos * freq_f;
            cos_cache[pos * rotary_half + i] = cosf(angle_f);
            sin_cache[pos * rotary_half + i] = sinf(angle_f);
        }
    }
}

// Apply RoPE to a single head's Q or K tensor in-place.
// x: [num_tokens, head_dim] for one head
// cos_cache, sin_cache: [max_seq_len, rotary_dim/2]
// pos_offset: starting position (for KV cache continuation)
// rotary_dim: number of dimensions to rotate (0 = use head_dim)
static inline void rope_apply_head(float *x,
                                   const float *cos_cache,
                                   const float *sin_cache,
                                   int num_tokens,
                                   int head_dim,
                                   int aligned_head_dim,
                                   int pos_offset,
                                   int rotary_dim)
{
    // Use head_dim if rotary_dim not specified or invalid
    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        rotary_dim = head_dim;
    }

    int rotary_half = rotary_dim / 2;

    for (int t = 0; t < num_tokens; ++t) {
        int pos = pos_offset + t;
        const float *cos_row = cos_cache + pos * rotary_half;
        const float *sin_row = sin_cache + pos * rotary_half;
        float *x_row = x + (size_t)t * (size_t)aligned_head_dim;

#if defined(__AVX512F__)
        // Process 16 pairs at a time (within rotary_half)
        int i = 0;
        for (; i + 16 <= rotary_half; i += 16) {
            __m512 x0 = _mm512_loadu_ps(&x_row[i]);
            __m512 x1 = _mm512_loadu_ps(&x_row[i + rotary_half]);
            __m512 c = _mm512_loadu_ps(&cos_row[i]);
            __m512 s = _mm512_loadu_ps(&sin_row[i]);

            // x'[i] = x0 * c - x1 * s
            __m512 r0 = _mm512_fmsub_ps(x0, c, _mm512_mul_ps(x1, s));
            // x'[i+half] = x0 * s + x1 * c
            __m512 r1 = _mm512_fmadd_ps(x0, s, _mm512_mul_ps(x1, c));

            _mm512_storeu_ps(&x_row[i], r0);
            _mm512_storeu_ps(&x_row[i + rotary_half], r1);
        }
        // Handle remaining elements in rotary portion
        for (; i < rotary_half; ++i) {
            float x0 = x_row[i];
            float x1 = x_row[i + rotary_half];
            float c = cos_row[i];
            float s = sin_row[i];
            x_row[i] = x0 * c - x1 * s;
            x_row[i + rotary_half] = x0 * s + x1 * c;
        }

#elif defined(__AVX__)
        // Process 8 pairs at a time (within rotary_half)
        int i = 0;
        for (; i + 8 <= rotary_half; i += 8) {
            __m256 x0 = _mm256_loadu_ps(&x_row[i]);
            __m256 x1 = _mm256_loadu_ps(&x_row[i + rotary_half]);
            __m256 c = _mm256_loadu_ps(&cos_row[i]);
            __m256 s = _mm256_loadu_ps(&sin_row[i]);

            // x'[i] = x0 * c - x1 * s (no FMA in AVX1)
            __m256 x0c = _mm256_mul_ps(x0, c);
            __m256 x1s = _mm256_mul_ps(x1, s);
            __m256 r0 = _mm256_sub_ps(x0c, x1s);

            // x'[i+half] = x0 * s + x1 * c
            __m256 x0s = _mm256_mul_ps(x0, s);
            __m256 x1c = _mm256_mul_ps(x1, c);
            __m256 r1 = _mm256_add_ps(x0s, x1c);

            _mm256_storeu_ps(&x_row[i], r0);
            _mm256_storeu_ps(&x_row[i + rotary_half], r1);
        }
        // Handle remaining elements in rotary portion
        for (; i < rotary_half; ++i) {
            float x0 = x_row[i];
            float x1 = x_row[i + rotary_half];
            float c = cos_row[i];
            float s = sin_row[i];
            x_row[i] = x0 * c - x1 * s;
            x_row[i + rotary_half] = x0 * s + x1 * c;
        }

#else
        // Scalar fallback
        for (int i = 0; i < rotary_half; ++i) {
            float x0 = x_row[i];
            float x1 = x_row[i + rotary_half];
            float c = cos_row[i];
            float s = sin_row[i];

            x_row[i] = x0 * c - x1 * s;
            x_row[i + rotary_half] = x0 * s + x1 * c;
        }
#endif

        // Channels [rotary_dim, head_dim) pass through unchanged - nothing to do
        // They're already in place and don't need rotation
    }
}

static inline void rope_apply_head_pairwise(float *x,
                                            const float *cos_cache,
                                            const float *sin_cache,
                                            int num_tokens,
                                            int head_dim,
                                            int aligned_head_dim,
                                            int pos_offset,
                                            int rotary_dim)
{
    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        rotary_dim = head_dim;
    }

    int rotary_half = rotary_dim / 2;

    for (int t = 0; t < num_tokens; ++t) {
        int pos = pos_offset + t;
        const float *cos_row = cos_cache + pos * rotary_half;
        const float *sin_row = sin_cache + pos * rotary_half;
        float *x_row = x + (size_t)t * (size_t)aligned_head_dim;

        for (int i = 0; i < rotary_half; ++i) {
            const int idx0 = 2 * i;
            const int idx1 = idx0 + 1;
            float x0 = x_row[idx0];
            float x1 = x_row[idx1];
            float c = cos_row[i];
            float s = sin_row[i];
            x_row[idx0] = x0 * c - x1 * s;
            x_row[idx1] = x0 * s + x1 * c;
        }
    }
}

static inline void rope_backward_apply_head_pairwise(const float *d_out,
                                                     float *d_x,
                                                     const float *cos_cache,
                                                     const float *sin_cache,
                                                     int num_tokens,
                                                     int head_dim,
                                                     int aligned_head_dim,
                                                     int pos_offset,
                                                     int rotary_dim)
{
    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        rotary_dim = head_dim;
    }

    int rotary_even = rotary_dim - (rotary_dim % 2);
    int rotary_half = rotary_even / 2;

    for (int t = 0; t < num_tokens; ++t) {
        int pos = pos_offset + t;
        const float *cos_row = cos_cache + pos * rotary_half;
        const float *sin_row = sin_cache + pos * rotary_half;
        const float *d_out_row = d_out + (size_t)t * (size_t)aligned_head_dim;
        float *d_x_row = d_x + (size_t)t * (size_t)aligned_head_dim;

        for (int i = 0; i < rotary_half; ++i) {
            const int idx0 = 2 * i;
            const int idx1 = idx0 + 1;
            float d0 = d_out_row[idx0];
            float d1 = d_out_row[idx1];
            float c = cos_row[i];
            float s = sin_row[i];
            d_x_row[idx0] = d0 * c + d1 * s;
            d_x_row[idx1] = -d0 * s + d1 * c;
        }

        for (int i = rotary_even; i < head_dim; ++i) {
            d_x_row[i] = d_out_row[i];
        }
        for (int i = head_dim; i < aligned_head_dim; ++i) {
            d_x_row[i] = 0.0f;
        }
    }
}

/**
 * RoPE forward (head-major layout, in-place)
 * @test test_rope.py::TestRoPEForward::test_rope_forward
 * @test test_rope.py::TestRoPEForward::test_rope_vs_separate
 * @test test_parity.py::test_rope_parity
 *
 * Applies rotary position embeddings in-place to Q or K tensor.
 * x: [num_heads, num_tokens, head_dim] head-major
 *
 * After changes: make test && make llamacpp-parity-full
 */
void rope_forward(float *x,
                  const float *cos_cache,
                  const float *sin_cache,
                  int num_heads,
                  int num_tokens,
                  int head_dim,
                  int aligned_head_dim,
                  int pos_offset)
{
    rope_forward_with_rotary_dim(x, cos_cache, sin_cache, num_heads, num_tokens,
                                 head_dim, aligned_head_dim, pos_offset, head_dim);
}

void rope_forward_with_rotary_dim(float *x,
                                  const float *cos_cache,
                                  const float *sin_cache,
                                  int num_heads,
                                  int num_tokens,
                                  int head_dim,
                                  int aligned_head_dim,
                                  int pos_offset,
                                  int rotary_dim)
{
    size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        rope_apply_head(x + h * head_stride,
                        cos_cache, sin_cache,
                        num_tokens, head_dim, aligned_head_dim, pos_offset, rotary_dim);
    }
}

/**
 * RoPE forward with custom head stride (for KV cache layouts)
 * @test test_rope.py::TestRoPEForward::test_rope_strided
 * @test test_kv_cache_attention.py::TestKVCacheAttention::test_rope_decode
 *
 * Variant with configurable head_stride_tokens for non-contiguous head layouts.
 *
 * After changes: make test
 */
void rope_forward_strided(float *x,
                          const float *cos_cache,
                          const float *sin_cache,
                          int num_heads,
                          int num_tokens,
                          int head_dim,
                          int aligned_head_dim,
                          int pos_offset,
                          int head_stride_tokens)
{
    rope_forward_strided_with_rotary_dim(x, cos_cache, sin_cache, num_heads, num_tokens,
                                         head_dim, aligned_head_dim, pos_offset,
                                         head_stride_tokens, head_dim);
}

void rope_forward_strided_with_rotary_dim(float *x,
                                          const float *cos_cache,
                                          const float *sin_cache,
                                          int num_heads,
                                          int num_tokens,
                                          int head_dim,
                                          int aligned_head_dim,
                                          int pos_offset,
                                          int head_stride_tokens,
                                          int rotary_dim)
{
    size_t head_stride = (size_t)head_stride_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        rope_apply_head(x + h * head_stride,
                        cos_cache, sin_cache,
                        num_tokens, head_dim, aligned_head_dim, pos_offset, rotary_dim);
    }
}

/**
 * RoPE backward (inverse rotation)
 * @test test_rope.py::TestRoPEBackward::test_rope_backward
 * @test test_rope.py::TestRoPEBackward::test_rope_backward_vs_separate
 *
 * RoPE backward: inverse rotation (rotate by -θ).
 * Since cos(-θ) = cos(θ) and sin(-θ) = -sin(θ):
 *   d_x[2i] = d0 * c + d1 * s
 *   d_x[2i+1] = -d0 * s + d1 * c
 *
 * After changes: make test
 */
void rope_backward(const float *d_out,
                   float *d_x,
                   const float *cos_cache,
                   const float *sin_cache,
                   int num_heads,
                   int num_tokens,
                   int head_dim,
                   int aligned_head_dim,
                   int pos_offset)
{
    size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    int half_dim = head_dim / 2;

    for (int h = 0; h < num_heads; ++h) {
        for (int t = 0; t < num_tokens; ++t) {
            int pos = pos_offset + t;
            const float *cos_row = cos_cache + pos * half_dim;
            const float *sin_row = sin_cache + pos * half_dim;

            size_t idx = h * head_stride + (size_t)t * (size_t)aligned_head_dim;
            const float *d_out_row = d_out + idx;
            float *d_x_row = d_x + idx;

#if defined(__AVX512F__)
            int i = 0;
            for (; i + 16 <= half_dim; i += 16) {
                __m512 d0 = _mm512_loadu_ps(&d_out_row[i]);
                __m512 d1 = _mm512_loadu_ps(&d_out_row[i + half_dim]);
                __m512 c = _mm512_loadu_ps(&cos_row[i]);
                __m512 s = _mm512_loadu_ps(&sin_row[i]);

                // Inverse: d_x[i] = d0 * c + d1 * s
                __m512 r0 = _mm512_fmadd_ps(d0, c, _mm512_mul_ps(d1, s));
                // Inverse: d_x[i+half] = -d0 * s + d1 * c
                __m512 r1 = _mm512_fmsub_ps(d1, c, _mm512_mul_ps(d0, s));

                _mm512_storeu_ps(&d_x_row[i], r0);
                _mm512_storeu_ps(&d_x_row[i + half_dim], r1);
            }
            for (; i < half_dim; ++i) {
                float d0 = d_out_row[i];
                float d1 = d_out_row[i + half_dim];
                float c = cos_row[i];
                float s = sin_row[i];
                d_x_row[i] = d0 * c + d1 * s;
                d_x_row[i + half_dim] = -d0 * s + d1 * c;
            }

#elif defined(__AVX__)
            int i = 0;
            for (; i + 8 <= half_dim; i += 8) {
                __m256 d0 = _mm256_loadu_ps(&d_out_row[i]);
                __m256 d1 = _mm256_loadu_ps(&d_out_row[i + half_dim]);
                __m256 c = _mm256_loadu_ps(&cos_row[i]);
                __m256 s = _mm256_loadu_ps(&sin_row[i]);

                // Inverse: d_x[i] = d0 * c + d1 * s
                __m256 d0c = _mm256_mul_ps(d0, c);
                __m256 d1s = _mm256_mul_ps(d1, s);
                __m256 r0 = _mm256_add_ps(d0c, d1s);

                // Inverse: d_x[i+half] = -d0 * s + d1 * c = d1 * c - d0 * s
                __m256 d1c = _mm256_mul_ps(d1, c);
                __m256 d0s = _mm256_mul_ps(d0, s);
                __m256 r1 = _mm256_sub_ps(d1c, d0s);

                _mm256_storeu_ps(&d_x_row[i], r0);
                _mm256_storeu_ps(&d_x_row[i + half_dim], r1);
            }
            for (; i < half_dim; ++i) {
                float d0 = d_out_row[i];
                float d1 = d_out_row[i + half_dim];
                float c = cos_row[i];
                float s = sin_row[i];
                d_x_row[i] = d0 * c + d1 * s;
                d_x_row[i + half_dim] = -d0 * s + d1 * c;
            }

#else
            for (int i = 0; i < half_dim; ++i) {
                float d0 = d_out_row[i];
                float d1 = d_out_row[i + half_dim];
                float c = cos_row[i];
                float s = sin_row[i];

                // Inverse rotation: rotate by -θ
                d_x_row[i] = d0 * c + d1 * s;
                d_x_row[i + half_dim] = -d0 * s + d1 * c;
            }
#endif

            for (int i = head_dim; i < aligned_head_dim; ++i) {
                d_x_row[i] = 0.0f;
            }
        }
    }
}

/**
 * RoPE backward in-place (overwrite with inverse rotation)
 * @test test_rope.py::TestRoPEBackward::test_rope_backward_inplace
 *
 * In-place backward: overwrite d_out with inverse-rotated gradients.
 * Useful when d_x == d_out is acceptable (saves memory).
 *
 * After changes: make test
 */
void rope_backward_inplace(float *d_x,
                           const float *cos_cache,
                           const float *sin_cache,
                           int num_heads,
                           int num_tokens,
                           int head_dim,
                           int aligned_head_dim,
                           int pos_offset)
{
    size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    int half_dim = head_dim / 2;

    for (int h = 0; h < num_heads; ++h) {
        for (int t = 0; t < num_tokens; ++t) {
            int pos = pos_offset + t;
            const float *cos_row = cos_cache + pos * half_dim;
            const float *sin_row = sin_cache + pos * half_dim;

            float *d_row = d_x + h * head_stride + (size_t)t * (size_t)aligned_head_dim;

#if defined(__AVX512F__)
            int i = 0;
            for (; i + 16 <= half_dim; i += 16) {
                __m512 d0 = _mm512_loadu_ps(&d_row[i]);
                __m512 d1 = _mm512_loadu_ps(&d_row[i + half_dim]);
                __m512 c = _mm512_loadu_ps(&cos_row[i]);
                __m512 s = _mm512_loadu_ps(&sin_row[i]);

                __m512 r0 = _mm512_fmadd_ps(d0, c, _mm512_mul_ps(d1, s));
                __m512 r1 = _mm512_fmsub_ps(d1, c, _mm512_mul_ps(d0, s));

                _mm512_storeu_ps(&d_row[i], r0);
                _mm512_storeu_ps(&d_row[i + half_dim], r1);
            }
            for (; i < half_dim; ++i) {
                float d0 = d_row[i];
                float d1 = d_row[i + half_dim];
                float c = cos_row[i];
                float s = sin_row[i];
                d_row[i] = d0 * c + d1 * s;
                d_row[i + half_dim] = -d0 * s + d1 * c;
            }

#elif defined(__AVX__)
            int i = 0;
            for (; i + 8 <= half_dim; i += 8) {
                __m256 d0 = _mm256_loadu_ps(&d_row[i]);
                __m256 d1 = _mm256_loadu_ps(&d_row[i + half_dim]);
                __m256 c = _mm256_loadu_ps(&cos_row[i]);
                __m256 s = _mm256_loadu_ps(&sin_row[i]);

                __m256 d0c = _mm256_mul_ps(d0, c);
                __m256 d1s = _mm256_mul_ps(d1, s);
                __m256 r0 = _mm256_add_ps(d0c, d1s);

                __m256 d1c = _mm256_mul_ps(d1, c);
                __m256 d0s = _mm256_mul_ps(d0, s);
                __m256 r1 = _mm256_sub_ps(d1c, d0s);

                _mm256_storeu_ps(&d_row[i], r0);
                _mm256_storeu_ps(&d_row[i + half_dim], r1);
            }
            for (; i < half_dim; ++i) {
                float d0 = d_row[i];
                float d1 = d_row[i + half_dim];
                float c = cos_row[i];
                float s = sin_row[i];
                d_row[i] = d0 * c + d1 * s;
                d_row[i + half_dim] = -d0 * s + d1 * c;
            }

#else
            for (int i = 0; i < half_dim; ++i) {
                float d0 = d_row[i];
                float d1 = d_row[i + half_dim];
                float c = cos_row[i];
                float s = sin_row[i];

                // Inverse rotation: rotate by -θ
                d_row[i] = d0 * c + d1 * s;
                d_row[i + half_dim] = -d0 * s + d1 * c;
            }
#endif

            for (int i = head_dim; i < aligned_head_dim; ++i) {
                d_row[i] = 0.0f;
            }
        }
    }
}

/**
 * RoPE forward for both Q and K (common inference pattern)
 * @test test_rope.py::TestRoPEForward::test_rope_forward_qk
 * @test test_fused_attention_decode.py::TestFusedAttentionDecode::test_qk_rope
 * @test test_parity.py::test_rope_qk_parity
 *
 * Combined RoPE forward for both Q and K in one call.
 * q: [num_heads, num_tokens, head_dim]
 * k: [num_kv_heads, num_tokens, head_dim]
 *
 * After changes: make test && make llamacpp-parity-full
 */
void rope_forward_qk(float *q,
                     float *k,
                     const float *cos_cache,
                     const float *sin_cache,
                     int num_heads,
                     int num_kv_heads,
                     int num_tokens,
                     int head_dim,
                     int aligned_head_dim,
                     int pos_offset)
{
    rope_forward_qk_with_rotary_dim(q, k, cos_cache, sin_cache, num_heads, num_kv_heads,
                                    num_tokens, head_dim, aligned_head_dim, pos_offset, head_dim);
}

void rope_forward_qk_with_rotary_dim(float *q,
                                     float *k,
                                     const float *cos_cache,
                                     const float *sin_cache,
                                     int num_heads,
                                     int num_kv_heads,
                                     int num_tokens,
                                     int head_dim,
                                     int aligned_head_dim,
                                     int pos_offset,
                                     int rotary_dim)
{
    rope_forward_with_rotary_dim(q, cos_cache, sin_cache, num_heads, num_tokens,
                                head_dim, aligned_head_dim, pos_offset, rotary_dim);
    rope_forward_with_rotary_dim(k, cos_cache, sin_cache, num_kv_heads, num_tokens,
                                head_dim, aligned_head_dim, pos_offset, rotary_dim);
}


static void rope_forward_split_direct_one(float *x,
                                          const float *freq_factors,
                                          int use_freq_factors,
                                          int num_heads,
                                          int num_tokens,
                                          int head_dim,
                                          int aligned_head_dim,
                                          int pos_offset,
                                          int rotary_dim,
                                          float freq_base)
{
    if (!x || num_heads <= 0 || num_tokens <= 0 || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }
    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        rotary_dim = head_dim;
    }
    if (freq_base <= 0.0f) {
        freq_base = 10000.0f;
    }

    /* Split-half rotate_half layout, matching HF/Llama-style RoPE:
     *   [x0...xH, y0...yH] -> [-y0...-yH, x0...xH]
     * Gemma4 is currently the first v8 model that needs direct per-layer
     * theta/rotary_dim control; keep this model-neutral and select it from
     * the IR via rope_layout=split_half + rope_param_mode=per_layer/direct.
     */
    const int rotary_half = rotary_dim / 2;
    const size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const float theta_scale = powf(freq_base, -2.0f / (float)rotary_dim);

    for (int h = 0; h < num_heads; ++h) {
        float *head = x + (size_t)h * head_stride;
        for (int t = 0; t < num_tokens; ++t) {
            const float pos = (float)(pos_offset + t);
            float *x_row = head + (size_t)t * (size_t)aligned_head_dim;
            for (int i = 0; i < rotary_half; ++i) {
                const int idx0 = i;
                const int idx1 = i + rotary_half;
                const float ff = (use_freq_factors && freq_factors) ? freq_factors[i] : 1.0f;
                const float theta = pos * powf(theta_scale, (float)i) / ff;
                const float c = cosf(theta);
                const float sv = sinf(theta);
                const float x0 = x_row[idx0];
                const float x1 = x_row[idx1];
                x_row[idx0] = x0 * c - x1 * sv;
                x_row[idx1] = x1 * c + x0 * sv;
            }
        }
    }
}

void rope_forward_qk_split_direct_f32(float *q,
                                      float *k,
                                      const float *freq_factors,
                                      int use_freq_factors,
                                      int num_heads,
                                      int num_kv_heads,
                                      int num_tokens,
                                      int head_dim,
                                      int aligned_head_dim,
                                      int pos_offset,
                                      int rotary_dim,
                                      float freq_base)
{
    rope_forward_split_direct_one(q, freq_factors, use_freq_factors,
                                  num_heads, num_tokens, head_dim, aligned_head_dim,
                                  pos_offset, rotary_dim, freq_base);
    rope_forward_split_direct_one(k, freq_factors, use_freq_factors,
                                  num_kv_heads, num_tokens, head_dim, aligned_head_dim,
                                  pos_offset, rotary_dim, freq_base);
}

void rope_forward_q_split_direct_f32(float *q,
                                     const float *freq_factors,
                                     int use_freq_factors,
                                     int num_heads,
                                     int num_tokens,
                                     int head_dim,
                                     int aligned_head_dim,
                                     int pos_offset,
                                     int rotary_dim,
                                     float freq_base)
{
    rope_forward_split_direct_one(q, freq_factors, use_freq_factors,
                                  num_heads, num_tokens, head_dim, aligned_head_dim,
                                  pos_offset, rotary_dim, freq_base);
}

void rope_forward_qk_gemma4_direct(float *q,
                                   float *k,
                                   const float *freq_factors,
                                   int use_freq_factors,
                                   int num_heads,
                                   int num_kv_heads,
                                   int num_tokens,
                                   int head_dim,
                                   int aligned_head_dim,
                                   int pos_offset,
                                   int rotary_dim,
                                   float freq_base)
{
    rope_forward_qk_split_direct_f32(q, k, freq_factors, use_freq_factors,
                                    num_heads, num_kv_heads, num_tokens,
                                    head_dim, aligned_head_dim, pos_offset,
                                    rotary_dim, freq_base);
}


static void rope_forward_gemma4v_vision_xy_one(float *x,
                                               int num_heads,
                                               int num_tokens,
                                               int head_dim,
                                               int aligned_head_dim,
                                               int grid_w,
                                               int rotary_dim,
                                               float freq_base)
{
    if (!x || num_heads <= 0 || num_tokens <= 0 || head_dim <= 0 || aligned_head_dim <= 0) {
        return;
    }
    if (grid_w <= 0) {
        grid_w = num_tokens;
    }
    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        rotary_dim = head_dim;
    }
    if (freq_base <= 0.0f) {
        freq_base = 100.0f;
    }

    /* Gemma4V vision uses llama.cpp's 2D NEOX RoPE contract:
     * rotate the first half of the rotary span by patch x, and the second
     * half by patch y. Each half is itself split-half/NEOX rotated.
     */
    const int half_span = rotary_dim / 2;
    const int segment_half = half_span / 2;
    if (segment_half <= 0) {
        return;
    }
    const size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    const float theta_scale = powf(freq_base, -2.0f / (float)half_span);

    for (int h = 0; h < num_heads; ++h) {
        float *head = x + (size_t)h * head_stride;
        for (int t = 0; t < num_tokens; ++t) {
            const int pos_x = t % grid_w;
            const int pos_y = t / grid_w;
            float *row = head + (size_t)t * (size_t)aligned_head_dim;
            for (int i = 0; i < segment_half; ++i) {
                const float inv_freq = powf(theta_scale, (float)i);

                const float theta_x = (float)pos_x * inv_freq;
                const float cx = cosf(theta_x);
                const float sx = sinf(theta_x);
                const int x0i = i;
                const int x1i = i + segment_half;
                const float x0 = row[x0i];
                const float x1 = row[x1i];
                row[x0i] = x0 * cx - x1 * sx;
                row[x1i] = x1 * cx + x0 * sx;

                const float theta_y = (float)pos_y * inv_freq;
                const float cy = cosf(theta_y);
                const float sy = sinf(theta_y);
                const int y0i = half_span + i;
                const int y1i = half_span + i + segment_half;
                const float y0 = row[y0i];
                const float y1 = row[y1i];
                row[y0i] = y0 * cy - y1 * sy;
                row[y1i] = y1 * cy + y0 * sy;
            }
        }
    }
}

void rope_forward_qk_gemma4v_vision_xy(float *q,
                                       float *k,
                                       int num_heads,
                                       int num_kv_heads,
                                       int num_tokens,
                                       int head_dim,
                                       int aligned_head_dim,
                                       int grid_w,
                                       int rotary_dim,
                                       float freq_base)
{
    rope_forward_gemma4v_vision_xy_one(q, num_heads, num_tokens, head_dim, aligned_head_dim, grid_w, rotary_dim, freq_base);
    rope_forward_gemma4v_vision_xy_one(k, num_kv_heads, num_tokens, head_dim, aligned_head_dim, grid_w, rotary_dim, freq_base);
}

void rope_forward_qk_with_rotary_dim_cache_stride(float *q,
                                                  float *k,
                                                  const float *cos_cache,
                                                  const float *sin_cache,
                                                  int num_heads,
                                                  int num_kv_heads,
                                                  int num_tokens,
                                                  int head_dim,
                                                  int aligned_head_dim,
                                                  int pos_offset,
                                                  int rotary_dim,
                                                  int cache_rotary_dim)
{
    if (rotary_dim <= 0 || rotary_dim > head_dim) {
        rotary_dim = head_dim;
    }
    if (cache_rotary_dim < rotary_dim) {
        cache_rotary_dim = rotary_dim;
    }
    const int rotary_half = rotary_dim / 2;
    const int cache_half = cache_rotary_dim / 2;
    const size_t head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        float *head = q + (size_t)h * head_stride;
        for (int t = 0; t < num_tokens; ++t) {
            const int pos = pos_offset + t;
            const float *cos_row = cos_cache + (size_t)pos * (size_t)cache_half;
            const float *sin_row = sin_cache + (size_t)pos * (size_t)cache_half;
            float *x_row = head + (size_t)t * (size_t)aligned_head_dim;
            for (int i = 0; i < rotary_half; ++i) {
                const int idx0 = 2 * i;
                const int idx1 = idx0 + 1;
                const float x0 = x_row[idx0];
                const float x1 = x_row[idx1];
                const float c = cos_row[i];
                const float sv = sin_row[i];
                x_row[idx0] = x0 * c - x1 * sv;
                x_row[idx1] = x0 * sv + x1 * c;
            }
        }
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        float *head = k + (size_t)h * head_stride;
        for (int t = 0; t < num_tokens; ++t) {
            const int pos = pos_offset + t;
            const float *cos_row = cos_cache + (size_t)pos * (size_t)cache_half;
            const float *sin_row = sin_cache + (size_t)pos * (size_t)cache_half;
            float *x_row = head + (size_t)t * (size_t)aligned_head_dim;
            for (int i = 0; i < rotary_half; ++i) {
                const int idx0 = 2 * i;
                const int idx1 = idx0 + 1;
                const float x0 = x_row[idx0];
                const float x1 = x_row[idx1];
                const float c = cos_row[i];
                const float sv = sin_row[i];
                x_row[idx0] = x0 * c - x1 * sv;
                x_row[idx1] = x0 * sv + x1 * c;
            }
        }
    }
}

void rope_forward_qk_pairwise_with_rotary_dim(float *q,
                                              float *k,
                                              const float *cos_cache,
                                              const float *sin_cache,
                                              int num_heads,
                                              int num_kv_heads,
                                              int num_tokens,
                                              int head_dim,
                                              int aligned_head_dim,
                                              int pos_offset,
                                              int rotary_dim)
{
    size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    size_t k_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        rope_apply_head_pairwise(q + (size_t) h * q_head_stride,
                                 cos_cache, sin_cache,
                                 num_tokens, head_dim, aligned_head_dim, pos_offset, rotary_dim);
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        rope_apply_head_pairwise(k + (size_t) h * k_head_stride,
                                 cos_cache, sin_cache,
                                 num_tokens, head_dim, aligned_head_dim, pos_offset, rotary_dim);
    }
}

static float vision_mrope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf((float) n_ctx_orig / (n_rot * 2.0f * (float) M_PI)) / (2.0f * logf(base));
}

static void vision_mrope_yarn_corr_dims(
    int n_dims,
    int n_ctx_orig,
    float freq_base,
    float beta_fast,
    float beta_slow,
    float dims[2]
) {
    float start = floorf(vision_mrope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceilf(vision_mrope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = start < 0.0f ? 0.0f : start;
    dims[1] = end > (float) (n_dims - 1) ? (float) (n_dims - 1) : end;
}

static float vision_mrope_yarn_ramp(float low, float high, int chan) {
    const float y = ((float) chan - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static void vision_mrope_yarn(
    float theta_extrap,
    float freq_scale,
    const float corr_dims[2],
    int chan,
    float ext_factor,
    float attn_factor,
    float *cos_theta,
    float *sin_theta
) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;

    if (ext_factor != 0.0f) {
        const float ramp_mix = vision_mrope_yarn_ramp(corr_dims[0], corr_dims[1], chan) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / fmaxf(freq_scale, 1e-6f));
    }

    *cos_theta = ck_rope_reference_cosf(theta) * mscale;
    *sin_theta = ck_rope_reference_sinf(theta) * mscale;
}

static void text_mrope_yarn(
    float theta_extrap,
    float freq_scale,
    const float corr_dims[2],
    int chan,
    float ext_factor,
    float attn_factor,
    float *cos_theta,
    float *sin_theta
) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;

    if (ext_factor != 0.0f) {
        const float ramp_mix =
            vision_mrope_yarn_ramp(corr_dims[0], corr_dims[1], chan) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / fmaxf(freq_scale, 1e-6f));
    }

    /*
     * Decoder text M-RoPE is part of the compiler-provenance contract.
     * Use the linked compiler math provider, exactly as ggml does. A mixed
     * GCC/ICX run must fail provenance checks instead of substituting glibc
     * trigonometry into an Intel-libimf production graph.
     */
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static inline void mrope_rotate_pair(
    float x0,
    float x1,
    float cos_theta,
    float sin_theta,
    float *out0,
    float *out1
) {
    const float x1_sin = x1 * sin_theta;
    const float x1_cos = x1 * cos_theta;
    *out0 = fmaf(x0, cos_theta, -x1_sin);
    *out1 = fmaf(x0, sin_theta, x1_cos);
}

static void vision_mrope_apply_head(
    float *x,
    const int32_t *positions,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int n_dims,
    const int sections[4],
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow
) {
    if (!x || !positions || num_tokens <= 0 || head_dim <= 0 || aligned_head_dim < head_dim || n_dims <= 0) {
        return;
    }

    int rotary_width = n_dims;
    if (rotary_width > head_dim) {
        rotary_width = head_dim;
    }
    if (rotary_width <= 0 || (rotary_width & 1) != 0 || rotary_width > aligned_head_dim) {
        return;
    }
    const int rope_pairs = rotary_width / 2;

    const int axis_y_pairs = sections[0];
    const int axis_x_pairs = sections[1];
    if (axis_y_pairs <= 0 || axis_x_pairs <= 0 || axis_y_pairs + axis_x_pairs > rope_pairs) {
        return;
    }

    const int num_pos = num_tokens;
    const float theta_scale = ck_rope_reference_powf(freq_base, -2.0f / (float) rope_pairs);
    float corr_dims[2] = {0.0f, (float) (rope_pairs - 1)};
    vision_mrope_yarn_corr_dims(rope_pairs, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    for (int tok = 0; tok < num_tokens; ++tok) {
        float theta_y = (float) positions[tok];
        float theta_x = (float) positions[tok + num_pos];
        float *row = x + (size_t) tok * (size_t) aligned_head_dim;

        for (int pair = 0; pair < rope_pairs; ++pair) {
            const int is_x_axis = pair >= axis_y_pairs && pair < axis_y_pairs + axis_x_pairs;
            if (pair == axis_y_pairs) {
                theta_x = (float) positions[tok + num_pos];
            }
            const float theta = is_x_axis ? theta_x : theta_y;

            float cos_theta = 0.0f;
            float sin_theta = 0.0f;
            vision_mrope_yarn(
                theta,
                freq_scale,
                corr_dims,
                pair * 2,
                ext_factor,
                attn_factor,
                &cos_theta,
                &sin_theta
            );

            const float x0 = row[pair];
            const float x1 = row[pair + rope_pairs];
            mrope_rotate_pair(
                x0,
                x1,
                cos_theta,
                sin_theta,
                &row[pair],
                &row[pair + rope_pairs]
            );

            theta_y *= theta_scale;
            theta_x *= theta_scale;
        }
    }
}

static int explicit_mrope_apply_ggml_exact(
    float *x,
    const int32_t *positions,
    int num_heads,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int n_dims,
    const int sections[4],
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    int rope_type
) {
    ck_ggml_cpu_init_fn ggml_cpu_init_fn = ck_resolve_ggml_cpu_init();
    ck_ggml_init_fn ggml_init_fn = ck_resolve_ggml_init();
    ck_ggml_free_fn ggml_free_fn = ck_resolve_ggml_free();
    ck_ggml_new_tensor_1d_fn ggml_new_tensor_1d_fn = ck_resolve_ggml_new_tensor_1d();
    ck_ggml_view_3d_fn ggml_view_3d_fn = ck_resolve_ggml_view_3d();
    ck_ggml_rope_multi_inplace_fn ggml_rope_multi_inplace_fn = ck_resolve_ggml_rope_multi_inplace();
    ck_ggml_new_graph_fn ggml_new_graph_fn = ck_resolve_ggml_new_graph();
    ck_ggml_build_forward_expand_fn ggml_build_forward_expand_fn = ck_resolve_ggml_build_forward_expand();
    ck_ggml_graph_compute_with_ctx_fn ggml_graph_compute_with_ctx_fn = ck_resolve_ggml_graph_compute_with_ctx();
    ck_ggml_get_data_fn ggml_get_data_fn = ck_resolve_ggml_get_data();

    if (!x || !positions || num_heads <= 0 || num_tokens <= 0 || head_dim <= 0 || aligned_head_dim < head_dim) {
        return 0;
    }
    if (!ggml_cpu_init_fn || !ggml_init_fn || !ggml_free_fn || !ggml_new_tensor_1d_fn ||
        !ggml_view_3d_fn || !ggml_rope_multi_inplace_fn || !ggml_new_graph_fn ||
        !ggml_build_forward_expand_fn || !ggml_graph_compute_with_ctx_fn || !ggml_get_data_fn) {
        return 0;
    }

    ggml_cpu_init_fn();

    const size_t row_bytes = (size_t) aligned_head_dim * sizeof(float);
    const size_t head_bytes = (size_t) num_tokens * row_bytes;
    const int64_t total_elems = (int64_t) num_heads * (int64_t) num_tokens * (int64_t) aligned_head_dim;
    const size_t mem_size =
        (size_t) 16 * 1024 * 1024 +
        (size_t) total_elems * sizeof(float) +
        (size_t) 4 * (size_t) num_tokens * sizeof(int32_t);

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
    struct ggml_tensor *x_base = ggml_new_tensor_1d_fn(ctx, GGML_TYPE_F32, total_elems);
    struct ggml_tensor *pos_base = ggml_new_tensor_1d_fn(ctx, GGML_TYPE_I32, (int64_t) 4 * (int64_t) num_tokens);
    if (!x_base || !pos_base) {
        ggml_free_fn(ctx);
        return 0;
    }

    void *x_base_data = ggml_get_data_fn(x_base);
    void *pos_base_data = ggml_get_data_fn(pos_base);
    if (!x_base_data || !pos_base_data) {
        ggml_free_fn(ctx);
        return 0;
    }

    memcpy(x_base_data, x, (size_t) total_elems * sizeof(float));
    memcpy(pos_base_data, positions, (size_t) 4 * (size_t) num_tokens * sizeof(int32_t));

    struct ggml_tensor *x_view = ggml_view_3d_fn(ctx,
                                                 x_base,
                                                 head_dim,
                                                 num_heads,
                                                 num_tokens,
                                                 head_bytes,
                                                 row_bytes,
                                                 0);
    if (!x_view) {
        ggml_free_fn(ctx);
        return 0;
    }

    int ggml_sections[GGML_MROPE_SECTIONS] = {
        sections[0], sections[1], sections[2], sections[3]
    };
    /* CK's vision contract names the full rotary width. ggml's VISION mode
     * names the split-half stride (the number of frequency pairs). Keep this
     * unit conversion inside the oracle adapter so production kernels and IR
     * continue to carry the canonical full-width value. */
    const int ggml_n_dims = rope_type == GGML_ROPE_TYPE_VISION ? n_dims / 2 : n_dims;
    if (ggml_n_dims <= 0 || (rope_type == GGML_ROPE_TYPE_VISION && (n_dims & 1) != 0)) {
        ggml_free_fn(ctx);
        return 0;
    }
    struct ggml_tensor *rope = ggml_rope_multi_inplace_fn(ctx,
                                                          x_view,
                                                          pos_base,
                                                          NULL,
                                                          ggml_n_dims,
                                                          ggml_sections,
                                                          rope_type,
                                                          n_ctx_orig,
                                                          freq_base,
                                                          freq_scale,
                                                          ext_factor,
                                                          attn_factor,
                                                          beta_fast,
                                                          beta_slow);
    if (!rope) {
        ggml_free_fn(ctx);
        return 0;
    }

    struct ggml_cgraph *gf = ggml_new_graph_fn(ctx);
    if (!gf) {
        ggml_free_fn(ctx);
        return 0;
    }
    ggml_build_forward_expand_fn(gf, rope);
    if (ggml_graph_compute_with_ctx_fn(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        ggml_free_fn(ctx);
        return 0;
    }

    memcpy(x, x_base_data, (size_t) total_elems * sizeof(float));
    ok = 1;
    ggml_free_fn(ctx);
    return ok;
}

static void explicit_mrope_apply_head(
    float *x,
    const int32_t *positions,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int n_dims,
    const int sections[4],
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    int is_imrope
) {
    if (!x || !positions || num_tokens <= 0 || head_dim <= 0 || aligned_head_dim < head_dim || n_dims <= 0) {
        return;
    }

    int rope_dims = n_dims;
    if (rope_dims > head_dim) {
        rope_dims = head_dim;
    }
    rope_dims &= ~1;
    if (rope_dims <= 0) {
        return;
    }
    const int rope_pairs = rope_dims / 2;

    const int num_pos = num_tokens;
    const int sec_w = sections[0] + sections[1];
    const int sec_e = sec_w + sections[2];
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const float theta_scale = powf(freq_base, -2.0f / (float) rope_dims);
    float corr_dims[2] = {0.0f, (float) (rope_dims - 1)};
    vision_mrope_yarn_corr_dims(rope_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    for (int tok = 0; tok < num_tokens; ++tok) {
        float theta_t = (float) positions[tok];
        float theta_h = (float) positions[tok + num_pos];
        float theta_w = (float) positions[tok + 2 * num_pos];
        float theta_e = (float) positions[tok + 3 * num_pos];
        float *row = x + (size_t) tok * (size_t) aligned_head_dim;

        for (int pair = 0; pair < rope_pairs; ++pair) {
            const int sector = sect_dims > 0 ? (pair % sect_dims) : pair;
            if (!is_imrope) {
                if (sector == 0) {
                    theta_t = (float) positions[tok];
                } else if (sector == sections[0]) {
                    theta_h = (float) positions[tok + num_pos];
                } else if (sector == sec_w) {
                    theta_w = (float) positions[tok + 2 * num_pos];
                } else if (sector == sec_e) {
                    theta_e = (float) positions[tok + 3 * num_pos];
                }
            }

            float theta = theta_t;
            if (is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sections[1]) {
                    theta = theta_h;
                } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                    theta = theta_w;
                } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                    theta = theta_t;
                } else {
                    theta = theta_e;
                }
            } else if (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            } else if (sector >= sec_w && sector < sec_e) {
                theta = theta_w;
            } else if (sector >= sec_e) {
                theta = theta_e;
            }

            float cos_theta = 0.0f;
            float sin_theta = 0.0f;
            vision_mrope_yarn(
                theta,
                freq_scale,
                corr_dims,
                pair * 2,
                ext_factor,
                attn_factor,
                &cos_theta,
                &sin_theta
            );

            const float x0 = row[pair];
            const float x1 = row[pair + rope_pairs];
            mrope_rotate_pair(
                x0,
                x1,
                cos_theta,
                sin_theta,
                &row[pair],
                &row[pair + rope_pairs]
            );

            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }
    }
}

static void text_mrope_apply_head(
    float *x,
    int num_tokens,
    int head_dim,
    int aligned_head_dim,
    int pos_offset,
    int n_dims,
    const int sections[4],
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    int is_imrope
) {
    if (!x || num_tokens <= 0 || head_dim <= 0 || aligned_head_dim < head_dim || n_dims <= 0) {
        return;
    }

    int rope_dims = n_dims;
    if (rope_dims > head_dim) {
        rope_dims = head_dim;
    }
    rope_dims &= ~1;
    if (rope_dims <= 0) {
        return;
    }
    const int rope_pairs = rope_dims / 2;

    const int sec_w = sections[0] + sections[1];
    const int sec_e = sec_w + sections[2];
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const float theta_scale = powf(freq_base, -2.0f / (float) rope_dims);
    float corr_dims[2] = {0.0f, (float) (rope_dims - 1)};
    vision_mrope_yarn_corr_dims(rope_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    for (int tok = 0; tok < num_tokens; ++tok) {
        const float base_pos = (float) (pos_offset + tok);
        float theta_t = base_pos;
        float theta_h = base_pos;
        float theta_w = base_pos;
        float theta_e = 0.0f;
        float *row = x + (size_t) tok * (size_t) aligned_head_dim;

        for (int pair = 0; pair < rope_pairs; ++pair) {
            const int sector = sect_dims > 0 ? (pair % sect_dims) : pair;
            float theta = theta_t;
            if (is_imrope && sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (is_imrope && sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (is_imrope && sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } else if (is_imrope) {
                theta = theta_e;
            } else if (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            } else if (sector >= sec_w && sector < sec_e) {
                theta = theta_w;
            } else if (sector >= sec_e) {
                theta = theta_e;
            }

            float cos_theta = 0.0f;
            float sin_theta = 0.0f;
            text_mrope_yarn(
                theta,
                freq_scale,
                corr_dims,
                pair * 2,
                ext_factor,
                attn_factor,
                &cos_theta,
                &sin_theta
            );

            const float x0 = row[pair];
            const float x1 = row[pair + rope_pairs];
            mrope_rotate_pair(
                x0,
                x1,
                cos_theta,
                sin_theta,
                &row[pair],
                &row[pair + rope_pairs]
            );

            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }
    }
}

void mrope_qk_text(float *q,
                   float *k,
                   int num_heads,
                   int num_kv_heads,
                   int num_tokens,
                   int head_dim,
                   int aligned_head_dim,
                   int pos_offset,
                   int n_dims,
                   int section_0,
                   int section_1,
                   int section_2,
                   int section_3,
                   int n_ctx_orig,
                   float freq_base,
                   float freq_scale,
                   float ext_factor,
                   float attn_factor,
                   float beta_fast,
                   float beta_slow)
{
    if (!q || !k || num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }

    const int sections[4] = {section_0, section_1, section_2, section_3};
    const size_t q_head_stride = (size_t) num_tokens * (size_t) aligned_head_dim;
    const size_t k_head_stride = (size_t) num_tokens * (size_t) aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        text_mrope_apply_head(
            q + (size_t) h * q_head_stride,
            num_tokens,
            head_dim,
            aligned_head_dim,
            pos_offset,
            n_dims,
            sections,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            0
        );
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        text_mrope_apply_head(
            k + (size_t) h * k_head_stride,
            num_tokens,
            head_dim,
            aligned_head_dim,
            pos_offset,
            n_dims,
            sections,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            0
        );
    }
}

void mrope_qk_text_imrope(float *q,
                          float *k,
                          int num_heads,
                          int num_kv_heads,
                          int num_tokens,
                          int head_dim,
                          int aligned_head_dim,
                          int pos_offset,
                          int n_dims,
                          int section_0,
                          int section_1,
                          int section_2,
                          int section_3,
                          int n_ctx_orig,
                          float freq_base,
                          float freq_scale,
                          float ext_factor,
                          float attn_factor,
                          float beta_fast,
                          float beta_slow)
{
    if (!q || !k || num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }

    const int sections[4] = {section_0, section_1, section_2, section_3};
    const size_t q_head_stride = (size_t) num_tokens * (size_t) aligned_head_dim;
    const size_t k_head_stride = (size_t) num_tokens * (size_t) aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        text_mrope_apply_head(
            q + (size_t) h * q_head_stride,
            num_tokens,
            head_dim,
            aligned_head_dim,
            pos_offset,
            n_dims,
            sections,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            1
        );
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        text_mrope_apply_head(
            k + (size_t) h * k_head_stride,
            num_tokens,
            head_dim,
            aligned_head_dim,
            pos_offset,
            n_dims,
            sections,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            1
        );
    }
}

void mrope_qk_vision(float *q,
                     float *k,
                     const int32_t *positions,
                     int num_heads,
                     int num_kv_heads,
                     int num_tokens,
                     int head_dim,
                     int aligned_head_dim,
                     int n_dims,
                     int section_0,
                     int section_1,
                     int section_2,
                     int section_3,
                     int n_ctx_orig,
                     float freq_base,
                     float freq_scale,
                     float ext_factor,
                     float attn_factor,
                     float beta_fast,
                     float beta_slow)
{
    if (!q || !k || !positions || num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }

    const int sections[4] = {section_0, section_1, section_2, section_3};

    if (ck_strict_parity_enabled()) {
        if (explicit_mrope_apply_ggml_exact(
                q, positions, num_heads, num_tokens, head_dim, aligned_head_dim, n_dims, sections,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, GGML_ROPE_TYPE_VISION) &&
            explicit_mrope_apply_ggml_exact(
                k, positions, num_kv_heads, num_tokens, head_dim, aligned_head_dim, n_dims, sections,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, GGML_ROPE_TYPE_VISION)) {
            return;
        }
    }

    const size_t q_head_stride = (size_t) num_tokens * (size_t) aligned_head_dim;
    const size_t k_head_stride = (size_t) num_tokens * (size_t) aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        vision_mrope_apply_head(
            q + (size_t) h * q_head_stride,
            positions,
            num_tokens,
            head_dim,
            aligned_head_dim,
            n_dims,
            sections,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow
        );
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        vision_mrope_apply_head(
            k + (size_t) h * k_head_stride,
            positions,
            num_tokens,
            head_dim,
            aligned_head_dim,
            n_dims,
            sections,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow
        );
    }
}


static void ck_mrope_round_storage(float *data, size_t count, int storage_kind)
{
    if (!data) return;
    for (size_t i = 0; i < count; ++i) {
        if (storage_kind == 1) {
            data[i] = bf16_to_float(float_to_bf16(data[i]));
        } else if (storage_kind == 2) {
            data[i] = ck_fp16_to_fp32(ck_fp32_to_fp16(data[i]));
        }
    }
}

#define CK_DEFINE_MROPE_STORAGE_WRAPPER(NAME, STORAGE_KIND) \
void NAME(float *q, float *k, const int32_t *positions, \
          int num_heads, int num_kv_heads, int num_tokens, \
          int head_dim, int aligned_head_dim, int n_dims, \
          int section_0, int section_1, int section_2, int section_3, \
          int n_ctx_orig, float freq_base, float freq_scale, \
          float ext_factor, float attn_factor, float beta_fast, float beta_slow) \
{ \
    mrope_qk_vision(q, k, positions, num_heads, num_kv_heads, num_tokens, \
                    head_dim, aligned_head_dim, n_dims, section_0, section_1, \
                    section_2, section_3, n_ctx_orig, freq_base, freq_scale, \
                    ext_factor, attn_factor, beta_fast, beta_slow); \
    const size_t q_count = (size_t) num_heads * (size_t) num_tokens * (size_t) aligned_head_dim; \
    const size_t k_count = (size_t) num_kv_heads * (size_t) num_tokens * (size_t) aligned_head_dim; \
    ck_mrope_round_storage(q, q_count, STORAGE_KIND); \
    ck_mrope_round_storage(k, k_count, STORAGE_KIND); \
}

CK_DEFINE_MROPE_STORAGE_WRAPPER(mrope_qk_vision_bf16_storage, 1)
CK_DEFINE_MROPE_STORAGE_WRAPPER(mrope_qk_vision_fp16_storage, 2)

void mrope_qk_imrope_positions(float *q,
                               float *k,
                               const int32_t *positions,
                               int num_heads,
                               int num_kv_heads,
                               int num_tokens,
                               int head_dim,
                               int aligned_head_dim,
                               int n_dims,
                               int section_0,
                               int section_1,
                               int section_2,
                               int section_3,
                               int n_ctx_orig,
                               float freq_base,
                               float freq_scale,
                               float ext_factor,
                               float attn_factor,
                               float beta_fast,
                               float beta_slow)
{
    if (!q || !k || !positions || num_heads <= 0 || num_kv_heads <= 0 || num_tokens <= 0) {
        return;
    }

    const int sections[4] = {section_0, section_1, section_2, section_3};

    if (ck_strict_parity_enabled()) {
        const int q_ok = explicit_mrope_apply_ggml_exact(
            q, positions, num_heads, num_tokens, head_dim, aligned_head_dim, n_dims, sections,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, GGML_ROPE_TYPE_IMROPE);
        const int k_ok = explicit_mrope_apply_ggml_exact(
            k, positions, num_kv_heads, num_tokens, head_dim, aligned_head_dim, n_dims, sections,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, GGML_ROPE_TYPE_IMROPE);
        if (q_ok && k_ok) {
            return;
        }
    }

    const size_t q_head_stride = (size_t) num_tokens * (size_t) aligned_head_dim;
    const size_t k_head_stride = (size_t) num_tokens * (size_t) aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        explicit_mrope_apply_head(
            q + (size_t) h * q_head_stride,
            positions,
            num_tokens,
            head_dim,
            aligned_head_dim,
            n_dims,
            sections,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            1
        );
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        explicit_mrope_apply_head(
            k + (size_t) h * k_head_stride,
            positions,
            num_tokens,
            head_dim,
            aligned_head_dim,
            n_dims,
            sections,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            1
        );
    }
}

/**
 * RoPE forward for both Q and K with custom strides (KV cache layouts)
 * @test test_rope.py::TestRoPEForward::test_rope_forward_qk_strided
 * @test test_kv_cache_attention.py::TestKVCacheAttention::test_qk_rope_strided
 *
 * Combined QK RoPE with configurable strides for KV cache layouts.
 *
 * After changes: make test
 */
void rope_forward_qk_strided(float *q,
                             float *k,
                             const float *cos_cache,
                             const float *sin_cache,
                             int num_heads,
                             int num_kv_heads,
                             int num_tokens,
                             int head_dim,
                             int aligned_head_dim,
                             int pos_offset,
                             int q_stride_tokens,
                             int k_stride_tokens)
{
    rope_forward_qk_strided_with_rotary_dim(q, k, cos_cache, sin_cache, num_heads, num_kv_heads,
                                           num_tokens, head_dim, aligned_head_dim, pos_offset,
                                           q_stride_tokens, k_stride_tokens, head_dim);
}

void rope_forward_qk_strided_with_rotary_dim(float *q,
                                             float *k,
                                             const float *cos_cache,
                                             const float *sin_cache,
                                             int num_heads,
                                             int num_kv_heads,
                                             int num_tokens,
                                             int head_dim,
                                             int aligned_head_dim,
                                             int pos_offset,
                                             int q_stride_tokens,
                                             int k_stride_tokens,
                                             int rotary_dim)
{
    rope_forward_strided_with_rotary_dim(q, cos_cache, sin_cache, num_heads, num_tokens,
                                       head_dim, aligned_head_dim, pos_offset,
                                       q_stride_tokens, rotary_dim);
    rope_forward_strided_with_rotary_dim(k, cos_cache, sin_cache, num_kv_heads, num_tokens,
                                       head_dim, aligned_head_dim, pos_offset,
                                       k_stride_tokens, rotary_dim);
}

/**
 * RoPE backward for both dQ and dK
 * @test test_rope.py::TestRoPEBackward::test_rope_backward_qk
 *
 * Combined RoPE backward for both dQ and dK gradients.
 *
 * After changes: make test
 */
void rope_backward_qk(const float *d_q_out,
                      const float *d_k_out,
                      float *d_q,
                      float *d_k,
                      const float *cos_cache,
                      const float *sin_cache,
                      int num_heads,
                      int num_kv_heads,
                      int num_tokens,
                      int head_dim,
                      int aligned_head_dim,
                      int pos_offset)
{
    rope_backward(d_q_out, d_q, cos_cache, sin_cache, num_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
    rope_backward(d_k_out, d_k, cos_cache, sin_cache, num_kv_heads, num_tokens, head_dim, aligned_head_dim, pos_offset);
}

void rope_backward_qk_pairwise_with_rotary_dim(const float *d_q_out,
                                               const float *d_k_out,
                                               float *d_q,
                                               float *d_k,
                                               const float *cos_cache,
                                               const float *sin_cache,
                                               int num_heads,
                                               int num_kv_heads,
                                               int num_tokens,
                                               int head_dim,
                                               int aligned_head_dim,
                                               int pos_offset,
                                               int rotary_dim)
{
    size_t q_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;
    size_t k_head_stride = (size_t)num_tokens * (size_t)aligned_head_dim;

    for (int h = 0; h < num_heads; ++h) {
        rope_backward_apply_head_pairwise(
            d_q_out + (size_t)h * q_head_stride,
            d_q + (size_t)h * q_head_stride,
            cos_cache,
            sin_cache,
            num_tokens,
            head_dim,
            aligned_head_dim,
            pos_offset,
            rotary_dim
        );
    }

    for (int h = 0; h < num_kv_heads; ++h) {
        rope_backward_apply_head_pairwise(
            d_k_out + (size_t)h * k_head_stride,
            d_k + (size_t)h * k_head_stride,
            cos_cache,
            sin_cache,
            num_tokens,
            head_dim,
            aligned_head_dim,
            pos_offset,
            rotary_dim
        );
    }
}
