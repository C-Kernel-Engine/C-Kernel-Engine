/**
 * @file deltanet_kernels.c
 * @brief FP32 Gated DeltaNet kernels for Qwen3.5-style recurrent attention.
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
 * This file implements the single-token recurrent update used by the
 * qwen3next / Gated DeltaNet path in llama.cpp.
 *
 * Per head, matching llama.cpp qwen35/qwen3next autoregressive DeltaNet:
 *   q_scaled = q / sqrt(state_dim)   // q and k arrive pre-normalized
 *   k_hat    = k
 *   beta_s   = sigmoid(beta)
 *   gate     = exp(g)
 *   S        = gate * S_prev
 *   kv_mem   = S^T * k_hat
 *   delta    = (v - kv_mem) * beta_s
 *   S_new    = S + outer(k_hat, delta)
 *   out      = S_new^T * q_scaled
 *
 * Design:
 *   - *_ref is the scalar reference implementation.
 *   - *_avx keeps a simple 1-row vector walk.
 *   - *_avx2 precomputes scaled q rows and unrolls the state sweep in
 *     row pairs to reduce loop overhead and layout churn.
 *   - The public dispatcher selects the best compiled ISA unless strict parity
 *     is enabled, in which case it falls back to *_ref.
 */

#include "bf16_utils.h"
#include "ckernel_engine.h"

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#define CK_DELTANET_MAX_STACK_DIM 4096
#define CK_DELTANET_LLAMA_CHUNK_SIZE 64
#define CK_DELTANET_LLAMA_CHUNK_MAX_DIM 256

#if defined(__GNUC__) || defined(__clang__)
#define CK_DELTANET_NOINLINE __attribute__((noinline))
#else
#define CK_DELTANET_NOINLINE
#endif

typedef float (*ck_deltanet_libm_f32_fn)(float);
static ck_deltanet_libm_f32_fn ck_deltanet_llama_expf = NULL;
static void *ck_deltanet_libm_handle = NULL;
static pthread_once_t ck_deltanet_libm_once = PTHREAD_ONCE_INIT;

static void ck_bind_deltanet_llama_libm(void)
{
    ck_deltanet_libm_handle = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
    if (ck_deltanet_libm_handle) {
        ck_deltanet_llama_expf =
            (ck_deltanet_libm_f32_fn)dlsym(ck_deltanet_libm_handle, "expf");
    }
    if (!ck_deltanet_llama_expf) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: llama.cpp DeltaNet requires "
                "expf from libm.so.6\n");
        abort();
    }
}

static inline float ck_deltanet_sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float ck_deltanet_llama_sigmoidf(float x)
{
    pthread_once(&ck_deltanet_libm_once, ck_bind_deltanet_llama_libm);
    return 1.0f / (1.0f + ck_deltanet_llama_expf(-x));
}

#if defined(__AVX512F__)
typedef __m512 (*ck_deltanet_sleef_expf16_fn)(__m512);
static ck_deltanet_sleef_expf16_fn ck_deltanet_pytorch_expf16 = NULL;
static void *ck_deltanet_sleef_handle = NULL;
#endif

typedef void (*ck_deltanet_mkl_vsexp_fn)(int, const float *, float *);
static ck_deltanet_mkl_vsexp_fn ck_deltanet_pytorch_vsexp = NULL;
static void *ck_deltanet_mkl_handle = NULL;
static pthread_once_t ck_deltanet_pytorch_primitives_once = PTHREAD_ONCE_INIT;

static void ck_bind_deltanet_pytorch_primitives(void)
{
    const char *mkl_library = getenv("CK_MKL_LIBRARY");
    if (mkl_library && *mkl_library) {
        ck_deltanet_mkl_handle = dlopen(mkl_library, RTLD_NOW | RTLD_LOCAL);
        if (ck_deltanet_mkl_handle) {
            ck_deltanet_pytorch_vsexp =
                (ck_deltanet_mkl_vsexp_fn)dlsym(
                    ck_deltanet_mkl_handle, "vsExp");
        }
    } else {
        ck_deltanet_pytorch_vsexp =
            (ck_deltanet_mkl_vsexp_fn)dlsym(RTLD_DEFAULT, "vsExp");
    }

#if defined(__AVX512F__)
    const char *library = getenv("CK_SLEEF_LIBRARY");
    if (library && *library) {
        ck_deltanet_sleef_handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        if (ck_deltanet_sleef_handle) {
            ck_deltanet_pytorch_expf16 =
                (ck_deltanet_sleef_expf16_fn)dlsym(
                    ck_deltanet_sleef_handle, "Sleef_expf16_u10");
        }
    } else {
        ck_deltanet_pytorch_expf16 =
            (ck_deltanet_sleef_expf16_fn)dlsym(
                RTLD_DEFAULT, "Sleef_expf16_u10");
    }
#endif
}

static void ck_deltanet_pytorch_gate_values(const float *g,
                                             const float *beta,
                                             float *gate_values,
                                             float *beta_values,
                                             int num_heads)
{
    pthread_once(
        &ck_deltanet_pytorch_primitives_once,
        ck_bind_deltanet_pytorch_primitives);
    if (!ck_deltanet_pytorch_vsexp) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: PyTorch DeltaNet requires "
                "MKL vsExp; set CK_MKL_LIBRARY\n");
        abort();
    }
    ck_deltanet_pytorch_vsexp(num_heads, g, gate_values);

    int h = 0;
#if defined(__AVX512F__)
    if (ck_deltanet_pytorch_expf16) {
        const __m512 one = _mm512_set1_ps(1.0f);
        for (; h + 15 < num_heads; h += 16) {
            const __m512 bv = _mm512_loadu_ps(beta + h);
            const __m512 beta_exp = ck_deltanet_pytorch_expf16(
                _mm512_sub_ps(_mm512_setzero_ps(), bv));
            const __m512 beta_sigmoid = _mm512_div_ps(
                one, _mm512_add_ps(one, beta_exp));
            float beta_lanes[16];
            _mm512_storeu_ps(beta_lanes, beta_sigmoid);
            for (int lane = 0; lane < 16; ++lane) {
                beta_values[h + lane] = bf16_to_float(
                    float_to_bf16(beta_lanes[lane]));
            }
        }
    }
#endif
    for (; h < num_heads; ++h) {
        beta_values[h] = bf16_to_float(float_to_bf16(
            ck_deltanet_sigmoidf(beta[h])));
    }
}

void gated_deltanet_pytorch_gate_values_debug(const float *g,
                                              const float *beta,
                                              float *gate_values,
                                              float *beta_values,
                                              int num_heads)
{
    if (!g || !beta || !gate_values || !beta_values || num_heads <= 0 ||
        num_heads > CK_DELTANET_MAX_STACK_DIM) {
        return;
    }
    ck_deltanet_pytorch_gate_values(
        g, beta, gate_values, beta_values, num_heads);
}

void gated_deltanet_autoregressive_forward_ref(const float *q,
                                               const float *k,
                                               const float *v,
                                               const float *g,
                                               const float *beta,
                                               const float *state_in,
                                               float *state_out,
                                               float *out,
                                               int num_heads,
                                               int state_dim,
                                               float norm_eps);

#if defined(__AVX2__)
#if defined(__AVX512F__)
static inline float ck_deltanet_gcc_reduce_add_ps(__m512 value)
{
    const __m256 hi = _mm512_extractf32x8_ps(value, 1);
    const __m256 lo = _mm512_castps512_ps256(value);
    const __m256 sum8 = _mm256_add_ps(hi, lo);
    const __m128 hi4 = _mm256_extractf128_ps(sum8, 1);
    const __m128 lo4 = _mm256_castps256_ps128(sum8);
    const __m128 sum4 = _mm_add_ps(hi4, lo4);
    const __m128 swapped = _mm_shuffle_ps(sum4, sum4, _MM_SHUFFLE(1, 0, 3, 2));
    const __m128 sum2 = _mm_add_ps(sum4, swapped);
    return _mm_cvtss_f32(_mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1)));
}
#endif

static CK_DELTANET_NOINLINE float ck_deltanet_llama_avx2_dot(
        const float *x, const float *y, int n)
{
#if defined(__AVX512F__)
    const int np = n & ~63;
    __m512 sum[4] = {
        _mm512_setzero_ps(), _mm512_setzero_ps(),
        _mm512_setzero_ps(), _mm512_setzero_ps()
    };
    for (int i = 0; i < np; i += 64) {
        for (int j = 0; j < 4; ++j) {
            const __m512 xv = _mm512_loadu_ps(x + i + j * 16);
            const __m512 yv = _mm512_loadu_ps(y + i + j * 16);
            sum[j] = _mm512_fmadd_ps(xv, yv, sum[j]);
        }
    }
    sum[0] = _mm512_add_ps(sum[0], sum[2]);
    sum[1] = _mm512_add_ps(sum[1], sum[3]);
    sum[0] = _mm512_add_ps(sum[0], sum[1]);
    float result = ck_deltanet_gcc_reduce_add_ps(sum[0]);
#else
    const int np = n & ~31;
    __m256 sum[4] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps()
    };
    for (int i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; ++j) {
            const __m256 xv = _mm256_loadu_ps(x + i + j * 8);
            const __m256 yv = _mm256_loadu_ps(y + i + j * 8);
#if defined(__FMA__)
            sum[j] = _mm256_fmadd_ps(xv, yv, sum[j]);
#else
            sum[j] = _mm256_add_ps(_mm256_mul_ps(xv, yv), sum[j]);
#endif
        }
    }
    sum[0] = _mm256_add_ps(sum[0], sum[2]);
    sum[1] = _mm256_add_ps(sum[1], sum[3]);
    sum[0] = _mm256_add_ps(sum[0], sum[1]);
    const __m128 halves = _mm_add_ps(
        _mm256_castps256_ps128(sum[0]), _mm256_extractf128_ps(sum[0], 1));
    const __m128 pairs = _mm_hadd_ps(halves, halves);
    float result = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
#endif
    for (int i = np; i < n; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

static inline float ck_deltanet_llama_scale(int state_dim)
{
#if defined(__SSE__)
    const __m128 dim = _mm_set_ss((float) state_dim);
    return _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(1.0f), _mm_sqrt_ss(dim)));
#else
    return 1.0f / sqrtf((float) state_dim);
#endif
}

static inline __m256 ck_deltanet_fmadd8(__m256 a, __m256 b, __m256 acc)
{
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, acc);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), acc);
#endif
}

/*
 * llama.cpp evaluates multi-token scalar-gate DeltaNet in 64-token chunks.
 * This is algebraically equivalent to the recurrent update above, but its
 * reduction tree is observably different after many recurrent layers.
 *
 * Layouts below are row-major:
 *   token vectors  [chunk, state_dim]
 *   recurrent S    [key_dim, value_dim]
 *   temporal mats  [chunk, chunk]
 *
 * The fixed upper bound keeps the provider allocation-free. Qwen3.5/3.6 use
 * state_dim=128; unsupported shapes retain the sequential fallback.
 */
static void gated_deltanet_llama_chunk64_head(
                                       const float *q,
                                       const float *k,
                                       const float *v,
                                       const float *g,
                                       const float *beta,
                                       const float *state_in,
                                       float *state_out,
                                       float *out,
                                       int rows,
                                       int num_heads,
                                       int group_count,
                                       int head,
                                       int state_dim)
{
    enum { C = CK_DELTANET_LLAMA_CHUNK_SIZE };
    const int group = head % group_count;
    const size_t qk_row_stride = (size_t)group_count * (size_t)state_dim;
    const size_t value_row_stride = (size_t)num_heads * (size_t)state_dim;
    const size_t gate_row_stride = (size_t)num_heads;
    const size_t state_count = (size_t)state_dim * (size_t)state_dim;
    const float scale = 1.0f / sqrtf((float)state_dim);

    float gcum[C];
    float beta_chunk[C];
    float decay[C * C];
    float transform[C * C];
    float q_chunk[C * CK_DELTANET_LLAMA_CHUNK_MAX_DIM];
    float k_chunk[C * CK_DELTANET_LLAMA_CHUNK_MAX_DIM];
    float value_beta[C * CK_DELTANET_LLAMA_CHUNK_MAX_DIM];
    float k_cumdecay[C * CK_DELTANET_LLAMA_CHUNK_MAX_DIM];
    float v_new[C * CK_DELTANET_LLAMA_CHUNK_MAX_DIM];
    float matrix_work[C * CK_DELTANET_LLAMA_CHUNK_MAX_DIM];
    float work_row[C];
    float gate_exp[C];

    float *state = state_out + (size_t)head * state_count;
    const float *initial = state_in + (size_t)head * state_count;
    if (state != initial) {
        for (size_t i = 0; i < state_count; ++i) {
            state[i] = initial[i];
        }
    }

    for (int chunk_start = 0; chunk_start < rows; chunk_start += C) {
        const int valid = rows - chunk_start < C ? rows - chunk_start : C;

        for (int i = 0; i < C; ++i) {
            const int token = chunk_start + i;
            const int present = i < valid;
            const float gate_value = present
                ? g[(size_t)token * gate_row_stride + (size_t)head]
                : 0.0f;
            gcum[i] = gate_value + (i ? gcum[i - 1] : 0.0f);
            gate_exp[i] = expf(gcum[i]);

            float beta_value = 0.0f;
            const float *q_src = NULL;
            const float *k_src = NULL;
            const float *v_src = NULL;
            if (present) {
                beta_value = ck_deltanet_sigmoidf(
                    beta[(size_t)token * gate_row_stride + (size_t)head]);
                q_src = q + (size_t)token * qk_row_stride +
                    (size_t)group * (size_t)state_dim;
                k_src = k + (size_t)token * qk_row_stride +
                    (size_t)group * (size_t)state_dim;
                v_src = v + (size_t)token * value_row_stride +
                    (size_t)head * (size_t)state_dim;
            }
            beta_chunk[i] = beta_value;
            for (int d = 0; d < state_dim; ++d) {
                const size_t offset = (size_t)i * (size_t)state_dim + (size_t)d;
                q_chunk[offset] = present ? q_src[d] * scale : 0.0f;
                k_chunk[offset] = present ? k_src[d] : 0.0f;
                value_beta[offset] = present ? v_src[d] * beta_value : 0.0f;
                k_cumdecay[offset] =
                    present ? k_src[d] * beta_value * gate_exp[i] : 0.0f;
            }
        }

        /*
         * kb[i,j] = beta_i * <k_i,k_j> * exp(gcum_i-gcum_j).
         * llama solves (I + tril(kb,-1)) X = -tril(kb,-1), then adds I.
         * Forward substitution is performed a column at a time to retain the
         * same dependency order as ggml_solve_tri.
         */
        for (int i = 0; i < C; ++i) {
            for (int j = 0; j < C; ++j) {
                const float d = j <= i ? expf(gcum[i] - gcum[j]) : 0.0f;
                decay[(size_t)i * C + (size_t)j] = d;
                transform[(size_t)i * C + (size_t)j] = j < i
                    ? ck_deltanet_llama_avx2_dot(
                          k_chunk + (size_t)i * (size_t)state_dim,
                          k_chunk + (size_t)j * (size_t)state_dim,
                          state_dim) * beta_chunk[i] * d
                    : 0.0f;
            }
        }
        for (int i = 0; i < C; ++i) {
            for (int j = 0; j < i; ++j) {
                work_row[j] = transform[(size_t)i * C + (size_t)j];
            }
            for (int col = 0; col <= i; ++col) {
                const float rhs = i == col ? 1.0f : 0.0f;
                float solved = rhs;
                for (int j = 0; j < i; ++j) {
                    solved -= work_row[j] * transform[(size_t)j * C + (size_t)col];
                }
                transform[(size_t)i * C + (size_t)col] = solved;
            }
        }

        /*
         * transformed V and cumulative-decay K use the solved temporal
         * transform.  Subtract the contribution of the incoming state to form
         * V_new, matching llama's v_t_new node.
         */
        for (int i = 0; i < C; ++i) {
            int d = 0;
            for (; d + 7 < state_dim; d += 8) {
                __m256 value_sum = _mm256_setzero_ps();
                __m256 key_sum = _mm256_setzero_ps();
                for (int j = 0; j < C; ++j) {
                    const __m256 coefficient = _mm256_set1_ps(
                        transform[(size_t)i * C + (size_t)j]);
                    value_sum = ck_deltanet_fmadd8(
                        coefficient,
                        _mm256_loadu_ps(value_beta +
                            (size_t)j * (size_t)state_dim + (size_t)d),
                        value_sum);
                    key_sum = ck_deltanet_fmadd8(
                        coefficient,
                        _mm256_loadu_ps(k_cumdecay +
                            (size_t)j * (size_t)state_dim + (size_t)d),
                        key_sum);
                }
                _mm256_storeu_ps(v_new +
                    (size_t)i * (size_t)state_dim + (size_t)d, value_sum);
                _mm256_storeu_ps(matrix_work +
                    (size_t)i * (size_t)state_dim + (size_t)d, key_sum);
            }
            for (; d < state_dim; ++d) {
                float value_sum = 0.0f;
                float key_sum = 0.0f;
                for (int j = 0; j < C; ++j) {
                    const float coefficient = transform[(size_t)i * C + (size_t)j];
                    value_sum += coefficient *
                        value_beta[(size_t)j * (size_t)state_dim + (size_t)d];
                    key_sum += coefficient *
                        k_cumdecay[(size_t)j * (size_t)state_dim + (size_t)d];
                }
                v_new[(size_t)i * (size_t)state_dim + (size_t)d] = value_sum;
                matrix_work[(size_t)i * (size_t)state_dim + (size_t)d] = key_sum;
            }
        }
        for (int i = 0; i < C; ++i) {
            int d = 0;
            for (; d + 7 < state_dim; d += 8) {
                __m256 v_prime = _mm256_setzero_ps();
                for (int r = 0; r < state_dim; ++r) {
                    v_prime = ck_deltanet_fmadd8(
                        _mm256_set1_ps(matrix_work[
                            (size_t)i * (size_t)state_dim + (size_t)r]),
                        _mm256_loadu_ps(state +
                            (size_t)r * (size_t)state_dim + (size_t)d),
                        v_prime);
                }
                float *dst = v_new +
                    (size_t)i * (size_t)state_dim + (size_t)d;
                _mm256_storeu_ps(dst, _mm256_sub_ps(_mm256_loadu_ps(dst), v_prime));
            }
            for (; d < state_dim; ++d) {
                float v_prime = 0.0f;
                for (int r = 0; r < state_dim; ++r) {
                    v_prime +=
                        matrix_work[(size_t)i * (size_t)state_dim + (size_t)r] *
                        state[(size_t)r * (size_t)state_dim + (size_t)d];
                }
                v_new[(size_t)i * (size_t)state_dim + (size_t)d] -= v_prime;
            }
        }

        for (int i = 0; i < valid; ++i) {
            float *out_token = out +
                (size_t)(chunk_start + i) * value_row_stride +
                (size_t)head * (size_t)state_dim;
            for (int j = 0; j <= i; ++j) {
                work_row[j] = ck_deltanet_llama_avx2_dot(
                    q_chunk + (size_t)i * (size_t)state_dim,
                    k_chunk + (size_t)j * (size_t)state_dim,
                    state_dim) * decay[(size_t)i * C + (size_t)j];
            }
            int d = 0;
            for (; d + 7 < state_dim; d += 8) {
                __m256 result = _mm256_setzero_ps();
                for (int r = 0; r < state_dim; ++r) {
                    const float q_gate =
                        q_chunk[(size_t)i * (size_t)state_dim + (size_t)r] *
                        gate_exp[i];
                    result = ck_deltanet_fmadd8(
                        _mm256_set1_ps(q_gate),
                        _mm256_loadu_ps(state +
                            (size_t)r * (size_t)state_dim + (size_t)d),
                        result);
                }
                for (int j = 0; j <= i; ++j) {
                    result = ck_deltanet_fmadd8(
                        _mm256_set1_ps(work_row[j]),
                        _mm256_loadu_ps(v_new +
                            (size_t)j * (size_t)state_dim + (size_t)d),
                        result);
                }
                _mm256_storeu_ps(out_token + d, result);
            }
            for (; d < state_dim; ++d) {
                float result = 0.0f;
                for (int r = 0; r < state_dim; ++r) {
                    result +=
                        q_chunk[(size_t)i * (size_t)state_dim + (size_t)r] *
                        gate_exp[i] *
                        state[(size_t)r * (size_t)state_dim + (size_t)d];
                }
                for (int j = 0; j <= i; ++j) {
                    result += work_row[j] *
                        v_new[(size_t)j * (size_t)state_dim + (size_t)d];
                }
                out_token[d] = result;
            }
        }

        const float last_decay = gate_exp[C - 1];
        for (int i = 0; i < C; ++i) {
            gate_exp[i] = expf(gcum[C - 1] - gcum[i]);
        }
        for (int r = 0; r < state_dim; ++r) {
            int d = 0;
            for (; d + 7 < state_dim; d += 8) {
                float *state_row = state +
                    (size_t)r * (size_t)state_dim + (size_t)d;
                __m256 updated = _mm256_mul_ps(
                    _mm256_loadu_ps(state_row), _mm256_set1_ps(last_decay));
                for (int i = 0; i < C; ++i) {
                    const float key_gate =
                        k_chunk[(size_t)i * (size_t)state_dim + (size_t)r] *
                        gate_exp[i];
                    updated = ck_deltanet_fmadd8(
                        _mm256_set1_ps(key_gate),
                        _mm256_loadu_ps(v_new +
                            (size_t)i * (size_t)state_dim + (size_t)d),
                        updated);
                }
                _mm256_storeu_ps(state_row, updated);
            }
            for (; d < state_dim; ++d) {
                float updated =
                    state[(size_t)r * (size_t)state_dim + (size_t)d] * last_decay;
                for (int i = 0; i < C; ++i) {
                    updated +=
                        k_chunk[(size_t)i * (size_t)state_dim + (size_t)r] *
                        gate_exp[i] *
                        v_new[(size_t)i * (size_t)state_dim + (size_t)d];
                }
                state[(size_t)r * (size_t)state_dim + (size_t)d] = updated;
            }
        }
    }
}

#endif

static void gated_deltanet_llama_avx2_grouped_forward_impl(
                                       const float *q,
                                       const float *k,
                                       const float *v,
                                       const float *g,
                                       const float *beta,
                                       const float *state_in,
                                       float *state_out,
                                       float *out,
                                       int num_heads,
                                       int group_count,
                                       int state_dim,
                                       float norm_eps,
                                       int head_begin,
                                       int head_end,
                                       int pytorch_bf16_boundaries)
{
#if defined(__AVX2__)
    (void) norm_eps;
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out ||
        num_heads <= 0 || num_heads > CK_DELTANET_MAX_STACK_DIM ||
        group_count <= 0 || num_heads % group_count != 0 ||
        state_dim <= 0 || state_dim > CK_DELTANET_MAX_STACK_DIM ||
        head_begin < 0 || head_end < head_begin || head_end > num_heads) {
        return;
    }
    const float scale = ck_deltanet_llama_scale(state_dim);
    const size_t vector_stride = (size_t) state_dim;
    const size_t state_stride = (size_t) state_dim * (size_t) state_dim;
    float column[CK_DELTANET_MAX_STACK_DIM];
    float q_scaled[CK_DELTANET_MAX_STACK_DIM];

    for (int h = head_begin; h < head_end; ++h) {
        /* llama.cpp ggml_repeat_4d tiles compact Q/K heads (0..G-1 repeated),
         * while the PyTorch Qwen3-Next reference uses repeat_interleave so
         * each compact head owns H/G adjacent value heads.  These layouts
         * are distinct numerical contracts even though their buffer shapes
         * are identical. */
        const int group = pytorch_bf16_boundaries
            ? h / (num_heads / group_count)
            : h % group_count;
        const float *q_head = q + (size_t) group * vector_stride;
        const float *k_head = k + (size_t) group * vector_stride;
        const float *v_head = v + (size_t) h * vector_stride;
        const float *state_prev = state_in + (size_t) h * state_stride;
        float *state_cur = state_out + (size_t) h * state_stride;
        float *out_head = out + (size_t) h * vector_stride;
        float gate;
        float beta_s;
        if (pytorch_bf16_boundaries) {
            gate = expf(g[h]);
            beta_s = ck_deltanet_sigmoidf(beta[h]);
        } else {
            pthread_once(&ck_deltanet_libm_once, ck_bind_deltanet_llama_libm);
            gate = ck_deltanet_llama_expf(g[h]);
            beta_s = ck_deltanet_llama_sigmoidf(beta[h]);
        }
        if (pytorch_bf16_boundaries) {
            beta_s = bf16_to_float(float_to_bf16(beta_s));
            for (int row = 0; row < state_dim; ++row) {
                q_scaled[row] = q_head[row] * scale;
            }
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_offset = (size_t) row * (size_t) state_dim;
            for (int col = 0; col < state_dim; ++col) {
                state_cur[row_offset + (size_t) col] =
                    state_prev[row_offset + (size_t) col] * gate;
            }
        }

        for (int col = 0; col < state_dim; ++col) {
            for (int row = 0; row < state_dim; ++row) {
                column[row] = state_cur[(size_t) row * (size_t) state_dim + (size_t) col];
            }
            const float memory = ck_deltanet_llama_avx2_dot(column, k_head, state_dim);
            const float delta = (v_head[col] - memory) * beta_s;
            for (int row = 0; row < state_dim; ++row) {
                const size_t offset = (size_t) row * (size_t) state_dim + (size_t) col;
#if defined(__FMA__)
                const float updated = fmaf(k_head[row], delta, state_cur[offset]);
#else
                const float updated = state_cur[offset] + k_head[row] * delta;
#endif
                state_cur[offset] = updated;
                column[row] = updated;
            }
            if (pytorch_bf16_boundaries) {
                out_head[col] =
                    ck_deltanet_llama_avx2_dot(column, q_scaled, state_dim);
            } else {
                out_head[col] =
                    ck_deltanet_llama_avx2_dot(column, q_head, state_dim) * scale;
            }
        }
    }
#else
    if (group_count != num_heads) {
        return;
    }
    gated_deltanet_autoregressive_forward_ref(
        q, k, v, g, beta, state_in, state_out, out,
        num_heads, state_dim, norm_eps);
#endif
}

/*
 * llama.cpp stores the recurrent matrix transposed so each logical state
 * column is contiguous.  Keeping that physical layout across prefill and
 * decode removes the gather/scatter copy from every state update while
 * preserving the same per-column dot and FMA order.
 */
static void gated_deltanet_llama_avx2_grouped_forward_transposed_impl(
                                       const float *q,
                                       const float *k,
                                       const float *v,
                                       const float *g,
                                       const float *beta,
                                       const float *state_in,
                                       float *state_out,
                                       float *out,
                                       int num_heads,
                                       int group_count,
                                       int state_dim,
                                       float norm_eps,
                                       int head_begin,
                                       int head_end)
{
#if defined(__AVX2__)
    (void) norm_eps;
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out ||
        num_heads <= 0 || group_count <= 0 || num_heads % group_count != 0 ||
        state_dim <= 0 || state_dim > CK_DELTANET_MAX_STACK_DIM ||
        head_begin < 0 || head_end < head_begin || head_end > num_heads) {
        return;
    }
    const float scale = ck_deltanet_llama_scale(state_dim);
    const size_t vector_stride = (size_t) state_dim;
    const size_t state_stride = (size_t) state_dim * (size_t) state_dim;

    pthread_once(&ck_deltanet_libm_once, ck_bind_deltanet_llama_libm);
    for (int h = head_begin; h < head_end; ++h) {
        const int group = h % group_count;
        const float *q_head = q + (size_t) group * vector_stride;
        const float *k_head = k + (size_t) group * vector_stride;
        const float *v_head = v + (size_t) h * vector_stride;
        const float *state_prev = state_in + (size_t) h * state_stride;
        float *state_cur = state_out + (size_t) h * state_stride;
        float *out_head = out + (size_t) h * vector_stride;
        const float gate = ck_deltanet_llama_expf(g[h]);
        const float beta_s = ck_deltanet_llama_sigmoidf(beta[h]);

        for (int col = 0; col < state_dim; ++col) {
            const float *prev_col = state_prev + (size_t) col * vector_stride;
            float *cur_col = state_cur + (size_t) col * vector_stride;
            int row = 0;
            const __m256 gate8 = _mm256_set1_ps(gate);
            for (; row + 7 < state_dim; row += 8) {
                const __m256 scaled = _mm256_mul_ps(
                    _mm256_loadu_ps(prev_col + row), gate8);
                _mm256_storeu_ps(cur_col + row, scaled);
            }
            for (; row < state_dim; ++row) {
                cur_col[row] = prev_col[row] * gate;
            }

            const float memory =
                ck_deltanet_llama_avx2_dot(cur_col, k_head, state_dim);
            const float delta = (v_head[col] - memory) * beta_s;
            row = 0;
            const __m256 delta8 = _mm256_set1_ps(delta);
            for (; row + 7 < state_dim; row += 8) {
                const __m256 updated = _mm256_fmadd_ps(
                    _mm256_loadu_ps(k_head + row), delta8,
                    _mm256_loadu_ps(cur_col + row));
                _mm256_storeu_ps(cur_col + row, updated);
            }
            for (; row < state_dim; ++row) {
                cur_col[row] = fmaf(k_head[row], delta, cur_col[row]);
            }
            out_head[col] =
                ck_deltanet_llama_avx2_dot(cur_col, q_head, state_dim) * scale;
        }
    }
#else
    (void)q; (void)k; (void)v; (void)g; (void)beta;
    (void)state_in; (void)state_out; (void)out;
    (void)num_heads; (void)group_count; (void)state_dim; (void)norm_eps;
    (void)head_begin; (void)head_end;
#endif
}

void gated_deltanet_llama_avx2_forward(const float *q,
                                       const float *k,
                                       const float *v,
                                       const float *g,
                                       const float *beta,
                                       const float *state_in,
                                       float *state_out,
                                       float *out,
                                       int num_heads,
                                       int group_count,
                                       int state_dim,
                                       float norm_eps)
{
    gated_deltanet_llama_avx2_grouped_forward_transposed_impl(
        q, k, v, g, beta, state_in, state_out, out,
        num_heads, group_count, state_dim, norm_eps, 0, num_heads);
}

/*
 * Orchestrator-facing range entry point. Heads own disjoint state/output
 * slices, so a threadpool can partition them without changing any per-head
 * reduction tree. Keep dispatch out of the numerical kernel itself.
 */
void gated_deltanet_llama_avx2_forward_head_range(
                                       const float *q,
                                       const float *k,
                                       const float *v,
                                       const float *g,
                                       const float *beta,
                                       const float *state_in,
                                       float *state_out,
                                       float *out,
                                       int num_heads,
                                       int group_count,
                                       int state_dim,
                                       float norm_eps,
                                       int head_begin,
                                       int head_end)
{
    gated_deltanet_llama_avx2_grouped_forward_transposed_impl(
        q, k, v, g, beta, state_in, state_out, out,
        num_heads, group_count, state_dim, norm_eps,
        head_begin, head_end);
}

static int ck_deltanet_ceil_log2(int value)
{
    int result = 0;
    int power = 1;
    while (power < value) {
        power <<= 1;
        ++result;
    }
    return result;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
static void ck_deltanet_pytorch_outer_sum(const float *matrix,
                                         const float *row_weights,
                                         float *output,
                                         int state_dim)
{
    const int num_levels = 4;
    int level_power = ck_deltanet_ceil_log2(state_dim) / num_levels;
    if (level_power < 4) {
        level_power = 4;
    }
    const int level_step = 1 << level_power;
    const int level_mask = level_step - 1;
    int col = 0;

#if defined(__AVX512F__)
    /* PyTorch vectorized_outer_sum reduces four adjacent vectors together. */
    for (; col + 63 < state_dim; col += 64) {
        __m512 acc[4][4];
        for (int level = 0; level < num_levels; ++level) {
            for (int block = 0; block < 4; ++block) {
                acc[level][block] = _mm512_setzero_ps();
            }
        }

        int i = 0;
        for (; i + level_step <= state_dim;) {
            for (int j = 0; j < level_step; ++j, ++i) {
                const float *row = matrix + (size_t)i * (size_t)state_dim + col;
                const __m512 weight = _mm512_set1_ps(row_weights[i]);
                for (int block = 0; block < 4; ++block) {
                    const __m512 product = _mm512_mul_ps(
                        _mm512_loadu_ps(row + block * 16), weight);
                    acc[0][block] = _mm512_add_ps(acc[0][block], product);
                }
            }

            for (int level = 1; level < num_levels; ++level) {
                for (int block = 0; block < 4; ++block) {
                    acc[level][block] = _mm512_add_ps(
                        acc[level][block], acc[level - 1][block]);
                    acc[level - 1][block] = _mm512_setzero_ps();
                }
                const int mask = level_mask << (level * level_power);
                if ((i & mask) != 0) {
                    break;
                }
            }
        }

        for (; i < state_dim; ++i) {
            const float *row = matrix + (size_t)i * (size_t)state_dim + col;
            const __m512 weight = _mm512_set1_ps(row_weights[i]);
            for (int block = 0; block < 4; ++block) {
                const __m512 product = _mm512_mul_ps(
                    _mm512_loadu_ps(row + block * 16), weight);
                acc[0][block] = _mm512_add_ps(acc[0][block], product);
            }
        }

        for (int level = 1; level < num_levels; ++level) {
            for (int block = 0; block < 4; ++block) {
                acc[0][block] = _mm512_add_ps(
                    acc[0][block], acc[level][block]);
            }
        }
        for (int block = 0; block < 4; ++block) {
            _mm512_storeu_ps(output + col + block * 16, acc[0][block]);
        }
    }
#elif defined(__AVX2__)
    for (; col + 31 < state_dim; col += 32) {
        __m256 acc[4][4];
        for (int level = 0; level < num_levels; ++level) {
            for (int block = 0; block < 4; ++block) {
                acc[level][block] = _mm256_setzero_ps();
            }
        }

        int i = 0;
        for (; i + level_step <= state_dim;) {
            for (int j = 0; j < level_step; ++j, ++i) {
                const float *row = matrix + (size_t)i * (size_t)state_dim + col;
                const __m256 weight = _mm256_set1_ps(row_weights[i]);
                for (int block = 0; block < 4; ++block) {
                    const __m256 product = _mm256_mul_ps(
                        _mm256_loadu_ps(row + block * 8), weight);
                    acc[0][block] = _mm256_add_ps(acc[0][block], product);
                }
            }
            for (int level = 1; level < num_levels; ++level) {
                for (int block = 0; block < 4; ++block) {
                    acc[level][block] = _mm256_add_ps(
                        acc[level][block], acc[level - 1][block]);
                    acc[level - 1][block] = _mm256_setzero_ps();
                }
                const int mask = level_mask << (level * level_power);
                if ((i & mask) != 0) {
                    break;
                }
            }
        }
        for (; i < state_dim; ++i) {
            const float *row = matrix + (size_t)i * (size_t)state_dim + col;
            const __m256 weight = _mm256_set1_ps(row_weights[i]);
            for (int block = 0; block < 4; ++block) {
                const __m256 product = _mm256_mul_ps(
                    _mm256_loadu_ps(row + block * 8), weight);
                acc[0][block] = _mm256_add_ps(acc[0][block], product);
            }
        }
        for (int level = 1; level < num_levels; ++level) {
            for (int block = 0; block < 4; ++block) {
                acc[0][block] = _mm256_add_ps(
                    acc[0][block], acc[level][block]);
            }
        }
        for (int block = 0; block < 4; ++block) {
            _mm256_storeu_ps(output + col + block * 8, acc[0][block]);
        }
    }
#endif

    /* Production Qwen dimensions are covered above. Keep a deterministic
     * scalar fallback for uncommon tail widths. */
    for (; col < state_dim; ++col) {
        float sum = 0.0f;
        for (int row = 0; row < state_dim; ++row) {
            sum += matrix[(size_t)row * (size_t)state_dim + col] *
                   row_weights[row];
        }
        output[col] = sum;
    }
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
static void gated_deltanet_pytorch_grouped_bf16_forward_impl(
                                                  const float *q,
                                                  const float *k,
                                                  const float *v,
                                                  const float *g,
                                                  const float *beta,
                                                  const float *state_in,
                                                  float *state_out,
                                                  float *out,
                                                  float *debug_decayed_state,
                                                  float *debug_memory,
                                                  float *debug_delta,
                                                  int num_heads,
                                                  int group_count,
                                                  int state_dim,
                                                  float norm_eps)
{
    (void)norm_eps;
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out ||
        num_heads <= 0 || group_count <= 0 || num_heads % group_count != 0 ||
        state_dim <= 0 || state_dim > CK_DELTANET_MAX_STACK_DIM) {
        return;
    }

    const size_t vector_stride = (size_t)state_dim;
    const size_t state_stride = vector_stride * vector_stride;
    const int heads_per_group = num_heads / group_count;
    const float sqrt_dim = sqrtf((float)state_dim);
    float gate_values[CK_DELTANET_MAX_STACK_DIM];
    float beta_values[CK_DELTANET_MAX_STACK_DIM];
    float memory[CK_DELTANET_MAX_STACK_DIM];
    float delta[CK_DELTANET_MAX_STACK_DIM];
    float q_scaled[CK_DELTANET_MAX_STACK_DIM];
    ck_deltanet_pytorch_gate_values(
        g, beta, gate_values, beta_values, num_heads);

    for (int h = 0; h < num_heads; ++h) {
        const int group = h / heads_per_group;
        const float *q_head = q + (size_t)group * vector_stride;
        const float *k_head = k + (size_t)group * vector_stride;
        const float *v_head = v + (size_t)h * vector_stride;
        const float *state_prev = state_in + (size_t)h * state_stride;
        float *state_cur = state_out + (size_t)h * state_stride;
        float *out_head = out + (size_t)h * vector_stride;

        for (int col = 0; col < state_dim; ++col) {
            q_scaled[col] = q_head[col] / sqrt_dim;
        }

        /* Materialize the decayed state before the separately ordered sum. */
        for (int row = 0; row < state_dim; ++row) {
            const size_t row_offset = (size_t)row * vector_stride;
            int col = 0;
#if defined(__AVX2__)
            const __m256 gate8 = _mm256_set1_ps(gate_values[h]);
            for (; col + 7 < state_dim; col += 8) {
                const __m256 state = _mm256_mul_ps(
                    _mm256_loadu_ps(state_prev + row_offset + (size_t)col),
                    gate8);
                _mm256_storeu_ps(state_cur + row_offset + (size_t)col, state);
            }
#endif
            for (; col < state_dim; ++col) {
                const size_t offset = row_offset + (size_t)col;
                const float state = state_prev[offset] * gate_values[h];
                state_cur[offset] = state;
            }
        }

        ck_deltanet_pytorch_outer_sum(
            state_cur, k_head, memory, state_dim);

        if (debug_decayed_state) {
            memcpy(
                debug_decayed_state + (size_t)h * state_stride,
                state_cur,
                state_stride * sizeof(float));
        }
        if (debug_memory) {
            memcpy(
                debug_memory + (size_t)h * vector_stride,
                memory,
                vector_stride * sizeof(float));
        }

        for (int col = 0; col < state_dim; ++col) {
            delta[col] = (v_head[col] - memory[col]) * beta_values[h];
        }
        if (debug_delta) {
            memcpy(
                debug_delta + (size_t)h * vector_stride,
                delta,
                vector_stride * sizeof(float));
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_offset = (size_t)row * vector_stride;
            const float key = k_head[row];
            int col = 0;
#if defined(__AVX2__)
            const __m256 key8 = _mm256_set1_ps(key);
            for (; col + 7 < state_dim; col += 8) {
                const __m256 update = _mm256_mul_ps(
                    key8, _mm256_loadu_ps(delta + col));
                const __m256 state = _mm256_add_ps(
                    _mm256_loadu_ps(state_cur + row_offset + (size_t)col),
                    update);
                _mm256_storeu_ps(state_cur + row_offset + (size_t)col, state);
            }
#endif
            for (; col < state_dim; ++col) {
                const size_t offset = row_offset + (size_t)col;
                const float state = state_cur[offset] + key * delta[col];
                state_cur[offset] = state;
            }
        }

        ck_deltanet_pytorch_outer_sum(
            state_cur, q_scaled, out_head, state_dim);

        for (int col = 0; col < state_dim; ++col) {
            out_head[col] = bf16_to_float(float_to_bf16(out_head[col]));
        }
    }
}

void gated_deltanet_pytorch_grouped_bf16_forward(const float *q,
                                                  const float *k,
                                                  const float *v,
                                                  const float *g,
                                                  const float *beta,
                                                  const float *state_in,
                                                  float *state_out,
                                                  float *out,
                                                  int num_heads,
                                                  int group_count,
                                                  int state_dim,
                                                  float norm_eps)
{
    gated_deltanet_pytorch_grouped_bf16_forward_impl(
        q, k, v, g, beta, state_in, state_out, out,
        NULL, NULL, NULL,
        num_heads, group_count, state_dim, norm_eps);
}

void gated_deltanet_pytorch_grouped_bf16_forward_debug(
                                                  const float *q,
                                                  const float *k,
                                                  const float *v,
                                                  const float *g,
                                                  const float *beta,
                                                  const float *state_in,
                                                  float *state_out,
                                                  float *out,
                                                  float *decayed_state,
                                                  float *memory,
                                                  float *delta,
                                                  int num_heads,
                                                  int group_count,
                                                  int state_dim,
                                                  float norm_eps)
{
    gated_deltanet_pytorch_grouped_bf16_forward_impl(
        q, k, v, g, beta, state_in, state_out, out,
        decayed_state, memory, delta,
        num_heads, group_count, state_dim, norm_eps);
}

void gated_deltanet_llama_avx2_prefill_forward(const float *q,
                                               const float *k,
                                               const float *v,
                                               const float *g,
                                               const float *beta,
                                               const float *state_in,
                                               float *state_out,
                                               float *out,
                                               int rows,
                                               int num_heads,
                                               int group_count,
                                               int state_dim,
                                               float norm_eps)
{
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out ||
        rows <= 0 || num_heads <= 0 || group_count <= 0 ||
        num_heads % group_count != 0 || state_dim <= 0) {
        return;
    }
    const size_t qk_stride = (size_t) group_count * (size_t) state_dim;
    const size_t value_stride = (size_t) num_heads * (size_t) state_dim;
    const size_t gate_stride = (size_t) num_heads;
    for (int row = 0; row < rows; ++row) {
        gated_deltanet_llama_avx2_forward(
            q + (size_t) row * qk_stride,
            k + (size_t) row * qk_stride,
            v + (size_t) row * value_stride,
            g + (size_t) row * gate_stride,
            beta + (size_t) row * gate_stride,
            row == 0 ? state_in : state_out,
            state_out,
            out + (size_t) row * value_stride,
            num_heads, group_count, state_dim, norm_eps);
    }
}

void gated_deltanet_llama_chunk64_prefill_forward(const float *q,
                                                  const float *k,
                                                  const float *v,
                                                  const float *g,
                                                  const float *beta,
                                                  const float *state_in,
                                                  float *state_out,
                                                  float *out,
                                                  int rows,
                                                  int num_heads,
                                                  int group_count,
                                                  int state_dim,
                                                  float norm_eps)
{
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out ||
        rows <= 0 || num_heads <= 0 || group_count <= 0 ||
        num_heads % group_count != 0 || state_dim <= 0) {
        return;
    }
#if defined(__AVX2__)
    (void)norm_eps;
    if (state_dim <= CK_DELTANET_LLAMA_CHUNK_MAX_DIM) {
        for (int head = 0; head < num_heads; ++head) {
            gated_deltanet_llama_chunk64_head(
                q, k, v, g, beta, state_in, state_out, out,
                rows, num_heads, group_count, head, state_dim);
        }
        return;
    }
#endif
    gated_deltanet_llama_avx2_prefill_forward(
        q, k, v, g, beta, state_in, state_out, out,
        rows, num_heads, group_count, state_dim, norm_eps);
}

void gated_deltanet_llama_chunk64_head_forward(const float *q,
                                               const float *k,
                                               const float *v,
                                               const float *g,
                                               const float *beta,
                                               const float *state_in,
                                               float *state_out,
                                               float *out,
                                               int rows,
                                               int num_heads,
                                               int group_count,
                                               int head,
                                               int state_dim)
{
#if defined(__AVX2__)
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out ||
        rows <= 0 || num_heads <= 0 || group_count <= 0 ||
        num_heads % group_count != 0 || head < 0 || head >= num_heads ||
        state_dim <= 0 || state_dim > CK_DELTANET_LLAMA_CHUNK_MAX_DIM) {
        return;
    }
    gated_deltanet_llama_chunk64_head(
        q, k, v, g, beta, state_in, state_out, out,
        rows, num_heads, group_count, head, state_dim);
#else
    (void)q;
    (void)k;
    (void)v;
    (void)g;
    (void)beta;
    (void)state_in;
    (void)state_out;
    (void)out;
    (void)rows;
    (void)num_heads;
    (void)group_count;
    (void)head;
    (void)state_dim;
#endif
}

void gated_deltanet_pytorch_grouped_bf16_prefill_forward(
                                               const float *q,
                                               const float *k,
                                               const float *v,
                                               const float *g,
                                               const float *beta,
                                               const float *state_in,
                                               float *state_out,
                                               float *out,
                                               int rows,
                                               int num_heads,
                                               int group_count,
                                               int state_dim,
                                               float norm_eps)
{
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out ||
        rows <= 0 || num_heads <= 0 || group_count <= 0 ||
        num_heads % group_count != 0 || state_dim <= 0) {
        return;
    }
    const size_t qk_stride = (size_t)group_count * (size_t)state_dim;
    const size_t value_stride = (size_t)num_heads * (size_t)state_dim;
    const size_t gate_stride = (size_t)num_heads;
    for (int row = 0; row < rows; ++row) {
        gated_deltanet_pytorch_grouped_bf16_forward(
            q + (size_t)row * qk_stride,
            k + (size_t)row * qk_stride,
            v + (size_t)row * value_stride,
            g + (size_t)row * gate_stride,
            beta + (size_t)row * gate_stride,
            row == 0 ? state_in : state_out,
            state_out,
            out + (size_t)row * value_stride,
            num_heads, group_count, state_dim, norm_eps);
    }
}

void gated_deltanet_autoregressive_forward_ref(const float *q,
                                               const float *k,
                                               const float *v,
                                               const float *g,
                                               const float *beta,
                                               const float *state_in,
                                               float *state_out,
                                               float *out,
                                               int num_heads,
                                               int state_dim,
                                               float norm_eps)
{
    const float q_scale = 1.0f / sqrtf((float)state_dim);
    const size_t vec_stride = (size_t)state_dim;
    const size_t state_stride = (size_t)state_dim * (size_t)state_dim;

    for (int h = 0; h < num_heads; ++h) {
        const float *q_head = q + (size_t)h * vec_stride;
        const float *k_head = k + (size_t)h * vec_stride;
        const float *v_head = v + (size_t)h * vec_stride;
        const float *state_prev = state_in + (size_t)h * state_stride;
        float *state_cur = state_out + (size_t)h * state_stride;
        float *out_head = out + (size_t)h * vec_stride;

        const float beta_s = ck_deltanet_sigmoidf(beta[h]);
        const float gate = expf(g[h]);

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            for (int col = 0; col < state_dim; ++col) {
                state_cur[row_off + (size_t)col] = state_prev[row_off + (size_t)col] * gate;
            }
        }

        for (int col = 0; col < state_dim; ++col) {
            float kv_mem = 0.0f;
            for (int row = 0; row < state_dim; ++row) {
                const float k_hat = k_head[row];
                kv_mem += state_cur[(size_t)row * (size_t)state_dim + (size_t)col] * k_hat;
            }

            const float delta = (v_head[col] - kv_mem) * beta_s;
            for (int row = 0; row < state_dim; ++row) {
                const float k_hat = k_head[row];
                state_cur[(size_t)row * (size_t)state_dim + (size_t)col] += k_hat * delta;
            }
        }

        for (int col = 0; col < state_dim; ++col) {
            float acc = 0.0f;
            for (int row = 0; row < state_dim; ++row) {
                const float q_hat = q_head[row] * q_scale;
                acc += state_cur[(size_t)row * (size_t)state_dim + (size_t)col] * q_hat;
            }
            out_head[col] = acc;
        }
    }
}

void gated_deltanet_autoregressive_backward_ref(const float *d_out,
                                                const float *d_state_out,
                                                const float *q,
                                                const float *k,
                                                const float *v,
                                                const float *g,
                                                const float *beta,
                                                const float *state_in,
                                                const float *state_out,
                                                float *d_q,
                                                float *d_k,
                                                float *d_v,
                                                float *d_g,
                                                float *d_beta,
                                                float *d_state_in,
                                                int num_heads,
                                                int state_dim,
                                                float norm_eps)
{
    const float q_scale = 1.0f / sqrtf((float)state_dim);
    const size_t vec_stride = (size_t)state_dim;
    const size_t state_stride = (size_t)state_dim * (size_t)state_dim;

    float q_hat[CK_DELTANET_MAX_STACK_DIM];
    float k_hat[CK_DELTANET_MAX_STACK_DIM];
    float kv_mem[CK_DELTANET_MAX_STACK_DIM];
    float delta[CK_DELTANET_MAX_STACK_DIM];
    float d_q_hat[CK_DELTANET_MAX_STACK_DIM];
    float d_k_hat[CK_DELTANET_MAX_STACK_DIM];
    float d_mem[CK_DELTANET_MAX_STACK_DIM];

    for (int h = 0; h < num_heads; ++h) {
        const float *d_out_head = d_out + (size_t)h * vec_stride;
        const float *d_state_out_head = d_state_out + (size_t)h * state_stride;
        const float *q_head = q + (size_t)h * vec_stride;
        const float *k_head = k + (size_t)h * vec_stride;
        const float *v_head = v + (size_t)h * vec_stride;
        const float *state_prev = state_in + (size_t)h * state_stride;
        const float *state_cur = state_out + (size_t)h * state_stride;
        float *d_q_head = d_q + (size_t)h * vec_stride;
        float *d_k_head = d_k + (size_t)h * vec_stride;
        float *d_v_head = d_v + (size_t)h * vec_stride;
        float *d_state_prev = d_state_in + (size_t)h * state_stride;

        const float beta_s = ck_deltanet_sigmoidf(beta[h]);
        const float gate = expf(g[h]);

        float qk_dot = 0.0f;
        float out_delta_dot = 0.0f;
        float beta_acc = 0.0f;
        float gate_acc = 0.0f;

        for (int i = 0; i < state_dim; ++i) {
            q_hat[i] = q_head[i] * q_scale;
            k_hat[i] = k_head[i];
            kv_mem[i] = 0.0f;
            d_q_hat[i] = 0.0f;
            d_k_hat[i] = 0.0f;
            d_mem[i] = 0.0f;
            d_v_head[i] = 0.0f;
            qk_dot += q_hat[i] * k_hat[i];
        }

        for (int col = 0; col < state_dim; ++col) {
            float mem = 0.0f;
            for (int row = 0; row < state_dim; ++row) {
                mem += (state_prev[(size_t)row * (size_t)state_dim + (size_t)col] * gate) * k_hat[row];
            }
            kv_mem[col] = mem;
            delta[col] = (v_head[col] - mem) * beta_s;
            out_delta_dot += d_out_head[col] * delta[col];
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            float dq_acc = 0.0f;
            float dk_acc = q_hat[row] * out_delta_dot;
            for (int col = 0; col < state_dim; ++col) {
                const float d_state_direct = d_state_out_head[row_off + (size_t)col];
                dq_acc += state_cur[row_off + (size_t)col] * d_out_head[col];
                dk_acc += d_state_direct * delta[col];
            }
            d_q_hat[row] = dq_acc;
            d_k_hat[row] = dk_acc;
        }

        for (int col = 0; col < state_dim; ++col) {
            float d_delta_acc = d_out_head[col] * qk_dot;
            for (int row = 0; row < state_dim; ++row) {
                d_delta_acc += d_state_out_head[(size_t)row * (size_t)state_dim + (size_t)col] * k_hat[row];
            }

            d_v_head[col] = beta_s * d_delta_acc;
            d_mem[col] = -beta_s * d_delta_acc;
            beta_acc += d_delta_acc * (v_head[col] - kv_mem[col]);
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            float s_dm_acc = 0.0f;
            for (int col = 0; col < state_dim; ++col) {
                s_dm_acc += (state_prev[row_off + (size_t)col] * gate) * d_mem[col];
            }
            d_k_hat[row] += s_dm_acc;
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            for (int col = 0; col < state_dim; ++col) {
                const float d_state_total = d_state_out_head[row_off + (size_t)col]
                                          + q_hat[row] * d_out_head[col]
                                          + k_hat[row] * d_mem[col];
                d_state_prev[row_off + (size_t)col] = gate * d_state_total;
                gate_acc += d_state_total * state_prev[row_off + (size_t)col];
            }
        }

        for (int i = 0; i < state_dim; ++i) {
            d_q_head[i] = d_q_hat[i] * q_scale;
            d_k_head[i] = d_k_hat[i];
        }

        d_g[h] = gate_acc * gate;
        d_beta[h] = beta_acc * beta_s * (1.0f - beta_s);
    }
}

#if defined(__AVX__)
static void ck_deltanet_scale_rows_avx(const float *src, float *dst, int dim, float scale)
{
    const __m256 scale_v = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, scale_v));
    }
    for (; i < dim; ++i) {
        dst[i] = src[i] * scale;
    }
}

void gated_deltanet_autoregressive_forward_avx(const float *q,
                                               const float *k,
                                               const float *v,
                                               const float *g,
                                               const float *beta,
                                               const float *state_in,
                                               float *state_out,
                                               float *out,
                                               int num_heads,
                                               int state_dim,
                                               float norm_eps)
{
    const float q_scale = 1.0f / sqrtf((float)state_dim);
    const size_t vec_stride = (size_t)state_dim;
    const size_t state_stride = (size_t)state_dim * (size_t)state_dim;

    float q_hat[CK_DELTANET_MAX_STACK_DIM];
    float k_hat[CK_DELTANET_MAX_STACK_DIM];
    float kv_mem[CK_DELTANET_MAX_STACK_DIM];
    float delta[CK_DELTANET_MAX_STACK_DIM];

    for (int h = 0; h < num_heads; ++h) {
        const float *q_head = q + (size_t)h * vec_stride;
        const float *k_head = k + (size_t)h * vec_stride;
        const float *v_head = v + (size_t)h * vec_stride;
        const float *state_prev = state_in + (size_t)h * state_stride;
        float *state_cur = state_out + (size_t)h * state_stride;
        float *out_head = out + (size_t)h * vec_stride;

        const float gate = expf(g[h]);
        const float beta_s = ck_deltanet_sigmoidf(beta[h]);

        /* q and k arrive pre-normalized by recurrent_qk_l2_norm. */
        ck_deltanet_scale_rows_avx(q_head, q_hat, state_dim, q_scale);
        ck_deltanet_scale_rows_avx(k_head, k_hat, state_dim, 1.0f);

        const __m256 beta_v = _mm256_set1_ps(beta_s);
        const __m256 zero_v = _mm256_setzero_ps();

        int col = 0;
        for (; col + 8 <= state_dim; col += 8) {
            _mm256_storeu_ps(kv_mem + col, zero_v);
            _mm256_storeu_ps(out_head + col, zero_v);
        }
        for (; col < state_dim; ++col) {
            kv_mem[col] = 0.0f;
            out_head[col] = 0.0f;
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            const __m256 k_hat_v = _mm256_set1_ps(k_hat[row]);
            const __m256 gate_v = _mm256_set1_ps(gate);

            col = 0;
            for (; col + 8 <= state_dim; col += 8) {
                __m256 prev_v = _mm256_loadu_ps(state_prev + row_off + (size_t)col);
                __m256 cur_v = _mm256_mul_ps(prev_v, gate_v);
                __m256 kv_v = _mm256_loadu_ps(kv_mem + col);
                kv_v = _mm256_add_ps(kv_v, _mm256_mul_ps(cur_v, k_hat_v));
                _mm256_storeu_ps(state_cur + row_off + (size_t)col, cur_v);
                _mm256_storeu_ps(kv_mem + col, kv_v);
            }
            for (; col < state_dim; ++col) {
                const float cur = state_prev[row_off + (size_t)col] * gate;
                state_cur[row_off + (size_t)col] = cur;
                kv_mem[col] += cur * k_hat[row];
            }
        }

        col = 0;
        for (; col + 8 <= state_dim; col += 8) {
            __m256 v_v = _mm256_loadu_ps(v_head + col);
            __m256 kv_v = _mm256_loadu_ps(kv_mem + col);
            __m256 delta_v = _mm256_mul_ps(_mm256_sub_ps(v_v, kv_v), beta_v);
            _mm256_storeu_ps(delta + col, delta_v);
        }
        for (; col < state_dim; ++col) {
            delta[col] = (v_head[col] - kv_mem[col]) * beta_s;
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            const __m256 k_hat_v = _mm256_set1_ps(k_hat[row]);
            const __m256 q_hat_v = _mm256_set1_ps(q_hat[row]);

            col = 0;
            for (; col + 8 <= state_dim; col += 8) {
                __m256 cur_v = _mm256_loadu_ps(state_cur + row_off + (size_t)col);
                __m256 delta_v = _mm256_loadu_ps(delta + col);
                __m256 out_v = _mm256_loadu_ps(out_head + col);
                __m256 updated_v = _mm256_add_ps(cur_v, _mm256_mul_ps(k_hat_v, delta_v));
                out_v = _mm256_add_ps(out_v, _mm256_mul_ps(updated_v, q_hat_v));
                _mm256_storeu_ps(state_cur + row_off + (size_t)col, updated_v);
                _mm256_storeu_ps(out_head + col, out_v);
            }
            for (; col < state_dim; ++col) {
                const float updated = state_cur[row_off + (size_t)col] + k_hat[row] * delta[col];
                state_cur[row_off + (size_t)col] = updated;
                out_head[col] += updated * q_hat[row];
            }
        }
    }
}
#endif

#if defined(__AVX2__)
static inline __m256 ck_deltanet_fmadd256(__m256 a, __m256 b, __m256 c)
{
#if defined(__FMA__)
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

void gated_deltanet_autoregressive_forward_avx2(const float *q,
                                                const float *k,
                                                const float *v,
                                                const float *g,
                                                const float *beta,
                                                const float *state_in,
                                                float *state_out,
                                                float *out,
                                                int num_heads,
                                                int state_dim,
                                                float norm_eps)
{
    const float q_scale = 1.0f / sqrtf((float)state_dim);
    const size_t vec_stride = (size_t)state_dim;
    const size_t state_stride = (size_t)state_dim * (size_t)state_dim;

    float q_hat[CK_DELTANET_MAX_STACK_DIM];
    float k_hat[CK_DELTANET_MAX_STACK_DIM];
    float kv_mem[CK_DELTANET_MAX_STACK_DIM];
    float delta[CK_DELTANET_MAX_STACK_DIM];

    for (int h = 0; h < num_heads; ++h) {
        const float *q_head = q + (size_t)h * vec_stride;
        const float *k_head = k + (size_t)h * vec_stride;
        const float *v_head = v + (size_t)h * vec_stride;
        const float *state_prev = state_in + (size_t)h * state_stride;
        float *state_cur = state_out + (size_t)h * state_stride;
        float *out_head = out + (size_t)h * vec_stride;

        const float gate = expf(g[h]);
        const float beta_s = ck_deltanet_sigmoidf(beta[h]);

        /* q and k arrive pre-normalized by recurrent_qk_l2_norm. */
        ck_deltanet_scale_rows_avx(q_head, q_hat, state_dim, q_scale);
        ck_deltanet_scale_rows_avx(k_head, k_hat, state_dim, 1.0f);

        const __m256 beta_v = _mm256_set1_ps(beta_s);
        const __m256 zero_v = _mm256_setzero_ps();

        int col = 0;
        for (; col + 8 <= state_dim; col += 8) {
            _mm256_storeu_ps(kv_mem + col, zero_v);
            _mm256_storeu_ps(out_head + col, zero_v);
        }
        for (; col < state_dim; ++col) {
            kv_mem[col] = 0.0f;
            out_head[col] = 0.0f;
        }

        int row = 0;
        for (; row + 2 <= state_dim; row += 2) {
            const size_t row0_off = (size_t)row * (size_t)state_dim;
            const size_t row1_off = (size_t)(row + 1) * (size_t)state_dim;
            const __m256 k0_v = _mm256_set1_ps(k_hat[row]);
            const __m256 k1_v = _mm256_set1_ps(k_hat[row + 1]);
            const __m256 gate0_v = _mm256_set1_ps(gate);
            const __m256 gate1_v = _mm256_set1_ps(gate);

            col = 0;
            for (; col + 8 <= state_dim; col += 8) {
                __m256 prev0_v = _mm256_loadu_ps(state_prev + row0_off + (size_t)col);
                __m256 prev1_v = _mm256_loadu_ps(state_prev + row1_off + (size_t)col);
                __m256 cur0_v = _mm256_mul_ps(prev0_v, gate0_v);
                __m256 cur1_v = _mm256_mul_ps(prev1_v, gate1_v);
                __m256 kv_v = _mm256_loadu_ps(kv_mem + col);
                kv_v = ck_deltanet_fmadd256(cur0_v, k0_v, kv_v);
                kv_v = ck_deltanet_fmadd256(cur1_v, k1_v, kv_v);
                _mm256_storeu_ps(state_cur + row0_off + (size_t)col, cur0_v);
                _mm256_storeu_ps(state_cur + row1_off + (size_t)col, cur1_v);
                _mm256_storeu_ps(kv_mem + col, kv_v);
            }
            for (; col < state_dim; ++col) {
                const float cur0 = state_prev[row0_off + (size_t)col] * gate;
                const float cur1 = state_prev[row1_off + (size_t)col] * gate;
                state_cur[row0_off + (size_t)col] = cur0;
                state_cur[row1_off + (size_t)col] = cur1;
                kv_mem[col] += cur0 * k_hat[row] + cur1 * k_hat[row + 1];
            }
        }
        for (; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            const __m256 k_hat_v = _mm256_set1_ps(k_hat[row]);
            const __m256 gate_v = _mm256_set1_ps(gate);
            col = 0;
            for (; col + 8 <= state_dim; col += 8) {
                __m256 prev_v = _mm256_loadu_ps(state_prev + row_off + (size_t)col);
                __m256 cur_v = _mm256_mul_ps(prev_v, gate_v);
                __m256 kv_v = _mm256_loadu_ps(kv_mem + col);
                kv_v = ck_deltanet_fmadd256(cur_v, k_hat_v, kv_v);
                _mm256_storeu_ps(state_cur + row_off + (size_t)col, cur_v);
                _mm256_storeu_ps(kv_mem + col, kv_v);
            }
            for (; col < state_dim; ++col) {
                const float cur = state_prev[row_off + (size_t)col] * gate;
                state_cur[row_off + (size_t)col] = cur;
                kv_mem[col] += cur * k_hat[row];
            }
        }

        col = 0;
        for (; col + 8 <= state_dim; col += 8) {
            __m256 v_v = _mm256_loadu_ps(v_head + col);
            __m256 kv_v = _mm256_loadu_ps(kv_mem + col);
            __m256 delta_v = _mm256_mul_ps(_mm256_sub_ps(v_v, kv_v), beta_v);
            _mm256_storeu_ps(delta + col, delta_v);
        }
        for (; col < state_dim; ++col) {
            delta[col] = (v_head[col] - kv_mem[col]) * beta_s;
        }

        row = 0;
        for (; row + 2 <= state_dim; row += 2) {
            const size_t row0_off = (size_t)row * (size_t)state_dim;
            const size_t row1_off = (size_t)(row + 1) * (size_t)state_dim;
            const __m256 k0_v = _mm256_set1_ps(k_hat[row]);
            const __m256 k1_v = _mm256_set1_ps(k_hat[row + 1]);
            const __m256 q0_v = _mm256_set1_ps(q_hat[row]);
            const __m256 q1_v = _mm256_set1_ps(q_hat[row + 1]);

            col = 0;
            for (; col + 8 <= state_dim; col += 8) {
                __m256 cur0_v = _mm256_loadu_ps(state_cur + row0_off + (size_t)col);
                __m256 cur1_v = _mm256_loadu_ps(state_cur + row1_off + (size_t)col);
                __m256 delta_v = _mm256_loadu_ps(delta + col);
                __m256 out_v = _mm256_loadu_ps(out_head + col);
                __m256 upd0_v = ck_deltanet_fmadd256(k0_v, delta_v, cur0_v);
                __m256 upd1_v = ck_deltanet_fmadd256(k1_v, delta_v, cur1_v);
                out_v = ck_deltanet_fmadd256(upd0_v, q0_v, out_v);
                out_v = ck_deltanet_fmadd256(upd1_v, q1_v, out_v);
                _mm256_storeu_ps(state_cur + row0_off + (size_t)col, upd0_v);
                _mm256_storeu_ps(state_cur + row1_off + (size_t)col, upd1_v);
                _mm256_storeu_ps(out_head + col, out_v);
            }
            for (; col < state_dim; ++col) {
                const float upd0 = state_cur[row0_off + (size_t)col] + k_hat[row] * delta[col];
                const float upd1 = state_cur[row1_off + (size_t)col] + k_hat[row + 1] * delta[col];
                state_cur[row0_off + (size_t)col] = upd0;
                state_cur[row1_off + (size_t)col] = upd1;
                out_head[col] += upd0 * q_hat[row] + upd1 * q_hat[row + 1];
            }
        }
        for (; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            const __m256 k_hat_v = _mm256_set1_ps(k_hat[row]);
            const __m256 q_hat_v = _mm256_set1_ps(q_hat[row]);
            col = 0;
            for (; col + 8 <= state_dim; col += 8) {
                __m256 cur_v = _mm256_loadu_ps(state_cur + row_off + (size_t)col);
                __m256 delta_v = _mm256_loadu_ps(delta + col);
                __m256 out_v = _mm256_loadu_ps(out_head + col);
                __m256 updated_v = ck_deltanet_fmadd256(k_hat_v, delta_v, cur_v);
                out_v = ck_deltanet_fmadd256(updated_v, q_hat_v, out_v);
                _mm256_storeu_ps(state_cur + row_off + (size_t)col, updated_v);
                _mm256_storeu_ps(out_head + col, out_v);
            }
            for (; col < state_dim; ++col) {
                const float updated = state_cur[row_off + (size_t)col] + k_hat[row] * delta[col];
                state_cur[row_off + (size_t)col] = updated;
                out_head[col] += updated * q_hat[row];
            }
        }
    }
}
#endif

#if defined(__AVX512F__)
static inline __m512 ck_deltanet_madd512(__m512 a, __m512 b, __m512 c)
{
    return _mm512_add_ps(_mm512_mul_ps(a, b), c);
}

static void ck_deltanet_scale_rows_avx512(const float *src, float *dst, int dim, float scale)
{
    const __m512 scale_v = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 x = _mm512_loadu_ps(src + i);
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(x, scale_v));
    }
    for (; i < dim; ++i) {
        dst[i] = src[i] * scale;
    }
}

void gated_deltanet_autoregressive_forward_avx512(const float *q,
                                                  const float *k,
                                                  const float *v,
                                                  const float *g,
                                                  const float *beta,
                                                  const float *state_in,
                                                  float *state_out,
                                                  float *out,
                                                  int num_heads,
                                                  int state_dim,
                                                  float norm_eps)
{
    const float q_scale = 1.0f / sqrtf((float)state_dim);
    const size_t vec_stride = (size_t)state_dim;
    const size_t state_stride = (size_t)state_dim * (size_t)state_dim;

    float q_hat[CK_DELTANET_MAX_STACK_DIM];
    float k_hat[CK_DELTANET_MAX_STACK_DIM];
    float kv_mem[CK_DELTANET_MAX_STACK_DIM];
    float delta[CK_DELTANET_MAX_STACK_DIM];

    for (int h = 0; h < num_heads; ++h) {
        const float *q_head = q + (size_t)h * vec_stride;
        const float *k_head = k + (size_t)h * vec_stride;
        const float *v_head = v + (size_t)h * vec_stride;
        const float *state_prev = state_in + (size_t)h * state_stride;
        float *state_cur = state_out + (size_t)h * state_stride;
        float *out_head = out + (size_t)h * vec_stride;

        const float gate = expf(g[h]);
        const float beta_s = ck_deltanet_sigmoidf(beta[h]);

        /* q and k arrive pre-normalized by recurrent_qk_l2_norm. */
        ck_deltanet_scale_rows_avx512(q_head, q_hat, state_dim, q_scale);
        ck_deltanet_scale_rows_avx512(k_head, k_hat, state_dim, 1.0f);

        const __m512 beta_v = _mm512_set1_ps(beta_s);
        const __m512 zero_v = _mm512_setzero_ps();

        int col = 0;
        for (; col + 16 <= state_dim; col += 16) {
            _mm512_storeu_ps(kv_mem + col, zero_v);
            _mm512_storeu_ps(out_head + col, zero_v);
        }
        for (; col < state_dim; ++col) {
            kv_mem[col] = 0.0f;
            out_head[col] = 0.0f;
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            const __m512 k_hat_v = _mm512_set1_ps(k_hat[row]);
            const __m512 gate_v = _mm512_set1_ps(gate);
            col = 0;
            for (; col + 16 <= state_dim; col += 16) {
                __m512 prev_v = _mm512_loadu_ps(state_prev + row_off + (size_t)col);
                __m512 cur_v = _mm512_mul_ps(prev_v, gate_v);
                __m512 kv_v = _mm512_loadu_ps(kv_mem + col);
                kv_v = ck_deltanet_madd512(cur_v, k_hat_v, kv_v);
                _mm512_storeu_ps(state_cur + row_off + (size_t)col, cur_v);
                _mm512_storeu_ps(kv_mem + col, kv_v);
            }
            for (; col < state_dim; ++col) {
                const float cur = state_prev[row_off + (size_t)col] * gate;
                state_cur[row_off + (size_t)col] = cur;
                kv_mem[col] += cur * k_hat[row];
            }
        }

        col = 0;
        for (; col + 16 <= state_dim; col += 16) {
            __m512 v_v = _mm512_loadu_ps(v_head + col);
            __m512 kv_v = _mm512_loadu_ps(kv_mem + col);
            __m512 delta_v = _mm512_mul_ps(_mm512_sub_ps(v_v, kv_v), beta_v);
            _mm512_storeu_ps(delta + col, delta_v);
        }
        for (; col < state_dim; ++col) {
            delta[col] = (v_head[col] - kv_mem[col]) * beta_s;
        }

        for (int row = 0; row < state_dim; ++row) {
            const size_t row_off = (size_t)row * (size_t)state_dim;
            const __m512 k_hat_v = _mm512_set1_ps(k_hat[row]);
            const __m512 q_hat_v = _mm512_set1_ps(q_hat[row]);
            col = 0;
            for (; col + 16 <= state_dim; col += 16) {
                __m512 cur_v = _mm512_loadu_ps(state_cur + row_off + (size_t)col);
                __m512 delta_v = _mm512_loadu_ps(delta + col);
                __m512 out_v = _mm512_loadu_ps(out_head + col);
                __m512 updated_v = ck_deltanet_madd512(k_hat_v, delta_v, cur_v);
                out_v = ck_deltanet_madd512(updated_v, q_hat_v, out_v);
                _mm512_storeu_ps(state_cur + row_off + (size_t)col, updated_v);
                _mm512_storeu_ps(out_head + col, out_v);
            }
            for (; col < state_dim; ++col) {
                const float updated = state_cur[row_off + (size_t)col] + k_hat[row] * delta[col];
                state_cur[row_off + (size_t)col] = updated;
                out_head[col] += updated * q_hat[row];
            }
        }
    }
}
#endif

static int ck_deltanet_force_ref(void)
{
    const char *env = getenv("CK_DELTANET_FORCE_REF");
    return env && atoi(env) != 0;
}

const char *gated_deltanet_impl_name(void)
{
    if (ck_strict_parity_enabled() || ck_deltanet_force_ref()) {
        return "REF";
    }
#if defined(__AVX512F__)
    return "AVX512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__AVX__)
    return "AVX";
#else
    return "REF";
#endif
}

void gated_deltanet_autoregressive_forward(const float *q,
                                           const float *k,
                                           const float *v,
                                           const float *g,
                                           const float *beta,
                                           const float *state_in,
                                           float *state_out,
                                           float *out,
                                           int num_heads,
                                           int state_dim,
                                           float norm_eps)
{
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out) {
        return;
    }
    if (num_heads <= 0 || state_dim <= 0) {
        return;
    }

    /*
     * q and k arrive pre-normalized by recurrent_qk_l2_norm, so the
     * ISA-specialized kernels can follow the same contract as the scalar ref.
     */
    if (ck_strict_parity_enabled() || ck_deltanet_force_ref()) {
        gated_deltanet_autoregressive_forward_ref(
            q, k, v, g, beta, state_in, state_out, out, num_heads, state_dim, norm_eps);
        return;
    }
#if defined(__AVX512F__)
    gated_deltanet_autoregressive_forward_avx512(
        q, k, v, g, beta, state_in, state_out, out, num_heads, state_dim, norm_eps);
#elif defined(__AVX2__)
    gated_deltanet_autoregressive_forward_avx2(
        q, k, v, g, beta, state_in, state_out, out, num_heads, state_dim, norm_eps);
#elif defined(__AVX__)
    gated_deltanet_autoregressive_forward_avx(
        q, k, v, g, beta, state_in, state_out, out, num_heads, state_dim, norm_eps);
#else
    gated_deltanet_autoregressive_forward_ref(
        q, k, v, g, beta, state_in, state_out, out, num_heads, state_dim, norm_eps);
#endif
}

void gated_deltanet_prefill_forward(const float *q,
                                    const float *k,
                                    const float *v,
                                    const float *g,
                                    const float *beta,
                                    const float *state_in,
                                    float *state_out,
                                    float *out,
                                    int rows,
                                    int num_heads,
                                    int state_dim,
                                    float norm_eps)
{
    if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out) {
        return;
    }
    if (rows <= 0 || num_heads <= 0 || state_dim <= 0) {
        return;
    }

    const size_t vector_stride = (size_t)num_heads * (size_t)state_dim;
    const size_t gate_stride = (size_t)num_heads;
    for (int row = 0; row < rows; ++row) {
        const float *row_state_in = row == 0 ? state_in : state_out;
        gated_deltanet_autoregressive_forward(
            q + (size_t)row * vector_stride,
            k + (size_t)row * vector_stride,
            v + (size_t)row * vector_stride,
            g + (size_t)row * gate_stride,
            beta + (size_t)row * gate_stride,
            row_state_in,
            state_out,
            out + (size_t)row * vector_stride,
            num_heads,
            state_dim,
            norm_eps);
    }
}

void gated_deltanet_autoregressive_backward(const float *d_out,
                                            const float *d_state_out,
                                            const float *q,
                                            const float *k,
                                            const float *v,
                                            const float *g,
                                            const float *beta,
                                            const float *state_in,
                                            const float *state_out,
                                            float *d_q,
                                            float *d_k,
                                            float *d_v,
                                            float *d_g,
                                            float *d_beta,
                                            float *d_state_in,
                                            int num_heads,
                                            int state_dim,
                                            float norm_eps)
{
    if (!d_out || !d_state_out || !q || !k || !v || !g || !beta || !state_in || !state_out ||
        !d_q || !d_k || !d_v || !d_g || !d_beta || !d_state_in) {
        return;
    }
    if (num_heads <= 0 || state_dim <= 0 || state_dim > CK_DELTANET_MAX_STACK_DIM) {
        return;
    }

    gated_deltanet_autoregressive_backward_ref(
        d_out,
        d_state_out,
        q,
        k,
        v,
        g,
        beta,
        state_in,
        state_out,
        d_q,
        d_k,
        d_v,
        d_g,
        d_beta,
        d_state_in,
        num_heads,
        state_dim,
        norm_eps);
}
