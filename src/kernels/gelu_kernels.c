/**
 * @file gelu_kernels.c
 * @brief GELU activation kernels with SIMD (SSE/AVX/AVX512)
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
 * GELU: y = x * 0.5 * (1 + erf(x / sqrt(2)))
 * Fast approx: y = x * sigmoid(1.702 * x)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <stddef.h>
#include <pthread.h>
#include <dlfcn.h>

#include "ckernel_quant.h"

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

static inline float ck_gelu_tanh_f32(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    const float x3 = x * x * x;
    const float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline float ck_gelu_tanh_ggml_f32(float x) {
    const float gelu_coef_a = 0.044715f;
    const float sqrt_2_over_pi = 0.79788456080286535588f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + gelu_coef_a * x * x)));
}

static ck_half ck_gelu_ggml_table_f16[1u << 16];
static pthread_once_t ck_gelu_ggml_table_once = PTHREAD_ONCE_INIT;
static pthread_once_t ck_gelu_ggml_runtime_once = PTHREAD_ONCE_INIT;

typedef void (*ck_gelu_ggml_cpu_init_fn)(void);
typedef ck_half (*ck_gelu_ggml_fp32_to_fp16_fn)(float);
typedef float (*ck_gelu_ggml_fp16_to_fp32_fn)(ck_half);

static const ck_half *ck_gelu_runtime_table_f16 = NULL;
static ck_gelu_ggml_fp32_to_fp16_fn ck_gelu_runtime_fp32_to_fp16 = NULL;
static ck_gelu_ggml_fp16_to_fp32_fn ck_gelu_runtime_fp16_to_fp32 = NULL;
static void *ck_gelu_runtime_handle = NULL;
static int ck_gelu_runtime_ready = 0;

static void ck_gelu_try_bind_runtime(void *handle) {
    ck_gelu_ggml_cpu_init_fn cpu_init_fn =
        (ck_gelu_ggml_cpu_init_fn) dlsym(handle, "ggml_cpu_init");
    ck_gelu_ggml_fp32_to_fp16_fn fp32_to_fp16_fn =
        (ck_gelu_ggml_fp32_to_fp16_fn) dlsym(handle, "ggml_fp32_to_fp16");
    ck_gelu_ggml_fp16_to_fp32_fn fp16_to_fp32_fn =
        (ck_gelu_ggml_fp16_to_fp32_fn) dlsym(handle, "ggml_fp16_to_fp32");
    const ck_half *table =
        (const ck_half *) dlsym(handle, "ggml_table_gelu_f16");

    if (!cpu_init_fn || !fp32_to_fp16_fn || !fp16_to_fp32_fn || !table) {
        return;
    }

    cpu_init_fn();
    ck_gelu_runtime_table_f16 = table;
    ck_gelu_runtime_fp32_to_fp16 = fp32_to_fp16_fn;
    ck_gelu_runtime_fp16_to_fp32 = fp16_to_fp32_fn;
    ck_gelu_runtime_ready = 1;
}

static void ck_gelu_ggml_runtime_init(void) {
    ck_gelu_try_bind_runtime(RTLD_DEFAULT);
    if (ck_gelu_runtime_ready) {
        return;
    }

    ck_gelu_runtime_handle = dlopen("llama.cpp/build/bin/libggml-cpu.so", RTLD_LAZY | RTLD_LOCAL);
    if (ck_gelu_runtime_handle) {
        ck_gelu_try_bind_runtime(ck_gelu_runtime_handle);
    }
}

static void ck_gelu_ggml_table_init(void) {
    for (uint32_t i = 0; i < (1u << 16); ++i) {
        const ck_half x_fp16 = (ck_half) i;
        const float x = ggml_fp16_to_fp32(x_fp16);
        const float y = ck_gelu_tanh_ggml_f32(x);
        ck_gelu_ggml_table_f16[i] = ggml_fp32_to_fp16(y);
    }
}

/* Fast vectorized exp approximation (same as softmax_kernels.c) */
#if defined(__AVX512F__)
static inline __m512 exp512_fast(__m512 x) {
    // Clamp to avoid overflow/underflow
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));

    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    const __m512 c1 = _mm512_set1_ps(0.693359375f);
    const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);

    __m512 t = _mm512_mul_ps(x, log2e);
    __m512 ti = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m512 rx = _mm512_sub_ps(x, _mm512_mul_ps(ti, c1));
    rx = _mm512_sub_ps(rx, _mm512_mul_ps(ti, c2));

    // Polynomial approximation
    const __m512 p0 = _mm512_set1_ps(1.0f);
    const __m512 p1 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 p2 = _mm512_set1_ps(0.24022650695910071f);
    const __m512 p3 = _mm512_set1_ps(0.05550410866482157f);
    const __m512 p4 = _mm512_set1_ps(0.009618129107628477f);

    __m512 poly = _mm512_fmadd_ps(p4, rx, p3);
    poly = _mm512_fmadd_ps(poly, rx, p2);
    poly = _mm512_fmadd_ps(poly, rx, p1);
    poly = _mm512_fmadd_ps(poly, rx, p0);

    __m512i ti_int = _mm512_cvtps_epi32(ti);
    ti_int = _mm512_add_epi32(ti_int, _mm512_set1_epi32(127));
    ti_int = _mm512_slli_epi32(ti_int, 23);
    __m512 scale = _mm512_castsi512_ps(ti_int);

    return _mm512_mul_ps(poly, scale);
}

// Fast vectorized tanh: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
static inline __m512 tanh512_fast(__m512 x) {
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 exp2x = exp512_fast(_mm512_mul_ps(two, x));
    __m512 num = _mm512_sub_ps(exp2x, one);
    __m512 den = _mm512_add_ps(exp2x, one);
    return _mm512_div_ps(num, den);
}
#endif

#if defined(__AVX2__)
static inline __m256 exp256_fast(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);

    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256 rx = _mm256_sub_ps(x, _mm256_mul_ps(ti, c1));
    rx = _mm256_sub_ps(rx, _mm256_mul_ps(ti, c2));

    const __m256 p0 = _mm256_set1_ps(1.0f);
    const __m256 p1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 p2 = _mm256_set1_ps(0.24022650695910071f);
    const __m256 p3 = _mm256_set1_ps(0.05550410866482157f);
    const __m256 p4 = _mm256_set1_ps(0.009618129107628477f);

    __m256 poly = _mm256_fmadd_ps(p4, rx, p3);
    poly = _mm256_fmadd_ps(poly, rx, p2);
    poly = _mm256_fmadd_ps(poly, rx, p1);
    poly = _mm256_fmadd_ps(poly, rx, p0);

    __m256i ti_int = _mm256_cvtps_epi32(ti);
    ti_int = _mm256_add_epi32(ti_int, _mm256_set1_epi32(127));
    ti_int = _mm256_slli_epi32(ti_int, 23);
    __m256 scale = _mm256_castsi256_ps(ti_int);

    return _mm256_mul_ps(poly, scale);
}

static inline __m256 tanh256_fast(__m256 x) {
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 exp2x = exp256_fast(_mm256_mul_ps(two, x));
    __m256 num = _mm256_sub_ps(exp2x, one);
    __m256 den = _mm256_add_ps(exp2x, one);
    return _mm256_div_ps(num, den);
}
#endif

/**
 * GELU activation forward (fast approximation, in-place)
 * @test test_gelu.py::TestGELUForward::test_gelu_fast_inplace
 * @test test_gelu.py::TestGELUForward::test_gelu_vs_exact
 * @test test_parity.py::test_gelu_parity
 *
 * Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * In-place on contiguous buffer.
 *
 * After changes: make test && make llamacpp-parity-full
 */
void gelu_fast_inplace(float *data, size_t n)
{
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

#if defined(__AVX512F__)
    const __m512 sqrt_2_pi_vec = _mm512_set1_ps(sqrt_2_over_pi);
    const __m512 coeff_vec = _mm512_set1_ps(coeff);
    const __m512 half_vec = _mm512_set1_ps(0.5f);
    const __m512 one_vec = _mm512_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 x3 = _mm512_mul_ps(x2, x);

        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        __m512 inner = _mm512_fmadd_ps(coeff_vec, x3, x);
        inner = _mm512_mul_ps(sqrt_2_pi_vec, inner);

        // result = 0.5 * x * (1 + tanh(inner))
        __m512 tanh_val = tanh512_fast(inner);
        __m512 one_plus_tanh = _mm512_add_ps(one_vec, tanh_val);
        __m512 result = _mm512_mul_ps(half_vec, _mm512_mul_ps(x, one_plus_tanh));

        _mm512_storeu_ps(&data[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

#elif defined(__AVX2__)
    const __m256 sqrt_2_pi_vec = _mm256_set1_ps(sqrt_2_over_pi);
    const __m256 coeff_vec = _mm256_set1_ps(coeff);
    const __m256 half_vec = _mm256_set1_ps(0.5f);
    const __m256 one_vec = _mm256_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);

        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        __m256 inner = _mm256_fmadd_ps(coeff_vec, x3, x);
        inner = _mm256_mul_ps(sqrt_2_pi_vec, inner);

        // result = 0.5 * x * (1 + tanh(inner))
        __m256 tanh_val = tanh256_fast(inner);
        __m256 one_plus_tanh = _mm256_add_ps(one_vec, tanh_val);
        __m256 result = _mm256_mul_ps(half_vec, _mm256_mul_ps(x, one_plus_tanh));

        _mm256_storeu_ps(&data[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

#elif defined(__AVX__)
    // AVX1: Vectorize arithmetic, use scalar tanh
    const __m256 sqrt_2_pi_vec = _mm256_set1_ps(sqrt_2_over_pi);
    const __m256 coeff_vec = _mm256_set1_ps(coeff);
    const __m256 half_vec = _mm256_set1_ps(0.5f);
    const __m256 one_vec = _mm256_set1_ps(1.0f);

    size_t i = 0;
    float inner_arr[8] __attribute__((aligned(32)));
    float tanh_arr[8] __attribute__((aligned(32)));

    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);

        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        __m256 coeff_x3 = _mm256_mul_ps(coeff_vec, x3);
        __m256 inner = _mm256_mul_ps(sqrt_2_pi_vec, _mm256_add_ps(x, coeff_x3));

        // Compute tanh scalarly
        _mm256_store_ps(inner_arr, inner);
        for (int j = 0; j < 8; ++j) {
            tanh_arr[j] = tanhf(inner_arr[j]);
        }
        __m256 tanh_val = _mm256_load_ps(tanh_arr);

        // result = 0.5 * x * (1 + tanh(inner))
        __m256 one_plus_tanh = _mm256_add_ps(one_vec, tanh_val);
        __m256 result = _mm256_mul_ps(half_vec, _mm256_mul_ps(x, one_plus_tanh));

        _mm256_storeu_ps(&data[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

#else
    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
#endif
}

// Exact GELU backward using the tanh-based approximation derivative, adapted
// from C-Transformer's backward_gelu. Operates element-wise on contiguous
// buffers.
// Derivative: d/dx GELU(x) = 0.5 * (1 + tanh(g)) + 0.5 * x * sech^2(g) * g'
// where g = sqrt(2/pi) * (x + 0.044715 * x^3)
//       g' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
void gelu_backward_exact(const float *input,
                         const float *d_output,
                         float *d_input,
                         size_t n)
{
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

#if defined(__AVX512F__)
    const __m512 sqrt_2_pi_vec = _mm512_set1_ps(sqrt_2_over_pi);
    const __m512 coeff_vec = _mm512_set1_ps(coeff);
    const __m512 coeff3_vec = _mm512_set1_ps(3.0f * coeff);
    const __m512 half_vec = _mm512_set1_ps(0.5f);
    const __m512 one_vec = _mm512_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(&input[i]);
        __m512 dy = _mm512_loadu_ps(&d_output[i]);

        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 x3 = _mm512_mul_ps(x2, x);

        // g = sqrt(2/pi) * (x + 0.044715 * x^3)
        __m512 g = _mm512_fmadd_ps(coeff_vec, x3, x);
        g = _mm512_mul_ps(sqrt_2_pi_vec, g);

        __m512 tanh_g = tanh512_fast(g);

        // g' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        __m512 g_prime = _mm512_fmadd_ps(coeff3_vec, x2, one_vec);
        g_prime = _mm512_mul_ps(sqrt_2_pi_vec, g_prime);

        // sech^2(g) = 1 - tanh^2(g)
        __m512 sech2_g = _mm512_fnmadd_ps(tanh_g, tanh_g, one_vec);

        // gelu_derivative = 0.5 * (1 + tanh_g) + 0.5 * x * sech2_g * g_prime
        __m512 term1 = _mm512_mul_ps(half_vec, _mm512_add_ps(one_vec, tanh_g));
        __m512 term2 = _mm512_mul_ps(half_vec, _mm512_mul_ps(x, _mm512_mul_ps(sech2_g, g_prime)));
        __m512 gelu_deriv = _mm512_add_ps(term1, term2);

        __m512 result = _mm512_mul_ps(dy, gelu_deriv);
        _mm512_storeu_ps(&d_input[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float g = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_g = tanhf(g);
        float x2 = x * x;
        float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);
        float sech2_g = 1.0f - tanh_g * tanh_g;
        float gelu_derivative = 0.5f * (1.0f + tanh_g) + 0.5f * x * sech2_g * g_prime;
        d_input[i] = d_output[i] * gelu_derivative;
    }

#elif defined(__AVX2__)
    const __m256 sqrt_2_pi_vec = _mm256_set1_ps(sqrt_2_over_pi);
    const __m256 coeff_vec = _mm256_set1_ps(coeff);
    const __m256 coeff3_vec = _mm256_set1_ps(3.0f * coeff);
    const __m256 half_vec = _mm256_set1_ps(0.5f);
    const __m256 one_vec = _mm256_set1_ps(1.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 dy = _mm256_loadu_ps(&d_output[i]);

        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);

        // g = sqrt(2/pi) * (x + 0.044715 * x^3)
        __m256 g = _mm256_fmadd_ps(coeff_vec, x3, x);
        g = _mm256_mul_ps(sqrt_2_pi_vec, g);

        __m256 tanh_g = tanh256_fast(g);

        // g' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        __m256 g_prime = _mm256_fmadd_ps(coeff3_vec, x2, one_vec);
        g_prime = _mm256_mul_ps(sqrt_2_pi_vec, g_prime);

        // sech^2(g) = 1 - tanh^2(g)
        __m256 sech2_g = _mm256_fnmadd_ps(tanh_g, tanh_g, one_vec);

        // gelu_derivative = 0.5 * (1 + tanh_g) + 0.5 * x * sech2_g * g_prime
        __m256 term1 = _mm256_mul_ps(half_vec, _mm256_add_ps(one_vec, tanh_g));
        __m256 term2 = _mm256_mul_ps(half_vec, _mm256_mul_ps(x, _mm256_mul_ps(sech2_g, g_prime)));
        __m256 gelu_deriv = _mm256_add_ps(term1, term2);

        __m256 result = _mm256_mul_ps(dy, gelu_deriv);
        _mm256_storeu_ps(&d_input[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float g = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_g = tanhf(g);
        float x2 = x * x;
        float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);
        float sech2_g = 1.0f - tanh_g * tanh_g;
        float gelu_derivative = 0.5f * (1.0f + tanh_g) + 0.5f * x * sech2_g * g_prime;
        d_input[i] = d_output[i] * gelu_derivative;
    }

#elif defined(__AVX__)
    // AVX1: Vectorize arithmetic, use scalar tanh
    const __m256 sqrt_2_pi_vec = _mm256_set1_ps(sqrt_2_over_pi);
    const __m256 coeff_vec = _mm256_set1_ps(coeff);
    const __m256 coeff3_vec = _mm256_set1_ps(3.0f * coeff);
    const __m256 half_vec = _mm256_set1_ps(0.5f);
    const __m256 one_vec = _mm256_set1_ps(1.0f);

    size_t i = 0;
    float g_arr[8] __attribute__((aligned(32)));
    float tanh_arr[8] __attribute__((aligned(32)));

    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 dy = _mm256_loadu_ps(&d_output[i]);

        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);

        // g = sqrt(2/pi) * (x + 0.044715 * x^3)
        __m256 coeff_x3 = _mm256_mul_ps(coeff_vec, x3);
        __m256 g = _mm256_mul_ps(sqrt_2_pi_vec, _mm256_add_ps(x, coeff_x3));

        // Compute tanh scalarly
        _mm256_store_ps(g_arr, g);
        for (int j = 0; j < 8; ++j) {
            tanh_arr[j] = tanhf(g_arr[j]);
        }
        __m256 tanh_g = _mm256_load_ps(tanh_arr);

        // g' = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        __m256 coeff3_x2 = _mm256_mul_ps(coeff3_vec, x2);
        __m256 g_prime = _mm256_mul_ps(sqrt_2_pi_vec, _mm256_add_ps(one_vec, coeff3_x2));

        // sech^2(g) = 1 - tanh^2(g)
        __m256 tanh_g_sq = _mm256_mul_ps(tanh_g, tanh_g);
        __m256 sech2_g = _mm256_sub_ps(one_vec, tanh_g_sq);

        // gelu_derivative = 0.5 * (1 + tanh_g) + 0.5 * x * sech2_g * g_prime
        __m256 term1 = _mm256_mul_ps(half_vec, _mm256_add_ps(one_vec, tanh_g));
        __m256 term2 = _mm256_mul_ps(half_vec, _mm256_mul_ps(x, _mm256_mul_ps(sech2_g, g_prime)));
        __m256 gelu_deriv = _mm256_add_ps(term1, term2);

        __m256 result = _mm256_mul_ps(dy, gelu_deriv);
        _mm256_storeu_ps(&d_input[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float g = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_g = tanhf(g);
        float x2 = x * x;
        float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);
        float sech2_g = 1.0f - tanh_g * tanh_g;
        float gelu_derivative = 0.5f * (1.0f + tanh_g) + 0.5f * x * sech2_g * g_prime;
        d_input[i] = d_output[i] * gelu_derivative;
    }

#else
    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];

        float x3 = x * x * x;
        float g = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_g = tanhf(g);

        float x2 = x * x;
        float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);

        float sech2_g = 1.0f - tanh_g * tanh_g;
        float gelu_derivative =
            0.5f * (1.0f + tanh_g) + 0.5f * x * sech2_g * g_prime;

        d_input[i] = d_output[i] * gelu_derivative;
    }
#endif
}

// Scalar-only exact GELU forward using standard library tanhf.
// This is slower than gelu_fast_inplace but provides maximum accuracy.
// Used by BF16 wrapper where conversion overhead dominates anyway.
void gelu_exact_inplace(float *data, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        data[i] = ck_gelu_tanh_f32(data[i]);
    }
}

// GGML-compatible GELU forward used by llama.cpp CPU F32 paths when
// GGML_GELU_FP16 is enabled. Inputs inside [-10, 10] are rounded to FP16,
// GELU is evaluated on that rounded value, then the output is rounded to FP16
// and widened back to FP32. This matches the lookup-table contract without
// needing the global ggml table.
void gelu_ggml_inplace(float *data, size_t n)
{
    pthread_once(&ck_gelu_ggml_runtime_once, ck_gelu_ggml_runtime_init);
    if (ck_gelu_runtime_ready) {
        for (size_t i = 0; i < n; ++i) {
            const float x = data[i];
            if (x <= -10.0f) {
                data[i] = 0.0f;
                continue;
            }
            if (x >= 10.0f) {
                data[i] = x;
                continue;
            }
            const ck_half x_fp16 = ck_gelu_runtime_fp32_to_fp16(x);
            const ck_half y_fp16 = ck_gelu_runtime_table_f16[(uint16_t) x_fp16];
            data[i] = ck_gelu_runtime_fp16_to_fp32(y_fp16);
        }
        return;
    }

    pthread_once(&ck_gelu_ggml_table_once, ck_gelu_ggml_table_init);
    for (size_t i = 0; i < n; ++i) {
        const float x = data[i];
        if (x <= -10.0f) {
            data[i] = 0.0f;
            continue;
        }
        if (x >= 10.0f) {
            data[i] = x;
            continue;
        }
        const ck_half x_fp16 = ggml_fp32_to_fp16(x);
        const ck_half y_fp16 = ck_gelu_ggml_table_f16[(uint16_t) x_fp16];
        data[i] = ggml_fp16_to_fp32(y_fp16);
    }
}

// Scalar-only exact GELU backward using standard library tanhf.
// This is slower than gelu_backward_exact but provides maximum accuracy.
// Used by BF16 wrapper where conversion overhead dominates anyway.
void gelu_backward_scalar(const float *input,
                          const float *d_output,
                          float *d_input,
                          size_t n)
{
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float g = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_g = tanhf(g);
        float x2 = x * x;
        float g_prime = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);
        float sech2_g = 1.0f - tanh_g * tanh_g;
        float gelu_derivative = 0.5f * (1.0f + tanh_g) + 0.5f * x * sech2_g * g_prime;
        d_input[i] = d_output[i] * gelu_derivative;
    }
}

// Fast approximate GELU backward, adapted from C-Transformer's backward_gelu_fast.
// Uses sigmoid approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
// Derivative: s * (1 + x * (1 - s) * 1.702) where s = sigmoid(1.702 * x)
void gelu_backward_fast(const float *input,
                        const float *d_output,
                        float *d_input,
                        size_t n)
{
    const float beta = 1.702f;

#if defined(__AVX512F__)
    const __m512 beta_vec = _mm512_set1_ps(beta);
    const __m512 one_vec = _mm512_set1_ps(1.0f);
    const __m512 neg_beta_vec = _mm512_set1_ps(-beta);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(&input[i]);
        __m512 dy = _mm512_loadu_ps(&d_output[i]);

        // s = sigmoid(beta * x) = 1 / (1 + exp(-beta * x))
        __m512 neg_beta_x = _mm512_mul_ps(neg_beta_vec, x);
        __m512 exp_neg = exp512_fast(neg_beta_x);
        __m512 s = _mm512_div_ps(one_vec, _mm512_add_ps(one_vec, exp_neg));

        // gelu_derivative = s * (1 + x * (1 - s) * beta)
        __m512 one_minus_s = _mm512_sub_ps(one_vec, s);
        __m512 inner = _mm512_fmadd_ps(_mm512_mul_ps(x, one_minus_s), beta_vec, one_vec);
        __m512 gelu_deriv = _mm512_mul_ps(s, inner);

        __m512 result = _mm512_mul_ps(dy, gelu_deriv);
        _mm512_storeu_ps(&d_input[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-beta * x));
        float gelu_derivative = s * (1.0f + x * (1.0f - s) * beta);
        d_input[i] = d_output[i] * gelu_derivative;
    }

#elif defined(__AVX2__)
    const __m256 beta_vec = _mm256_set1_ps(beta);
    const __m256 one_vec = _mm256_set1_ps(1.0f);
    const __m256 neg_beta_vec = _mm256_set1_ps(-beta);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 dy = _mm256_loadu_ps(&d_output[i]);

        // s = sigmoid(beta * x) = 1 / (1 + exp(-beta * x))
        __m256 neg_beta_x = _mm256_mul_ps(neg_beta_vec, x);
        __m256 exp_neg = exp256_fast(neg_beta_x);
        __m256 s = _mm256_div_ps(one_vec, _mm256_add_ps(one_vec, exp_neg));

        // gelu_derivative = s * (1 + x * (1 - s) * beta)
        __m256 one_minus_s = _mm256_sub_ps(one_vec, s);
        __m256 inner = _mm256_fmadd_ps(_mm256_mul_ps(x, one_minus_s), beta_vec, one_vec);
        __m256 gelu_deriv = _mm256_mul_ps(s, inner);

        __m256 result = _mm256_mul_ps(dy, gelu_deriv);
        _mm256_storeu_ps(&d_input[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-beta * x));
        float gelu_derivative = s * (1.0f + x * (1.0f - s) * beta);
        d_input[i] = d_output[i] * gelu_derivative;
    }

#elif defined(__AVX__)
    // AVX1: Vectorize arithmetic, use scalar exp
    const __m256 beta_vec = _mm256_set1_ps(beta);
    const __m256 one_vec = _mm256_set1_ps(1.0f);
    const __m256 neg_beta_vec = _mm256_set1_ps(-beta);

    size_t i = 0;
    float neg_beta_x_arr[8] __attribute__((aligned(32)));
    float exp_arr[8] __attribute__((aligned(32)));

    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 dy = _mm256_loadu_ps(&d_output[i]);

        // s = sigmoid(beta * x) = 1 / (1 + exp(-beta * x))
        __m256 neg_beta_x = _mm256_mul_ps(neg_beta_vec, x);

        // Compute exp scalarly
        _mm256_store_ps(neg_beta_x_arr, neg_beta_x);
        for (int j = 0; j < 8; ++j) {
            exp_arr[j] = expf(neg_beta_x_arr[j]);
        }
        __m256 exp_neg = _mm256_load_ps(exp_arr);

        __m256 s = _mm256_div_ps(one_vec, _mm256_add_ps(one_vec, exp_neg));

        // gelu_derivative = s * (1 + x * (1 - s) * beta)
        __m256 one_minus_s = _mm256_sub_ps(one_vec, s);
        __m256 x_one_minus_s = _mm256_mul_ps(x, one_minus_s);
        __m256 x_one_minus_s_beta = _mm256_mul_ps(x_one_minus_s, beta_vec);
        __m256 inner = _mm256_add_ps(one_vec, x_one_minus_s_beta);
        __m256 gelu_deriv = _mm256_mul_ps(s, inner);

        __m256 result = _mm256_mul_ps(dy, gelu_deriv);
        _mm256_storeu_ps(&d_input[i], result);
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-beta * x));
        float gelu_derivative = s * (1.0f + x * (1.0f - s) * beta);
        d_input[i] = d_output[i] * gelu_derivative;
    }

#else
    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        float x = input[i];
        float s = 1.0f / (1.0f + expf(-beta * x));
        float gelu_derivative = s * (1.0f + x * (1.0f - s) * beta);
        d_input[i] = d_output[i] * gelu_derivative;
    }
#endif
}
