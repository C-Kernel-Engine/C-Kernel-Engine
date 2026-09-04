#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "bf16_utils.h"
#include "ckernel_engine.h"

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

typedef float (*ck_recurrent_libm_f32_fn)(float);
static ck_recurrent_libm_f32_fn ck_recurrent_llama_expf = NULL;
static ck_recurrent_libm_f32_fn ck_recurrent_llama_logf = NULL;
static ck_recurrent_libm_f32_fn ck_recurrent_pytorch_log1pf = NULL;
static void *ck_recurrent_libm_handle = NULL;
static pthread_once_t ck_recurrent_libm_once = PTHREAD_ONCE_INIT;

static void ck_bind_recurrent_llama_libm(void)
{
    ck_recurrent_libm_handle = dlopen("libm.so.6", RTLD_NOW | RTLD_LOCAL);
    if (ck_recurrent_libm_handle) {
        ck_recurrent_llama_expf =
            (ck_recurrent_libm_f32_fn)dlsym(ck_recurrent_libm_handle, "expf");
        ck_recurrent_llama_logf =
            (ck_recurrent_libm_f32_fn)dlsym(ck_recurrent_libm_handle, "logf");
        ck_recurrent_pytorch_log1pf =
            (ck_recurrent_libm_f32_fn)dlsym(ck_recurrent_libm_handle, "log1pf");
    }
    if (!ck_recurrent_llama_expf || !ck_recurrent_llama_logf) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: llama.cpp recurrent softplus "
                "requires expf/logf from libm.so.6\n");
        abort();
    }
}

static inline float recurrent_softplus(float x) {
    pthread_once(&ck_recurrent_libm_once, ck_bind_recurrent_llama_libm);
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return ck_recurrent_llama_expf(x);
    }
    return ck_recurrent_llama_logf(1.0f + ck_recurrent_llama_expf(x));
}

static inline float recurrent_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

void recurrent_dt_gate_forward(const float *alpha,
                               const float *dt_bias,
                               const float *a,
                               float *gate,
                               int rows,
                               int num_heads,
                               int state_dim) {
    const int dim = num_heads * state_dim;
    for (int row = 0; row < rows; ++row) {
        const float *alpha_row = alpha + (size_t) row * (size_t) dim;
        float *gate_row = gate + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float x = alpha_row[col] + dt_bias[col];
            gate_row[col] = recurrent_softplus(x) * a[col];
        }
    }
}

void recurrent_dt_gate_expanded_forward(const float *alpha,
                                        const float *dt_bias,
                                        const float *a,
                                        float *gate,
                                        int rows,
                                        int num_heads,
                                        int state_dim) {
    for (int row = 0; row < rows; ++row) {
        const float *alpha_row = alpha + (size_t) row * (size_t) num_heads;
        float *gate_row = gate + (size_t) row * (size_t) num_heads * (size_t) state_dim;
        for (int h = 0; h < num_heads; ++h) {
            const float sp = recurrent_softplus(alpha_row[h] + dt_bias[h]);
            const float *a_head = a + (size_t) h * (size_t) state_dim;
            float *gate_head = gate_row + (size_t) h * (size_t) state_dim;
            for (int col = 0; col < state_dim; ++col) {
                gate_head[col] = sp * a_head[col];
            }
        }
    }
}

void recurrent_dt_gate_backward(const float *d_gate,
                                const float *alpha,
                                const float *dt_bias,
                                const float *a,
                                float *d_alpha,
                                float *d_dt_bias,
                                float *d_a,
                                int rows,
                                int dim) {
    for (int col = 0; col < dim; ++col) {
        d_dt_bias[col] = 0.0f;
        d_a[col] = 0.0f;
    }

    for (int row = 0; row < rows; ++row) {
        const float *d_gate_row = d_gate + (size_t) row * (size_t) dim;
        const float *alpha_row = alpha + (size_t) row * (size_t) dim;
        float *d_alpha_row = d_alpha + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float x = alpha_row[col] + dt_bias[col];
            const float sp = recurrent_softplus(x);
            const float sig = recurrent_sigmoid(x);
            const float d_out = d_gate_row[col];
            d_a[col] += d_out * sp;
            {
                const float d_x = d_out * a[col] * sig;
                d_alpha_row[col] = d_x;
                d_dt_bias[col] += d_x;
            }
        }
    }
}

void recurrent_silu_forward(const float *x,
                            float *out,
                            int rows,
                            int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t) row * (size_t) dim;
        float *out_row = out + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float xv = x_row[col];
            out_row[col] = xv * recurrent_sigmoid(xv);
        }
    }
}

#if defined(__AVX512F__)
typedef __m512 (*ck_recurrent_sleef_expf16_fn)(__m512);
typedef __m512 (*ck_recurrent_sleef_log1pf16_fn)(__m512);
static ck_recurrent_sleef_expf16_fn ck_recurrent_pytorch_expf16 = NULL;
static ck_recurrent_sleef_log1pf16_fn ck_recurrent_pytorch_log1pf16 = NULL;
static void *ck_recurrent_sleef_handle = NULL;
static pthread_once_t ck_recurrent_sleef_once = PTHREAD_ONCE_INIT;

static void ck_bind_recurrent_pytorch_sleef(void)
{
    const char *library = getenv("CK_SLEEF_LIBRARY");
    if (library && *library) {
        ck_recurrent_sleef_handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        if (ck_recurrent_sleef_handle) {
            ck_recurrent_pytorch_expf16 = (ck_recurrent_sleef_expf16_fn)dlsym(
                ck_recurrent_sleef_handle, "Sleef_expf16_u10");
            ck_recurrent_pytorch_log1pf16 = (ck_recurrent_sleef_log1pf16_fn)dlsym(
                ck_recurrent_sleef_handle, "Sleef_log1pf16_u10");
        }
    } else {
        ck_recurrent_pytorch_expf16 =
            (ck_recurrent_sleef_expf16_fn)dlsym(RTLD_DEFAULT, "Sleef_expf16_u10");
        ck_recurrent_pytorch_log1pf16 =
            (ck_recurrent_sleef_log1pf16_fn)dlsym(RTLD_DEFAULT, "Sleef_log1pf16_u10");
    }
}
#endif

void recurrent_dt_gate_forward_pytorch_fp32(const float *alpha,
                                             const float *dt_bias,
                                             const float *a,
                                             float *gate,
                                             int rows,
                                             int num_heads,
                                             int state_dim)
{
    if (!alpha || !dt_bias || !a || !gate || rows < 0 || num_heads < 0 || state_dim != 1) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: invalid PyTorch FP32 recurrent dt-gate arguments\n");
        abort();
    }
#if defined(__AVX512F__)
    pthread_once(&ck_recurrent_sleef_once, ck_bind_recurrent_pytorch_sleef);
    if (!ck_recurrent_pytorch_expf16 || !ck_recurrent_pytorch_log1pf16) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: PyTorch FP32 recurrent dt gate requires "
                "SLEEF Sleef_expf16_u10 and Sleef_log1pf16_u10; set CK_SLEEF_LIBRARY\n");
        abort();
    }
    if ((num_heads & 15) != 0) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: PyTorch AVX-512 recurrent dt gate requires "
                "a head count divisible by 16 (got %d)\n",
                num_heads);
        abort();
    }
    pthread_once(&ck_recurrent_libm_once, ck_bind_recurrent_llama_libm);
    if (!ck_recurrent_pytorch_log1pf) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: PyTorch FP32 recurrent dt gate requires "
                "log1pf from libm.so.6\n");
        abort();
    }
    const __m512 threshold = _mm512_set1_ps(20.0f);
    const int count = rows * num_heads;
    int index = 0;
    for (; index + 32 <= count; index += 32) {
        for (int half = 0; half < 2; ++half) {
            float x_lanes[16] __attribute__((aligned(64)));
            float a_lanes[16] __attribute__((aligned(64)));
            const int base = index + half * 16;
            for (int lane = 0; lane < 16; ++lane) {
                const int head = (base + lane) % num_heads;
                x_lanes[lane] = alpha[base + lane] + dt_bias[head];
                a_lanes[lane] = a[head];
            }
            const __m512 x = _mm512_load_ps(x_lanes);
            const __m512 softplus = _mm512_mask_blend_ps(
                _mm512_cmp_ps_mask(x, threshold, _CMP_GT_OQ),
                ck_recurrent_pytorch_log1pf16(ck_recurrent_pytorch_expf16(x)),
                x);
            _mm512_storeu_ps(
                gate + base,
                _mm512_mul_ps(softplus, _mm512_load_ps(a_lanes)));
        }
    }
    for (; index < count; ++index) {
        const int head = index % num_heads;
        const float x = alpha[index] + dt_bias[head];
        const float softplus = x > 20.0f
            ? x
            : ck_recurrent_pytorch_log1pf(ck_recurrent_llama_expf(x));
        gate[index] = softplus * a[head];
    }
#else
    fprintf(stderr,
            "HARD KERNEL CONTRACT FAULT: PyTorch FP32 recurrent dt gate requires AVX-512\n");
    abort();
#endif
}

void recurrent_silu_forward_pytorch_bf16_storage(const float *x,
                                                  float *out,
                                                  int rows,
                                                  int dim)
{
    if (!x || !out || rows < 0 || dim < 0) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: invalid PyTorch BF16 recurrent SiLU arguments\n");
        abort();
    }
#if defined(__AVX512F__)
    pthread_once(&ck_recurrent_sleef_once, ck_bind_recurrent_pytorch_sleef);
    if (!ck_recurrent_pytorch_expf16) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: PyTorch BF16 recurrent SiLU requires "
                "SLEEF Sleef_expf16_u10; set CK_SLEEF_LIBRARY\n");
        abort();
    }
#endif
    for (int row = 0; row < rows; ++row) {
        const float *src = x + (size_t)row * (size_t)dim;
        float *dst = out + (size_t)row * (size_t)dim;
        int col = 0;
#if defined(__AVX512F__)
        for (; col + 16 <= dim; col += 16) {
            float lanes[16] __attribute__((aligned(64)));
            for (int lane = 0; lane < 16; ++lane) {
                lanes[lane] = bf16_to_float(float_to_bf16(src[col + lane]));
            }
            const __m512 values = _mm512_load_ps(lanes);
            const __m512 denominator = _mm512_add_ps(
                _mm512_set1_ps(1.0f),
                ck_recurrent_pytorch_expf16(
                    _mm512_sub_ps(_mm512_setzero_ps(), values)));
            _mm512_store_ps(lanes, _mm512_div_ps(values, denominator));
            for (int lane = 0; lane < 16; ++lane) {
                dst[col + lane] = bf16_to_float(float_to_bf16(lanes[lane]));
            }
        }
#endif
        for (; col < dim; ++col) {
            const float value = bf16_to_float(float_to_bf16(src[col]));
            const float silu = value / (1.0f + expf(-value));
            dst[col] = bf16_to_float(float_to_bf16(silu));
        }
    }
}

void recurrent_silu_forward_pytorch_bf16_input_fp32_output(const float *x,
                                                            float *out,
                                                            int rows,
                                                            int dim)
{
    if (!x || !out || rows < 0 || dim < 0) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: invalid BF16-input FP32-output SiLU arguments\n");
        abort();
    }
#if defined(__AVX512F__)
    pthread_once(&ck_recurrent_sleef_once, ck_bind_recurrent_pytorch_sleef);
    if (!ck_recurrent_pytorch_expf16) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: PyTorch BF16-input SiLU requires "
                "SLEEF Sleef_expf16_u10; set CK_SLEEF_LIBRARY\n");
        abort();
    }
#endif
    const int count = rows * dim;
    int i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= count; i += 16) {
        float lanes[16] __attribute__((aligned(64)));
        for (int lane = 0; lane < 16; ++lane) {
            lanes[lane] = bf16_to_float(float_to_bf16(x[i + lane]));
        }
        const __m512 values = _mm512_load_ps(lanes);
        const __m512 denominator = _mm512_add_ps(
            _mm512_set1_ps(1.0f),
            ck_recurrent_pytorch_expf16(
                _mm512_sub_ps(_mm512_setzero_ps(), values)));
        _mm512_storeu_ps(out + i, _mm512_div_ps(values, denominator));
    }
#endif
    for (; i < count; ++i) {
        const float value = bf16_to_float(float_to_bf16(x[i]));
        out[i] = value / (1.0f + expf(-value));
    }
}

void recurrent_sigmoid_forward_pytorch_bf16_input_fp32_output(
                                                            const float *x,
                                                            float *out,
                                                            int rows,
                                                            int dim)
{
    if (!x || !out || rows < 0 || dim < 0) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: invalid BF16-input FP32-output sigmoid arguments\n");
        abort();
    }
#if defined(__AVX512F__)
    pthread_once(&ck_recurrent_sleef_once, ck_bind_recurrent_pytorch_sleef);
    if (!ck_recurrent_pytorch_expf16) {
        fprintf(stderr,
                "HARD KERNEL CONTRACT FAULT: PyTorch BF16-input sigmoid requires "
                "SLEEF Sleef_expf16_u10; set CK_SLEEF_LIBRARY\n");
        abort();
    }
#endif
    const int count = rows * dim;
    int i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= count; i += 16) {
        float lanes[16] __attribute__((aligned(64)));
        for (int lane = 0; lane < 16; ++lane) {
            lanes[lane] = bf16_to_float(float_to_bf16(x[i + lane]));
        }
        const __m512 values = _mm512_load_ps(lanes);
        const __m512 denominator = _mm512_add_ps(
            _mm512_set1_ps(1.0f),
            ck_recurrent_pytorch_expf16(
                _mm512_sub_ps(_mm512_setzero_ps(), values)));
        _mm512_storeu_ps(out + i,
                         _mm512_div_ps(_mm512_set1_ps(1.0f), denominator));
    }
#endif
    for (; i < count; ++i) {
        const float value = bf16_to_float(float_to_bf16(x[i]));
        out[i] = 1.0f / (1.0f + expf(-value));
    }
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)
/* Match ggml's AVX-512 exp approximation and instruction grouping exactly. */
static inline __m512 recurrent_ggml_expf_avx512(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b = _mm512_fnmadd_ps(
        n, _mm512_set1_ps(0x1.7f7d1cp-20f),
        _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __mmask16 d = _mm512_cmp_ps_mask(
        _mm512_abs_ps(n), _mm512_set1_ps(192.0f), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(
            _mm512_fmadd_ps(
                _mm512_set1_ps(0x1.0e4020p-7f), b,
                _mm512_set1_ps(0x1.573e2ep-5f)),
            u,
            _mm512_fmadd_ps(
                _mm512_set1_ps(0x1.555e66p-3f), b,
                _mm512_set1_ps(0x1.fffdb6p-2f))),
        u,
        _mm512_fmadd_ps(
            _mm512_set1_ps(0x1.ffffecp-1f), b,
            _mm512_set1_ps(1.0f)));
    const __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d)) {
        return res;
    }
    const __m512 zero = _mm512_setzero_ps();
    const __m512 alt = _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ),
        _mm512_set1_ps(INFINITY),
        zero);
    return _mm512_mask_blend_ps(d, res, alt);
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
/* Match ggml's AVX2 exp approximation exactly; it is the fallback provider ABI. */
static inline __m256 recurrent_ggml_expf_avx2(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(
        n, _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(
        _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
    const __m256i c = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
                      _mm256_set1_ps(126.0f), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                            _mm256_set1_ps(0x1.573e2ep-5f)),
            u,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                            _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
        return _mm256_fmadd_ps(j, k, k);
    }
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32((int) 0x82000000u));
    const __m256 s1 = _mm256_castsi256_ps(
        _mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
                      _mm256_set1_ps(192.0f), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(
                    _mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}
#endif

void recurrent_silu_forward_ggml(const float *x,
                                 float *out,
                                 int rows,
                                 int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t) row * (size_t) dim;
        float *out_row = out + (size_t) row * (size_t) dim;
        int col = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
        for (; col + 16 <= dim; col += 16) {
            const __m512 xv = _mm512_loadu_ps(x_row + col);
            const __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), xv);
            const __m512 denom = _mm512_add_ps(
                _mm512_set1_ps(1.0f), recurrent_ggml_expf_avx512(neg));
            _mm512_storeu_ps(out_row + col, _mm512_div_ps(xv, denom));
        }
#elif defined(__AVX2__) && defined(__FMA__)
        for (; col + 8 <= dim; col += 8) {
            const __m256 xv = _mm256_loadu_ps(x_row + col);
            const __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), xv);
            const __m256 denom = _mm256_add_ps(
                _mm256_set1_ps(1.0f), recurrent_ggml_expf_avx2(neg));
            _mm256_storeu_ps(out_row + col, _mm256_div_ps(xv, denom));
        }
#endif
        float (*volatile llama_expf)(float) = expf;
        for (; col < dim; ++col) {
            const float xv = x_row[col];
            out_row[col] = xv / (1.0f + llama_expf(-xv));
        }
    }
}

void recurrent_sigmoid_forward_ggml(const float *x,
                                    float *out,
                                    int rows,
                                    int dim) {
    float (*volatile llama_expf)(float) = expf;
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t) row * (size_t) dim;
        float *out_row = out + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            out_row[col] = 1.0f / (1.0f + llama_expf(-x_row[col]));
        }
    }
}

void recurrent_silu_backward(const float *d_out,
                             const float *x,
                             float *d_x,
                             int rows,
                             int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *d_out_row = d_out + (size_t) row * (size_t) dim;
        const float *x_row = x + (size_t) row * (size_t) dim;
        float *d_x_row = d_x + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float xv = x_row[col];
            const float sig = recurrent_sigmoid(xv);
            d_x_row[col] = d_out_row[col] * (sig + xv * sig * (1.0f - sig));
        }
    }
}
