/**
 * @file swiglu_kernels.c
 * @brief SwiGLU activation kernels with SIMD (SSE/AVX/AVX512)
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
 * SwiGLU: y = silu(gate) * up = (gate * sigmoid(gate)) * up
 */

#include "ckernel_engine.h"
#include "ckernel_quant.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/*
 * PyTorch-parity sigmoid for strict/exact SwiGLU paths.
 * Keep this in fp32 to match ATen CPU opmath more closely.
 */
static inline float sigmoid_scalar_parity(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* ========================================================================== */
/* Fast exp approximation for SIMD                                            */
/* ========================================================================== */

#if defined(__AVX512F__)
// AVX-512 fast exp approximation
static inline __m512 exp512_fast(__m512 x) {
    // Clamp to avoid overflow/underflow
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));

    // exp(x) = 2^(x * log2(e))
    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    __m512 z = _mm512_mul_ps(x, log2e);

    // Split into integer and fractional parts
    __m512 zf = _mm512_roundscale_ps(z, _MM_FROUND_TO_NEAREST_INT);
    __m512 f = _mm512_sub_ps(z, zf);

    // Polynomial for 2^f, f in [-0.5, 0.5]
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 c2 = _mm512_set1_ps(0.2402265069591007f);
    const __m512 c3 = _mm512_set1_ps(0.05550410866482158f);
    const __m512 c4 = _mm512_set1_ps(0.009618129107628478f);

    __m512 poly = _mm512_fmadd_ps(f, c4, c3);
    poly = _mm512_fmadd_ps(f, poly, c2);
    poly = _mm512_fmadd_ps(f, poly, c1);
    poly = _mm512_fmadd_ps(f, poly, c0);

    // Scale by 2^n
    __m512i zi = _mm512_cvtps_epi32(zf);
    zi = _mm512_add_epi32(zi, _mm512_set1_epi32(127));
    zi = _mm512_slli_epi32(zi, 23);
    __m512 scale = _mm512_castsi512_ps(zi);

    return _mm512_mul_ps(poly, scale);
}

// AVX-512 sigmoid: 1 / (1 + exp(-x))
static inline __m512 sigmoid512_fast(__m512 x) {
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    __m512 exp_neg = exp512_fast(neg_x);
    __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(one, _mm512_add_ps(one, exp_neg));
}
#endif

#if defined(__AVX2__)
// AVX2 fast exp approximation (needs FMA and integer ops)
static inline __m256 exp256_fast(__m256 x) {
    // Clamp
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    // exp(x) = 2^(x * log2(e))
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    __m256 z = _mm256_mul_ps(x, log2e);

    // Round to nearest integer
    __m256 zf = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 f = _mm256_sub_ps(z, zf);

    // Polynomial for 2^f
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628478f);

    __m256 poly = _mm256_fmadd_ps(f, c4, c3);
    poly = _mm256_fmadd_ps(f, poly, c2);
    poly = _mm256_fmadd_ps(f, poly, c1);
    poly = _mm256_fmadd_ps(f, poly, c0);

    // Scale by 2^n
    __m256i zi = _mm256_cvtps_epi32(zf);
    zi = _mm256_add_epi32(zi, _mm256_set1_epi32(127));
    zi = _mm256_slli_epi32(zi, 23);
    __m256 scale = _mm256_castsi256_ps(zi);

    return _mm256_mul_ps(poly, scale);
}

// AVX2 sigmoid
static inline __m256 sigmoid256_fast(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = exp256_fast(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
}

/* GGML's parity path uses this specific vector exponential approximation. */
static inline __m256 ck_ggml_expf_avx2(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(
        _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
                      _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                            _mm256_set1_ps(0x1.573e2ep-5f)),
            u,
            _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                            _mm256_set1_ps(0x1.fffdb6p-2f))),
        u,
        _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
        return _mm256_fmadd_ps(j, k, k);
    }
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32((int)0x82000000u));
    const __m256 s1 = _mm256_castsi256_ps(
        _mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), n),
                      _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(_mm256_castsi256_ps(c),
                              _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(_mm256_castsi256_ps(c),
                                 _mm256_fmadd_ps(k, j, k)))));
}
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
/* Keep the production parity provider aligned with llama.cpp's AVX-512
 * exponential approximation and instruction grouping. */
static inline __m512 ck_ggml_expf_avx512(__m512 x) {
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

/**
 * SwiGLU forward pass
 * @test test_swiglu.py::TestSwiGLUForward::test_forward_tokens
 * @test test_swiglu.py::TestSwiGLUForward::test_forward_single
 * @test test_mlp.py::TestMLPForward::test_swiglu_mlp
 * @test test_fused_swiglu_decode.py::TestFusedSwiGLUDecode::test_fused_swiglu_decode
 * @test test_parity.py::test_swiglu_parity
 *
 * SwiGLU: y = silu(gate) * up where silu(x) = x * sigmoid(x)
 *
 * After changes: make test && make llamacpp-parity-full
 */
void swiglu_forward(const float *input,
                    float *output,
                    int tokens,
                    int dim)
{
    const char *fast_env = getenv("CK_SWIGLU_FAST");
    const char *exact_env = getenv("CK_SWIGLU_EXACT");
    if (ck_strict_parity_enabled() ||
        !(fast_env && atoi(fast_env) != 0) ||
        (exact_env && atoi(exact_env) != 0)) {
        swiglu_forward_exact(input, output, tokens, dim);
        return;
    }

    int T = tokens;
    int D = dim;

    for (int t = 0; t < T; ++t) {
        const float *row = input + (size_t)t * (2 * D);
        float *out_row = output + (size_t)t * D;
        int d = 0;

#if defined(__AVX512F__)
        // AVX-512: Process 16 floats at a time
        for (; d + 16 <= D; d += 16) {
            __m512 a = _mm512_loadu_ps(&row[d]);         // gate
            __m512 b = _mm512_loadu_ps(&row[D + d]);     // value

            __m512 s = sigmoid512_fast(a);              // sigmoid(a)
            __m512 silu = _mm512_mul_ps(a, s);          // silu(a) = a * sigmoid(a)
            __m512 y = _mm512_mul_ps(silu, b);          // y = silu(a) * b

            _mm512_storeu_ps(&out_row[d], y);
        }
#elif defined(__AVX2__)
        // AVX2: Process 8 floats at a time
        for (; d + 8 <= D; d += 8) {
            __m256 a = _mm256_loadu_ps(&row[d]);         // gate
            __m256 b = _mm256_loadu_ps(&row[D + d]);     // value

            __m256 s = sigmoid256_fast(a);              // sigmoid(a)
            __m256 silu = _mm256_mul_ps(a, s);          // silu(a) = a * sigmoid(a)
            __m256 y = _mm256_mul_ps(silu, b);          // y = silu(a) * b

            _mm256_storeu_ps(&out_row[d], y);
        }
#elif defined(__AVX__)
        // AVX1: Vectorize arithmetic, use scalar sigmoid
        float a_arr[8] __attribute__((aligned(32)));
        float s_arr[8] __attribute__((aligned(32)));

        for (; d + 8 <= D; d += 8) {
            __m256 a = _mm256_loadu_ps(&row[d]);         // gate
            __m256 b = _mm256_loadu_ps(&row[D + d]);     // value

            // Compute sigmoid scalarly
            _mm256_store_ps(a_arr, a);
            for (int j = 0; j < 8; ++j) {
                s_arr[j] = sigmoid_scalar(a_arr[j]);
            }
            __m256 s = _mm256_load_ps(s_arr);

            __m256 silu = _mm256_mul_ps(a, s);          // silu(a) = a * sigmoid(a)
            __m256 y = _mm256_mul_ps(silu, b);          // y = silu(a) * b

            _mm256_storeu_ps(&out_row[d], y);
        }
#endif

        // Scalar fallback for remaining elements
        for (; d < D; ++d) {
            float a = row[d];       // gate
            float b = row[D + d];   // value

            float s = sigmoid_scalar(a);         // sigmoid(a)
            float silu = a * s;                  // silu(a) = a * sigmoid(a)

            out_row[d] = silu * b;
        }
    }
}

void swiglu_forward_q8_k(const float *input,
                         void *output_q8,
                         int tokens,
                         int dim)
{
    if (!input || !output_q8 || tokens <= 0 || dim <= 0) {
        return;
    }
    if ((dim % QK_K) != 0) {
        return;
    }

    const char *fast_env = getenv("CK_SWIGLU_FAST");
    const char *exact_env = getenv("CK_SWIGLU_EXACT");
    const int use_fast = !ck_strict_parity_enabled() &&
                         (fast_env && atoi(fast_env) != 0) &&
                         !(exact_env && atoi(exact_env) != 0);

    const int blocks_per_row = dim / QK_K;
    block_q8_K *q8 = (block_q8_K *)output_q8;
    float tmp[QK_K];

    for (int t = 0; t < tokens; ++t) {
        const float *row = input + (size_t)t * (size_t)(2 * dim);
        block_q8_K *q8_row = q8 + (size_t)t * (size_t)blocks_per_row;

        for (int block = 0; block < blocks_per_row; ++block) {
            const int base = block * QK_K;
            int d = 0;

#if defined(__AVX2__)
            if (use_fast) {
                for (; d + 8 <= QK_K; d += 8) {
                    const __m256 a = _mm256_loadu_ps(row + base + d);
                    const __m256 b = _mm256_loadu_ps(row + dim + base + d);
                    const __m256 s = sigmoid256_fast(a);
                    const __m256 y = _mm256_mul_ps(_mm256_mul_ps(a, s), b);
                    _mm256_storeu_ps(tmp + d, y);
                }
            }
#else
            (void)use_fast;
#endif

            for (; d < QK_K; ++d) {
                const float a = row[base + d];
                const float b = row[dim + base + d];
                const float s = use_fast ? sigmoid_scalar(a) : sigmoid_scalar_parity(a);
                tmp[d] = (a * s) * b;
            }

            quantize_row_q8_k(tmp, (void *)&q8_row[block], QK_K);
        }
    }
}

/**
 * SwiGLU backward pass
 * @test test_swiglu.py::TestSwiGLUBackward::test_backward_tokens
 * @test test_swiglu.py::TestSwiGLUBackward::test_backward_single
 * @test test_parity.py::test_swiglu_backward_parity
 *
 * Computes dGate and dUp given dY.
 * dGate = dy * b * silu'(a), dUp = dy * silu(a)
 *
 * After changes: make test && make llamacpp-parity-full
 */
void swiglu_backward(const float *input,
                     const float *d_output,
                     float *d_input,
                     int tokens,
                     int dim)
{
    if (ck_strict_parity_enabled()) {
        swiglu_backward_exact(input, d_output, d_input, tokens, dim);
        return;
    }

    int T = tokens;
    int D = dim;

    for (int t = 0; t < T; ++t) {
        const float *row = input + (size_t)t * (2 * D);
        const float *dy_row = d_output + (size_t)t * D;
        float *dx_row = d_input + (size_t)t * (2 * D);
        int d = 0;

#if defined(__AVX512F__)
        // AVX-512: Process 16 floats at a time
        __m512 one = _mm512_set1_ps(1.0f);
        for (; d + 16 <= D; d += 16) {
            __m512 a = _mm512_loadu_ps(&row[d]);         // gate
            __m512 b = _mm512_loadu_ps(&row[D + d]);     // value
            __m512 dy = _mm512_loadu_ps(&dy_row[d]);

            __m512 s = sigmoid512_fast(a);              // sigmoid(a)
            __m512 silu = _mm512_mul_ps(a, s);          // silu(a) = a * s
            __m512 one_minus_s = _mm512_sub_ps(one, s);
            __m512 inner = _mm512_fmadd_ps(a, one_minus_s, one);      // 1 + a * (1 - s)
            __m512 silu_prime = _mm512_mul_ps(s, inner);              // s * (1 + a * (1 - s))

            // dA = dy * b * silu_prime
            __m512 dA = _mm512_mul_ps(dy, _mm512_mul_ps(b, silu_prime));
            // dB = dy * silu
            __m512 dB = _mm512_mul_ps(dy, silu);

            _mm512_storeu_ps(&dx_row[d], dA);
            _mm512_storeu_ps(&dx_row[D + d], dB);
        }
#elif defined(__AVX2__)
        // AVX2: Process 8 floats at a time
        __m256 one = _mm256_set1_ps(1.0f);
        for (; d + 8 <= D; d += 8) {
            __m256 a = _mm256_loadu_ps(&row[d]);         // gate
            __m256 b = _mm256_loadu_ps(&row[D + d]);     // value
            __m256 dy = _mm256_loadu_ps(&dy_row[d]);

            __m256 s = sigmoid256_fast(a);              // sigmoid(a)
            __m256 silu = _mm256_mul_ps(a, s);          // silu(a) = a * s
            __m256 one_minus_s = _mm256_sub_ps(one, s);
            __m256 inner = _mm256_fmadd_ps(a, one_minus_s, one);      // 1 + a * (1 - s)
            __m256 silu_prime = _mm256_mul_ps(s, inner);              // s * (1 + a * (1 - s))

            // dA = dy * b * silu_prime
            __m256 dA = _mm256_mul_ps(dy, _mm256_mul_ps(b, silu_prime));
            // dB = dy * silu
            __m256 dB = _mm256_mul_ps(dy, silu);

            _mm256_storeu_ps(&dx_row[d], dA);
            _mm256_storeu_ps(&dx_row[D + d], dB);
        }
#elif defined(__AVX__)
        // AVX1: Vectorize arithmetic, use scalar sigmoid
        __m256 one = _mm256_set1_ps(1.0f);
        float a_arr[8] __attribute__((aligned(32)));
        float s_arr[8] __attribute__((aligned(32)));

        for (; d + 8 <= D; d += 8) {
            __m256 a = _mm256_loadu_ps(&row[d]);         // gate
            __m256 b = _mm256_loadu_ps(&row[D + d]);     // value
            __m256 dy = _mm256_loadu_ps(&dy_row[d]);

            // Compute sigmoid scalarly
            _mm256_store_ps(a_arr, a);
            for (int j = 0; j < 8; ++j) {
                s_arr[j] = sigmoid_scalar(a_arr[j]);
            }
            __m256 s = _mm256_load_ps(s_arr);

            __m256 silu = _mm256_mul_ps(a, s);                        // silu(a) = a * s
            __m256 one_minus_s = _mm256_sub_ps(one, s);
            __m256 a_one_minus_s = _mm256_mul_ps(a, one_minus_s);
            __m256 inner = _mm256_add_ps(one, a_one_minus_s);         // 1 + a * (1 - s)
            __m256 silu_prime = _mm256_mul_ps(s, inner);              // s * (1 + a * (1 - s))

            // dA = dy * b * silu_prime
            __m256 dA = _mm256_mul_ps(dy, _mm256_mul_ps(b, silu_prime));
            // dB = dy * silu
            __m256 dB = _mm256_mul_ps(dy, silu);

            _mm256_storeu_ps(&dx_row[d], dA);
            _mm256_storeu_ps(&dx_row[D + d], dB);
        }
#endif

        // Scalar fallback for remaining elements
        for (; d < D; ++d) {
            float a = row[d];       // gate
            float b = row[D + d];   // value
            float dy = dy_row[d];

            float s = sigmoid_scalar(a);               // sigmoid(a)
            float silu = a * s;                        // silu(a)
            float silu_prime = s * (1.0f + a * (1.0f - s)); // silu'(a), PyTorch form

            float dA = dy * b * silu_prime;
            float dB = dy * silu;

            dx_row[d] = dA;
            dx_row[D + d] = dB;
        }
    }
}
// ============================================================================
// Exact versions using standard library expf (slower but accurate)
// ============================================================================

/**
 * SwiGLU forward pass (exact version using stdlib sigmoid)
 * @test test_swiglu.py::TestSwiGLUForward::test_exact_vs_fast
 * @test test_swiglu.py::TestSwiGLUForward::test_exact_single
 *
 * Uses standard library expf for numerical accuracy reference.
 *
 * After changes: make test
 */
void swiglu_forward_exact(const float *input,
                          float *output,
                          int tokens,
                          int dim)
{
    int T = tokens;
    int D = dim;

    for (int t = 0; t < T; ++t) {
        const float *row = input + (size_t)t * (2 * D);
        float *out_row = output + (size_t)t * D;

        for (int d = 0; d < D; ++d) {
            float a = row[d];       // gate
            float b = row[D + d];   // value

            float s = sigmoid_scalar_parity(a); // sigmoid(a)
            float silu = a * s;                 // silu(a)
            out_row[d] = silu * b;
        }
    }
}

void swiglu_forward_ggml(const float *input,
                         float *output,
                         int tokens,
                         int dim)
{
    for (int t = 0; t < tokens; ++t) {
        const float *row = input + (size_t)t * (2 * dim);
        float *out_row = output + (size_t)t * dim;
        int d = 0;

#if defined(__AVX512F__) && defined(__AVX512DQ__)
        for (; d + 16 <= dim; d += 16) {
            const __m512 gate = _mm512_loadu_ps(row + d);
            const __m512 up = _mm512_loadu_ps(row + dim + d);
            const __m512 neg_gate = _mm512_sub_ps(_mm512_setzero_ps(), gate);
            const __m512 denom = _mm512_add_ps(
                _mm512_set1_ps(1.0f), ck_ggml_expf_avx512(neg_gate));
            const __m512 silu = _mm512_div_ps(gate, denom);
            _mm512_storeu_ps(out_row + d, _mm512_mul_ps(silu, up));
        }
#elif defined(__AVX2__) && defined(__FMA__)
        for (; d + 8 <= dim; d += 8) {
            const __m256 gate = _mm256_loadu_ps(row + d);
            const __m256 up = _mm256_loadu_ps(row + dim + d);
            const __m256 neg_gate = _mm256_sub_ps(_mm256_setzero_ps(), gate);
            const __m256 denom = _mm256_add_ps(
                _mm256_set1_ps(1.0f), ck_ggml_expf_avx2(neg_gate));
            const __m256 silu = _mm256_div_ps(gate, denom);
            _mm256_storeu_ps(out_row + d, _mm256_mul_ps(silu, up));
        }
#endif
        for (; d < dim; ++d) {
            const float gate = row[d];
            out_row[d] = (gate / (1.0f + expf(-gate))) * row[dim + d];
        }
    }
}

/**
 * SwiGLU backward pass (exact version using stdlib sigmoid)
 * @test test_swiglu.py::TestSwiGLUBackward::test_exact_vs_fast
 * @test test_swiglu.py::TestSwiGLUBackward::test_exact_single
 *
 * Uses standard library expf for numerical accuracy reference.
 *
 * After changes: make test
 */
void swiglu_backward_exact(const float *input,
                           const float *d_output,
                           float *d_input,
                           int tokens,
                           int dim)
{
    int T = tokens;
    int D = dim;

    for (int t = 0; t < T; ++t) {
        const float *row = input + (size_t)t * (2 * D);
        const float *dy_row = d_output + (size_t)t * D;
        float *dx_row = d_input + (size_t)t * (2 * D);

        for (int d = 0; d < D; ++d) {
            float a = row[d];       // gate
            float b = row[D + d];   // value
            float dy = dy_row[d];

            float s = sigmoid_scalar_parity(a); // sigmoid(a)
            float silu = a * s;                 // silu(a)
            float silu_prime = s * (1.0f + a * (1.0f - s)); // silu'(a), PyTorch form

            float dA = dy * b * silu_prime;
            float dB = dy * silu;

            dx_row[d] = dA;
            dx_row[D + d] = dB;
        }
    }
}
