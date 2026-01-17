/**
 * @file softmax_kernels.c
 * @brief Softmax forward/backward kernels with SIMD (SSE/AVX/AVX512)
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
 * Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
 */

#include <math.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* Fast vectorized exp approximation (good for softmax, ~1e-4 relative error) */
// Based on Schraudolph's algorithm with improved coefficients
#if defined(__AVX512F__)
static inline __m512 exp512_approx(__m512 x) {
    // Clamp to avoid overflow/underflow
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));

    // exp(x) = 2^(x * log2(e)) = 2^(x * 1.4426950408889634)
    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    const __m512 c1 = _mm512_set1_ps(0.693359375f);
    const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);

    __m512 t = _mm512_mul_ps(x, log2e);
    __m512 ti = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // Reconstruct remainder: rx = x - ti * ln(2)
    __m512 rx = _mm512_sub_ps(x, _mm512_mul_ps(ti, c1));
    rx = _mm512_sub_ps(rx, _mm512_mul_ps(ti, c2));

    // Polynomial approximation for 2^tf on [-0.5, 0.5]
    const __m512 p0 = _mm512_set1_ps(1.0f);
    const __m512 p1 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 p2 = _mm512_set1_ps(0.24022650695910071f);
    const __m512 p3 = _mm512_set1_ps(0.05550410866482157f);
    const __m512 p4 = _mm512_set1_ps(0.009618129107628477f);

    __m512 poly = _mm512_fmadd_ps(p4, rx, p3);
    poly = _mm512_fmadd_ps(poly, rx, p2);
    poly = _mm512_fmadd_ps(poly, rx, p1);
    poly = _mm512_fmadd_ps(poly, rx, p0);

    // Scale by 2^ti using integer manipulation
    __m512i ti_int = _mm512_cvtps_epi32(ti);
    ti_int = _mm512_add_epi32(ti_int, _mm512_set1_epi32(127));
    ti_int = _mm512_slli_epi32(ti_int, 23);
    __m512 scale = _mm512_castsi512_ps(ti_int);

    return _mm512_mul_ps(poly, scale);
}
#endif

#if defined(__AVX2__)
// AVX2 version with integer operations
static inline __m256 exp256_approx(__m256 x) {
    // Clamp to avoid overflow/underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);

    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256 rx = _mm256_sub_ps(x, _mm256_mul_ps(ti, c1));
    rx = _mm256_sub_ps(rx, _mm256_mul_ps(ti, c2));

    // Polynomial (use FMA if available)
    const __m256 p0 = _mm256_set1_ps(1.0f);
    const __m256 p1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 p2 = _mm256_set1_ps(0.24022650695910071f);
    const __m256 p3 = _mm256_set1_ps(0.05550410866482157f);
    const __m256 p4 = _mm256_set1_ps(0.009618129107628477f);

    __m256 poly = _mm256_fmadd_ps(p4, rx, p3);
    poly = _mm256_fmadd_ps(poly, rx, p2);
    poly = _mm256_fmadd_ps(poly, rx, p1);
    poly = _mm256_fmadd_ps(poly, rx, p0);

    // Scale by 2^ti using AVX2 integer ops
    __m256i ti_int = _mm256_cvtps_epi32(ti);
    ti_int = _mm256_add_epi32(ti_int, _mm256_set1_epi32(127));
    ti_int = _mm256_slli_epi32(ti_int, 23);
    __m256 scale = _mm256_castsi256_ps(ti_int);

    return _mm256_mul_ps(poly, scale);
}
#endif

// AVX/AVX2 horizontal max helper (works for both, uses 256-bit ops only)
#if defined(__AVX__) || defined(__AVX2__)
static inline float hmax256_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 max128 = _mm_max_ps(lo, hi);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max128);
}

// AVX/AVX2 horizontal sum helper
static inline float hsum256_ps_softmax(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif

// Causal softmax on head-major attention scores, copied and generalized
// from C-Transformer's apply_causal_softmax_head_major.
//
// scores layout: [head][query_token][key_token] with aligned_context_window stride:
//   index = h * aligned_context_window * aligned_context_window
//         + i * aligned_context_window
//         + j
/**
 * Causal softmax (in-place, row-wise)
 * @test test_softmax.py::TestSoftmaxForward::test_causal_softmax
 * @test test_softmax.py::TestSoftmaxForward::test_causal_vs_softmax
 * @test test_attention.py::TestAttentionForward::test_softmax_correctness
 *
 * Applies causal mask (j > i => 0) and softmax to scores matrix.
 * In-place on [num_heads, T, T] scores matrix.
 *
 * After changes: make test && make llamacpp-parity-full
 */
void causal_softmax_head_major(float *scores,
                               int num_heads,
                               int num_tokens,
                               int aligned_context_window)
{
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            int base = h * aligned_context_window * aligned_context_window
                     + i * aligned_context_window;
            float *row = &scores[base];
            int len = i + 1;  // Number of valid elements (0..i inclusive)

#if defined(__AVX512F__)
            // Find max (vectorized)
            __m512 max_vec = _mm512_set1_ps(-INFINITY);
            int j = 0;
            for (; j + 16 <= len; j += 16) {
                __m512 v = _mm512_loadu_ps(&row[j]);
                max_vec = _mm512_max_ps(max_vec, v);
            }
            float max_val = _mm512_reduce_max_ps(max_vec);
            for (; j < len; ++j) {
                if (row[j] > max_val) max_val = row[j];
            }

            // Compute exp and sum (vectorized)
            __m512 max_broadcast = _mm512_set1_ps(max_val);
            __m512 sum_vec = _mm512_setzero_ps();
            j = 0;
            for (; j + 16 <= len; j += 16) {
                __m512 v = _mm512_loadu_ps(&row[j]);
                __m512 e = exp512_approx(_mm512_sub_ps(v, max_broadcast));
                _mm512_storeu_ps(&row[j], e);
                sum_vec = _mm512_add_ps(sum_vec, e);
            }
            float sum = _mm512_reduce_add_ps(sum_vec);
            for (; j < len; ++j) {
                float e = expf(row[j] - max_val);
                row[j] = e;
                sum += e;
            }

            // Normalize (vectorized)
            float inv_sum = 1.0f / sum;
            __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
            j = 0;
            for (; j + 16 <= len; j += 16) {
                __m512 v = _mm512_loadu_ps(&row[j]);
                _mm512_storeu_ps(&row[j], _mm512_mul_ps(v, inv_sum_vec));
            }
            for (; j < len; ++j) {
                row[j] *= inv_sum;
            }

            // Zero out future tokens (vectorized)
            __m512 zero = _mm512_setzero_ps();
            for (; j + 16 <= num_tokens; j += 16) {
                _mm512_storeu_ps(&row[j], zero);
            }
            for (; j < num_tokens; ++j) {
                row[j] = 0.0f;
            }

#elif defined(__AVX2__)
            // AVX2: Find max (vectorized)
            __m256 max_vec = _mm256_set1_ps(-INFINITY);
            int j = 0;
            for (; j + 8 <= len; j += 8) {
                __m256 v = _mm256_loadu_ps(&row[j]);
                max_vec = _mm256_max_ps(max_vec, v);
            }
            float max_val = hmax256_ps(max_vec);
            for (; j < len; ++j) {
                if (row[j] > max_val) max_val = row[j];
            }

            // Compute exp and sum (vectorized with fast exp)
            __m256 max_broadcast = _mm256_set1_ps(max_val);
            __m256 sum_vec = _mm256_setzero_ps();
            j = 0;
            for (; j + 8 <= len; j += 8) {
                __m256 v = _mm256_loadu_ps(&row[j]);
                __m256 e = exp256_approx(_mm256_sub_ps(v, max_broadcast));
                _mm256_storeu_ps(&row[j], e);
                sum_vec = _mm256_add_ps(sum_vec, e);
            }
            float sum = hsum256_ps_softmax(sum_vec);
            for (; j < len; ++j) {
                float e = expf(row[j] - max_val);
                row[j] = e;
                sum += e;
            }

            // Normalize (vectorized)
            float inv_sum = 1.0f / sum;
            __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
            j = 0;
            for (; j + 8 <= len; j += 8) {
                __m256 v = _mm256_loadu_ps(&row[j]);
                _mm256_storeu_ps(&row[j], _mm256_mul_ps(v, inv_sum_vec));
            }
            for (; j < len; ++j) {
                row[j] *= inv_sum;
            }

            // Zero out future tokens (vectorized)
            __m256 zero = _mm256_setzero_ps();
            for (; j + 8 <= num_tokens; j += 8) {
                _mm256_storeu_ps(&row[j], zero);
            }
            for (; j < num_tokens; ++j) {
                row[j] = 0.0f;
            }

#elif defined(__AVX__)
            // AVX1: vectorized max/sum/normalize, scalar exp
            __m256 max_vec = _mm256_set1_ps(-INFINITY);
            int j = 0;
            for (; j + 8 <= len; j += 8) {
                __m256 v = _mm256_loadu_ps(&row[j]);
                max_vec = _mm256_max_ps(max_vec, v);
            }
            float max_val = hmax256_ps(max_vec);
            for (; j < len; ++j) {
                if (row[j] > max_val) max_val = row[j];
            }

            // Compute exp and sum (scalar exp, no fast approx for AVX1)
            float sum = 0.0f;
            for (j = 0; j < len; ++j) {
                float e = expf(row[j] - max_val);
                row[j] = e;
                sum += e;
            }

            // Normalize (vectorized)
            float inv_sum = 1.0f / sum;
            __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
            j = 0;
            for (; j + 8 <= len; j += 8) {
                __m256 v = _mm256_loadu_ps(&row[j]);
                _mm256_storeu_ps(&row[j], _mm256_mul_ps(v, inv_sum_vec));
            }
            for (; j < len; ++j) {
                row[j] *= inv_sum;
            }

            // Zero out future tokens (vectorized)
            __m256 zero = _mm256_setzero_ps();
            for (; j + 8 <= num_tokens; j += 8) {
                _mm256_storeu_ps(&row[j], zero);
            }
            for (; j < num_tokens; ++j) {
                row[j] = 0.0f;
            }

#else
            // Scalar fallback
            float max_val = row[0];
            for (int j = 1; j < len; ++j) {
                if (row[j] > max_val) max_val = row[j];
            }

            float sum = 0.0f;
            for (int j = 0; j < len; ++j) {
                float e = expf(row[j] - max_val);
                row[j] = e;
                sum += e;
            }

            float inv_sum = 1.0f / sum;
            for (int j = 0; j < len; ++j) {
                row[j] *= inv_sum;
            }

            for (int j = len; j < num_tokens; ++j) {
                row[j] = 0.0f;
            }
#endif
        }
    }
}

// Scalar-only exact causal softmax using standard library expf.
// This is slower than causal_softmax_head_major but provides maximum accuracy.
// Used by BF16 attention wrapper where approximation error accumulates.
/**
 * Causal softmax (exact version using stdlib expf)
 * @test test_softmax.py::TestSoftmaxForward::test_causal_softmax_exact
 * @test test_softmax.py::TestSoftmaxForward::test_exact_vs_fast
 *
 * Exact causal softmax using standard library expf for numerical accuracy reference.
 *
 * After changes: make test
 */
void causal_softmax_head_major_exact(float *scores,
                                      int num_heads,
                                      int num_tokens,
                                      int aligned_context_window)
{
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < num_tokens; ++i) {
            int base = h * aligned_context_window * aligned_context_window
                     + i * aligned_context_window;
            float *row = &scores[base];
            int len = i + 1;

            // Find max
            float max_val = -INFINITY;
            for (int j = 0; j < len; ++j) {
                if (row[j] > max_val) max_val = row[j];
            }

            // Compute exp and sum using standard library expf
            float sum = 0.0f;
            for (int j = 0; j < len; ++j) {
                float e = expf(row[j] - max_val);
                row[j] = e;
                sum += e;
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < len; ++j) {
                row[j] *= inv_sum;
            }

            // Zero out future tokens
            for (int j = len; j < num_tokens; ++j) {
                row[j] = 0.0f;
            }
        }
    }
}

// Backward pass for causal softmax on head-major scores, adapted from
// C-Transformer's backward_causal_softmax. Operates in-place on d_scores,
// using the cached forward softmax output `weights`.
void backward_causal_softmax_head_major(float *d_scores,
                                        const float *weights,
                                        int num_heads,
                                        int num_tokens,
                                        int aligned_context_window)
{
    int H = num_heads;
    int T = num_tokens;

    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < T; ++i) {
            int base = h * aligned_context_window * aligned_context_window
                     + i * aligned_context_window;
            float *drow = &d_scores[base];
            const float *wrow = &weights[base];
            int len = i + 1;

#if defined(__AVX512F__)
            // Compute dot product (vectorized)
            __m512 dot_vec = _mm512_setzero_ps();
            int j = 0;
            for (; j + 16 <= len; j += 16) {
                __m512 w = _mm512_loadu_ps(&wrow[j]);
                __m512 dw = _mm512_loadu_ps(&drow[j]);
                dot_vec = _mm512_fmadd_ps(w, dw, dot_vec);
            }
            float dot_product = _mm512_reduce_add_ps(dot_vec);
            for (; j < len; ++j) {
                dot_product += wrow[j] * drow[j];
            }

            // Compute gradient: d_scores = w * (dw - dot_product)
            __m512 dot_broadcast = _mm512_set1_ps(dot_product);
            j = 0;
            for (; j + 16 <= len; j += 16) {
                __m512 w = _mm512_loadu_ps(&wrow[j]);
                __m512 dw = _mm512_loadu_ps(&drow[j]);
                __m512 diff = _mm512_sub_ps(dw, dot_broadcast);
                __m512 result = _mm512_mul_ps(w, diff);
                _mm512_storeu_ps(&drow[j], result);
            }
            for (; j < len; ++j) {
                drow[j] = wrow[j] * (drow[j] - dot_product);
            }

            // Zero out future tokens
            __m512 zero = _mm512_setzero_ps();
            for (; j + 16 <= T; j += 16) {
                _mm512_storeu_ps(&drow[j], zero);
            }
            for (; j < T; ++j) {
                drow[j] = 0.0f;
            }

#elif defined(__AVX__)
            // Compute dot product (vectorized)
            __m256 dot_vec = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= len; j += 8) {
                __m256 w = _mm256_loadu_ps(&wrow[j]);
                __m256 dw = _mm256_loadu_ps(&drow[j]);
                // No FMA in AVX1: use mul + add
                __m256 prod = _mm256_mul_ps(w, dw);
                dot_vec = _mm256_add_ps(dot_vec, prod);
            }
            float dot_product = hsum256_ps_softmax(dot_vec);
            for (; j < len; ++j) {
                dot_product += wrow[j] * drow[j];
            }

            // Compute gradient: d_scores = w * (dw - dot_product)
            __m256 dot_broadcast = _mm256_set1_ps(dot_product);
            j = 0;
            for (; j + 8 <= len; j += 8) {
                __m256 w = _mm256_loadu_ps(&wrow[j]);
                __m256 dw = _mm256_loadu_ps(&drow[j]);
                __m256 diff = _mm256_sub_ps(dw, dot_broadcast);
                __m256 result = _mm256_mul_ps(w, diff);
                _mm256_storeu_ps(&drow[j], result);
            }
            for (; j < len; ++j) {
                drow[j] = wrow[j] * (drow[j] - dot_product);
            }

            // Zero out future tokens
            __m256 zero = _mm256_setzero_ps();
            for (; j + 8 <= T; j += 8) {
                _mm256_storeu_ps(&drow[j], zero);
            }
            for (; j < T; ++j) {
                drow[j] = 0.0f;
            }

#else
            // Scalar fallback
            float dot_product = 0.0f;
            for (int j = 0; j < len; ++j) {
                dot_product += wrow[j] * drow[j];
            }

            for (int j = 0; j < len; ++j) {
                drow[j] = wrow[j] * (drow[j] - dot_product);
            }

            for (int j = len; j < T; ++j) {
                drow[j] = 0.0f;
            }
#endif
        }
    }
}

