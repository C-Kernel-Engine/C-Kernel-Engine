/**
 * @file rmsnorm_kernels.c
 * @brief RMSNorm forward/backward kernels with SIMD (SSE/AVX/AVX512)
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
 * RMSNorm: y[i] = gamma[i] * x[i] / sqrt(mean(x^2) + eps)
 */

#include "ckernel_engine.h"
#include <math.h>
#include <stddef.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* AVX1 horizontal sum helper (no _mm256_reduce_add_ps in AVX1) */
#if defined(__AVX__) && !defined(__AVX512F__)
static inline float hsum256_ps_rmsnorm(__m256 v) {
    // Sum upper and lower 128-bit lanes
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    // Horizontal add within 128-bit lane
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif
static void rmsnorm_forward_strict_scalar(const float *input,
                                          const float *gamma,
                                          float *output,
                                          float *rstd_cache,
                                          int tokens,
                                          int d_model,
                                          int aligned_embed_dim,
                                          float eps)
{
    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)aligned_embed_dim;
        float *y = output + (size_t)t * (size_t)aligned_embed_dim;

        double sum_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            const double v = (double)x[d];
            sum_sq += v * v;
        }
        const double mean_sq = sum_sq / (double)d_model;
        const double rstd64 = 1.0 / sqrt(mean_sq + (double)eps);
        const float rstd = (float)rstd64;
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }

        for (int d = 0; d < d_model; ++d) {
            y[d] = (float)(((double)x[d]) * ((double)gamma[d]) * rstd64);
        }
        for (int d = d_model; d < aligned_embed_dim; ++d) {
            y[d] = 0.0f;
        }
    }
}

static void rmsnorm_backward_strict_scalar(const float *d_output,
                                           const float *input,
                                           const float *gamma,
                                           const float *rstd_cache,
                                           float *d_input,
                                           float *d_gamma,
                                           int tokens,
                                           int d_model,
                                           int aligned_embed_dim)
{
    for (int d = 0; d < d_model; ++d) {
        d_gamma[d] = 0.0f;
    }

    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)aligned_embed_dim;
        const float *dY = d_output + (size_t)t * (size_t)aligned_embed_dim;
        float *dX = d_input + (size_t)t * (size_t)aligned_embed_dim;
        const double rstd = (double)rstd_cache[t];

        double sum_dY_g_xhat = 0.0;
        for (int d = 0; d < d_model; ++d) {
            const double x_hat = (double)x[d] * rstd;
            sum_dY_g_xhat += (double)dY[d] * (double)gamma[d] * x_hat;
        }
        const double m = sum_dY_g_xhat / (double)d_model;

        for (int d = 0; d < d_model; ++d) {
            const double x_hat = (double)x[d] * rstd;
            const double dy = (double)dY[d];
            dX[d] = (float)(rstd * (dy * (double)gamma[d] - x_hat * m));
        }
        for (int d = d_model; d < aligned_embed_dim; ++d) {
            dX[d] = 0.0f;
        }
    }

    for (int d = 0; d < d_model; ++d) {
        double acc = 0.0;
        for (int t = 0; t < tokens; ++t) {
            const float *x = input + (size_t)t * (size_t)aligned_embed_dim;
            const float *dY = d_output + (size_t)t * (size_t)aligned_embed_dim;
            const double rstd = (double)rstd_cache[t];
            const double x_hat = (double)x[d] * rstd;
            acc += (double)dY[d] * x_hat;
        }
        d_gamma[d] = (float)acc;
    }
}


/**
 * RMSNorm forward pass
 * @test test_rmsnorm.py::TestRMSNormForward::test_fp32_tokens
 * @test test_rmsnorm.py::TestRMSNormForward::test_fp32_single
 * @test test_rmsnorm.py::TestRMSNormForward::test_perf_rolled
 * @test test_layernorm.py::TestLayerNormForward::test_rmsnorm_compat
 * @test test_parity.py::test_rmsnorm_parity
 *
 * RMSNorm: y[i] = gamma[i] * x[i] / sqrt(mean(x^2) + eps)
 *
 * After changes: make test && make llamacpp-parity-full
 */
void rmsnorm_forward(const float *input,
                     const float *gamma,
                     float *output,
                     float *rstd_cache,
                     int tokens,
                     int d_model,
                     int aligned_embed_dim,
                     float eps)
{
    int T = tokens;
    int D = d_model;
    int aligned = aligned_embed_dim;

    if (ck_strict_parity_enabled()) {
        rmsnorm_forward_strict_scalar(input, gamma, output, rstd_cache, T, D, aligned, eps);
        return;
    }

    for (int t = 0; t < T; ++t) {
        const float *x = input + (size_t)t * aligned;
        float *y = output + (size_t)t * aligned;

#if defined(__AVX512F__)
        // AVX-512: Process 16 floats at a time
        __m512 sum_sq_vec = _mm512_setzero_ps();
        int d = 0;

        // Vectorized sum of squares
        for (; d + 16 <= D; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            sum_sq_vec = _mm512_fmadd_ps(xv, xv, sum_sq_vec);
        }
        float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            sum_sq += x[d] * x[d];
        }

        float mean_sq = sum_sq / (float)D;
        float rstd = 1.0f / sqrtf(mean_sq + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }

        // Apply normalization and scale (vectorized)
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);
            __m512 yv = _mm512_mul_ps(x_hat, gv);
            _mm512_storeu_ps(&y[d], yv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            y[d] = x[d] * rstd * gamma[d];
        }

#elif defined(__AVX__)
        // AVX: Process 8 floats at a time
        __m256 sum_sq_vec = _mm256_setzero_ps();
        int d = 0;

        // Vectorized sum of squares (no FMA in AVX1, use mul + add)
        for (; d + 8 <= D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 xv_sq = _mm256_mul_ps(xv, xv);
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, xv_sq);
        }
        float sum_sq = hsum256_ps_rmsnorm(sum_sq_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            sum_sq += x[d] * x[d];
        }

        float mean_sq = sum_sq / (float)D;
        float rstd = 1.0f / sqrtf(mean_sq + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }

        // Apply normalization and scale (vectorized)
        __m256 rstd_vec = _mm256_set1_ps(rstd);
        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 gv = _mm256_loadu_ps(&gamma[d]);
            __m256 x_hat = _mm256_mul_ps(xv, rstd_vec);
            __m256 yv = _mm256_mul_ps(x_hat, gv);
            _mm256_storeu_ps(&y[d], yv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            y[d] = x[d] * rstd * gamma[d];
        }

#else
        // Scalar fallback
        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            double v = (double)x[d];
            sum_sq += v * v;
        }
        double mean_sq = sum_sq / (double)D;
        double r = sqrt(mean_sq + (double)eps);
        float rstd = (float)(1.0 / r);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }

        // Apply normalization and scale
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            y[d] = x_hat * gamma[d];
        }
#endif

        // Zero padding (if any)
        for (int d = D; d < aligned; ++d) {
            y[d] = 0.0f;
        }
    }
}

/**
 * RMSNorm backward pass
 * @test test_rmsnorm.py::TestRMSNormBackward::test_backward_tokens
 * @test test_rmsnorm.py::TestRMSNormBackward::test_backward_single
 * @test test_parity.py::test_rmsnorm_backward_parity
 *
 * Computes dX and dGamma given dY, X, gamma, and cached rstd.
 * dX_i = rstd * (dY_i * gamma_i - x_hat_i * m)
 * dGamma_i = sum_t (dY_i * x_hat_i)
 *
 * After changes: make test && make llamacpp-parity-full
 */
void rmsnorm_backward(const float *d_output,
                      const float *input,
                      const float *gamma,
                      const float *rstd_cache,
                      float *d_input,
                      float *d_gamma,
                      int tokens,
                      int d_model,
                      int aligned_embed_dim)
{
    int T = tokens;
    int D = d_model;
    int aligned = aligned_embed_dim;

    if (ck_strict_parity_enabled()) {
        rmsnorm_backward_strict_scalar(d_output, input, gamma, rstd_cache, d_input, d_gamma, T, D, aligned);
        return;
    }

    // Zero parameter gradients
#if defined(__AVX512F__)
    {
        int d = 0;
        for (; d + 16 <= D; d += 16) {
            _mm512_storeu_ps(&d_gamma[d], _mm512_setzero_ps());
        }
        for (; d < D; ++d) {
            d_gamma[d] = 0.0f;
        }
    }
#elif defined(__AVX__)
    {
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            _mm256_storeu_ps(&d_gamma[d], _mm256_setzero_ps());
        }
        for (; d < D; ++d) {
            d_gamma[d] = 0.0f;
        }
    }
#else
    for (int d = 0; d < D; ++d) {
        d_gamma[d] = 0.0f;
    }
#endif

    for (int t = 0; t < T; ++t) {
        const float *x = input + (size_t)t * aligned;
        const float *dY = d_output + (size_t)t * aligned;
        float *dX = d_input + (size_t)t * aligned;

        float rstd = rstd_cache[t];

#if defined(__AVX512F__)
        // Compute m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        __m512 sum_vec = _mm512_setzero_ps();
        int d = 0;

        for (; d + 16 <= D; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            __m512 dyv = _mm512_loadu_ps(&dY[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);
            // sum += dY * gamma * x_hat
            __m512 prod = _mm512_mul_ps(dyv, gv);
            sum_vec = _mm512_fmadd_ps(prod, x_hat, sum_vec);
        }
        float sum_dY_g_xhat = _mm512_reduce_add_ps(sum_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            float x_hat = x[d] * rstd;
            sum_dY_g_xhat += dY[d] * gamma[d] * x_hat;
        }
        float m = sum_dY_g_xhat / (float)D;

        // Compute dX and accumulate dGamma (vectorized)
        __m512 m_vec = _mm512_set1_ps(m);
        d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            __m512 dyv = _mm512_loadu_ps(&dY[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 dgv = _mm512_loadu_ps(&d_gamma[d]);

            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);

            // dX = rstd * (dY * gamma - x_hat * m)
            __m512 dy_g = _mm512_mul_ps(dyv, gv);
            __m512 xhat_m = _mm512_mul_ps(x_hat, m_vec);
            __m512 diff = _mm512_sub_ps(dy_g, xhat_m);
            __m512 dxv = _mm512_mul_ps(rstd_vec, diff);
            _mm512_storeu_ps(&dX[d], dxv);

            // d_gamma += dY * x_hat
            dgv = _mm512_fmadd_ps(dyv, x_hat, dgv);
            _mm512_storeu_ps(&d_gamma[d], dgv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            float x_hat = x[d] * rstd;
            float dy = dY[d];
            dX[d] = rstd * (dy * gamma[d] - x_hat * m);
            d_gamma[d] += dy * x_hat;
        }

#elif defined(__AVX__)
        // Compute m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
        __m256 rstd_vec = _mm256_set1_ps(rstd);
        __m256 sum_vec = _mm256_setzero_ps();
        int d = 0;

        for (; d + 8 <= D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 dyv = _mm256_loadu_ps(&dY[d]);
            __m256 gv = _mm256_loadu_ps(&gamma[d]);
            __m256 x_hat = _mm256_mul_ps(xv, rstd_vec);
            // sum += dY * gamma * x_hat (no FMA, use mul + mul + add)
            __m256 prod = _mm256_mul_ps(dyv, gv);
            __m256 prod2 = _mm256_mul_ps(prod, x_hat);
            sum_vec = _mm256_add_ps(sum_vec, prod2);
        }
        float sum_dY_g_xhat = hsum256_ps_rmsnorm(sum_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            float x_hat = x[d] * rstd;
            sum_dY_g_xhat += dY[d] * gamma[d] * x_hat;
        }
        float m = sum_dY_g_xhat / (float)D;

        // Compute dX and accumulate dGamma (vectorized)
        __m256 m_vec = _mm256_set1_ps(m);
        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 dyv = _mm256_loadu_ps(&dY[d]);
            __m256 gv = _mm256_loadu_ps(&gamma[d]);
            __m256 dgv = _mm256_loadu_ps(&d_gamma[d]);

            __m256 x_hat = _mm256_mul_ps(xv, rstd_vec);

            // dX = rstd * (dY * gamma - x_hat * m)
            __m256 dy_g = _mm256_mul_ps(dyv, gv);
            __m256 xhat_m = _mm256_mul_ps(x_hat, m_vec);
            __m256 diff = _mm256_sub_ps(dy_g, xhat_m);
            __m256 dxv = _mm256_mul_ps(rstd_vec, diff);
            _mm256_storeu_ps(&dX[d], dxv);

            // d_gamma += dY * x_hat
            __m256 dy_xhat = _mm256_mul_ps(dyv, x_hat);
            dgv = _mm256_add_ps(dgv, dy_xhat);
            _mm256_storeu_ps(&d_gamma[d], dgv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            float x_hat = x[d] * rstd;
            float dy = dY[d];
            dX[d] = rstd * (dy * gamma[d] - x_hat * m);
            d_gamma[d] += dy * x_hat;
        }

#else
        // Scalar fallback
        // Compute m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
        double sum_dY_g_xhat = 0.0;
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            sum_dY_g_xhat += (double)dY[d] * (double)gamma[d] * (double)x_hat;
        }
        float m = (float)(sum_dY_g_xhat / (double)D);

        // Compute dX and accumulate dGamma
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            float dy = dY[d];
            dX[d] = rstd * (dy * gamma[d] - x_hat * m);
            d_gamma[d] += dy * x_hat;
        }
#endif

        // Zero padding gradients (if any)
        for (int d = D; d < aligned; ++d) {
            dX[d] = 0.0f;
        }
    }
}
