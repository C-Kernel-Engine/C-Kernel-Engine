/**
 * @file rmsnorm_kernels_bf16.c
 * @brief RMSNorm kernels for BF16 tensors
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

#include "bf16_utils.h"
#include "ckernel_engine.h"

#include <math.h>
#include <stdint.h>

/* RMSNorm forward for BF16 inputs/outputs; gamma stays in float for precision */
void rmsnorm_forward_bf16(const uint16_t *input,
                          const float *gamma,
                          uint16_t *output,
                          float *rstd_cache,
                          int tokens,
                          int d_model,
                          int aligned_embed_dim,
                          float eps)
{
    int T = tokens;
    int D = d_model;
    int aligned = aligned_embed_dim;

    for (int t = 0; t < T; ++t) {
        const uint16_t *x_bf16 = input + (size_t)t * aligned;
        float *rstd_ptr = rstd_cache ? (rstd_cache + t) : NULL;
        uint16_t *out_bf16 = output + (size_t)t * aligned;

#if defined(__AVX512F__)
        // AVX-512: Process 16 floats at a time
        __m512 sum_sq_vec = _mm512_setzero_ps();
        int d = 0;

        // Vectorized sum of squares
        for (; d + 16 <= D; d += 16) {
            __m512 xv = bf16_loadu_cvt_fp32(&x_bf16[d]);
            sum_sq_vec = _mm512_fmadd_ps(xv, xv, sum_sq_vec);
        }
        float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            sum_sq += x * x;
        }

        float mean_sq = sum_sq / (float)D;
        float rstd = 1.0f / sqrtf(mean_sq + eps);
        if (rstd_ptr) {
            *rstd_ptr = rstd;
        }

        // Apply normalization and scale (vectorized)
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 xv = bf16_loadu_cvt_fp32(&x_bf16[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);
            __m512 yv = _mm512_mul_ps(x_hat, gv);
            fp32_cvt_storeu_bf16(&out_bf16[d], yv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float y = x * rstd * gamma[d];
            out_bf16[d] = float_to_bf16(y);
        }

#else
        // Scalar fallback
        double sum_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            sum_sq += (double)x * (double)x;
        }
        double mean_sq = sum_sq / (double)D;
        double r = sqrt(mean_sq + (double)eps);
        float rstd = (float)(1.0 / r);
        if (rstd_ptr) {
            *rstd_ptr = rstd;
        }

        for (int d = 0; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float x_hat = x * rstd;
            float y = x_hat * gamma[d];
            out_bf16[d] = float_to_bf16(y);
        }
#endif

        // Zero padding
        for (int d = D; d < aligned; ++d) {
            out_bf16[d] = 0;
        }
    }
}

// RMSNorm backward for BF16 inputs/outputs; gradients accumulate in float.
void rmsnorm_backward_bf16(const uint16_t *d_output,
                           const uint16_t *input,
                           const float *gamma,
                           const float *rstd_cache,
                           uint16_t *d_input,
                           float *d_gamma,
                           int tokens,
                           int d_model,
                           int aligned_embed_dim)
{
    int T = tokens;
    int D = d_model;
    int aligned = aligned_embed_dim;

    if (!d_output || !input || !gamma || !rstd_cache || !d_input || !d_gamma) {
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
#else
    for (int d = 0; d < D; ++d) {
        d_gamma[d] = 0.0f;
    }
#endif

    for (int t = 0; t < T; ++t) {
        const uint16_t *x_bf16 = input + (size_t)t * aligned;
        const uint16_t *dY_bf16 = d_output + (size_t)t * aligned;
        uint16_t *dX_bf16 = d_input + (size_t)t * aligned;
        float rstd = rstd_cache[t];

#if defined(__AVX512F__)
        // Compute m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        __m512 sum_vec = _mm512_setzero_ps();
        int d = 0;

        for (; d + 16 <= D; d += 16) {
            __m512 xv = bf16_loadu_cvt_fp32(&x_bf16[d]);
            __m512 dyv = bf16_loadu_cvt_fp32(&dY_bf16[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);
            // sum += dY * gamma * x_hat
            __m512 prod = _mm512_mul_ps(dyv, gv);
            sum_vec = _mm512_fmadd_ps(prod, x_hat, sum_vec);
        }
        float sum_dY_g_xhat = _mm512_reduce_add_ps(sum_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float x_hat = x * rstd;
            float dy = bf16_to_float(dY_bf16[d]);
            sum_dY_g_xhat += dy * gamma[d] * x_hat;
        }
        float m = sum_dY_g_xhat / (float)D;

        // Compute dX and accumulate dGamma (vectorized)
        __m512 m_vec = _mm512_set1_ps(m);
        d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 xv = bf16_loadu_cvt_fp32(&x_bf16[d]);
            __m512 dyv = bf16_loadu_cvt_fp32(&dY_bf16[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 dgv = _mm512_loadu_ps(&d_gamma[d]);

            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);

            // dX = rstd * (dY * gamma - x_hat * m)
            __m512 dy_g = _mm512_mul_ps(dyv, gv);
            __m512 xhat_m = _mm512_mul_ps(x_hat, m_vec);
            __m512 diff = _mm512_sub_ps(dy_g, xhat_m);
            __m512 dxv = _mm512_mul_ps(rstd_vec, diff);
            fp32_cvt_storeu_bf16(&dX_bf16[d], dxv);

            // d_gamma += dY * x_hat
            dgv = _mm512_fmadd_ps(dyv, x_hat, dgv);
            _mm512_storeu_ps(&d_gamma[d], dgv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float x_hat = x * rstd;
            float dy = bf16_to_float(dY_bf16[d]);
            float dx = rstd * (dy * gamma[d] - x_hat * m);
            dX_bf16[d] = float_to_bf16(dx);
            d_gamma[d] += dy * x_hat;
        }

#else
        // Scalar fallback
        double sum_dY_g_xhat = 0.0;
        for (int d = 0; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float x_hat = x * rstd;
            float dy = bf16_to_float(dY_bf16[d]);
            sum_dY_g_xhat += (double)dy * (double)gamma[d] * (double)x_hat;
        }
        float m = (float)(sum_dY_g_xhat / (double)D);

        for (int d = 0; d < D; ++d) {
            float x = bf16_to_float(x_bf16[d]);
            float x_hat = x * rstd;
            float dy = bf16_to_float(dY_bf16[d]);
            float dx = rstd * (dy * gamma[d] - x_hat * m);
            dX_bf16[d] = float_to_bf16(dx);
            d_gamma[d] += dy * x_hat;
        }
#endif

        // Zero padding gradients
        for (int d = D; d < aligned; ++d) {
            dX_bf16[d] = 0;
        }
    }
}
