/**
 * @file optimizer_kernels_bf16.c
 * @brief BF16 optimizer kernels for training
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
 * Note: Optimizer state (m, v) is always kept in fp32 for numerical stability.
 * Only weights and gradients are in bf16.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bf16_utils.h"

/* Forward declarations of fp32 kernels */
extern void adamw_update_f32(
    const float *grad, float *weight, float *m, float *v, size_t numel,
    float lr, float beta1, float beta2, float eps, float weight_decay, int step);

extern void sgd_momentum_update_f32(
    const float *grad, float *weight, float *velocity, size_t numel,
    float lr, float momentum, float weight_decay);

extern void gradient_accumulate_f32(float *dst, const float *src, size_t numel);
extern void gradient_scale_f32(float *grad, size_t numel, float scale);


/**
 * @brief AdamW optimizer update (bf16 weights/gradients, fp32 optimizer state)
 *
 * Weights and gradients are in bf16 for memory efficiency.
 * Momentum (m) and variance (v) are in fp32 for numerical stability.
 *
 * @param grad       Gradient tensor (bf16) [numel]
 * @param weight     Weight tensor to update (bf16, in-place) [numel]
 * @param m          First moment buffer (fp32, in-place) [numel]
 * @param v          Second moment buffer (fp32, in-place) [numel]
 * @param numel      Number of elements
 * @param lr         Learning rate
 * @param beta1      First moment decay (typically 0.9)
 * @param beta2      Second moment decay (typically 0.999)
 * @param eps        Numerical stability constant (typically 1e-8)
 * @param weight_decay Weight decay coefficient
 * @param step       Current step number (1-indexed)
 */
void adamw_update_bf16(
    const uint16_t *grad,
    uint16_t *weight,
    float *m,
    float *v,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step)
{
    if (!grad || !weight || !m || !v || numel == 0) {
        return;
    }

    // Bias correction terms
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

#if defined(__AVX512F__)
    // Vectorized path: process 16 elements at a time
    __m512 v_beta1 = _mm512_set1_ps(beta1);
    __m512 v_beta2 = _mm512_set1_ps(beta2);
    __m512 v_one_minus_beta1 = _mm512_set1_ps(one_minus_beta1);
    __m512 v_one_minus_beta2 = _mm512_set1_ps(one_minus_beta2);
    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_eps = _mm512_set1_ps(eps);
    __m512 v_weight_decay = _mm512_set1_ps(weight_decay);
    __m512 v_bc1_inv = _mm512_set1_ps(1.0f / bias_correction1);
    __m512 v_bc2_inv = _mm512_set1_ps(1.0f / bias_correction2);

    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        // Load bf16 gradient and weight, convert to fp32
        __m512 g = bf16_loadu_cvt_fp32(&grad[i]);
        __m512 w = bf16_loadu_cvt_fp32(&weight[i]);

        // Load fp32 optimizer state
        __m512 m_val = _mm512_loadu_ps(&m[i]);
        __m512 v_val = _mm512_loadu_ps(&v[i]);

        // Update m: m = beta1 * m + (1 - beta1) * g
        m_val = _mm512_fmadd_ps(v_beta1, m_val, _mm512_mul_ps(v_one_minus_beta1, g));

        // Update v: v = beta2 * v + (1 - beta2) * g^2
        __m512 g_sq = _mm512_mul_ps(g, g);
        v_val = _mm512_fmadd_ps(v_beta2, v_val, _mm512_mul_ps(v_one_minus_beta2, g_sq));

        // Bias-corrected estimates
        __m512 m_hat = _mm512_mul_ps(m_val, v_bc1_inv);
        __m512 v_hat = _mm512_mul_ps(v_val, v_bc2_inv);

        // Update weight: w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
        __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), v_eps);
        __m512 update = _mm512_div_ps(m_hat, denom);
        update = _mm512_fmadd_ps(v_weight_decay, w, update);
        w = _mm512_fnmadd_ps(v_lr, update, w);

        // Store updated weight as bf16
        fp32_cvt_storeu_bf16(&weight[i], w);

        // Store updated optimizer state (stays fp32)
        _mm512_storeu_ps(&m[i], m_val);
        _mm512_storeu_ps(&v[i], v_val);
    }

    // Scalar tail
    for (; i < numel; ++i) {
        float g = bf16_to_float(grad[i]);
        float w = bf16_to_float(weight[i]);

        m[i] = beta1 * m[i] + one_minus_beta1 * g;
        v[i] = beta2 * v[i] + one_minus_beta2 * g * g;

        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        w = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
        weight[i] = float_to_bf16(w);
    }
#else
    // Scalar path
    for (size_t i = 0; i < numel; ++i) {
        float g = bf16_to_float(grad[i]);
        float w = bf16_to_float(weight[i]);

        m[i] = beta1 * m[i] + one_minus_beta1 * g;
        v[i] = beta2 * v[i] + one_minus_beta2 * g * g;

        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        w = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
        weight[i] = float_to_bf16(w);
    }
#endif
}


/**
 * @brief SGD with momentum (bf16 weights/gradients)
 */
void sgd_momentum_update_bf16(
    const uint16_t *grad,
    uint16_t *weight,
    float *velocity,
    size_t numel,
    float lr,
    float momentum,
    float weight_decay)
{
    if (!grad || !weight || !velocity || numel == 0) {
        return;
    }

#if defined(__AVX512F__)
    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_momentum = _mm512_set1_ps(momentum);
    __m512 v_weight_decay = _mm512_set1_ps(weight_decay);

    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = bf16_loadu_cvt_fp32(&grad[i]);
        __m512 w = bf16_loadu_cvt_fp32(&weight[i]);
        __m512 vel = _mm512_loadu_ps(&velocity[i]);

        vel = _mm512_fmadd_ps(v_momentum, vel, g);
        __m512 update = _mm512_fmadd_ps(v_weight_decay, w, vel);
        w = _mm512_fnmadd_ps(v_lr, update, w);

        fp32_cvt_storeu_bf16(&weight[i], w);
        _mm512_storeu_ps(&velocity[i], vel);
    }

    for (; i < numel; ++i) {
        float g = bf16_to_float(grad[i]);
        float w = bf16_to_float(weight[i]);
        velocity[i] = momentum * velocity[i] + g;
        w = w - lr * (velocity[i] + weight_decay * w);
        weight[i] = float_to_bf16(w);
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        float g = bf16_to_float(grad[i]);
        float w = bf16_to_float(weight[i]);
        velocity[i] = momentum * velocity[i] + g;
        w = w - lr * (velocity[i] + weight_decay * w);
        weight[i] = float_to_bf16(w);
    }
#endif
}


/**
 * @brief Zero out gradient buffer (bf16)
 */
void zero_gradients_bf16(uint16_t *grad, size_t numel)
{
    if (!grad || numel == 0) {
        return;
    }
    memset(grad, 0, numel * sizeof(uint16_t));
}


/**
 * @brief Accumulate gradients: dst += src (bf16)
 */
void gradient_accumulate_bf16(uint16_t *dst, const uint16_t *src, size_t numel)
{
    if (!dst || !src || numel == 0) {
        return;
    }

#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 d = bf16_loadu_cvt_fp32(&dst[i]);
        __m512 s = bf16_loadu_cvt_fp32(&src[i]);
        fp32_cvt_storeu_bf16(&dst[i], _mm512_add_ps(d, s));
    }
    for (; i < numel; ++i) {
        float d = bf16_to_float(dst[i]);
        float s = bf16_to_float(src[i]);
        dst[i] = float_to_bf16(d + s);
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        float d = bf16_to_float(dst[i]);
        float s = bf16_to_float(src[i]);
        dst[i] = float_to_bf16(d + s);
    }
#endif
}


/**
 * @brief Scale gradients: grad *= scale (bf16)
 */
void gradient_scale_bf16(uint16_t *grad, size_t numel, float scale)
{
    if (!grad || numel == 0) {
        return;
    }

#if defined(__AVX512F__)
    __m512 v_scale = _mm512_set1_ps(scale);
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = bf16_loadu_cvt_fp32(&grad[i]);
        fp32_cvt_storeu_bf16(&grad[i], _mm512_mul_ps(g, v_scale));
    }
    for (; i < numel; ++i) {
        float g = bf16_to_float(grad[i]);
        grad[i] = float_to_bf16(g * scale);
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        float g = bf16_to_float(grad[i]);
        grad[i] = float_to_bf16(g * scale);
    }
#endif
}


/**
 * @brief Clip gradient norm (bf16)
 *
 * @return The original L2 norm before clipping
 */
float gradient_clip_norm_bf16(uint16_t *grad, size_t numel, float max_norm)
{
    if (!grad || numel == 0 || max_norm <= 0.0f) {
        return 0.0f;
    }

    // Compute L2 norm in fp32 for accuracy
    double sum_sq = 0.0;
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = bf16_loadu_cvt_fp32(&grad[i]);
        acc = _mm512_fmadd_ps(g, g, acc);
    }
    sum_sq = _mm512_reduce_add_ps(acc);
    for (; i < numel; ++i) {
        float g = bf16_to_float(grad[i]);
        sum_sq += (double)g * (double)g;
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        float g = bf16_to_float(grad[i]);
        sum_sq += (double)g * (double)g;
    }
#endif

    float norm = sqrtf((float)sum_sq);

    if (norm > max_norm) {
        float scale = max_norm / norm;
        gradient_scale_bf16(grad, numel, scale);
    }

    return norm;
}
