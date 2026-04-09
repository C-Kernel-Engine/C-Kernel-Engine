/**
 * @file optimizer_kernels.c
 * @brief Optimizer kernels for training (AdamW, SGD)
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
 * AdamW Algorithm:
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 *   m_hat = m_t / (1 - beta1^t)
 *   v_hat = v_t / (1 - beta2^t)
 *   w_t = w_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w_{t-1})
 *
 * Note: AdamW applies weight decay directly to weights, not to gradients.
 * This is different from L2 regularization (Adam with L2 adds decay to gradient).
 *
 * Epsilon amplification at early steps: at step 1 with bc2=0.001, elements where
 * v≈0 (sparse/near-zero gradients) produce sqrt(v_hat)+eps ≈ eps=1e-8, amplifying
 * any fp32 rounding in the accumulated gradient by up to lr/eps = 1e6. This is
 * expected AdamW behavior, not a bug. In parity tests, gate on mean_param_diff
 * (not max_param_diff) for grad_accum > 1 to avoid false alarms from these outliers.
 *
 * Long-horizon fp32 risk: at lr=1e-3 with all-fp32 SIMD paths, known drift begins
 * around step ~800 due to accumulated rounding. Use ck_strict_parity_enabled() (fp64
 * path) for parity validation; for production training keep lr < 1e-3 or monitor.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "ck_threadpool.h"
#include "ckernel_engine.h"

/* Include SIMD headers based on available instruction sets */
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__)
#include <immintrin.h>
#endif

#define CK_OPT_PAR_MIN_NUMEL ((size_t)262144)
#define CK_OPT_PAR_MAX_THREADS 256

typedef struct {
    const float *grad;
    float *weight;
    float *m;
    float *v;
    size_t numel;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step;
} ck_adamw_parallel_args_t;

typedef struct {
    float *dst;
    const float *src;
    size_t numel;
} ck_accum_parallel_args_t;

typedef struct {
    float *grad;
    size_t numel;
    float scale;
} ck_scale_parallel_args_t;

typedef struct {
    const float *grad;
    size_t numel;
    double partial[CK_OPT_PAR_MAX_THREADS];
} ck_sum_sq_parallel_args_t;

typedef struct {
    const float *const *grads;
    const size_t *numels;
    int tensor_count;
    double partial[CK_OPT_PAR_MAX_THREADS];
} ck_sum_sq_multi_parallel_args_t;

typedef struct {
    float *const *grads;
    float *const *weights;
    float *const *m_states;
    float *const *v_states;
    const size_t *numels;
    int tensor_count;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float grad_scale;
    int step;
} ck_adamw_multi_parallel_args_t;

static void adamw_update_f32_impl(
    const float *grad,
    float *weight,
    float *m,
    float *v,
    size_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step);

static void gradient_accumulate_f32_impl(float *dst, const float *src, size_t numel);
static void gradient_scale_f32_impl(float *grad, size_t numel, float scale);
static double gradient_sum_sq_f32_impl(const float *grad, size_t numel);

static void ck_adamw_parallel_work(int ith, int nth, void *argp)
{
    ck_adamw_parallel_args_t *a = (ck_adamw_parallel_args_t *)argp;
    if (!a || !a->grad || !a->weight || !a->m || !a->v || a->numel == 0) {
        return;
    }
    size_t chunk = (a->numel + (size_t)nth - 1u) / (size_t)nth;
    size_t start = (size_t)ith * chunk;
    if (start >= a->numel) {
        return;
    }
    size_t end = start + chunk;
    if (end > a->numel) {
        end = a->numel;
    }
    adamw_update_f32_impl(
        a->grad + start,
        a->weight + start,
        a->m + start,
        a->v + start,
        end - start,
        a->lr,
        a->beta1,
        a->beta2,
        a->eps,
        a->weight_decay,
        a->step);
}

static void ck_accum_parallel_work(int ith, int nth, void *argp)
{
    ck_accum_parallel_args_t *a = (ck_accum_parallel_args_t *)argp;
    if (!a || !a->dst || !a->src || a->numel == 0) {
        return;
    }
    size_t chunk = (a->numel + (size_t)nth - 1u) / (size_t)nth;
    size_t start = (size_t)ith * chunk;
    if (start >= a->numel) {
        return;
    }
    size_t end = start + chunk;
    if (end > a->numel) {
        end = a->numel;
    }
    gradient_accumulate_f32_impl(a->dst + start, a->src + start, end - start);
}

static void ck_scale_parallel_work(int ith, int nth, void *argp)
{
    ck_scale_parallel_args_t *a = (ck_scale_parallel_args_t *)argp;
    if (!a || !a->grad || a->numel == 0) {
        return;
    }
    size_t chunk = (a->numel + (size_t)nth - 1u) / (size_t)nth;
    size_t start = (size_t)ith * chunk;
    if (start >= a->numel) {
        return;
    }
    size_t end = start + chunk;
    if (end > a->numel) {
        end = a->numel;
    }
    gradient_scale_f32_impl(a->grad + start, end - start, a->scale);
}

static void ck_sum_sq_parallel_work(int ith, int nth, void *argp)
{
    ck_sum_sq_parallel_args_t *a = (ck_sum_sq_parallel_args_t *)argp;
    if (!a || !a->grad || a->numel == 0 || ith < 0 || ith >= CK_OPT_PAR_MAX_THREADS) {
        return;
    }
    size_t chunk = (a->numel + (size_t)nth - 1u) / (size_t)nth;
    size_t start = (size_t)ith * chunk;
    if (start >= a->numel) {
        a->partial[ith] = 0.0;
        return;
    }
    size_t end = start + chunk;
    if (end > a->numel) {
        end = a->numel;
    }
    a->partial[ith] = gradient_sum_sq_f32_impl(a->grad + start, end - start);
}

static void ck_sum_sq_multi_parallel_work(int ith, int nth, void *argp)
{
    ck_sum_sq_multi_parallel_args_t *a = (ck_sum_sq_multi_parallel_args_t *)argp;
    if (!a || !a->grads || !a->numels || a->tensor_count <= 0 ||
        ith < 0 || ith >= CK_OPT_PAR_MAX_THREADS) {
        return;
    }

    double sum_sq = 0.0;
    for (int ti = ith; ti < a->tensor_count; ti += nth) {
        const float *g = a->grads[ti];
        size_t n = a->numels[ti];
        if (!g || n == 0) {
            continue;
        }
        sum_sq += gradient_sum_sq_f32_impl(g, n);
    }
    a->partial[ith] = sum_sq;
}

static void ck_adamw_multi_parallel_work(int ith, int nth, void *argp)
{
    ck_adamw_multi_parallel_args_t *a = (ck_adamw_multi_parallel_args_t *)argp;
    if (!a || !a->grads || !a->weights || !a->m_states || !a->v_states || !a->numels ||
        a->tensor_count <= 0) {
        return;
    }

    const int per = (a->tensor_count + nth - 1) / nth;
    const int t0 = per * ith;
    int t1 = t0 + per;
    if (t0 >= a->tensor_count) {
        return;
    }
    if (t1 > a->tensor_count) {
        t1 = a->tensor_count;
    }

    for (int ti = t0; ti < t1; ++ti) {
        float *g = a->grads[ti];
        float *w = a->weights[ti];
        float *m = a->m_states[ti];
        float *v = a->v_states[ti];
        size_t n = a->numels[ti];
        if (!g || !w || !m || !v || n == 0) {
            continue;
        }

        if (a->grad_scale != 1.0f) {
            gradient_scale_f32_impl(g, n, a->grad_scale);
        }

        adamw_update_f32_impl(
            g,
            w,
            m,
            v,
            n,
            a->lr,
            a->beta1,
            a->beta2,
            a->eps,
            a->weight_decay,
            a->step);
    }
}


/**
 * @brief AdamW optimizer update (fp32 version)
 *
 * Updates weights in-place using the AdamW algorithm.
 * Momentum (m) and variance (v) are stored in fp32 for numerical stability.
 *
 * @param grad       Gradient tensor (fp32) [numel]
 * @param weight     Weight tensor to update (fp32, in-place) [numel]
 * @param m          First moment (momentum) buffer (fp32, in-place) [numel]
 * @param v          Second moment (variance) buffer (fp32, in-place) [numel]
 * @param numel      Number of elements
 * @param lr         Learning rate
 * @param beta1      Exponential decay rate for first moment (typically 0.9)
 * @param beta2      Exponential decay rate for second moment (typically 0.999)
 * @param eps        Small constant for numerical stability (typically 1e-8)
 * @param weight_decay Weight decay coefficient (typically 0.01)
 * @param step       Current step number (1-indexed for bias correction)
 */
static void adamw_update_f32_impl(
    const float *grad,
    float *weight,
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

    // Precompute constants
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;

    if (ck_strict_parity_enabled()) {
        const double beta1_d = (double)beta1;
        const double beta2_d = (double)beta2;
        const double one_minus_beta1_d = 1.0 - beta1_d;
        const double one_minus_beta2_d = 1.0 - beta2_d;
        const double lr_d = (double)lr;
        const double eps_d = (double)eps;
        const double wd_d = (double)weight_decay;

        const double bc1 = 1.0 - pow(beta1_d, (double)step);
        const double bc2 = 1.0 - pow(beta2_d, (double)step);
        const double step_size = lr_d / bc1;
        const double bc2_sqrt = sqrt(bc2);
        const double wd_scale = 1.0 - lr_d * wd_d;

        for (size_t i = 0; i < numel; ++i) {
            double g = (double)grad[i];
            double w = (double)weight[i];
            double m_i = (double)m[i];
            double v_i = (double)v[i];

            m_i = beta1_d * m_i + one_minus_beta1_d * g;
            v_i = beta2_d * v_i + one_minus_beta2_d * g * g;

            // Match PyTorch AdamW op order: decoupled weight decay first.
            w *= wd_scale;

            double denom = sqrt(v_i) / bc2_sqrt + eps_d;
            w -= step_size * (m_i / denom);

            m[i] = (float)m_i;
            v[i] = (float)v_i;
            weight[i] = (float)w;
        }
        return;
    }

#if defined(__AVX512F__)
    // AVX-512 path: process 16 floats at a time
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
        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 w = _mm512_loadu_ps(&weight[i]);
        __m512 m_val = _mm512_loadu_ps(&m[i]);
        __m512 v_val = _mm512_loadu_ps(&v[i]);

        // m = beta1 * m + (1 - beta1) * g
        m_val = _mm512_fmadd_ps(v_beta1, m_val, _mm512_mul_ps(v_one_minus_beta1, g));

        // v = beta2 * v + (1 - beta2) * g^2
        __m512 g_sq = _mm512_mul_ps(g, g);
        v_val = _mm512_fmadd_ps(v_beta2, v_val, _mm512_mul_ps(v_one_minus_beta2, g_sq));

        // Bias-corrected estimates
        __m512 m_hat = _mm512_mul_ps(m_val, v_bc1_inv);
        __m512 v_hat = _mm512_mul_ps(v_val, v_bc2_inv);

        // w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
        __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), v_eps);
        __m512 update = _mm512_div_ps(m_hat, denom);
        update = _mm512_fmadd_ps(v_weight_decay, w, update);
        w = _mm512_fnmadd_ps(v_lr, update, w);

        _mm512_storeu_ps(&weight[i], w);
        _mm512_storeu_ps(&m[i], m_val);
        _mm512_storeu_ps(&v[i], v_val);
    }

    // Scalar tail
    for (; i < numel; ++i) {
        float g = grad[i];
        float w = weight[i];
        m[i] = beta1 * m[i] + one_minus_beta1 * g;
        v[i] = beta2 * v[i] + one_minus_beta2 * g * g;
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        weight[i] = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    }

#elif defined(__AVX__)
    // AVX path: process 8 floats at a time (no FMA on older CPUs like Ivy Bridge)
    __m256 v_beta1 = _mm256_set1_ps(beta1);
    __m256 v_beta2 = _mm256_set1_ps(beta2);
    __m256 v_one_minus_beta1 = _mm256_set1_ps(one_minus_beta1);
    __m256 v_one_minus_beta2 = _mm256_set1_ps(one_minus_beta2);
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_eps = _mm256_set1_ps(eps);
    __m256 v_weight_decay = _mm256_set1_ps(weight_decay);
    __m256 v_bc1_inv = _mm256_set1_ps(1.0f / bias_correction1);
    __m256 v_bc2_inv = _mm256_set1_ps(1.0f / bias_correction2);

    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad[i]);
        __m256 w = _mm256_loadu_ps(&weight[i]);
        __m256 m_val = _mm256_loadu_ps(&m[i]);
        __m256 v_val = _mm256_loadu_ps(&v[i]);

        // m = beta1 * m + (1 - beta1) * g
        m_val = _mm256_add_ps(_mm256_mul_ps(v_beta1, m_val),
                              _mm256_mul_ps(v_one_minus_beta1, g));

        // v = beta2 * v + (1 - beta2) * g^2
        __m256 g_sq = _mm256_mul_ps(g, g);
        v_val = _mm256_add_ps(_mm256_mul_ps(v_beta2, v_val),
                              _mm256_mul_ps(v_one_minus_beta2, g_sq));

        // Bias-corrected estimates
        __m256 m_hat = _mm256_mul_ps(m_val, v_bc1_inv);
        __m256 v_hat = _mm256_mul_ps(v_val, v_bc2_inv);

        // w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), v_eps);
        __m256 update = _mm256_div_ps(m_hat, denom);
        update = _mm256_add_ps(update, _mm256_mul_ps(v_weight_decay, w));
        w = _mm256_sub_ps(w, _mm256_mul_ps(v_lr, update));

        _mm256_storeu_ps(&weight[i], w);
        _mm256_storeu_ps(&m[i], m_val);
        _mm256_storeu_ps(&v[i], v_val);
    }

    // Scalar tail
    for (; i < numel; ++i) {
        float g = grad[i];
        float w = weight[i];
        m[i] = beta1 * m[i] + one_minus_beta1 * g;
        v[i] = beta2 * v[i] + one_minus_beta2 * g * g;
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        weight[i] = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    }

#elif defined(__SSE2__)
    // SSE2 path: process 4 floats at a time
    __m128 v_beta1 = _mm_set1_ps(beta1);
    __m128 v_beta2 = _mm_set1_ps(beta2);
    __m128 v_one_minus_beta1 = _mm_set1_ps(one_minus_beta1);
    __m128 v_one_minus_beta2 = _mm_set1_ps(one_minus_beta2);
    __m128 v_lr = _mm_set1_ps(lr);
    __m128 v_eps = _mm_set1_ps(eps);
    __m128 v_weight_decay = _mm_set1_ps(weight_decay);
    __m128 v_bc1_inv = _mm_set1_ps(1.0f / bias_correction1);
    __m128 v_bc2_inv = _mm_set1_ps(1.0f / bias_correction2);

    size_t i = 0;
    for (; i + 4 <= numel; i += 4) {
        __m128 g = _mm_loadu_ps(&grad[i]);
        __m128 w = _mm_loadu_ps(&weight[i]);
        __m128 m_val = _mm_loadu_ps(&m[i]);
        __m128 v_val = _mm_loadu_ps(&v[i]);

        // m = beta1 * m + (1 - beta1) * g
        m_val = _mm_add_ps(_mm_mul_ps(v_beta1, m_val),
                           _mm_mul_ps(v_one_minus_beta1, g));

        // v = beta2 * v + (1 - beta2) * g^2
        __m128 g_sq = _mm_mul_ps(g, g);
        v_val = _mm_add_ps(_mm_mul_ps(v_beta2, v_val),
                           _mm_mul_ps(v_one_minus_beta2, g_sq));

        // Bias-corrected estimates
        __m128 m_hat = _mm_mul_ps(m_val, v_bc1_inv);
        __m128 v_hat = _mm_mul_ps(v_val, v_bc2_inv);

        // w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
        __m128 denom = _mm_add_ps(_mm_sqrt_ps(v_hat), v_eps);
        __m128 update = _mm_div_ps(m_hat, denom);
        update = _mm_add_ps(update, _mm_mul_ps(v_weight_decay, w));
        w = _mm_sub_ps(w, _mm_mul_ps(v_lr, update));

        _mm_storeu_ps(&weight[i], w);
        _mm_storeu_ps(&m[i], m_val);
        _mm_storeu_ps(&v[i], v_val);
    }

    // Scalar tail
    for (; i < numel; ++i) {
        float g = grad[i];
        float w = weight[i];
        m[i] = beta1 * m[i] + one_minus_beta1 * g;
        v[i] = beta2 * v[i] + one_minus_beta2 * g * g;
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        weight[i] = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    }

#else
    // Scalar path
    for (size_t i = 0; i < numel; ++i) {
        float g = grad[i];
        float w = weight[i];
        m[i] = beta1 * m[i] + one_minus_beta1 * g;
        v[i] = beta2 * v[i] + one_minus_beta2 * g * g;
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        weight[i] = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    }
#endif
}


void adamw_update_f32(
    const float *grad,
    float *weight,
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

    ck_threadpool_t *pool = ck_threadpool_global();
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;
    if (!pool || nth <= 1 || nth > CK_OPT_PAR_MAX_THREADS || numel < CK_OPT_PAR_MIN_NUMEL) {
        adamw_update_f32_impl(grad, weight, m, v, numel, lr, beta1, beta2, eps, weight_decay, step);
        return;
    }

    ck_adamw_parallel_args_t args = {
        .grad = grad,
        .weight = weight,
        .m = m,
        .v = v,
        .numel = numel,
        .lr = lr,
        .beta1 = beta1,
        .beta2 = beta2,
        .eps = eps,
        .weight_decay = weight_decay,
        .step = step,
    };
    ck_threadpool_dispatch(pool, ck_adamw_parallel_work, &args);
}


void adamw_clip_update_multi_f32(
    float *const *grads,
    float *const *weights,
    float *const *m_states,
    float *const *v_states,
    const size_t *numels,
    int tensor_count,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float max_grad_norm,
    int step)
{
    if (!grads || !weights || !m_states || !v_states || !numels || tensor_count <= 0) {
        return;
    }

    size_t total_numel = 0;
    int valid_tensors = 0;
    for (int i = 0; i < tensor_count; ++i) {
        if (grads[i] && weights[i] && m_states[i] && v_states[i] && numels[i] > 0) {
            total_numel += numels[i];
            valid_tensors += 1;
        }
    }
    if (valid_tensors == 0 || total_numel == 0) {
        return;
    }

    float grad_scale = 1.0f;
    if (max_grad_norm > 0.0f) {
        float global_norm = gradient_global_norm_multi_f32((const float *const *)grads, numels, tensor_count);
        if (global_norm > max_grad_norm) {
            grad_scale = max_grad_norm / global_norm;
        }
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;

    if (!pool || nth <= 1 || nth > CK_OPT_PAR_MAX_THREADS ||
        total_numel < CK_OPT_PAR_MIN_NUMEL || valid_tensors < 2) {
        for (int i = 0; i < tensor_count; ++i) {
            float *g = grads[i];
            float *w = weights[i];
            float *m = m_states[i];
            float *v = v_states[i];
            size_t n = numels[i];
            if (!g || !w || !m || !v || n == 0) {
                continue;
            }
            if (grad_scale != 1.0f) {
                gradient_scale_f32_impl(g, n, grad_scale);
            }
            adamw_update_f32_impl(g, w, m, v, n, lr, beta1, beta2, eps, weight_decay, step);
        }
        return;
    }

    ck_adamw_multi_parallel_args_t args = {
        .grads = grads,
        .weights = weights,
        .m_states = m_states,
        .v_states = v_states,
        .numels = numels,
        .tensor_count = tensor_count,
        .lr = lr,
        .beta1 = beta1,
        .beta2 = beta2,
        .eps = eps,
        .weight_decay = weight_decay,
        .grad_scale = grad_scale,
        .step = step,
    };
    ck_threadpool_dispatch(pool, ck_adamw_multi_parallel_work, &args);
}


/**
 * @brief SGD with momentum optimizer update (fp32 version)
 *
 * v_t = momentum * v_{t-1} + g_t
 * w_t = w_{t-1} - lr * (v_t + weight_decay * w_{t-1})
 *
 * @param grad       Gradient tensor (fp32) [numel]
 * @param weight     Weight tensor to update (fp32, in-place) [numel]
 * @param velocity   Velocity buffer (fp32, in-place) [numel]
 * @param numel      Number of elements
 * @param lr         Learning rate
 * @param momentum   Momentum coefficient (typically 0.9)
 * @param weight_decay Weight decay coefficient
 */
void sgd_momentum_update_f32(
    const float *grad,
    float *weight,
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
    // AVX-512 path: process 16 floats at a time
    __m512 v_lr = _mm512_set1_ps(lr);
    __m512 v_momentum = _mm512_set1_ps(momentum);
    __m512 v_weight_decay = _mm512_set1_ps(weight_decay);

    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 w = _mm512_loadu_ps(&weight[i]);
        __m512 vel = _mm512_loadu_ps(&velocity[i]);

        vel = _mm512_fmadd_ps(v_momentum, vel, g);
        __m512 update = _mm512_fmadd_ps(v_weight_decay, w, vel);
        w = _mm512_fnmadd_ps(v_lr, update, w);

        _mm512_storeu_ps(&weight[i], w);
        _mm512_storeu_ps(&velocity[i], vel);
    }

    for (; i < numel; ++i) {
        velocity[i] = momentum * velocity[i] + grad[i];
        weight[i] = weight[i] - lr * (velocity[i] + weight_decay * weight[i]);
    }

#elif defined(__AVX__)
    // AVX path: process 8 floats at a time
    __m256 v_lr = _mm256_set1_ps(lr);
    __m256 v_momentum = _mm256_set1_ps(momentum);
    __m256 v_weight_decay = _mm256_set1_ps(weight_decay);

    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad[i]);
        __m256 w = _mm256_loadu_ps(&weight[i]);
        __m256 vel = _mm256_loadu_ps(&velocity[i]);

        // v = momentum * v + g
        vel = _mm256_add_ps(_mm256_mul_ps(v_momentum, vel), g);

        // w = w - lr * (v + weight_decay * w)
        __m256 update = _mm256_add_ps(vel, _mm256_mul_ps(v_weight_decay, w));
        w = _mm256_sub_ps(w, _mm256_mul_ps(v_lr, update));

        _mm256_storeu_ps(&weight[i], w);
        _mm256_storeu_ps(&velocity[i], vel);
    }

    for (; i < numel; ++i) {
        velocity[i] = momentum * velocity[i] + grad[i];
        weight[i] = weight[i] - lr * (velocity[i] + weight_decay * weight[i]);
    }

#elif defined(__SSE2__)
    // SSE2 path: process 4 floats at a time
    __m128 v_lr = _mm_set1_ps(lr);
    __m128 v_momentum = _mm_set1_ps(momentum);
    __m128 v_weight_decay = _mm_set1_ps(weight_decay);

    size_t i = 0;
    for (; i + 4 <= numel; i += 4) {
        __m128 g = _mm_loadu_ps(&grad[i]);
        __m128 w = _mm_loadu_ps(&weight[i]);
        __m128 vel = _mm_loadu_ps(&velocity[i]);

        vel = _mm_add_ps(_mm_mul_ps(v_momentum, vel), g);
        __m128 update = _mm_add_ps(vel, _mm_mul_ps(v_weight_decay, w));
        w = _mm_sub_ps(w, _mm_mul_ps(v_lr, update));

        _mm_storeu_ps(&weight[i], w);
        _mm_storeu_ps(&velocity[i], vel);
    }

    for (; i < numel; ++i) {
        velocity[i] = momentum * velocity[i] + grad[i];
        weight[i] = weight[i] - lr * (velocity[i] + weight_decay * weight[i]);
    }

#else
    // Scalar path
    for (size_t i = 0; i < numel; ++i) {
        velocity[i] = momentum * velocity[i] + grad[i];
        weight[i] = weight[i] - lr * (velocity[i] + weight_decay * weight[i]);
    }
#endif
}


/**
 * @brief Zero out gradient buffer (fp32)
 *
 * @param grad  Gradient tensor to zero [numel]
 * @param numel Number of elements
 */
void zero_gradients_f32(float *grad, size_t numel)
{
    if (!grad || numel == 0) {
        return;
    }
    memset(grad, 0, numel * sizeof(float));
}


/**
 * @brief Accumulate gradients: dst += src (fp32)
 *
 * Used for gradient accumulation across micro-batches.
 *
 * @param dst   Destination gradient buffer (in-place) [numel]
 * @param src   Source gradient buffer [numel]
 * @param numel Number of elements
 */
static void gradient_accumulate_f32_impl(float *dst, const float *src, size_t numel)
{
    if (!dst || !src || numel == 0) {
        return;
    }

#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 d = _mm512_loadu_ps(&dst[i]);
        __m512 s = _mm512_loadu_ps(&src[i]);
        _mm512_storeu_ps(&dst[i], _mm512_add_ps(d, s));
    }
    for (; i < numel; ++i) {
        dst[i] += src[i];
    }

#elif defined(__AVX__)
    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 d = _mm256_loadu_ps(&dst[i]);
        __m256 s = _mm256_loadu_ps(&src[i]);
        _mm256_storeu_ps(&dst[i], _mm256_add_ps(d, s));
    }
    for (; i < numel; ++i) {
        dst[i] += src[i];
    }

#elif defined(__SSE2__)
    size_t i = 0;
    for (; i + 4 <= numel; i += 4) {
        __m128 d = _mm_loadu_ps(&dst[i]);
        __m128 s = _mm_loadu_ps(&src[i]);
        _mm_storeu_ps(&dst[i], _mm_add_ps(d, s));
    }
    for (; i < numel; ++i) {
        dst[i] += src[i];
    }

#else
    for (size_t i = 0; i < numel; ++i) {
        dst[i] += src[i];
    }
#endif
}


void gradient_accumulate_f32(float *dst, const float *src, size_t numel)
{
    if (!dst || !src || numel == 0) {
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;
    if (!pool || nth <= 1 || nth > CK_OPT_PAR_MAX_THREADS || numel < CK_OPT_PAR_MIN_NUMEL) {
        gradient_accumulate_f32_impl(dst, src, numel);
        return;
    }

    ck_accum_parallel_args_t args = {
        .dst = dst,
        .src = src,
        .numel = numel,
    };
    ck_threadpool_dispatch(pool, ck_accum_parallel_work, &args);
}


/**
 * @brief Scale gradients by a constant: grad *= scale (fp32)
 *
 * Used for averaging gradients after accumulation: grad /= batch_size
 *
 * @param grad  Gradient tensor to scale (in-place) [numel]
 * @param numel Number of elements
 * @param scale Scale factor (typically 1.0 / batch_size)
 */
static void gradient_scale_f32_impl(float *grad, size_t numel, float scale)
{
    if (!grad || numel == 0) {
        return;
    }

#if defined(__AVX512F__)
    __m512 v_scale = _mm512_set1_ps(scale);
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        _mm512_storeu_ps(&grad[i], _mm512_mul_ps(g, v_scale));
    }
    for (; i < numel; ++i) {
        grad[i] *= scale;
    }

#elif defined(__AVX__)
    __m256 v_scale = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad[i]);
        _mm256_storeu_ps(&grad[i], _mm256_mul_ps(g, v_scale));
    }
    for (; i < numel; ++i) {
        grad[i] *= scale;
    }

#elif defined(__SSE2__)
    __m128 v_scale = _mm_set1_ps(scale);
    size_t i = 0;
    for (; i + 4 <= numel; i += 4) {
        __m128 g = _mm_loadu_ps(&grad[i]);
        _mm_storeu_ps(&grad[i], _mm_mul_ps(g, v_scale));
    }
    for (; i < numel; ++i) {
        grad[i] *= scale;
    }

#else
    for (size_t i = 0; i < numel; ++i) {
        grad[i] *= scale;
    }
#endif
}


void gradient_scale_f32(float *grad, size_t numel, float scale)
{
    if (!grad || numel == 0) {
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;
    if (!pool || nth <= 1 || nth > CK_OPT_PAR_MAX_THREADS || numel < CK_OPT_PAR_MIN_NUMEL) {
        gradient_scale_f32_impl(grad, numel, scale);
        return;
    }

    ck_scale_parallel_args_t args = {
        .grad = grad,
        .numel = numel,
        .scale = scale,
    };
    ck_threadpool_dispatch(pool, ck_scale_parallel_work, &args);
}

static double gradient_sum_sq_f32_impl(const float *grad, size_t numel)
{
    if (!grad || numel == 0) {
        return 0.0;
    }

    double sum_sq = 0.0;
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= numel; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        acc = _mm512_fmadd_ps(g, g, acc);
    }
    sum_sq = _mm512_reduce_add_ps(acc);
    for (; i < numel; ++i) {
        sum_sq += (double)grad[i] * (double)grad[i];
    }
#elif defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad[i]);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(g, g));
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sums = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum_sq = _mm_cvtss_f32(sums);
    for (; i < numel; ++i) {
        sum_sq += (double)grad[i] * (double)grad[i];
    }
#elif defined(__SSE2__)
    __m128 acc = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= numel; i += 4) {
        __m128 g = _mm_loadu_ps(&grad[i]);
        acc = _mm_add_ps(acc, _mm_mul_ps(g, g));
    }
    __m128 shuf = _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(acc, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum_sq = _mm_cvtss_f32(sums);
    for (; i < numel; ++i) {
        sum_sq += (double)grad[i] * (double)grad[i];
    }
#else
    for (size_t i = 0; i < numel; ++i) {
        sum_sq += (double)grad[i] * (double)grad[i];
    }
#endif
    return sum_sq;
}


/**
 * @brief Clip gradient norm (fp32)
 *
 * If ||grad||_2 > max_norm, scale grad so that ||grad||_2 = max_norm
 *
 * @param grad     Gradient tensor to clip (in-place) [numel]
 * @param numel    Number of elements
 * @param max_norm Maximum allowed L2 norm
 * @return         The original L2 norm before clipping
 */
float gradient_clip_norm_f32(float *grad, size_t numel, float max_norm)
{
    if (!grad || numel == 0 || max_norm <= 0.0f) {
        return 0.0f;
    }

    double sum_sq = 0.0;
    ck_threadpool_t *pool = ck_threadpool_global();
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;

    if (pool && nth > 1 && nth <= CK_OPT_PAR_MAX_THREADS && numel >= CK_OPT_PAR_MIN_NUMEL) {
        ck_sum_sq_parallel_args_t args;
        args.grad = grad;
        args.numel = numel;
        for (int i = 0; i < CK_OPT_PAR_MAX_THREADS; ++i) {
            args.partial[i] = 0.0;
        }
        ck_threadpool_dispatch(pool, ck_sum_sq_parallel_work, &args);
        for (int i = 0; i < nth; ++i) {
            sum_sq += args.partial[i];
        }
    } else {
        sum_sq = gradient_sum_sq_f32_impl(grad, numel);
    }

    float norm = sqrtf((float)sum_sq);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        gradient_scale_f32(grad, numel, scale);
    }
    return norm;
}

float gradient_global_norm_multi_f32(const float *const *grads, const size_t *numels, int tensor_count)
{
    if (!grads || !numels || tensor_count <= 0) {
        return 0.0f;
    }

    size_t total_numel = 0;
    int valid_tensors = 0;
    for (int i = 0; i < tensor_count; ++i) {
        if (!grads[i] || numels[i] == 0) {
            continue;
        }
        total_numel += numels[i];
        valid_tensors += 1;
    }
    if (total_numel == 0 || valid_tensors == 0) {
        return 0.0f;
    }

    double sum_sq = 0.0;
    ck_threadpool_t *pool = ck_threadpool_global();
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;

    if (pool && nth > 1 && nth <= CK_OPT_PAR_MAX_THREADS &&
        total_numel >= CK_OPT_PAR_MIN_NUMEL && valid_tensors > 1) {
        ck_sum_sq_multi_parallel_args_t args;
        args.grads = grads;
        args.numels = numels;
        args.tensor_count = tensor_count;
        for (int i = 0; i < CK_OPT_PAR_MAX_THREADS; ++i) {
            args.partial[i] = 0.0;
        }
        ck_threadpool_dispatch(pool, ck_sum_sq_multi_parallel_work, &args);
        for (int i = 0; i < nth; ++i) {
            sum_sq += args.partial[i];
        }
    } else {
        for (int i = 0; i < tensor_count; ++i) {
            const float *g = grads[i];
            size_t n = numels[i];
            if (!g || n == 0) {
                continue;
            }
            sum_sq += gradient_sum_sq_f32_impl(g, n);
        }
    }

    if (!(sum_sq > 0.0)) {
        return 0.0f;
    }
    return sqrtf((float)sum_sq);
}
