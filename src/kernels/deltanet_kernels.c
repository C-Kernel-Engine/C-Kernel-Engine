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
 * Per head:
 *   q_hat   = l2_norm(q) / sqrt(state_dim)
 *   k_hat   = l2_norm(k)
 *   beta_s  = sigmoid(beta)
 *   gate    = exp(g)
 *   S       = gate * S_prev
 *   kv_mem  = S^T * k_hat
 *   delta   = (v - kv_mem) * beta_s
 *   S_new   = S + outer(k_hat, delta)
 *   out     = S_new^T * q_hat
 *
 * Design:
 *   - *_ref is the scalar reference implementation.
 *   - *_avx keeps a simple 1-row vector walk.
 *   - *_avx2 precomputes normalized q/k rows and unrolls the state sweep in
 *     row pairs to reduce loop overhead and layout churn.
 *   - The public dispatcher selects the best compiled ISA unless strict parity
 *     is enabled, in which case it falls back to *_ref.
 */

#include "ckernel_engine.h"

#include <math.h>
#include <stddef.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#define CK_DELTANET_MAX_STACK_DIM 4096

static inline float ck_deltanet_sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float ck_deltanet_l2_inv_norm_scalar(const float *x, int dim, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    return 1.0f / sqrtf(sum_sq + eps);
}

static void ck_deltanet_scaled_l2_norm_backward_ref(const float *x,
                                                    const float *d_y,
                                                    float *d_x,
                                                    int dim,
                                                    float inv_norm,
                                                    float scale)
{
    float dot = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += d_y[i] * x[i];
    }

    const float scaled_inv = scale * inv_norm;
    const float proj_scale = scale * inv_norm * inv_norm * inv_norm * dot;
    for (int i = 0; i < dim; ++i) {
        d_x[i] = scaled_inv * d_y[i] - proj_scale * x[i];
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

        const float q_inv_norm = ck_deltanet_l2_inv_norm_scalar(q_head, state_dim, norm_eps);
        const float k_inv_norm = ck_deltanet_l2_inv_norm_scalar(k_head, state_dim, norm_eps);
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
                const float k_hat = k_head[row] * k_inv_norm;
                kv_mem += state_cur[(size_t)row * (size_t)state_dim + (size_t)col] * k_hat;
            }

            const float delta = (v_head[col] - kv_mem) * beta_s;
            for (int row = 0; row < state_dim; ++row) {
                const float k_hat = k_head[row] * k_inv_norm;
                state_cur[(size_t)row * (size_t)state_dim + (size_t)col] += k_hat * delta;
            }
        }

        for (int col = 0; col < state_dim; ++col) {
            float acc = 0.0f;
            for (int row = 0; row < state_dim; ++row) {
                const float q_hat = q_head[row] * q_inv_norm * q_scale;
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

        const float q_inv_norm = ck_deltanet_l2_inv_norm_scalar(q_head, state_dim, norm_eps);
        const float k_inv_norm = ck_deltanet_l2_inv_norm_scalar(k_head, state_dim, norm_eps);
        const float beta_s = ck_deltanet_sigmoidf(beta[h]);
        const float gate = expf(g[h]);

        float qk_dot = 0.0f;
        float out_delta_dot = 0.0f;
        float beta_acc = 0.0f;
        float gate_acc = 0.0f;

        for (int i = 0; i < state_dim; ++i) {
            q_hat[i] = q_head[i] * q_inv_norm * q_scale;
            k_hat[i] = k_head[i] * k_inv_norm;
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

        ck_deltanet_scaled_l2_norm_backward_ref(
            q_head, d_q_hat, d_q_head, state_dim, q_inv_norm, q_scale);
        ck_deltanet_scaled_l2_norm_backward_ref(
            k_head, d_k_hat, d_k_head, state_dim, k_inv_norm, 1.0f);

        d_g[h] = gate_acc * gate;
        d_beta[h] = beta_acc * beta_s * (1.0f - beta_s);
    }
}

#if defined(__AVX__)
static inline float ck_deltanet_hsum256(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static float ck_deltanet_l2_inv_norm_avx(const float *x, int dim, float eps)
{
    __m256 sum_sq_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        sum_sq_v = _mm256_add_ps(sum_sq_v, _mm256_mul_ps(xv, xv));
    }

    float sum_sq = ck_deltanet_hsum256(sum_sq_v);
    for (; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }

    return 1.0f / sqrtf(sum_sq + eps);
}

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

        const float q_inv_norm = ck_deltanet_l2_inv_norm_avx(q_head, state_dim, norm_eps);
        const float k_inv_norm = ck_deltanet_l2_inv_norm_avx(k_head, state_dim, norm_eps);
        const float beta_s = ck_deltanet_sigmoidf(beta[h]);
        const float gate = expf(g[h]);

        ck_deltanet_scale_rows_avx(q_head, q_hat, state_dim, q_inv_norm * q_scale);
        ck_deltanet_scale_rows_avx(k_head, k_hat, state_dim, k_inv_norm);

        const __m256 gate_v = _mm256_set1_ps(gate);
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
static float ck_deltanet_l2_inv_norm_avx2(const float *x, int dim, float eps)
{
    __m256 sum_sq_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
#if defined(__FMA__)
        sum_sq_v = _mm256_fmadd_ps(xv, xv, sum_sq_v);
#else
        sum_sq_v = _mm256_add_ps(sum_sq_v, _mm256_mul_ps(xv, xv));
#endif
    }

    float sum_sq = ck_deltanet_hsum256(sum_sq_v);
    for (; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }

    return 1.0f / sqrtf(sum_sq + eps);
}

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

        const float q_inv_norm = ck_deltanet_l2_inv_norm_avx2(q_head, state_dim, norm_eps);
        const float k_inv_norm = ck_deltanet_l2_inv_norm_avx2(k_head, state_dim, norm_eps);
        const float beta_s = ck_deltanet_sigmoidf(beta[h]);
        const float gate = expf(g[h]);

        ck_deltanet_scale_rows_avx(q_head, q_hat, state_dim, q_inv_norm * q_scale);
        ck_deltanet_scale_rows_avx(k_head, k_hat, state_dim, k_inv_norm);

        const __m256 gate_v = _mm256_set1_ps(gate);
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

            col = 0;
            for (; col + 8 <= state_dim; col += 8) {
                __m256 prev0_v = _mm256_loadu_ps(state_prev + row0_off + (size_t)col);
                __m256 prev1_v = _mm256_loadu_ps(state_prev + row1_off + (size_t)col);
                __m256 cur0_v = _mm256_mul_ps(prev0_v, gate_v);
                __m256 cur1_v = _mm256_mul_ps(prev1_v, gate_v);
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
static inline float ck_deltanet_hsum512(__m512 v)
{
    return _mm512_reduce_add_ps(v);
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

static float ck_deltanet_l2_inv_norm_avx512(const float *x, int dim, float eps)
{
    __m512 sum_sq_v = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        sum_sq_v = _mm512_fmadd_ps(xv, xv, sum_sq_v);
    }
    float sum_sq = ck_deltanet_hsum512(sum_sq_v);
    for (; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    return 1.0f / sqrtf(sum_sq + eps);
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

        const float q_inv_norm = ck_deltanet_l2_inv_norm_avx512(q_head, state_dim, norm_eps);
        const float k_inv_norm = ck_deltanet_l2_inv_norm_avx512(k_head, state_dim, norm_eps);
        const float beta_s = ck_deltanet_sigmoidf(beta[h]);
        const float gate = expf(g[h]);

        ck_deltanet_scale_rows_avx512(q_head, q_hat, state_dim, q_inv_norm * q_scale);
        ck_deltanet_scale_rows_avx512(k_head, k_hat, state_dim, k_inv_norm);

        const __m512 gate_v = _mm512_set1_ps(gate);
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
            col = 0;
            for (; col + 16 <= state_dim; col += 16) {
                __m512 prev_v = _mm512_loadu_ps(state_prev + row_off + (size_t)col);
                __m512 cur_v = _mm512_mul_ps(prev_v, gate_v);
                __m512 kv_v = _mm512_loadu_ps(kv_mem + col);
                kv_v = _mm512_fmadd_ps(cur_v, k_hat_v, kv_v);
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
                __m512 updated_v = _mm512_fmadd_ps(k_hat_v, delta_v, cur_v);
                out_v = _mm512_fmadd_ps(updated_v, q_hat_v, out_v);
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

const char *gated_deltanet_impl_name(void)
{
    if (ck_strict_parity_enabled()) {
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

    if (ck_strict_parity_enabled() || state_dim > CK_DELTANET_MAX_STACK_DIM) {
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
