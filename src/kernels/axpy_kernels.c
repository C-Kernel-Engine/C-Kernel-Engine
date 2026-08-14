/**
 * @file axpy_kernels.c
 * @brief AXPY kernels for FP32: y = y + alpha * x
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
 * Classic BLAS Level-1 operation used in MoE expert output accumulation.
 * When gathering expert outputs: output += weight[i] * expert_output[i]
 *
 * Operations:
 *   - axpy_f32: y += alpha * x (in-place)
 *   - axpy_strided_f32: strided version for non-contiguous memory
 *   - weighted_sum_f32: sum multiple vectors with weights
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ckernel_engine.h"
#include "ckernel_dtype.h"
#include "bf16_utils.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* =============================================================================
 * AXPY: y = y + alpha * x
 *
 * Core operation for MoE expert gathering:
 *   output = sum_i(weight_i * expert_output_i)
 *
 * Implemented as: output += weight * expert_output (called for each expert)
 * ============================================================================= */

/**
 * @brief In-place AXPY: y += alpha * x
 * @test test_axpy.py::TestAXPY::test_axpy_f32
 * @test test_axpy.py::TestAXPY::test_axpy_vs_naive
 *
 * In-place scaled vector addition: y += alpha * x
 * BLAS-like axpy operation.
 *
 * After changes: make test
 */
void axpy_f32(float *y,
              const float *x,
              float alpha,
              int n)
{
    if (!y || !x || n <= 0) {
        return;
    }

    int i = 0;

#ifdef __AVX512F__
    __m512 valpha = _mm512_set1_ps(alpha);
    for (; i + 16 <= n; i += 16) {
        __m512 vy = _mm512_loadu_ps(&y[i]);
        __m512 vx = _mm512_loadu_ps(&x[i]);
        vy = _mm512_fmadd_ps(vx, valpha, vy);  /* y = y + alpha * x */
        _mm512_storeu_ps(&y[i], vy);
    }
#endif

#ifdef __AVX2__
    __m256 valpha256 = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8) {
        __m256 vy = _mm256_loadu_ps(&y[i]);
        __m256 vx = _mm256_loadu_ps(&x[i]);
        vy = _mm256_fmadd_ps(vx, valpha256, vy);
        _mm256_storeu_ps(&y[i], vy);
    }
#endif

    /* Scalar remainder */
    for (; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

/* =============================================================================
 * Scaled copy: y = alpha * x
 *
 * First step when accumulating: initialize output with first expert's result.
 * ============================================================================= */

/**
 * @brief Scaled copy: y = alpha * x
 *
 * @param y Output vector [n]
 * @param x Input vector [n]
 * @param alpha Scalar multiplier
 * @param n Vector length
 */
void scal_copy_f32(float *y,
                   const float *x,
                   float alpha,
                   int n)
{
    if (!y || !x || n <= 0) {
        return;
    }

    int i = 0;

#ifdef __AVX512F__
    __m512 valpha = _mm512_set1_ps(alpha);
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vy = _mm512_mul_ps(vx, valpha);
        _mm512_storeu_ps(&y[i], vy);
    }
#endif

#ifdef __AVX2__
    __m256 valpha256 = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_mul_ps(vx, valpha256);
        _mm256_storeu_ps(&y[i], vy);
    }
#endif

    for (; i < n; i++) {
        y[i] = alpha * x[i];
    }
}

/* =============================================================================
 * Weighted sum: y = sum_i(weights[i] * x[i])
 *
 * Combine multiple expert outputs with their routing weights in one pass.
 * More efficient than multiple axpy calls when all inputs are available.
 * ============================================================================= */

/**
 * @brief Weighted sum of k vectors: y = sum_i(weights[i] * vectors[i])
 *
 * @param y Output vector [n]
 * @param vectors Array of k input vector pointers, each [n]
 * @param weights Array of k scalar weights
 * @param k Number of vectors to combine
 * @param n Vector length
 */
void weighted_sum_f32(float *y,
                      const float **vectors,
                      const float *weights,
                      int k,
                      int n)
{
    if (!y || !vectors || !weights || k <= 0 || n <= 0) {
        return;
    }

    /* Initialize with first vector */
    scal_copy_f32(y, vectors[0], weights[0], n);

    /* Accumulate rest */
    for (int i = 1; i < k; i++) {
        axpy_f32(y, vectors[i], weights[i], n);
    }
}

/* =============================================================================
 * Zero-initialized AXPY accumulation
 *
 * Zero output first, then accumulate. Useful when output may contain garbage.
 * ============================================================================= */

/**
 * @brief Zero output then accumulate: y = 0; y += alpha * x
 *
 * @param y Output vector [n], zeroed then accumulated
 * @param x Input vector [n]
 * @param alpha Scalar multiplier
 * @param n Vector length
 */
void axpy_zero_f32(float *y,
                   const float *x,
                   float alpha,
                   int n)
{
    if (!y || n <= 0) {
        return;
    }

    memset(y, 0, n * sizeof(float));

    if (x) {
        axpy_f32(y, x, alpha, n);
    }
}

/* =============================================================================
 * 2D batched AXPY for [tokens, hidden] shaped tensors
 *
 * Process multiple tokens at once, common in transformer inference.
 * ============================================================================= */

/**
 * @brief Batched AXPY for 2D tensors: Y[t,:] += alpha * X[t,:]
 *
 * @param Y Output tensor [num_tokens, dim]
 * @param X Input tensor [num_tokens, dim]
 * @param alpha Scalar multiplier
 * @param num_tokens Number of tokens
 * @param dim Hidden dimension
 * @param y_stride Stride between Y rows (for alignment)
 * @param x_stride Stride between X rows
 */
void axpy_2d_f32(float *Y,
                 const float *X,
                 float alpha,
                 int num_tokens,
                 int dim,
                 int y_stride,
                 int x_stride)
{
    if (!Y || !X || num_tokens <= 0 || dim <= 0) {
        return;
    }

    /* Default strides if not specified */
    if (y_stride <= 0) y_stride = dim;
    if (x_stride <= 0) x_stride = dim;

    for (int t = 0; t < num_tokens; t++) {
        axpy_f32(Y + t * y_stride, X + t * x_stride, alpha, dim);
    }
}

/* =============================================================================
 * MoE-specific: Accumulate expert output with routing weight
 *
 * Convenience wrapper with clear semantics for MoE usage.
 * ============================================================================= */

/**
 * @brief Accumulate expert output: output += routing_weight * expert_output
 *
 * @param output Token output buffer [hidden_dim], accumulated in place
 * @param expert_output Expert's output for this token [hidden_dim]
 * @param routing_weight Softmax routing weight for this expert
 * @param hidden_dim Hidden dimension
 */
void moe_accumulate_expert_f32(float *output,
                               const float *expert_output,
                               float routing_weight,
                               int hidden_dim)
{
    axpy_f32(output, expert_output, routing_weight, hidden_dim);
}


/* =============================================================================
 * Routed MoE expert MLP with ReLU2 activation.
 *
 * expert_up   layout: [n_experts, intermediate_dim, hidden_dim]
 * expert_down layout: [n_experts, hidden_dim, intermediate_dim]
 * output      layout: [rows, hidden_dim]
 * ============================================================================= */
static inline size_t ck_moe_up_idx(int e, int i, int h, int intermediate_dim, int hidden_dim)
{
    return ((size_t)e * (size_t)intermediate_dim + (size_t)i) * (size_t)hidden_dim + (size_t)h;
}

static inline size_t ck_moe_down_idx(int e, int h, int i, int hidden_dim, int intermediate_dim)
{
    return ((size_t)e * (size_t)hidden_dim + (size_t)h) * (size_t)intermediate_dim + (size_t)i;
}

static int ck_moe_debug_enabled(void)
{
    const char *v = getenv("CK_DEBUG_MOE");
    return v && v[0] && v[0] != '0';
}

static void ck_moe_debug_finite(const char *name, const float *x, size_t n)
{
    if (!ck_moe_debug_enabled() || !x) {
        return;
    }
    size_t finite = 0;
    size_t nan = 0;
    size_t inf = 0;
    float min_v = 0.0f;
    float max_v = 0.0f;
    int have = 0;
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        if (isnan(v)) {
            ++nan;
        } else if (isinf(v)) {
            ++inf;
        } else {
            ++finite;
            if (!have || v < min_v) min_v = v;
            if (!have || v > max_v) max_v = v;
            have = 1;
        }
    }
    fprintf(stderr,
            "[CK_DEBUG_MOE] %s finite=%zu/%zu nan=%zu inf=%zu min=%g max=%g\n",
            name,
            finite,
            n,
            nan,
            inf,
            have ? min_v : 0.0f,
            have ? max_v : 0.0f);
}

void moe_relu2_expert_forward_f32(const float *hidden,
                                  const int *indices,
                                  const float *routing_weights,
                                  const float *expert_up,
                                  const float *expert_down,
                                  float *output,
                                  int rows,
                                  int hidden_dim,
                                  int intermediate_dim,
                                  int n_experts,
                                  int top_k)
{
    if (!hidden || !indices || !routing_weights || !expert_up || !expert_down || !output ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 || n_experts <= 0 || top_k <= 0) {
        return;
    }

    const size_t out_count = (size_t)rows * (size_t)hidden_dim;
    for (size_t p = 0; p < out_count; ++p) output[p] = 0.0f;

    float pre[intermediate_dim];
    float gate[intermediate_dim];
    float up[intermediate_dim];
    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        float *y = output + (size_t)r * (size_t)hidden_dim;
        for (int slot = 0; slot < top_k; ++slot) {
            const int e = indices[(size_t)r * (size_t)top_k + (size_t)slot];
            if (e < 0 || e >= n_experts) continue;
            const float route_w = routing_weights[(size_t)r * (size_t)top_k + (size_t)slot];

            for (int i = 0; i < intermediate_dim; ++i) {
                float v = 0.0f;
                for (int h = 0; h < hidden_dim; ++h) {
                    v += expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] * x[h];
                }
                pre[i] = v;
                act[i] = (v > 0.0f) ? v * v : 0.0f;
            }

            for (int h = 0; h < hidden_dim; ++h) {
                float v = 0.0f;
                for (int i = 0; i < intermediate_dim; ++i) {
                    v += expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)] * act[i];
                }
                y[h] += route_w * v;
            }
        }
    }
}



static inline float ck_moe_sigmoid_f32(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float ck_moe_silu_f32(float x)
{
    return x * ck_moe_sigmoid_f32(x);
}

static inline float ck_moe_dsilu_f32(float x)
{
    const float sig = ck_moe_sigmoid_f32(x);
    return sig + x * sig * (1.0f - sig);
}

void moe_swiglu_expert_forward_f32(const float *hidden,
                                   const int *indices,
                                   const float *routing_weights,
                                   const float *expert_gate,
                                   const float *expert_up,
                                   const float *expert_down,
                                   float *output,
                                   int rows,
                                   int hidden_dim,
                                   int intermediate_dim,
                                   int n_experts,
                                   int top_k)
{
    if (!hidden || !indices || !routing_weights || !expert_gate || !expert_up || !expert_down || !output ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 || n_experts <= 0 || top_k <= 0) {
        return;
    }

    for (size_t p = 0; p < (size_t)rows * (size_t)hidden_dim; ++p) output[p] = 0.0f;

    float gate[intermediate_dim];
    float up[intermediate_dim];
    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        float *y = output + (size_t)r * (size_t)hidden_dim;
        for (int slot = 0; slot < top_k; ++slot) {
            const int e = indices[(size_t)r * (size_t)top_k + (size_t)slot];
            if (e < 0 || e >= n_experts) continue;
            const float route_w = routing_weights[(size_t)r * (size_t)top_k + (size_t)slot];

            for (int i = 0; i < intermediate_dim; ++i) {
                float gv = 0.0f;
                float uv = 0.0f;
                for (int h = 0; h < hidden_dim; ++h) {
                    gv += expert_gate[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] * x[h];
                    uv += expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] * x[h];
                }
                gate[i] = gv;
                up[i] = uv;
                act[i] = ck_moe_silu_f32(gv) * uv;
            }

            for (int h = 0; h < hidden_dim; ++h) {
                float v = 0.0f;
                for (int i = 0; i < intermediate_dim; ++i) {
                    v += expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)] * act[i];
                }
                y[h] += route_w * v;
            }
        }
    }
}

void moe_swiglu_expert_forward_bf16(const float *hidden,
                                    const int *indices,
                                    const float *routing_weights,
                                    const uint16_t *expert_gate,
                                    const uint16_t *expert_up,
                                    const uint16_t *expert_down,
                                    float *output,
                                    int rows,
                                    int hidden_dim,
                                    int intermediate_dim,
                                    int n_experts,
                                    int top_k)
{
    if (!hidden || !indices || !routing_weights || !expert_gate || !expert_up || !expert_down || !output ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 || n_experts <= 0 || top_k <= 0) {
        return;
    }

    for (size_t p = 0; p < (size_t)rows * (size_t)hidden_dim; ++p) output[p] = 0.0f;

    float gate[intermediate_dim];
    float up[intermediate_dim];
    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        float *y = output + (size_t)r * (size_t)hidden_dim;
        for (int slot = 0; slot < top_k; ++slot) {
            const int e = indices[(size_t)r * (size_t)top_k + (size_t)slot];
            if (e < 0 || e >= n_experts) continue;
            const float route_w = routing_weights[(size_t)r * (size_t)top_k + (size_t)slot];

            for (int i = 0; i < intermediate_dim; ++i) {
                float gv = 0.0f;
                float uv = 0.0f;
                for (int h = 0; h < hidden_dim; ++h) {
                    gv += bf16_to_float(expert_gate[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)]) * x[h];
                    uv += bf16_to_float(expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)]) * x[h];
                }
                gate[i] = gv;
                up[i] = uv;
                act[i] = ck_moe_silu_f32(gv) * uv;
            }

            for (int h = 0; h < hidden_dim; ++h) {
                float v = 0.0f;
                for (int i = 0; i < intermediate_dim; ++i) {
                    v += bf16_to_float(expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)]) * act[i];
                }
                y[h] += route_w * v;
            }
        }
    }
}

static size_t ck_moe_align64(size_t value)
{
    return (value + 63u) & ~(size_t)63u;
}

size_t moe_swiglu_expert_q4k_q5k_workspace_bytes(int hidden_dim,
                                                  int intermediate_dim)
{
    if (hidden_dim <= 0 || intermediate_dim <= 0 ||
        hidden_dim % 256 != 0 || intermediate_dim % 256 != 0) {
        return 0;
    }

    size_t bytes = ck_moe_align64(ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)hidden_dim));
    bytes += ck_moe_align64(2u * (size_t)intermediate_dim * sizeof(float));
    bytes += ck_moe_align64(ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)intermediate_dim));
    bytes += ck_moe_align64((size_t)hidden_dim * sizeof(float));
    return bytes;
}

int moe_swiglu_expert_forward_q4k_q5k_workspace(
    const float *hidden,
    const int *indices,
    const float *routing_weights,
    const void *expert_gate,
    const void *expert_up,
    const void *expert_down,
    float *output,
    int rows,
    int hidden_dim,
    int intermediate_dim,
    int n_experts,
    int top_k,
    void *workspace,
    size_t workspace_bytes)
{
    const size_t required = moe_swiglu_expert_q4k_q5k_workspace_bytes(
        hidden_dim, intermediate_dim);
    if (!hidden || !indices || !routing_weights || !expert_gate || !expert_up ||
        !expert_down || !output || !workspace || required == 0 ||
        workspace_bytes < required || rows <= 0 || n_experts <= 0 ||
        top_k <= 0 || top_k > n_experts) {
        return -1;
    }

    const size_t hidden_q8_bytes = ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)hidden_dim));
    const size_t gate_up_bytes = ck_moe_align64(
        2u * (size_t)intermediate_dim * sizeof(float));
    const size_t act_q8_bytes = ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)intermediate_dim));
    uint8_t *cursor = (uint8_t *)workspace;
    void *hidden_q8 = cursor;
    cursor += hidden_q8_bytes;
    float *gate_up = (float *)cursor;
    cursor += gate_up_bytes;
    void *act_q8 = cursor;
    cursor += act_q8_bytes;
    float *expert_output = (float *)cursor;

    const size_t q4_row_bytes = ck_dtype_row_bytes(CK_DT_Q4_K, (size_t)hidden_dim);
    const size_t q5_row_bytes = ck_dtype_row_bytes(CK_DT_Q5_K, (size_t)intermediate_dim);
    const uint8_t *gate_base = (const uint8_t *)expert_gate;
    const uint8_t *up_base = (const uint8_t *)expert_up;
    const uint8_t *down_base = (const uint8_t *)expert_down;

    memset(output, 0, (size_t)rows * (size_t)hidden_dim * sizeof(float));
    for (int row = 0; row < rows; ++row) {
        const float *x = hidden + (size_t)row * (size_t)hidden_dim;
        float *y = output + (size_t)row * (size_t)hidden_dim;
        quantize_row_q8_k(x, hidden_q8, hidden_dim);

        for (int slot = 0; slot < top_k; ++slot) {
            const size_t route_index = (size_t)row * (size_t)top_k + (size_t)slot;
            const int expert = indices[route_index];
            if (expert < 0 || expert >= n_experts) {
                return -2;
            }

            const size_t up_expert_offset =
                (size_t)expert * (size_t)intermediate_dim * q4_row_bytes;
            const size_t down_expert_offset =
                (size_t)expert * (size_t)hidden_dim * q5_row_bytes;
            gemv_q4_k_q8_k(gate_up,
                           gate_base + up_expert_offset,
                           hidden_q8,
                           intermediate_dim,
                           hidden_dim);
            gemv_q4_k_q8_k(gate_up + intermediate_dim,
                           up_base + up_expert_offset,
                           hidden_q8,
                           intermediate_dim,
                           hidden_dim);
            swiglu_forward_ggml(gate_up, gate_up, 1, intermediate_dim);
            quantize_row_q8_k(gate_up, act_q8, intermediate_dim);
            gemv_q5_k_q8_k(expert_output,
                           down_base + down_expert_offset,
                           act_q8,
                           hidden_dim,
                           intermediate_dim);

            const float route_weight = routing_weights[route_index];
            axpy_f32(y, expert_output, route_weight, hidden_dim);
        }
    }
    return 0;
}

void moe_swiglu_expert_backward_f32(const float *d_output,
                                    const float *hidden,
                                    const int *indices,
                                    const float *routing_weights,
                                    const float *expert_gate,
                                    const float *expert_up,
                                    const float *expert_down,
                                    float *d_hidden,
                                    float *d_routing_weights,
                                    float *d_expert_gate,
                                    float *d_expert_up,
                                    float *d_expert_down,
                                    int rows,
                                    int hidden_dim,
                                    int intermediate_dim,
                                    int n_experts,
                                    int top_k)
{
    if (!d_output || !hidden || !indices || !routing_weights || !expert_gate || !expert_up || !expert_down ||
        !d_hidden || !d_routing_weights || !d_expert_gate || !d_expert_up || !d_expert_down ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 || n_experts <= 0 || top_k <= 0) {
        return;
    }

    for (size_t p = 0; p < (size_t)rows * (size_t)hidden_dim; ++p) d_hidden[p] = 0.0f;
    for (size_t p = 0; p < (size_t)rows * (size_t)top_k; ++p) d_routing_weights[p] = 0.0f;
    for (size_t p = 0; p < (size_t)n_experts * (size_t)intermediate_dim * (size_t)hidden_dim; ++p) {
        d_expert_gate[p] = 0.0f;
        d_expert_up[p] = 0.0f;
    }
    for (size_t p = 0; p < (size_t)n_experts * (size_t)hidden_dim * (size_t)intermediate_dim; ++p) d_expert_down[p] = 0.0f;

    float gate[intermediate_dim];
    float up[intermediate_dim];
    float silu_gate[intermediate_dim];
    float act[intermediate_dim];
    float d_act[intermediate_dim];
    float expert_out[hidden_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        const float *dy = d_output + (size_t)r * (size_t)hidden_dim;
        float *dx = d_hidden + (size_t)r * (size_t)hidden_dim;

        for (int slot = 0; slot < top_k; ++slot) {
            const int e = indices[(size_t)r * (size_t)top_k + (size_t)slot];
            if (e < 0 || e >= n_experts) continue;
            const float route_w = routing_weights[(size_t)r * (size_t)top_k + (size_t)slot];

            for (int i = 0; i < intermediate_dim; ++i) {
                float gv = 0.0f;
                float uv = 0.0f;
                for (int h = 0; h < hidden_dim; ++h) {
                    gv += expert_gate[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] * x[h];
                    uv += expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] * x[h];
                }
                gate[i] = gv;
                up[i] = uv;
                silu_gate[i] = ck_moe_silu_f32(gv);
                act[i] = silu_gate[i] * uv;
                d_act[i] = 0.0f;
            }

            for (int h = 0; h < hidden_dim; ++h) {
                float v = 0.0f;
                for (int i = 0; i < intermediate_dim; ++i) {
                    v += expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)] * act[i];
                }
                expert_out[h] = v;
            }

            float d_route = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                const float d_expert_out = dy[h] * route_w;
                d_route += dy[h] * expert_out[h];
                for (int i = 0; i < intermediate_dim; ++i) {
                    d_expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)] += d_expert_out * act[i];
                    d_act[i] += d_expert_out * expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)];
                }
            }
            d_routing_weights[(size_t)r * (size_t)top_k + (size_t)slot] += d_route;

            for (int i = 0; i < intermediate_dim; ++i) {
                const float d_up = d_act[i] * silu_gate[i];
                const float d_gate = d_act[i] * up[i] * ck_moe_dsilu_f32(gate[i]);
                for (int h = 0; h < hidden_dim; ++h) {
                    d_expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] += d_up * x[h];
                    d_expert_gate[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] += d_gate * x[h];
                    dx[h] += d_up * expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] +
                             d_gate * expert_gate[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)];
                }
            }
        }
    }
}

void moe_swiglu_shared_forward_f32(const float *hidden,
                                   const float *routed,
                                   const float *shared_gate,
                                   const float *shared_up,
                                   const float *shared_down,
                                   float *output,
                                   int rows,
                                   int hidden_dim,
                                   int intermediate_dim)
{
    if (!hidden || !shared_gate || !shared_up || !shared_down || !output || rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0) {
        return;
    }

    float gate[intermediate_dim];
    float up[intermediate_dim];
    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        const float *route = routed ? (routed + (size_t)r * (size_t)hidden_dim) : NULL;
        float *y = output + (size_t)r * (size_t)hidden_dim;
        for (int i = 0; i < intermediate_dim; ++i) {
            float gv = 0.0f;
            float uv = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                gv += shared_gate[(size_t)i * (size_t)hidden_dim + (size_t)h] * x[h];
                uv += shared_up[(size_t)i * (size_t)hidden_dim + (size_t)h] * x[h];
            }
            act[i] = ck_moe_silu_f32(gv) * uv;
        }
        for (int h = 0; h < hidden_dim; ++h) {
            float v = route ? route[h] : 0.0f;
            for (int i = 0; i < intermediate_dim; ++i) {
                v += shared_down[(size_t)h * (size_t)intermediate_dim + (size_t)i] * act[i];
            }
            y[h] = v;
        }
    }
}

void moe_swiglu_shared_forward_bf16(const float *hidden,
                                    const float *routed,
                                    const uint16_t *shared_gate,
                                    const uint16_t *shared_up,
                                    const uint16_t *shared_down,
                                    float *output,
                                    int rows,
                                    int hidden_dim,
                                    int intermediate_dim)
{
    if (!hidden || !shared_gate || !shared_up || !shared_down || !output ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0) {
        return;
    }

    float gate[intermediate_dim];
    float up[intermediate_dim];
    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        const float *route = routed ? (routed + (size_t)r * (size_t)hidden_dim) : NULL;
        float *y = output + (size_t)r * (size_t)hidden_dim;
        for (int i = 0; i < intermediate_dim; ++i) {
            float gv = 0.0f;
            float uv = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                gv += bf16_to_float(shared_gate[(size_t)i * (size_t)hidden_dim + (size_t)h]) * x[h];
                uv += bf16_to_float(shared_up[(size_t)i * (size_t)hidden_dim + (size_t)h]) * x[h];
            }
            gate[i] = gv;
            up[i] = uv;
            act[i] = ck_moe_silu_f32(gv) * uv;
        }
        for (int h = 0; h < hidden_dim; ++h) {
            float v = route ? route[h] : 0.0f;
            for (int i = 0; i < intermediate_dim; ++i) {
                v += bf16_to_float(shared_down[(size_t)h * (size_t)intermediate_dim + (size_t)i]) * act[i];
            }
            y[h] = v;
        }
    }
}

/*
 * FarSkip shared-expert combine used by Instella-MoE.
 *
 * PyTorch constructs these values as:
 *   mlp_output        = routed + shared
 *   residual_no_route = post_attn_residual + shared
 *   main_output       = post_attn_residual + mlp_output
 *
 * Keep the shared down projection in its own ascending FP32 reduction and
 * preserve those two explicit addition boundaries.  Folding routed into the
 * down-projection accumulator changes the rounding contract.
 */
void farskip_swiglu_shared_combine_bf16(const float *hidden,
                                        const float *routed,
                                        const float *post_attn_residual,
                                        const uint16_t *shared_gate,
                                        const uint16_t *shared_up,
                                        const uint16_t *shared_down,
                                        float *main_output,
                                        float *routed_free_output,
                                        int rows,
                                        int hidden_dim,
                                        int intermediate_dim)
{
    if (!hidden || !routed || !post_attn_residual || !shared_gate || !shared_up ||
        !shared_down || !main_output || !routed_free_output || rows <= 0 ||
        hidden_dim <= 0 || intermediate_dim <= 0) {
        return;
    }

    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        const float *route = routed + (size_t)r * (size_t)hidden_dim;
        const float *residual = post_attn_residual + (size_t)r * (size_t)hidden_dim;
        float *main = main_output + (size_t)r * (size_t)hidden_dim;
        float *routed_free = routed_free_output + (size_t)r * (size_t)hidden_dim;

        for (int i = 0; i < intermediate_dim; ++i) {
            float gv = 0.0f;
            float uv = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                gv += bf16_to_float(shared_gate[(size_t)i * (size_t)hidden_dim + (size_t)h]) * x[h];
                uv += bf16_to_float(shared_up[(size_t)i * (size_t)hidden_dim + (size_t)h]) * x[h];
            }
            act[i] = ck_moe_silu_f32(gv) * uv;
        }

        for (int h = 0; h < hidden_dim; ++h) {
            float shared = 0.0f;
            for (int i = 0; i < intermediate_dim; ++i) {
                shared += bf16_to_float(shared_down[(size_t)h * (size_t)intermediate_dim + (size_t)i]) * act[i];
            }
            const float mlp_output = route[h] + shared;
            routed_free[h] = residual[h] + shared;
            main[h] = residual[h] + mlp_output;
        }
    }
}

void moe_swiglu_shared_backward_f32(const float *d_output,
                                    const float *hidden,
                                    const float *shared_gate,
                                    const float *shared_up,
                                    const float *shared_down,
                                    float *d_hidden,
                                    float *d_routed,
                                    float *d_shared_gate,
                                    float *d_shared_up,
                                    float *d_shared_down,
                                    int rows,
                                    int hidden_dim,
                                    int intermediate_dim)
{
    if (!d_output || !hidden || !shared_gate || !shared_up || !shared_down ||
        !d_hidden || !d_routed || !d_shared_gate || !d_shared_up || !d_shared_down ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0) {
        return;
    }

    for (size_t p = 0; p < (size_t)rows * (size_t)hidden_dim; ++p) {
        d_hidden[p] = 0.0f;
        d_routed[p] = d_output[p];
    }
    for (size_t p = 0; p < (size_t)intermediate_dim * (size_t)hidden_dim; ++p) {
        d_shared_gate[p] = 0.0f;
        d_shared_up[p] = 0.0f;
    }
    for (size_t p = 0; p < (size_t)hidden_dim * (size_t)intermediate_dim; ++p) d_shared_down[p] = 0.0f;

    float gate[intermediate_dim];
    float up[intermediate_dim];
    float silu_gate[intermediate_dim];
    float act[intermediate_dim];
    float d_act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        const float *dy = d_output + (size_t)r * (size_t)hidden_dim;
        float *dx = d_hidden + (size_t)r * (size_t)hidden_dim;

        for (int i = 0; i < intermediate_dim; ++i) {
            float gv = 0.0f;
            float uv = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                gv += shared_gate[(size_t)i * (size_t)hidden_dim + (size_t)h] * x[h];
                uv += shared_up[(size_t)i * (size_t)hidden_dim + (size_t)h] * x[h];
            }
            gate[i] = gv;
            up[i] = uv;
            silu_gate[i] = ck_moe_silu_f32(gv);
            act[i] = silu_gate[i] * uv;
            d_act[i] = 0.0f;
        }

        for (int h = 0; h < hidden_dim; ++h) {
            for (int i = 0; i < intermediate_dim; ++i) {
                d_shared_down[(size_t)h * (size_t)intermediate_dim + (size_t)i] += dy[h] * act[i];
                d_act[i] += dy[h] * shared_down[(size_t)h * (size_t)intermediate_dim + (size_t)i];
            }
        }

        for (int i = 0; i < intermediate_dim; ++i) {
            const float d_up = d_act[i] * silu_gate[i];
            const float d_gate = d_act[i] * up[i] * ck_moe_dsilu_f32(gate[i]);
            for (int h = 0; h < hidden_dim; ++h) {
                d_shared_up[(size_t)i * (size_t)hidden_dim + (size_t)h] += d_up * x[h];
                d_shared_gate[(size_t)i * (size_t)hidden_dim + (size_t)h] += d_gate * x[h];
                dx[h] += d_up * shared_up[(size_t)i * (size_t)hidden_dim + (size_t)h] +
                         d_gate * shared_gate[(size_t)i * (size_t)hidden_dim + (size_t)h];
            }
        }
    }
}

void moe_relu2_expert_forward_q5_0_q8_0(const float *hidden,
                                        const int *indices,
                                        const float *routing_weights,
                                        const void *expert_up,
                                        const void *expert_down,
                                        float *output,
                                        int rows,
                                        int hidden_dim,
                                        int intermediate_dim,
                                        int n_experts,
                                        int top_k)
{
    if (!hidden || !indices || !routing_weights || !expert_up || !expert_down || !output ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 || n_experts <= 0 || top_k <= 0) {
        return;
    }

    const size_t out_count = (size_t)rows * (size_t)hidden_dim;
    for (size_t p = 0; p < out_count; ++p) output[p] = 0.0f;

    const size_t up_row_bytes = ck_dtype_row_bytes(CK_DT_Q5_0, (size_t)hidden_dim);
    const size_t down_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)intermediate_dim);
    const uint8_t *up_base = (const uint8_t *)expert_up;
    const uint8_t *down_base = (const uint8_t *)expert_down;

    float up_row[hidden_dim];
    float down_row[intermediate_dim];
    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        float *y = output + (size_t)r * (size_t)hidden_dim;
        if (ck_moe_debug_enabled() && r == 0) {
            fprintf(stderr,
                    "[CK_DEBUG_MOE] routed_q5q8 rows=%d hidden=%d intermediate=%d experts=%d top_k=%d up_row_bytes=%zu down_row_bytes=%zu\n",
                    rows,
                    hidden_dim,
                    intermediate_dim,
                    n_experts,
                    top_k,
                    up_row_bytes,
                    down_row_bytes);
            fprintf(stderr, "[CK_DEBUG_MOE] routed slots:");
            for (int dbg_slot = 0; dbg_slot < top_k; ++dbg_slot) {
                fprintf(stderr,
                        " (%d,%g)",
                        indices[(size_t)r * (size_t)top_k + (size_t)dbg_slot],
                        routing_weights[(size_t)r * (size_t)top_k + (size_t)dbg_slot]);
            }
            fprintf(stderr, "\n");
            ck_moe_debug_finite("routed.hidden[0]", x, (size_t)hidden_dim);
            ck_moe_debug_finite("routed.hidden_all", hidden, (size_t)rows * (size_t)hidden_dim);
        }
        for (int slot = 0; slot < top_k; ++slot) {
            const int e = indices[(size_t)r * (size_t)top_k + (size_t)slot];
            if (e < 0 || e >= n_experts) continue;
            const float route_w = routing_weights[(size_t)r * (size_t)top_k + (size_t)slot];
            const uint8_t *expert_up_base = up_base + (size_t)e * (size_t)intermediate_dim * up_row_bytes;
            const uint8_t *expert_down_base = down_base + (size_t)e * (size_t)hidden_dim * down_row_bytes;

            for (int i = 0; i < intermediate_dim; ++i) {
                dequant_q5_0_row(expert_up_base + (size_t)i * up_row_bytes, up_row, (size_t)hidden_dim);
                float v = 0.0f;
                for (int h = 0; h < hidden_dim; ++h) v += up_row[h] * x[h];
                act[i] = (v > 0.0f) ? v * v : 0.0f;
            }

            for (int h = 0; h < hidden_dim; ++h) {
                dequant_q8_0_row(expert_down_base + (size_t)h * down_row_bytes, down_row, (size_t)intermediate_dim);
                float v = 0.0f;
                for (int i = 0; i < intermediate_dim; ++i) v += down_row[i] * act[i];
                y[h] += route_w * v;
            }
        }
        if (ck_moe_debug_enabled() && r == 0) {
            ck_moe_debug_finite("routed.output[0]", y, (size_t)hidden_dim);
            ck_moe_debug_finite("routed.output_all", output, (size_t)rows * (size_t)hidden_dim);
        }
    }
}


void moe_relu2_expert_forward_q5_0_q5_0(const float *hidden,
                                        const int *indices,
                                        const float *routing_weights,
                                        const void *expert_up,
                                        const void *expert_down,
                                        float *output,
                                        int rows,
                                        int hidden_dim,
                                        int intermediate_dim,
                                        int n_experts,
                                        int top_k)
{
    if (!hidden || !indices || !routing_weights || !expert_up || !expert_down || !output ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 || n_experts <= 0 || top_k <= 0) {
        return;
    }

    const size_t out_count = (size_t)rows * (size_t)hidden_dim;
    for (size_t p = 0; p < out_count; ++p) output[p] = 0.0f;

    const size_t up_row_bytes = ck_dtype_row_bytes(CK_DT_Q5_0, (size_t)hidden_dim);
    const size_t down_row_bytes = ck_dtype_row_bytes(CK_DT_Q5_0, (size_t)intermediate_dim);
    const uint8_t *up_base = (const uint8_t *)expert_up;
    const uint8_t *down_base = (const uint8_t *)expert_down;

    float up_row[hidden_dim];
    float down_row[intermediate_dim];
    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        float *y = output + (size_t)r * (size_t)hidden_dim;
        if (ck_moe_debug_enabled() && r == 0) {
            fprintf(stderr,
                    "[CK_DEBUG_MOE] routed_q5q5 rows=%d hidden=%d intermediate=%d experts=%d top_k=%d up_row_bytes=%zu down_row_bytes=%zu\n",
                    rows,
                    hidden_dim,
                    intermediate_dim,
                    n_experts,
                    top_k,
                    up_row_bytes,
                    down_row_bytes);
            fprintf(stderr, "[CK_DEBUG_MOE] routed slots:");
            for (int dbg_slot = 0; dbg_slot < top_k; ++dbg_slot) {
                fprintf(stderr,
                        " (%d,%g)",
                        indices[(size_t)r * (size_t)top_k + (size_t)dbg_slot],
                        routing_weights[(size_t)r * (size_t)top_k + (size_t)dbg_slot]);
            }
            fprintf(stderr, "\n");
            ck_moe_debug_finite("routed.hidden[0]", x, (size_t)hidden_dim);
            ck_moe_debug_finite("routed.hidden_all", hidden, (size_t)rows * (size_t)hidden_dim);
        }
        for (int slot = 0; slot < top_k; ++slot) {
            const int e = indices[(size_t)r * (size_t)top_k + (size_t)slot];
            if (e < 0 || e >= n_experts) continue;
            const float route_w = routing_weights[(size_t)r * (size_t)top_k + (size_t)slot];
            const uint8_t *expert_up_base = up_base + (size_t)e * (size_t)intermediate_dim * up_row_bytes;
            const uint8_t *expert_down_base = down_base + (size_t)e * (size_t)hidden_dim * down_row_bytes;

            for (int i = 0; i < intermediate_dim; ++i) {
                dequant_q5_0_row(expert_up_base + (size_t)i * up_row_bytes, up_row, (size_t)hidden_dim);
                float v = 0.0f;
                for (int h = 0; h < hidden_dim; ++h) v += up_row[h] * x[h];
                act[i] = (v > 0.0f) ? v * v : 0.0f;
            }

            for (int h = 0; h < hidden_dim; ++h) {
                dequant_q5_0_row(expert_down_base + (size_t)h * down_row_bytes, down_row, (size_t)intermediate_dim);
                float v = 0.0f;
                for (int i = 0; i < intermediate_dim; ++i) v += down_row[i] * act[i];
                y[h] += route_w * v;
            }
        }
        if (ck_moe_debug_enabled() && r == 0) {
            ck_moe_debug_finite("routed.output[0]", y, (size_t)hidden_dim);
            ck_moe_debug_finite("routed.output_all", output, (size_t)rows * (size_t)hidden_dim);
        }
    }
}

void moe_relu2_shared_forward_q5_1_q8_0(const float *hidden,
                                        const float *routed,
                                        const void *shared_up,
                                        const void *shared_down,
                                        float *output,
                                        int rows,
                                        int hidden_dim,
                                        int intermediate_dim)
{
    if (!hidden || !shared_up || !shared_down || !output || rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0) {
        return;
    }

    const size_t up_row_bytes = ck_dtype_row_bytes(CK_DT_Q5_1, (size_t)hidden_dim);
    const size_t down_row_bytes = ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)intermediate_dim);
    const uint8_t *up_base = (const uint8_t *)shared_up;
    const uint8_t *down_base = (const uint8_t *)shared_down;

    float up_row[hidden_dim];
    float down_row[intermediate_dim];
    float act[intermediate_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        const float *route = routed ? (routed + (size_t)r * (size_t)hidden_dim) : NULL;
        float *y = output + (size_t)r * (size_t)hidden_dim;
        float x_alias[hidden_dim];
        if (output == hidden) {
            memcpy(x_alias, x, (size_t)hidden_dim * sizeof(float));
            x = x_alias;
        }

        if (ck_moe_debug_enabled() && r == 0) {
            fprintf(stderr,
                    "[CK_DEBUG_MOE] shared_q5q8 rows=%d hidden=%d intermediate=%d up_row_bytes=%zu down_row_bytes=%zu alias=%d\n",
                    rows,
                    hidden_dim,
                    intermediate_dim,
                    up_row_bytes,
                    down_row_bytes,
                    output == hidden);
            ck_moe_debug_finite("shared.hidden[0]", x, (size_t)hidden_dim);
            ck_moe_debug_finite("shared.hidden_all", hidden, (size_t)rows * (size_t)hidden_dim);
            ck_moe_debug_finite("shared.routed[0]", route, (size_t)hidden_dim);
            if (routed) ck_moe_debug_finite("shared.routed_all", routed, (size_t)rows * (size_t)hidden_dim);
        }

        for (int i = 0; i < intermediate_dim; ++i) {
            dequant_q5_1_row(up_base + (size_t)i * up_row_bytes, up_row, (size_t)hidden_dim);
            float v = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) v += up_row[h] * x[h];
            act[i] = (v > 0.0f) ? v * v : 0.0f;
        }

        for (int h = 0; h < hidden_dim; ++h) {
            dequant_q8_0_row(down_base + (size_t)h * down_row_bytes, down_row, (size_t)intermediate_dim);
            float v = route ? route[h] : 0.0f;
            for (int i = 0; i < intermediate_dim; ++i) v += down_row[i] * act[i];
            y[h] = v;
        }

        if (ck_moe_debug_enabled() && r == 0) {
            ck_moe_debug_finite("shared.output[0]", y, (size_t)hidden_dim);
            ck_moe_debug_finite("shared.output_all", output, (size_t)rows * (size_t)hidden_dim);
        }
    }
}

void moe_relu2_expert_backward_f32(const float *d_output,
                                   const float *hidden,
                                   const int *indices,
                                   const float *routing_weights,
                                   const float *expert_up,
                                   const float *expert_down,
                                   float *d_hidden,
                                   float *d_routing_weights,
                                   float *d_expert_up,
                                   float *d_expert_down,
                                   int rows,
                                   int hidden_dim,
                                   int intermediate_dim,
                                   int n_experts,
                                   int top_k)
{
    if (!d_output || !hidden || !indices || !routing_weights || !expert_up || !expert_down ||
        !d_hidden || !d_routing_weights || !d_expert_up || !d_expert_down ||
        rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 || n_experts <= 0 || top_k <= 0) {
        return;
    }

    for (size_t p = 0; p < (size_t)rows * (size_t)hidden_dim; ++p) d_hidden[p] = 0.0f;
    for (size_t p = 0; p < (size_t)rows * (size_t)top_k; ++p) d_routing_weights[p] = 0.0f;
    for (size_t p = 0; p < (size_t)n_experts * (size_t)intermediate_dim * (size_t)hidden_dim; ++p) d_expert_up[p] = 0.0f;
    for (size_t p = 0; p < (size_t)n_experts * (size_t)hidden_dim * (size_t)intermediate_dim; ++p) d_expert_down[p] = 0.0f;

    float pre[intermediate_dim];
    float act[intermediate_dim];
    float d_act[intermediate_dim];
    float d_pre[intermediate_dim];
    float expert_out[hidden_dim];

    for (int r = 0; r < rows; ++r) {
        const float *x = hidden + (size_t)r * (size_t)hidden_dim;
        const float *dy = d_output + (size_t)r * (size_t)hidden_dim;
        float *dx = d_hidden + (size_t)r * (size_t)hidden_dim;

        for (int slot = 0; slot < top_k; ++slot) {
            const int e = indices[(size_t)r * (size_t)top_k + (size_t)slot];
            if (e < 0 || e >= n_experts) continue;
            const float route_w = routing_weights[(size_t)r * (size_t)top_k + (size_t)slot];

            for (int i = 0; i < intermediate_dim; ++i) {
                float v = 0.0f;
                for (int h = 0; h < hidden_dim; ++h) {
                    v += expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] * x[h];
                }
                pre[i] = v;
                act[i] = (v > 0.0f) ? v * v : 0.0f;
                d_act[i] = 0.0f;
            }

            for (int h = 0; h < hidden_dim; ++h) {
                float v = 0.0f;
                for (int i = 0; i < intermediate_dim; ++i) {
                    v += expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)] * act[i];
                }
                expert_out[h] = v;
            }

            float d_route = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                const float d_expert_out = dy[h] * route_w;
                d_route += dy[h] * expert_out[h];
                for (int i = 0; i < intermediate_dim; ++i) {
                    d_expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)] += d_expert_out * act[i];
                    d_act[i] += d_expert_out * expert_down[ck_moe_down_idx(e, h, i, hidden_dim, intermediate_dim)];
                }
            }
            d_routing_weights[(size_t)r * (size_t)top_k + (size_t)slot] += d_route;

            for (int i = 0; i < intermediate_dim; ++i) {
                d_pre[i] = (pre[i] > 0.0f) ? d_act[i] * 2.0f * pre[i] : 0.0f;
            }

            for (int i = 0; i < intermediate_dim; ++i) {
                const float dpi = d_pre[i];
                for (int h = 0; h < hidden_dim; ++h) {
                    d_expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)] += dpi * x[h];
                    dx[h] += dpi * expert_up[ck_moe_up_idx(e, i, h, intermediate_dim, hidden_dim)];
                }
            }
        }
    }
}
