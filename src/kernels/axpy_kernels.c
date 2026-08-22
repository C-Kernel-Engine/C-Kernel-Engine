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
#include <stdatomic.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ckernel_engine.h"
#include "ckernel_dtype.h"
#include "ck_threadpool.h"
#include "bf16_utils.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

void gemm_q4_k_q8_k_compact_rows4(float *output,
                                   int output_stride,
                                   const void *weights,
                                   const void *const input_rows[4],
                                   int rows,
                                   int output_dim,
                                   int input_dim);
void gemm_q5_k_q8_k_compact_rows4(float *output,
                                   int output_stride,
                                   const void *weights,
                                   const void *const input_rows[4],
                                   int rows,
                                   int output_dim,
                                   int input_dim);
void gemm_q4_k_q8_k_packed_vnni_x8_compact_order_rows4(
    float *output,
    const void *weights_packed,
    const void *input_q8,
    int rows,
    int output_dim,
    int input_dim);
size_t q4_k_packed_vnni_x8_block_size(void);

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

typedef struct {
    const float *hidden;
    const int *indices;
    const float *routing_weights;
    const void *expert_gate;
    const void *expert_up;
    const void *expert_down;
    float *output;
    int rows;
    int hidden_dim;
    int intermediate_dim;
    int n_experts;
    int top_k;
    uint8_t *workspace;
    size_t workspace_stride;
    int status[CK_THREADPOOL_MAX_THREADS];
} ck_moe_q4k_q5k_parallel_args_t;

static void ck_moe_q4k_q5k_parallel_work(int ith, int nth, void *opaque)
{
    ck_moe_q4k_q5k_parallel_args_t *args =
        (ck_moe_q4k_q5k_parallel_args_t *)opaque;
    const int begin = (args->rows * ith) / nth;
    const int end = (args->rows * (ith + 1)) / nth;
    if (begin >= end) {
        args->status[ith] = 0;
        return;
    }
    args->status[ith] = moe_swiglu_expert_forward_q4k_q5k_workspace(
        args->hidden + (size_t)begin * (size_t)args->hidden_dim,
        args->indices + (size_t)begin * (size_t)args->top_k,
        args->routing_weights + (size_t)begin * (size_t)args->top_k,
        args->expert_gate,
        args->expert_up,
        args->expert_down,
        args->output + (size_t)begin * (size_t)args->hidden_dim,
        end - begin,
        args->hidden_dim,
        args->intermediate_dim,
        args->n_experts,
        args->top_k,
        args->workspace + (size_t)ith * args->workspace_stride,
        args->workspace_stride);
}

typedef struct {
    const int *indices;
    const void *expert_gate;
    const void *expert_up;
    const void *expert_down;
    const void *hidden_q8;
    uint8_t *workspace;
    size_t workspace_stride;
    size_t hidden_q8_bytes;
    int hidden_dim;
    int intermediate_dim;
    int n_experts;
    int top_k;
    float *expert_output[CK_THREADPOOL_MAX_THREADS];
    int status[CK_THREADPOOL_MAX_THREADS];
} ck_moe_q4k_q5k_route_args_t;

static void ck_moe_q4k_q5k_route_work(int ith, int nth, void *opaque)
{
    ck_moe_q4k_q5k_route_args_t *args =
        (ck_moe_q4k_q5k_route_args_t *)opaque;
    if (ith >= nth || ith >= args->top_k) return;

    const int expert = args->indices[ith];
    if (expert < 0 || expert >= args->n_experts) {
        args->status[ith] = -2;
        return;
    }

    const size_t gate_up_bytes = ck_moe_align64(
        2u * (size_t)args->intermediate_dim * sizeof(float));
    const size_t act_q8_bytes = ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)args->intermediate_dim));
    uint8_t *cursor = args->workspace + (size_t)ith * args->workspace_stride;
    cursor += args->hidden_q8_bytes;
    float *gate_up = (float *)cursor;
    cursor += gate_up_bytes;
    void *act_q8 = cursor;
    cursor += act_q8_bytes;
    float *expert_output = (float *)cursor;
    args->expert_output[ith] = expert_output;

    const size_t q4_row_bytes = ck_dtype_row_bytes(
        CK_DT_Q4_K, (size_t)args->hidden_dim);
    const size_t q5_row_bytes = ck_dtype_row_bytes(
        CK_DT_Q5_K, (size_t)args->intermediate_dim);
    const size_t q4_expert_offset =
        (size_t)expert * (size_t)args->intermediate_dim * q4_row_bytes;
    const size_t q5_expert_offset =
        (size_t)expert * (size_t)args->hidden_dim * q5_row_bytes;

    gemv_q4_k_q8_k(
        gate_up,
        (const uint8_t *)args->expert_gate + q4_expert_offset,
        args->hidden_q8, args->intermediate_dim, args->hidden_dim);
    gemv_q4_k_q8_k(
        gate_up + args->intermediate_dim,
        (const uint8_t *)args->expert_up + q4_expert_offset,
        args->hidden_q8, args->intermediate_dim, args->hidden_dim);
    swiglu_forward_ggml(gate_up, gate_up, 1, args->intermediate_dim);
    quantize_row_q8_k(gate_up, act_q8, args->intermediate_dim);
    gemv_q5_k_q8_k(
        expert_output,
        (const uint8_t *)args->expert_down + q5_expert_offset,
        act_q8, args->hidden_dim, args->intermediate_dim);
    args->status[ith] = 0;
}

static int ck_moe_q4k_q5k_route_parallel(
    const float *hidden,
    const int *indices,
    const float *routing_weights,
    const void *expert_gate,
    const void *expert_up,
    const void *expert_down,
    float *output,
    int hidden_dim,
    int intermediate_dim,
    int n_experts,
    int top_k,
    void *workspace,
    size_t workspace_bytes,
    size_t workspace_stride,
    ck_threadpool_t *pool)
{
    if (!pool || top_k <= 1 || top_k > CK_THREADPOOL_MAX_THREADS ||
        ck_threadpool_n_threads(pool) < top_k ||
        workspace_bytes < workspace_stride * (size_t)top_k) {
        return 1;
    }

    const size_t hidden_q8_bytes = ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)hidden_dim));
    void *hidden_q8 = workspace;
    quantize_row_q8_k(hidden, hidden_q8, hidden_dim);

    ck_moe_q4k_q5k_route_args_t args = {
        .indices = indices,
        .expert_gate = expert_gate,
        .expert_up = expert_up,
        .expert_down = expert_down,
        .hidden_q8 = hidden_q8,
        .workspace = (uint8_t *)workspace,
        .workspace_stride = workspace_stride,
        .hidden_q8_bytes = hidden_q8_bytes,
        .hidden_dim = hidden_dim,
        .intermediate_dim = intermediate_dim,
        .n_experts = n_experts,
        .top_k = top_k,
        .expert_output = {0},
        .status = {0},
    };
    ck_threadpool_dispatch_n(
        pool, top_k, ck_moe_q4k_q5k_route_work, &args);

    memset(output, 0, (size_t)hidden_dim * sizeof(float));
    for (int slot = 0; slot < top_k; ++slot) {
        if (args.status[slot] != 0 || !args.expert_output[slot]) {
            return args.status[slot] != 0 ? args.status[slot] : -1;
        }
        axpy_f32(
            output, args.expert_output[slot], routing_weights[slot], hidden_dim);
    }
    return 0;
}

int moe_swiglu_expert_forward_q4k_q5k_parallel_workspace(
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
    const size_t stride = moe_swiglu_expert_q4k_q5k_workspace_bytes(
        hidden_dim, intermediate_dim);
    if (!hidden || !indices || !routing_weights || !expert_gate || !expert_up ||
        !expert_down || !output || !workspace || stride == 0 || rows <= 0 ||
        n_experts <= 0 || top_k <= 0 || top_k > n_experts) {
        return -1;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    if (rows == 1) {
        const int route_status = ck_moe_q4k_q5k_route_parallel(
            hidden, indices, routing_weights,
            expert_gate, expert_up, expert_down, output,
            hidden_dim, intermediate_dim, n_experts, top_k,
            workspace, workspace_bytes, stride, pool);
        if (route_status <= 0) return route_status;
    }

    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > rows) active = rows;
    if (active > CK_THREADPOOL_MAX_THREADS) active = CK_THREADPOOL_MAX_THREADS;
    if (workspace_bytes < stride * (size_t)active) return -1;
    if (active <= 1) {
        return moe_swiglu_expert_forward_q4k_q5k_workspace(
            hidden, indices, routing_weights, expert_gate, expert_up, expert_down,
            output, rows, hidden_dim, intermediate_dim, n_experts, top_k,
            workspace, stride);
    }

    ck_moe_q4k_q5k_parallel_args_t args = {
        .hidden = hidden,
        .indices = indices,
        .routing_weights = routing_weights,
        .expert_gate = expert_gate,
        .expert_up = expert_up,
        .expert_down = expert_down,
        .output = output,
        .rows = rows,
        .hidden_dim = hidden_dim,
        .intermediate_dim = intermediate_dim,
        .n_experts = n_experts,
        .top_k = top_k,
        .workspace = (uint8_t *)workspace,
        .workspace_stride = stride,
        .status = {0},
    };
    ck_threadpool_dispatch_n(pool, active, ck_moe_q4k_q5k_parallel_work, &args);
    for (int ith = 0; ith < active; ++ith) {
        if (args.status[ith] != 0) return args.status[ith];
    }
    return 0;
}

typedef struct {
    size_t hidden_q8_offset;
    size_t route_rows_offset;
    size_t slot_offsets_offset;
    size_t counts_offset;
    size_t cursors_offset;
    size_t workers_offset;
    size_t hidden_q8_row_bytes;
    size_t worker_stride;
    size_t total_bytes;
} ck_moe_q4k_q5k_bucket_layout_t;

static int ck_moe_size_add(size_t a, size_t b, size_t *result)
{
    if (!result || a > SIZE_MAX - b) return -1;
    *result = a + b;
    return 0;
}

static int ck_moe_size_mul(size_t a, size_t b, size_t *result)
{
    if (!result || (a != 0 && b > SIZE_MAX / a)) return -1;
    *result = a * b;
    return 0;
}

static int ck_moe_bucket_layout(int rows,
                                int hidden_dim,
                                int intermediate_dim,
                                int n_experts,
                                int top_k,
                                ck_moe_q4k_q5k_bucket_layout_t *layout)
{
    if (!layout || rows <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 ||
        n_experts <= 0 || top_k <= 0 || top_k > n_experts ||
        hidden_dim % 256 != 0 || intermediate_dim % 256 != 0) {
        return -1;
    }

    memset(layout, 0, sizeof(*layout));
    layout->hidden_q8_row_bytes = ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)hidden_dim));

    size_t gate_up_bytes = 0;
    size_t worker_bytes = 0;
    if (ck_moe_size_mul(8u * sizeof(float), (size_t)intermediate_dim,
                        &gate_up_bytes) != 0) {
        return -1;
    }
    worker_bytes = ck_moe_align64(gate_up_bytes);
    if (ck_moe_size_add(
            worker_bytes,
            ck_moe_align64(4u * ck_dtype_row_bytes(
                CK_DT_Q8_K, (size_t)hidden_dim)),
            &worker_bytes) != 0 ||
        ck_moe_size_add(
            worker_bytes,
            ck_moe_align64(4u * ck_dtype_row_bytes(
                CK_DT_Q8_K, (size_t)intermediate_dim)),
            &worker_bytes) != 0 ||
        ck_moe_size_add(
            worker_bytes,
            ck_moe_align64(4u * (size_t)hidden_dim * sizeof(float)),
            &worker_bytes) != 0) {
        return -1;
    }
    layout->worker_stride = worker_bytes;

    size_t cursor = 0;
    size_t bytes = 0;
    layout->hidden_q8_offset = cursor;
    if (ck_moe_size_mul((size_t)rows, layout->hidden_q8_row_bytes, &bytes) != 0 ||
        ck_moe_size_add(cursor, ck_moe_align64(bytes), &cursor) != 0) {
        return -1;
    }

    layout->route_rows_offset = cursor;
    if (ck_moe_size_mul((size_t)rows, (size_t)top_k, &bytes) != 0 ||
        ck_moe_size_mul(bytes, sizeof(int), &bytes) != 0 ||
        ck_moe_size_add(cursor, ck_moe_align64(bytes), &cursor) != 0) {
        return -1;
    }

    layout->slot_offsets_offset = cursor;
    if (ck_moe_size_mul((size_t)top_k, (size_t)n_experts + 1u, &bytes) != 0 ||
        ck_moe_size_mul(bytes, sizeof(int), &bytes) != 0 ||
        ck_moe_size_add(cursor, ck_moe_align64(bytes), &cursor) != 0) {
        return -1;
    }

    layout->counts_offset = cursor;
    if (ck_moe_size_mul((size_t)n_experts, sizeof(int), &bytes) != 0 ||
        ck_moe_size_add(cursor, ck_moe_align64(bytes), &cursor) != 0) {
        return -1;
    }

    layout->cursors_offset = cursor;
    if (ck_moe_size_mul((size_t)n_experts, sizeof(int), &bytes) != 0 ||
        ck_moe_size_add(cursor, ck_moe_align64(bytes), &cursor) != 0) {
        return -1;
    }

    layout->workers_offset = cursor;
    if (ck_moe_size_mul((size_t)CK_THREADPOOL_MAX_THREADS,
                        layout->worker_stride, &bytes) != 0 ||
        ck_moe_size_add(cursor, ck_moe_align64(bytes), &cursor) != 0) {
        return -1;
    }
    layout->total_bytes = cursor;
    return 0;
}

size_t moe_swiglu_expert_q4k_q5k_bucketed_workspace_bytes(
    int rows,
    int hidden_dim,
    int intermediate_dim,
    int n_experts,
    int top_k)
{
    ck_moe_q4k_q5k_bucket_layout_t layout;
    if (ck_moe_bucket_layout(rows, hidden_dim, intermediate_dim, n_experts,
                             top_k, &layout) != 0) {
        return 0;
    }
    return layout.total_bytes;
}

typedef struct {
    const float *hidden;
    uint8_t *hidden_q8;
    int rows;
    int hidden_dim;
    size_t hidden_q8_row_bytes;
} ck_moe_q4k_q5k_quantize_args_t;

static void ck_moe_q4k_q5k_quantize_work(int ith, int nth, void *opaque)
{
    ck_moe_q4k_q5k_quantize_args_t *args =
        (ck_moe_q4k_q5k_quantize_args_t *)opaque;
    const int begin = (args->rows * ith) / nth;
    const int end = (args->rows * (ith + 1)) / nth;
    for (int row = begin; row < end; ++row) {
        quantize_row_q8_k(
            args->hidden + (size_t)row * (size_t)args->hidden_dim,
            args->hidden_q8 + (size_t)row * args->hidden_q8_row_bytes,
            args->hidden_dim);
    }
}

typedef struct {
    const int *bucket_rows;
    const int *bucket_offsets;
    const float *routing_weights;
    const uint8_t *hidden_q8;
    const uint8_t *gate_base;
    const uint8_t *up_base;
    const uint8_t *gate_packed_base;
    const uint8_t *up_packed_base;
    const uint8_t *down_base;
    float *output;
    uint8_t *workers;
    size_t worker_stride;
    size_t hidden_q8_row_bytes;
    size_t q4_expert_stride;
    size_t q4_packed_expert_stride;
    size_t q5_expert_stride;
    int hidden_dim;
    int intermediate_dim;
    int n_experts;
    int top_k;
    int slot;
    int total_tasks;
    atomic_int next_task;
} ck_moe_q4k_q5k_bucket_work_t;

enum { CK_MOE_Q4K_Q5K_TASK_ROWS = 16 };

static int ck_moe_bucket_expert_for_position(const int *offsets,
                                             int n_experts,
                                             int position)
{
    int lo = 0;
    int hi = n_experts;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (offsets[mid + 1] <= position) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static void ck_moe_q4k_q5k_bucket_work(int ith, int nth, void *opaque)
{
    (void)nth;
    ck_moe_q4k_q5k_bucket_work_t *args =
        (ck_moe_q4k_q5k_bucket_work_t *)opaque;
    uint8_t *cursor = args->workers + (size_t)ith * args->worker_stride;
    float *gate_up = (float *)cursor;
    cursor += ck_moe_align64(
        8u * (size_t)args->intermediate_dim * sizeof(float));
    uint8_t *hidden_q8_batch = cursor;
    const size_t hidden_q8_batch_row_bytes = ck_dtype_row_bytes(
        CK_DT_Q8_K, (size_t)args->hidden_dim);
    cursor += ck_moe_align64(4u * hidden_q8_batch_row_bytes);
    void *act_q8 = cursor;
    cursor += ck_moe_align64(
        4u * ck_dtype_row_bytes(CK_DT_Q8_K, (size_t)args->intermediate_dim));
    float *expert_output = (float *)cursor;
    const size_t act_q8_row_bytes = ck_dtype_row_bytes(
        CK_DT_Q8_K, (size_t)args->intermediate_dim);

    for (;;) {
        const int task = atomic_fetch_add_explicit(
            &args->next_task, 1, memory_order_relaxed);
        if (task >= args->total_tasks) break;

        int position = task * CK_MOE_Q4K_Q5K_TASK_ROWS;
        const int task_end = position + CK_MOE_Q4K_Q5K_TASK_ROWS <
                args->bucket_offsets[args->n_experts]
            ? position + CK_MOE_Q4K_Q5K_TASK_ROWS
            : args->bucket_offsets[args->n_experts];
        int expert = ck_moe_bucket_expert_for_position(
            args->bucket_offsets, args->n_experts, position);

        while (position < task_end && expert < args->n_experts) {
            const int expert_end = args->bucket_offsets[expert + 1];
            const int segment_end = expert_end < task_end
                ? expert_end : task_end;
            const uint8_t *gate = args->gate_base +
                (size_t)expert * args->q4_expert_stride;
            const uint8_t *up = args->up_base +
                (size_t)expert * args->q4_expert_stride;
            const uint8_t *gate_packed = args->gate_packed_base
                ? args->gate_packed_base +
                    (size_t)expert * args->q4_packed_expert_stride
                : NULL;
            const uint8_t *up_packed = args->up_packed_base
                ? args->up_packed_base +
                    (size_t)expert * args->q4_packed_expert_stride
                : NULL;
            const uint8_t *down = args->down_base +
                (size_t)expert * args->q5_expert_stride;

            for (int i = position; i < segment_end; i += 4) {
                const int batch_rows = segment_end - i < 4
                    ? segment_end - i : 4;
                const void *hidden_rows[4] = {NULL, NULL, NULL, NULL};
                const void *activation_rows[4] = {NULL, NULL, NULL, NULL};
                int output_rows[4] = {0, 0, 0, 0};
                for (int batch_row = 0; batch_row < batch_rows; ++batch_row) {
                    const int row = args->bucket_rows[i + batch_row];
                    output_rows[batch_row] = row;
                    hidden_rows[batch_row] = args->hidden_q8 +
                        (size_t)row * args->hidden_q8_row_bytes;
                    if (gate_packed && up_packed) {
                        memcpy(
                            hidden_q8_batch +
                                (size_t)batch_row * hidden_q8_batch_row_bytes,
                            hidden_rows[batch_row],
                            args->hidden_q8_row_bytes);
                    }
                }
                for (int batch_row = batch_rows; batch_row < 4; ++batch_row) {
                    hidden_rows[batch_row] = hidden_rows[0];
                }

                if (gate_packed && up_packed) {
                    float *up_rows = gate_up +
                        4u * (size_t)args->intermediate_dim;
                    gemm_q4_k_q8_k_packed_vnni_x8_compact_order_rows4(
                        gate_up, gate_packed, hidden_q8_batch,
                        batch_rows, args->intermediate_dim,
                        args->hidden_dim);
                    gemm_q4_k_q8_k_packed_vnni_x8_compact_order_rows4(
                        up_rows, up_packed, hidden_q8_batch,
                        batch_rows, args->intermediate_dim,
                        args->hidden_dim);
                    swiglu_forward_ggml_split(
                        gate_up, up_rows, gate_up, batch_rows,
                        args->intermediate_dim);
                } else {
                    const int gate_up_stride = 2 * args->intermediate_dim;
                    gemm_q4_k_q8_k_compact_rows4(
                        gate_up, gate_up_stride, gate, hidden_rows, batch_rows,
                        args->intermediate_dim, args->hidden_dim);
                    gemm_q4_k_q8_k_compact_rows4(
                        gate_up + args->intermediate_dim, gate_up_stride, up,
                        hidden_rows, batch_rows,
                        args->intermediate_dim, args->hidden_dim);
                    swiglu_forward_ggml(
                        gate_up, gate_up, batch_rows, args->intermediate_dim);
                }
                for (int batch_row = 0; batch_row < batch_rows; ++batch_row) {
                    void *activation = (uint8_t *)act_q8 +
                        (size_t)batch_row * act_q8_row_bytes;
                    quantize_row_q8_k(
                        gate_up + (size_t)batch_row *
                            (size_t)args->intermediate_dim,
                        activation, args->intermediate_dim);
                    activation_rows[batch_row] = activation;
                }
                for (int batch_row = batch_rows; batch_row < 4; ++batch_row) {
                    activation_rows[batch_row] = activation_rows[0];
                }
                gemm_q5_k_q8_k_compact_rows4(
                    expert_output, args->hidden_dim, down, activation_rows,
                    batch_rows, args->hidden_dim, args->intermediate_dim);

                for (int batch_row = 0; batch_row < batch_rows; ++batch_row) {
                    const int row = output_rows[batch_row];
                    const size_t route_index = (size_t)row *
                        (size_t)args->top_k + (size_t)args->slot;
                    axpy_f32(
                        args->output + (size_t)row *
                            (size_t)args->hidden_dim,
                        expert_output + (size_t)batch_row *
                            (size_t)args->hidden_dim,
                        args->routing_weights[route_index], args->hidden_dim);
                }
            }
            position = segment_end;
            ++expert;
        }
    }
}

static int ck_moe_swiglu_expert_forward_q4k_q5k_bucketed_impl(
    const float *hidden,
    const int *indices,
    const float *routing_weights,
    const void *expert_gate,
    const void *expert_up,
    const void *expert_down,
    const void *expert_gate_packed,
    const void *expert_up_packed,
    float *output,
    int rows,
    int hidden_dim,
    int intermediate_dim,
    int n_experts,
    int top_k,
    void *workspace,
    size_t workspace_bytes)
{
    ck_moe_q4k_q5k_bucket_layout_t layout;
    if (!hidden || !indices || !routing_weights || !expert_gate || !expert_up ||
        !expert_down || !output || !workspace ||
        ((expert_gate_packed == NULL) != (expert_up_packed == NULL)) ||
        ck_moe_bucket_layout(rows, hidden_dim, intermediate_dim, n_experts,
                             top_k, &layout) != 0 ||
        workspace_bytes < layout.total_bytes) {
        return -1;
    }

    uint8_t *base = (uint8_t *)workspace;
    uint8_t *hidden_q8 = base + layout.hidden_q8_offset;
    int *route_rows = (int *)(base + layout.route_rows_offset);
    int *slot_offsets = (int *)(base + layout.slot_offsets_offset);
    int *counts = (int *)(base + layout.counts_offset);
    int *cursors = (int *)(base + layout.cursors_offset);
    uint8_t *workers = base + layout.workers_offset;

    for (int slot = 0; slot < top_k; ++slot) {
        memset(counts, 0, (size_t)n_experts * sizeof(*counts));
        for (int row = 0; row < rows; ++row) {
            const int expert = indices[(size_t)row * (size_t)top_k +
                                       (size_t)slot];
            if (expert < 0 || expert >= n_experts) return -2;
            counts[expert] += 1;
        }

        int *offsets = slot_offsets + (size_t)slot * ((size_t)n_experts + 1u);
        offsets[0] = 0;
        for (int expert = 0; expert < n_experts; ++expert) {
            offsets[expert + 1] = offsets[expert] + counts[expert];
            cursors[expert] = offsets[expert];
        }
        int *rows_for_slot = route_rows + (size_t)slot * (size_t)rows;
        for (int row = 0; row < rows; ++row) {
            const int expert = indices[(size_t)row * (size_t)top_k +
                                       (size_t)slot];
            rows_for_slot[cursors[expert]++] = row;
        }
    }

    memset(output, 0, (size_t)rows * (size_t)hidden_dim * sizeof(float));
    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > rows) active = rows;
    if (active > CK_THREADPOOL_MAX_THREADS) active = CK_THREADPOOL_MAX_THREADS;
    if (active < 1) active = 1;

    ck_moe_q4k_q5k_quantize_args_t quantize_args = {
        .hidden = hidden,
        .hidden_q8 = hidden_q8,
        .rows = rows,
        .hidden_dim = hidden_dim,
        .hidden_q8_row_bytes = layout.hidden_q8_row_bytes,
    };
    if (active > 1 && pool) {
        ck_threadpool_dispatch_n(
            pool, active, ck_moe_q4k_q5k_quantize_work, &quantize_args);
    } else {
        ck_moe_q4k_q5k_quantize_work(0, 1, &quantize_args);
    }

    const size_t q4_expert_stride = (size_t)intermediate_dim *
        ck_dtype_row_bytes(CK_DT_Q4_K, (size_t)hidden_dim);
    const size_t q4_packed_expert_stride =
        (size_t)((intermediate_dim + 7) / 8) *
        (size_t)(hidden_dim / 256) * q4_k_packed_vnni_x8_block_size();
    const size_t q5_expert_stride = (size_t)hidden_dim *
        ck_dtype_row_bytes(CK_DT_Q5_K, (size_t)intermediate_dim);
    for (int slot = 0; slot < top_k; ++slot) {
        ck_moe_q4k_q5k_bucket_work_t args = {
            .bucket_rows = route_rows + (size_t)slot * (size_t)rows,
            .bucket_offsets = slot_offsets +
                (size_t)slot * ((size_t)n_experts + 1u),
            .routing_weights = routing_weights,
            .hidden_q8 = hidden_q8,
            .gate_base = (const uint8_t *)expert_gate,
            .up_base = (const uint8_t *)expert_up,
            .gate_packed_base = (const uint8_t *)expert_gate_packed,
            .up_packed_base = (const uint8_t *)expert_up_packed,
            .down_base = (const uint8_t *)expert_down,
            .output = output,
            .workers = workers,
            .worker_stride = layout.worker_stride,
            .hidden_q8_row_bytes = layout.hidden_q8_row_bytes,
            .q4_expert_stride = q4_expert_stride,
            .q4_packed_expert_stride = q4_packed_expert_stride,
            .q5_expert_stride = q5_expert_stride,
            .hidden_dim = hidden_dim,
            .intermediate_dim = intermediate_dim,
            .n_experts = n_experts,
            .top_k = top_k,
            .slot = slot,
            .total_tasks = (rows + CK_MOE_Q4K_Q5K_TASK_ROWS - 1) /
                CK_MOE_Q4K_Q5K_TASK_ROWS,
        };
        atomic_init(&args.next_task, 0);
        int task_threads = active;
        if (task_threads > args.total_tasks) task_threads = args.total_tasks;
        if (active > 1 && pool) {
            ck_threadpool_dispatch_n(
                pool, task_threads, ck_moe_q4k_q5k_bucket_work, &args);
        } else {
            ck_moe_q4k_q5k_bucket_work(0, 1, &args);
        }
    }
    return 0;
}

int moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace(
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
    return ck_moe_swiglu_expert_forward_q4k_q5k_bucketed_impl(
        hidden, indices, routing_weights,
        expert_gate, expert_up, expert_down, NULL, NULL, output,
        rows, hidden_dim, intermediate_dim, n_experts, top_k,
        workspace, workspace_bytes);
}

int moe_swiglu_expert_forward_q4k_q5k_bucketed_prepared_workspace(
    const float *hidden,
    const int *indices,
    const float *routing_weights,
    const void *expert_gate,
    const void *expert_up,
    const void *expert_down,
    const void *expert_gate_packed,
    const void *expert_up_packed,
    float *output,
    int rows,
    int hidden_dim,
    int intermediate_dim,
    int n_experts,
    int top_k,
    void *workspace,
    size_t workspace_bytes)
{
    return ck_moe_swiglu_expert_forward_q4k_q5k_bucketed_impl(
        hidden, indices, routing_weights,
        expert_gate, expert_up, expert_down,
        expert_gate_packed, expert_up_packed, output,
        rows, hidden_dim, intermediate_dim, n_experts, top_k,
        workspace, workspace_bytes);
}

int moe_swiglu_expert_forward_q4k_q5k_auto_workspace(
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
    if (rows < 512) {
        return moe_swiglu_expert_forward_q4k_q5k_parallel_workspace(
            hidden, indices, routing_weights,
            expert_gate, expert_up, expert_down, output,
            rows, hidden_dim, intermediate_dim, n_experts, top_k,
            workspace, workspace_bytes);
    }
    return moe_swiglu_expert_forward_q4k_q5k_bucketed_workspace(
        hidden, indices, routing_weights,
        expert_gate, expert_up, expert_down, output,
        rows, hidden_dim, intermediate_dim, n_experts, top_k,
        workspace, workspace_bytes);
}

size_t moe_swiglu_shared_q8_0_gated_workspace_bytes(int hidden_dim,
                                                    int intermediate_dim)
{
    if (hidden_dim <= 0 || intermediate_dim <= 0 ||
        hidden_dim % 32 != 0 || intermediate_dim % 32 != 0) {
        return 0;
    }

    size_t bytes = ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)hidden_dim));
    bytes += ck_moe_align64(2u * (size_t)intermediate_dim * sizeof(float));
    bytes += ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)intermediate_dim));
    bytes += ck_moe_align64((size_t)hidden_dim * sizeof(float));
    return bytes;
}

int moe_swiglu_shared_forward_q8_0_gated_workspace(
    const float *hidden,
    const float *routed,
    const void *shared_gate,
    const void *shared_up,
    const void *shared_down,
    const float *shared_gate_input,
    float *output,
    int rows,
    int hidden_dim,
    int intermediate_dim,
    void *workspace,
    size_t workspace_bytes)
{
    const size_t required = moe_swiglu_shared_q8_0_gated_workspace_bytes(
        hidden_dim, intermediate_dim);
    if (!hidden || !shared_gate || !shared_up || !shared_down ||
        !shared_gate_input || !output || !workspace || required == 0 ||
        workspace_bytes < required || rows <= 0) {
        return -1;
    }

    const size_t hidden_q8_bytes = ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)hidden_dim));
    const size_t gate_up_bytes = ck_moe_align64(
        2u * (size_t)intermediate_dim * sizeof(float));
    const size_t activation_q8_bytes = ck_moe_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)intermediate_dim));
    uint8_t *cursor = (uint8_t *)workspace;
    void *hidden_q8 = cursor;
    cursor += hidden_q8_bytes;
    float *gate_up = (float *)cursor;
    cursor += gate_up_bytes;
    void *activation_q8 = cursor;
    cursor += activation_q8_bytes;
    float *shared_output = (float *)cursor;

    for (int row = 0; row < rows; ++row) {
        const float *x = hidden + (size_t)row * (size_t)hidden_dim;
        const float *routed_row = routed
            ? routed + (size_t)row * (size_t)hidden_dim
            : NULL;
        float *output_row = output + (size_t)row * (size_t)hidden_dim;

        quantize_row_q8_0(x, hidden_q8, hidden_dim);
        gemv_q8_0_q8_0(gate_up, shared_gate, hidden_q8,
                       intermediate_dim, hidden_dim);
        gemv_q8_0_q8_0(gate_up + intermediate_dim, shared_up, hidden_q8,
                       intermediate_dim, hidden_dim);
        swiglu_forward_ggml(gate_up, gate_up, 1, intermediate_dim);
        quantize_row_q8_0(gate_up, activation_q8, intermediate_dim);
        gemv_q8_0_q8_0(shared_output, shared_down, activation_q8,
                       hidden_dim, intermediate_dim);

        float gate_value = 0.0f;
        gemm_nt_f32_llama_production(
            x, shared_gate_input, NULL, &gate_value, 1, 1, hidden_dim);
        const float gate_scale = 1.0f / (1.0f + expf(-gate_value));
        for (int h = 0; h < hidden_dim; ++h) {
            const float routed_value = routed_row ? routed_row[h] : 0.0f;
            volatile float gated_shared = shared_output[h] * gate_scale;
            output_row[h] = routed_value + gated_shared;
        }
    }
    return 0;
}

typedef struct {
    const float *hidden;
    const float *routed;
    const void *shared_gate;
    const void *shared_up;
    const void *shared_down;
    const float *shared_gate_input;
    float *output;
    int rows;
    int hidden_dim;
    int intermediate_dim;
    uint8_t *workspace;
    size_t workspace_stride;
    int status[CK_THREADPOOL_MAX_THREADS];
} ck_moe_shared_q8_0_parallel_args_t;

static void ck_moe_shared_q8_0_parallel_work(int ith, int nth, void *opaque)
{
    ck_moe_shared_q8_0_parallel_args_t *args =
        (ck_moe_shared_q8_0_parallel_args_t *)opaque;
    const int begin = (args->rows * ith) / nth;
    const int end = (args->rows * (ith + 1)) / nth;
    if (begin >= end) {
        args->status[ith] = 0;
        return;
    }

    args->status[ith] = moe_swiglu_shared_forward_q8_0_gated_workspace(
        args->hidden + (size_t)begin * (size_t)args->hidden_dim,
        args->routed
            ? args->routed + (size_t)begin * (size_t)args->hidden_dim
            : NULL,
        args->shared_gate,
        args->shared_up,
        args->shared_down,
        args->shared_gate_input,
        args->output + (size_t)begin * (size_t)args->hidden_dim,
        end - begin,
        args->hidden_dim,
        args->intermediate_dim,
        args->workspace + (size_t)ith * args->workspace_stride,
        args->workspace_stride);
}

int moe_swiglu_shared_forward_q8_0_gated_parallel_workspace(
    const float *hidden,
    const float *routed,
    const void *shared_gate,
    const void *shared_up,
    const void *shared_down,
    const float *shared_gate_input,
    float *output,
    int rows,
    int hidden_dim,
    int intermediate_dim,
    void *workspace,
    size_t workspace_bytes)
{
    const size_t stride = moe_swiglu_shared_q8_0_gated_workspace_bytes(
        hidden_dim, intermediate_dim);
    if (!hidden || !shared_gate || !shared_up || !shared_down ||
        !shared_gate_input || !output || !workspace || stride == 0 || rows <= 0) {
        return -1;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active > rows) active = rows;
    if (active > CK_THREADPOOL_MAX_THREADS) active = CK_THREADPOOL_MAX_THREADS;
    if ((size_t)active > SIZE_MAX / stride ||
        workspace_bytes < stride * (size_t)active) {
        return -1;
    }
    if (active <= 1) {
        return moe_swiglu_shared_forward_q8_0_gated_workspace(
            hidden, routed, shared_gate, shared_up, shared_down,
            shared_gate_input, output, rows, hidden_dim, intermediate_dim,
            workspace, stride);
    }

    ck_moe_shared_q8_0_parallel_args_t args = {
        .hidden = hidden,
        .routed = routed,
        .shared_gate = shared_gate,
        .shared_up = shared_up,
        .shared_down = shared_down,
        .shared_gate_input = shared_gate_input,
        .output = output,
        .rows = rows,
        .hidden_dim = hidden_dim,
        .intermediate_dim = intermediate_dim,
        .workspace = (uint8_t *)workspace,
        .workspace_stride = stride,
        .status = {0},
    };
    ck_threadpool_dispatch_n(
        pool, active, ck_moe_shared_q8_0_parallel_work, &args);
    for (int ith = 0; ith < active; ++ith) {
        if (args.status[ith] != 0) return args.status[ith];
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
