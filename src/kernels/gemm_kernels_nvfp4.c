/**
 * @file gemm_kernels_nvfp4.c
 * @brief Packed NVFP4 weight kernels for CPU inference.
 *
 * The storage ABI keeps E2M1 weights and E4M3 block scales packed. The
 * checkpoint's reciprocal tensor/expert scale is supplied separately so no
 * weight expansion or scale re-encoding is required during conversion.
 */

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ck_threadpool.h"
#include "ckernel_quant.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* Twice the represented E2M1 value. The 0.5 factor is carried by E4M3. */
static const int8_t ck_nvfp4_e2m1_x2[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

/* UE4M3 values include the 0.5 compensation for the doubled E2M1 table. */
static const float ck_nvfp4_ue4m3[128] = {
    0.0f, 0.0009765625f, 0.001953125f, 0.0029296875f, 0.00390625f, 0.0048828125f, 0.005859375f, 0.0068359375f,
    0.0078125f, 0.0087890625f, 0.009765625f, 0.0107421875f, 0.01171875f, 0.0126953125f, 0.013671875f, 0.0146484375f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 0.0f,
};

static inline float ck_ue4m3_to_fp32_inline(uint8_t value)
{
    return ck_nvfp4_ue4m3[value & UINT8_C(0x7f)];
}

float ck_ue4m3_to_fp32(uint8_t value)
{
    return ck_ue4m3_to_fp32_inline(value);
}

void dequantize_row_nvfp4(const void *weights, float *output, int k,
                          float weight_scale)
{
    assert(k >= 0 && k % QK_NVFP4 == 0);
    const block_nvfp4 *blocks = (const block_nvfp4 *)weights;
    const int block_count = k / QK_NVFP4;

    for (int block_index = 0; block_index < block_count; ++block_index) {
        const block_nvfp4 *block = &blocks[block_index];
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            const float scale = ck_ue4m3_to_fp32_inline(block->d[sub]) * weight_scale;
            const uint8_t *packed = &block->qs[sub * (QK_NVFP4_SUB / 2)];
            float *dst = &output[block_index * QK_NVFP4 + sub * QK_NVFP4_SUB];
            for (int lane = 0; lane < QK_NVFP4_SUB / 2; ++lane) {
                const uint8_t pair = packed[lane];
                dst[lane] = scale * (float)ck_nvfp4_e2m1_x2[pair & 0x0f];
                dst[lane + QK_NVFP4_SUB / 2] =
                    scale * (float)ck_nvfp4_e2m1_x2[pair >> 4];
            }
        }
    }
}

void vec_dot_nvfp4_q8_0_ref(int n, float *output, const void *weights,
                            const void *activations, float weight_scale)
{
    assert(n >= 0 && n % QK_NVFP4 == 0);
    const block_nvfp4 *w = (const block_nvfp4 *)weights;
    const block_q8_0 *x = (const block_q8_0 *)activations;
    const int block_count = n / QK_NVFP4;
    float sum = 0.0f;

    for (int block_index = 0; block_index < block_count; ++block_index) {
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            const int q8_block = sub / 2;
            const int q8_offset = (sub % 2) * QK_NVFP4_SUB;
            const float scale = ck_ue4m3_to_fp32_inline(w[block_index].d[sub]) *
                                CK_FP16_TO_FP32(x[2 * block_index + q8_block].d) *
                                weight_scale;
            const uint8_t *packed =
                &w[block_index].qs[sub * (QK_NVFP4_SUB / 2)];
            const int8_t *q8 = &x[2 * block_index + q8_block].qs[q8_offset];
            int integer_sum = 0;
            for (int lane = 0; lane < QK_NVFP4_SUB / 2; ++lane) {
                const uint8_t pair = packed[lane];
                integer_sum += (int)q8[lane] *
                               (int)ck_nvfp4_e2m1_x2[pair & 0x0f];
                integer_sum += (int)q8[lane + QK_NVFP4_SUB / 2] *
                               (int)ck_nvfp4_e2m1_x2[pair >> 4];
            }
            sum += scale * (float)integer_sum;
        }
    }
    *output = sum;
}

#if defined(__AVX2__)
static inline __m256i ck_nvfp4_mul_add_i8_avx2(__m256i weights,
                                               __m256i activations)
{
    const __m256i absolute_weights = _mm256_sign_epi8(weights, weights);
    const __m256i signed_activations =
        _mm256_sign_epi8(activations, weights);
    return _mm256_maddubs_epi16(absolute_weights, signed_activations);
}
#endif

void vec_dot_nvfp4_q8_0(int n, float *output, const void *weights,
                        const void *activations, float weight_scale)
{
#if defined(__AVX2__)
    assert(n >= 0 && n % QK_NVFP4 == 0);
    const block_nvfp4 *w = (const block_nvfp4 *)weights;
    const block_q8_0 *x = (const block_q8_0 *)activations;
    const int block_count = n / QK_NVFP4;
    const __m128i lut = _mm_loadu_si128(
        (const __m128i *)ck_nvfp4_e2m1_x2);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 accumulated = _mm256_setzero_ps();

    for (int block_index = 0; block_index < block_count; ++block_index) {
        const block_nvfp4 *block = &w[block_index];
        const __m128i packed01 = _mm_loadu_si128(
            (const __m128i *)(block->qs + 0));
        const __m128i packed23 = _mm_loadu_si128(
            (const __m128i *)(block->qs + 16));
        const __m128i low01 = _mm_shuffle_epi8(
            lut, _mm_and_si128(packed01, nibble_mask));
        const __m128i high01 = _mm_shuffle_epi8(
            lut, _mm_and_si128(_mm_srli_epi16(packed01, 4), nibble_mask));
        const __m128i low23 = _mm_shuffle_epi8(
            lut, _mm_and_si128(packed23, nibble_mask));
        const __m128i high23 = _mm_shuffle_epi8(
            lut, _mm_and_si128(_mm_srli_epi16(packed23, 4), nibble_mask));

        __m256i values01 = _mm256_castsi128_si256(
            _mm_unpacklo_epi64(low01, high01));
        values01 = _mm256_inserti128_si256(
            values01, _mm_unpackhi_epi64(low01, high01), 1);
        __m256i values23 = _mm256_castsi128_si256(
            _mm_unpacklo_epi64(low23, high23));
        values23 = _mm256_inserti128_si256(
            values23, _mm_unpackhi_epi64(low23, high23), 1);

        const __m256i q8_01 = _mm256_loadu_si256(
            (const __m256i *)x[2 * block_index + 0].qs);
        const __m256i q8_23 = _mm256_loadu_si256(
            (const __m256i *)x[2 * block_index + 1].qs);
        const __m256i dot01 = _mm256_madd_epi16(
            ck_nvfp4_mul_add_i8_avx2(values01, q8_01), ones);
        const __m256i dot23 = _mm256_madd_epi16(
            ck_nvfp4_mul_add_i8_avx2(values23, q8_23), ones);

        const float q8_scale0 =
            CK_FP16_TO_FP32(x[2 * block_index + 0].d);
        const float q8_scale1 =
            CK_FP16_TO_FP32(x[2 * block_index + 1].d);
        const float scale0 = ck_ue4m3_to_fp32_inline(block->d[0]) * q8_scale0;
        const float scale1 = ck_ue4m3_to_fp32_inline(block->d[1]) * q8_scale0;
        const float scale2 = ck_ue4m3_to_fp32_inline(block->d[2]) * q8_scale1;
        const float scale3 = ck_ue4m3_to_fp32_inline(block->d[3]) * q8_scale1;
        const __m256 scales01 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_set1_ps(scale0)),
            _mm_set1_ps(scale1), 1);
        const __m256 scales23 = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_set1_ps(scale2)),
            _mm_set1_ps(scale3), 1);
        accumulated = _mm256_fmadd_ps(
            scales01, _mm256_cvtepi32_ps(dot01), accumulated);
        accumulated = _mm256_fmadd_ps(
            scales23, _mm256_cvtepi32_ps(dot23), accumulated);
    }

    __m128 sum4 = _mm_add_ps(
        _mm256_castps256_ps128(accumulated),
        _mm256_extractf128_ps(accumulated, 1));
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    *output = _mm_cvtss_f32(sum4) * weight_scale;
#else
    vec_dot_nvfp4_q8_0_ref(n, output, weights, activations, weight_scale);
#endif
}

void gemv_nvfp4_q8_0(float *output, const void *weights,
                     const float *weight_scales, const void *activations,
                     int rows, int cols)
{
    assert(rows >= 0 && cols >= 0 && cols % QK_NVFP4 == 0);
    const size_t row_bytes = (size_t)(cols / QK_NVFP4) * sizeof(block_nvfp4);
    const uint8_t *weight_bytes = (const uint8_t *)weights;
    for (int row = 0; row < rows; ++row) {
        const float scale = weight_scales ? weight_scales[row] : 1.0f;
        vec_dot_nvfp4_q8_0(cols, &output[row],
                          weight_bytes + (size_t)row * row_bytes,
                          activations, scale);
    }
}

static size_t ck_nvfp4_align64(size_t value)
{
    return (value + 63u) & ~(size_t)63u;
}

typedef struct {
    float *output;
    const uint8_t *weights;
    const void *activations;
    float weight_scale;
    size_t row_bytes;
    int cols;
} ck_nvfp4_gemv_rows_args_t;

static void ck_nvfp4_gemv_rows(int begin, int end, void *opaque)
{
    const ck_nvfp4_gemv_rows_args_t *args =
        (const ck_nvfp4_gemv_rows_args_t *)opaque;
    for (int row = begin; row < end; ++row) {
        vec_dot_nvfp4_q8_0(
            args->cols, &args->output[row],
            args->weights + (size_t)row * args->row_bytes,
            args->activations, args->weight_scale);
    }
}

static void gemv_nvfp4_q8_0_uniform(float *output, const void *weights,
                                    float weight_scale,
                                    const void *activations, int rows, int cols)
{
    ck_nvfp4_gemv_rows_args_t args = {
        .output = output,
        .weights = (const uint8_t *)weights,
        .activations = activations,
        .weight_scale = weight_scale,
        .row_bytes = ck_dtype_row_bytes(CK_DT_NVFP4, (size_t)cols),
        .cols = cols,
    };
    ck_threadpool_t *pool = ck_threadpool_global();
    int active_threads = pool ? ck_threadpool_n_threads(pool) : 1;
    if (active_threads > rows) {
        active_threads = rows;
    }
    if (!pool || active_threads <= 1 || rows < 64) {
        ck_nvfp4_gemv_rows(0, rows, &args);
        return;
    }

    int grain = rows / (active_threads * 4);
    if (grain < 8) {
        grain = 8;
    }
    ck_threadpool_parallel_for_n(
        pool, active_threads, 0, rows, grain, ck_nvfp4_gemv_rows, &args);
}

size_t moe_swiglu_nvfp4_workspace_bytes(int hidden_dim, int intermediate_dim)
{
    if (hidden_dim <= 0 || intermediate_dim <= 0 || hidden_dim % 64 != 0 ||
        intermediate_dim % 64 != 0) {
        return 0;
    }
    size_t bytes = ck_nvfp4_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)hidden_dim));
    bytes += ck_nvfp4_align64(2u * (size_t)intermediate_dim * sizeof(float));
    bytes += ck_nvfp4_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)intermediate_dim));
    bytes += ck_nvfp4_align64((size_t)hidden_dim * sizeof(float));
    return bytes;
}

static int ck_moe_swiglu_nvfp4_projection(
    const float *hidden, const void *gate, float gate_scale,
    const void *up, float up_scale, const void *down, float down_scale,
    float *result, int hidden_dim, int intermediate_dim, void *workspace)
{
    uint8_t *cursor = (uint8_t *)workspace;
    void *hidden_q8 = cursor;
    cursor += ck_nvfp4_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)hidden_dim));
    float *gate_up = (float *)cursor;
    cursor += ck_nvfp4_align64(2u * (size_t)intermediate_dim * sizeof(float));
    void *act_q8 = cursor;

    quantize_row_q8_0(hidden, hidden_q8, hidden_dim);
    gemv_nvfp4_q8_0_uniform(gate_up, gate, gate_scale, hidden_q8,
                            intermediate_dim, hidden_dim);
    gemv_nvfp4_q8_0_uniform(gate_up + intermediate_dim, up, up_scale,
                            hidden_q8, intermediate_dim, hidden_dim);
    for (int i = 0; i < intermediate_dim; ++i) {
        const float value = gate_up[i];
        gate_up[i] = (value / (1.0f + expf(-value))) *
                     gate_up[intermediate_dim + i];
    }
    quantize_row_q8_0(gate_up, act_q8, intermediate_dim);
    gemv_nvfp4_q8_0_uniform(result, down, down_scale, act_q8,
                            hidden_dim, intermediate_dim);
    return 0;
}

int moe_swiglu_expert_forward_nvfp4_workspace(
    const float *hidden, const int *indices, const float *routing_weights,
    const void *expert_gate, const float *expert_gate_scales,
    const void *expert_up, const float *expert_up_scales,
    const void *expert_down, const float *expert_down_scales,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    int n_experts, int top_k, void *workspace, size_t workspace_bytes)
{
    const size_t required = moe_swiglu_nvfp4_workspace_bytes(
        hidden_dim, intermediate_dim);
    if (!hidden || !indices || !routing_weights || !expert_gate ||
        !expert_gate_scales || !expert_up || !expert_up_scales ||
        !expert_down || !expert_down_scales || !output || !workspace ||
        required == 0 || workspace_bytes < required || rows <= 0 ||
        n_experts <= 0 || top_k <= 0 || top_k > n_experts) {
        return -1;
    }

    const size_t hidden_q8_bytes = ck_nvfp4_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)hidden_dim));
    const size_t gate_up_bytes = ck_nvfp4_align64(
        2u * (size_t)intermediate_dim * sizeof(float));
    const size_t act_q8_bytes = ck_nvfp4_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)intermediate_dim));
    float *expert_output = (float *)((uint8_t *)workspace + hidden_q8_bytes +
                                     gate_up_bytes + act_q8_bytes);
    const size_t up_expert_bytes = (size_t)intermediate_dim *
        ck_dtype_row_bytes(CK_DT_NVFP4, (size_t)hidden_dim);
    const size_t down_expert_bytes = (size_t)hidden_dim *
        ck_dtype_row_bytes(CK_DT_NVFP4, (size_t)intermediate_dim);
    memset(output, 0, (size_t)rows * (size_t)hidden_dim * sizeof(float));

    for (int row = 0; row < rows; ++row) {
        const float *x = hidden + (size_t)row * (size_t)hidden_dim;
        float *y = output + (size_t)row * (size_t)hidden_dim;
        for (int slot = 0; slot < top_k; ++slot) {
            const size_t route = (size_t)row * (size_t)top_k + (size_t)slot;
            const int expert = indices[route];
            if (expert < 0 || expert >= n_experts) {
                return -2;
            }
            ck_moe_swiglu_nvfp4_projection(
                x,
                (const uint8_t *)expert_gate + (size_t)expert * up_expert_bytes,
                expert_gate_scales[expert],
                (const uint8_t *)expert_up + (size_t)expert * up_expert_bytes,
                expert_up_scales[expert],
                (const uint8_t *)expert_down + (size_t)expert * down_expert_bytes,
                expert_down_scales[expert], expert_output,
                hidden_dim, intermediate_dim, workspace);
            const float route_weight = routing_weights[route];
            for (int h = 0; h < hidden_dim; ++h) {
                y[h] += route_weight * expert_output[h];
            }
        }
    }
    return 0;
}

int moe_swiglu_shared_forward_nvfp4_workspace(
    const float *hidden, const float *routed,
    const void *shared_gate, const float *shared_gate_scale,
    const void *shared_up, const float *shared_up_scale,
    const void *shared_down, const float *shared_down_scale,
    float *output, int rows, int hidden_dim, int intermediate_dim,
    float combination_scale, void *workspace, size_t workspace_bytes)
{
    const size_t required = moe_swiglu_nvfp4_workspace_bytes(
        hidden_dim, intermediate_dim);
    if (!hidden || !shared_gate || !shared_gate_scale || !shared_up ||
        !shared_up_scale || !shared_down || !shared_down_scale || !output ||
        !workspace || required == 0 || workspace_bytes < required || rows <= 0) {
        return -1;
    }
    const size_t hidden_q8_bytes = ck_nvfp4_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)hidden_dim));
    const size_t gate_up_bytes = ck_nvfp4_align64(
        2u * (size_t)intermediate_dim * sizeof(float));
    const size_t act_q8_bytes = ck_nvfp4_align64(
        ck_dtype_row_bytes(CK_DT_Q8_0, (size_t)intermediate_dim));
    float *shared_output = (float *)((uint8_t *)workspace + hidden_q8_bytes +
                                     gate_up_bytes + act_q8_bytes);
    for (int row = 0; row < rows; ++row) {
        ck_moe_swiglu_nvfp4_projection(
            hidden + (size_t)row * (size_t)hidden_dim,
            shared_gate, shared_gate_scale[0], shared_up, shared_up_scale[0],
            shared_down, shared_down_scale[0], shared_output,
            hidden_dim, intermediate_dim, workspace);
        float *y = output + (size_t)row * (size_t)hidden_dim;
        const float *route = routed ? routed + (size_t)row * (size_t)hidden_dim : NULL;
        for (int h = 0; h < hidden_dim; ++h) {
            y[h] = combination_scale *
                   (shared_output[h] + (route ? route[h] : 0.0f));
        }
    }
    return 0;
}
