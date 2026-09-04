#include "ckernel_engine.h"
#include "bf16_utils.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);
void gemv_q4_k_q8_k_avx2(float *y,
                         const void *W,
                         const void *x_q8,
                         int M,
                         int K);
void gemm_nt_q5_0_q8_0_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);
void gemm_nt_q6_k_q8_k_parallel_dispatch(
    const void *A, const void *B, const float *bias, float *C,
    int M, int N, int K);

static inline float ck_bf16_round(float value) {
    return bf16_to_float(float_to_bf16(value));
}

static inline float ck_sigmoid_bf16(float value) {
    return ck_bf16_round(1.0f / (1.0f + expf(-value)));
}

void hyper_stream_expand_f32(const float *input,
                             float *output,
                             int rows,
                             int streams,
                             int hidden_dim) {
    if (!input || !output || rows <= 0 || streams <= 0 || hidden_dim <= 0) {
        return;
    }
    for (int row = 0; row < rows; ++row) {
        const float *src = input + (size_t)row * (size_t)hidden_dim;
        float *dst = output + (size_t)row * (size_t)streams * (size_t)hidden_dim;
        for (int stream = 0; stream < streams; ++stream) {
            for (int col = 0; col < hidden_dim; ++col) {
                dst[(size_t)stream * (size_t)hidden_dim + (size_t)col] = src[col];
            }
        }
    }
}

void hyper_stream_expand_bf16(const float *input,
                              float *output,
                              int rows,
                              int streams,
                              int hidden_dim) {
    if (!input || !output || rows <= 0 || streams <= 0 || hidden_dim <= 0) {
        return;
    }
    for (int row = 0; row < rows; ++row) {
        const float *src = input + (size_t)row * (size_t)hidden_dim;
        float *dst = output + (size_t)row * (size_t)streams * (size_t)hidden_dim;
        for (int stream = 0; stream < streams; ++stream) {
            for (int col = 0; col < hidden_dim; ++col) {
                dst[(size_t)stream * (size_t)hidden_dim + (size_t)col] =
                    ck_bf16_round(src[col]);
            }
        }
    }
}

void hyper_connection_mix_bf16(const float *hyper_input,
                               const float *norm_weight,
                               const uint16_t *mix_down_weight,
                               const uint16_t *mix_up_weight,
                               const uint16_t *inject_weight,
                               float *mixed_output,
                               float *injection_output,
                               float *normalized_scratch,
                               float *dynamic_scratch,
                               float *mix_scratch,
                               int rows,
                               int streams,
                               int hidden_dim,
                               int dynamic_dim,
                               float eps,
                               int emit_injection) {
    if (!hyper_input || !norm_weight || !mix_down_weight || !mix_up_weight ||
        !mixed_output || !normalized_scratch || !dynamic_scratch || !mix_scratch ||
        rows <= 0 || streams <= 0 || hidden_dim <= 0 || dynamic_dim <= 0) {
        return;
    }
    if (emit_injection && (!inject_weight || !injection_output)) {
        return;
    }

    const int hyper_dim = streams * hidden_dim;
    const float inv_streams = 1.0f / (float)streams;

    for (int row = 0; row < rows; ++row) {
        const float *input_row =
            hyper_input + (size_t)row * (size_t)hyper_dim;
        float *norm_row =
            normalized_scratch + (size_t)row * (size_t)hyper_dim;
        float *dynamic_row =
            dynamic_scratch + (size_t)row * (size_t)dynamic_dim;
        float *mix_row = mix_scratch + (size_t)row * (size_t)hyper_dim;

        for (int stream = 0; stream < streams; ++stream) {
            const int base = stream * hidden_dim;
            rmsnorm_forward_qwen3next_pytorch_bf16_storage(
                input_row + base,
                norm_weight + base,
                norm_row + base,
                NULL,
                1,
                hidden_dim,
                hidden_dim,
                eps
            );
        }

        for (int out = 0; out < dynamic_dim; ++out) {
            const uint16_t *weight_row =
                mix_down_weight + (size_t)out * (size_t)hyper_dim;
            float sum = 0.0f;
            for (int col = 0; col < hyper_dim; ++col) {
                sum += norm_row[col] * bf16_to_float(weight_row[col]);
            }
            const float projected = ck_bf16_round(sum * inv_streams);
            dynamic_row[out] = ck_bf16_round(
                projected / (1.0f + expf(-projected)));
        }

        for (int out = 0; out < hyper_dim; ++out) {
            const uint16_t *weight_row =
                mix_up_weight + (size_t)out * (size_t)dynamic_dim;
            float sum = 0.0f;
            for (int col = 0; col < dynamic_dim; ++col) {
                sum += dynamic_row[col] * bf16_to_float(weight_row[col]);
            }
            mix_row[out] = ck_sigmoid_bf16(ck_bf16_round(sum));
        }

        float *mixed_row =
            mixed_output + (size_t)row * (size_t)hidden_dim;
        for (int col = 0; col < hidden_dim; ++col) {
            float sum = 0.0f;
            for (int stream = 0; stream < streams; ++stream) {
                const int index = stream * hidden_dim + col;
                sum += ck_bf16_round(norm_row[index] * mix_row[index]);
            }
            mixed_row[col] = ck_bf16_round(sum * inv_streams);
        }

        if (emit_injection) {
            float *injection_row =
                injection_output + (size_t)row * (size_t)streams;
            for (int stream = 0; stream < streams; ++stream) {
                const uint16_t *weight_row =
                    inject_weight + (size_t)stream * (size_t)hyper_dim;
                float sum = 0.0f;
                for (int col = 0; col < hyper_dim; ++col) {
                    sum += norm_row[col] * bf16_to_float(weight_row[col]);
                }
                injection_row[stream] = ck_bf16_round(
                    2.0f * ck_sigmoid_bf16(ck_bf16_round(sum * inv_streams)));
            }
        }
    }
}

typedef void (*ck_hyper_q8k_gemm_fn)(
    const void *, const void *, const float *, float *, int, int, int);

static void hyper_injection_q4k_q8k_llama_dispatch(
    const void *input,
    const void *weight,
    const float *bias,
    float *output,
    int rows,
    int output_dim,
    int input_dim) {
    if (!input || !weight || !output || rows <= 0 || output_dim <= 0 ||
        input_dim <= 0 || input_dim % QK_K != 0) {
        return;
    }

    const size_t input_row_bytes =
        (size_t)(input_dim / QK_K) * sizeof(block_q8_K);
    for (int row = 0; row < rows; ++row) {
        float *output_row = output + (size_t)row * (size_t)output_dim;
        const void *input_row =
            (const uint8_t *)input + (size_t)row * input_row_bytes;
        gemv_q4_k_q8_k_avx2(
            output_row, weight, input_row, output_dim, input_dim);
        if (bias) {
            for (int col = 0; col < output_dim; ++col) {
                output_row[col] += bias[col];
            }
        }
    }
}

static void hyper_connection_mix_quantized(const float *hyper_input,
                                           const float *norm_weight,
                                           const void *mix_down_weight,
                                           const void *mix_up_weight,
                                           const void *inject_weight,
                                           float *mixed_output,
                                           float *injection_output,
                                           float *normalized_scratch,
                                           float *dynamic_scratch,
                                           float *mix_scratch,
                                           int rows,
                                           int streams,
                                           int hidden_dim,
                                           int dynamic_dim,
                                           float eps,
                                           int emit_injection,
                                           ck_hyper_q8k_gemm_fn injection_gemm,
                                           ck_hyper_q8k_gemm_fn down_gemm) {
    if (!hyper_input || !norm_weight || !mix_down_weight || !mix_up_weight ||
        !mixed_output || !normalized_scratch || !dynamic_scratch || !mix_scratch ||
        !injection_gemm || !down_gemm || rows <= 0 || streams <= 0 || hidden_dim <= 0 ||
        dynamic_dim <= 0) {
        return;
    }
    if (emit_injection && (!inject_weight || !injection_output)) {
        return;
    }

    const int hyper_dim = streams * hidden_dim;
    const float inv_streams = 1.0f / (float)streams;
    if (hyper_dim % QK_K != 0 || dynamic_dim % QK8_0 != 0) {
        return;
    }

    const size_t normalized_q8_row_bytes =
        (size_t)(hyper_dim / QK_K) * sizeof(block_q8_K);
    const size_t dynamic_q8_row_bytes =
        (size_t)(dynamic_dim / QK8_0) * sizeof(block_q8_0);
    block_q8_K *normalized_q8 = (block_q8_K *)mix_scratch;
    block_q8_0 dynamic_q8[dynamic_dim / QK8_0];

    for (int row = 0; row < rows; ++row) {
        const float *input_row = hyper_input + (size_t)row * (size_t)hyper_dim;
        float *norm_row = normalized_scratch + (size_t)row * (size_t)hyper_dim;
        for (int stream = 0; stream < streams; ++stream) {
            const int base = stream * hidden_dim;
            double sum_sq = 0.0;
            for (int col = 0; col < hidden_dim; ++col) {
                const float value = input_row[base + col];
                sum_sq += (double)(value * value);
            }
            const float mean = (float)(sum_sq / (double)hidden_dim);
            const float rstd = 1.0f / sqrtf(mean + eps);
            for (int col = 0; col < hidden_dim; ++col) {
                const int index = base + col;
                norm_row[index] = input_row[index] * rstd * norm_weight[index];
            }
        }

        quantize_row_q8_k(
            norm_row,
            (uint8_t *)normalized_q8 + (size_t)row * normalized_q8_row_bytes,
            hyper_dim);
    }

    down_gemm(
        normalized_q8,
        mix_down_weight,
        NULL,
        dynamic_scratch,
        rows,
        dynamic_dim,
        hyper_dim);

    if (emit_injection) {
        injection_gemm(
            normalized_q8,
            inject_weight,
            NULL,
            injection_output,
            rows,
            streams,
            hyper_dim);
        for (int row = 0; row < rows; ++row) {
            float *injection_row = injection_output + (size_t)row * (size_t)streams;
            for (int stream = 0; stream < streams; ++stream) {
                injection_row[stream] *= inv_streams;
            }
            recurrent_sigmoid_forward_ggml(
                injection_row, injection_row, 1, streams);
            for (int stream = 0; stream < streams; ++stream) {
                injection_row[stream] *= 2.0f;
            }
        }
    }

    for (int row = 0; row < rows; ++row) {
        float *dynamic_row = dynamic_scratch + (size_t)row * (size_t)dynamic_dim;
        for (int col = 0; col < dynamic_dim; ++col) {
            dynamic_row[col] *= inv_streams;
        }
        recurrent_silu_forward_ggml(
            dynamic_row, dynamic_row, 1, dynamic_dim);

        quantize_row_q8_0(dynamic_row, dynamic_q8, dynamic_dim);
        memcpy(
            (uint8_t *)dynamic_scratch + (size_t)row * dynamic_q8_row_bytes,
            dynamic_q8,
            dynamic_q8_row_bytes);
    }

    gemm_nt_q5_0_q8_0_parallel_dispatch(
        dynamic_scratch,
        mix_up_weight,
        NULL,
        mix_scratch,
        rows,
        hyper_dim,
        dynamic_dim);

    for (int row = 0; row < rows; ++row) {
        const float *norm_row =
            normalized_scratch + (size_t)row * (size_t)hyper_dim;
        float *mix_row = mix_scratch + (size_t)row * (size_t)hyper_dim;
        recurrent_sigmoid_forward_ggml(
            mix_row, mix_row, 1, hyper_dim);

        float *mixed_row = mixed_output + (size_t)row * (size_t)hidden_dim;
        for (int col = 0; col < hidden_dim; ++col) {
            float sum = 0.0f;
            for (int stream = 0; stream < streams; ++stream) {
                const int index = stream * hidden_dim + col;
                sum += norm_row[index] * mix_row[index];
            }
            mixed_row[col] = sum * inv_streams;
        }

    }
}

void hyper_connection_mix_q4k_q5_0_q4k(const float *hyper_input,
                                       const float *norm_weight,
                                       const void *mix_down_weight,
                                       const void *mix_up_weight,
                                       const void *inject_weight,
                                       float *mixed_output,
                                       float *injection_output,
                                       float *normalized_scratch,
                                       float *dynamic_scratch,
                                       float *mix_scratch,
                                       int rows,
                                       int streams,
                                       int hidden_dim,
                                       int dynamic_dim,
                                       float eps,
                                       int emit_injection) {
    hyper_connection_mix_quantized(
        hyper_input, norm_weight, mix_down_weight, mix_up_weight, inject_weight,
        mixed_output, injection_output, normalized_scratch, dynamic_scratch,
        mix_scratch, rows, streams, hidden_dim, dynamic_dim, eps,
        emit_injection, hyper_injection_q4k_q8k_llama_dispatch,
        gemm_nt_q4_k_q8_k_pairwise_split_min_parallel_dispatch);
}

void hyper_connection_mix_q6k_q5_0_q4k(const float *hyper_input,
                                       const float *norm_weight,
                                       const void *mix_down_weight,
                                       const void *mix_up_weight,
                                       const void *inject_weight,
                                       float *mixed_output,
                                       float *injection_output,
                                       float *normalized_scratch,
                                       float *dynamic_scratch,
                                       float *mix_scratch,
                                       int rows,
                                       int streams,
                                       int hidden_dim,
                                       int dynamic_dim,
                                       float eps,
                                       int emit_injection) {
    hyper_connection_mix_quantized(
        hyper_input, norm_weight, mix_down_weight, mix_up_weight, inject_weight,
        mixed_output, injection_output, normalized_scratch, dynamic_scratch,
        mix_scratch, rows, streams, hidden_dim, dynamic_dim, eps,
        emit_injection, hyper_injection_q4k_q8k_llama_dispatch,
        gemm_nt_q6_k_q8_k_parallel_dispatch);
}

void hyper_stream_inject_bf16(const float *hyper_input,
                              const float *block_output,
                              const float *injection_weight,
                              float *output,
                              int rows,
                              int streams,
                              int hidden_dim) {
    if (!hyper_input || !block_output || !injection_weight || !output ||
        rows <= 0 || streams <= 0 || hidden_dim <= 0) {
        return;
    }
    const int hyper_dim = streams * hidden_dim;
    for (int row = 0; row < rows; ++row) {
        const float *hyper_row =
            hyper_input + (size_t)row * (size_t)hyper_dim;
        const float *block_row =
            block_output + (size_t)row * (size_t)hidden_dim;
        const float *inject_row =
            injection_weight + (size_t)row * (size_t)streams;
        float *output_row = output + (size_t)row * (size_t)hyper_dim;
        for (int stream = 0; stream < streams; ++stream) {
            for (int col = 0; col < hidden_dim; ++col) {
                const int index = stream * hidden_dim + col;
                output_row[index] = ck_bf16_round(
                    hyper_row[index] +
                    ck_bf16_round(block_row[col] * inject_row[stream]));
            }
        }
    }
}

void hyper_stream_inject_f32(const float *hyper_input,
                             const float *block_output,
                             const float *injection_weight,
                             float *output,
                             int rows,
                             int streams,
                             int hidden_dim) {
    if (!hyper_input || !block_output || !injection_weight || !output ||
        rows <= 0 || streams <= 0 || hidden_dim <= 0) {
        return;
    }
    const int hyper_dim = streams * hidden_dim;
    for (int row = 0; row < rows; ++row) {
        const float *hyper_row =
            hyper_input + (size_t)row * (size_t)hyper_dim;
        const float *block_row =
            block_output + (size_t)row * (size_t)hidden_dim;
        const float *inject_row =
            injection_weight + (size_t)row * (size_t)streams;
        float *output_row = output + (size_t)row * (size_t)hyper_dim;
        for (int stream = 0; stream < streams; ++stream) {
            for (int col = 0; col < hidden_dim; ++col) {
                const int index = stream * hidden_dim + col;
                volatile const float weighted =
                    block_row[col] * inject_row[stream];
                output_row[index] = hyper_row[index] + weighted;
            }
        }
    }
}
