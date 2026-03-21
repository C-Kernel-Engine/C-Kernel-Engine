#include "ckernel_engine.h"

#include <math.h>
#include <string.h>

static inline float recurrent_sigmoid_local(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

void recurrent_norm_gate_forward(const float *x,
                                 const float *gate,
                                 const float *weight,
                                 float *out,
                                 int rows,
                                 int num_heads,
                                 int head_dim,
                                 float eps) {
    const int inner_dim = num_heads * head_dim;
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t) row * (size_t) inner_dim;
        const float *gate_row = gate + (size_t) row * (size_t) inner_dim;
        float *out_row = out + (size_t) row * (size_t) inner_dim;

        for (int head = 0; head < num_heads; ++head) {
            const float *x_head = x_row + (size_t) head * (size_t) head_dim;
            const float *gate_head = gate_row + (size_t) head * (size_t) head_dim;
            float *out_head = out_row + (size_t) head * (size_t) head_dim;

            float ms = 0.0f;
            for (int col = 0; col < head_dim; ++col) {
                ms += x_head[col] * x_head[col];
            }
            ms /= (float) head_dim;
            const float inv_rms = 1.0f / sqrtf(ms + eps);

            for (int col = 0; col < head_dim; ++col) {
                const float g = gate_head[col];
                const float silu = g * recurrent_sigmoid_local(g);
                out_head[col] = x_head[col] * inv_rms * weight[col] * silu;
            }
        }
    }
}

void recurrent_norm_gate_backward(const float *d_out,
                                  const float *x,
                                  const float *gate,
                                  const float *weight,
                                  float *d_x,
                                  float *d_gate,
                                  float *d_weight,
                                  int rows,
                                  int num_heads,
                                  int head_dim,
                                  float eps) {
    const int inner_dim = num_heads * head_dim;
    memset(d_weight, 0, (size_t) head_dim * sizeof(float));

    for (int row = 0; row < rows; ++row) {
        const float *d_out_row = d_out + (size_t) row * (size_t) inner_dim;
        const float *x_row = x + (size_t) row * (size_t) inner_dim;
        const float *gate_row = gate + (size_t) row * (size_t) inner_dim;
        float *d_x_row = d_x + (size_t) row * (size_t) inner_dim;
        float *d_gate_row = d_gate + (size_t) row * (size_t) inner_dim;

        for (int head = 0; head < num_heads; ++head) {
            const float *x_head = x_row + (size_t) head * (size_t) head_dim;
            const float *gate_head = gate_row + (size_t) head * (size_t) head_dim;
            const float *d_out_head = d_out_row + (size_t) head * (size_t) head_dim;
            float *d_x_head = d_x_row + (size_t) head * (size_t) head_dim;
            float *d_gate_head = d_gate_row + (size_t) head * (size_t) head_dim;

            float ms = 0.0f;
            for (int col = 0; col < head_dim; ++col) {
                ms += x_head[col] * x_head[col];
            }
            ms /= (float) head_dim;
            const float inv_rms = 1.0f / sqrtf(ms + eps);
            const float inv_rms3_over_dim = (inv_rms * inv_rms * inv_rms) / (float) head_dim;

            float dot = 0.0f;
            for (int col = 0; col < head_dim; ++col) {
                const float g = gate_head[col];
                const float sig = recurrent_sigmoid_local(g);
                const float silu = g * sig;
                dot += d_out_head[col] * weight[col] * silu * x_head[col];
            }

            for (int col = 0; col < head_dim; ++col) {
                const float g = gate_head[col];
                const float sig = recurrent_sigmoid_local(g);
                const float silu = g * sig;
                const float scaled = inv_rms * weight[col] * silu;
                d_x_head[col] = d_out_head[col] * scaled - x_head[col] * inv_rms3_over_dim * dot;
                d_weight[col] += d_out_head[col] * (x_head[col] * inv_rms) * silu;
                d_gate_head[col] = d_out_head[col] * (x_head[col] * inv_rms * weight[col]) *
                                   (sig + g * sig * (1.0f - sig));
            }
        }
    }
}
