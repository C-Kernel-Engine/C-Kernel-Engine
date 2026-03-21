#include "ckernel_engine.h"

#include <math.h>

static void recurrent_l2_norm_rows_forward_one(float *x,
                                               int rows,
                                               int dim,
                                               int head_dim,
                                               float eps) {
    if (!x || rows <= 0 || dim <= 0 || head_dim <= 0) {
        return;
    }
    const int num_heads = dim / head_dim;
    if (num_heads <= 0 || num_heads * head_dim != dim) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        float *row_ptr = x + (size_t) row * (size_t) dim;
        for (int head = 0; head < num_heads; ++head) {
            float *head_ptr = row_ptr + (size_t) head * (size_t) head_dim;
            float sum_sq = 0.0f;
            for (int col = 0; col < head_dim; ++col) {
                sum_sq += head_ptr[col] * head_ptr[col];
            }
            const float inv_norm = 1.0f / sqrtf(sum_sq + eps);
            for (int col = 0; col < head_dim; ++col) {
                head_ptr[col] *= inv_norm;
            }
        }
    }
}

static void recurrent_l2_norm_rows_backward_one(const float *d_out,
                                                const float *x,
                                                float *d_x,
                                                int rows,
                                                int dim,
                                                int head_dim,
                                                float eps) {
    if (!d_out || !x || !d_x || rows <= 0 || dim <= 0 || head_dim <= 0) {
        return;
    }
    const int num_heads = dim / head_dim;
    if (num_heads <= 0 || num_heads * head_dim != dim) {
        return;
    }

    for (int row = 0; row < rows; ++row) {
        const float *d_row = d_out + (size_t) row * (size_t) dim;
        const float *x_row = x + (size_t) row * (size_t) dim;
        float *dx_row = d_x + (size_t) row * (size_t) dim;
        for (int head = 0; head < num_heads; ++head) {
            const float *d_head = d_row + (size_t) head * (size_t) head_dim;
            const float *x_head = x_row + (size_t) head * (size_t) head_dim;
            float *dx_head = dx_row + (size_t) head * (size_t) head_dim;

            float sum_sq = 0.0f;
            float dot = 0.0f;
            for (int col = 0; col < head_dim; ++col) {
                sum_sq += x_head[col] * x_head[col];
                dot += d_head[col] * x_head[col];
            }
            const float inv_norm = 1.0f / sqrtf(sum_sq + eps);
            const float proj_scale = inv_norm * inv_norm * inv_norm * dot;
            for (int col = 0; col < head_dim; ++col) {
                dx_head[col] = inv_norm * d_head[col] - proj_scale * x_head[col];
            }
        }
    }
}

void recurrent_qk_l2_norm_forward(float *q,
                                  float *k,
                                  int rows,
                                  int q_dim,
                                  int k_dim,
                                  int head_dim,
                                  float eps) {
    recurrent_l2_norm_rows_forward_one(q, rows, q_dim, head_dim, eps);
    recurrent_l2_norm_rows_forward_one(k, rows, k_dim, head_dim, eps);
}

void recurrent_qk_l2_norm_backward(const float *d_q_out,
                                   const float *d_k_out,
                                   const float *q,
                                   const float *k,
                                   float *d_q,
                                   float *d_k,
                                   int rows,
                                   int q_dim,
                                   int k_dim,
                                   int head_dim,
                                   float eps) {
    recurrent_l2_norm_rows_backward_one(d_q_out, q, d_q, rows, q_dim, head_dim, eps);
    recurrent_l2_norm_rows_backward_one(d_k_out, k, d_k, rows, k_dim, head_dim, eps);
}
