#include "ckernel_engine.h"

#include <math.h>
#include <string.h>

static inline float hybrid_sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

void split_q_gate_forward(const float *packed_qg,
                          float *q,
                          float *gate,
                          int rows,
                          int q_dim,
                          int gate_dim,
                          int group_dim) {
    const int packed_dim = q_dim + gate_dim;
    if (!packed_qg || !q || !gate || rows <= 0 || q_dim <= 0 || gate_dim <= 0) {
        return;
    }
    if (group_dim <= 0) {
        group_dim = q_dim;
    }
    const int q_groups = q_dim / group_dim;
    const int gate_group_dim = (q_groups > 0 && gate_dim % q_groups == 0) ? (gate_dim / q_groups) : gate_dim;
    for (int row = 0; row < rows; ++row) {
        const float *src = packed_qg + (size_t) row * (size_t) packed_dim;
        float *q_dst = q + (size_t) row * (size_t) q_dim;
        float *gate_dst = gate + (size_t) row * (size_t) gate_dim;
        if (q_groups > 0 && q_groups * group_dim == q_dim && q_groups * gate_group_dim == gate_dim) {
            for (int group = 0; group < q_groups; ++group) {
                const size_t src_group_off = (size_t) group * (size_t) (group_dim + gate_group_dim);
                memcpy(
                    q_dst + (size_t) group * (size_t) group_dim,
                    src + src_group_off,
                    (size_t) group_dim * sizeof(float));
                memcpy(
                    gate_dst + (size_t) group * (size_t) gate_group_dim,
                    src + src_group_off + (size_t) group_dim,
                    (size_t) gate_group_dim * sizeof(float));
            }
        } else {
            memcpy(q_dst, src, (size_t) q_dim * sizeof(float));
            memcpy(gate_dst, src + q_dim, (size_t) gate_dim * sizeof(float));
        }
    }
}

void split_q_gate_backward(const float *d_q,
                           const float *d_gate,
                           float *d_packed_qg,
                           int rows,
                           int q_dim,
                           int gate_dim,
                           int group_dim) {
    const int packed_dim = q_dim + gate_dim;
    if (!d_q || !d_gate || !d_packed_qg || rows <= 0 || q_dim <= 0 || gate_dim <= 0) {
        return;
    }
    if (group_dim <= 0) {
        group_dim = q_dim;
    }
    const int q_groups = q_dim / group_dim;
    const int gate_group_dim = (q_groups > 0 && gate_dim % q_groups == 0) ? (gate_dim / q_groups) : gate_dim;
    for (int row = 0; row < rows; ++row) {
        const float *dq_src = d_q + (size_t) row * (size_t) q_dim;
        const float *dg_src = d_gate + (size_t) row * (size_t) gate_dim;
        float *dst = d_packed_qg + (size_t) row * (size_t) packed_dim;
        if (q_groups > 0 && q_groups * group_dim == q_dim && q_groups * gate_group_dim == gate_dim) {
            for (int group = 0; group < q_groups; ++group) {
                const size_t dst_group_off = (size_t) group * (size_t) (group_dim + gate_group_dim);
                memcpy(
                    dst + dst_group_off,
                    dq_src + (size_t) group * (size_t) group_dim,
                    (size_t) group_dim * sizeof(float));
                memcpy(
                    dst + dst_group_off + (size_t) group_dim,
                    dg_src + (size_t) group * (size_t) gate_group_dim,
                    (size_t) gate_group_dim * sizeof(float));
            }
        } else {
            memcpy(dst, dq_src, (size_t) q_dim * sizeof(float));
            memcpy(dst + q_dim, dg_src, (size_t) gate_dim * sizeof(float));
        }
    }
}

void attn_gate_sigmoid_mul_forward(const float *x,
                                   const float *gate,
                                   float *out,
                                   int rows,
                                   int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *x_row = x + (size_t) row * (size_t) dim;
        const float *gate_row = gate + (size_t) row * (size_t) dim;
        float *out_row = out + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            out_row[col] = x_row[col] * hybrid_sigmoid(gate_row[col]);
        }
    }
}

void attn_gate_sigmoid_mul_backward(const float *d_out,
                                    const float *x,
                                    const float *gate,
                                    float *d_x,
                                    float *d_gate,
                                    int rows,
                                    int dim) {
    for (int row = 0; row < rows; ++row) {
        const float *d_out_row = d_out + (size_t) row * (size_t) dim;
        const float *x_row = x + (size_t) row * (size_t) dim;
        const float *gate_row = gate + (size_t) row * (size_t) dim;
        float *d_x_row = d_x + (size_t) row * (size_t) dim;
        float *d_gate_row = d_gate + (size_t) row * (size_t) dim;
        for (int col = 0; col < dim; ++col) {
            const float sig = hybrid_sigmoid(gate_row[col]);
            d_x_row[col] = d_out_row[col] * sig;
            d_gate_row[col] = d_out_row[col] * x_row[col] * sig * (1.0f - sig);
        }
    }
}
