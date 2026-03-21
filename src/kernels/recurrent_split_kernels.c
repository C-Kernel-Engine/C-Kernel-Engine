#include "ckernel_engine.h"

#include <string.h>

void recurrent_split_qkv_forward(const float *packed_qkv,
                                 float *q,
                                 float *k,
                                 float *v,
                                 int rows,
                                 int q_dim,
                                 int k_dim,
                                 int v_dim) {
    const int packed_dim = q_dim + k_dim + v_dim;
    for (int row = 0; row < rows; ++row) {
        const float *src = packed_qkv + (size_t) row * (size_t) packed_dim;
        float *q_dst = q + (size_t) row * (size_t) q_dim;
        float *k_dst = k + (size_t) row * (size_t) k_dim;
        float *v_dst = v + (size_t) row * (size_t) v_dim;
        memcpy(q_dst, src, (size_t) q_dim * sizeof(float));
        memcpy(k_dst, src + q_dim, (size_t) k_dim * sizeof(float));
        memcpy(v_dst, src + q_dim + k_dim, (size_t) v_dim * sizeof(float));
    }
}

void recurrent_split_qkv_backward(const float *d_q,
                                  const float *d_k,
                                  const float *d_v,
                                  float *d_packed_qkv,
                                  int rows,
                                  int q_dim,
                                  int k_dim,
                                  int v_dim) {
    const int packed_dim = q_dim + k_dim + v_dim;
    for (int row = 0; row < rows; ++row) {
        const float *dq_src = d_q + (size_t) row * (size_t) q_dim;
        const float *dk_src = d_k + (size_t) row * (size_t) k_dim;
        const float *dv_src = d_v + (size_t) row * (size_t) v_dim;
        float *dst = d_packed_qkv + (size_t) row * (size_t) packed_dim;
        memcpy(dst, dq_src, (size_t) q_dim * sizeof(float));
        memcpy(dst + q_dim, dk_src, (size_t) k_dim * sizeof(float));
        memcpy(dst + q_dim + k_dim, dv_src, (size_t) v_dim * sizeof(float));
    }
}

void recurrent_split_conv_qkv_forward(const float *packed_qkv,
                                      float *q,
                                      float *k,
                                      float *v,
                                      int rows,
                                      int q_dim,
                                      int k_dim,
                                      int v_dim) {
    recurrent_split_qkv_forward(packed_qkv, q, k, v, rows, q_dim, k_dim, v_dim);
}

void recurrent_split_conv_qkv_backward(const float *d_q,
                                       const float *d_k,
                                       const float *d_v,
                                       float *d_packed_qkv,
                                       int rows,
                                       int q_dim,
                                       int k_dim,
                                       int v_dim) {
    recurrent_split_qkv_backward(d_q, d_k, d_v, d_packed_qkv, rows, q_dim, k_dim, v_dim);
}
