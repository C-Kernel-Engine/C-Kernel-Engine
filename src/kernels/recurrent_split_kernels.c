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

void split_qkv_packed_head_major_forward(const float *packed_qkv,
                                         float *q,
                                         float *k,
                                         float *v,
                                         int rows,
                                         int q_dim,
                                         int k_dim,
                                         int v_dim,
                                         int num_heads,
                                         int num_kv_heads) {
    if (!packed_qkv || !q || !k || !v || rows <= 0 || q_dim <= 0 || k_dim <= 0 || v_dim <= 0 ||
        num_heads <= 0 || num_kv_heads <= 0) {
        return;
    }

    const int q_head_dim = q_dim / num_heads;
    const int k_head_dim = k_dim / num_kv_heads;
    const int v_head_dim = v_dim / num_kv_heads;
    if (q_head_dim <= 0 || k_head_dim <= 0 || v_head_dim <= 0) {
        return;
    }
    if (q_head_dim * num_heads != q_dim || k_head_dim * num_kv_heads != k_dim || v_head_dim * num_kv_heads != v_dim) {
        return;
    }

    const int packed_dim = q_dim + k_dim + v_dim;
    const size_t q_head_stride = (size_t) rows * (size_t) q_head_dim;
    const size_t k_head_stride = (size_t) rows * (size_t) k_head_dim;
    const size_t v_head_stride = (size_t) rows * (size_t) v_head_dim;

    for (int row = 0; row < rows; ++row) {
        const float *src = packed_qkv + (size_t) row * (size_t) packed_dim;

        for (int head = 0; head < num_heads; ++head) {
            const float *src_q = src + (size_t) head * (size_t) q_head_dim;
            float *dst_q = q + (size_t) head * q_head_stride + (size_t) row * (size_t) q_head_dim;
            memcpy(dst_q, src_q, (size_t) q_head_dim * sizeof(float));
        }

        const float *src_k_base = src + (size_t) q_dim;
        const float *src_v_base = src + (size_t) q_dim + (size_t) k_dim;
        for (int head = 0; head < num_kv_heads; ++head) {
            const float *src_k = src_k_base + (size_t) head * (size_t) k_head_dim;
            const float *src_v = src_v_base + (size_t) head * (size_t) v_head_dim;
            float *dst_k = k + (size_t) head * k_head_stride + (size_t) row * (size_t) k_head_dim;
            float *dst_v = v + (size_t) head * v_head_stride + (size_t) row * (size_t) v_head_dim;
            memcpy(dst_k, src_k, (size_t) k_head_dim * sizeof(float));
            memcpy(dst_v, src_v, (size_t) v_head_dim * sizeof(float));
        }
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
