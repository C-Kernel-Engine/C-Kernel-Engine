#include "ckernel_engine.h"

#include <stdlib.h>
#include <string.h>

void recurrent_conv_state_update_forward(const float *state_in,
                                         const float *q,
                                         const float *k,
                                         const float *v,
                                         float *conv_x,
                                         float *state_out,
                                         int history_len,
                                         int num_seqs,
                                         int num_tokens,
                                         int q_dim,
                                         int k_dim,
                                         int v_dim) {
    const int channels = q_dim + k_dim + v_dim;
    const int total_len = history_len + num_tokens;
    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *state_seq = state_in + (size_t) seq * (size_t) channels * (size_t) history_len;
        float *conv_seq = conv_x + (size_t) seq * (size_t) channels * (size_t) total_len;
        float *state_out_seq = state_out + (size_t) seq * (size_t) channels * (size_t) history_len;
        for (int ch = 0; ch < channels; ++ch) {
            memcpy(
                conv_seq + (size_t) ch * (size_t) total_len,
                state_seq + (size_t) ch * (size_t) history_len,
                (size_t) history_len * sizeof(float));
        }

        for (int tok = 0; tok < num_tokens; ++tok) {
            const int row = seq * num_tokens + tok;
            const float *q_row = q + (size_t) row * (size_t) q_dim;
            const float *k_row = k + (size_t) row * (size_t) k_dim;
            const float *v_row = v + (size_t) row * (size_t) v_dim;
            for (int col = 0; col < q_dim; ++col) {
                conv_seq[(size_t) col * (size_t) total_len + (size_t) (history_len + tok)] = q_row[col];
            }
            for (int col = 0; col < k_dim; ++col) {
                conv_seq[(size_t) (q_dim + col) * (size_t) total_len + (size_t) (history_len + tok)] = k_row[col];
            }
            for (int col = 0; col < v_dim; ++col) {
                conv_seq[(size_t) (q_dim + k_dim + col) * (size_t) total_len + (size_t) (history_len + tok)] = v_row[col];
            }
        }

        for (int ch = 0; ch < channels; ++ch) {
            memcpy(
                state_out_seq + (size_t) ch * (size_t) history_len,
                conv_seq + (size_t) ch * (size_t) total_len + (size_t) num_tokens,
                (size_t) history_len * sizeof(float));
        }
    }
}

void recurrent_conv_state_update_backward(const float *d_conv_x,
                                          const float *d_state_out,
                                          float *d_state_in,
                                          float *d_q,
                                          float *d_k,
                                          float *d_v,
                                          int history_len,
                                          int num_seqs,
                                          int num_tokens,
                                          int q_dim,
                                          int k_dim,
                                          int v_dim) {
    const int channels = q_dim + k_dim + v_dim;
    const int total_len = history_len + num_tokens;
    float *d_conv_total = (float *) malloc((size_t) num_seqs * (size_t) total_len * (size_t) channels * sizeof(float));
    if (!d_conv_total) {
        return;
    }

    memcpy(d_conv_total, d_conv_x, (size_t) num_seqs * (size_t) total_len * (size_t) channels * sizeof(float));

    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *d_state_out_seq = d_state_out + (size_t) seq * (size_t) channels * (size_t) history_len;
        float *d_conv_seq = d_conv_total + (size_t) seq * (size_t) channels * (size_t) total_len;
        for (int ch = 0; ch < channels; ++ch) {
            float *dst = d_conv_seq + (size_t) ch * (size_t) total_len + (size_t) num_tokens;
            const float *src = d_state_out_seq + (size_t) ch * (size_t) history_len;
            for (int idx = 0; idx < history_len; ++idx) {
                dst[idx] += src[idx];
            }
        }
    }

    for (int seq = 0; seq < num_seqs; ++seq) {
        const float *d_conv_seq = d_conv_total + (size_t) seq * (size_t) channels * (size_t) total_len;
        float *d_state_in_seq = d_state_in + (size_t) seq * (size_t) channels * (size_t) history_len;

        for (int ch = 0; ch < channels; ++ch) {
            memcpy(
                d_state_in_seq + (size_t) ch * (size_t) history_len,
                d_conv_seq + (size_t) ch * (size_t) total_len,
                (size_t) history_len * sizeof(float));
        }

        for (int tok = 0; tok < num_tokens; ++tok) {
            const int row = seq * num_tokens + tok;
            float *d_q_row = d_q + (size_t) row * (size_t) q_dim;
            float *d_k_row = d_k + (size_t) row * (size_t) k_dim;
            float *d_v_row = d_v + (size_t) row * (size_t) v_dim;
            for (int col = 0; col < q_dim; ++col) {
                d_q_row[col] = d_conv_seq[(size_t) col * (size_t) total_len + (size_t) (history_len + tok)];
            }
            for (int col = 0; col < k_dim; ++col) {
                d_k_row[col] = d_conv_seq[(size_t) (q_dim + col) * (size_t) total_len + (size_t) (history_len + tok)];
            }
            for (int col = 0; col < v_dim; ++col) {
                d_v_row[col] = d_conv_seq[(size_t) (q_dim + k_dim + col) * (size_t) total_len + (size_t) (history_len + tok)];
            }
        }
    }

    free(d_conv_total);
}
