#include "ckernel_engine.h"
#include "ck_threadpool.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const float *state_in;
    const float *q;
    const float *k;
    const float *v;
    float *conv_x;
    float *state_out;
    int history_len;
    int num_tokens;
    int q_dim;
    int k_dim;
    int v_dim;
    int channels;
    int total_len;
} ck_recurrent_conv_state_args_t;

static void ck_recurrent_conv_state_channel_range(int begin,
                                                  int end,
                                                  void *opaque) {
    const ck_recurrent_conv_state_args_t *args =
        (const ck_recurrent_conv_state_args_t *)opaque;

    const int first_seq = begin / args->channels;
    const int last_seq = (end - 1) / args->channels;
    for (int seq = first_seq; seq <= last_seq; ++seq) {
        int ch_begin = begin - seq * args->channels;
        int ch_end = end - seq * args->channels;
        if (ch_begin < 0) ch_begin = 0;
        if (ch_end > args->channels) ch_end = args->channels;

        const size_t state_seq_offset =
            (size_t)seq * (size_t)args->channels *
            (size_t)args->history_len;
        const size_t conv_seq_offset =
            (size_t)seq * (size_t)args->channels *
            (size_t)args->total_len;
        for (int ch = ch_begin; ch < ch_end; ++ch) {
            memcpy(
                args->conv_x + conv_seq_offset +
                    (size_t)ch * (size_t)args->total_len,
                args->state_in + state_seq_offset +
                    (size_t)ch * (size_t)args->history_len,
                (size_t)args->history_len * sizeof(float));
        }

        const int q_begin = ch_begin;
        const int q_end = ch_end < args->q_dim ? ch_end : args->q_dim;
        const int k_begin = ch_begin > args->q_dim ?
            ch_begin - args->q_dim : 0;
        const int k_end = ch_end < args->q_dim + args->k_dim ?
            ch_end - args->q_dim : args->k_dim;
        const int v_begin = ch_begin > args->q_dim + args->k_dim ?
            ch_begin - args->q_dim - args->k_dim : 0;
        const int v_end = ch_end - args->q_dim - args->k_dim < args->v_dim ?
            ch_end - args->q_dim - args->k_dim : args->v_dim;

        for (int tok = 0; tok < args->num_tokens; ++tok) {
            const int row = seq * args->num_tokens + tok;
            for (int col = q_begin; col < q_end; ++col) {
                args->conv_x[conv_seq_offset +
                    (size_t)col * (size_t)args->total_len +
                    (size_t)(args->history_len + tok)] =
                    args->q[(size_t)row * (size_t)args->q_dim + (size_t)col];
            }
            for (int col = k_begin; col < k_end; ++col) {
                args->conv_x[conv_seq_offset +
                    (size_t)(args->q_dim + col) * (size_t)args->total_len +
                    (size_t)(args->history_len + tok)] =
                    args->k[(size_t)row * (size_t)args->k_dim + (size_t)col];
            }
            for (int col = v_begin; col < v_end; ++col) {
                args->conv_x[conv_seq_offset +
                    (size_t)(args->q_dim + args->k_dim + col) *
                        (size_t)args->total_len +
                    (size_t)(args->history_len + tok)] =
                    args->v[(size_t)row * (size_t)args->v_dim + (size_t)col];
            }
        }

        for (int ch = ch_begin; ch < ch_end; ++ch) {
            memcpy(
                args->state_out + state_seq_offset +
                    (size_t)ch * (size_t)args->history_len,
                args->conv_x + conv_seq_offset +
                    (size_t)ch * (size_t)args->total_len +
                    (size_t)args->num_tokens,
                (size_t)args->history_len * sizeof(float));
        }
    }
}

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
    if (!state_in || !q || !k || !v || !conv_x || !state_out ||
        history_len < 0 || num_seqs <= 0 || num_tokens < 0 ||
        q_dim < 0 || k_dim < 0 || v_dim < 0 ||
        q_dim > INT_MAX - k_dim || q_dim + k_dim > INT_MAX - v_dim ||
        history_len > INT_MAX - num_tokens) {
        return;
    }
    const int channels = q_dim + k_dim + v_dim;
    const int total_len = history_len + num_tokens;
    if (channels <= 0 || total_len <= 0 || num_seqs > INT_MAX / channels) {
        return;
    }

    ck_recurrent_conv_state_args_t args = {
        state_in, q, k, v, conv_x, state_out,
        history_len, num_tokens, q_dim, k_dim, v_dim, channels, total_len,
    };
    const int jobs = num_seqs * channels;
    const size_t elements = (size_t)jobs * (size_t)num_tokens;
    const size_t minimum_elements_per_thread = 32768u;
    if (num_tokens <= 1 || elements < 2u * minimum_elements_per_thread) {
        ck_recurrent_conv_state_channel_range(0, jobs, &args);
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int active = pool ? ck_threadpool_n_threads(pool) : 1;
    size_t useful_threads =
        (elements + minimum_elements_per_thread - 1u) /
        minimum_elements_per_thread;
    if ((size_t)active > useful_threads) active = (int)useful_threads;
    if (active > jobs) active = jobs;

    if (active > 1 && ck_threadpool_thread_id(pool) <= 0) {
        ck_threadpool_parallel_for_n(
            pool, active, 0, jobs, 32,
            ck_recurrent_conv_state_channel_range, &args);
    } else {
        ck_recurrent_conv_state_channel_range(0, jobs, &args);
    }
}

static int recurrent_conv_backward_extents(int history_len,
                                           int num_seqs,
                                           int num_tokens,
                                           int q_dim,
                                           int k_dim,
                                           int v_dim,
                                           int *channels_out,
                                           int *total_len_out,
                                           size_t *elements_out) {
    if (history_len < 0 || num_seqs <= 0 || num_tokens < 0 || q_dim < 0 ||
        k_dim < 0 || v_dim < 0 || q_dim > INT_MAX - k_dim ||
        q_dim + k_dim > INT_MAX - v_dim || history_len > INT_MAX - num_tokens) {
        return 0;
    }
    const int channels = q_dim + k_dim + v_dim;
    const int total_len = history_len + num_tokens;
    if (channels == 0 || total_len == 0) {
        return 0;
    }
    size_t elements = (size_t)num_seqs;
    if ((size_t)total_len > SIZE_MAX / elements) {
        return 0;
    }
    elements *= (size_t)total_len;
    if ((size_t)channels > SIZE_MAX / elements) {
        return 0;
    }
    *channels_out = channels;
    *total_len_out = total_len;
    *elements_out = elements * (size_t)channels;
    return 1;
}

void recurrent_conv_state_update_backward_workspace(const float *d_conv_x,
                                                     const float *d_state_out,
                                                     float *d_state_in,
                                                     float *d_q,
                                                     float *d_k,
                                                     float *d_v,
                                                     float *d_conv_total,
                                                     int history_len,
                                                     int num_seqs,
                                                     int num_tokens,
                                                     int q_dim,
                                                     int k_dim,
                                                     int v_dim) {
    int channels = 0;
    int total_len = 0;
    size_t elements = 0;
    if (!d_conv_x || !d_state_out || !d_state_in || !d_q || !d_k || !d_v ||
        !d_conv_total || !recurrent_conv_backward_extents(
            history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim,
            &channels, &total_len, &elements)) {
        return;
    }
    if (elements > SIZE_MAX / sizeof(float)) {
        return;
    }

    memcpy(d_conv_total, d_conv_x, elements * sizeof(float));

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
    int channels = 0;
    int total_len = 0;
    size_t elements = 0;
    if (!recurrent_conv_backward_extents(
            history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim,
            &channels, &total_len, &elements) ||
        elements > SIZE_MAX / sizeof(float)) {
        return;
    }
    (void)channels;
    (void)total_len;
    float *workspace = (float *)malloc(elements * sizeof(float));
    if (!workspace) {
        return;
    }
    recurrent_conv_state_update_backward_workspace(
        d_conv_x, d_state_out, d_state_in, d_q, d_k, d_v, workspace,
        history_len, num_seqs, num_tokens, q_dim, k_dim, v_dim);
    free(workspace);
}
