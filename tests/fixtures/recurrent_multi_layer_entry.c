/* Asset-free two/three-layer composition fixture. The Python test inserts
 * the real emitted batch entry at BATCH_API below. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#define CK_EXPORT
#define VOCAB_SIZE 16
#define MAX_SEQ_LEN 16
#define KV_CACHE_SIZE 64
#define A_EMBEDDED_INPUT 128
#define A_RESIDUAL 1152
#define A_MAIN_STREAM_Q8 2176
#define A_RECURRENT_PACKED 2496
#define A_RECURRENT_NORMED 3008
#define A_RECURRENT_Z 3200
#define W_LAYER_0_ATTN_QKV 3400
#define W_LAYER_0_ATTN_GATE 3416
#define W_LAYER_0_SSM_OUT 3432
#define W_LAYER_1_ATTN_QKV 3448
#define W_LAYER_1_ATTN_GATE 3464
#define W_LAYER_1_SSM_OUT 3480
#define W_LAYER_2_ATTN_QKV 3496
#define W_LAYER_2_ATTN_GATE 3512
#define W_LAYER_2_SSM_OUT 3528
#ifdef RECUR_THREE_LAYER
#define STATE_BYTES 96
#else
#define STATE_BYTES 64
#endif

typedef struct {
    uint64_t sequence_handle;
    int token, position;
    size_t row_offset, token_count;
    float *logits;
} CKModelBatchDecodeRowV8;
typedef struct {
    uint8_t bump[4096];
    size_t bump_size;
    int pos;
    float logits[VOCAB_SIZE];
} CKModel;
typedef struct {
    int occupied, pos;
    uint64_t handle;
    uint8_t *kv, *conv, *ssm;
} CKSequenceStateV8;
static CKModel object = {.bump_size = 4096}, *g_model = &object;
static CKSequenceStateV8 g_sequence_states[3], *g_active_sequence;
static int g_ck_skip_decode_logits;

static CKSequenceStateV8 *ck_sequence_find(uint64_t handle) {
    for (int i = 0; i < 3; ++i)
        if (g_sequence_states[i].occupied && g_sequence_states[i].handle == handle)
            return &g_sequence_states[i];
    return NULL;
}
static int ck_model_sequence_state_activate(uint64_t handle) {
    CKSequenceStateV8 *next = ck_sequence_find(handle);
    if (!next) return -1;
    if (g_active_sequence && g_active_sequence != next)
        g_active_sequence->pos = g_model->pos;
    if (g_active_sequence != next) {
        g_model->pos = next->pos;
        g_active_sequence = next;
    }
    return 0;
}
static int ck_model_cancel_requested(void) { return 0; }

static void gemm_nt_q6_k_q8_k_m2(const void *input, const void *weight,
                                  const float *bias, float *output,
                                  int rows, int channels, int width) {
    (void)bias; (void)width;
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < channels; ++col)
            output[row * channels + col] =
                (float)(((const uint8_t *)input)[row * 292] +
                        ((const uint8_t *)weight)[0] + col);
}
static void gemm_nt_q4_k_q8_k_decode_m2(const void *input, const void *weight,
                                         const float *bias, float *output,
                                         int rows, int channels, int width) {
    (void)bias; (void)width;
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < channels; ++col)
            output[row * channels + col] =
                (float)(((const uint8_t *)input)[row * 292] +
                        ((const uint8_t *)weight)[0] + col + 5);
}
static void gemm_nt_q8_0_q8_0_contract_m2(const void *input, const void *weight,
                                           const float *bias, float *output,
                                           int rows, int channels, int width) {
    (void)bias;
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < channels; ++col)
            output[row * channels + col] =
                ((const float *)input)[row * width] +
                (float)((const uint8_t *)weight)[0] + (float)col;
}

static void ck_batch_stage_0(CKModel *model, int token) {
    ((float *)(model->bump + A_EMBEDDED_INPUT))[0] = (float)token;
    ((float *)(model->bump + A_RESIDUAL))[0] = (float)(token + 20);
    model->bump[A_MAIN_STREAM_Q8] = (uint8_t)(token + 1);
}
static void ck_batch_stage_2(CKModel *model, int token) {
    (void)token;
    CKSequenceStateV8 *s = g_active_sequence;
    float q = ((float *)(model->bump + A_RECURRENT_PACKED))[0];
    float gate = ((float *)(model->bump + A_RECURRENT_Z))[0];
    s->conv[0] += (uint8_t)q;
    s->ssm[0] += (uint8_t)gate;
    s->kv[0] = (uint8_t)(q + gate);
    ((float *)(model->bump + A_RECURRENT_NORMED))[0] =
        q + gate + (float)s->conv[0] + (float)s->ssm[0] +
        ((float *)(model->bump + A_RESIDUAL))[0];
}
static void ck_batch_stage_4(CKModel *model, int token) {
    (void)token;
    float hidden = ((float *)(model->bump + A_EMBEDDED_INPUT))[0];
    ((float *)(model->bump + A_RESIDUAL))[0] = hidden;
    model->bump[A_MAIN_STREAM_Q8] = (uint8_t)((int)hidden % 251);
}
static void ck_batch_stage_6(CKModel *model, int token) {
    (void)token;
    CKSequenceStateV8 *s = g_active_sequence;
    float q = ((float *)(model->bump + A_RECURRENT_PACKED))[0];
    float gate = ((float *)(model->bump + A_RECURRENT_Z))[0];
    s->conv[32] += (uint8_t)q;
    s->ssm[32] += (uint8_t)gate;
    s->kv[1] = (uint8_t)(q + gate);
    ((float *)(model->bump + A_RECURRENT_NORMED))[0] =
        q + gate + (float)s->conv[32] + (float)s->ssm[32] +
        ((float *)(model->bump + A_RESIDUAL))[0];
}
#ifdef RECUR_THREE_LAYER
static void ck_batch_stage_8(CKModel *model, int token) {
    (void)token;
    float hidden = ((float *)(model->bump + A_EMBEDDED_INPUT))[0];
    ((float *)(model->bump + A_RESIDUAL))[0] = hidden;
    model->bump[A_MAIN_STREAM_Q8] = (uint8_t)((int)hidden % 251);
}
static void ck_batch_stage_10(CKModel *model, int token) {
    (void)token;
    CKSequenceStateV8 *s = g_active_sequence;
    float q = ((float *)(model->bump + A_RECURRENT_PACKED))[0];
    float gate = ((float *)(model->bump + A_RECURRENT_Z))[0];
    s->conv[64] += (uint8_t)q;
    s->ssm[64] += (uint8_t)gate;
    s->kv[2] = (uint8_t)(q + gate);
    ((float *)(model->bump + A_RECURRENT_NORMED))[0] =
        q + gate + (float)s->conv[64] + (float)s->ssm[64] +
        ((float *)(model->bump + A_RESIDUAL))[0];
}
static void ck_batch_stage_12(CKModel *model, int token) {
#else
static void ck_batch_stage_8(CKModel *model, int token) {
#endif
    (void)token;
    CKSequenceStateV8 *s = g_active_sequence;
    memset(model->logits, 0, sizeof(model->logits));
    model->logits[0] = ((float *)(model->bump + A_EMBEDDED_INPUT))[0];
    model->logits[1] = (float)s->conv[0] + (float)s->conv[32];
    model->logits[2] = (float)s->ssm[0] + (float)s->ssm[32];
    model->logits[3] = ((float *)(model->bump + A_RESIDUAL))[0];
    model->logits[4] = (float)s->kv[0] + (float)s->kv[1];
#ifdef RECUR_THREE_LAYER
    model->logits[5] = (float)s->conv[64] + (float)s->ssm[64] + (float)s->kv[2];
#endif
    ++model->pos;
    s->pos = model->pos;
}

/* BATCH_API */

static void isolated_step(int token) {
    ck_batch_stage_0(g_model, token);
    gemm_nt_q6_k_q8_k_m2(g_model->bump + A_MAIN_STREAM_Q8,
        g_model->bump + W_LAYER_0_ATTN_QKV, NULL,
        (float *)(g_model->bump + A_RECURRENT_PACKED), 1, 16, 256);
    gemm_nt_q4_k_q8_k_decode_m2(g_model->bump + A_MAIN_STREAM_Q8,
        g_model->bump + W_LAYER_0_ATTN_GATE, NULL,
        (float *)(g_model->bump + A_RECURRENT_Z), 1, 16, 256);
    ck_batch_stage_2(g_model, token);
    gemm_nt_q8_0_q8_0_contract_m2(g_model->bump + A_RECURRENT_NORMED,
        g_model->bump + W_LAYER_0_SSM_OUT, NULL,
        (float *)(g_model->bump + A_EMBEDDED_INPUT), 1, 16, 32);
    ck_batch_stage_4(g_model, token);
    gemm_nt_q6_k_q8_k_m2(g_model->bump + A_MAIN_STREAM_Q8,
        g_model->bump + W_LAYER_1_ATTN_QKV, NULL,
        (float *)(g_model->bump + A_RECURRENT_PACKED), 1, 16, 256);
    gemm_nt_q4_k_q8_k_decode_m2(g_model->bump + A_MAIN_STREAM_Q8,
        g_model->bump + W_LAYER_1_ATTN_GATE, NULL,
        (float *)(g_model->bump + A_RECURRENT_Z), 1, 16, 256);
    ck_batch_stage_6(g_model, token);
    gemm_nt_q8_0_q8_0_contract_m2(g_model->bump + A_RECURRENT_NORMED,
        g_model->bump + W_LAYER_1_SSM_OUT, NULL,
        (float *)(g_model->bump + A_EMBEDDED_INPUT), 1, 16, 32);
#ifdef RECUR_THREE_LAYER
    ck_batch_stage_8(g_model, token);
    gemm_nt_q6_k_q8_k_m2(g_model->bump + A_MAIN_STREAM_Q8,
        g_model->bump + W_LAYER_2_ATTN_QKV, NULL,
        (float *)(g_model->bump + A_RECURRENT_PACKED), 1, 16, 256);
    gemm_nt_q4_k_q8_k_decode_m2(g_model->bump + A_MAIN_STREAM_Q8,
        g_model->bump + W_LAYER_2_ATTN_GATE, NULL,
        (float *)(g_model->bump + A_RECURRENT_Z), 1, 16, 256);
    ck_batch_stage_10(g_model, token);
    gemm_nt_q8_0_q8_0_contract_m2(g_model->bump + A_RECURRENT_NORMED,
        g_model->bump + W_LAYER_2_SSM_OUT, NULL,
        (float *)(g_model->bump + A_EMBEDDED_INPUT), 1, 16, 32);
    ck_batch_stage_12(g_model, token);
#else
    ck_batch_stage_8(g_model, token);
#endif
}

static void seed_states(uint8_t kv[2][64], uint8_t conv[2][STATE_BYTES],
                        uint8_t ssm[2][STATE_BYTES]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < STATE_BYTES; ++j) {
            conv[i][j] = (uint8_t)(5 + i * 19 + j % 11);
            ssm[i][j] = (uint8_t)(7 + i * 23 + j % 13);
        }
        for (int j = 0; j < 64; ++j)
            kv[i][j] = (uint8_t)(3 + i * 17 + j % 7);
        g_sequence_states[i].pos = i == 0 ? 2 : 5;
    }
    g_active_sequence = NULL;
    g_model->pos = 0;
}

int main(void) {
    uint8_t kv[2][64] = {{0}}, conv[2][STATE_BYTES] = {{0}};
    uint8_t ssm[2][STATE_BYTES] = {{0}};
    uint8_t expected_kv[3][2][64], expected_conv[3][2][STATE_BYTES];
    uint8_t expected_ssm[3][2][STATE_BYTES];
    float isolated[3][2][VOCAB_SIZE] = {{{0}}};
    float batched[3][2][VOCAB_SIZE] = {{{0}}};
    int expected_position[3][2];
    const int tokens[3][2] = {{3, 7}, {5, 11}, {9, 4}};
    for (int i = 0; i < 2; ++i)
        g_sequence_states[i] = (CKSequenceStateV8){
            .occupied = 1, .handle = (uint64_t)(i + 1),
            .kv = kv[i], .conv = conv[i], .ssm = ssm[i]};
    g_model->bump[W_LAYER_0_ATTN_QKV] = 1;
    g_model->bump[W_LAYER_0_ATTN_GATE] = 2;
    g_model->bump[W_LAYER_0_SSM_OUT] = 3;
    g_model->bump[W_LAYER_1_ATTN_QKV] = 4;
    g_model->bump[W_LAYER_1_ATTN_GATE] = 5;
    g_model->bump[W_LAYER_1_SSM_OUT] = 6;
#ifdef RECUR_THREE_LAYER
    g_model->bump[W_LAYER_2_ATTN_QKV] = 7;
    g_model->bump[W_LAYER_2_ATTN_GATE] = 8;
    g_model->bump[W_LAYER_2_SSM_OUT] = 9;
#endif
    seed_states(kv, conv, ssm);
    for (int step = 0; step < 3; ++step) {
        for (int i = 0; i < 2; ++i) {
            if (ck_model_sequence_state_activate((uint64_t)(i + 1))) return 1;
            isolated_step(tokens[step][i]);
            memcpy(isolated[step][i], g_model->logits,
                   sizeof(isolated[step][i]));
            memcpy(expected_kv[step][i], kv[i], 64);
            memcpy(expected_conv[step][i], conv[i], STATE_BYTES);
            memcpy(expected_ssm[step][i], ssm[i], STATE_BYTES);
            expected_position[step][i] = g_model->pos;
        }
    }
    seed_states(kv, conv, ssm);
    size_t bytes = 0, alignment = 0;
    if (ck_model_batch_decode_workspace(&bytes, &alignment) || alignment != 64)
        return 2;
    void *workspace = aligned_alloc(alignment,
        (bytes + alignment - 1) / alignment * alignment);
    if (!workspace) return 3;
    for (int step = 0; step < 3; ++step) {
        CKModelBatchDecodeRowV8 rows[2];
        for (int row = 0; row < 2; ++row) {
            int i = step % 2 ? 1 - row : row;
            rows[row] = (CKModelBatchDecodeRowV8){
                .sequence_handle = (uint64_t)(i + 1),
                .token = tokens[step][i],
                .position = g_sequence_states[i].pos,
                .row_offset = (size_t)row, .token_count = 1,
                .logits = batched[step][i]};
        }
        if (ck_model_decode_batch2(rows, 2, workspace, bytes)) return 3;
        for (int i = 0; i < 2; ++i) {
            if (ck_model_sequence_state_activate((uint64_t)(i + 1)) ||
                memcmp(isolated[step][i], batched[step][i],
                       sizeof(isolated[step][i])) ||
                memcmp(expected_kv[step][i], kv[i], 64) ||
                memcmp(expected_conv[step][i], conv[i], STATE_BYTES) ||
                memcmp(expected_ssm[step][i], ssm[i], STATE_BYTES) ||
                g_model->pos != expected_position[step][i])
                return 4 + step * 2 + i;
        }
    }
    free(workspace);
    return 0;
}
