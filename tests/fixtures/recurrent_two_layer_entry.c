/* Asset-free generated-entry composition fixture; the Python test inserts
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
static void ck_batch_stage_8(CKModel *model, int token) {
    (void)token;
    CKSequenceStateV8 *s = g_active_sequence;
    model->logits[0] = ((float *)(model->bump + A_EMBEDDED_INPUT))[0];
    model->logits[1] = (float)s->conv[0] + (float)s->conv[32];
    model->logits[2] = (float)s->ssm[0] + (float)s->ssm[32];
    model->logits[3] = ((float *)(model->bump + A_RESIDUAL))[0];
    model->logits[4] = (float)s->kv[0] + (float)s->kv[1];
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
    ck_batch_stage_8(g_model, token);
}

int main(void) {
    uint8_t kv[2][64] = {{0}}, conv[2][64] = {{0}}, ssm[2][64] = {{0}};
    uint8_t expected_kv[2][64], expected_conv[2][64], expected_ssm[2][64];
    float isolated[2][VOCAB_SIZE] = {{0}}, batched[2][VOCAB_SIZE] = {{0}};
    int expected_position[2];
    const int tokens[2] = {3, 7};
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
    for (int i = 0; i < 2; ++i) {
        if (ck_model_sequence_state_activate((uint64_t)(i + 1))) return 1;
        isolated_step(tokens[i]);
        memcpy(isolated[i], g_model->logits, sizeof(isolated[i]));
        memcpy(expected_kv[i], kv[i], 64);
        memcpy(expected_conv[i], conv[i], 64);
        memcpy(expected_ssm[i], ssm[i], 64);
        expected_position[i] = g_model->pos;
    }
    memset(kv, 0, sizeof(kv)); memset(conv, 0, sizeof(conv));
    memset(ssm, 0, sizeof(ssm));
    for (int i = 0; i < 2; ++i) g_sequence_states[i].pos = 0;
    g_active_sequence = NULL; g_model->pos = 0;
    CKModelBatchDecodeRowV8 rows[2] = {
        {.sequence_handle = 1, .token = 3, .position = 0,
         .row_offset = 0, .token_count = 1, .logits = batched[0]},
        {.sequence_handle = 2, .token = 7, .position = 0,
         .row_offset = 1, .token_count = 1, .logits = batched[1]},
    };
    size_t bytes = 0, alignment = 0;
    if (ck_model_batch_decode_workspace(&bytes, &alignment) || alignment != 64)
        return 2;
    void *workspace = aligned_alloc(alignment,
        (bytes + alignment - 1) / alignment * alignment);
    if (!workspace || ck_model_decode_batch2(rows, 2, workspace, bytes)) return 3;
    for (int i = 0; i < 2; ++i) {
        if (ck_model_sequence_state_activate((uint64_t)(i + 1)) ||
            memcmp(isolated[i], batched[i], sizeof(isolated[i])) ||
            memcmp(expected_kv[i], kv[i], 64) ||
            memcmp(expected_conv[i], conv[i], 64) ||
            memcmp(expected_ssm[i], ssm[i], 64) ||
            g_model->pos != expected_position[i]) return 4 + i;
    }
    free(workspace);
    return 0;
}
