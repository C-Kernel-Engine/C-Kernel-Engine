/*
 * Chunked BF16 gated DeltaNet with a persistent FP32 recurrent state.
 * The provider uses 64-token temporal blocks and FP32 BLAS products.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "bf16_utils.h"
#include "ckernel_engine.h"

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { CHUNK = 64, HEAD_WIDTH = 128, MATRIX = CHUNK * HEAD_WIDTH };
enum { TEMPORAL = CHUNK * CHUNK, STATE = HEAD_WIDTH * HEAD_WIDTH };

typedef struct {
    float query[MATRIX], key[MATRIX], value_beta[MATRIX], key_beta[MATRIX];
    float weighted_key[MATRIX], decay[TEMPORAL], decay_input[TEMPORAL];
    float transform[TEMPORAL], value[MATRIX], cumulative_key[MATRIX];
    float weighted_query[MATRIX], value_new[MATRIX], previous_value[MATRIX];
    float history[MATRIX], query_key[TEMPORAL], local[MATRIX];
    float final_key[MATRIX], update[STATE];
    float cumulative_gate[CHUNK], gate_exp[CHUNK], final_delta[CHUNK], final_exp[CHUNK];
} chunk_workspace;

_Static_assert(sizeof(chunk_workspace) == 558080, "DeltaNet map workspace size");

typedef void (*sgemm_fn)(
    const char *, const char *, const int *, const int *, const int *,
    const float *, const float *, const int *, const float *, const int *,
    const float *, float *, const int *);
typedef void (*vsexp_fn)(int, const float *, float *);

static pthread_once_t bind_once = PTHREAD_ONCE_INIT;
static sgemm_fn sgemm;
static vsexp_fn vsexp;

static void bind_math(void)
{
    const char *library = getenv("CK_MKL_LIBRARY");
    void *handle = library && *library ? dlopen(library, RTLD_NOW | RTLD_LOCAL) : RTLD_DEFAULT;
    if (!handle && library && *library)
        return;
    sgemm = (sgemm_fn)dlsym(handle, "sgemm_");
    vsexp = (vsexp_fn)dlsym(handle, "vsExp");
}

static void matmul(int rows, int columns, int depth, const float *a, const float *b,
                   int transpose_a, int transpose_b, float *c)
{
    const char ta = transpose_a ? 'T' : 'N';
    const char tb = transpose_b ? 'T' : 'N';
    const int lda = transpose_a ? rows : depth;
    const int ldb = transpose_b ? depth : columns;
    const float one = 1.0f, zero = 0.0f;
    /* Fortran BLAS uses column-major storage. Exchange operands and axes. */
    sgemm(&tb, &ta, &columns, &rows, &depth, &one, b, &ldb, a, &lda,
          &zero, c, &columns);
}

static float bf16_round(float x)
{
    return bf16_to_float(float_to_bf16(x));
}

static float reduce_previous_rows(const float *matrix, int row, int column)
{
    const float *coefficients = matrix + row * CHUNK;
    if ((row < 8 && column < (row / 4) * 4) || (row >= 32 && column < 32)) {
        float blocks = 0.0f, sum = 0.0f;
        for (int i = 0; i < row; ++i) {
            if (i > column)
                sum += coefficients[i] * matrix[i * CHUNK + column];
            if (i % 16 == 15) {
                blocks += sum;
                sum = 0.0f;
            }
        }
        return sum + blocks;
    }

    /* CPU outer sum uses four interleaved accumulators for the remaining columns. */
    float partials[4] = {0};
    const int complete = (row / 4) * 4;
    for (int i = column + 1; i < complete; ++i)
        partials[i % 4] += coefficients[i] * matrix[i * CHUNK + column];
    for (int i = complete > column ? complete : column + 1; i < row; ++i)
        partials[0] += coefficients[i] * matrix[i * CHUNK + column];
    return ((partials[0] + partials[1]) + partials[2]) + partials[3];
}

static void temporal_transform(chunk_workspace *w)
{
    vsexp(CHUNK, w->cumulative_gate, w->gate_exp);
    for (int row = 0; row < CHUNK; ++row) {
        w->final_delta[row] = w->cumulative_gate[CHUNK - 1] - w->cumulative_gate[row];
        for (int column = 0; column < CHUNK; ++column)
            w->decay_input[row * CHUNK + column] = column <= row
                ? w->cumulative_gate[row] - w->cumulative_gate[column] : 0.0f;
    }
    vsexp(TEMPORAL, w->decay_input, w->decay);
    vsexp(CHUNK, w->final_delta, w->final_exp);
    matmul(CHUNK, CHUNK, HEAD_WIDTH, w->key_beta, w->key, 0, 1, w->transform);

    for (int row = 0; row < CHUNK; ++row) {
        for (int column = 0; column < CHUNK; ++column) {
            const int offset = row * CHUNK + column;
            if (column > row)
                w->decay[offset] = 0.0f;
            w->transform[offset] = column < row
                ? -w->transform[offset] * w->decay[offset] : 0.0f;
        }
    }
    /* Updating columns left to right keeps each row coefficient needed for
       later columns intact until its final use. */
    for (int row = 1; row < CHUNK; ++row)
        for (int column = 0; column < row; ++column)
            w->transform[row * CHUNK + column] += reduce_previous_rows(w->transform, row, column);
    for (int row = 0; row < CHUNK; ++row)
        w->transform[row * CHUNK + row] = 1.0f;
}

static void apply_chunk(chunk_workspace *w, float *state)
{
    temporal_transform(w);
    for (int row = 0; row < CHUNK; ++row) {
        for (int d = 0; d < HEAD_WIDTH; ++d) {
            const int at = row * HEAD_WIDTH + d;
            w->weighted_key[at] = w->key_beta[at] * w->gate_exp[row];
            w->weighted_query[at] = w->query[at] * w->gate_exp[row];
            w->final_key[at] = w->key[at] * w->final_exp[row];
        }
    }
    matmul(CHUNK, HEAD_WIDTH, CHUNK, w->transform, w->value_beta, 0, 0, w->value);
    matmul(CHUNK, HEAD_WIDTH, CHUNK, w->transform, w->weighted_key, 0, 0, w->cumulative_key);
    matmul(CHUNK, HEAD_WIDTH, HEAD_WIDTH, w->cumulative_key, state, 0, 0, w->previous_value);
    for (int i = 0; i < MATRIX; ++i)
        w->value_new[i] = w->value[i] - w->previous_value[i];

    matmul(CHUNK, HEAD_WIDTH, HEAD_WIDTH, w->weighted_query, state, 0, 0, w->history);
    matmul(CHUNK, CHUNK, HEAD_WIDTH, w->query, w->key, 0, 1, w->query_key);
    for (int i = 0; i < TEMPORAL; ++i)
        w->query_key[i] *= w->decay[i];
    matmul(CHUNK, HEAD_WIDTH, CHUNK, w->query_key, w->value_new, 0, 0, w->local);

    matmul(HEAD_WIDTH, HEAD_WIDTH, CHUNK, w->final_key, w->value_new, 1, 0, w->update);
    for (int i = 0; i < STATE; ++i)
        state[i] = state[i] * w->gate_exp[CHUNK - 1] + w->update[i];
}

void gated_deltanet_chunked_bf16_prefill_forward(
    const float *q, const float *k, const float *v, const float *packed_qkv,
    const float *g, const float *beta, const float *state_in, float *state_out,
    float *out, void *workspace, int rows, int heads, int groups, int width, float eps)
{
    pthread_once(&bind_once, bind_math);
    if (!sgemm || !vsexp || !q || !k || !v || !packed_qkv || !g || !beta ||
        !state_in || !state_out || !out || !workspace || rows < 1 || heads < 1 ||
        groups < 1 || heads % groups || width != HEAD_WIDTH) {
        fprintf(stderr, "HARD KERNEL CONTRACT FAULT: chunked BF16 DeltaNet requires MKL sgemm_ and vsExp, 128-wide heads and valid buffers\n");
        abort();
    }

    chunk_workspace *w = workspace;
    const int repeat = heads / groups;
    const size_t compact_row = (size_t)groups * HEAD_WIDTH;
    const size_t value_row = (size_t)heads * HEAD_WIDTH;
    const size_t packed_row = 2 * compact_row + value_row;
    const size_t vector_limit = (size_t)rows * heads / 32 * 32;
    const size_t boundary_row = vector_limit / heads;
    const int boundary_head = (int)(vector_limit % heads);
    const int boundary_group = boundary_head / repeat;
    const int mixed_group = boundary_head % repeat;
    const float scale = (float)(1.0 / sqrt((double)HEAD_WIDTH));
    float scalar_q[HEAD_WIDTH], scalar_k[HEAD_WIDTH];
    if (mixed_group) {
        const float *raw = packed_qkv + boundary_row * packed_row + (size_t)boundary_group * HEAD_WIDTH;
        memcpy(scalar_q, raw, sizeof(scalar_q));
        memcpy(scalar_k, raw + compact_row, sizeof(scalar_k));
        /* The repeated group can straddle the vector end of BF16 rsqrt. */
        recurrent_qk_l2_norm_pytorch_bf16_storage(
            scalar_q, scalar_k, 1, HEAD_WIDTH, HEAD_WIDTH, 1, HEAD_WIDTH, eps);
    }

    for (int head = 0; head < heads; ++head) {
        const int group = head / repeat;
        const size_t compact_offset = (size_t)group * HEAD_WIDTH;
        const size_t head_offset = (size_t)head * HEAD_WIDTH;
        const float *initial = state_in + (size_t)head * STATE;
        float *state = state_out + (size_t)head * STATE;
        if (state != initial)
            memcpy(state, initial, STATE * sizeof(float));
        for (int start = 0; start < rows;) {
            const int valid = rows - start < CHUNK ? rows - start : CHUNK;
            double cumulative = 0.0;
            for (int row = 0; row < CHUNK; ++row) {
                const size_t token = (size_t)start + row;
                const int present = row < valid;
                const size_t gate_index = token * heads + head;
                if (present)
                    cumulative += g[gate_index];
                w->cumulative_gate[row] = (float)cumulative;
                const float factor = present
                    ? bf16_round(1.0f / (1.0f + expf(-beta[gate_index]))) : 0.0f;
                const int scalar = mixed_group && token == boundary_row &&
                    group == boundary_group && head >= boundary_head;
                for (int d = 0; d < HEAD_WIDTH; ++d) {
                    const int at = row * HEAD_WIDTH + d;
                    const size_t compact = token * compact_row + compact_offset + d;
                    const size_t expanded = token * value_row + head_offset + d;
                    w->query[at] = present ? (scalar ? scalar_q[d] : q[compact]) * scale : 0.0f;
                    w->key[at] = present ? (scalar ? scalar_k[d] : k[compact]) : 0.0f;
                    w->value_beta[at] = present ? v[expanded] * factor : 0.0f;
                    w->key_beta[at] = w->key[at] * factor;
                }
            }
            apply_chunk(w, state);
            for (int row = 0; row < valid; ++row) {
                float *destination = out + ((size_t)start + row) * value_row + head_offset;
                for (int d = 0; d < HEAD_WIDTH; ++d) {
                    const int at = row * HEAD_WIDTH + d;
                    destination[d] = bf16_round(w->history[at] + w->local[at]);
                }
            }
            start += valid;
        }
    }
}
