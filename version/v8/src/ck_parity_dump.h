/**
 * @file ck_parity_dump.h
 * @brief Parity dump instrumentation for comparing CKE outputs with llama.cpp
 *
 * Usage:
 *   1. Compile with -DCK_PARITY_DUMP
 *   2. Call ck_dump_init() before running inference
 *   3. Call ck_dump_close() after inference
 *   4. Use parity_test.py to compare dumps
 *
 * File Format:
 *   - 128-byte header per tensor
 *   - Raw float32 data
 */

#ifndef CK_PARITY_DUMP_H
#define CK_PARITY_DUMP_H

#ifdef CK_PARITY_DUMP

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CKDUMP_MAGIC "CKDMP\0\0\0"
#define CKDUMP_VERSION 1

typedef struct {
    char magic[8];         /* "CKDMP\0\0\0" */
    uint32_t version;      /* = 1 */
    int32_t layer_id;      /* -1 for global ops, 0-27 for layers */
    char op_name[32];      /* e.g., "q_proj", "attn_out", "logits" */
    uint32_t dtype;        /* 0=fp32, 1=fp16 */
    uint32_t rank;         /* 1-4 */
    int64_t shape[4];      /* dimensions */
    uint32_t elem_count;   /* total elements */
    int32_t token_id;      /* current token being decoded */
    uint8_t reserved[32];  /* padding (header = 128 bytes) */
} __attribute__((packed)) CKDumpFileHeader;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(CKDumpFileHeader) == 128, "CKDumpFileHeader must be 128 bytes");
#endif

static FILE *g_ck_dump_file = NULL;
static int g_ck_dump_token = 0;

static inline int ck_dump_filter_token_matches(const char *filter, const char *candidate) {
    if (!filter || !filter[0] || !candidate || !candidate[0]) return 0;
    const char *p = filter;
    const size_t clen = strlen(candidate);
    while (*p) {
        while (*p == ',' || *p == ';' || *p == ' ' || *p == '\t' || *p == '\n') p++;
        const char *start = p;
        while (*p && *p != ',' && *p != ';' && *p != ' ' && *p != '\t' && *p != '\n') p++;
        const size_t len = (size_t)(p - start);
        if (len == clen && strncmp(start, candidate, clen) == 0) return 1;
    }
    return 0;
}

static inline int ck_dump_layer_allowed(int layer_id) {
    const char *filter = getenv("CK_PARITY_LAYER_FILTER");
    if (!filter || !filter[0]) return 1;
    if (ck_dump_filter_token_matches(filter, "all") || ck_dump_filter_token_matches(filter, "*")) return 1;

    char layer_buf[32];
    snprintf(layer_buf, sizeof(layer_buf), "%d", layer_id);
    return ck_dump_filter_token_matches(filter, layer_buf);
}

static inline int ck_dump_op_allowed(int layer_id, const char *op_name) {
    const char *filter = getenv("CK_PARITY_OP_FILTER");
    if (!filter || !filter[0]) return 1;
    if (ck_dump_filter_token_matches(filter, "all") || ck_dump_filter_token_matches(filter, "*")) return 1;
    if (ck_dump_filter_token_matches(filter, op_name)) return 1;

    char layered_name[96];
    snprintf(layered_name, sizeof(layered_name), "%s-%d", op_name, layer_id);
    if (ck_dump_filter_token_matches(filter, layered_name)) return 1;
    snprintf(layered_name, sizeof(layered_name), "%s:%d", op_name, layer_id);
    if (ck_dump_filter_token_matches(filter, layered_name)) return 1;
    return 0;
}

static inline int ck_dump_should_emit(int layer_id, const char *op_name) {
    const char *enabled = getenv("CK_PARITY_CAPTURE_ENABLED");
    if (enabled && enabled[0] && atoi(enabled) == 0) return 0;
    return ck_dump_layer_allowed(layer_id) && ck_dump_op_allowed(layer_id, op_name);
}

/**
 * Initialize dumping. Call before any inference.
 * @param dump_dir Directory to write dump.bin (default: uses CK_PARITY_DIR env, or "ck_parity_dumps")
 */
static inline void ck_dump_init(const char *dump_dir) {
    char path[512];

    // Use CK_PARITY_DIR environment variable if set, otherwise use provided dir
    const char *env_dir = getenv("CK_PARITY_DIR");
    if (env_dir && env_dir[0]) {
        snprintf(path, sizeof(path), "%s/dump.bin", env_dir);
    } else if (dump_dir) {
        snprintf(path, sizeof(path), "%s/dump.bin", dump_dir);
    } else {
        snprintf(path, sizeof(path), "ck_parity_dumps/dump.bin");
    }

    g_ck_dump_file = fopen(path, "wb");
    if (g_ck_dump_file) {
        fprintf(stderr, "[CK_DUMP] Writing to: %s\n", path);
    } else {
        fprintf(stderr, "[CK_DUMP] Failed to open: %s\n", path);
    }
}

/**
 * Set current token ID for tracking.
 */
static inline void ck_dump_set_token(int token_id) {
    g_ck_dump_token = token_id;
}

/**
 * Dump a tensor to file.
 * @param data Pointer to float32 data
 * @param layer_id Layer index (-1 for global ops like logits)
 * @param op_name Operation name (e.g., "q_proj", "attn_out")
 * @param elem_count Number of elements
 */
static inline void ck_dump_tensor(
    const float *data,
    int layer_id,
    const char *op_name,
    int elem_count
) {
    if (!g_ck_dump_file || !data) return;
    if (!ck_dump_should_emit(layer_id, op_name)) return;

    CKDumpFileHeader header = {0};
    memcpy(header.magic, CKDUMP_MAGIC, 8);
    header.version = CKDUMP_VERSION;
    header.layer_id = layer_id;
    strncpy(header.op_name, op_name, 31);
    header.op_name[31] = '\0';
    header.dtype = 0;  /* fp32 */
    header.rank = 1;
    header.shape[0] = elem_count;
    header.elem_count = elem_count;
    header.token_id = g_ck_dump_token;

    fwrite(&header, sizeof(CKDumpFileHeader), 1, g_ck_dump_file);
    fwrite(data, elem_count * sizeof(float), 1, g_ck_dump_file);
    fflush(g_ck_dump_file);
}

/**
 * Dump a 2D tensor (e.g., attention matrix).
 */
static inline void ck_dump_tensor_2d(
    const float *data,
    int layer_id,
    const char *op_name,
    int dim0,
    int dim1
) {
    if (!g_ck_dump_file || !data) return;
    if (!ck_dump_should_emit(layer_id, op_name)) return;

    CKDumpFileHeader header = {0};
    memcpy(header.magic, CKDUMP_MAGIC, 8);
    header.version = CKDUMP_VERSION;
    header.layer_id = layer_id;
    strncpy(header.op_name, op_name, 31);
    header.op_name[31] = '\0';
    header.dtype = 0;
    header.rank = 2;
    header.shape[0] = dim0;
    header.shape[1] = dim1;
    header.elem_count = dim0 * dim1;
    header.token_id = g_ck_dump_token;

    fwrite(&header, sizeof(CKDumpFileHeader), 1, g_ck_dump_file);
    fwrite(data, dim0 * dim1 * sizeof(float), 1, g_ck_dump_file);
    fflush(g_ck_dump_file);
}

/**
 * Dump a 3D head-major tensor after reordering it to token-major logical order.
 *
 * Source layout:
 *   data[head][token][dim]
 *
 * Dumped logical flatten order:
 *   data[token][head][dim]
 *
 * This is used for parity on attention Q/K/V buffers, where CK stores
 * head-major scratch but llama.cpp checkpoint views flatten logically by token.
 */
static inline void ck_dump_tensor_head_major_token_major_strided(
    const float *data,
    int layer_id,
    const char *op_name,
    int num_heads,
    int num_tokens,
    int head_dim,
    int physical_head_dim
) {
    if (!g_ck_dump_file || !data || num_heads <= 0 || num_tokens <= 0 ||
        head_dim <= 0 || physical_head_dim < head_dim) return;
    if (!ck_dump_should_emit(layer_id, op_name)) return;

    const size_t heads = (size_t) num_heads;
    const size_t tokens = (size_t) num_tokens;
    const size_t width = (size_t) head_dim;
    const size_t stride = (size_t) physical_head_dim;
    if (heads > SIZE_MAX / tokens || heads * tokens > SIZE_MAX / stride ||
        heads * tokens > UINT32_MAX / width ||
        heads * tokens * width > SIZE_MAX / sizeof(float)) return;
    const size_t elem_count = heads * tokens * width;
    float *tmp = (float *) malloc(elem_count * sizeof(float));
    if (!tmp) return;

    size_t dst = 0;
    for (int token = 0; token < num_tokens; ++token) {
        for (int head = 0; head < num_heads; ++head) {
            const float *src = data +
                (size_t) head * tokens * stride +
                (size_t) token * stride;
            memcpy(&tmp[dst], src, (size_t) head_dim * sizeof(float));
            dst += (size_t) head_dim;
        }
    }

    CKDumpFileHeader header = {0};
    memcpy(header.magic, CKDUMP_MAGIC, 8);
    header.version = CKDUMP_VERSION;
    header.layer_id = layer_id;
    strncpy(header.op_name, op_name, 31);
    header.op_name[31] = '\0';
    header.dtype = 0;  /* fp32 */
    header.rank = 1;
    header.shape[0] = (int64_t) elem_count;
    header.elem_count = (uint32_t) elem_count;
    header.token_id = g_ck_dump_token;

    fwrite(&header, sizeof(CKDumpFileHeader), 1, g_ck_dump_file);
    fwrite(tmp, elem_count * sizeof(float), 1, g_ck_dump_file);
    fflush(g_ck_dump_file);
    free(tmp);
}

static inline void ck_dump_tensor_head_major_token_major(
    const float *data, int layer_id, const char *op_name,
    int num_heads, int num_tokens, int head_dim
) {
    ck_dump_tensor_head_major_token_major_strided(
        data, layer_id, op_name, num_heads, num_tokens, head_dim, head_dim
    );
}

/**
 * Close dump file. Call after inference completes.
 */
static inline void ck_dump_close(void) {
    if (g_ck_dump_file) {
        fclose(g_ck_dump_file);
        g_ck_dump_file = NULL;
        fprintf(stderr, "[CK_DUMP] Closed\n");
    }
}

#ifdef __cplusplus
}
#endif

#else  /* !CK_PARITY_DUMP */

#define ck_dump_init(dir)
#define ck_dump_set_token(token)
#define ck_dump_tensor(data, layer, name, count)
#define ck_dump_tensor_2d(data, layer, name, d0, d1)
#define ck_dump_tensor_head_major_token_major(data, layer, name, h, t, d)
#define ck_dump_tensor_head_major_token_major_strided(data, layer, name, h, t, d, stride)
#define ck_dump_close()

#endif  /* CK_PARITY_DUMP */

#endif  /* CK_PARITY_DUMP_H */
