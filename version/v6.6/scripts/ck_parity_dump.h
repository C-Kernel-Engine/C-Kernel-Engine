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

#ifdef __cplusplus
extern "C" {
#endif

#define CKDUMP_MAGIC "CKDMP\0\0"
#define CKDUMP_VERSION 1

typedef struct {
    char magic[8];         /* "CKDMP\0\0" */
    uint32_t version;      /* = 1 */
    int32_t layer_id;      /* -1 for global ops, 0-27 for layers */
    char op_name[32];      /* e.g., "q_proj", "attn_out", "logits" */
    uint32_t dtype;        /* 0=fp32, 1=fp16 */
    uint32_t rank;         /* 1-4 */
    int64_t shape[4];      /* dimensions */
    uint32_t elem_count;   /* total elements */
    int32_t token_id;      /* current token being decoded */
    uint8_t reserved[28];  /* padding */
} __attribute__((packed)) CKDumpFileHeader;

static FILE *g_ck_dump_file = NULL;
static int g_ck_dump_token = 0;

/**
 * Initialize dumping. Call before any inference.
 * @param dump_dir Directory to write dump.bin (default: "ck_parity_dumps")
 */
static inline void ck_dump_init(const char *dump_dir) {
    if (!dump_dir) dump_dir = "ck_parity_dumps";

    char path[512];
    snprintf(path, sizeof(path), "%s/dump.bin", dump_dir);

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
#define ck_dump_close()

#endif  /* CK_PARITY_DUMP */

#endif  /* CK_PARITY_DUMP_H */
