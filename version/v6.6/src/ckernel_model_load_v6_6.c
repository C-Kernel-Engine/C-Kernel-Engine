/*
 * ckernel_model_load_v6_6.c - Load BUMPWGT4/BUMPWGT5 weights using a manifest map.
 *
 * WEIGHT FILE FORMAT:
 * ===================
 * BUMPWGT4 (Legacy):
 *   - 8-byte magic: "BUMPWGT4"
 *   - Raw tensor data packed sequentially in a flat binary format
 *
 * BUMPWGT5 (Current - v6.6):
 *   - 8-byte magic: "BUMPWGT5"
 *   - Includes metadata footer for template/config/quant summary
 *   - Manifest-driven loading: uses separate .manifest file to map tensors
 *
 * IMPORTANT: This loader ignores BUMPWGT5 metadata and uses only the
 * sidecar manifest map for tensor offsets. The magic check validates format.
 *
 * MANIFEST FILE FORMAT (weights_manifest.map):
 * =====================
 * Each line: name|dtype|file_offset|size|runtime_offset
 *   - name:       Tensor identifier (e.g., "model.layers.0.attn.k.weight")
 *   - dtype:      Data type (e.g., "f16", "q4k", "i8")
 *   - file_offset: Byte offset in the weights.bump file where tensor starts
 *   - size:       Total bytes to read for this tensor
 *   - runtime_offset: Destination offset in the runtime buffer (layout output)
 *
 * Example manifest line:
 *   model.layers.0.attn.k.weight|q4k|16777216|1048576|8192
 *   ^                              ^     ^           ^       ^
 *   |                              |     |           |       +-- Load to addr + 8192
 *   |                              |     |           +-- Read 1MB from file
 *   |                              |     +-- Start at byte offset 16MB in file
 *   |                              +-- Quantized 4-bit K attention weights
 *   +-- Layer 0 K attention weight tensor
 *
 * This file handles BOTH BUMPWGT4 and BUMPWGT5 formats for backward compatibility.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ckernel_model_load_v6.6.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * CONSTANTS
 * =========
 * MANIFEST_LINE_MAX: Max line length for manifest parsing (4096 bytes)
 * COPY_CHUNK:        1MB chunks for memory-safe copying (avoids huge allocations)
 */
#define MANIFEST_LINE_MAX 4096
#define COPY_CHUNK (1 << 20)

/*
 * Helper: Check if string 's' starts with 'prefix'
 * Used for detecting comment lines (#) and empty lines in manifest
 */
static int starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

/*
 * Helper: Parse unsigned 64-bit integer from string
 * Handles decimal (123), hex (0x7B), and octal (0173) formats
 */
static unsigned long long parse_u64(const char *s) {
    if (!s) return 0;
    return strtoull(s, NULL, 0);
}

static int dtype_from_str(const char *s) {
    if (!s) return 0;
    if (strcmp(s, "fp32") == 0 || strcmp(s, "f32") == 0) return 0;
    if (strcmp(s, "bf16") == 0) return 1;
    if (strcmp(s, "fp16") == 0 || strcmp(s, "f16") == 0) return 2;
    if (strcmp(s, "int8") == 0 || strcmp(s, "i8") == 0) return 3;
    if (strcmp(s, "int4") == 0 || strcmp(s, "i4") == 0) return 4;
    if (strcmp(s, "q4_0") == 0) return 5;
    if (strcmp(s, "q4_1") == 0) return 6;
    if (strcmp(s, "q4_k") == 0 || strcmp(s, "q4k") == 0) return 7;
    if (strcmp(s, "q6_k") == 0 || strcmp(s, "q6k") == 0) return 8;
    if (strcmp(s, "q8_0") == 0 || strcmp(s, "q8_0") == 0) return 9;
    if (strcmp(s, "q8_k") == 0 || strcmp(s, "q8k") == 0) return 10;
    if (strcmp(s, "q5_0") == 0) return 11;
    if (strcmp(s, "q5_1") == 0) return 12;
    if (strcmp(s, "u8") == 0) return 13;
    if (strcmp(s, "i32") == 0) return 14;
    return 0;
}

/*
 * ck_load_weights_manifest_v6.6()
 * ===============================
 * Loads model weights from a BUMPWGT4/BUMPWGT5 binary file using a manifest map.
 *
 * This is the primary entry point for loading weights in v6.6. It supports both:
 *   - BUMPWGT4: Legacy flat binary format (sequential tensors)
 *   - BUMPWGT5: New manifest-driven format (random access via offsets)
 *
 * IMPORTANT: This loader IGNORES any metadata header in the BUMPWGT5 file.
 * It relies entirely on the sidecar .manifest file for tensor mapping.
 * The weights file's magic bytes only serve as a format validation check.
 *
 * Parameters:
 *   base:          Pointer to pre-allocated memory region (mmap'd or malloc'd)
 *   weights_path:  Path to .bin weights file (starts with BUMPWGT4/BUMPWGT5 magic)
 *   manifest_path: Path to .manifest file (maps tensor names to offsets)
 *
 * Returns:
 *   Manifest map handle or NULL on error
 *
 * MEMORY LAYOUT:
 * ==============
 *   [base + runtime_offset_0] = Tensor 0 data
 *   [base + runtime_offset_1] = Tensor 1 data
 *   ...
 *
 * The manifest specifies where each tensor lives in the weights file and where
 * it should be placed in the runtime memory region.
 */
ck_manifest_map_t *ck_load_weights_manifest_v66(void *base,
                                const char *weights_path,
                                const char *manifest_path)
{
    /* Validate input parameters */
    if (!base || !weights_path || !manifest_path) {
        fprintf(stderr, "ck_load_weights_manifest_v66: invalid arguments\n");
        return NULL;
    }

    /* STEP 1: Open the weights binary file */
    FILE *wf = fopen(weights_path, "rb");
    if (!wf) {
        fprintf(stderr, "ck_load_weights_manifest_v66: failed to open %s: %s\n",
                weights_path, strerror(errno));
        return NULL;
    }

    /* STEP 2: Verify magic bytes to detect BUMPWGT4 vs BUMPWGT5 */
    /* BUMPWGT4 = legacy format, BUMPWGT5 = current manifest format */
    char magic[8] = {0};
    if (fread(magic, 1, 8, wf) != 8 ||
        (memcmp(magic, "BUMPWGT4", 8) != 0 && memcmp(magic, "BUMPWGT5", 8) != 0)) {
        fprintf(stderr, "ck_load_weights_manifest_v6.6: invalid BUMPWGT4/BUMPWGT5 magic\n");
        fprintf(stderr, "  Expected: 'BUMPWGT4' or 'BUMPWGT5'\n");
        fprintf(stderr, "  Got: '%.8s' (file may be corrupted or wrong format)\n", magic);
        fclose(wf);
        return NULL;
    }

    /* STEP 3: Open the manifest file (pipe-delimited mapping) */
    FILE *mf = fopen(manifest_path, "r");
    if (!mf) {
        fprintf(stderr, "ck_load_weights_manifest_v6.6: failed to open %s: %s\n",
                manifest_path, strerror(errno));
        fclose(wf);
        return NULL;
    }

    ck_manifest_map_t *manifest = calloc(1, sizeof(*manifest));
    if (!manifest) {
        fprintf(stderr, "ck_load_weights_manifest_v6.6: manifest alloc failed\n");
        fclose(mf);
        fclose(wf);
        return NULL;
    }
    manifest->mapped_base = (uint8_t *)base;

    /* STEP 4: Allocate chunk buffer for streaming copies
     * Using 1MB chunks avoids allocating massive single buffers
     * and works well with OS read-ahead caching */
    char line[MANIFEST_LINE_MAX];
    unsigned char *buf = malloc(COPY_CHUNK);
    if (!buf) {
        fprintf(stderr, "ck_load_weights_manifest_v6.6: malloc failed\n");
        free(manifest);
        fclose(mf);
        fclose(wf);
        return NULL;
    }

    size_t cap = 0;
    size_t count = 0;
    ck_weight_info_t *entries = NULL;

    /* STEP 5: Process each line in the manifest
     * Format: name|dtype|file_offset|size|runtime_offset */
    while (fgets(line, sizeof(line), mf)) {
        /* Skip comments (#) and empty lines */
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }

        /* Remove trailing newlines/carriage returns */
        line[strcspn(line, "\r\n")] = '\0';

        /* Parse manifest fields (pipe-delimited) */
        char *name = strtok(line, "|");
        char *dtype = strtok(NULL, "|");
        char *file_off = strtok(NULL, "|");
        char *size_str = strtok(NULL, "|");
        char *rt_off = strtok(NULL, "|");

        /* Validate all 5 fields are present */
        if (!name || !dtype || !file_off || !size_str || !rt_off) {
            fprintf(stderr, "ck_load_weights_manifest_v6.6: malformed manifest line\n");
            fprintf(stderr, "  Expected: name|dtype|file_offset|size|runtime_offset\n");
            fprintf(stderr, "  Got: %s\n", line);
            free(buf);
            fclose(mf);
            fclose(wf);
            return NULL;
        }

        /* Parse numeric fields as unsigned 64-bit integers */
        unsigned long long file_offset = parse_u64(file_off);
        unsigned long long size = parse_u64(size_str);
        unsigned long long runtime_offset = parse_u64(rt_off);

        if (count == cap) {
            size_t next = cap ? cap * 2 : 256;
            ck_weight_info_t *grow = realloc(entries, next * sizeof(*entries));
            if (!grow) {
                fprintf(stderr, "ck_load_weights_manifest_v6.6: realloc failed\n");
                free(buf);
                fclose(mf);
                fclose(wf);
                for (size_t i = 0; i < count; ++i) {
                    free((void *)entries[i].name);
                }
                free(entries);
                free(manifest);
                return NULL;
            }
            entries = grow;
            cap = next;
        }

        entries[count].name = strdup(name);
        entries[count].dtype = dtype_from_str(dtype);
        entries[count].file_offset = file_offset;
        entries[count].size = size;
        entries[count].runtime_offset = runtime_offset;
        count++;

        /* STEP 6: Seek to tensor location in weights file */
        if (fseek(wf, (long)file_offset, SEEK_SET) != 0) {
            fprintf(stderr, "ck_load_weights_manifest_v6.6: fseek failed to offset %llu\n",
                    (unsigned long long)file_offset);
            free(buf);
            fclose(mf);
            fclose(wf);
            return NULL;
        }

        /* STEP 7: Copy tensor data from file to memory
         * Uses chunked copying for large tensors (>1MB) */
        unsigned char *dst = (unsigned char *)base + runtime_offset;
        unsigned long long remaining = size;

        while (remaining > 0) {
            /* Determine chunk size (cap at COPY_CHUNK = 1MB) */
            size_t take = remaining > COPY_CHUNK ? COPY_CHUNK : (size_t)remaining;

            /* Read chunk from file */
            size_t n = fread(buf, 1, take, wf);
            if (n != take) {
                fprintf(stderr, "ck_load_weights_manifest_v6.6: short read at offset %llu\n",
                        (unsigned long long)file_offset);
                fprintf(stderr, "  Expected %zu bytes, got %zu (EOF or disk error)\n", take, n);
                free(buf);
                fclose(mf);
                fclose(wf);
                for (size_t i = 0; i < count; ++i) {
                    free((void *)entries[i].name);
                }
                free(entries);
                free(manifest);
                return NULL;
            }

            /* Copy chunk to destination */
            memcpy(dst, buf, take);
            dst += take;
            remaining -= take;
        }

        /* Debug output (optional - useful for tracing) */
        /* fprintf(stderr, "Loaded %s (%llu bytes) -> offset %llu\n",
                name, size, runtime_offset); */
    }

    /* STEP 8: Cleanup and return success */
    free(buf);
    fclose(mf);
    fclose(wf);
    manifest->entries = entries;
    manifest->count = (int)count;
    return manifest;
}

void ck_unload_manifest_map(ck_manifest_map_t *manifest) {
    if (!manifest) return;
    if (manifest->entries) {
        for (int i = 0; i < manifest->count; ++i) {
            free((void *)manifest->entries[i].name);
        }
        free(manifest->entries);
    }
    if (manifest->file) fclose(manifest->file);
    free(manifest);
}

ck_weight_info_t *ck_get_weight_info(ck_manifest_map_t *manifest, const char *name) {
    if (!manifest || !name) return NULL;
    for (int i = 0; i < manifest->count; ++i) {
        if (manifest->entries[i].name && strcmp(manifest->entries[i].name, name) == 0) {
            return &manifest->entries[i];
        }
    }
    return NULL;
}

int ck_load_weights(ck_manifest_map_t *manifest,
                    const char *bump_path,
                    void *arena,
                    size_t arena_size)
{
    if (!manifest || !bump_path || !arena) return -1;
    (void)arena_size;
    FILE *wf = fopen(bump_path, "rb");
    if (!wf) return -1;

    char magic[8] = {0};
    if (fread(magic, 1, 8, wf) != 8 ||
        (memcmp(magic, "BUMPWGT4", 8) != 0 && memcmp(magic, "BUMPWGT5", 8) != 0)) {
        fclose(wf);
        return -1;
    }

    unsigned char *buf = malloc(COPY_CHUNK);
    if (!buf) {
        fclose(wf);
        return -1;
    }

    for (int i = 0; i < manifest->count; ++i) {
        ck_weight_info_t *e = &manifest->entries[i];
        if (fseek(wf, (long)e->file_offset, SEEK_SET) != 0) {
            free(buf);
            fclose(wf);
            return -1;
        }
        unsigned char *dst = (unsigned char *)arena + e->runtime_offset;
        unsigned long long remaining = e->size;
        while (remaining > 0) {
            size_t take = remaining > COPY_CHUNK ? COPY_CHUNK : (size_t)remaining;
            size_t n = fread(buf, 1, take, wf);
            if (n != take) {
                free(buf);
                fclose(wf);
                return -1;
            }
            memcpy(dst, buf, take);
            dst += take;
            remaining -= take;
        }
    }

    free(buf);
    fclose(wf);
    return 0;
}

int64_t ck_load_weight_by_name(ck_manifest_map_t *manifest,
                               const char *bump_path,
                               const char *weight_name,
                               void *dest)
{
    if (!manifest || !bump_path || !weight_name || !dest) return -1;
    ck_weight_info_t *e = ck_get_weight_info(manifest, weight_name);
    if (!e) return -1;

    FILE *wf = fopen(bump_path, "rb");
    if (!wf) return -1;

    char magic[8] = {0};
    if (fread(magic, 1, 8, wf) != 8 ||
        (memcmp(magic, "BUMPWGT4", 8) != 0 && memcmp(magic, "BUMPWGT5", 8) != 0)) {
        fclose(wf);
        return -1;
    }

    if (fseek(wf, (long)e->file_offset, SEEK_SET) != 0) {
        fclose(wf);
        return -1;
    }
    size_t n = fread(dest, 1, (size_t)e->size, wf);
    fclose(wf);
    return (n == (size_t)e->size) ? (int64_t)n : -1;
}
