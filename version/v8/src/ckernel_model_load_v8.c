/*
 * ckernel_model_load_v8.c - Load BUMPWGT4/BUMPWGT5 weights using a manifest map.
 *
 * WEIGHT FILE FORMAT:
 * ===================
 * BUMPWGT4 (Legacy):
 *   - 8-byte magic: "BUMPWGT4"
 *   - Raw tensor data packed sequentially in a flat binary format
 *
 * BUMPWGT5 (Current - v7):
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

#include "ckernel_model_load_v8.h"

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

static ck_manifest_map_t *ck_parse_manifest_map(void *base,
                                                const char *manifest_path)
{
    FILE *mf = NULL;
    ck_manifest_map_t *manifest = NULL;
    ck_weight_info_t *entries = NULL;
    char line[MANIFEST_LINE_MAX];
    size_t cap = 0;
    size_t count = 0;

    if (!base || !manifest_path) {
        fprintf(stderr, "ck_parse_manifest_map: invalid arguments\n");
        return NULL;
    }

    mf = fopen(manifest_path, "r");
    if (!mf) {
        fprintf(stderr, "ck_parse_manifest_map: failed to open %s: %s\n",
                manifest_path, strerror(errno));
        return NULL;
    }

    manifest = calloc(1, sizeof(*manifest));
    if (!manifest) {
        fprintf(stderr, "ck_parse_manifest_map: manifest alloc failed\n");
        fclose(mf);
        return NULL;
    }
    manifest->mapped_base = (uint8_t *)base;
    manifest->weights_materialized = 0;

    while (fgets(line, sizeof(line), mf)) {
        char *name = NULL;
        char *dtype = NULL;
        char *file_off = NULL;
        char *size_str = NULL;
        char *rt_off = NULL;

        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        line[strcspn(line, "\r\n")] = '\0';

        name = strtok(line, "|");
        dtype = strtok(NULL, "|");
        file_off = strtok(NULL, "|");
        size_str = strtok(NULL, "|");
        rt_off = strtok(NULL, "|");

        if (!name || !dtype || !file_off || !size_str || !rt_off) {
            fprintf(stderr, "ck_parse_manifest_map: malformed manifest line\n");
            fprintf(stderr, "  Expected: name|dtype|file_offset|size|runtime_offset\n");
            fprintf(stderr, "  Got: %s\n", line);
            fclose(mf);
            free(manifest);
            return NULL;
        }

        if (count == cap) {
            size_t next = cap ? cap * 2 : 256;
            ck_weight_info_t *grow = realloc(entries, next * sizeof(*entries));
            if (!grow) {
                fprintf(stderr, "ck_parse_manifest_map: realloc failed\n");
                fclose(mf);
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
        if (!entries[count].name) {
            fprintf(stderr, "ck_parse_manifest_map: strdup failed\n");
            fclose(mf);
            for (size_t i = 0; i < count; ++i) {
                free((void *)entries[i].name);
            }
            free(entries);
            free(manifest);
            return NULL;
        }
        entries[count].dtype = dtype_from_str(dtype);
        entries[count].file_offset = parse_u64(file_off);
        entries[count].size = parse_u64(size_str);
        entries[count].runtime_offset = parse_u64(rt_off);
        count++;
    }

    fclose(mf);
    manifest->entries = entries;
    manifest->count = (int)count;
    return manifest;
}

ck_manifest_map_t *ck_open_weights_manifest_v7(void *base,
                                               const char *manifest_path)
{
    return ck_parse_manifest_map(base, manifest_path);
}

static int validate_weights_magic(FILE *wf, const char *weights_path)
{
    char magic[8] = {0};
    if (fread(magic, 1, 8, wf) != 8 ||
        (memcmp(magic, "BUMPWGT4", 8) != 0 && memcmp(magic, "BUMPWGT5", 8) != 0)) {
        fprintf(stderr, "ck_open_weights_manifest_v8: invalid BUMPWGT4/BUMPWGT5 magic in %s\n",
                weights_path);
        fprintf(stderr, "  Expected: 'BUMPWGT4' or 'BUMPWGT5'\n");
        fprintf(stderr, "  Got: '%.8s' (file may be corrupted or wrong format)\n", magic);
        return -1;
    }
    return 0;
}

ck_manifest_map_t *ck_open_weights_manifest_v8(void *base,
                                               const char *weights_path,
                                               const char *manifest_path,
                                               int materialize_weights)
{
    FILE *wf = NULL;
    ck_manifest_map_t *manifest = NULL;
    unsigned char *buf = NULL;

    if (!base || !weights_path || !manifest_path) {
        fprintf(stderr, "ck_open_weights_manifest_v8: invalid arguments\n");
        return NULL;
    }

    wf = fopen(weights_path, "rb");
    if (!wf) {
        fprintf(stderr, "ck_open_weights_manifest_v8: failed to open %s: %s\n",
                weights_path, strerror(errno));
        return NULL;
    }
    if (validate_weights_magic(wf, weights_path) != 0) {
        fclose(wf);
        return NULL;
    }

    manifest = ck_parse_manifest_map(base, manifest_path);
    if (!manifest) {
        fclose(wf);
        return NULL;
    }
    manifest->weights_materialized = materialize_weights ? 1 : 0;

    if (!materialize_weights) {
        for (int i = 0; i < manifest->count; ++i) {
            ck_weight_info_t *entry = &manifest->entries[i];
            if (entry->file_offset != entry->runtime_offset) {
                fprintf(stderr,
                        "ck_open_weights_manifest_v8: mixed-backed mode requires file_offset == runtime_offset for %s\n",
                        entry->name ? entry->name : "(unknown)");
                fclose(wf);
                ck_unload_manifest_map(manifest);
                return NULL;
            }
        }
        fclose(wf);
        return manifest;
    }

    buf = malloc(COPY_CHUNK);
    if (!buf) {
        fprintf(stderr, "ck_open_weights_manifest_v8: malloc failed\n");
        fclose(wf);
        ck_unload_manifest_map(manifest);
        return NULL;
    }

    for (int i = 0; i < manifest->count; ++i) {
        ck_weight_info_t *entry = &manifest->entries[i];
        unsigned char *dst = (unsigned char *)base + entry->runtime_offset;
        unsigned long long remaining = entry->size;

        if (fseek(wf, (long)entry->file_offset, SEEK_SET) != 0) {
            fprintf(stderr, "ck_open_weights_manifest_v8: fseek failed to offset %llu\n",
                    (unsigned long long)entry->file_offset);
            free(buf);
            fclose(wf);
            ck_unload_manifest_map(manifest);
            return NULL;
        }

        while (remaining > 0) {
            size_t take = remaining > COPY_CHUNK ? COPY_CHUNK : (size_t)remaining;
            size_t n = fread(buf, 1, take, wf);
            if (n != take) {
                fprintf(stderr, "ck_open_weights_manifest_v8: short read at offset %llu\n",
                        (unsigned long long)entry->file_offset);
                free(buf);
                fclose(wf);
                ck_unload_manifest_map(manifest);
                return NULL;
            }
            memcpy(dst, buf, take);
            dst += take;
            remaining -= take;
        }
    }

    free(buf);
    fclose(wf);
    return manifest;
}

ck_manifest_map_t *ck_load_weights_manifest_v7(void *base,
                                               const char *weights_path,
                                               const char *manifest_path)
{
    return ck_open_weights_manifest_v8(base, weights_path, manifest_path, 1);
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
