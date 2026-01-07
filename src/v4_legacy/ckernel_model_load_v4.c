/*
 * ckernel_model_load_v4.c - Load BUMPWGT4 weights using a manifest map.
 */

#include "ckernel_model_load_v4.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MANIFEST_LINE_MAX 4096
#define COPY_CHUNK (1 << 20)

static int starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static unsigned long long parse_u64(const char *s) {
    if (!s) return 0;
    return strtoull(s, NULL, 0);
}

int ck_load_weights_manifest_v4(void *base,
                                const char *weights_path,
                                const char *manifest_path)
{
    if (!base || !weights_path || !manifest_path) {
        fprintf(stderr, "ck_load_weights_manifest_v4: invalid arguments\n");
        return -1;
    }

    FILE *wf = fopen(weights_path, "rb");
    if (!wf) {
        fprintf(stderr, "ck_load_weights_manifest_v4: failed to open %s: %s\n",
                weights_path, strerror(errno));
        return -1;
    }

    char magic[8] = {0};
    if (fread(magic, 1, 8, wf) != 8 || memcmp(magic, "BUMPWGT4", 8) != 0) {
        fprintf(stderr, "ck_load_weights_manifest_v4: invalid BUMPWGT4 magic\n");
        fclose(wf);
        return -1;
    }

    FILE *mf = fopen(manifest_path, "r");
    if (!mf) {
        fprintf(stderr, "ck_load_weights_manifest_v4: failed to open %s: %s\n",
                manifest_path, strerror(errno));
        fclose(wf);
        return -1;
    }

    char line[MANIFEST_LINE_MAX];
    unsigned char *buf = malloc(COPY_CHUNK);
    if (!buf) {
        fprintf(stderr, "ck_load_weights_manifest_v4: malloc failed\n");
        fclose(mf);
        fclose(wf);
        return -1;
    }

    while (fgets(line, sizeof(line), mf)) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        line[strcspn(line, "\r\n")] = '\0';

        char *name = strtok(line, "|");
        char *dtype = strtok(NULL, "|");
        char *file_off = strtok(NULL, "|");
        char *size_str = strtok(NULL, "|");
        char *rt_off = strtok(NULL, "|");

        if (!name || !dtype || !file_off || !size_str || !rt_off) {
            fprintf(stderr, "ck_load_weights_manifest_v4: malformed line\n");
            free(buf);
            fclose(mf);
            fclose(wf);
            return -1;
        }

        (void)name;
        (void)dtype;

        unsigned long long file_offset = parse_u64(file_off);
        unsigned long long size = parse_u64(size_str);
        unsigned long long runtime_offset = parse_u64(rt_off);

        if (fseek(wf, (long)file_offset, SEEK_SET) != 0) {
            fprintf(stderr, "ck_load_weights_manifest_v4: fseek failed\n");
            free(buf);
            fclose(mf);
            fclose(wf);
            return -1;
        }

        unsigned char *dst = (unsigned char *)base + runtime_offset;
        unsigned long long remaining = size;

        while (remaining > 0) {
            size_t take = remaining > COPY_CHUNK ? COPY_CHUNK : (size_t)remaining;
            size_t n = fread(buf, 1, take, wf);
            if (n != take) {
                fprintf(stderr, "ck_load_weights_manifest_v4: short read\n");
                free(buf);
                fclose(mf);
                fclose(wf);
                return -1;
            }
            memcpy(dst, buf, take);
            dst += take;
            remaining -= take;
        }
    }

    free(buf);
    fclose(mf);
    fclose(wf);
    return 0;
}
