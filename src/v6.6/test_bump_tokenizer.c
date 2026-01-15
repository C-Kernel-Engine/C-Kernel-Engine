/*
 * test_bump_tokenizer.c - Test reading tokenizer from BUMP file
 *
 * This test verifies that tokenizer data is correctly embedded in the BUMP file
 * and can be read directly without requiring a separate tokenizer.json file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

/* BUMP file header structure (must match convert_gguf_to_bump_v6_5.py) */
#pragma pack(push, 1)
typedef struct {
    char magic[8];        /* "BUMPWGT4" */
    uint32_t version;     /* 4 */
    uint32_t model_type;  /* 1 = legacy */
    uint32_t num_layers;
    uint32_t vocab_size;
    uint32_t embed_dim;
    uint32_t intermediate_size;
    uint32_t context_length;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint64_t aligned_embed_dim;
    uint64_t aligned_head_dim;
    uint64_t aligned_intermediate;
    uint64_t aligned_context;
    /* Tokenizer metadata (v4.1+) */
    uint32_t num_merges;
    uint32_t total_vocab_bytes;
    uint8_t checksum[32]; /* SHA-256 */
} BumpHeader;
#pragma pack(pop)

#define BUMP_HEADER_SIZE 128

/* Read manifest from JSON file to get tokenizer offsets */
typedef struct {
    const char *name;
    const char *dtype;
    size_t offset;
    size_t size;
} ManifestEntry;

static bool read_manifest_entry(const char *json_path, const char *entry_name,
                               size_t *out_offset, size_t *out_size) {
    FILE *f = fopen(json_path, "r");
    if (!f) return false;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = malloc(len + 1);
    if (!buf) { fclose(f); return false; }
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);

    /* Simple JSON parsing for our specific format:
     * entries: [ { "name": "vocab_offsets", "file_offset": 123, "size": 456 }, ... ]
     */
    char search[256];
    snprintf(search, sizeof(search), "\"name\": \"%s\"", entry_name);
    char *pos = strstr(buf, search);
    if (!pos) { free(buf); return false; }

    /* Find file_offset field within this entry */
    char *offset_pos = strstr(pos, "\"file_offset\":");
    if (!offset_pos) { free(buf); return false; }
    *out_offset = strtoull(offset_pos + 14, NULL, 0);

    /* Find size field within this entry */
    char *size_pos = strstr(pos, "\"size\":");
    if (!size_pos) { free(buf); return false; }
    *out_size = strtoull(size_pos + 7, NULL, 0);

    free(buf);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <weights.bump> [manifest.json]\n", argv[0]);
        fprintf(stderr, "\nTests reading tokenizer data from BUMP file\n");
        return 1;
    }

    const char *bump_path = argv[1];
    const char *manifest_path = argc > 2 ? argv[2] : NULL;

    printf("Testing BUMP tokenizer extraction\n");
    printf("==================================\n");
    printf("BUMP file: %s\n", bump_path);
    if (manifest_path) printf("Manifest: %s\n", manifest_path);
    printf("\n");

    /* Open BUMP file */
    FILE *f = fopen(bump_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", bump_path);
        return 1;
    }

    /* Read header */
    BumpHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "Error: Cannot read header\n");
        fclose(f);
        return 1;
    }

    /* Verify magic */
    if (memcmp(header.magic, "BUMPWGT4", 8) != 0) {
        fprintf(stderr, "Error: Invalid magic: %.8s\n", header.magic);
        fclose(f);
        return 1;
    }

    printf("BUMP Header:\n");
    printf("  Version: %u\n", header.version);
    printf("  Layers: %u\n", header.num_layers);
    printf("  Vocab size: %u\n", header.vocab_size);
    printf("  Embed dim: %u\n", header.embed_dim);
    printf("  Heads: %u/%u\n", header.num_heads, header.num_kv_heads);
    printf("  Head dim: %u\n", header.head_dim);
    printf("  Intermediate: %u\n", header.intermediate_size);
    printf("  Context: %u\n", header.context_length);
    printf("  Num merges: %u\n", header.num_merges);
    printf("  Vocab bytes: %u\n", header.total_vocab_bytes);
    printf("\n");

    /* If we have a manifest, use it to find tokenizer entries */
    if (manifest_path) {
        size_t vocab_offsets_off, vocab_offsets_size;
        size_t vocab_strings_off, vocab_strings_size;
        size_t vocab_merges_off, vocab_merges_size;

        bool have_offsets = read_manifest_entry(manifest_path, "vocab_offsets",
                                               &vocab_offsets_off, &vocab_offsets_size);
        bool have_strings = read_manifest_entry(manifest_path, "vocab_strings",
                                               &vocab_strings_off, &vocab_strings_size);
        bool have_merges = read_manifest_entry(manifest_path, "vocab_merges",
                                              &vocab_merges_off, &vocab_merges_size);

        printf("Manifest entries:\n");
        if (have_offsets) {
            printf("  vocab_offsets: offset=0x%lx size=%zu bytes\n",
                   vocab_offsets_off, vocab_offsets_size);
        } else {
            printf("  vocab_offsets: NOT FOUND\n");
        }
        if (have_strings) {
            printf("  vocab_strings: offset=0x%lx size=%zu bytes\n",
                   vocab_strings_off, vocab_strings_size);
        } else {
            printf("  vocab_strings: NOT FOUND\n");
        }
        if (have_merges) {
            printf("  vocab_merges: offset=0x%lx size=%zu bytes\n",
                   vocab_merges_off, vocab_merges_size);
        } else {
            printf("  vocab_merges: NOT FOUND\n");
        }
        printf("\n");

        if (have_offsets && have_strings) {
            /* Read and display some tokens */
            printf("Sample tokens from BUMP:\n");

            /* Read vocab_offsets */
            int32_t *offsets = malloc(vocab_offsets_size);
            fseek(f, vocab_offsets_off, SEEK_SET);
            fread(offsets, 1, vocab_offsets_size, f);

            /* Read vocab_strings */
            char *strings = malloc(vocab_strings_size);
            fseek(f, vocab_strings_off, SEEK_SET);
            fread(strings, 1, vocab_strings_size, f);

            int num_tokens = vocab_offsets_size / 4;
            for (int i = 0; i < 10 && i < num_tokens; i++) {
                const char *token = strings + offsets[i];
                printf("  [%d] '%s'\n", i, token);
            }
            printf("  ...\n");
            for (int i = num_tokens - 5; i < num_tokens; i++) {
                if (i >= 10) {
                    const char *token = strings + offsets[i];
                    printf("  [%d] '%s'\n", i, token);
                }
            }

            free(offsets);
            free(strings);

            printf("\nTokenizer successfully read from BUMP file!\n");
        }
    } else {
        printf("No manifest provided - cannot locate tokenizer entries.\n");
        printf("Run converter with --manifest-out to generate manifest.\n");
    }

    fclose(f);
    return 0;
}
