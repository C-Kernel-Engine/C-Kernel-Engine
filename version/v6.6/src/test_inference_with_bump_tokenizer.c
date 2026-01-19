/*
 * test_inference_with_bump_tokenizer.c - Test tokenizer loading from BUMP
 *
 * This demonstrates loading a tokenizer embedded in a BUMP file without
 * needing a separate tokenizer.json file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "tokenizer/true_bpe.h"

/* Simple JSON parsing to extract manifest entries */
static bool parse_manifest_entry(const char *json, const char *name,
                                 size_t *offset, size_t *size) {
    char search[256];
    snprintf(search, sizeof(search), "\"name\": \"%s\"", name);
    char *pos = strstr(json, search);
    if (!pos) return false;

    char *off_pos = strstr(pos, "\"file_offset\":");
    if (!off_pos) return false;
    *offset = strtoull(off_pos + 14, NULL, 0);

    char *size_pos = strstr(pos, "\"size\":");
    if (!size_pos) return false;
    *size = strtoull(size_pos + 7, NULL, 0);

    return true;
}

/* Parse num_merges from manifest header */
static int parse_manifest_int(const char *json, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":", key);
    char *pos = strstr(json, search);
    if (!pos) return 0;
    return atoi(pos + strlen(search));
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <weights.bump> <manifest.json> [prompt]\n", argv[0]);
        return 1;
    }

    const char *bump_path = argv[1];
    const char *manifest_path = argv[2];
    const char *prompt = argc > 3 ? argv[3] : "Hello, world!";

    printf("Loading tokenizer from BUMP file...\n");
    printf("  BUMP: %s\n", bump_path);
    printf("  Manifest: %s\n", manifest_path);
    printf("  Prompt: \"%s\"\n\n", prompt);

    /* Read manifest JSON */
    FILE *mf = fopen(manifest_path, "r");
    if (!mf) {
        fprintf(stderr, "Cannot open manifest: %s\n", manifest_path);
        return 1;
    }
    fseek(mf, 0, SEEK_END);
    long mlen = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    char *manifest = malloc(mlen + 1);
    fread(manifest, 1, mlen, mf);
    manifest[mlen] = '\0';
    fclose(mf);

    /* Parse manifest entries */
    size_t vocab_off_offset, vocab_off_size;
    size_t vocab_str_offset, vocab_str_size;
    size_t vocab_merge_offset, vocab_merge_size;

    if (!parse_manifest_entry(manifest, "vocab_offsets", &vocab_off_offset, &vocab_off_size)) {
        fprintf(stderr, "vocab_offsets not found in manifest\n");
        free(manifest);
        return 1;
    }
    if (!parse_manifest_entry(manifest, "vocab_strings", &vocab_str_offset, &vocab_str_size)) {
        fprintf(stderr, "vocab_strings not found in manifest\n");
        free(manifest);
        return 1;
    }
    if (!parse_manifest_entry(manifest, "vocab_merges", &vocab_merge_offset, &vocab_merge_size)) {
        fprintf(stderr, "vocab_merges not found in manifest\n");
        free(manifest);
        return 1;
    }

    int vocab_size = parse_manifest_int(manifest, "vocab_size");
    int num_merges = parse_manifest_int(manifest, "num_merges");
    free(manifest);

    printf("Manifest entries:\n");
    printf("  vocab_size: %d\n", vocab_size);
    printf("  num_merges: %d\n", num_merges);
    printf("  vocab_offsets: offset=%zu size=%zu\n", vocab_off_offset, vocab_off_size);
    printf("  vocab_strings: offset=%zu size=%zu\n", vocab_str_offset, vocab_str_size);
    printf("  vocab_merges: offset=%zu size=%zu\n", vocab_merge_offset, vocab_merge_size);
    printf("\n");

    /* Open BUMP file and read tokenizer data */
    FILE *bf = fopen(bump_path, "rb");
    if (!bf) {
        fprintf(stderr, "Cannot open BUMP: %s\n", bump_path);
        return 1;
    }

    /* Read vocab offsets */
    int32_t *vocab_offsets = malloc(vocab_off_size);
    fseek(bf, vocab_off_offset, SEEK_SET);
    fread(vocab_offsets, 1, vocab_off_size, bf);

    /* Read vocab strings */
    char *vocab_strings = malloc(vocab_str_size);
    fseek(bf, vocab_str_offset, SEEK_SET);
    fread(vocab_strings, 1, vocab_str_size, bf);

    /* Read merges */
    int32_t *vocab_merges = NULL;
    if (num_merges > 0) {
        vocab_merges = malloc(vocab_merge_size);
        fseek(bf, vocab_merge_offset, SEEK_SET);
        fread(vocab_merges, 1, vocab_merge_size, bf);
    }

    fclose(bf);

    /* Initialize tokenizer */
    CKTrueBPE *tokenizer = ck_true_bpe_create();
    if (!tokenizer) {
        fprintf(stderr, "Failed to create tokenizer\n");
        free(vocab_offsets);
        free(vocab_strings);
        free(vocab_merges);
        return 1;
    }

    /* Load tokenizer from BUMP data */
    if (ck_true_bpe_load_binary(tokenizer, vocab_size, vocab_offsets,
                                vocab_strings, num_merges, vocab_merges) != 0) {
        fprintf(stderr, "Failed to load tokenizer from BUMP\n");
        ck_true_bpe_free(tokenizer);
        free(vocab_offsets);
        free(vocab_strings);
        free(vocab_merges);
        return 1;
    }

    printf("Tokenizer loaded successfully!\n");
    printf("  Vocab size: %d\n", vocab_size);
    printf("  Num merges: %d\n", num_merges);
    printf("\n");

    /* Tokenize the prompt */
    int32_t tokens[512];
    int num_tokens = ck_true_bpe_encode(tokenizer, prompt, -1, tokens, 512);

    printf("Tokenization results:\n");
    printf("  Input: \"%s\"\n", prompt);
    printf("  Tokens (%d):", num_tokens);
    for (int i = 0; i < num_tokens; i++) {
        const char *tok_str = ck_true_bpe_id_to_token(tokenizer, tokens[i]);
        printf(" [%d:'%s']", tokens[i], tok_str ? tok_str : "?");
    }
    printf("\n");

    /* Decode tokens back to string */
    char decoded[1024] = {0};
    int decoded_len = 0;
    for (int i = 0; i < num_tokens && decoded_len < 1020; i++) {
        const char *tok_str = ck_true_bpe_id_to_token(tokenizer, tokens[i]);
        if (tok_str) {
            int len = strlen(tok_str);
            if (decoded_len + len < 1020) {
                strcpy(decoded + decoded_len, tok_str);
                decoded_len += len;
            }
        }
    }
    printf("  Decoded: \"%s\"\n", decoded);

    /* Cleanup */
    ck_true_bpe_free(tokenizer);
    free(vocab_offsets);
    free(vocab_strings);
    free(vocab_merges);

    printf("\nTokenizer test PASSED!\n");
    return 0;
}
