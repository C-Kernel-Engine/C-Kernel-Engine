/*
 * Simple tokenizer test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer/tokenizer.h"

int main(int argc, char **argv) {
    printf("=== C-Kernel-Engine Tokenizer Test ===\n\n");

    /* Create BPE tokenizer */
    CKTokenizer *tok = ck_tokenizer_create_bpe();
    if (!tok) {
        fprintf(stderr, "Failed to create tokenizer\n");
        return 1;
    }

    printf("Tokenizer type: %s\n\n", ck_tokenizer_type_name(tok));

    /* Add some test tokens */
    printf("Adding test vocabulary...\n");

    /* Special tokens */
    ck_tokenizer_add_special_token(tok, "<unk>", 0);
    ck_tokenizer_add_special_token(tok, "<s>", 1);
    ck_tokenizer_add_special_token(tok, "</s>", 2);
    ck_tokenizer_add_special_token(tok, "<pad>", 3);

    /* Common tokens */
    ck_tokenizer_add_token(tok, "hello", 100, 0.0f);
    ck_tokenizer_add_token(tok, "world", 101, 0.0f);
    ck_tokenizer_add_token(tok, "hello world", 102, 0.0f);
    ck_tokenizer_add_token(tok, "test", 103, 0.0f);
    ck_tokenizer_add_token(tok, "ing", 104, 0.0f);
    ck_tokenizer_add_token(tok, "testing", 105, 0.0f);
    ck_tokenizer_add_token(tok, "token", 106, 0.0f);
    ck_tokenizer_add_token(tok, "izer", 107, 0.0f);
    ck_tokenizer_add_token(tok, "hello</s>", 108, 0.0f);

    /* WordPiece style with ## prefix */
    ck_tokenizer_add_token(tok, "##ing", 200, 0.0f);
    ck_tokenizer_add_token(tok, "##er", 201, 0.0f);

    printf("Vocabulary size: %zu\n\n", ck_tokenizer_vocab_size(tok));

    /* Test encoding */
    const char *test_strings[] = {
        "hello world",
        "testing",
        "tokenizer",
        "hello world testing tokenizer",
        NULL
    };

    printf("=== Encoding Tests ===\n\n");

    for (int i = 0; test_strings[i] != NULL; i++) {
        const char *text = test_strings[i];
        int32_t ids[256];
        int max_ids = 256;

        /* Enable BOS/EOS */
        tok->config.add_bos = true;
        tok->config.add_eos = true;

        int num_ids = ck_tokenizer_encode(tok, text, -1, ids, max_ids);

        printf("Input: \"%s\"\n", text);
        printf("Tokens [%d]: ", num_ids);

        for (int j = 0; j < num_ids; j++) {
            const char *token = ck_tokenizer_id_to_token(tok, ids[j]);
            printf("%d", ids[j]);
            if (token) {
                printf("(%s)", token);
            }
            if (j < num_ids - 1) printf(", ");
        }
        printf("\n");

        /* Test decoding */
        char decoded[1024];
        ck_tokenizer_decode(tok, ids, num_ids, decoded, sizeof(decoded));
        printf("Decoded: \"%s\"\n\n", decoded);
    }

    /* Test lookup */
    printf("=== Lookup Tests ===\n\n");
    printf("'hello' -> id %d\n", ck_tokenizer_lookup(tok, "hello"));
    printf("'world' -> id %d\n", ck_tokenizer_lookup(tok, "world"));
    printf("'unknown' -> id %d (should be unk_id=0)\n", ck_tokenizer_lookup(tok, "unknown"));

    /* Clean up */
    ck_tokenizer_free(tok);

    printf("\n=== Test Complete ===\n");
    return 0;
}
