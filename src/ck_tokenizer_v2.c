#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

#include "ck_tokenizer.h"

static uint32_t hash_string(const char *s, int len) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= (uint8_t)s[i];
        hash *= 16777619u;
    }
    return hash;
}

int ck_tokenizer_load_binary(CKTokenizer *tok,
                             int vocab_size,
                             const int32_t *offsets,
                             const char *strings,
                             int num_merges,
                             const int32_t *merges) {
    if (!tok || !offsets || !strings) return -1;

    // We assume ck_tokenizer_init was already called to alloc hash tables
    tok->vocab_size = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        const char *token = strings + offsets[i];
        int len = (int)strlen(token);
        
        CKVocabEntry *entry = (CKVocabEntry *)ck_pool_alloc(&tok->pool, sizeof(CKVocabEntry));
        entry->token = (char *)token; 
        entry->token_len = len;
        entry->id = i;

        uint32_t bucket = hash_string(token, len) % tok->vocab_hash_size;
        entry->next = tok->vocab_hash[bucket];
        tok->vocab_hash[bucket] = entry;

        tok->id_to_token[i] = entry->token;
        tok->vocab_size++;
    }

    if (merges && num_merges > 0) {
        for (int i = 0; i < num_merges; i++) {
            int32_t left = merges[i*3 + 0];
            int32_t right = merges[i*3 + 1];
            int32_t merged = merges[i*3 + 2];
            ck_tokenizer_add_merge(tok, left, right, merged);
        }
    }

    return 0;
}
