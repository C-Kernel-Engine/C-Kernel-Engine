/*
 * Hash Table with AVX-512 Optimized String Comparison
 *
 * Direct copy from HPC_Embeddings with SIMD optimizations.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include "tokenizer/hash_table.h"
#include "tokenizer/murmurhash3.h"

/* Hash seed constant from HPC_Embeddings */
#define CK_TOKENIZER_HASH_SEED 0x9747b28c

uint32_t ck_tokenizer_hash(const char *key, size_t len) {
    return ck_murmurhash3(key, (uint32_t)len, CK_TOKENIZER_HASH_SEED);
}

uint32_t ck_tokenizer_hash_str(const char *key) {
    return ck_murmurhash3_str(key, CK_TOKENIZER_HASH_SEED);
}

/* SIMD-optimized string comparison using AVX-512 */
/* Falls back to regular strcmp if AVX-512 is not available at compile time */
static inline int simd_strcmp(const char *s1, const char *s2) {
    size_t len1 = strlen(s1);
    size_t len2 = strlen(s2);

    /* Fallback to regular strcmp for short strings */
    if (len1 < 64 || len2 < 64) {
        return strcmp(s1, s2);
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    /* AVX-512 path */
    while (1) {
        /* Load 64 bytes from each string into AVX-512 registers */
        __m512i chunk1 = _mm512_loadu_si512((const __m512i *)s1);
        __m512i chunk2 = _mm512_loadu_si512((const __m512i *)s2);

        /* Compare the chunks byte by byte */
        __mmask64 cmp_mask = _mm512_cmpeq_epu8_mask(chunk1, chunk2);

        /* Check if all bytes are equal */
        if (cmp_mask != 0xFFFFFFFFFFFFFFFF) {
            /* Find the first differing byte */
            int first_diff = __builtin_ctzll(~cmp_mask);
            return (unsigned char)s1[first_diff] - (unsigned char)s2[first_diff];
        }

        /* Check if we hit a null character in s1 or s2 */
        __mmask64 null_mask1 = _mm512_test_epi8_mask(chunk1, _mm512_set1_epi8('\0'));
        __mmask64 null_mask2 = _mm512_test_epi8_mask(chunk2, _mm512_set1_epi8('\0'));

        if (null_mask1 || null_mask2) {
            if (len1 == len2) {
                return 0;
            } else {
                return (len1 < len2) ? -1 : 1;
            }
        }

        /* Advance the pointers by 64 bytes for the next iteration */
        s1 += 64;
        s2 += 64;
    }
#else
    /* SSE/AVX fallback - just use regular strcmp for simplicity */
    (void)s1;
    (void)s2;
    (void)len1;
    (void)len2;
    return strcmp(s1, s2);
#endif
}

CKTokenizerHashTable *ck_tokenizer_hash_table_create(size_t bucket_count) {
    if (bucket_count == 0) {
        bucket_count = CK_TOKENIZER_HT_BUCKETS_SMALL;
    }

    CKTokenizerHashTable *table = (CKTokenizerHashTable *)malloc(sizeof(CKTokenizerHashTable));
    if (!table) {
        return NULL;
    }

    table->entries = (CKTokenizerHashEntry **)calloc(bucket_count, sizeof(CKTokenizerHashEntry *));
    if (!table->entries) {
        free(table);
        return NULL;
    }

    table->size = bucket_count;
    table->count = 0;
    table->load_factor = 0.75f;

    return table;
}

static CKTokenizerHashEntry *create_entry(const char *key, const void *value, size_t value_size) {
    CKTokenizerHashEntry *entry = (CKTokenizerHashEntry *)malloc(sizeof(CKTokenizerHashEntry));
    if (!entry) {
        return NULL;
    }

    entry->key = strdup(key);
    if (!entry->key) {
        free(entry);
        return NULL;
    }

    if (value) {
        entry->value = malloc(value_size);
        if (!entry->value) {
            free(entry->key);
            free(entry);
            return NULL;
        }
        memcpy(entry->value, value, value_size);
    } else {
        entry->value = NULL;
    }

    entry->next = NULL;
    return entry;
}

static void free_entry(CKTokenizerHashEntry *entry, bool free_value) {
    if (!entry) return;
    free(entry->key);
    if (free_value && entry->value) {
        free(entry->value);
    }
    free(entry);
}

void ck_tokenizer_hash_table_free(CKTokenizerHashTable *table, bool free_values) {
    if (!table) {
        return;
    }

    for (size_t i = 0; i < table->size; i++) {
        CKTokenizerHashEntry *entry = table->entries[i];
        while (entry) {
            CKTokenizerHashEntry *next = entry->next;
            free_entry(entry, free_values);
            entry = next;
        }
    }

    free(table->entries);
    free(table);
}

int ck_tokenizer_hash_table_insert(CKTokenizerHashTable *table,
                                   const char *key,
                                   void *value) {
    if (!table || !key) {
        return -1;
    }

    uint32_t bucket = ck_tokenizer_hash_str(key) % table->size;
    CKTokenizerHashEntry *entry = table->entries[bucket];

    /* Check if key already exists */
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            /* Update existing entry - just replace value pointer */
            entry->value = value;
            return 0;
        }
        entry = entry->next;
    }

    /* Create new entry with NULL value (caller manages memory) */
    CKTokenizerHashEntry *new_entry = (CKTokenizerHashEntry *)malloc(sizeof(CKTokenizerHashEntry));
    if (!new_entry) {
        return -1;
    }

    new_entry->key = strdup(key);
    if (!new_entry->key) {
        free(new_entry);
        return -1;
    }

    new_entry->value = value;
    new_entry->next = table->entries[bucket];
    table->entries[bucket] = new_entry;
    table->count++;

    return 0;
}

void *ck_tokenizer_hash_table_lookup(CKTokenizerHashTable *table, const char *key) {
    if (!table || !key) {
        return NULL;
    }

    uint32_t bucket = ck_tokenizer_hash_str(key) % table->size;
    CKTokenizerHashEntry *entry = table->entries[bucket];

    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }

    return NULL;
}

/* AVX-512 optimized lookup */
void *ck_tokenizer_hash_table_lookup_avx(CKTokenizerHashTable *table, const char *key) {
    if (!table || !key) {
        return NULL;
    }

    uint32_t bucket = ck_tokenizer_hash_str(key) % table->size;
    CKTokenizerHashEntry *entry = table->entries[bucket];

    while (entry) {
        if (simd_strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }

    return NULL;
}

int ck_tokenizer_hash_table_delete(CKTokenizerHashTable *table,
                                   const char *key,
                                   bool free_value) {
    if (!table || !key) {
        return -1;
    }

    uint32_t bucket = ck_tokenizer_hash_str(key) % table->size;
    CKTokenizerHashEntry *entry = table->entries[bucket];
    CKTokenizerHashEntry *prev = NULL;

    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            if (prev) {
                prev->next = entry->next;
            } else {
                table->entries[bucket] = entry->next;
            }
            free_entry(entry, free_value);
            table->count--;
            return 0;
        }
        prev = entry;
        entry = entry->next;
    }

    return -1;
}

size_t ck_tokenizer_hash_table_count(CKTokenizerHashTable *table) {
    return table ? table->count : 0;
}

bool ck_tokenizer_hash_table_contains(CKTokenizerHashTable *table, const char *key) {
    return ck_tokenizer_hash_table_lookup(table, key) != NULL;
}

int ck_tokenizer_hash_table_iterate(CKTokenizerHashTable *table,
                                    CKTokenizerHashCallback callback,
                                    void *user_data) {
    if (!table || !callback) {
        return -1;
    }

    for (size_t i = 0; i < table->size; i++) {
        CKTokenizerHashEntry *entry = table->entries[i];
        while (entry) {
            int ret = callback(entry->key, entry->value, user_data);
            if (ret != 0) {
                return ret;
            }
            entry = entry->next;
        }
    }

    return 0;
}

size_t ck_tokenizer_hash_table_keys(CKTokenizerHashTable *table,
                                    const char **out_keys,
                                    size_t max_keys) {
    if (!table || !out_keys) {
        return 0;
    }

    size_t written = 0;
    for (size_t i = 0; i < table->size && written < max_keys; i++) {
        CKTokenizerHashEntry *entry = table->entries[i];
        while (entry && written < max_keys) {
            out_keys[written++] = entry->key;
            entry = entry->next;
        }
    }

    return written;
}

void ck_tokenizer_hash_table_clear(CKTokenizerHashTable *table, bool free_values) {
    if (!table) {
        return;
    }

    for (size_t i = 0; i < table->size; i++) {
        CKTokenizerHashEntry *entry = table->entries[i];
        while (entry) {
            CKTokenizerHashEntry *next = entry->next;
            free_entry(entry, free_values);
            entry = next;
        }
        table->entries[i] = NULL;
    }

    table->count = 0;
}
