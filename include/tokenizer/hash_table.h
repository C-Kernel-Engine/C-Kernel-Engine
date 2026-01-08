/*
 * Hash Table - Optimized with AVX-512
 *
 * Ported from HPC_Embeddings with SIMD-optimized string comparison.
 * Uses MurmurHash3 for hashing and supports AVX-512 for fast lookups.
 *
 * By Anthony Shivakumar
 */

#ifndef CK_TOKENIZER_HASH_TABLE_H
#define CK_TOKENIZER_HASH_TABLE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Hash table entry */
typedef struct CKTokenizerHashEntry {
    char *key;           /* Null-terminated key string */
    void *value;         /* Associated value */
    struct CKTokenizerHashEntry *next;  /* Chain for collisions */
} CKTokenizerHashEntry;

/* Hash table structure */
typedef struct {
    CKTokenizerHashEntry **entries;  /* Bucket array */
    size_t size;                     /* Number of buckets */
    size_t count;                    /* Number of entries */
    float load_factor;               /* Max load factor before resize */
} CKTokenizerHashTable;

/**
 * Create a hash table.
 *
 * @param bucket_count  Number of buckets (0 = auto-size)
 * @return              Newly allocated hash table, or NULL on error
 */
CKTokenizerHashTable *ck_tokenizer_hash_table_create(size_t bucket_count);

/**
 * Free a hash table.
 *
 * @param table   Hash table to free
 * @param free_values  If true, also free all value pointers
 */
void ck_tokenizer_hash_table_free(CKTokenizerHashTable *table, bool free_values);

/**
 * Insert a key-value pair.
 *
 * @param table   Hash table
 * @param key     Key string
 * @param value   Value pointer
 * @return        0 on success, -1 on error
 */
int ck_tokenizer_hash_table_insert(CKTokenizerHashTable *table,
                                   const char *key,
                                   void *value);

/**
 * Look up a key.
 *
 * @param table   Hash table
 * @param key     Key to look up
 * @return        Value pointer, or NULL if not found
 */
void *ck_tokenizer_hash_table_lookup(CKTokenizerHashTable *table,
                                     const char *key);

/**
 * Delete a key.
 *
 * @param table   Hash table
 * @param key     Key to delete
 * @param free_value  If true, free the value pointer
 * @return        0 if found and deleted, -1 if not found
 */
int ck_tokenizer_hash_table_delete(CKTokenizerHashTable *table,
                                   const char *key,
                                   bool free_value);

/**
 * Get the number of entries.
 *
 * @param table   Hash table
 * @return        Number of entries
 */
size_t ck_tokenizer_hash_table_count(CKTokenizerHashTable *table);

/**
 * Check if key exists.
 *
 * @param table   Hash table
 * @param key     Key to check
 * @return        true if key exists
 */
bool ck_tokenizer_hash_table_contains(CKTokenizerHashTable *table,
                                      const char *key);

/**
 * Iterate over all entries.
 *
 * @param table   Hash table
 * @param callback  Function to call for each entry
 * @param user_data  User-provided data for callback
 * @return        0 if all entries processed, non-zero to stop
 */
typedef int (*CKTokenizerHashCallback)(const char *key, void *value, void *user_data);

int ck_tokenizer_hash_table_iterate(CKTokenizerHashTable *table,
                                    CKTokenizerHashCallback callback,
                                    void *user_data);

/**
 * Get all keys as an array.
 *
 * @param table   Hash table
 * @param out_keys  Output array for keys (must be pre-allocated)
 * @param max_keys  Maximum keys to write
 * @return        Number of keys written
 */
size_t ck_tokenizer_hash_table_keys(CKTokenizerHashTable *table,
                                    const char **out_keys,
                                    size_t max_keys);

/**
 * Clear all entries (but keep bucket array).
 *
 * @param table       Hash table
 * @param free_values If true, free all value pointers
 */
void ck_tokenizer_hash_table_clear(CKTokenizerHashTable *table,
                                   bool free_values);

/* Pre-defined bucket counts (prime numbers for good distribution) */
#define CK_TOKENIZER_HT_BUCKETS_SMALL    1024
#define CK_TOKENIZER_HT_BUCKETS_MEDIUM   8192
#define CK_TOKENIZER_HT_BUCKETS_LARGE   65536
#define CK_TOKENIZER_HT_BUCKETS_XL      262144

#ifdef __cplusplus
}
#endif

#endif /* CK_TOKENIZER_HASH_TABLE_H */
