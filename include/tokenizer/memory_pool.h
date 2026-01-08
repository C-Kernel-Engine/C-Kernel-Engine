/*
 * Memory Pool - Thread-safe allocator
 *
 * Ported from HPC_Embeddings for use in C-Kernel-Engine.
 * Provides fast allocations with optional thread safety.
 *
 * By Anthony Shivakumar
 */

#ifndef CK_TOKENIZER_MEMPOOL_H
#define CK_TOKENIZER_MEMPOOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Default pool size: 16MB */
#define CK_TOKENIZER_POOL_DEFAULT_SIZE (16 * 1024 * 1024)

/* Block size: 1MB */
#define CK_TOKENIZER_POOL_BLOCK_SIZE (1024 * 1024)

/* Memory pool structure */
typedef struct CKTokenizerMemPool {
    char *memory;          /* Pool memory */
    size_t size;           /* Total pool size */
    size_t used;           /* Bytes used */
    size_t alignment;      /* Allocation alignment */
} CKTokenizerMemPool;

/**
 * Initialize a memory pool.
 *
 * @param pool     Pool to initialize
 * @param size     Pool size in bytes (0 = use default)
 * @return         0 on success, -1 on error
 */
int ck_tokenizer_mempool_init(CKTokenizerMemPool *pool, size_t size);

/**
 * Free a memory pool.
 *
 * @param pool     Pool to free
 */
void ck_tokenizer_mempool_free(CKTokenizerMemPool *pool);

/**
 * Allocate from pool.
 *
 * @param pool     Pool to allocate from
 * @param size     Size in bytes
 * @return         Pointer to allocated memory, or NULL on error
 */
void *ck_tokenizer_mempool_alloc(CKTokenizerMemPool *pool, size_t size);

/**
 * Allocate aligned memory from pool.
 *
 * @param pool     Pool to allocate from
 * @param size     Size in bytes
 * @param align    Alignment (must be power of 2)
 * @return         Pointer to allocated memory, or NULL on error
 */
void *ck_tokenizer_mempool_alloc_aligned(CKTokenizerMemPool *pool, size_t size, size_t align);

/**
 * Allocate and copy string (strdup equivalent).
 *
 * @param pool     Pool to allocate from
 * @param str      String to copy
 * @return         Pointer to copied string, or NULL on error
 */
char *ck_tokenizer_mempool_strdup(CKTokenizerMemPool *pool, const char *str);

/**
 * Allocate and copy string with length.
 *
 * @param pool     Pool to allocate from
 * @param str      String to copy
 * @param len      Length to copy (-1 for null-terminated)
 * @return         Pointer to copied string, or NULL on error
 */
char *ck_tokenizer_mempool_strndup(CKTokenizerMemPool *pool, const char *str, int len);

/**
 * Reset pool (mark all memory as free).
 *
 * @param pool     Pool to reset
 */
void ck_tokenizer_mempool_reset(CKTokenizerMemPool *pool);

/**
 * Get used bytes in pool.
 *
 * @param pool     Pool to query
 * @return         Number of bytes used
 */
size_t ck_tokenizer_mempool_used(CKTokenizerMemPool *pool);

/**
 * Get available bytes in pool.
 *
 * @param pool     Pool to query
 * @return         Number of bytes available
 */
size_t ck_tokenizer_mempool_available(CKTokenizerMemPool *pool);

/**
 * Get allocation count.
 *
 * @param pool     Pool to query
 * @return         Number of allocations
 */
size_t ck_tokenizer_mempool_alloc_count(CKTokenizerMemPool *pool);

#ifdef __cplusplus
}
#endif

#endif /* CK_TOKENIZER_MEMPOOL_H */
