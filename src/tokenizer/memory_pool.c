/*
 * Memory Pool Implementation
 *
 * Ported from HPC_Embeddings for use in C-Kernel-Engine tokenizer.
 */

#include "tokenizer/memory_pool.h"
#include <stdlib.h>
#include <string.h>

int ck_tokenizer_mempool_init(CKTokenizerMemPool *pool, size_t size) {
    if (size == 0) {
        size = CK_TOKENIZER_POOL_DEFAULT_SIZE;
    }

    pool->memory = (char *)malloc(size);
    if (!pool->memory) {
        return -1;
    }

    pool->size = size;
    pool->used = 0;
    pool->alignment = 8;

    return 0;
}

void ck_tokenizer_mempool_free(CKTokenizerMemPool *pool) {
    if (pool->memory) {
        free(pool->memory);
        pool->memory = NULL;
    }
    pool->size = 0;
    pool->used = 0;
}

static size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

void *ck_tokenizer_mempool_alloc(CKTokenizerMemPool *pool, size_t size) {
    if (!pool || !pool->memory || size == 0) {
        return NULL;
    }

    size = align_up(size, pool->alignment);

    if (pool->used + size > pool->size) {
        return NULL;
    }

    void *ptr = pool->memory + pool->used;
    pool->used += size;

    return ptr;
}

void *ck_tokenizer_mempool_alloc_aligned(CKTokenizerMemPool *pool, size_t size, size_t align) {
    if (!pool || !pool->memory || size == 0) {
        return NULL;
    }

    /* Validate alignment is power of 2 */
    if (align & (align - 1)) {
        return NULL;
    }

    /* Align up the allocation size */
    size = align_up(size, align);

    /* Align the current position */
    size_t misalignment = pool->used & (align - 1);
    size_t padding = (misalignment == 0) ? 0 : (align - misalignment);

    if (pool->used + padding + size > pool->size) {
        return NULL;
    }

    pool->used += padding;
    void *ptr = pool->memory + pool->used;
    pool->used += size;

    return ptr;
}

char *ck_tokenizer_mempool_strdup(CKTokenizerMemPool *pool, const char *str) {
    if (!pool || !str) {
        return NULL;
    }
    return ck_tokenizer_mempool_strndup(pool, str, -1);
}

char *ck_tokenizer_mempool_strndup(CKTokenizerMemPool *pool, const char *str, int len) {
    if (!pool || !str) {
        return NULL;
    }

    if (len < 0) {
        len = (int)strlen(str);
    }

    char *copy = (char *)ck_tokenizer_mempool_alloc(pool, len + 1);
    if (!copy) {
        return NULL;
    }

    memcpy(copy, str, len);
    copy[len] = '\0';

    return copy;
}

void ck_tokenizer_mempool_reset(CKTokenizerMemPool *pool) {
    if (pool) {
        pool->used = 0;
    }
}

size_t ck_tokenizer_mempool_used(CKTokenizerMemPool *pool) {
    return pool ? pool->used : 0;
}

size_t ck_tokenizer_mempool_available(CKTokenizerMemPool *pool) {
    if (!pool) {
        return 0;
    }
    return pool->size - pool->used;
}
