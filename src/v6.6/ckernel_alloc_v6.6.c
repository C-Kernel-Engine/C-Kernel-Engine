#define _GNU_SOURCE
#include "ckernel_alloc.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

/* 2MB huge page size on Linux. */
#ifndef HUGE_PAGE_SIZE
#define HUGE_PAGE_SIZE (2UL * 1024UL * 1024UL)
#endif

typedef struct ck_huge_alloc_entry {
    void *ptr;
    size_t len;
    int was_mmap;
    struct ck_huge_alloc_entry *next;
} ck_huge_alloc_entry_t;

static pthread_mutex_t g_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;
static ck_huge_alloc_entry_t *g_alloc_list = NULL;

static size_t align_up_bytes(size_t n, size_t align)
{
    if (align == 0) return n;
    return (n + align - 1) & ~(align - 1);
}

static int record_allocation(void *ptr, size_t len, int was_mmap)
{
    ck_huge_alloc_entry_t *entry = malloc(sizeof(*entry));
    if (!entry) {
        return 0;
    }
    entry->ptr = ptr;
    entry->len = len;
    entry->was_mmap = was_mmap;
    pthread_mutex_lock(&g_alloc_mutex);
    entry->next = g_alloc_list;
    g_alloc_list = entry;
    pthread_mutex_unlock(&g_alloc_mutex);
    return 1;
}

static ck_huge_alloc_entry_t *detach_allocation(void *ptr)
{
    pthread_mutex_lock(&g_alloc_mutex);
    ck_huge_alloc_entry_t **node = &g_alloc_list;
    while (*node) {
        if ((*node)->ptr == ptr) {
            ck_huge_alloc_entry_t *entry = *node;
            *node = entry->next;
            pthread_mutex_unlock(&g_alloc_mutex);
            return entry;
        }
        node = &(*node)->next;
    }
    pthread_mutex_unlock(&g_alloc_mutex);
    return NULL;
}

void *ck_huge_alloc(size_t bytes)
{
    size_t len = align_up_bytes(bytes, HUGE_PAGE_SIZE);

    /* First, try explicit huge pages via mmap + MAP_HUGETLB. */
    void *p = mmap(NULL, len,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                   -1, 0);
    if (p != MAP_FAILED) {
        if (!record_allocation(p, len, 1)) {
            munmap(p, len);
            return NULL;
        }
        return p;
    }

    /* Fallback: aligned_alloc with transparent hugepage hint. */
    void *q = aligned_alloc(HUGE_PAGE_SIZE, len);
    if (!q) {
        fprintf(stderr, "ck_huge_alloc: aligned_alloc failed for %zu bytes: %s\n",
                len, strerror(errno));
        return NULL;
    }

    /* Best-effort hint; ignore errors. */
    (void)madvise(q, len, MADV_HUGEPAGE);
    if (!record_allocation(q, len, 0)) {
        free(q);
        return NULL;
    }
    return q;
}

void ck_huge_free(void *ptr, size_t bytes)
{
    if (!ptr || bytes == 0) {
        return;
    }

    ck_huge_alloc_entry_t *entry = detach_allocation(ptr);
    if (!entry) {
        /* Fall back to malloc/free if the allocation wasn't tracked. */
        free(ptr);
        return;
    }

    if (entry->was_mmap) {
        munmap(ptr, entry->len);
    } else {
        free(ptr);
    }

    free(entry);
}
