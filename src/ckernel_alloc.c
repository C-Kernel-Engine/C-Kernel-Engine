#define _GNU_SOURCE
#include "ckernel_alloc.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
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

static size_t align_down_bytes(size_t n, size_t align)
{
    if (align == 0) return n;
    return n & ~(align - 1);
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

static int env_flag_enabled(const char *name)
{
    const char *value = getenv(name);
    if (!value || !*value) {
        return 0;
    }
    if (strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "no") == 0) {
        return 0;
    }
    return 1;
}

static void ck_bump_alloc_reset(ck_bump_alloc_t *alloc)
{
    if (!alloc) {
        return;
    }
    alloc->base = NULL;
    alloc->total_size = 0;
    alloc->weights_base = 0;
    alloc->activations_base = 0;
    alloc->mapped_len = 0;
    alloc->weights_file_size = 0;
    alloc->mode = CK_BUMP_MODE_UNINITIALIZED;
}

#ifdef __linux__
static int ck_bump_alloc_try_mixed(ck_bump_alloc_t *alloc, const char *weights_path)
{
    long page_size_raw = sysconf(_SC_PAGESIZE);
    size_t page_size = page_size_raw > 0 ? (size_t)page_size_raw : 4096U;
    size_t mapped_len = 0;
    size_t prefix_len = 0;
    size_t weights_len = 0;
    size_t weights_map_len = 0;
    size_t runtime_map_start = 0;
    int fd = -1;
    struct stat st;
    uint8_t *base = NULL;

    if (!alloc || !weights_path || !*weights_path) {
        return -1;
    }
    if (alloc->weights_base > alloc->activations_base || alloc->activations_base > alloc->total_size) {
        fprintf(stderr, "ck_bump_alloc_init: invalid bump layout for mixed fallback\n");
        return -1;
    }
    if (alloc->weights_base != 0) {
        fprintf(stderr,
                "ck_bump_alloc_init: mixed fallback currently requires weights_base == 0 (got %zu)\n",
                alloc->weights_base);
        return -1;
    }
    if (alloc->weights_base != align_down_bytes(alloc->weights_base, page_size)) {
        fprintf(stderr,
                "ck_bump_alloc_init: mixed fallback requires page-aligned weights_base (got %zu, page %zu)\n",
                alloc->weights_base, page_size);
        return -1;
    }

    mapped_len = align_up_bytes(alloc->total_size, page_size);
    prefix_len = alloc->weights_base;
    weights_len = alloc->activations_base - alloc->weights_base;
    weights_map_len = align_up_bytes(weights_len, page_size);
    runtime_map_start = align_up_bytes(alloc->activations_base, page_size);

    fd = open(weights_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "ck_bump_alloc_init: failed to open %s: %s\n", weights_path, strerror(errno));
        return -1;
    }
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "ck_bump_alloc_init: fstat failed for %s: %s\n", weights_path, strerror(errno));
        close(fd);
        return -1;
    }
    if (!S_ISREG(st.st_mode)) {
        fprintf(stderr, "ck_bump_alloc_init: %s is not a regular file\n", weights_path);
        close(fd);
        return -1;
    }
    alloc->weights_file_size = (size_t)st.st_size;
    if (weights_len > 0 && weights_map_len > align_up_bytes(alloc->weights_file_size, page_size)) {
        fprintf(stderr,
                "ck_bump_alloc_init: weights file too small for mixed mapping (%zu < %zu bytes)\n",
                alloc->weights_file_size, weights_map_len);
        close(fd);
        return -1;
    }

    base = mmap(NULL, mapped_len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "ck_bump_alloc_init: reserve mmap failed for %zu bytes: %s\n",
                mapped_len, strerror(errno));
        close(fd);
        return -1;
    }

    if (prefix_len > 0) {
        void *prefix = mmap(base, prefix_len, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (prefix == MAP_FAILED) {
            fprintf(stderr, "ck_bump_alloc_init: prefix mmap failed: %s\n", strerror(errno));
            munmap(base, mapped_len);
            close(fd);
            return -1;
        }
    }

    if (weights_map_len > 0) {
        void *mapped = mmap(base + alloc->weights_base, weights_map_len,
                            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd, 0);
        if (mapped == MAP_FAILED) {
            fprintf(stderr, "ck_bump_alloc_init: weights mmap failed: %s\n", strerror(errno));
            munmap(base, mapped_len);
            close(fd);
            return -1;
        }
    }

    if (runtime_map_start < mapped_len) {
        void *runtime = mmap(base + runtime_map_start, mapped_len - runtime_map_start,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (runtime == MAP_FAILED) {
            fprintf(stderr, "ck_bump_alloc_init: runtime mmap failed: %s\n", strerror(errno));
            munmap(base, mapped_len);
            close(fd);
            return -1;
        }
    }

    close(fd);
    alloc->base = base;
    alloc->mapped_len = mapped_len;
    alloc->mode = CK_BUMP_MODE_MIXED_FILE_BACKED;
    fprintf(stderr,
            "ck_bump_alloc_init: fallback to mixed-backed bump (%zu MiB total, weights file-backed=%zu MiB)\n",
            alloc->total_size / (1024U * 1024U), weights_len / (1024U * 1024U));
    return 0;
}
#endif

int ck_bump_alloc_init(ck_bump_alloc_t *alloc,
                       const char *weights_path,
                       size_t total_size,
                       size_t weights_base,
                       size_t activations_base)
{
    int force_mixed = 0;
    int disable_mixed = 0;

    if (!alloc || total_size == 0) {
        fprintf(stderr, "ck_bump_alloc_init: invalid arguments\n");
        return -1;
    }

    ck_bump_alloc_reset(alloc);
    alloc->total_size = total_size;
    alloc->weights_base = weights_base;
    alloc->activations_base = activations_base;
    force_mixed = env_flag_enabled("CK_BUMP_FORCE_MIXED");
    disable_mixed = env_flag_enabled("CK_BUMP_DISABLE_MIXED");

    if (weights_base > activations_base || activations_base > total_size) {
        fprintf(stderr,
                "ck_bump_alloc_init: invalid bump layout weights=%zu activations=%zu total=%zu\n",
                weights_base, activations_base, total_size);
        ck_bump_alloc_reset(alloc);
        return -1;
    }

    if (!force_mixed) {
        void *anon = ck_huge_alloc(total_size);
        if (anon) {
            alloc->base = (uint8_t *)anon;
            alloc->mapped_len = align_up_bytes(total_size, HUGE_PAGE_SIZE);
            alloc->mode = CK_BUMP_MODE_ANON;
            return 0;
        }
    }

#ifdef __linux__
    if (!disable_mixed) {
        if (ck_bump_alloc_try_mixed(alloc, weights_path) == 0) {
            return 0;
        }
    }
#else
    (void)weights_path;
#endif

    fprintf(stderr,
            "ck_bump_alloc_init: failed to allocate bump arena (%zu bytes, %.2f MiB)\n",
            total_size, (double)total_size / (1024.0 * 1024.0));
    ck_bump_alloc_reset(alloc);
    return -1;
}

void ck_bump_alloc_free(ck_bump_alloc_t *alloc)
{
    if (!alloc || !alloc->base) {
        ck_bump_alloc_reset(alloc);
        return;
    }

    if (alloc->mode == CK_BUMP_MODE_MIXED_FILE_BACKED) {
        if (alloc->mapped_len > 0) {
            munmap(alloc->base, alloc->mapped_len);
        }
    } else if (alloc->mode == CK_BUMP_MODE_ANON) {
        ck_huge_free(alloc->base, alloc->total_size);
    }

    ck_bump_alloc_reset(alloc);
}

int ck_bump_alloc_needs_weight_materialization(const ck_bump_alloc_t *alloc)
{
    if (!alloc) {
        return 1;
    }
    return alloc->mode != CK_BUMP_MODE_MIXED_FILE_BACKED;
}
