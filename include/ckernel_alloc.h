#ifndef CKERNEL_ALLOC_H
#define CKERNEL_ALLOC_H

#include <stddef.h>
#include <stdint.h>

/**
 * Allocate a large, contiguous memory region for model weights/activations.
 *
 * Implementation strategy:
 *  - Try to allocate 2MB-aligned memory backed by huge pages where possible.
 *  - Fall back to aligned_alloc + madvise(MADV_HUGEPAGE) when explicit
 *    huge pages are not available.
 *
 * Returns NULL on failure.
 */
void *ck_huge_alloc(size_t bytes);

/**
 * Free memory allocated by ck_huge_alloc.
 *
 * The bytes parameter should be the same size passed to ck_huge_alloc.
 */
void ck_huge_free(void *ptr, size_t bytes);

typedef enum {
    CK_BUMP_MODE_UNINITIALIZED = 0,
    CK_BUMP_MODE_ANON = 1,
    CK_BUMP_MODE_MIXED_FILE_BACKED = 2,
} ck_bump_mode_t;

typedef struct {
    uint8_t *base;
    size_t total_size;
    size_t weights_base;
    size_t activations_base;
    size_t mapped_len;
    size_t weights_file_size;
    ck_bump_mode_t mode;
} ck_bump_alloc_t;

/*
 * Initialize a logical bump arena.
 *
 * The allocator first tries the existing anonymous DRAM path. On Linux, if that
 * fails, it can fall back to a mixed-backed arena where the weights region is
 * file-backed from weights.bump and the runtime region is anonymous memory,
 * while preserving one contiguous virtual address space.
 */
int ck_bump_alloc_init(ck_bump_alloc_t *alloc,
                       const char *weights_path,
                       size_t total_size,
                       size_t weights_base,
                       size_t activations_base);

/*
 * Release a logical bump arena previously initialized by ck_bump_alloc_init().
 */
void ck_bump_alloc_free(ck_bump_alloc_t *alloc);

/*
 * Return non-zero when weights still need to be copied into RAM.
 */
int ck_bump_alloc_needs_weight_materialization(const ck_bump_alloc_t *alloc);

#endif /* CKERNEL_ALLOC_H */
