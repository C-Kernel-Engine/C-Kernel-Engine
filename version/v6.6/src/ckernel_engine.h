/*
 * ckernel_engine.h - Minimal v6.6 engine definitions
 *
 * Provides dtype enums and helper functions for generated code.
 */

#ifndef CKERNEL_ENGINE_H
#define CKERNEL_ENGINE_H

#include <stdint.h>

/* ============================================================================
 * DATA TYPE ENUMS
 * ============================================================================ */

#define CK_DT_FP32    0
#define CK_DT_FP16    1
#define CK_DT_BF16    2
#define CK_DT_Q8_0    3
#define CK_DT_Q5_0    4
#define CK_DT_Q4_K    5
#define CK_DT_Q6_K    6
#define CK_DT_I8      7
#define CK_DT_INT32   8

/* ============================================================================
 * DTYPE HELPERS
 * ============================================================================ */

/* Get bytes per element for a dtype */
static inline size_t ck_dtype_bytes(int dtype) {
    switch (dtype) {
        case CK_DT_FP32:  return 4;
        case CK_DT_FP16:
        case CK_DT_BF16:  return 2;
        case CK_DT_I8:    return 1;
        case CK_DT_INT32: return 4;
        case CK_DT_Q8_0:  return 1;
        case CK_DT_Q5_0:  return 0.6875f;  /* 22 bytes per 32 elements */
        case CK_DT_Q4_K:  return 0.5625f;  /* 144 bytes per 256 elements */
        case CK_DT_Q6_K:  return 0.8203125f; /* 210 bytes per 256 elements */
        default:          return 4;
    }
}

/* Get bytes for a row of given element count and dtype */
static inline size_t ck_dtype_row_bytes(int dtype, size_t elements) {
    float bytes_per_elem = ck_dtype_bytes(dtype);
    return (size_t)(elements * bytes_per_elem);
}

/* Get block size for quantized dtypes */
static inline size_t ck_dtype_block_size(int dtype) {
    switch (dtype) {
        case CK_DT_Q8_0:  return 32;
        case CK_DT_Q5_0:  return 32;
        case CK_DT_Q4_K:
        case CK_DT_Q6_K:  return 256;
        default:          return 1;
    }
}

#endif /* CKERNEL_ENGINE_H */
