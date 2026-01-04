/**
 * @file ckernel_quant.h
 * @brief Quantization block structures for weight-only quantization
 *
 * Defines block structures for various quantization formats used in LLM inference.
 * Primary focus on Q4_K_M which is commonly used for LLM weight compression.
 *
 * Block structures are compatible with llama.cpp/GGML for model loading.
 */

#ifndef CKERNEL_QUANT_H
#define CKERNEL_QUANT_H

#include <stdint.h>
#include <stddef.h>
#include "ckernel_dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Half-Precision Type (FP16 - IEEE 754)
 * ============================================================================ */

typedef uint16_t ck_half;

/* ============================================================================
 * Q4_0: Simple 4-bit Quantization
 * - 32 weights per block
 * - 1 FP16 scale per block
 * - 18 bytes per 32 weights = 4.5 bits/weight
 * ============================================================================ */

#define QK4_0 32

typedef struct {
    ck_half d;             /* 2 bytes: scale (delta) */
    uint8_t qs[QK4_0 / 2]; /* 16 bytes: 32 x 4-bit weights (2 per byte) */
} block_q4_0;
/* Total: 18 bytes per 32 weights */

/* ============================================================================
 * Q4_1: Simple 4-bit Quantization with Min
 * - 32 weights per block
 * - 2 FP16 values: scale (d) and min (m)
 * - 20 bytes per 32 weights = 5.0 bits/weight
 * ============================================================================ */

#define QK4_1 32

typedef struct {
    ck_half d;             /* 2 bytes: scale (delta) */
    ck_half m;             /* 2 bytes: minimum */
    uint8_t qs[QK4_1 / 2]; /* 16 bytes: 32 x 4-bit weights (2 per byte) */
} block_q4_1;
/* Total: 20 bytes per 32 weights */

/* ============================================================================
 * Q5_0: Simple 5-bit Quantization
 * - 32 weights per block
 * - 1 FP16 scale per block
 * - Low 4 bits stored like Q4_0, high 1 bit packed separately
 * - 22 bytes per 32 weights = 5.5 bits/weight
 * ============================================================================ */

#define QK5_0 32

typedef struct {
    ck_half d;             /* 2 bytes: scale (delta) */
    uint8_t qh[4];         /* 4 bytes: high 1-bit of each weight (32 bits total) */
    uint8_t qs[QK5_0 / 2]; /* 16 bytes: low 4-bits of 32 weights (2 per byte) */
} block_q5_0;
/* Total: 22 bytes per 32 weights */

/* ============================================================================
 * Q5_1: Simple 5-bit Quantization with Min
 * - 32 weights per block
 * - 2 FP16 values: scale (d) and min (m)
 * - Low 4 bits stored like Q4_1, high 1 bit packed separately
 * - 24 bytes per 32 weights = 6.0 bits/weight
 * ============================================================================ */

#define QK5_1 32

typedef struct {
    ck_half d;             /* 2 bytes: scale (delta) */
    ck_half m;             /* 2 bytes: minimum */
    uint8_t qh[4];         /* 4 bytes: high 1-bit of each weight (32 bits total) */
    uint8_t qs[QK5_1 / 2]; /* 16 bytes: low 4-bits of 32 weights (2 per byte) */
} block_q5_1;
/* Total: 24 bytes per 32 weights */

/* ============================================================================
 * Q8_0: Simple 8-bit Quantization
 * - 32 weights per block
 * - 1 FP16 scale per block
 * - 34 bytes per 32 weights = 8.5 bits/weight
 * ============================================================================ */

#define QK8_0 32

typedef struct {
    ck_half d;             /* 2 bytes: scale */
    int8_t qs[QK8_0];      /* 32 bytes: 32 x 8-bit signed weights */
} block_q8_0;
/* Total: 34 bytes per 32 weights */

/* ============================================================================
 * Q4_K: K-Quant 4-bit with Nested Scales (Primary Target)
 * - 256 weights per super-block
 * - 8 sub-blocks of 32 weights each
 * - Two-level scaling: super-block FP16 + sub-block 6-bit
 * - 144 bytes per 256 weights = 4.5 bits/weight
 *
 * This is the format used by Q4_K_M, Q4_K_S, Q4_K_L variants.
 * The M/S/L suffix indicates quantization aggressiveness, not structure.
 * ============================================================================ */

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    ck_half d;                    /* 2 bytes: super-block scale */
    ck_half dmin;                 /* 2 bytes: super-block minimum */
    uint8_t scales[K_SCALE_SIZE]; /* 12 bytes: 8 sub-block scales + 8 sub-block mins (6-bit packed) */
    uint8_t qs[QK_K / 2];         /* 128 bytes: 256 x 4-bit weights */
} block_q4_K;
/* Total: 144 bytes per 256 weights */

/* ============================================================================
 * Q6_K: K-Quant 6-bit (per-16 scales)
 * - 256 weights per block
 * - 16 sub-blocks of 16 weights each
 * - Stored as low 4 bits (ql) + high 2 bits (qh) + int8 scales
 * ============================================================================ */

typedef struct {
    uint8_t ql[QK_K / 2];      /* 128 bytes: low 4 bits */
    uint8_t qh[QK_K / 4];      /* 64 bytes: high 2 bits */
    int8_t scales[QK_K / 16];  /* 16 bytes: 16 sub-block scales */
    ck_half d;                 /* 2 bytes: super-block scale */
} block_q6_K;
/* Total: 210 bytes per 256 weights */

/* ============================================================================
 * Q8_K: K-Quant 8-bit (used for activations in some ops)
 * - 256 weights per super-block
 * - 1 FP32 scale per block (not FP16 like others!)
 * ============================================================================ */

typedef struct {
    float d;                  /* 4 bytes: scale */
    int8_t qs[QK_K];          /* 256 bytes: 256 x 8-bit signed weights */
    int16_t bsums[QK_K / 16]; /* 32 bytes: block sums for optimization */
} block_q8_K;
/* Total: 292 bytes per 256 weights */

/* ============================================================================
 * Size Calculation Utilities
 * ============================================================================ */

/**
 * @brief Get the block size (number of weights per block) for a quant type
 */
static inline size_t ck_quant_block_size(int type) {
    switch (type) {
        case 0: return QK4_0;    /* Q4_0 */
        case 1: return QK8_0;    /* Q8_0 */
        case 2: return QK_K;     /* Q4_K */
        case 3: return QK_K;     /* Q8_K */
        case CK_DT_Q4_1: return QK4_1;
        case CK_DT_Q5_0: return QK5_0;
        case CK_DT_Q5_1: return QK5_1;
        case CK_DT_Q6_K: return QK_K;
        default: return 1;
    }
}

/**
 * @brief Get the byte size per block for a quant type
 */
static inline size_t ck_quant_type_size(int type) {
    switch (type) {
        case 0: return sizeof(block_q4_0);
        case 1: return sizeof(block_q8_0);
        case 2: return sizeof(block_q4_K);
        case 3: return sizeof(block_q8_K);
        case CK_DT_Q4_1: return sizeof(block_q4_1);
        case CK_DT_Q5_0: return sizeof(block_q5_0);
        case CK_DT_Q5_1: return sizeof(block_q5_1);
        case CK_DT_Q6_K: return sizeof(block_q6_K);
        default: return 4; /* FP32 */
    }
}

/**
 * @brief Calculate total bytes needed for n_elements with given quant type
 */
static inline size_t ck_quant_row_size(int type, int64_t n_elements) {
    size_t block_size = ck_quant_block_size(type);
    size_t type_size = ck_quant_type_size(type);
    return (n_elements / block_size) * type_size;
}

/* ============================================================================
 * Q4_K Scale Unpacking Utilities
 *
 * The scales[12] array packs 8 scales and 8 mins in 6-bit format.
 * Unpacking is non-trivial due to the bit packing.
 * ============================================================================ */

/**
 * @brief Unpack Q4_K sub-block scales and mins
 *
 * @param scales The packed scales[12] array from block_q4_K
 * @param sc Output: 8 unpacked scale values (multiply by super-block d)
 * @param m Output: 8 unpacked min values (multiply by super-block dmin)
 */
static inline void unpack_q4_k_scales(const uint8_t *scales,
                                       uint8_t *sc, uint8_t *m) {
    /*
     * The 12-byte scales array encodes 8 scales and 8 mins in 6-bit each.
     * Each 6-bit value is split across byte boundaries.
     */

    /* Unpack scales (sc[0..7]) from bytes 0-5 */
    sc[0] = (scales[0] & 0x3F);
    sc[1] = (scales[0] >> 6) | ((scales[1] & 0x0F) << 2);
    sc[2] = (scales[1] >> 4) | ((scales[2] & 0x03) << 4);
    sc[3] = (scales[2] >> 2);
    sc[4] = (scales[3] & 0x3F);
    sc[5] = (scales[3] >> 6) | ((scales[4] & 0x0F) << 2);
    sc[6] = (scales[4] >> 4) | ((scales[5] & 0x03) << 4);
    sc[7] = (scales[5] >> 2);

    /* Unpack mins (m[0..7]) from bytes 6-11 */
    m[0] = (scales[6] & 0x3F);
    m[1] = (scales[6] >> 6) | ((scales[7] & 0x0F) << 2);
    m[2] = (scales[7] >> 4) | ((scales[8] & 0x03) << 4);
    m[3] = (scales[8] >> 2);
    m[4] = (scales[9] & 0x3F);
    m[5] = (scales[9] >> 6) | ((scales[10] & 0x0F) << 2);
    m[6] = (scales[10] >> 4) | ((scales[11] & 0x03) << 4);
    m[7] = (scales[11] >> 2);
}

/* ============================================================================
 * FP16 Conversion Utilities
 * ============================================================================ */

/**
 * @brief Convert FP16 (ck_half) to FP32
 */
static inline float ck_fp16_to_fp32(ck_half h) {
    /*
     * FP16: 1 sign + 5 exponent + 10 mantissa
     * FP32: 1 sign + 8 exponent + 23 mantissa
     */
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t result;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            result = sign;
        } else {
            /* Denormalized - convert to normalized FP32 */
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            result = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        /* Inf or NaN */
        result = sign | 0x7F800000 | (mant << 13);
    } else {
        /* Normalized */
        result = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    union { uint32_t u; float f; } u;
    u.u = result;
    return u.f;
}

/**
 * @brief Convert FP32 to FP16 (ck_half)
 */
static inline ck_half ck_fp32_to_fp16(float f) {
    union { uint32_t u; float f; } u;
    u.f = f;

    uint32_t sign = (u.u >> 16) & 0x8000;
    int32_t exp = ((u.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u.u >> 13) & 0x3FF;

    if (exp <= 0) {
        if (exp < -10) {
            /* Underflow to zero */
            return sign;
        }
        /* Denormalized */
        mant = (mant | 0x400) >> (1 - exp);
        return sign | mant;
    } else if (exp >= 31) {
        /* Overflow to infinity */
        return sign | 0x7C00;
    }

    return sign | (exp << 10) | mant;
}

/* Convenience macros */
#define CK_FP16_TO_FP32(x) ck_fp16_to_fp32(x)
#define CK_FP32_TO_FP16(x) ck_fp32_to_fp16(x)

/* Legacy compatibility (for files that used the old names) */
typedef ck_half ggml_half;
#define ggml_fp16_to_fp32 ck_fp16_to_fp32
#define ggml_fp32_to_fp16 ck_fp32_to_fp16
#define GGML_FP16_TO_FP32 CK_FP16_TO_FP32
#define GGML_FP32_TO_FP16 CK_FP32_TO_FP16

#ifdef __cplusplus
}
#endif

#endif /* CKERNEL_QUANT_H */
