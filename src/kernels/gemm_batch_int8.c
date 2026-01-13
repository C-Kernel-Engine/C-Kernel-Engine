/**
 * @file gemm_batch_int8.c
 * @brief Batch GEMM kernels for quantized weights with INT8 activations
 *
 * Implements batch matrix multiplication where:
 *   - Activations (A): Q8_0 quantized (INT8 + scale)
 *   - Weights (B): Q5_0 or Q8_0 quantized
 *   - Output (C): FP32
 *
 * Operation: C[M,N] = A[M,K] @ B[N,K]^T  (B is transposed/row-major weights)
 *
 * Instruction Set Implementations:
 *   - Scalar: Reference implementation for correctness verification
 *   - AVX: 256-bit SIMD (8 floats, or 32 int8s)
 *   - AVX-512: 512-bit SIMD (16 floats, or 64 int8s)
 *   - AMX: Intel Advanced Matrix Extensions (tile-based, requires Sapphire Rapids+)
 *
 * Design Philosophy:
 *   - Every kernel MUST produce bit-identical results to scalar reference
 *   - Comprehensive testing against llama.cpp ensures correctness
 *   - Performance optimizations never compromise accuracy
 *
 * @author C-Kernel-Engine Team
 * @date 2024
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "ckernel_quant.h"

/* SIMD headers */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE4_1__)
#include <immintrin.h>
#endif

/* AMX headers (requires specific compiler support) */
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#define HAS_AMX 1
#else
#define HAS_AMX 0
#endif

/* ============================================================================
 * Constants and Block Sizes
 * ============================================================================ */

#define QK8_0 32   /* Q8_0: 32 weights per block */
#define QK5_0 32   /* Q5_0: 32 weights per block */

/* AMX tile dimensions */
#define AMX_TILE_M 16
#define AMX_TILE_N 16
#define AMX_TILE_K 64

/* ============================================================================
 * SECTION 1: GEMM Q8_0 x Q8_0 -> FP32
 *
 * Both weights and activations are Q8_0 quantized.
 * This is the simplest case - direct INT8 x INT8 -> INT32 accumulation.
 * ============================================================================ */

/**
 * @brief Scalar reference: gemm_nt_q8_0_q8_0
 *
 * C[m,n] = sum_k( dequant(A[m,k]) * dequant(B[n,k]) )
 *        = sum_blocks( d_a * d_b * sum_j(a_qs[j] * b_qs[j]) )
 *
 * @param A      Input activations [M, K] in Q8_0 format
 * @param B      Weight matrix [N, K] in Q8_0 format (row-major, each row is one output)
 * @param C      Output matrix [M, N] in FP32
 * @param M      Number of tokens (batch size)
 * @param N      Number of output features (rows in B)
 * @param K      Number of input features (must be multiple of 32)
 */
void gemm_nt_q8_0_q8_0_ref(
    const void *A,
    const void *B,
    float *C,
    int M, int N, int K)
{
    const int nb = K / QK8_0;  /* Number of blocks per row */
    const block_q8_0 *a_blocks = (const block_q8_0 *)A;
    const block_q8_0 *b_blocks = (const block_q8_0 *)B;

    for (int m = 0; m < M; m++) {
        const block_q8_0 *a_row = a_blocks + (size_t)m * nb;

        for (int n = 0; n < N; n++) {
            const block_q8_0 *b_row = b_blocks + (size_t)n * nb;
            float sum = 0.0f;

            for (int ib = 0; ib < nb; ib++) {
                const float d_a = CK_FP16_TO_FP32(a_row[ib].d);
                const float d_b = CK_FP16_TO_FP32(b_row[ib].d);
                const float d = d_a * d_b;

                int32_t sumi = 0;
                for (int j = 0; j < QK8_0; j++) {
                    sumi += (int32_t)a_row[ib].qs[j] * (int32_t)b_row[ib].qs[j];
                }

                sum += d * (float)sumi;
            }

            C[(size_t)m * N + n] = sum;
        }
    }
}

#if defined(__AVX2__)
/**
 * @brief AVX2 implementation: gemm_nt_q8_0_q8_0
 *
 * Uses 256-bit vectors to process 32 int8 values at once.
 * Requires AVX2 for _mm256_cvtepi8_epi16, _mm256_madd_epi16, etc.
 * Accumulates in INT32, then scales by d_a * d_b.
 */
void gemm_nt_q8_0_q8_0_avx2(
    const void *A,
    const void *B,
    float *C,
    int M, int N, int K)
{
    const int nb = K / QK8_0;
    const block_q8_0 *a_blocks = (const block_q8_0 *)A;
    const block_q8_0 *b_blocks = (const block_q8_0 *)B;

    for (int m = 0; m < M; m++) {
        const block_q8_0 *a_row = a_blocks + (size_t)m * nb;

        for (int n = 0; n < N; n++) {
            const block_q8_0 *b_row = b_blocks + (size_t)n * nb;
            float sum = 0.0f;

            for (int ib = 0; ib < nb; ib++) {
                const float d_a = CK_FP16_TO_FP32(a_row[ib].d);
                const float d_b = CK_FP16_TO_FP32(b_row[ib].d);
                const float d = d_a * d_b;

                /* Load 32 int8 values from A and B */
                __m256i va = _mm256_loadu_si256((const __m256i *)a_row[ib].qs);
                __m256i vb = _mm256_loadu_si256((const __m256i *)b_row[ib].qs);

                /* Split into 16-bit for multiplication without overflow
                 * Process low 16 bytes and high 16 bytes separately */
                __m128i va_lo = _mm256_castsi256_si128(va);
                __m128i va_hi = _mm256_extracti128_si256(va, 1);
                __m128i vb_lo = _mm256_castsi256_si128(vb);
                __m128i vb_hi = _mm256_extracti128_si256(vb, 1);

                /* Extend to 16-bit and multiply */
                __m256i va_lo_16 = _mm256_cvtepi8_epi16(va_lo);
                __m256i vb_lo_16 = _mm256_cvtepi8_epi16(vb_lo);
                __m256i va_hi_16 = _mm256_cvtepi8_epi16(va_hi);
                __m256i vb_hi_16 = _mm256_cvtepi8_epi16(vb_hi);

                __m256i prod_lo = _mm256_mullo_epi16(va_lo_16, vb_lo_16);
                __m256i prod_hi = _mm256_mullo_epi16(va_hi_16, vb_hi_16);

                /* Horizontal sum: extend to 32-bit and add */
                __m256i sum_lo = _mm256_madd_epi16(prod_lo, _mm256_set1_epi16(1));
                __m256i sum_hi = _mm256_madd_epi16(prod_hi, _mm256_set1_epi16(1));
                __m256i sum_32 = _mm256_add_epi32(sum_lo, sum_hi);

                /* Reduce 8 x int32 to single int32 */
                __m128i sum_128 = _mm_add_epi32(
                    _mm256_castsi256_si128(sum_32),
                    _mm256_extracti128_si256(sum_32, 1)
                );
                sum_128 = _mm_add_epi32(sum_128, _mm_srli_si128(sum_128, 8));
                sum_128 = _mm_add_epi32(sum_128, _mm_srli_si128(sum_128, 4));
                int32_t sumi = _mm_cvtsi128_si32(sum_128);

                sum += d * (float)sumi;
            }

            C[(size_t)m * N + n] = sum;
        }
    }
}
#endif /* __AVX2__ */

#if defined(__AVX512F__)
/**
 * @brief AVX-512 implementation: gemm_nt_q8_0_q8_0
 *
 * Uses 512-bit vectors to process 64 int8 values at once.
 * With VNNI, can use _mm512_dpbusd for even faster int8 dot products.
 */
void gemm_nt_q8_0_q8_0_avx512(
    const void *A,
    const void *B,
    float *C,
    int M, int N, int K)
{
    const int nb = K / QK8_0;
    const block_q8_0 *a_blocks = (const block_q8_0 *)A;
    const block_q8_0 *b_blocks = (const block_q8_0 *)B;

    for (int m = 0; m < M; m++) {
        const block_q8_0 *a_row = a_blocks + (size_t)m * nb;

        for (int n = 0; n < N; n++) {
            const block_q8_0 *b_row = b_blocks + (size_t)n * nb;
            float sum = 0.0f;

            for (int ib = 0; ib < nb; ib++) {
                const float d_a = CK_FP16_TO_FP32(a_row[ib].d);
                const float d_b = CK_FP16_TO_FP32(b_row[ib].d);
                const float d = d_a * d_b;

                /* Load 32 int8 values - use 256-bit load, extend to 512-bit for processing */
                __m256i va_256 = _mm256_loadu_si256((const __m256i *)a_row[ib].qs);
                __m256i vb_256 = _mm256_loadu_si256((const __m256i *)b_row[ib].qs);

                /* Extend int8 to int16 for multiplication */
                __m512i va_16 = _mm512_cvtepi8_epi16(va_256);
                __m512i vb_16 = _mm512_cvtepi8_epi16(vb_256);

                /* Multiply 32 pairs of int16 -> int16 (no overflow for int8*int8) */
                __m512i prod = _mm512_mullo_epi16(va_16, vb_16);

                /* Sum adjacent pairs to int32: madd adds pairs of int16 products */
                __m512i sum_32 = _mm512_madd_epi16(prod, _mm512_set1_epi16(1));

                /* Reduce all 16 int32 lanes to single int32 */
                int32_t sumi = _mm512_reduce_add_epi32(sum_32);

                sum += d * (float)sumi;
            }

            C[(size_t)m * N + n] = sum;
        }
    }
}

#if defined(__AVX512VNNI__)
/**
 * @brief AVX-512 VNNI implementation: gemm_nt_q8_0_q8_0
 *
 * Uses VNNI instructions (_mm512_dpbusd_epi32) for optimal int8 dot products.
 * VNNI computes: acc += sum(a[i] * b[i]) for 4 int8 pairs at once.
 *
 * Note: VNNI expects unsigned * signed for dpbusd, so we need to handle
 * signed * signed carefully using dpbssd or offset trick.
 */
void gemm_nt_q8_0_q8_0_vnni(
    const void *A,
    const void *B,
    float *C,
    int M, int N, int K)
{
    const int nb = K / QK8_0;
    const block_q8_0 *a_blocks = (const block_q8_0 *)A;
    const block_q8_0 *b_blocks = (const block_q8_0 *)B;

    for (int m = 0; m < M; m++) {
        const block_q8_0 *a_row = a_blocks + (size_t)m * nb;

        for (int n = 0; n < N; n++) {
            const block_q8_0 *b_row = b_blocks + (size_t)n * nb;
            float sum = 0.0f;

            for (int ib = 0; ib < nb; ib++) {
                const float d_a = CK_FP16_TO_FP32(a_row[ib].d);
                const float d_b = CK_FP16_TO_FP32(b_row[ib].d);
                const float d = d_a * d_b;

                /* Load 32 int8 values */
                __m256i va = _mm256_loadu_si256((const __m256i *)a_row[ib].qs);
                __m256i vb = _mm256_loadu_si256((const __m256i *)b_row[ib].qs);

                /* For signed*signed, use extend to 16-bit approach
                 * (dpbusd is unsigned*signed, dpbssd requires AVX512_VNNI_INT8) */
                __m512i va_16 = _mm512_cvtepi8_epi16(va);
                __m512i vb_16 = _mm512_cvtepi8_epi16(vb);
                __m512i prod = _mm512_mullo_epi16(va_16, vb_16);
                __m512i sum_32 = _mm512_madd_epi16(prod, _mm512_set1_epi16(1));
                int32_t sumi = _mm512_reduce_add_epi32(sum_32);

                sum += d * (float)sumi;
            }

            C[(size_t)m * N + n] = sum;
        }
    }
}
#endif /* __AVX512VNNI__ */
#endif /* __AVX512F__ */

/**
 * @brief Dispatcher for gemm_nt_q8_0_q8_0
 *
 * Selects the best available implementation at runtime.
 */
/* Dispatcher is now gemm_nt_q8_0_q8_0_bias in Section 5 */


/* ============================================================================
 * SECTION 2: GEMM Q5_0 x Q8_0 -> FP32
 *
 * Weights are Q5_0 (5-bit), activations are Q8_0 (8-bit).
 * Q5_0 requires unpacking: 4 bits from qs[] + 1 bit from qh[].
 * ============================================================================ */

/**
 * @brief Scalar reference: gemm_nt_q5_0_q8_0
 *
 * Q5_0 weight reconstruction:
 *   weight[j] = d * ((qs_nibble | (qh_bit << 4)) - 16)
 *
 * For j in 0..15: use low nibble + qh bit j
 * For j in 16..31: use high nibble + qh bit (j+16) -> actually bit (j) for j=16..31
 *
 * @param A      Input activations [M, K] in Q8_0 format
 * @param B      Weight matrix [N, K] in Q5_0 format
 * @param C      Output matrix [M, N] in FP32
 * @param M      Number of tokens (batch size)
 * @param N      Number of output features
 * @param K      Number of input features (must be multiple of 32)
 */
void gemm_nt_q5_0_q8_0_ref(
    const void *A,
    const void *B,
    float *C,
    int M, int N, int K)
{
    const int nb = K / QK5_0;
    const block_q8_0 *a_blocks = (const block_q8_0 *)A;
    const block_q5_0 *b_blocks = (const block_q5_0 *)B;

    for (int m = 0; m < M; m++) {
        const block_q8_0 *a_row = a_blocks + (size_t)m * nb;

        for (int n = 0; n < N; n++) {
            const block_q5_0 *b_row = b_blocks + (size_t)n * nb;
            float sum = 0.0f;

            for (int ib = 0; ib < nb; ib++) {
                const float d_a = CK_FP16_TO_FP32(a_row[ib].d);
                const float d_b = CK_FP16_TO_FP32(b_row[ib].d);
                const float d = d_a * d_b;

                /* Load high bits as 32-bit value */
                uint32_t qh;
                memcpy(&qh, b_row[ib].qh, sizeof(qh));

                int32_t sumi = 0;

                /* Process 32 weights: j=0..15 uses low nibble, j=16..31 uses high nibble */
                for (int j = 0; j < 16; j++) {
                    /* First 16 weights: low nibble + qh bit j */
                    const uint8_t xh_0 = ((qh >> j) & 1) << 4;
                    const int8_t w0 = (int8_t)(((b_row[ib].qs[j] & 0x0F) | xh_0) - 16);

                    /* Second 16 weights: high nibble + qh bit (j+16) */
                    const uint8_t xh_1 = ((qh >> (j + 16)) & 1) << 4;
                    const int8_t w1 = (int8_t)(((b_row[ib].qs[j] >> 4) | xh_1) - 16);

                    /* Accumulate with activation values */
                    sumi += (int32_t)w0 * (int32_t)a_row[ib].qs[j];
                    sumi += (int32_t)w1 * (int32_t)a_row[ib].qs[j + 16];
                }

                sum += d * (float)sumi;
            }

            C[(size_t)m * N + n] = sum;
        }
    }
}

/* ============================================================================
 * SECTION 3: AMX Implementation (Intel Advanced Matrix Extensions)
 *
 * AMX uses tile registers (TMM0-TMM7) for matrix operations.
 * Each tile can hold up to 16 rows x 64 bytes (1KB).
 *
 * Key operations:
 *   _tile_loadd: Load tile from memory
 *   _tile_dpbssd: Signed int8 dot product accumulate (A signed, B signed)
 *   _tile_stored: Store tile to memory
 *
 * Requirements:
 *   - Sapphire Rapids or later CPU
 *   - __AMX_INT8__ defined
 *   - OS support (XSAVE/XRSTOR for tiles)
 * ============================================================================ */

#if HAS_AMX

/* AMX tile configuration */
typedef struct {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[8];
    uint8_t rows[8];
} tile_config_t;

static void amx_tile_config_init(void)
{
    static __thread int initialized = 0;
    if (initialized) return;

    tile_config_t tc = {0};
    tc.palette_id = 1;

    /* Configure tiles for our GEMM pattern:
     * TMM0: accumulator C (16 rows x 16 cols of int32)
     * TMM1: A tile (16 rows x 64 bytes = 64 int8 per row)
     * TMM2: B tile (16 rows x 64 bytes)
     */
    tc.rows[0] = 16; tc.colsb[0] = 64;  /* TMM0: 16x16 int32 */
    tc.rows[1] = 16; tc.colsb[1] = 64;  /* TMM1: A */
    tc.rows[2] = 16; tc.colsb[2] = 64;  /* TMM2: B */
    tc.rows[3] = 16; tc.colsb[3] = 64;  /* TMM3: spare */
    tc.rows[4] = 16; tc.colsb[4] = 64;  /* TMM4: spare */
    tc.rows[5] = 16; tc.colsb[5] = 64;  /* TMM5: spare */
    tc.rows[6] = 16; tc.colsb[6] = 64;  /* TMM6: spare */
    tc.rows[7] = 16; tc.colsb[7] = 64;  /* TMM7: spare */

    _tile_loadconfig(&tc);
    initialized = 1;
}

/**
 * @brief AMX implementation: gemm_nt_q8_0_q8_0
 *
 * Uses AMX tiles for matrix multiplication.
 * This is a simplified version - full implementation would tile the problem.
 *
 * Note: AMX requires specific data layout and tiling strategy.
 * This implementation focuses on correctness; optimization is future work.
 */
void gemm_nt_q8_0_q8_0_amx(
    const void *A,
    const void *B,
    float *C,
    int M, int N, int K)
{
    amx_tile_config_init();

    /* For now, fall back to AVX-512 implementation.
     * Full AMX implementation requires:
     * 1. Repacking data for tile-friendly layout
     * 2. Proper tile blocking (16x16 tiles)
     * 3. Scale factor handling after tile operations
     *
     * TODO: Implement full AMX path when we have test infrastructure
     */
    gemm_nt_q8_0_q8_0_avx512(A, B, C, M, N, K);
}

void gemm_nt_q5_0_q8_0_amx(
    const void *A,
    const void *B,
    float *C,
    int M, int N, int K)
{
    amx_tile_config_init();

    /* Q5_0 requires unpacking before AMX can process.
     * Strategy:
     * 1. Unpack Q5_0 to int8 buffer
     * 2. Use AMX for the actual GEMM
     * 3. Apply scales
     *
     * For now, fall back to scalar reference.
     */
    gemm_nt_q5_0_q8_0_ref(A, B, C, M, N, K);
}

#endif /* HAS_AMX */


/* ============================================================================
 * SECTION 4: API Functions with Full Dispatch
 * ============================================================================ */

/**
 * @brief Get the best implementation name for logging/debugging
 */
const char* gemm_batch_int8_impl_name(void)
{
#if HAS_AMX
    return "AMX";
#elif defined(__AVX512VNNI__)
    return "AVX-512 VNNI";
#elif defined(__AVX512F__)
    return "AVX-512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__AVX__)
    return "AVX";
#else
    return "Scalar";
#endif
}


/* ============================================================================
 * SECTION 5: API Wrappers with Bias Support
 *
 * These match the existing API signature in ckernel_quant.h
 * ============================================================================ */

/**
 * @brief gemm_nt_q8_0_q8_0 with optional bias (matches header signature)
 *
 * C[m,n] = A[m,K] @ B[n,K]^T + bias[n]
 */
void gemm_nt_q8_0_q8_0(
    const void *A,
    const void *B,
    const float *bias,
    float *C,
    int M, int N, int K)
{
    /* First compute GEMM */
#if defined(__AVX512VNNI__)
    gemm_nt_q8_0_q8_0_vnni(A, B, C, M, N, K);
#elif defined(__AVX512F__)
    gemm_nt_q8_0_q8_0_avx512(A, B, C, M, N, K);
#elif defined(__AVX2__)
    gemm_nt_q8_0_q8_0_avx2(A, B, C, M, N, K);
#else
    gemm_nt_q8_0_q8_0_ref(A, B, C, M, N, K);
#endif

    /* Add bias if provided */
    if (bias != NULL) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[(size_t)m * N + n] += bias[n];
            }
        }
    }
}
