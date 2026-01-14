/**
 * @file gemm_kernels_amx.c
 * @brief AMX (Advanced Matrix Extensions) GEMM kernels
 *
 * Intel AMX provides dedicated matrix multiply hardware:
 *   - 8 tile registers (TMM0-TMM7), each up to 1KB
 *   - TDPBSSD: INT8 signed dot product (A signed, B signed)
 *   - TDPBSUD: INT8 mixed sign (A signed, B unsigned)
 *   - TDPBUSD: INT8 mixed sign (A unsigned, B signed)
 *   - TDPBUUD: INT8 unsigned dot product
 *   - TDPBF16PS: BF16 dot product to FP32
 *
 * Tile dimensions:
 *   - Max: 16 rows x 64 bytes (1024 bytes per tile)
 *   - For INT8: 16x64 elements
 *   - For BF16: 16x32 elements
 *
 * Performance:
 *   - AMX INT8: ~2000 INT8 ops/cycle (vs ~256 for AVX-512 VNNI)
 *   - AMX BF16: ~1000 BF16 ops/cycle
 *   - Expected 8-16x speedup over AVX-512 for large GEMM
 *
 * Requirements:
 *   - Sapphire Rapids or newer (4th Gen Xeon)
 *   - Linux kernel 5.16+ with AMX support
 *   - Compiler: GCC 11+, Clang 12+, ICX 2022+
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

#include "ckernel_quant.h"

/* AMX requires specific compiler support */
#if defined(__AMX_INT8__) || defined(__AMX_TILE__)

#include <immintrin.h>

/* Tile configuration structure */
typedef struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];  /* Columns in bytes for each tile */
    uint8_t rows[16];    /* Rows for each tile */
} __tile_config;

/* AMX tile dimensions for our use case */
#define AMX_TILE_M 16      /* Rows per tile (matches hardware max) */
#define AMX_TILE_N 16      /* Output columns (16 int32 = 64 bytes) */
#define AMX_TILE_K 64      /* K dimension (64 int8 = 64 bytes) */

/* Tile register assignments for GEMM:
 * TMM0: A tile (activations)
 * TMM1: B tile (weights)
 * TMM2: C tile (accumulator)
 * TMM3-7: Reserved for larger blocking
 */
#define TILE_A 0
#define TILE_B 1
#define TILE_C 2

/**
 * @brief Configure AMX tiles for GEMM operation
 *
 * Must be called before using tile instructions.
 * Tiles:
 *   - TILE_A: M x K bytes (activations, INT8)
 *   - TILE_B: K x N bytes (weights, INT8, must be K rows for TDPB*)
 *   - TILE_C: M x (N*4) bytes (accumulator, INT32)
 */
static void configure_tiles_gemm(int M, int N, int K) {
    __tile_config config = {0};

    config.palette_id = 1;  /* Use palette 1 (standard) */

    /* Tile A: M rows, K columns (K bytes per row) */
    int tile_m = (M > AMX_TILE_M) ? AMX_TILE_M : M;
    int tile_k = (K > AMX_TILE_K) ? AMX_TILE_K : K;
    int tile_n = (N > AMX_TILE_N) ? AMX_TILE_N : N;

    /* TILE_A: Input activations (INT8) */
    config.rows[TILE_A] = tile_m;
    config.colsb[TILE_A] = tile_k;

    /* TILE_B: Weights (INT8) - note: for TDPBSSD, B has K rows, N cols */
    config.rows[TILE_B] = tile_k;
    config.colsb[TILE_B] = tile_n * 4;  /* N int32 outputs = N*4 bytes */

    /* TILE_C: Accumulator (INT32) */
    config.rows[TILE_C] = tile_m;
    config.colsb[TILE_C] = tile_n * 4;  /* N int32 outputs = N*4 bytes */

    _tile_loadconfig(&config);
}

/**
 * @brief Release AMX tile configuration
 */
static void release_tiles(void) {
    _tile_release();
}

/**
 * @brief AMX INT8 GEMM: C[M,N] += A[M,K] @ B[K,N]
 *
 * Uses TDPBSSD (signed int8 x signed int8 -> int32 accumulate)
 *
 * @param A      INT8 activations [M, K], row-major
 * @param B      INT8 weights [K, N], column-major (transposed for efficiency)
 * @param C      INT32 accumulator [M, N], row-major
 * @param M      Output rows
 * @param N      Output columns
 * @param K      Inner dimension
 */
void gemm_amx_int8_core(
    const int8_t *A,
    const int8_t *B,
    int32_t *C,
    int M, int N, int K)
{
    /* Configure tiles for this GEMM size */
    configure_tiles_gemm(M, N, K);

    /* Process in tiles */
    for (int m = 0; m < M; m += AMX_TILE_M) {
        int tile_m = (m + AMX_TILE_M <= M) ? AMX_TILE_M : (M - m);

        for (int n = 0; n < N; n += AMX_TILE_N) {
            int tile_n = (n + AMX_TILE_N <= N) ? AMX_TILE_N : (N - n);

            /* Zero accumulator tile */
            _tile_zero(TILE_C);

            /* Accumulate over K dimension */
            for (int k = 0; k < K; k += AMX_TILE_K) {
                int tile_k = (k + AMX_TILE_K <= K) ? AMX_TILE_K : (K - k);

                /* Load A tile: A[m:m+tile_m, k:k+tile_k] */
                _tile_loadd(TILE_A, A + m * K + k, K);

                /* Load B tile: B[k:k+tile_k, n:n+tile_n]
                 * Note: B is stored column-major for efficient AMX access */
                _tile_loadd(TILE_B, B + k * N + n, N * 4);

                /* TDPBSSD: C += A @ B (signed int8 dot product) */
                _tile_dpbssd(TILE_C, TILE_A, TILE_B);
            }

            /* Store C tile: C[m:m+tile_m, n:n+tile_n] */
            _tile_stored(TILE_C, C + m * N + n, N * 4);
        }
    }

    release_tiles();
}

/**
 * @brief AMX GEMV for Q4_K x Q8_K: y[M] = W[M,K] @ x[K]
 *
 * Adapts the tile-based AMX for vector operations by treating the vector
 * as a 1-row matrix.
 *
 * For decode (single token), M=1, so we batch multiple output rows together
 * to better utilize AMX tiles.
 *
 * @param y         Output (FP32), shape [M]
 * @param W         Weights in Q4_K format, shape [M, K]
 * @param x_q8      Input in Q8_K format, shape [K]
 * @param M         Output dimension
 * @param K         Input dimension (must be multiple of 256)
 */
void gemv_q4_k_q8_k_amx(float *y,
                         const void *W,
                         const void *x_q8,
                         int M, int K)
{
    /* For single-token decode, AMX overhead may not be worth it.
     * Use AMX only if M >= 16 (can fill a tile row).
     *
     * For M < 16, fall back to VNNI which has lower setup overhead.
     */
    if (M < AMX_TILE_M) {
        /* Fall back to VNNI for small M */
        extern void gemv_q4_k_q8_k_vnni(float *y, const void *W, const void *x_q8, int M, int K);
        gemv_q4_k_q8_k_vnni(y, W, x_q8, M, K);
        return;
    }

    /* For larger M (prefill), use AMX */
    const block_q4_K *blocks = (const block_q4_K *)W;
    const block_q8_K *bx = (const block_q8_K *)x_q8;
    const int nb = K / QK_K;

    /* TODO: Implement full AMX path for Q4_K x Q8_K
     *
     * The challenge: Q4_K uses 4-bit quantization with per-block scales,
     * while AMX works best with uniform INT8 data.
     *
     * Options:
     * 1. Dequantize Q4_K to INT8 first, then use AMX
     * 2. Use AMX for partial computation, scale adjustment outside
     * 3. Keep Q4_K in VNNI, use AMX only for Q8_0 x Q8_0
     *
     * For now, fall back to VNNI for Q4_K.
     */
    extern void gemv_q4_k_q8_k_vnni(float *y, const void *W, const void *x_q8, int M, int K);
    gemv_q4_k_q8_k_vnni(y, W, x_q8, M, K);
}

/**
 * @brief AMX GEMM for Q8_0 x Q8_0: C[M,N] = A[M,K] @ B[N,K]^T
 *
 * This is the ideal case for AMX - uniform INT8 data on both sides.
 * AMX can directly compute INT8 x INT8 -> INT32 accumulation.
 *
 * @param A         Input activations in Q8_0 format [M, K]
 * @param B         Weight matrix in Q8_0 format [N, K]
 * @param C         Output in FP32 [M, N]
 * @param M         Number of tokens (batch size)
 * @param N         Number of output features
 * @param K         Number of input features (must be multiple of 32)
 */
void gemm_nt_q8_0_q8_0_amx(
    const void *A,
    const void *B,
    float *C,
    int M, int N, int K)
{
    const int nb = K / QK8_0;
    const block_q8_0 *a_blocks = (const block_q8_0 *)A;
    const block_q8_0 *b_blocks = (const block_q8_0 *)B;

    /* For small M (decode), AMX overhead > benefit */
    if (M < AMX_TILE_M) {
        extern void gemm_nt_q8_0_q8_0_avx512(const void *A, const void *B, float *C, int M, int N, int K);
        gemm_nt_q8_0_q8_0_avx512(A, B, C, M, N, K);
        return;
    }

    configure_tiles_gemm(M, N, K);

    /* Process output in tiles */
    for (int m = 0; m < M; m += AMX_TILE_M) {
        int tile_m = (m + AMX_TILE_M <= M) ? AMX_TILE_M : (M - m);

        for (int n = 0; n < N; n += AMX_TILE_N) {
            int tile_n = (n + AMX_TILE_N <= N) ? AMX_TILE_N : (N - n);

            /* Accumulator for this tile (in INT32) */
            int32_t acc[AMX_TILE_M * AMX_TILE_N] = {0};

            /* Scale accumulator (for FP32 output) */
            float scale_acc[AMX_TILE_M * AMX_TILE_N] = {0};

            /* Process blocks */
            for (int ib = 0; ib < nb; ib++) {
                /* Extract scales for this block */
                float d_a[AMX_TILE_M];
                float d_b[AMX_TILE_N];

                for (int i = 0; i < tile_m && (m + i) < M; i++) {
                    const block_q8_0 *a_row = a_blocks + (size_t)(m + i) * nb;
                    d_a[i] = CK_FP16_TO_FP32(a_row[ib].d);
                }

                for (int j = 0; j < tile_n && (n + j) < N; j++) {
                    const block_q8_0 *b_row = b_blocks + (size_t)(n + j) * nb;
                    d_b[j] = CK_FP16_TO_FP32(b_row[ib].d);
                }

                /* Prepare INT8 tiles for AMX
                 * A_tile[i,k] = a_blocks[m+i].qs[k]
                 * B_tile[k,j] = b_blocks[n+j].qs[k]
                 */
                int8_t A_tile[AMX_TILE_M * QK8_0] __attribute__((aligned(64)));
                int8_t B_tile[QK8_0 * AMX_TILE_N] __attribute__((aligned(64)));

                /* Fill A tile */
                for (int i = 0; i < tile_m && (m + i) < M; i++) {
                    const block_q8_0 *a_row = a_blocks + (size_t)(m + i) * nb;
                    memcpy(&A_tile[i * QK8_0], a_row[ib].qs, QK8_0);
                }

                /* Fill B tile (transposed for column-major access) */
                for (int j = 0; j < tile_n && (n + j) < N; j++) {
                    const block_q8_0 *b_row = b_blocks + (size_t)(n + j) * nb;
                    for (int k = 0; k < QK8_0; k++) {
                        B_tile[k * AMX_TILE_N + j] = b_row[ib].qs[k];
                    }
                }

                /* Zero accumulator tile */
                _tile_zero(TILE_C);

                /* Load tiles and compute */
                _tile_loadd(TILE_A, A_tile, QK8_0);
                _tile_loadd(TILE_B, B_tile, AMX_TILE_N);
                _tile_dpbssd(TILE_C, TILE_A, TILE_B);

                /* Store and accumulate with scales */
                int32_t C_tile[AMX_TILE_M * AMX_TILE_N] __attribute__((aligned(64)));
                _tile_stored(TILE_C, C_tile, AMX_TILE_N * 4);

                /* Apply scales and accumulate to FP32 */
                for (int i = 0; i < tile_m && (m + i) < M; i++) {
                    for (int j = 0; j < tile_n && (n + j) < N; j++) {
                        float d = d_a[i] * d_b[j];
                        scale_acc[i * AMX_TILE_N + j] += d * (float)C_tile[i * AMX_TILE_N + j];
                    }
                }
            }

            /* Write final output */
            for (int i = 0; i < tile_m && (m + i) < M; i++) {
                for (int j = 0; j < tile_n && (n + j) < N; j++) {
                    C[(m + i) * N + (n + j)] = scale_acc[i * AMX_TILE_N + j];
                }
            }
        }
    }

    release_tiles();
}

/**
 * @brief Check if AMX is available at runtime
 */
bool amx_available(void) {
    /* Check CPUID for AMX support */
    unsigned int eax, ebx, ecx, edx;

    /* CPUID leaf 7, subleaf 0 */
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );

    /* AMX-TILE: EDX bit 24 */
    /* AMX-INT8: EDX bit 25 */
    /* AMX-BF16: EDX bit 22 */
    bool has_amx_tile = (edx >> 24) & 1;
    bool has_amx_int8 = (edx >> 25) & 1;

    return has_amx_tile && has_amx_int8;
}

#else /* No AMX support */

#include <stdbool.h>

void gemv_q4_k_q8_k_amx(float *y, const void *W, const void *x_q8, int M, int K) {
    /* Fall back to VNNI or AVX-512 */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    extern void gemv_q4_k_q8_k_vnni(float *y, const void *W, const void *x_q8, int M, int K);
    gemv_q4_k_q8_k_vnni(y, W, x_q8, M, K);
#elif defined(__AVX512F__)
    extern void gemv_q4_k_q8_k_avx2(float *y, const void *W, const void *x_q8, int M, int K);
    gemv_q4_k_q8_k_avx2(y, W, x_q8, M, K);
#else
    extern void gemv_q4_k_q8_k_ref(float *y, const void *W, const void *x_q8, int M, int K);
    gemv_q4_k_q8_k_ref(y, W, x_q8, M, K);
#endif
}

void gemm_nt_q8_0_q8_0_amx(const void *A, const void *B, float *C, int M, int N, int K) {
    extern void gemm_nt_q8_0_q8_0_ref(const void *A, const void *B, float *C, int M, int N, int K);
    gemm_nt_q8_0_q8_0_ref(A, B, C, M, N, K);
}

bool amx_available(void) {
    return false;
}

#endif /* __AMX_INT8__ */
