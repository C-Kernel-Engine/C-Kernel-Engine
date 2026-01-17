/**
 * @file gemm_microkernel.c
 * @brief GEMM Microkernel - High-Performance Register-Blocked Matrix Multiplication
 *
 * CK-ENGINE KERNEL RULES:
 * =======================
 * 1. NO malloc/free - memory via bump allocator, pointers passed in
 * 2. NO OpenMP - parallelization at orchestrator/codegen layer
 * 3. API must define: inputs, outputs, workspace, and memory layouts
 * 4. Pure computation - deterministic, no side effects
 *
 * After changes: make test && make llamacpp-parity-full
 *
 * This file implements optimized GEMM microkernels with multiple backends:
 *
 * 1. USE_MKL: Intel MKL cblas_sgemm (best performance on Intel CPUs)
 * 2. USE_ONEDNN: Intel oneDNN matmul primitive (Apache 2.0 licensed)
 * 3. Native: Our own AVX-512/AVX2/AVX microkernels (no dependencies)
 *
 * Build with:
 *   make USE_MKL=1      # Use Intel MKL
 *   make USE_ONEDNN=1   # Use Intel oneDNN
 *   make                # Use native kernels
 *
 * Layout: C[M,N] = A[M,K] @ B[K,N] (row-major)
 */

#include "ckernel_engine.h"
#include "cpu_features.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// =============================================================================
// Backend Selection: MKL > oneDNN > Native
// =============================================================================

#if defined(USE_MKL)
    #include <mkl.h>
    #define GEMM_BACKEND "MKL"
#elif defined(USE_ONEDNN)
    #include <dnnl.h>
    #define GEMM_BACKEND "oneDNN"
#else
    #define GEMM_BACKEND "Native"
#endif

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// MKL Backend Implementation
// =============================================================================

#if defined(USE_MKL)

void gemm_microkernel(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K,
    int B_transposed
)
{
    // MKL uses column-major by default, but CblasRowMajor handles row-major
    // C = alpha * A @ B + beta * C
    // For B_transposed: C = A @ B^T
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        B_transposed ? CblasTrans : CblasNoTrans,
        M, N, K,
        1.0f,           // alpha
        A, K,                       // lda (A is always [M,K] row-major)
        B, B_transposed ? K : N,  // ldb
        0.0f,           // beta
        C, N            // ldc
    );
}

// Stub implementations for blocked versions (MKL handles everything)
void gemm_microkernel_blocked(const float *A, const float *B, float *C, int M, int N, int K) {
    gemm_microkernel(A, B, C, M, N, K, 0);
}
void gemm_microkernel_packed(const float *A, const float *B, float *C, int M, int N, int K) {
    gemm_microkernel(A, B, C, M, N, K, 0);
}
void gemm_microkernel_blocked_bt(const float *A, const float *B, float *C, int M, int N, int K) {
    gemm_microkernel(A, B, C, M, N, K, 1);
}

#elif defined(USE_ONEDNN)

// =============================================================================
// oneDNN Backend Implementation
// =============================================================================

// Global oneDNN engine and stream (initialized once)
static dnnl_engine_t g_engine = NULL;
static dnnl_stream_t g_stream = NULL;

static void onednn_init(void) {
    if (g_engine) return;
    dnnl_engine_create(&g_engine, dnnl_cpu, 0);
    dnnl_stream_create(&g_stream, g_engine, dnnl_stream_default_flags);
}

void gemm_microkernel(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K,
    int B_transposed
)
{
    onednn_init();

    // Create memory descriptors for row-major matrices
    dnnl_memory_desc_t a_md, b_md, c_md;
    dnnl_dims_t a_dims = {M, K};
    dnnl_dims_t b_dims = {K, N};
    dnnl_dims_t c_dims = {M, N};
    dnnl_dims_t a_strides = {K, 1};
    dnnl_dims_t b_strides = {N, 1}; /* default: B is [K,N] row-major */
    dnnl_dims_t c_strides = {N, 1};

    if (B_transposed) {
        /*
         * Our "B_transposed" convention means: caller stores B as [N,K] row-major,
         * but wants C = A[M,K] @ B[N,K]^T => treat weights as [K,N].
         *
         * oneDNN matmul has no transpose flag, so represent B^T as a strided view:
         *   dims    = [K, N]
         *   strides = [1, K]  (offset(k,n) = k + n*K == B[n*K + k])
         */
        b_strides[0] = 1;
        b_strides[1] = K;
    }

    dnnl_memory_desc_create_with_strides(&a_md, 2, a_dims, dnnl_f32, a_strides);
    dnnl_memory_desc_create_with_strides(&b_md, 2, b_dims, dnnl_f32, b_strides);
    dnnl_memory_desc_create_with_strides(&c_md, 2, c_dims, dnnl_f32, c_strides);

    // Create matmul primitive descriptor
    dnnl_primitive_desc_t matmul_pd;
    dnnl_matmul_primitive_desc_create(&matmul_pd, g_engine, a_md, b_md, NULL, c_md, NULL);

    // Create primitive
    dnnl_primitive_t matmul;
    dnnl_primitive_create(&matmul, matmul_pd);

    // Create memory objects
    dnnl_memory_t a_mem, b_mem, c_mem;
    dnnl_memory_create(&a_mem, a_md, g_engine, (void*)A);
    dnnl_memory_create(&b_mem, b_md, g_engine, (void*)B);
    dnnl_memory_create(&c_mem, c_md, g_engine, (void*)C);

    // Execute
    dnnl_exec_arg_t args[3] = {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    };
    dnnl_primitive_execute(matmul, g_stream, 3, args);
    dnnl_stream_wait(g_stream);

    // Cleanup
    dnnl_primitive_destroy(matmul);
    dnnl_primitive_desc_destroy(matmul_pd);
    dnnl_memory_destroy(a_mem);
    dnnl_memory_destroy(b_mem);
    dnnl_memory_destroy(c_mem);
    dnnl_memory_desc_destroy(a_md);
    dnnl_memory_desc_destroy(b_md);
    dnnl_memory_desc_destroy(c_md);
}

void gemm_microkernel_blocked(const float *A, const float *B, float *C, int M, int N, int K) {
    gemm_microkernel(A, B, C, M, N, K, 0);
}
void gemm_microkernel_packed(const float *A, const float *B, float *C, int M, int N, int K) {
    gemm_microkernel(A, B, C, M, N, K, 0);
}
void gemm_microkernel_blocked_bt(const float *A, const float *B, float *C, int M, int N, int K) {
    gemm_microkernel(A, B, C, M, N, K, 1);
}

#else
// =============================================================================
// Native Backend (our own AVX-512/AVX2/AVX kernels)
// =============================================================================

// =============================================================================
// Microkernel Configuration
//
// MR/NR are fixed at compile time (microkernel register usage)
// MC/NC/KC are determined at runtime based on detected cache sizes
// =============================================================================

#if defined(__AVX512F__)
    // AVX-512: 6x32 microkernel (6 rows, 32 cols = 2 ZMM per row = 12 ZMM accumulators)
    // 32 ZMM registers available - no spilling
    #define MR_FIXED 6
    #define NR_FIXED 32
#elif defined(__FMA__)
    // AVX2+FMA: 6x16 microkernel - FMA hides latency, some spilling acceptable
    #define MR_FIXED 6
    #define NR_FIXED 16
#elif defined(__AVX__)
    // AVX (no FMA): 4x16 microkernel to avoid register spilling
    // Only 16 YMM registers: 8 accum + 2 B + 4 A + 2 temp = 16 (fits!)
    #define MR_FIXED 4
    #define NR_FIXED 16
#else
    // Scalar fallback
    #define MR_FIXED 4
    #define NR_FIXED 4
#endif

// These macros use runtime-detected values (initialized once at startup)
#define MR (MR_FIXED)
#define NR (NR_FIXED)
#define MC (get_gemm_params()->MC)
#define NC (get_gemm_params()->NC)
#define KC (get_gemm_params()->KC)

// =============================================================================
// AVX-512 6x32 Microkernel - oneDNN style
//
// Computes: C[0:6, 0:32] += A[0:6, 0:K] @ B[0:K, 0:32]
//
// Register usage (32 ZMM registers available):
// - c0_lo, c0_hi, c1_lo, c1_hi, ... c5_hi: 12 accumulators (2 per row)
// - b_lo, b_hi: 2 registers for B row
// - a0-a5: 6 registers for A broadcasts
// - Remaining for prefetch and temp
// =============================================================================

#if defined(__AVX512F__)
static inline void gemm_microkernel_6x32_avx512(
    int K,
    const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,
    float * __restrict__ C, int ldc,
    int first_k
)
{
    // 12 accumulators: 6 rows x 2 ZMM (32 floats) per row
    __m512 c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi;
    __m512 c3_lo, c3_hi, c4_lo, c4_hi, c5_lo, c5_hi;

    if (first_k) {
        c0_lo = _mm512_setzero_ps(); c0_hi = _mm512_setzero_ps();
        c1_lo = _mm512_setzero_ps(); c1_hi = _mm512_setzero_ps();
        c2_lo = _mm512_setzero_ps(); c2_hi = _mm512_setzero_ps();
        c3_lo = _mm512_setzero_ps(); c3_hi = _mm512_setzero_ps();
        c4_lo = _mm512_setzero_ps(); c4_hi = _mm512_setzero_ps();
        c5_lo = _mm512_setzero_ps(); c5_hi = _mm512_setzero_ps();
    } else {
        c0_lo = _mm512_loadu_ps(&C[0 * ldc]);      c0_hi = _mm512_loadu_ps(&C[0 * ldc + 16]);
        c1_lo = _mm512_loadu_ps(&C[1 * ldc]);      c1_hi = _mm512_loadu_ps(&C[1 * ldc + 16]);
        c2_lo = _mm512_loadu_ps(&C[2 * ldc]);      c2_hi = _mm512_loadu_ps(&C[2 * ldc + 16]);
        c3_lo = _mm512_loadu_ps(&C[3 * ldc]);      c3_hi = _mm512_loadu_ps(&C[3 * ldc + 16]);
        c4_lo = _mm512_loadu_ps(&C[4 * ldc]);      c4_hi = _mm512_loadu_ps(&C[4 * ldc + 16]);
        c5_lo = _mm512_loadu_ps(&C[5 * ldc]);      c5_hi = _mm512_loadu_ps(&C[5 * ldc + 16]);
    }

    // Prefetch first cache lines
    _mm_prefetch((const char*)&B[0], _MM_HINT_T0);
    _mm_prefetch((const char*)&B[64], _MM_HINT_T0);

    // Main K loop - unrolled by 8 with software pipelining for better ILP
    int k = 0;
    for (; k <= K - 8; k += 8) {
        // Prefetch ahead - 16 rows ahead for L1, 32 for L2
        _mm_prefetch((const char*)&B[(k + 16) * ldb], _MM_HINT_T0);
        _mm_prefetch((const char*)&B[(k + 16) * ldb + 64], _MM_HINT_T0);
        _mm_prefetch((const char*)&B[(k + 32) * ldb], _MM_HINT_T1);

        // Software pipelining: preload first B row
        __m512 b_lo_next = _mm512_loadu_ps(&B[k * ldb]);
        __m512 b_hi_next = _mm512_loadu_ps(&B[k * ldb + 16]);

        #define AVX512_ITER(koff) { \
            __m512 b_lo = b_lo_next; \
            __m512 b_hi = b_hi_next; \
            if ((koff) < 7) { \
                b_lo_next = _mm512_loadu_ps(&B[(k + (koff) + 1) * ldb]); \
                b_hi_next = _mm512_loadu_ps(&B[(k + (koff) + 1) * ldb + 16]); \
            } \
            __m512 a0 = _mm512_set1_ps(A[0 * lda + k + (koff)]); \
            __m512 a1 = _mm512_set1_ps(A[1 * lda + k + (koff)]); \
            __m512 a2 = _mm512_set1_ps(A[2 * lda + k + (koff)]); \
            __m512 a3 = _mm512_set1_ps(A[3 * lda + k + (koff)]); \
            __m512 a4 = _mm512_set1_ps(A[4 * lda + k + (koff)]); \
            __m512 a5 = _mm512_set1_ps(A[5 * lda + k + (koff)]); \
            c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi); \
            c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi); \
            c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi); \
            c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi); \
            c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi); \
            c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi); \
        }

        AVX512_ITER(0);
        AVX512_ITER(1);
        AVX512_ITER(2);
        AVX512_ITER(3);
        AVX512_ITER(4);
        AVX512_ITER(5);
        AVX512_ITER(6);
        AVX512_ITER(7);

        #undef AVX512_ITER
    }

    // Handle K % 8 remainder with 4-unroll
    for (; k <= K - 4; k += 4) {
        #define AVX512_ITER4(koff) { \
            __m512 b_lo = _mm512_loadu_ps(&B[(k + koff) * ldb]); \
            __m512 b_hi = _mm512_loadu_ps(&B[(k + koff) * ldb + 16]); \
            __m512 a0 = _mm512_set1_ps(A[0 * lda + k + koff]); \
            __m512 a1 = _mm512_set1_ps(A[1 * lda + k + koff]); \
            __m512 a2 = _mm512_set1_ps(A[2 * lda + k + koff]); \
            __m512 a3 = _mm512_set1_ps(A[3 * lda + k + koff]); \
            __m512 a4 = _mm512_set1_ps(A[4 * lda + k + koff]); \
            __m512 a5 = _mm512_set1_ps(A[5 * lda + k + koff]); \
            c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi); \
            c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi); \
            c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi); \
            c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi); \
            c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi); \
            c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi); \
        }
        AVX512_ITER4(0);
        AVX512_ITER4(1);
        AVX512_ITER4(2);
        AVX512_ITER4(3);
        #undef AVX512_ITER4
    }

    // Handle remaining K
    for (; k < K; k++) {
        __m512 b_lo = _mm512_loadu_ps(&B[k * ldb]);
        __m512 b_hi = _mm512_loadu_ps(&B[k * ldb + 16]);

        c0_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * lda + k]), b_lo, c0_lo);
        c0_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * lda + k]), b_hi, c0_hi);
        c1_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * lda + k]), b_lo, c1_lo);
        c1_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * lda + k]), b_hi, c1_hi);
        c2_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * lda + k]), b_lo, c2_lo);
        c2_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * lda + k]), b_hi, c2_hi);
        c3_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * lda + k]), b_lo, c3_lo);
        c3_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * lda + k]), b_hi, c3_hi);
        c4_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[4 * lda + k]), b_lo, c4_lo);
        c4_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[4 * lda + k]), b_hi, c4_hi);
        c5_lo = _mm512_fmadd_ps(_mm512_set1_ps(A[5 * lda + k]), b_lo, c5_lo);
        c5_hi = _mm512_fmadd_ps(_mm512_set1_ps(A[5 * lda + k]), b_hi, c5_hi);
    }

    // Store results
    _mm512_storeu_ps(&C[0 * ldc], c0_lo);      _mm512_storeu_ps(&C[0 * ldc + 16], c0_hi);
    _mm512_storeu_ps(&C[1 * ldc], c1_lo);      _mm512_storeu_ps(&C[1 * ldc + 16], c1_hi);
    _mm512_storeu_ps(&C[2 * ldc], c2_lo);      _mm512_storeu_ps(&C[2 * ldc + 16], c2_hi);
    _mm512_storeu_ps(&C[3 * ldc], c3_lo);      _mm512_storeu_ps(&C[3 * ldc + 16], c3_hi);
    _mm512_storeu_ps(&C[4 * ldc], c4_lo);      _mm512_storeu_ps(&C[4 * ldc + 16], c4_hi);
    _mm512_storeu_ps(&C[5 * ldc], c5_lo);      _mm512_storeu_ps(&C[5 * ldc + 16], c5_hi);
}

// Packed version for large matrices
static inline void gemm_microkernel_6x32_packed_avx512(
    int K,
    const float * __restrict__ Ap,  // Packed A: [MR, K] contiguous
    const float * __restrict__ Bp,  // Packed B: [K, NR] contiguous
    float * __restrict__ C, int ldc,
    int first_k
)
{
    __m512 c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi;
    __m512 c3_lo, c3_hi, c4_lo, c4_hi, c5_lo, c5_hi;

    if (first_k) {
        c0_lo = _mm512_setzero_ps(); c0_hi = _mm512_setzero_ps();
        c1_lo = _mm512_setzero_ps(); c1_hi = _mm512_setzero_ps();
        c2_lo = _mm512_setzero_ps(); c2_hi = _mm512_setzero_ps();
        c3_lo = _mm512_setzero_ps(); c3_hi = _mm512_setzero_ps();
        c4_lo = _mm512_setzero_ps(); c4_hi = _mm512_setzero_ps();
        c5_lo = _mm512_setzero_ps(); c5_hi = _mm512_setzero_ps();
    } else {
        c0_lo = _mm512_loadu_ps(&C[0 * ldc]);      c0_hi = _mm512_loadu_ps(&C[0 * ldc + 16]);
        c1_lo = _mm512_loadu_ps(&C[1 * ldc]);      c1_hi = _mm512_loadu_ps(&C[1 * ldc + 16]);
        c2_lo = _mm512_loadu_ps(&C[2 * ldc]);      c2_hi = _mm512_loadu_ps(&C[2 * ldc + 16]);
        c3_lo = _mm512_loadu_ps(&C[3 * ldc]);      c3_hi = _mm512_loadu_ps(&C[3 * ldc + 16]);
        c4_lo = _mm512_loadu_ps(&C[4 * ldc]);      c4_hi = _mm512_loadu_ps(&C[4 * ldc + 16]);
        c5_lo = _mm512_loadu_ps(&C[5 * ldc]);      c5_hi = _mm512_loadu_ps(&C[5 * ldc + 16]);
    }

    // Packed B is contiguous: B[k, 0:32] at Bp[k * 32]
    _mm_prefetch((const char*)Bp, _MM_HINT_T0);
    _mm_prefetch((const char*)(Bp + 16), _MM_HINT_T0);

    int k = 0;
    for (; k <= K - 4; k += 4) {
        _mm_prefetch((const char*)(Bp + (k + 8) * NR), _MM_HINT_T0);
        _mm_prefetch((const char*)(Bp + (k + 8) * NR + 16), _MM_HINT_T0);

        #define PACKED_ITER(koff) { \
            __m512 b_lo = _mm512_load_ps(&Bp[(k + koff) * NR]); \
            __m512 b_hi = _mm512_load_ps(&Bp[(k + koff) * NR + 16]); \
            __m512 a0 = _mm512_set1_ps(Ap[0 * K + k + koff]); \
            __m512 a1 = _mm512_set1_ps(Ap[1 * K + k + koff]); \
            __m512 a2 = _mm512_set1_ps(Ap[2 * K + k + koff]); \
            __m512 a3 = _mm512_set1_ps(Ap[3 * K + k + koff]); \
            __m512 a4 = _mm512_set1_ps(Ap[4 * K + k + koff]); \
            __m512 a5 = _mm512_set1_ps(Ap[5 * K + k + koff]); \
            c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi); \
            c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi); \
            c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi); \
            c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi); \
            c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi); \
            c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi); \
        }

        PACKED_ITER(0);
        PACKED_ITER(1);
        PACKED_ITER(2);
        PACKED_ITER(3);

        #undef PACKED_ITER
    }

    for (; k < K; k++) {
        __m512 b_lo = _mm512_load_ps(&Bp[k * NR]);
        __m512 b_hi = _mm512_load_ps(&Bp[k * NR + 16]);

        c0_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[0 * K + k]), b_lo, c0_lo);
        c0_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[0 * K + k]), b_hi, c0_hi);
        c1_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[1 * K + k]), b_lo, c1_lo);
        c1_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[1 * K + k]), b_hi, c1_hi);
        c2_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[2 * K + k]), b_lo, c2_lo);
        c2_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[2 * K + k]), b_hi, c2_hi);
        c3_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[3 * K + k]), b_lo, c3_lo);
        c3_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[3 * K + k]), b_hi, c3_hi);
        c4_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[4 * K + k]), b_lo, c4_lo);
        c4_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[4 * K + k]), b_hi, c4_hi);
        c5_lo = _mm512_fmadd_ps(_mm512_set1_ps(Ap[5 * K + k]), b_lo, c5_lo);
        c5_hi = _mm512_fmadd_ps(_mm512_set1_ps(Ap[5 * K + k]), b_hi, c5_hi);
    }

    _mm512_storeu_ps(&C[0 * ldc], c0_lo);      _mm512_storeu_ps(&C[0 * ldc + 16], c0_hi);
    _mm512_storeu_ps(&C[1 * ldc], c1_lo);      _mm512_storeu_ps(&C[1 * ldc + 16], c1_hi);
    _mm512_storeu_ps(&C[2 * ldc], c2_lo);      _mm512_storeu_ps(&C[2 * ldc + 16], c2_hi);
    _mm512_storeu_ps(&C[3 * ldc], c3_lo);      _mm512_storeu_ps(&C[3 * ldc + 16], c3_hi);
    _mm512_storeu_ps(&C[4 * ldc], c4_lo);      _mm512_storeu_ps(&C[4 * ldc + 16], c4_hi);
    _mm512_storeu_ps(&C[5 * ldc], c5_lo);      _mm512_storeu_ps(&C[5 * ldc + 16], c5_hi);
}
#endif // __AVX512F__

// =============================================================================
// AVX/AVX2 6x16 Microkernel with FMA
//
// KEY FIX: Use _mm256_fmadd_ps instead of separate mul+add
// FMA fuses multiply-add into single instruction: c = a*b + c
// This gives ~2x throughput improvement on FMA-capable CPUs
// =============================================================================

#if defined(__AVX__)
static inline void gemm_microkernel_6x16_avx(
    int K,
    const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,
    float * __restrict__ C, int ldc,
    int first_k
)
{
    // 12 accumulators: 6 rows x 2 YMM (16 floats) per row
    __m256 c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi;
    __m256 c3_lo, c3_hi, c4_lo, c4_hi, c5_lo, c5_hi;

    if (first_k) {
        c0_lo = _mm256_setzero_ps(); c0_hi = _mm256_setzero_ps();
        c1_lo = _mm256_setzero_ps(); c1_hi = _mm256_setzero_ps();
        c2_lo = _mm256_setzero_ps(); c2_hi = _mm256_setzero_ps();
        c3_lo = _mm256_setzero_ps(); c3_hi = _mm256_setzero_ps();
        c4_lo = _mm256_setzero_ps(); c4_hi = _mm256_setzero_ps();
        c5_lo = _mm256_setzero_ps(); c5_hi = _mm256_setzero_ps();
    } else {
        c0_lo = _mm256_loadu_ps(&C[0 * ldc]);      c0_hi = _mm256_loadu_ps(&C[0 * ldc + 8]);
        c1_lo = _mm256_loadu_ps(&C[1 * ldc]);      c1_hi = _mm256_loadu_ps(&C[1 * ldc + 8]);
        c2_lo = _mm256_loadu_ps(&C[2 * ldc]);      c2_hi = _mm256_loadu_ps(&C[2 * ldc + 8]);
        c3_lo = _mm256_loadu_ps(&C[3 * ldc]);      c3_hi = _mm256_loadu_ps(&C[3 * ldc + 8]);
        c4_lo = _mm256_loadu_ps(&C[4 * ldc]);      c4_hi = _mm256_loadu_ps(&C[4 * ldc + 8]);
        c5_lo = _mm256_loadu_ps(&C[5 * ldc]);      c5_hi = _mm256_loadu_ps(&C[5 * ldc + 8]);
    }

    // Prefetch first cache lines of B
    _mm_prefetch((const char*)B, _MM_HINT_T0);
    _mm_prefetch((const char*)(B + 32), _MM_HINT_T0);

    // Main K loop - unrolled by 8 for better ILP and prefetch hiding
    int k = 0;

#if defined(__FMA__)
    // FMA path - uses fused multiply-add (single instruction)
    for (; k <= K - 8; k += 8) {
        // Prefetch ahead - 16 rows ahead for L1
        _mm_prefetch((const char*)&B[(k + 16) * ldb], _MM_HINT_T0);
        _mm_prefetch((const char*)&B[(k + 16) * ldb + 32], _MM_HINT_T0);
        _mm_prefetch((const char*)&B[(k + 17) * ldb], _MM_HINT_T0);

        // Software pipelining: load B for iteration 0 before the loop body
        __m256 b_lo_next = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b_hi_next = _mm256_loadu_ps(&B[k * ldb + 8]);

        #define FMA_ITER(koff) { \
            __m256 b_lo = b_lo_next; \
            __m256 b_hi = b_hi_next; \
            if ((koff) < 7) { \
                b_lo_next = _mm256_loadu_ps(&B[(k + (koff) + 1) * ldb]); \
                b_hi_next = _mm256_loadu_ps(&B[(k + (koff) + 1) * ldb + 8]); \
            } \
            __m256 a0 = _mm256_set1_ps(A[0 * lda + k + (koff)]); \
            __m256 a1 = _mm256_set1_ps(A[1 * lda + k + (koff)]); \
            __m256 a2 = _mm256_set1_ps(A[2 * lda + k + (koff)]); \
            __m256 a3 = _mm256_set1_ps(A[3 * lda + k + (koff)]); \
            __m256 a4 = _mm256_set1_ps(A[4 * lda + k + (koff)]); \
            __m256 a5 = _mm256_set1_ps(A[5 * lda + k + (koff)]); \
            c0_lo = _mm256_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm256_fmadd_ps(a0, b_hi, c0_hi); \
            c1_lo = _mm256_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm256_fmadd_ps(a1, b_hi, c1_hi); \
            c2_lo = _mm256_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm256_fmadd_ps(a2, b_hi, c2_hi); \
            c3_lo = _mm256_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm256_fmadd_ps(a3, b_hi, c3_hi); \
            c4_lo = _mm256_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm256_fmadd_ps(a4, b_hi, c4_hi); \
            c5_lo = _mm256_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm256_fmadd_ps(a5, b_hi, c5_hi); \
        }

        FMA_ITER(0);
        FMA_ITER(1);
        FMA_ITER(2);
        FMA_ITER(3);
        FMA_ITER(4);
        FMA_ITER(5);
        FMA_ITER(6);
        FMA_ITER(7);

        #undef FMA_ITER
    }

    // Handle remaining K with FMA
    for (; k < K; k++) {
        __m256 b_lo = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b_hi = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        __m256 a4 = _mm256_set1_ps(A[4 * lda + k]);
        __m256 a5 = _mm256_set1_ps(A[5 * lda + k]);

        c0_lo = _mm256_fmadd_ps(a0, b_lo, c0_lo); c0_hi = _mm256_fmadd_ps(a0, b_hi, c0_hi);
        c1_lo = _mm256_fmadd_ps(a1, b_lo, c1_lo); c1_hi = _mm256_fmadd_ps(a1, b_hi, c1_hi);
        c2_lo = _mm256_fmadd_ps(a2, b_lo, c2_lo); c2_hi = _mm256_fmadd_ps(a2, b_hi, c2_hi);
        c3_lo = _mm256_fmadd_ps(a3, b_lo, c3_lo); c3_hi = _mm256_fmadd_ps(a3, b_hi, c3_hi);
        c4_lo = _mm256_fmadd_ps(a4, b_lo, c4_lo); c4_hi = _mm256_fmadd_ps(a4, b_hi, c4_hi);
        c5_lo = _mm256_fmadd_ps(a5, b_lo, c5_lo); c5_hi = _mm256_fmadd_ps(a5, b_hi, c5_hi);
    }
#else
    // Non-FMA fallback (older CPUs without FMA)
    for (; k <= K - 4; k += 4) {
        _mm_prefetch((const char*)&B[(k + 8) * ldb], _MM_HINT_T0);

        #define AVX_ITER(koff) { \
            __m256 b_lo = _mm256_loadu_ps(&B[(k + koff) * ldb]); \
            __m256 b_hi = _mm256_loadu_ps(&B[(k + koff) * ldb + 8]); \
            __m256 a0 = _mm256_set1_ps(A[0 * lda + k + koff]); \
            __m256 a1 = _mm256_set1_ps(A[1 * lda + k + koff]); \
            __m256 a2 = _mm256_set1_ps(A[2 * lda + k + koff]); \
            __m256 a3 = _mm256_set1_ps(A[3 * lda + k + koff]); \
            __m256 a4 = _mm256_set1_ps(A[4 * lda + k + koff]); \
            __m256 a5 = _mm256_set1_ps(A[5 * lda + k + koff]); \
            c0_lo = _mm256_add_ps(c0_lo, _mm256_mul_ps(a0, b_lo)); \
            c0_hi = _mm256_add_ps(c0_hi, _mm256_mul_ps(a0, b_hi)); \
            c1_lo = _mm256_add_ps(c1_lo, _mm256_mul_ps(a1, b_lo)); \
            c1_hi = _mm256_add_ps(c1_hi, _mm256_mul_ps(a1, b_hi)); \
            c2_lo = _mm256_add_ps(c2_lo, _mm256_mul_ps(a2, b_lo)); \
            c2_hi = _mm256_add_ps(c2_hi, _mm256_mul_ps(a2, b_hi)); \
            c3_lo = _mm256_add_ps(c3_lo, _mm256_mul_ps(a3, b_lo)); \
            c3_hi = _mm256_add_ps(c3_hi, _mm256_mul_ps(a3, b_hi)); \
            c4_lo = _mm256_add_ps(c4_lo, _mm256_mul_ps(a4, b_lo)); \
            c4_hi = _mm256_add_ps(c4_hi, _mm256_mul_ps(a4, b_hi)); \
            c5_lo = _mm256_add_ps(c5_lo, _mm256_mul_ps(a5, b_lo)); \
            c5_hi = _mm256_add_ps(c5_hi, _mm256_mul_ps(a5, b_hi)); \
        }

        AVX_ITER(0);
        AVX_ITER(1);
        AVX_ITER(2);
        AVX_ITER(3);

        #undef AVX_ITER
    }

    for (; k < K; k++) {
        __m256 b_lo = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b_hi = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        __m256 a4 = _mm256_set1_ps(A[4 * lda + k]);
        __m256 a5 = _mm256_set1_ps(A[5 * lda + k]);

        c0_lo = _mm256_add_ps(c0_lo, _mm256_mul_ps(a0, b_lo));
        c0_hi = _mm256_add_ps(c0_hi, _mm256_mul_ps(a0, b_hi));
        c1_lo = _mm256_add_ps(c1_lo, _mm256_mul_ps(a1, b_lo));
        c1_hi = _mm256_add_ps(c1_hi, _mm256_mul_ps(a1, b_hi));
        c2_lo = _mm256_add_ps(c2_lo, _mm256_mul_ps(a2, b_lo));
        c2_hi = _mm256_add_ps(c2_hi, _mm256_mul_ps(a2, b_hi));
        c3_lo = _mm256_add_ps(c3_lo, _mm256_mul_ps(a3, b_lo));
        c3_hi = _mm256_add_ps(c3_hi, _mm256_mul_ps(a3, b_hi));
        c4_lo = _mm256_add_ps(c4_lo, _mm256_mul_ps(a4, b_lo));
        c4_hi = _mm256_add_ps(c4_hi, _mm256_mul_ps(a4, b_hi));
        c5_lo = _mm256_add_ps(c5_lo, _mm256_mul_ps(a5, b_lo));
        c5_hi = _mm256_add_ps(c5_hi, _mm256_mul_ps(a5, b_hi));
    }
#endif

    _mm256_storeu_ps(&C[0 * ldc], c0_lo);      _mm256_storeu_ps(&C[0 * ldc + 8], c0_hi);
    _mm256_storeu_ps(&C[1 * ldc], c1_lo);      _mm256_storeu_ps(&C[1 * ldc + 8], c1_hi);
    _mm256_storeu_ps(&C[2 * ldc], c2_lo);      _mm256_storeu_ps(&C[2 * ldc + 8], c2_hi);
    _mm256_storeu_ps(&C[3 * ldc], c3_lo);      _mm256_storeu_ps(&C[3 * ldc + 8], c3_hi);
    _mm256_storeu_ps(&C[4 * ldc], c4_lo);      _mm256_storeu_ps(&C[4 * ldc + 8], c4_hi);
    _mm256_storeu_ps(&C[5 * ldc], c5_lo);      _mm256_storeu_ps(&C[5 * ldc + 8], c5_hi);
}
#endif

// =============================================================================
// AVX 4x16 Microkernel (for AVX-only CPUs without FMA)
//
// This smaller tile avoids register spilling on CPUs with only 16 YMM registers.
// Register allocation: 8 accumulators + 2 B + 4 A + 2 temp = 16 registers
// =============================================================================

#if defined(__AVX__) && !defined(__FMA__)
static inline void gemm_microkernel_4x16_avx(
    int K,
    const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,
    float * __restrict__ C, int ldc,
    int first_k
)
{
    // 8 accumulators: 4 rows x 2 YMM (16 floats) per row
    __m256 c0_lo, c0_hi, c1_lo, c1_hi, c2_lo, c2_hi, c3_lo, c3_hi;

    if (first_k) {
        c0_lo = _mm256_setzero_ps(); c0_hi = _mm256_setzero_ps();
        c1_lo = _mm256_setzero_ps(); c1_hi = _mm256_setzero_ps();
        c2_lo = _mm256_setzero_ps(); c2_hi = _mm256_setzero_ps();
        c3_lo = _mm256_setzero_ps(); c3_hi = _mm256_setzero_ps();
    } else {
        c0_lo = _mm256_loadu_ps(&C[0 * ldc]);      c0_hi = _mm256_loadu_ps(&C[0 * ldc + 8]);
        c1_lo = _mm256_loadu_ps(&C[1 * ldc]);      c1_hi = _mm256_loadu_ps(&C[1 * ldc + 8]);
        c2_lo = _mm256_loadu_ps(&C[2 * ldc]);      c2_hi = _mm256_loadu_ps(&C[2 * ldc + 8]);
        c3_lo = _mm256_loadu_ps(&C[3 * ldc]);      c3_hi = _mm256_loadu_ps(&C[3 * ldc + 8]);
    }

    _mm_prefetch((const char*)B, _MM_HINT_T0);

    // K loop - unrolled by 4 for better ILP
    int k = 0;
    for (; k <= K - 4; k += 4) {
        _mm_prefetch((const char*)&B[(k + 8) * ldb], _MM_HINT_T0);

        #define AVX4_ITER(koff) { \
            __m256 b_lo = _mm256_loadu_ps(&B[(k + koff) * ldb]); \
            __m256 b_hi = _mm256_loadu_ps(&B[(k + koff) * ldb + 8]); \
            __m256 a0 = _mm256_set1_ps(A[0 * lda + k + koff]); \
            __m256 a1 = _mm256_set1_ps(A[1 * lda + k + koff]); \
            __m256 a2 = _mm256_set1_ps(A[2 * lda + k + koff]); \
            __m256 a3 = _mm256_set1_ps(A[3 * lda + k + koff]); \
            c0_lo = _mm256_add_ps(c0_lo, _mm256_mul_ps(a0, b_lo)); \
            c0_hi = _mm256_add_ps(c0_hi, _mm256_mul_ps(a0, b_hi)); \
            c1_lo = _mm256_add_ps(c1_lo, _mm256_mul_ps(a1, b_lo)); \
            c1_hi = _mm256_add_ps(c1_hi, _mm256_mul_ps(a1, b_hi)); \
            c2_lo = _mm256_add_ps(c2_lo, _mm256_mul_ps(a2, b_lo)); \
            c2_hi = _mm256_add_ps(c2_hi, _mm256_mul_ps(a2, b_hi)); \
            c3_lo = _mm256_add_ps(c3_lo, _mm256_mul_ps(a3, b_lo)); \
            c3_hi = _mm256_add_ps(c3_hi, _mm256_mul_ps(a3, b_hi)); \
        }

        AVX4_ITER(0);
        AVX4_ITER(1);
        AVX4_ITER(2);
        AVX4_ITER(3);

        #undef AVX4_ITER
    }

    // Handle remaining K
    for (; k < K; k++) {
        __m256 b_lo = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b_hi = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);

        c0_lo = _mm256_add_ps(c0_lo, _mm256_mul_ps(a0, b_lo));
        c0_hi = _mm256_add_ps(c0_hi, _mm256_mul_ps(a0, b_hi));
        c1_lo = _mm256_add_ps(c1_lo, _mm256_mul_ps(a1, b_lo));
        c1_hi = _mm256_add_ps(c1_hi, _mm256_mul_ps(a1, b_hi));
        c2_lo = _mm256_add_ps(c2_lo, _mm256_mul_ps(a2, b_lo));
        c2_hi = _mm256_add_ps(c2_hi, _mm256_mul_ps(a2, b_hi));
        c3_lo = _mm256_add_ps(c3_lo, _mm256_mul_ps(a3, b_lo));
        c3_hi = _mm256_add_ps(c3_hi, _mm256_mul_ps(a3, b_hi));
    }

    _mm256_storeu_ps(&C[0 * ldc], c0_lo);      _mm256_storeu_ps(&C[0 * ldc + 8], c0_hi);
    _mm256_storeu_ps(&C[1 * ldc], c1_lo);      _mm256_storeu_ps(&C[1 * ldc + 8], c1_hi);
    _mm256_storeu_ps(&C[2 * ldc], c2_lo);      _mm256_storeu_ps(&C[2 * ldc + 8], c2_hi);
    _mm256_storeu_ps(&C[3 * ldc], c3_lo);      _mm256_storeu_ps(&C[3 * ldc + 8], c3_hi);
}
#endif

// =============================================================================
// Edge case handler for non-MRxNR aligned tiles
// =============================================================================

static void gemm_microkernel_edge(
    int m, int n, int K,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int first_k
)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = first_k ? 0.0f : C[i * ldc + j];
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// =============================================================================
// Matrix Packing Functions - Parallel for large matrices
// =============================================================================

// Pack A panel: A[m0:m0+mc, k0:k0+kc] -> Ap[mc, kc] in row-panel format
static void pack_a_panel(
    const float *A, int lda,
    float *Ap,
    int mc, int kc, int mr
)
{
    #pragma omp parallel for schedule(static) if(mc > 64)
    for (int i = 0; i < mc; i += mr) {
        int rows = (i + mr <= mc) ? mr : (mc - i);
        float *Ap_panel = &Ap[(i / mr) * mr * kc];

        for (int p = 0; p < rows; p++) {
            const float *A_row = &A[(i + p) * lda];
            float *Ap_row = &Ap_panel[p * kc];

            // Vectorized copy
            int k = 0;
#if defined(__AVX__)
            for (; k <= kc - 8; k += 8) {
                _mm256_storeu_ps(&Ap_row[k], _mm256_loadu_ps(&A_row[k]));
            }
#endif
            for (; k < kc; k++) {
                Ap_row[k] = A_row[k];
            }
        }
        // Zero pad if partial panel
        for (int p = rows; p < mr; p++) {
            memset(&Ap_panel[p * kc], 0, kc * sizeof(float));
        }
    }
}

// Pack B panel: B[k0:k0+kc, n0:n0+nc] -> Bp[kc, nc] in column-panel format
static void pack_b_panel(
    const float *B, int ldb,
    float *Bp,
    int kc, int nc, int nr
)
{
    #pragma omp parallel for schedule(static) if(nc > 128)
    for (int j = 0; j < nc; j += nr) {
        int cols = (j + nr <= nc) ? nr : (nc - j);
        float *Bp_panel = &Bp[(j / nr) * nr * kc];

        for (int k = 0; k < kc; k++) {
            const float *B_row = &B[k * ldb + j];
            float *Bp_row = &Bp_panel[k * nr];

            // Copy cols and zero-pad
            int c = 0;
#if defined(__AVX512F__)
            for (; c <= cols - 16; c += 16) {
                _mm512_store_ps(&Bp_row[c], _mm512_loadu_ps(&B_row[c]));
            }
#elif defined(__AVX__)
            for (; c <= cols - 8; c += 8) {
                _mm256_store_ps(&Bp_row[c], _mm256_loadu_ps(&B_row[c]));
            }
#endif
            for (; c < cols; c++) {
                Bp_row[c] = B_row[c];
            }
            for (; c < nr; c++) {
                Bp_row[c] = 0.0f;
            }
        }
    }
}

// =============================================================================
// High-Performance GEMM with 2D Threading and Packing
//
// This is the main entry point for large matrices. Uses:
// 1. 2D thread partitioning (across M and N blocks)
// 2. Parallel matrix packing
// 3. Optimized microkernels
// =============================================================================

void gemm_microkernel_packed(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
    // Use tile-parallel blocked version - scales better on many-core systems
    gemm_microkernel_blocked(A, B, C, M, N, K);
}

// =============================================================================
// Cache-Blocked GEMM with 2D Threading
//
// KEY FIX: Use 2D parallelization across both M and N tile dimensions.
// For 48-core Xeon, we need at least 48 parallel tasks. With 1024x1024:
// - M_tiles = ceil(1024 / MR) = ~170 tiles
// - N_tiles = ceil(1024 / NR) = ~32 tiles (for NR=32)
// - Total = 5440 tiles - excellent parallelism!
// =============================================================================

// Sequential version for small matrices (avoids OpenMP overhead)
static void gemm_microkernel_sequential(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
    // Zero output
    for (int i = 0; i < M; i++) {
        memset(&C[i * N], 0, N * sizeof(float));
    }

    const int mr = MR;
    const int nr = NR;

    // Block over K
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kb = (k0 + KC <= K) ? KC : (K - k0);
        int first_k = (k0 == 0);

        // Loop over tiles
        for (int m0 = 0; m0 < M; m0 += mr) {
            int mr_actual = (m0 + mr <= M) ? mr : (M - m0);

            for (int n0 = 0; n0 < N; n0 += nr) {
                int nr_actual = (n0 + nr <= N) ? nr : (N - n0);

                const float *A_tile = &A[m0 * K + k0];
                const float *B_tile = &B[k0 * N + n0];
                float *C_tile = &C[m0 * N + n0];

                if (mr_actual == mr && nr_actual == nr) {
#if defined(__AVX512F__)
                    gemm_microkernel_6x32_avx512(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#elif defined(__FMA__)
                    gemm_microkernel_6x16_avx(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#elif defined(__AVX__)
                    gemm_microkernel_4x16_avx(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#else
                    gemm_microkernel_edge(mr_actual, nr_actual, kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#endif
                } else {
                    gemm_microkernel_edge(mr_actual, nr_actual, kb, A_tile, K, B_tile, N, C_tile, N, first_k);
                }
            }
        }
    }
}

// Flag to track if we've set optimal thread count (only used in native backend)
static int g_threads_initialized = 0;

// Set optimal thread count for GEMM (physical cores only, no hyperthreading)
static void gemm_init_threads(void) {
    if (g_threads_initialized) return;

#ifdef _OPENMP
    const CPUInfo* cpu = get_cpu_info();
    int physical_cores = cpu->num_cores;

    // Only use physical cores - hyperthreading hurts compute-bound GEMM
    if (physical_cores > 0) {
        int current_max = omp_get_max_threads();
        // Only reduce if we have more threads than physical cores
        if (current_max > physical_cores) {
            omp_set_num_threads(physical_cores);
        }
    }
#endif
    g_threads_initialized = 1;
}

void gemm_microkernel_blocked(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
    const int mr = MR;
    const int nr = NR;

    // Use sequential version for small matrices to avoid OpenMP overhead
    // Threshold tuned for typical 4-8 core systems
    if ((size_t)M * N * K <= 512ULL * 512 * 512) {
        gemm_microkernel_sequential(A, B, C, M, N, K);
        return;
    }

    // Initialize thread count to physical cores (once)
    gemm_init_threads();

    // Zero output first
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        memset(&C[i * N], 0, N * sizeof(float));
    }

    // Block over K (outermost - for accumulation across all threads)
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kb = (k0 + KC <= K) ? KC : (K - k0);
        int first_k = (k0 == 0);

        // Parallelize over M rows - each thread gets a chunk of M
        // This gives better cache locality than tile-level parallelism
        #pragma omp parallel for schedule(static)
        for (int m0 = 0; m0 < M; m0 += mr) {
            int mr_actual = (m0 + mr <= M) ? mr : (M - m0);

            // Each thread processes all N tiles for its M rows
            for (int n0 = 0; n0 < N; n0 += nr) {
                int nr_actual = (n0 + nr <= N) ? nr : (N - n0);

                const float *A_tile = &A[m0 * K + k0];
                const float *B_tile = &B[k0 * N + n0];
                float *C_tile = &C[m0 * N + n0];

                if (mr_actual == mr && nr_actual == nr) {
#if defined(__AVX512F__)
                    gemm_microkernel_6x32_avx512(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#elif defined(__FMA__)
                    gemm_microkernel_6x16_avx(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#elif defined(__AVX__)
                    gemm_microkernel_4x16_avx(kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#else
                    gemm_microkernel_edge(mr_actual, nr_actual, kb, A_tile, K, B_tile, N, C_tile, N, first_k);
#endif
                } else {
                    gemm_microkernel_edge(mr_actual, nr_actual, kb, A_tile, K, B_tile, N, C_tile, N, first_k);
                }
            }
        }
    }
}

// =============================================================================
// B-transposed GEMM: C[M,N] = A[M,K] @ B[N,K].T
// =============================================================================

#if defined(__AVX512F__)
static inline void gemm_microkernel_6x32_bt_avx512(
    int K,
    const float * __restrict__ A, int lda,
    const float * __restrict__ B, int ldb,  // B is [N, K] transposed
    float * __restrict__ C, int ldc,
    int first_k
)
{
    // For B transposed, we need different access pattern
    // C[i,j] = sum_k A[i,k] * B[j,k]

    if (first_k) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 32; j++) {
                C[i * ldc + j] = 0.0f;
            }
        }
    }

    // Process K in chunks of 16 for SIMD
    int k = 0;
    for (; k <= K - 16; k += 16) {
        // Load A[0:6, k:k+16] - 6 rows
        __m512 a0 = _mm512_loadu_ps(&A[0 * lda + k]);
        __m512 a1 = _mm512_loadu_ps(&A[1 * lda + k]);
        __m512 a2 = _mm512_loadu_ps(&A[2 * lda + k]);
        __m512 a3 = _mm512_loadu_ps(&A[3 * lda + k]);
        __m512 a4 = _mm512_loadu_ps(&A[4 * lda + k]);
        __m512 a5 = _mm512_loadu_ps(&A[5 * lda + k]);

        // For each column j of C (row j of B)
        for (int j = 0; j < 32; j++) {
            __m512 b = _mm512_loadu_ps(&B[j * ldb + k]);

            // Compute dot products using reduction
            C[0 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a0, b));
            C[1 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a1, b));
            C[2 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a2, b));
            C[3 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a3, b));
            C[4 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a4, b));
            C[5 * ldc + j] += _mm512_reduce_add_ps(_mm512_mul_ps(a5, b));
        }
    }

    // Handle remaining K
    for (; k < K; k++) {
        for (int i = 0; i < 6; i++) {
            float a = A[i * lda + k];
            for (int j = 0; j < 32; j++) {
                C[i * ldc + j] += a * B[j * ldb + k];
            }
        }
    }
}
#endif

void gemm_microkernel_blocked_bt(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
)
{
    // Zero output first
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        memset(&C[i * N], 0, N * sizeof(float));
    }

    const int mr = MR;
    const int nr = NR;

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int m0 = 0; m0 < M; m0 += MC) {
        for (int n0 = 0; n0 < N; n0 += NC) {
            int mb = (m0 + MC <= M) ? MC : (M - m0);
            int nb = (n0 + NC <= N) ? NC : (N - n0);

            for (int k0 = 0; k0 < K; k0 += KC) {
                int kb = (k0 + KC <= K) ? KC : (K - k0);
                int first_k = (k0 == 0);

                for (int m1 = 0; m1 < mb; m1 += mr) {
                    int mr_actual = (m1 + mr <= mb) ? mr : (mb - m1);

                    for (int n1 = 0; n1 < nb; n1 += nr) {
                        int nr_actual = (n1 + nr <= nb) ? nr : (nb - n1);

                        const float *A_tile = &A[(m0 + m1) * K + k0];
                        const float *B_tile = &B[(n0 + n1) * K + k0];
                        float *C_tile = &C[(m0 + m1) * N + (n0 + n1)];

                        if (mr_actual == mr && nr_actual == nr) {
#if defined(__AVX512F__)
                            gemm_microkernel_6x32_bt_avx512(kb, A_tile, K, B_tile, K, C_tile, N, first_k);
#else
                            // Scalar fallback for B-transposed
                            for (int i = 0; i < mr; i++) {
                                for (int j = 0; j < nr; j++) {
                                    float sum = first_k ? 0.0f : C_tile[i * N + j];
                                    for (int kk = 0; kk < kb; kk++) {
                                        sum += A_tile[i * K + kk] * B_tile[j * K + kk];
                                    }
                                    C_tile[i * N + j] = sum;
                                }
                            }
#endif
                        } else {
                            // Edge case
                            for (int i = 0; i < mr_actual; i++) {
                                for (int j = 0; j < nr_actual; j++) {
                                    float sum = first_k ? 0.0f : C_tile[i * N + j];
                                    for (int kk = 0; kk < kb; kk++) {
                                        sum += A_tile[i * K + kk] * B_tile[j * K + kk];
                                    }
                                    C_tile[i * N + j] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

#define PACK_THRESHOLD 256  // Use packing for matrices >= 256

void gemm_microkernel(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K,
    int B_transposed
)
{
    if (B_transposed) {
        gemm_microkernel_blocked_bt(A, B, C, M, N, K);
    } else {
        // Use packed version for large matrices
        if (M >= PACK_THRESHOLD && N >= PACK_THRESHOLD && K >= PACK_THRESHOLD) {
            gemm_microkernel_packed(A, B, C, M, N, K);
        } else {
            gemm_microkernel_blocked(A, B, C, M, N, K);
        }
    }
}

#endif // !USE_MKL && !USE_ONEDNN (Native backend)

// =============================================================================
// Query which backend is in use
// =============================================================================

const char* gemm_get_backend(void) {
    return GEMM_BACKEND;
}
