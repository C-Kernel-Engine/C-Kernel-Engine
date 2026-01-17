/**
 * @file gemm_fused_kernels.c
 * @brief Fused GEMM Kernels with activations
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
 * GEMM operations fused with activations (ReLU, GELU, SiLU) and SwiGLU.
 * The key benefit: intermediate results stay in registers, avoiding DRAM
 * round-trips between operations.
 *
 * Supported operations:
 * - gemm_bias_relu_fused:  C = ReLU(A @ B^T + bias)
 * - gemm_bias_gelu_fused:  C = GELU(A @ B^T + bias)
 * - gemm_bias_silu_fused:  C = SiLU(A @ B^T + bias)
 * - gemm_swiglu_fused:     C = SiLU(x @ W_gate) * (x @ W_up)
 *
 * All kernels support:
 * - AVX1 SIMD (256-bit vectors, no FMA)
 * - OpenMP parallelization
 * - Scalar fallback
 */

#include "ckernel_engine.h"
#include <math.h>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// SIMD Helpers
// =============================================================================

#if defined(__AVX__) || defined(__AVX512F__)
// Horizontal sum of 8 floats in __m256 (works for both AVX and AVX-512)
static inline float hsum256_ps_fused(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

#if defined(__AVX512F__)
// Horizontal sum of 16 floats in __m512
static inline float hsum512_ps_fused(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(lo, hi);
    return hsum256_ps_fused(sum256);
}
#endif

// =============================================================================
// Fast activation approximations (scalar)
// =============================================================================

// GELU approximation: x * sigmoid(1.702 * x) (QuickGELU)
static inline float fast_gelu_scalar(float x) {
    float sx = 1.702f * x;
    float sig = 1.0f / (1.0f + expf(-sx));
    return x * sig;
}

// =============================================================================
// GEMM + Bias + ReLU fused
// C[i,j] = max(0, sum_k(A[i,k] * B[j,k]) + bias[j])
// =============================================================================
void gemm_bias_relu_fused(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K)
{
#if defined(__AVX__)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k;
            for (k = 0; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }
            float sum = hsum256_ps_fused(sum_vec);
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            // Fused: add bias and ReLU while still in register
            sum += bias[j];
            C[i * N + j] = sum > 0.0f ? sum : 0.0f;
        }
    }
#else
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            sum += bias[j];
            C[i * N + j] = sum > 0.0f ? sum : 0.0f;
        }
    }
#endif
}

// =============================================================================
// GEMM + Bias + GELU fused
// C[i,j] = GELU(sum_k(A[i,k] * B[j,k]) + bias[j])
// Uses QuickGELU approximation: x * sigmoid(1.702 * x)
// =============================================================================
void gemm_bias_gelu_fused(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K)
{
#if defined(__AVX__)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k;
            for (k = 0; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }
            float sum = hsum256_ps_fused(sum_vec);
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            sum += bias[j];
            C[i * N + j] = fast_gelu_scalar(sum);
        }
    }
#else
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            sum += bias[j];
            C[i * N + j] = fast_gelu_scalar(sum);
        }
    }
#endif
}

// =============================================================================
// GEMM + Bias + SiLU/Swish fused
// C[i,j] = SiLU(sum_k(A[i,k] * B[j,k]) + bias[j])
// SiLU(x) = x * sigmoid(x)
// =============================================================================
void gemm_bias_silu_fused(const float *A,
                          const float *B,
                          const float *bias,
                          float *C,
                          int M, int N, int K)
{
#if defined(__AVX__)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k;
            for (k = 0; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }
            float sum = hsum256_ps_fused(sum_vec);
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            sum += bias[j];
            // SiLU: x * sigmoid(x)
            float sig = 1.0f / (1.0f + expf(-sum));
            C[i * N + j] = sum * sig;
        }
    }
#else
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            sum += bias[j];
            float sig = 1.0f / (1.0f + expf(-sum));
            C[i * N + j] = sum * sig;
        }
    }
#endif
}

// =============================================================================
// GEMM + SwiGLU Fused (LLaMA/SmolLM style MLP)
//
// Computes: output = SiLU(x @ W_gate + b_gate) * (x @ W_up + b_up)
//
// This fuses TWO GEMMs + SwiGLU activation into one pass:
// - gate = x @ W_gate + b_gate   (GEMM 1)
// - up   = x @ W_up + b_up       (GEMM 2)
// - out  = SiLU(gate) * up       (SwiGLU)
//
// Layout:
//   x:       [M, K]      input activations
//   W_gate:  [N, K]      gate projection weights (transposed)
//   W_up:    [N, K]      up projection weights (transposed)
//   b_gate:  [N]         gate bias (can be NULL)
//   b_up:    [N]         up bias (can be NULL)
//   output:  [M, N]      result
//
// The key insight: gate and up values stay in registers, never written to DRAM
// =============================================================================
void gemm_swiglu_fused(const float *x,
                       const float *W_gate,
                       const float *W_up,
                       const float *b_gate,
                       const float *b_up,
                       float *output,
                       int M, int N, int K)
{
#if defined(__AVX__)
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        const float *x_row = &x[i * K];
        float *out_row = &output[i * N];

        for (int j = 0; j < N; j++) {
            const float *w_gate_row = &W_gate[j * K];
            const float *w_up_row = &W_up[j * K];

            // Compute both dot products in parallel using SIMD
            __m256 gate_vec = _mm256_setzero_ps();
            __m256 up_vec = _mm256_setzero_ps();

            int k;
            for (k = 0; k <= K - 8; k += 8) {
                __m256 x_vec = _mm256_loadu_ps(&x_row[k]);
                __m256 wg_vec = _mm256_loadu_ps(&w_gate_row[k]);
                __m256 wu_vec = _mm256_loadu_ps(&w_up_row[k]);

                // gate += x * W_gate
                gate_vec = _mm256_add_ps(gate_vec, _mm256_mul_ps(x_vec, wg_vec));
                // up += x * W_up
                up_vec = _mm256_add_ps(up_vec, _mm256_mul_ps(x_vec, wu_vec));
            }

            // Horizontal sum
            float gate = hsum256_ps_fused(gate_vec);
            float up = hsum256_ps_fused(up_vec);

            // Scalar remainder
            for (; k < K; k++) {
                gate += x_row[k] * w_gate_row[k];
                up += x_row[k] * w_up_row[k];
            }

            // Add biases
            if (b_gate) gate += b_gate[j];
            if (b_up) up += b_up[j];

            // SwiGLU: SiLU(gate) * up = gate * sigmoid(gate) * up
            float sig = 1.0f / (1.0f + expf(-gate));
            out_row[j] = gate * sig * up;
        }
    }
#else
    // Scalar fallback
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float gate = 0.0f;
            float up = 0.0f;

            for (int k = 0; k < K; k++) {
                gate += x[i * K + k] * W_gate[j * K + k];
                up += x[i * K + k] * W_up[j * K + k];
            }

            if (b_gate) gate += b_gate[j];
            if (b_up) up += b_up[j];

            // SwiGLU: SiLU(gate) * up
            float sig = 1.0f / (1.0f + expf(-gate));
            output[i * N + j] = gate * sig * up;
        }
    }
#endif
}
