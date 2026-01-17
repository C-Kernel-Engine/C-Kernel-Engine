/**
 * @file mlp_fused_decode.c
 * @brief Fully fused MLP decode kernel (T=1 token generation)
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
 * LEGACY: This file is from v6/v6.5 and kept for backward compatibility.
 *
 * This kernel fuses the ENTIRE MLP block into a single pass:
 *   output = Down(SwiGLU(Gate(x), Up(x))) + residual
 *
 * Key optimization: The intermediate SwiGLU values (~4864 floats = 19KB for Qwen2)
 * NEVER touch DRAM. They stay in L1/L2 cache through tiling.
 *
 * Target: Intel Xeon 5th Gen (Emerald Rapids) with AVX-512 and AMX
 *
 * Memory traffic comparison (Qwen2-0.5B, D=896, Hff=4864):
 *   Unfused: 76 KB activation traffic (38KB write + 38KB read)
 *   Fused:   0 KB activation traffic (tiles stay in L1)
 *
 * Weight layout expected: Row-major, transposed for matvec
 *   W_gate[Hff, D], W_up[Hff, D], W_down[D, Hff]
 */

#include "ckernel_engine.h"
#include <math.h>
#include <stdlib.h>  // for aligned_alloc in v1
#include <string.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// Configuration for Xeon 5th Gen
// =============================================================================

// L1 data cache: 48 KB per core on Sapphire/Emerald Rapids
// L2 cache: 2 MB per core
// We use a tile size that fits comfortably in L1 with room for weights
#define MLP_TILE_SIZE 64   // 64 intermediate values = 256 bytes

// For down projection accumulation, we tile the output dimension
#define OUTPUT_TILE_SIZE 32  // 32 output values accumulated at once

// =============================================================================
// AVX-512 Helpers
// =============================================================================

#if defined(__AVX512F__)
// Fast SiLU (x * sigmoid(x)) using AVX-512
// sigmoid(x) = 1 / (1 + exp(-x))
static inline __m512 silu_avx512(__m512 x) {
    // Compute -x
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);

    // exp(-x) approximation using polynomial (faster than _mm512_exp_ps)
    // We use the identity: exp(-x) = 2^(-x/ln2)
    // For better accuracy with larger ranges, we clamp
    __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);
    __m512 log2e = _mm512_set1_ps(1.4426950408889634f);

    // Clamp to avoid overflow/underflow
    neg_x = _mm512_max_ps(neg_x, _mm512_set1_ps(-88.0f));
    neg_x = _mm512_min_ps(neg_x, _mm512_set1_ps(88.0f));

    // Use the built-in exp if available (Xeon has fast transcendentals)
    // Otherwise fall back to polynomial approximation
#if defined(__AVX512ER__)  // Knights Landing/Mill have fast exp
    __m512 exp_neg_x = _mm512_exp2a23_ps(_mm512_mul_ps(neg_x, log2e));
#else
    // Polynomial approximation for exp(-x)
    // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 (good for |x| < 4)
    // For larger x, we use range reduction
    __m512 t = _mm512_mul_ps(neg_x, log2e);
    __m512i ti = _mm512_cvtps_epi32(t);
    __m512 tf = _mm512_sub_ps(t, _mm512_cvtepi32_ps(ti));
    tf = _mm512_mul_ps(tf, ln2);

    // Polynomial for 2^frac
    __m512 c0 = _mm512_set1_ps(1.0f);
    __m512 c1 = _mm512_set1_ps(0.6931471805599453f);
    __m512 c2 = _mm512_set1_ps(0.2402265069591007f);
    __m512 c3 = _mm512_set1_ps(0.05550410866482158f);
    __m512 c4 = _mm512_set1_ps(0.009618129107628477f);

    __m512 p = _mm512_fmadd_ps(c4, tf, c3);
    p = _mm512_fmadd_ps(p, tf, c2);
    p = _mm512_fmadd_ps(p, tf, c1);
    p = _mm512_fmadd_ps(p, tf, c0);

    // Scale by 2^int
    __m512 exp_neg_x = _mm512_scalef_ps(p, _mm512_cvtepi32_ps(ti));
#endif

    // sigmoid = 1 / (1 + exp(-x))
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 sigmoid = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg_x));

    // silu = x * sigmoid(x)
    return _mm512_mul_ps(x, sigmoid);
}

// Horizontal sum of __m512 (AVX-512F only, no DQ required)
static inline float hsum512_ps(__m512 v) {
    // Use shuffle-based reduction (AVX-512F compatible)
    // Reduce 16 -> 8
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
    __m256 sum256 = _mm256_add_ps(lo, hi);
    // Reduce 8 -> 4
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    // Reduce 4 -> 2 -> 1
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif

// Scalar SiLU for fallback and remainder
static inline float silu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Fully Fused MLP Decode (Main Kernel)
// =============================================================================
//
// Computes: output[D] = SwiGLU_MLP(x[D]) where
//   gate = x @ W_gate^T + b_gate
//   up   = x @ W_up^T + b_up
//   swiglu = SiLU(gate) * up
//   output = swiglu @ W_down^T + b_down
//
// Tiling strategy:
//   - Process intermediate dimension in tiles of MLP_TILE_SIZE
//   - For each tile: compute gate, up, swiglu (stays in registers/L1)
//   - Immediately accumulate into output via W_down
//   - Swiglu tile NEVER written to DRAM
//
void fused_mlp_swiglu_decode(
    const float *x,           // [D] input (after RMSNorm)
    const float *W_gate,      // [Hff, D] gate projection weights
    const float *W_up,        // [Hff, D] up projection weights
    const float *W_down,      // [D, Hff] down projection weights
    const float *b_gate,      // [Hff] gate bias (can be NULL)
    const float *b_up,        // [Hff] up bias (can be NULL)
    const float *b_down,      // [D] down bias (can be NULL)
    float *output,            // [D] output
    int D,                    // hidden dimension (e.g., 896)
    int Hff)                  // intermediate dimension (e.g., 4864)
{
#if defined(__AVX512F__)
    // Initialize output with bias or zero
    if (b_down) {
        memcpy(output, b_down, D * sizeof(float));
    } else {
        memset(output, 0, D * sizeof(float));
    }

    // Process intermediate dimension in tiles
    // Each tile computes MLP_TILE_SIZE swiglu values and immediately
    // accumulates them into the output

    /* Bounds check for stack allocation */
    if (D > 4096) return;

    #pragma omp parallel
    {
        /* Thread-local accumulator on stack (no malloc!) */
        float local_output[4096] __attribute__((aligned(64)));
        memset(local_output, 0, D * sizeof(float));

        #pragma omp for schedule(static)
        for (int t = 0; t < Hff; t += MLP_TILE_SIZE) {
            int tile_end = (t + MLP_TILE_SIZE < Hff) ? t + MLP_TILE_SIZE : Hff;
            int tile_size = tile_end - t;

            // Compute SwiGLU for this tile (stays in L1 cache)
            float swiglu_tile[MLP_TILE_SIZE] __attribute__((aligned(64)));

            for (int j = t; j < tile_end; j++) {
                const float *wg_row = &W_gate[j * D];
                const float *wu_row = &W_up[j * D];

                // Compute gate = x @ W_gate[j] using AVX-512
                __m512 gate_acc = _mm512_setzero_ps();
                __m512 up_acc = _mm512_setzero_ps();

                int k = 0;
                for (; k <= D - 16; k += 16) {
                    __m512 x_vec = _mm512_loadu_ps(&x[k]);
                    __m512 wg_vec = _mm512_loadu_ps(&wg_row[k]);
                    __m512 wu_vec = _mm512_loadu_ps(&wu_row[k]);

                    gate_acc = _mm512_fmadd_ps(x_vec, wg_vec, gate_acc);
                    up_acc = _mm512_fmadd_ps(x_vec, wu_vec, up_acc);
                }

                float gate = hsum512_ps(gate_acc);
                float up = hsum512_ps(up_acc);

                // Scalar remainder
                for (; k < D; k++) {
                    gate += x[k] * wg_row[k];
                    up += x[k] * wu_row[k];
                }

                // Add biases
                if (b_gate) gate += b_gate[j];
                if (b_up) up += b_up[j];

                // SwiGLU: SiLU(gate) * up
                swiglu_tile[j - t] = silu_scalar(gate) * up;
            }

            // Accumulate into output via W_down
            // output[i] += sum_j(swiglu_tile[j] * W_down[i, t+j])
            for (int i = 0; i < D; i++) {
                const float *wd_row = &W_down[i * Hff + t];

                __m512 acc = _mm512_setzero_ps();
                int j = 0;
                for (; j <= tile_size - 16; j += 16) {
                    __m512 sw_vec = _mm512_loadu_ps(&swiglu_tile[j]);
                    __m512 wd_vec = _mm512_loadu_ps(&wd_row[j]);
                    acc = _mm512_fmadd_ps(sw_vec, wd_vec, acc);
                }

                float sum = hsum512_ps(acc);
                for (; j < tile_size; j++) {
                    sum += swiglu_tile[j] * wd_row[j];
                }

                local_output[i] += sum;
            }
        }

        // Reduce thread-local outputs
        #pragma omp critical
        {
            for (int i = 0; i < D; i++) {
                output[i] += local_output[i];
            }
        }
        /* No free - stack buffer auto-deallocates */
    }

#else
    // Scalar fallback (same algorithm, no SIMD)
    if (b_down) {
        memcpy(output, b_down, D * sizeof(float));
    } else {
        memset(output, 0, D * sizeof(float));
    }

    for (int t = 0; t < Hff; t += MLP_TILE_SIZE) {
        int tile_end = (t + MLP_TILE_SIZE < Hff) ? t + MLP_TILE_SIZE : Hff;
        int tile_size = tile_end - t;

        float swiglu_tile[MLP_TILE_SIZE];

        for (int j = t; j < tile_end; j++) {
            float gate = 0.0f;
            float up = 0.0f;

            for (int k = 0; k < D; k++) {
                gate += x[k] * W_gate[j * D + k];
                up += x[k] * W_up[j * D + k];
            }

            if (b_gate) gate += b_gate[j];
            if (b_up) up += b_up[j];

            swiglu_tile[j - t] = silu_scalar(gate) * up;
        }

        for (int i = 0; i < D; i++) {
            for (int j = 0; j < tile_size; j++) {
                output[i] += swiglu_tile[j] * W_down[i * Hff + t + j];
            }
        }
    }
#endif
}

// Forward declaration for fallback
void fused_mlp_swiglu_decode_tiled(
    const float *x, const float *W_gate, const float *W_up, const float *W_down,
    const float *b_gate, const float *b_up, const float *b_down,
    float *output, int D, int Hff);

// =============================================================================
// Optimized Version: Two-Phase with Stack Buffer (Best for 24+ cores)
// =============================================================================
//
// Phase 1: All threads compute swiglu values in parallel (no reduction needed)
// Phase 2: All threads compute output values in parallel (no reduction needed)
//
// Uses a stack-allocated buffer that fits in L2 cache.
// For Hff > MAX_SWIGLU_STACK, falls back to tiled version.
//
#define MAX_SWIGLU_STACK 8192  // 32KB buffer, fits in L2

void fused_mlp_swiglu_decode_v2(
    const float *x,           // [D]
    const float *W_gate,      // [Hff, D]
    const float *W_up,        // [Hff, D]
    const float *W_down,      // [D, Hff]
    const float *b_gate,      // [Hff] or NULL
    const float *b_up,        // [Hff] or NULL
    const float *b_down,      // [D] or NULL
    float *output,            // [D]
    int D,
    int Hff)
{
    // For large Hff, use tiled version to avoid stack overflow
    if (Hff > MAX_SWIGLU_STACK) {
        fused_mlp_swiglu_decode_tiled(x, W_gate, W_up, W_down,
                                      b_gate, b_up, b_down, output, D, Hff);
        return;
    }

#if defined(__AVX512F__)
    // Stack-allocated swiglu buffer (max 32KB)
    float swiglu[MAX_SWIGLU_STACK] __attribute__((aligned(64)));

    // Phase 1: Compute all swiglu values (parallelize over Hff)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < Hff; j++) {
        const float *wg_row = &W_gate[j * D];
        const float *wu_row = &W_up[j * D];

        __m512 gate_acc = _mm512_setzero_ps();
        __m512 up_acc = _mm512_setzero_ps();

        int k = 0;
        for (; k <= D - 16; k += 16) {
            __m512 x_vec = _mm512_loadu_ps(&x[k]);
            __m512 wg_vec = _mm512_loadu_ps(&wg_row[k]);
            __m512 wu_vec = _mm512_loadu_ps(&wu_row[k]);

            gate_acc = _mm512_fmadd_ps(x_vec, wg_vec, gate_acc);
            up_acc = _mm512_fmadd_ps(x_vec, wu_vec, up_acc);
        }

        float gate = hsum512_ps(gate_acc);
        float up = hsum512_ps(up_acc);

        for (; k < D; k++) {
            gate += x[k] * wg_row[k];
            up += x[k] * wu_row[k];
        }

        if (b_gate) gate += b_gate[j];
        if (b_up) up += b_up[j];

        swiglu[j] = silu_scalar(gate) * up;
    }

    // Phase 2: Down projection (parallelize over D)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < D; i++) {
        const float *wd_row = &W_down[i * Hff];

        __m512 acc = _mm512_setzero_ps();
        int j = 0;
        for (; j <= Hff - 16; j += 16) {
            __m512 sw_vec = _mm512_loadu_ps(&swiglu[j]);
            __m512 wd_vec = _mm512_loadu_ps(&wd_row[j]);
            acc = _mm512_fmadd_ps(sw_vec, wd_vec, acc);
        }

        float sum = hsum512_ps(acc);
        for (; j < Hff; j++) {
            sum += swiglu[j] * wd_row[j];
        }

        output[i] = sum + (b_down ? b_down[i] : 0.0f);
    }

#else
    // Scalar fallback with stack buffer
    float swiglu[MAX_SWIGLU_STACK];

    for (int j = 0; j < Hff; j++) {
        float gate = 0.0f, up = 0.0f;
        for (int k = 0; k < D; k++) {
            gate += x[k] * W_gate[j * D + k];
            up += x[k] * W_up[j * D + k];
        }
        if (b_gate) gate += b_gate[j];
        if (b_up) up += b_up[j];
        swiglu[j] = silu_scalar(gate) * up;
    }

    for (int i = 0; i < D; i++) {
        float sum = 0.0f;
        for (int j = 0; j < Hff; j++) {
            sum += swiglu[j] * W_down[i * Hff + j];
        }
        output[i] = sum + (b_down ? b_down[i] : 0.0f);
    }
#endif
}

// =============================================================================
// Version 3: True Zero-Copy Tiled Fusion (Best for Large L2)
// =============================================================================
//
// This version processes tiles of the intermediate dimension and immediately
// accumulates into output, without any intermediate buffer allocation.
//
// Optimal for Xeon 5th gen with 2MB L2 per core.
//
void fused_mlp_swiglu_decode_tiled(
    const float *x,           // [D]
    const float *W_gate,      // [Hff, D]
    const float *W_up,        // [Hff, D]
    const float *W_down,      // [D, Hff]
    const float *b_gate,      // [Hff] or NULL
    const float *b_up,        // [Hff] or NULL
    const float *b_down,      // [D] or NULL
    float *output,            // [D]
    int D,
    int Hff)
{
    // Tile size chosen to fit in L2 with W_down tile
    // Tile of swiglu: 256 floats = 1KB
    // Tile of W_down: 256 * D floats = 256 * 896 * 4 = 896KB
    // Fits in 2MB L2 with room for x and prefetch
    const int TILE = 256;

#if defined(__AVX512F__)
    // Initialize output
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < D; i++) {
        output[i] = b_down ? b_down[i] : 0.0f;
    }

    // Process tiles of intermediate dimension
    for (int t = 0; t < Hff; t += TILE) {
        int tile_end = (t + TILE < Hff) ? t + TILE : Hff;
        int tile_size = tile_end - t;

        // Compute swiglu tile
        float swiglu_tile[256] __attribute__((aligned(64)));

        #pragma omp parallel for schedule(static)
        for (int jj = 0; jj < tile_size; jj++) {
            int j = t + jj;
            const float *wg_row = &W_gate[j * D];
            const float *wu_row = &W_up[j * D];

            __m512 gate_acc = _mm512_setzero_ps();
            __m512 up_acc = _mm512_setzero_ps();

            int k = 0;
            for (; k <= D - 16; k += 16) {
                __m512 x_vec = _mm512_loadu_ps(&x[k]);
                __m512 wg_vec = _mm512_loadu_ps(&wg_row[k]);
                __m512 wu_vec = _mm512_loadu_ps(&wu_row[k]);

                gate_acc = _mm512_fmadd_ps(x_vec, wg_vec, gate_acc);
                up_acc = _mm512_fmadd_ps(x_vec, wu_vec, up_acc);
            }

            float gate = hsum512_ps(gate_acc);
            float up = hsum512_ps(up_acc);

            for (; k < D; k++) {
                gate += x[k] * wg_row[k];
                up += x[k] * wu_row[k];
            }

            if (b_gate) gate += b_gate[j];
            if (b_up) up += b_up[j];

            swiglu_tile[jj] = silu_scalar(gate) * up;
        }

        // Accumulate into output (parallelize over D)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < D; i++) {
            const float *wd_row = &W_down[i * Hff + t];

            __m512 acc = _mm512_setzero_ps();
            int j = 0;
            for (; j <= tile_size - 16; j += 16) {
                __m512 sw_vec = _mm512_loadu_ps(&swiglu_tile[j]);
                __m512 wd_vec = _mm512_loadu_ps(&wd_row[j]);
                acc = _mm512_fmadd_ps(sw_vec, wd_vec, acc);
            }

            float sum = hsum512_ps(acc);
            for (; j < tile_size; j++) {
                sum += swiglu_tile[j] * wd_row[j];
            }

            // Atomic add (or use thread-local buffers for better perf)
            #pragma omp atomic
            output[i] += sum;
        }
    }

#else
    // Scalar fallback
    for (int i = 0; i < D; i++) {
        output[i] = b_down ? b_down[i] : 0.0f;
    }

    for (int t = 0; t < Hff; t += TILE) {
        int tile_end = (t + TILE < Hff) ? t + TILE : Hff;

        float swiglu_tile[256];

        for (int j = t; j < tile_end; j++) {
            float gate = 0.0f, up = 0.0f;
            for (int k = 0; k < D; k++) {
                gate += x[k] * W_gate[j * D + k];
                up += x[k] * W_up[j * D + k];
            }
            if (b_gate) gate += b_gate[j];
            if (b_up) up += b_up[j];
            swiglu_tile[j - t] = silu_scalar(gate) * up;
        }

        for (int i = 0; i < D; i++) {
            for (int j = t; j < tile_end; j++) {
                output[i] += swiglu_tile[j - t] * W_down[i * Hff + j];
            }
        }
    }
#endif
}
