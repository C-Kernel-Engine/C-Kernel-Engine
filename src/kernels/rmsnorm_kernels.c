/**
 * @file rmsnorm_kernels.c
 * @brief RMSNorm forward/backward kernels with SIMD (SSE/AVX/AVX512)
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
 * RMSNorm: y[i] = gamma[i] * x[i] / sqrt(mean(x^2) + eps)
 */

#include "bf16_utils.h"
#include "ckernel_engine.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
static inline __m256 rmsnorm_square_avx2_no_contract(__m256 values)
{
#if defined(__GNUC__) || defined(__clang__)
    __m256 squared;
    __asm__ volatile ("vmulps %1, %1, %0" : "=x"(squared) : "x"(values));
    return squared;
#else
    /* Preserve the materialized pow(2) boundary on compilers where inline
     * assembly is unavailable. */
    _Alignas(32) volatile float materialized[8];
    _mm256_store_ps((float *)materialized, _mm256_mul_ps(values, values));
    return _mm256_load_ps((const float *)materialized);
#endif
}

static inline __m256 rmsnorm_add_avx2_ordered(__m256 left, __m256 right)
{
#if defined(__GNUC__) || defined(__clang__)
    __m256 sum;
    __asm__ volatile ("vaddps %2, %1, %0" : "=x"(sum) : "x"(left), "x"(right));
    return sum;
#else
    _Alignas(32) volatile float materialized[8];
    _mm256_store_ps((float *)materialized, _mm256_add_ps(left, right));
    return _mm256_load_ps((const float *)materialized);
#endif
}

static inline __m256 rmsnorm_load_bf16_values_avx2(const float *values)
{
    _Alignas(32) float rounded[8];
    for (int lane = 0; lane < 8; ++lane) {
        rounded[lane] = bf16_to_float(float_to_bf16(values[lane]));
    }
    return _mm256_load_ps(rounded);
}
#endif

#if defined(__i386__) || defined(__x86_64__)
static inline float rmsnorm_div_f32_ordered(float numerator, float denominator)
{
#if defined(__GNUC__) || defined(__clang__)
    float quotient;
    __asm__ volatile ("vdivss %2, %1, %0"
                      : "=x"(quotient)
                      : "x"(numerator), "x"(denominator));
    return quotient;
#else
    volatile float ordered_numerator = numerator;
    volatile float ordered_denominator = denominator;
    return ordered_numerator / ordered_denominator;
#endif
}
#endif

/* AVX1 horizontal sum helper (no _mm256_reduce_add_ps in AVX1) */
#if defined(__AVX__) && !defined(__AVX512F__)
static inline float hsum256_ps_rmsnorm(__m256 v) {
    // Sum upper and lower 128-bit lanes
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(lo, hi);
    // Horizontal add within 128-bit lane
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
#endif
static void rmsnorm_forward_strict_scalar(const float *input,
                                          const float *gamma,
                                          float *output,
                                          float *rstd_cache,
                                          int tokens,
                                          int d_model,
                                          int input_stride,
                                          int output_stride,
                                          float eps)
{
    const float inv_d = 1.0f / (float)d_model;
    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)input_stride;
        float *y = output + (size_t)t * (size_t)output_stride;

        float sum_sq = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            const float v = x[d];
            sum_sq += v * v;
        }
        const float mean_sq = sum_sq * inv_d;
        const float rstd = 1.0f / sqrtf(mean_sq + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }

        for (int d = 0; d < d_model; ++d) {
            const float x_hat = x[d] * rstd;
            y[d] = x_hat * gamma[d];
        }
        for (int d = d_model; d < output_stride; ++d) {
            y[d] = 0.0f;
        }
    }
}

#if defined(__clang__)
__attribute__((optnone, noinline))
#elif defined(__GNUC__)
__attribute__((optimize("O1,no-tree-vectorize,no-tree-slp-vectorize"), noinline))
#endif
void rmsnorm_forward_fp64_sum(const float *input,
                              const float *gamma,
                              float *output,
                              float *rstd_cache,
                              int tokens,
                              int d_model,
                              int aligned_embed_dim,
                              float eps)
{
    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)aligned_embed_dim;
        float *y = output + (size_t)t * (size_t)aligned_embed_dim;
        /* This provider's contract requires an ascending scalar reduction.
         * Keep the accumulator volatile so whole-program optimization cannot
         * reassociate the sum or replace it with SIMD partial reductions. */
        volatile double sum_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            const float square = x[d] * x[d];
            sum_sq = sum_sq + (double)square;
        }
        const float mean_sq = (float)(sum_sq / (double)d_model);
        const float rstd = 1.0f / sqrtf(mean_sq + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }
        for (int d = 0; d < d_model; ++d) {
            const float normalized = x[d] * rstd;
            y[d] = normalized * gamma[d];
        }
        for (int d = d_model; d < aligned_embed_dim; ++d) {
            y[d] = 0.0f;
        }
    }
}

static inline float rmsnorm_llama_production_rstd(float mean_eps)
{
    /*
     * ggml's production CPU RMSNorm emits scalar sqrt followed by scalar
     * division.  With -fno-math-errno ICX otherwise strength-reduces the C
     * expression to vrsqrt14ss plus one Newton step.  That estimate differs by
     * one ULP for some rows and the error is amplified by quantized
     * projections in deep recurrent models.
     */
#if defined(CK_TARGET_X86)
    const __m128 value = _mm_set_ss(mean_eps);
    const __m128 root = _mm_sqrt_ss(value);
    return _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(1.0f), root));
#else
    const volatile float root = sqrtf(mean_eps);
    return 1.0f / root;
#endif
}

#if defined(__clang__)
__attribute__((optnone, noinline))
#elif defined(__GNUC__)
__attribute__((optimize("O1,no-tree-vectorize,no-tree-slp-vectorize"), noinline))
#endif
void rmsnorm_forward_llama_production(const float *input,
                                      const float *gamma,
                                      float *output,
                                      float *rstd_cache,
                                      int tokens,
                                      int d_model,
                                      int aligned_embed_dim,
                                      float eps)
{
    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)aligned_embed_dim;
        float *y = output + (size_t)t * (size_t)aligned_embed_dim;
        volatile double sum_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            const float square = x[d] * x[d];
            sum_sq = sum_sq + (double)square;
        }
        const float mean_sq = (float)(sum_sq / (double)d_model);
        const float rstd = rmsnorm_llama_production_rstd(mean_sq + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }
        for (int d = 0; d < d_model; ++d) {
            /*
             * Keep the RMSNorm + scale expression fused at the source level.
             * llama.cpp's CPU graph fuses GGML_OP_RMS_NORM followed by
             * GGML_OP_MUL and evaluates this left-associative expression in
             * one kernel.  Materializing the normalized value as a named
             * float introduces a store/load rounding boundary under ICX and
             * differs by one ULP for otherwise identical inputs.
             */
            y[d] = x[d] * rstd * gamma[d];
        }
        for (int d = d_model; d < aligned_embed_dim; ++d) {
            y[d] = 0.0f;
        }
    }
}

/*
 * Backend-matched PyTorch BF16 RMSNorm contract.
 *
 * This is intentionally separate from the generic FP32 and strict/FP64
 * RMSNorm providers. FP64 accumulation lowers mathematical error, but it does
 * not reproduce PyTorch's BF16 CPU boundary values. The installed PyTorch 2.8
 * binary dispatches mean(sum(pow(x, 2))) through ATen's AVX2 float reduction:
 *
 *   BF16 load -> FP32 materialized square -> four-stream/four-level AVX2
 *   cascade -> ordered lane fold -> FP32 mean/div/sqrt -> BF16 normalized
 *   value -> BF16 gamma multiply -> BF16 output.
 *
 * ICX may otherwise contract the square with accumulation, tree-reduce the
 * scalar lane fold, or replace scalar division. The ordered intrinsic helpers
 * above prevent those transformations. This implementation follows PyTorch
 * aten/src/ATen/native/cpu/SumKernel.cpp (cascade_sum) at commit
 * a1cb3cc05d46d198467bebbb6e8fba50a325d4e7. oneDNN is not the RMSNorm
 * oracle; it is used by adjacent BF16 linear projections.
 */
static void rmsnorm_forward_pytorch_bf16_storage_impl(const float *input,
                                                       const float *gamma,
                                                       float *output,
                                                       float *rstd_cache,
                                                       int tokens,
                                                       int d_model,
                                                       int input_stride,
                                                       int output_stride,
                                                       float eps,
                                                       int qwen3next_weight_order)
{
    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)input_stride;
        float *y = output + (size_t)t * (size_t)output_stride;
        float sum_sq = 0.0f;

#if defined(__AVX2__)
        /* Match the installed ATen build's AVX2 floating-point cascade_sum
         * contract. It treats the
         * contiguous row as four interleaved vector streams, accumulates
         * 16-item blocks through four hierarchy levels, then folds the four
         * streams and vector lanes left-to-right. PyTorch materializes pow(2)
         * before mean(), so keep the multiply separate from accumulation.
         * The host supports AVX-512, but this PyTorch build has AVX2 reduction
         * providers only; provider ISA is part of the numerical contract. */
        __m256 level[4][4];
        for (int hierarchy = 0; hierarchy < 4; ++hierarchy) {
            for (int stream = 0; stream < 4; ++stream) {
                level[hierarchy][stream] = _mm256_setzero_ps();
            }
        }
        int d = 0;
        const int vector_count = d_model / 8;
        const int cascade_items = vector_count / 4;
        int level_power = 4;
        if (cascade_items > 1) {
            int ceil_log2 = 0;
            unsigned int value = (unsigned int)(cascade_items - 1);
            while (value != 0) {
                value >>= 1;
                ++ceil_log2;
            }
            const int candidate = ceil_log2 / 4;
            if (candidate > level_power) level_power = candidate;
        }
        const int level_step = 1 << level_power;
        const int level_mask = level_step - 1;
        int item = 0;
        for (; item + level_step <= cascade_items;) {
            for (int block = 0; block < level_step; ++block, ++item) {
                for (int stream = 0; stream < 4; ++stream) {
                    const int offset = (item * 4 + stream) * 8;
                    const __m256 values = rmsnorm_load_bf16_values_avx2(x + offset);
                    const __m256 squared = rmsnorm_square_avx2_no_contract(values);
                    level[0][stream] = rmsnorm_add_avx2_ordered(
                        level[0][stream], squared
                    );
                }
            }
            for (int hierarchy = 1; hierarchy < 4; ++hierarchy) {
                for (int stream = 0; stream < 4; ++stream) {
                    level[hierarchy][stream] = rmsnorm_add_avx2_ordered(
                        level[hierarchy][stream], level[hierarchy - 1][stream]
                    );
                    level[hierarchy - 1][stream] = _mm256_setzero_ps();
                }
                const int mask = level_mask << (hierarchy * level_power);
                if ((item & mask) != 0) break;
            }
        }
        for (; item < cascade_items; ++item) {
            for (int stream = 0; stream < 4; ++stream) {
                const int offset = (item * 4 + stream) * 8;
                const __m256 values = rmsnorm_load_bf16_values_avx2(x + offset);
                const __m256 squared = rmsnorm_square_avx2_no_contract(values);
                level[0][stream] = rmsnorm_add_avx2_ordered(
                    level[0][stream], squared
                );
            }
        }
        for (int hierarchy = 1; hierarchy < 4; ++hierarchy) {
            for (int stream = 0; stream < 4; ++stream) {
                level[0][stream] = rmsnorm_add_avx2_ordered(
                    level[0][stream], level[hierarchy][stream]
                );
            }
        }
        __m256 reduced = level[0][0];
        for (int stream = 1; stream < 4; ++stream) {
            reduced = rmsnorm_add_avx2_ordered(reduced, level[0][stream]);
        }
        d = cascade_items * 4 * 8;
        for (; d + 8 <= d_model; d += 8) {
            const __m256 values = rmsnorm_load_bf16_values_avx2(x + d);
            const __m256 squared = rmsnorm_square_avx2_no_contract(values);
            reduced = rmsnorm_add_avx2_ordered(reduced, squared);
        }
        _Alignas(32) float lanes[8];
        _mm256_store_ps(lanes, reduced);
        volatile float ordered_sum = 0.0f;
        for (int lane = 0; lane < 8; ++lane) {
            ordered_sum = ordered_sum + lanes[lane];
        }
        sum_sq = ordered_sum;
        for (; d < d_model; ++d) {
            const float value = bf16_to_float(float_to_bf16(x[d]));
            sum_sq += value * value;
        }
#else
        for (int d = 0; d < d_model; ++d) {
            const float value = bf16_to_float(float_to_bf16(x[d]));
            sum_sq += value * value;
        }
#endif

#if defined(__i386__) || defined(__x86_64__)
        const float variance = rmsnorm_div_f32_ordered(sum_sq, (float)d_model);
        const float rstd = rmsnorm_div_f32_ordered(1.0f, sqrtf(variance + eps));
#else
        const float variance = sum_sq / (float)d_model;
        const float rstd = 1.0f / sqrtf(variance + eps);
#endif
        if (rstd_cache) rstd_cache[t] = rstd;
        for (int d = 0; d < d_model; ++d) {
            const float value = bf16_to_float(float_to_bf16(x[d]));
            if (qwen3next_weight_order) {
                /* Qwen3Next: (FP32 normalized * FP32 weight).to(BF16). */
                y[d] = bf16_to_float(float_to_bf16(
                    (value * rstd) * gamma[d]));
            } else {
                const float weight =
                    bf16_to_float(float_to_bf16(gamma[d]));
                const float normalized =
                    bf16_to_float(float_to_bf16(value * rstd));
                y[d] = bf16_to_float(float_to_bf16(normalized * weight));
            }
        }
        for (int d = d_model; d < output_stride; ++d) y[d] = 0.0f;
    }
}

void rmsnorm_forward_pytorch_bf16_storage(const float *input,
                                          const float *gamma,
                                          float *output,
                                          float *rstd_cache,
                                          int tokens,
                                          int d_model,
                                          int aligned_embed_dim,
                                          float eps)
{
    rmsnorm_forward_pytorch_bf16_storage_impl(
        input, gamma, output, rstd_cache, tokens, d_model,
        aligned_embed_dim, aligned_embed_dim, eps, 0);
}

void rmsnorm_forward_strided_pytorch_bf16_storage(const float *input,
                                                   const float *gamma,
                                                   float *output,
                                                   float *rstd_cache,
                                                   int tokens,
                                                   int d_model,
                                                   int input_stride,
                                                   int output_stride,
                                                   float eps)
{
    rmsnorm_forward_pytorch_bf16_storage_impl(
        input, gamma, output, rstd_cache, tokens, d_model,
        input_stride, output_stride, eps, 0);
}

void rmsnorm_forward_qwen3next_pytorch_bf16_storage(
                                          const float *input,
                                          const float *gamma,
                                          float *output,
                                          float *rstd_cache,
                                          int tokens,
                                          int d_model,
                                          int aligned_embed_dim,
                                          float eps)
{
    rmsnorm_forward_pytorch_bf16_storage_impl(
        input, gamma, output, rstd_cache, tokens, d_model,
        aligned_embed_dim, aligned_embed_dim, eps, 1);
}

static void rmsnorm_backward_strict_scalar(const float *d_output,
                                           const float *input,
                                           const float *gamma,
                                           const float *rstd_cache,
                                           float *d_input,
                                           float *d_gamma,
                                           int tokens,
                                           int d_model,
                                           int aligned_embed_dim)
{
    const float inv_d = 1.0f / (float)d_model;
    for (int d = 0; d < d_model; ++d) {
        d_gamma[d] = 0.0f;
    }

    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)aligned_embed_dim;
        const float *dY = d_output + (size_t)t * (size_t)aligned_embed_dim;
        float *dX = d_input + (size_t)t * (size_t)aligned_embed_dim;
        const float rstd = rstd_cache[t];

        float sum_dY_g_xhat = 0.0f;
        for (int d = 0; d < d_model; ++d) {
            const float x_hat = x[d] * rstd;
            const float grad_x_hat = dY[d] * gamma[d];
            sum_dY_g_xhat += x_hat * grad_x_hat;
        }

        for (int d = 0; d < d_model; ++d) {
            const float x_hat = x[d] * rstd;
            const float grad_x_hat = dY[d] * gamma[d];
            dX[d] = (grad_x_hat - (x_hat * inv_d) * sum_dY_g_xhat) * rstd;
            d_gamma[d] += dY[d] * x_hat;
        }
        for (int d = d_model; d < aligned_embed_dim; ++d) {
            dX[d] = 0.0f;
        }
    }
}


/**
 * RMSNorm forward pass
 * @test test_rmsnorm.py::TestRMSNormForward::test_fp32_tokens
 * @test test_rmsnorm.py::TestRMSNormForward::test_fp32_single
 * @test test_rmsnorm.py::TestRMSNormForward::test_perf_rolled
 * @test test_layernorm.py::TestLayerNormForward::test_rmsnorm_compat
 * @test test_parity.py::test_rmsnorm_parity
 *
 * RMSNorm: y[i] = gamma[i] * x[i] / sqrt(mean(x^2) + eps)
 *
 * After changes: make test && make llamacpp-parity-full
 */
void rmsnorm_forward_strided_f32(const float *input,
                                 const float *gamma,
                                 float *output,
                                 float *rstd_cache,
                                 int tokens,
                                 int d_model,
                                 int input_stride,
                                 int output_stride,
                                 float eps)
{
    int T = tokens;
    int D = d_model;

    const char *exact_env = getenv("CK_RMSNORM_EXACT");
    if (ck_strict_parity_enabled() || (exact_env && atoi(exact_env) != 0)) {
        rmsnorm_forward_strict_scalar(
            input, gamma, output, rstd_cache, T, D, input_stride, output_stride, eps
        );
        return;
    }

    for (int t = 0; t < T; ++t) {
        const float *x = input + (size_t)t * (size_t)input_stride;
        float *y = output + (size_t)t * (size_t)output_stride;

#if defined(__AVX512F__)
        // AVX-512: Process 16 floats at a time
        __m512 sum_sq_vec = _mm512_setzero_ps();
        int d = 0;

        // Vectorized sum of squares
        for (; d + 16 <= D; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            sum_sq_vec = _mm512_fmadd_ps(xv, xv, sum_sq_vec);
        }
        float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            sum_sq += x[d] * x[d];
        }

        float mean_sq = sum_sq / (float)D;
        float rstd = 1.0f / sqrtf(mean_sq + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }

        // Apply normalization and scale (vectorized)
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);
            __m512 yv = _mm512_mul_ps(x_hat, gv);
            _mm512_storeu_ps(&y[d], yv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            y[d] = x[d] * rstd * gamma[d];
        }

#elif defined(__AVX__)
        // AVX: Process 8 floats at a time
        __m256 sum_sq_vec = _mm256_setzero_ps();
        int d = 0;

        // Vectorized sum of squares (no FMA in AVX1, use mul + add)
        for (; d + 8 <= D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 xv_sq = _mm256_mul_ps(xv, xv);
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, xv_sq);
        }
        float sum_sq = hsum256_ps_rmsnorm(sum_sq_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            sum_sq += x[d] * x[d];
        }

        float mean_sq = sum_sq / (float)D;
        float rstd = 1.0f / sqrtf(mean_sq + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }

        // Apply normalization and scale (vectorized)
        __m256 rstd_vec = _mm256_set1_ps(rstd);
        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 gv = _mm256_loadu_ps(&gamma[d]);
            __m256 x_hat = _mm256_mul_ps(xv, rstd_vec);
            __m256 yv = _mm256_mul_ps(x_hat, gv);
            _mm256_storeu_ps(&y[d], yv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            y[d] = x[d] * rstd * gamma[d];
        }

#else
        // Scalar fallback
        float sum_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            float v = x[d];
            sum_sq += v * v;
        }
        float mean_sq = sum_sq / (float)D;
        float rstd = 1.0f / sqrtf(mean_sq + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }

        // Apply normalization and scale
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            y[d] = x_hat * gamma[d];
        }
#endif

        // Zero padding (if any)
        for (int d = D; d < output_stride; ++d) {
            y[d] = 0.0f;
        }
    }
}

void rmsnorm_forward(const float *input,
                     const float *gamma,
                     float *output,
                     float *rstd_cache,
                     int tokens,
                     int d_model,
                     int aligned_embed_dim,
                     float eps)
{
    rmsnorm_forward_strided_f32(
        input,
        gamma,
        output,
        rstd_cache,
        tokens,
        d_model,
        aligned_embed_dim,
        aligned_embed_dim,
        eps
    );
}

void rmsnorm_forward_kv_lora(const float *input,
                             const float *gamma,
                             float *output,
                             float *rstd_cache,
                             int tokens,
                             int d_model,
                             int aligned_embed_dim,
                             float eps)
{
    rmsnorm_forward(input, gamma, output, rstd_cache, tokens, d_model, aligned_embed_dim, eps);
}

void rmsnorm_forward_no_weight(const float *input,
                               float *output,
                               float *rstd_cache,
                               int tokens,
                               int d_model,
                               int aligned_embed_dim,
                               float eps)
{
    if (!input || !output || tokens <= 0 || d_model <= 0 || aligned_embed_dim <= 0) {
        return;
    }
    const float inv_d = 1.0f / (float)d_model;
    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)aligned_embed_dim;
        float *y = output + (size_t)t * (size_t)aligned_embed_dim;
        double sum_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            sum_sq += (double)x[d] * (double)x[d];
        }
        const float rstd = 1.0f / sqrtf((float)(sum_sq * (double)inv_d) + eps);
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }
        for (int d = 0; d < d_model; ++d) {
            y[d] = x[d] * rstd;
        }
        for (int d = d_model; d < aligned_embed_dim; ++d) {
            y[d] = 0.0f;
        }
    }
}

void gemma4_v_norm_forward(const float *input,
                           float *output,
                           float *rstd_cache,
                           int tokens,
                           int num_kv_heads,
                           int head_dim,
                           float eps)
{
    if (!input || !output || tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        return;
    }
    rmsnorm_forward_no_weight(input, output, rstd_cache,
                              tokens * num_kv_heads, head_dim, head_dim, eps);
}


/**
 * RMSNorm backward pass
 * @test test_rmsnorm.py::TestRMSNormBackward::test_backward_tokens
 * @test test_rmsnorm.py::TestRMSNormBackward::test_backward_single
 * @test test_parity.py::test_rmsnorm_backward_parity
 *
 * Computes dX and dGamma given dY, X, gamma, and cached rstd.
 * dX_i = rstd * (dY_i * gamma_i - x_hat_i * m)
 * dGamma_i = sum_t (dY_i * x_hat_i)
 *
 * After changes: make test && make llamacpp-parity-full
 */
void rmsnorm_backward(const float *d_output,
                      const float *input,
                      const float *gamma,
                      const float *rstd_cache,
                      float *d_input,
                      float *d_gamma,
                      int tokens,
                      int d_model,
                      int aligned_embed_dim)
{
    int T = tokens;
    int D = d_model;
    int aligned = aligned_embed_dim;

    if (ck_strict_parity_enabled()) {
        rmsnorm_backward_strict_scalar(d_output, input, gamma, rstd_cache, d_input, d_gamma, T, D, aligned);
        return;
    }

    // Zero parameter gradients
#if defined(__AVX512F__)
    {
        int d = 0;
        for (; d + 16 <= D; d += 16) {
            _mm512_storeu_ps(&d_gamma[d], _mm512_setzero_ps());
        }
        for (; d < D; ++d) {
            d_gamma[d] = 0.0f;
        }
    }
#elif defined(__AVX__)
    {
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            _mm256_storeu_ps(&d_gamma[d], _mm256_setzero_ps());
        }
        for (; d < D; ++d) {
            d_gamma[d] = 0.0f;
        }
    }
#else
    for (int d = 0; d < D; ++d) {
        d_gamma[d] = 0.0f;
    }
#endif

    for (int t = 0; t < T; ++t) {
        const float *x = input + (size_t)t * aligned;
        const float *dY = d_output + (size_t)t * aligned;
        float *dX = d_input + (size_t)t * aligned;

        float rstd = rstd_cache[t];

#if defined(__AVX512F__)
        // Compute m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
        __m512 rstd_vec = _mm512_set1_ps(rstd);
        __m512 sum_vec = _mm512_setzero_ps();
        int d = 0;

        for (; d + 16 <= D; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            __m512 dyv = _mm512_loadu_ps(&dY[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);
            // sum += dY * gamma * x_hat
            __m512 prod = _mm512_mul_ps(dyv, gv);
            sum_vec = _mm512_fmadd_ps(prod, x_hat, sum_vec);
        }
        float sum_dY_g_xhat = _mm512_reduce_add_ps(sum_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            float x_hat = x[d] * rstd;
            sum_dY_g_xhat += dY[d] * gamma[d] * x_hat;
        }
        float m = sum_dY_g_xhat / (float)D;

        // Compute dX and accumulate dGamma (vectorized)
        __m512 m_vec = _mm512_set1_ps(m);
        d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 xv = _mm512_loadu_ps(&x[d]);
            __m512 dyv = _mm512_loadu_ps(&dY[d]);
            __m512 gv = _mm512_loadu_ps(&gamma[d]);
            __m512 dgv = _mm512_loadu_ps(&d_gamma[d]);

            __m512 x_hat = _mm512_mul_ps(xv, rstd_vec);

            // dX = rstd * (dY * gamma - x_hat * m)
            __m512 dy_g = _mm512_mul_ps(dyv, gv);
            __m512 xhat_m = _mm512_mul_ps(x_hat, m_vec);
            __m512 diff = _mm512_sub_ps(dy_g, xhat_m);
            __m512 dxv = _mm512_mul_ps(rstd_vec, diff);
            _mm512_storeu_ps(&dX[d], dxv);

            // d_gamma += dY * x_hat
            dgv = _mm512_fmadd_ps(dyv, x_hat, dgv);
            _mm512_storeu_ps(&d_gamma[d], dgv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            float x_hat = x[d] * rstd;
            float dy = dY[d];
            dX[d] = rstd * (dy * gamma[d] - x_hat * m);
            d_gamma[d] += dy * x_hat;
        }

#elif defined(__AVX__)
        // Compute m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
        __m256 rstd_vec = _mm256_set1_ps(rstd);
        __m256 sum_vec = _mm256_setzero_ps();
        int d = 0;

        for (; d + 8 <= D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 dyv = _mm256_loadu_ps(&dY[d]);
            __m256 gv = _mm256_loadu_ps(&gamma[d]);
            __m256 x_hat = _mm256_mul_ps(xv, rstd_vec);
            // sum += dY * gamma * x_hat (no FMA, use mul + mul + add)
            __m256 prod = _mm256_mul_ps(dyv, gv);
            __m256 prod2 = _mm256_mul_ps(prod, x_hat);
            sum_vec = _mm256_add_ps(sum_vec, prod2);
        }
        float sum_dY_g_xhat = hsum256_ps_rmsnorm(sum_vec);

        // Handle remaining elements
        for (; d < D; ++d) {
            float x_hat = x[d] * rstd;
            sum_dY_g_xhat += dY[d] * gamma[d] * x_hat;
        }
        float m = sum_dY_g_xhat / (float)D;

        // Compute dX and accumulate dGamma (vectorized)
        __m256 m_vec = _mm256_set1_ps(m);
        d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 xv = _mm256_loadu_ps(&x[d]);
            __m256 dyv = _mm256_loadu_ps(&dY[d]);
            __m256 gv = _mm256_loadu_ps(&gamma[d]);
            __m256 dgv = _mm256_loadu_ps(&d_gamma[d]);

            __m256 x_hat = _mm256_mul_ps(xv, rstd_vec);

            // dX = rstd * (dY * gamma - x_hat * m)
            __m256 dy_g = _mm256_mul_ps(dyv, gv);
            __m256 xhat_m = _mm256_mul_ps(x_hat, m_vec);
            __m256 diff = _mm256_sub_ps(dy_g, xhat_m);
            __m256 dxv = _mm256_mul_ps(rstd_vec, diff);
            _mm256_storeu_ps(&dX[d], dxv);

            // d_gamma += dY * x_hat
            __m256 dy_xhat = _mm256_mul_ps(dyv, x_hat);
            dgv = _mm256_add_ps(dgv, dy_xhat);
            _mm256_storeu_ps(&d_gamma[d], dgv);
        }
        // Handle remaining elements
        for (; d < D; ++d) {
            float x_hat = x[d] * rstd;
            float dy = dY[d];
            dX[d] = rstd * (dy * gamma[d] - x_hat * m);
            d_gamma[d] += dy * x_hat;
        }

#else
        // Scalar fallback
        // Compute m = (1/D) * sum_j (dY_j * gamma_j * x_hat_j)
        float sum_dY_g_xhat = 0.0f;
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            sum_dY_g_xhat += dY[d] * gamma[d] * x_hat;
        }
        float m = sum_dY_g_xhat / (float)D;

        // Compute dX and accumulate dGamma
        for (int d = 0; d < D; ++d) {
            float x_hat = x[d] * rstd;
            float dy = dY[d];
            dX[d] = rstd * (dy * gamma[d] - x_hat * m);
            d_gamma[d] += dy * x_hat;
        }
#endif

        // Zero padding gradients (if any)
        for (int d = D; d < aligned; ++d) {
            dX[d] = 0.0f;
        }
    }
}
