/**
 * @file layernorm_kernels.c
 * @brief LayerNorm forward/backward kernels with SIMD (SSE/AVX/AVX512)
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
 * LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
 */

#include "ckernel_engine.h"
#include "ckernel_normalization_checked.h"
#include "bf16_utils.h"

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__) || defined(__SSE2__)
#include <immintrin.h>
#endif
#include <math.h>
#include <stdlib.h>

static inline void zero_layernorm_padding(float *out_ptr,
                                          int d_model,
                                          int aligned_embed_dim)
{
    for (int idx = d_model; idx < aligned_embed_dim; ++idx) {
        out_ptr[idx] = 0.0f;
    }
}

void layernorm_naive_serial_matched_precision(const float *input,
                                              const float *gamma,
                                              const float *beta,
                                              float *output,
                                              float *mean_cache,
                                              float *rstd_cache,
                                              int tokens, int d_model, float eps);

#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma float_control(precise, on, push)
#endif
static void layernorm_forward_ggml_exact(const float *input,
                                         const float *gamma,
                                         const float *beta,
                                         float *output,
                                         float *mean_cache,
                                         float *rstd_cache,
                                         int tokens,
                                         int d_model,
                                         int input_stride,
                                         int output_stride,
                                         int aligned_embed_dim,
                                         float eps,
                                         int fixed_chunk4)
{
    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)input_stride;
        float *y = output + (size_t)t * (size_t)output_stride;

        double sum_acc = 0.0;
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
#endif
        for (int i = 0; i < d_model; ++i) {
            sum_acc += (double)x[i];
        }
        const float sum = (float)sum_acc;
        const float mean = sum / (float)d_model;

        double var_acc = 0.0;
        int i = 0;
        if (fixed_chunk4) {
            /* Match the checked provider's baseline four-value variance
             * reduction on all ISAs, including AVX2 and AVX-512 builds. */
            for (; i + 3 < d_model; i += 4) {
                const float a = x[i] - mean;
                const float b = x[i + 1] - mean;
                const float c = x[i + 2] - mean;
                const float d = x[i + 3] - mean;
                y[i] = a;
                y[i + 1] = b;
                y[i + 2] = c;
                y[i + 3] = d;
                /* Products round to FP32 before the pairwise sums. */
                volatile float aa = a * a;
                volatile float bb = b * b;
                volatile float cc = c * c;
                volatile float dd = d * d;
                var_acc += (double)((aa + bb) + (cc + dd));
            }
        } else {
#if defined(__AVX2__) && defined(__FMA__)
        /* This provider promises ggml's ordered AVX2 reduction. Keep that
         * arithmetic identity on AVX-512 hosts instead of widening it based
         * on the compiler target. */
        for (; i + 7 < d_model; i += 8) {
            __m256 val = _mm256_sub_ps(_mm256_loadu_ps(x + i),
                                       _mm256_set1_ps(mean));
            _mm256_storeu_ps(y + i, val);
            val = _mm256_mul_ps(val, val);
            __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                     _mm256_castps256_ps128(val));
            val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
            val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
            var_acc += (double)_mm_cvtss_f32(val2);
        }
#elif defined(__SSE2__)
        for (; i + 3 < d_model; i += 4) {
            __m128 val = _mm_sub_ps(_mm_loadu_ps(x + i),
                                    _mm_set1_ps(mean));
            _mm_storeu_ps(y + i, val);
            val = _mm_mul_ps(val, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
            val = _mm_add_ps(val, _mm_movehl_ps(val, val));
            val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
            __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
            val = _mm_add_ps(val, tmp);
            tmp = _mm_movehl_ps(tmp, val);
            val = _mm_add_ss(val, tmp);
#endif
            var_acc += (double)_mm_cvtss_f32(val);
        }
#endif
        }
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
#endif
        for (; i < d_model; ++i) {
            const float centered = x[i] - mean;
            y[i] = centered;
            var_acc += (double)(centered * centered);
        }
        const float variance = (float)(var_acc / (double)d_model);
        const float scale = 1.0f / sqrtf(variance + eps);

        if (mean_cache) {
            mean_cache[t] = mean;
        }
        if (rstd_cache) {
            rstd_cache[t] = scale;
        }

#if defined(__clang__)
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
#endif
        for (int i = 0; i < d_model; ++i) {
            y[i] *= scale;
        }
        if (gamma) {
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
#endif
            for (int i = 0; i < d_model; ++i) {
                y[i] *= gamma[i];
            }
        }
        if (beta) {
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
#endif
            for (int i = 0; i < d_model; ++i) {
                y[i] += beta[i];
            }
        }

        if (aligned_embed_dim > d_model) {
            zero_layernorm_padding(y, d_model, aligned_embed_dim);
        }
    }
}
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma float_control(pop)
#endif

#if defined(__AVX2__) || defined(__AVX__)
static inline float hsum256_ps(__m256 v)
{
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif
// Naive serial LayerNorm implementation (forward only), copied from C-Transformer.
void layernorm_naive_serial(const float *input,
                            const float *gamma,
                            const float *beta,
                            float *output,
                            float *mean_cache,
                            float *rstd_cache,
                            int tokens, int d_model, int aligned_embed_dim,
                            float eps)
{
    for (int t = 0; t < tokens; ++t) {
        const float *in_ptr = input + t * aligned_embed_dim;
        float *out_ptr = output + t * aligned_embed_dim;

        float sum_val = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            sum_val += in_ptr[i];
        }
        float mean = sum_val / (float)d_model;

        float sum_sq_diff = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            float diff = in_ptr[i] - mean;
            sum_sq_diff += diff * diff;
        }
        float variance = sum_sq_diff / (float)d_model + eps;

        double var_double = (double)variance;
        float inv_std = (float)(1.0 / sqrt(var_double));

        for (int i = 0; i < d_model; ++i) {
            float normalized_val = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized_val * gamma[i] + beta[i];
        }

        if (mean_cache) {
            mean_cache[t] = mean;
        }
        if (rstd_cache) {
            rstd_cache[t] = inv_std;
        }
        /* Keep aligned padding quiet so future GEMMs see deterministic memory. */
        if (aligned_embed_dim > d_model) {
            /* Keep padded lanes zeroed so subsequent GEMMs never read stale data. */
            for (int i = d_model; i < aligned_embed_dim; ++i) {
                out_ptr[i] = 0.0f;
            }
        }
    }
}

#if defined(__AVX512F__)
// AVX-512 rolled slice kernel, copied from C-Transformer (model-agnostic).
static void layernorm_forward_rolled_slice_avx512(const float *__restrict input_slice_base,
                                    const float *__restrict gamma,
                                    const float *__restrict beta,
                                    float *__restrict output_slice_base,
                                    float *__restrict mean_cache_slice,
                                    float *__restrict rstd_cache_slice,
                                    int num_tokens_in_slice,
                                    int d_model,
                                    int aligned_embed_dim,
                                    float eps)
{
    for (int t = 0; t < num_tokens_in_slice; ++t) {
        const float *in_ptr_token = input_slice_base + t * aligned_embed_dim;
        float *out_ptr_token = output_slice_base + t * aligned_embed_dim;

        __m512 acc_sum_vec = _mm512_setzero_ps();
        int j = 0;
        for (; j <= d_model - 16; j += 16) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            __m512 v = _mm512_load_ps(in_ptr_token + j);
            acc_sum_vec = _mm512_add_ps(acc_sum_vec, v);
        }
        float mean = _mm512_reduce_add_ps(acc_sum_vec);
        for (; j < d_model; ++j) {
            mean += in_ptr_token[j];
        }
        mean /= (float)d_model;
        __m512 mean_vec = _mm512_set1_ps(mean);

        __m512 acc_var_vec = _mm512_setzero_ps();
        j = 0;
        for (; j <= d_model - 16; j += 16) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            __m512 v = _mm512_load_ps(in_ptr_token + j);
            __m512 diff = _mm512_sub_ps(v, mean_vec);
            acc_var_vec = _mm512_fmadd_ps(diff, diff, acc_var_vec);
        }
        float var = _mm512_reduce_add_ps(acc_var_vec);
        for (; j < d_model; ++j) {
            float diff = in_ptr_token[j] - mean;
            var += diff * diff;
        }
        var = var / (float)d_model + eps;
        double var_double = (double)var;
        float inv_std = (float)(1.0 / sqrt(var_double));
        __m512 inv_std_vec = _mm512_set1_ps(inv_std);

        if (mean_cache_slice) {
            mean_cache_slice[t] = mean;
        }
        if (rstd_cache_slice) {
            rstd_cache_slice[t] = inv_std;
        }

        j = 0;
        for (; j <= d_model - 16; j += 16) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(gamma + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(beta + j + 128), _MM_HINT_T0);

            __m512 v = _mm512_load_ps(in_ptr_token + j);
            __m512 g = _mm512_load_ps(gamma + j);
            __m512 b = _mm512_load_ps(beta + j);

            __m512 n = _mm512_mul_ps(_mm512_sub_ps(v, mean_vec), inv_std_vec);
            __m512 o = _mm512_fmadd_ps(n, g, b);

            _mm512_store_ps(out_ptr_token + j, o);
        }
        for (; j < d_model; ++j) {
            float normed = (in_ptr_token[j] - mean) * inv_std;
            out_ptr_token[j] = normed * gamma[j] + beta[j];
        }

        if (aligned_embed_dim > d_model) {
            /* Keep the padded lanes zeroed so later GEMMs see deterministic memory. */
            zero_layernorm_padding(out_ptr_token, d_model, aligned_embed_dim);
        }
    }
}
#elif defined(__AVX2__) || defined(__AVX__)
// AVX/AVX2 rolled slice kernel (8-float vectors).
static void layernorm_forward_rolled_slice_avx256(const float *__restrict input_slice_base,
                                    const float *__restrict gamma,
                                    const float *__restrict beta,
                                    float *__restrict output_slice_base,
                                    float *__restrict mean_cache_slice,
                                    float *__restrict rstd_cache_slice,
                                    int num_tokens_in_slice,
                                    int d_model,
                                    int aligned_embed_dim,
                                    float eps)
{
    for (int t = 0; t < num_tokens_in_slice; ++t) {
        const float *in_ptr_token = input_slice_base + t * aligned_embed_dim;
        float *out_ptr_token = output_slice_base + t * aligned_embed_dim;

        __m256 acc_sum_vec = _mm256_setzero_ps();
        int j = 0;
        for (; j <= d_model - 8; j += 8) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            __m256 v = _mm256_load_ps(in_ptr_token + j);
            acc_sum_vec = _mm256_add_ps(acc_sum_vec, v);
        }
        float mean = hsum256_ps(acc_sum_vec);
        for (; j < d_model; ++j) {
            mean += in_ptr_token[j];
        }
        mean /= (float)d_model;
        __m256 mean_vec = _mm256_set1_ps(mean);

        __m256 acc_var_vec = _mm256_setzero_ps();
        j = 0;
        for (; j <= d_model - 8; j += 8) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            __m256 v = _mm256_load_ps(in_ptr_token + j);
            __m256 diff = _mm256_sub_ps(v, mean_vec);
#if defined(__FMA__)
            acc_var_vec = _mm256_fmadd_ps(diff, diff, acc_var_vec);
#else
            acc_var_vec = _mm256_add_ps(acc_var_vec, _mm256_mul_ps(diff, diff));
#endif
        }
        float var = hsum256_ps(acc_var_vec);
        for (; j < d_model; ++j) {
            float diff = in_ptr_token[j] - mean;
            var += diff * diff;
        }
        var = var / (float)d_model + eps;
        double var_double = (double)var;
        float inv_std = (float)(1.0 / sqrt(var_double));
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);

        if (mean_cache_slice) {
            mean_cache_slice[t] = mean;
        }
        if (rstd_cache_slice) {
            rstd_cache_slice[t] = inv_std;
        }

        j = 0;
        for (; j <= d_model - 8; j += 8) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(gamma + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(beta + j + 128), _MM_HINT_T0);

            __m256 v = _mm256_load_ps(in_ptr_token + j);
            __m256 g = _mm256_load_ps(gamma + j);
            __m256 b = _mm256_load_ps(beta + j);

            __m256 n = _mm256_mul_ps(_mm256_sub_ps(v, mean_vec), inv_std_vec);
#if defined(__FMA__)
            __m256 o = _mm256_fmadd_ps(n, g, b);
#else
            __m256 o = _mm256_add_ps(_mm256_mul_ps(n, g), b);
#endif

            _mm256_store_ps(out_ptr_token + j, o);
        }
        for (; j < d_model; ++j) {
            float normed = (in_ptr_token[j] - mean) * inv_std;
            out_ptr_token[j] = normed * gamma[j] + beta[j];
        }

        if (aligned_embed_dim > d_model) {
            zero_layernorm_padding(out_ptr_token, d_model, aligned_embed_dim);
        }
    }
}
#endif

void layernorm_forward_rolled_slice(const float *__restrict input_slice_base,
                                    const float *__restrict gamma,
                                    const float *__restrict beta,
                                    float *__restrict output_slice_base,
                                    float *__restrict mean_cache_slice,
                                    float *__restrict rstd_cache_slice,
                                    int num_tokens_in_slice,
                                    int d_model,
                                    int aligned_embed_dim,
                                    float eps)
{
    if (ck_strict_parity_enabled()) {
        layernorm_forward_ggml_exact(input_slice_base, gamma, beta,
                                     output_slice_base, mean_cache_slice, rstd_cache_slice,
                                     num_tokens_in_slice, d_model,
                                     aligned_embed_dim, aligned_embed_dim, aligned_embed_dim, eps, 0);
        return;
    }

#if defined(__AVX512F__)
    layernorm_forward_rolled_slice_avx512(input_slice_base, gamma, beta,
                                           output_slice_base, mean_cache_slice, rstd_cache_slice,
                                           num_tokens_in_slice, d_model, aligned_embed_dim, eps);
#elif defined(__AVX2__) || defined(__AVX__)
    layernorm_forward_rolled_slice_avx256(input_slice_base, gamma, beta,
                                           output_slice_base, mean_cache_slice, rstd_cache_slice,
                                           num_tokens_in_slice, d_model, aligned_embed_dim, eps);
#else
    layernorm_naive_serial(input_slice_base, gamma, beta,
                           output_slice_base, mean_cache_slice, rstd_cache_slice,
                           num_tokens_in_slice, d_model, aligned_embed_dim, eps);
#endif
}

#if defined(__AVX512F__)
// AVX-512 unrolled slice kernel, copied from C-Transformer (model-agnostic).
static void layernorm_forward_unrolled_slice_avx512(const float *__restrict input_slice_base,
                                    const float *__restrict gamma,
                                    const float *__restrict beta,
                                    float *__restrict output_slice_base,
                                      float *__restrict mean_cache_slice,
                                      float *__restrict rstd_cache_slice,
                                      int num_tokens_in_slice,
                                      int d_model,
                                      float eps)
{
    for (int t = 0; t < num_tokens_in_slice; ++t) {
        const float *in_ptr_token = input_slice_base + t * d_model;
        float *out_ptr_token = output_slice_base + t * d_model;

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        int j = 0;
        int unroll_factor_floats = 64;

        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);

            __m512 v0 = _mm512_load_ps(in_ptr_token + j);
            __m512 v1 = _mm512_load_ps(in_ptr_token + j + 16);
            __m512 v2 = _mm512_load_ps(in_ptr_token + j + 32);
            __m512 v3 = _mm512_load_ps(in_ptr_token + j + 48);

            acc0 = _mm512_add_ps(acc0, v0);
            acc1 = _mm512_add_ps(acc1, v1);
            acc2 = _mm512_add_ps(acc2, v2);
            acc3 = _mm512_add_ps(acc3, v3);
        }
        __m512 acc_sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1),
                                       _mm512_add_ps(acc2, acc3));
        float mean = _mm512_reduce_add_ps(acc_sum);

        for (; j < d_model; ++j) {
            mean += in_ptr_token[j];
        }
        mean /= (float)d_model;
        __m512 mean_vec = _mm512_set1_ps(mean);

        acc0 = _mm512_setzero_ps();
        acc1 = _mm512_setzero_ps();
        acc2 = _mm512_setzero_ps();
        acc3 = _mm512_setzero_ps();

        j = 0;
        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);

            __m512 v0 = _mm512_load_ps(in_ptr_token + j);
            __m512 v1 = _mm512_load_ps(in_ptr_token + j + 16);
            __m512 v2 = _mm512_load_ps(in_ptr_token + j + 32);
            __m512 v3 = _mm512_load_ps(in_ptr_token + j + 48);

            __m512 d0 = _mm512_sub_ps(v0, mean_vec);
            __m512 d1 = _mm512_sub_ps(v1, mean_vec);
            __m512 d2 = _mm512_sub_ps(v2, mean_vec);
            __m512 d3 = _mm512_sub_ps(v3, mean_vec);

            acc0 = _mm512_fmadd_ps(d0, d0, acc0);
            acc1 = _mm512_fmadd_ps(d1, d1, acc1);
            acc2 = _mm512_fmadd_ps(d2, d2, acc2);
            acc3 = _mm512_fmadd_ps(d3, d3, acc3);
        }
        acc_sum = _mm512_add_ps(_mm512_add_ps(acc0, acc1),
                                _mm512_add_ps(acc2, acc3));
        float var = _mm512_reduce_add_ps(acc_sum);

        for (; j < d_model; ++j) {
            float diff = in_ptr_token[j] - mean;
            var += diff * diff;
        }
        var = var / (float)d_model + eps;
        double var_double = (double)var;
        float inv_std = (float)(1.0 / sqrt(var_double));
        __m512 inv_std_vec = _mm512_set1_ps(inv_std);

        if (mean_cache_slice) {
            mean_cache_slice[t] = mean;
        }
        if (rstd_cache_slice) {
            rstd_cache_slice[t] = inv_std;
        }

        j = 0;
        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(gamma + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(beta + j + 128), _MM_HINT_T0);

            __m512 v0 = _mm512_load_ps(in_ptr_token + j);
            __m512 v1 = _mm512_load_ps(in_ptr_token + j + 16);
            __m512 v2 = _mm512_load_ps(in_ptr_token + j + 32);
            __m512 v3 = _mm512_load_ps(in_ptr_token + j + 48);

            __m512 g0 = _mm512_load_ps(gamma + j);
            __m512 g1 = _mm512_load_ps(gamma + j + 16);
            __m512 g2 = _mm512_load_ps(gamma + j + 32);
            __m512 g3 = _mm512_load_ps(gamma + j + 48);

            __m512 b0 = _mm512_load_ps(beta + j);
            __m512 b1 = _mm512_load_ps(beta + j + 16);
            __m512 b2 = _mm512_load_ps(beta + j + 32);
            __m512 b3 = _mm512_load_ps(beta + j + 48);

            __m512 n0 = _mm512_mul_ps(_mm512_sub_ps(v0, mean_vec), inv_std_vec);
            __m512 n1 = _mm512_mul_ps(_mm512_sub_ps(v1, mean_vec), inv_std_vec);
            __m512 n2 = _mm512_mul_ps(_mm512_sub_ps(v2, mean_vec), inv_std_vec);
            __m512 n3 = _mm512_mul_ps(_mm512_sub_ps(v3, mean_vec), inv_std_vec);

            __m512 o0 = _mm512_fmadd_ps(n0, g0, b0);
            __m512 o1 = _mm512_fmadd_ps(n1, g1, b1);
            __m512 o2 = _mm512_fmadd_ps(n2, g2, b2);
            __m512 o3 = _mm512_fmadd_ps(n3, g3, b3);

            _mm512_store_ps(out_ptr_token + j, o0);
            _mm512_store_ps(out_ptr_token + j + 16, o1);
            _mm512_store_ps(out_ptr_token + j + 32, o2);
            _mm512_store_ps(out_ptr_token + j + 48, o3);
        }
        for (; j < d_model; ++j) {
            float normed = (in_ptr_token[j] - mean) * inv_std;
            out_ptr_token[j] = normed * gamma[j] + beta[j];
        }
    }
}
#elif defined(__AVX2__) || defined(__AVX__)
// AVX/AVX2 unrolled slice kernel (8-float vectors).
static void layernorm_forward_unrolled_slice_avx256(const float *__restrict input_slice_base,
                                    const float *__restrict gamma,
                                    const float *__restrict beta,
                                    float *__restrict output_slice_base,
                                      float *__restrict mean_cache_slice,
                                      float *__restrict rstd_cache_slice,
                                      int num_tokens_in_slice,
                                      int d_model,
                                      float eps)
{
    for (int t = 0; t < num_tokens_in_slice; ++t) {
        const float *in_ptr_token = input_slice_base + t * d_model;
        float *out_ptr_token = output_slice_base + t * d_model;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int j = 0;
        int unroll_factor_floats = 32;

        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);

            __m256 v0 = _mm256_load_ps(in_ptr_token + j);
            __m256 v1 = _mm256_load_ps(in_ptr_token + j + 8);
            __m256 v2 = _mm256_load_ps(in_ptr_token + j + 16);
            __m256 v3 = _mm256_load_ps(in_ptr_token + j + 24);

            acc0 = _mm256_add_ps(acc0, v0);
            acc1 = _mm256_add_ps(acc1, v1);
            acc2 = _mm256_add_ps(acc2, v2);
            acc3 = _mm256_add_ps(acc3, v3);
        }
        __m256 acc_sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                       _mm256_add_ps(acc2, acc3));
        float mean = hsum256_ps(acc_sum);

        for (; j < d_model; ++j) {
            mean += in_ptr_token[j];
        }
        mean /= (float)d_model;
        __m256 mean_vec = _mm256_set1_ps(mean);

        acc0 = _mm256_setzero_ps();
        acc1 = _mm256_setzero_ps();
        acc2 = _mm256_setzero_ps();
        acc3 = _mm256_setzero_ps();

        j = 0;
        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);

            __m256 v0 = _mm256_load_ps(in_ptr_token + j);
            __m256 v1 = _mm256_load_ps(in_ptr_token + j + 8);
            __m256 v2 = _mm256_load_ps(in_ptr_token + j + 16);
            __m256 v3 = _mm256_load_ps(in_ptr_token + j + 24);

            __m256 d0 = _mm256_sub_ps(v0, mean_vec);
            __m256 d1 = _mm256_sub_ps(v1, mean_vec);
            __m256 d2 = _mm256_sub_ps(v2, mean_vec);
            __m256 d3 = _mm256_sub_ps(v3, mean_vec);

#if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
#else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
#endif
        }
        acc_sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                _mm256_add_ps(acc2, acc3));
        float var = hsum256_ps(acc_sum);

        for (; j < d_model; ++j) {
            float diff = in_ptr_token[j] - mean;
            var += diff * diff;
        }
        var = var / (float)d_model + eps;
        double var_double = (double)var;
        float inv_std = (float)(1.0 / sqrt(var_double));
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);

        if (mean_cache_slice) {
            mean_cache_slice[t] = mean;
        }
        if (rstd_cache_slice) {
            rstd_cache_slice[t] = inv_std;
        }

        j = 0;
        for (; j <= d_model - unroll_factor_floats; j += unroll_factor_floats) {
            _mm_prefetch((const char *)(in_ptr_token + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(gamma + j + 128), _MM_HINT_T0);
            _mm_prefetch((const char *)(beta + j + 128), _MM_HINT_T0);

            __m256 v0 = _mm256_load_ps(in_ptr_token + j);
            __m256 v1 = _mm256_load_ps(in_ptr_token + j + 8);
            __m256 v2 = _mm256_load_ps(in_ptr_token + j + 16);
            __m256 v3 = _mm256_load_ps(in_ptr_token + j + 24);

            __m256 g0 = _mm256_load_ps(gamma + j);
            __m256 g1 = _mm256_load_ps(gamma + j + 8);
            __m256 g2 = _mm256_load_ps(gamma + j + 16);
            __m256 g3 = _mm256_load_ps(gamma + j + 24);

            __m256 b0 = _mm256_load_ps(beta + j);
            __m256 b1 = _mm256_load_ps(beta + j + 8);
            __m256 b2 = _mm256_load_ps(beta + j + 16);
            __m256 b3 = _mm256_load_ps(beta + j + 24);

            __m256 n0 = _mm256_mul_ps(_mm256_sub_ps(v0, mean_vec), inv_std_vec);
            __m256 n1 = _mm256_mul_ps(_mm256_sub_ps(v1, mean_vec), inv_std_vec);
            __m256 n2 = _mm256_mul_ps(_mm256_sub_ps(v2, mean_vec), inv_std_vec);
            __m256 n3 = _mm256_mul_ps(_mm256_sub_ps(v3, mean_vec), inv_std_vec);

#if defined(__FMA__)
            __m256 o0 = _mm256_fmadd_ps(n0, g0, b0);
            __m256 o1 = _mm256_fmadd_ps(n1, g1, b1);
            __m256 o2 = _mm256_fmadd_ps(n2, g2, b2);
            __m256 o3 = _mm256_fmadd_ps(n3, g3, b3);
#else
            __m256 o0 = _mm256_add_ps(_mm256_mul_ps(n0, g0), b0);
            __m256 o1 = _mm256_add_ps(_mm256_mul_ps(n1, g1), b1);
            __m256 o2 = _mm256_add_ps(_mm256_mul_ps(n2, g2), b2);
            __m256 o3 = _mm256_add_ps(_mm256_mul_ps(n3, g3), b3);
#endif

            _mm256_store_ps(out_ptr_token + j, o0);
            _mm256_store_ps(out_ptr_token + j + 8, o1);
            _mm256_store_ps(out_ptr_token + j + 16, o2);
            _mm256_store_ps(out_ptr_token + j + 24, o3);
        }
        for (; j < d_model; ++j) {
            float normed = (in_ptr_token[j] - mean) * inv_std;
            out_ptr_token[j] = normed * gamma[j] + beta[j];
        }
    }
}
#else
// Scalar fallback when AVX-512 is unavailable.
static void layernorm_forward_unrolled_slice_scalar(const float *__restrict input_slice_base,
                                      const float *__restrict gamma,
                                      const float *__restrict beta,
                                      float *__restrict output_slice_base,
                                      float *__restrict mean_cache_slice,
                                      float *__restrict rstd_cache_slice,
                                      int num_tokens_in_slice,
                                      int d_model,
                                      float eps)
{
    layernorm_naive_serial_matched_precision(input_slice_base, gamma, beta,
                                             output_slice_base, mean_cache_slice, rstd_cache_slice,
                                             num_tokens_in_slice, d_model, eps);
}
#endif

void layernorm_forward_unrolled_slice(const float *__restrict input_slice_base,
                                      const float *__restrict gamma,
                                      const float *__restrict beta,
                                      float *__restrict output_slice_base,
                                      float *__restrict mean_cache_slice,
                                      float *__restrict rstd_cache_slice,
                                      int num_tokens_in_slice,
                                      int d_model,
                                      float eps)
{
    if (ck_strict_parity_enabled()) {
        layernorm_forward_ggml_exact(input_slice_base, gamma, beta,
                                     output_slice_base, mean_cache_slice, rstd_cache_slice,
                                     num_tokens_in_slice, d_model,
                                     d_model, d_model, d_model, eps, 0);
        return;
    }

#if defined(__AVX512F__)
    layernorm_forward_unrolled_slice_avx512(input_slice_base, gamma, beta,
                                             output_slice_base, mean_cache_slice, rstd_cache_slice,
                                             num_tokens_in_slice, d_model, eps);
#elif defined(__AVX2__) || defined(__AVX__)
    layernorm_forward_unrolled_slice_avx256(input_slice_base, gamma, beta,
                                             output_slice_base, mean_cache_slice, rstd_cache_slice,
                                             num_tokens_in_slice, d_model, eps);
#else
    layernorm_forward_unrolled_slice_scalar(input_slice_base, gamma, beta,
                                            output_slice_base, mean_cache_slice, rstd_cache_slice,
                                            num_tokens_in_slice, d_model, eps);
#endif
}

// Precision-matched naive LayerNorm used for benchmarking, copied from C-Transformer.
void layernorm_naive_serial_matched_precision(const float *input,
                                              const float *gamma,
                                              const float *beta,
                                              float *output,
                                              float *mean_cache,
                                              float *rstd_cache,
                                              int tokens, int d_model, float eps)
{
    layernorm_forward_ggml_exact(input, gamma, beta,
                                 output, mean_cache, rstd_cache,
                                 tokens, d_model,
                                 d_model, d_model, d_model, eps, 0);
}

void ck_layernorm_row_fixed_chunk4_f32(const float *input, const float *gamma,
    const float *beta, float *output, float *mean, float *rstd, int channels,
    float epsilon)
{
    layernorm_forward_ggml_exact(input, gamma, beta, output, mean, rstd,
                                 1, channels, channels, channels, channels,
                                 epsilon, 1);
}

/*
 * Float-buffer LayerNorm with BF16 storage boundaries.
 *
 * The physical activation arena remains FP32 so this variant composes with the
 * existing graph ABI. Inputs are first rounded as if loaded from BF16 storage,
 * the established matched reduction executes in FP32, and outputs are rounded
 * back to BF16 values while remaining represented as float.
 */
void layernorm_naive_serial_bf16_storage(const float *input,
                                         const float *gamma,
                                         const float *beta,
                                         float *output,
                                         float *mean_cache,
                                         float *rstd_cache,
                                         int tokens, int d_model, float eps)
{
    const size_t count = (size_t)tokens * (size_t)d_model;
    for (size_t i = 0; i < count; ++i) {
        output[i] = bf16_to_float(float_to_bf16(input[i]));
    }
    layernorm_forward_ggml_exact(output, gamma, beta,
                                 output, mean_cache, rstd_cache,
                                 tokens, d_model,
                                 d_model, d_model, d_model, eps, 0);
    for (size_t i = 0; i < count; ++i) {
        output[i] = bf16_to_float(float_to_bf16(output[i]));
    }
}

#if defined(__AVX2__) && defined(__FMA__)
static inline void layernorm_pytorch_add_moments_vec(
    int m0_add,
    __m256 m1_add,
    __m256 m2_add,
    int *m0,
    __m256 *m1,
    __m256 *m2)
{
    const int n = *m0 + m0_add;
    const float c = n == 0 ? 0.0f : (float)m0_add / (float)n;
    const __m256 delta = _mm256_sub_ps(m1_add, *m1);
    const __m256 c_vec = _mm256_set1_ps(c);
    const __m256 m2_tmp = _mm256_add_ps(*m2, m2_add);
    const __m256 c_delta = _mm256_mul_ps(c_vec, delta);
    const __m256 m0_delta = _mm256_mul_ps(
        delta, _mm256_set1_ps((float)*m0));
    *m1 = _mm256_add_ps(*m1, c_delta);
    *m2 = _mm256_fmadd_ps(m0_delta, c_delta, m2_tmp);
    *m0 = n;
}

static inline void layernorm_pytorch_add_moments_scalar(
    int m0_add,
    float m1_add,
    float m2_add,
    int *m0,
    float *m1,
    float *m2)
{
    const int n = *m0 + m0_add;
    const float c = n == 0 ? 0.0f : (float)m0_add / (float)n;
    const float delta = m1_add - *m1;
    *m1 = fmaf(c, delta, *m1);
    const float delta_term = (delta * delta) * c;
    *m2 += fmaf(delta_term, (float)*m0, m2_add);
    *m0 = n;
}

/*
 * Match the AVX2 ATen BF16 RowwiseMoments provider selected by PyTorch 2.8.
 * It consumes 16 BF16 values per vector, converts its lower and upper halves
 * to two 8-lane FP32 vectors, and combines 16-vector Welford chunks
 * through a binary cascade. The activation arena contains BF16-rounded FP32
 * values, so the same lane grouping can be reproduced without changing ABI.
 */
static inline void layernorm_pytorch_bf16_rowwise_moments_avx2(
    const float *x,
    int n_values,
    float *mean,
    float *variance)
{
    enum { BF16_VEC_SIZE = 16, ACC_VEC_SIZE = 8, CHUNK_SIZE = 16, MAX_DEPTH = 64 };
    const int n_vec = n_values / BF16_VEC_SIZE;
    const int chunks = (n_vec + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int depth = 0;
    for (int v = chunks; v > 1; v = (v + 1) / 2) {
        ++depth;
    }
    if (depth == 0) {
        depth = 1;
    }

    int m0_stack[MAX_DEPTH] = {0};
    __m256 m1_stack[MAX_DEPTH];
    __m256 m2_stack[MAX_DEPTH];
    const __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < depth; ++i) {
        m1_stack[i] = zero;
        m2_stack[i] = zero;
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int remaining = n_vec - chunk * CHUNK_SIZE;
        const int count = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
        __m256 local_m1_lo = zero;
        __m256 local_m1_hi = zero;
        __m256 local_m2_lo = zero;
        __m256 local_m2_hi = zero;
        const float *chunk_x = x + (size_t)chunk * CHUNK_SIZE * BF16_VEC_SIZE;

        for (int j = 0; j < count; ++j) {
            const float c = 1.0f / (float)(j + 1);
            const __m256 c_vec = _mm256_set1_ps(c);
            const float *row = chunk_x + (size_t)j * BF16_VEC_SIZE;
            const __m256 x_lo = _mm256_loadu_ps(row);
            const __m256 x_hi = _mm256_loadu_ps(row + ACC_VEC_SIZE);
            const __m256 delta_lo = _mm256_sub_ps(x_lo, local_m1_lo);
            const __m256 delta_hi = _mm256_sub_ps(x_hi, local_m1_hi);
            local_m1_lo = _mm256_fmadd_ps(delta_lo, c_vec, local_m1_lo);
            local_m1_hi = _mm256_fmadd_ps(delta_hi, c_vec, local_m1_hi);
            local_m2_lo = _mm256_fmadd_ps(
                delta_lo, _mm256_sub_ps(x_lo, local_m1_lo), local_m2_lo);
            local_m2_hi = _mm256_fmadd_ps(
                delta_hi, _mm256_sub_ps(x_hi, local_m1_hi), local_m2_hi);
        }

        layernorm_pytorch_add_moments_vec(
            count, local_m1_lo, local_m2_lo,
            &m0_stack[0], &m1_stack[0], &m2_stack[0]);
        layernorm_pytorch_add_moments_vec(
            count, local_m1_hi, local_m2_hi,
            &m0_stack[0], &m1_stack[0], &m2_stack[0]);

        int mask = chunk + 1;
        for (int level = 1; level < depth && (mask & 1) == 0; ++level) {
            layernorm_pytorch_add_moments_vec(
                m0_stack[level - 1], m1_stack[level - 1], m2_stack[level - 1],
                &m0_stack[level], &m1_stack[level], &m2_stack[level]);
            m0_stack[level - 1] = 0;
            m1_stack[level - 1] = zero;
            m2_stack[level - 1] = zero;
            mask >>= 1;
        }
    }
    for (int level = 1; level < depth; ++level) {
        layernorm_pytorch_add_moments_vec(
            m0_stack[level], m1_stack[level], m2_stack[level],
            &m0_stack[0], &m1_stack[0], &m2_stack[0]);
    }

    float m1_lanes[ACC_VEC_SIZE];
    float m2_lanes[ACC_VEC_SIZE];
    _mm256_storeu_ps(m1_lanes, m1_stack[0]);
    _mm256_storeu_ps(m2_lanes, m2_stack[0]);
    int scalar_count = 0;
    float scalar_m1 = 0.0f;
    float scalar_m2 = 0.0f;
    for (int i = n_vec * BF16_VEC_SIZE; i < n_values; ++i) {
        const float delta = x[i] - scalar_m1;
        ++scalar_count;
        scalar_m1 += delta / (float)scalar_count;
        scalar_m2 += delta * (x[i] - scalar_m1);
    }
    const int lane_count = n_vec * BF16_VEC_SIZE / ACC_VEC_SIZE;
    for (int lane = 0; lane < ACC_VEC_SIZE; ++lane) {
        layernorm_pytorch_add_moments_scalar(
            lane_count, m1_lanes[lane], m2_lanes[lane],
            &scalar_count, &scalar_m1, &scalar_m2);
    }
    *mean = scalar_m1;
    *variance = scalar_m2 / (float)n_values;
}
#endif

void layernorm_pytorch_welford_bf16_storage(const float *input,
                                             const float *gamma,
                                             const float *beta,
                                             float *output,
                                             float *mean_cache,
                                             float *rstd_cache,
                                             int tokens,
                                             int d_model,
                                             float eps)
{
#if !defined(__AVX2__) || !defined(__FMA__)
    (void)input; (void)gamma; (void)beta; (void)output;
    (void)mean_cache; (void)rstd_cache; (void)tokens; (void)d_model; (void)eps;
    abort();
#else
    for (int t = 0; t < tokens; ++t) {
        const float *x = input + (size_t)t * (size_t)d_model;
        float *y = output + (size_t)t * (size_t)d_model;
        float mean;
        float variance;
        layernorm_pytorch_bf16_rowwise_moments_avx2(x, d_model, &mean, &variance);
        const float rstd = 1.0f / sqrtf(variance + eps);
        const float bias = -rstd * mean;
        int i = 0;
        for (; i + 7 < d_model; i += 8) {
            const __m256 x_vec = _mm256_loadu_ps(x + i);
            const __m256 gamma_vec = gamma ? _mm256_loadu_ps(gamma + i) : _mm256_set1_ps(1.0f);
            const __m256 beta_vec = beta ? _mm256_loadu_ps(beta + i) : _mm256_setzero_ps();
            const __m256 normalized = _mm256_fmadd_ps(
                x_vec, _mm256_set1_ps(rstd), _mm256_set1_ps(bias));
            const __m256 transformed = _mm256_fmadd_ps(normalized, gamma_vec, beta_vec);
            float lanes[8];
            _mm256_storeu_ps(lanes, transformed);
            for (int lane = 0; lane < 8; ++lane) {
                y[i + lane] = bf16_to_float(float_to_bf16(lanes[lane]));
            }
        }
        for (; i < d_model; ++i) {
            const float gamma_v = gamma ? gamma[i] : 1.0f;
            const float beta_v = beta ? beta[i] : 0.0f;
            const float value = fmaf(fmaf(x[i], rstd, bias), gamma_v, beta_v);
            y[i] = bf16_to_float(float_to_bf16(value));
        }
        if (mean_cache) {
            mean_cache[t] = mean;
        }
        if (rstd_cache) {
            rstd_cache[t] = rstd;
        }
    }
#endif
}

// LayerNorm backward kernel (model-agnostic), adapted from C-Transformer's
// backward_layernorm. Computes gradients w.r.t. input, gamma, and beta.
void layernorm_backward_kernel(const float *d_output,  // [T×aligned_D]
                               const float *input,     // [T×aligned_D]
                               const float *gamma,     // [D]
                               const float *mean,      // [T]
                               const float *rstd,      // [T]
                               float *d_input,         // [T×aligned_D]
                               float *d_gamma,         // [D] (accumulated)
                               float *d_beta,          // [D] (accumulated)
                               int tokens, int d_model, int aligned_embed_dim)
{
    int T = tokens;
    int D = d_model;
    int aligned_D = aligned_embed_dim;

    // Per-token input gradients
    for (int t = 0; t < T; ++t) {
        float mean_t = mean[t];
        float rstd_t = rstd[t];

        float d_y_gamma_sum = 0.0f;
        float d_y_gamma_xhat_sum = 0.0f;

        // First pass: compute sums
        for (int d = 0; d < D; ++d) {
            float x = input[t * aligned_D + d];
            float x_hat = (x - mean_t) * rstd_t;
            float d_y = d_output[t * aligned_D + d];
            float d_y_gamma = d_y * gamma[d];

            d_y_gamma_sum += d_y_gamma;
            d_y_gamma_xhat_sum += d_y_gamma * x_hat;
        }

        // Second pass: compute input gradients
        float scale = rstd_t / (float)D;
        for (int d = 0; d < D; ++d) {
            float x = input[t * aligned_D + d];
            float x_hat = (x - mean_t) * rstd_t;
            float d_y = d_output[t * aligned_D + d];

            d_input[t * aligned_D + d] =
                scale * ((float)D * d_y * gamma[d] - d_y_gamma_sum - x_hat * d_y_gamma_xhat_sum);
        }

        // Zero padding for aligned dimension beyond D
        for (int d = D; d < aligned_D; ++d) {
            d_input[t * aligned_D + d] = 0.0f;
        }
    }

    // Parameter gradients (gamma, beta)
    for (int d = 0; d < D; ++d) {
        float gamma_grad = 0.0f;
        float beta_grad = 0.0f;

        for (int t = 0; t < T; ++t) {
            float x = input[t * aligned_D + d];
            float x_hat = (x - mean[t]) * rstd[t];
            float d_y = d_output[t * aligned_D + d];

            gamma_grad += d_y * x_hat;
            beta_grad += d_y;
        }

        d_gamma[d] += gamma_grad;
        d_beta[d] += beta_grad;
    }
}
