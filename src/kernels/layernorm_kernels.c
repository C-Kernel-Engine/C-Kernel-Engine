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

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif
#include <math.h>

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
    for (int t = 0; t < tokens; ++t) {
        const float *in_ptr = input + t * d_model;
        float *out_ptr = output + t * d_model;

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
    }
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
