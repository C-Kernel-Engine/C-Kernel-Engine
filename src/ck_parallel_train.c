/**
 * @file ck_parallel_train.c
 * @brief Thread-pool dispatch wrapper for FP32 training GEMM.
 *
 * The training runtime currently emits many gemm_blocked_serial calls.
 * This wrapper keeps kernel math unchanged and only parallelizes dispatch.
 */

#include "ckernel_engine.h"
#include "ck_threadpool.h"

#include <stddef.h>
#if defined(__AVX2__) || defined(__AVX__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

typedef struct {
    const float *A;
    const float *B;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int split_n;
} ck_train_gemm_args_t;

static void ck_train_gemm_work(int ith, int nth, void *argp) {
    ck_train_gemm_args_t *a = (ck_train_gemm_args_t *)argp;
    if (!a || a->M <= 0 || a->N <= 0 || a->K <= 0) {
        return;
    }

    if (a->split_n) {
        /* Decode/microbatch path (M=1): split output columns. */
        int dn = (a->N + nth - 1) / nth;
        int n0 = dn * ith;
        int n1 = n0 + dn;
        if (n0 >= a->N) {
            return;
        }
        if (n1 > a->N) {
            n1 = a->N;
        }
        const int n_chunk = n1 - n0;
        if (n_chunk <= 0) {
            return;
        }

        const float *B_chunk = a->B + (size_t)n0 * (size_t)a->K;
        const float *bias_chunk = a->bias ? (a->bias + n0) : NULL;
        float *C_chunk = a->C + n0; /* M=1 layout */

        gemm_blocked_serial(a->A, B_chunk, bias_chunk, C_chunk, 1, n_chunk, a->K);
        return;
    }

    /* Prefill/train path (M>1): split rows. */
    int dm = (a->M + nth - 1) / nth;
    int m0 = dm * ith;
    int m1 = m0 + dm;
    if (m0 >= a->M) {
        return;
    }
    if (m1 > a->M) {
        m1 = a->M;
    }
    const int m_chunk = m1 - m0;
    if (m_chunk <= 0) {
        return;
    }

    const float *A_chunk = a->A + (size_t)m0 * (size_t)a->K;
    float *C_chunk = a->C + (size_t)m0 * (size_t)a->N;
    gemm_blocked_serial(A_chunk, a->B, a->bias, C_chunk, m_chunk, a->N, a->K);
}

void gemm_blocked_serial_train_parallel_dispatch(const float *A,
                                                 const float *B,
                                                 const float *bias,
                                                 float *C,
                                                 int M,
                                                 int N,
                                                 int K) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    const int nth = pool ? ck_threadpool_n_threads(pool) : 1;

    /* Keep small shapes on serial path to avoid dispatch overhead. */
    const size_t work = (size_t)M * (size_t)N * (size_t)K;
    if (!pool || nth <= 1 || work < (size_t)131072) {
        gemm_blocked_serial(A, B, bias, C, M, N, K);
        return;
    }

    int split_n = 0;
    if (M == 1) {
        /* Tensor-parallel decode-style split only when each worker has enough columns. */
        const int cols_per_worker = (N + nth - 1) / nth;
        if (cols_per_worker >= 256 && K >= 512 && work >= (size_t)2097152) {
            split_n = 1;
        } else {
            gemm_blocked_serial(A, B, bias, C, M, N, K);
            return;
        }
    } else {
        /* Row split for prefill/train batches. */
        if (M < nth * 2) {
            gemm_blocked_serial(A, B, bias, C, M, N, K);
            return;
        }
    }

    ck_train_gemm_args_t args = {
        .A = A,
        .B = B,
        .bias = bias,
        .C = C,
        .M = M,
        .N = N,
        .K = K,
        .split_n = split_n,
    };

    ck_threadpool_dispatch(pool, ck_train_gemm_work, &args);
}

typedef struct {
    const float *A;
    const float *B;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
    int split_n;
} ck_train_gemm_nn_args_t;

typedef struct {
    const float *A;
    const float *B;
    const float *bias;
    float *C;
    int M;
    int N;
    int K;
} ck_train_gemm_tn_args_t;

typedef struct {
    const float *d_output;
    float *d_bias;
    int T;
    int aligned_out;
} ck_train_bias_reduce_args_t;

typedef struct {
    const float *d_output;
    const float *input;
    float *d_W;
    float *d_b;
    int aligned_in;
    int aligned_out;
} ck_train_outer_t1_args_t;

static void ck_train_gemm_nn_work(int ith, int nth, void *argp) {
    ck_train_gemm_nn_args_t *a = (ck_train_gemm_nn_args_t *)argp;
    if (!a || a->M <= 0 || a->N <= 0 || a->K <= 0) {
        return;
    }

    if (a->split_n) {
        int dn = (a->N + nth - 1) / nth;
        int n0 = dn * ith;
        int n1 = n0 + dn;
        if (n0 >= a->N) {
            return;
        }
        if (n1 > a->N) {
            n1 = a->N;
        }
#if defined(__AVX2__)
        int j = n0;
        for (; j <= n1 - 8; j += 8) {
            __m256 sum = a->bias ? _mm256_loadu_ps(a->bias + j) : _mm256_setzero_ps();
            for (int k = 0; k < a->K; ++k) {
                __m256 av = _mm256_set1_ps(a->A[k]);
                __m256 bv = _mm256_loadu_ps(a->B + (size_t)k * (size_t)a->N + (size_t)j);
#if defined(__FMA__)
                sum = _mm256_fmadd_ps(av, bv, sum);
#else
                sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
#endif
            }
            _mm256_storeu_ps(a->C + j, sum);
        }
        for (; j < n1; ++j) {
            float sum = a->bias ? a->bias[j] : 0.0f;
            for (int k = 0; k < a->K; ++k) {
                sum += a->A[k] * a->B[(size_t)k * (size_t)a->N + (size_t)j];
            }
            a->C[j] = sum;
        }
#else
        for (int j = n0; j < n1; ++j) {
            float sum = a->bias ? a->bias[j] : 0.0f;
            for (int k = 0; k < a->K; ++k) {
                sum += a->A[k] * a->B[(size_t)k * (size_t)a->N + (size_t)j];
            }
            a->C[j] = sum;
        }
#endif
        return;
    }

    int dm = (a->M + nth - 1) / nth;
    int m0 = dm * ith;
    int m1 = m0 + dm;
    if (m0 >= a->M) {
        return;
    }
    if (m1 > a->M) {
        m1 = a->M;
    }

    const int m_chunk = m1 - m0;
    if (m_chunk <= 0) {
        return;
    }

    const float *A_chunk = a->A + (size_t)m0 * (size_t)a->K;
    float *C_chunk = a->C + (size_t)m0 * (size_t)a->N;
    gemm_nn_simd(A_chunk, a->B, a->bias, C_chunk, m_chunk, a->N, a->K);
}

static void ck_train_gemm_tn_work(int ith, int nth, void *argp) {
    ck_train_gemm_tn_args_t *a = (ck_train_gemm_tn_args_t *)argp;
    if (!a || a->M <= 0 || a->N <= 0 || a->K <= 0) {
        return;
    }

    int dm = (a->M + nth - 1) / nth;
    int m0 = dm * ith;
    int m1 = m0 + dm;
    if (m0 >= a->M) {
        return;
    }
    if (m1 > a->M) {
        m1 = a->M;
    }

    for (int i = m0; i < m1; ++i) {
        float *c_row = a->C + (size_t)i * (size_t)a->N;
#if defined(__AVX2__)
        int j = 0;
        for (; j <= a->N - 8; j += 8) {
            __m256 sum = a->bias ? _mm256_loadu_ps(a->bias + j) : _mm256_setzero_ps();
            for (int k = 0; k < a->K; ++k) {
                const float aval = a->A[(size_t)k * (size_t)a->M + (size_t)i];
                __m256 av = _mm256_set1_ps(aval);
                __m256 bv = _mm256_loadu_ps(a->B + (size_t)k * (size_t)a->N + (size_t)j);
#if defined(__FMA__)
                sum = _mm256_fmadd_ps(av, bv, sum);
#else
                sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
#endif
            }
            _mm256_storeu_ps(c_row + j, sum);
        }
        for (; j < a->N; ++j) {
            float sum = a->bias ? a->bias[j] : 0.0f;
            for (int k = 0; k < a->K; ++k) {
                sum += a->A[(size_t)k * (size_t)a->M + (size_t)i] *
                       a->B[(size_t)k * (size_t)a->N + (size_t)j];
            }
            c_row[j] = sum;
        }
#else
        for (int j = 0; j < a->N; ++j) {
            float sum = a->bias ? a->bias[j] : 0.0f;
            for (int k = 0; k < a->K; ++k) {
                sum += a->A[(size_t)k * (size_t)a->M + (size_t)i] *
                       a->B[(size_t)k * (size_t)a->N + (size_t)j];
            }
            c_row[j] = sum;
        }
#endif
    }
}

static void ck_train_bias_reduce_work(int ith, int nth, void *argp) {
    ck_train_bias_reduce_args_t *a = (ck_train_bias_reduce_args_t *)argp;
    if (!a || a->T <= 0 || a->aligned_out <= 0) {
        return;
    }

    int dn = (a->aligned_out + nth - 1) / nth;
    int n0 = dn * ith;
    int n1 = n0 + dn;
    if (n0 >= a->aligned_out) {
        return;
    }
    if (n1 > a->aligned_out) {
        n1 = a->aligned_out;
    }

    for (int out_idx = n0; out_idx < n1; ++out_idx) {
        float bias_grad = 0.0f;
        for (int t = 0; t < a->T; ++t) {
            bias_grad += a->d_output[(size_t)t * (size_t)a->aligned_out + (size_t)out_idx];
        }
        a->d_bias[out_idx] += bias_grad;
    }
}


static void ck_train_outer_t1_work(int ith, int nth, void *argp) {
    ck_train_outer_t1_args_t *a = (ck_train_outer_t1_args_t *)argp;
    if (!a || a->aligned_in <= 0 || a->aligned_out <= 0) {
        return;
    }

    int dn = (a->aligned_out + nth - 1) / nth;
    int n0 = dn * ith;
    int n1 = n0 + dn;
    if (n0 >= a->aligned_out) {
        return;
    }
    if (n1 > a->aligned_out) {
        n1 = a->aligned_out;
    }

    for (int out_idx = n0; out_idx < n1; ++out_idx) {
        float g = a->d_output[out_idx];
        a->d_b[out_idx] += g;
        float *dw_row = a->d_W + (size_t)out_idx * (size_t)a->aligned_in;
        for (int j = 0; j < a->aligned_in; ++j) {
            dw_row[j] = g * a->input[j];
        }
    }
}

void gemm_backward_f32_train_parallel_dispatch(const float *d_output,
                                               const float *input,
                                               const float *W,
                                               float *d_input,
                                               float *d_W,
                                               float *d_b,
                                               int T,
                                               int aligned_in,
                                               int aligned_out,
                                               int num_threads) {
    if (!d_output || !input || !W || !d_input || !d_W || !d_b) {
        return;
    }
    if (T <= 0 || aligned_in <= 0 || aligned_out <= 0) {
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;
    if (num_threads > 0 && num_threads < nth) {
        nth = num_threads;
    }

    if (!pool || nth <= 1) {
        fc2_backward_kernel(d_output,
                            input,
                            W,
                            d_input,
                            d_W,
                            d_b,
                            T,
                            aligned_in,
                            aligned_out,
                            1);
        return;
    }

    /* T=1 is the dominant runtime shape in generated train microsteps.
     * Use vectorized d_input and parallel outer-product for dW/db. */
    if (T == 1) {
        gemm_nn_simd(d_output, W, NULL, d_input, 1, aligned_in, aligned_out);

        const size_t outer_work = (size_t)aligned_out * (size_t)aligned_in;
        if (aligned_out < nth * 2 || aligned_in < 64 || outer_work < (size_t)524288) {
            for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
                float g = d_output[out_idx];
                d_b[out_idx] += g;
                float *dw_row = d_W + (size_t)out_idx * (size_t)aligned_in;
                for (int j = 0; j < aligned_in; ++j) {
                    dw_row[j] = g * input[j];
                }
            }
        } else {
            ck_train_outer_t1_args_t t1_args = {
                .d_output = d_output,
                .input = input,
                .d_W = d_W,
                .d_b = d_b,
                .aligned_in = aligned_in,
                .aligned_out = aligned_out,
            };
            ck_threadpool_dispatch(pool, ck_train_outer_t1_work, &t1_args);
        }
        return;
    }

    /*
     * Stability guard for T>1 train micro-batches:
     * use the proven FC2 reference backward path for d_input/dW/db.
     * The custom threaded TN worker path is only kept for T==1 currently.
     */
    fc2_backward_kernel(d_output,
                        input,
                        W,
                        d_input,
                        d_W,
                        d_b,
                        T,
                        aligned_in,
                        aligned_out,
                        1);
    return;

    ck_train_gemm_nn_args_t nn_args = {
        .A = d_output,
        .B = W,
        .bias = NULL,
        .C = d_input,
        .M = T,
        .N = aligned_in,
        .K = aligned_out,
        .split_n = 0,
    };
    ck_threadpool_dispatch(pool, ck_train_gemm_nn_work, &nn_args);

    ck_train_gemm_tn_args_t tn_args = {
        .A = d_output,
        .B = input,
        .bias = NULL,
        .C = d_W,
        .M = aligned_out,
        .N = aligned_in,
        .K = T,
    };
    ck_threadpool_dispatch(pool, ck_train_gemm_tn_work, &tn_args);

    ck_train_bias_reduce_args_t b_args = {
        .d_output = d_output,
        .d_bias = d_b,
        .T = T,
        .aligned_out = aligned_out,
    };
    ck_threadpool_dispatch(pool, ck_train_bias_reduce_work, &b_args);
}

/*
 * v2 training backward wrapper:
 * - keeps kernel math contract untouched
 * - uses more aggressive threadpool partitioning for T=1 microstep shapes
 */
void gemm_backward_f32_train_parallel_dispatch_v2(const float *d_output,
                                                  const float *input,
                                                  const float *W,
                                                  float *d_input,
                                                  float *d_W,
                                                  float *d_b,
                                                  int T,
                                                  int aligned_in,
                                                  int aligned_out,
                                                  int num_threads) {
    if (!d_output || !input || !W || !d_input || !d_W || !d_b) {
        return;
    }
    if (T <= 0 || aligned_in <= 0 || aligned_out <= 0) {
        return;
    }

    ck_threadpool_t *pool = ck_threadpool_global();
    int nth = pool ? ck_threadpool_n_threads(pool) : 1;
    if (num_threads > 0 && num_threads < nth) {
        nth = num_threads;
    }

    if (!pool || nth <= 1) {
        gemm_backward_f32_train_parallel_dispatch(
            d_output, input, W, d_input, d_W, d_b, T, aligned_in, aligned_out, 1);
        return;
    }

    if (T == 1) {
        const size_t nn_work = (size_t)aligned_in * (size_t)aligned_out;
        if (aligned_in >= nth * 32 && nn_work >= (size_t)131072) {
            ck_train_gemm_nn_args_t nn_args = {
                .A = d_output,
                .B = W,
                .bias = NULL,
                .C = d_input,
                .M = 1,
                .N = aligned_in,
                .K = aligned_out,
                .split_n = 1,
            };
            ck_threadpool_dispatch(pool, ck_train_gemm_nn_work, &nn_args);
        } else {
            gemm_nn_simd(d_output, W, NULL, d_input, 1, aligned_in, aligned_out);
        }

        const size_t outer_work = (size_t)aligned_out * (size_t)aligned_in;
        if (aligned_out >= nth && outer_work >= (size_t)131072) {
            ck_train_outer_t1_args_t t1_args = {
                .d_output = d_output,
                .input = input,
                .d_W = d_W,
                .d_b = d_b,
                .aligned_in = aligned_in,
                .aligned_out = aligned_out,
            };
            ck_threadpool_dispatch(pool, ck_train_outer_t1_work, &t1_args);
        } else {
            for (int out_idx = 0; out_idx < aligned_out; ++out_idx) {
                float g = d_output[out_idx];
                d_b[out_idx] += g;
                float *dw_row = d_W + (size_t)out_idx * (size_t)aligned_in;
                for (int j = 0; j < aligned_in; ++j) {
                    dw_row[j] = g * input[j];
                }
            }
        }
        return;
    }

    /*
     * Stability-first fallback for T>1:
     * keep v2 fast-threaded path for T==1 only, and route larger T to
     * the reference backward kernel while we harden the TN threaded worker.
     */
    fc2_backward_kernel(d_output,
                        input,
                        W,
                        d_input,
                        d_W,
                        d_b,
                        T,
                        aligned_in,
                        aligned_out,
                        1);
    return;

    ck_train_gemm_nn_args_t nn_args = {
        .A = d_output,
        .B = W,
        .bias = NULL,
        .C = d_input,
        .M = T,
        .N = aligned_in,
        .K = aligned_out,
        .split_n = 0,
    };
    ck_threadpool_dispatch(pool, ck_train_gemm_nn_work, &nn_args);

    ck_train_gemm_tn_args_t tn_args = {
        .A = d_output,
        .B = input,
        .bias = NULL,
        .C = d_W,
        .M = aligned_out,
        .N = aligned_in,
        .K = T,
    };
    ck_threadpool_dispatch(pool, ck_train_gemm_tn_work, &tn_args);

    ck_train_bias_reduce_args_t b_args = {
        .d_output = d_output,
        .d_bias = d_b,
        .T = T,
        .aligned_out = aligned_out,
    };
    ck_threadpool_dispatch(pool, ck_train_bias_reduce_work, &b_args);
}
