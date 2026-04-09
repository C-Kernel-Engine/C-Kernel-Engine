/**
 * @file ck_parallel_train.c
 * @brief Thread-pool dispatch wrapper for FP32 training GEMM.
 *
 * The training runtime currently emits many gemm_blocked_serial calls.
 * This wrapper keeps kernel math unchanged and only parallelizes dispatch.
 *
 * OpenMP removal note:
 * Training backward used to fall back into legacy OpenMP GEMM paths for T>1
 * micro-batches, which mixed libiomp barriers with the CK threadpool and
 * showed up as fork/barrier overhead in profiling. Keep the hot training
 * path on the CK threadpool here, and leave older kernel-level OpenMP paths
 * as compatibility fallbacks until they are fully retired.
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

#if defined(__AVX__) && !defined(__AVX512F__)
static inline float ck_train_hsum256_ps(__m256 v) {
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

static void ck_train_gemm_nt_compute_rows(const float *A,
                                          const float *B,
                                          const float *bias,
                                          float *C,
                                          int row_start,
                                          int row_end,
                                          int N,
                                          int K) {
    if (!A || !B || !C || row_start >= row_end || N <= 0 || K <= 0) {
        return;
    }

    for (int i = row_start; i < row_end; ++i) {
        const float *a_row = A + (size_t)i * (size_t)K;
        float *c_row = C + (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) {
            const float *b_row = B + (size_t)j * (size_t)K;
            float sum = bias ? bias[j] : 0.0f;
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            int k = 0;
            for (; k <= K - 16; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(a_row + k);
                __m512 b_vec = _mm512_loadu_ps(b_row + k);
                acc = _mm512_fmadd_ps(a_vec, b_vec, acc);
            }
            sum += _mm512_reduce_add_ps(acc);
            for (; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
#elif defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(a_row + k);
                __m256 b_vec = _mm256_loadu_ps(b_row + k);
#if defined(__FMA__)
                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a_vec, b_vec));
#endif
            }
            sum += ck_train_hsum256_ps(acc);
            for (; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
#else
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
#endif
            c_row[j] = sum;
        }
    }
}

static int ck_train_pick_active_threads(int nth, size_t work_items, size_t min_chunk)
{
    if (nth <= 1 || work_items == 0 || min_chunk == 0) {
        return 1;
    }
    size_t active = (work_items + min_chunk - 1u) / min_chunk;
    if (active < 1u) {
        active = 1u;
    }
    if (active > (size_t)nth) {
        active = (size_t)nth;
    }
    return (int)active;
}

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

    ck_train_gemm_nt_compute_rows(a->A, a->B, a->bias, a->C, m0, m1, a->N, a->K);
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
    int active_nth = nth;
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
        /* Row split for prefill/train batches with coarser chunks. */
        active_nth = M / 2;
        if (active_nth > nth) {
            active_nth = nth;
        }
        if (active_nth <= 1) {
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

    ck_threadpool_dispatch_n(pool, active_nth, ck_train_gemm_work, &args);
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

typedef struct {
    const float *d_output;
    const float *input;
    const float *W;
    float *d_input;
    float *d_W;
    float *d_b;
    int T;
    int aligned_in;
    int aligned_out;
} ck_train_gemm_backward_args_t;

static void ck_train_gemm_nn_compute_rows(const float *A,
                                          const float *B,
                                          const float *bias,
                                          float *C,
                                          int row_start,
                                          int row_end,
                                          int N,
                                          int K) {
    if (!A || !B || !C || row_start >= row_end || N <= 0 || K <= 0) {
        return;
    }

    for (int i = row_start; i < row_end; ++i) {
        const float *a_row = A + (size_t)i * (size_t)K;
        float *c_row = C + (size_t)i * (size_t)N;
#if defined(__AVX512F__)
        int j = 0;
        for (; j <= N - 16; j += 16) {
            __m512 sum = bias ? _mm512_loadu_ps(bias + j) : _mm512_setzero_ps();
            for (int k = 0; k < K; ++k) {
                __m512 av = _mm512_set1_ps(a_row[k]);
                __m512 bv = _mm512_loadu_ps(B + (size_t)k * (size_t)N + (size_t)j);
                sum = _mm512_fmadd_ps(av, bv, sum);
            }
            _mm512_storeu_ps(c_row + j, sum);
        }
        for (; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * B[(size_t)k * (size_t)N + (size_t)j];
            }
            c_row[j] = sum;
        }
#elif defined(__AVX2__)
        int j = 0;
        for (; j <= N - 8; j += 8) {
            __m256 sum = bias ? _mm256_loadu_ps(bias + j) : _mm256_setzero_ps();
            for (int k = 0; k < K; ++k) {
                __m256 av = _mm256_set1_ps(a_row[k]);
                __m256 bv = _mm256_loadu_ps(B + (size_t)k * (size_t)N + (size_t)j);
#if defined(__FMA__)
                sum = _mm256_fmadd_ps(av, bv, sum);
#else
                sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
#endif
            }
            _mm256_storeu_ps(c_row + j, sum);
        }
        for (; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * B[(size_t)k * (size_t)N + (size_t)j];
            }
            c_row[j] = sum;
        }
#else
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * B[(size_t)k * (size_t)N + (size_t)j];
            }
            c_row[j] = sum;
        }
#endif
    }
}

static void ck_train_gemm_tn_compute_rows(const float *A,
                                          const float *B,
                                          const float *bias,
                                          float *C,
                                          int row_start,
                                          int row_end,
                                          int M,
                                          int N,
                                          int K) {
    if (!A || !B || !C || row_start >= row_end || M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    for (int i = row_start; i < row_end; ++i) {
        float *c_row = C + (size_t)i * (size_t)N;
#if defined(__AVX512F__)
        int j = 0;
        for (; j <= N - 16; j += 16) {
            __m512 sum = bias ? _mm512_loadu_ps(bias + j) : _mm512_setzero_ps();
            for (int k = 0; k < K; ++k) {
                __m512 av = _mm512_set1_ps(A[(size_t)k * (size_t)M + (size_t)i]);
                __m512 bv = _mm512_loadu_ps(B + (size_t)k * (size_t)N + (size_t)j);
                sum = _mm512_fmadd_ps(av, bv, sum);
            }
            _mm512_storeu_ps(c_row + j, sum);
        }
        for (; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[(size_t)k * (size_t)M + (size_t)i] *
                       B[(size_t)k * (size_t)N + (size_t)j];
            }
            c_row[j] = sum;
        }
#elif defined(__AVX2__)
        int j = 0;
        for (; j <= N - 8; j += 8) {
            __m256 sum = bias ? _mm256_loadu_ps(bias + j) : _mm256_setzero_ps();
            for (int k = 0; k < K; ++k) {
                __m256 av = _mm256_set1_ps(A[(size_t)k * (size_t)M + (size_t)i]);
                __m256 bv = _mm256_loadu_ps(B + (size_t)k * (size_t)N + (size_t)j);
#if defined(__FMA__)
                sum = _mm256_fmadd_ps(av, bv, sum);
#else
                sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
#endif
            }
            _mm256_storeu_ps(c_row + j, sum);
        }
        for (; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[(size_t)k * (size_t)M + (size_t)i] *
                       B[(size_t)k * (size_t)N + (size_t)j];
            }
            c_row[j] = sum;
        }
#else
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[(size_t)k * (size_t)M + (size_t)i] *
                       B[(size_t)k * (size_t)N + (size_t)j];
            }
            c_row[j] = sum;
        }
#endif
    }
}

static void ck_train_bias_reduce_compute_range(const float *d_output,
                                               float *d_bias,
                                               int T,
                                               int out_start,
                                               int out_end,
                                               int aligned_out) {
    if (!d_output || !d_bias || T <= 0 || out_start >= out_end || aligned_out <= 0) {
        return;
    }

    for (int out_idx = out_start; out_idx < out_end; ++out_idx) {
        float bias_grad = 0.0f;
        for (int t = 0; t < T; ++t) {
            bias_grad += d_output[(size_t)t * (size_t)aligned_out + (size_t)out_idx];
        }
        d_bias[out_idx] += bias_grad;
    }
}

static void ck_train_outer_t1_compute_range(const float *d_output,
                                            const float *input,
                                            float *d_W,
                                            float *d_b,
                                            int out_start,
                                            int out_end,
                                            int aligned_in) {
    if (!d_output || !input || !d_W || out_start >= out_end || aligned_in <= 0) {
        return;
    }

    for (int out_idx = out_start; out_idx < out_end; ++out_idx) {
        float g = d_output[out_idx];
        if (d_b) {
            d_b[out_idx] += g;
        }
        float *dw_row = d_W + (size_t)out_idx * (size_t)aligned_in;
        for (int j = 0; j < aligned_in; ++j) {
            dw_row[j] = g * input[j];
        }
    }
}

static void ck_train_gemm_backward_serial(const float *d_output,
                                          const float *input,
                                          const float *W,
                                          float *d_input,
                                          float *d_W,
                                          float *d_b,
                                          int T,
                                          int aligned_in,
                                          int aligned_out) {
    ck_train_gemm_nn_compute_rows(d_output, W, NULL, d_input, 0, T, aligned_in, aligned_out);
    ck_train_gemm_tn_compute_rows(d_output, input, NULL, d_W, 0, aligned_out, aligned_out, aligned_in, T);
    if (d_b) {
        ck_train_bias_reduce_compute_range(d_output, d_b, T, 0, aligned_out, aligned_out);
    }
}

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

    ck_train_gemm_nn_compute_rows(a->A, a->B, a->bias, a->C, m0, m1, a->N, a->K);
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

    ck_train_outer_t1_compute_range(a->d_output, a->input, a->d_W, a->d_b, n0, n1, a->aligned_in);
}

static void ck_train_gemm_backward_work(int ith, int nth, void *argp) {
    ck_train_gemm_backward_args_t *a = (ck_train_gemm_backward_args_t *)argp;
    if (!a || !a->d_output || !a->input || !a->W || !a->d_input || !a->d_W ||
        a->T <= 0 || a->aligned_in <= 0 || a->aligned_out <= 0) {
        return;
    }

    int dt = (a->T + nth - 1) / nth;
    int t0 = dt * ith;
    int t1 = t0 + dt;
    if (t1 > a->T) {
        t1 = a->T;
    }
    if (t0 < t1) {
        ck_train_gemm_nn_compute_rows(
            a->d_output,
            a->W,
            NULL,
            a->d_input,
            t0,
            t1,
            a->aligned_in,
            a->aligned_out);
    }

    int dn = (a->aligned_out + nth - 1) / nth;
    int n0 = dn * ith;
    int n1 = n0 + dn;
    if (n1 > a->aligned_out) {
        n1 = a->aligned_out;
    }
    if (n0 < n1) {
        ck_train_gemm_tn_compute_rows(
            a->d_output,
            a->input,
            NULL,
            a->d_W,
            n0,
            n1,
            a->aligned_out,
            a->aligned_in,
            a->T);
        if (a->d_b) {
            ck_train_bias_reduce_compute_range(
                a->d_output,
                a->d_b,
                a->T,
                n0,
                n1,
                a->aligned_out);
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
    if (!d_output || !input || !W || !d_input || !d_W) {
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
        ck_train_gemm_backward_serial(d_output, input, W, d_input, d_W, d_b, T, aligned_in, aligned_out);
        return;
    }

    /* T=1 is the dominant runtime shape in generated train microsteps.
     * Use vectorized d_input and parallel outer-product for dW/db. */
    if (T == 1) {
        gemm_nn_simd(d_output, W, NULL, d_input, 1, aligned_in, aligned_out);

        const size_t outer_work = (size_t)aligned_out * (size_t)aligned_in;
        if (aligned_out < nth * 2 || aligned_in < 64 || outer_work < (size_t)524288) {
            ck_train_outer_t1_compute_range(d_output, input, d_W, d_b, 0, aligned_out, aligned_in);
        } else {
            const int active_outer = ck_train_pick_active_threads(nth, (size_t)aligned_out, (size_t)64);
            ck_train_outer_t1_args_t t1_args = {
                .d_output = d_output,
                .input = input,
                .d_W = d_W,
                .d_b = d_b,
                .aligned_in = aligned_in,
                .aligned_out = aligned_out,
            };
            if (active_outer <= 1) {
                ck_train_outer_t1_compute_range(d_output, input, d_W, d_b, 0, aligned_out, aligned_in);
            } else {
                ck_threadpool_dispatch_n(pool, active_outer, ck_train_outer_t1_work, &t1_args);
            }
        }
        return;
    }

    const size_t nn_work = (size_t)T * (size_t)aligned_in * (size_t)aligned_out;
    const size_t tn_work = (size_t)aligned_out * (size_t)aligned_in * (size_t)T;
    if ((T < 2 && aligned_out < nth * 2) || (nn_work < (size_t)131072 && tn_work < (size_t)131072)) {
        ck_train_gemm_backward_serial(d_output, input, W, d_input, d_W, d_b, T, aligned_in, aligned_out);
        return;
    }

    ck_train_gemm_backward_args_t bw_args = {
        .d_output = d_output,
        .input = input,
        .W = W,
        .d_input = d_input,
        .d_W = d_W,
        .d_b = d_b,
        .T = T,
        .aligned_in = aligned_in,
        .aligned_out = aligned_out,
    };
    {
        const size_t bw_rows = (size_t)aligned_out > (size_t)T ? (size_t)aligned_out : (size_t)T;
        const int active_bw = ck_train_pick_active_threads(nth, bw_rows, (size_t)64);
        if (active_bw <= 1) {
            ck_train_gemm_backward_serial(d_output, input, W, d_input, d_W, d_b, T, aligned_in, aligned_out);
        } else {
            ck_threadpool_dispatch_n(pool, active_bw, ck_train_gemm_backward_work, &bw_args);
        }
    }
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
    if (!d_output || !input || !W || !d_input || !d_W) {
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
        ck_train_gemm_backward_serial(d_output, input, W, d_input, d_W, d_b, T, aligned_in, aligned_out);
        return;
    }

    if (T == 1) {
        const size_t nn_work = (size_t)aligned_in * (size_t)aligned_out;
        if (aligned_in >= nth * 32 && nn_work >= (size_t)131072) {
            const int active_nn = ck_train_pick_active_threads(nth, (size_t)aligned_in, (size_t)64);
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
            if (active_nn <= 1) {
                gemm_nn_simd(d_output, W, NULL, d_input, 1, aligned_in, aligned_out);
            } else {
                ck_threadpool_dispatch_n(pool, active_nn, ck_train_gemm_nn_work, &nn_args);
            }
        } else {
            gemm_nn_simd(d_output, W, NULL, d_input, 1, aligned_in, aligned_out);
        }

        const size_t outer_work = (size_t)aligned_out * (size_t)aligned_in;
        if (aligned_out >= nth && outer_work >= (size_t)131072) {
            const int active_outer = ck_train_pick_active_threads(nth, (size_t)aligned_out, (size_t)64);
            ck_train_outer_t1_args_t t1_args = {
                .d_output = d_output,
                .input = input,
                .d_W = d_W,
                .d_b = d_b,
                .aligned_in = aligned_in,
                .aligned_out = aligned_out,
            };
            if (active_outer <= 1) {
                ck_train_outer_t1_compute_range(d_output, input, d_W, d_b, 0, aligned_out, aligned_in);
            } else {
                ck_threadpool_dispatch_n(pool, active_outer, ck_train_outer_t1_work, &t1_args);
            }
        } else {
            ck_train_outer_t1_compute_range(d_output, input, d_W, d_b, 0, aligned_out, aligned_in);
        }
        return;
    }

    const size_t nn_work = (size_t)T * (size_t)aligned_in * (size_t)aligned_out;
    const size_t tn_work = (size_t)aligned_out * (size_t)aligned_in * (size_t)T;
    if ((T < nth && aligned_out < nth) || (nn_work < (size_t)131072 && tn_work < (size_t)131072)) {
        ck_train_gemm_backward_serial(d_output, input, W, d_input, d_W, d_b, T, aligned_in, aligned_out);
        return;
    }

    ck_train_gemm_backward_args_t bw_args = {
        .d_output = d_output,
        .input = input,
        .W = W,
        .d_input = d_input,
        .d_W = d_W,
        .d_b = d_b,
        .T = T,
        .aligned_in = aligned_in,
        .aligned_out = aligned_out,
    };
    {
        const size_t bw_rows = (size_t)aligned_out > (size_t)T ? (size_t)aligned_out : (size_t)T;
        const int active_bw = ck_train_pick_active_threads(nth, bw_rows, (size_t)64);
        if (active_bw <= 1) {
            ck_train_gemm_backward_serial(d_output, input, W, d_input, d_W, d_b, T, aligned_in, aligned_out);
        } else {
            ck_threadpool_dispatch_n(pool, active_bw, ck_train_gemm_backward_work, &bw_args);
        }
    }
}
