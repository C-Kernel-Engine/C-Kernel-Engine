#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "../include/ckernel_quant.h"

// Forward declaration of kernels (usually in .c files)
void gemv_q4_k_q8_k_ref(float *y, const void *W, const void *x_q8, int M, int K);
void gemv_q4_k_q8_k_sse(float *y, const void *W, const void *x_q8, int M, int K);

// Helper to fill Q4_K with random data
void fill_random_q4k(void *buffer, int k) {
    int num_blocks = k / QK_K;
    block_q4_K *blocks = (block_q4_K *)buffer;
    for (int i = 0; i < num_blocks; ++i) {
        blocks[i].d = (float)(rand() % 100) / 1000.0f; // float16
        blocks[i].dmin = (float)(rand() % 100) / 1000.0f; // float16
        for (int j = 0; j < 12; ++j) blocks[i].scales[j] = rand() % 256;
        for (int j = 0; j < QK_K / 2; ++j) blocks[i].qs[j] = rand() % 256;
    }
}

// Helper to fill Q8_K with random data
void fill_random_q8k(void *buffer, int k) {
    int num_blocks = k / QK_K;
    block_q8_K *blocks = (block_q8_K *)buffer;
    for (int i = 0; i < num_blocks; ++i) {
        blocks[i].d = (float)(rand() % 100) / 1000.0f;
        for (int j = 0; j < QK_K; ++j) blocks[i].qs[j] = (int8_t)(rand() % 256 - 128);
        for (int j = 0; j < QK_K / 16; ++j) blocks[i].bsums[j] = rand() % 32768;
    }
}

double get_time_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int K = 4096; // Typical hidden size
    int M = 1;    // Batch size 1 (decode)
    int ITER = 1000;

    if (argc > 1) K = atoi(argv[1]);
    if (argc > 2) ITER = atoi(argv[2]);

    size_t q4_size = (K / QK_K) * sizeof(block_q4_K);
    size_t q8_size = (K / QK_K) * sizeof(block_q8_K);

    void *W = malloc(q4_size);
    void *X = malloc(q8_size);
    float Y_ref[M];
    float Y_avx[M];

    srand(42);
    fill_random_q4k(W, K);
    fill_random_q8k(X, K);

    printf("Benchmarking Q4_K x Q8_K (K=%d, M=%d, ITER=%d)\n", K, M, ITER);

    // Warmup
    gemv_q4_k_q8_k_ref(Y_ref, W, X, M, K);

    // Bench Ref
    double start = get_time_s();
    for (int i = 0; i < ITER; ++i) {
        gemv_q4_k_q8_k_ref(Y_ref, W, X, M, K);
    }
    double dt_ref = get_time_s() - start;
    printf("REF: %.3f ms/iter (%.2f GFLOPS)\n", 
           dt_ref * 1000.0 / ITER, 
           (2.0 * K * M * ITER) / dt_ref / 1e9);

    // Bench SSE
    start = get_time_s();
    for (int i = 0; i < ITER; ++i) {
        gemv_q4_k_q8_k_sse(Y_avx, W, X, M, K);
    }
    double dt_avx = get_time_s() - start;
    printf("SSE: %.3f ms/iter (%.2f GFLOPS)\n", 
           dt_avx * 1000.0 / ITER, 
           (2.0 * K * M * ITER) / dt_avx / 1e9);

    // Verify
    float diff = fabsf(Y_ref[0] - Y_avx[0]);
    if (diff > 1e-3) {
        printf("MISMATCH: Ref=%.4f, SSE=%.4f, Diff=%.4f\n", Y_ref[0], Y_avx[0], diff);
    } else {
        printf("VERIFIED: Match (diff %.4e)\n", diff);
    }

    free(W);
    free(X);
    return 0;
}
