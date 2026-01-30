/**
 * @file test_threadpool.c
 * @brief Unit tests and benchmarks for ck_threadpool
 *
 * Tests:
 *   1. Create / destroy lifecycle
 *   2. Single-thread dispatch
 *   3. Multi-thread dispatch with correct ith/nth
 *   4. Barrier correctness (all threads sync)
 *   5. Multiple sequential dispatches
 *   6. Pause / resume
 *   7. Dispatch latency benchmark
 *   8. Barrier latency benchmark
 *   9. Parallel sum correctness (simulated GEMV row split)
 *
 * Usage:
 *   ./build/test_threadpool           # Run all tests
 *   ./build/test_threadpool --bench   # Run benchmarks only
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <math.h>
#include <time.h>

#include "ck_threadpool.h"

/* ============================================================================
 * Test Helpers
 * ============================================================================ */

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) do {                                    \
    g_tests_run++;                                                     \
    if (!(cond)) {                                                     \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", msg, __LINE__,  \
                #cond);                                                \
        return 0;                                                      \
    }                                                                  \
    g_tests_passed++;                                                  \
} while (0)

static double time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================================
 * Test 1: Create / Destroy
 * ============================================================================ */

static int test_create_destroy(void)
{
    printf("  [1] Create / destroy...\n");

    ck_threadpool_t *pool = ck_threadpool_create(4);
    TEST_ASSERT(pool != NULL, "create with 4 threads");
    TEST_ASSERT(ck_threadpool_n_threads(pool) == 4, "n_threads == 4");
    ck_threadpool_destroy(pool);

    /* Single thread */
    pool = ck_threadpool_create(1);
    TEST_ASSERT(pool != NULL, "create with 1 thread");
    TEST_ASSERT(ck_threadpool_n_threads(pool) == 1, "n_threads == 1");
    ck_threadpool_destroy(pool);

    /* NULL safety */
    ck_threadpool_destroy(NULL);

    return 1;
}

/* ============================================================================
 * Test 2: Single-Thread Dispatch
 * ============================================================================ */

static atomic_int g_single_count;

static void single_work(int ith, int nth, void *args)
{
    (void)args;
    if (ith == 0 && nth == 1) {
        atomic_fetch_add(&g_single_count, 1);
    }
}

static int test_single_dispatch(void)
{
    printf("  [2] Single-thread dispatch...\n");

    ck_threadpool_t *pool = ck_threadpool_create(1);
    atomic_store(&g_single_count, 0);

    ck_threadpool_dispatch(pool, single_work, NULL);
    TEST_ASSERT(atomic_load(&g_single_count) == 1, "work called once");

    ck_threadpool_dispatch(pool, single_work, NULL);
    TEST_ASSERT(atomic_load(&g_single_count) == 2, "work called twice");

    ck_threadpool_destroy(pool);
    return 1;
}

/* ============================================================================
 * Test 3: Multi-Thread Dispatch
 * ============================================================================ */

#define MAX_TEST_THREADS 16
static atomic_int g_thread_seen[MAX_TEST_THREADS];

static void multi_work(int ith, int nth, void *args)
{
    int *expected_nth = (int *)args;
    if (ith >= 0 && ith < MAX_TEST_THREADS && nth == *expected_nth) {
        atomic_fetch_add(&g_thread_seen[ith], 1);
    }
}

static int test_multi_dispatch(void)
{
    printf("  [3] Multi-thread dispatch...\n");

    int n = 4;
    ck_threadpool_t *pool = ck_threadpool_create(n);

    for (int i = 0; i < MAX_TEST_THREADS; i++) {
        atomic_store(&g_thread_seen[i], 0);
    }

    ck_threadpool_dispatch(pool, multi_work, &n);

    /* Every thread should have been called exactly once */
    int actual_n = ck_threadpool_n_threads(pool);
    for (int i = 0; i < actual_n; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "thread %d was called", i);
        TEST_ASSERT(atomic_load(&g_thread_seen[i]) == 1, msg);
    }

    ck_threadpool_destroy(pool);
    return 1;
}

/* ============================================================================
 * Test 4: Barrier Correctness
 * ============================================================================ */

static atomic_int g_phase1_done;
static atomic_int g_phase2_started;
static int g_barrier_ok;

static void barrier_work(int ith, int nth, void *args)
{
    ck_threadpool_t *pool = (ck_threadpool_t *)args;

    /* Phase 1: all threads increment counter */
    atomic_fetch_add(&g_phase1_done, 1);

    /* Barrier: wait for all threads */
    ck_threadpool_barrier(pool);

    /* Phase 2: check that ALL threads completed phase 1 */
    int p1 = atomic_load(&g_phase1_done);
    if (p1 != nth) {
        g_barrier_ok = 0;
    }
    atomic_fetch_add(&g_phase2_started, 1);
}

static int test_barrier(void)
{
    printf("  [4] Barrier correctness...\n");

    int n = 4;
    ck_threadpool_t *pool = ck_threadpool_create(n);
    int actual_n = ck_threadpool_n_threads(pool);

    atomic_store(&g_phase1_done, 0);
    atomic_store(&g_phase2_started, 0);
    g_barrier_ok = 1;

    ck_threadpool_dispatch(pool, barrier_work, pool);

    TEST_ASSERT(g_barrier_ok, "barrier synchronized correctly");
    TEST_ASSERT(atomic_load(&g_phase2_started) == actual_n,
                "all threads reached phase 2");

    ck_threadpool_destroy(pool);
    return 1;
}

/* ============================================================================
 * Test 5: Multiple Sequential Dispatches
 * ============================================================================ */

static atomic_int g_seq_sum;

static void seq_work(int ith, int nth, void *args)
{
    (void)nth; (void)args;
    atomic_fetch_add(&g_seq_sum, ith + 1);  /* thread 0 adds 1, thread 1 adds 2, etc */
}

static int test_sequential_dispatch(void)
{
    printf("  [5] Sequential dispatches...\n");

    int n = 4;
    ck_threadpool_t *pool = ck_threadpool_create(n);
    int actual_n = ck_threadpool_n_threads(pool);
    int expected_per = actual_n * (actual_n + 1) / 2;  /* sum(1..n) */

    atomic_store(&g_seq_sum, 0);

    for (int d = 0; d < 10; d++) {
        ck_threadpool_dispatch(pool, seq_work, NULL);
    }

    TEST_ASSERT(atomic_load(&g_seq_sum) == expected_per * 10,
                "10 dispatches sum correct");

    ck_threadpool_destroy(pool);
    return 1;
}

/* ============================================================================
 * Test 6: Pause / Resume
 * ============================================================================ */

static int test_pause_resume(void)
{
    printf("  [6] Pause / resume...\n");

    int n = 4;
    ck_threadpool_t *pool = ck_threadpool_create(n);

    atomic_store(&g_seq_sum, 0);
    ck_threadpool_dispatch(pool, seq_work, NULL);
    int sum1 = atomic_load(&g_seq_sum);
    TEST_ASSERT(sum1 > 0, "dispatch before pause works");

    ck_threadpool_pause(pool);

    /* After pause, dispatch should still work (wakes workers) */
    ck_threadpool_dispatch(pool, seq_work, NULL);
    int sum2 = atomic_load(&g_seq_sum);
    TEST_ASSERT(sum2 > sum1, "dispatch after pause works");

    ck_threadpool_resume(pool);

    ck_threadpool_dispatch(pool, seq_work, NULL);
    int sum3 = atomic_load(&g_seq_sum);
    TEST_ASSERT(sum3 > sum2, "dispatch after resume works");

    ck_threadpool_destroy(pool);
    return 1;
}

/* ============================================================================
 * Test 7: Parallel Sum (simulated GEMV row split)
 * ============================================================================ */

#define PSUM_N 4096
static float g_psum_data[PSUM_N];
static float g_psum_partial[MAX_TEST_THREADS];

static void psum_work(int ith, int nth, void *args)
{
    (void)args;
    int dr = (PSUM_N + nth - 1) / nth;
    int r0 = dr * ith;
    int r1 = (r0 + dr < PSUM_N) ? (r0 + dr) : PSUM_N;

    float sum = 0.0f;
    for (int i = r0; i < r1; i++) {
        sum += g_psum_data[i];
    }
    g_psum_partial[ith] = sum;
}

static int test_parallel_sum(void)
{
    printf("  [7] Parallel sum (GEMV row split)...\n");

    /* Fill data: 1, 2, 3, ..., N */
    for (int i = 0; i < PSUM_N; i++) {
        g_psum_data[i] = (float)(i + 1);
    }
    float expected = (float)PSUM_N * (float)(PSUM_N + 1) / 2.0f;

    int n = 4;
    ck_threadpool_t *pool = ck_threadpool_create(n);
    int actual_n = ck_threadpool_n_threads(pool);
    memset(g_psum_partial, 0, sizeof(g_psum_partial));

    ck_threadpool_dispatch(pool, psum_work, NULL);

    /* Sum partial results */
    float total = 0.0f;
    for (int i = 0; i < actual_n; i++) {
        total += g_psum_partial[i];
    }

    TEST_ASSERT(fabsf(total - expected) < 1.0f,
                "parallel sum matches serial");

    ck_threadpool_destroy(pool);
    return 1;
}

/* ============================================================================
 * Benchmark: Dispatch Latency
 * ============================================================================ */

static void noop_work(int ith, int nth, void *args)
{
    (void)ith; (void)nth; (void)args;
}

static void bench_dispatch(void)
{
    printf("\n  Dispatch latency benchmark:\n");

    int thread_counts[] = {1, 2, 4, 8};
    int n_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int t = 0; t < n_counts; t++) {
        int n = thread_counts[t];
        ck_threadpool_t *pool = ck_threadpool_create(n);
        if (!pool) continue;

        int actual_n = ck_threadpool_n_threads(pool);
        int warmup = 100;
        int iters = 10000;

        /* Warmup */
        for (int i = 0; i < warmup; i++) {
            ck_threadpool_dispatch(pool, noop_work, NULL);
        }

        double t0 = time_ms();
        for (int i = 0; i < iters; i++) {
            ck_threadpool_dispatch(pool, noop_work, NULL);
        }
        double elapsed = time_ms() - t0;
        double per_dispatch_us = (elapsed / iters) * 1000.0;

        printf("    %d threads: %.1f us/dispatch (%.0f dispatches/ms)\n",
               actual_n, per_dispatch_us, iters / elapsed);

        ck_threadpool_destroy(pool);
    }
}

/* ============================================================================
 * Benchmark: Barrier Latency
 * ============================================================================ */

#define BARRIER_ITERS 1000

static void barrier_bench_work(int ith, int nth, void *args)
{
    ck_threadpool_t *pool = (ck_threadpool_t *)args;
    for (int i = 0; i < BARRIER_ITERS; i++) {
        ck_threadpool_barrier(pool);
    }
}

static void bench_barrier(void)
{
    printf("\n  Barrier latency benchmark:\n");

    int thread_counts[] = {2, 4, 8};
    int n_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int t = 0; t < n_counts; t++) {
        int n = thread_counts[t];
        ck_threadpool_t *pool = ck_threadpool_create(n);
        if (!pool) continue;

        int actual_n = ck_threadpool_n_threads(pool);
        if (actual_n < 2) {
            ck_threadpool_destroy(pool);
            continue;
        }

        /* Warmup */
        ck_threadpool_dispatch(pool, barrier_bench_work, pool);

        double t0 = time_ms();
        ck_threadpool_dispatch(pool, barrier_bench_work, pool);
        double elapsed = time_ms() - t0;
        double per_barrier_us = (elapsed / BARRIER_ITERS) * 1000.0;

        printf("    %d threads: %.2f us/barrier (%d barriers in %.1f ms)\n",
               actual_n, per_barrier_us, BARRIER_ITERS, elapsed);

        ck_threadpool_destroy(pool);
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv)
{
    int bench_only = 0;
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        bench_only = 1;
    }

    printf("========================================\n");
    printf("  CK Thread Pool Tests\n");
    printf("========================================\n\n");

    if (!bench_only) {
        int ok = 1;
        ok &= test_create_destroy();
        ok &= test_single_dispatch();
        ok &= test_multi_dispatch();
        ok &= test_barrier();
        ok &= test_sequential_dispatch();
        ok &= test_pause_resume();
        ok &= test_parallel_sum();

        printf("\n  Results: %d/%d passed\n", g_tests_passed, g_tests_run);

        if (!ok || g_tests_passed != g_tests_run) {
            printf("  SOME TESTS FAILED\n");
        } else {
            printf("  ALL TESTS PASSED\n");
        }
    }

    /* Benchmarks */
    printf("\n========================================\n");
    printf("  Thread Pool Benchmarks\n");
    printf("========================================\n");
    bench_dispatch();
    bench_barrier();

    printf("\n");
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
