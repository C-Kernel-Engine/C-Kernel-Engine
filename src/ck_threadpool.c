#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/**
 * @file ck_threadpool.c
 * @brief Persistent pthread thread pool for CK-Engine inference
 *
 * Architecture:
 *   - N-1 worker pthreads created at startup, main thread is thread 0
 *   - Workers spin on atomic dispatch counter waiting for work
 *   - Barriers use atomic counter + spin-wait with _mm_pause()
 *   - Hybrid polling: spin CK_THREADPOOL_SPIN_COUNT rounds, then condvar
 *   - All atomics on separate cache lines to avoid false sharing
 *
 * Based on the ggml_threadpool design from llama.cpp, adapted for
 * CK-Engine's kernel dispatch model.
 */

#include "ck_threadpool.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#ifdef __linux__
#include <sched.h>
#endif

#ifdef __x86_64__
#include <immintrin.h>
#define CK_SPIN_PAUSE() _mm_pause()
#else
#define CK_SPIN_PAUSE() ((void)0)
#endif

static atomic_int g_gemm_schedule = CK_GEMM_SCHEDULE_AUTO;

int ck_set_gemm_schedule(int policy)
{
    if (policy < CK_GEMM_SCHEDULE_AUTO || policy > CK_GEMM_SCHEDULE_DYNAMIC) {
        return -1;
    }
    atomic_store_explicit(&g_gemm_schedule, policy, memory_order_release);
    return 0;
}

int ck_get_gemm_schedule(void)
{
    return atomic_load_explicit(&g_gemm_schedule, memory_order_acquire);
}

int ck_gemm_dynamic_schedule_enabled(void)
{
    const int policy = ck_get_gemm_schedule();
    return policy == CK_GEMM_SCHEDULE_AUTO || policy == CK_GEMM_SCHEDULE_DYNAMIC;
}

/* ============================================================================
 * Internal Structures (cache-line aligned)
 * ============================================================================ */

/** Per-worker state */
typedef struct {
    pthread_t thread;
    int       id;           /* 0 = main, 1..n-1 = workers */
    struct ck_threadpool *pool;
} ck_worker_t;

/** Barrier state — all fields on separate cache lines */
typedef struct {
    _Alignas(CK_CACHE_LINE) atomic_int n_arrived;
    _Alignas(CK_CACHE_LINE) atomic_int n_phase;
    int n_threads;
    char _pad[CK_CACHE_LINE - sizeof(int)];
} ck_barrier_t;

/** Thread pool (opaque) */
struct ck_threadpool {
    /* Dispatch state — cache-line aligned */
    _Alignas(CK_CACHE_LINE) atomic_int      n_dispatch;    /* bumped to wake workers */
    _Alignas(CK_CACHE_LINE) atomic_int      n_complete;    /* workers signal completion */
    _Alignas(CK_CACHE_LINE) atomic_int      active_threads; /* active threads for current dispatch */
    _Alignas(CK_CACHE_LINE) ck_work_fn_t    work_fn;       /* current work function */
    void                                    *work_args;     /* current work arguments */

    /* Barrier for intra-dispatch synchronization */
    ck_barrier_t barrier;

    /* Worker management */
    int          n_threads;       /* worker capacity (including main) */
    int          default_threads; /* ordinary dispatch width */
    ck_worker_t  workers[CK_THREADPOOL_MAX_THREADS];

    /* Shutdown / pause signals */
    _Alignas(CK_CACHE_LINE) atomic_int stop;
    _Alignas(CK_CACHE_LINE) atomic_int paused;

    /* Condvar for sleep/wake (hybrid polling) */
    pthread_mutex_t mutex;
    pthread_cond_t  cond_dispatch;  /* workers wait here when sleeping */
    pthread_cond_t  cond_done;      /* main waits here for completion */

    _Alignas(CK_CACHE_LINE) atomic_int profile_enabled;
    atomic_uint_fast64_t profile_dispatch_count;
    atomic_uint_fast64_t profile_dispatch_total_ns;
    atomic_uint_fast64_t profile_main_work_ns;
    atomic_uint_fast64_t profile_completion_wait_ns;
};

static uint64_t monotonic_ns(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

/* ============================================================================
 * Barrier Implementation
 * ============================================================================ */

static void barrier_init(ck_barrier_t *b, int n_threads)
{
    atomic_store(&b->n_arrived, 0);
    atomic_store(&b->n_phase, 0);
    b->n_threads = n_threads;
}

/**
 * Spin-wait barrier. All threads must call this.
 * Uses phase counter to allow re-use without reset.
 */
static void barrier_wait(ck_barrier_t *b)
{
    const int n = b->n_threads;
    const int phase = atomic_load_explicit(&b->n_phase, memory_order_relaxed);

    if (atomic_fetch_add_explicit(&b->n_arrived, 1, memory_order_acq_rel) == n - 1) {
        /* Last thread to arrive — reset and advance phase */
        atomic_store_explicit(&b->n_arrived, 0, memory_order_relaxed);
        atomic_store_explicit(&b->n_phase, phase + 1, memory_order_release);
    } else {
        /* Spin until phase advances */
        int spins = 0;
        while (atomic_load_explicit(&b->n_phase, memory_order_acquire) == phase) {
            CK_SPIN_PAUSE();
            spins++;
            /* After many spins, yield to avoid wasting CPU on oversubscribed systems */
            if (spins > CK_THREADPOOL_SPIN_COUNT * 16) {
                sched_yield();
                spins = 0;
            }
        }
    }
}

/* ============================================================================
 * Worker Thread
 * ============================================================================ */

static void *worker_main(void *arg)
{
    ck_worker_t *w = (ck_worker_t *)arg;
    ck_threadpool_t *pool = w->pool;
    const int ith = w->id;
    int last_dispatch = 0;

    for (;;) {
        /* Spin-wait for new dispatch */
        int spins = 0;
        int active = 0;
        ck_work_fn_t fn = NULL;
        void *args = NULL;
        for (;;) {
            /* Check shutdown */
            if (atomic_load_explicit(&pool->stop, memory_order_acquire)) {
                return NULL;
            }

            /* Check for new work */
            int current = atomic_load_explicit(&pool->n_dispatch, memory_order_acquire);
            active = atomic_load_explicit(&pool->active_threads, memory_order_acquire);
            if (current != last_dispatch) {
                /* Snapshot the epoch and descriptor together. An inactive
                 * worker may still be catching up with an older dispatch. */
                pthread_mutex_lock(&pool->mutex);
                current = atomic_load_explicit(&pool->n_dispatch, memory_order_acquire);
                active = atomic_load_explicit(&pool->active_threads, memory_order_acquire);
                if (current != last_dispatch) {
                    last_dispatch = current;
                    if (ith < active) {
                        fn = pool->work_fn;
                        args = pool->work_args;
                        pthread_mutex_unlock(&pool->mutex);
                        break;
                    }
                }
                pthread_mutex_unlock(&pool->mutex);
                spins = 0;
            }

            /* Threads outside the active subset sleep instead of spinning. */
            if (ith >= active || spins >= CK_THREADPOOL_SPIN_COUNT) {
                pthread_mutex_lock(&pool->mutex);
                for (;;) {
                    if (atomic_load_explicit(&pool->stop, memory_order_acquire)) {
                        pthread_mutex_unlock(&pool->mutex);
                        return NULL;
                    }
                    current = atomic_load_explicit(&pool->n_dispatch, memory_order_acquire);
                    active = atomic_load_explicit(&pool->active_threads, memory_order_acquire);
                    if (current != last_dispatch) {
                        last_dispatch = current;
                        if (ith < active) {
                            fn = pool->work_fn;
                            args = pool->work_args;
                            pthread_mutex_unlock(&pool->mutex);
                            goto worker_have_work;
                        }
                    }
                    pthread_cond_wait(&pool->cond_dispatch, &pool->mutex);
                }
            }

            CK_SPIN_PAUSE();
            spins++;
        }

worker_have_work:
        /* Execute work */
        if (fn) {
            fn(ith, active, args);
        }

        /* Signal completion */
        if (atomic_fetch_add_explicit(&pool->n_complete, 1, memory_order_acq_rel)
            == active - 2) {
            /* Last worker done — wake main thread if it's waiting */
            pthread_mutex_lock(&pool->mutex);
            pthread_cond_signal(&pool->cond_done);
            pthread_mutex_unlock(&pool->mutex);
        }
    }

    return NULL;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

extern int ck_get_physical_cores(void);

int ck_threadpool_bounded_capacity(int default_threads, int logical_threads)
{
    if (default_threads < 1) default_threads = 1;
    if (default_threads > CK_THREADPOOL_MAX_THREADS) {
        default_threads = CK_THREADPOOL_MAX_THREADS;
    }
    if (logical_threads <= default_threads) return default_threads;

    int capacity = default_threads + (logical_threads - default_threads) / 2;
    if (capacity > CK_THREADPOOL_MAX_THREADS) {
        capacity = CK_THREADPOOL_MAX_THREADS;
    }
    return capacity;
}

ck_threadpool_t *ck_threadpool_create_capacity(int default_threads,
                                                int capacity_threads)
{
    if (default_threads <= 0) {
        default_threads = ck_get_physical_cores();
        if (default_threads <= 0) default_threads = 1;
        /* Cap at reasonable default for memory-bound workloads */
        if (default_threads > 8) default_threads = 8;
    }
    if (capacity_threads < default_threads) {
        capacity_threads = default_threads;
    }
    if (capacity_threads > CK_THREADPOOL_MAX_THREADS) {
        capacity_threads = CK_THREADPOOL_MAX_THREADS;
    }
    if (default_threads > capacity_threads) {
        default_threads = capacity_threads;
    }

    ck_threadpool_t *pool = aligned_alloc(CK_CACHE_LINE, sizeof(ck_threadpool_t));
    if (!pool) return NULL;
    memset(pool, 0, sizeof(*pool));

    pool->n_threads = capacity_threads;
    pool->default_threads = default_threads;
    atomic_store(&pool->n_dispatch, 0);
    atomic_store(&pool->n_complete, 0);
    atomic_store(&pool->active_threads, default_threads);
    atomic_store(&pool->stop, 0);
    atomic_store(&pool->paused, 0);
    atomic_store(&pool->profile_enabled, 0);
    pool->work_fn = NULL;
    pool->work_args = NULL;

    barrier_init(&pool->barrier, default_threads);

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_dispatch, NULL);
    pthread_cond_init(&pool->cond_done, NULL);

    /* Thread 0 = main thread (no pthread created) */
    pool->workers[0].id = 0;
    pool->workers[0].pool = pool;
    pool->workers[0].thread = pthread_self();

    /* Spawn N-1 worker threads */
    for (int i = 1; i < capacity_threads; i++) {
        pool->workers[i].id = i;
        pool->workers[i].pool = pool;

        int rc = pthread_create(&pool->workers[i].thread, NULL,
                                worker_main, &pool->workers[i]);
        if (rc != 0) {
            fprintf(stderr, "[CK threadpool] Failed to create worker %d: %s\n",
                    i, strerror(rc));
            /* Reduce thread count to what we managed to create */
            pool->n_threads = i;
            barrier_init(&pool->barrier, i);
            break;
        }
    }

    if (pool->n_threads > 1) {
        fprintf(stderr,
                "[CK threadpool] Created %d threads (default=%d, 1 main + %d workers)\n",
                pool->n_threads, pool->default_threads, pool->n_threads - 1);
    }

    return pool;
}

ck_threadpool_t *ck_threadpool_create(int n_threads)
{
    return ck_threadpool_create_capacity(n_threads, n_threads);
}

void ck_threadpool_destroy(ck_threadpool_t *pool)
{
    if (!pool) return;

    /* Signal shutdown */
    atomic_store_explicit(&pool->stop, 1, memory_order_release);

    /* Wake all sleeping workers */
    pthread_mutex_lock(&pool->mutex);
    pthread_cond_broadcast(&pool->cond_dispatch);
    pthread_mutex_unlock(&pool->mutex);

    /* Join all worker threads */
    for (int i = 1; i < pool->n_threads; i++) {
        pthread_join(pool->workers[i].thread, NULL);
    }

    pthread_cond_destroy(&pool->cond_dispatch);
    pthread_cond_destroy(&pool->cond_done);
    pthread_mutex_destroy(&pool->mutex);

    free(pool);
}

/* ============================================================================
 * Dispatch & Synchronization
 * ============================================================================ */

void ck_threadpool_dispatch_n(ck_threadpool_t *pool, int active_threads, ck_work_fn_t fn, void *args)
{
    if (!pool || !fn) return;
    if (active_threads <= 0) {
        active_threads = 1;
    }
    if (active_threads > pool->n_threads) {
        active_threads = pool->n_threads;
    }

    const int profile = atomic_load_explicit(
        &pool->profile_enabled, memory_order_relaxed);
    const uint64_t dispatch_start = profile ? monotonic_ns() : 0;

    /* Single-thread fast path: just call directly */
    if (active_threads == 1 || pool->n_threads == 1) {
        fn(0, 1, args);
        if (profile) {
            const uint64_t dispatch_end = monotonic_ns();
            atomic_fetch_add_explicit(&pool->profile_dispatch_count, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(
                &pool->profile_dispatch_total_ns,
                dispatch_end - dispatch_start,
                memory_order_relaxed);
            atomic_fetch_add_explicit(
                &pool->profile_main_work_ns,
                dispatch_end - dispatch_start,
                memory_order_relaxed);
        }
        return;
    }

    /* Reset barrier phase for this dispatch */
    barrier_init(&pool->barrier, active_threads);

    /* Set work descriptor */
    pthread_mutex_lock(&pool->mutex);
    pool->work_fn = fn;
    pool->work_args = args;
    atomic_store_explicit(&pool->active_threads, active_threads, memory_order_release);
    atomic_store_explicit(&pool->n_complete, 0, memory_order_release);

    /* Wake workers by bumping dispatch counter */
    atomic_fetch_add_explicit(&pool->n_dispatch, 1, memory_order_release);

    /* Also signal condvar for sleeping workers */
    pthread_cond_broadcast(&pool->cond_dispatch);
    pthread_mutex_unlock(&pool->mutex);

    /* Main thread (ith=0) does its share */
    const uint64_t main_start = profile ? monotonic_ns() : 0;
    fn(0, active_threads, args);
    const uint64_t main_end = profile ? monotonic_ns() : 0;

    /* Wait for all workers to complete */
    if (active_threads > 1) {
        int spins = 0;
        while (atomic_load_explicit(&pool->n_complete, memory_order_acquire)
               < active_threads - 1) {
            CK_SPIN_PAUSE();
            spins++;
            if (spins >= CK_THREADPOOL_SPIN_COUNT) {
                pthread_mutex_lock(&pool->mutex);
                if (atomic_load_explicit(&pool->n_complete, memory_order_acquire)
                    < active_threads - 1) {
                    pthread_cond_wait(&pool->cond_done, &pool->mutex);
                }
                pthread_mutex_unlock(&pool->mutex);
                spins = 0;
            }
        }
    }
    if (profile) {
        const uint64_t dispatch_end = monotonic_ns();
        atomic_fetch_add_explicit(&pool->profile_dispatch_count, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &pool->profile_dispatch_total_ns,
            dispatch_end - dispatch_start,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &pool->profile_main_work_ns,
            main_end - main_start,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &pool->profile_completion_wait_ns,
            dispatch_end - main_end,
            memory_order_relaxed);
    }
}

void ck_threadpool_dispatch(ck_threadpool_t *pool, ck_work_fn_t fn, void *args)
{
    if (!pool) return;
    ck_threadpool_dispatch_n(pool, pool->default_threads, fn, args);
}

typedef struct {
    _Alignas(CK_CACHE_LINE) atomic_int next;
    int end;
    int grain_size;
    ck_range_fn_t fn;
    void *args;
} ck_parallel_for_work_t;

static void ck_parallel_for_worker(int ith, int nth, void *opaque)
{
    (void)ith;
    (void)nth;
    ck_parallel_for_work_t *work = (ck_parallel_for_work_t *)opaque;
    for (;;) {
        const int begin = atomic_fetch_add_explicit(
            &work->next, work->grain_size, memory_order_relaxed);
        if (begin >= work->end) break;
        int end = begin + work->grain_size;
        if (end > work->end) end = work->end;
        work->fn(begin, end, work->args);
    }
}

void ck_threadpool_parallel_for_n(ck_threadpool_t *pool,
                                  int active_threads,
                                  int begin,
                                  int end,
                                  int grain_size,
                                  ck_range_fn_t fn,
                                  void *args)
{
    if (!fn || begin >= end) return;
    if (grain_size <= 0) grain_size = 1;
    if (!pool || active_threads <= 1) {
        fn(begin, end, args);
        return;
    }

    ck_parallel_for_work_t work = {
        .end = end,
        .grain_size = grain_size,
        .fn = fn,
        .args = args,
    };
    atomic_init(&work.next, begin);
    ck_threadpool_dispatch_n(
        pool, active_threads, ck_parallel_for_worker, &work);
}

void ck_threadpool_barrier(ck_threadpool_t *pool)
{
    if (!pool || pool->n_threads <= 1) return;
    barrier_wait(&pool->barrier);
}

/* ============================================================================
 * Power Management
 * ============================================================================ */

void ck_threadpool_pause(ck_threadpool_t *pool)
{
    if (!pool) return;
    atomic_store_explicit(&pool->paused, 1, memory_order_release);
}

void ck_threadpool_resume(ck_threadpool_t *pool)
{
    if (!pool) return;
    atomic_store_explicit(&pool->paused, 0, memory_order_release);

    /* Wake sleeping workers */
    pthread_mutex_lock(&pool->mutex);
    pthread_cond_broadcast(&pool->cond_dispatch);
    pthread_mutex_unlock(&pool->mutex);
}

/* ============================================================================
 * Queries
 * ============================================================================ */

int ck_threadpool_n_threads(const ck_threadpool_t *pool)
{
    return pool ? pool->default_threads : 1;
}

int ck_threadpool_capacity(const ck_threadpool_t *pool)
{
    return pool ? pool->n_threads : 1;
}

int ck_threadpool_thread_id(const ck_threadpool_t *pool)
{
    if (!pool) return -1;
    pthread_t self = pthread_self();
    for (int i = 0; i < pool->n_threads; i++) {
        if (pthread_equal(self, pool->workers[i].thread)) {
            return i;
        }
    }
    return -1;
}

void ck_threadpool_profile_reset(ck_threadpool_t *pool)
{
    if (!pool) return;
    atomic_store_explicit(&pool->profile_dispatch_count, 0, memory_order_relaxed);
    atomic_store_explicit(&pool->profile_dispatch_total_ns, 0, memory_order_relaxed);
    atomic_store_explicit(&pool->profile_main_work_ns, 0, memory_order_relaxed);
    atomic_store_explicit(&pool->profile_completion_wait_ns, 0, memory_order_relaxed);
    atomic_store_explicit(&pool->profile_enabled, 1, memory_order_release);
}

void ck_threadpool_profile_snapshot(
    const ck_threadpool_t *pool, ck_threadpool_profile_t *profile)
{
    if (!profile) return;
    memset(profile, 0, sizeof(*profile));
    if (!pool) return;
    profile->dispatch_count = atomic_load_explicit(
        &pool->profile_dispatch_count, memory_order_relaxed);
    profile->dispatch_total_ns = atomic_load_explicit(
        &pool->profile_dispatch_total_ns, memory_order_relaxed);
    profile->main_work_ns = atomic_load_explicit(
        &pool->profile_main_work_ns, memory_order_relaxed);
    profile->completion_wait_ns = atomic_load_explicit(
        &pool->profile_completion_wait_ns, memory_order_relaxed);
}

/* ============================================================================
 * Global Thread Pool
 * ============================================================================ */

static ck_threadpool_t *g_threadpool = NULL;
static pthread_once_t g_threadpool_once = PTHREAD_ONCE_INIT;

extern int ck_get_num_threads(void);

static int ck_available_logical_cpus(void)
{
#ifdef __linux__
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        int count = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &allowed)) ++count;
        }
        if (count > 0) return count;
    }
#endif
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    return online > 0 ? (int)online : 1;
}

static void global_pool_init(void)
{
    const int available_threads = ck_available_logical_cpus();
    int physical_threads = ck_get_physical_cores();
    if (physical_threads > available_threads) physical_threads = available_threads;
    int default_threads = ck_get_num_threads();
    if (default_threads > available_threads) {
        default_threads = available_threads;
    }
    int capacity_threads = default_threads;
    const char *capacity_env = getenv("CK_THREADPOOL_CAPACITY");
    if (capacity_env && atoi(capacity_env) > 0) {
        capacity_threads = atoi(capacity_env);
    } else if (!getenv("CK_NUM_THREADS") &&
               default_threads == physical_threads) {
        /* Generated runtimes set OMP_NUM_THREADS=1 to keep OpenMP dormant and
         * configure the CK pool separately. Do not mistake that isolation
         * setting for a CK capacity cap. A non-physical default remains an
         * explicit width and does not gain automatic SMT workers. */
        capacity_threads = ck_threadpool_bounded_capacity(
                default_threads, available_threads);
    }
    if (capacity_threads > available_threads) capacity_threads = available_threads;
    g_threadpool = ck_threadpool_create_capacity(
            default_threads, capacity_threads);
}

ck_threadpool_t *ck_threadpool_global(void)
{
    pthread_once(&g_threadpool_once, global_pool_init);
    return g_threadpool;
}

void ck_threadpool_global_destroy(void)
{
    if (g_threadpool) {
        ck_threadpool_destroy(g_threadpool);
        g_threadpool = NULL;
        /* Reset once control so pool can be re-created if needed */
        g_threadpool_once = PTHREAD_ONCE_INIT;
    }
}
