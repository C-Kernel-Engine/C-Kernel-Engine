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

#ifdef __x86_64__
#include <immintrin.h>
#define CK_SPIN_PAUSE() _mm_pause()
#else
#define CK_SPIN_PAUSE() ((void)0)
#endif

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
    int          n_threads;     /* total threads (including main) */
    ck_worker_t  workers[CK_THREADPOOL_MAX_THREADS];

    /* Shutdown / pause signals */
    _Alignas(CK_CACHE_LINE) atomic_int stop;
    _Alignas(CK_CACHE_LINE) atomic_int paused;

    /* Condvar for sleep/wake (hybrid polling) */
    pthread_mutex_t mutex;
    pthread_cond_t  cond_dispatch;  /* workers wait here when sleeping */
    pthread_cond_t  cond_done;      /* main waits here for completion */
};

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
                last_dispatch = current;
                if (ith < active) {
                    break;
                }
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
        fn = pool->work_fn;
        args = pool->work_args;
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

ck_threadpool_t *ck_threadpool_create(int n_threads)
{
    if (n_threads <= 0) {
        n_threads = ck_get_physical_cores();
        if (n_threads <= 0) n_threads = 1;
        /* Cap at reasonable default for memory-bound workloads */
        if (n_threads > 8) n_threads = 8;
    }
    if (n_threads > CK_THREADPOOL_MAX_THREADS) {
        n_threads = CK_THREADPOOL_MAX_THREADS;
    }

    ck_threadpool_t *pool = aligned_alloc(CK_CACHE_LINE, sizeof(ck_threadpool_t));
    if (!pool) return NULL;
    memset(pool, 0, sizeof(*pool));

    pool->n_threads = n_threads;
    atomic_store(&pool->n_dispatch, 0);
    atomic_store(&pool->n_complete, 0);
    atomic_store(&pool->active_threads, n_threads);
    atomic_store(&pool->stop, 0);
    atomic_store(&pool->paused, 0);
    pool->work_fn = NULL;
    pool->work_args = NULL;

    barrier_init(&pool->barrier, n_threads);

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_dispatch, NULL);
    pthread_cond_init(&pool->cond_done, NULL);

    /* Thread 0 = main thread (no pthread created) */
    pool->workers[0].id = 0;
    pool->workers[0].pool = pool;
    pool->workers[0].thread = pthread_self();

    /* Spawn N-1 worker threads */
    for (int i = 1; i < n_threads; i++) {
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
        fprintf(stderr, "[CK threadpool] Created %d threads (1 main + %d workers)\n",
                pool->n_threads, pool->n_threads - 1);
    }

    return pool;
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

    /* Single-thread fast path: just call directly */
    if (active_threads == 1 || pool->n_threads == 1) {
        fn(0, 1, args);
        return;
    }

    /* Reset barrier phase for this dispatch */
    barrier_init(&pool->barrier, active_threads);

    /* Set work descriptor */
    pool->work_fn = fn;
    pool->work_args = args;
    atomic_store_explicit(&pool->active_threads, active_threads, memory_order_release);
    atomic_store_explicit(&pool->n_complete, 0, memory_order_release);

    /* Wake workers by bumping dispatch counter */
    atomic_fetch_add_explicit(&pool->n_dispatch, 1, memory_order_release);

    /* Also signal condvar for sleeping workers */
    pthread_mutex_lock(&pool->mutex);
    pthread_cond_broadcast(&pool->cond_dispatch);
    pthread_mutex_unlock(&pool->mutex);

    /* Main thread (ith=0) does its share */
    fn(0, active_threads, args);

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
}

void ck_threadpool_dispatch(ck_threadpool_t *pool, ck_work_fn_t fn, void *args)
{
    if (!pool) return;
    ck_threadpool_dispatch_n(pool, pool->n_threads, fn, args);
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

/* ============================================================================
 * Global Thread Pool
 * ============================================================================ */

static ck_threadpool_t *g_threadpool = NULL;
static pthread_once_t g_threadpool_once = PTHREAD_ONCE_INIT;

extern int ck_get_num_threads(void);

static void global_pool_init(void)
{
    int n = ck_get_num_threads();
    g_threadpool = ck_threadpool_create(n);
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
