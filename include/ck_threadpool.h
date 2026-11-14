/**
 * @file ck_threadpool.h
 * @brief Persistent pthread thread pool for CK-Engine inference
 *
 * Design goals:
 * - Sub-microsecond dispatch latency (spin-wait barriers)
 * - Zero allocation after init (all memory pre-allocated)
 * - Cache-line aligned atomics to avoid false sharing
 * - Hybrid polling: spin N rounds, then fall back to condvar
 * - Thread 0 = calling thread (does serial ops + its share of parallel work)
 *
 * Usage:
 *   ck_threadpool_t *pool = ck_threadpool_create(4);  // 4 threads total
 *
 *   // In decode loop:
 *   ck_threadpool_dispatch(pool, my_work_fn, args);
 *   // my_work_fn called on all threads with (ith, nth, args)
 *
 *   // Between batches:
 *   ck_threadpool_pause(pool);   // workers sleep (0% CPU)
 *   ck_threadpool_resume(pool);  // wake workers
 *
 *   ck_threadpool_destroy(pool);
 *
 * Architecture:
 *   STARTUP:  Main creates N-1 worker pthreads, all spin on atomic counter
 *   DISPATCH: Main writes work desc, bumps counter, all threads execute
 *   BARRIER:  Atomic counter + spin-wait with _mm_pause()
 *   PAUSE:    Workers sleep on pthread_cond_t (0% CPU between batches)
 */

#ifndef CK_THREADPOOL_H
#define CK_THREADPOOL_H

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */

/** Maximum threads supported (main + workers) */
#define CK_THREADPOOL_MAX_THREADS 64

/** Number of spin iterations before falling back to condvar wait */
#define CK_THREADPOOL_SPIN_COUNT 1024

/** Cache line size for alignment (x86-64) */
#define CK_CACHE_LINE 64

/* ============================================================================
 * Types
 * ============================================================================ */

/**
 * Work function signature.
 * Called on ALL threads (including calling thread 0).
 *
 * @param ith   Thread index (0 = calling thread)
 * @param nth   Total number of threads
 * @param args  Opaque argument pointer (set via dispatch)
 */
typedef void (*ck_work_fn_t)(int ith, int nth, void *args);

/**
 * Thread pool state (opaque).
 *
 * All atomics are cache-line aligned to prevent false sharing.
 * Workers spin on n_dispatch, checking for new work or shutdown.
 */
typedef struct ck_threadpool ck_threadpool_t;

typedef struct {
    uint64_t dispatch_count;
    uint64_t dispatch_total_ns;
    uint64_t main_work_ns;
    uint64_t completion_wait_ns;
} ck_threadpool_profile_t;

/** Process the half-open interval [begin, end). */
typedef void (*ck_range_fn_t)(int begin, int end, void *args);

/** Scheduling policy for independent GEMM output tiles. */
typedef enum {
    CK_GEMM_SCHEDULE_AUTO = 0,
    CK_GEMM_SCHEDULE_STATIC = 1,
    CK_GEMM_SCHEDULE_DYNAMIC = 2,
} ck_gemm_schedule_t;

/**
 * Set the process-wide GEMM tile scheduling policy.
 *
 * AUTO is the production default and currently selects dynamic work claiming
 * for providers whose jobs write independent output tiles. Providers with
 * ordered/shared reductions do not consult this policy.
 *
 * @return 0 on success, -1 for an invalid policy.
 */
int ck_set_gemm_schedule(int policy);

/** Return the configured process-wide GEMM scheduling policy. */
int ck_get_gemm_schedule(void);

/** Return non-zero when independent GEMM tiles should use dynamic claiming. */
int ck_gemm_dynamic_schedule_enabled(void);

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/**
 * Create a thread pool with `n_threads` total threads.
 * Thread 0 is the calling thread; n_threads-1 workers are spawned.
 *
 * @param n_threads  Total thread count (including main). Must be >= 1.
 *                   Pass 0 for auto-detect (physical cores).
 * @return Pool handle, or NULL on failure.
 */
ck_threadpool_t *ck_threadpool_create(int n_threads);

/**
 * Create a pool whose ordinary dispatch width is smaller than its worker
 * capacity. Exact providers may opt into the additional workers with
 * ck_threadpool_dispatch_n(); ordinary dispatch remains at default_threads.
 */
ck_threadpool_t *ck_threadpool_create_capacity(int default_threads,
                                                int capacity_threads);

/**
 * Compute the bounded capacity for an SMT-safe provider. The default width is
 * preserved and at most half of the additional logical CPUs are reserved.
 */
int ck_threadpool_bounded_capacity(int default_threads, int logical_threads);

/**
 * Destroy the thread pool. Signals all workers to exit and joins them.
 * Safe to call with NULL.
 */
void ck_threadpool_destroy(ck_threadpool_t *pool);

/* ============================================================================
 * Dispatch & Synchronization
 * ============================================================================ */

/**
 * Dispatch work to all threads and wait for completion.
 *
 * 1. Sets the work function and args
 * 2. Bumps the dispatch counter (wakes workers)
 * 3. Main thread (ith=0) executes its share
 * 4. Waits for all worker callbacks to complete
 *
 * This is a blocking call — returns when ALL threads have finished.
 * Concurrent multi-worker dispatches on the same pool are serialized.
 * Single-worker calls may execute concurrently because they do not publish
 * a shared work descriptor. Do not destroy a pool while a call is active or
 * recursively dispatch the same pool from one of its work functions.
 *
 * @param pool  Thread pool
 * @param fn    Work function (called on each thread)
 * @param args  Argument passed to fn
 */
void ck_threadpool_dispatch(ck_threadpool_t *pool, ck_work_fn_t fn, void *args);

/**
 * Dispatch work to a subset of the pool and wait for completion.
 *
 * Threads with ith >= active_threads remain idle for this dispatch.
 * The work function sees nth == active_threads.
 *
 * @param pool            Thread pool
 * @param active_threads  Number of active threads including the calling thread
 * @param fn              Work function
 * @param args            Argument passed to fn
 */
void ck_threadpool_dispatch_n(ck_threadpool_t *pool,
                              int active_threads,
                              ck_work_fn_t fn,
                              void *args);

/**
 * Dynamically distribute independent ranges through the persistent pool.
 *
 * Workers claim `grain_size` consecutive indices until [begin, end) is empty.
 * This changes ownership only; callers remain responsible for ensuring that
 * ranges write disjoint outputs and preserve each output's reduction order.
 * Do not use this helper for unordered shared reductions.
 */
void ck_threadpool_parallel_for_n(ck_threadpool_t *pool,
                                  int active_threads,
                                  int begin,
                                  int end,
                                  int grain_size,
                                  ck_range_fn_t fn,
                                  void *args);

/**
 * Barrier synchronization within a dispatched work function.
 *
 * ALL participants in the current dispatch must call this at the same point.
 * A single-worker dispatch returns immediately, even if the pool has more
 * workers or another caller is executing a multi-worker dispatch.
 *
 * Must only be called from within a work function (during dispatch).
 *
 * @param pool  Thread pool
 */
void ck_threadpool_barrier(ck_threadpool_t *pool);

/* ============================================================================
 * Power Management
 * ============================================================================ */

/**
 * Pause workers — they sleep on condvar (0% CPU).
 * Call between batches or during interactive waiting.
 * Workers wake on next dispatch or resume.
 */
void ck_threadpool_pause(ck_threadpool_t *pool);

/**
 * Resume workers — transition from sleep to spin-wait.
 * Call before starting a new batch of work.
 */
void ck_threadpool_resume(ck_threadpool_t *pool);

/* ============================================================================
 * Queries
 * ============================================================================ */

/** Get the ordinary/default dispatch width (including the calling thread). */
int ck_threadpool_n_threads(const ck_threadpool_t *pool);

/** Get the maximum worker capacity available to explicit dispatch_n calls. */
int ck_threadpool_capacity(const ck_threadpool_t *pool);

/**
 * Get the current callback's ith for this pool. Outside a callback, return
 * the physical worker index (or the pool creator's legacy index 0), and -1
 * for an unrelated thread. The current dispatch caller is always index 0,
 * including callers that did not create the pool.
 */
int ck_threadpool_thread_id(const ck_threadpool_t *pool);

/** True inside a callback on this pool, including its calling-thread slot 0. */
int ck_threadpool_in_dispatch(const ck_threadpool_t *pool);

/** Enable profiling and reset cumulative dispatch timing counters. */
void ck_threadpool_profile_reset(ck_threadpool_t *pool);

/** Snapshot cumulative dispatch timing counters without stopping workers. */
void ck_threadpool_profile_snapshot(
    const ck_threadpool_t *pool, ck_threadpool_profile_t *profile);

/* ============================================================================
 * Global Thread Pool (convenience)
 * ============================================================================ */

/**
 * Get or create the global thread pool.
 * Thread-safe (uses pthread_once internally).
 * Uses ck_get_num_threads() for the default width. In automatic mode the pool
 * may reserve a bounded subset of SMT siblings as explicit provider capacity;
 * ordinary dispatch remains at the default width.
 *
 * @return Global pool, never NULL after successful first call.
 */
ck_threadpool_t *ck_threadpool_global(void);

/**
 * Destroy the global thread pool.
 * Called during engine shutdown.
 */
void ck_threadpool_global_destroy(void);

#ifdef __cplusplus
}
#endif

#endif /* CK_THREADPOOL_H */
