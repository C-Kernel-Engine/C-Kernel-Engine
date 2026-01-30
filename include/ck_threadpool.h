/**
 * @file ck_threadpool.h
 * @brief Persistent pthread thread pool for CK-Engine inference
 *
 * Design goals:
 * - Sub-microsecond dispatch latency (spin-wait barriers)
 * - Zero allocation after init (all memory pre-allocated)
 * - Cache-line aligned atomics to avoid false sharing
 * - Hybrid polling: spin N rounds, then fall back to condvar
 * - Thread 0 = main thread (does serial ops + its share of parallel work)
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
 * Called on ALL threads (including main thread 0).
 *
 * @param ith   Thread index (0 = main thread)
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

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

/**
 * Create a thread pool with `n_threads` total threads.
 * Thread 0 is the calling (main) thread; n_threads-1 workers are spawned.
 *
 * @param n_threads  Total thread count (including main). Must be >= 1.
 *                   Pass 0 for auto-detect (physical cores).
 * @return Pool handle, or NULL on failure.
 */
ck_threadpool_t *ck_threadpool_create(int n_threads);

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
 * 4. Waits for all threads to complete via barrier
 *
 * This is a blocking call — returns when ALL threads have finished.
 *
 * @param pool  Thread pool
 * @param fn    Work function (called on each thread)
 * @param args  Argument passed to fn
 */
void ck_threadpool_dispatch(ck_threadpool_t *pool, ck_work_fn_t fn, void *args);

/**
 * Barrier synchronization within a dispatched work function.
 *
 * ALL threads must call this at the same point. Threads spin-wait
 * until all have arrived, then proceed.
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

/** Get total thread count (including main thread) */
int ck_threadpool_n_threads(const ck_threadpool_t *pool);

/** Get thread index for current thread (0 = main, -1 if not in pool) */
int ck_threadpool_thread_id(const ck_threadpool_t *pool);

/* ============================================================================
 * Global Thread Pool (convenience)
 * ============================================================================ */

/**
 * Get or create the global thread pool.
 * Thread-safe (uses pthread_once internally).
 * Uses ck_get_num_threads() for auto-detection.
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
