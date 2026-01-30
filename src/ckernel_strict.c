#include "ckernel_engine.h"
#include "ck_threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(USE_MKL)
#include <mkl.h>
#endif

// =============================================================================
// Strict parity mode (for numerical reproducibility)
// =============================================================================

static int ck_strict_parity = 0;

void ck_set_strict_parity(int enabled)
{
    ck_strict_parity = enabled ? 1 : 0;
#ifdef _OPENMP
    if (ck_strict_parity) {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
    }
#endif
}

int ck_strict_parity_enabled(void)
{
    return ck_strict_parity;
}

// =============================================================================
// Thread configuration
// =============================================================================

static int g_num_threads = 0;
static int g_threads_initialized = 0;

static int ck_parse_env_int(const char *name)
{
    const char *val = getenv(name);
    if (!val || !val[0]) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    long n = strtol(val, &end, 10);
    if (errno != 0 || end == val || n <= 0 || n > (1L << 20)) {
        return 0;
    }
    return (int)n;
}

// Detect physical CPU cores (not hyperthreads) when possible.
int ck_get_physical_cores(void)
{
    int physical_cores = 0;
    int logical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (logical_cores <= 0) {
        logical_cores = 1;
    }

    // Read from /proc/cpuinfo (Linux) and count unique (physical id, core id) pairs.
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        int physical_id = -1;
        int core_id = -1;

        struct {
            int physical_id;
            int core_id;
        } seen[8192];
        int seen_count = 0;

        const int seen_cap = (int)(sizeof(seen) / sizeof(seen[0]));

        // Helper: add (pid,cid) to set if not present.
        #define CK_ADD_PAIR(pid, cid)                                            \
            do {                                                                 \
                if ((pid) >= 0 && (cid) >= 0) {                                  \
                    int exists = 0;                                              \
                    for (int ii = 0; ii < seen_count; ++ii) {                    \
                        if (seen[ii].physical_id == (pid) &&                     \
                            seen[ii].core_id == (cid)) {                         \
                            exists = 1;                                          \
                            break;                                               \
                        }                                                        \
                    }                                                            \
                    if (!exists && seen_count < seen_cap) {                      \
                        seen[seen_count].physical_id = (pid);                    \
                        seen[seen_count].core_id = (cid);                        \
                        ++seen_count;                                            \
                    }                                                            \
                }                                                                \
            } while (0)

        while (fgets(line, sizeof(line), f)) {
            int val;

            // Blank line separates processor blocks.
            if (line[0] == '\n' || line[0] == '\0') {
                CK_ADD_PAIR(physical_id, core_id);
                physical_id = -1;
                core_id = -1;
                continue;
            }

            if (sscanf(line, "physical id : %d", &val) == 1) {
                physical_id = val;
                continue;
            }
            if (sscanf(line, "core id : %d", &val) == 1) {
                core_id = val;
                continue;
            }
        }
        fclose(f);

        // Handle file without trailing blank line.
        CK_ADD_PAIR(physical_id, core_id);

        #undef CK_ADD_PAIR

        physical_cores = seen_count;
    }

    // If we couldn't reliably detect physical cores (common in containers),
    // fall back to logical CPUs instead of incorrectly forcing single-thread execution.
    if (physical_cores <= 1 && logical_cores > 1) {
        return logical_cores;
    }

    if (physical_cores > 1) {
        return physical_cores;
    }

    return logical_cores;
}

void ck_set_num_threads(int num_threads)
{
    // 0 = auto-detect
    if (num_threads <= 0) {
        // Prefer explicit env controls when present:
        // - CK_NUM_THREADS: engine-level override
        // - OMP_NUM_THREADS: standard OpenMP control (set by `ck run --threads`)
        int env_threads = ck_parse_env_int("CK_NUM_THREADS");
        if (env_threads <= 0) {
            env_threads = ck_parse_env_int("OMP_NUM_THREADS");
        }
        num_threads = env_threads > 0 ? env_threads : ck_get_physical_cores();
    }

    g_num_threads = num_threads;
    g_threads_initialized = 1;

#ifdef _OPENMP
    omp_set_dynamic(0);  // Disable dynamic adjustment
    omp_set_num_threads(num_threads);
#endif

#if defined(USE_MKL)
    mkl_set_num_threads(num_threads);
#endif

    fprintf(stderr, "[CK] Set %d threads (auto=%d)\n",
            num_threads, ck_get_physical_cores());
}

int ck_get_num_threads(void)
{
    // Auto-initialize if not set
    if (!g_threads_initialized) {
        ck_set_num_threads(0);  // Auto-detect
    }
    return g_num_threads;
}

// =============================================================================
// Thread pool lifecycle
// =============================================================================

/**
 * Initialize the global thread pool.
 * Called once during engine startup (e.g., from ck_model_init).
 * Uses ck_get_num_threads() for thread count (respects CK_NUM_THREADS env).
 *
 * Safe to call multiple times — subsequent calls are no-ops.
 */
void ck_threadpool_init(void)
{
    /* ck_threadpool_global() uses pthread_once internally */
    ck_threadpool_t *pool = ck_threadpool_global();
    (void)pool;
}

/**
 * Shut down the global thread pool.
 * Called during engine teardown. Workers are joined and freed.
 */
void ck_threadpool_shutdown(void)
{
    ck_threadpool_global_destroy();
}

/**
 * Get the global thread pool handle for dispatch.
 * Convenience wrapper — initializes on first call.
 */
ck_threadpool_t *ck_get_threadpool(void)
{
    return ck_threadpool_global();
}
