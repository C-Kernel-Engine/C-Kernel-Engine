#include "ckernel_engine.h"
#include "ck_threadpool.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
static float *ck_strict_next_gemm_a = NULL;
static size_t ck_strict_next_gemm_a_size = 0;
static size_t ck_strict_next_gemm_a_cap = 0;
static int ck_strict_next_gemm_a_valid = 0;

void ck_set_strict_parity(int enabled)
{
    ck_strict_parity = enabled ? 1 : 0;
    if (!ck_strict_parity) {
        ck_strict_next_gemm_a_valid = 0;
        ck_strict_next_gemm_a_size = 0;
    }
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

void ck_strict_store_next_gemm_a(const float *data, size_t elems)
{
    if (!ck_strict_parity || !data || elems == 0) {
        ck_strict_next_gemm_a_valid = 0;
        ck_strict_next_gemm_a_size = 0;
        return;
    }
    if (elems > ck_strict_next_gemm_a_cap) {
        float *next = (float *) realloc(ck_strict_next_gemm_a, elems * sizeof(float));
        if (!next) {
            ck_strict_next_gemm_a_valid = 0;
            ck_strict_next_gemm_a_size = 0;
            return;
        }
        ck_strict_next_gemm_a = next;
        ck_strict_next_gemm_a_cap = elems;
    }
    memcpy(ck_strict_next_gemm_a, data, elems * sizeof(float));
    ck_strict_next_gemm_a_size = elems;
    ck_strict_next_gemm_a_valid = 1;
}

const float *ck_strict_consume_next_gemm_a(size_t elems)
{
    if (!ck_strict_parity || !ck_strict_next_gemm_a_valid || ck_strict_next_gemm_a_size != elems) {
        return NULL;
    }
    ck_strict_next_gemm_a_valid = 0;
    return ck_strict_next_gemm_a;
}

typedef void *(*ck_strict_mtmd_clip_init_fn)(const char *, int, int, int, int, int);
typedef void (*ck_strict_mtmd_clip_free_fn)(void *);
typedef size_t (*ck_strict_mtmd_clip_embd_nbytes_by_img_fn)(void *, int, int);
typedef int (*ck_strict_mtmd_clip_encode_float_image_fn)(void *, int, float *, int, int, float *);

int ck_strict_mtmd_clip_encode_planar_f32(const float *planar,
                                          int channels,
                                          int height,
                                          int width,
                                          float *out,
                                          size_t out_elems)
{
    const char *gguf_path = getenv("CK_STRICT_GGUF_PATH");
    const char *shim_path = getenv("CK_STRICT_MTMD_SHIM_SO");
    if (!gguf_path || !gguf_path[0] || !shim_path || !shim_path[0]) {
        return 0;
    }
    if (!planar || !out || channels != 3 || height <= 0 || width <= 0) {
        return 0;
    }

    void *shim = dlopen(shim_path, RTLD_LAZY | RTLD_LOCAL);
    if (!shim) {
        return 0;
    }

    ck_strict_mtmd_clip_init_fn init_fn =
        (ck_strict_mtmd_clip_init_fn) dlsym(shim, "ck_mtmd_clip_init");
    ck_strict_mtmd_clip_free_fn free_fn =
        (ck_strict_mtmd_clip_free_fn) dlsym(shim, "ck_mtmd_clip_free");
    ck_strict_mtmd_clip_embd_nbytes_by_img_fn embd_nbytes_fn =
        (ck_strict_mtmd_clip_embd_nbytes_by_img_fn) dlsym(shim, "ck_mtmd_clip_embd_nbytes_by_img");
    ck_strict_mtmd_clip_encode_float_image_fn encode_fn =
        (ck_strict_mtmd_clip_encode_float_image_fn) dlsym(shim, "ck_mtmd_clip_encode_float_image");

    if (!init_fn || !free_fn || !embd_nbytes_fn || !encode_fn) {
        dlclose(shim);
        return 0;
    }

    const size_t pixel_count = (size_t) height * (size_t) width;
    float *interleaved = (float *) malloc(pixel_count * (size_t) channels * sizeof(float));
    if (!interleaved) {
        dlclose(shim);
        return 0;
    }
    for (size_t idx = 0; idx < pixel_count; ++idx) {
        interleaved[idx * 3 + 0] = planar[idx];
        interleaved[idx * 3 + 1] = planar[pixel_count + idx];
        interleaved[idx * 3 + 2] = planar[2 * pixel_count + idx];
    }

    int ok = 0;
    void *handle = init_fn(gguf_path, 0, 0, 0, 0, 0);
    if (handle) {
        const size_t needed_bytes = embd_nbytes_fn(handle, width, height);
        if (needed_bytes > 0 && needed_bytes <= out_elems * sizeof(float)) {
            ok = encode_fn(handle, 1, interleaved, height, width, out) ? 1 : 0;
        }
        free_fn(handle);
    }

    free(interleaved);
    dlclose(shim);
    return ok;
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

    int cpu_cores_hint = 0;
    int siblings_hint = 0;

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
            if (sscanf(line, "cpu cores : %d", &val) == 1) {
                if (val > cpu_cores_hint) cpu_cores_hint = val;
                continue;
            }
            if (sscanf(line, "siblings : %d", &val) == 1) {
                if (val > siblings_hint) siblings_hint = val;
                continue;
            }
        }
        fclose(f);

        // Handle file without trailing blank line.
        CK_ADD_PAIR(physical_id, core_id);

        #undef CK_ADD_PAIR

        physical_cores = seen_count;
    }

    // Fallback: infer threads-per-core from siblings/cpu cores when pair data
    // is missing (common in containers/VMs).
    if (physical_cores <= 1 && logical_cores > 1) {
        int threads_per_core = 0;
        if (siblings_hint > 0 && cpu_cores_hint > 0 && siblings_hint >= cpu_cores_hint) {
            threads_per_core = siblings_hint / cpu_cores_hint;
        }
        if (threads_per_core > 1) {
            int inferred_physical = logical_cores / threads_per_core;
            if (inferred_physical > 1) {
                return inferred_physical;
            }
        }
        if (cpu_cores_hint > 1 && cpu_cores_hint <= logical_cores) {
            return cpu_cores_hint;
        }
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
