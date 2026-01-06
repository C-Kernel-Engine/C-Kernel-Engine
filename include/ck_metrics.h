/**
 * ck_metrics.h - C-Kernel Engine Training Metrics API
 *
 * Lightweight metrics logging for real-time training dashboards.
 * Sends metrics to the ANTSAND live training dashboard via HTTP POST
 * or writes to a local file for batch upload.
 *
 * Usage:
 *   ck_metrics_init("run_001", "http://localhost/labs/metrics/log");
 *
 *   for (int step = 0; step < max_steps; step++) {
 *       // ... training step ...
 *       ck_metrics_log_f("loss", loss);
 *       ck_metrics_log_f("lr", lr);
 *       ck_metrics_log_f("grad_norm", grad_norm);
 *       ck_metrics_log_i("tokens_per_sec", tokens_per_sec);
 *       ck_metrics_log_i("memory_mb", get_memory_usage_mb());
 *       ck_metrics_step(step);  // Flush metrics for this step
 *   }
 *
 *   ck_metrics_end("completed");
 *
 * Build:
 *   Link with -lcurl for HTTP mode, or use file mode for no dependencies.
 */

#ifndef CK_METRICS_H
#define CK_METRICS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

#define CK_METRICS_MAX_NAME_LEN    64
#define CK_METRICS_MAX_METRICS     32
#define CK_METRICS_MAX_URL_LEN     256
#define CK_METRICS_BUFFER_SIZE     4096

// Transport modes
typedef enum {
    CK_METRICS_MODE_HTTP,      // Send via HTTP POST (requires libcurl)
    CK_METRICS_MODE_FILE,      // Write to local JSONL file
    CK_METRICS_MODE_STDOUT,    // Print to stdout (debug)
    CK_METRICS_MODE_DISABLED   // No-op (for benchmarking without overhead)
} CKMetricsMode;

// Metric types
typedef enum {
    CK_METRIC_FLOAT,
    CK_METRIC_INT,
    CK_METRIC_STRING
} CKMetricType;

// Single metric entry
typedef struct {
    char name[CK_METRICS_MAX_NAME_LEN];
    CKMetricType type;
    union {
        double f;
        int64_t i;
        char s[CK_METRICS_MAX_NAME_LEN];
    } value;
} CKMetric;

// Metrics context
typedef struct {
    char run_id[CK_METRICS_MAX_NAME_LEN];
    char endpoint[CK_METRICS_MAX_URL_LEN];
    char file_path[CK_METRICS_MAX_URL_LEN];
    CKMetricsMode mode;

    // Current step metrics
    CKMetric metrics[CK_METRICS_MAX_METRICS];
    int metric_count;
    int64_t current_step;

    // Config (sent once at start)
    char model_name[CK_METRICS_MAX_NAME_LEN];
    char dataset_name[CK_METRICS_MAX_NAME_LEN];
    int batch_size;
    double learning_rate;
    int max_steps;

    // Internal state
    bool initialized;
    void* curl_handle;  // libcurl easy handle
    FILE* file_handle;  // For file mode
} CKMetricsContext;

// Global context (for convenience API)
extern CKMetricsContext* ck_metrics_ctx;

// ═══════════════════════════════════════════════════════════════════════════════
// Core API
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Initialize metrics logging
 *
 * @param run_id     Unique identifier for this training run (e.g., "run_20260104_093000")
 * @param endpoint   HTTP endpoint or file path depending on mode
 * @param mode       Transport mode (HTTP, FILE, STDOUT, DISABLED)
 * @return           true on success
 */
bool ck_metrics_init(const char* run_id, const char* endpoint, CKMetricsMode mode);

/**
 * Initialize with full configuration
 */
bool ck_metrics_init_full(const char* run_id, const char* endpoint, CKMetricsMode mode,
                          const char* model, const char* dataset,
                          int batch_size, double lr, int max_steps);

/**
 * Log a float metric (e.g., loss, learning rate)
 */
void ck_metrics_log_f(const char* name, double value);

/**
 * Log an integer metric (e.g., step, tokens_per_sec)
 */
void ck_metrics_log_i(const char* name, int64_t value);

/**
 * Log a string metric (e.g., phase, status)
 */
void ck_metrics_log_s(const char* name, const char* value);

/**
 * Flush metrics for the current step and advance to next step
 * Call this at the end of each training step
 *
 * @param step    Current training step number
 */
void ck_metrics_step(int64_t step);

/**
 * End the training run
 *
 * @param status  Final status ("completed", "failed", "interrupted")
 */
void ck_metrics_end(const char* status);

/**
 * Cleanup and free resources
 */
void ck_metrics_cleanup(void);

// ═══════════════════════════════════════════════════════════════════════════════
// Context-based API (for multiple concurrent runs)
// ═══════════════════════════════════════════════════════════════════════════════

CKMetricsContext* ck_metrics_create_context(void);
void ck_metrics_destroy_context(CKMetricsContext* ctx);

bool ck_metrics_ctx_init(CKMetricsContext* ctx, const char* run_id,
                         const char* endpoint, CKMetricsMode mode);
void ck_metrics_ctx_log_f(CKMetricsContext* ctx, const char* name, double value);
void ck_metrics_ctx_log_i(CKMetricsContext* ctx, const char* name, int64_t value);
void ck_metrics_ctx_step(CKMetricsContext* ctx, int64_t step);
void ck_metrics_ctx_end(CKMetricsContext* ctx, const char* status);

// ═══════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Generate a unique run ID based on current timestamp
 */
void ck_metrics_generate_run_id(char* buffer, size_t size);

/**
 * Get current memory usage in MB (platform-specific)
 */
int64_t ck_metrics_get_memory_mb(void);

/**
 * Get current timestamp in seconds with microsecond precision
 */
double ck_metrics_timestamp(void);

#ifdef __cplusplus
}
#endif

#endif // CK_METRICS_H

// ═══════════════════════════════════════════════════════════════════════════════
// IMPLEMENTATION (define CK_METRICS_IMPLEMENTATION in ONE .c file)
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef CK_METRICS_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

// Optional: libcurl for HTTP mode
#ifdef CK_METRICS_USE_CURL
#include <curl/curl.h>
#endif

// Global context
CKMetricsContext* ck_metrics_ctx = NULL;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static void metrics_build_json(CKMetricsContext* ctx, char* buffer, size_t size) {
    int offset = 0;
    offset += snprintf(buffer + offset, size - offset,
        "{\"run_id\":\"%s\",\"step\":%lld,\"timestamp\":%.6f,\"metrics\":{",
        ctx->run_id, (long long)ctx->current_step, ck_metrics_timestamp());

    for (int i = 0; i < ctx->metric_count; i++) {
        CKMetric* m = &ctx->metrics[i];
        if (i > 0) offset += snprintf(buffer + offset, size - offset, ",");

        switch (m->type) {
            case CK_METRIC_FLOAT:
                offset += snprintf(buffer + offset, size - offset,
                    "\"%s\":%.6g", m->name, m->value.f);
                break;
            case CK_METRIC_INT:
                offset += snprintf(buffer + offset, size - offset,
                    "\"%s\":%lld", m->name, (long long)m->value.i);
                break;
            case CK_METRIC_STRING:
                offset += snprintf(buffer + offset, size - offset,
                    "\"%s\":\"%s\"", m->name, m->value.s);
                break;
        }
    }

    snprintf(buffer + offset, size - offset, "}}");
}

static void metrics_send_http(CKMetricsContext* ctx, const char* json) {
#ifdef CK_METRICS_USE_CURL
    if (!ctx->curl_handle) return;

    CURL* curl = (CURL*)ctx->curl_handle;
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, ctx->endpoint);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1L);  // 1 second timeout

    curl_easy_perform(curl);
    curl_slist_free_all(headers);
#else
    (void)ctx; (void)json;
    // HTTP mode requires libcurl
#endif
}

static void metrics_write_file(CKMetricsContext* ctx, const char* json) {
    if (!ctx->file_handle) return;
    fprintf(ctx->file_handle, "%s\n", json);
    fflush(ctx->file_handle);
}

static void metrics_flush(CKMetricsContext* ctx) {
    if (ctx->metric_count == 0) return;

    char buffer[CK_METRICS_BUFFER_SIZE];
    metrics_build_json(ctx, buffer, sizeof(buffer));

    switch (ctx->mode) {
        case CK_METRICS_MODE_HTTP:
            metrics_send_http(ctx, buffer);
            break;
        case CK_METRICS_MODE_FILE:
            metrics_write_file(ctx, buffer);
            break;
        case CK_METRICS_MODE_STDOUT:
            printf("%s\n", buffer);
            break;
        case CK_METRICS_MODE_DISABLED:
            break;
    }

    ctx->metric_count = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

bool ck_metrics_init(const char* run_id, const char* endpoint, CKMetricsMode mode) {
    return ck_metrics_init_full(run_id, endpoint, mode, "unknown", "unknown", 0, 0, 0);
}

bool ck_metrics_init_full(const char* run_id, const char* endpoint, CKMetricsMode mode,
                          const char* model, const char* dataset,
                          int batch_size, double lr, int max_steps) {
    if (!ck_metrics_ctx) {
        ck_metrics_ctx = ck_metrics_create_context();
    }

    CKMetricsContext* ctx = ck_metrics_ctx;

    strncpy(ctx->run_id, run_id, CK_METRICS_MAX_NAME_LEN - 1);
    strncpy(ctx->endpoint, endpoint, CK_METRICS_MAX_URL_LEN - 1);
    strncpy(ctx->model_name, model, CK_METRICS_MAX_NAME_LEN - 1);
    strncpy(ctx->dataset_name, dataset, CK_METRICS_MAX_NAME_LEN - 1);
    ctx->mode = mode;
    ctx->batch_size = batch_size;
    ctx->learning_rate = lr;
    ctx->max_steps = max_steps;
    ctx->metric_count = 0;
    ctx->current_step = 0;

    switch (mode) {
        case CK_METRICS_MODE_HTTP:
#ifdef CK_METRICS_USE_CURL
            curl_global_init(CURL_GLOBAL_DEFAULT);
            ctx->curl_handle = curl_easy_init();
#endif
            break;
        case CK_METRICS_MODE_FILE:
            ctx->file_handle = fopen(endpoint, "a");
            if (!ctx->file_handle) return false;
            break;
        default:
            break;
    }

    ctx->initialized = true;
    return true;
}

void ck_metrics_log_f(const char* name, double value) {
    if (!ck_metrics_ctx || !ck_metrics_ctx->initialized) return;
    ck_metrics_ctx_log_f(ck_metrics_ctx, name, value);
}

void ck_metrics_log_i(const char* name, int64_t value) {
    if (!ck_metrics_ctx || !ck_metrics_ctx->initialized) return;
    ck_metrics_ctx_log_i(ck_metrics_ctx, name, value);
}

void ck_metrics_log_s(const char* name, const char* value) {
    if (!ck_metrics_ctx || !ck_metrics_ctx->initialized) return;
    CKMetricsContext* ctx = ck_metrics_ctx;
    if (ctx->metric_count >= CK_METRICS_MAX_METRICS) return;

    CKMetric* m = &ctx->metrics[ctx->metric_count++];
    strncpy(m->name, name, CK_METRICS_MAX_NAME_LEN - 1);
    m->type = CK_METRIC_STRING;
    strncpy(m->value.s, value, CK_METRICS_MAX_NAME_LEN - 1);
}

void ck_metrics_step(int64_t step) {
    if (!ck_metrics_ctx || !ck_metrics_ctx->initialized) return;
    ck_metrics_ctx->current_step = step;
    metrics_flush(ck_metrics_ctx);
}

void ck_metrics_end(const char* status) {
    if (!ck_metrics_ctx || !ck_metrics_ctx->initialized) return;

    // Log final status
    ck_metrics_log_s("status", status);
    metrics_flush(ck_metrics_ctx);

    ck_metrics_cleanup();
}

void ck_metrics_cleanup(void) {
    if (!ck_metrics_ctx) return;

#ifdef CK_METRICS_USE_CURL
    if (ck_metrics_ctx->curl_handle) {
        curl_easy_cleanup((CURL*)ck_metrics_ctx->curl_handle);
        curl_global_cleanup();
    }
#endif

    if (ck_metrics_ctx->file_handle) {
        fclose(ck_metrics_ctx->file_handle);
    }

    ck_metrics_destroy_context(ck_metrics_ctx);
    ck_metrics_ctx = NULL;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context-based API
// ─────────────────────────────────────────────────────────────────────────────

CKMetricsContext* ck_metrics_create_context(void) {
    CKMetricsContext* ctx = (CKMetricsContext*)calloc(1, sizeof(CKMetricsContext));
    return ctx;
}

void ck_metrics_destroy_context(CKMetricsContext* ctx) {
    if (ctx) free(ctx);
}

bool ck_metrics_ctx_init(CKMetricsContext* ctx, const char* run_id,
                         const char* endpoint, CKMetricsMode mode) {
    if (!ctx) return false;
    strncpy(ctx->run_id, run_id, CK_METRICS_MAX_NAME_LEN - 1);
    strncpy(ctx->endpoint, endpoint, CK_METRICS_MAX_URL_LEN - 1);
    ctx->mode = mode;
    ctx->initialized = true;
    return true;
}

void ck_metrics_ctx_log_f(CKMetricsContext* ctx, const char* name, double value) {
    if (!ctx || ctx->metric_count >= CK_METRICS_MAX_METRICS) return;
    CKMetric* m = &ctx->metrics[ctx->metric_count++];
    strncpy(m->name, name, CK_METRICS_MAX_NAME_LEN - 1);
    m->type = CK_METRIC_FLOAT;
    m->value.f = value;
}

void ck_metrics_ctx_log_i(CKMetricsContext* ctx, const char* name, int64_t value) {
    if (!ctx || ctx->metric_count >= CK_METRICS_MAX_METRICS) return;
    CKMetric* m = &ctx->metrics[ctx->metric_count++];
    strncpy(m->name, name, CK_METRICS_MAX_NAME_LEN - 1);
    m->type = CK_METRIC_INT;
    m->value.i = value;
}

void ck_metrics_ctx_step(CKMetricsContext* ctx, int64_t step) {
    if (!ctx) return;
    ctx->current_step = step;
    metrics_flush(ctx);
}

void ck_metrics_ctx_end(CKMetricsContext* ctx, const char* status) {
    if (!ctx) return;
    ck_metrics_ctx_log_f(ctx, "status", 0);  // Simplified
    metrics_flush(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────────────────────────────────────

void ck_metrics_generate_run_id(char* buffer, size_t size) {
    time_t t = time(NULL);
    struct tm* tm = localtime(&t);
    snprintf(buffer, size, "run_%04d%02d%02d_%02d%02d%02d",
        tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
        tm->tm_hour, tm->tm_min, tm->tm_sec);
}

int64_t ck_metrics_get_memory_mb(void) {
#ifdef __linux__
    FILE* f = fopen("/proc/self/statm", "r");
    if (f) {
        long pages = 0;
        fscanf(f, "%ld", &pages);
        fclose(f);
        return (pages * sysconf(_SC_PAGESIZE)) / (1024 * 1024);
    }
#endif
    return 0;
}

double ck_metrics_timestamp(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

#endif // CK_METRICS_IMPLEMENTATION
