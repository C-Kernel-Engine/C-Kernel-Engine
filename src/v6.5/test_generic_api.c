/**
 * @file test_generic_api.c
 * @brief Generic test/benchmark harness using ck_model_* API
 *
 * This file works with ANY model - just link with different inference.c
 *
 * Build:
 *   gcc test_generic_api.c inference.c kernels/*.c -o test_bench -lm -lpthread
 *
 * Run:
 *   ./test_bench --weights weights.bump --benchmark 100
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "ck_model_api.h"

/* ============================================================================
 * TIMING UTILITIES
 * ============================================================================ */

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ============================================================================
 * WEIGHT LOADING
 * ============================================================================ */

static int load_weights_from_bump(void *model, const char *bump_path) {
    int fd = open(bump_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[ERROR] Cannot open: %s\n", bump_path);
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return -1;
    }

    size_t file_size = st.st_size;
    size_t model_bytes = ck_model_get_total_bytes(model);
    void *base = ck_model_get_base(model);

    /* Map file */
    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED) {
        fprintf(stderr, "[ERROR] mmap failed\n");
        return -1;
    }

    /* Copy weights (skip 64-byte header in BUMP file) */
    size_t header_size = 64;
    size_t weight_bytes = ck_model_get_config()->weight_bytes;

    if (file_size < header_size + weight_bytes) {
        fprintf(stderr, "[ERROR] BUMP file too small: %zu < %zu\n",
                file_size, header_size + weight_bytes);
        munmap(mapped, file_size);
        return -1;
    }

    /* Copy weights to model base (after header) */
    memcpy((char*)base + header_size, (char*)mapped + header_size, weight_bytes);

    munmap(mapped, file_size);
    printf("[INFO] Loaded %zu bytes from %s\n", weight_bytes, bump_path);
    return 0;
}

/* ============================================================================
 * SAMPLING
 * ============================================================================ */

static int sample_argmax(const float *logits, int vocab_size) {
    int best_idx = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

/* ============================================================================
 * BENCHMARK
 * ============================================================================ */

static void run_benchmark(void *model, int num_tokens) {
    const CKModelConfig *cfg = ck_model_get_config();

    printf("\n");
    printf("============================================\n");
    printf("  BENCHMARK: %s\n", cfg->model_name);
    printf("============================================\n");
    printf("  Layers:      %d\n", cfg->num_layers);
    printf("  Embed dim:   %d\n", cfg->embed_dim);
    printf("  Heads:       %d (KV: %d)\n", cfg->num_heads, cfg->num_kv_heads);
    printf("  Vocab:       %d\n", cfg->vocab_size);
    printf("  Tokens:      %d\n", num_tokens);
    printf("============================================\n\n");

    /* Warmup */
    printf("[WARMUP] Running 3 warmup iterations...\n");
    int token = 1;
    for (int i = 0; i < 3; i++) {
        ck_model_decode(model, &token, i);
    }

    /* Benchmark decode */
    printf("[BENCH] Running %d decode iterations...\n", num_tokens);

    double start = get_time_ms();
    for (int i = 0; i < num_tokens; i++) {
        ck_model_decode(model, &token, i + 3);  /* Offset by warmup */
    }
    double end = get_time_ms();

    double elapsed_ms = end - start;
    double tokens_per_sec = num_tokens / (elapsed_ms / 1000.0);
    double ms_per_token = elapsed_ms / num_tokens;

    printf("\n");
    printf("============================================\n");
    printf("  RESULTS\n");
    printf("============================================\n");
    printf("  Total time:     %.2f ms\n", elapsed_ms);
    printf("  Tokens/sec:     %.2f\n", tokens_per_sec);
    printf("  ms/token:       %.2f\n", ms_per_token);
    printf("============================================\n");

    /* Verify canaries */
    int errors = ck_model_verify_canaries(model);
    if (errors > 0) {
        printf("[WARN] %d canary corruptions detected!\n", errors);
    } else {
        printf("[OK] Memory canaries intact\n");
    }
}

/* ============================================================================
 * SIMPLE GENERATION TEST
 * ============================================================================ */

static void run_generation_test(void *model, int num_tokens) {
    const CKModelConfig *cfg = ck_model_get_config();

    printf("\n[TEST] Generation test (%d tokens)...\n", num_tokens);

    /* Start with token 1 (usually <s> or similar) */
    int token = 1;

    printf("[GEN] Token IDs: ");
    for (int i = 0; i < num_tokens; i++) {
        ck_model_decode(model, &token, i);
        float *logits = ck_model_get_logits(model);
        token = sample_argmax(logits, cfg->vocab_size);
        printf("%d ", token);
        fflush(stdout);
    }
    printf("\n");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --weights <path>    Path to weights.bump file\n");
    printf("  --benchmark <n>     Run benchmark with n tokens (default: 100)\n");
    printf("  --generate <n>      Run generation test with n tokens\n");
    printf("  --info              Print model info and exit\n");
    printf("  --help              Show this help\n");
}

int main(int argc, char **argv) {
    const char *weights_path = NULL;
    int benchmark_tokens = 0;
    int generate_tokens = 0;
    int info_only = 0;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
            benchmark_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--info") == 0) {
            info_only = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* Print model info */
    const CKModelConfig *cfg = ck_model_get_config();
    printf("\n");
    printf("============================================\n");
    printf("  CK-Engine Generic Test Harness\n");
    printf("============================================\n");
    printf("  Model:       %s\n", cfg->model_name);
    printf("  Family:      %s\n", cfg->model_family);
    printf("  Layers:      %d\n", cfg->num_layers);
    printf("  Embed:       %d\n", cfg->embed_dim);
    printf("  Heads:       %d / %d (Q/KV)\n", cfg->num_heads, cfg->num_kv_heads);
    printf("  Intermediate:%d\n", cfg->intermediate_size);
    printf("  Vocab:       %d\n", cfg->vocab_size);
    printf("  Max seq:     %d\n", cfg->max_seq_len);
    printf("  Total mem:   %.2f GB\n", cfg->total_bytes / 1e9);
    printf("  Weight mem:  %.2f GB\n", cfg->weight_bytes / 1e9);
    printf("============================================\n");

    if (info_only) {
        return 0;
    }

    if (!weights_path) {
        fprintf(stderr, "[ERROR] --weights required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Create model */
    printf("\n[INIT] Creating model...\n");
    void *model = ck_model_create();
    if (!model) {
        fprintf(stderr, "[ERROR] Failed to create model\n");
        return 1;
    }
    printf("[INIT] Model created (%.2f GB allocated)\n", cfg->total_bytes / 1e9);

    /* Load weights */
    printf("[INIT] Loading weights from %s...\n", weights_path);
    if (load_weights_from_bump(model, weights_path) != 0) {
        fprintf(stderr, "[ERROR] Failed to load weights\n");
        ck_model_free(model);
        return 1;
    }

    /* Precompute RoPE */
    printf("[INIT] Precomputing RoPE...\n");
    ck_model_precompute_rope(model);

    /* Run tests */
    if (benchmark_tokens > 0) {
        run_benchmark(model, benchmark_tokens);
    }

    if (generate_tokens > 0) {
        run_generation_test(model, generate_tokens);
    }

    if (benchmark_tokens == 0 && generate_tokens == 0) {
        /* Default: quick benchmark */
        run_benchmark(model, 100);
    }

    /* Cleanup */
    printf("\n[CLEANUP] Freeing model...\n");
    ck_model_free(model);
    printf("[DONE]\n");

    return 0;
}
