/**
 * @file ck_cli_template.c
 * @brief C-Kernel-Engine v7 CLI Template
 *
 * This is a standalone CLI template that demonstrates the proper usage pattern:
 *   1. Allocate memory (via ck_model_init)
 *   2. Load weights (from bump file)
 *   3. Initialize tokenizer (from vocab data in model)
 *   4. Wait for user prompt
 *   5. Tokenize input
 *   6. Run prefill (embed + forward)
 *   7. Run decode loop (sample + decode)
 *
 * Build:
 *   gcc -O3 -o ck-cli ck_cli_template.c -ldl -lm
 *
 * Usage:
 *   ./ck-cli <libmodel.so> <weights.bump> [-p "prompt"]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <time.h>
#include <math.h>

/* ============================================================================
 * API Function Types (loaded via dlsym)
 * ============================================================================ */

typedef int (*init_fn_t)(const char *weights_path);
typedef void (*free_fn_t)(void);
typedef int (*embed_fn_t)(const int32_t *tokens, int num_tokens);
typedef int (*forward_fn_t)(float *logits_out);
typedef int (*decode_fn_t)(int32_t token, float *logits_out);
typedef int32_t (*sample_fn_t)(void);
typedef float* (*get_logits_fn_t)(void);
typedef int (*get_int_fn_t)(void);
typedef void (*kv_reset_fn_t)(void);
typedef const int32_t* (*get_vocab_offsets_fn_t)(void);
typedef const char* (*get_vocab_strings_fn_t)(void);
typedef const int32_t* (*get_vocab_merges_fn_t)(void);

/* ============================================================================
 * Model API Structure
 * ============================================================================ */

typedef struct {
    void *handle;

    /* Core API */
    init_fn_t init;
    free_fn_t free_model;
    embed_fn_t embed_tokens;
    forward_fn_t forward;
    decode_fn_t decode;
    sample_fn_t sample_argmax;
    get_logits_fn_t get_logits;
    kv_reset_fn_t kv_reset;

    /* Model info */
    get_int_fn_t get_vocab_size;
    get_int_fn_t get_context_window;
    get_int_fn_t get_active_tokens;

    /* Vocab accessors (for tokenizer) */
    get_int_fn_t get_num_merges;
    get_int_fn_t get_vocab_strings_size;
    get_vocab_offsets_fn_t get_vocab_offsets;
    get_vocab_strings_fn_t get_vocab_strings;
    get_vocab_merges_fn_t get_vocab_merges;
} ModelAPI;

/* ============================================================================
 * Simple Tokenizer (placeholder - in production use true_bpe.h)
 * ============================================================================ */

typedef struct {
    int vocab_size;
    const int32_t *vocab_offsets;
    const char *vocab_strings;
    int num_merges;
    const int32_t *vocab_merges;
} SimpleTokenizer;

static SimpleTokenizer* tokenizer_create(void) {
    return (SimpleTokenizer*)calloc(1, sizeof(SimpleTokenizer));
}

static void tokenizer_free(SimpleTokenizer *tok) {
    if (tok) free(tok);
}

static int tokenizer_init(SimpleTokenizer *tok,
                          int vocab_size,
                          const int32_t *vocab_offsets,
                          const char *vocab_strings,
                          int num_merges,
                          const int32_t *vocab_merges) {
    tok->vocab_size = vocab_size;
    tok->vocab_offsets = vocab_offsets;
    tok->vocab_strings = vocab_strings;
    tok->num_merges = num_merges;
    tok->vocab_merges = vocab_merges;
    return 0;
}

/* Simple byte-level tokenization (for demo - use proper BPE in production) */
static int tokenizer_encode(SimpleTokenizer *tok, const char *text, int text_len,
                           int32_t *ids, int max_ids) {
    if (text_len < 0) text_len = strlen(text);

    /* Simple: one byte = one token (demo only) */
    int n = 0;
    for (int i = 0; i < text_len && n < max_ids; i++) {
        /* Find matching token in vocab (slow linear search for demo) */
        unsigned char c = (unsigned char)text[i];
        ids[n++] = c;  /* Just use byte value as token ID */
    }
    return n;
}

/* Decode token ID to string */
static const char* tokenizer_decode(SimpleTokenizer *tok, int32_t id) {
    static char buf[256];
    if (id < 0 || id >= tok->vocab_size || !tok->vocab_offsets || !tok->vocab_strings) {
        buf[0] = '\0';
        return buf;
    }

    int offset = tok->vocab_offsets[id];
    int next_offset = (id + 1 < tok->vocab_size) ? tok->vocab_offsets[id + 1] : offset;
    int len = next_offset - offset;
    if (len < 0 || len > 255) len = 0;

    memcpy(buf, tok->vocab_strings + offset, len);
    buf[len] = '\0';
    return buf;
}

/* ============================================================================
 * Model Loading
 * ============================================================================ */

static bool load_symbol(void *handle, const char *name, void **out, bool required) {
    *out = dlsym(handle, name);
    if (!*out && required) {
        fprintf(stderr, "Error: missing symbol '%s'\n", name);
        return false;
    }
    return true;
}

static bool load_model(const char *lib_path, ModelAPI *api) {
    memset(api, 0, sizeof(*api));

    api->handle = dlopen(lib_path, RTLD_NOW);
    if (!api->handle) {
        fprintf(stderr, "Error: dlopen failed: %s\n", dlerror());
        return false;
    }

    /* Required functions */
    if (!load_symbol(api->handle, "ck_model_init", (void**)&api->init, true)) return false;
    if (!load_symbol(api->handle, "ck_model_embed_tokens", (void**)&api->embed_tokens, true)) return false;
    if (!load_symbol(api->handle, "ck_model_forward", (void**)&api->forward, true)) return false;
    if (!load_symbol(api->handle, "ck_model_decode", (void**)&api->decode, true)) return false;
    if (!load_symbol(api->handle, "ck_model_sample_argmax", (void**)&api->sample_argmax, true)) return false;

    /* Optional functions */
    load_symbol(api->handle, "ck_model_free", (void**)&api->free_model, false);
    load_symbol(api->handle, "ck_model_get_logits", (void**)&api->get_logits, false);
    load_symbol(api->handle, "ck_model_kv_cache_reset", (void**)&api->kv_reset, false);
    load_symbol(api->handle, "ck_model_get_vocab_size", (void**)&api->get_vocab_size, false);
    load_symbol(api->handle, "ck_model_get_context_window", (void**)&api->get_context_window, false);
    load_symbol(api->handle, "ck_model_get_active_tokens", (void**)&api->get_active_tokens, false);
    load_symbol(api->handle, "ck_model_get_num_merges", (void**)&api->get_num_merges, false);
    load_symbol(api->handle, "ck_model_get_vocab_strings_size", (void**)&api->get_vocab_strings_size, false);
    load_symbol(api->handle, "ck_model_get_vocab_offsets", (void**)&api->get_vocab_offsets, false);
    load_symbol(api->handle, "ck_model_get_vocab_strings", (void**)&api->get_vocab_strings, false);
    load_symbol(api->handle, "ck_model_get_vocab_merges", (void**)&api->get_vocab_merges, false);

    return true;
}

/* ============================================================================
 * Inference
 * ============================================================================ */

static void run_inference(ModelAPI *api, SimpleTokenizer *tok,
                         const char *prompt, int max_tokens) {
    int vocab_size = api->get_vocab_size ? api->get_vocab_size() : 32000;
    int context_len = api->get_context_window ? api->get_context_window() : 2048;

    /* Tokenize input */
    int32_t *tokens = (int32_t*)malloc(context_len * sizeof(int32_t));
    int num_tokens = tokenizer_encode(tok, prompt, -1, tokens, context_len);

    printf("Prompt: \"%s\" (%d tokens)\n", prompt, num_tokens);
    printf("Generating...\n\n");

    /* Reset KV cache */
    if (api->kv_reset) {
        api->kv_reset();
    }

    /* STEP 1: Prefill - embed all prompt tokens and run forward */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    api->embed_tokens(tokens, num_tokens);
    api->forward(NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;

    /* Sample first token */
    int32_t next_token = api->sample_argmax();

    /* STEP 2: Decode loop - generate tokens one at a time */
    double decode_ms = 0.0;
    int generated = 0;

    printf("Output: ");
    fflush(stdout);

    for (int i = 0; i < max_tokens; i++) {
        /* Output token */
        const char *word = tokenizer_decode(tok, next_token);
        printf("%s", word);
        fflush(stdout);

        /* Check for EOS (commonly token 2 or high IDs like 151643) */
        if (next_token == 2 || next_token == 151643 || next_token == 151645) {
            break;
        }

        /* Decode next token */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        api->decode(next_token, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        decode_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (t1.tv_nsec - t0.tv_nsec) / 1e6;

        /* Sample next */
        next_token = api->sample_argmax();
        generated++;
    }

    printf("\n\n");

    /* Print timing stats */
    printf("--- Stats ---\n");
    printf("Prefill: %d tokens in %.1f ms (%.1f tok/s)\n",
           num_tokens, prefill_ms, num_tokens / (prefill_ms / 1000.0));
    printf("Decode:  %d tokens in %.1f ms (%.1f tok/s)\n",
           generated, decode_ms, generated > 0 ? generated / (decode_ms / 1000.0) : 0);

    free(tokens);
}

/* ============================================================================
 * REPL (Read-Eval-Print Loop)
 * ============================================================================ */

static void run_repl(ModelAPI *api, SimpleTokenizer *tok) {
    char line[4096];

    printf("\nC-Kernel-Engine v7 CLI\n");
    printf("Type your prompt and press Enter. Type 'quit' to exit.\n\n");

    while (1) {
        printf("You: ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }

        /* Remove trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }

        /* Check for quit command */
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) {
            break;
        }

        /* Skip empty lines */
        if (line[0] == '\0') {
            continue;
        }

        /* Reset KV cache for new conversation */
        if (api->kv_reset) {
            api->kv_reset();
        }

        printf("Assistant: ");
        fflush(stdout);

        /* Run inference */
        run_inference(api, tok, line, 256);
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

static void print_usage(const char *prog) {
    printf("Usage: %s <libmodel.so> <weights.bump> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -p, --prompt TEXT    Run single prompt instead of REPL\n");
    printf("  -n, --max-tokens N   Max tokens to generate (default: 256)\n");
    printf("  -h, --help           Show this help\n");
    printf("\nExamples:\n");
    printf("  %s ./model.so ./weights.bump\n", prog);
    printf("  %s ./model.so ./weights.bump -p \"Hello, world!\"\n", prog);
}

int main(int argc, char **argv) {
    const char *lib_path = NULL;
    const char *weights_path = NULL;
    const char *prompt = NULL;
    int max_tokens = 256;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            if (!lib_path) lib_path = argv[i];
            else if (!weights_path) weights_path = argv[i];
        }
    }

    if (!lib_path || !weights_path) {
        print_usage(argv[0]);
        return 1;
    }

    /* Load model library */
    printf("Loading model: %s\n", lib_path);

    ModelAPI api;
    if (!load_model(lib_path, &api)) {
        return 1;
    }

    /* Initialize model (allocate memory, load weights) */
    printf("Initializing model: %s\n", weights_path);

    if (api.init(weights_path) != 0) {
        fprintf(stderr, "Error: ck_model_init failed\n");
        return 1;
    }

    /* Get model info */
    int vocab_size = api.get_vocab_size ? api.get_vocab_size() : 32000;
    int context_len = api.get_context_window ? api.get_context_window() : 2048;

    printf("Model loaded: vocab=%d, context=%d\n", vocab_size, context_len);

    /* Initialize tokenizer */
    SimpleTokenizer *tok = tokenizer_create();

    if (api.get_vocab_offsets && api.get_vocab_strings) {
        int num_merges = api.get_num_merges ? api.get_num_merges() : 0;
        tokenizer_init(tok, vocab_size,
                       api.get_vocab_offsets(),
                       api.get_vocab_strings(),
                       num_merges,
                       api.get_vocab_merges ? api.get_vocab_merges() : NULL);
        printf("Tokenizer initialized from model\n");
    } else {
        printf("Warning: Using fallback tokenizer (no vocab in model)\n");
        tokenizer_init(tok, vocab_size, NULL, NULL, 0, NULL);
    }

    /* Run inference */
    if (prompt) {
        run_inference(&api, tok, prompt, max_tokens);
    } else {
        run_repl(&api, tok);
    }

    /* Cleanup */
    tokenizer_free(tok);
    if (api.free_model) api.free_model();
    dlclose(api.handle);

    printf("Done.\n");
    return 0;
}
