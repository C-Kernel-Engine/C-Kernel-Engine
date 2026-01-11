/*
 * ck_profile_v6.c - Standalone profiling binary for CK-Engine v6
 *
 * Build:
 *   gcc -O3 -g -fno-omit-frame-pointer -mavx -fopenmp \
 *       tools/ck_profile_v6.c \
 *       src/*.c src/kernels/*.c \
 *       -I include -lm -o build/ck-profile-v6
 *
 * Profile:
 *   perf record -g -F 999 ./build/ck-profile-v6 \
 *       ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF 50
 *   perf report --stdio --sort=symbol | head -50
 *
 * Flamegraph:
 *   perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>

/* Timer helpers */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Simple token sampler (argmax) */
static int argmax_f32(const float *logits, int vocab_size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/* Load model from shared library */
typedef struct {
    void *handle;
    void *model;
    void *(*create)(const char *weights_path);
    void (*destroy)(void *model);
    void (*prefill)(void *model, const int *tokens, int num_tokens);
    void (*decode)(void *model, const int *token, int token_index);
    float *(*get_logits)(void *model);
    int vocab_size;
    int context_window;
} CKModel;

static int load_model_so(CKModel *m, const char *model_dir) {
    char so_path[4096];
    snprintf(so_path, sizeof(so_path), "%s/ck-kernel-inference.so", model_dir);

    m->handle = dlopen(so_path, RTLD_NOW);
    if (!m->handle) {
        fprintf(stderr, "Failed to load %s: %s\n", so_path, dlerror());
        return -1;
    }

    /* Find symbols - try common naming patterns */
    /* The model struct and functions are named based on the model */
    /* For Qwen2-0.5B, they might be: qwen2_0_5b_instruct_q4_k_m_* */

    /* Try to find the _create function */
    void *sym = NULL;
    char *prefixes[] = {
        "qwen2_0_5b_instruct_q4_k_m",
        "qwen2_0_5b_instruct",
        "qwen2_0_5b",
        "model",
        NULL
    };

    char sym_name[256];
    for (int i = 0; prefixes[i]; i++) {
        snprintf(sym_name, sizeof(sym_name), "%s_create", prefixes[i]);
        sym = dlsym(m->handle, sym_name);
        if (sym) {
            printf("[PROFILE] Found prefix: %s\n", prefixes[i]);

            m->create = sym;
            snprintf(sym_name, sizeof(sym_name), "%s_destroy", prefixes[i]);
            m->destroy = dlsym(m->handle, sym_name);
            snprintf(sym_name, sizeof(sym_name), "%s_prefill", prefixes[i]);
            m->prefill = dlsym(m->handle, sym_name);
            snprintf(sym_name, sizeof(sym_name), "%s_decode", prefixes[i]);
            m->decode = dlsym(m->handle, sym_name);
            snprintf(sym_name, sizeof(sym_name), "%s_get_logits", prefixes[i]);
            m->get_logits = dlsym(m->handle, sym_name);

            /* Get constants */
            snprintf(sym_name, sizeof(sym_name), "%s_VOCAB_SIZE", prefixes[i]);
            int *vocab_ptr = dlsym(m->handle, sym_name);
            m->vocab_size = vocab_ptr ? *vocab_ptr : 151936;

            snprintf(sym_name, sizeof(sym_name), "%s_CONTEXT_WINDOW", prefixes[i]);
            int *ctx_ptr = dlsym(m->handle, sym_name);
            m->context_window = ctx_ptr ? *ctx_ptr : 32768;

            break;
        }
    }

    if (!m->create) {
        fprintf(stderr, "Failed to find model symbols\n");

        /* List available symbols for debugging */
        printf("\nAvailable symbols containing 'create':\n");
        /* Can't easily enumerate dloaded symbols, user needs to check nm output */
        printf("Run: nm -D %s | grep create\n", so_path);

        dlclose(m->handle);
        return -1;
    }

    if (!m->prefill || !m->decode || !m->get_logits) {
        fprintf(stderr, "Missing required functions (prefill=%p, decode=%p, get_logits=%p)\n",
                m->prefill, m->decode, m->get_logits);
        dlclose(m->handle);
        return -1;
    }

    return 0;
}

static void print_usage(const char *prog) {
    printf("Usage: %s <model_dir> [num_tokens] [--decode-only]\n", prog);
    printf("\n");
    printf("Arguments:\n");
    printf("  model_dir     Directory containing ck-kernel-inference.so and weights.bump\n");
    printf("  num_tokens    Number of tokens to generate (default: 20)\n");
    printf("  --decode-only Skip prefill, only measure decode\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s ~/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF 50\n", prog);
    printf("\n");
    printf("Profiling:\n");
    printf("  perf record -g -F 999 %s <model_dir> 50\n", prog);
    printf("  perf report --stdio --sort=symbol\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *model_dir = argv[1];
    int num_tokens = (argc > 2 && argv[2][0] != '-') ? atoi(argv[2]) : 20;
    int decode_only = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--decode-only") == 0) {
            decode_only = 1;
        }
    }

    printf("=== CK-Engine v6 Profiler ===\n");
    printf("Model dir: %s\n", model_dir);
    printf("Tokens to generate: %d\n", num_tokens);
    printf("\n");

    /* Load model */
    CKModel m = {0};
    if (load_model_so(&m, model_dir) != 0) {
        return 1;
    }

    printf("Vocab size: %d\n", m.vocab_size);
    printf("Context window: %d\n", m.context_window);
    printf("\n");

    /* Create model instance */
    char weights_path[4096];
    snprintf(weights_path, sizeof(weights_path), "%s/weights.bump", model_dir);

    printf("Loading weights from: %s\n", weights_path);
    double t0 = get_time_ms();
    m.model = m.create(weights_path);
    double t1 = get_time_ms();

    if (!m.model) {
        fprintf(stderr, "Failed to create model\n");
        dlclose(m.handle);
        return 1;
    }
    printf("Model loaded in %.1f ms\n\n", t1 - t0);

    /* Prepare prompt tokens */
    /* "Hello" in Qwen2 tokenizer: [9707] (simplified) */
    int prompt_tokens[] = {
        151644,  /* <|im_start|> */
        872,     /* user */
        198,     /* \n */
        9707,    /* Hello */
        0,       /* ! */
        151645,  /* <|im_end|> */
        198,     /* \n */
        151644,  /* <|im_start|> */
        77091,   /* assistant */
        198,     /* \n */
    };
    int prompt_len = sizeof(prompt_tokens) / sizeof(prompt_tokens[0]);

    /* Storage for generated tokens */
    int *generated = malloc((prompt_len + num_tokens) * sizeof(int));
    memcpy(generated, prompt_tokens, prompt_len * sizeof(int));

    /* === Prefill Phase === */
    double prefill_ms = 0;
    if (!decode_only) {
        printf("=== PREFILL (%d tokens) ===\n", prompt_len);
        t0 = get_time_ms();
        m.prefill(m.model, prompt_tokens, prompt_len);
        t1 = get_time_ms();
        prefill_ms = t1 - t0;

        printf("Prefill: %.1f ms (%.2f ms/tok, %.2f tok/s)\n",
               prefill_ms, prefill_ms / prompt_len, prompt_len / (prefill_ms / 1000.0));
    }

    /* === Decode Phase === */
    printf("\n=== DECODE (%d tokens) ===\n", num_tokens);

    double decode_total_ms = 0;
    double decode_times[num_tokens];
    int current_pos = prompt_len;

    /* First token comes from prefill logits */
    float *logits = m.get_logits(m.model);
    int next_token = argmax_f32(logits, m.vocab_size);
    generated[current_pos] = next_token;

    printf("Generating tokens...\n");

    for (int i = 0; i < num_tokens; i++) {
        t0 = get_time_ms();

        /* Decode single token */
        m.decode(m.model, &generated[current_pos], current_pos);

        t1 = get_time_ms();
        decode_times[i] = t1 - t0;
        decode_total_ms += decode_times[i];

        /* Sample next token */
        logits = m.get_logits(m.model);
        next_token = argmax_f32(logits, m.vocab_size);

        current_pos++;
        if (current_pos < prompt_len + num_tokens) {
            generated[current_pos] = next_token;
        }

        /* Progress indicator */
        if ((i + 1) % 10 == 0 || i == num_tokens - 1) {
            printf("  Token %d/%d: %.1f ms (%.2f tok/s avg)\n",
                   i + 1, num_tokens, decode_times[i],
                   (i + 1) / (decode_total_ms / 1000.0));
        }

        /* Stop on EOS */
        if (next_token == 151645 || next_token == 151643) {
            printf("  [EOS at token %d]\n", i + 1);
            num_tokens = i + 1;
            break;
        }
    }

    /* === Summary === */
    printf("\n=== SUMMARY ===\n");
    if (!decode_only) {
        printf("Prefill: %.1f ms for %d tokens (%.2f tok/s)\n",
               prefill_ms, prompt_len, prompt_len / (prefill_ms / 1000.0));
    }
    printf("Decode:  %.1f ms for %d tokens (%.2f tok/s)\n",
           decode_total_ms, num_tokens, num_tokens / (decode_total_ms / 1000.0));
    printf("Average decode latency: %.1f ms/tok\n", decode_total_ms / num_tokens);

    /* Timing distribution */
    double min_t = decode_times[0], max_t = decode_times[0], sum_t = 0;
    for (int i = 0; i < num_tokens; i++) {
        if (decode_times[i] < min_t) min_t = decode_times[i];
        if (decode_times[i] > max_t) max_t = decode_times[i];
        sum_t += decode_times[i];
    }
    printf("Decode time range: %.1f - %.1f ms (avg %.1f ms)\n", min_t, max_t, sum_t / num_tokens);

    printf("\n=== PERFORMANCE GAP ESTIMATE ===\n");
    double llama_cpp_tok_s = 35.0;  /* From user's measurement */
    double our_tok_s = num_tokens / (decode_total_ms / 1000.0);
    printf("llama.cpp: ~%.1f tok/s\n", llama_cpp_tok_s);
    printf("CK-Engine: ~%.1f tok/s\n", our_tok_s);
    printf("Gap: %.1fx slower\n", llama_cpp_tok_s / our_tok_s);

    /* Cleanup */
    free(generated);
    if (m.destroy) {
        m.destroy(m.model);
    }
    dlclose(m.handle);

    return 0;
}
