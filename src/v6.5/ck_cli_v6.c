/*
 * Native v6 CLI: true-BPE + prefill/decode loop
 *
 * Usage:
 *   ck-cli-v6 <libmodel.so> <weights.bump> [--prompt TEXT] [--max-tokens N]
 *   ck-cli-v6 --lib <libmodel.so> --weights <weights.bump> [options]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <dlfcn.h>
#include <unistd.h>
#include <time.h>

#include "tokenizer/true_bpe.h"

#define CK_CLI_DEFAULT_MAX_TOKENS  128
#define CK_CLI_EOS_MAX             8
#define CK_CLI_OUTPUT_BUF_SIZE     4096

static volatile sig_atomic_t g_exit_requested = 0;

/* Timing globals */
static double g_prefill_time_ms = 0.0;
static double g_decode_time_ms = 0.0;
static int g_decode_count = 0;

static void handle_sigint(int sig) {
    (void)sig;
    g_exit_requested = 1;
}

typedef int (*init_t)(const char *weights_path);
typedef int (*embed_t)(const int32_t *tokens, int num_tokens);
typedef int (*forward_t)(float *logits_out);
typedef int (*kv_enable_t)(int capacity);
typedef void (*kv_reset_t)(void);
typedef int (*decode_t)(int32_t token, float *logits_out);
typedef int (*sample_argmax_t)(void);
typedef int (*get_int_t)(void);
typedef void *(*get_ptr_t)(void);
typedef void (*free_t)(void);

typedef struct {
    void *handle;
    init_t init;
    embed_t embed;
    forward_t forward;
    kv_enable_t kv_enable;
    kv_reset_t kv_reset;
    decode_t decode;
    sample_argmax_t sample;
    get_int_t get_context;
    get_int_t get_vocab_size;
    get_int_t get_num_merges;
    get_int_t get_vocab_bytes;
    get_ptr_t get_offsets;
    get_ptr_t get_strings;
    get_ptr_t get_merges;
    free_t free_fn;
} ModelAPI;

typedef struct {
    const char *lib_path;
    const char *weights_path;
    const char *prompt_once;
    int max_tokens;
    int context_override;
    bool ignore_eos;
    bool stream;            /* Stream tokens as generated */
    bool timing;            /* Show timing breakdown */
    int eos_ids[CK_CLI_EOS_MAX];
    int eos_count;
} CLIOptions;

static void print_help(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s <libmodel.so> <weights.bump> [options]\n", prog);
    fprintf(stderr, "  %s --lib <libmodel.so> --weights <weights.bump> [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --prompt, -p TEXT       Run single prompt (non-interactive)\n");
    fprintf(stderr, "  --max-tokens, -t N       Max tokens to generate (default: %d)\n", CK_CLI_DEFAULT_MAX_TOKENS);
    fprintf(stderr, "  --context, -c N          Override context/KV cache size\n");
    fprintf(stderr, "  --stream, -s             Stream tokens as generated\n");
    fprintf(stderr, "  --timing, -T             Show timing breakdown\n");
    fprintf(stderr, "  --eos IDS               Comma-separated EOS token IDs\n");
    fprintf(stderr, "  --ignore-eos            Do not stop on EOS tokens\n");
    fprintf(stderr, "  --help, -h              Show this help\n");
}

static char *read_prompt_file(const char *path) {
    if (!path || path[0] != '@') {
        return NULL;
    }
    FILE *f = fopen(path + 1, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open prompt file: %s\n", path + 1);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t n = fread(buf, 1, (size_t)size, f);
    buf[n] = '\0';
    fclose(f);
    return buf;
}

static bool parse_eos_ids(const char *arg, CLIOptions *opt) {
    if (!arg || !opt) return false;
    opt->eos_count = 0;
    const char *p = arg;
    while (*p && opt->eos_count < CK_CLI_EOS_MAX) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        opt->eos_ids[opt->eos_count++] = (int)v;
        p = end;
        if (*p == ',') p++;
    }
    return opt->eos_count > 0;
}

static bool is_eos_token(const CLIOptions *opt, int token) {
    if (!opt || opt->ignore_eos) return false;
    for (int i = 0; i < opt->eos_count; i++) {
        if (opt->eos_ids[i] == token) return true;
    }
    return false;
}

static void output_flush(char *buf, size_t *len) {
    if (*len == 0) return;
    fwrite(buf, 1, *len, stdout);
    *len = 0;
}

static void output_append(char *buf, size_t *len, const char *text) {
    if (!text || !*text) return;
    size_t n = strlen(text);
    if (*len + n >= CK_CLI_OUTPUT_BUF_SIZE) {
        output_flush(buf, len);
    }
    if (n >= CK_CLI_OUTPUT_BUF_SIZE) {
        fwrite(text, 1, n, stdout);
        return;
    }
    memcpy(buf + *len, text, n);
    *len += n;
}

static void output_token(char *buf, size_t *len, const char *token) {
    if (!token || !*token) return;
    const unsigned char *b = (const unsigned char *)token;
    /* "Ġ" prefix (0xC4 0xA0) */
    if (b[0] == 0xC4 && b[1] == 0xA0) {
        output_append(buf, len, " ");
        output_append(buf, len, token + 2);
        return;
    }
    /* "▁" prefix (0xE2 0x96 0x81) */
    if (b[0] == 0xE2 && b[1] == 0x96 && b[2] == 0x81) {
        output_append(buf, len, " ");
        output_append(buf, len, token + 3);
        return;
    }
    output_append(buf, len, token);
}

static bool resolve_symbol(void *handle, const char *name, void **out_ptr, bool required) {
    void *sym = dlsym(handle, name);
    if (!sym && required) {
        fprintf(stderr, "Error: missing symbol %s\n", name);
        return false;
    }
    if (out_ptr) *out_ptr = sym;
    return true;
}

static bool load_model_api(const char *lib_path, ModelAPI *api) {
    if (!lib_path || !api) return false;
    memset(api, 0, sizeof(*api));
    api->handle = dlopen(lib_path, RTLD_NOW);
    if (!api->handle) {
        fprintf(stderr, "Error: dlopen failed: %s\n", dlerror());
        return false;
    }

    if (!resolve_symbol(api->handle, "ck_model_init", (void **)&api->init, true)) return false;
    if (!resolve_symbol(api->handle, "ck_model_embed_tokens", (void **)&api->embed, true)) return false;
    if (!resolve_symbol(api->handle, "ck_model_forward", (void **)&api->forward, true)) return false;
    if (!resolve_symbol(api->handle, "ck_model_decode", (void **)&api->decode, true)) return false;
    if (!resolve_symbol(api->handle, "ck_model_sample_argmax", (void **)&api->sample, true)) return false;
    resolve_symbol(api->handle, "ck_model_kv_cache_enable", (void **)&api->kv_enable, false);
    resolve_symbol(api->handle, "ck_model_kv_cache_reset", (void **)&api->kv_reset, false);
    resolve_symbol(api->handle, "ck_model_get_context_window", (void **)&api->get_context, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_size", (void **)&api->get_vocab_size, false);
    resolve_symbol(api->handle, "ck_model_get_num_merges", (void **)&api->get_num_merges, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_strings_size", (void **)&api->get_vocab_bytes, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_offsets", (void **)&api->get_offsets, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_strings", (void **)&api->get_strings, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_merges", (void **)&api->get_merges, false);
    resolve_symbol(api->handle, "ck_model_free", (void **)&api->free_fn, false);

    if (!api->get_vocab_size || !api->get_vocab_bytes || !api->get_offsets || !api->get_strings) {
        fprintf(stderr, "Error: vocab accessors missing from model\n");
        return false;
    }
    return true;
}

static int run_prompt(ModelAPI *api, CKTrueBPE *tokenizer, const CLIOptions *opt, const char *input) {
    if (!api || !tokenizer || !opt || !input) return -1;
    if (g_exit_requested) return -1;

    int ctx = opt->context_override;
    if (ctx <= 0 && api->get_context) {
        ctx = api->get_context();
    }
    if (ctx <= 0) ctx = 4096;

    int max_tokens = opt->max_tokens > 0 ? opt->max_tokens : CK_CLI_DEFAULT_MAX_TOKENS;

    int32_t *ids = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
    if (!ids) {
        fprintf(stderr, "Error: failed to allocate token buffer\n");
        return -1;
    }

    int n = ck_true_bpe_encode(tokenizer, input, -1, ids, ctx);
    if (n <= 0) {
        fprintf(stderr, "[Tokenizer] failed to encode prompt\n");
        free(ids);
        return -1;
    }
    if (n > ctx) n = ctx;

    /* Reset timing globals */
    g_prefill_time_ms = 0.0;
    g_decode_time_ms = 0.0;
    g_decode_count = 0;

    if (api->kv_reset) {
        api->kv_reset();
    }

    if (api->embed(ids, n) != 0) {
        fprintf(stderr, "[Model] embed failed\n");
        free(ids);
        return -1;
    }

    /* Time prefill (forward) */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (api->forward(NULL) != 0) {
        fprintf(stderr, "[Model] forward failed\n");
        free(ids);
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_prefill_time_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1000000.0;

    int next_token = api->sample();
    char out_buf[CK_CLI_OUTPUT_BUF_SIZE];
    size_t out_len = 0;

    if (!opt->timing) {
        printf("Assistant: ");
        fflush(stdout);
    }

    for (int generated = 0; generated < max_tokens && !g_exit_requested; generated++) {
        if (next_token < 0) break;
        if (is_eos_token(opt, next_token)) break;

        const char *word = ck_true_bpe_id_to_token(tokenizer, next_token);
        output_token(out_buf, &out_len, word);

        /* Flush immediately if streaming enabled */
        if (opt->stream) {
            output_flush(out_buf, &out_len);
            fflush(stdout);
        } else if (out_len > (CK_CLI_OUTPUT_BUF_SIZE / 2)) {
            output_flush(out_buf, &out_len);
            fflush(stdout);
        }

        if (generated + 1 >= max_tokens) break;

        /* Time decode */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (api->decode(next_token, NULL) != 0) {
            fprintf(stderr, "\n[Model] decode failed\n");
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        g_decode_time_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                            (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
        g_decode_count++;

        next_token = api->sample();
    }

    output_flush(out_buf, &out_len);

    /* Print timing if requested */
    if (opt->timing) {
        printf("\n");
        double total_ms = g_prefill_time_ms + g_decode_time_ms;
        double prefill_rate = n / (g_prefill_time_ms / 1000.0);
        double decode_rate = g_decode_count / (g_decode_time_ms / 1000.0);
        double avg_decode = g_decode_count > 0 ? g_decode_time_ms / g_decode_count : 0.0;

        printf("Timing:\n");
        printf("  Prefill:  %3d tokens in %7.2f ms (%6.2f tok/s)\n",
               n, g_prefill_time_ms, prefill_rate);
        printf("  Decode:   %3d tokens in %7.2f ms (%6.2f tok/s) avg %.2f ms/tok\n",
               g_decode_count, g_decode_time_ms, decode_rate, avg_decode);
        printf("  Total:    %3d tokens in %7.2f ms\n",
               n + g_decode_count, total_ms);
    } else {
        printf("\n");
    }
    fflush(stdout);

    free(ids);
    return 0;
}

static bool parse_args(int argc, char **argv, CLIOptions *opt) {
    if (!opt) return false;
    memset(opt, 0, sizeof(*opt));
    opt->max_tokens = CK_CLI_DEFAULT_MAX_TOKENS;
    opt->eos_ids[0] = 151643;
    opt->eos_ids[1] = 151645;
    opt->eos_count = 2;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            print_help(argv[0]);
            return false;
        } else if (!strcmp(arg, "--lib") && i + 1 < argc) {
            opt->lib_path = argv[++i];
        } else if (!strcmp(arg, "--weights") && i + 1 < argc) {
            opt->weights_path = argv[++i];
        } else if ((!strcmp(arg, "--prompt") || !strcmp(arg, "-p")) && i + 1 < argc) {
            opt->prompt_once = argv[++i];
        } else if ((!strcmp(arg, "--max-tokens") || !strcmp(arg, "-t")) && i + 1 < argc) {
            opt->max_tokens = atoi(argv[++i]);
        } else if ((!strcmp(arg, "--context") || !strcmp(arg, "-c")) && i + 1 < argc) {
            opt->context_override = atoi(argv[++i]);
        } else if (!strcmp(arg, "--stream") || !strcmp(arg, "-s")) {
            opt->stream = true;
        } else if (!strcmp(arg, "--timing") || !strcmp(arg, "-T")) {
            opt->timing = true;
        } else if (!strcmp(arg, "--eos") && i + 1 < argc) {
            parse_eos_ids(argv[++i], opt);
        } else if (!strcmp(arg, "--ignore-eos")) {
            opt->ignore_eos = true;
        } else if (arg[0] != '-') {
            if (!opt->lib_path) opt->lib_path = arg;
            else if (!opt->weights_path) opt->weights_path = arg;
            else {
                fprintf(stderr, "Unknown argument: %s\n", arg);
                return false;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            return false;
        }
    }

    if (!opt->lib_path || !opt->weights_path) {
        print_help(argv[0]);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    signal(SIGINT, handle_sigint);

    CLIOptions opt;
    if (!parse_args(argc, argv, &opt)) {
        return 1;
    }

    ModelAPI api;
    if (!load_model_api(opt.lib_path, &api)) {
        return 1;
    }

    if (api.init(opt.weights_path) != 0) {
        fprintf(stderr, "Error: ck_model_init failed\n");
        return 1;
    }

    int ctx = opt.context_override;
    if (ctx <= 0 && api.get_context) ctx = api.get_context();
    if (api.kv_enable && ctx > 0) {
        api.kv_enable(ctx);
    }

    CKTrueBPE *tokenizer = ck_true_bpe_create();
    if (!tokenizer) {
        fprintf(stderr, "[Tokenizer] failed to create\n");
        return 1;
    }

    int vocab_size = api.get_vocab_size ? api.get_vocab_size() : 0;
    int vocab_bytes = api.get_vocab_bytes ? api.get_vocab_bytes() : 0;
    int num_merges = api.get_num_merges ? api.get_num_merges() : 0;
    const int32_t *offsets = (const int32_t *)api.get_offsets();
    const char *strings = (const char *)api.get_strings();
    const int32_t *merges = api.get_merges ? (const int32_t *)api.get_merges() : NULL;

    if (vocab_size <= 0 || vocab_bytes <= 0 || !offsets || !strings) {
        fprintf(stderr, "[Tokenizer] missing vocab data in model\n");
        ck_true_bpe_free(tokenizer);
        return 1;
    }

    if (ck_true_bpe_load_binary(tokenizer, vocab_size, offsets, strings, num_merges, merges) != 0) {
        fprintf(stderr, "[Tokenizer] failed to load vocab\n");
        ck_true_bpe_free(tokenizer);
        return 1;
    }

    setvbuf(stdout, NULL, _IOFBF, 1 << 20);

    if (opt.prompt_once) {
        char *file_prompt = read_prompt_file(opt.prompt_once);
        const char *prompt_text = file_prompt ? file_prompt : opt.prompt_once;
        run_prompt(&api, tokenizer, &opt, prompt_text);
        free(file_prompt);
    } else {
        char line[4096];
        while (!g_exit_requested) {
            printf("\nYou: ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) {
                if (feof(stdin) || g_exit_requested) {
                    break;
                }
                if (errno == EINTR) {
                    break;
                }
                continue;
            }
            if (line[0] == '\n' || line[0] == '\0') continue;
            if (!strncmp(line, "/exit", 5) || !strncmp(line, "/quit", 5)) break;
            if (!strncmp(line, "/help", 5)) {
                print_help(argv[0]);
                continue;
            }
            if (!strncmp(line, "/reset", 6)) {
                if (api.kv_reset) api.kv_reset();
                continue;
            }
            char *file_prompt = read_prompt_file(line);
            const char *prompt_text = file_prompt ? file_prompt : line;
            run_prompt(&api, tokenizer, &opt, prompt_text);
            free(file_prompt);
        }
    }

    ck_true_bpe_free(tokenizer);
    if (api.free_fn) api.free_fn();
    if (api.handle) dlclose(api.handle);
    return 0;
}
