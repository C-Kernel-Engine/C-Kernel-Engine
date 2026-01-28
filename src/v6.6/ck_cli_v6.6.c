/*
 * C-Kernel-Engine v6.6 Native CLI
 *
 * Features:
 *   - Model auto-discovery from cache
 *   - Readline support for history/editing
 *   - Chat template support (Qwen, LLaMA, etc.)
 *   - Temperature/top-p sampling
 *   - Streaming output
 *
 * Usage:
 *   ck-cli-v6.6 --model <name>                    # Auto-discover from cache
 *   ck-cli-v6.6 <libmodel.so> <weights.bump>      # Direct paths
 *   ck-cli-v6.6 --lib <.so> --weights <.bump>     # Named args
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
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

#include "tokenizer/true_bpe.h"
#include "ck_features.h"

#define CK_CLI_VERSION         "6.6.0"
#define CK_CLI_DEFAULT_MAX_TOKENS  256
#define CK_CLI_EOS_MAX             8
#define CK_CLI_OUTPUT_BUF_SIZE     4096
#define CK_CLI_MAX_CONTEXT         32768
#define CK_CLI_HISTORY_FILE        ".ck_cli_history"

static volatile sig_atomic_t g_exit_requested = 0;
static volatile sig_atomic_t g_generation_active = 0;

/* Timing globals */
static double g_prefill_time_ms = 0.0;
static double g_decode_time_ms = 0.0;
static int g_decode_count = 0;
static int g_prompt_tokens = 0;

static void handle_sigint(int sig) {
    (void)sig;
    if (g_generation_active) {
        g_generation_active = 0;  /* Stop generation but don't exit */
    } else {
        g_exit_requested = 1;
    }
}

/* ============================================================================
 * Model API Types
 * ============================================================================ */

typedef int (*init_t)(const char *weights_path);
typedef int (*embed_t)(const int32_t *tokens, int num_tokens);
typedef int (*forward_t)(float *logits_out);
typedef int (*kv_enable_t)(int capacity);
typedef void (*kv_reset_t)(void);
typedef int (*decode_t)(int32_t token, float *logits_out);
typedef int (*sample_argmax_t)(void);
typedef float *(*get_logits_t)(void);
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
    get_logits_t get_logits;
    get_int_t get_context;
    get_int_t get_vocab_size;
    get_int_t get_num_merges;
    get_int_t get_vocab_bytes;
    get_int_t get_active_tokens;
    get_ptr_t get_offsets;
    get_ptr_t get_strings;
    get_ptr_t get_merges;
    free_t free_fn;
} ModelAPI;

/* ============================================================================
 * Chat Template Types
 * ============================================================================ */

typedef enum {
    CHAT_TEMPLATE_NONE = 0,
    CHAT_TEMPLATE_QWEN,
    CHAT_TEMPLATE_LLAMA,
    CHAT_TEMPLATE_CHATML,
    CHAT_TEMPLATE_MISTRAL,
} ChatTemplateType;

typedef struct {
    ChatTemplateType type;
    const char *system_prefix;
    const char *system_suffix;
    const char *user_prefix;
    const char *user_suffix;
    const char *assistant_prefix;
    const char *assistant_suffix;
} ChatTemplate;

static const ChatTemplate g_templates[] = {
    [CHAT_TEMPLATE_NONE] = {
        .type = CHAT_TEMPLATE_NONE,
        .system_prefix = "", .system_suffix = "\n",
        .user_prefix = "", .user_suffix = "\n",
        .assistant_prefix = "", .assistant_suffix = "",
    },
    [CHAT_TEMPLATE_QWEN] = {
        .type = CHAT_TEMPLATE_QWEN,
        .system_prefix = "<|im_start|>system\n",
        .system_suffix = "<|im_end|>\n",
        .user_prefix = "<|im_start|>user\n",
        .user_suffix = "<|im_end|>\n",
        .assistant_prefix = "<|im_start|>assistant\n",
        .assistant_suffix = "<|im_end|>",
    },
    [CHAT_TEMPLATE_LLAMA] = {
        .type = CHAT_TEMPLATE_LLAMA,
        .system_prefix = "[INST] <<SYS>>\n",
        .system_suffix = "\n<</SYS>>\n\n",
        .user_prefix = "",
        .user_suffix = " [/INST]",
        .assistant_prefix = " ",
        .assistant_suffix = " </s><s>[INST] ",
    },
    [CHAT_TEMPLATE_CHATML] = {
        .type = CHAT_TEMPLATE_CHATML,
        .system_prefix = "<|im_start|>system\n",
        .system_suffix = "<|im_end|>\n",
        .user_prefix = "<|im_start|>user\n",
        .user_suffix = "<|im_end|>\n",
        .assistant_prefix = "<|im_start|>assistant\n",
        .assistant_suffix = "<|im_end|>",
    },
    [CHAT_TEMPLATE_MISTRAL] = {
        .type = CHAT_TEMPLATE_MISTRAL,
        .system_prefix = "",
        .system_suffix = "\n\n",
        .user_prefix = "[INST] ",
        .user_suffix = " [/INST]",
        .assistant_prefix = "",
        .assistant_suffix = "</s> ",
    },
};

/* ============================================================================
 * CLI Options
 * ============================================================================ */

typedef struct {
    const char *model_name;     /* Model name for auto-discovery */
    const char *lib_path;
    const char *weights_path;
    const char *prompt_once;
    const char *system_prompt;
    int max_tokens;
    int context_override;
    float temperature;
    float top_p;
    bool ignore_eos;
    bool stream;
    bool timing;
    bool verbose;
    bool no_chat_template;
    ChatTemplateType chat_template;
    int eos_ids[CK_CLI_EOS_MAX];
    int eos_count;
} CLIOptions;

/* ============================================================================
 * Cache Discovery
 * ============================================================================ */

static const char *get_cache_dir(void) {
    static char cache_path[4096];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(cache_path, sizeof(cache_path), "%s/.cache/ck-engine-v6.6/models", home);
    return cache_path;
}

static bool find_model_in_cache(const char *model_name, char *lib_out, char *weights_out, size_t out_size) {
    const char *cache_dir = get_cache_dir();
    DIR *dir = opendir(cache_dir);
    if (!dir) return false;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        /* Check if directory name contains model_name */
        if (strstr(entry->d_name, model_name) != NULL) {
            char model_dir[4096];
            snprintf(model_dir, sizeof(model_dir), "%s/%s", cache_dir, entry->d_name);

            /* Check for required files */
            char so_path[4096], bump_path[4096];
            snprintf(so_path, sizeof(so_path), "%s/ck-kernel-inference.so", model_dir);
            snprintf(bump_path, sizeof(bump_path), "%s/weights.bump", model_dir);

            struct stat st;
            if (stat(so_path, &st) == 0 && stat(bump_path, &st) == 0) {
                strncpy(lib_out, so_path, out_size - 1);
                strncpy(weights_out, bump_path, out_size - 1);
                closedir(dir);
                return true;
            }
        }
    }
    closedir(dir);
    return false;
}

/* ============================================================================
 * EOS Token Loading
 * ============================================================================ */

static bool load_eos_from_vocab_json(const char *weights_path, CLIOptions *opt) {
    if (!weights_path || !opt) return false;

    /* Construct vocab.json path from weights path */
    char vocab_path[4096];
    const char *slash = strrchr(weights_path, '/');
    if (!slash) return false;

    size_t dir_len = (size_t)(slash - weights_path);
    if (dir_len + 12 >= sizeof(vocab_path)) return false;

    memcpy(vocab_path, weights_path, dir_len);
    vocab_path[dir_len] = '\0';
    strcat(vocab_path, "/vocab.json");

    FILE *f = fopen(vocab_path, "r");
    if (!f) return false;

    /* Simple JSON parsing for special_tokens */
    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';

    /* Look for "special_tokens" section */
    const char *st = strstr(buf, "\"special_tokens\"");
    if (!st) return false;

    /* Extract eos token */
    const char *eos = strstr(st, "\"eos\"");
    if (eos) {
        const char *colon = strchr(eos, ':');
        if (colon) {
            int eos_id = atoi(colon + 1);
            if (eos_id > 0) {
                opt->eos_ids[0] = eos_id;
                opt->eos_count = 1;
            }
        }
    }

    /* Extract bos token (often used as im_end for chat) */
    const char *bos = strstr(st, "\"bos\"");
    if (bos) {
        const char *colon = strchr(bos, ':');
        if (colon) {
            int bos_id = atoi(colon + 1);
            if (bos_id > 0 && bos_id != opt->eos_ids[0]) {
                opt->eos_ids[opt->eos_count++] = bos_id;
            }
        }
    }

    return opt->eos_count > 0;
}

static void list_available_models(void) {
    const char *cache_dir = get_cache_dir();
    DIR *dir = opendir(cache_dir);
    if (!dir) {
        fprintf(stderr, "No models found in %s\n", cache_dir);
        return;
    }

    printf("Available models in %s:\n", cache_dir);
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        char model_dir[4096];
        snprintf(model_dir, sizeof(model_dir), "%s/%s", cache_dir, entry->d_name);

        char so_path[4096];
        snprintf(so_path, sizeof(so_path), "%s/ck-kernel-inference.so", model_dir);

        struct stat st;
        if (stat(so_path, &st) == 0) {
            printf("  - %s\n", entry->d_name);
            count++;
        }
    }
    closedir(dir);

    if (count == 0) {
        printf("  (none found)\n");
    }
}

/* ============================================================================
 * Sampling
 * ============================================================================ */

static int sample_top_p(float *logits, int vocab_size, float temperature, float top_p) {
    if (temperature <= 0.0f || top_p <= 0.0f) {
        /* Argmax */
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return best;
    }

    /* Apply temperature */
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum += logits[i];
    }

    /* Normalize to probabilities */
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }

    /* Sort indices by probability (simple selection for top-p) */
    /* For efficiency, we'll do nucleus sampling with cumulative sum */
    float cumsum = 0.0f;
    float threshold = (float)rand() / (float)RAND_MAX * top_p;

    /* Find nucleus tokens and sample */
    int *indices = (int *)malloc(vocab_size * sizeof(int));
    float *probs = (float *)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
        probs[i] = logits[i];
    }

    /* Simple sort (for small vocab, bubble sort is fine; for large, use qsort) */
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs[j] > probs[i]) {
                float tmp_p = probs[i]; probs[i] = probs[j]; probs[j] = tmp_p;
                int tmp_i = indices[i]; indices[i] = indices[j]; indices[j] = tmp_i;
            }
        }
        cumsum += probs[i];
        if (cumsum >= top_p) break;
    }

    /* Sample from nucleus */
    float r = (float)rand() / (float)RAND_MAX * cumsum;
    float acc = 0.0f;
    int result = indices[0];
    for (int i = 0; cumsum > 0 && i < vocab_size; i++) {
        acc += probs[i];
        if (acc >= r) {
            result = indices[i];
            break;
        }
        if (acc >= cumsum) break;
    }

    free(indices);
    free(probs);
    return result;
}

/* ============================================================================
 * Output Helpers
 * ============================================================================ */

/**
 * Decode GPT-2 byte-level BPE representation back to actual bytes.
 *
 * GPT-2's tokenizer maps certain bytes to Unicode code points:
 * - Bytes 0x00-0x20 → U+0100-U+0120 (Ā Ć ċ ... Ġ)
 * - Bytes 0x7F-0xA0 → U+017F-U+01A0
 * - Printable ASCII (0x21-0x7E) stays as-is
 *
 * This function reverses that mapping.
 *
 * @param token  Input BPE token string (UTF-8)
 * @param out    Output buffer for decoded bytes
 * @param max    Size of output buffer
 * @return       Number of bytes written (not including NUL)
 */
static int decode_bpe_token(const char *token, char *out, int max) {
    if (!token || max <= 0) return 0;

    const unsigned char *src = (const unsigned char *)token;
    int out_len = 0;

    while (*src && out_len < max - 1) {
        unsigned int codepoint;
        int bytes;

        /* Decode UTF-8 to codepoint */
        if ((src[0] & 0x80) == 0) {
            /* Single byte ASCII */
            codepoint = src[0];
            bytes = 1;
        } else if ((src[0] & 0xE0) == 0xC0 && (src[1] & 0xC0) == 0x80) {
            /* Two byte sequence */
            codepoint = ((src[0] & 0x1F) << 6) | (src[1] & 0x3F);
            bytes = 2;
        } else if ((src[0] & 0xF0) == 0xE0 && (src[1] & 0xC0) == 0x80 && (src[2] & 0xC0) == 0x80) {
            /* Three byte sequence */
            codepoint = ((src[0] & 0x0F) << 12) | ((src[1] & 0x3F) << 6) | (src[2] & 0x3F);
            bytes = 3;
        } else if ((src[0] & 0xF8) == 0xF0 && (src[1] & 0xC0) == 0x80 &&
                   (src[2] & 0xC0) == 0x80 && (src[3] & 0xC0) == 0x80) {
            /* Four byte sequence */
            codepoint = ((src[0] & 0x07) << 18) | ((src[1] & 0x3F) << 12) |
                        ((src[2] & 0x3F) << 6) | (src[3] & 0x3F);
            bytes = 4;
        } else {
            /* Invalid UTF-8, copy byte as-is */
            out[out_len++] = (char)*src;
            src++;
            continue;
        }

        /* Check if this is a GPT-2 byte-encoded character */
        if (codepoint >= 0x100 && codepoint <= 0x120) {
            /* Bytes 0x00-0x20: U+0100-U+0120 → byte = codepoint - 0x100 */
            out[out_len++] = (char)(codepoint - 0x100);
        } else if (codepoint >= 0x17F && codepoint <= 0x1A0) {
            /* Bytes 0x7F-0xA0: U+017F-U+01A0 → byte = codepoint - 0x100 */
            out[out_len++] = (char)(codepoint - 0x100);
        } else if (codepoint < 0x80) {
            /* Regular ASCII - copy as-is */
            out[out_len++] = (char)codepoint;
        } else if (codepoint == 0x2581) {
            /* SentencePiece space marker ▁ (U+2581) → space */
            out[out_len++] = ' ';
        } else {
            /* Other UTF-8 characters - copy original bytes */
            for (int i = 0; i < bytes && out_len < max - 1; i++) {
                out[out_len++] = (char)src[i];
            }
        }

        src += bytes;
    }

    out[out_len] = '\0';
    return out_len;
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

    /* Decode BPE byte-level encoding to actual bytes */
    char decoded[1024];
    int n = decode_bpe_token(token, decoded, sizeof(decoded));
    if (n > 0) {
        output_append(buf, len, decoded);
    }
}

/* ============================================================================
 * Model Loading
 * ============================================================================ */

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
    resolve_symbol(api->handle, "ck_model_sample_argmax", (void **)&api->sample, false);  /* Optional - we can sample from logits */
    resolve_symbol(api->handle, "ck_model_get_logits", (void **)&api->get_logits, false);
    resolve_symbol(api->handle, "ck_model_kv_cache_enable", (void **)&api->kv_enable, false);
    resolve_symbol(api->handle, "ck_model_kv_cache_reset", (void **)&api->kv_reset, false);
    resolve_symbol(api->handle, "ck_model_get_context_window", (void **)&api->get_context, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_size", (void **)&api->get_vocab_size, false);
    resolve_symbol(api->handle, "ck_model_get_num_merges", (void **)&api->get_num_merges, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_strings_size", (void **)&api->get_vocab_bytes, false);
    resolve_symbol(api->handle, "ck_model_get_active_tokens", (void **)&api->get_active_tokens, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_offsets", (void **)&api->get_offsets, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_strings", (void **)&api->get_strings, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_merges", (void **)&api->get_merges, false);
    resolve_symbol(api->handle, "ck_model_free", (void **)&api->free_fn, false);

    if (!api->get_vocab_size || !api->get_offsets || !api->get_strings) {
        fprintf(stderr, "Error: vocab accessors missing from model\n");
        return false;
    }
    return true;
}

/* ============================================================================
 * Chat Template Application
 * ============================================================================ */

static ChatTemplateType detect_chat_template(const char *model_name) {
    if (!model_name) return CHAT_TEMPLATE_CHATML;

    /* Lowercase comparison */
    char lower[256];
    strncpy(lower, model_name, sizeof(lower) - 1);
    for (char *p = lower; *p; p++) *p = (*p >= 'A' && *p <= 'Z') ? *p + 32 : *p;

    if (strstr(lower, "qwen")) return CHAT_TEMPLATE_QWEN;
    if (strstr(lower, "llama")) return CHAT_TEMPLATE_LLAMA;
    if (strstr(lower, "mistral")) return CHAT_TEMPLATE_MISTRAL;

    return CHAT_TEMPLATE_CHATML;  /* Default */
}

static char *apply_chat_template(const ChatTemplate *tmpl, const char *system, const char *user) {
    size_t needed = 0;
    if (system && *system) {
        needed += strlen(tmpl->system_prefix) + strlen(system) + strlen(tmpl->system_suffix);
    }
    needed += strlen(tmpl->user_prefix) + strlen(user) + strlen(tmpl->user_suffix);
    needed += strlen(tmpl->assistant_prefix);
    needed += 1;  /* null terminator */

    char *result = (char *)malloc(needed);
    if (!result) return NULL;

    result[0] = '\0';
    if (system && *system) {
        strcat(result, tmpl->system_prefix);
        strcat(result, system);
        strcat(result, tmpl->system_suffix);
    }
    strcat(result, tmpl->user_prefix);
    strcat(result, user);
    strcat(result, tmpl->user_suffix);
    strcat(result, tmpl->assistant_prefix);

    return result;
}

/* ============================================================================
 * EOS Token Handling
 * ============================================================================ */

static bool is_eos_token(const CLIOptions *opt, int token) {
    if (!opt || opt->ignore_eos) return false;
    for (int i = 0; i < opt->eos_count; i++) {
        if (opt->eos_ids[i] == token) return true;
    }
    return false;
}

/**
 * Text-based EOS pattern detection with pending output buffering.
 *
 * When special tokens like <|im_end|> are tokenized as regular text
 * (e.g., !, im, _end, !), we need to detect the pattern in the output
 * and avoid outputting the partial pattern tokens.
 *
 * This is a workaround for tokenizers that don't properly encode special tokens.
 */
#define EOS_PATTERN_BUF_SIZE 64
#define EOS_PENDING_MAX 8

typedef struct {
    char pattern_buf[EOS_PATTERN_BUF_SIZE];  /* Accumulated text for pattern matching */
    int pattern_len;
    char *pending[EOS_PENDING_MAX];          /* Pending token texts (not yet output) */
    int pending_count;
    const char *target_pattern;              /* Pattern to detect */
    const char *partial_prefix;              /* Prefix that might start the pattern */
} EOSPatternState;

static EOSPatternState g_eos_state = {0};

static void eos_pattern_reset(void) {
    g_eos_state.pattern_len = 0;
    g_eos_state.pattern_buf[0] = '\0';
    for (int i = 0; i < g_eos_state.pending_count; i++) {
        free(g_eos_state.pending[i]);
        g_eos_state.pending[i] = NULL;
    }
    g_eos_state.pending_count = 0;
    g_eos_state.target_pattern = NULL;
    g_eos_state.partial_prefix = NULL;
}

static void eos_pattern_init(ChatTemplateType tmpl) {
    eos_pattern_reset();
    switch (tmpl) {
        case CHAT_TEMPLATE_QWEN:
        case CHAT_TEMPLATE_CHATML:
            g_eos_state.target_pattern = "im_end";
            g_eos_state.partial_prefix = "im";
            break;
        case CHAT_TEMPLATE_LLAMA:
        case CHAT_TEMPLATE_MISTRAL:
            g_eos_state.target_pattern = "</s>";
            g_eos_state.partial_prefix = "</";
            break;
        default:
            break;
    }
}

/**
 * Check if token might be start of EOS pattern.
 */
static bool eos_is_potential_prefix(const char *token) {
    if (!token || !g_eos_state.partial_prefix) return false;

    /* Check if current accumulated buffer + token could start the pattern */
    size_t tlen = strlen(token);
    size_t plen = g_eos_state.pattern_len;
    size_t target_len = g_eos_state.target_pattern ? strlen(g_eos_state.target_pattern) : 0;

    /* If buffer + token contains partial match of target, it's a potential prefix */
    if (target_len == 0) return false;

    /* Build temp buffer */
    char temp[EOS_PATTERN_BUF_SIZE];
    if (plen + tlen >= EOS_PATTERN_BUF_SIZE) return false;
    memcpy(temp, g_eos_state.pattern_buf, plen);
    memcpy(temp + plen, token, tlen);
    temp[plen + tlen] = '\0';

    /* Check if temp is a prefix of target or contains start of target */
    const char *target = g_eos_state.target_pattern;
    size_t temp_len = plen + tlen;

    /* Look for any suffix of temp that is a prefix of target */
    for (size_t i = 0; i < temp_len; i++) {
        size_t remaining = temp_len - i;
        if (remaining > target_len) remaining = target_len;
        if (strncmp(temp + i, target, remaining) == 0) {
            return true;
        }
    }

    return false;
}

/**
 * Process a token for EOS pattern detection.
 *
 * @param token_text  The token text to process
 * @param out_buf     Output buffer for safe-to-output text
 * @param out_len     Current length of output buffer
 * @param tmpl        Chat template type
 * @return            true if EOS pattern detected, false otherwise
 */
static bool eos_pattern_process(const char *token_text, char *out_buf, size_t *out_len,
                                 void (*output_fn)(char*, size_t*, const char*),
                                 ChatTemplateType tmpl) {
    if (!token_text || !g_eos_state.target_pattern) {
        /* No pattern to match - output directly */
        if (token_text && output_fn) output_fn(out_buf, out_len, token_text);
        return false;
    }

    /* Append to pattern buffer */
    size_t tlen = strlen(token_text);
    if (g_eos_state.pattern_len + (int)tlen < EOS_PATTERN_BUF_SIZE - 1) {
        memcpy(g_eos_state.pattern_buf + g_eos_state.pattern_len, token_text, tlen);
        g_eos_state.pattern_len += (int)tlen;
        g_eos_state.pattern_buf[g_eos_state.pattern_len] = '\0';
    }

    /* Check if pattern is complete */
    if (strstr(g_eos_state.pattern_buf, g_eos_state.target_pattern)) {
        /* EOS detected - don't output pending tokens */
        eos_pattern_reset();
        return true;
    }

    /* Check if this could still be part of the pattern */
    if (eos_is_potential_prefix(token_text)) {
        /* Hold this token - might be part of EOS */
        if (g_eos_state.pending_count < EOS_PENDING_MAX) {
            g_eos_state.pending[g_eos_state.pending_count] = strdup(token_text);
            g_eos_state.pending_count++;
        }
        return false;
    }

    /* Not part of pattern - flush pending tokens and this one */
    for (int i = 0; i < g_eos_state.pending_count; i++) {
        if (output_fn) output_fn(out_buf, out_len, g_eos_state.pending[i]);
        free(g_eos_state.pending[i]);
        g_eos_state.pending[i] = NULL;
    }
    g_eos_state.pending_count = 0;
    g_eos_state.pattern_len = 0;
    g_eos_state.pattern_buf[0] = '\0';

    if (output_fn) output_fn(out_buf, out_len, token_text);
    return false;
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

/* ============================================================================
 * Prompt Execution
 * ============================================================================ */

static int run_prompt(ModelAPI *api, CKTrueBPE *tokenizer, CLIOptions *opt, const char *input) {
    if (!api || !tokenizer || !opt || !input) return -1;
    if (g_exit_requested) return -1;

    int ctx = opt->context_override;
    if (ctx <= 0 && api->get_context) ctx = api->get_context();
    if (ctx <= 0) ctx = 4096;
    if (ctx > CK_CLI_MAX_CONTEXT) ctx = CK_CLI_MAX_CONTEXT;

    int max_tokens = opt->max_tokens > 0 ? opt->max_tokens : CK_CLI_DEFAULT_MAX_TOKENS;

    /* Apply chat template if enabled */
    const ChatTemplate *tmpl = &g_templates[opt->no_chat_template ? CHAT_TEMPLATE_NONE : opt->chat_template];
    char *formatted = apply_chat_template(tmpl, opt->system_prompt, input);
    if (!formatted) {
        fprintf(stderr, "Error: failed to format prompt\n");
        return -1;
    }

    if (opt->verbose) {
        printf("[DEBUG] Formatted prompt:\n%s\n", formatted);
    }

    int32_t *ids = (int32_t *)malloc((size_t)ctx * sizeof(int32_t));
    if (!ids) {
        fprintf(stderr, "Error: failed to allocate token buffer\n");
        free(formatted);
        return -1;
    }

    int n = ck_true_bpe_encode(tokenizer, formatted, -1, ids, ctx);
    free(formatted);

    if (n <= 0) {
        fprintf(stderr, "[Tokenizer] failed to encode prompt\n");
        free(ids);
        return -1;
    }
    if (n > ctx - max_tokens) {
        n = ctx - max_tokens;
        if (opt->verbose) {
            printf("[DEBUG] Truncated prompt to %d tokens\n", n);
        }
    }

    g_prefill_time_ms = 0.0;
    g_decode_time_ms = 0.0;
    g_decode_count = 0;
    g_prompt_tokens = n;

    if (api->kv_reset) api->kv_reset();

    if (api->embed(ids, n) != 0) {
        fprintf(stderr, "[Model] embed failed\n");
        free(ids);
        return -1;
    }

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

    /* Get vocab size for sampling */
    int vocab_size = api->get_vocab_size ? api->get_vocab_size() : 0;

    /* Helper: sample next token from logits */
    #define SAMPLE_NEXT_TOKEN() do { \
        if (api->get_logits && vocab_size > 0) { \
            float *logits = api->get_logits(); \
            if (logits) { \
                int active = api->get_active_tokens ? api->get_active_tokens() : 1; \
                float *last_logits = logits + (size_t)(active - 1) * vocab_size; \
                float *logits_copy = (float *)malloc(vocab_size * sizeof(float)); \
                memcpy(logits_copy, last_logits, vocab_size * sizeof(float)); \
                next_token = sample_top_p(logits_copy, vocab_size, opt->temperature, opt->top_p); \
                free(logits_copy); \
            } else if (api->sample) { \
                next_token = api->sample(); \
            } else { \
                next_token = -1; \
            } \
        } else if (api->sample) { \
            next_token = api->sample(); \
        } else { \
            next_token = -1; \
        } \
    } while(0)

    /* Sample first token */
    int next_token;
    SAMPLE_NEXT_TOKEN();

    char out_buf[CK_CLI_OUTPUT_BUF_SIZE];
    size_t out_len = 0;

    /* Initialize EOS pattern detection for this prompt */
    eos_pattern_init(opt->chat_template);

    g_generation_active = 1;

    for (int generated = 0; generated < max_tokens && !g_exit_requested && g_generation_active; generated++) {
        if (next_token < 0) break;

        if (opt->verbose) {
            const char *tok_str = ck_true_bpe_id_to_token(tokenizer, next_token);
            fprintf(stderr, "[DEBUG] Token %d: %d (%s)\n", generated, next_token, tok_str ? tok_str : "NULL");
        }

        if (is_eos_token(opt, next_token)) {
            if (opt->verbose) {
                fprintf(stderr, "[DEBUG] EOS detected (token ID), stopping\n");
            }
            break;
        }

        const char *word = ck_true_bpe_id_to_token(tokenizer, next_token);

        /* Process token through EOS pattern detection (buffers potential EOS tokens) */
        if (!opt->ignore_eos &&
            eos_pattern_process(word, out_buf, &out_len, output_token, opt->chat_template)) {
            if (opt->verbose) {
                fprintf(stderr, "[DEBUG] EOS detected (text pattern), stopping\n");
            }
            break;
        }

        if (opt->stream) {
            output_flush(out_buf, &out_len);
            fflush(stdout);
        } else if (out_len > (CK_CLI_OUTPUT_BUF_SIZE / 2)) {
            output_flush(out_buf, &out_len);
            fflush(stdout);
        }

        if (generated + 1 >= max_tokens) break;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (api->decode(next_token, NULL) != 0) {
            fprintf(stderr, "\n[Model] decode failed\n");
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        g_decode_time_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                            (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
        g_decode_count++;

        /* Sample next token */
        SAMPLE_NEXT_TOKEN();
    }

    #undef SAMPLE_NEXT_TOKEN
    g_generation_active = 0;
    output_flush(out_buf, &out_len);
    printf("\n");

    if (opt->timing) {
        double total_ms = g_prefill_time_ms + g_decode_time_ms;
        double prefill_rate = g_prompt_tokens / (g_prefill_time_ms / 1000.0);
        double decode_rate = g_decode_count > 0 ? g_decode_count / (g_decode_time_ms / 1000.0) : 0.0;
        double avg_decode = g_decode_count > 0 ? g_decode_time_ms / g_decode_count : 0.0;

        printf("\033[90m");  /* Gray text */
        printf("prompt: %3d tok / %7.1f ms (%5.1f tok/s) | ", g_prompt_tokens, g_prefill_time_ms, prefill_rate);
        printf("decode: %3d tok / %7.1f ms (%5.1f tok/s, %5.1f ms/tok)\033[0m\n",
               g_decode_count, g_decode_time_ms, decode_rate, avg_decode);
    }
    fflush(stdout);

    free(ids);
    return 0;
}

/* ============================================================================
 * Help & Argument Parsing
 * ============================================================================ */

static void print_banner(void) {
    printf("\n");
    printf("  \033[1;36mC-Kernel-Engine v%s\033[0m\n", CK_CLI_VERSION);
    printf("  Native inference CLI with true-BPE tokenization\n");
    printf("\n");
}

static void print_help(const char *prog) {
    print_banner();
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --model <name>                       Auto-discover model from cache\n", prog);
    fprintf(stderr, "  %s <libmodel.so> <weights.bump>         Direct paths\n", prog);
    fprintf(stderr, "  %s --lib <.so> --weights <.bump>        Named arguments\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --model, -m NAME        Model name (searches in cache)\n");
    fprintf(stderr, "  --lib PATH              Path to compiled model .so\n");
    fprintf(stderr, "  --weights PATH          Path to weights .bump file\n");
    fprintf(stderr, "  --prompt, -p TEXT       Run single prompt (non-interactive)\n");
    fprintf(stderr, "  --system, -S TEXT       System prompt\n");
    fprintf(stderr, "  --max-tokens, -n N      Max tokens to generate (default: %d)\n", CK_CLI_DEFAULT_MAX_TOKENS);
    fprintf(stderr, "  --context, -c N         Override context/KV cache size\n");
    fprintf(stderr, "  --temperature, -T F     Sampling temperature (default: 0.0 = greedy)\n");
    fprintf(stderr, "  --top-p F               Nucleus sampling top-p (default: 0.9)\n");
    fprintf(stderr, "  --stream, -s            Stream tokens as generated\n");
    fprintf(stderr, "  --timing, -t            Show timing breakdown\n");
    fprintf(stderr, "  --no-chat-template      Disable chat template formatting\n");
    fprintf(stderr, "  --eos IDS               Comma-separated EOS token IDs\n");
    fprintf(stderr, "  --ignore-eos            Do not stop on EOS tokens\n");
    fprintf(stderr, "  --list                  List available models\n");
    fprintf(stderr, "  --verbose, -v           Verbose output\n");
    fprintf(stderr, "  --help, -h              Show this help\n");
    fprintf(stderr, "\nREPL Commands:\n");
    fprintf(stderr, "  /exit, /quit            Exit the REPL\n");
    fprintf(stderr, "  /reset                  Reset KV cache\n");
    fprintf(stderr, "  /timing                 Toggle timing display\n");
    fprintf(stderr, "  /temp <value>           Set temperature\n");
    fprintf(stderr, "  /system <text>          Set system prompt\n");
    fprintf(stderr, "  /help                   Show help\n");
}

static bool parse_args(int argc, char **argv, CLIOptions *opt) {
    if (!opt) return false;
    memset(opt, 0, sizeof(*opt));
    opt->max_tokens = CK_CLI_DEFAULT_MAX_TOKENS;
    opt->temperature = 0.0f;  /* Greedy by default */
    opt->top_p = 0.9f;
    opt->stream = true;  /* Stream by default */
    opt->timing = true;  /* Show timing by default */
    /* Default EOS tokens for Qwen/ChatML */
    opt->eos_ids[0] = 151643;  /* <|im_end|> */
    opt->eos_ids[1] = 151645;  /* <|endoftext|> */
    opt->eos_ids[2] = 151644;  /* <|im_sep|> */
    opt->eos_count = 3;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            print_help(argv[0]);
            return false;
        } else if (!strcmp(arg, "--list")) {
            list_available_models();
            return false;
        } else if ((!strcmp(arg, "--model") || !strcmp(arg, "-m")) && i + 1 < argc) {
            opt->model_name = argv[++i];
        } else if (!strcmp(arg, "--lib") && i + 1 < argc) {
            opt->lib_path = argv[++i];
        } else if (!strcmp(arg, "--weights") && i + 1 < argc) {
            opt->weights_path = argv[++i];
        } else if ((!strcmp(arg, "--prompt") || !strcmp(arg, "-p")) && i + 1 < argc) {
            opt->prompt_once = argv[++i];
        } else if ((!strcmp(arg, "--system") || !strcmp(arg, "-S")) && i + 1 < argc) {
            opt->system_prompt = argv[++i];
        } else if ((!strcmp(arg, "--max-tokens") || !strcmp(arg, "-n")) && i + 1 < argc) {
            opt->max_tokens = atoi(argv[++i]);
        } else if ((!strcmp(arg, "--context") || !strcmp(arg, "-c")) && i + 1 < argc) {
            opt->context_override = atoi(argv[++i]);
        } else if ((!strcmp(arg, "--temperature") || !strcmp(arg, "-T")) && i + 1 < argc) {
            opt->temperature = (float)atof(argv[++i]);
        } else if (!strcmp(arg, "--top-p") && i + 1 < argc) {
            opt->top_p = (float)atof(argv[++i]);
        } else if (!strcmp(arg, "--stream") || !strcmp(arg, "-s")) {
            opt->stream = true;
        } else if (!strcmp(arg, "--no-stream")) {
            opt->stream = false;
        } else if (!strcmp(arg, "--timing") || !strcmp(arg, "-t")) {
            opt->timing = true;
        } else if (!strcmp(arg, "--no-timing")) {
            opt->timing = false;
        } else if (!strcmp(arg, "--no-chat-template")) {
            opt->no_chat_template = true;
        } else if (!strcmp(arg, "--eos") && i + 1 < argc) {
            parse_eos_ids(argv[++i], opt);
        } else if (!strcmp(arg, "--ignore-eos")) {
            opt->ignore_eos = true;
        } else if (!strcmp(arg, "--verbose") || !strcmp(arg, "-v")) {
            opt->verbose = true;
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

    /* Auto-discover model if --model specified */
    if (opt->model_name && (!opt->lib_path || !opt->weights_path)) {
        static char lib_buf[4096], weights_buf[4096];
        if (find_model_in_cache(opt->model_name, lib_buf, weights_buf, sizeof(lib_buf))) {
            opt->lib_path = lib_buf;
            opt->weights_path = weights_buf;
        } else {
            fprintf(stderr, "Error: model '%s' not found in cache\n", opt->model_name);
            fprintf(stderr, "Run with --list to see available models\n");
            return false;
        }
    }

    if (!opt->lib_path || !opt->weights_path) {
        print_help(argv[0]);
        return false;
    }

    /* Auto-detect chat template from model name/path */
    const char *name_for_template = opt->model_name ? opt->model_name : opt->lib_path;
    opt->chat_template = detect_chat_template(name_for_template);

    /* Load EOS tokens from vocab.json if available */
    if (load_eos_from_vocab_json(opt->weights_path, opt)) {
        if (opt->verbose) {
            printf("[DEBUG] Loaded %d EOS tokens: ", opt->eos_count);
            for (int i = 0; i < opt->eos_count; i++) {
                printf("%d ", opt->eos_ids[i]);
            }
            printf("\n");
        }
    }

    return true;
}

/* ============================================================================
 * REPL Command Processing
 * ============================================================================ */

static bool process_repl_command(const char *line, CLIOptions *opt, ModelAPI *api) {
    if (!line || line[0] != '/') return false;

    if (!strncmp(line, "/exit", 5) || !strncmp(line, "/quit", 5)) {
        g_exit_requested = 1;
        return true;
    }
    if (!strncmp(line, "/help", 5)) {
        printf("REPL Commands:\n");
        printf("  /exit, /quit        Exit\n");
        printf("  /reset              Reset KV cache\n");
        printf("  /timing             Toggle timing display\n");
        printf("  /temp <value>       Set temperature (0 = greedy)\n");
        printf("  /top-p <value>      Set top-p\n");
        printf("  /system <text>      Set system prompt\n");
        printf("  /clear              Clear system prompt\n");
        printf("  /verbose            Toggle verbose mode\n");
        return true;
    }
    if (!strncmp(line, "/reset", 6)) {
        if (api->kv_reset) {
            api->kv_reset();
            printf("[KV cache reset]\n");
        }
        return true;
    }
    if (!strncmp(line, "/timing", 7)) {
        opt->timing = !opt->timing;
        printf("[Timing %s]\n", opt->timing ? "enabled" : "disabled");
        return true;
    }
    if (!strncmp(line, "/verbose", 8)) {
        opt->verbose = !opt->verbose;
        printf("[Verbose %s]\n", opt->verbose ? "enabled" : "disabled");
        return true;
    }
    if (!strncmp(line, "/temp ", 6)) {
        opt->temperature = (float)atof(line + 6);
        printf("[Temperature set to %.2f]\n", opt->temperature);
        return true;
    }
    if (!strncmp(line, "/top-p ", 7)) {
        opt->top_p = (float)atof(line + 7);
        printf("[Top-p set to %.2f]\n", opt->top_p);
        return true;
    }
    if (!strncmp(line, "/system ", 8)) {
        opt->system_prompt = strdup(line + 8);
        printf("[System prompt set]\n");
        return true;
    }
    if (!strncmp(line, "/clear", 6)) {
        opt->system_prompt = NULL;
        printf("[System prompt cleared]\n");
        return true;
    }

    printf("Unknown command: %s\n", line);
    return true;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char **argv) {
    signal(SIGINT, handle_sigint);
    srand((unsigned int)time(NULL));

    CLIOptions opt;
    if (!parse_args(argc, argv, &opt)) {
        return 1;
    }

    print_banner();
    printf("Loading: %s\n", opt.lib_path);

    ModelAPI api;
    if (!load_model_api(opt.lib_path, &api)) {
        return 1;
    }

    printf("Initializing model...\n");
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

    /* Register special tokens for pre-BPE matching.
     * This is done in the CLI (orchestrator), NOT the generated model code.
     * The generated model code stays "dumb" - just inference.
     * Model-specific token handling is the CLI's responsibility.
     */
    {
        /* Common special tokens across model families */
        static const char *special_tokens[] = {
            /* Qwen/ChatML */
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            /* Llama 3 */
            "<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>",
            /* Generic */
            "</s>", "<s>", "<pad>", "<unk>",
            NULL
        };
        int registered = 0;
        for (int i = 0; special_tokens[i] != NULL; i++) {
            int32_t id = ck_true_bpe_lookup(tokenizer, special_tokens[i]);
            /* Verify it's actually this token (not unk) via round-trip */
            const char *check = ck_true_bpe_id_to_token(tokenizer, id);
            if (check && strcmp(check, special_tokens[i]) == 0) {
                ck_true_bpe_add_special_token(tokenizer, special_tokens[i], id);
                registered++;
                if (opt.verbose) {
                    printf("[Tokenizer] Registered special: %s -> %d\n", special_tokens[i], id);
                }
            }
        }
        if (opt.verbose) {
            printf("[Tokenizer] Registered %d special tokens for pre-BPE matching\n", registered);
        }
    }

    printf("Ready! Vocab: %d, Context: %d, Template: %s\n",
           vocab_size, ctx,
           opt.no_chat_template ? "none" :
           opt.chat_template == CHAT_TEMPLATE_QWEN ? "qwen" :
           opt.chat_template == CHAT_TEMPLATE_LLAMA ? "llama" :
           opt.chat_template == CHAT_TEMPLATE_MISTRAL ? "mistral" : "chatml");

    /* Print CPU capability info */
    ck_capability_t cap = ck_get_capabilities();
    printf("[Hardware] %s | Vector: %d-bit | FMA: %s | AI Accel: %s | Kernel: %s\n",
           cap.name, cap.width, cap.has_fma ? "Yes" : "No",
           cap.has_ai_accel ? "Yes" : "No", cap.best_kernel);

    printf("Type /help for commands, Ctrl+C to stop generation\n\n");

    setvbuf(stdout, NULL, _IOFBF, 1 << 20);

    if (opt.prompt_once) {
        run_prompt(&api, tokenizer, &opt, opt.prompt_once);
    } else {
        /* REPL */
#ifdef HAVE_READLINE
        char *home = getenv("HOME");
        char history_path[4096];
        if (home) {
            snprintf(history_path, sizeof(history_path), "%s/%s", home, CK_CLI_HISTORY_FILE);
            read_history(history_path);
        }
#endif

        while (!g_exit_requested) {
#ifdef HAVE_READLINE
            char *line = readline("\033[1;32mYou:\033[0m ");
            if (!line) break;
            if (*line) add_history(line);
#else
            printf("\033[1;32mYou:\033[0m ");
            fflush(stdout);
            char line_buf[4096];
            if (!fgets(line_buf, sizeof(line_buf), stdin)) {
                if (feof(stdin) || g_exit_requested) break;
                if (errno == EINTR) break;
                continue;
            }
            /* Remove trailing newline */
            size_t len = strlen(line_buf);
            if (len > 0 && line_buf[len-1] == '\n') line_buf[len-1] = '\0';
            char *line = line_buf;
#endif

            if (line[0] == '\0') {
#ifdef HAVE_READLINE
                free(line);
#endif
                continue;
            }

            if (line[0] == '/') {
                process_repl_command(line, &opt, &api);
#ifdef HAVE_READLINE
                free(line);
#endif
                continue;
            }

            printf("\033[1;34mAssistant:\033[0m ");
            fflush(stdout);
            run_prompt(&api, tokenizer, &opt, line);

#ifdef HAVE_READLINE
            free(line);
#endif
        }

#ifdef HAVE_READLINE
        if (home) {
            write_history(history_path);
        }
#endif
    }

    ck_true_bpe_free(tokenizer);
    if (api.free_fn) api.free_fn();
    if (api.handle) dlclose(api.handle);

    printf("\nGoodbye!\n");
    return 0;
}
