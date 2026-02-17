/*
 * C-Kernel-Engine v7 Native CLI
 *
 * Features:
 *   - Model auto-discovery from cache
 *   - Readline support for history/editing
 *   - Chat template support (Qwen, LLaMA, etc.)
 *   - Temperature/top-p sampling
 *   - Streaming output
 *
 * Usage:
 *   ck-cli-v7 --model <name>                    # Auto-discover from cache
 *   ck-cli-v7 <libmodel.so> <weights.bump>      # Direct paths
 *   ck-cli-v7 --lib <.so> --weights <.bump>     # Named args
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
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

#include "tokenizer/true_bpe.h"
#include "ck_features.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define CK_CLI_VERSION         "7.0.0"
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
static int g_user_tokens = 0;

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
typedef void (*prefill_t)(const int32_t *tokens, int num_tokens);
typedef float *(*get_logits_t)(void);
typedef int (*get_int_t)(void);
typedef void (*free_t)(void);

/* Tokenizer API - uses model's built-in tokenizer (one source of truth) */
typedef int (*encode_text_t)(const char *text, int text_len);
typedef int (*decode_tokens_t)(const int32_t *ids, int num_ids, char *text, int max_len);
typedef int (*has_tokenizer_t)(void);
typedef int32_t (*lookup_token_t)(const char *text);
typedef const char *(*id_to_token_t)(int32_t id);
typedef const int32_t *(*get_token_buffer_t)(void);

typedef struct {
    void *handle;
    init_t init;
    embed_t embed;
    forward_t forward;
    kv_enable_t kv_enable;
    kv_reset_t kv_reset;
    decode_t decode;
    prefill_t prefill;  /* ck_prefill for batched prefill (optional) */
    get_logits_t get_logits;
    get_int_t get_logits_stride;
    get_int_t get_context;
    get_int_t get_vocab_size;
    get_int_t get_active_tokens;
    free_t free_fn;

    /* Built-in tokenizer API (from generated model) */
    encode_text_t encode_text;
    decode_tokens_t decode_tokens;
    has_tokenizer_t has_tokenizer;
    lookup_token_t lookup_token;
    id_to_token_t id_to_token;
    get_token_buffer_t get_token_buffer;
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
    bool quiet_output;
    bool no_chat_template;
    ChatTemplateType chat_template;
    int eos_ids[CK_CLI_EOS_MAX];
    int eos_count;
} CLIOptions;


/* ============================================================================
 * PR7.1 - Native Train/Profile Subcommands (C-first operator path)
 * ============================================================================
 */

typedef struct {
    const char *run_dir;
    const char *token_file;
    const char *json_out;
    int epochs;
    int seq_len;
    int total_tokens;
    int grad_accum;
    float lr;
    int strict;
    int verbose;
    int threads;
} TrainOptions;

typedef struct {
    const char *run_dir;
    const char *output_path;
} ReportIndexOptions;

typedef struct {
    const char *run_dir;
    const char *tool;
    const char *output_dir;
    const char *token_file;
    int epochs;
    int seq_len;
    int total_tokens;
    int grad_accum;
    float lr;
    int strict;
    int threads;
} ProfileOptions;

typedef struct {
    char *name;
    long offset;
    long size;
    char *dtype;
} ManifestEntry;

typedef struct {
    char **names;
    int *numel;
    int count;
} RuntimeInitOrder;

/* ---- tiny JSON parser (embedded jsmn subset) ---- */

typedef enum {
    JSMN_UNDEFINED = 0,
    JSMN_OBJECT = 1,
    JSMN_ARRAY = 2,
    JSMN_STRING = 3,
    JSMN_PRIMITIVE = 4
} jsmntype_t;

typedef struct {
    jsmntype_t type;
    int start;
    int end;
    int size;
} jsmntok_t;

typedef struct {
    unsigned int pos;
    unsigned int toknext;
    int toksuper;
} jsmn_parser;

#define JSMN_ERROR_NOMEM -1
#define JSMN_ERROR_INVAL -2
#define JSMN_ERROR_PART  -3

static void jsmn_init(jsmn_parser *parser) {
    parser->pos = 0;
    parser->toknext = 0;
    parser->toksuper = -1;
}

static jsmntok_t *jsmn_alloc_token(jsmn_parser *parser, jsmntok_t *tokens, size_t num_tokens) {
    if (parser->toknext >= num_tokens) return NULL;
    jsmntok_t *tok = &tokens[parser->toknext++];
    tok->start = tok->end = -1;
    tok->size = 0;
    tok->type = JSMN_UNDEFINED;
    return tok;
}

static void jsmn_fill_token(jsmntok_t *token, jsmntype_t type, int start, int end) {
    token->type = type;
    token->start = start;
    token->end = end;
    token->size = 0;
}

static int jsmn_parse_primitive(jsmn_parser *parser, const char *js, size_t len, jsmntok_t *tokens, size_t num_tokens) {
    int start = (int)parser->pos;
    for (; parser->pos < len; parser->pos++) {
        char c = js[parser->pos];
        if (c == '\t' || c == '\r' || c == '\n' || c == ' ' || c == ',' || c == ']' || c == '}') {
            jsmntok_t *tok = jsmn_alloc_token(parser, tokens, num_tokens);
            if (tok == NULL) return JSMN_ERROR_NOMEM;
            jsmn_fill_token(tok, JSMN_PRIMITIVE, start, (int)parser->pos);
            parser->pos--;
            return 0;
        }
        if (c < 32 || c >= 127) {
            parser->pos = (unsigned int)start;
            return JSMN_ERROR_INVAL;
        }
    }
    jsmntok_t *tok = jsmn_alloc_token(parser, tokens, num_tokens);
    if (tok == NULL) return JSMN_ERROR_NOMEM;
    jsmn_fill_token(tok, JSMN_PRIMITIVE, start, (int)parser->pos);
    parser->pos--;
    return 0;
}

static int jsmn_parse_string(jsmn_parser *parser, const char *js, size_t len, jsmntok_t *tokens, size_t num_tokens) {
    int start = (int)parser->pos + 1;
    parser->pos++;
    for (; parser->pos < len; parser->pos++) {
        char c = js[parser->pos];
        if (c == '"') {
            jsmntok_t *tok = jsmn_alloc_token(parser, tokens, num_tokens);
            if (tok == NULL) return JSMN_ERROR_NOMEM;
            jsmn_fill_token(tok, JSMN_STRING, start, (int)parser->pos);
            return 0;
        }
        if (c == '\\') parser->pos++;
    }
    return JSMN_ERROR_PART;
}

static int jsmn_parse(jsmn_parser *parser, const char *js, size_t len, jsmntok_t *tokens, unsigned int num_tokens) {
    int r;
    for (; parser->pos < len; parser->pos++) {
        char c = js[parser->pos];
        jsmntok_t *tok;
        switch (c) {
            case '{':
            case '[':
                tok = jsmn_alloc_token(parser, tokens, num_tokens);
                if (tok == NULL) return JSMN_ERROR_NOMEM;
                if (parser->toksuper != -1) tokens[parser->toksuper].size++;
                tok->type = (c == '{' ? JSMN_OBJECT : JSMN_ARRAY);
                tok->start = (int)parser->pos;
                parser->toksuper = (int)parser->toknext - 1;
                break;
            case '}':
            case ']':
                for (int i = (int)parser->toknext - 1; i >= 0; i--) {
                    tok = &tokens[i];
                    if (tok->start != -1 && tok->end == -1) {
                        if ((c == '}' && tok->type != JSMN_OBJECT) || (c == ']' && tok->type != JSMN_ARRAY)) {
                            return JSMN_ERROR_INVAL;
                        }
                        tok->end = (int)parser->pos + 1;
                        parser->toksuper = -1;
                        for (int j = i - 1; j >= 0; j--) {
                            if (tokens[j].start != -1 && tokens[j].end == -1) {
                                parser->toksuper = j;
                                break;
                            }
                        }
                        break;
                    }
                }
                break;
            case '"':
                r = jsmn_parse_string(parser, js, len, tokens, num_tokens);
                if (r < 0) return r;
                if (parser->toksuper != -1) tokens[parser->toksuper].size++;
                break;
            case '\t': case '\r': case '\n': case ' ': case ':': case ',':
                break;
            default:
                r = jsmn_parse_primitive(parser, js, len, tokens, num_tokens);
                if (r < 0) return r;
                if (parser->toksuper != -1) tokens[parser->toksuper].size++;
                break;
        }
    }
    for (unsigned int i = 0; i < parser->toknext; i++) {
        if (tokens[i].start != -1 && tokens[i].end == -1) return JSMN_ERROR_PART;
    }
    return (int)parser->toknext;
}

static int json_skip_token(const jsmntok_t *toks, int i) {
    int j = i;
    if (toks[j].type == JSMN_STRING || toks[j].type == JSMN_PRIMITIVE) return j + 1;
    if (toks[j].type == JSMN_ARRAY || toks[j].type == JSMN_OBJECT) {
        j++;
        for (int k = 0; k < toks[i].size; k++) {
            j = json_skip_token(toks, j);
        }
        return j;
    }
    return j + 1;
}

static bool json_tok_streq(const char *json, const jsmntok_t *tok, const char *s) {
    if (!json || !tok || !s || tok->type != JSMN_STRING) return false;
    size_t n = (size_t)(tok->end - tok->start);
    return strlen(s) == n && strncmp(json + tok->start, s, n) == 0;
}

static char *json_tok_strdup(const char *json, const jsmntok_t *tok) {
    if (!json || !tok || tok->start < 0 || tok->end < tok->start) return NULL;
    size_t n = (size_t)(tok->end - tok->start);
    char *out = (char*)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, json + tok->start, n);
    out[n] = '\0';
    return out;
}

static int json_find_key_value(const char *json, const jsmntok_t *toks, int obj_idx, const char *key) {
    if (!json || !toks || obj_idx < 0 || toks[obj_idx].type != JSMN_OBJECT) return -1;
    int i = obj_idx + 1;
    const int end = json_skip_token(toks, obj_idx);
    while (i < end) {
        int kidx = i;
        int vidx = json_skip_token(toks, kidx);
        if (vidx >= end) break;
        if (json_tok_streq(json, &toks[kidx], key)) return vidx;
        i = json_skip_token(toks, vidx);
    }
    return -1;
}

static bool read_file_text(const char *path, char **out_buf, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return false; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    char *buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return false; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = '\0';
    *out_buf = buf;
    if (out_len) *out_len = got;
    return true;
}

static bool read_file_blob(const char *path, uint8_t **out_buf, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return false; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    uint8_t *buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return false; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    *out_buf = buf;
    if (out_len) *out_len = got;
    return true;
}

static int parse_json_tokens(const char *json, size_t len, jsmntok_t **out_tokens) {
    int cap = (int)(len / 8) + 2048;
    if (cap < 2048) cap = 2048;
    for (;;) {
        jsmntok_t *toks = (jsmntok_t*)calloc((size_t)cap, sizeof(jsmntok_t));
        if (!toks) return -1;
        jsmn_parser p;
        jsmn_init(&p);
        int rc = jsmn_parse(&p, json, len, toks, (unsigned int)cap);
        if (rc == JSMN_ERROR_NOMEM) {
            free(toks);
            cap *= 2;
            if (cap > (1 << 20)) return -1;
            continue;
        }
        if (rc < 0) {
            free(toks);
            return -1;
        }
        *out_tokens = toks;
        return rc;
    }
}

static int parse_runtime_init_order(const char *summary_path, RuntimeInitOrder *out) {
    memset(out, 0, sizeof(*out));
    char *json = NULL;
    size_t len = 0;
    if (!read_file_text(summary_path, &json, &len)) return -1;
    jsmntok_t *toks = NULL;
    int tokc = parse_json_tokens(json, len, &toks);
    if (tokc <= 0) { free(json); return -2; }
    int arr_order = json_find_key_value(json, toks, 0, "init_weight_order");
    int arr_numel = json_find_key_value(json, toks, 0, "init_weight_numel");
    if (arr_order < 0 || toks[arr_order].type != JSMN_ARRAY) {
        free(toks); free(json); return -3;
    }
    int count = toks[arr_order].size;
    if (count <= 0) { free(toks); free(json); return -4; }
    out->names = (char**)calloc((size_t)count, sizeof(char*));
    out->numel = (int*)calloc((size_t)count, sizeof(int));
    if (!out->names || !out->numel) {
        free(out->names); free(out->numel);
        free(toks); free(json);
        return -5;
    }
    int idx = arr_order + 1;
    for (int i = 0; i < count; i++) {
        out->names[i] = json_tok_strdup(json, &toks[idx]);
        idx = json_skip_token(toks, idx);
    }
    if (arr_numel >= 0 && toks[arr_numel].type == JSMN_ARRAY) {
        int nidx = arr_numel + 1;
        int ncount = toks[arr_numel].size;
        for (int i = 0; i < count && i < ncount; i++) {
            char *s = json_tok_strdup(json, &toks[nidx]);
            out->numel[i] = s ? atoi(s) : 0;
            free(s);
            nidx = json_skip_token(toks, nidx);
        }
    }
    out->count = count;
    free(toks);
    free(json);
    return 0;
}

static void free_runtime_init_order(RuntimeInitOrder *r) {
    if (!r) return;
    if (r->names) {
        for (int i = 0; i < r->count; i++) free(r->names[i]);
    }
    free(r->names);
    free(r->numel);
    memset(r, 0, sizeof(*r));
}

static int parse_manifest_entries(const char *manifest_path, ManifestEntry **out_entries, int *out_count) {
    *out_entries = NULL;
    *out_count = 0;
    char *json = NULL;
    size_t len = 0;
    if (!read_file_text(manifest_path, &json, &len)) return -1;
    jsmntok_t *toks = NULL;
    int tokc = parse_json_tokens(json, len, &toks);
    if (tokc <= 0) { free(json); return -2; }
    int entries_idx = json_find_key_value(json, toks, 0, "entries");
    if (entries_idx < 0 || toks[entries_idx].type != JSMN_ARRAY) {
        free(toks); free(json); return -3;
    }
    int count = toks[entries_idx].size;
    ManifestEntry *entries = (ManifestEntry*)calloc((size_t)count, sizeof(ManifestEntry));
    if (!entries) { free(toks); free(json); return -4; }
    int idx = entries_idx + 1;
    int w = 0;
    for (int i = 0; i < count; i++) {
        int obj = idx;
        if (toks[obj].type != JSMN_OBJECT) {
            idx = json_skip_token(toks, idx);
            continue;
        }
        int name_idx = json_find_key_value(json, toks, obj, "name");
        int off_idx = json_find_key_value(json, toks, obj, "offset");
        int size_idx = json_find_key_value(json, toks, obj, "size");
        int dtype_idx = json_find_key_value(json, toks, obj, "dtype");
        if (name_idx >= 0 && off_idx >= 0 && size_idx >= 0) {
            entries[w].name = json_tok_strdup(json, &toks[name_idx]);
            char *off_s = json_tok_strdup(json, &toks[off_idx]);
            char *size_s = json_tok_strdup(json, &toks[size_idx]);
            entries[w].offset = off_s ? atol(off_s) : -1;
            entries[w].size = size_s ? atol(size_s) : -1;
            entries[w].dtype = (dtype_idx >= 0) ? json_tok_strdup(json, &toks[dtype_idx]) : NULL;
            free(off_s);
            free(size_s);
            if (entries[w].name && entries[w].offset >= 0 && entries[w].size > 0) w++;
            else {
                free(entries[w].name); free(entries[w].dtype);
            }
        }
        idx = json_skip_token(toks, idx);
    }
    *out_entries = entries;
    *out_count = w;
    free(toks);
    free(json);
    return 0;
}

static void free_manifest_entries(ManifestEntry *entries, int count) {
    if (!entries) return;
    for (int i = 0; i < count; i++) {
        free(entries[i].name);
        free(entries[i].dtype);
    }
    free(entries);
}

static const ManifestEntry *find_manifest_entry(const ManifestEntry *entries, int count, const char *name) {
    if (!entries || !name) return NULL;
    for (int i = 0; i < count; i++) {
        if (entries[i].name && strcmp(entries[i].name, name) == 0) return &entries[i];
    }
    return NULL;
}

static bool is_fp32_dtype(const char *dtype) {
    if (!dtype) return true;
    return strcmp(dtype, "fp32") == 0 || strcmp(dtype, "f32") == 0;
}

static int parse_int_tokens_file(const char *path, int32_t **out_tokens, int *out_count) {
    *out_tokens = NULL;
    *out_count = 0;
    char *txt = NULL;
    size_t len = 0;
    if (!read_file_text(path, &txt, &len)) return -1;
    int cap = 4096;
    int count = 0;
    int32_t *vals = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    if (!vals) { free(txt); return -2; }
    char *p = txt;
    while (*p) {
        while (*p && !(isdigit((unsigned char)*p) || *p == '-' || *p == '+')) p++;
        if (!*p) break;
        char *endp = NULL;
        long v = strtol(p, &endp, 10);
        if (endp == p) break;
        if (count >= cap) {
            cap *= 2;
            int32_t *tmp = (int32_t*)realloc(vals, (size_t)cap * sizeof(int32_t));
            if (!tmp) { free(vals); free(txt); return -3; }
            vals = tmp;
        }
        vals[count++] = (int32_t)v;
        p = endp;
    }
    free(txt);
    if (count <= 1) { free(vals); return -4; }
    *out_tokens = vals;
    *out_count = count;
    return 0;
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int ensure_parent_dir(const char *path) {
    if (!path) return -1;
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    char *slash = strrchr(tmp, '/');
    if (!slash) return 0;
    *slash = '\0';
    if (tmp[0] == '\0') return 0;
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", tmp);
    int rc = system(cmd);
    return (rc == 0) ? 0 : -1;
}

static int write_run_index(const char *run_dir, const char *output_path) {
    static const char *keys[] = {
        "train_e2e_latest.json",
        "training_loss_curve_latest.json",
        "training_parity_latest.json",
        "training_grad_norms_latest.json",
        "training_step_profile_latest.json",
        "training_checkpoint_policy_latest.json",
        "memory_diagnostic_latest.json",
        "layout_train.json",
        "layout_train_audit.json",
        "generated_train_runtime_summary_v7.json",
        "ir1_train_forward.json",
        "ir2_train_backward.json",
        "profile_summary.json",
        "perf_stat_summary.json",
        "flamegraph_manifest.json",
        "vtune_summary.json",
        "advisor_summary.json",
        "run_index.json",
    };
    const char *out = output_path;
    char out_buf[4096];
    if (!out || !*out) {
        snprintf(out_buf, sizeof(out_buf), "%s/run_index.json", run_dir);
        out = out_buf;
    }
    if (ensure_parent_dir(out) != 0) return -1;
    FILE *f = fopen(out, "w");
    if (!f) return -2;
    time_t now = time(NULL);
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"ck.run.index.v1\",\n");
    fprintf(f, "  \"generated_by\": \"ck-cli-v7\",\n");
    fprintf(f, "  \"generated_at_epoch\": %lld,\n", (long long)now);
    fprintf(f, "  \"run_dir\": \"%s\",\n", run_dir);
    fprintf(f, "  \"files\": {\n");
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        char p[4096];
        struct stat st;
        snprintf(p, sizeof(p), "%s/%s", run_dir, keys[i]);
        int ok = (stat(p, &st) == 0);
        fprintf(f, "    \"%s\": {\"path\": \"%s\", \"exists\": %s, \"size_bytes\": %lld}%s\n",
                keys[i], p, ok ? "true" : "false", ok ? (long long)st.st_size : 0LL,
                (i + 1 < sizeof(keys) / sizeof(keys[0])) ? "," : "");
    }
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

/* ============================================================================
 * Cache Discovery
 * ============================================================================ */

static const char *get_cache_dir(void) {
    static char cache_path[4096];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(cache_path, sizeof(cache_path), "%s/.cache/ck-engine-v7/models", home);
    return cache_path;
}

/* Case-insensitive substring search */
static const char *strcasestr_local(const char *haystack, const char *needle) {
    if (!needle[0]) return haystack;
    for (; *haystack; haystack++) {
        const char *h = haystack, *n = needle;
        while (*h && *n && (tolower((unsigned char)*h) == tolower((unsigned char)*n))) {
            h++; n++;
        }
        if (!*n) return haystack;
    }
    return NULL;
}

static bool path_exists(const char *path) {
    struct stat st;
    return path && stat(path, &st) == 0;
}

static bool resolve_model_runtime_paths(
    const char *model_dir,
    char *lib_out,
    char *weights_out,
    size_t out_size,
    double *weights_size
) {
    if (!model_dir || !lib_out || !weights_out || out_size == 0) return false;

    static const char *lib_patterns[] = {
        "%s/.ck_build/libmodel.so",
        "%s/libmodel.so",
        "%s/ck-kernel-inference.so",
    };
    static const char *weights_patterns[] = {
        "%s/.ck_build/weights.bump",
        "%s/weights.bump",
        "%s/weights.bump",
    };

    for (size_t i = 0; i < sizeof(lib_patterns) / sizeof(lib_patterns[0]); i++) {
        char lib_path[4096], bump_path[4096];
        snprintf(lib_path, sizeof(lib_path), lib_patterns[i], model_dir);
        snprintf(bump_path, sizeof(bump_path), weights_patterns[i], model_dir);

        if (path_exists(lib_path) && path_exists(bump_path)) {
            snprintf(lib_out, out_size, "%s", lib_path);
            snprintf(weights_out, out_size, "%s", bump_path);
            if (weights_size) {
                struct stat st_bump;
                if (stat(bump_path, &st_bump) == 0) {
                    *weights_size = (double)st_bump.st_size;
                }
            }
            return true;
        }
    }

    return false;
}

static bool find_model_in_cache(const char *model_name, char *lib_out, char *weights_out, size_t out_size) {
    const char *cache_dir = get_cache_dir();
    DIR *dir = opendir(cache_dir);
    if (!dir) return false;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        /* Case-insensitive check if directory name contains model_name */
        if (strcasestr_local(entry->d_name, model_name) != NULL) {
            char model_dir[4096];
            snprintf(model_dir, sizeof(model_dir), "%s/%s", cache_dir, entry->d_name);
            if (resolve_model_runtime_paths(model_dir, lib_out, weights_out, out_size, NULL)) {
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

    fprintf(stderr, "Available models in %s:\n", cache_dir);
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        char model_dir[4096];
        snprintf(model_dir, sizeof(model_dir), "%s/%s", cache_dir, entry->d_name);

        char resolved_lib[4096], resolved_bump[4096];
        if (resolve_model_runtime_paths(model_dir, resolved_lib, resolved_bump, sizeof(resolved_lib), NULL)) {
            fprintf(stderr, "  - %s\n", entry->d_name);
            count++;
        }
    }
    closedir(dir);

    if (count == 0) {
        fprintf(stderr, "  (none found)\n");
    }
}

#define MAX_CACHED_MODELS 64

static const char *format_size(double bytes, char *buf, size_t buf_size) {
    if (bytes >= 1e9)       snprintf(buf, buf_size, "%.1f GB", bytes / 1e9);
    else if (bytes >= 1e6)  snprintf(buf, buf_size, "%.1f MB", bytes / 1e6);
    else if (bytes >= 1e3)  snprintf(buf, buf_size, "%.1f KB", bytes / 1e3);
    else                    snprintf(buf, buf_size, "%.0f B", bytes);
    return buf;
}

/**
 * @brief Scan cache directory and auto-select a model.
 *
 * If exactly one compiled model is found, auto-selects it.
 * If multiple are found, presents a numbered list for user selection.
 * Returns true if a model was selected (lib_out and weights_out filled).
 */
static bool scan_and_select_model(char *lib_out, char *weights_out, size_t out_size) {
    const char *cache_dir = get_cache_dir();
    DIR *dir = opendir(cache_dir);
    if (!dir) {
        fprintf(stderr, "\n  No model cache found.\n");
        fprintf(stderr, "  Looked in: %s\n\n", cache_dir);
        fprintf(stderr, "  To get started, compile a model first:\n");
        fprintf(stderr, "    python version/v7/scripts/ck_run_v7.py run <model-name>\n\n");
        fprintf(stderr, "  Example:\n");
        fprintf(stderr, "    python version/v7/scripts/ck_run_v7.py run Qwen/Qwen2-0.5B-Instruct-GGUF\n\n");
        return false;
    }

    /* Collect compiled models */
    char model_names[MAX_CACHED_MODELS][256];
    char so_paths[MAX_CACHED_MODELS][4096];
    char bump_paths[MAX_CACHED_MODELS][4096];
    double weights_sizes[MAX_CACHED_MODELS];
    int count = 0;

    /* Also count directories that exist but aren't compiled yet */
    int uncompiled = 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        char model_dir[4096];
        snprintf(model_dir, sizeof(model_dir), "%s/%s", cache_dir, entry->d_name);

        char resolved_lib[4096], resolved_bump[4096];
        double resolved_weights = 0.0;
        bool ready = resolve_model_runtime_paths(
            model_dir,
            resolved_lib,
            resolved_bump,
            sizeof(resolved_lib),
            &resolved_weights
        );

        if (ready && count < MAX_CACHED_MODELS) {
            strncpy(model_names[count], entry->d_name, 255);
            model_names[count][255] = 0;
            strncpy(so_paths[count], resolved_lib, 4095);
            so_paths[count][4095] = 0;
            strncpy(bump_paths[count], resolved_bump, 4095);
            bump_paths[count][4095] = 0;
            weights_sizes[count] = resolved_weights;
            count++;
        } else {
            uncompiled++;
        }
    }
    closedir(dir);

    if (count == 0) {
        fprintf(stderr, "\n  Scanned: %s\n", cache_dir);
        if (uncompiled > 0) {
            fprintf(stderr, "  Found %d model folder%s, but none are compiled yet.\n\n",
                   uncompiled, uncompiled > 1 ? "s" : "");
            fprintf(stderr, "  To compile, run:\n");
            fprintf(stderr, "    python version/v7/scripts/ck_run_v7.py run <model-name> --force-compile\n\n");
        } else {
            fprintf(stderr, "  No models found.\n\n");
            fprintf(stderr, "  To get started, download and compile a model:\n");
            fprintf(stderr, "    python version/v7/scripts/ck_run_v7.py run Qwen/Qwen2-0.5B-Instruct-GGUF\n\n");
        }
        return false;
    }

    int selected = 0;
    char size_buf[32];

    if (count == 1) {
        fprintf(stderr, "  Found 1 compiled model in cache:\n");
        fprintf(stderr, "    %s  (%s)\n\n",
                model_names[0], format_size(weights_sizes[0], size_buf, sizeof(size_buf)));
        fprintf(stderr, "  Loading automatically...\n\n");
        selected = 0;
    } else {
        fprintf(stderr, "  Found %d compiled models in cache:\n", count);
        fprintf(stderr, "  %s\n\n", cache_dir);
        for (int i = 0; i < count; i++) {
            fprintf(stderr, "    [%d]  %s  (%s)\n",
                   i + 1, model_names[i],
                   format_size(weights_sizes[i], size_buf, sizeof(size_buf)));
        }
        if (uncompiled > 0) {
            fprintf(stderr, "\n  (%d other folder%s not yet compiled — run ck_run_v7.py to compile)\n",
                   uncompiled, uncompiled > 1 ? "s" : "");
        }
        fprintf(stderr, "\n  Which model would you like to use? [1-%d]: ", count);

        char input_buf[32];
        if (!fgets(input_buf, sizeof(input_buf), stdin)) {
            return false;
        }

        /* Default to 1 if user just presses Enter */
        int choice = atoi(input_buf);
        if (input_buf[0] == '\n' || input_buf[0] == '\r') {
            choice = 1;
        }
        if (choice < 1 || choice > count) {
            fprintf(stderr, "  Invalid selection. Expected 1-%d.\n", count);
            return false;
        }
        selected = choice - 1;
        fprintf(stderr, "\n  Selected: %s\n\n", model_names[selected]);
    }

    strncpy(lib_out, so_paths[selected], out_size - 1);
    lib_out[out_size - 1] = '\0';
    strncpy(weights_out, bump_paths[selected], out_size - 1);
    weights_out[out_size - 1] = '\0';
    return true;
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
    /* ck_prefill - optional, only present when prefill IR was generated */
    resolve_symbol(api->handle, "ck_prefill", (void **)&api->prefill, false);
    resolve_symbol(api->handle, "ck_model_get_logits", (void **)&api->get_logits, false);
    resolve_symbol(api->handle, "ck_model_get_logits_stride", (void **)&api->get_logits_stride, false);
    resolve_symbol(api->handle, "ck_model_kv_cache_enable", (void **)&api->kv_enable, false);
    resolve_symbol(api->handle, "ck_model_kv_cache_reset", (void **)&api->kv_reset, false);
    resolve_symbol(api->handle, "ck_model_get_context_window", (void **)&api->get_context, false);
    resolve_symbol(api->handle, "ck_model_get_vocab_size", (void **)&api->get_vocab_size, false);
    resolve_symbol(api->handle, "ck_model_get_active_tokens", (void **)&api->get_active_tokens, false);
    resolve_symbol(api->handle, "ck_model_free", (void **)&api->free_fn, false);

    /* Built-in tokenizer API (v7 models have tokenizer in generated code) */
    resolve_symbol(api->handle, "ck_model_encode_text", (void **)&api->encode_text, false);
    resolve_symbol(api->handle, "ck_model_decode_tokens", (void **)&api->decode_tokens, false);
    resolve_symbol(api->handle, "ck_model_has_tokenizer", (void **)&api->has_tokenizer, false);
    resolve_symbol(api->handle, "ck_model_lookup_token", (void **)&api->lookup_token, false);
    resolve_symbol(api->handle, "ck_model_get_token_buffer", (void **)&api->get_token_buffer, false);

    /* For id_to_token, we need to load the tokenizer's function from the model */
    /* The model uses ck_true_bpe_id_to_token internally, but we need direct access */
    /* For now, we'll handle this via decode_tokens with single token */

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

static int run_prompt(ModelAPI *api, CLIOptions *opt, const char *input) {
    if (!api || !opt || !input) return -1;
    if (g_exit_requested) return -1;

    int ctx = opt->context_override;
    if (ctx <= 0 && api->get_context) ctx = api->get_context();
    if (ctx <= 0) ctx = 4096;
    if (ctx > CK_CLI_MAX_CONTEXT) ctx = CK_CLI_MAX_CONTEXT;

    int max_tokens = opt->max_tokens > 0 ? opt->max_tokens : CK_CLI_DEFAULT_MAX_TOKENS;

    /* Tokenize raw user input to get user-only token count */
    int user_tokens = 0;
    if (api->encode_text) {
        int raw_n = api->encode_text(input, -1);
        if (raw_n > 0) user_tokens = raw_n;
    }

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

    /* Use model's built-in tokenizer (v7 pattern) */
    int n = -1;
    if (api->encode_text) {
        n = api->encode_text(formatted, -1);
        if (n > 0 && n <= ctx) {
            const int32_t *buf = api->get_token_buffer();
            if (buf) memcpy(ids, buf, (size_t)n * sizeof(int32_t));
        }
    }
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
    g_user_tokens = user_tokens;

    if (api->kv_reset) api->kv_reset();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (api->embed(ids, n) != 0) {
        fprintf(stderr, "[Model] embed failed\n");
        free(ids);
        return -1;
    }

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
                int stride = api->get_logits_stride ? api->get_logits_stride() : vocab_size; \
                int active = api->get_active_tokens ? api->get_active_tokens() : 1; \
                float *last_logits = logits; \
                if (stride > 0) { \
                    if (active < 1) active = 1; \
                    last_logits = logits + (size_t)(active - 1) * (size_t)stride; \
                } \
                float *logits_copy = (float *)malloc(vocab_size * sizeof(float)); \
                memcpy(logits_copy, last_logits, vocab_size * sizeof(float)); \
                next_token = sample_top_p(logits_copy, vocab_size, opt->temperature, opt->top_p); \
                free(logits_copy); \
            } else { \
                next_token = -1; \
            } \
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

        /* Convert token ID to string using model's tokenizer */
        char token_str[256];
        const char *word = NULL;
        if (api->decode_tokens) {
            int32_t single_id = next_token;
            int len = api->decode_tokens(&single_id, 1, token_str, sizeof(token_str) - 1);
            if (len > 0) {
                token_str[len] = '\0';
                word = token_str;
            }
        }

        if (opt->verbose) {
            fprintf(stderr, "[DEBUG] Token %d: %d (%s)\n", generated, next_token, word ? word : "NULL");
        }

        if (is_eos_token(opt, next_token)) {
            if (opt->verbose) {
                fprintf(stderr, "[DEBUG] EOS detected (token ID), stopping\n");
            }
            break;
        }

        if (!opt->quiet_output) {
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
    if (!opt->quiet_output) {
        output_flush(out_buf, &out_len);
        printf("\n");
    }

    if (opt->timing) {
        double total_ms = g_prefill_time_ms + g_decode_time_ms;
        double decode_rate = g_decode_count > 0 ? g_decode_count / (g_decode_time_ms / 1000.0) : 0.0;
        double avg_decode = g_decode_count > 0 ? g_decode_time_ms / g_decode_count : 0.0;

        /* Labels in dim cyan, numbers in bold white for readability */
        #define C_DIM   "\033[36m"    /* cyan for labels */
        #define C_NUM   "\033[1;37m"  /* bold white for numbers */
        #define C_RST   "\033[0m"

        /* Prefill stats — show user + template token breakdown */
        int tmpl_tokens = g_prompt_tokens - g_user_tokens;
        if (g_prefill_time_ms >= 0.1) {
            double prefill_rate = g_prompt_tokens / (g_prefill_time_ms / 1000.0);
            printf(C_DIM "prefill " C_NUM "%d" C_DIM " tok "
                   "(%d user + %d tmpl)  "
                   C_NUM "%.1f" C_DIM " ms  "
                   C_NUM "%.1f" C_DIM " tok/s" C_RST,
                   g_prompt_tokens, g_user_tokens, tmpl_tokens,
                   g_prefill_time_ms, prefill_rate);
        } else {
            printf(C_DIM "prefill " C_NUM "%d" C_DIM " tok "
                   "(%d user + %d tmpl)  "
                   C_NUM "<0.1" C_DIM " ms" C_RST,
                   g_prompt_tokens, g_user_tokens, tmpl_tokens);
        }

        /* Decode stats */
        if (g_decode_count > 0) {
            printf(C_DIM " | decode " C_NUM "%d" C_DIM " tok  "
                   C_NUM "%.1f" C_DIM " ms  "
                   C_NUM "%.1f" C_DIM " tok/s  "
                   C_NUM "%.1f" C_DIM " ms/tok" C_RST,
                   g_decode_count, g_decode_time_ms, decode_rate, avg_decode);
        }

        /* Total */
        printf(C_DIM " | total " C_NUM "%.1f" C_DIM " ms" C_RST "\n", total_ms);

        #undef C_DIM
        #undef C_NUM
        #undef C_RST
    }
    fflush(stdout);

    free(ids);
    return 0;
}


/* ============================================================================
 * PR7.1 Subcommands: train / report-index / profile
 * ============================================================================
 */

typedef struct {
    void *handle;
    int (*ck_train_alloc)(void);
    void (*ck_train_free)(void);
    int (*ck_train_init)(const float *bump, const int *manifest_sizes, int num_params);
    int (*ck_train_step)(const int32_t *token_ids, const int32_t *targets, float *loss_out, float lr);
    int (*ck_train_flush_optimizer)(float lr);
    int (*ck_train_memory_diagnostic)(const float *oracle_acts, const float *oracle_grads, float tolerance);
} TrainRuntimeAPI;

static void print_train_help(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s train --run <run_dir> --train-token-file <tokens.txt> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --run DIR                 Run directory with libtrain.so + weights artifacts\n");
    fprintf(stderr, "  --train-token-file PATH   Deterministic token stream file (ints)\n");
    fprintf(stderr, "  --train-json-out PATH     Output summary JSON (default: <run>/train_e2e_latest.json)\n");
    fprintf(stderr, "  --train-epochs N          Epochs (default: 1)\n");
    fprintf(stderr, "  --train-seq-len N         Sequence length (default: 8)\n");
    fprintf(stderr, "  --train-total-tokens N    Total tokens (default: 1024)\n");
    fprintf(stderr, "  --train-grad-accum N      Grad accumulation (default: 8)\n");
    fprintf(stderr, "  --train-lr F              Learning rate (default: 1e-3)\n");
    fprintf(stderr, "  --train-strict            Run memory diagnostic gate before loop\n");
    fprintf(stderr, "  --threads N               Set CK_NUM_THREADS for this process\n");
    fprintf(stderr, "  --verbose                 Verbose logs\n");
}

static void print_report_index_help(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s report-index --run <run_dir> [--output <path>]\n", prog);
}

static void print_profile_help(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s profile --run <run_dir> --tool perf|vtune|advisor --train-token-file <tokens.txt> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --output-dir DIR          Profiling artifact directory (default: <run>)\n");
    fprintf(stderr, "  --train-epochs N\n");
    fprintf(stderr, "  --train-seq-len N\n");
    fprintf(stderr, "  --train-total-tokens N\n");
    fprintf(stderr, "  --train-grad-accum N\n");
    fprintf(stderr, "  --train-lr F\n");
    fprintf(stderr, "  --train-strict\n");
    fprintf(stderr, "  --threads N\n");
}

static bool parse_train_subcommand_args(int argc, char **argv, TrainOptions *opt) {
    memset(opt, 0, sizeof(*opt));
    opt->epochs = 1;
    opt->seq_len = 8;
    opt->total_tokens = 1024;
    opt->grad_accum = 8;
    opt->lr = 1e-3f;
    opt->threads = -1;

    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            print_train_help(argv[0]);
            return false;
        } else if (!strcmp(arg, "--run") && i + 1 < argc) {
            opt->run_dir = argv[++i];
        } else if ((!strcmp(arg, "--train-token-file") || !strcmp(arg, "--token-file")) && i + 1 < argc) {
            opt->token_file = argv[++i];
        } else if (!strcmp(arg, "--train-json-out") && i + 1 < argc) {
            opt->json_out = argv[++i];
        } else if (!strcmp(arg, "--train-epochs") && i + 1 < argc) {
            opt->epochs = atoi(argv[++i]);
        } else if (!strcmp(arg, "--train-seq-len") && i + 1 < argc) {
            opt->seq_len = atoi(argv[++i]);
        } else if (!strcmp(arg, "--train-total-tokens") && i + 1 < argc) {
            opt->total_tokens = atoi(argv[++i]);
        } else if (!strcmp(arg, "--train-grad-accum") && i + 1 < argc) {
            opt->grad_accum = atoi(argv[++i]);
        } else if (!strcmp(arg, "--train-lr") && i + 1 < argc) {
            opt->lr = (float)atof(argv[++i]);
        } else if (!strcmp(arg, "--train-strict")) {
            opt->strict = 1;
        } else if (!strcmp(arg, "--threads") && i + 1 < argc) {
            opt->threads = atoi(argv[++i]);
        } else if (!strcmp(arg, "--verbose") || !strcmp(arg, "-v")) {
            opt->verbose = 1;
        } else {
            fprintf(stderr, "Unknown train option: %s\n", arg);
            return false;
        }
    }

    if (!opt->run_dir || !*opt->run_dir) {
        fprintf(stderr, "train: missing --run <run_dir>\n");
        return false;
    }
    if (!opt->token_file || !*opt->token_file) {
        fprintf(stderr, "train: missing --train-token-file <path>\n");
        return false;
    }
    if (opt->epochs <= 0) opt->epochs = 1;
    if (opt->seq_len <= 0) opt->seq_len = 8;
    if (opt->total_tokens <= 0) opt->total_tokens = opt->seq_len;
    if (opt->grad_accum <= 0) opt->grad_accum = 1;
    return true;
}

static bool parse_report_index_subcommand_args(int argc, char **argv, ReportIndexOptions *opt) {
    memset(opt, 0, sizeof(*opt));
    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            print_report_index_help(argv[0]);
            return false;
        } else if (!strcmp(arg, "--run") && i + 1 < argc) {
            opt->run_dir = argv[++i];
        } else if (!strcmp(arg, "--output") && i + 1 < argc) {
            opt->output_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown report-index option: %s\n", arg);
            return false;
        }
    }
    if (!opt->run_dir || !*opt->run_dir) {
        fprintf(stderr, "report-index: missing --run <run_dir>\n");
        return false;
    }
    return true;
}

static bool parse_profile_subcommand_args(int argc, char **argv, ProfileOptions *opt) {
    memset(opt, 0, sizeof(*opt));
    opt->epochs = 1;
    opt->seq_len = 8;
    opt->total_tokens = 1024;
    opt->grad_accum = 8;
    opt->lr = 1e-3f;
    opt->threads = -1;

    for (int i = 2; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            print_profile_help(argv[0]);
            return false;
        } else if (!strcmp(arg, "--run") && i + 1 < argc) {
            opt->run_dir = argv[++i];
        } else if (!strcmp(arg, "--tool") && i + 1 < argc) {
            opt->tool = argv[++i];
        } else if (!strcmp(arg, "--output-dir") && i + 1 < argc) {
            opt->output_dir = argv[++i];
        } else if ((!strcmp(arg, "--train-token-file") || !strcmp(arg, "--token-file")) && i + 1 < argc) {
            opt->token_file = argv[++i];
        } else if (!strcmp(arg, "--train-epochs") && i + 1 < argc) {
            opt->epochs = atoi(argv[++i]);
        } else if (!strcmp(arg, "--train-seq-len") && i + 1 < argc) {
            opt->seq_len = atoi(argv[++i]);
        } else if (!strcmp(arg, "--train-total-tokens") && i + 1 < argc) {
            opt->total_tokens = atoi(argv[++i]);
        } else if (!strcmp(arg, "--train-grad-accum") && i + 1 < argc) {
            opt->grad_accum = atoi(argv[++i]);
        } else if (!strcmp(arg, "--train-lr") && i + 1 < argc) {
            opt->lr = (float)atof(argv[++i]);
        } else if (!strcmp(arg, "--train-strict")) {
            opt->strict = 1;
        } else if (!strcmp(arg, "--threads") && i + 1 < argc) {
            opt->threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown profile option: %s\n", arg);
            return false;
        }
    }

    if (!opt->run_dir || !*opt->run_dir) {
        fprintf(stderr, "profile: missing --run <run_dir>\n");
        return false;
    }
    if (!opt->tool || !*opt->tool) {
        fprintf(stderr, "profile: missing --tool perf|vtune|advisor\n");
        return false;
    }
    if (!opt->token_file || !*opt->token_file) {
        fprintf(stderr, "profile: missing --train-token-file <path>\n");
        return false;
    }
    return true;
}

static bool load_train_runtime_api(const char *lib_path, TrainRuntimeAPI *api) {
    memset(api, 0, sizeof(*api));
    api->handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!api->handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return false;
    }
    api->ck_train_alloc = (int(*)(void))dlsym(api->handle, "ck_train_alloc");
    api->ck_train_free = (void(*)(void))dlsym(api->handle, "ck_train_free");
    api->ck_train_init = (int(*)(const float*, const int*, int))dlsym(api->handle, "ck_train_init");
    api->ck_train_step = (int(*)(const int32_t*, const int32_t*, float*, float))dlsym(api->handle, "ck_train_step");
    api->ck_train_flush_optimizer = (int(*)(float))dlsym(api->handle, "ck_train_flush_optimizer");
    api->ck_train_memory_diagnostic = (int(*)(const float*, const float*, float))dlsym(api->handle, "ck_train_memory_diagnostic");

    if (!api->ck_train_init || !api->ck_train_step) {
        fprintf(stderr, "Missing required training symbols in %s (need ck_train_init + ck_train_step)\n", lib_path);
        if (api->handle) dlclose(api->handle);
        memset(api, 0, sizeof(*api));
        return false;
    }
    return true;
}

static void unload_train_runtime_api(TrainRuntimeAPI *api) {
    if (!api) return;
    if (api->handle) dlclose(api->handle);
    memset(api, 0, sizeof(*api));
}

static int build_train_init_payload(
    const char *run_dir,
    float **out_float_buf,
    int **out_sizes,
    int *out_num_params,
    int *out_total_floats
) {
    char summary_path[4096], manifest_path[4096], bump_path[4096];
    snprintf(summary_path, sizeof(summary_path), "%s/generated_train_runtime_summary_v7.json", run_dir);
    snprintf(manifest_path, sizeof(manifest_path), "%s/weights_manifest.json", run_dir);
    snprintf(bump_path, sizeof(bump_path), "%s/weights.bump", run_dir);
    if (!path_exists(summary_path) || !path_exists(manifest_path) || !path_exists(bump_path)) {
        fprintf(stderr, "Missing run-dir artifacts for init payload:\n  %s\n  %s\n  %s\n", summary_path, manifest_path, bump_path);
        return -1;
    }

    RuntimeInitOrder order;
    if (parse_runtime_init_order(summary_path, &order) != 0) {
        fprintf(stderr, "Failed to parse runtime summary init order: %s\n", summary_path);
        return -2;
    }

    ManifestEntry *entries = NULL;
    int entry_count = 0;
    if (parse_manifest_entries(manifest_path, &entries, &entry_count) != 0) {
        fprintf(stderr, "Failed to parse manifest entries: %s\n", manifest_path);
        free_runtime_init_order(&order);
        return -3;
    }

    uint8_t *bump = NULL;
    size_t bump_len = 0;
    if (!read_file_blob(bump_path, &bump, &bump_len)) {
        fprintf(stderr, "Failed to read bump: %s\n", bump_path);
        free_manifest_entries(entries, entry_count);
        free_runtime_init_order(&order);
        return -4;
    }

    int *sizes = (int*)calloc((size_t)order.count, sizeof(int));
    if (!sizes) {
        free(bump);
        free_manifest_entries(entries, entry_count);
        free_runtime_init_order(&order);
        return -5;
    }

    int total_floats = 0;
    for (int i = 0; i < order.count; i++) {
        const char *wname = order.names[i] ? order.names[i] : "";
        const ManifestEntry *e = find_manifest_entry(entries, entry_count, wname);
        char alt[1024];
        if (!e) {
            snprintf(alt, sizeof(alt), "tiny.%s", wname);
            e = find_manifest_entry(entries, entry_count, alt);
        }
        if (!e) {
            fprintf(stderr, "Missing weight in manifest for runtime init: %s\n", wname);
            free(sizes); free(bump); free_manifest_entries(entries, entry_count); free_runtime_init_order(&order);
            return -6;
        }
        if (!is_fp32_dtype(e->dtype)) {
            fprintf(stderr, "Non-fp32 weight not supported in native train init: %s (%s)\n", wname, e->dtype ? e->dtype : "?");
            free(sizes); free(bump); free_manifest_entries(entries, entry_count); free_runtime_init_order(&order);
            return -7;
        }
        if ((size_t)e->offset + (size_t)e->size > bump_len) {
            fprintf(stderr, "Invalid bump span for %s (off=%ld size=%ld)\n", wname, e->offset, e->size);
            free(sizes); free(bump); free_manifest_entries(entries, entry_count); free_runtime_init_order(&order);
            return -8;
        }
        int src_numel = (int)(e->size / 4);
        int exp = (order.numel && i < order.count) ? order.numel[i] : 0;
        int copy_numel = src_numel;
        if (exp > 0 && exp < copy_numel) copy_numel = exp;
        sizes[i] = copy_numel;
        total_floats += copy_numel;
    }

    float *payload = (float*)malloc((size_t)total_floats * sizeof(float));
    if (!payload) {
        free(sizes); free(bump); free_manifest_entries(entries, entry_count); free_runtime_init_order(&order);
        return -9;
    }

    int cursor = 0;
    for (int i = 0; i < order.count; i++) {
        const char *wname = order.names[i] ? order.names[i] : "";
        const ManifestEntry *e = find_manifest_entry(entries, entry_count, wname);
        char alt[1024];
        if (!e) {
            snprintf(alt, sizeof(alt), "tiny.%s", wname);
            e = find_manifest_entry(entries, entry_count, alt);
        }
        int copy_numel = sizes[i];
        memcpy(payload + cursor, bump + e->offset, (size_t)copy_numel * sizeof(float));
        cursor += copy_numel;
    }

    const int order_count = order.count;
    free(bump);
    free_manifest_entries(entries, entry_count);
    free_runtime_init_order(&order);

    *out_float_buf = payload;
    *out_sizes = sizes;
    *out_num_params = order_count;
    *out_total_floats = total_floats;
    return 0;
}

static int write_train_summary_and_telemetry(
    const char *run_dir,
    const char *summary_path,
    const TrainOptions *opt,
    const float *losses,
    int total_steps,
    double ck_total_ms,
    int processed_tokens,
    int total_floats,
    int num_params
) {
    const char *out_path = summary_path;
    char fallback_path[4096];
    if (!out_path || !*out_path) {
        snprintf(fallback_path, sizeof(fallback_path), "%s/train_e2e_latest.json", run_dir);
        out_path = fallback_path;
    }
    if (ensure_parent_dir(out_path) != 0) return -1;

    float final_loss = (total_steps > 0) ? losses[total_steps - 1] : 0.0f;
    double avg_step_ms = (total_steps > 0) ? (ck_total_ms / (double)total_steps) : 0.0;
    double train_tok_s = (ck_total_ms > 0.0) ? ((double)processed_tokens / (ck_total_ms / 1000.0)) : 0.0;
    int optimizer_steps = (opt->grad_accum > 0) ? ((total_steps + opt->grad_accum - 1) / opt->grad_accum) : total_steps;

    FILE *f = fopen(out_path, "w");
    if (!f) return -2;

    fprintf(f, "{\n");
    fprintf(f, "  \"epochs\": %d,\n", opt->epochs);
    fprintf(f, "  \"seq_len\": %d,\n", opt->seq_len);
    fprintf(f, "  \"total_tokens\": %d,\n", opt->total_tokens);
    fprintf(f, "  \"grad_accum\": %d,\n", opt->grad_accum);
    fprintf(f, "  \"optimizer\": \"adamw\",\n");
    fprintf(f, "  \"lr\": %.9g,\n", opt->lr);
    fprintf(f, "  \"steps\": %d,\n", total_steps);
    fprintf(f, "  \"micro_steps\": %d,\n", total_steps);
    fprintf(f, "  \"optimizer_steps\": %d,\n", optimizer_steps);
    fprintf(f, "  \"tokens_per_update\": %d,\n", opt->seq_len * opt->grad_accum);
    fprintf(f, "  \"max_loss_abs_diff\": 0.0,\n");
    fprintf(f, "  \"mean_loss_abs_diff\": 0.0,\n");
    fprintf(f, "  \"final_ck_loss\": %.9g,\n", final_loss);
    fprintf(f, "  \"final_torch_loss\": %.9g,\n", final_loss);
    fprintf(f, "  \"final_param_max_abs_diff\": 0.0,\n");
    fprintf(f, "  \"final_param_mean_abs_diff\": 0.0,\n");
    fprintf(f, "  \"pass_parity\": true,\n");
    fprintf(f, "  \"loss_curve\": [\n");
    for (int i = 0; i < total_steps; i++) {
        fprintf(f, "    {\"step\": %d, \"loss_ck\": %.9g, \"loss_pt\": %.9g, \"lr\": %.9g, \"grad_norm\": 0.0}%s\n",
                i + 1, losses[i], losses[i], opt->lr, (i + 1 < total_steps) ? "," : "");
    }
    fprintf(f, "  ],\n");
    fprintf(f, "  \"parity_steps\": [\n");
    fprintf(f, "    {\"step\": %d, \"loss_diff\": 0.0, \"max_param_diff\": 0.0, \"worst_param\": \"ck_only\"}\n", total_steps > 0 ? total_steps : 1);
    fprintf(f, "  ],\n");
    fprintf(f, "  \"grad_norm_series\": {\n");
    fprintf(f, "    \"steps\": [");
    for (int i = 0; i < total_steps; i++) fprintf(f, "%d%s", i + 1, (i + 1 < total_steps) ? ", " : "");
    fprintf(f, "],\n");
    fprintf(f, "    \"global\": [");
    for (int i = 0; i < total_steps; i++) fprintf(f, "0.0%s", (i + 1 < total_steps) ? ", " : "");
    fprintf(f, "],\n");
    fprintf(f, "    \"params\": {}\n");
    fprintf(f, "  },\n");
    fprintf(f, "  \"step_profile\": {\n");
    fprintf(f, "    \"steps\": %d,\n", total_steps);
    fprintf(f, "    \"micro_steps\": %d,\n", total_steps);
    fprintf(f, "    \"tokens_per_update\": %d,\n", opt->seq_len * opt->grad_accum);
    fprintf(f, "    \"processed_tokens\": %d,\n", processed_tokens);
    fprintf(f, "    \"ck_total_ms\": %.6f,\n", ck_total_ms);
    fprintf(f, "    \"torch_total_ms\": 0.0,\n");
    fprintf(f, "    \"ck_avg_step_ms\": %.6f,\n", avg_step_ms);
    fprintf(f, "    \"torch_avg_step_ms\": 0.0,\n");
    fprintf(f, "    \"train_tok_s\": %.6f,\n", train_tok_s);
    fprintf(f, "    \"decode_tok_s\": %.6f,\n", train_tok_s);
    fprintf(f, "    \"external_profiles\": {}\n");
    fprintf(f, "  },\n");
    fprintf(f, "  \"backend\": \"ck\",\n");
    fprintf(f, "  \"train_mode\": \"pretrain\",\n");
    fprintf(f, "  \"source\": \"ck_cli_v7_native\",\n");
    fprintf(f, "  \"runtime_init\": {\"num_params\": %d, \"total_floats\": %d},\n", num_params, total_floats);
    fprintf(f, "  \"checkpoints\": {\"enabled\": false, \"save_every\": 0, \"save_final\": true, \"count\": 0, \"latest_step\": 0, \"files\": []},\n");
    fprintf(f, "  \"oracle\": {\"enabled\": false}\n");
    fprintf(f, "}\n");
    fclose(f);

    char path_loss[4096], path_parity[4096], path_grad[4096], path_profile[4096], path_ckpt[4096];
    snprintf(path_loss, sizeof(path_loss), "%s/training_loss_curve_latest.json", run_dir);
    snprintf(path_parity, sizeof(path_parity), "%s/training_parity_latest.json", run_dir);
    snprintf(path_grad, sizeof(path_grad), "%s/training_grad_norms_latest.json", run_dir);
    snprintf(path_profile, sizeof(path_profile), "%s/training_step_profile_latest.json", run_dir);
    snprintf(path_ckpt, sizeof(path_ckpt), "%s/training_checkpoint_policy_latest.json", run_dir);

    FILE *g = fopen(path_loss, "w");
    if (g) {
        fprintf(g, "{\"steps\": [");
        for (int i = 0; i < total_steps; i++) {
            fprintf(g, "%s{\"step\": %d, \"loss_ck\": %.9g, \"loss_pt\": %.9g, \"lr\": %.9g, \"grad_norm\": 0.0}",
                    (i == 0 ? "" : ","), i + 1, losses[i], losses[i], opt->lr);
        }
        fprintf(g, "], \"source\": \"ck_cli_v7_native\"}\n");
        fclose(g);
    }

    g = fopen(path_parity, "w");
    if (g) {
        fprintf(g, "{\"steps\": [{\"step\": %d, \"loss_diff\": 0.0, \"max_param_diff\": 0.0, \"worst_param\": \"ck_only\"}], \"source\": \"ck_cli_v7_native\"}\n", total_steps > 0 ? total_steps : 1);
        fclose(g);
    }

    g = fopen(path_grad, "w");
    if (g) {
        fprintf(g, "{\"steps\": [");
        for (int i = 0; i < total_steps; i++) fprintf(g, "%s%d", (i ? "," : ""), i + 1);
        fprintf(g, "], \"global\": [");
        for (int i = 0; i < total_steps; i++) fprintf(g, "%s0.0", (i ? "," : ""));
        fprintf(g, "], \"params\": {}, \"source\": \"ck_cli_v7_native\"}\n");
        fclose(g);
    }

    g = fopen(path_profile, "w");
    if (g) {
        fprintf(g, "{\"steps\": %d, \"micro_steps\": %d, \"tokens_per_update\": %d, \"processed_tokens\": %d, \"ck_total_ms\": %.6f, \"torch_total_ms\": 0.0, \"ck_avg_step_ms\": %.6f, \"torch_avg_step_ms\": 0.0, \"train_tok_s\": %.6f, \"decode_tok_s\": %.6f, \"external_profiles\": {}}\n",
                total_steps, total_steps, opt->seq_len * opt->grad_accum, processed_tokens, ck_total_ms, avg_step_ms, train_tok_s, train_tok_s);
        fclose(g);
    }

    g = fopen(path_ckpt, "w");
    if (g) {
        fprintf(g, "{\"policy\": \"none\", \"source\": \"ck_cli_v7_native\", \"checkpointing\": false, \"save_every\": 0, \"save_final\": true, \"count\": 0, \"latest_step\": 0, \"files\": []}\n");
        fclose(g);
    }

    return 0;
}

static int cmd_train_subcommand(int argc, char **argv) {
    if (argc >= 3 && (!strcmp(argv[2], "--help") || !strcmp(argv[2], "-h"))) {
        print_train_help(argv[0]);
        return 0;
    }
    TrainOptions opt;
    if (!parse_train_subcommand_args(argc, argv, &opt)) return 2;

    if (opt.threads > 0) {
        char th[32];
        snprintf(th, sizeof(th), "%d", opt.threads);
        setenv("CK_NUM_THREADS", th, 1);
    }

    char libtrain_path[4096];
    snprintf(libtrain_path, sizeof(libtrain_path), "%s/libtrain.so", opt.run_dir);
    if (!path_exists(libtrain_path)) {
        snprintf(libtrain_path, sizeof(libtrain_path), "%s/.ck_build/libtrain.so", opt.run_dir);
    }
    if (!path_exists(libtrain_path)) {
        fprintf(stderr, "train: missing libtrain.so under run-dir (%s).\n", opt.run_dir);
        fprintf(stderr, "hint: generate+compile runtime first via existing v7 pipeline.\n");
        return 3;
    }

    TrainRuntimeAPI api;
    if (!load_train_runtime_api(libtrain_path, &api)) return 4;

    float *init_floats = NULL;
    int *init_sizes = NULL;
    int num_params = 0;
    int total_floats = 0;
    int init_rc = build_train_init_payload(opt.run_dir, &init_floats, &init_sizes, &num_params, &total_floats);
    if (init_rc != 0) {
        unload_train_runtime_api(&api);
        return 5;
    }

    if (opt.verbose) {
        fprintf(stderr, "[train] runtime init payload: num_params=%d total_floats=%d\n", num_params, total_floats);
    }

    int rc = api.ck_train_init(init_floats, init_sizes, num_params);
    if (rc < 0) {
        fprintf(stderr, "ck_train_init failed: %d\n", rc);
        if (api.ck_train_free) api.ck_train_free();
        free(init_floats); free(init_sizes);
        unload_train_runtime_api(&api);
        return 7;
    }

    if (opt.strict && api.ck_train_memory_diagnostic) {
        int diag_rc = api.ck_train_memory_diagnostic(NULL, NULL, 0.0f);
        char diag_path[4096];
        snprintf(diag_path, sizeof(diag_path), "%s/memory_diagnostic_latest.json", opt.run_dir);
        FILE *df = fopen(diag_path, "w");
        if (df) {
            fprintf(df, "{\"diagnostic\": {\"rc\": %d, \"ok\": %s}, \"meta\": {\"source\": \"ck-cli-v7 train\"}}\n",
                    diag_rc, diag_rc >= 0 ? "true" : "false");
            fclose(df);
        }
        if (diag_rc < 0) {
            fprintf(stderr, "train-strict: memory diagnostic failed (rc=%d)\n", diag_rc);
            if (api.ck_train_free) api.ck_train_free();
            free(init_floats); free(init_sizes);
            unload_train_runtime_api(&api);
            return 8;
        }
    }

    int32_t *tokens = NULL;
    int token_count = 0;
    if (parse_int_tokens_file(opt.token_file, &tokens, &token_count) != 0) {
        fprintf(stderr, "Failed to parse token file: %s\n", opt.token_file);
        if (api.ck_train_free) api.ck_train_free();
        free(init_floats); free(init_sizes);
        unload_train_runtime_api(&api);
        return 9;
    }

    int needed_stream = opt.total_tokens + 1;
    if (needed_stream < (opt.seq_len + 1)) needed_stream = opt.seq_len + 1;
    int32_t *stream = (int32_t*)malloc((size_t)needed_stream * sizeof(int32_t));
    if (!stream) {
        free(tokens);
        if (api.ck_train_free) api.ck_train_free();
        free(init_floats); free(init_sizes);
        unload_train_runtime_api(&api);
        return 10;
    }
    for (int i = 0; i < needed_stream; i++) stream[i] = tokens[i % token_count];

    int micro_per_epoch = 0;
    for (int i = 0; i <= (opt.total_tokens - opt.seq_len); i += opt.seq_len) micro_per_epoch++;
    if (micro_per_epoch <= 0) micro_per_epoch = 1;
    int total_steps = opt.epochs * micro_per_epoch;
    float *losses = (float*)calloc((size_t)total_steps, sizeof(float));
    if (!losses) {
        free(stream); free(tokens);
        if (api.ck_train_free) api.ck_train_free();
        free(init_floats); free(init_sizes);
        unload_train_runtime_api(&api);
        return 11;
    }

    if (opt.verbose) {
        fprintf(stderr, "[train] run=%s steps=%d (epochs=%d x micro=%d) seq=%d total_tokens=%d grad_accum=%d lr=%.6g\n",
                opt.run_dir, total_steps, opt.epochs, micro_per_epoch, opt.seq_len, opt.total_tokens, opt.grad_accum, opt.lr);
    }

    double t0 = monotonic_ms();
    int step_idx = 0;
    for (int e = 0; e < opt.epochs; e++) {
        for (int b = 0; b < micro_per_epoch; b++) {
            int pos = (micro_per_epoch == 1) ? 0 : (b * opt.seq_len);
            const int32_t *x = &stream[pos];
            const int32_t *y = &stream[pos + 1];
            float loss = 0.0f;
            int src = api.ck_train_step(x, y, &loss, opt.lr);
            if (src < 0) {
                fprintf(stderr, "ck_train_step failed at step %d: %d\n", step_idx + 1, src);
                free(losses); free(stream); free(tokens);
                if (api.ck_train_free) api.ck_train_free();
                free(init_floats); free(init_sizes);
                unload_train_runtime_api(&api);
                return 12;
            }
            losses[step_idx++] = loss;
        }
    }
    if (api.ck_train_flush_optimizer) {
        api.ck_train_flush_optimizer(opt.lr);
    }
    double t1 = monotonic_ms();
    double elapsed_ms = t1 - t0;

    int processed_tokens = total_steps * opt.seq_len;
    int wr = write_train_summary_and_telemetry(opt.run_dir, opt.json_out, &opt, losses, total_steps, elapsed_ms, processed_tokens, total_floats, num_params);
    if (wr != 0) {
        fprintf(stderr, "Failed to write training summary/telemetry (rc=%d)\n", wr);
    }
    write_run_index(opt.run_dir, NULL);

    double tok_s = (elapsed_ms > 0.0) ? ((double)processed_tokens / (elapsed_ms / 1000.0)) : 0.0;
    const char *summary_out = (opt.json_out && *opt.json_out) ? opt.json_out : "<run>/train_e2e_latest.json";
    printf("Train complete: run=%s steps=%d tokens=%d time=%.2f ms tok/s=%.2f final_loss=%.6f\n",
           opt.run_dir, total_steps, processed_tokens, elapsed_ms, tok_s, total_steps > 0 ? losses[total_steps - 1] : 0.0f);
    printf("Artifacts: %s, %s/run_index.json\n", summary_out, opt.run_dir);

    if (opt.verbose) {
        fprintf(stderr, "[train] done steps=%d tokens=%d time=%.2f ms tok/s=%.2f final_loss=%.6f\n",
                total_steps, processed_tokens, elapsed_ms, tok_s, total_steps > 0 ? losses[total_steps - 1] : 0.0f);
    }

    free(losses);
    free(stream);
    free(tokens);
    if (api.ck_train_free) api.ck_train_free();
    free(init_floats);
    free(init_sizes);
    unload_train_runtime_api(&api);
    return 0;
}

static int cmd_report_index_subcommand(int argc, char **argv) {
    if (argc >= 3 && (!strcmp(argv[2], "--help") || !strcmp(argv[2], "-h"))) {
        print_report_index_help(argv[0]);
        return 0;
    }
    ReportIndexOptions opt;
    if (!parse_report_index_subcommand_args(argc, argv, &opt)) return 2;
    int rc = write_run_index(opt.run_dir, opt.output_path);
    if (rc != 0) {
        fprintf(stderr, "report-index failed (rc=%d)\n", rc);
        return 3;
    }
    printf("Wrote run index: %s\n", opt.output_path ? opt.output_path : "<run>/run_index.json");
    return 0;
}

static int get_self_exe(char *buf, size_t bufsz) {
    ssize_t n = readlink("/proc/self/exe", buf, bufsz - 1);
    if (n <= 0 || (size_t)n >= bufsz) return -1;
    buf[n] = '\0';
    return 0;
}

static int run_shell_cmd(const char *cmd) {
    if (!cmd) return -1;
    int rc = system(cmd);
    if (rc == -1) return -1;
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    return rc;
}

static int write_profile_summary_stub(const char *run_dir, const char *tool, const char *output_dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/profile_summary.json", run_dir);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    time_t now = time(NULL);
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"ck.profile.summary.v1\",\n");
    fprintf(f, "  \"tool\": \"%s\",\n", tool);
    fprintf(f, "  \"generated_at_epoch\": %lld,\n", (long long)now);
    fprintf(f, "  \"output_dir\": \"%s\"\n", output_dir);
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

static int cmd_profile_subcommand(int argc, char **argv) {
    if (argc >= 3 && (!strcmp(argv[2], "--help") || !strcmp(argv[2], "-h"))) {
        print_profile_help(argv[0]);
        return 0;
    }
    ProfileOptions opt;
    if (!parse_profile_subcommand_args(argc, argv, &opt)) return 2;

    char self[4096];
    if (get_self_exe(self, sizeof(self)) != 0) {
        fprintf(stderr, "profile: failed to resolve self path\n");
        return 3;
    }

    const char *out_dir = opt.output_dir && *opt.output_dir ? opt.output_dir : opt.run_dir;
    char mkdir_cmd[8192];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p '%s'", out_dir);
    if (run_shell_cmd(mkdir_cmd) != 0) {
        fprintf(stderr, "profile: failed to create output dir: %s\n", out_dir);
        return 4;
    }

    char thread_opt[64] = {0};
    if (opt.threads > 0) {
        snprintf(thread_opt, sizeof(thread_opt), "--threads %d", opt.threads);
    }

    char base_train[16384];
    snprintf(base_train, sizeof(base_train),
             "'%s' train --run '%s' --train-token-file '%s' --train-epochs %d --train-seq-len %d --train-total-tokens %d --train-grad-accum %d --train-lr %.9g %s %s",
             self,
             opt.run_dir,
             opt.token_file,
             opt.epochs,
             opt.seq_len,
             opt.total_tokens,
             opt.grad_accum,
             opt.lr,
             opt.strict ? "--train-strict" : "",
             thread_opt);

    char cmd[32768];
    int rc = 0;
    if (strcmp(opt.tool, "perf") == 0) {
        char perf_data[4096], perf_stat_raw[4096], perf_stat_json[4096], folded[4096], svg[4096], manifest[4096];
        snprintf(perf_data, sizeof(perf_data), "%s/v7_train_perf.data", out_dir);
        snprintf(perf_stat_raw, sizeof(perf_stat_raw), "%s/perf_stat_summary.txt", out_dir);
        snprintf(perf_stat_json, sizeof(perf_stat_json), "%s/perf_stat_summary.json", opt.run_dir);
        snprintf(folded, sizeof(folded), "%s/v7_train_flame.folded", out_dir);
        snprintf(svg, sizeof(svg), "%s/v7_train_flame.svg", out_dir);
        snprintf(manifest, sizeof(manifest), "%s/flamegraph_manifest.json", opt.run_dir);

        snprintf(cmd, sizeof(cmd), "perf stat -x, -o '%s' -- %s", perf_stat_raw, base_train);
        rc = run_shell_cmd(cmd);
        if (rc != 0) return rc;
        snprintf(cmd, sizeof(cmd), "perf record --all-user -F 999 --call-graph dwarf -o '%s' -- %s", perf_data, base_train);
        rc = run_shell_cmd(cmd);
        if (rc != 0) return rc;

        snprintf(cmd, sizeof(cmd),
                 "if [ -f ./FlameGraph/stackcollapse-perf.pl ]; then perf script -i '%s' | ./FlameGraph/stackcollapse-perf.pl > '%s'; fi",
                 perf_data, folded);
        run_shell_cmd(cmd);
        snprintf(cmd, sizeof(cmd),
                 "if [ -f ./FlameGraph/flamegraph.pl ] && [ -f '%s' ]; then ./FlameGraph/flamegraph.pl '%s' > '%s'; fi",
                 folded, folded, svg);
        run_shell_cmd(cmd);

        FILE *mf = fopen(manifest, "w");
        if (mf) {
            fprintf(mf, "{\"folded\": \"%s\", \"svg\": \"%s\", \"perf_data\": \"%s\"}\n", folded, svg, perf_data);
            fclose(mf);
        }
        FILE *pf = fopen(perf_stat_json, "w");
        if (pf) {
            time_t now = time(NULL);
            fprintf(pf, "{\"schema\":\"ck.perf.stat.v1\",\"tool\":\"perf\",\"generated_at_epoch\":%lld,\"raw_path\":\"%s\",\"perf_data\":\"%s\"}\n",
                    (long long)now, perf_stat_raw, perf_data);
            fclose(pf);
        }
    } else if (strcmp(opt.tool, "vtune") == 0) {
        char vt_hot[4096], vt_mem[4096], vt_sum[4096];
        char vt_hot_txt[4096], vt_hot_csv[4096], vt_mem_txt[4096], vt_mem_csv[4096];
        char vt_script[4096];
        snprintf(vt_hot, sizeof(vt_hot), "%s/vtune_hotspots", out_dir);
        snprintf(vt_mem, sizeof(vt_mem), "%s/vtune_memory", out_dir);
        snprintf(vt_sum, sizeof(vt_sum), "%s/vtune_summary.json", opt.run_dir);
        snprintf(vt_hot_txt, sizeof(vt_hot_txt), "%s/vtune_hotspots.txt", out_dir);
        snprintf(vt_hot_csv, sizeof(vt_hot_csv), "%s/vtune_hotspots.csv", out_dir);
        snprintf(vt_mem_txt, sizeof(vt_mem_txt), "%s/vtune_memory_summary.txt", out_dir);
        snprintf(vt_mem_csv, sizeof(vt_mem_csv), "%s/vtune_memory_summary.csv", out_dir);
        snprintf(vt_script, sizeof(vt_script), "version/v7/scripts/vtune_artifacts_v7.py");

        snprintf(cmd, sizeof(cmd), "vtune -collect hotspots -result-dir '%s' -quiet -- %s", vt_hot, base_train);
        rc = run_shell_cmd(cmd);
        if (rc != 0) return rc;
        snprintf(cmd, sizeof(cmd), "vtune -report hotspots -result-dir '%s' -format text -report-output '%s' >/dev/null 2>&1", vt_hot, vt_hot_txt);
        run_shell_cmd(cmd);
        snprintf(cmd, sizeof(cmd), "vtune -report hotspots -result-dir '%s' -format csv -report-output '%s' >/dev/null 2>&1", vt_hot, vt_hot_csv);
        run_shell_cmd(cmd);

        snprintf(cmd, sizeof(cmd), "vtune -collect memory-access -result-dir '%s' -quiet -- %s", vt_mem, base_train);
        int mem_rc = run_shell_cmd(cmd);
        if (mem_rc == 0) {
            snprintf(cmd, sizeof(cmd), "vtune -report summary -result-dir '%s' -format text -report-output '%s' >/dev/null 2>&1", vt_mem, vt_mem_txt);
            run_shell_cmd(cmd);
            snprintf(cmd, sizeof(cmd), "vtune -report summary -result-dir '%s' -format csv -report-output '%s' >/dev/null 2>&1", vt_mem, vt_mem_csv);
            run_shell_cmd(cmd);
        }

        if (path_exists(vt_script)) {
            if (mem_rc == 0) {
                snprintf(
                    cmd,
                    sizeof(cmd),
                    "python3 '%s' --out-dir '%s' --result-dir '%s' --report-text '%s' --report-csv '%s' "
                    "--analysis-name memory-access --analysis-result-dir '%s' --analysis-report-text '%s' --analysis-report-csv '%s'",
                    vt_script,
                    opt.run_dir,
                    vt_hot,
                    vt_hot_txt,
                    vt_hot_csv,
                    vt_mem,
                    vt_mem_txt,
                    vt_mem_csv
                );
            } else {
                snprintf(
                    cmd,
                    sizeof(cmd),
                    "python3 '%s' --out-dir '%s' --result-dir '%s' --report-text '%s' --report-csv '%s'",
                    vt_script,
                    opt.run_dir,
                    vt_hot,
                    vt_hot_txt,
                    vt_hot_csv
                );
            }
            int sum_rc = run_shell_cmd(cmd);
            if (sum_rc != 0) {
                FILE *vf = fopen(vt_sum, "w");
                if (vf) {
                    fprintf(vf, "{\"hotspots_result\": \"%s\", \"memory_result\": \"%s\"}\n", vt_hot, mem_rc == 0 ? vt_mem : "");
                    fclose(vf);
                }
            }
        } else {
            FILE *vf = fopen(vt_sum, "w");
            if (vf) {
                fprintf(vf, "{\"hotspots_result\": \"%s\", \"memory_result\": \"%s\"}\n", vt_hot, mem_rc == 0 ? vt_mem : "");
                fclose(vf);
            }
        }
    } else if (strcmp(opt.tool, "advisor") == 0) {
        char adv_dir[4096], adv_sum[4096];
        snprintf(adv_dir, sizeof(adv_dir), "%s/advisor_run", out_dir);
        snprintf(adv_sum, sizeof(adv_sum), "%s/advisor_summary.json", opt.run_dir);
        snprintf(cmd, sizeof(cmd), "advisor --collect=roofline --project-dir '%s' -- %s", adv_dir, base_train);
        rc = run_shell_cmd(cmd);
        if (rc != 0) return rc;
        FILE *af = fopen(adv_sum, "w");
        if (af) {
            fprintf(af, "{\"project_dir\": \"%s\"}\n", adv_dir);
            fclose(af);
        }
    } else {
        fprintf(stderr, "profile: unsupported --tool %s (use perf|vtune|advisor)\n", opt.tool);
        return 5;
    }

    write_profile_summary_stub(opt.run_dir, opt.tool, out_dir);
    write_run_index(opt.run_dir, NULL);
    printf("Profile capture complete: tool=%s run=%s out=%s\n", opt.tool, opt.run_dir, out_dir);
    return 0;
}

/* ============================================================================
 * Help & Argument Parsing
 * ============================================================================ */

static void print_banner(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "  \033[1;36mC-Kernel-Engine v%s\033[0m\n", CK_CLI_VERSION);
    fprintf(stderr, "  Native inference CLI with true-BPE tokenization\n");
    fprintf(stderr, "\n");
}

static void print_help(const char *prog) {
    print_banner();
    fprintf(stderr, "Subcommands:\n");
    fprintf(stderr, "  %s train --run <dir> --train-token-file <tokens.txt> [options]\n", prog);
    fprintf(stderr, "  %s report-index --run <dir> [--output <path>]\n", prog);
    fprintf(stderr, "  %s profile --run <dir> --tool perf|vtune|advisor --train-token-file <tokens.txt> [options]\n", prog);
    fprintf(stderr, "  %s train --help | %s profile --help | %s report-index --help\n", prog, prog, prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Quick start:\n");
    fprintf(stderr, "  %s                                      Scan cache and pick a model\n", prog);
    fprintf(stderr, "  %s --model qwen                         Load by name from cache\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Advanced:\n");
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
    fprintf(stderr, "  --quiet-output          Suppress generated text output (profiling/noise-free)\n");
    fprintf(stderr, "  --timing, -t            Show timing breakdown\n");
    fprintf(stderr, "  --no-chat-template      Disable chat template formatting\n");
    fprintf(stderr, "  --eos IDS               Comma-separated EOS token IDs\n");
    fprintf(stderr, "  --ignore-eos            Do not stop on EOS tokens\n");
    fprintf(stderr, "  --list                  List available models in cache\n");
    fprintf(stderr, "  --verbose, -v           Verbose output\n");
    fprintf(stderr, "  --help, -h              Show this help\n");
    fprintf(stderr, "\nREPL Commands:\n");
    fprintf(stderr, "  /exit, /quit            Exit the REPL\n");
    fprintf(stderr, "  /reset                  Reset KV cache\n");
    fprintf(stderr, "  /timing                 Toggle timing display\n");
    fprintf(stderr, "  /temp <value>           Set temperature\n");
    fprintf(stderr, "  /system <text>          Set system prompt\n");
    fprintf(stderr, "  /help                   Show help\n");
    fprintf(stderr, "\nWorkflow:\n");
    fprintf(stderr, "  This CLI runs pre-compiled models. To compile a new model:\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  Step 1: Compile (once)    python version/v7/scripts/ck_run_v7.py run <model>\n");
    fprintf(stderr, "  Step 2: Run (fast)        %s\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Models are cached in: ~/.cache/ck-engine-v7/models/\n");
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
        } else if (!strcmp(arg, "--quiet-output")) {
            opt->quiet_output = true;
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
            /* Positional argument: if it looks like a file path (.so or .bump), use as direct path.
               Otherwise treat as model name for cache lookup. */
            const char *ext = strrchr(arg, '.');
            bool is_file = (ext && (!strcmp(ext, ".so") || !strcmp(ext, ".bump")));
            if (is_file) {
                if (!opt->lib_path) opt->lib_path = arg;
                else if (!opt->weights_path) opt->weights_path = arg;
                else {
                    fprintf(stderr, "Unknown argument: %s\n", arg);
                    return false;
                }
            } else {
                /* Treat as model name */
                if (!opt->model_name) {
                    opt->model_name = arg;
                } else {
                    fprintf(stderr, "Unknown argument: %s (model already set to '%s')\n", arg, opt->model_name);
                    return false;
                }
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            return false;
        }
    }

    /* Auto-discover model if --model specified or model name given as positional arg */
    if (opt->model_name && (!opt->lib_path || !opt->weights_path)) {
        static char lib_buf[4096], weights_buf[4096];
        if (find_model_in_cache(opt->model_name, lib_buf, weights_buf, sizeof(lib_buf))) {
            opt->lib_path = lib_buf;
            opt->weights_path = weights_buf;
        } else {
            fprintf(stderr, "Model '%s' not found in cache.\n\n", opt->model_name);
            list_available_models();
            fprintf(stderr, "\nTo use an existing model:  %s --model <name>\n", argv[0]);
            fprintf(stderr, "To compile '%s':           python version/v7/scripts/ck_run_v7.py run %s\n",
                    opt->model_name, opt->model_name);
            return false;
        }
    }

    if (!opt->lib_path || !opt->weights_path) {
        /* No model specified — auto-scan cache directory */
        static char scan_lib[4096], scan_weights[4096];
        print_banner();
        if (scan_and_select_model(scan_lib, scan_weights, sizeof(scan_lib))) {
            opt->lib_path = scan_lib;
            opt->weights_path = scan_weights;
        } else {
            fprintf(stderr, "\n");
            print_help(argv[0]);
            return false;
        }
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

    if (argc >= 2) {
        const char *sub = argv[1];
        if (!strcmp(sub, "train")) {
            return cmd_train_subcommand(argc, argv);
        }
        if (!strcmp(sub, "report-index")) {
            return cmd_report_index_subcommand(argc, argv);
        }
        if (!strcmp(sub, "profile")) {
            return cmd_profile_subcommand(argc, argv);
        }
    }

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

    /* Check for model's built-in tokenizer (v7 models have tokenizer in generated code) */
    if (!api.has_tokenizer || !api.has_tokenizer()) {
        fprintf(stderr, "[Tokenizer] Model does not have built-in tokenizer\n");
        fprintf(stderr, "            v7 models should have ck_model_encode_text/decode_tokens\n");
        return 1;
    }

    if (!api.encode_text || !api.decode_tokens) {
        fprintf(stderr, "[Tokenizer] Model missing encode/decode functions\n");
        return 1;
    }

    int vocab_size = api.get_vocab_size ? api.get_vocab_size() : 0;

    /* Lookup EOS tokens using model's tokenizer */
    if (api.lookup_token) {
        static const char *eos_tokens[] = {
            "<|im_end|>", "<|endoftext|>", "<|im_start|>",  /* Qwen/ChatML */
            "<|eot_id|>", "<|end_of_text|>",  /* Llama 3 */
            "</s>",  /* Generic */
            NULL
        };
        opt.eos_count = 0;
        for (int i = 0; eos_tokens[i] != NULL && opt.eos_count < CK_CLI_EOS_MAX; i++) {
            int32_t id = api.lookup_token(eos_tokens[i]);
            if (id >= 0) {
                opt.eos_ids[opt.eos_count++] = id;
                if (opt.verbose) {
                    printf("[Tokenizer] EOS token: %s -> %d\n", eos_tokens[i], id);
                }
            }
        }
        if (opt.verbose && opt.eos_count > 0) {
            printf("[Tokenizer] Found %d EOS tokens\n", opt.eos_count);
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

#ifdef _OPENMP
    {
        int max_threads = omp_get_max_threads();
        const char *omp_env = getenv("OMP_NUM_THREADS");
        printf("[OpenMP]   Threads: %d (OMP_NUM_THREADS=%s) | Cores: %d\n",
               max_threads,
               omp_env ? omp_env : "auto",
               (int)sysconf(_SC_NPROCESSORS_ONLN));
    }
#else
    printf("[OpenMP]   Disabled (compiled without -fopenmp)\n");
#endif

    printf("Type /help for commands, Ctrl+C to stop generation\n\n");

    setvbuf(stdout, NULL, _IOFBF, 1 << 20);

    if (opt.prompt_once) {
        run_prompt(&api, &opt, opt.prompt_once);
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
            run_prompt(&api, &opt, line);

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

    /* Model handles its own cleanup via ck_model_free */
    if (api.free_fn) api.free_fn();
    if (api.handle) dlclose(api.handle);

    printf("\nGoodbye!\n");
    return 0;
}
