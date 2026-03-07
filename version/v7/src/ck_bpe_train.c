/*
 * ck_bpe_train.c
 *
 * Standalone C BPE trainer for v7 workflows.
 *
 * Goals:
 *   - Pure C (no Python dependency)
 *   - Deterministic training from a local text corpus
 *   - Threaded pair counting/rewrite with a persistent worker team
 *   - Memory-pool style allocation (one big arena for core training state)
 *   - Emits tokenizer.json + optional binary artifacts for v7 pipelines
 *
 * Notes on token representation:
 *   - Default mode: GPT-2 byte-level UTF-8 pieces (ByteLevel-compatible).
 *   - --ascii-only mode: raw ASCII bytes (\t,\n,\r,0x20-0x7E) with no UTF-8 marker mapping.
 *   - Learned merges concatenate base symbols into larger token strings.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define CK_BPE_VERSION "0.1.0"

#define SPECIAL_COUNT 4
#define ID_UNK 0
#define ID_BOS 1
#define ID_EOS 2
#define ID_PAD 3
#define BASE_BYTE_ID 4
#define BASE_BYTE_COUNT 256
#define ASCII_PRINT_MIN 0x20
#define ASCII_PRINT_MAX 0x7E
#define ASCII_BASE_COUNT (3 + (ASCII_PRINT_MAX - ASCII_PRINT_MIN + 1))  /* \\t,\\n,\\r + printable */

#define DEFAULT_VOCAB_SIZE 1024
#define DEFAULT_MIN_FREQ 2
#define DEFAULT_EXTS ".svg,.txt,.xml,.html,.md,.json"

#define MAX_TOKEN_TEXT 1048576

/* ------------------------------ Utility Types ------------------------------ */

typedef struct {
    char *path;
    size_t size;
} FileEntry;

typedef struct {
    FileEntry *items;
    int count;
    int cap;
} FileList;

typedef struct {
    uint8_t *base;
    size_t size;
    size_t used;
} Arena;

typedef struct {
    int32_t left;
    int32_t right;
    uint8_t byte;
    uint8_t is_base;
} SymbolNode;

typedef struct {
    int32_t left;
    int32_t right;
    int32_t merged;
    uint32_t freq;
} MergeRule;

typedef struct {
    uint64_t key;   /* key+1 (0 = empty) */
    uint32_t count;
} PairSlot;

typedef enum {
    JOB_NONE = 0,
    JOB_COUNT = 1,
    JOB_REPLACE = 2,
} JobType;

struct Trainer;

typedef struct {
    int tid;
    struct Trainer *tr;
    PairSlot *local_table;
} Worker;

typedef struct {
    const char *corpus_dir;
    const char *out_json;
    const char *binary_out_dir;
    const char *special_tokens_file;
    const char *exts_csv;
    int vocab_size;
    int min_freq;
    int max_piece_bytes;
    int threads;
    int verbose;
    int ascii_only;
} Options;

typedef struct Trainer {
    Options opt;

    int num_files;
    size_t total_tokens;
    size_t max_seq_len;

    int *seq_offsets;
    int *seq_lens;
    int *seq_caps;

    int32_t *seq_a;
    int32_t *seq_b;
    int use_a;

    SymbolNode *symbols;
    int32_t *symbol_text_lens;
    int num_symbols;
    int max_symbols;
    int base_symbol_count;
    int base_byte_id;
    char **reserved_specials;
    int num_reserved_specials;
    int32_t byte_to_symbol[256]; /* byte -> token id (or -1 if disallowed in mode) */

    MergeRule *merges;
    int num_merges;

    int32_t merge_left;
    int32_t merge_right;
    int32_t merge_new;

    size_t local_cap;
    size_t global_cap;
    PairSlot *global_table;

    int nthreads;
    Worker *workers;
    pthread_t *threads;

    pthread_mutex_t mu;
    pthread_cond_t cv_job;
    pthread_cond_t cv_done;
    int job_serial;
    int done_workers;
    int stop;
    JobType job;
} Trainer;

/* ------------------------------ Small Helpers ------------------------------ */

static void fatalf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) fatalf("out of memory (malloc %zu)", n);
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) fatalf("out of memory (calloc %zu x %zu)", n, sz);
    return p;
}

static size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

static void arena_init(Arena *a, size_t bytes) {
    a->base = (uint8_t *)calloc(1, bytes);
    if (!a->base) fatalf("failed to allocate arena (%zu bytes)", bytes);
    a->size = bytes;
    a->used = 0;
}

static void *arena_alloc(Arena *a, size_t bytes, size_t align) {
    size_t off = align_up(a->used, align ? align : 8);
    if (off + bytes > a->size) {
        fatalf("arena overflow: need %zu bytes (used=%zu size=%zu)", bytes, a->used, a->size);
    }
    void *p = a->base + off;
    a->used = off + bytes;
    return p;
}

static bool path_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = (char *)xmalloc(n);
    memcpy(out, s, n);
    return out;
}

static size_t next_pow2_size(size_t v) {
    if (v < 2) return 2;
    v--;
    for (size_t i = 1; i < sizeof(size_t) * 8; i <<= 1) v |= (v >> i);
    return v + 1;
}

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static inline uint64_t pair_key(int32_t a, int32_t b) {
    return (((uint64_t)(uint32_t)a) << 32) | (uint64_t)(uint32_t)b;
}

static inline uint64_t key_slot(uint64_t k) {
    return k + 1ULL; /* 0 reserved for empty */
}

static inline uint64_t slot_key_decode(uint64_t slot_key) {
    return slot_key - 1ULL;
}

static inline void pair_table_add(PairSlot *tab, size_t cap, uint64_t slot_k, uint32_t delta) {
    size_t mask = cap - 1;
    size_t i = (size_t)splitmix64(slot_k) & mask;
    for (;;) {
        if (tab[i].key == 0) {
            tab[i].key = slot_k;
            tab[i].count = delta;
            return;
        }
        if (tab[i].key == slot_k) {
            uint64_t sum = (uint64_t)tab[i].count + (uint64_t)delta;
            tab[i].count = (sum > UINT32_MAX) ? UINT32_MAX : (uint32_t)sum;
            return;
        }
        i = (i + 1) & mask;
    }
}

static void ensure_parent_dir(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    char *slash = strrchr(tmp, '/');
    if (!slash) return;
    *slash = '\0';
    if (tmp[0] == '\0') return;

    char buf[4096];
    size_t n = strlen(tmp);
    size_t j = 0;
    for (size_t i = 0; i <= n; i++) {
        if (i == n || tmp[i] == '/') {
            if (j > 0) {
                buf[j] = '\0';
                if (mkdir(buf, 0775) != 0 && errno != EEXIST) {
                    fatalf("mkdir failed: %s (%s)", buf, strerror(errno));
                }
            }
            if (i < n) {
                buf[j++] = '/';
            }
        } else {
            buf[j++] = tmp[i];
        }
    }
}

/*
 * Create a directory tree in-place (mkdir -p semantics) without shelling out.
 * This keeps the trainer self-contained and avoids quoting/injection issues.
 */
static void ensure_dir_recursive(const char *dir) {
    if (!dir || !*dir) return;

    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", dir);
    size_t n = strlen(tmp);
    if (n == 0) return;

    /* Trim trailing slash so we do not attempt mkdir("") at the end. */
    while (n > 1 && tmp[n - 1] == '/') {
        tmp[--n] = '\0';
    }

    for (size_t i = 1; i <= n; i++) {
        if (tmp[i] == '/' || tmp[i] == '\0') {
            char saved = tmp[i];
            tmp[i] = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                fatalf("mkdir failed: %s (%s)", tmp, strerror(errno));
            }
            tmp[i] = saved;
        }
    }
}

/* ------------------------------ File Discovery ----------------------------- */

static bool extension_matches(const char *filename, const char *exts_csv) {
    if (!exts_csv || !*exts_csv) return true;
    if (strcmp(exts_csv, "*") == 0) return true;

    const char *dot = strrchr(filename, '.');
    if (!dot) return false;

    char ext_l[64];
    snprintf(ext_l, sizeof(ext_l), "%s", dot);
    for (char *p = ext_l; *p; ++p) *p = (char)tolower((unsigned char)*p);

    char csv[512];
    snprintf(csv, sizeof(csv), "%s", exts_csv);
    char *save = NULL;
    char *tok = strtok_r(csv, ",", &save);
    while (tok) {
        while (*tok && isspace((unsigned char)*tok)) tok++;
        char cand[64];
        snprintf(cand, sizeof(cand), "%s", tok);
        size_t m = strlen(cand);
        while (m > 0 && isspace((unsigned char)cand[m - 1])) cand[--m] = '\0';
        for (char *p = cand; *p; ++p) *p = (char)tolower((unsigned char)*p);
        if (strcmp(cand, ext_l) == 0) return true;
        tok = strtok_r(NULL, ",", &save);
    }
    return false;
}

static void file_list_push(FileList *fl, const char *path, size_t sz) {
    if (fl->count == fl->cap) {
        int new_cap = fl->cap ? fl->cap * 2 : 64;
        fl->items = (FileEntry *)realloc(fl->items, (size_t)new_cap * sizeof(FileEntry));
        if (!fl->items) fatalf("out of memory expanding file list");
        fl->cap = new_cap;
    }
    fl->items[fl->count].path = strdup(path);
    fl->items[fl->count].size = sz;
    if (!fl->items[fl->count].path) fatalf("out of memory duplicating path");
    fl->count++;
}

static void collect_files_recursive(const char *dir, const char *exts_csv, FileList *out) {
    DIR *d = opendir(dir);
    if (!d) fatalf("cannot open corpus dir: %s", dir);

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (lstat(path, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            collect_files_recursive(path, exts_csv, out);
            continue;
        }
        if (!S_ISREG(st.st_mode)) continue;
        if (st.st_size <= 0) continue;
        if (!extension_matches(ent->d_name, exts_csv)) continue;

        file_list_push(out, path, (size_t)st.st_size);
    }
    closedir(d);
}

static int file_entry_cmp(const void *a, const void *b) {
    const FileEntry *fa = (const FileEntry *)a;
    const FileEntry *fb = (const FileEntry *)b;
    return strcmp(fa->path, fb->path);
}

static void file_list_free(FileList *fl) {
    if (!fl) return;
    for (int i = 0; i < fl->count; i++) free(fl->items[i].path);
    free(fl->items);
    fl->items = NULL;
    fl->count = fl->cap = 0;
}

static bool token_list_contains(char **items, int count, const char *needle) {
    for (int i = 0; i < count; i++) {
        if (strcmp(items[i], needle) == 0) return true;
    }
    return false;
}

static void load_special_tokens_file(const char *path, char ***out_tokens, int *out_count) {
    *out_tokens = NULL;
    *out_count = 0;
    if (!path || !*path) return;

    FILE *f = fopen(path, "r");
    if (!f) fatalf("cannot open special tokens file: %s", path);

    char **tokens = NULL;
    int count = 0;
    int cap = 0;
    char *line = NULL;
    size_t line_cap = 0;

    while (getline(&line, &line_cap, f) != -1) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (n == 0) continue;
        if (!strcmp(line, "<|unk|>") || !strcmp(line, "<|bos|>") ||
            !strcmp(line, "<|eos|>") || !strcmp(line, "<|pad|>")) {
            continue;
        }
        if (token_list_contains(tokens, count, line)) continue;
        if (count == cap) {
            int new_cap = cap ? cap * 2 : 32;
            tokens = (char **)realloc(tokens, (size_t)new_cap * sizeof(char *));
            if (!tokens) fatalf("out of memory loading special tokens");
            cap = new_cap;
        }
        tokens[count++] = xstrdup(line);
    }

    free(line);
    fclose(f);
    *out_tokens = tokens;
    *out_count = count;
}

/* ------------------------------ Worker Engine ------------------------------ */

static void shard_bounds(int n_items, int n_shards, int shard_id, int *out_start, int *out_end) {
    int s = (n_items * shard_id) / n_shards;
    int e = (n_items * (shard_id + 1)) / n_shards;
    *out_start = s;
    *out_end = e;
}

static void worker_job_count(Worker *w) {
    Trainer *tr = w->tr;
    PairSlot *tab = w->local_table;
    memset(tab, 0, tr->local_cap * sizeof(PairSlot));

    int start = 0, end = 0;
    shard_bounds(tr->num_files, tr->nthreads, w->tid, &start, &end);

    int32_t *src = tr->use_a ? tr->seq_a : tr->seq_b;

    for (int s = start; s < end; s++) {
        int off = tr->seq_offsets[s];
        int len = tr->seq_lens[s];
        for (int i = 0; i + 1 < len; i++) {
            if (src[off + i] < tr->base_byte_id || src[off + i + 1] < tr->base_byte_id) continue;
            uint64_t k = pair_key(src[off + i], src[off + i + 1]);
            pair_table_add(tab, tr->local_cap, key_slot(k), 1);
        }
    }
}

static void worker_job_replace(Worker *w) {
    Trainer *tr = w->tr;
    int start = 0, end = 0;
    shard_bounds(tr->num_files, tr->nthreads, w->tid, &start, &end);

    int32_t *src = tr->use_a ? tr->seq_a : tr->seq_b;
    int32_t *dst = tr->use_a ? tr->seq_b : tr->seq_a;

    int32_t L = tr->merge_left;
    int32_t R = tr->merge_right;
    int32_t M = tr->merge_new;

    for (int s = start; s < end; s++) {
        int off = tr->seq_offsets[s];
        int len = tr->seq_lens[s];
        int out = 0;

        int i = 0;
        while (i < len) {
            if (i + 1 < len && src[off + i] == L && src[off + i + 1] == R) {
                dst[off + out++] = M;
                i += 2;
            } else {
                dst[off + out++] = src[off + i++];
            }
        }
        tr->seq_lens[s] = out;
    }
}

static void *worker_main(void *arg) {
    Worker *w = (Worker *)arg;
    Trainer *tr = w->tr;
    int seen_serial = 0;

    for (;;) {
        pthread_mutex_lock(&tr->mu);
        while (!tr->stop && seen_serial == tr->job_serial) {
            pthread_cond_wait(&tr->cv_job, &tr->mu);
        }
        if (tr->stop) {
            pthread_mutex_unlock(&tr->mu);
            break;
        }

        JobType job = tr->job;
        seen_serial = tr->job_serial;
        pthread_mutex_unlock(&tr->mu);

        if (job == JOB_COUNT) {
            worker_job_count(w);
        } else if (job == JOB_REPLACE) {
            worker_job_replace(w);
        }

        pthread_mutex_lock(&tr->mu);
        tr->done_workers++;
        if (tr->done_workers >= tr->nthreads) {
            pthread_cond_signal(&tr->cv_done);
        }
        pthread_mutex_unlock(&tr->mu);
    }
    return NULL;
}

static void dispatch_job_and_wait(Trainer *tr, JobType job) {
    pthread_mutex_lock(&tr->mu);
    tr->job = job;
    tr->done_workers = 0;
    tr->job_serial++;
    pthread_cond_broadcast(&tr->cv_job);
    while (tr->done_workers < tr->nthreads) {
        pthread_cond_wait(&tr->cv_done, &tr->mu);
    }
    pthread_mutex_unlock(&tr->mu);
}

/* ------------------------------- Training Core ------------------------------ */

static void merge_local_counts(Trainer *tr) {
    memset(tr->global_table, 0, tr->global_cap * sizeof(PairSlot));

    for (int t = 0; t < tr->nthreads; t++) {
        PairSlot *lt = tr->workers[t].local_table;
        for (size_t i = 0; i < tr->local_cap; i++) {
            if (lt[i].key == 0 || lt[i].count == 0) continue;
            pair_table_add(tr->global_table, tr->global_cap, lt[i].key, lt[i].count);
        }
    }
}

static bool select_best_pair(Trainer *tr, int32_t *out_left, int32_t *out_right, uint32_t *out_freq) {
    uint64_t best_slot_key = 0;
    uint32_t best_count = 0;
    const int max_piece_bytes = tr->opt.max_piece_bytes;

    for (size_t i = 0; i < tr->global_cap; i++) {
        uint64_t sk = tr->global_table[i].key;
        uint32_t ct = tr->global_table[i].count;
        if (sk == 0 || ct == 0) continue;
        uint64_t k = slot_key_decode(sk);
        int32_t L = (int32_t)(k >> 32);
        int32_t R = (int32_t)(k & 0xFFFFFFFFu);
        if (L < tr->base_byte_id || R < tr->base_byte_id) continue;
        if (max_piece_bytes > 0) {
            if (L < 0 || R < 0 || L >= tr->num_symbols || R >= tr->num_symbols) continue;
            int64_t merged_len = (int64_t)tr->symbol_text_lens[L] + (int64_t)tr->symbol_text_lens[R];
            if (merged_len > (int64_t)max_piece_bytes) continue;
        }
        if (ct > best_count || (ct == best_count && sk < best_slot_key)) {
            best_count = ct;
            best_slot_key = sk;
        }
    }

    if (best_slot_key == 0 || best_count < (uint32_t)tr->opt.min_freq) return false;

    uint64_t k = slot_key_decode(best_slot_key);
    int32_t L = (int32_t)(k >> 32);
    int32_t R = (int32_t)(k & 0xFFFFFFFFu);

    *out_left = L;
    *out_right = R;
    *out_freq = best_count;
    return true;
}

static size_t total_active_tokens(const Trainer *tr) {
    size_t sum = 0;
    for (int i = 0; i < tr->num_files; i++) sum += (size_t)tr->seq_lens[i];
    return sum;
}

static bool is_ascii_train_byte(uint8_t b) {
    return b == '\t' || b == '\n' || b == '\r' ||
           (b >= ASCII_PRINT_MIN && b <= ASCII_PRINT_MAX);
}

static int base_piece_text_len(uint8_t byte, int ascii_only) {
    if (ascii_only) return 1;
    if (byte >= 0x21 && byte <= 0x7E) return 1;
    unsigned int codepoint;
    if (byte <= 0x20) codepoint = 0x100u + byte;
    else if (byte >= 0x7F && byte <= 0xA0) codepoint = 0x100u + byte;
    else codepoint = byte;
    if (codepoint < 0x80u) return 1;
    if (codepoint < 0x800u) return 2;
    return 3;
}

static void init_symbols(Trainer *tr) {
    for (int i = 0; i < 256; i++) tr->byte_to_symbol[i] = -1;

    tr->symbols[ID_UNK].left = -1;
    tr->symbols[ID_UNK].right = -1;
    tr->symbols[ID_UNK].is_base = 0;
    tr->symbol_text_lens[ID_UNK] = (int32_t)(sizeof("<|unk|>") - 1);
    tr->symbols[ID_BOS].left = -1;
    tr->symbols[ID_BOS].right = -1;
    tr->symbols[ID_BOS].is_base = 0;
    tr->symbol_text_lens[ID_BOS] = (int32_t)(sizeof("<|bos|>") - 1);
    tr->symbols[ID_EOS].left = -1;
    tr->symbols[ID_EOS].right = -1;
    tr->symbols[ID_EOS].is_base = 0;
    tr->symbol_text_lens[ID_EOS] = (int32_t)(sizeof("<|eos|>") - 1);
    tr->symbols[ID_PAD].left = -1;
    tr->symbols[ID_PAD].right = -1;
    tr->symbols[ID_PAD].is_base = 0;
    tr->symbol_text_lens[ID_PAD] = (int32_t)(sizeof("<|pad|>") - 1);

    int next_id = BASE_BYTE_ID;
    for (int i = 0; i < tr->num_reserved_specials; i++) {
        int id = next_id++;
        tr->symbols[id].left = -1;
        tr->symbols[id].right = -1;
        tr->symbols[id].is_base = 0;
        tr->symbol_text_lens[id] = (int32_t)strlen(tr->reserved_specials[i]);
    }

    tr->base_byte_id = next_id;
    if (tr->opt.ascii_only) {
        for (int b = 0; b < 256; b++) {
            if (!is_ascii_train_byte((uint8_t)b)) continue;
            tr->symbols[next_id].left = -1;
            tr->symbols[next_id].right = -1;
            tr->symbols[next_id].byte = (uint8_t)b;
            tr->symbols[next_id].is_base = 1;
            tr->symbol_text_lens[next_id] = (int32_t)base_piece_text_len((uint8_t)b, 1);
            tr->byte_to_symbol[b] = next_id;
            next_id++;
        }
    } else {
        for (int b = 0; b < BASE_BYTE_COUNT; b++) {
            int id = BASE_BYTE_ID + b;
            tr->symbols[id].left = -1;
            tr->symbols[id].right = -1;
            tr->symbols[id].byte = (uint8_t)b;
            tr->symbols[id].is_base = 1;
            tr->symbol_text_lens[id] = (int32_t)base_piece_text_len((uint8_t)b, 0);
            tr->byte_to_symbol[b] = id;
            next_id++;
        }
    }

    tr->base_symbol_count = next_id - tr->base_byte_id;
    tr->num_symbols = next_id;
    tr->num_merges = 0;
}

static void run_bpe_training(Trainer *tr) {
    size_t start_tokens = total_active_tokens(tr);
    if (tr->opt.verbose) {
        fprintf(stderr, "[bpe] start tokens=%zu symbols=%d target_vocab=%d min_freq=%d threads=%d\n",
                start_tokens, tr->num_symbols, tr->opt.vocab_size, tr->opt.min_freq, tr->nthreads);
    }

    int iter = 0;
    while (tr->num_symbols < tr->max_symbols) {
        dispatch_job_and_wait(tr, JOB_COUNT);
        merge_local_counts(tr);

        int32_t L = -1, R = -1;
        uint32_t freq = 0;
        if (!select_best_pair(tr, &L, &R, &freq)) {
            if (tr->opt.verbose) {
                fprintf(stderr, "[bpe] stopping: no pair >= min_freq (%d)\n", tr->opt.min_freq);
            }
            break;
        }

        int new_id = tr->num_symbols;
        tr->merge_left = L;
        tr->merge_right = R;
        tr->merge_new = new_id;

        tr->symbols[new_id].left = L;
        tr->symbols[new_id].right = R;
        tr->symbols[new_id].byte = 0;
        tr->symbols[new_id].is_base = 0;
        tr->symbol_text_lens[new_id] = tr->symbol_text_lens[L] + tr->symbol_text_lens[R];

        tr->merges[tr->num_merges].left = L;
        tr->merges[tr->num_merges].right = R;
        tr->merges[tr->num_merges].merged = new_id;
        tr->merges[tr->num_merges].freq = freq;

        dispatch_job_and_wait(tr, JOB_REPLACE);
        tr->use_a = !tr->use_a;

        tr->num_merges++;
        tr->num_symbols++;
        iter++;

        if (tr->opt.verbose && (iter <= 20 || (iter % 50) == 0)) {
            size_t alive = total_active_tokens(tr);
            fprintf(stderr, "[bpe] iter=%d merged=(%d,%d)->%d freq=%u active_tokens=%zu\n",
                    iter, L, R, new_id, freq, alive);
        }
    }

    if (tr->opt.verbose) {
        size_t end_tokens = total_active_tokens(tr);
        fprintf(stderr, "[bpe] done merges=%d vocab=%d tokens: %zu -> %zu (%.2fx)\n",
                tr->num_merges, tr->num_symbols, start_tokens, end_tokens,
                end_tokens > 0 ? (double)start_tokens / (double)end_tokens : 0.0);
    }
}

/* ------------------------------ Token Rendering ---------------------------- */

/* Match CK true_bpe's byte-level preprocessing mapping for tokenizer compatibility. */
static int byte_to_gpt2_piece(uint8_t byte, char out[4]) {
    if (byte >= 0x21 && byte <= 0x7E) {
        out[0] = (char)byte;
        out[1] = '\0';
        return 1;
    }

    unsigned int codepoint;
    if (byte <= 0x20) codepoint = 0x100u + byte;
    else if (byte >= 0x7F && byte <= 0xA0) codepoint = 0x100u + byte;
    else codepoint = byte;

    if (codepoint < 0x80) {
        out[0] = (char)codepoint;
        out[1] = '\0';
        return 1;
    }
    if (codepoint < 0x800) {
        out[0] = (char)(0xC0u | (codepoint >> 6));
        out[1] = (char)(0x80u | (codepoint & 0x3Fu));
        out[2] = '\0';
        return 2;
    }

    out[0] = (char)(0xE0u | (codepoint >> 12));
    out[1] = (char)(0x80u | ((codepoint >> 6) & 0x3Fu));
    out[2] = (char)(0x80u | (codepoint & 0x3Fu));
    out[3] = '\0';
    return 3;
}

static int byte_to_ascii_piece(uint8_t byte, char out[2]) {
    out[0] = (char)byte;
    out[1] = '\0';
    return 1;
}

static const char *special_name(const Trainer *tr, int id) {
    switch (id) {
        case ID_UNK: return "<|unk|>";
        case ID_BOS: return "<|bos|>";
        case ID_EOS: return "<|eos|>";
        case ID_PAD: return "<|pad|>";
        default:
            if (tr && id >= SPECIAL_COUNT && id < tr->base_byte_id) {
                return tr->reserved_specials[id - SPECIAL_COUNT];
            }
            return NULL;
    }
}

typedef struct {
    char *buf;
    size_t cap;
    size_t len;
    int overflow;
} StrBuilder;

static void sb_append(StrBuilder *sb, const char *s) {
    if (sb->overflow) return;
    size_t n = strlen(s);
    if (sb->len + n + 1 > sb->cap) {
        sb->overflow = 1;
        return;
    }
    memcpy(sb->buf + sb->len, s, n);
    sb->len += n;
    sb->buf[sb->len] = '\0';
}

static int emit_symbol_text_recursive(const Trainer *tr, int32_t id, StrBuilder *sb, int depth) {
    if (depth > 8192) return -1;
    if (id < 0 || id >= tr->num_symbols) return -1;

    const char *sp = special_name(tr, id);
    if (sp) {
        sb_append(sb, sp);
        return sb->overflow ? -1 : 0;
    }

    const SymbolNode *s = &tr->symbols[id];
    if (s->is_base) {
        if (tr->opt.ascii_only) {
            char seg_ascii[2];
            byte_to_ascii_piece((uint8_t)s->byte, seg_ascii);
            sb_append(sb, seg_ascii);
        } else {
            char seg[4];
            byte_to_gpt2_piece((uint8_t)s->byte, seg);
            sb_append(sb, seg);
        }
        return sb->overflow ? -1 : 0;
    }

    if (emit_symbol_text_recursive(tr, s->left, sb, depth + 1) != 0) return -1;
    if (emit_symbol_text_recursive(tr, s->right, sb, depth + 1) != 0) return -1;
    return sb->overflow ? -1 : 0;
}

static int render_symbol_text(const Trainer *tr, int32_t id, char *out, size_t out_cap) {
    if (!out || out_cap == 0) return -1;
    out[0] = '\0';
    StrBuilder sb = { .buf = out, .cap = out_cap, .len = 0, .overflow = 0 };
    if (emit_symbol_text_recursive(tr, id, &sb, 0) != 0) return -1;
    return sb.overflow ? -1 : 0;
}

static void json_escape(const char *in, char *out, size_t cap) {
    size_t o = 0;
    for (size_t i = 0; in[i] != '\0'; i++) {
        unsigned char c = (unsigned char)in[i];
        const char *esc = NULL;
        char tmp[8];

        if (c == '"') esc = "\\\"";
        else if (c == '\\') esc = "\\\\";
        else if (c == '\b') esc = "\\b";
        else if (c == '\f') esc = "\\f";
        else if (c == '\n') esc = "\\n";
        else if (c == '\r') esc = "\\r";
        else if (c == '\t') esc = "\\t";
        else if (c < 0x20) {
            snprintf(tmp, sizeof(tmp), "\\u%04X", (unsigned int)c);
            esc = tmp;
        }

        if (esc) {
            size_t n = strlen(esc);
            if (o + n + 1 >= cap) break;
            memcpy(out + o, esc, n);
            o += n;
        } else {
            if (o + 2 >= cap) break;
            out[o++] = (char)c;
        }
    }
    out[o] = '\0';
}

/* ------------------------------ Output Writers ----------------------------- */

static void write_tokenizer_json(const char *path, const Trainer *tr) {
    ensure_parent_dir(path);
    FILE *f = fopen(path, "w");
    if (!f) fatalf("cannot write tokenizer.json: %s", path);

    char *tok_buf = (char *)xmalloc(MAX_TOKEN_TEXT);
    char *esc_a = (char *)xmalloc(MAX_TOKEN_TEXT * 2);
    char *esc_b = (char *)xmalloc(MAX_TOKEN_TEXT * 2);
    const char *gpt2_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    char gpt2_regex_json[MAX_TOKEN_TEXT];
    json_escape(gpt2_regex, gpt2_regex_json, sizeof(gpt2_regex_json));

    fprintf(f, "{\n");
    fprintf(f, "  \"version\": \"1.0\",\n");
    fprintf(f, "  \"ck_mode\": \"%s\",\n", tr->opt.ascii_only ? "ascii_bpe" : "bytelevel_bpe");
    fprintf(f, "  \"truncation\": null,\n");
    fprintf(f, "  \"padding\": null,\n");
    fprintf(f, "  \"added_tokens\": [\n");
    fprintf(f, "    {\"id\": %d, \"content\": \"<|unk|>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true},\n", ID_UNK);
    fprintf(f, "    {\"id\": %d, \"content\": \"<|bos|>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true},\n", ID_BOS);
    fprintf(f, "    {\"id\": %d, \"content\": \"<|eos|>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true},\n", ID_EOS);
    fprintf(f, "    {\"id\": %d, \"content\": \"<|pad|>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}", ID_PAD);
    for (int i = 0; i < tr->num_reserved_specials; i++) {
        json_escape(tr->reserved_specials[i], esc_a, MAX_TOKEN_TEXT * 2);
        fprintf(f, ",\n    {\"id\": %d, \"content\": \"%s\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true}", SPECIAL_COUNT + i, esc_a);
    }
    fprintf(f, "\n");
    fprintf(f, "  ],\n");

    fprintf(f, "  \"normalizer\": null,\n");
    if (tr->opt.ascii_only) {
        fprintf(f, "  \"pre_tokenizer\": null,\n");
        fprintf(f, "  \"post_processor\": null,\n");
        fprintf(f, "  \"decoder\": null,\n");
    } else {
        fprintf(f, "  \"pre_tokenizer\": {\"type\": \"Sequence\", \"pretokenizers\": [{\"type\": \"Split\", \"pattern\": {\"Regex\": \"%s\"}, \"behavior\": \"Isolated\", \"invert\": false}, {\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": false, \"use_regex\": false}]},\n", gpt2_regex_json);
        fprintf(f, "  \"post_processor\": null,\n");
        fprintf(f, "  \"decoder\": {\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": false, \"use_regex\": false},\n");
    }

    fprintf(f, "  \"model\": {\n");
    fprintf(f, "    \"type\": \"BPE\",\n");
    fprintf(f, "    \"dropout\": null,\n");
    fprintf(f, "    \"unk_token\": \"<|unk|>\",\n");
    fprintf(f, "    \"continuing_subword_prefix\": null,\n");
    fprintf(f, "    \"end_of_word_suffix\": null,\n");
    fprintf(f, "    \"fuse_unk\": false,\n");
    fprintf(f, "    \"byte_fallback\": true,\n");

    fprintf(f, "    \"vocab\": {\n");
    for (int id = 0; id < tr->num_symbols; id++) {
        if (render_symbol_text(tr, id, tok_buf, MAX_TOKEN_TEXT) != 0) {
            fatalf("token render overflow for id=%d", id);
        }
        json_escape(tok_buf, esc_a, MAX_TOKEN_TEXT * 2);
        fprintf(f, "      \"%s\": %d%s\n", esc_a, id, (id + 1 < tr->num_symbols) ? "," : "");
    }
    fprintf(f, "    },\n");

    fprintf(f, "    \"merges\": [\n");
    for (int i = 0; i < tr->num_merges; i++) {
        int32_t L = tr->merges[i].left;
        int32_t R = tr->merges[i].right;
        if (render_symbol_text(tr, L, tok_buf, MAX_TOKEN_TEXT) != 0) fatalf("merge render left overflow");
        json_escape(tok_buf, esc_a, MAX_TOKEN_TEXT * 2);
        if (render_symbol_text(tr, R, tok_buf, MAX_TOKEN_TEXT) != 0) fatalf("merge render right overflow");
        json_escape(tok_buf, esc_b, MAX_TOKEN_TEXT * 2);
        fprintf(f, "      \"%s %s\"%s\n", esc_a, esc_b, (i + 1 < tr->num_merges) ? "," : "");
    }
    fprintf(f, "    ]\n");
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    free(tok_buf);
    free(esc_a);
    free(esc_b);
}

static void write_binary_exports(const char *out_dir, const Trainer *tr) {
    if (!out_dir || !*out_dir) return;
    ensure_dir_recursive(out_dir);

    char *tok_buf = (char *)xmalloc(MAX_TOKEN_TEXT);

    int32_t *offsets = (int32_t *)xcalloc((size_t)tr->num_symbols, sizeof(int32_t));

    size_t blob_bytes = 0;
    for (int id = 0; id < tr->num_symbols; id++) {
        if (render_symbol_text(tr, id, tok_buf, MAX_TOKEN_TEXT) != 0) fatalf("binary render overflow");
        offsets[id] = (int32_t)blob_bytes;
        blob_bytes += strlen(tok_buf) + 1;
    }

    char *blob = (char *)xmalloc(blob_bytes > 0 ? blob_bytes : 1);
    size_t cursor = 0;
    for (int id = 0; id < tr->num_symbols; id++) {
        if (render_symbol_text(tr, id, tok_buf, MAX_TOKEN_TEXT) != 0) fatalf("binary render overflow");
        size_t n = strlen(tok_buf) + 1;
        memcpy(blob + cursor, tok_buf, n);
        cursor += n;
    }

    char path_offsets[4096], path_strings[4096], path_merges[4096], path_meta[4096];
    snprintf(path_offsets, sizeof(path_offsets), "%s/vocab_offsets.bin", out_dir);
    snprintf(path_strings, sizeof(path_strings), "%s/vocab_strings.bin", out_dir);
    snprintf(path_merges, sizeof(path_merges), "%s/vocab_merges.bin", out_dir);
    snprintf(path_meta, sizeof(path_meta), "%s/tokenizer_meta.json", out_dir);

    FILE *f = fopen(path_offsets, "wb");
    if (!f) fatalf("cannot write %s", path_offsets);
    fwrite(offsets, sizeof(int32_t), (size_t)tr->num_symbols, f);
    fclose(f);

    f = fopen(path_strings, "wb");
    if (!f) fatalf("cannot write %s", path_strings);
    fwrite(blob, 1, blob_bytes, f);
    fclose(f);

    f = fopen(path_merges, "wb");
    if (!f) fatalf("cannot write %s", path_merges);
    for (int i = 0; i < tr->num_merges; i++) {
        int32_t triple[3] = { tr->merges[i].left, tr->merges[i].right, tr->merges[i].merged };
        fwrite(triple, sizeof(int32_t), 3, f);
    }
    fclose(f);

    f = fopen(path_meta, "w");
    if (!f) fatalf("cannot write %s", path_meta);
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"ck.bpe.binary.v1\",\n");
    fprintf(f, "  \"mode\": \"%s\",\n", tr->opt.ascii_only ? "ascii_bpe" : "bytelevel_bpe");
    fprintf(f, "  \"vocab_size\": %d,\n", tr->num_symbols);
    fprintf(f, "  \"num_merges\": %d,\n", tr->num_merges);
    fprintf(f, "  \"num_reserved_special_tokens\": %d,\n", tr->num_reserved_specials);
    fprintf(f, "  \"max_piece_bytes\": %d,\n", tr->opt.max_piece_bytes);
    fprintf(f, "  \"offsets\": \"%s\",\n", path_offsets);
    fprintf(f, "  \"strings\": \"%s\",\n", path_strings);
    fprintf(f, "  \"merges\": \"%s\"\n", path_merges);
    fprintf(f, "}\n");
    fclose(f);

    free(tok_buf);
    free(offsets);
    free(blob);
}

/* ------------------------------- CLI Parsing ------------------------------- */

static void print_help(const char *prog) {
    fprintf(stderr,
        "ck-bpe-train %s\n"
        "Standalone C BPE trainer (threaded, arena-backed).\n\n"
        "Usage:\n"
        "  %s --corpus-dir DIR --out tokenizer.json [options]\n\n"
        "Required:\n"
        "  --corpus-dir DIR         Directory containing training text files\n"
        "  --out PATH               Output tokenizer.json path\n\n"
        "Options:\n"
        "  --vocab-size N           Target vocab size (default: %d)\n"
        "  --min-freq N             Minimum pair frequency to merge (default: %d)\n"
        "  --max-piece-bytes N      Max token byte-length after merges (default: %d, 0=unbounded)\n"
        "  --threads N              Worker threads (default: cpu cores)\n"
        "  --ext LIST               Comma-separated file extensions (default: %s)\n"
        "                           Use '*' for all regular files\n"
        "  --binary-out-dir DIR     Also write vocab_offsets.bin, vocab_strings.bin, vocab_merges.bin\n"
        "  --special-tokens-file P  One token per line; reserve these as atomic special tokens\n"
        "  --ascii-only             Train ASCII-identity BPE (no ByteLevel UTF-8 mapping)\n"
        "                           Allowed bytes: \\t, \\n, \\r, and 0x20-0x7E\n"
        "  --verbose                Print training progress\n"
        "  --help                   Show this help\n",
        CK_BPE_VERSION, prog, DEFAULT_VOCAB_SIZE, DEFAULT_MIN_FREQ, 0, DEFAULT_EXTS);
}

static void parse_args(int argc, char **argv, Options *opt) {
    memset(opt, 0, sizeof(*opt));
    opt->vocab_size = DEFAULT_VOCAB_SIZE;
    opt->min_freq = DEFAULT_MIN_FREQ;
    opt->max_piece_bytes = 0;
    opt->threads = 0;
    opt->exts_csv = DEFAULT_EXTS;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            print_help(argv[0]);
            exit(0);
        } else if (!strcmp(a, "--corpus-dir") && i + 1 < argc) {
            opt->corpus_dir = argv[++i];
        } else if (!strcmp(a, "--out") && i + 1 < argc) {
            opt->out_json = argv[++i];
        } else if (!strcmp(a, "--binary-out-dir") && i + 1 < argc) {
            opt->binary_out_dir = argv[++i];
        } else if (!strcmp(a, "--special-tokens-file") && i + 1 < argc) {
            opt->special_tokens_file = argv[++i];
        } else if (!strcmp(a, "--vocab-size") && i + 1 < argc) {
            opt->vocab_size = atoi(argv[++i]);
        } else if (!strcmp(a, "--min-freq") && i + 1 < argc) {
            opt->min_freq = atoi(argv[++i]);
        } else if (!strcmp(a, "--max-piece-bytes") && i + 1 < argc) {
            opt->max_piece_bytes = atoi(argv[++i]);
        } else if (!strcmp(a, "--threads") && i + 1 < argc) {
            opt->threads = atoi(argv[++i]);
        } else if (!strcmp(a, "--ext") && i + 1 < argc) {
            opt->exts_csv = argv[++i];
        } else if (!strcmp(a, "--ascii-only")) {
            opt->ascii_only = 1;
        } else if (!strcmp(a, "--verbose") || !strcmp(a, "-v")) {
            opt->verbose = 1;
        } else {
            fatalf("unknown option: %s", a);
        }
    }

    if (!opt->corpus_dir || !*opt->corpus_dir) fatalf("missing --corpus-dir");
    if (!opt->out_json || !*opt->out_json) fatalf("missing --out");
    if (!path_exists(opt->corpus_dir)) fatalf("corpus dir not found: %s", opt->corpus_dir);
    if (opt->special_tokens_file && !path_exists(opt->special_tokens_file)) {
        fatalf("special tokens file not found: %s", opt->special_tokens_file);
    }
    int base_count = opt->ascii_only ? ASCII_BASE_COUNT : BASE_BYTE_COUNT;
    if (opt->min_freq < 1) opt->min_freq = 1;
    if (opt->max_piece_bytes < 0) fatalf("--max-piece-bytes must be >= 0");
    if (opt->threads <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        opt->threads = (n > 0) ? (int)n : 4;
    }
    if (opt->threads > 64) opt->threads = 64;
}

/* ------------------------------- Main Driver ------------------------------- */

static int match_reserved_special(const Trainer *tr, const char *buf, size_t len, size_t pos, int32_t *out_id, size_t *out_advance) {
    int32_t best_id = -1;
    size_t best_len = 0;
    for (int i = 0; i < tr->num_reserved_specials; i++) {
        const char *tok = tr->reserved_specials[i];
        size_t tok_len = strlen(tok);
        if (tok_len == 0 || pos + tok_len > len) continue;
        if (memcmp(buf + pos, tok, tok_len) != 0) continue;
        if (tok_len > best_len) {
            best_len = tok_len;
            best_id = SPECIAL_COUNT + i;
        }
    }
    if (best_id < 0) return 0;
    *out_id = best_id;
    *out_advance = best_len;
    return 1;
}

static void load_corpus_into_sequences(const FileList *fl, Trainer *tr) {
    size_t cursor = 0;
    for (int i = 0; i < fl->count; i++) {
        tr->seq_offsets[i] = (int)cursor;
        tr->seq_caps[i] = (int)fl->items[i].size;
        tr->seq_lens[i] = 0;

        FILE *f = fopen(fl->items[i].path, "rb");
        if (!f) fatalf("cannot open corpus file: %s", fl->items[i].path);
        size_t file_bytes = fl->items[i].size;
        char *buf = (char *)xmalloc(file_bytes);
        size_t got = fread(buf, 1, file_bytes, f);
        fclose(f);
        if (got != file_bytes) {
            free(buf);
            fatalf("short read from corpus file: %s", fl->items[i].path);
        }

        int len = 0;
        size_t pos = 0;
        while (pos < file_bytes) {
            if (cursor >= tr->total_tokens) {
                free(buf);
                fatalf("corpus grew during read; rerun training");
            }
            int32_t special_id = -1;
            size_t advance = 0;
            if (tr->num_reserved_specials > 0 &&
                match_reserved_special(tr, buf, file_bytes, pos, &special_id, &advance)) {
                tr->seq_a[cursor++] = special_id;
                len++;
                pos += advance;
                continue;
            }
            uint8_t b = (uint8_t)buf[pos++];
            int32_t id = tr->byte_to_symbol[b];
            if (id < 0) {
                free(buf);
                if (tr->opt.ascii_only) {
                    fatalf("non-ASCII byte in --ascii-only corpus: file=%s byte=0x%02X", fl->items[i].path, (unsigned)b);
                }
                fatalf("unmapped byte in corpus: file=%s byte=0x%02X", fl->items[i].path, (unsigned)b);
            }
            tr->seq_a[cursor++] = id;
            len++;
        }
        free(buf);
        tr->seq_lens[i] = len;
        if ((size_t)len > tr->max_seq_len) tr->max_seq_len = (size_t)len;
    }
    if (cursor != tr->total_tokens) {
        tr->total_tokens = cursor;
    }
}

int main(int argc, char **argv) {
    Options opt;
    parse_args(argc, argv, &opt);
    int base_count = opt.ascii_only ? ASCII_BASE_COUNT : BASE_BYTE_COUNT;

    char **reserved_specials = NULL;
    int num_reserved_specials = 0;
    load_special_tokens_file(opt.special_tokens_file, &reserved_specials, &num_reserved_specials);

    FileList files = {0};
    collect_files_recursive(opt.corpus_dir, opt.exts_csv, &files);
    if (files.count == 0) {
        fatalf("no corpus files found under %s (ext=%s)", opt.corpus_dir, opt.exts_csv ? opt.exts_csv : "*");
    }
    qsort(files.items, (size_t)files.count, sizeof(FileEntry), file_entry_cmp);

    size_t total_tokens = 0;
    for (int i = 0; i < files.count; i++) total_tokens += files.items[i].size;
    if (total_tokens < 2) fatalf("corpus too small (tokens=%zu)", total_tokens);

    int nthreads = opt.threads;
    if (nthreads > files.count) nthreads = files.count;
    if (nthreads < 1) nthreads = 1;

    size_t est_pairs = (total_tokens > 1) ? (total_tokens - 1) : 1;
    size_t global_cap = next_pow2_size(est_pairs * 2 + 1024);
    size_t local_cap = next_pow2_size((est_pairs / (size_t)nthreads) * 4 + 4096);

    int min_vocab_size = SPECIAL_COUNT + num_reserved_specials + base_count;
    if (opt.vocab_size < min_vocab_size) {
        fatalf("--vocab-size must be >= %d for selected mode + reserved specials", min_vocab_size);
    }

    size_t arena_bytes = 0;
    arena_bytes += align_up((size_t)files.count * sizeof(int), 64) * 3;           /* offsets/lens/caps */
    arena_bytes += align_up(total_tokens * sizeof(int32_t), 64) * 2;               /* seq_a, seq_b */
    arena_bytes += align_up((size_t)opt.vocab_size * sizeof(SymbolNode), 64);      /* symbols */
    arena_bytes += align_up((size_t)opt.vocab_size * sizeof(int32_t), 64);          /* symbol text lengths */
    arena_bytes += align_up((size_t)opt.vocab_size * sizeof(MergeRule), 64);       /* merges */
    arena_bytes += align_up((size_t)nthreads * sizeof(Worker), 64);                 /* workers */
    arena_bytes += align_up((size_t)nthreads * sizeof(pthread_t), 64);              /* threads */
    arena_bytes += align_up(global_cap * sizeof(PairSlot), 64);                     /* global pair table */
    arena_bytes += align_up((size_t)nthreads * local_cap * sizeof(PairSlot), 64);   /* local pair tables */

    Arena arena;
    arena_init(&arena, arena_bytes);

    Trainer tr;
    memset(&tr, 0, sizeof(tr));
    tr.opt = opt;
    tr.num_files = files.count;
    tr.total_tokens = total_tokens;
    tr.max_seq_len = 0;
    tr.max_symbols = opt.vocab_size;
    tr.local_cap = local_cap;
    tr.global_cap = global_cap;
    tr.nthreads = nthreads;
    tr.use_a = 1;
    tr.reserved_specials = reserved_specials;
    tr.num_reserved_specials = num_reserved_specials;

    tr.seq_offsets = (int *)arena_alloc(&arena, (size_t)files.count * sizeof(int), 64);
    tr.seq_lens = (int *)arena_alloc(&arena, (size_t)files.count * sizeof(int), 64);
    tr.seq_caps = (int *)arena_alloc(&arena, (size_t)files.count * sizeof(int), 64);
    tr.seq_a = (int32_t *)arena_alloc(&arena, total_tokens * sizeof(int32_t), 64);
    tr.seq_b = (int32_t *)arena_alloc(&arena, total_tokens * sizeof(int32_t), 64);
    tr.symbols = (SymbolNode *)arena_alloc(&arena, (size_t)opt.vocab_size * sizeof(SymbolNode), 64);
    tr.symbol_text_lens = (int32_t *)arena_alloc(&arena, (size_t)opt.vocab_size * sizeof(int32_t), 64);
    tr.merges = (MergeRule *)arena_alloc(&arena, (size_t)opt.vocab_size * sizeof(MergeRule), 64);
    tr.workers = (Worker *)arena_alloc(&arena, (size_t)nthreads * sizeof(Worker), 64);
    tr.threads = (pthread_t *)arena_alloc(&arena, (size_t)nthreads * sizeof(pthread_t), 64);
    tr.global_table = (PairSlot *)arena_alloc(&arena, global_cap * sizeof(PairSlot), 64);

    PairSlot *local_tables = (PairSlot *)arena_alloc(&arena, (size_t)nthreads * local_cap * sizeof(PairSlot), 64);

    init_symbols(&tr);
    load_corpus_into_sequences(&files, &tr);

    pthread_mutex_init(&tr.mu, NULL);
    pthread_cond_init(&tr.cv_job, NULL);
    pthread_cond_init(&tr.cv_done, NULL);

    for (int t = 0; t < nthreads; t++) {
        tr.workers[t].tid = t;
        tr.workers[t].tr = &tr;
        tr.workers[t].local_table = local_tables + ((size_t)t * local_cap);
        if (pthread_create(&tr.threads[t], NULL, worker_main, &tr.workers[t]) != 0) {
            fatalf("pthread_create failed for worker %d", t);
        }
    }

    if (opt.verbose) {
        fprintf(stderr, "[bpe] files=%d total_tokens=%zu max_seq_len=%zu threads=%d\n",
                tr.num_files, tr.total_tokens, tr.max_seq_len, tr.nthreads);
        fprintf(stderr, "[bpe] arena=%zu bytes used=%zu local_cap=%zu global_cap=%zu\n",
                arena.size, arena.used, tr.local_cap, tr.global_cap);
    }

    run_bpe_training(&tr);

    pthread_mutex_lock(&tr.mu);
    tr.stop = 1;
    pthread_cond_broadcast(&tr.cv_job);
    pthread_mutex_unlock(&tr.mu);
    for (int t = 0; t < nthreads; t++) {
        pthread_join(tr.threads[t], NULL);
    }

    write_tokenizer_json(opt.out_json, &tr);
    write_binary_exports(opt.binary_out_dir, &tr);

    printf("ck-bpe-train complete\n");
    printf("  corpus_dir: %s\n", opt.corpus_dir);
    printf("  files:      %d\n", tr.num_files);
    printf("  mode:       %s\n", opt.ascii_only ? "ascii_bpe" : "bytelevel_bpe");
    printf("  vocab_size: %d\n", tr.num_symbols);
    printf("  merges:     %d\n", tr.num_merges);
    printf("  reserved_special_tokens: %d\n", tr.num_reserved_specials);
    printf("  max_piece_bytes: %d\n", opt.max_piece_bytes);
    printf("  out:        %s\n", opt.out_json);
    if (opt.binary_out_dir && *opt.binary_out_dir) {
        printf("  binary_out: %s\n", opt.binary_out_dir);
    }

    pthread_cond_destroy(&tr.cv_job);
    pthread_cond_destroy(&tr.cv_done);
    pthread_mutex_destroy(&tr.mu);

    free(arena.base);
    for (int i = 0; i < num_reserved_specials; i++) free(reserved_specials[i]);
    free(reserved_specials);
    file_list_free(&files);
    return 0;
}
