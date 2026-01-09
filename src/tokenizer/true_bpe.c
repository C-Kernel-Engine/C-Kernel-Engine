/*
 * True BPE (Byte-Pair Encoding) Tokenizer
 *
 * Implements the actual BPE algorithm used by GPT-2, LLaMA, Qwen, etc.
 * Unlike greedy longest-match, this applies merge rules in priority order.
 *
 * Algorithm:
 * 1. Split text into initial tokens (characters or bytes)
 * 2. Find the highest-priority merge that can be applied
 * 3. Apply that merge (combine two adjacent tokens into one)
 * 4. Repeat until no more merges possible
 * 5. Look up final tokens in vocabulary to get IDs
 *
 * Data Structures:
 * - Vocabulary: token string -> token ID (hash table)
 * - Merge rules: (left_id, right_id) -> (merged_id, priority) (hash table)
 * - Token list: dynamic array for the working token sequence
 *
 * By Anthony Shivakumar
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

#include "tokenizer/true_bpe.h"

/* ═══════════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define MERGE_HASH_SIZE 65536      /* Size of merge lookup hash table */
#define INITIAL_TOKEN_CAPACITY 256 /* Initial capacity for token list */
#define MAX_TOKEN_LEN 128          /* Maximum length of a single token string */

/* ═══════════════════════════════════════════════════════════════════════════════
 * Data Structures
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* A single merge rule */
typedef struct {
    int32_t left_id;      /* Left token ID */
    int32_t right_id;     /* Right token ID */
    int32_t merged_id;    /* Resulting merged token ID */
    int32_t priority;     /* Lower = higher priority (applied first) */
} CKBPEMerge;

/* Hash table entry for merge lookup */
typedef struct CKMergeEntry {
    uint64_t key;                /* Hash key: (left_id << 32) | right_id */
    CKBPEMerge merge;            /* The merge rule */
    struct CKMergeEntry *next;   /* Chain for collision handling */
} CKMergeEntry;

/* Merge lookup hash table */
typedef struct {
    CKMergeEntry **buckets;
    size_t num_buckets;
    size_t num_entries;
} CKMergeTable;

/* A working token during BPE encoding */
typedef struct {
    char *str;           /* Token string */
    int32_t id;          /* Token ID (-1 if not yet looked up) */
    uint16_t len;        /* String length */
    bool is_merged;      /* True if this is result of a merge */
} CKBPEToken;

/* Dynamic token list for BPE processing */
typedef struct {
    CKBPEToken *tokens;
    size_t count;
    size_t capacity;
} CKBPETokenList;

/* Main True BPE tokenizer structure */
struct CKTrueBPE {
    /* Vocabulary: token string -> token ID */
    CKTokenizerHashTable *vocab;

    /* Reverse vocabulary: token ID -> token string */
    char **id_to_token;
    size_t vocab_size;
    size_t vocab_capacity;

    /* Merge rules: (left_id, right_id) -> merged_id with priority */
    CKMergeTable *merges;
    int32_t num_merges;

    /* Special token IDs */
    int32_t unk_id;
    int32_t bos_id;
    int32_t eos_id;
    int32_t pad_id;

    /* Configuration */
    CKBPEConfig config;

    /* String buffer for token operations */
    char *str_buffer;
    size_t str_buffer_size;
};

/* ═══════════════════════════════════════════════════════════════════════════════
 * Merge Table Operations
 * ═══════════════════════════════════════════════════════════════════════════════ */

static uint64_t merge_key(int32_t left_id, int32_t right_id) {
    return ((uint64_t)left_id << 32) | (uint32_t)right_id;
}

static size_t merge_hash(uint64_t key, size_t num_buckets) {
    /* Simple hash mixing */
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key % num_buckets;
}

static CKMergeTable *merge_table_create(size_t num_buckets) {
    CKMergeTable *table = (CKMergeTable *)malloc(sizeof(CKMergeTable));
    if (!table) return NULL;

    table->buckets = (CKMergeEntry **)calloc(num_buckets, sizeof(CKMergeEntry *));
    if (!table->buckets) {
        free(table);
        return NULL;
    }

    table->num_buckets = num_buckets;
    table->num_entries = 0;
    return table;
}

static void merge_table_free(CKMergeTable *table) {
    if (!table) return;

    for (size_t i = 0; i < table->num_buckets; i++) {
        CKMergeEntry *entry = table->buckets[i];
        while (entry) {
            CKMergeEntry *next = entry->next;
            free(entry);
            entry = next;
        }
    }

    free(table->buckets);
    free(table);
}

static int merge_table_insert(CKMergeTable *table, const CKBPEMerge *merge) {
    uint64_t key = merge_key(merge->left_id, merge->right_id);
    size_t bucket = merge_hash(key, table->num_buckets);

    /* Check if already exists */
    CKMergeEntry *entry = table->buckets[bucket];
    while (entry) {
        if (entry->key == key) {
            /* Update existing */
            entry->merge = *merge;
            return 0;
        }
        entry = entry->next;
    }

    /* Create new entry */
    entry = (CKMergeEntry *)malloc(sizeof(CKMergeEntry));
    if (!entry) return -1;

    entry->key = key;
    entry->merge = *merge;
    entry->next = table->buckets[bucket];
    table->buckets[bucket] = entry;
    table->num_entries++;

    return 0;
}

static const CKBPEMerge *merge_table_lookup(const CKMergeTable *table, int32_t left_id, int32_t right_id) {
    uint64_t key = merge_key(left_id, right_id);
    size_t bucket = merge_hash(key, table->num_buckets);

    CKMergeEntry *entry = table->buckets[bucket];
    while (entry) {
        if (entry->key == key) {
            return &entry->merge;
        }
        entry = entry->next;
    }

    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Token List Operations
 * ═══════════════════════════════════════════════════════════════════════════════ */

static CKBPETokenList *token_list_create(size_t initial_capacity) {
    CKBPETokenList *list = (CKBPETokenList *)malloc(sizeof(CKBPETokenList));
    if (!list) return NULL;

    list->tokens = (CKBPEToken *)calloc(initial_capacity, sizeof(CKBPEToken));
    if (!list->tokens) {
        free(list);
        return NULL;
    }

    list->count = 0;
    list->capacity = initial_capacity;
    return list;
}

static void token_list_free(CKBPETokenList *list) {
    if (!list) return;

    for (size_t i = 0; i < list->count; i++) {
        if (list->tokens[i].str) {
            free(list->tokens[i].str);
        }
    }

    free(list->tokens);
    free(list);
}

static void token_list_clear(CKBPETokenList *list) {
    for (size_t i = 0; i < list->count; i++) {
        if (list->tokens[i].str) {
            free(list->tokens[i].str);
            list->tokens[i].str = NULL;
        }
    }
    list->count = 0;
}

static int token_list_append(CKBPETokenList *list, const char *str, size_t len, int32_t id) {
    if (list->count >= list->capacity) {
        size_t new_cap = list->capacity * 2;
        CKBPEToken *new_tokens = (CKBPEToken *)realloc(list->tokens, new_cap * sizeof(CKBPEToken));
        if (!new_tokens) return -1;
        list->tokens = new_tokens;
        list->capacity = new_cap;
        /* Zero new entries */
        memset(list->tokens + list->count, 0, (new_cap - list->count) * sizeof(CKBPEToken));
    }

    CKBPEToken *tok = &list->tokens[list->count];
    tok->str = (char *)malloc(len + 1);
    if (!tok->str) return -1;

    memcpy(tok->str, str, len);
    tok->str[len] = '\0';
    tok->len = (uint16_t)len;
    tok->id = id;
    tok->is_merged = false;

    list->count++;
    return 0;
}

/* Merge tokens at positions i and i+1 into a single token */
static int token_list_merge_at(CKBPETokenList *list, size_t pos, const char *merged_str, size_t merged_len, int32_t merged_id) {
    if (pos + 1 >= list->count) return -1;

    /* Free old strings */
    free(list->tokens[pos].str);
    free(list->tokens[pos + 1].str);

    /* Create merged token */
    list->tokens[pos].str = (char *)malloc(merged_len + 1);
    if (!list->tokens[pos].str) return -1;

    memcpy(list->tokens[pos].str, merged_str, merged_len);
    list->tokens[pos].str[merged_len] = '\0';
    list->tokens[pos].len = (uint16_t)merged_len;
    list->tokens[pos].id = merged_id;
    list->tokens[pos].is_merged = true;

    /* Shift remaining tokens left */
    for (size_t i = pos + 1; i < list->count - 1; i++) {
        list->tokens[i] = list->tokens[i + 1];
    }
    list->count--;

    /* Clear the now-unused last slot */
    list->tokens[list->count].str = NULL;

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * True BPE Tokenizer API
 * ═══════════════════════════════════════════════════════════════════════════════ */

CKTrueBPE *ck_true_bpe_create(void) {
    CKTrueBPE *bpe = (CKTrueBPE *)calloc(1, sizeof(CKTrueBPE));
    if (!bpe) return NULL;

    /* Create vocabulary hash table */
    bpe->vocab = ck_tokenizer_hash_table_create(CK_TOKENIZER_HT_BUCKETS_LARGE);
    if (!bpe->vocab) {
        free(bpe);
        return NULL;
    }

    /* Create merge table */
    bpe->merges = merge_table_create(MERGE_HASH_SIZE);
    if (!bpe->merges) {
        ck_tokenizer_hash_table_free(bpe->vocab, true);
        free(bpe);
        return NULL;
    }

    /* Initialize reverse vocabulary */
    bpe->vocab_capacity = 4096;
    bpe->id_to_token = (char **)calloc(bpe->vocab_capacity, sizeof(char *));
    if (!bpe->id_to_token) {
        merge_table_free(bpe->merges);
        ck_tokenizer_hash_table_free(bpe->vocab, true);
        free(bpe);
        return NULL;
    }

    /* String buffer for token operations */
    bpe->str_buffer_size = 4096;
    bpe->str_buffer = (char *)malloc(bpe->str_buffer_size);
    if (!bpe->str_buffer) {
        free(bpe->id_to_token);
        merge_table_free(bpe->merges);
        ck_tokenizer_hash_table_free(bpe->vocab, true);
        free(bpe);
        return NULL;
    }

    /* Default special token IDs */
    bpe->unk_id = 0;
    bpe->bos_id = -1;
    bpe->eos_id = -1;
    bpe->pad_id = -1;

    /* Default config */
    bpe->config.add_bos = false;
    bpe->config.add_eos = false;
    bpe->config.byte_fallback = true;
    bpe->config.space_prefix_style = CK_SPACE_PREFIX_AUTO;

    return bpe;
}

void ck_true_bpe_free(CKTrueBPE *bpe) {
    if (!bpe) return;

    if (bpe->vocab) {
        ck_tokenizer_hash_table_free(bpe->vocab, true);
    }

    if (bpe->merges) {
        merge_table_free(bpe->merges);
    }

    if (bpe->id_to_token) {
        for (size_t i = 0; i < bpe->vocab_size; i++) {
            if (bpe->id_to_token[i]) {
                free(bpe->id_to_token[i]);
            }
        }
        free(bpe->id_to_token);
    }

    if (bpe->str_buffer) {
        free(bpe->str_buffer);
    }

    free(bpe);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Vocabulary Management
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Token info stored in vocab hash table */
typedef struct {
    int32_t id;
    float score;
} BPETokenInfo;

int ck_true_bpe_add_token(CKTrueBPE *bpe, const char *token, int32_t id, float score) {
    if (!bpe || !token) return -1;

    /* Ensure reverse vocab has space */
    if (id >= (int32_t)bpe->vocab_capacity) {
        size_t new_cap = bpe->vocab_capacity * 2;
        while (new_cap <= (size_t)id) new_cap *= 2;

        char **new_array = (char **)realloc(bpe->id_to_token, new_cap * sizeof(char *));
        if (!new_array) return -1;

        memset(new_array + bpe->vocab_capacity, 0, (new_cap - bpe->vocab_capacity) * sizeof(char *));
        bpe->id_to_token = new_array;
        bpe->vocab_capacity = new_cap;
    }

    /* Check if token exists */
    BPETokenInfo *existing = (BPETokenInfo *)ck_tokenizer_hash_table_lookup(bpe->vocab, token);
    if (existing) {
        existing->id = id;
        existing->score = score;
        if (bpe->id_to_token[id]) free(bpe->id_to_token[id]);
        bpe->id_to_token[id] = strdup(token);
        return 0;
    }

    /* Create new token info */
    BPETokenInfo *info = (BPETokenInfo *)malloc(sizeof(BPETokenInfo));
    if (!info) return -1;

    info->id = id;
    info->score = score;

    if (ck_tokenizer_hash_table_insert(bpe->vocab, token, info) != 0) {
        free(info);
        return -1;
    }

    if (id >= (int32_t)bpe->vocab_size) {
        bpe->vocab_size = id + 1;
    }

    if (bpe->id_to_token[id]) free(bpe->id_to_token[id]);
    bpe->id_to_token[id] = strdup(token);

    return 0;
}

int ck_true_bpe_add_merge(CKTrueBPE *bpe, int32_t left_id, int32_t right_id, int32_t merged_id, int32_t priority) {
    if (!bpe) return -1;

    CKBPEMerge merge = {
        .left_id = left_id,
        .right_id = right_id,
        .merged_id = merged_id,
        .priority = priority
    };

    int ret = merge_table_insert(bpe->merges, &merge);
    if (ret == 0) {
        bpe->num_merges++;
    }
    return ret;
}

int ck_true_bpe_add_merge_by_tokens(CKTrueBPE *bpe, const char *left, const char *right, int32_t priority) {
    if (!bpe || !left || !right) return -1;

    /* Look up token IDs */
    BPETokenInfo *left_info = (BPETokenInfo *)ck_tokenizer_hash_table_lookup(bpe->vocab, left);
    BPETokenInfo *right_info = (BPETokenInfo *)ck_tokenizer_hash_table_lookup(bpe->vocab, right);

    if (!left_info || !right_info) {
        return -1;  /* Tokens not in vocabulary */
    }

    /* Create merged token string */
    size_t left_len = strlen(left);
    size_t right_len = strlen(right);
    size_t merged_len = left_len + right_len;

    if (merged_len >= bpe->str_buffer_size) {
        return -1;  /* Too long */
    }

    memcpy(bpe->str_buffer, left, left_len);
    memcpy(bpe->str_buffer + left_len, right, right_len);
    bpe->str_buffer[merged_len] = '\0';

    /* Look up or create merged token */
    BPETokenInfo *merged_info = (BPETokenInfo *)ck_tokenizer_hash_table_lookup(bpe->vocab, bpe->str_buffer);
    int32_t merged_id;

    if (merged_info) {
        merged_id = merged_info->id;
    } else {
        /* Merged token should already exist in vocabulary */
        return -1;
    }

    return ck_true_bpe_add_merge(bpe, left_info->id, right_info->id, merged_id, priority);
}

void ck_true_bpe_set_special_ids(CKTrueBPE *bpe, int32_t unk, int32_t bos, int32_t eos, int32_t pad) {
    if (!bpe) return;
    bpe->unk_id = unk;
    bpe->bos_id = bos;
    bpe->eos_id = eos;
    bpe->pad_id = pad;
}

void ck_true_bpe_set_config(CKTrueBPE *bpe, const CKBPEConfig *config) {
    if (!bpe || !config) return;
    bpe->config = *config;
}

int32_t ck_true_bpe_lookup(const CKTrueBPE *bpe, const char *token) {
    if (!bpe || !token) return -1;

    BPETokenInfo *info = (BPETokenInfo *)ck_tokenizer_hash_table_lookup(bpe->vocab, token);
    return info ? info->id : bpe->unk_id;
}

const char *ck_true_bpe_id_to_token(const CKTrueBPE *bpe, int32_t id) {
    if (!bpe || id < 0 || id >= (int32_t)bpe->vocab_size) return NULL;
    return bpe->id_to_token[id];
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Space Prefix Detection
 * ═══════════════════════════════════════════════════════════════════════════════ */

CKSpacePrefixStyle ck_true_bpe_detect_space_style(CKTrueBPE *bpe) {
    if (!bpe) return CK_SPACE_PREFIX_GPT2;

    if (bpe->config.space_prefix_style != CK_SPACE_PREFIX_AUTO) {
        return bpe->config.space_prefix_style;
    }

    /* Count tokens starting with each style */
    int gpt2_count = 0;  /* Ġ (0xC4 0xA0) */
    int spm_count = 0;   /* ▁ (0xE2 0x96 0x81) */

    for (size_t i = 0; i < bpe->vocab_size && i < 10000; i++) {
        const char *token = bpe->id_to_token[i];
        if (!token) continue;

        unsigned char c0 = (unsigned char)token[0];
        unsigned char c1 = (unsigned char)token[1];

        if (c0 == 0xC4 && c1 == 0xA0) {
            gpt2_count++;
        } else if (c0 == 0xE2 && c1 == 0x96 && (unsigned char)token[2] == 0x81) {
            spm_count++;
        }
    }

    CKSpacePrefixStyle detected = (spm_count > gpt2_count * 2) ? CK_SPACE_PREFIX_SPM : CK_SPACE_PREFIX_GPT2;
    bpe->config.space_prefix_style = detected;

    return detected;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * True BPE Encoding
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * GPT-2 Byte-Level BPE Character Mapping
 *
 * GPT-2 uses a byte-level encoding where certain bytes are mapped to
 * special Unicode characters to avoid issues with control characters:
 *
 * - Space (0x20) → Ġ (U+0120, bytes 0xC4 0xA0)
 * - Newline (0x0A) → Ċ (U+010A, bytes 0xC4 0x8A)
 * - Tab (0x09) → ĉ (U+0109, bytes 0xC4 0x89)
 * - Carriage return (0x0D) → č (U+010D, bytes 0xC4 0x8D)
 *
 * The mapping is: byte 0x00-0x20 → U+0100 + byte
 * Regular printable ASCII (0x21-0x7E) stays as-is
 */

/* Convert a byte to GPT-2 byte-level BPE representation */
static int byte_to_gpt2(unsigned char byte, char *out) {
    if (byte >= 0x21 && byte <= 0x7E && byte != '!') {
        /* Printable ASCII (except control chars) stays as-is */
        out[0] = (char)byte;
        return 1;
    }

    /* Control chars and special bytes map to U+0100 + byte */
    /* This gives us characters like Ġ (U+0120), Ċ (U+010A), etc. */
    unsigned int codepoint;

    /* GPT-2's byte_encoder mapping */
    if (byte == '!') codepoint = byte;
    else if (byte == '"') codepoint = byte;
    else if (byte >= '#' && byte <= '~') codepoint = byte;
    else if (byte == 0x21) codepoint = '!';  /* Already handled above */
    else {
        /* Map 0x00-0x20, 0x7F-0xFF to offset range */
        if (byte <= 0x20) {
            codepoint = 0x100 + byte;  /* 0x00-0x20 → U+0100-U+0120 */
        } else if (byte >= 0x7F && byte <= 0xA0) {
            codepoint = 0x100 + byte;  /* 0x7F-0xA0 → U+017F-U+01A0 */
        } else {
            codepoint = byte;  /* Others: 0xA1-0xFF stay as-is */
        }
    }

    /* Encode as UTF-8 */
    if (codepoint < 0x80) {
        out[0] = (char)codepoint;
        return 1;
    } else if (codepoint < 0x800) {
        out[0] = (char)(0xC0 | (codepoint >> 6));
        out[1] = (char)(0x80 | (codepoint & 0x3F));
        return 2;
    } else {
        out[0] = (char)(0xE0 | (codepoint >> 12));
        out[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        out[2] = (char)(0x80 | (codepoint & 0x3F));
        return 3;
    }
}

/* Preprocess text: convert to byte-level BPE representation */
static int preprocess_text(const CKTrueBPE *bpe, const char *text, int text_len, char *out, int out_max) {
    CKSpacePrefixStyle style = bpe->config.space_prefix_style;
    int out_len = 0;

    /* For SentencePiece, add ▁ at start */
    if (style == CK_SPACE_PREFIX_SPM && text_len > 0 && text[0] != ' ') {
        if (out_len + 3 > out_max) return -1;
        out[out_len++] = (char)0xE2;
        out[out_len++] = (char)0x96;
        out[out_len++] = (char)0x81;
    }

    for (int i = 0; i < text_len; i++) {
        unsigned char byte = (unsigned char)text[i];

        if (style == CK_SPACE_PREFIX_SPM) {
            /* SentencePiece style: only convert spaces */
            if (byte == ' ') {
                if (out_len + 3 > out_max) return -1;
                out[out_len++] = (char)0xE2;
                out[out_len++] = (char)0x96;
                out[out_len++] = (char)0x81;
            } else {
                if (out_len + 1 > out_max) return -1;
                out[out_len++] = (char)byte;
            }
        } else {
            /* GPT-2 style: full byte-level encoding */
            char encoded[4];
            int enc_len = byte_to_gpt2(byte, encoded);
            if (out_len + enc_len > out_max) return -1;
            for (int j = 0; j < enc_len; j++) {
                out[out_len++] = encoded[j];
            }
        }
    }

    return out_len;
}

/* Get UTF-8 character length */
static int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;       /* 0xxxxxxx */
    if ((c & 0xE0) == 0xC0) return 2;    /* 110xxxxx */
    if ((c & 0xF0) == 0xE0) return 3;    /* 1110xxxx */
    if ((c & 0xF8) == 0xF0) return 4;    /* 11110xxx */
    return 1;  /* Invalid, treat as single byte */
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GPT-2 Style Pretokenizer
 *
 * The GPT-2 pretokenizer uses a regex to split text into chunks BEFORE BPE:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
 *   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *
 * Key behaviors:
 * - Words get an optional leading space attached: "hello world" -> ["hello", " world"]
 * - Multiple spaces before a word: "   hello" -> ["  ", " hello"] (n-1 spaces, then space+word)
 * - Numbers stay together: "123" -> ["123"]
 * - Punctuation may get leading space: " ," -> [" ,"]
 *
 * This implementation provides a simplified version that handles common cases.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Character classification helpers */
static bool is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool is_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}

static bool is_whitespace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/* Check if this is the GPT-2 Ġ character (0xC4 0xA0) */
static bool is_gpt2_space(const char *s, int len) {
    return len >= 2 && (unsigned char)s[0] == 0xC4 && (unsigned char)s[1] == 0xA0;
}

/* Check if this is a GPT-2 encoded letter (regular ASCII letter in byte-level) */
static bool is_bpe_letter(const char *s, int len) {
    if (len == 1) {
        unsigned char c = (unsigned char)s[0];
        return is_letter(c);
    }
    return false;
}

/* Check if this is a GPT-2 encoded digit */
static bool is_bpe_digit(const char *s, int len) {
    if (len == 1) {
        unsigned char c = (unsigned char)s[0];
        return is_digit(c);
    }
    return false;
}

/* Pretokenizer chunk types */
typedef enum {
    CHUNK_WORD,        /* Letters (with optional leading space) */
    CHUNK_NUMBER,      /* Digits */
    CHUNK_WHITESPACE,  /* Whitespace (not attached to word) */
    CHUNK_OTHER        /* Punctuation, etc. */
} ChunkType;

/* A chunk from pretokenization */
typedef struct {
    const char *start;
    int len;
    ChunkType type;
} PretokChunk;

/* Check if character is a newline (Ċ = 0xC4 0x8A in byte-level BPE) */
static bool is_bpe_newline(const char *s, int len) {
    return len >= 2 && (unsigned char)s[0] == 0xC4 && (unsigned char)s[1] == 0x8A;
}

/* Check if this is a non-letter, non-digit, non-newline character that can prefix a word */
static bool is_word_prefix_char(const char *s, int len) {
    /* In GPT-2 regex: [^\r\n\p{L}\p{N}]? matches any char except newline, letter, digit */
    if (len == 1) {
        unsigned char c = (unsigned char)s[0];
        return !is_letter(c) && !is_digit(c) && c != '\n' && c != '\r';
    }
    /* Multi-byte: check if it's not a letter/digit (newline is Ċ which we handle separately) */
    if (is_gpt2_space(s, len)) return true;  /* Space can prefix */
    if (is_bpe_newline(s, len)) return false;  /* Newline cannot prefix */
    return true;  /* Other multi-byte chars can prefix */
}

/* Check if character is punctuation (not space, not letter, not digit) */
static bool is_bpe_punct(const char *s, int len) {
    if (len == 1) {
        unsigned char c = (unsigned char)s[0];
        return !is_letter(c) && !is_digit(c) && c != ' ' && c != '\t' && c != '\n' && c != '\r';
    }
    /* Multi-byte: not Ġ (space) or Ċ (newline) */
    if (is_gpt2_space(s, len)) return false;
    if (len >= 2 && (unsigned char)s[0] == 0xC4) {
        unsigned char c1 = (unsigned char)s[1];
        /* Ċ (newline), ĉ (tab), č (CR) etc. are not punctuation */
        if (c1 == 0x8A || c1 == 0x89 || c1 == 0x8D) return false;
    }
    return true;
}

/*
 * GPT-2 Pretokenizer
 *
 * Splits byte-level encoded text into chunks for independent BPE processing.
 *
 * The GPT-2 regex pattern (in order of matching):
 *   1. (?i:'s|'t|'re|'ve|'m|'ll|'d)      - Contractions
 *   2. [^\r\n\p{L}\p{N}]?\p{L}+          - Words with optional prefix
 *   3. \p{N}                              - Single digit
 *   4.  ?[^\s\p{L}\p{N}]+[\r\n]*         - Optional space + punctuation + newlines
 *   5. \s*[\r\n]+                         - Whitespace + newlines
 *   6. \s+(?!\S)                          - Trailing whitespace
 *   7. \s+                                - Whitespace
 *
 * Returns array of chunks (caller provides buffer).
 */
static int gpt2_pretokenize(const char *text, int text_len, PretokChunk *chunks, int max_chunks) {
    int num_chunks = 0;
    int pos = 0;

    while (pos < text_len && num_chunks < max_chunks) {
        int chunk_start = pos;
        int char_len = utf8_char_len((unsigned char)text[pos]);
        if (pos + char_len > text_len) char_len = text_len - pos;

        /* Pattern 2: [^\r\n\p{L}\p{N}]?\p{L}+ - Words with optional punctuation prefix */
        /* This pattern MUST be checked before pattern 4 (punctuation) */
        /* Check if we have: letter, or punctuation followed by letter */
        bool is_word = false;
        int word_start = pos;
        int prefix_len = 0;

        if (is_bpe_letter(text + pos, char_len)) {
            /* Word without prefix */
            is_word = true;
        } else if (is_bpe_punct(text + pos, char_len) && !is_gpt2_space(text + pos, text_len - pos)) {
            /* Check if punctuation is followed by a letter */
            int after = pos + char_len;
            if (after < text_len) {
                int next_len = utf8_char_len((unsigned char)text[after]);
                if (is_bpe_letter(text + after, next_len)) {
                    /* This is pattern 2: punctuation + letters */
                    is_word = true;
                    prefix_len = char_len;
                }
            }
        }

        if (is_word) {
            /* Collect the word (with optional prefix) */
            pos = word_start + prefix_len;  /* Skip prefix if any */
            while (pos < text_len) {
                int clen = utf8_char_len((unsigned char)text[pos]);
                if (is_bpe_letter(text + pos, clen)) {
                    pos += clen;
                } else {
                    break;
                }
            }
            chunks[num_chunks].start = text + chunk_start;
            chunks[num_chunks].len = pos - chunk_start;
            chunks[num_chunks].type = CHUNK_WORD;
            num_chunks++;
            continue;
        }

        /* Pattern 3: \p{N} - Single digit */
        if (is_bpe_digit(text + pos, char_len)) {
            pos += char_len;
            chunks[num_chunks].start = text + chunk_start;
            chunks[num_chunks].len = pos - chunk_start;
            chunks[num_chunks].type = CHUNK_NUMBER;
            num_chunks++;
            continue;
        }

        /* Pattern 4:  ?[^\s\p{L}\p{N}]+[\r\n]* - Optional space + punctuation + newlines */
        /* At this point, we know the current char is NOT followed by letters (checked above) */
        /* Check for space (Ġ) followed by punctuation, OR just punctuation */
        bool has_leading_space = is_gpt2_space(text + pos, text_len - pos);
        int punct_start = has_leading_space ? pos + 2 : pos;

        if (punct_start < text_len) {
            int pchar_len = utf8_char_len((unsigned char)text[punct_start]);
            if (is_bpe_punct(text + punct_start, pchar_len)) {
                /* This matches pattern 4: space? + punctuation + newlines? */
                if (has_leading_space) {
                    pos += 2;  /* Include the leading space */
                }
                /* Collect punctuation characters */
                /* Pattern 4 collects ALL consecutive punctuation, NOT stopping before letters */
                /* The key insight: if we have " (" before "int", the " (" is pattern 4, */
                /* and "(int" would be pattern 2, but pattern 4 matches first since " (" */
                /* is a complete match for pattern 4 (space + one punctuation). */
                while (pos < text_len) {
                    int clen = utf8_char_len((unsigned char)text[pos]);
                    if (is_bpe_punct(text + pos, clen)) {
                        pos += clen;
                    } else {
                        break;
                    }
                }
                /* Include trailing newlines */
                while (pos < text_len && is_bpe_newline(text + pos, text_len - pos)) {
                    pos += 2;
                }
                chunks[num_chunks].start = text + chunk_start;
                chunks[num_chunks].len = pos - chunk_start;
                chunks[num_chunks].type = CHUNK_OTHER;
                num_chunks++;
                continue;
            }
        }

        /* Pattern 5/6/7: Whitespace handling */
        if (is_gpt2_space(text + pos, text_len - pos)) {
            /* Count consecutive spaces */
            int space_count = 0;
            int space_end = pos;
            while (space_end < text_len && is_gpt2_space(text + space_end, text_len - space_end)) {
                space_count++;
                space_end += 2;
            }

            /* Check if spaces are followed by newlines (pattern 5) */
            if (space_end < text_len && is_bpe_newline(text + space_end, text_len - space_end)) {
                /* Whitespace + newlines */
                while (space_end < text_len && is_bpe_newline(text + space_end, text_len - space_end)) {
                    space_end += 2;
                }
                chunks[num_chunks].start = text + pos;
                chunks[num_chunks].len = space_end - pos;
                chunks[num_chunks].type = CHUNK_WHITESPACE;
                num_chunks++;
                pos = space_end;
                continue;
            }

            /* Check what follows the spaces */
            if (space_end < text_len) {
                int next_len = utf8_char_len((unsigned char)text[space_end]);
                if (is_bpe_letter(text + space_end, next_len)) {
                    /* Letters follow - output (n-1) spaces, then space+word (pattern 2) */
                    if (space_count > 1) {
                        chunks[num_chunks].start = text + pos;
                        chunks[num_chunks].len = (space_count - 1) * 2;
                        chunks[num_chunks].type = CHUNK_WHITESPACE;
                        num_chunks++;
                        pos += (space_count - 1) * 2;
                        if (num_chunks >= max_chunks) break;
                    }
                    /* Collect space + word */
                    chunk_start = pos;
                    pos += 2;  /* Skip the Ġ */
                    while (pos < text_len) {
                        int clen = utf8_char_len((unsigned char)text[pos]);
                        if (is_bpe_letter(text + pos, clen)) {
                            pos += clen;
                        } else {
                            break;
                        }
                    }
                    chunks[num_chunks].start = text + chunk_start;
                    chunks[num_chunks].len = pos - chunk_start;
                    chunks[num_chunks].type = CHUNK_WORD;
                    num_chunks++;
                    continue;
                } else if (is_bpe_digit(text + space_end, next_len)) {
                    /* Digit follows - output (n-1) spaces, then space+digit */
                    if (space_count > 1) {
                        chunks[num_chunks].start = text + pos;
                        chunks[num_chunks].len = (space_count - 1) * 2;
                        chunks[num_chunks].type = CHUNK_WHITESPACE;
                        num_chunks++;
                        pos += (space_count - 1) * 2;
                        if (num_chunks >= max_chunks) break;
                    }
                    /* Space + single digit */
                    chunk_start = pos;
                    pos += 2 + 1;  /* Ġ + digit */
                    chunks[num_chunks].start = text + chunk_start;
                    chunks[num_chunks].len = pos - chunk_start;
                    chunks[num_chunks].type = CHUNK_NUMBER;
                    num_chunks++;
                    continue;
                }
                /* Pattern 4 would have caught space+punct, so this is trailing space before something else */
            }

            /* Just whitespace */
            chunks[num_chunks].start = text + pos;
            chunks[num_chunks].len = space_count * 2;
            chunks[num_chunks].type = CHUNK_WHITESPACE;
            num_chunks++;
            pos = space_end;
            continue;
        }

        /* Pattern 5: Newlines (Ċ) */
        if (is_bpe_newline(text + pos, text_len - pos)) {
            while (pos < text_len && is_bpe_newline(text + pos, text_len - pos)) {
                pos += 2;
            }
            chunks[num_chunks].start = text + chunk_start;
            chunks[num_chunks].len = pos - chunk_start;
            chunks[num_chunks].type = CHUNK_OTHER;
            num_chunks++;
            continue;
        }

        /* Fallback: single character chunk */
        pos += char_len;
        chunks[num_chunks].start = text + chunk_start;
        chunks[num_chunks].len = pos - chunk_start;
        chunks[num_chunks].type = CHUNK_OTHER;
        num_chunks++;
    }

    return num_chunks;
}

/* Initialize token list from preprocessed text */
static int init_tokens_from_text(CKTrueBPE *bpe, CKBPETokenList *list, const char *text, int text_len) {
    token_list_clear(list);

    int pos = 0;
    while (pos < text_len) {
        int char_len = utf8_char_len((unsigned char)text[pos]);
        if (pos + char_len > text_len) {
            char_len = text_len - pos;  /* Truncated UTF-8 */
        }

        /* Look up this character/byte in vocabulary */
        char char_buf[8];
        memcpy(char_buf, text + pos, char_len);
        char_buf[char_len] = '\0';

        int32_t id = ck_true_bpe_lookup(bpe, char_buf);

        if (token_list_append(list, char_buf, char_len, id) != 0) {
            return -1;
        }

        pos += char_len;
    }

    return 0;
}

/* Find the best (highest priority = lowest number) merge in the token list */
static int find_best_merge(const CKTrueBPE *bpe, const CKBPETokenList *list,
                           size_t *best_pos, const CKBPEMerge **best_merge) {
    *best_pos = 0;
    *best_merge = NULL;
    int32_t best_priority = INT32_MAX;

    for (size_t i = 0; i + 1 < list->count; i++) {
        int32_t left_id = list->tokens[i].id;
        int32_t right_id = list->tokens[i + 1].id;

        if (left_id < 0 || right_id < 0) continue;  /* Unknown tokens can't merge */

        const CKBPEMerge *merge = merge_table_lookup(bpe->merges, left_id, right_id);
        if (merge && merge->priority < best_priority) {
            best_priority = merge->priority;
            *best_pos = i;
            *best_merge = merge;
        }
    }

    return (*best_merge != NULL) ? 0 : -1;
}

/* Apply BPE merges until no more possible */
static int apply_bpe_merges(CKTrueBPE *bpe, CKBPETokenList *list) {
    char merged_buf[MAX_TOKEN_LEN * 2];

    while (list->count > 1) {
        size_t best_pos;
        const CKBPEMerge *best_merge;

        if (find_best_merge(bpe, list, &best_pos, &best_merge) != 0) {
            break;  /* No more merges possible */
        }

        /* Get merged token string */
        const char *merged_str = bpe->id_to_token[best_merge->merged_id];
        if (!merged_str) {
            /* Construct from left + right */
            size_t left_len = list->tokens[best_pos].len;
            size_t right_len = list->tokens[best_pos + 1].len;

            if (left_len + right_len >= sizeof(merged_buf)) {
                break;  /* Too long */
            }

            memcpy(merged_buf, list->tokens[best_pos].str, left_len);
            memcpy(merged_buf + left_len, list->tokens[best_pos + 1].str, right_len);
            merged_buf[left_len + right_len] = '\0';
            merged_str = merged_buf;
        }

        /* Apply the merge */
        if (token_list_merge_at(list, best_pos, merged_str, strlen(merged_str), best_merge->merged_id) != 0) {
            break;
        }
    }

    return 0;
}

/*
 * Encode a single chunk (after pretokenization) using BPE
 */
static int encode_chunk(CKTrueBPE *bpe, const char *chunk, int chunk_len,
                        int32_t *ids, int max_ids, CKBPETokenList *list) {
    if (chunk_len <= 0) return 0;

    /* First, try to look up the entire chunk as a single token */
    char chunk_buf[256];
    if (chunk_len < (int)sizeof(chunk_buf)) {
        memcpy(chunk_buf, chunk, chunk_len);
        chunk_buf[chunk_len] = '\0';
        int32_t chunk_id = ck_true_bpe_lookup(bpe, chunk_buf);
        if (chunk_id >= 0) {
            /* Entire chunk is a single token */
            if (max_ids >= 1) {
                ids[0] = chunk_id;
                return 1;
            }
            return 0;
        }
    }

    /* Initialize token list from chunk characters */
    if (init_tokens_from_text(bpe, list, chunk, chunk_len) != 0) {
        return 0;
    }

    /* Apply BPE merges to this chunk */
    apply_bpe_merges(bpe, list);

    /* Extract token IDs from this chunk */
    int out_idx = 0;
    for (size_t i = 0; i < list->count && out_idx < max_ids; i++) {
        int32_t id = list->tokens[i].id;

        /* Handle unknown tokens */
        if (id < 0) {
            if (bpe->config.byte_fallback) {
                /* Output each byte as separate token (byte fallback) */
                for (size_t j = 0; j < list->tokens[i].len && out_idx < max_ids; j++) {
                    char byte_token[8];
                    snprintf(byte_token, sizeof(byte_token), "<0x%02X>", (unsigned char)list->tokens[i].str[j]);
                    int32_t byte_id = ck_true_bpe_lookup(bpe, byte_token);
                    ids[out_idx++] = (byte_id >= 0) ? byte_id : bpe->unk_id;
                }
            } else {
                ids[out_idx++] = bpe->unk_id;
            }
        } else {
            ids[out_idx++] = id;
        }
    }

    return out_idx;
}

int ck_true_bpe_encode(CKTrueBPE *bpe, const char *text, int text_len, int32_t *ids, int max_ids) {
    if (!bpe || !text || !ids || max_ids <= 0) return 0;
    if (text_len < 0) text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    /* Auto-detect space style if needed */
    if (bpe->config.space_prefix_style == CK_SPACE_PREFIX_AUTO) {
        ck_true_bpe_detect_space_style(bpe);
    }

    /* Preprocess text (byte-level encoding) */
    char preprocessed[16384];
    int pp_len = preprocess_text(bpe, text, text_len, preprocessed, sizeof(preprocessed) - 1);
    if (pp_len < 0) {
        return 0;  /* Preprocessing failed */
    }
    preprocessed[pp_len] = '\0';

    int out_idx = 0;

    /* Add BOS token if configured */
    if (bpe->config.add_bos && bpe->bos_id >= 0 && out_idx < max_ids) {
        ids[out_idx++] = bpe->bos_id;
    }

    /* For GPT-2 style, use pretokenizer to split into chunks */
    if (bpe->config.space_prefix_style == CK_SPACE_PREFIX_GPT2 ||
        bpe->config.space_prefix_style == CK_SPACE_PREFIX_AUTO) {

        /* Pretokenize */
        PretokChunk chunks[1024];
        int num_chunks = gpt2_pretokenize(preprocessed, pp_len, chunks, 1024);

        /* Create reusable token list */
        CKBPETokenList *list = token_list_create(INITIAL_TOKEN_CAPACITY);
        if (!list) return out_idx;

        /* Process each chunk independently with BPE */
        for (int c = 0; c < num_chunks && out_idx < max_ids; c++) {
            int chunk_ids = encode_chunk(bpe, chunks[c].start, chunks[c].len,
                                         ids + out_idx, max_ids - out_idx, list);
            out_idx += chunk_ids;
        }

        token_list_free(list);
    } else {
        /* SentencePiece style: no pretokenization, process entire text */
        CKBPETokenList *list = token_list_create(INITIAL_TOKEN_CAPACITY);
        if (!list) return out_idx;

        int chunk_ids = encode_chunk(bpe, preprocessed, pp_len,
                                     ids + out_idx, max_ids - out_idx, list);
        out_idx += chunk_ids;

        token_list_free(list);
    }

    /* Add EOS token if configured */
    if (bpe->config.add_eos && bpe->eos_id >= 0 && out_idx < max_ids) {
        ids[out_idx++] = bpe->eos_id;
    }

    return out_idx;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Decoding
 * ═══════════════════════════════════════════════════════════════════════════════ */

int ck_true_bpe_decode(const CKTrueBPE *bpe, const int32_t *ids, int num_ids, char *text, int max_len) {
    if (!bpe || !ids || !text || max_len <= 0) return 0;

    int len = 0;
    for (int i = 0; i < num_ids && len < max_len - 1; i++) {
        int32_t id = ids[i];
        if (id < 0) continue;

        /* Skip special tokens */
        if (id == bpe->bos_id || id == bpe->eos_id || id == bpe->pad_id) {
            continue;
        }

        const char *token = ck_true_bpe_id_to_token(bpe, id);
        if (!token) continue;

        int token_len = (int)strlen(token);
        unsigned char c0 = (unsigned char)token[0];
        unsigned char c1 = (unsigned char)token[1];

        /* Convert space prefix markers back to space */
        if (c0 == 0xC4 && c1 == 0xA0) {
            /* Ġ -> space */
            if (len < max_len - 1) text[len++] = ' ';
            token += 2;
            token_len -= 2;
        } else if (c0 == 0xE2 && c1 == 0x96 && (unsigned char)token[2] == 0x81) {
            /* ▁ -> space */
            if (len < max_len - 1) text[len++] = ' ';
            token += 3;
            token_len -= 3;
        }

        /* Copy rest of token */
        for (int j = 0; j < token_len && len < max_len - 1; j++) {
            text[len++] = token[j];
        }
    }

    text[len] = '\0';
    return len;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Statistics
 * ═══════════════════════════════════════════════════════════════════════════════ */

size_t ck_true_bpe_vocab_size(const CKTrueBPE *bpe) {
    return bpe ? bpe->vocab_size : 0;
}

int32_t ck_true_bpe_num_merges(const CKTrueBPE *bpe) {
    return bpe ? bpe->num_merges : 0;
}
