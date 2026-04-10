/*
 * C-Kernel-Engine BPE Tokenizer Implementation  [v4 - correctness complete, optimization pending]
 *
 * Pure C implementation of Byte-Pair Encoding.
 * Reads HuggingFace tokenizer.json format.
 *
 * Token IDs are dense indices into the embedding table:
 *   embedding[token_id] gives the vector for that token.
 *
 * TODO: Performance optimizations (correctness is done, these are for throughput):
 *
 *   1. Pre-allocate encode buffer
 *      - ck_tokenizer_encode() mallocs int32_t[] per call (line ~629)
 *      - Add encode_buf + encode_buf_cap to CKTokenizer struct
 *      - Allocate in ck_tokenizer_init(), realloc only if input exceeds cap
 *      - Eliminates malloc/free syscall per encode call
 *
 *   2. Linked list instead of array shift in merge loop
 *      - Merge at line ~686 shifts entire tail of tokens[] left by 1: O(n) per merge
 *      - With n initial tokens and m merges, total shifting is O(n * m)
 *      - Doubly-linked list would make each merge O(1)
 *
 *   3. Priority queue for best-merge scan
 *      - find_best_merge scans ALL pairs each iteration (line ~673): O(n) per merge
 *      - Min-heap of valid merge candidates: O(n log n) total instead of O(n * m)
 *
 *   Current profile: tokenizer runs 2x per prompt (not per token). GEMM kernels
 *   dominate at 57% of compute. These optimizations matter for batch/server use.
 *
 * By Anthony Shivakumar
 */

#include "ck_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Simple JSON parser state */
typedef struct {
    const char *data;
    const char *pos;
    const char *end;
} JSONParser;

/* ========================================================================== */
/* Memory Pool                                                                 */
/* ========================================================================== */

void ck_pool_init(CKMemPool *pool) {
    memset(pool, 0, sizeof(*pool));
}

static CKPoolBlock *pool_new_block(size_t capacity) {
    CKPoolBlock *block = (CKPoolBlock *)malloc(sizeof(CKPoolBlock));
    if (!block) return NULL;
    block->data = (uint8_t *)malloc(capacity);
    if (!block->data) {
        free(block);
        return NULL;
    }
    block->used = 0;
    block->capacity = capacity;
    block->next = NULL;
    return block;
}

void *ck_pool_alloc(CKMemPool *pool, size_t size) {
    /* Align to 8 bytes */
    size = (size + 7) & ~7;

    /* Check if current block has space */
    if (pool->current && pool->current->used + size <= pool->current->capacity) {
        void *ptr = pool->current->data + pool->current->used;
        pool->current->used += size;
        pool->total_allocated += size;
        return ptr;
    }

    /* Need new block */
    size_t block_size = CK_POOL_BLOCK_SIZE;
    if (size > block_size) block_size = size;

    CKPoolBlock *block = pool_new_block(block_size);
    if (!block) return NULL;

    block->next = pool->head;
    pool->head = block;
    pool->current = block;

    void *ptr = block->data;
    block->used = size;
    pool->total_allocated += size;
    return ptr;
}

char *ck_pool_strdup(CKMemPool *pool, const char *s, int len) {
    if (len < 0) len = (int)strlen(s);
    char *copy = (char *)ck_pool_alloc(pool, len + 1);
    if (!copy) return NULL;
    memcpy(copy, s, len);
    copy[len] = '\0';
    return copy;
}

void ck_pool_free(CKMemPool *pool) {
    CKPoolBlock *block = pool->head;
    while (block) {
        CKPoolBlock *next = block->next;
        free(block->data);
        free(block);
        block = next;
    }
    memset(pool, 0, sizeof(*pool));
}

/* ========================================================================== */
/* Hash Functions                                                              */
/* ========================================================================== */

/* FNV-1a hash for strings */
static uint32_t hash_string(const char *s, int len) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= (uint8_t)s[i];
        hash *= 16777619u;
    }
    return hash;
}

/* Hash for merge pair (left_id, right_id) */
static uint32_t hash_pair(int32_t left, int32_t right) {
    uint64_t combined = ((uint64_t)left << 32) | (uint32_t)right;
    /* MurmurHash3 finalizer */
    combined ^= combined >> 33;
    combined *= 0xff51afd7ed558ccdULL;
    combined ^= combined >> 33;
    combined *= 0xc4ceb9fe1a85ec53ULL;
    combined ^= combined >> 33;
    return (uint32_t)combined;
}

/* ========================================================================== */
/* Tokenizer Init/Free                                                         */
/* ========================================================================== */

int ck_tokenizer_init(CKTokenizer *tok) {
    memset(tok, 0, sizeof(*tok));
    ck_pool_init(&tok->pool);

    /* Default special tokens */
    tok->unk_id = 0;
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->pad_id = 3;

    /* Allocate vocab hash table */
    tok->vocab_hash_size = 65536;  /* 64K buckets */
    tok->vocab_hash = (CKVocabEntry **)calloc(tok->vocab_hash_size, sizeof(CKVocabEntry *));
    if (!tok->vocab_hash) return -1;

    /* Allocate reverse vocab */
    tok->id_to_token = (char **)calloc(CK_MAX_VOCAB_SIZE, sizeof(char *));
    if (!tok->id_to_token) {
        free(tok->vocab_hash);
        return -1;
    }

    /* Allocate merge hash table */
    tok->merge_hash_size = 262144;  /* 256K buckets */
    tok->merge_hash = (int *)malloc(tok->merge_hash_size * sizeof(int));
    if (!tok->merge_hash) {
        free(tok->vocab_hash);
        free(tok->id_to_token);
        return -1;
    }
    memset(tok->merge_hash, -1, tok->merge_hash_size * sizeof(int));

    return 0;
}

void ck_tokenizer_free(CKTokenizer *tok) {
    ck_pool_free(&tok->pool);
    free(tok->vocab_hash);
    free(tok->id_to_token);
    free(tok->merges);
    free(tok->merge_hash);
    memset(tok, 0, sizeof(*tok));
}

/* ========================================================================== */
/* Vocabulary Operations                                                       */
/* ========================================================================== */

int32_t ck_tokenizer_add_token(CKTokenizer *tok, const char *token, int len) {
    if (len < 0) len = (int)strlen(token);
    if (tok->vocab_size >= CK_MAX_VOCAB_SIZE) return -1;

    /* Check if already exists */
    int32_t existing = ck_tokenizer_lookup(tok, token, len);
    if (existing != tok->unk_id || (len == 0)) {
        return existing;
    }

    /* Create new entry */
    CKVocabEntry *entry = (CKVocabEntry *)ck_pool_alloc(&tok->pool, sizeof(CKVocabEntry));
    if (!entry) return -1;

    entry->token = ck_pool_strdup(&tok->pool, token, len);
    if (!entry->token) return -1;
    entry->token_len = len;
    entry->id = tok->vocab_size;

    /* Add to hash table */
    uint32_t bucket = hash_string(token, len) % tok->vocab_hash_size;
    entry->next = tok->vocab_hash[bucket];
    tok->vocab_hash[bucket] = entry;

    /* Add to reverse lookup */
    tok->id_to_token[tok->vocab_size] = entry->token;

    tok->vocab_size++;
    return entry->id;
}

int32_t ck_tokenizer_lookup(const CKTokenizer *tok, const char *token, int len) {
    if (len < 0) len = (int)strlen(token);
    uint32_t bucket = hash_string(token, len) % tok->vocab_hash_size;

    for (CKVocabEntry *e = tok->vocab_hash[bucket]; e; e = e->next) {
        if (e->token_len == len && memcmp(e->token, token, len) == 0) {
            return e->id;
        }
    }
    return tok->unk_id;
}

const char *ck_tokenizer_id_to_token(const CKTokenizer *tok, int32_t id) {
    if (id < 0 || id >= tok->vocab_size) return NULL;
    return tok->id_to_token[id];
}

/* ========================================================================== */
/* Merge Operations                                                            */
/* ========================================================================== */

int ck_tokenizer_add_merge(CKTokenizer *tok, int32_t left, int32_t right, int32_t merged) {
    int idx = tok->num_merges;

    /* Grow merges array if needed */
    if (idx % 4096 == 0) {
        size_t new_cap = (idx + 4096) * sizeof(CKMergeRule);
        CKMergeRule *new_merges = (CKMergeRule *)realloc(tok->merges, new_cap);
        if (!new_merges) return -1;
        tok->merges = new_merges;
    }

    tok->merges[idx].left = left;
    tok->merges[idx].right = right;
    tok->merges[idx].merged = merged;
    tok->merges[idx].priority = idx;  /* Earlier = higher priority */

    /* Add to hash table */
    uint32_t bucket = hash_pair(left, right) % tok->merge_hash_size;
    /* Linear probing */
    while (tok->merge_hash[bucket] >= 0) {
        bucket = (bucket + 1) % tok->merge_hash_size;
    }
    tok->merge_hash[bucket] = idx;

    tok->num_merges++;
    return 0;
}

int ck_tokenizer_lookup_merge(const CKTokenizer *tok, int32_t left, int32_t right) {
    uint32_t bucket = hash_pair(left, right) % tok->merge_hash_size;

    /* Linear probing */
    int probes = 0;
    while (tok->merge_hash[bucket] >= 0 && probes < tok->merge_hash_size) {
        int idx = tok->merge_hash[bucket];
        if (tok->merges[idx].left == left && tok->merges[idx].right == right) {
            return idx;
        }
        bucket = (bucket + 1) % tok->merge_hash_size;
        probes++;
    }
    return -1;
}

/* ========================================================================== */
/* JSON Parser (minimal, just for tokenizer.json)                              */
/* ========================================================================== */

static void json_skip_whitespace(JSONParser *p) {
    while (p->pos < p->end && isspace((unsigned char)*p->pos)) {
        p->pos++;
    }
}

static int json_match_char(JSONParser *p, char c) {
    json_skip_whitespace(p);
    if (p->pos < p->end && *p->pos == c) {
        p->pos++;
        return 1;
    }
    return 0;
}

static int json_parse_string(JSONParser *p, char *buf, int max_len) {
    json_skip_whitespace(p);
    if (p->pos >= p->end || *p->pos != '"') return -1;
    p->pos++;

    int len = 0;
    while (p->pos < p->end && *p->pos != '"') {
        char c = *p->pos++;
        if (c == '\\' && p->pos < p->end) {
            c = *p->pos++;
            switch (c) {
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case '\\': c = '\\'; break;
                case '"': c = '"'; break;
                case 'u': {
                    /* Unicode escape \uXXXX */
                    if (p->pos + 4 <= p->end) {
                        char hex[5] = {p->pos[0], p->pos[1], p->pos[2], p->pos[3], 0};
                        unsigned int codepoint = (unsigned int)strtol(hex, NULL, 16);
                        p->pos += 4;
                        /* Convert to UTF-8 */
                        if (codepoint < 0x80) {
                            if (len < max_len - 1) buf[len++] = (char)codepoint;
                        } else if (codepoint < 0x800) {
                            if (len < max_len - 2) {
                                buf[len++] = (char)(0xC0 | (codepoint >> 6));
                                buf[len++] = (char)(0x80 | (codepoint & 0x3F));
                            }
                        } else {
                            if (len < max_len - 3) {
                                buf[len++] = (char)(0xE0 | (codepoint >> 12));
                                buf[len++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
                                buf[len++] = (char)(0x80 | (codepoint & 0x3F));
                            }
                        }
                        continue;
                    }
                    break;
                }
                default: break;
            }
        }
        if (len < max_len - 1) buf[len++] = c;
    }
    buf[len] = '\0';

    if (p->pos < p->end && *p->pos == '"') p->pos++;
    return len;
}

static int json_parse_int(JSONParser *p, int *out) {
    json_skip_whitespace(p);
    if (p->pos >= p->end) return -1;

    int neg = 0;
    if (*p->pos == '-') {
        neg = 1;
        p->pos++;
    }

    if (p->pos >= p->end || !isdigit((unsigned char)*p->pos)) return -1;

    int val = 0;
    while (p->pos < p->end && isdigit((unsigned char)*p->pos)) {
        val = val * 10 + (*p->pos - '0');
        p->pos++;
    }

    *out = neg ? -val : val;
    return 0;
}

static void json_skip_value(JSONParser *p) {
    json_skip_whitespace(p);
    if (p->pos >= p->end) return;

    char c = *p->pos;
    if (c == '"') {
        char buf[1024];
        json_parse_string(p, buf, sizeof(buf));
    } else if (c == '{') {
        int depth = 1;
        p->pos++;
        while (p->pos < p->end && depth > 0) {
            if (*p->pos == '{') depth++;
            else if (*p->pos == '}') depth--;
            else if (*p->pos == '"') {
                char buf[1024];
                json_parse_string(p, buf, sizeof(buf));
                continue;
            }
            p->pos++;
        }
    } else if (c == '[') {
        int depth = 1;
        p->pos++;
        while (p->pos < p->end && depth > 0) {
            if (*p->pos == '[') depth++;
            else if (*p->pos == ']') depth--;
            else if (*p->pos == '"') {
                char buf[1024];
                json_parse_string(p, buf, sizeof(buf));
                continue;
            }
            p->pos++;
        }
    } else {
        /* Number, bool, null */
        while (p->pos < p->end && !isspace((unsigned char)*p->pos) &&
               *p->pos != ',' && *p->pos != '}' && *p->pos != ']') {
            p->pos++;
        }
    }
}

/* ========================================================================== */
/* Load from tokenizer.json                                                    */
/* ========================================================================== */

int ck_tokenizer_load(CKTokenizer *tok, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open tokenizer: %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *data = (char *)malloc(size + 1);
    if (!data) {
        fclose(f);
        return -1;
    }
    fread(data, 1, size, f);
    data[size] = '\0';
    fclose(f);

    JSONParser parser = {data, data, data + size};
    JSONParser *p = &parser;

    /* Parse top-level object */
    if (!json_match_char(p, '{')) {
        free(data);
        return -1;
    }

    char key[256];
    while (p->pos < p->end && *p->pos != '}') {
        if (json_parse_string(p, key, sizeof(key)) < 0) break;
        if (!json_match_char(p, ':')) break;

        if (strcmp(key, "model") == 0) {
            /* Parse model object */
            if (!json_match_char(p, '{')) {
                json_skip_value(p);
                json_match_char(p, ',');
                continue;
            }

            while (p->pos < p->end && *p->pos != '}') {
                if (json_parse_string(p, key, sizeof(key)) < 0) break;
                if (!json_match_char(p, ':')) break;

                if (strcmp(key, "vocab") == 0) {
                    /* Parse vocab object: {"token": id, ...} */
                    if (!json_match_char(p, '{')) {
                        json_skip_value(p);
                        json_match_char(p, ',');
                        continue;
                    }

                    char token[CK_MAX_TOKEN_LEN];
                    while (p->pos < p->end && *p->pos != '}') {
                        int token_len = json_parse_string(p, token, sizeof(token));
                        if (token_len < 0) break;
                        if (!json_match_char(p, ':')) break;

                        int id;
                        if (json_parse_int(p, &id) < 0) break;

                        /* Ensure we have space up to this ID */
                        while (tok->vocab_size <= id) {
                            ck_tokenizer_add_token(tok, "", 0);
                        }

                        /* Add/update token */
                        uint32_t bucket = hash_string(token, token_len) % tok->vocab_hash_size;
                        CKVocabEntry *entry = (CKVocabEntry *)ck_pool_alloc(&tok->pool, sizeof(CKVocabEntry));
                        entry->token = ck_pool_strdup(&tok->pool, token, token_len);
                        entry->token_len = token_len;
                        entry->id = id;
                        entry->next = tok->vocab_hash[bucket];
                        tok->vocab_hash[bucket] = entry;
                        tok->id_to_token[id] = entry->token;
                        if (id >= tok->vocab_size) tok->vocab_size = id + 1;

                        json_match_char(p, ',');
                    }
                    json_match_char(p, '}');

                } else if (strcmp(key, "merges") == 0) {
                    /* Parse merges array: ["tok1 tok2", ...] */
                    if (!json_match_char(p, '[')) {
                        json_skip_value(p);
                        json_match_char(p, ',');
                        continue;
                    }

                    char merge_str[512];
                    while (p->pos < p->end && *p->pos != ']') {
                        int merge_len = json_parse_string(p, merge_str, sizeof(merge_str));
                        if (merge_len < 0) break;

                        /* Parse "token1 token2" */
                        char *space = strchr(merge_str, ' ');
                        if (space) {
                            *space = '\0';
                            char *tok1 = merge_str;
                            char *tok2 = space + 1;

                            int32_t id1 = ck_tokenizer_lookup(tok, tok1, -1);
                            int32_t id2 = ck_tokenizer_lookup(tok, tok2, -1);

                            /* Create merged token */
                            char merged[512];
                            snprintf(merged, sizeof(merged), "%s%s", tok1, tok2);
                            int32_t merged_id = ck_tokenizer_lookup(tok, merged, -1);

                            if (merged_id == tok->unk_id) {
                                merged_id = ck_tokenizer_add_token(tok, merged, -1);
                            }

                            ck_tokenizer_add_merge(tok, id1, id2, merged_id);
                        }

                        json_match_char(p, ',');
                    }
                    json_match_char(p, ']');

                } else {
                    json_skip_value(p);
                }

                json_match_char(p, ',');
            }
            json_match_char(p, '}');

        } else if (strcmp(key, "added_tokens") == 0) {
            /* Parse added_tokens array for special tokens */
            if (!json_match_char(p, '[')) {
                json_skip_value(p);
                json_match_char(p, ',');
                continue;
            }

            while (p->pos < p->end && *p->pos != ']') {
                if (!json_match_char(p, '{')) {
                    json_skip_value(p);
                    json_match_char(p, ',');
                    continue;
                }

                char content[256] = "";
                int id = -1;
                while (p->pos < p->end && *p->pos != '}') {
                    if (json_parse_string(p, key, sizeof(key)) < 0) break;
                    if (!json_match_char(p, ':')) break;

                    if (strcmp(key, "content") == 0) {
                        json_parse_string(p, content, sizeof(content));
                    } else if (strcmp(key, "id") == 0) {
                        json_parse_int(p, &id);
                    } else if (strcmp(key, "special") == 0) {
                        json_skip_whitespace(p);
                        json_skip_value(p);
                    } else {
                        json_skip_value(p);
                    }
                    json_match_char(p, ',');
                }
                json_match_char(p, '}');

                if (id >= 0 && content[0]) {
                    /* Identify special tokens */
                    if (strcmp(content, "<unk>") == 0 || strcmp(content, "[UNK]") == 0) {
                        tok->unk_id = id;
                    } else if (strcmp(content, "<s>") == 0 || strcmp(content, "<bos>") == 0 ||
                               strcmp(content, "[BOS]") == 0) {
                        tok->bos_id = id;
                    } else if (strcmp(content, "</s>") == 0 || strcmp(content, "<eos>") == 0 ||
                               strcmp(content, "[EOS]") == 0 || strcmp(content, "<|endoftext|>") == 0) {
                        tok->eos_id = id;
                    } else if (strcmp(content, "<pad>") == 0 || strcmp(content, "[PAD]") == 0) {
                        tok->pad_id = id;
                    }
                }

                json_match_char(p, ',');
            }
            json_match_char(p, ']');

        } else {
            json_skip_value(p);
        }

        json_match_char(p, ',');
    }

    free(data);

    printf("Loaded tokenizer: %d tokens, %d merges\n", tok->vocab_size, tok->num_merges);
    printf("  UNK=%d BOS=%d EOS=%d PAD=%d\n", tok->unk_id, tok->bos_id, tok->eos_id, tok->pad_id);

    return 0;
}

/* ========================================================================== */
/* BPE Encode                                                                  */
/* ========================================================================== */

int ck_tokenizer_encode(const CKTokenizer *tok,
                        const char *text,
                        int text_len,
                        int32_t *ids,
                        int max_ids) {
    if (text_len < 0) text_len = (int)strlen(text);

    /* Pre-tokenize: split on whitespace, keep spaces as tokens */
    /* For simplicity, treat each byte as initial token, then apply BPE */

    /* Initial tokens: one per byte */
    int32_t *tokens = (int32_t *)malloc(text_len * sizeof(int32_t));
    int num_tokens = 0;

    for (int i = 0; i < text_len; i++) {
        /* Look up single-character token */
        char c[2] = {text[i], '\0'};
        int32_t id = ck_tokenizer_lookup(tok, c, 1);

        /* Handle special byte tokens like <0xXX> */
        if (id == tok->unk_id) {
            char byte_token[8];
            snprintf(byte_token, sizeof(byte_token), "<0x%02X>", (unsigned char)text[i]);
            id = ck_tokenizer_lookup(tok, byte_token, -1);
        }

        /* Try UTF-8 multi-byte sequences */
        if (id == tok->unk_id && (unsigned char)text[i] >= 0x80) {
            int utf8_len = 1;
            if ((text[i] & 0xE0) == 0xC0) utf8_len = 2;
            else if ((text[i] & 0xF0) == 0xE0) utf8_len = 3;
            else if ((text[i] & 0xF8) == 0xF0) utf8_len = 4;

            if (i + utf8_len <= text_len) {
                id = ck_tokenizer_lookup(tok, text + i, utf8_len);
                if (id != tok->unk_id) {
                    tokens[num_tokens++] = id;
                    i += utf8_len - 1;
                    continue;
                }
            }
        }

        tokens[num_tokens++] = id;
    }

    /* Apply BPE merges iteratively */
    bool changed = true;
    while (changed && num_tokens > 1) {
        changed = false;

        /* Find best merge (lowest priority = earliest in merge list) */
        int best_pos = -1;
        int best_priority = tok->num_merges;

        for (int i = 0; i < num_tokens - 1; i++) {
            int merge_idx = ck_tokenizer_lookup_merge(tok, tokens[i], tokens[i + 1]);
            if (merge_idx >= 0 && tok->merges[merge_idx].priority < best_priority) {
                best_pos = i;
                best_priority = tok->merges[merge_idx].priority;
            }
        }

        if (best_pos >= 0) {
            int merge_idx = ck_tokenizer_lookup_merge(tok, tokens[best_pos], tokens[best_pos + 1]);
            tokens[best_pos] = tok->merges[merge_idx].merged;

            /* Shift remaining tokens */
            for (int i = best_pos + 1; i < num_tokens - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            num_tokens--;
            changed = true;
        }
    }

    /* Copy to output */
    int out_len = 0;

    if (tok->add_bos && out_len < max_ids) {
        ids[out_len++] = tok->bos_id;
    }

    for (int i = 0; i < num_tokens && out_len < max_ids; i++) {
        ids[out_len++] = tokens[i];
    }

    if (tok->add_eos && out_len < max_ids) {
        ids[out_len++] = tok->eos_id;
    }

    free(tokens);
    return out_len;
}

/* ========================================================================== */
/* Decode                                                                      */
/* ========================================================================== */

int ck_tokenizer_decode(const CKTokenizer *tok,
                        const int32_t *ids,
                        int num_ids,
                        char *text,
                        int max_len) {
    int len = 0;

    for (int i = 0; i < num_ids; i++) {
        /* Skip special tokens */
        if (ids[i] == tok->bos_id || ids[i] == tok->eos_id || ids[i] == tok->pad_id) {
            continue;
        }

        const char *token = ck_tokenizer_id_to_token(tok, ids[i]);
        if (!token) continue;

        int token_len = (int)strlen(token);

        /* Handle byte tokens <0xXX> */
        if (token_len == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x') {
            char hex[3] = {token[3], token[4], 0};
            unsigned int byte = (unsigned int)strtol(hex, NULL, 16);
            if (len < max_len - 1) {
                text[len++] = (char)byte;
            }
            continue;
        }

        /* Handle GPT-style space prefix (Ġ = 0xC4 0xA0 in UTF-8) */
        const char *src = token;
        if ((unsigned char)token[0] == 0xC4 && (unsigned char)token[1] == 0xA0) {
            if (len < max_len - 1) {
                text[len++] = ' ';
            }
            src = token + 2;
            token_len -= 2;
        }

        /* Copy token */
        for (int j = 0; j < token_len && len < max_len - 1; j++) {
            text[len++] = src[j];
        }
    }

    text[len] = '\0';
    return len;
}
