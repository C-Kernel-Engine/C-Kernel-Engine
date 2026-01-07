/*
 * C-Kernel-Engine Greedy Tokenizer
 *
 * High-performance tokenizer with:
 * - Greedy longest-match encoding
 * - BPE and WordPiece support
 * - MurmurHash3 for fast lookups
 * - AVX-512 string comparison (when available)
 *
 * By Anthony Shivakumar
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

#include "tokenizer/tokenizer.h"
#include "tokenizer/murmurhash3.h"
#include "tokenizer/hash_table.h"

/* Token info stored in hash table value */
typedef struct {
    int32_t id;
    float score;
    bool is_special;
} TokenInfo;

/* Tokenizer structure */
struct CKTokenizer {
    /* Vocabulary: token string -> TokenInfo */
    CKTokenizerHashTable *vocab;

    /* Reverse vocab: id -> token string */
    char **id_to_token;
    size_t vocab_size;
    size_t vocab_capacity;

    /* Special token IDs */
    int32_t unk_id;
    int32_t bos_id;
    int32_t eos_id;
    int32_t pad_id;
    int32_t mask_id;

    /* Configuration */
    CKTokenizerConfig config;
};

/* Create a new tokenizer */
CKTokenizer *ck_tokenizer_create(CKTokenizerType type) {
    CKTokenizer *tok = (CKTokenizer *)malloc(sizeof(CKTokenizer));
    if (!tok) {
        return NULL;
    }

    memset(tok, 0, sizeof(*tok));

    /* Create hash table for vocabulary */
    tok->vocab = ck_tokenizer_hash_table_create(CK_TOKENIZER_HT_BUCKETS_LARGE);
    if (!tok->vocab) {
        free(tok);
        return NULL;
    }

    /* Initialize reverse vocab */
    tok->vocab_capacity = 4096;
    tok->id_to_token = (char **)calloc(tok->vocab_capacity, sizeof(char *));
    if (!tok->id_to_token) {
        ck_tokenizer_hash_table_free(tok->vocab, true);
        free(tok);
        return NULL;
    }

    /* Set default special tokens */
    tok->unk_id = 0;
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->pad_id = -1;
    tok->mask_id = -1;

    /* Set config */
    tok->config.type = type;
    tok->config.add_bos = false;
    tok->config.add_eos = false;
    tok->config.unk_score = -1e10f;

    return tok;
}

/* Free a tokenizer */
void ck_tokenizer_free(CKTokenizer *tok) {
    if (!tok) return;

    /* Free vocabulary entries */
    if (tok->vocab) {
        ck_tokenizer_hash_table_free(tok->vocab, true);
    }

    /* Free reverse vocab */
    if (tok->id_to_token) {
        for (size_t i = 0; i < tok->vocab_size; i++) {
            if (tok->id_to_token[i]) {
                free(tok->id_to_token[i]);
            }
        }
        free(tok->id_to_token);
    }

    free(tok);
}

/* Reset tokenizer state */
void ck_tokenizer_reset(CKTokenizer *tok) {
    if (!tok) return;

    ck_tokenizer_hash_table_clear(tok->vocab, true);

    for (size_t i = 0; i < tok->vocab_size; i++) {
        if (tok->id_to_token[i]) {
            free(tok->id_to_token[i]);
            tok->id_to_token[i] = NULL;
        }
    }

    tok->vocab_size = 0;
}

/* Add a token to vocabulary */
int ck_tokenizer_add_token(CKTokenizer *tok, const char *token, int32_t id, float score) {
    if (!tok || !token) {
        return -1;
    }

    /* Ensure we have space in reverse vocab */
    if (id >= (int32_t)tok->vocab_capacity) {
        size_t new_cap = tok->vocab_capacity * 2;
        while (new_cap <= (size_t)id) {
            new_cap *= 2;
        }
        char **new_array = (char **)realloc(tok->id_to_token, new_cap * sizeof(char *));
        if (!new_array) {
            return -1;
        }
        memset(new_array + tok->vocab_capacity, 0, (new_cap - tok->vocab_capacity) * sizeof(char *));
        tok->id_to_token = new_array;
        tok->vocab_capacity = new_cap;
    }

    /* Check if token already exists */
    TokenInfo *existing = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, token);
    if (existing) {
        /* Update existing */
        existing->id = id;
        existing->score = score;
        if (id >= (int32_t)tok->vocab_size) {
            tok->vocab_size = id + 1;
        }
        if (tok->id_to_token[id]) {
            free(tok->id_to_token[id]);
        }
        tok->id_to_token[id] = strdup(token);
        return 0;
    }

    /* Create new token info */
    TokenInfo *info = (TokenInfo *)malloc(sizeof(TokenInfo));
    if (!info) {
        return -1;
    }

    info->id = id;
    info->score = score;
    info->is_special = false;

    /* Insert into hash table */
    if (ck_tokenizer_hash_table_insert(tok->vocab, token, info) != 0) {
        free(info);
        return -1;
    }

    /* Add to reverse vocab */
    if (id >= (int32_t)tok->vocab_size) {
        tok->vocab_size = id + 1;
    }
    if (tok->id_to_token[id]) {
        free(tok->id_to_token[id]);
    }
    tok->id_to_token[id] = strdup(token);

    return 0;
}

/* Add special token */
int ck_tokenizer_add_special_token(CKTokenizer *tok, const char *name, int32_t id) {
    if (!tok || !name) {
        return -1;
    }

    if (ck_tokenizer_add_token(tok, name, id, -1e10f) != 0) {
        return -1;
    }

    TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, name);
    if (info) {
        info->is_special = true;
    }

    /* Map special token names to IDs */
    if (strcmp(name, "<unk>") == 0 || strcmp(name, "[UNK]") == 0) {
        tok->unk_id = id;
    } else if (strcmp(name, "<s>") == 0 || strcmp(name, "<bos>") == 0 || strcmp(name, "[BOS]") == 0) {
        tok->bos_id = id;
    } else if (strcmp(name, "</s>") == 0 || strcmp(name, "<eos>") == 0 || strcmp(name, "[EOS]") == 0) {
        tok->eos_id = id;
    } else if (strcmp(name, "<pad>") == 0 || strcmp(name, "[PAD]") == 0) {
        tok->pad_id = id;
    } else if (strcmp(name, "<mask>") == 0 || strcmp(name, "[MASK]") == 0) {
        tok->mask_id = id;
    }

    return 0;
}

/* Set special token IDs */
void ck_tokenizer_set_special_ids(CKTokenizer *tok, int32_t unk, int32_t bos, int32_t eos, int32_t pad, int32_t mask) {
    if (!tok) return;
    tok->unk_id = unk;
    tok->bos_id = bos;
    tok->eos_id = eos;
    tok->pad_id = pad;
    tok->mask_id = mask;
}

/* Look up token ID */
int32_t ck_tokenizer_lookup(const CKTokenizer *tok, const char *token) {
    if (!tok || !token) {
        return -1;
    }

    TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, token);
    if (info) {
        return info->id;
    }
    return tok->unk_id;
}

/* Get token string by ID */
const char *ck_tokenizer_id_to_token(const CKTokenizer *tok, int32_t id) {
    if (!tok || id < 0 || id >= (int32_t)tok->vocab_size) {
        return NULL;
    }
    return tok->id_to_token[id];
}

/* Get vocabulary size */
/* Note: ck_tokenizer_vocab_size() is defined in tokenizer.h as inline */

/* Check if token is special */
bool ck_tokenizer_is_special(const CKTokenizer *tok, int32_t id) {
    if (!tok || id < 0) {
        return false;
    }

    const char *token = ck_tokenizer_id_to_token(tok, id);
    if (!token) {
        return false;
    }

    TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, token);
    return info && info->is_special;
}

/* Find longest matching token at position */
static int32_t find_longest_match(const CKTokenizer *tok,
                                  const char *text,
                                  size_t text_len,
                                  size_t pos,
                                  size_t *match_len) {
    if (!tok || !text || pos >= text_len) {
        *match_len = 0;
        return tok ? tok->unk_id : -1;
    }

    /* Maximum token length is 256, but typical max is much smaller */
    size_t max_len = 64;
    if (pos + max_len > text_len) {
        max_len = text_len - pos;
    }

    int32_t best_id = tok->unk_id;
    size_t best_len = 0;

    /* Try from longest to shortest */
    for (size_t len = max_len; len >= 1; len--) {
        /* Use a temporary buffer for lookup */
        char tmp[65];
        memcpy(tmp, text + pos, len);
        tmp[len] = '\0';

        TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, tmp);
        if (info) {
            best_id = info->id;
            best_len = len;
            break;
        }

        /* Check if this is a special token */
        if (len > 1 && text[pos] == '<' && text[pos + len - 1] == '>') {
            info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, tmp);
            if (info) {
                best_id = info->id;
                best_len = len;
                break;
            }
        }
    }

    *match_len = best_len;
    return best_id;
}

/* Encode text to token IDs using greedy longest-match */
int ck_tokenizer_encode(const CKTokenizer *tok,
                        const char *text,
                        int text_len,
                        int32_t *ids,
                        int max_ids) {
    if (!tok || !text || !ids || max_ids <= 0) {
        return 0;
    }

    if (text_len < 0) {
        text_len = (int)strlen(text);
    }

    /* Add BOS token if configured */
    int out_idx = 0;
    if (tok->config.add_bos && tok->bos_id >= 0 && out_idx < max_ids) {
        ids[out_idx++] = tok->bos_id;
    }

    /* Tokenize using greedy longest-match */
    size_t pos = 0;
    while (pos < (size_t)text_len && out_idx < max_ids) {
        /* Skip whitespace for WordPiece-style tokenization */
        if (tok->config.type == CK_TOKENIZER_WORDPIECE) {
            while (pos < (size_t)text_len && isspace((unsigned char)text[pos])) {
                pos++;
            }
            if (pos >= (size_t)text_len) break;
        }

        size_t match_len = 0;
        int32_t id = find_longest_match(tok, text, text_len, pos, &match_len);

        if (match_len == 0) {
            /* No match found, emit UNK */
            if (tok->unk_id >= 0) {
                ids[out_idx++] = tok->unk_id;
            }
            pos++;
        } else {
            ids[out_idx++] = id;
            pos += match_len;
        }
    }

    /* Add EOS token if configured */
    if (tok->config.add_eos && tok->eos_id >= 0 && out_idx < max_ids) {
        ids[out_idx++] = tok->eos_id;
    }

    return out_idx;
}

/* Encode with special tokens */
int ck_tokenizer_encode_with_special(CKTokenizer *tok,
                                     const char *text,
                                     int text_len,
                                     int32_t *ids,
                                     int max_ids,
                                     bool add_special) {
    if (!tok) return 0;

    bool saved_bos = tok->config.add_bos;
    bool saved_eos = tok->config.add_eos;

    if (add_special) {
        tok->config.add_bos = true;
        tok->config.add_eos = true;
    }

    int result = ck_tokenizer_encode(tok, text, text_len, ids, max_ids);

    tok->config.add_bos = saved_bos;
    tok->config.add_eos = saved_eos;

    return result;
}

/* Decode token IDs to text */
int ck_tokenizer_decode(const CKTokenizer *tok,
                        const int32_t *ids,
                        int num_ids,
                        char *text,
                        int max_len) {
    if (!tok || !ids || !text || max_len <= 0) {
        return 0;
    }

    int len = 0;

    for (int i = 0; i < num_ids; i++) {
        int32_t id = ids[i];

        /* Skip special tokens */
        if (tok->unk_id >= 0 && id == tok->unk_id) continue;
        if (tok->bos_id >= 0 && id == tok->bos_id) continue;
        if (tok->eos_id >= 0 && id == tok->eos_id) continue;
        if (tok->pad_id >= 0 && id == tok->pad_id) continue;
        if (tok->mask_id >= 0 && id == tok->mask_id) continue;

        const char *token = ck_tokenizer_id_to_token(tok, id);
        if (!token) continue;

        int token_len = (int)strlen(token);

        /* Handle GPT-2 space marker (Ġ = 0xC4 0xA0) */
        if ((unsigned char)token[0] == 0xC4 && (unsigned char)token[1] == 0xA0) {
            if (len < max_len - 1) {
                text[len++] = ' ';
            }
            token++;
            token_len--;
        }

        /* Copy token */
        for (int j = 0; j < token_len && len < max_len - 1; j++) {
            text[len++] = token[j];
        }
    }

    text[len] = '\0';
    return len;
}

/* Decode to allocated string */
char *ck_tokenizer_decode_alloc(const CKTokenizer *tok,
                                const int32_t *ids,
                                int num_ids,
                                int *out_len) {
    if (!tok || !ids || !out_len) {
        return NULL;
    }

    /* Estimate output size: roughly 4 chars per token */
    int max_len = num_ids * 8 + 1;
    char *text = (char *)malloc(max_len);
    if (!text) {
        return NULL;
    }

    int len = ck_tokenizer_decode(tok, ids, num_ids, text, max_len);

    if (len <= 0) {
        free(text);
        return NULL;
    }

    /* Resize to fit */
    char *result = (char *)realloc(text, len + 1);
    if (!result) {
        result = text;
    }

    *out_len = len;
    return result;
}

/* Get tokenizer type name */
const char *ck_tokenizer_type_name(const CKTokenizer *tok) {
    if (!tok) return "unknown";

    switch (tok->config.type) {
        case CK_TOKENIZER_BPE: return "BPE";
        case CK_TOKENIZER_WORDPIECE: return "WordPiece";
        case CK_TOKENIZER_SPM: return "SentencePiece";
        default: return "unknown";
    }
}

/* Estimate token count */
size_t ck_tokenizer_estimate_tokens(const CKTokenizer *tok, const char *text) {
    if (!tok || !text) return 0;
    /* Rough estimate: ~4 chars per token */
    return strlen(text) / 4 + 2;
}

/* Get last error */
const char *ck_tokenizer_last_error(void) {
    /* TODO: Add error handling */
    return NULL;
}

/* Load vocabulary from text file (one token per line) */
int ck_tokenizer_load_text(CKTokenizer *tok, const char *path) {
    if (!tok || !path) {
        return -1;
    }

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open vocabulary file: %s\n", path);
        return -1;
    }

    char line[1024];
    int32_t id = 0;

    while (fgets(line, sizeof(line), f)) {
        /* Remove trailing newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }

        /* Skip empty lines and comments */
        if (len == 0 || line[0] == '#') {
            continue;
        }

        if (ck_tokenizer_add_token(tok, line, id, 0.0f) != 0) {
            fprintf(stderr, "Failed to add token: %s\n", line);
            fclose(f);
            return -1;
        }

        id++;
    }

    fclose(f);
    printf("Loaded %d tokens from %s\n", (int)ck_tokenizer_vocab_size(tok), path);
    return 0;
}

/* Load merges from text file (for BPE) */
int ck_tokenizer_load_merges(CKTokenizer *tok, const char *path) {
    /* BPE merges are typically applied during training, not loading */
    /* For now, this is a placeholder */
    (void)tok;
    (void)path;
    return 0;
}
