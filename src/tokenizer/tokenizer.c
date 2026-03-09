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

/* Tokenizer structure - defined in tokenizer.h via typedef */
int ck_tokenizer_encode_spm_dispatch(const CKTokenizer *tok,
                                     const char *text,
                                     int text_len,
                                     int32_t *ids,
                                     int max_ids);

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

    /* Create trie for fast lookups (1M nodes for ~50k vocab) */
    tok->vocab_trie = ck_trie_create(1000000);
    if (!tok->vocab_trie) {
        ck_tokenizer_hash_table_free(tok->vocab, true);
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

    /* Initialize scores and types for SPM */
    tok->scores = NULL;
    tok->types = NULL;

    /* Set config */
    tok->config.type = type;
    tok->config.add_bos = false;
    tok->config.add_eos = false;
    tok->config.add_space_prefix = true;
    tok->config.unk_score = -1e10f;
    tok->config.spm_mode = CK_SPM_MODE_UNIGRAM;

    ck_tokenizer_mempool_init(&tok->pool, 1024 * 1024);

    return tok;
}

/* Free a tokenizer */
void ck_tokenizer_free(CKTokenizer *tok) {
    if (!tok) return;

    /* Free vocabulary entries */
    if (tok->vocab) {
        ck_tokenizer_hash_table_free(tok->vocab, true);
    }

    /* Free trie */
    if (tok->vocab_trie) {
        ck_trie_free(tok->vocab_trie);
    }

    /* Free reverse vocab strings */
    if (tok->id_to_token) {
        /* Note: strings were strdup'd in add_token */
        for (size_t i = 0; i < tok->vocab_size; i++) {
            if (tok->id_to_token[i]) {
                free(tok->id_to_token[i]);
            }
        }
        free(tok->id_to_token);
    }

    /* Free SPM-related arrays */
    if (tok->scores) free(tok->scores);
    if (tok->types) free(tok->types);
    if (tok->byte_token_id) free(tok->byte_token_id);

    ck_tokenizer_mempool_free(&tok->pool);
    free(tok);
}

/* Reset tokenizer state */
void ck_tokenizer_reset(CKTokenizer *tok) {
    if (!tok) return;

    ck_tokenizer_hash_table_clear(tok->vocab, true);

    if (tok->vocab_trie) {
        ck_trie_clear(tok->vocab_trie);
    }

    for (size_t i = 0; i < tok->vocab_size; i++) {
        if (tok->id_to_token[i]) {
            free(tok->id_to_token[i]);
            tok->id_to_token[i] = NULL;
        }
    }

    tok->vocab_size = 0;

    /* Reset SPM-related arrays using actual allocated sizes */
    if (tok->scores && tok->scores_size > 0) {
        memset(tok->scores, 0, tok->scores_size * sizeof(float));
    }
    if (tok->types && tok->types_size > 0) {
        memset(tok->types, 0, tok->types_size * sizeof(uint8_t));
    }
    /* Clear byte lookup table */
    if (tok->byte_token_id) {
        memset(tok->byte_token_id, -1, 256 * sizeof(int32_t));
    }
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
        existing->id = id;
        existing->score = score;
        if (id >= (int32_t)tok->vocab_size) tok->vocab_size = id + 1;
        if (tok->id_to_token[id]) free(tok->id_to_token[id]);
        tok->id_to_token[id] = strdup(token);
        return 0;
    }

    /* Create new token info */
    TokenInfo *info = (TokenInfo *)malloc(sizeof(TokenInfo));
    if (!info) return -1;
    info->id = id;
    info->score = score;
    info->is_special = false;

    if (ck_tokenizer_hash_table_insert(tok->vocab, token, info) != 0) {
        free(info);
        return -1;
    }

    /* Also add to trie for fast longest-match lookups */
    if (tok->vocab_trie) {
        ck_trie_insert(tok->vocab_trie, token, id, false, 0);
    }

    if (id >= (int32_t)tok->vocab_size) tok->vocab_size = id + 1;
    if (tok->id_to_token[id]) free(tok->id_to_token[id]);
    tok->id_to_token[id] = strdup(token);

    return 0;
}

/* Add special token */
int ck_tokenizer_add_special_token(CKTokenizer *tok, const char *name, int32_t id) {
    if (!tok || !name) return -1;
    if (ck_tokenizer_add_token(tok, name, id, -1e10f) != 0) return -1;

    TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, name);
    if (info) info->is_special = true;

    /* Also add to trie as special */
    if (tok->vocab_trie) {
        ck_trie_insert(tok->vocab_trie, name, id, true, 0);
    }

    if (strcmp(name, "<unk>") == 0 || strcmp(name, "[UNK]") == 0) tok->unk_id = id;
    else if (strcmp(name, "<s>") == 0 || strcmp(name, "<bos>") == 0 || strcmp(name, "[BOS]") == 0) tok->bos_id = id;
    else if (strcmp(name, "</s>") == 0 || strcmp(name, "<eos>") == 0 || strcmp(name, "[EOS]") == 0) tok->eos_id = id;
    else if (strcmp(name, "<pad>") == 0 || strcmp(name, "[PAD]") == 0) tok->pad_id = id;

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

void ck_tokenizer_set_add_bos_eos(CKTokenizer *tok, bool add_bos, bool add_eos) {
    if (!tok) return;
    tok->config.add_bos = add_bos;
    tok->config.add_eos = add_eos;
}

void ck_tokenizer_set_add_space_prefix(CKTokenizer *tok, bool add_space_prefix) {
    if (!tok) return;
    tok->config.add_space_prefix = add_space_prefix;
}

void ck_tokenizer_set_spm_mode(CKTokenizer *tok, CKSpmMode spm_mode) {
    if (!tok) return;
    tok->config.spm_mode = spm_mode;
}

/* Set whether to use trie for lookups */
void ck_tokenizer_set_use_trie(CKTokenizer *tok, bool use_trie) {
    if (!tok) return;
    tok->config.use_trie = use_trie;
}

/* Set space prefix style for BPE tokenizers */
void ck_tokenizer_set_space_prefix_style(CKTokenizer *tok, CKSpacePrefixStyle style) {
    if (!tok) return;
    tok->config.space_prefix_style = style;
    if (style != CK_SPACE_PREFIX_AUTO) {
        tok->config.space_prefix_detected = true;
    }
}

/* Auto-detect space prefix style from vocabulary.
 * Checks for presence of tokens starting with Ġ (GPT-2) vs ▁ (SentencePiece). */
CKSpacePrefixStyle ck_tokenizer_detect_space_prefix_style(CKTokenizer *tok) {
    if (!tok) return CK_SPACE_PREFIX_GPT2;

    /* Already detected? */
    if (tok->config.space_prefix_detected && tok->config.space_prefix_style != CK_SPACE_PREFIX_AUTO) {
        return tok->config.space_prefix_style;
    }

    /* Count tokens starting with each style:
     * Ġ (U+0120) = bytes 0xC4 0xA0
     * ▁ (U+2581) = bytes 0xE2 0x96 0x81
     */
    int gpt2_count = 0;
    int spm_count = 0;

    for (size_t i = 0; i < tok->vocab_size && i < 10000; i++) {  /* Sample first 10k tokens */
        const char *token = tok->id_to_token[i];
        if (!token) continue;

        unsigned char c0 = (unsigned char)token[0];
        unsigned char c1 = (unsigned char)token[1];

        /* Check for Ġ (0xC4 0xA0) */
        if (c0 == 0xC4 && c1 == 0xA0) {
            gpt2_count++;
        }
        /* Check for ▁ (0xE2 0x96 0x81) */
        else if (c0 == 0xE2 && c1 == 0x96 && (unsigned char)token[2] == 0x81) {
            spm_count++;
        }
    }

    /* Determine style based on counts */
    CKSpacePrefixStyle detected;
    if (spm_count > gpt2_count * 2 && spm_count > 0) {
        detected = CK_SPACE_PREFIX_SPM;
    } else if (gpt2_count > 0) {
        detected = CK_SPACE_PREFIX_GPT2;
    } else {
        detected = CK_SPACE_PREFIX_ASCII;
    }

    tok->config.space_prefix_style = detected;
    tok->config.space_prefix_detected = true;

    return detected;
}

/* Look up token ID */
int32_t ck_tokenizer_lookup(const CKTokenizer *tok, const char *token) {
    if (!tok || !token) return -1;
    TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, token);
    return info ? info->id : tok->unk_id;
}

/* Get token string by ID */
const char *ck_tokenizer_id_to_token(const CKTokenizer *tok, int32_t id) {
    if (!tok || id < 0 || id >= (int32_t)tok->vocab_size) return NULL;
    return tok->id_to_token[id];
}

/* Find longest matching token at position using trie (O(k) where k = token length) */
static int32_t find_longest_match_trie(const CKTokenizer *tok, const char *text, size_t text_len, size_t pos, size_t *match_len) {
    if (!tok || !tok->vocab_trie || !text || pos >= text_len) {
        *match_len = 0;
        return tok ? tok->unk_id : -1;
    }

    int32_t token_id = ck_trie_find_longest(tok->vocab_trie, text, text_len, pos, match_len);
    return token_id >= 0 ? token_id : tok->unk_id;
}

/* Find longest matching token at position using hash table (O(n*k) worst case) */
static int32_t find_longest_match_hash(const CKTokenizer *tok, const char *text, size_t text_len, size_t pos, size_t *match_len) {
    if (!tok || !text || pos >= text_len) {
        *match_len = 0;
        return tok ? tok->unk_id : -1;
    }

    size_t max_len = 64;
    if (pos + max_len > text_len) max_len = text_len - pos;

    int32_t best_id = tok->unk_id;
    size_t best_len = 0;

    for (size_t len = max_len; len >= 1; len--) {
        char tmp[65];
        memcpy(tmp, text + pos, len);
        tmp[len] = '\0';

        TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, tmp);
        if (info) {
            best_id = info->id;
            best_len = len;
            break;
        }
    }

    *match_len = best_len;
    return best_id;
}

/* Find longest matching token at position - dispatches to trie or hash table */
static int32_t find_longest_match(const CKTokenizer *tok, const char *text, size_t text_len, size_t pos, size_t *match_len) {
    if (tok->config.use_trie) {
        return find_longest_match_trie(tok, text, text_len, pos, match_len);
    } else {
        return find_longest_match_hash(tok, text, text_len, pos, match_len);
    }
}

/* Convert ASCII spaces to space prefix marker.
 * GPT-2/Qwen use Ġ (U+0120, bytes 0xC4 0xA0) - replaces spaces only
 * LLaMA/SentencePiece use ▁ (U+2581, bytes 0xE2 0x96 0x81) - adds prefix at start AND replaces spaces
 * Returns new length, or -1 if buffer too small. */
static int preprocess_bpe_spaces(const char *text, int text_len, char *out, int out_max, CKSpacePrefixStyle style) {
    int out_len = 0;

    if (style == CK_SPACE_PREFIX_ASCII) {
        if (text_len > out_max) return -1;
        memcpy(out, text, (size_t)text_len);
        return text_len;
    }

    /* For SentencePiece, add ▁ at the start of text (unless text starts with space) */
    if (style == CK_SPACE_PREFIX_SPM && text_len > 0 && text[0] != ' ') {
        if (out_len + 3 > out_max) return -1;
        out[out_len++] = (char)0xE2;
        out[out_len++] = (char)0x96;
        out[out_len++] = (char)0x81;
    }

    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            if (style == CK_SPACE_PREFIX_SPM) {
                /* SentencePiece style: ▁ (3 bytes: 0xE2 0x96 0x81) */
                if (out_len + 3 > out_max) return -1;
                out[out_len++] = (char)0xE2;
                out[out_len++] = (char)0x96;
                out[out_len++] = (char)0x81;
            } else {
                /* GPT-2 style: Ġ (2 bytes: 0xC4 0xA0) */
                if (out_len + 2 > out_max) return -1;
                out[out_len++] = (char)0xC4;
                out[out_len++] = (char)0xA0;
            }
        } else {
            if (out_len + 1 > out_max) return -1;
            out[out_len++] = text[i];
        }
    }
    return out_len;
}

/* Encode text to token IDs using greedy longest-match or Viterbi for SPM */
int ck_tokenizer_encode(const CKTokenizer *tok, const char *text, int text_len, int32_t *ids, int max_ids) {
    if (!tok || !text || !ids || max_ids <= 0) return 0;
    if (text_len < 0) text_len = (int)strlen(text);

    /* SentencePiece implementation lives in tokenizer_spm.c */
    if (tok->config.type == CK_TOKENIZER_SPM) {
        return ck_tokenizer_encode_spm_dispatch(tok, text, text_len, ids, max_ids);
    }

    /* For BPE tokenizers, convert spaces to appropriate prefix marker.
     * Auto-detect style from vocabulary if not already set. */
    char preprocessed[8192];
    const char *input = text;
    int input_len = text_len;

    if (tok->config.type == CK_TOKENIZER_BPE) {
        /* Get or detect space prefix style */
        CKSpacePrefixStyle style = ((CKTokenizer *)tok)->config.space_prefix_style;
        if (!((CKTokenizer *)tok)->config.space_prefix_detected) {
            style = ck_tokenizer_detect_space_prefix_style((CKTokenizer *)tok);
        }

        int pp_len = preprocess_bpe_spaces(text, text_len, preprocessed, sizeof(preprocessed) - 1, style);
        if (pp_len > 0) {
            preprocessed[pp_len] = '\0';
            input = preprocessed;
            input_len = pp_len;
        }
    }

    int out_idx = 0;
    if (tok->config.add_bos && tok->bos_id >= 0 && out_idx < max_ids) {
        ids[out_idx++] = tok->bos_id;
    }

    size_t pos = 0;
    while (pos < (size_t)input_len && out_idx < max_ids) {
        size_t match_len = 0;
        int32_t id = find_longest_match(tok, input, input_len, pos, &match_len);

        if (match_len == 0) {
            /* Emit UNK for unknown characters */
            if (tok->unk_id >= 0) ids[out_idx++] = tok->unk_id;
            pos++;
        } else {
            ids[out_idx++] = id;
            pos += match_len;
        }
    }

    if (tok->config.add_eos && tok->eos_id >= 0 && out_idx < max_ids) {
        ids[out_idx++] = tok->eos_id;
    }

    return out_idx;
}

/* Decode token IDs to text */
int ck_tokenizer_decode(const CKTokenizer *tok, const int32_t *ids, int num_ids, char *text, int max_len) {
    if (!tok || !ids || !text || max_len <= 0) return 0;
    int len = 0;
    for (int i = 0; i < num_ids; i++) {
        int32_t id = ids[i];
        if (id < 0) continue;
        const char *token = ck_tokenizer_id_to_token(tok, id);
        if (!token) continue;
        int token_len = (int)strlen(token);

        /* Check for space prefix markers and convert to ASCII space */
        unsigned char c0 = (unsigned char)token[0];
        unsigned char c1 = (unsigned char)token[1];

        if (c0 == 0xC4 && c1 == 0xA0) {
            /* Ġ (U+0120) is 2 bytes - convert to space */
            if (len < max_len - 1) text[len++] = ' ';
            token += 2; token_len -= 2;
        } else if (c0 == 0xE2 && c1 == 0x96 && (unsigned char)token[2] == 0x81) {
            /* ▁ (U+2581) is 3 bytes - convert to space */
            if (len < max_len - 1) text[len++] = ' ';
            token += 3; token_len -= 3;
        }

        /* SentencePiece byte tokens should decode back to the raw byte rather
         * than printing the literal vocabulary piece "<0xXX>". */
        if (token_len == 6 && token[0] == '<' && token[1] == '0' &&
            token[2] == 'x' && token[5] == '>') {
            unsigned int byte_val = 0;
            if (sscanf(token, "<0x%02X>", &byte_val) == 1 && byte_val < 256) {
                if (len < max_len - 1) text[len++] = (char)byte_val;
                continue;
            }
        }

        for (int j = 0; j < token_len && len < max_len - 1; j++) text[len++] = token[j];
    }
    text[len] = '\0';
    return len;
}

/* Load vocabulary from memory-mapped binary data */
int ck_tokenizer_load_binary(CKTokenizer *tok,
                             int vocab_size,
                             const int32_t *offsets,
                             const char *strings,
                             int num_merges,
                             const int32_t *merges) {
    return ck_tokenizer_load_binary_with_scores(tok, vocab_size, offsets, strings, NULL, NULL, num_merges, merges);
}

/* Placeholders for header compliance */
int ck_tokenizer_load_gguf(CKTokenizer *tok, const char *path) { (void)tok; (void)path; return -1; }
int ck_tokenizer_load_json(CKTokenizer *tok, const char *path) { (void)tok; (void)path; return -1; }
int ck_tokenizer_load_text(CKTokenizer *tok, const char *path) { (void)tok; (void)path; return -1; }
int ck_tokenizer_load_merges(CKTokenizer *tok, const char *path) { (void)tok; (void)path; return -1; }
int ck_tokenizer_add_merge(CKTokenizer *tok, int32_t left, int32_t right, int32_t merged, int32_t priority) {
    (void)tok; (void)left; (void)right; (void)merged; (void)priority; return 0;
}
