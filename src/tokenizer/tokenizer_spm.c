/*
 * SentencePiece-specific tokenizer implementation split from tokenizer.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "tokenizer/tokenizer.h"
#include "tokenizer/hash_table.h"

/* Token info stored in hash table value */
typedef struct {
    int32_t id;
    float score;
    bool is_special;
} TokenInfo;

/* Internal exact lookup (returns -1 if token string is not in vocab). */
static int32_t ck_tokenizer_lookup_exact(const CKTokenizer *tok, const char *token) {
    if (!tok || !token) return -1;
    TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, token);
    return info ? info->id : -1;
}

/* Internal exact lookup for non-null-terminated text slices. */
static int32_t ck_tokenizer_lookup_exact_n(const CKTokenizer *tok, const char *text, int text_len) {
    if (!tok || !text || text_len <= 0) return -1;
    char stack_buf[512];
    char *tmp = stack_buf;
    if (text_len >= (int)sizeof(stack_buf)) {
        tmp = (char *)malloc((size_t)text_len + 1);
        if (!tmp) return -1;
    }
    memcpy(tmp, text, (size_t)text_len);
    tmp[text_len] = '\0';
    int32_t id = ck_tokenizer_lookup_exact(tok, tmp);
    if (tmp != stack_buf) free(tmp);
    return id;
}

/* Find the longest registered special token starting at a byte offset. */
static int32_t spm_find_special_token_at_pos(const CKTokenizer *tok,
                                             const char *text,
                                             int text_len,
                                             size_t pos,
                                             size_t *match_len) {
    if (match_len) *match_len = 0;
    if (!tok || !text || text_len <= 0 || pos >= (size_t)text_len) return -1;

    if (tok->vocab_trie && tok->vocab_trie->root) {
        CKTrieNode *node = tok->vocab_trie->root;
        CKTrieNode *best_node = NULL;
        size_t best_len = 0;
        size_t cur = pos;

        while (cur < (size_t)text_len && node) {
            unsigned char c = (unsigned char)text[cur];
            if (!node->children[c]) break;
            node = node->children[c];
            cur++;
            if (node->token_id >= 0 && node->is_special) {
                best_node = node;
                best_len = cur - pos;
            }
        }

        if (best_node) {
            if (match_len) *match_len = best_len;
            return best_node->token_id;
        }
    }

    int max_len = CK_TOKENIZER_MAX_TOKEN_LEN;
    if (pos + (size_t)max_len > (size_t)text_len) {
        max_len = (int)((size_t)text_len - pos);
    }
    char tmp[CK_TOKENIZER_MAX_TOKEN_LEN + 1];
    for (int len = max_len; len >= 1; len--) {
        memcpy(tmp, text + pos, (size_t)len);
        tmp[len] = '\0';
        TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, tmp);
        if (info && info->id >= 0 && info->is_special) {
            if (match_len) *match_len = (size_t)len;
            return info->id;
        }
    }
    return -1;
}

/* ============================================================================
 * SPM (SentencePiece) Tokenization with Viterbi/DP
 * ============================================================================ */

/*
 * GGUF Token Type enum values (from llama.cpp gguf/constants.py):
 *   NORMAL       = 1
 *   UNKNOWN      = 2
 *   CONTROL      = 3
 *   USER_DEFINED = 4
 *   UNUSED       = 5
 *   BYTE         = 6
 *
 * IMPORTANT: These must match exactly or token filtering will be incorrect!
 */
#define GGUF_TOKEN_NORMAL       1
#define GGUF_TOKEN_UNKNOWN      2
#define GGUF_TOKEN_CONTROL      3
#define GGUF_TOKEN_USER_DEFINED 4
#define GGUF_TOKEN_UNUSED      5
#define GGUF_TOKEN_BYTE         6

/* Check if token type allows inclusion in DP path (exclude CONTROL, UNUSED, BYTE)
 * Note: UNKNOWN tokens are allowed because they're needed for unknown content */
static inline bool spm_token_allowed_in_dp(const CKTokenizer *tok, int32_t token_id) {
    if (!tok->types || token_id < 0 || token_id >= (int32_t)tok->vocab_size) {
        return true;  /* No type info, allow all */
    }
    uint8_t t = tok->types[token_id];
    /* Reject CONTROL, UNUSED, and BYTE tokens (but allow UNKNOWN for fallback) */
    return t != GGUF_TOKEN_CONTROL && t != GGUF_TOKEN_UNUSED && t != GGUF_TOKEN_BYTE;
}

/* Check if token is a byte token (for identification) */
static inline bool spm_is_byte_token(const CKTokenizer *tok, int32_t token_id) {
    if (!tok->types || token_id < 0 || token_id >= (int32_t)tok->vocab_size) {
        return false;
    }
    return tok->types[token_id] == GGUF_TOKEN_BYTE;
}

/* Find byte token ID using fast lookup table (primary) or <0xXX> fallback */
static inline int32_t spm_get_byte_token(const CKTokenizer *tok, unsigned char byte_val) {
    /* Try fast lookup table first */
    if (tok->byte_token_id && tok->byte_token_id[byte_val] >= 0) {
        return tok->byte_token_id[byte_val];
    }
    /* Fallback to <0xXX> format */
    char byte_token[16];
    int len = snprintf(byte_token, sizeof(byte_token), "<0x%02X>", byte_val);
    if (len <= 0) return tok->unk_id;
    return ck_tokenizer_lookup(tok, byte_token);
}

/* Check if a token string represents a byte token (<0xXX> format) */
static inline bool spm_token_is_byte_format(const char *token) {
    return token && token[0] == '<' && token[1] == '0' &&
           token[2] == 'x' && token[3] >= '0' && token[3] <= 'F' &&
           token[4] >= '0' && token[4] <= 'F' && token[5] == '>';
}

/* Build byte token lookup table from vocab (called during load) */
static void spm_build_byte_lookup(CKTokenizer *tok, const char *strings, const int32_t *offsets, int vocab_size) {
    /* Reuse existing array or allocate new one */
    if (!tok->byte_token_id) {
        tok->byte_token_id = (int32_t *)malloc(256 * sizeof(int32_t));
        if (!tok->byte_token_id) return;
    }

    /* Initialize all entries to -1 */
    for (int i = 0; i < 256; i++) {
        tok->byte_token_id[i] = -1;
    }

    /* Scan vocab for byte tokens */
    for (int i = 0; i < vocab_size; i++) {
        if (!tok->types || tok->types[i] != GGUF_TOKEN_BYTE) continue;

        const char *token = strings + offsets[i];
        size_t len = strlen(token);

        if (len == 1) {
            /* Raw byte token (single byte) */
            unsigned char byte_val = (unsigned char)token[0];
            tok->byte_token_id[byte_val] = i;
        } else if (spm_token_is_byte_format(token)) {
            /* <0xXX> format - parse the hex value */
            unsigned int byte_val;
            if (sscanf(token, "<0x%02X>", &byte_val) == 1 && byte_val < 256) {
                tok->byte_token_id[byte_val] = i;
            }
        }
    }
}

/* Get length of UTF-8 codepoint starting at c (0 if invalid) */
static inline int utf8_len(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;  /* ASCII */
    if ((c & 0xE0) == 0xC0) return 2;  /* 2-byte sequence */
    if ((c & 0xF0) == 0xE0) return 3;  /* 3-byte sequence */
    if ((c & 0xF8) == 0xF0) return 4;  /* 4-byte sequence */
    return 1;  /* Invalid, treat as 1 byte */
}

/* llama.cpp SPM whitespace handling:
 * - Optional dummy prefix as ASCII space
 * - Replace each ASCII space with ▁ (U+2581)
 * - Do not trim or collapse whitespace
 */
static int preprocess_spm_llama_text(const char *text, int text_len, char *out, int out_max, bool add_space_prefix) {
    int out_len = 0;
    if (text_len < 0) text_len = (int)strlen(text);

    if (add_space_prefix && text_len > 0) {
        if (out_len + 3 > out_max) return -1;
        out[out_len++] = (char)0xE2;
        out[out_len++] = (char)0x96;
        out[out_len++] = (char)0x81;
    }

    for (int i = 0; i < text_len;) {
        if (text[i] == ' ') {
            int j = i;
            while (j < text_len && text[j] == ' ') {
                j++;
            }
            int run = j - i;

            /* Match llama.cpp behavior for this GGUF family:
             * single separators map to ▁, but multi-space runs remain literal. */
            if (run == 1) {
                if (out_len + 3 > out_max) return -1;
                out[out_len++] = (char)0xE2;
                out[out_len++] = (char)0x96;
                out[out_len++] = (char)0x81;
            } else {
                if (out_len + run > out_max) return -1;
                for (int k = 0; k < run; k++) {
                    out[out_len++] = ' ';
                }
            }
            i = j;
        } else {
            if (out_len + 1 > out_max) return -1;
            out[out_len++] = text[i++];
        }
    }

    return out_len;
}

typedef struct {
    int prev;
    int next;
    const char *text;
    int n;
    int node_id;
} SpmLlamaSymbol;

typedef struct {
    const char *text;
    int n;
    int left;
    int right;
} SpmLlamaNode;

static int spm_llama_resegment_node(const CKTokenizer *tok,
                                    const SpmLlamaNode *nodes,
                                    int node_id,
                                    int32_t *ids,
                                    int max_ids,
                                    int out_idx) {
    if (!tok || !nodes || node_id < 0 || !ids || out_idx >= max_ids) {
        return out_idx;
    }

    const SpmLlamaNode *node = &nodes[node_id];
    int32_t token_id = ck_tokenizer_lookup_exact_n(tok, node->text, node->n);
    if (token_id >= 0) {
        ids[out_idx++] = token_id;
        return out_idx;
    }

    if (node->left >= 0 && node->right >= 0) {
        out_idx = spm_llama_resegment_node(tok, nodes, node->left, ids, max_ids, out_idx);
        out_idx = spm_llama_resegment_node(tok, nodes, node->right, ids, max_ids, out_idx);
        return out_idx;
    }

    for (int i = 0; i < node->n && out_idx < max_ids; i++) {
        int32_t byte_token = spm_get_byte_token(tok, (unsigned char)node->text[i]);
        ids[out_idx++] = (byte_token >= 0) ? byte_token : tok->unk_id;
    }
    return out_idx;
}

/* llama.cpp merge-style SPM path (LLAMA_VOCAB_TYPE_SPM). */
static int ck_tokenizer_encode_spm_llama_impl(const CKTokenizer *tok,
                                              const char *text,
                                              int text_len,
                                              int32_t *ids,
                                              int max_ids) {
    if (!tok || !text || !ids || max_ids <= 0) return 0;
    if (text_len < 0) text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    char preprocessed[8192];
    int pp_len = preprocess_spm_llama_text(text, text_len, preprocessed, (int)sizeof(preprocessed) - 1,
                                           tok->config.add_space_prefix);
    if (pp_len < 0) return 0;
    preprocessed[pp_len] = '\0';

    int num_symbols = 0;
    for (int offs = 0; offs < pp_len;) {
        int char_len = utf8_len((unsigned char)preprocessed[offs]);
        if (char_len <= 0) char_len = 1;
        if (offs + char_len > pp_len) char_len = pp_len - offs;
        offs += char_len;
        num_symbols++;
    }
    if (num_symbols <= 0) return 0;

    SpmLlamaSymbol *symbols = (SpmLlamaSymbol *)calloc((size_t)num_symbols, sizeof(SpmLlamaSymbol));
    int node_cap = 2 * num_symbols + 1;
    SpmLlamaNode *nodes = (SpmLlamaNode *)calloc((size_t)node_cap, sizeof(SpmLlamaNode));
    if (!symbols || !nodes) {
        if (symbols) free(symbols);
        if (nodes) free(nodes);
        return 0;
    }

    int index = 0;
    for (int offs = 0; offs < pp_len && index < num_symbols;) {
        int char_len = utf8_len((unsigned char)preprocessed[offs]);
        if (char_len <= 0) char_len = 1;
        if (offs + char_len > pp_len) char_len = pp_len - offs;

        symbols[index].text = preprocessed + offs;
        symbols[index].n = char_len;
        symbols[index].prev = index - 1;
        symbols[index].next = (index + 1 < num_symbols) ? (index + 1) : -1;
        symbols[index].node_id = index;

        nodes[index].text = preprocessed + offs;
        nodes[index].n = char_len;
        nodes[index].left = -1;
        nodes[index].right = -1;

        offs += char_len;
        index++;
    }

    int node_count = num_symbols;
    for (;;) {
        int best_left = -1;
        int best_right = -1;
        float best_score = -1e30f;

        for (int left = 0; left != -1; left = symbols[left].next) {
            int right = symbols[left].next;
            if (right < 0) continue;

            int pair_len = symbols[left].n + symbols[right].n;
            int32_t token_id = ck_tokenizer_lookup_exact_n(tok, symbols[left].text, pair_len);
            if (token_id < 0 || token_id >= (int32_t)tok->vocab_size) continue;

            float score = 0.0f;
            if (tok->scores && token_id >= 0 && token_id < (int32_t)tok->scores_size) {
                score = tok->scores[token_id];
            }

            if (best_left < 0 || score > best_score || (score == best_score && left < best_left)) {
                best_left = left;
                best_right = right;
                best_score = score;
            }
        }

        if (best_left < 0 || best_right < 0) break;
        if (node_count >= node_cap) break;

        SpmLlamaSymbol *left = &symbols[best_left];
        SpmLlamaSymbol *right = &symbols[best_right];

        int new_node_id = node_count++;
        nodes[new_node_id].text = left->text;
        nodes[new_node_id].n = left->n + right->n;
        nodes[new_node_id].left = left->node_id;
        nodes[new_node_id].right = right->node_id;

        left->n += right->n;
        left->node_id = new_node_id;
        left->next = right->next;
        if (right->next >= 0) {
            symbols[right->next].prev = best_left;
        }

        right->n = 0;
        right->prev = -1;
        right->next = -1;
    }

    int out_idx = 0;
    for (int i = 0; i != -1 && out_idx < max_ids; i = symbols[i].next) {
        out_idx = spm_llama_resegment_node(tok, nodes, symbols[i].node_id, ids, max_ids, out_idx);
    }

    free(symbols);
    free(nodes);
    return out_idx;
}

/* Replace spaces with SentencePiece underscore (U+2581)
 * Also normalize whitespace similarly to SPM:
 * - Leading spaces: consume them (SPM adds dummy prefix)
 * - Multiple spaces: collapse to single space
 * - Trailing spaces: consume them
 */
static int preprocess_spm_text(const char *text, int text_len, char *out, int out_max, bool add_space_prefix) {
    int out_len = 0;

    /* Count leading spaces */
    int lead_spaces = 0;
    while (lead_spaces < text_len && text[lead_spaces] == ' ') {
        lead_spaces++;
    }

    /* Count trailing spaces */
    int trail_spaces = 0;
    while (trail_spaces < text_len - lead_spaces &&
           text[text_len - 1 - trail_spaces] == ' ') {
        trail_spaces++;
    }

    /* Add ▁ at start if there's any non-space content AND text doesn't already start with ▁ */
    int content_len = text_len - lead_spaces - trail_spaces;
    int starts_with_prefix = (text_len >= 3 &&
                              (unsigned char)text[0] == 0xE2 &&
                              (unsigned char)text[1] == 0x96 &&
                              (unsigned char)text[2] == 0x81);
    int inserted_prefix = 0;
    if (content_len > 0 && !starts_with_prefix && add_space_prefix) {
        if (out_len + 3 > out_max) return -1;
        out[out_len++] = (char)0xE2;
        out[out_len++] = (char)0x96;
        out[out_len++] = (char)0x81;
        inserted_prefix = 1;
    }

    /* Process middle content: collapse multiple spaces to single ▁ */
    int i = lead_spaces;
    int last_was_space = (starts_with_prefix || inserted_prefix) ? 1 : 0;
    while (i < text_len - trail_spaces) {
        if (text[i] == ' ') {
            if (!last_was_space) {
                /* First space after content - add ▁ */
                if (out_len + 3 > out_max) return -1;
                out[out_len++] = (char)0xE2;
                out[out_len++] = (char)0x96;
                out[out_len++] = (char)0x81;
                last_was_space = 1;
            }
            /* Skip additional consecutive spaces */
        } else {
            if (out_len + 1 > out_max) return -1;
            out[out_len++] = text[i];
            last_was_space = 0;
        }
        i++;
    }

    return out_len;
}

/* Forward declaration for SPM Viterbi */
static int spm_find_candidates_at_pos(const CKTokenizer *tok, const char *text, int text_len,
                                      size_t pos, int32_t *candidates, int max_candidates);

/* Forward declaration for unknown run counting */
static int spm_count_unknown_run(const CKTokenizer *tok, const char *text, int text_len, size_t pos);

/* Forward declaration for byte fallback */
static int spm_encode_byte_fallback(const CKTokenizer *tok,
                                    const char *text, int text_len,
                                    int32_t *ids, int max_ids);

/* SPM Viterbi/DP encoding - finds best token sequence using token scores */
static int ck_tokenizer_encode_spm_impl(const CKTokenizer *tok,
                                       const char *text,
                                       int text_len,
                                       int32_t *ids,
                                       int max_ids) {
    if (!tok || !text || !ids || max_ids <= 0) return 0;
    if (text_len < 0) text_len = (int)strlen(text);
    if (text_len == 0) return 0;
    const int dbg = getenv("CK_DEBUG_SPM_ENCODE") ? 1 : 0;
    if (dbg) {
        fprintf(stderr, "[SPM] encode start: text_len=%d max_ids=%d\n", text_len, max_ids);
    }

    /* Preprocess: replace spaces with ▁ */
    char preprocessed[8192];
    int pp_len = preprocess_spm_text(text, text_len, preprocessed, sizeof(preprocessed) - 1,
                                     tok->config.add_space_prefix);
    if (pp_len < 0) return 0;
    preprocessed[pp_len] = '\0';
    if (dbg) {
        fprintf(stderr, "[SPM] preprocessed len=%d: \"%.*s\"\n", pp_len, pp_len, preprocessed);
    }

    /* DP arrays - use malloc for large inputs */
    size_t n = (size_t)pp_len + 1;
    float *best_score = (float *)malloc(n * sizeof(float));
    int32_t *best_prev = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *best_token = (int32_t *)malloc(n * sizeof(int32_t));
    if (dbg) {
        fprintf(stderr, "[SPM] DP alloc n=%zu\n", n);
    }

    if (!best_score || !best_prev || !best_token) {
        if (best_score) free(best_score);
        if (best_prev) free(best_prev);
        if (best_token) free(best_token);
        return 0;
    }

    /* Initialize DP */
    const float neg_inf = -1e30f;
    const float unknown_penalty = -10.0f;  /* SentencePiece-style UNK penalty */
    for (size_t i = 0; i < n; i++) {
        best_score[i] = neg_inf;
        best_prev[i] = -1;
        best_token[i] = -1;
    }
    best_score[0] = 0.0f;

    /* DP: for each position, find best way to reach it */
    for (size_t pos = 0; pos < n; pos++) {
        if (best_score[pos] == neg_inf) continue;

        /* Find all tokens that match at this position */
        int32_t candidates[64];
        int num_cand = spm_find_candidates_at_pos(tok, preprocessed, pp_len, pos, candidates, 64);
        if (dbg && pos < 8) {
            fprintf(stderr, "[SPM] pos=%zu cand=%d\n", pos, num_cand);
        }

        for (int c = 0; c < num_cand; c++) {
            int32_t token_id = candidates[c];

            /* Skip disallowed token types in DP */
            if (!spm_token_allowed_in_dp(tok, token_id)) {
                continue;
            }

            /* Get token string and length */
            const char *token = ck_tokenizer_id_to_token(tok, token_id);
            if (!token) continue;

            /* Calculate token length in bytes */
            int token_len = (int)strlen(token);

            /* For UNK token, use the unknown run length to cover all consecutive unknown bytes */
            if (token_id == tok->unk_id) {
                token_len = spm_count_unknown_run(tok, preprocessed, pp_len, pos);
                if (token_len == 0) token_len = 1;  /* At least 1 byte */
            }

            size_t next_pos = pos + token_len;

            if (next_pos >= n) continue;

            /* Get token score for Viterbi */
            float token_score = 0.0f;
            if (tok->scores && token_id >= 0 && token_id < (int32_t)tok->vocab_size) {
                token_score = tok->scores[token_id];
            }

            /* USER_DEFINED tokens get score 0 (like llama.cpp) */
            if (tok->types && token_id >= 0 && token_id < (int32_t)tok->types_size) {
                if (tok->types[token_id] == GGUF_TOKEN_USER_DEFINED) {
                    token_score = 0.0f;
                }
            }
            /* Apply UNK penalty (SentencePiece behavior) */
            if (token_id == tok->unk_id) {
                token_score += unknown_penalty;
            }

            /* Transition: score = best_score[pos] + token_score */
            float new_score = best_score[pos] + token_score;

            if (new_score > best_score[next_pos]) {
                best_score[next_pos] = new_score;
                best_prev[next_pos] = (int32_t)pos;
                best_token[next_pos] = token_id;
            }
        }
    }

    /* Backtrack to find best token sequence */
    int32_t *reverse_ids = (int32_t *)malloc(max_ids * sizeof(int32_t));
    if (!reverse_ids) {
        free(best_score);
        free(best_prev);
        free(best_token);
        return 0;
    }

    int num_tokens = 0;
    int32_t curr = (int32_t)(n - 1);

    /* Handle trailingUNK by finding valid end */
    while (curr > 0 && best_token[curr] < 0) {
        curr = best_prev[curr];
    }

    /* Backtrack from end to start, collecting tokens.
     * We track the token's start position to avoid duplicates. */
    int last_start = -1;  /* Track the start position of last added token */
    while (curr > 0 && num_tokens < max_ids) {
        int32_t token_id = best_token[curr];
        if (token_id >= 0) {
            /* Use the DP backpointer as the true token start */
            int token_start = best_prev[curr];

            /* Only add if this is a new token (different start position) */
            if (token_start != last_start) {
                reverse_ids[num_tokens++] = token_id;
                last_start = token_start;
            }
        }
        curr = best_prev[curr];
    }
    if (dbg) {
        fprintf(stderr, "[SPM] backtrack tokens=%d curr=%d\n", num_tokens, curr);
    }

    /* Free DP arrays before using reverse_ids */
    free(best_score);
    free(best_prev);
    free(best_token);

    if (num_tokens > max_ids) num_tokens = max_ids;

    /* Backtracking collected tokens in reverse order, so reverse once */
    for (int i = 0; i < num_tokens / 2; i++) {
        int32_t tmp = reverse_ids[i];
        reverse_ids[i] = reverse_ids[num_tokens - 1 - i];
        reverse_ids[num_tokens - 1 - i] = tmp;
    }

    /* Copy to output and merge consecutive UNK tokens (SPM behavior) */
    int out_idx = 0;
    for (int i = 0; i < num_tokens && out_idx < max_ids; i++) {
        int32_t token_id = reverse_ids[i];

        /* Merge consecutive UNK tokens into one */
        if (token_id == tok->unk_id && out_idx > 0 && ids[out_idx - 1] == tok->unk_id) {
            continue;  /* Skip - already have UNK */
        }
        ids[out_idx++] = token_id;
    }
    if (dbg) {
        fprintf(stderr, "[SPM] encode done: out=%d\n", out_idx);
    }

    free(reverse_ids);

    /* If DP failed to produce valid tokens, use byte-fallback */
    if (num_tokens == 0) {
        return spm_encode_byte_fallback(tok, text, text_len, ids, max_ids);
    }

    return out_idx;
}

static int ck_tokenizer_encode_spm_plain_segment(const CKTokenizer *tok,
                                                 const char *text,
                                                 int text_len,
                                                 int32_t *ids,
                                                 int max_ids) {
    if (!tok || !text || text_len <= 0 || !ids || max_ids <= 0) return 0;
    if (tok->config.spm_mode == CK_SPM_MODE_LLAMA) {
        return ck_tokenizer_encode_spm_llama_impl(tok, text, text_len, ids, max_ids);
    }
    return ck_tokenizer_encode_spm_impl(tok, text, text_len, ids, max_ids);
}

/* Fallback: encode using byte tokens for any unmatched content.
 * Uses the ORIGINAL text (not preprocessed), mapping each byte to a byte token. */
static int spm_encode_byte_fallback(const CKTokenizer *tok,
                                    const char *text, int text_len,
                                    int32_t *ids, int max_ids) {
    if (!tok || !text || !ids || max_ids <= 0) return 0;
    if (text_len < 0) text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    int count = 0;
    for (int i = 0; i < text_len && count < max_ids; i++) {
        unsigned char byte_val = (unsigned char)text[i];
        int32_t byte_token = spm_get_byte_token(tok, byte_val);

        /* If we have a byte token, use it; otherwise use UNK */
        if (byte_token >= 0 && byte_token != tok->unk_id) {
            ids[count++] = byte_token;
        } else {
            ids[count++] = tok->unk_id;
        }
    }
    return count;
}

/* Find all candidate tokens matching at position */
static int spm_find_candidates_at_pos(const CKTokenizer *tok, const char *text, int text_len,
                                      size_t pos, int32_t *candidates, int max_candidates) {
    if (!tok || !text || pos >= (size_t)text_len) return 0;

    int num_found = 0;
    int max_len = 64;
    if (pos + max_len > (size_t)text_len) max_len = (int)(text_len - pos);

    /* Iterate from longest to shortest to find all matches */
    char tmp[65];
    for (int len = max_len; len >= 1 && num_found < max_candidates; len--) {
        memcpy(tmp, text + pos, len);
        tmp[len] = '\0';

        TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, tmp);
        if (info && info->id >= 0 && info->id != tok->unk_id) {
            /* Skip disallowed token types */
            if (!spm_token_allowed_in_dp(tok, info->id)) {
                continue;
            }

            /* Check if already added */
            int dup = 0;
            for (int j = 0; j < num_found; j++) {
                if (candidates[j] == info->id) {
                    dup = 1;
                    break;
                }
            }
            if (!dup) {
                candidates[num_found++] = info->id;
            }
        }
    }

    /* If no candidates found, add UNK token as fallback.
     * For SPM, UNK should cover all consecutive unknown bytes until a known token or end.
     * We handle this by adding UNK with a special marker - we'll check at runtime
     * how far we can extend it. */
    if (num_found == 0 && tok->unk_id >= 0 && max_candidates > 0) {
        /* Only add UNK if it's allowed in DP */
        if (spm_token_allowed_in_dp(tok, tok->unk_id)) {
            candidates[num_found++] = tok->unk_id;
        }
    }

    return num_found;
}

/* Count how many consecutive bytes at text[pos] are not start of any vocab token.
 * Also stop at UTF-8 encoded '▁' (U+2581 = 0xE2 0x96 0x81) since that's a known token. */
static int spm_count_unknown_run(const CKTokenizer *tok, const char *text, int text_len, size_t pos) {
    int run = 0;
    while (pos + run < (size_t)text_len) {
        /* Stop at '▁' (U+2581 = 0xE2 0x96 0x81) since that's a known token */
        if (pos + run + 3 <= (size_t)text_len &&
            (unsigned char)text[pos + run] == 0xE2 &&
            (unsigned char)text[pos + run + 1] == 0x96 &&
            (unsigned char)text[pos + run + 2] == 0x81) {
            break;
        }

        /* Check if any vocab token matches at this position */
        int max_len = 64;
        if (pos + run + max_len > (size_t)text_len) {
            max_len = (int)(text_len - pos - run);
        }
        int found = 0;
        for (int len = max_len; len >= 1; len--) {
            char tmp[65];
            memcpy(tmp, text + pos + run, len);
            tmp[len] = '\0';
            TokenInfo *info = (TokenInfo *)ck_tokenizer_hash_table_lookup(tok->vocab, tmp);
            if (info && info->id >= 0 && info->id != tok->unk_id && spm_token_allowed_in_dp(tok, info->id)) {
                found = 1;
                break;
            }
        }
        if (found) break;
        run++;
    }
    return run;
}

/* Encode text to token IDs using SentencePiece paths only. */
int ck_tokenizer_encode_spm_dispatch(const CKTokenizer *tok,
                                     const char *text,
                                     int text_len,
                                     int32_t *ids,
                                     int max_ids) {
    if (!tok || !text || !ids || max_ids <= 0) return 0;
    if (text_len < 0) text_len = (int)strlen(text);

    int out_idx = 0;
    if (tok->config.add_bos && tok->bos_id >= 0 && out_idx < max_ids) {
        ids[out_idx++] = tok->bos_id;
    }
    if (text_len == 0) {
        if (tok->config.add_eos && tok->eos_id >= 0 && out_idx < max_ids) {
            ids[out_idx++] = tok->eos_id;
        }
        return out_idx;
    }

    int segment_start = 0;
    for (int pos = 0; pos < text_len && out_idx < max_ids;) {
        size_t special_len = 0;
        int32_t special_id = spm_find_special_token_at_pos(tok, text, text_len, (size_t)pos, &special_len);
        if (special_id < 0 || special_len == 0) {
            pos++;
            continue;
        }

        if (segment_start < pos) {
            int n = ck_tokenizer_encode_spm_plain_segment(
                tok,
                text + segment_start,
                pos - segment_start,
                ids + out_idx,
                max_ids - out_idx
            );
            if (n <= 0) return n;
            out_idx += n;
            if (out_idx >= max_ids) break;
        }

        ids[out_idx++] = special_id;
        pos += (int)special_len;
        segment_start = pos;
    }

    if (segment_start < text_len && out_idx < max_ids) {
        int n = ck_tokenizer_encode_spm_plain_segment(
            tok,
            text + segment_start,
            text_len - segment_start,
            ids + out_idx,
            max_ids - out_idx
        );
        if (n <= 0) return n;
        out_idx += n;
    }

    if (tok->config.add_eos && tok->eos_id >= 0 && out_idx < max_ids) {
        ids[out_idx++] = tok->eos_id;
    }

    return out_idx;
}

/* Load vocabulary from memory-mapped binary data with scores and types */
int ck_tokenizer_load_binary_with_scores(CKTokenizer *tok,
                                         int vocab_size,
                                         const int32_t *offsets,
                                         const char *strings,
                                         const float *scores,
                                         const uint8_t *types,
                                         int num_merges,
                                         const int32_t *merges) {
    if (!tok || !offsets || !strings) return -1;
    ck_tokenizer_reset(tok);

    /* Free any existing scores/types arrays before reallocating */
    if (tok->scores) {
        free(tok->scores);
        tok->scores = NULL;
        tok->scores_size = 0;
    }
    if (tok->types) {
        free(tok->types);
        tok->types = NULL;
        tok->types_size = 0;
    }

    /* Allocate scores and types arrays if provided */
    if (scores && vocab_size > 0) {
        tok->scores = (float *)malloc(vocab_size * sizeof(float));
        if (!tok->scores) return -1;
        memcpy(tok->scores, scores, vocab_size * sizeof(float));
        tok->scores_size = (size_t)vocab_size;
    }
    if (types && vocab_size > 0) {
        tok->types = (uint8_t *)malloc(vocab_size * sizeof(uint8_t));
        if (!tok->types) {
            if (tok->scores) {
                free(tok->scores);
                tok->scores = NULL;
            }
            return -1;
        }
        memcpy(tok->types, types, vocab_size * sizeof(uint8_t));
        tok->types_size = (size_t)vocab_size;
    }

    for (int i = 0; i < vocab_size; i++) {
        const char *token = strings + offsets[i];
        float score = scores ? scores[i] : 0.0f;
        ck_tokenizer_add_token(tok, token, i, score);
    }

    /* Build byte token lookup table if types are available */
    if (types && vocab_size > 0) {
        spm_build_byte_lookup(tok, strings, offsets, vocab_size);

        /* Log token type statistics */
        int count_normal = 0, count_unknown = 0, count_control = 0, count_byte = 0, count_other = 0;
        int max_type = 0;
        for (int i = 0; i < vocab_size; i++) {
            uint8_t t = tok->types[i];
            if (t > max_type) max_type = t;
            switch (t) {
                case GGUF_TOKEN_NORMAL: count_normal++; break;
                case GGUF_TOKEN_UNKNOWN: count_unknown++; break;
                case GGUF_TOKEN_CONTROL: count_control++; break;
                case GGUF_TOKEN_BYTE: count_byte++; break;
                default: count_other++; break;
            }
        }
        fprintf(stderr, "[TOKENIZER] Loaded %d tokens: normal=%d, unknown=%d, control=%d, byte=%d, other=%d\n",
                vocab_size, count_normal, count_unknown, count_control, count_byte, count_other);
        if (max_type > GGUF_TOKEN_BYTE) {
            fprintf(stderr, "[TOKENIZER] Warning: Unexpected token type %d\n", max_type);
        }
    }

    /* TODO: Merges */
    (void)num_merges; (void)merges;
    return 0;
}
