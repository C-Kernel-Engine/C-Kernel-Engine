/*
 * C-Kernel-Engine Tokenizer
 *
 * High-performance tokenizer supporting:
 * - BPE (Byte-Pair Encoding): GPT-2, LLaMA, Qwen
 * - WordPiece: BERT, RoBERTa
 * - SentencePiece (unigram): LLaMA, T5
 *
 * Features:
 * - MurmurHash3 hashing
 * - AVX-512 optimized string comparison
 * - Greedy longest-match encoding
 * - Full UTF-8 support
 * - GGUF vocab loading
 *
 * By Anthony Shivakumar
 */

#ifndef CK_TOKENIZER_H
#define CK_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "tokenizer/murmurhash3.h"
#include "tokenizer/memory_pool.h"
#include "tokenizer/hash_table.h"
#include "tokenizer/utf8.h"
#include "data_structures/tries/trie.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Export macro */
#ifdef _WIN32
#define CK_TOKENIZER_API __declspec(dllexport)
#else
#define CK_TOKENIZER_API __attribute__((visibility("default")))
#endif

/* Maximum token length */
#define CK_TOKENIZER_MAX_TOKEN_LEN 256

/* Maximum vocabulary size */
#define CK_TOKENIZER_MAX_VOCAB_SIZE 256000

/* Default hash table size */
#define CK_TOKENIZER_DEFAULT_HT_SIZE 65536

/* Tokenizer model type */
typedef enum {
    CK_TOKENIZER_BPE = 0,        /* Byte-Pair Encoding (GPT-2, LLaMA, Qwen) */
    CK_TOKENIZER_WORDPIECE = 1,  /* WordPiece (BERT, RoBERTa) */
    CK_TOKENIZER_SPM = 2         /* SentencePiece (unigram) */
} CKTokenizerType;

/* Space prefix style for BPE tokenizers */
typedef enum {
    CK_SPACE_PREFIX_AUTO = 0,    /* Auto-detect from vocabulary */
    CK_SPACE_PREFIX_GPT2 = 1,    /* GPT-2 style: Ġ (U+0120, bytes 0xC4 0xA0) */
    CK_SPACE_PREFIX_SPM = 2      /* SentencePiece style: ▁ (U+2581, bytes 0xE2 0x96 0x81) */
} CKSpacePrefixStyle;

/* Tokenizer configuration */
typedef struct {
    CKTokenizerType type;           /* Tokenization algorithm */
    bool add_bos;                   /* Add beginning-of-sequence token */
    bool add_eos;                   /* Add end-of-sequence token */
    bool lowercase;                 /* Convert text to lowercase before tokenizing */
    bool treat_whitespace_as_suffix; /* For SentencePiece */
    float unk_score;               /* Unknown token score (for SPM) */
    bool use_trie;                  /* Use trie for lookups (faster), false = use hash table */
    CKSpacePrefixStyle space_prefix_style; /* Space prefix style (GPT-2 Ġ vs SentencePiece ▁) */
    bool space_prefix_detected;     /* True if auto-detection has run */
} CKTokenizerConfig;

/* Vocabulary entry */
typedef struct {
    int32_t id;                     /* Token ID */
    float score;                    /* Score (for SPM) */
    bool is_special;                /* Is special token */
} CKTokenizerToken;

/* Main tokenizer structure */
typedef struct {
    /* Configuration */
    CKTokenizerConfig config;

    /* Vocabulary: token string -> token info */
    CKTokenizerHashTable *vocab;

    /* Trie for fast longest-match lookups (O(k) instead of O(n*k)) */
    CKTrie *vocab_trie;

    /* Reverse vocabulary: ID -> token string */
    char **id_to_token;
    size_t vocab_size;
    size_t vocab_capacity;

    /* Special token IDs */
    int32_t unk_id;
    int32_t bos_id;
    int32_t eos_id;
    int32_t pad_id;
    int32_t mask_id;

    /* Memory pool for allocations */
    CKTokenizerMemPool pool;

    /* For BPE: merge rules */
    int32_t *merge_pairs;           /* left_id * vocab_size + right_id -> merge priority */
    size_t merge_pairs_size;
    int32_t *merge_result;          /* merge priority -> merged token ID */
    size_t merge_result_size;
    int32_t num_merges;

    /* Cache for encoding */
    char *encode_buffer;
    size_t encode_buffer_size;
} CKTokenizer;

/* ============================================================================
 * Initialization and Cleanup
 * ============================================================================ */

/**
 * Create a new tokenizer.
 *
 * @param type     Tokenizer type (BPE, WordPiece, SPM)
 * @return         Newly allocated tokenizer, or NULL on error
 */
CK_TOKENIZER_API CKTokenizer *ck_tokenizer_create(CKTokenizerType type);

/**
 * Create tokenizer with default BPE config.
 */
static inline CKTokenizer *ck_tokenizer_create_bpe(void) {
    return ck_tokenizer_create(CK_TOKENIZER_BPE);
}

/**
 * Create tokenizer with default WordPiece config.
 */
static inline CKTokenizer *ck_tokenizer_create_wordpiece(void) {
    return ck_tokenizer_create(CK_TOKENIZER_WORDPIECE);
}

/**
 * Create tokenizer with default SPM config.
 */
static inline CKTokenizer *ck_tokenizer_create_spm(void) {
    return ck_tokenizer_create(CK_TOKENIZER_SPM);
}

/**
 * Free a tokenizer.
 *
 * @param tok  Tokenizer to free
 */
CK_TOKENIZER_API void ck_tokenizer_free(CKTokenizer *tok);

/**
 * Reset tokenizer state (clear vocab but keep config).
 *
 * @param tok  Tokenizer to reset
 */
CK_TOKENIZER_API void ck_tokenizer_reset(CKTokenizer *tok);

/* ============================================================================
 * Vocabulary Management
 * ============================================================================ */

/**
 * Add a token to vocabulary.
 *
 * @param tok    Tokenizer
 * @param token  Token string
 * @param id     Token ID
 * @param score  Token score (for SPM)
 * @return       0 on success, -1 on error
 */
int ck_tokenizer_add_token(CKTokenizer *tok,
                           const char *token,
                           int32_t id,
                           float score);

/**
 * Add special token (UNK, BOS, EOS, PAD, MASK).
 *
 * @param tok    Tokenizer
 * @param name   Special token name ("unk", "bos", "eos", "pad", "mask")
 * @param id     Token ID
 * @return       0 on success, -1 on error
 */
int ck_tokenizer_add_special_token(CKTokenizer *tok,
                                   const char *name,
                                   int32_t id);

/**
 * Set special token IDs.
 *
 * @param tok    Tokenizer
 * @param unk    Unknown token ID
 * @param bos    Beginning-of-sequence token ID
 * @param eos    End-of-sequence token ID
 * @param pad    Padding token ID
 * @param mask   Mask token ID
 */
CK_TOKENIZER_API void ck_tokenizer_set_special_ids(CKTokenizer *tok,
                                  int32_t unk,
                                  int32_t bos,
                                  int32_t eos,
                                  int32_t pad,
                                  int32_t mask);

/**
 * Set whether to lowercase input text before tokenizing.
 *
 * @param tok        Tokenizer
 * @param lowercase  If true, convert text to lowercase
 */
CK_TOKENIZER_API void ck_tokenizer_set_lowercase(CKTokenizer *tok, bool lowercase);

/**
 * Set lookup method (trie vs hash table).
 *
 * @param tok      Tokenizer
 * @param use_trie If true, use trie (faster for longest-match), false = hash table
 */
CK_TOKENIZER_API void ck_tokenizer_set_use_trie(CKTokenizer *tok, bool use_trie);

/**
 * Set space prefix style for BPE tokenizers.
 *
 * GPT-2/Qwen use Ġ (U+0120), LLaMA/SentencePiece use ▁ (U+2581).
 * Default is AUTO which auto-detects from vocabulary.
 *
 * @param tok    Tokenizer
 * @param style  Space prefix style (AUTO, GPT2, or SPM)
 */
CK_TOKENIZER_API void ck_tokenizer_set_space_prefix_style(CKTokenizer *tok, CKSpacePrefixStyle style);

/**
 * Auto-detect space prefix style from vocabulary.
 *
 * Checks for presence of tokens starting with Ġ vs ▁ to determine style.
 *
 * @param tok    Tokenizer
 * @return       Detected style (GPT2 or SPM)
 */
CK_TOKENIZER_API CKSpacePrefixStyle ck_tokenizer_detect_space_prefix_style(CKTokenizer *tok);

/**
 * Look up token ID by string.
 *
 * @param tok    Tokenizer
 * @param token  Token string
 * @return       Token ID, or unk_id if not found
 */
CK_TOKENIZER_API int32_t ck_tokenizer_lookup(const CKTokenizer *tok, const char *token);

/**
 * Get token string by ID.
 *
 * @param tok    Tokenizer
 * @param id     Token ID
 * @return       Token string, or NULL if invalid
 */
CK_TOKENIZER_API const char *ck_tokenizer_id_to_token(const CKTokenizer *tok, int32_t id);

/**
 * Get token info by ID.
 *
 * @param tok    Tokenizer
 * @param id     Token ID
 * @param score  Output: token score
 * @return       Token string, or NULL if invalid
 */
CK_TOKENIZER_API const char *ck_tokenizer_id_to_token_info(const CKTokenizer *tok,
                                          int32_t id,
                                          float *score);

/**
 * Get vocabulary size.
 */
static inline size_t ck_tokenizer_vocab_size(const CKTokenizer *tok) {
    return tok ? tok->vocab_size : 0;
}

/* ============================================================================
 * BPE Merge Rules
 * ============================================================================ */

/**
 * Add a BPE merge rule.
 *
 * @param tok        Tokenizer
 * @param left_id    Left token ID
 * @param right_id   Right token ID
 * @param merged_id  Merged token ID
 * @param priority   Lower = higher priority (applied first)
 * @return           0 on success, -1 on error
 */
int ck_tokenizer_add_merge(CKTokenizer *tok,
                           int32_t left_id,
                           int32_t right_id,
                           int32_t merged_id,
                           int32_t priority);

/* ============================================================================
 * Encoding (Text -> Token IDs)
 * ============================================================================ */

/**
 * Encode text to token IDs using greedy longest-match.
 *
 * For BPE: applies merge rules iteratively.
 * For WordPiece/SPM: greedy longest-match from vocabulary.
 *
 * @param tok        Tokenizer
 * @param text       Input text
 * @param text_len   Text length, or -1 for null-terminated
 * @param ids        Output token IDs
 * @param max_ids    Maximum IDs to write
 * @return           Number of tokens written
 */
int ck_tokenizer_encode(const CKTokenizer *tok,
                        const char *text,
                        int text_len,
                        int32_t *ids,
                        int max_ids);

/**
 * Encode with special token handling.
 *
 * @param tok        Tokenizer
 * @param text       Input text
 * @param text_len   Text length, or -1 for null-terminated
 * @param ids        Output token IDs
 * @param max_ids    Maximum IDs to write
 * @param add_special Add BOS/EOS tokens
 * @return           Number of tokens written
 */
int ck_tokenizer_encode_with_special(CKTokenizer *tok,
                                     const char *text,
                                     int text_len,
                                     int32_t *ids,
                                     int max_ids,
                                     bool add_special);

/**
 * Encode and return tokens as array of strings.
 *
 * @param tok        Tokenizer
 * @param text       Input text
 * @param text_len   Text length
 * @param out_tokens Output token strings (caller must free each)
 * @param max_tokens Maximum tokens
 * @return           Number of tokens written
 */
int ck_tokenizer_encode_tokens(const CKTokenizer *tok,
                               const char *text,
                               int text_len,
                               const char **out_tokens,
                               int max_tokens);

/* ============================================================================
 * Decoding (Token IDs -> Text)
 * ============================================================================ */

/**
 * Decode token IDs to text.
 *
 * @param tok      Tokenizer
 * @param ids      Input token IDs
 * @param num_ids  Number of IDs
 * @param text     Output text buffer
 * @param max_len  Maximum text length
 * @return         Number of bytes written
 */
int ck_tokenizer_decode(const CKTokenizer *tok,
                        const int32_t *ids,
                        int num_ids,
                        char *text,
                        int max_len);

/**
 * Decode to buffer allocated by caller.
 *
 * @param tok      Tokenizer
 * @param ids      Input token IDs
 * @param num_ids  Number of IDs
 * @param out_len  Output: length of decoded string
 * @return         Newly allocated string, or NULL on error
 */
CK_TOKENIZER_API char *ck_tokenizer_decode_alloc(const CKTokenizer *tok,
                                const int32_t *ids,
                                int num_ids,
                                int *out_len);

/* ============================================================================
 * File Loading
 * ============================================================================ */

/**
 * Load vocabulary from memory-mapped binary data.
 *
 * @param tok         Tokenizer
 * @param vocab_size  Number of tokens
 * @param offsets     Array of offsets into strings pool
 * @param strings     String pool containing null-terminated tokens
 * @param num_merges  Number of BPE merges
 * @param merges      Merge rules as (left, right, merged) triplets
 * @return            0 on success, -1 on error
 */
int ck_tokenizer_load_binary(CKTokenizer *tok,
                             int vocab_size,
                             const int32_t *offsets,
                             const char *strings,
                             int num_merges,
                             const int32_t *merges);

/**
 * Load vocabulary from GGUF file.
 *
 * @param tok    Tokenizer
 * @param path   Path to GGUF file
 * @return       0 on success, -1 on error
 */
int ck_tokenizer_load_gguf(CKTokenizer *tok, const char *path);

/**
 * Load vocabulary from JSON file (HuggingFace format).
 *
 * @param tok    Tokenizer
 * @param path   Path to vocab.json or tokenizer.json
 * @return       0 on success, -1 on error
 */
int ck_tokenizer_load_json(CKTokenizer *tok, const char *path);

/**
 * Load vocabulary from text file (one token per line).
 *
 * Format: token_string [id] [score]
 * Lines starting with # are comments.
 *
 * @param tok    Tokenizer
 * @param path   Path to vocabulary file
 * @return       0 on success, -1 on error
 */
int ck_tokenizer_load_text(CKTokenizer *tok, const char *path);

/**
 * Load BPE merges from text file.
 *
 * Format: token1 token2 (one merge per line)
 *
 * @param tok    Tokenizer
 * @param path   Path to merges.txt
 * @return       0 on success, -1 on error
 */
int ck_tokenizer_load_merges(CKTokenizer *tok, const char *path);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get the tokenizer type name.
 *
 * @param tok    Tokenizer
 * @return       Type name string
 */
CK_TOKENIZER_API const char *ck_tokenizer_type_name(const CKTokenizer *tok);

/**
 * Check if token is special.
 *
 * @param tok    Tokenizer
 * @param id     Token ID
 * @return       true if special token
 */
CK_TOKENIZER_API bool ck_tokenizer_is_special(const CKTokenizer *tok, int32_t id);

/**
 * Estimate encoded token count.
 *
 * @param tok    Tokenizer
 * @param text   Input text
 * @return       Estimated number of tokens
 */
CK_TOKENIZER_API size_t ck_tokenizer_estimate_tokens(const CKTokenizer *tok, const char *text);

/**
 * Get last error message.
 *
 * @return       Last error message, or NULL if no error
 */
CK_TOKENIZER_API const char *ck_tokenizer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* CK_TOKENIZER_H */
