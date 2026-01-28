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
 * This provides 100% parity with HuggingFace tokenizers when vocabulary
 * and merge rules are loaded correctly.
 *
 * By Anthony Shivakumar
 */

#ifndef CK_TRUE_BPE_H
#define CK_TRUE_BPE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "tokenizer/hash_table.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Export macro */
#ifdef _WIN32
#define CK_TRUE_BPE_API __declspec(dllexport)
#else
#define CK_TRUE_BPE_API __attribute__((visibility("default")))
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * Types
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Space prefix style for BPE tokenizers (same as tokenizer.h for compatibility) */
typedef enum {
    CK_SPACE_PREFIX_AUTO = 0,    /* Auto-detect from vocabulary */
    CK_SPACE_PREFIX_GPT2 = 1,    /* GPT-2 style: Ġ (U+0120, bytes 0xC4 0xA0) */
    CK_SPACE_PREFIX_SPM = 2      /* SentencePiece style: ▁ (U+2581, bytes 0xE2 0x96 0x81) */
} CKSpacePrefixStyle;

/* True BPE configuration */
typedef struct {
    bool add_bos;                    /* Add beginning-of-sequence token */
    bool add_eos;                    /* Add end-of-sequence token */
    bool byte_fallback;              /* Fall back to byte tokens for unknown chars */
    CKSpacePrefixStyle space_prefix_style;  /* Space prefix style (Ġ vs ▁) */
} CKBPEConfig;

/* Opaque tokenizer handle */
typedef struct CKTrueBPE CKTrueBPE;

/* ═══════════════════════════════════════════════════════════════════════════════
 * Creation and Destruction
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Create a new True BPE tokenizer.
 *
 * @return  Newly allocated tokenizer, or NULL on error
 */
CK_TRUE_BPE_API CKTrueBPE *ck_true_bpe_create(void);

/**
 * Free a True BPE tokenizer.
 *
 * @param bpe  Tokenizer to free
 */
CK_TRUE_BPE_API void ck_true_bpe_free(CKTrueBPE *bpe);

/* ═══════════════════════════════════════════════════════════════════════════════
 * Vocabulary Management
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Add a token to the vocabulary.
 *
 * @param bpe    Tokenizer
 * @param token  Token string (UTF-8)
 * @param id     Token ID
 * @param score  Token score (for unigram models, 0.0 for BPE)
 * @return       0 on success, -1 on error
 */
CK_TRUE_BPE_API int ck_true_bpe_add_token(CKTrueBPE *bpe,
                                          const char *token,
                                          int32_t id,
                                          float score);

/**
 * Add a BPE merge rule by token IDs.
 *
 * Merge rules define how tokens are combined during encoding.
 * Rules with lower priority numbers are applied first.
 *
 * @param bpe        Tokenizer
 * @param left_id    Left token ID
 * @param right_id   Right token ID
 * @param merged_id  Resulting merged token ID
 * @param priority   Merge priority (lower = applied first)
 * @return           0 on success, -1 on error
 */
CK_TRUE_BPE_API int ck_true_bpe_add_merge(CKTrueBPE *bpe,
                                          int32_t left_id,
                                          int32_t right_id,
                                          int32_t merged_id,
                                          int32_t priority);

/**
 * Add a BPE merge rule by token strings.
 *
 * This looks up the token IDs automatically and determines the merged token.
 * The merged token must already exist in the vocabulary.
 *
 * @param bpe       Tokenizer
 * @param left      Left token string
 * @param right     Right token string
 * @param priority  Merge priority (lower = applied first)
 * @return          0 on success, -1 on error
 */
CK_TRUE_BPE_API int ck_true_bpe_add_merge_by_tokens(CKTrueBPE *bpe,
                                                     const char *left,
                                                     const char *right,
                                                     int32_t priority);

/**
 * Set special token IDs.
 *
 * @param bpe  Tokenizer
 * @param unk  Unknown token ID (-1 to disable)
 * @param bos  Beginning-of-sequence token ID (-1 to disable)
 * @param eos  End-of-sequence token ID (-1 to disable)
 * @param pad  Padding token ID (-1 to disable)
 */
CK_TRUE_BPE_API void ck_true_bpe_set_special_ids(CKTrueBPE *bpe,
                                                  int32_t unk,
                                                  int32_t bos,
                                                  int32_t eos,
                                                  int32_t pad);

/**
 * Add a special token that should be matched BEFORE BPE encoding.
 *
 * Special tokens like <|im_start|>, <|im_end|>, <|endoftext|> are matched
 * literally in the input text before BPE processing. Without this, BPE would
 * break them into individual characters.
 *
 * @param bpe    Tokenizer
 * @param token  Token string to match literally (e.g., "<|im_end|>")
 * @param id     Token ID to output when matched
 * @return       0 on success, -1 on error
 */
CK_TRUE_BPE_API int ck_true_bpe_add_special_token(CKTrueBPE *bpe,
                                                   const char *token,
                                                   int32_t id);

/**
 * Set tokenizer configuration.
 *
 * @param bpe     Tokenizer
 * @param config  Configuration to apply
 */
CK_TRUE_BPE_API void ck_true_bpe_set_config(CKTrueBPE *bpe, const CKBPEConfig *config);

/**
 * Load vocabulary + merges from binary buffers.
 *
 * @param bpe         Tokenizer
 * @param vocab_size  Number of tokens
 * @param offsets     Offsets array (length vocab_size)
 * @param strings     Null-terminated token strings blob
 * @param num_merges  Number of merge rules
 * @param merges      Merge triples [left_id, right_id, merged_id] (length num_merges*3)
 * @return            0 on success, -1 on error
 */
CK_TRUE_BPE_API int ck_true_bpe_load_binary(CKTrueBPE *bpe,
                                            int vocab_size,
                                            const int32_t *offsets,
                                            const char *strings,
                                            int num_merges,
                                            const int32_t *merges);

/* ═══════════════════════════════════════════════════════════════════════════════
 * Token Lookup
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Look up a token ID by string.
 *
 * @param bpe    Tokenizer
 * @param token  Token string
 * @return       Token ID, or unk_id if not found
 */
CK_TRUE_BPE_API int32_t ck_true_bpe_lookup(const CKTrueBPE *bpe, const char *token);

/**
 * Get a token string by ID.
 *
 * @param bpe  Tokenizer
 * @param id   Token ID
 * @return     Token string, or NULL if invalid
 */
CK_TRUE_BPE_API const char *ck_true_bpe_id_to_token(const CKTrueBPE *bpe, int32_t id);

/**
 * Get vocabulary size.
 *
 * @param bpe  Tokenizer
 * @return     Number of tokens in vocabulary
 */
CK_TRUE_BPE_API size_t ck_true_bpe_vocab_size(const CKTrueBPE *bpe);

/**
 * Get number of merge rules.
 *
 * @param bpe  Tokenizer
 * @return     Number of merge rules
 */
CK_TRUE_BPE_API int32_t ck_true_bpe_num_merges(const CKTrueBPE *bpe);

/* ═══════════════════════════════════════════════════════════════════════════════
 * Space Prefix Detection
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Auto-detect space prefix style from vocabulary.
 *
 * Counts tokens starting with Ġ (GPT-2) vs ▁ (SentencePiece) to determine style.
 * The detected style is cached in the config.
 *
 * @param bpe  Tokenizer
 * @return     Detected style (GPT2 or SPM)
 */
CK_TRUE_BPE_API CKSpacePrefixStyle ck_true_bpe_detect_space_style(CKTrueBPE *bpe);

/* ═══════════════════════════════════════════════════════════════════════════════
 * Encoding and Decoding
 * ═══════════════════════════════════════════════════════════════════════════════ */

/**
 * Encode text to token IDs using true BPE algorithm.
 *
 * This applies merge rules in priority order (not greedy longest-match).
 *
 * @param bpe       Tokenizer
 * @param text      Input text (UTF-8)
 * @param text_len  Text length in bytes, or -1 for null-terminated
 * @param ids       Output token IDs array
 * @param max_ids   Maximum IDs to write
 * @return          Number of tokens written
 */
CK_TRUE_BPE_API int ck_true_bpe_encode(CKTrueBPE *bpe,
                                        const char *text,
                                        int text_len,
                                        int32_t *ids,
                                        int max_ids);

/**
 * Decode token IDs to text.
 *
 * @param bpe      Tokenizer
 * @param ids      Input token IDs
 * @param num_ids  Number of IDs
 * @param text     Output text buffer
 * @param max_len  Maximum text length
 * @return         Number of bytes written (excluding null terminator)
 */
CK_TRUE_BPE_API int ck_true_bpe_decode(const CKTrueBPE *bpe,
                                        const int32_t *ids,
                                        int num_ids,
                                        char *text,
                                        int max_len);

#ifdef __cplusplus
}
#endif

#endif /* CK_TRUE_BPE_H */
