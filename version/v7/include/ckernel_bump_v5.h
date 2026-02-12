#ifndef CKERNEL_BUMP_V5_H
#define CKERNEL_BUMP_V5_H

/*
 * BUMPWGT5 format definition (v7).
 *
 * Design goals:
 *  - Preserve existing weight offsets (payload layout unchanged).
 *  - Append metadata at EOF with a footer for discovery.
 *  - Keep BUMPWGT4-compatible header size (128 bytes).
 */

#include <stdint.h>

#define CK_BUMP_MAGIC_V5 "BUMPWGT5"
#define CK_BUMP_META_FOOTER_MAGIC "BUMPV5MD"

#pragma pack(push, 1)
typedef struct {
    char magic[8];        /* "BUMPWGT5" */
    uint32_t version;     /* 5 */
    uint32_t model_type;  /* legacy placeholder (1) */
    uint32_t num_layers;
    uint32_t vocab_size;
    uint32_t embed_dim;
    uint32_t intermediate_size;
    uint32_t context_length;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint64_t aligned_embed_dim;
    uint64_t aligned_head_dim;
    uint64_t aligned_intermediate;
    uint64_t aligned_context;
    /* Tokenizer metadata (v4.1+) */
    uint32_t num_merges;
    uint32_t total_vocab_bytes;
    uint8_t checksum[32]; /* SHA-256 of dtype table + weights payload */
    uint8_t reserved[8];  /* Pad to 128 bytes (reserved) */
} CKBumpHeaderV5;
#pragma pack(pop)

#define CK_BUMP_HEADER_SIZE 128
#define CK_BUMP_EXT_META_SIZE 24
#define CK_BUMP_DATA_START (CK_BUMP_HEADER_SIZE + CK_BUMP_EXT_META_SIZE)

#pragma pack(push, 1)
typedef struct {
    char magic[8];        /* "BUMPV5MD" */
    uint64_t meta_size;   /* JSON blob size (bytes) */
    uint8_t meta_sha256[32]; /* SHA-256 of metadata JSON blob */
} CKBumpMetaFooterV5;
#pragma pack(pop)

#define CK_BUMP_META_FOOTER_SIZE ((uint64_t)(8 + 8 + 32))

#endif /* CKERNEL_BUMP_V5_H */
