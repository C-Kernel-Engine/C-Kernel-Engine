/*
 * MurmurHash3 - 32-bit and 128-bit versions
 *
 * Ported from HPC_Embeddings for use in C-Kernel-Engine tokenizer.
 * Provides consistent, high-quality hashing for vocabulary lookups.
 *
 * Original by Austin Appleby, ported for C-Kernel-Engine.
 */

#ifndef CK_MURMURHASH3_H
#define CK_MURMURHASH3_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * MurmurHash3-32bit hash function (original HPC_Embeddings version).
 *
 * @param key      Data to hash
 * @param len      Length in bytes
 * @param seed     Seed value
 * @return         32-bit hash value
 */
uint32_t ck_murmurhash3(const char *key, uint32_t len, uint32_t seed);

/**
 * MurmurHash3-32bit hash function (alternative name).
 *
 * @param key      Data to hash
 * @param len      Length in bytes
 * @param seed     Optional seed value (use 0 for default)
 * @return         32-bit hash value
 */
uint32_t ck_murmurhash3_32(const void *key, size_t len, uint32_t seed);

/**
 * MurmurHash3-128bit hash function (produces two 64-bit values).
 *
 * @param key      Data to hash
 * @param len      Length in bytes
 * @param seed     Optional seed value (use 0 for default)
 * @param out1     First 64 bits of hash output
 * @param out2     Second 64 bits of hash output
 */
void ck_murmurhash3_128(const void *key, size_t len, uint32_t seed,
                        uint64_t *out1, uint64_t *out2);

/**
 * MurmurHash3-32bit for string (null-terminated).
 */
static inline uint32_t ck_murmurhash3_str(const char *str, uint32_t seed) {
    return ck_murmurhash3_32(str, strlen(str), seed);
}

/**
 * MurmurHash3-32bit for string with length.
 */
static inline uint32_t ck_murmurhash3_strn(const char *str, size_t len, uint32_t seed) {
    return ck_murmurhash3(str, len, seed);
}

#ifdef __cplusplus
}
#endif

#endif /* CK_MURMURHASH3_H */
